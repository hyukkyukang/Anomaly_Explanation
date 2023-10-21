import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_factory.data_loader import get_loader_segment
from model.AnomalyTransformer import AnomalyTransformer
from src.data_factory.dbsherlock.utils import anomaly_causes
from utils.utils import *

filtered_anomaly_causes = ["combined"] + [
    c for c in anomaly_causes if c != "Poor Physical Design"
]


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


class EarlyStopping:
    def __init__(
        self, patience=7, verbose=False, dataset_name="", delta=0, cause="all"
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name
        self.cause = cause

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif (
            score < self.best_score + self.delta
            or score2 < self.best_score2 + self.delta
        ):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(
            model.state_dict(),
            os.path.join(
                path,
                str(self.dataset)
                + f"_{self.cause}".replace(" ", "_").replace("/", "")
                + "_checkpoint.pth",
            ),
        )
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loaders = {}
        self.vali_loaders = {}
        self.test_loaders = {}
        self.thre_loaders = {}
        for cause in filtered_anomaly_causes:
            self.train_loaders[cause] = get_loader_segment(
                self.data_path,
                batch_size=self.batch_size,
                win_size=self.win_size,
                step=self.step_size,
                mode="train",
                dataset=self.dataset,
                cause=cause,
            )
            self.vali_loaders[cause] = get_loader_segment(
                self.data_path,
                batch_size=self.batch_size,
                win_size=self.win_size,
                step=self.step_size,
                mode="val",
                dataset=self.dataset,
                cause=cause,
            )
            self.test_loaders[cause] = get_loader_segment(
                self.data_path,
                batch_size=self.batch_size,
                win_size=self.win_size,
                step=self.step_size,
                mode="test",
                dataset=self.dataset,
                cause=cause,
            )
            self.thre_loaders[cause] = get_loader_segment(
                self.data_path,
                batch_size=1,
                win_size=self.win_size,
                step=self.step_size,
                mode="thre",
                dataset=self.dataset,
                cause=cause,
            )
        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self):
        self.models = {}
        self.optimizers = {}
        for cause in filtered_anomaly_causes:
            self.models[cause] = AnomalyTransformer(
                win_size=self.win_size,
                enc_in=self.input_c,
                c_out=self.output_c,
                e_layers=3,
            )
            self.optimizers[cause] = torch.optim.Adam(
                self.models[cause].parameters(), lr=self.lr
            )

            if torch.cuda.is_available():
                self.models[cause].cuda()

    def vali(self, vali_loader, cause):
        for cause in filtered_anomaly_causes:
            self.models[cause].eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.models[cause](input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += torch.mean(
                    my_kl_loss(
                        series[u],
                        (
                            prior[u]
                            / torch.unsqueeze(
                                torch.sum(prior[u], dim=-1), dim=-1
                            ).repeat(1, 1, 1, self.win_size)
                        ).detach(),
                    )
                ) + torch.mean(
                    my_kl_loss(
                        (
                            prior[u]
                            / torch.unsqueeze(
                                torch.sum(prior[u], dim=-1), dim=-1
                            ).repeat(1, 1, 1, self.win_size)
                        ).detach(),
                        series[u],
                    )
                )
                prior_loss += torch.mean(
                    my_kl_loss(
                        (
                            prior[u]
                            / torch.unsqueeze(
                                torch.sum(prior[u], dim=-1), dim=-1
                            ).repeat(1, 1, 1, self.win_size)
                        ),
                        series[u].detach(),
                    )
                ) + torch.mean(
                    my_kl_loss(
                        series[u].detach(),
                        (
                            prior[u]
                            / torch.unsqueeze(
                                torch.sum(prior[u], dim=-1), dim=-1
                            ).repeat(1, 1, 1, self.win_size)
                        ),
                    )
                )
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):
        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stoppings = {}
        for cause in filtered_anomaly_causes:
            early_stoppings[cause] = EarlyStopping(
                patience=3, verbose=True, dataset_name=self.dataset, cause=cause
            )
            train_steps = len(self.train_loaders[cause])

            for epoch in range(self.num_epochs):
                iter_count = 0
                loss1_list = []

                epoch_time = time.time()
                self.models[cause].train()
                for i, (input_data, labels, classes) in enumerate(
                    self.train_loaders[cause]
                ):
                    self.optimizers[cause].zero_grad()
                    iter_count += 1
                    input = input_data.float().to(self.device)

                    output, series, prior, _ = self.models[cause](input)

                    # calculate Association discrepancy
                    series_loss = 0.0
                    prior_loss = 0.0
                    for u in range(len(prior)):
                        series_loss += torch.mean(
                            my_kl_loss(
                                series[u],
                                (
                                    prior[u]
                                    / torch.unsqueeze(
                                        torch.sum(prior[u], dim=-1), dim=-1
                                    ).repeat(1, 1, 1, self.win_size)
                                ).detach(),
                            )
                        ) + torch.mean(
                            my_kl_loss(
                                (
                                    prior[u]
                                    / torch.unsqueeze(
                                        torch.sum(prior[u], dim=-1), dim=-1
                                    ).repeat(1, 1, 1, self.win_size)
                                ).detach(),
                                series[u],
                            )
                        )
                        prior_loss += torch.mean(
                            my_kl_loss(
                                (
                                    prior[u]
                                    / torch.unsqueeze(
                                        torch.sum(prior[u], dim=-1), dim=-1
                                    ).repeat(1, 1, 1, self.win_size)
                                ),
                                series[u].detach(),
                            )
                        ) + torch.mean(
                            my_kl_loss(
                                series[u].detach(),
                                (
                                    prior[u]
                                    / torch.unsqueeze(
                                        torch.sum(prior[u], dim=-1), dim=-1
                                    ).repeat(1, 1, 1, self.win_size)
                                ),
                            )
                        )
                    series_loss = series_loss / len(prior)
                    prior_loss = prior_loss / len(prior)

                    rec_loss = self.criterion(output, input)

                    loss1_list.append((rec_loss - self.k * series_loss).item())
                    loss1 = rec_loss - self.k * series_loss
                    loss2 = rec_loss + self.k * prior_loss

                    if (i + 1) % 100 == 0:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (
                            (self.num_epochs - epoch) * train_steps - i
                        )
                        print(
                            "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                                speed, left_time
                            )
                        )
                        iter_count = 0
                        time_now = time.time()

                    # Minimax strategy
                    loss1.backward(retain_graph=True)
                    loss2.backward()
                    self.optimizers[cause].step()

                print(
                    "Epoch: {} cost time: {}".format(
                        epoch + 1, time.time() - epoch_time
                    )
                )
                train_loss = np.average(loss1_list)

                vali_loss1, vali_loss2 = self.vali(self.test_loaders[cause], cause)

                print(
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                        epoch + 1, train_steps, train_loss, vali_loss1
                    )
                )
                early_stoppings[cause](vali_loss1, vali_loss2, self.models[cause], path)
                if early_stoppings[cause].early_stop:
                    print("Early stopping")
                    break
                adjust_learning_rate(self.optimizers[cause], epoch + 1, self.lr)

    def test(self):
        for anomaly_cause in filtered_anomaly_causes:
            self.models[anomaly_cause].load_state_dict(
                torch.load(
                    os.path.join(
                        str(self.model_save_path),
                        str(self.dataset)
                        + f"_{self.cause}".replace(" ", "_").replace("/", "")
                        + "_checkpoint.pth",
                    )
                )
            )
            self.models[anomaly_cause].eval()
        temperature = 50

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduce=False)
        threshs = {}
        for anomaly_cause in filtered_anomaly_causes:
            # (1) stastic on the train set
            attens_energy = []
            for i, (input_data, labels, classes) in enumerate(
                self.train_loaders[anomaly_cause]
            ):
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.models[anomaly_cause](input)
                loss = torch.mean(criterion(input, output), dim=-1)
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    if u == 0:
                        series_loss = (
                            my_kl_loss(
                                series[u],
                                (
                                    prior[u]
                                    / torch.unsqueeze(
                                        torch.sum(prior[u], dim=-1), dim=-1
                                    ).repeat(1, 1, 1, self.win_size)
                                ).detach(),
                            )
                            * temperature
                        )
                        prior_loss = (
                            my_kl_loss(
                                (
                                    prior[u]
                                    / torch.unsqueeze(
                                        torch.sum(prior[u], dim=-1), dim=-1
                                    ).repeat(1, 1, 1, self.win_size)
                                ),
                                series[u].detach(),
                            )
                            * temperature
                        )
                    else:
                        series_loss += (
                            my_kl_loss(
                                series[u],
                                (
                                    prior[u]
                                    / torch.unsqueeze(
                                        torch.sum(prior[u], dim=-1), dim=-1
                                    ).repeat(1, 1, 1, self.win_size)
                                ).detach(),
                            )
                            * temperature
                        )
                        prior_loss += (
                            my_kl_loss(
                                (
                                    prior[u]
                                    / torch.unsqueeze(
                                        torch.sum(prior[u], dim=-1), dim=-1
                                    ).repeat(1, 1, 1, self.win_size)
                                ),
                                series[u].detach(),
                            )
                            * temperature
                        )

                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)

            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            train_energy = np.array(attens_energy)

            # (2) find the threshold
            attens_energy = []
            for i, (input_data, labels, classes) in enumerate(
                self.thre_loaders[anomaly_cause]
            ):
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.models[anomaly_cause](input)

                loss = torch.mean(criterion(input, output), dim=-1)

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    if u == 0:
                        series_loss = (
                            my_kl_loss(
                                series[u],
                                (
                                    prior[u]
                                    / torch.unsqueeze(
                                        torch.sum(prior[u], dim=-1), dim=-1
                                    ).repeat(1, 1, 1, self.win_size)
                                ).detach(),
                            )
                            * temperature
                        )
                        prior_loss = (
                            my_kl_loss(
                                (
                                    prior[u]
                                    / torch.unsqueeze(
                                        torch.sum(prior[u], dim=-1), dim=-1
                                    ).repeat(1, 1, 1, self.win_size)
                                ),
                                series[u].detach(),
                            )
                            * temperature
                        )
                    else:
                        series_loss += (
                            my_kl_loss(
                                series[u],
                                (
                                    prior[u]
                                    / torch.unsqueeze(
                                        torch.sum(prior[u], dim=-1), dim=-1
                                    ).repeat(1, 1, 1, self.win_size)
                                ).detach(),
                            )
                            * temperature
                        )
                        prior_loss += (
                            my_kl_loss(
                                (
                                    prior[u]
                                    / torch.unsqueeze(
                                        torch.sum(prior[u], dim=-1), dim=-1
                                    ).repeat(1, 1, 1, self.win_size)
                                ),
                                series[u].detach(),
                            )
                            * temperature
                        )
                # Metric
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)

            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            test_energy = np.array(attens_energy)
            combined_energy = np.concatenate([train_energy, test_energy], axis=0)
            thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
            threshs[anomaly_cause] = thresh
            print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        test_classes = []
        attens_energys = {cause: [] for cause in filtered_anomaly_causes}
        for i, (input_data, labels, classes) in enumerate(
            self.thre_loaders["combined"]
        ):
            input = input_data.float().to(self.device)
            for anomaly_cause in filtered_anomaly_causes:
                output, series, prior, _ = self.models[anomaly_cause](input)

                loss = torch.mean(criterion(input, output), dim=-1)

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    if u == 0:
                        series_loss = (
                            my_kl_loss(
                                series[u],
                                (
                                    prior[u]
                                    / torch.unsqueeze(
                                        torch.sum(prior[u], dim=-1), dim=-1
                                    ).repeat(1, 1, 1, self.win_size)
                                ).detach(),
                            )
                            * temperature
                        )
                        prior_loss = (
                            my_kl_loss(
                                (
                                    prior[u]
                                    / torch.unsqueeze(
                                        torch.sum(prior[u], dim=-1), dim=-1
                                    ).repeat(1, 1, 1, self.win_size)
                                ),
                                series[u].detach(),
                            )
                            * temperature
                        )
                    else:
                        series_loss += (
                            my_kl_loss(
                                series[u],
                                (
                                    prior[u]
                                    / torch.unsqueeze(
                                        torch.sum(prior[u], dim=-1), dim=-1
                                    ).repeat(1, 1, 1, self.win_size)
                                ).detach(),
                            )
                            * temperature
                        )
                        prior_loss += (
                            my_kl_loss(
                                (
                                    prior[u]
                                    / torch.unsqueeze(
                                        torch.sum(prior[u], dim=-1), dim=-1
                                    ).repeat(1, 1, 1, self.win_size)
                                ),
                                series[u].detach(),
                            )
                            * temperature
                        )
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)

                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                attens_energys[anomaly_cause].append(cri)
            test_labels.append(labels)
            test_classes.append(classes)

        test_energys = {cause: [] for cause in filtered_anomaly_causes}
        preds = {cause: [] for cause in filtered_anomaly_causes}
        for anomaly_cause in filtered_anomaly_causes:
            attens_energys[anomaly_cause] = np.concatenate(
                attens_energys[anomaly_cause], axis=0
            ).reshape(-1)
            test_energys[anomaly_cause] = np.array(attens_energys[anomaly_cause])
            preds[anomaly_cause] = (test_energys[anomaly_cause] > thresh).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        test_classes = np.concatenate(test_classes, axis=0).reshape(-1)
        test_classes = np.array(test_classes)

        gt = test_labels.astype(int)
        gt_classes = test_classes.astype(int)

        for anomaly_cause in filtered_anomaly_causes:
            print("pred:   ", preds[anomaly_cause].shape)
        print("gt:     ", gt.shape)

        # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
        for anomaly_cause in filtered_anomaly_causes:
            anomaly_state = False
            for i in range(len(gt)):
                if gt[i] == 1 and preds[anomaly_cause][i] == 1 and not anomaly_state:
                    anomaly_state = True
                    for j in range(i, 0, -1):
                        if gt[j] == 0:
                            break
                        else:
                            if preds[anomaly_cause][j] == 0:
                                preds[anomaly_cause][j] = 1
                    for j in range(i, len(gt)):
                        if gt[j] == 0:
                            break
                        else:
                            if preds[anomaly_cause][j] == 0:
                                preds[anomaly_cause][j] = 1
                elif gt[i] == 0:
                    anomaly_state = False
                if anomaly_state:
                    preds[anomaly_cause][i] = 1

            preds[anomaly_cause] = np.array(preds[anomaly_cause])
            print("pred: ", preds[anomaly_cause].shape)
        gt = np.array(gt)
        print("gt:   ", gt.shape)

        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        # Calculate accuracy
        for i in range(len(gt_classes)):
            # Calculate precision
            if gt[i] == preds["combined"][i] and gt[i] == 1:
                # Find out the cause of the anomaly
                possible_causes = []
                for key, pred in preds.items():
                    if pred[i] == 1:
                        possible_causes.append(key)
                if len(possible_causes) == 10:
                    stop = 1
                else:
                    stop = 1

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(
            gt, pred, average="binary"
        )
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision, recall, f_score
            )
        )

        return accuracy, precision, recall, f_score
