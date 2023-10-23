import os
import time
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data_factory.dataloader import get_dataloader
from model.AnomalyTransformer import AnomalyTransformer
from utils.utils import *


def calculate_classification_accuracy(cls_probs, classes, labels) -> Tuple[int, int]:
    if type(cls_probs) == np.ndarray and len(cls_probs.shape) == 1:
        predicted = cls_probs
        gold = classes
    else:
        _, predicted = torch.max(cls_probs.data, 2)
        _, gold = torch.max(classes, 2)

    # Filter only anomaly regions
    predicted_for_anomaly_regions = predicted[labels == 1]
    gold_for_anomaly_regions = gold[labels == 1]
    if type(gold_for_anomaly_regions) == torch.Tensor:
        gold_for_anomaly_regions = gold_for_anomaly_regions.cuda()

    # Count correct
    cls_correct_cnt = (
        (predicted_for_anomaly_regions == gold_for_anomaly_regions).sum().item()
    )
    cls_total_num_cnt = len(predicted_for_anomaly_regions)

    (
        cls_precision,
        cls_recall,
        cls_f_score,
        cls_support,
    ) = precision_recall_fscore_support(
        gold_for_anomaly_regions,
        predicted_for_anomaly_regions,
        average="micro",
    )

    return cls_correct_cnt, cls_total_num_cnt


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
    def __init__(self, patience=7, verbose=False, dataset_name="", delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.best_score3 = None
        self.best_accuracy = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, val_loss3, accuracy, model, path):
        score = -val_loss
        score2 = -val_loss2
        score3 = -val_loss3
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.best_score3 = score3
            self.best_accuracy = accuracy
            self.save_checkpoint(val_loss, val_loss2, val_loss3, model, path)
        elif (
            (
                score < self.best_score + self.delta
                or score2 < self.best_score2 + self.delta
            )
            and score3 < self.best_score3 + self.delta
            and accuracy < self.best_accuracy + self.delta
        ):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.best_score3 = score3
            self.best_accuracy = accuracy
            self.save_checkpoint(val_loss, val_loss2, val_loss3, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, val_loss3, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(
            model.state_dict(),
            os.path.join(
                path,
                str(self.dataset) + "_checkpoint.pth",
            ),
        )
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2
        self.val_loss3_min = val_loss3


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_dataloader(
            data_path=self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            step=self.step_size,
            mode="train",
            dataset=self.dataset,
        )
        self.vali_loader = get_dataloader(
            data_path=self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            step=self.step_size,
            mode="val",
            dataset=self.dataset,
        )
        self.test_loader = get_dataloader(
            data_path=self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            step=self.step_size,
            mode="test",
            dataset=self.dataset,
        )
        self.thre_loader = get_dataloader(
            data_path=self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            step=self.step_size,
            mode="thre",
            dataset=self.dataset,
        )
        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.criterion2 = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=2)
        self.temperature = 50
        self.find_best = config["find_best"]

    def build_model(self):
        self.model = AnomalyTransformer(
            win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, data_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        loss_3 = []
        all_accuracy = []
        cls_num_cnt = 0
        cls_correct_cnt = 0
        for i, (input_data, labels, classes) in enumerate(data_loader):
            input = input_data.float().to(self.device)
            cls_output, output, series, prior, _ = self.model(input)
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
            cls_probs = self.softmax(cls_output)
            cls_loss = self.criterion2(cls_probs, classes.cuda())
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())
            loss_3.append(cls_loss.item())

            # Accumulate accuracy
            correct_cnt, total_cnt = calculate_classification_accuracy(
                cls_probs, classes, labels
            )
            cls_correct_cnt += correct_cnt
            cls_num_cnt += total_cnt

        accuracy = cls_correct_cnt / cls_num_cnt

        return np.average(loss_1), np.average(loss_2), np.average(loss_3), accuracy

    def train(self):
        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(
            patience=20, verbose=True, dataset_name=self.dataset
        )
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels, classes, is_overlaps) in enumerate(
                self.train_loader
            ):
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                cls_out, output, series, prior, _ = self.model(input)

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
                cls_probs = self.softmax(cls_out)
                # Filter only anomaly regions
                cls_probs_for_anomaly_regions = cls_probs[labels == 1]
                classes_for_anomaly_regions = classes[labels == 1]
                classification_loss = self.criterion2(
                    cls_probs_for_anomaly_regions, classes_for_anomaly_regions.cuda()
                )

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss
                # Classification loss
                loss3 = classification_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward(retain_graph=True)
                loss3.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2, vali_loss3, accuracy = self.vali(self.test_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Vali acc: {4:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1, accuracy
                )
            )
            early_stopping(
                vali_loss1, vali_loss2, vali_loss3, accuracy, self.model, path
            )
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(
                    str(self.model_save_path),
                    str(self.dataset) + "_checkpoint.pth",
                )
            )
        )
        self.model.eval()

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        train_labels = []
        attens_energy = []
        for i, (input_data, labels, classes) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            cls_output, output, series, prior, _ = self.model(input)
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
                        * self.temperature
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
                        * self.temperature
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
                        * self.temperature
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
                        * self.temperature
                    )

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            train_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        val_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.vali_loader):
            input = input_data.float().to(self.device)
            cls_output, output, series, prior, _ = self.model(input)

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
                        * self.temperature
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
                        * self.temperature
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
                        * self.temperature
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
                        * self.temperature
                    )
            # Metric
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            val_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        train_labels = np.concatenate(train_labels, axis=0).reshape(-1)
        train_labels = np.array(train_labels)
        val_labels = np.concatenate(val_labels, axis=0).reshape(-1)
        val_labels = np.array(val_labels)

        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        combined_labels = np.concatenate([train_labels, val_labels], axis=0)
        if self.find_best:
            thresh = self.find_best_threshold(combined_energy, combined_labels)
        else:
            thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        cls_preds = []
        cls_golds = []
        cls_num_cnt = 0
        cls_correct_cnt = 0
        for i, (input_data, labels, classes) in enumerate(self.test_loader):
            input = input_data.float().to(self.device)
            cls_output, output, series, prior, _ = self.model(input)

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
                        * self.temperature
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
                        * self.temperature
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
                        * self.temperature
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
                        * self.temperature
                    )
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)

            # Compute classification accuracy
            cls_prob = self.softmax(cls_output)
            _, cls_predicted = torch.max(cls_prob.data, 2)
            _, cls_gold = torch.max(classes, 2)

            # Filter only anomaly regions
            # predicted_for_anomaly_regions = predicted[labels == 1]
            # gold_for_anomaly_regions = gold[labels == 1]

            # cls_correct_cnt += (
            #     (predicted_for_anomaly_regions == gold_for_anomaly_regions.cuda())
            #     .sum()
            #     .item()
            # )
            # cls_num_cnt += predicted_for_anomaly_regions.size(0)

            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)
            cls_preds.append(cls_predicted)
            cls_golds.append(cls_gold)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)
        cls_preds = np.array(torch.stack(cls_preds).cpu()).reshape(-1)
        cls_golds = np.array(torch.stack(cls_golds).cpu()).reshape(-1)
        accuracy, precision, recall, f_score = self.get_metrics_for_threshold(
            test_energy,
            test_labels,
            thresh,
            cls_preds,
            cls_golds,
        )
        return accuracy, precision, recall, f_score

    def get_metrics_for_threshold(self, energy, labels, thresh, cls_preds, cls_golds):
        pred = (energy > thresh).astype(int)

        gt = labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        # Compute accuracy
        before_correct_cnt, before_total_cnt = calculate_classification_accuracy(
            cls_preds, cls_golds, labels
        )

        # Modify the gt labels
        visited_indices = []
        for start_idx in range(len(gt)):
            if start_idx in visited_indices:
                continue
            if gt[start_idx] == 1:
                # Find the range
                for end_idx in range(start_idx, len(gt)):
                    if gt[end_idx] == 0:
                        break
                # get cls preds
                cls_pred_sub = cls_preds[start_idx:end_idx]
                # Find the most frequent class
                tmp = {}
                for cls in cls_pred_sub:
                    if cls not in tmp.keys():
                        tmp[cls] = 0
                    tmp[cls] += 1
                # Sort the dict by value
                tmp = sorted(tmp.items(), key=lambda x: x[1], reverse=True)
                most_frequent_cls = tmp[0][0]
                cls_preds[start_idx:end_idx] = most_frequent_cls
                visited_indices += list(range(start_idx, end_idx))
            else:
                continue

        after_correct_cnt, after_total_cnt = calculate_classification_accuracy(
            cls_preds, cls_golds, labels
        )

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(
            gt, pred, average="binary"
        )

        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision, recall, f_score
            )
        )
        print(
            f"Before acc: {before_correct_cnt / before_total_cnt} ({before_correct_cnt}/{before_total_cnt})"
        )
        print(
            f"After acc: {after_correct_cnt / after_total_cnt} ({after_correct_cnt}/{after_total_cnt})"
        )

        return accuracy, precision, recall, f_score

    def find_best_threshold(
        self, combined_energy, combined_labels, ar_range=np.arange(0, 5.1, 0.1)
    ):
        best_f_score = 0
        best_thresh = None
        best_ar = None
        print("Finding best threshold...")
        for anomaly_ratio in ar_range:
            print(f"Anomaly Ratio: {anomaly_ratio}")
            thresh = np.percentile(combined_energy, 100 - anomaly_ratio)
            accuracy, precision, recall, f_score = self.get_metrics_for_threshold(
                combined_energy, combined_labels, thresh
            )
            if f_score > best_f_score:
                best_f_score = f_score
                best_thresh = thresh
                best_ar = anomaly_ratio
        print(f"Best F1 Score: {best_f_score}")
        print(f"Best Anomaly Ratio: {best_ar}")
        return best_thresh
