import json
import logging
import os
import pickle
from typing import *

import hydra
import numpy as np
import omegaconf
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from DBAnomTransformer.config.utils import default_config, recursive_override
from DBAnomTransformer.data_factory.data import AnomalyData
from DBAnomTransformer.data_factory.dataloader import get_dataloader
from DBAnomTransformer.data_factory.dataset.base import AnomalyTransformerDataset
from DBAnomTransformer.model.AnomalyTransformer import AnomalyTransformer
from DBAnomTransformer.solver import (
    EarlyStopping,
    adjust_learning_rate,
    calculate_classification_accuracy,
    compute_loss,
    compute_series_prior_loss,
    find_best_threshold,
    get_attention_energy,
    get_distances,
    load_dataset,
)

logger = logging.getLogger("AnomalyDetector")


class DBAnomDector:
    def __init__(
        self,
        config: Optional[omegaconf.DictConfig] = default_config,
        override_config: Optional[omegaconf.DictConfig] = None,
    ) -> None:
        self.config = config
        self.override_config = override_config
        self.model: AnomalyTransformer = None
        self.scaler: StandardScaler = None
        self.stats: Dict[str, Any] = None
        # For computing loss
        self.criterion = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=2)
        self.__post_init__()

    def __post_init__(self) -> None:
        # Override configs
        if self.override_config:
            assert isinstance(self.override_config, omegaconf.DictConfig)
            self.config = recursive_override(self.config, self.override_config)
        # Set device
        if "cuda" in self.config.device:
            assert torch.cuda.is_available(), "CUDA is not available"
        self.device = torch.device(self.config.device)

        # Load model
        self.model = AnomalyTransformer(
            win_size=self.config.model.win_size,
            enc_in=self.config.model.num_feature,
            c_out=self.config.model.num_feature,
            n_classes=self.config.model.num_anomaly_cause,
            e_layers=self.config.model.num_e_layer,
        )
        self.load_checkpoint()
        self.model.to(self.device)

    def __call__(self, *args, **kwargs) -> None:
        return self.infer(*args, **kwargs)

    def load_checkpoint(self) -> None:
        if os.path.exists(self.config.model_path):
            logger.info(f"Loading model from {self.config.model_path}")
            self.model.load_state_dict(torch.load(self.config.model_path))
            logger.info(f"Loading scaler from {self.config.scaler_path}")
            # Load scaler
            with open(self.config.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            # Load stats
            logger.info(f"Loading config from {self.config.stats_path}")
            with open(self.config.stats_path, "r") as f:
                self.stats = json.load(f)
        else:
            logger.warn(
                f"Model path {self.config.model_path} does not exist. Make sure to train model first."
            )

    def get_dataloaders(
        self, dataset_path: str, dataset_name: Optional[str] = None
    ) -> Tuple[torch.utils.data.DataLoader]:
        logger.info(f"Loading dataset from {dataset_path}")
        train_dataloader = get_dataloader(
            dataset_path=dataset_path,
            batch_size=self.config.train.batch_size,
            win_size=self.config.model.win_size,
            step=self.config.model.step_size,
            mode="train",
            dataset=dataset_name if dataset_name else self.config.dataset.name,
        )
        val_dataloader = get_dataloader(
            dataset_path=dataset_path,
            batch_size=self.config.train.batch_size,
            win_size=self.config.model.win_size,
            step=self.config.model.step_size,
            mode="val",
            dataset=dataset_name if dataset_name else self.config.dataset.name,
        )
        # Check num anomaly & num feature
        num_anomly_from_data = len(train_dataloader.dataset.anomaly_causes)
        num_feature_from_data = len(
            train_dataloader.dataset.dataset.data[0].valid_attributes
        )

        assert (
            self.config.model.num_anomaly_cause == num_anomly_from_data
        ), f"num_anomaly_cause in config is {self.config.model.num_anomaly_cause}, but num_anomaly_cause from data is {num_anomly_from_data}"
        assert (
            self.config.model.num_feature == num_feature_from_data
        ), f"num_feature in config is {self.config.model.num_feature}, but num_feature from data is {num_feature_from_data}"

        return train_dataloader, val_dataloader

    def train(
        self, dataset_path: Optional[str] = None, dataset_name: Optional[str] = None
    ) -> None:
        # To train model
        self.model.train()
        # Set optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.optimizer.lr
        )
        # Set dataset path
        if dataset_path is None:
            dataset_path = self.config.dataset.path
        if dataset_name is None:
            dataset_name = self.config.dataset.name
        # Get dataset loaders
        train_dataloader, val_dataloader = self.get_dataloaders(
            dataset_path, dataset_name
        )
        # Dataset for threshold finding
        loaded_dataset = load_dataset(train_dataloader, val_dataloader, None)
        early_stopping = EarlyStopping(
            patience=self.config.train.patience,
            verbose=self.config.train.verbose,
            dataset_name=dataset_name if dataset_name else self.config.dataset.name,
        )
        self.scaler = train_dataloader.dataset.scaler

        for epoch in range(self.config.optimizer.num_epoch):
            loss1_list = []
            for input_data, labels, classes, _ in train_dataloader:
                optimizer.zero_grad()
                input = input_data.float().to(self.device)
                cls_out, output, series, prior, _ = self.model(input)
                loss1, loss2, loss3 = compute_loss(
                    input=input,
                    recon_output=output,
                    classify_output=cls_out,
                    anomaly_labels=labels,
                    anomaly_causes=classes,
                    series=series,
                    prior=prior,
                    win_size=self.config.model.win_size,
                    k=self.config.model.k,
                )
                loss1_list.append(loss1.item())
                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward(retain_graph=True)
                loss3.backward()
                optimizer.step()

            avg_train_loss = np.average(loss1_list)

            # Compute validateion loss and accuracy
            vali_loss1, vali_loss2, vali_loss3, accuracy = self.vali(val_dataloader)
            self.model.train()

            logger.info(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Vali acc: {4:.7f} ".format(
                    epoch + 1,
                    len(train_dataloader),
                    avg_train_loss,
                    vali_loss1,
                    accuracy,
                )
            )
            early_stopping(
                vali_loss1,
                vali_loss2,
                vali_loss3,
                accuracy,
                self.model,
                train_dataloader.dataset.scaler,
                self.config.model_path,
                self.config.scaler_path,
            )
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

            adjust_learning_rate(optimizer, epoch + 1, self.config.optimizer.lr)

        # Figure out the best threshold in validation set
        with torch.no_grad():
            self.model.eval()
            val_energy, val_labels, val_overlap_mask = get_attention_energy(
                model=self.model,
                dataloader=val_dataloader,
                win_size=self.config.model.win_size,
                device=self.device,
                return_overlap_mask=True,
            )
            # Get threshold
            if self.config.find_best:
                (
                    anomaly_threshold,
                    stats_weight,
                    stats_feat_dim,
                ) = find_best_threshold(
                    val_energy=val_energy,
                    val_labels=val_labels,
                    overlapping_flags=val_overlap_mask,
                    step_size=self.config.model.step_size,
                    loaded_dataset=loaded_dataset,
                    add_stats=self.config.add_stats,
                )
            else:
                anomaly_threshold = np.percentile(
                    val_energy, 100 - self.config.anomaly_ratio
                )
                stats_weight = 0
                stats_feat_dim = 0

            self.stats = {
                "anomaly_threshold": float(anomaly_threshold),
                "stats_weight": float(stats_weight),
                "stats_feat_dim": int(stats_feat_dim),
            }
            # Save stats
            logger.info(f"Saving stats to {self.config.stats_path}")
            with open(self.config.stats_path, "w") as f:
                f.write(json.dumps(self.stats, indent=4))

    @torch.no_grad()
    def vali(self, data_loader):
        criterion = nn.MSELoss()
        cross_entropy = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=2)
        self.model.eval()

        loss_1 = []
        loss_2 = []
        loss_3 = []

        cls_num_cnt = 0
        cls_correct_cnt = 0
        for i, (input_data, labels, classes, _) in enumerate(data_loader):
            input = input_data.float().to(self.device)
            cls_output, output, series, prior, _ = self.model(input)

            # Compute series loss and prior loss
            series_loss, prior_loss = compute_series_prior_loss(
                series, prior, win_size=self.config.model.win_size, do_mean=True
            )

            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = criterion(output, input)
            pred_cls_probs = softmax(cls_output)
            gold_cls_probs = (
                torch.nn.functional.one_hot(
                    classes.long(),
                    num_classes=len(data_loader.dataset.anomaly_causes),
                )
                .float()
                .to(self.device)
            )
            cls_loss = cross_entropy(pred_cls_probs, gold_cls_probs)
            loss_1.append((rec_loss - self.config.model.k * series_loss).item())
            loss_2.append((rec_loss + self.config.model.k * prior_loss).item())
            loss_3.append(cls_loss.item())

            # Accumulate accuracy
            correct_cnt, total_cnt = calculate_classification_accuracy(
                pred_cls_probs, gold_cls_probs, labels
            )
            cls_correct_cnt += correct_cnt
            cls_num_cnt += total_cnt

        accuracy = cls_correct_cnt / cls_num_cnt if cls_num_cnt else 0

        return np.average(loss_1), np.average(loss_2), np.average(loss_3), accuracy

    @torch.no_grad()
    def infer(self, data: pd.DataFrame) -> Tuple[List[float], List[bool], List[int]]:
        self.model.eval()
        # Preprocess data
        anomaly_data = AnomalyData.from_dataframe(data)
        segments = AnomalyTransformerDataset.create_time_segments(
            data=anomaly_data, win_size=self.config.model.win_size, anomaly_causes=[]
        )
        # To batch
        attens_energy = []
        cls_preds = []
        overlapping_flags = []
        for segment in segments:
            value, label, cause, is_overlap = segment.to_item()

            # To tensor
            value = torch.tensor(value).unsqueeze(0).float().to(self.device)
            is_overlap = torch.tensor(is_overlap).unsqueeze(0).float().to(self.device)

            # Forward
            cls_output, output, series, prior, _ = self.model(value)

            # Compute series loss and prior loss
            loss = torch.mean(self.criterion(value, output), dim=-1)
            series_loss, prior_loss = compute_series_prior_loss(
                series, prior, win_size=self.config.model.win_size
            )
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()

            # Compute classification accuracy
            cls_prob = self.softmax(cls_output)
            _, cls_predicted = torch.max(cls_prob.data, 2)

            attens_energy.append(cri)
            cls_preds.append(cls_predicted.cpu())
            overlapping_flags.append(is_overlap.cpu())

        # Aggregate
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        cls_preds = np.array(torch.stack(cls_preds)).reshape(-1)
        overlapping_flags = np.concatenate(overlapping_flags, axis=0).reshape(-1)

        if self.stats and self.stats["stats_weight"] and False:
            # Calculate distance
            if self.stats["stats_feat_dim"]:
                stats = get_distances(
                    loaded_dataset=self.loaded_dataset,
                    step_size=self.step_size,
                    mode="test",
                    option="pca",
                    dim=self.stats["stats_feat_dim"],
                )
            else:
                stats = get_distances(
                    loaded_dataset=self.loaded_dataset,
                    step_size=self.step_size,
                    mode="test",
                )

            # Modify energy
            attens_energy = attens_energy + self.stats["stats_weight"] * stats

        # Filter only non-overlapping regions
        if self.stats and self.stats["anomaly_threshold"]:
            pred_is_anomaly = attens_energy > self.stats["anomaly_threshold"]
        else:
            pred_is_anomaly = attens_energy > self.config.default_anomaly_threshold
        attens_energy = attens_energy[overlapping_flags == 0]
        pred_is_anomaly = pred_is_anomaly[overlapping_flags == 0]
        pred_anomaly_cause = cls_preds[overlapping_flags == 0]

        # Post processing
        score = attens_energy.tolist()
        pred_is_anomaly = pred_is_anomaly.tolist()

        return score, pred_is_anomaly, pred_anomaly_cause


@hydra.main(
    config_path="/root/Anomaly_Explanation/configs/",
    config_name="base",
    version_base="1.3",
)
def main(config: omegaconf.DictConfig) -> None:
    # Preprocess data
    data = np.random.rand(130, 29)
    data_df = pd.DataFrame(data, columns=[f"attr_{i}" for i in range(29)])

    # Load model
    detector = DBAnomDector()
    # detector.train()

    # # Load checkpoint again
    # detector.load_checkpoint()

    score, is_anomaly, anomaly_cause = detector.infer(data=data_df)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )

    main()
    logger.info("Done!")
