import abc
import math
import random
from typing import *

import hkkang_utils.data as data_utils
import hkkang_utils.list as list_utils
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from DBAnomTransformer.data_factory.data import AnomalyData, AnomalyDataset


@data_utils.dataclass
class TimeSegment:
    start_time: int
    end_time: int
    value: np.ndarray  # dimension: (time, attribute)
    is_anomaly: List[bool]
    is_overlap: List[bool]
    anomaly_cause: List[int]

    def to_item(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        value: np.ndarray = self.value
        label: List[bool] = self.is_anomaly
        cause: List[int] = self.anomaly_cause
        is_overlap: List[bool] = self.is_overlap

        # Return
        return value, np.float32(label), np.float32(cause), np.float32(is_overlap)


class AnomalyTransformerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        win_size: int,
        step: int,
        mode: str,
        data_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        data_split_num: Optional[Tuple[int, int, int]] = None,
        scaler: Optional[StandardScaler] = None,
    ):
        self.data_path = data_path
        self.dataset_path = dataset_path
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.data_split_num = (
            data_split_num if data_split_num else (8 / 11, 1 / 11, 2 / 11)
        )
        self.time_segments: List[TimeSegment] = []
        self.scaler = scaler if scaler else StandardScaler()
        self.__post__init__(scaler=scaler)

    def __post__init__(self, scaler: Optional[StandardScaler] = None) -> None:
        # Check arguments
        assert (
            self.anomaly_causes is not None
        ), f"Override anomaly_causes in the child class"
        assert self.skip_causes is not None, f"Override skip_causes in the child class"
        assert (
            self.data_path != self.dataset_path
        ), f"Only one of data_path or dataset_path should be given"
        assert (
            self.data_path is not None or self.dataset_path is not None
        ), f"Either data_path or dataset should be given"
        # We expect scaler to be given in data_path mode, because we don't have access to the whole dataset
        assert (
            self.data_path is None or scaler is not None
        ), f"scaler should be given for data_path mode"

        # Load dataset
        if self.data_path:
            # Create time segments
            self.dataset: AnomalyDataset = self.load_data(self.data_path)
            time_segments = list_utils.do_flatten_list(
                [
                    self.create_time_segments(d, win_size=self.win_size)
                    for d in self.dataset.data
                ]
            )
            self.logger.info(
                f"Created {len(time_segments)} time segments from 1 anomaly data)"
            )
        else:
            self.dataset: AnomalyDataset = self.load_dataset(self.dataset_path)

            # Split dataset
            train_data_list, val_data_list, test_data_list = self.split_dataset(
                self.dataset, seed=120
            )

            # Save data list
            self.train_data_list = train_data_list
            self.val_data_list = val_data_list
            self.test_data_list = test_data_list

            if self.mode == "train":
                data_list = train_data_list
            elif self.mode == "val":
                data_list = val_data_list
            elif self.mode == "test":
                data_list = test_data_list
            else:
                data_list = test_data_list
            data_for_fitting = train_data_list

            # Fit scaler
            segments_for_fitting = list_utils.do_flatten_list(
                [
                    self.create_time_segments(
                        d, win_size=self.win_size, anomaly_causes=self.anomaly_causes
                    )
                    for d in data_for_fitting
                ]
            )
            self.fit_scaler(segments_for_fitting=segments_for_fitting)

            # Create time segments
            time_segments = list_utils.do_flatten_list(
                [
                    self.create_time_segments(
                        d, win_size=self.win_size, anomaly_causes=self.anomaly_causes
                    )
                    for d in data_list
                ]
            )
            self.logger.info(
                f"Created {len(time_segments)} time segments from {len(data_list)} anomaly data)"
            )
        # Scale values
        self.logger.info(f"Scaling values for {self.mode} mode")
        self.time_segments = self.scale_values(segments_to_fit=time_segments)
        self.logger.info(f"{self.mode} dataset is ready!\n")

    def __len__(self) -> int:
        return len(self.time_segments)

    def __getitem__(
        self, index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Get data
        return self.time_segments[index].to_item()

    @abc.abstractmethod
    def load_data(self, path: str) -> AnomalyDataset:
        """Load single anomaly data from the given path (Return as AnomalyDataset)"""
        raise NotImplementedError("Implement this method on the child class")

    @abc.abstractmethod
    def load_dataset(self, path: str) -> AnomalyDataset:
        """Load anomaly dataset from the given path"""
        raise NotImplementedError("Implement this method on the child class")

    @classmethod
    def create_time_segments(
        self,
        data: AnomalyData,
        win_size: int,
        anomaly_causes: Optional[List[str]] = None,
    ) -> List[TimeSegment]:
        segments: List[TimeSegment] = []
        total_time = len(data.values)
        assert total_time > win_size, f"total_time: {total_time}, win_size: {win_size}"

        # Create segments from the given anomaly data
        for start_time in range(0, total_time, win_size):
            end_time = min(start_time + win_size, total_time)

            # Check if the data is enough for the window size.
            # If not, use overlapping data from the previous segment.
            is_overlap = [False] * win_size
            if end_time - start_time < win_size:
                # Use overlapping data
                overlap_size = win_size - (end_time - start_time)
                is_overlap[:overlap_size] = [True] * overlap_size
                start_time = end_time - win_size

            # Create a segment
            value = np.array(data.values[start_time:end_time])
            if data.skip_first_two_attributes:
                value = value[
                    :, 2:
                ]  # We ignore first two attributes (following DBSherlock)
            is_anomaly = [
                idx in data.valid_abnormal_regions
                for idx in range(start_time, end_time)
            ]
            if anomaly_causes:
                cause = anomaly_causes.index(data.cause)
            else:
                cause = 0
            anomaly_cause = [cause if is_anomaly[idx] else 0 for idx in range(win_size)]
            segments.append(
                TimeSegment(
                    start_time=start_time,
                    end_time=end_time,
                    value=value,
                    is_anomaly=is_anomaly,
                    is_overlap=is_overlap,
                    anomaly_cause=anomaly_cause,
                )
            )

        return segments

    def split_dataset(
        self, dataset: AnomalyDataset, seed: Optional[int] = None
    ) -> Tuple[List[AnomalyData], List[AnomalyData], List[AnomalyData]]:
        train_data = []
        val_data = []
        test_data = []

        # Set random seed
        seed = 0 if seed is None else seed
        random.seed(seed)

        for cause in dataset.causes:
            if cause in self.skip_causes:
                continue

            # Get split indices for each cause
            num_data_per_cause = len(dataset.get_data_of_cause(cause))
            indices = list(range(num_data_per_cause))
            random.shuffle(indices)
            cut1 = math.ceil(self.data_split_num[0] * num_data_per_cause)
            cut2 = cut1 + math.ceil(self.data_split_num[1] * num_data_per_cause)
            train_indicies = indices[:cut1]
            val_indicies = indices[cut1:cut2]

            data_of_cause = dataset.get_data_of_cause(cause)
            for id in range(num_data_per_cause):
                if id in train_indicies:
                    train_data.append(data_of_cause[id])
                elif id in val_indicies:
                    val_data.append(data_of_cause[id])
                else:
                    test_data.append(data_of_cause[id])

        return train_data, val_data, test_data

    def scale_values(self, segments_to_fit: List[TimeSegment]) -> List[TimeSegment]:
        # Scale all values
        for time_segment in segments_to_fit:
            time_segment.value = self.scaler.transform(time_segment.value)

        return segments_to_fit

    def fit_scaler(self, segments_for_fitting: List[TimeSegment]) -> None:
        assert segments_for_fitting is not None

        # Check attribute size of all time segments are the same
        assert len(set([s.value.shape[1] for s in segments_for_fitting])) == 1

        # Calculate means for each attribute
        values = np.vstack([seg.value for seg in segments_for_fitting])

        self.scaler.fit(values)
