import datetime
import logging
import os
from typing import *

import hkkang_utils.file as file_utils
import numpy as np

from DBAnomTransformer.data_factory.data import AnomalyData, AnomalyDataset
from DBAnomTransformer.data_factory.dataset.base import AnomalyTransformerDataset

logger = logging.getLogger("EDADataset")

SKIP_CAUSES = []

ANOMALY_CAUSES = [
    "No Anomaly",
    "db_backup",
    "index",
    "workload_spike",
    "poor_query",
    "mem",
    "cpu",
]

SKIP_FEATURES = ["timestamp", "cause"]

FEATURES = [
    "checkpoints_timed",
    "checkpoints_req",
    "checkpoint_write_time",
    "checkpoint_sync_time",
    "buffers_checkpoint",
    "buffers_clean",
    "maxwritten_clean",
    "buffers_backend",
    "buffers_backend_fsync",
    "buffers_alloc",
    "cpu_percent",
    "mem_total",
    "mem_available",
    "mem_percent",
    "mem_used",
    "mem_free",
    "mem_active",
    "mem_inactive",
    "mem_buffers",
    "mem_cached",
    "mem_shared",
    "mem_slab",
    "disk_percent",
    "disk_read_count",
    "disk_write_count",
    "disk_read_bytes",
    "disk_write_bytes",
    "tps",
    "latency_95th_percentile",
]


class EDADataset(AnomalyTransformerDataset):
    def __init__(self, *args, **kwargs):
        self.logger = logger
        self.anomaly_causes = ANOMALY_CAUSES
        self.skip_causes = SKIP_CAUSES
        super(EDADataset, self).__init__(*args, **kwargs)

    def load_data(self, path: str) -> "EDADataset":
        # Create AnomalyData
        data: AnomalyData = self._csv_to_anomaly_data(file_path=path, meta_data=None)

        # Return AnomalyDataset
        return AnomalyDataset(
            causes=[data.cause],
            data=[data],
        )

    def load_dataset(self, path: str) -> AnomalyDataset:
        assert os.path.isdir(path), f"{path} is not a directory"
        meta_dir_path = os.path.join(path, "meta_data")
        raw_dir_path = os.path.join(path, "raw_data")

        # Read meta data
        meta_data = self._read_meta_data(dir_path=meta_dir_path)

        # Create AnomalyData list
        anomaly_data_list: List[AnomalyData] = []
        for file_path in file_utils.get_files_in_directory(
            raw_dir_path, return_with_dir=True
        ):
            anomaly_data: AnomalyData = self._csv_to_anomaly_data(
                file_path=file_path, meta_data=meta_data
            )
            anomaly_data_list.append(anomaly_data)

        # Create AnomalyDataset
        return AnomalyDataset(
            causes=list(set([d.cause for d in anomaly_data_list])),
            data=anomaly_data_list,
        )

    def _read_meta_data(self, dir_path: str) -> Dict[str, List[Dict[str, Any]]]:
        assert os.path.isdir(dir_path), f"{dir_path} directory does not exist"

        # List all files in the directory
        file_paths = file_utils.get_files_in_directory(dir_path, return_with_dir=True)

        # Extract meta information for each anomaly cause
        meta_data = {}
        for file_path in file_paths:
            anomaly_cause = os.path.basename(file_path).replace(".csv", "")
            meta_data[anomaly_cause] = file_utils.read_csv_file(file_path)

        return meta_data

    @classmethod
    def _csv_to_anomaly_data(
        self, file_path: str, meta_data: Dict = None
    ) -> AnomalyData:
        # Parse anomaly cause and data instance number
        file_name = os.path.basename(file_path).replace(".csv", "")
        file_name_split = file_name.split("_")
        anomaly_cause = "_".join(file_name_split[:-1])
        data_instance_num = int(file_name_split[-1]) - 1

        # Get anomaly start and end time
        if meta_data:
            anomaly_start_time_str = meta_data[anomaly_cause][data_instance_num][
                "Anomaly Start Time"
            ]
            anomaly_end_time_str = meta_data[anomaly_cause][data_instance_num][
                "Anomaly End Time"
            ]
            anomaly_start_time = datetime.datetime.strptime(
                anomaly_start_time_str, "%Y-%m-%d %H:%M:%S"
            )
            anomaly_end_time = datetime.datetime.strptime(
                anomaly_end_time_str, "%Y-%m-%d %H:%M:%S"
            )
        else:
            anomaly_start_time = None
            anomaly_end_time = None

        # Read data
        data: List[Dict] = file_utils.read_csv_file(file_path)
        assert (
            data[0]["cause"] == anomaly_cause
        ), f"Different anomaly cause ({data[0]['cause']}) detected in {file_path}"

        # Initialize values
        normal_regions = []
        abnormal_regions = []
        values = np.zeros((len(data), len(FEATURES))).tolist()

        # Fill values
        for time_idx, datum in enumerate(data):
            # Parse timestamp and check if it is in the anomaly region
            timestamp = datetime.datetime.strptime(
                datum["timestamp"], "%Y-%m-%d %H:%M:%S"
            )
            if anomaly_start_time is not None and anomaly_end_time is not None:
                if anomaly_start_time <= timestamp and timestamp <= anomaly_end_time:
                    abnormal_regions.append(time_idx)
                else:
                    normal_regions.append(time_idx)

            # Add feature values for this time step
            for key, value in datum.items():
                if key in SKIP_FEATURES:
                    continue
                assert key in FEATURES, f"{key} is not a valid feature"
                feat_idx = FEATURES.index(key)
                values[time_idx][feat_idx] = value

        # Create anomaly data
        anomaly_data = AnomalyData(
            cause=anomaly_cause,
            attributes=FEATURES,
            values=values,
            normal_regions=normal_regions,
            abnormal_regions=abnormal_regions,
            skip_first_two_attributes=False,
        )

        return anomaly_data
