import logging
import os
from typing import *

import hkkang_utils.file as file_utils

from DBAnomTransformer.data_factory.data import AnomalyDataset
from DBAnomTransformer.data_factory.dataset.base import AnomalyTransformerDataset

logger = logging.getLogger("DBSherlockDataset")

SKIP_CAUSES = ["Poor Physical Design"]

ANOMALY_CAUSES = [
    "No Anomaly",
    "Poorly Written Query",
    "Poor Physical Design",
    "Workload Spike",
    "I/O Saturation",
    "DB Backup",
    "Table Restore",
    "CPU Saturation",
    "Flush Log/Table",
    "Network Congestion",
    "Lock Contention",
]


class DBSherlockDataset(AnomalyTransformerDataset):
    def __init__(self, *args, **kwargs):
        self.logger = logger
        self.anomaly_causes = ANOMALY_CAUSES
        self.skip_causes = SKIP_CAUSES
        super(DBSherlockDataset, self).__init__(*args, **kwargs)

    def load_data(self, path: str) -> AnomalyDataset:
        raise NotImplementedError

    def load_dataset(self, path: str) -> AnomalyDataset:
        assert path.endswith(".json")
        cache_path = path.replace(".json", ".pkl")
        # Load from cache if exists
        if os.path.isfile(cache_path):
            logger.info(f"Loading dataset from {cache_path}")
            return file_utils.read_pickle_file(cache_path)
        # Load from json file
        logger.info(f"Loading dataset from {path}")
        dataset_dic: Dict = file_utils.read_json_file(path)
        dataset = AnomalyDataset.from_dict(data=dataset_dic)
        # Add skip_first_two_attributes
        for datum in dataset.data:
            datum.skip_first_two_attributes = True
        # Save to cache
        logger.info(f"Saving dataset to {cache_path}")
        file_utils.write_pickle_file(dataset, cache_path)
        return dataset
