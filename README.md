# Anomaly Detection and Explanation
We develop deep learning model that detects and explain anomaly in multivariate time series data.

Our model is based on [Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy (ICLR'22)](https://openreview.net/forum?id=LzQQ89U1qm_). We train and evaluate the model on [DBSherlock dataset](https://github.com/hyukkyukang/DBSherlock).

## Anomaly Transformer

Anomaly transformer is a transformer-based model that detects anomaly in multivariate time series data. It is based on the assumption that the normal data is highly correlated, while the abnormal data is not. It uses a transformer encoder to learn the correlation between different time steps, and then uses a discriminator to distinguish the normal and abnormal data based on the learned correlation.

- An inherent distinguishable criterion as **Association Discrepancy** for detection.
- A new **Anomaly-Attention** mechanism to compute the association discrepancy.
- A **minimax strategy** to amplify the normal-abnormal distinguishability of the association discrepancy.

<p align="center">
<img src=".\pics\structure.png" height = "350" alt="" align=center />
</p>

For more details, please refer to the [paper](https://openreview.net/forum?id=LzQQ89U1qm_).

## Environment Setup
Start docker container using docker compose, and login to the container

```bash
docker compose up -d
```
Install python packages
```bash
pip install -r requirements.txt
```

## Prepare Dataset
### Download
Download DBSherlock dataset.
```bash
python scripts/dataset/download_datasets.py
```

Append `--download_all` argument to download all datasets (i.e., SMD, SMAP, PSM, MSL, and DBSherlock).
```bash
python scripts/dataset/download_datasets.py --download_all
```

### Preprocess data

Convert DBSherlock data (.mat file to .json file):
```bash
python src/DBAnomTransformer/data_factory/convert_dbsherlock.py \
    --input dataset/dbsherlock/tpcc_16w.mat \
    --out_dir dataset/dbsherlock/converted/ \
    --prefix tpcc_16w

python src/DBAnomTransformer/data_factory/convert_dbsherlock.py \
    --input dataset/dbsherlock/tpcc_500w.mat \
    --out_dir dataset/dbsherlock/converted/ \
    --prefix tpcc_500w

python src/DBAnomTransformer/data_factory/convert_dbsherlock.py \
    --input dataset/dbsherlock/tpce_3000.mat \
    --out_dir dataset/dbsherlock/converted/ \
    --prefix tpce_3000
```

Convert DBSherlock data into train & validate data for Anomaly Transformer:
```bash
python src/DBAnomTransformer/data_factory/process.py \
    --input_path dataset/dbsherlock/converted/tpcc_16w_test.json \
    --output_path dataset/dbsherlock/processed/tpcc_16w/

python src/DBAnomTransformer/data_factory/process.py \
    --input_path dataset/dbsherlock/converted/tpcc_500w_test.json \
    --output_path dataset/dbsherlock/processed/tpcc_500w/

python src/DBAnomTransformer/data_factory/process.py \
    --input_path dataset/dbsherlock/converted/tpce_3000_test.json \
    --output_path dataset/dbsherlock/processed/tpce_3000/
```

## Reproducing Experiments
We provide the experiment scripts under the folder `./scripts`. You can reproduce the experiment results with the below script:
```bash
bash ./scripts/experiment/DBS.sh
```
or you can run the below commands to train and evaluate the model step by step.

### Training
Train the model on DBSherlock dataset:
```bash
python src/DBAnomTransformer/main.py \
    --dataset EDA \
    --dataset_path dataset/EDA/ \
    --mode train
```

### Evaluating
Evaluate the trained model on the test split of the same dataset:
```bash
python src/DBAnomTransformer/main.py \
    --dataset EDA \
    --dataset_path dataset/EDA/ \
    --mode test 
```

## Inference
Download the package through pip
```bash
pip install DBAnomTransformer
```
Load the trained model and use it to detect anomaly in new data.
Below is an example of using the model to detect anomaly in dummy data (as DBS or EDA dataset).
```python
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from DBAnomTransformer.config.utils import default_config
from DBAnomTransformer.detector import DBAnomDector

# dataset_name = "DBS"
dataset_name = "EDA"

# Create config
eda_config = default_config
dbsherlock_config = OmegaConf.create(
    {
        "model": {"num_anomaly_cause": 11, "num_feature": 200},
        "model_path": "checkpoints/DBS_checkpoint.pth",
        "scaler_path": "checkpoints/DBS_scaler.pkl",
        "stats_path": "checkpoints/DBS_stats.json",
    }
)


# Create dummy data
if dataset_name == "EDA":
    feature_num = 29
elif dataset_name == "DBS":
    feature_num = 200
dummy_data = np.random.rand(130, feature_num)
dummy_data = pd.DataFrame(dummy_data, columns=[f"attr_{i}" for i in range(feature_num)])


# Initialize and train model
if dataset_name == "EDA":
    detector = DBAnomDector()
    detector.train(dataset_path="dataset/EDA/")
elif dataset_name == "DBS":
    detector = DBAnomDector(override_config=dbsherlock_config)
    detector.train(
        dataset_path="dataset/dbsherlock/converted/tpcc_500w_test.json",
        dataset_name="DBS",
    )

# Run inference (detect anomaly)
anomaly_score, is_anomaly, anomaly_cause = detector.infer(data=dummy_data)
```

Note that the dataset folder should be organized as follows:
```text
dataset
├── EDA
│   ├── meta_data
│   │   ├── db_backup.csv
│   │   ├── index.csv
│   │   ├── ...
│   │   └── workload_spike.csv
│   ├── raw_data
│   │   ├── db_backup_1.csv
│   │   ├── db_backup_2.csv
│   │   ├── ...
│   │   ├── workload_spike_1.csv
│   │   ├── workload_spike_2.csv
│   │   ├── ...
```

## Reference
This respository is based on [Anomaly Transformer](https://github.com/thuml/Anomaly-Transformer).

```
@inproceedings{
xu2022anomaly,
title={Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy},
author={Jiehui Xu and Haixu Wu and Jianmin Wang and Mingsheng Long},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=LzQQ89U1qm_}
}
```