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
python src/data_factory/dbsherlock/convert.py \
    --input dataset/dbsherlock/tpcc_16w.mat \
    --out_dir dataset/dbsherlock/converted/ \
    --prefix tpcc_16w

python src/data_factory/dbsherlock/convert.py \
    --input dataset/dbsherlock/tpcc_500w.mat \
    --out_dir dataset/dbsherlock/converted/ \
    --prefix tpcc_500w

python src/data_factory/dbsherlock/convert.py \
    --input dataset/dbsherlock/tpce_3000.mat \
    --out_dir dataset/dbsherlock/converted/ \
    --prefix tpce_3000
```

Convert DBSherlock data into train & validate data for Anomaly Transformer:
```bash
python src/data_factory/dbsherlock/process.py \
    --input_path dataset/dbsherlock/converted/tpcc_16w_test.json \
    --output_path dataset/dbsherlock/processed/tpcc_16w/

python src/data_factory/dbsherlock/process.py \
    --input_path dataset/dbsherlock/converted/tpcc_500w_test.json \
    --output_path dataset/dbsherlock/processed/tpcc_500w/

python src/data_factory/dbsherlock/process.py \
    --input_path dataset/dbsherlock/converted/tpce_3000_test.json \
    --output_path dataset/dbsherlock/processed/tpce_3000/
```

## Train and Evaluate
We provide the experiment scripts under the folder `./scripts`. You can reproduce the experiment results with the below script:
```bash
bash ./scripts/experiment/DBS.sh
```
or you can run the below commands to train and evaluate the model step by step.

### Training
Train the model on DBSherlock dataset:
```bash
python main.py \
    --dataset EDA \
    --dataset_path dataset/EDA/ \
    --mode train
```

### Evaluating
Evaluate the trained model on the test split of the same dataset:
```bash
python main.py \
    --dataset EDA \
    --dataset_path dataset/EDA/ \
    --mode test 
```

### Inference
Perform inference on time series data with the trained model:
```bash
python main.py \
    --dataset EDA \
    --dataset_path dataset/EDA/ \
    --mode infer
    --output_path results/EDA/
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