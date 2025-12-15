export CUDA_VISIBLE_DEVICES=0

python src/DBAnomTransformer/main.py --anormly_ratio 2 --num_epochs 10  --batch_size 256  --mode train --dataset DBS --dataset_path /root/Anomaly_Explanation/dataset/dbsherlock/converted/tpcc_16w_test.json --input_c 200 --output_c 200 --win_size 25 --step_size 25
python src/DBAnomTransformer/main.py --anormly_ratio 2 --num_epochs 10  --batch_size 256  --mode test  --dataset DBS --dataset_path /root/Anomaly_Explanation/dataset/dbsherlock/converted/tpcc_16w_test.json --input_c 200 --output_c 200 --win_size 25 --step_size 25 --pretrained_model 20
