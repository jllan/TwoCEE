
current_dir=`pwd`
DATA_DIR="${current_dir}/dataset/CHIP-CDEE"

python3 src/ee/run_ee_pipeline_predict_cdee.py \
--task_name 'cdee-pipeline' \
--eval_data_file $DATA_DIR/CHIP-CDEE_dev.json \
--test_data_file $DATA_DIR/CHIP-CDEE_test.json \
--schema_file $DATA_DIR/schema.json \
--per_device_eval_batch_size 16 \
--gpu_ids '1'
