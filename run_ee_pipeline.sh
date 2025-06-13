current_date=$(date +%Y%m%d)

current_dir=`pwd`
dataset="CHIP-CDEE"

DATA_DIR="${current_dir}/dataset/${dataset}"
task_name='cdee-pipeline'
stage="argument_embedding"

max_length=300
doc_stride=none

gpu_ids='2'
SEED=42
num_epochs=20
method="global-pointer"

PRETRAINED_MODEL_PATH=$current_dir/../pretrained_models/chinese_roberta_wwm_ext_pytorch

pretrained_model_name='roberta-wwm-med'

train_batch_size=8
eval_batch_size=16
learning_rate=2e-5

OUTPUT_DIR="${current_dir}/outputs/${task_name}/${stage}/${pretrained_model_name}_lr${learning_rate}_bs${train_batch_size}_maxlen${max_length}_stride${doc_stride}_${current_date}_seed${SEED}_fgm"


CUDA_VISIBLE_DEVICES=$gpu_ids python3 src/ee/run_ee_pipeline.py \
--task_name $task_name \
--data_dir $DATA_DIR \
--train_data_file $DATA_DIR/CHIP-CDEE_train.json \
--eval_data_file $DATA_DIR/CHIP-CDEE_dev.json \
--test_data_file $DATA_DIR/CHIP-CDEE_test.json \
--schema_file $DATA_DIR/schema.json \
--model_name_or_path $PRETRAINED_MODEL_PATH \
--output_dir $OUTPUT_DIR \
--max_length $max_length \
--num_train_epochs $num_epochs \
--per_device_train_batch_size $train_batch_size \
--per_device_eval_batch_size $eval_batch_size \
--learning_rate $learning_rate \
--do_train \
--do_eval \
--logging_steps 100 \
--evaluate_during_training \
--seed $SEED \
--do_lower_case \
--gpu_ids $gpu_ids \
--method $method \
--adv_type fgm