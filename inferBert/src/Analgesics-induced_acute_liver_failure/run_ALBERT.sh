# ALBERT_BASE_DIR= ../../dat/albert_base_zh
ALBERT_BASE_DIR = "https://tfhub.dev/google/albert_base/3"
DATA_DIR=../../dat/Analgesics-induced_acute_liver_failure/proc
OUTPUT_DIR=../../dat/Analgesics-induced_acute_liver_failure/output/
python ../albert/run_classifier.py \
    --data_dir=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --init_checkpoint=$ALBERT_BASE_DIR/model.ckpt-best \
    --albert_config_file=$ALBERT_BASE_DIR/albert_config.json \
    --vocab_file=$ALBERT_BASE_DIR/30k-clean.vocab \
    --spm_model_file=$ALBERT_BASE_DIR/30k-clean.model \
    --do_train \
    --do_eval \
    --do_predict \
    --do_lower_case \
    --max_seq_length=128 \
    --optimizer=adamw \
    --task_name=causal \
    --warmup_step=200 \
    --learning_rate=2e-5 \
    --train_step=30000 \
    --save_checkpoints_steps=3000 \
    --train_batch_size=32

# ALBERT_MODEL_HUB="https://tfhub.dev/google/albert_base/3"
# python -m ../albert/run_classifier.py \
#     --data_dir=$DATA_DIR \
#     --output_dir=$OUTPUT_DIR \
#     --albert_hub_module_handle=$ALBERT_MODEL_HUB \
#     --spm_model_file="from_tf_hub" \
#     --do_train \
#     --do_eval \
#     --do_predict \
#     --do_lower_case \
#     --max_seq_length=128 \
#     --optimizer=adamw \
#     --task_name=causal \
#     --warmup_step=200 \
#     --learning_rate=2e-5 \
#     --train_step=30000 \
#     --save_checkpoints_steps=3000 \
#     --train_batch_size=32
