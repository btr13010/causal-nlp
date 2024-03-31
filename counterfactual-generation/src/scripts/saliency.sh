WORKING_DIR=/home/ubuntu/baotq4/causal-nlp/counterfactual-generation

MODEL_CONFIG=FacebookAI/roberta-large

TASK=nli

GPUs=1

for SPLIT in train dev
do
    TEST_PATH=${WORKING_DIR}/data/snli/ori_data/${SPLIT}.json

    # path to your fine-tuned classifier checkpint
    CKPT_PATH=${WORKING_DIR}/results/roberta-large/lightning_logs/version_3/checkpoints/epoch=1-step=626.ckpt

    OUTPUT_PATH=${WORKING_DIR}/data/snli/ori_data/${SPLIT}_saliency.json

    CUDA_VISIBLE_DEVICES=1,2 python nlu.py \
        --task $TASK \
        --saliency \
        --output_path $OUTPUT_PATH \
        --test_set $TEST_PATH \
        --load_dir $CKPT_PATH \
        --gpus $GPUs \
        --model_config $MODEL_CONFIG \
        --batch_size 64 \
        --saliency_mode gradient
done