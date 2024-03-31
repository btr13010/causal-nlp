WORKING_DIR=/home/ubuntu/baotq4/causal-nlp/counterfactual-generation

# model
MODEL_CONFIG=roberta-large

# data
TASK=nli

DATASET=snli

for SPLIT in train dev
do
    TEST_FILE=${WORKING_DIR}/data/${DATASET}/${SPLIT}.json
    PREDICT_FILE=${WORKING_DIR}/data/${DATASET}/${SPLIT}_predict.json

    # path to your fine-tuned classifier checkpint
    CKPT_PATH=${WORKING_DIR}/results/roberta-large/lightning_logs/version_3/checkpoints/epoch=1-step=626.ckpt

    CUDA_VISIBLE_DEVICES=1 python nlu.py \
        --task $TASK \
        --predict \
        --predict_file $PREDICT_FILE \
        --test_set $TEST_FILE \
        --load_dir $CKPT_PATH \
        --gpus 1 \
        --model_config $MODEL_CONFIG \
        --batch_size 1024
done