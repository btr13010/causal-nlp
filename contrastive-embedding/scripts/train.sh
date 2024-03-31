WORKING_DIR=/home/ubuntu/baotq4/causal-nlp

MODEL_CONFIG=FacebookAI/roberta-large

TASK=nli

DATASET=snli

TRAIN_PATH=${WORKING_DIR}/counterfactual-generation/results/t5-large/lightning_logs/version_5/checkpoints/train_generate.json
DEV_PATH=${WORKING_DIR}/counterfactual-generation/results/t5-large/lightning_logs/version_5/checkpoints/dev_generate.json

# save_dir
SAVE_DIR=${WORKING_DIR}/contrastive-embedding/results/roberta-large-test

GPUs=1

echo "Using GPUs: [$GPUs]"
echo "TASK: $TASK"

CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false python src/train-bert-classifier.py \
    --task $TASK \
    --train_set $TRAIN_PATH \
    --dev_set $DEV_PATH \
    --save_dir $SAVE_DIR \
    --lr 1e-5 \
    --max_epochs 20 \
    --gpus $GPUs \
    --model_config $MODEL_CONFIG \
    --warm_up 0 \
    --weight_decay 0.1 \
    --batch_size 32 \