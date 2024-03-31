WORKING_DIR=/home/ubuntu/baotq4/causal-nlp/counterfactual-generation

MODEL_CONFIG=t5-large

TASK=nli

for SPLIT in train dev
do
    TEST_PATH=${WORKING_DIR}/data/snli/rationale_mask/${SPLIT}_cad.json

    # path to your fine-tuned classifier checkpint
    CKPT_PATH=${WORKING_DIR}/results/t5-large/lightning_logs/version_5/checkpoints/epoch=7-step=17407.ckpt
    SAVE_FILE=${SPLIT}_generate.json

    CUDA_VISIBLE_DEVICES=1 python generator.py \
        --task $TASK \
        --generate \
        --test_set $TEST_PATH \
        --load_dir $CKPT_PATH \
        --save_file $SAVE_FILE \
        --gpus 1 \
        --model_config $MODEL_CONFIG \
        --batch_size 512
done