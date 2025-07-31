#!/bin/bash

cuda=${1:-0}
export CUDA_VISIBLE_DEVICES=${cuda}

SEQ_LEN=2048
WINDOW_SIZE=1536
BP_PER_TOKEN=1
MODEL_NAME_OR_PATH="zhihan1996/DNABERT-2-117M"
MODEL="dnabert2"
MUTBPE_METHOD="base"
REVERSE_STRAND="False"
USE_CLS="False"
BATCH_SIZE=64
CACHE_DIR="./data"

for EXPERIMENT in "variant_effect_causal_eqtl"; do
    DOWNSTREAM_SAVE_DIR="output/$EXPERIMENT"
    echo "Processing experiment: $EXPERIMENT"
    echo "Output directory: $DOWNSTREAM_SAVE_DIR"
    
for USE_CLS in False
do
    for REVERSE_STRAND in False True
    do
        python MutBPE_vep_embedding.py \
            --seq_len $SEQ_LEN \
            --window_size $WINDOW_SIZE \
            --bp_per_token $BP_PER_TOKEN \
            --model_name_or_path $MODEL_NAME_OR_PATH \
            --downstream_save_dir $DOWNSTREAM_SAVE_DIR \
            --model $MODEL \
            --mutbpe_method $MUTBPE_METHOD \
            --distance_metrics $DISTANCE_METRICS \
            --reverse_strand $REVERSE_STRAND \
            --BPEfromMut $BPEFROMMUT \
            --use_CLS $USE_CLS \
            --embed_dump_batch_size $BATCH_SIZE \
            --cache_dir $CACHE_DIR \
            --experiment $EXPERIMENT
        done
    done
done