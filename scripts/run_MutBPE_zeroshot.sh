#!/bin/bash

MODELS=(dnabert2)
METHODS=(base mutbpe)
BPEFROMMUTS=(False)
REVERSE_STRANDS=(True False)
USE_CLS_LIST=(False)
DISTANCE_METRICS=(cosine pearson spearman)
DOWNSTREAM_SAVE_DIR="output"

EXPERIMENTS=(
  "variant_effect_causal_eqtl"
)

for EXPERIMENT in "${EXPERIMENTS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
      for BPEFROMMUT in "${BPEFROMMUTS[@]}"; do
        for REVERSE_STRAND in "${REVERSE_STRANDS[@]}"; do
          for USE_CLS in "${USE_CLS_LIST[@]}"; do
            for DISTANCE_METRIC in "${DISTANCE_METRICS[@]}"; do
              echo "==== run: experiment=$EXPERIMENT, model=$MODEL, mutbpe_method=$METHOD, reverse_strand=$REVERSE_STRAND, use_CLS=$USE_CLS, BPEfromMut=$BPEFROMMUT, distance_metrics=$DISTANCE_METRIC ===="
              python BPE_zeroshot.py \
                --experiment $EXPERIMENT \
                --downstream_save_dir $DOWNSTREAM_SAVE_DIR \
                --model $MODEL \
                --mutbpe_method $METHOD \
                --distance_metrics $DISTANCE_METRIC \
                --reverse_strand $REVERSE_STRAND \
                --BPEfromMut $BPEFROMMUT \
                --use_CLS $USE_CLS
            done
          done
        done
      done
    done
  done
done