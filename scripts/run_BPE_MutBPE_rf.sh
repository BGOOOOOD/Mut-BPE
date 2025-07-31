#!/bin/bash

MODELS=(dnabert2)
METHODS=(base mutbpe)
BPEFROMMUTS=(False)
REVERSE_STRANDS=(True False)
USE_CLS_LIST=(False)
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
            echo "==== run: experiment=$EXPERIMENT, model=$MODEL, mutbpe_method=$METHOD, reverse_strand=$REVERSE_STRAND, use_CLS=$USE_CLS, BPEfromMut=$BPEFROMMUT ===="
            python BPE_vep_rf.py \
              --model $MODEL \
              --mutbpe_method $METHOD \
              --experiment $EXPERIMENT \
              --BPEfromMut $BPEFROMMUT \
              --reverse_strand $REVERSE_STRAND \
              --use_CLS $USE_CLS \
              --downstream_save_dir $DOWNSTREAM_SAVE_DIR
          done
        done
      done
    done
  done
done
