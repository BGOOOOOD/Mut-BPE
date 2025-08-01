#!/bin/bash

cuda=${1:-0}
export CUDA_VISIBLE_DEVICES=${cuda}

mutbpe_methods=("base")
BPEfromMuts=("False")
reverse_strands=("False" "True")
use_CLSs=("False")
experiments=("variant_effect_pathogenic_ClinVar")

for experiment in "${experiments[@]}"; do
  output_dir="./output/$experiment"
  echo "==== process experiment: $experiment ===="
  echo "output dir: $output_dir"
  
  for mutbpe_method in "${mutbpe_methods[@]}"; do
      for reverse_strand in "${reverse_strands[@]}"; do
        for use_CLS in "${use_CLSs[@]}"; do
          echo "==== run: experiment=$experiment, mutbpe_method=$mutbpe_method, BPEfromMut=$BPEfromMut, reverse_strand=$reverse_strand, use_CLS=$use_CLS ===="
          python MutBPE_finetune.py \
            --model_name_or_path "zhihan1996/DNABERT-2-117M" \
            --experiment "$experiment" \
            --seq_len 1024 \
            --mutbpe_pad_length 256 \
            --mutbpe_method "$mutbpe_method" \
            --window_size 768 \
            --reverse_strand "$reverse_strand" \
            --use_CLS "$use_CLS" \
            --output_dir "$output_dir" \
            --run_name "dnabert2" \
            --num_train_epochs 3 \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 16 \
            --learning_rate 1e-4 \
            --logging_steps 50 \
            --eval_and_save_results True
        done
      done
    done
  done
done