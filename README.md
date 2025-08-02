# Mut-BPE: Mutation-aware Byte Pair Encoding for DNA Sequence Analysis

## Overview

Mut-BPE is a mutation-aware Byte Pair Encoding (BPE) approach for DNA sequence analysis, specifically designed for variant effect prediction tasks. This project implements various methods for processing DNA sequences with mutations and evaluating their performance on downstream tasks.

## Project Structure

```
Mut-BPE/
├── README.md                    # This file
├── MutBPE_finetune.py          # Fine-tuning script for DNA models
├── MutBPE_vep_embedding.py     # Embedding extraction script
├── BPE_zeroshot.py             # Zero-shot evaluation script
└── scripts/                    # Utility scripts
```

## Main Scripts

### 1. MutBPE_finetune.py
Fine-tunes DNA language models (DNABERT-2, MutBERT, etc.) using mutation-aware BPE tokenization. Finetuned models will be uploaded to huggingface soon.

**Key Features:**
- Supports multiple DNA models (DNABERT-2, MutBERT, HyenaDNA, etc.)
- Implements mutation-aware BPE tokenization
- Configurable sequence length and window size
- Support for reverse strand sequences
- LoRA fine-tuning support

**Usage:**
```bash
python MutBPE_finetune.py \
    --model_name_or_path zhihan1996/DNABERT-2-117M \
    --experiment variant_effect_pathogenic_ClinVar \
    --seq_len 1024 \
    --mutbpe_pad_length 256 \
    --mutbpe_method mutbpe \
    --window_size 768 \
    --BPEfromMut False \
    --reverse_strand False \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --learning_rate 1e-4
```

### 2. MutBPE_vep_embedding.py
Extracts embeddings from fine-tuned models for downstream analysis. You need to generate embeddings first before doing the zero-shot.

**Key Features:**
- Extracts embeddings from reference and alternative sequences
- Supports both forward and reverse strand processing
- Configurable embedding extraction methods (CLS token vs. average pooling)
- Handles different model architectures

**Usage:**
```bash
python MutBPE_vep_embedding.py \
    --model_name_or_path zhihan1996/DNABERT-2-117M \
    --experiment variant_effect_pathogenic_ClinVar \
    --seq_len 1024 \
    --mutbpe_method mutbpe \
    --window_size 768 \
    --reverse_strand False \
    --use_CLS False
```

### 3. BPE_zeroshot.py
Performs zero-shot evaluation using embedding distances.

**Key Features:**
- Calculates distances between reference and alternative embeddings
- Supports multiple distance metrics (cosine, Pearson, Spearman)
- Computes AUROC and AUPRC metrics
- No training required

**Usage:**
```bash
python BPE_zeroshot.py \
    --experiment variant_effect_pathogenic_ClinVar \
    --model dnabert2 \
    --reverse_strand False \
    --use_CLS False \
    --distance_metrics cosine
```

## Experiments

The project supports various variant effect prediction experiments:

- `variant_effect_causal_eQTL`: causal eQTL pathogenic variants
- `variant_effect_pathogenic_ClinVar`: ClinVar pathogenic variants
- `variant_effect_pathogenic_ClinVar_chr1`: Chromosome 1 variants
- `variant_effect_pathogenic_Cosmic_chr1`: COSMIC variants
- `variant_effect_pathogenic_Complex_chr1`: Complex variants
- `variant_effect_pathogenic_mendelian_chr11`: Mendelian variants

`variant_effect_causal_eQTL` data is supoorted for embedding & zero-shot demo now and `variant_effect_pathogenic_ClinVar` data is supported for finetune demo now, if you want to use these data you need to put hg38 data under ./Mut-BPE/data/download.
Other experiment datasets are coming soon on huggingface.

## Output Structure

Results are organized in the following structure:
```
output/
└── {experiment}/
    └── {model}/
        └── method={mutbpe_method}/
            └── BPEfromMut={BPEfromMut}_rs={reverse_strand}_use_CLS={use_CLS}/
                ├── train_embeds.pt
                ├── test_embeds.pt
                ├── zero_shot/
                └── fintune_model_seq_{seq_len}
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- Transformers 4.20+
- Datasets
- Scikit-learn
- NumPy
- CUDA (for GPU acceleration)

## Installation

GPU is needed in embeddings generation and finetune.

```bash
# Clone the repository
git clone <repository-url>
cd Mut-BPE

# Install dependencies
pip install -r requirements.txt
```