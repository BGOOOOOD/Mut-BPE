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
├── BPE_vep_rf.py               # Random Forest evaluation script
├── data/                       # Data directory
├── output/                     # Output directory for results
└── scripts/                    # Utility scripts
```

## Main Scripts

### 1. MutBPE_finetune.py
Fine-tunes DNA language models (DNABERT-2, MutBERT, etc.) using mutation-aware BPE tokenization.

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
    --use_CLS False \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --learning_rate 1e-4
```

### 2. MutBPE_vep_embedding.py
Extracts embeddings from fine-tuned models for downstream analysis.

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
    --BPEfromMut False \
    --reverse_strand False \
    --use_CLS False
```

### 3. BPE_zeroshot.py
Performs zero-shot evaluation using embedding distances.

**Key Features:**
- Calculates distances between reference and alternative embeddings
- Supports multiple distance metrics (cosine, L1, L2, Pearson, Spearman)
- Computes AUROC and AUPRC metrics
- No training required

**Usage:**
```bash
python BPE_zeroshot.py \
    --experiment variant_effect_pathogenic_ClinVar \
    --model dnabert2 \
    --mutbpe_method mutbpe \
    --BPEfromMut False \
    --reverse_strand False \
    --use_CLS False \
    --distance_metrics cosine
```

### 4. BPE_vep_rf.py
Evaluates embeddings using Random Forest classifier.

**Key Features:**
- Trains Random Forest on extracted embeddings
- Configurable hyperparameters (number of trees, tissue usage)
- Cross-validation support
- Performance evaluation with AUROC and AUPRC

**Usage:**
```bash
python BPE_vep_rf.py \
    --experiment variant_effect_pathogenic_ClinVar \
    --model dnabert2 \
    --mutbpe_method mutbpe \
    --BPEfromMut False \
    --reverse_strand False \
    --use_CLS False
```

## Key Parameters

### Model Parameters
- `--model_name_or_path`: Pre-trained model path
- `--seq_len`: Input sequence length
- `--window_size`: Window size for mutation analysis
- `--bp_per_token`: Base pairs per token

### MutBPE Parameters
- `--mutbpe_method`: BPE method ('base', 'mutbpe', or None)
- `--BPEfromMut`: Whether to perform BPE from mutation point
- `--mutbpe_pad_length`: Padding length for BPE tokenization

### Processing Parameters
- `--reverse_strand`: Use reverse strand sequences
- `--use_CLS`: Use CLS token for embeddings, default false

### Training Parameters
- `--num_train_epochs`: Number of training epochs
- `--per_device_train_batch_size`: Batch size per device
- `--learning_rate`: Learning rate
- `--logging_steps`: Logging frequency

## Supported Models

- **DNABERT-2**: `zhihan1996/DNABERT-2-117M`
- **MutBERT**: Various MutBERT variants
- **HyenaDNA**: Long-range DNA models
- **Nucleotide Transformer**: 6-mer based models
- **Gena-LM**: BPE models
- **Caduceus**: State space models

## Experiments

The project supports various variant effect prediction experiments:

- `variant_effect_causal_eQTL`: causal eQTL pathogenic variants
- `variant_effect_pathogenic_ClinVar`: ClinVar pathogenic variants
- `variant_effect_pathogenic_ClinVar_chr1`: Chromosome 1 variants
- `variant_effect_pathogenic_Cosmic_chr1`: COSMIC variants
- `variant_effect_pathogenic_Complex_chr1`: Complex variants
- `variant_effect_pathogenic_mendelian_chr11`: Mendelian variants

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
                └── vep_rf/
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

```bash
# Clone the repository
git clone <repository-url>
cd Mut-BPE

# Install dependencies
pip install -r requirements.txt
```