import os
import csv
import json
import logging
from functools import partial
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union
from os import path as osp

import torch
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from datetime import datetime

from peft import LoraConfig, get_peft_model
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments as HfTrainingArguments,
    Trainer,
    EvalPrediction,
    DataCollatorWithPadding,
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# =====================================================================================
# Dataclasses for Arguments
# =====================================================================================

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="zhihan1996/DNA_bert_2", metadata={"help": "Path to pretrained model."})
    use_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA."})
    lora_r: int = field(default=8, metadata={"help": "LoRA r parameter."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha parameter."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_target_modules: str = field(default="query,value", metadata={"help": "LoRA target modules."})

@dataclass
class DataArguments:
    experiment: str = field(default='variant_effect_pathogenic_ClinVar', metadata={"help": "experiment: 'variant_effect_pathogenic_ClinVar', 'variant_effect_pathogenic_ClinVar_chr1', 'variant_effect_pathogenic_Cosmic_chr1', 'variant_effect_pathogenic_Complex_chr1', 'variant_effect_pathogenic_mendelian_chr11'"})
    cache_dir: str = field(default="./data_cache", metadata={"help": "Directory to cache raw data."})
    model: str = field(default="dnabert2", metadata={"help": "Model name."})
    seq_len: int = field(default=512, metadata={"help": "Sequence length."})
    bp_per_token: int = field(default=1, metadata={"help": "Base pairs per token."}) # For models like DNABERT
    window_size: int = field(default=1536, metadata={"help": "Window size for MutBPE."})
    mutbpe_method: Optional[str] = field(default=None, metadata={"help": "MutBPE method: 'mutbpe', 'base'"})
    reverse_strand: str = field(default="False", metadata={"help": "Use reverse strand sequences."})
    use_CLS: str = field(default="False", metadata={"help": "Whether to use CLS token."})
    mutbpe_pad_length: int = field(default=512, metadata={"help": "Padding length for MutBPE tokenization."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    factor: float = field(default=1.0)
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    logging_dir: str = field(default="runs")
    tb_name: str = field(default="run") 
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps")
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="auroc")
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)

# =====================================================================================
# Data Processing Logic (Preserved and Refactored)
# =====================================================================================

def string_reverse_complement(seq): # get the inverse dna seq
    """Reverse complement a DNA sequence."""
    STRING_COMPLEMENT_MAP = {
        "A": "T", "C": "G", "G": "C", "T": "A", "a": "t", "c": "g", "g": "c", "t": "a",
        "N": "N", "n": "n",
    }
    
    rev_comp = ""
    for base in seq[::-1]:
        if base in STRING_COMPLEMENT_MAP:
            rev_comp += STRING_COMPLEMENT_MAP[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp

def ids_to_tokens(dna_original_ids, tokenizer):
    """
    from dna_original_ids to generate dna_stable_tokens
    
    """
    
    dna_original_token = []
    for n in range(len(dna_original_ids)):
        tokens = tokenizer.convert_ids_to_tokens(dna_original_ids[n])
        if tokens not in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:

            #tokens = " ".join(tokens)
            #tokens = "".join(tokens)
            dna_original_token.append(tokens)
    return dna_original_token


def recast_chromosome_tissue_dist2TSS(examples): # extranct the seq related information
    """Recast chromosome to int."""
    return {
        "chromosome": -1 if examples["chromosome"] == "X" else int(examples["chromosome"]),
        "tissue": examples["tissue"],
        "distance_to_nearest_tss": examples["distance_to_nearest_tss"]
    }

def extract_window_seq(examples, data_args):
    """
    Extract window_size length fragments centered on variant_idx from ref and alt sequences.
    Supports batched calls.
    """
    half_window = data_args.window_size // 2
    ref_windows = []
    alt_windows = []
    for ref_seq, alt_seq, idx in zip(
        examples["ref_forward_sequence"], examples["alt_forward_sequence"], examples["variant_idx"]
    ):
        idx = int(idx)
        start = max(0, idx - half_window)
        end = min(idx + half_window + 1, len(ref_seq))
        ref_window = ref_seq[start:end]
        alt_window = alt_seq[start:end]
        ref_windows.append(ref_window)
        alt_windows.append(alt_window)
    return {
            "ref_window_seq": ref_windows,
            "alt_window_seq": alt_windows,
        }

def MutBPE(original_seq, data_args, examples, batch_idx):
    # Step 1: Merge list into complete string
    merged_str = ''.join(original_seq)
    if not merged_str:
        return original_seq
    
    # Step 2: Calculate middle index
    middle_index = len(merged_str) // 2
    
    # Step 3: Build index mapping table
    index_map = []
    current_start = 0
    for idx, sub in enumerate(original_seq):
        sub_length = len(sub)
        if sub_length == 0:
            index_map.append((current_start, current_start - 1))
            continue
        end = current_start + sub_length - 1
        index_map.append((current_start, end))
        current_start = end + 1
    
    # Step 4: Locate the sub-item containing the middle character
    target_sub = None
    target_idx = -1
    for idx, (start, end) in enumerate(index_map):
        if start <= middle_index <= end:
            target_sub = original_seq[idx]
            target_idx = idx
            break
    if target_sub is None:
        return original_seq
    
    # Step 5: Calculate split position within sub-item
    sub_start = index_map[target_idx][0]
    split_pos = middle_index - sub_start
    
    if data_args.mutbpe_method == 'mutbpe':
        idx = data_args.window_size // 2
        left_part = target_sub[:split_pos]
        ref_middle_char = target_sub[split_pos]
        alt_middle_char = examples["alt_window_seq"][batch_idx][idx]
        right_part = target_sub[split_pos + 1:]
        
        ref_parts = []
        if left_part != '':
            ref_parts.append(left_part)
        ref_parts.append(ref_middle_char)
        if right_part != '':
            ref_parts.append(right_part)

        alt_parts = []
        if left_part != '':
            alt_parts.append(left_part)
        alt_parts.append(alt_middle_char)
        if right_part != '':
            alt_parts.append(right_part)    

        return {
            "ref_tokens": original_seq[:target_idx] + ref_parts + original_seq[target_idx + 1:],
            "alt_tokens": original_seq[:target_idx] + alt_parts + original_seq[target_idx + 1:]
        }

    else:
        return original_seq
        
def tokenize_with_special_tokens_and_padding(ref_tokens, alt_tokens, tokenizer, pad_length=512):
    """
    Tokenize ref and alt sequences separately, manually add [CLS] and [SEP], then pad to specified length.
    ref_tokens/alt_tokens: List[List[int]]
    Returns: ref_input_ids, alt_input_ids, both List[List[int]], each sublist length=pad_length
    """
    
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    ref_input_ids = []
    alt_input_ids = []
    for tokens in ref_tokens:
        tokens = [cls_id] + tokens + [sep_id]
        tokens = tokens[:pad_length] + [pad_id] * max(0, pad_length - len(tokens))
        ref_input_ids.append(tokens)
    for tokens in alt_tokens:
        tokens = [cls_id] + tokens + [sep_id]
        tokens = tokens[:pad_length] + [pad_id] * max(0, pad_length - len(tokens))
        alt_input_ids.append(tokens)
    return ref_input_ids, alt_input_ids


def MutBPE_tokenize(examples, tokenizer, data_args): # read and tokenize the eqtl data and return the innput_ids
    """Tokenize sequence.
    Args:
        examples: (batch of) items from the dataset.
        tokenizer: AutoTokenizer.
        max_length: int.
    Returns:
        dict with values as list of token ids.
    """
    
    pad_len = data_args.mutbpe_pad_length
    # batch_encode_plus does not do padding and max_length
    ref_tokenized = tokenizer.batch_encode_plus(
        examples["ref_window_seq"],
        add_special_tokens=False,
        return_attention_mask=False,
        truncation=True,
        padding='do_not_pad'
    )
    alt_tokenized = tokenizer.batch_encode_plus(
        examples["alt_window_seq"],
        add_special_tokens=False,
        return_attention_mask=False,
        truncation=True,
        padding='do_not_pad'
    )
    ref_tokens = []
    alt_tokens = []
    for batch_idx, (ref, alt) in enumerate(zip(ref_tokenized["input_ids"], alt_tokenized["input_ids"])):
        if data_args.mutbpe_method == 'mutbpe':
            ref_tokens.append(tokenizer.convert_tokens_to_ids(MutBPE(tokenizer.convert_ids_to_tokens(ref), data_args, examples, batch_idx)["ref_tokens"]))
            alt_tokens.append(tokenizer.convert_tokens_to_ids(MutBPE(tokenizer.convert_ids_to_tokens(ref), data_args, examples, batch_idx)["alt_tokens"]))
        else:
            ref_tokens.append(tokenizer.convert_tokens_to_ids(MutBPE(tokenizer.convert_ids_to_tokens(ref), data_args, examples, batch_idx)))
            alt_tokens.append(tokenizer.convert_tokens_to_ids(MutBPE(tokenizer.convert_ids_to_tokens(alt), data_args, examples, batch_idx)))
    ref_input_ids, alt_input_ids = tokenize_with_special_tokens_and_padding(
        ref_tokens=ref_tokens, alt_tokens=alt_tokens, tokenizer=tokenizer, pad_length=pad_len
    )
    if data_args.reverse_strand == "True":
        ref_rc_tokenized = tokenizer.batch_encode_plus(
            [string_reverse_complement(seq) for seq in examples["ref_window_seq"]],
            add_special_tokens=False,
            return_attention_mask=False,
            truncation=True,
            padding='do_not_pad'
        )
        alt_rc_tokenized = tokenizer.batch_encode_plus(
            [string_reverse_complement(seq) for seq in examples["alt_window_seq"]],
            add_special_tokens=False,
            return_attention_mask=False,
            truncation=True,
            padding='do_not_pad'
        )
        ref_rc_tokens = []
        alt_rc_tokens = []
        for batch_idx, (ref, alt) in enumerate(zip(ref_rc_tokenized["input_ids"], alt_rc_tokenized["input_ids"])):
            if data_args.mutbpe_method == 'mutbpe':
                ref_rc_tokens.append(tokenizer.convert_tokens_to_ids(MutBPE(tokenizer.convert_ids_to_tokens(ref), data_args, examples, batch_idx)["ref_tokens"]))
                alt_rc_tokens.append(tokenizer.convert_tokens_to_ids(MutBPE(tokenizer.convert_ids_to_tokens(ref), data_args, examples, batch_idx)["alt_tokens"]))
            else:
                ref_rc_tokens.append(tokenizer.convert_tokens_to_ids(MutBPE(tokenizer.convert_ids_to_tokens(ref), data_args, examples, batch_idx)))
                alt_rc_tokens.append(tokenizer.convert_tokens_to_ids(MutBPE(tokenizer.convert_ids_to_tokens(alt), data_args, examples, batch_idx)))
        ref_rc_input_ids, alt_rc_input_ids = tokenize_with_special_tokens_and_padding(
            ref_tokens=ref_rc_tokens, alt_tokens=alt_rc_tokens, tokenizer=tokenizer, pad_length=pad_len
        )
        return {
            "ref_input_ids": ref_input_ids,
            "alt_input_ids": alt_input_ids,
            "ref_rc_input_ids": ref_rc_input_ids,
            "alt_rc_input_ids": alt_rc_input_ids,
        }
    else:
        return {
            "ref_input_ids": ref_input_ids,
            "alt_input_ids": alt_input_ids,
        }
     

def tokenize_variants(examples, tokenizer, max_length: int, data_args):
    """Tokenize sequence.

    Args:
        examples: (batch of) items from the dataset.
        tokenizer: AutoTokenizer.
    Returns:
        dict with values as list of token ids.
    """
    ref_tokenized = tokenizer.batch_encode_plus(
        examples["ref_forward_sequence"],
        add_special_tokens=False,
        return_attention_mask=False,
        max_length=max_length,
        truncation=True,
        padding='max_length',
    )
    alt_tokenized = tokenizer.batch_encode_plus(
        examples["alt_forward_sequence"],
        add_special_tokens=False,
        return_attention_mask=False,
        max_length=max_length,
        truncation=True,
        padding='max_length',
    )
    ref_tokenized_mutbert = tokenizer.batch_encode_plus(
            examples["ref_forward_sequence"],
            add_special_tokens=False,
            return_attention_mask=False,
            max_length=max_length,
            truncation=True,
        )
    alt_tokenized_mutbert = tokenizer.batch_encode_plus(
            examples["alt_forward_sequence"],
            add_special_tokens=False,
            return_attention_mask=False,
            max_length=max_length,
            truncation=True,
        )

    if data_args.reverse_strand == "True":  # if reverse strand is used, get the reverse complement seq
        # Reverse complement sequences
        ref_rc_tokenized = tokenizer.batch_encode_plus(
            [string_reverse_complement(seq) for seq in examples["ref_forward_sequence"]],
            add_special_tokens=False,
            return_attention_mask=False,
            max_length=max_length,
            truncation=True,
            padding='max_length'
        )
        alt_rc_tokenized = tokenizer.batch_encode_plus(
            [string_reverse_complement(seq) for seq in examples["alt_forward_sequence"]],
            add_special_tokens=False,
            return_attention_mask=False,
            max_length=max_length,
            truncation=True,
            padding='max_length'
        )
        ref_rc_tokenized_mutbert = tokenizer.batch_encode_plus(
            [string_reverse_complement(seq) for seq in examples["ref_forward_sequence"]],
            add_special_tokens=False,
            return_attention_mask=False,
            max_length=max_length,
            truncation=True,
        )
        alt_rc_tokenized_mutbert = tokenizer.batch_encode_plus(
            [string_reverse_complement(seq) for seq in examples["alt_forward_sequence"]],
            add_special_tokens=False,
            return_attention_mask=False,
            max_length=max_length,
            truncation=True,
        )

        if "MutBERT" in data_args.model:
            return {
                "ref_input_ids": ref_tokenized_mutbert["input_ids"],
                "alt_input_ids": alt_tokenized_mutbert["input_ids"],
                "ref_rc_input_ids": ref_rc_tokenized_mutbert["input_ids"],
                "alt_rc_input_ids": alt_rc_tokenized_mutbert["input_ids"],
            }
        return {
            "ref_input_ids": ref_tokenized["input_ids"],
            "alt_input_ids": alt_tokenized["input_ids"],
            "ref_rc_input_ids": ref_rc_tokenized["input_ids"],
            "alt_rc_input_ids": alt_rc_tokenized["input_ids"],
        }
    else:
        if "MutBERT" in data_args.model:
            return {
                "ref_input_ids": ref_tokenized_mutbert["input_ids"],
                "alt_input_ids": alt_tokenized_mutbert["input_ids"],
            }
        return {
            "ref_input_ids": ref_tokenized["input_ids"],
            "alt_input_ids": alt_tokenized["input_ids"],
        }


def find_bpe_variant_idx(examples, data_args):
    """Find token location that differs between reference and variant sequence.

    Args:
        examples: items from the dataset (not batched).
    Returns:
        dict with values index of difference.
    """
    # Guess that variant is at halfway point
    idx = data_args.seq_len // 2 if data_args.seq_len % 2 == 0 else (data_args.seq_len - 1) // 2
    if examples["ref_forward_sequence"][idx] == examples["alt_forward_sequence"][idx]:
        # If no, loop through sequence and find variant location
        idx = -1
        for i, (ref, alt) in enumerate(zip(examples["ref_forward_sequence"], examples["alt_forward_sequence"])):
            if ref != alt:
                idx = i
    # Same as above, but for reverse complement
    if data_args.reverse_strand == "True":
        rc_idx = data_args.seq_len // 2 if data_args.seq_len % 2 == 0 else (data_args.seq_len - 1) // 2
        if examples["ref_forward_sequence"][rc_idx] == examples["alt_forward_sequence"][rc_idx]:
            rc_idx = -1
            for i, (ref, alt) in enumerate(zip(examples["ref_forward_sequence"], examples["alt_forward_sequence"])):
                if ref != alt:
                    rc_idx = i
        return {"variant_idx": idx, "rc_variant_idx": rc_idx}
    else:
        return {"variant_idx": idx}

def find_variant_idx(examples, data_args):
    """Find token location that differs between reference and variant sequence.

    Args:
        examples: items from the dataset (batched).
    Returns:
        dict with values as lists of indices.
    """
    variant_indices = []
    rc_variant_indices = []
    
    for i in range(len(examples["ref_input_ids"])):
        # Guess that variant is at halfway point
        idx = len(examples["ref_input_ids"][i]) // 2
        if examples["ref_input_ids"][i][idx] == examples["alt_input_ids"][i][idx]:
            # If no, loop through sequence and find variant location
            idx = -1
            for j, (ref, alt) in enumerate(zip(examples["ref_input_ids"][i], examples["alt_input_ids"][i])):
                if ref != alt:
                    idx = j
                    break
        variant_indices.append(idx)
        
        # Same as above, but for reverse complement
        if data_args.reverse_strand == "True":
            rc_idx = len(examples["ref_rc_input_ids"][i]) // 2 - 1
            if examples["ref_rc_input_ids"][i][rc_idx] == examples["alt_rc_input_ids"][i][rc_idx]:
                rc_idx = -1
                for j, (ref, alt) in enumerate(zip(examples["ref_rc_input_ids"][i], examples["alt_rc_input_ids"][i])):
                    if ref != alt:
                        rc_idx = j
                        break
            rc_variant_indices.append(rc_idx)
    
    if data_args.reverse_strand == "True":
        return {"variant_idx": variant_indices, "rc_variant_idx": rc_variant_indices}
    else:
        return {"variant_idx": variant_indices}


def prepare_dataset(args, tokenizer, model_name_or_path):
    """Prepare or load the tokenized dataset."""
    # Data Preprocessing
    num_tokens = args.seq_len // args.bp_per_token

    # Load data
    cache_dir = osp.join(
        "data", f"{args.experiment}", f"seqlen={args.seq_len}"
        # "InstaDeepAI_genomics-long-range-benchmark"
    )
    if "nucleotide-transformer" in model_name_or_path:  # NT uses 6-mers, so tokenization is different
        preprocessed_cache_file = osp.join(cache_dir, "6mer_token_preprocessed", f'{args.model}', f'rs={args.reverse_strand}')

    elif "hyena" in model_name_or_path:
        preprocessed_cache_file = osp.join(cache_dir, "char_token_preprocessed", f'{args.model}', f'rs={args.reverse_strand}')
    
    elif "MutBERT" in model_name_or_path:
        preprocessed_cache_file = osp.join(cache_dir, "char_token_preprocessed", f'{args.model}', f'rs={args.reverse_strand}')

    else:
        preprocessed_cache_file = osp.join(
            cache_dir,
            f"bpe_token_preprocessed/{args.model}",
            f'method={args.mutbpe_method}', 
            f'rs={args.reverse_strand}'
        )
    print(f"Cache dir: {cache_dir}")
    print(f"Cache dir preprocessed: {preprocessed_cache_file}")

    if not os.path.exists(preprocessed_cache_file):
        os.makedirs(preprocessed_cache_file, exist_ok=True)
        print(f"Cache: yes")
        if args.experiment == 'variant_effect_pathogenic_ClinVar':
            dataset = load_dataset(
                "InstaDeepAI/genomics-long-range-benchmark",
                task_name="variant_effect_pathogenic_clinvar",
                sequence_length=args.seq_len,
                cache_dir=args.cache_dir,
                load_from_cache=False,
                force_download=True,
            )
        else:
            raise ValueError(f"Experiment {args.experiment} not supported")
        
        print("Dataset loaded. Cached to disk:")
        print(osp.dirname(list(dataset.cache_files.values())[0][0]["filename"]))
        try:
            del dataset["validation"]  # `validation` split is empty
        except KeyError:
            pass

        # Process data
        dataset = dataset.filter(
            lambda example: example["ref_forward_sequence"].count('N') < 0.005 * args.seq_len,
            desc="Filter N's"
        )
        # Check if dataset contains required columns
        if all(col in dataset.column_names for col in ["chromosome", "tissue", "distance_to_nearest_tss"]):
            dataset = dataset.map(
                recast_chromosome_tissue_dist2TSS,
                remove_columns=["chromosome", "tissue", "distance_to_nearest_tss"],
                desc="Recast chromosome"
            )

        if args.mutbpe_method == None: # if not BPE model
            dataset = dataset.map(
                partial(tokenize_variants, tokenizer=tokenizer, max_length=num_tokens, data_args=args),
                batch_size=1000,
                batched=True,
                #remove_columns=["ref_forward_sequence", "alt_forward_sequence"],
                desc="Tokenize"
            )
            dataset = dataset.map(find_variant_idx, batched=True, fn_kwargs={'data_args': args}, desc="Find variant idx") # find the variant index in the sequence
        
        else: # if BPE model
            print( "BPE model tokenization start" )
            dataset = dataset.map(partial(find_bpe_variant_idx, data_args=args), desc="Find variant idx") # find the variant index in the sequence
            dataset = dataset.map(partial(extract_window_seq, data_args=args), batched=True, desc="Extract window sequences") # extract the window sequence
            dataset = dataset.map(
                partial(MutBPE_tokenize, tokenizer=tokenizer, data_args=args),
                batch_size=1000,
                batched=True,
                #remove_columns=["ref_forward_sequence", "alt_forward_sequence"],
                desc="Tokenize with MutBPE"
            )
        
        dataset.save_to_disk(preprocessed_cache_file)

    dataset = load_from_disk(preprocessed_cache_file)
    print("Final dataset columns:", dataset.column_names)
    return dataset

# =====================================================================================
# Custom Data Collator for Sequence Pair Classification
# =====================================================================================
@dataclass
class VariantDataCollator:
    tokenizer: Any
    data_args: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.data_args.reverse_strand == "False":
            batch = {
                'labels': torch.tensor([f['label'] for f in features], dtype=torch.long),
                'ref_input_ids': torch.tensor([f['ref_input_ids'] for f in features], dtype=torch.long),
                'alt_input_ids': torch.tensor([f['alt_input_ids'] for f in features], dtype=torch.long),
            }
        else:
            batch = {
                'labels': torch.tensor([f['label'] for f in features], dtype=torch.long),
                'ref_input_ids': torch.tensor([f['ref_input_ids'] for f in features], dtype=torch.long),
                'alt_input_ids': torch.tensor([f['alt_input_ids'] for f in features], dtype=torch.long),
                'ref_rc_input_ids': torch.tensor([f['ref_rc_input_ids'] for f in features], dtype=torch.long),
                'alt_rc_input_ids': torch.tensor([f['alt_rc_input_ids'] for f in features], dtype=torch.long),
            }
        all_input_ids = []

        for i, feature in enumerate(features):  
            if getattr(self.data_args, "reverse_strand", "False") == "True":
                # Use both forward and reverse strands, concatenation format: [CLS] ref [SEP] ref_rc [SEP] alt [SEP] alt_rc [SEP]
                ref_ids = feature['ref_input_ids']
                ref_rc_ids = feature['ref_rc_input_ids']
                alt_ids = feature['alt_input_ids']
                alt_rc_ids = feature['alt_rc_input_ids']

                # Print length
                #print(f"[DEBUG] ref_ids len: {len(ref_ids)}, ref_rc_ids len: {len(ref_rc_ids)}, alt_ids len: {len(alt_ids)}, alt_rc_ids len: {len(alt_rc_ids)}")

                # Truncate to ensure not exceeding maximum length
                if "mutbert" in self.data_args.model.lower():
                    max_len_per_seq = self.data_args.window_size
                else:
                    max_len_per_seq = self.tokenizer.model_max_length // 4
                ref_ids = ref_ids[:max_len_per_seq]
                ref_rc_ids = ref_rc_ids[:max_len_per_seq]
                alt_ids = alt_ids[:max_len_per_seq]
                alt_rc_ids = alt_rc_ids[:max_len_per_seq]

                # Concatenate
                combined_ids = ref_ids + ref_rc_ids + alt_ids + alt_rc_ids
                all_input_ids.append(torch.tensor(combined_ids))
            else:
                # Only use forward strand, format: [CLS] ref [SEP] alt [SEP]
                ref_ids = feature['ref_input_ids']
                alt_ids = feature['alt_input_ids']

                # Print length
                #print(f"[DEBUG] ref_ids len: {len(ref_ids)}, alt_ids len: {len(alt_ids)}")

                # Truncate to ensure not exceeding maximum length
                if "mutbert" in self.data_args.model.lower():
                    max_len_per_seq = self.data_args.window_size
                else:
                    max_len_per_seq = self.tokenizer.model_max_length // 4
                ref_ids = ref_ids[:max_len_per_seq]
                alt_ids = alt_ids[:max_len_per_seq]

                combined_ids = ref_ids + alt_ids
                all_input_ids.append(torch.tensor(combined_ids))

        # padding
        padded = torch.nn.utils.rnn.pad_sequence(
            all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        # MutBERT model need to use one-hot encoding
        if "mutbert" in self.data_args.model.lower():
            #print("num_classes:", len(self.tokenizer))
            # Ensure padded values are within valid range
            padded = torch.clamp(padded, 0, len(self.tokenizer) - 1)
            # Create attention mask (before one-hot conversion)
            attention_mask = padded.ne(self.tokenizer.pad_token_id)
            # Convert to one-hot encoding
            padded = F.one_hot(padded, num_classes=len(self.tokenizer)).float()
            batch['attention_mask'] = attention_mask
        else:
            batch['attention_mask'] = padded.ne(self.tokenizer.pad_token_id)

        batch['input_ids'] = padded
        for k in ["ref_input_ids", "alt_input_ids", "ref_rc_input_ids", "alt_rc_input_ids"]:
            batch.pop(k, None)
        return batch

# =====================================================================================
# Metrics Calculation
# =====================================================================================

def compute_metrics(p: EvalPrediction) -> Dict:
    logits, labels = p
            # If logits is a tuple, take the first element
    if isinstance(logits, tuple):
        logits = logits[0]
    # print("logits type:", type(logits))
    # print("logits shape:", np.shape(logits))
    # print("logits example:", logits[:2])
    predictions = np.argmax(logits, axis=-1)
    acc = sklearn.metrics.accuracy_score(labels, predictions)
    mcc = sklearn.metrics.matthews_corrcoef(labels, predictions)
    f1 = sklearn.metrics.f1_score(labels, predictions, average="binary")
    try:
        probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
        auroc = sklearn.metrics.roc_auc_score(labels, probs[:, 1])
    except ValueError as e:
        logging.warning(f"Cannot calculate AUROC: {e}")
        auroc = 0.0
    
    try:
        auprc = sklearn.metrics.average_precision_score(labels, probs[:, 1])
    except ValueError as e:
        logging.warning(f"Cannot calculate AUPRC: {e}")
        auprc = 0.0
    
    return {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": acc,
        "eval_matthews_correlation": mcc,
        "f1": f1
    }

# =====================================================================================
# Main Training Function
# =====================================================================================

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(level=logging.INFO)

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=data_args.seq_len,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    # --- Prepare Datasets ---
    processed_dataset = prepare_dataset(data_args, tokenizer, model_args.model_name_or_path)
    train_dataset = processed_dataset["train"]
    print("train_dataset type:", type(train_dataset))
    print("train_dataset[0] keys:", train_dataset[0].keys())
    eval_dataset = processed_dataset["test"] # Using test set for evaluation

    # --- Model ---
    if "MutBERT" in model_args.model_name_or_path or "mutbert" in model_args.model_name_or_path.lower():
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            rope_scaling={'type': 'dynamic','factor': 4.0},
            num_labels=len(train_dataset.unique('label')),
            trust_remote_code=True,
        )
        print(f"Loaded MutBERT model with {len(train_dataset.unique('label'))} labels")
    elif "hyena" in model_args.model_name_or_path:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            num_labels=len(train_dataset.unique('label')),
            trust_remote_code=True,
        )
        print(f"Loaded HyenaDNA model with {len(train_dataset.unique('label'))} labels")
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            num_labels=len(train_dataset.unique('label')),
            trust_remote_code=True,
        )
    # --- LoRA ---
    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # --- Trainer ---
    custom_model_save_path = osp.join(
        training_args.output_dir, f'{training_args.run_name}',
        f'method={data_args.mutbpe_method}', 
        '_'.join([f'rs={data_args.reverse_strand}', f'use_CLS={data_args.use_CLS}']),
        f'ft_model_seq_{data_args.mutbpe_pad_length}'
    )
    os.makedirs(custom_model_save_path, exist_ok=True)

    # Modify training_args.output_dir
    training_args.output_dir = custom_model_save_path

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=VariantDataCollator(tokenizer, data_args),
        compute_metrics=compute_metrics,
    )
    
    # --- Train and Evaluate ---
    trainer.train()
    
    model.save_pretrained(custom_model_save_path)
    tokenizer.save_pretrained(custom_model_save_path)

    if training_args.eval_and_save_results:
        logging.info("Evaluating on test set...")
        results = trainer.evaluate(eval_dataset=eval_dataset)
        
        results_path = os.path.join(training_args.output_dir, 
                                    'finetune_results')
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, f"{training_args.run_name}_eval_results.json"), "w") as f:
            json.dump(results, f, indent=4)
        
        print("\nTest set results:")
        print(results)

if __name__ == "__main__":
    main()
