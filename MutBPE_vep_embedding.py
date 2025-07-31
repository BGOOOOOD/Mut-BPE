from functools import partial
import argparse
from os import path as osp
import os
from typing import Dict, Iterable, Optional

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, DefaultDataCollator, AutoModel, AutoModelForMaskedLM
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn import preprocessing
from tqdm.auto  import tqdm

import numpy as np
from sklearn.metrics import roc_auc_score
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seq_len", type=int, default=2048,  # 2048
                        help="Sequence length (in bp)..")
    parser.add_argument("--window_size", type=int, default=1536,  # 2048
                        help="window size (in bp) for calculating statistics around the variant.")
    parser.add_argument("--bp_per_token", type=int, default=1,
                        help="Number of base pairs per token.")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--downstream_save_dir", type=str, default="output",
                        help="Directory to save downstream task.")
    parser.add_argument("--embed_dump_batch_size", type=int, default=1,
                        help="Batch size for embedding dump.")
    parser.add_argument("--model", type=str, default=None, choices=['dnabert2', 'mutbert', 'NT', 'GROVER', 'gena-lm', 'caduceus'], help="Embeddings model name.")
    parser.add_argument("--mutbpe_method", type=str, default=None, choices=['base', 'mutbpe'], help="MutBPE Methods.")
    parser.add_argument("--reverse_strand", type=str, choices=["True", "False"], default="False", help="Use reverse strand sequences.")
    parser.add_argument("--use_CLS", type=str, choices=["True", "False"], default="False", help="Use CLS token for classification.")
    parser.add_argument("--experiment", type=str, default='variant_effect_pathogenic_ClinVar', choices=['variant_effect_pathogenic_ClinVar', 
    'variant_effect_pathogenic_ClinVar_chr1', 'variant_effect_pathogenic_Cosmic_chr1', 'variant_effect_pathogenic_Complex_chr1', 'variant_effect_pathogenic_mendelian_chr11'], help="experiment name.")
    parser.add_argument("--cache_dir", type=str, default='/hpc2hdd/home/yxu662/jhupload/MutBPE/data')
    parser.add_argument("--mutbpe_pad_length", type=int, default=512, help="Padding length for MutBPE tokenization.")
    args = parser.parse_args()
    return args



class DNAEmbeddingModel(nn.Module): # get the final embeddings of the input
    """Wrapper around HF model.

    Args:
        model_name_or_path: str, path to HF model.
    """
    def __init__(
            self,
            model_name_or_path: str,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        if "MutBERT" in model_name_or_path:
            self.backbone = AutoModel.from_pretrained(model_name_or_path,
                                                      trust_remote_code=True,
                                                      rope_scaling={'type': 'dynamic','factor': 4.0}
                                                      )
        elif "nucleotide-transformer" in model_name_or_path:
            # NT LM `backbone` is under the `.esm` attribute
            self.backbone = AutoModelForMaskedLM.from_pretrained(model_name_or_path, trust_remote_code=True).esm
        elif "GROVER" in model_name_or_path:
            self.backbone = AutoModelForMaskedLM.from_pretrained(model_name_or_path, trust_remote_code=True)
        elif "caduceus" in model_name_or_path:
            self.backbone = AutoModelForMaskedLM.from_pretrained(model_name_or_path, trust_remote_code=True)
        else:
            self.backbone = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)

    def forward(self, input_ids):
        """Backbone forward pass to retrieve last_hidden_state."""
        if "DNABERT" in self.model_name_or_path:
            return self.backbone(input_ids)[0]
        elif "gena-lm" in self.model_name_or_path:
            return self.backbone(input_ids)[0]
        elif "GROVER" in self.model_name_or_path:
            return self.backbone(input_ids)[0]

        else:
            return self.backbone(input_ids).last_hidden_state
    


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
    # Compatible with int input
    if isinstance(dna_original_ids, int):
        dna_original_ids = [dna_original_ids]
    dna_original_token = []
    for n in range(len(dna_original_ids)):
        tokens = tokenizer.convert_ids_to_tokens(dna_original_ids[n])
        if tokens not in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            dna_original_token.append(tokens)
    return dna_original_token


def recast_chromosome_tissue_dist2TSS(examples): # extranct the seq related information
    """Recast chromosome to int."""
    return {
        "chromosome": -1 if examples["chromosome"] == "X" else int(examples["chromosome"]),
        "tissue": examples["tissue"],
        "distance_to_nearest_tss": examples["distance_to_nearest_tss"]
    }


def extract_window_seq(examples):
    """
    Extract window_size length fragments centered on variant_idx from ref and alt sequences.
    Supports batched calls.
    """
    half_window = args.window_size // 2
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


def MutBPE(original_seq, examples,batch_idx):
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
    
    if args.mutbpe_method == 'mutbpe':
        idx = args.window_size // 2
        #print(f"idx: {idx}", f'len(examples["alt_window_seq"]): {len(examples["alt_window_seq"][batch_idx])}')
        left_part = target_sub[:split_pos]
        ref_middle_char = target_sub[split_pos]
        alt_middle_char = examples["alt_window_seq"][batch_idx][idx]
        right_part = target_sub[split_pos + 1:]
    
        # Step 7: Build parts
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
        print("="*60)
        print(f"mutbpe_method: {args.mutbpe_method}")
        print("="*60)
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


def MutBPE_tokenize(examples, tokenizer): # read and tokenize the eqtl data and return the innput_ids
    """Tokenize sequence.
    Args:
        examples: (batch of) items from the dataset.
        tokenizer: AutoTokenizer.
        max_length: int.
    Returns:
        dict with values as list of token ids.
    """
    pad_len = args.mutbpe_pad_length
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
        if args.mutbpe_method == 'mutbpe':
            ref_tokens.append(tokenizer.convert_tokens_to_ids(MutBPE(tokenizer.convert_ids_to_tokens(ref), examples, batch_idx)["ref_tokens"]))
            alt_tokens.append(tokenizer.convert_tokens_to_ids(MutBPE(tokenizer.convert_ids_to_tokens(ref), examples, batch_idx)["alt_tokens"]))
        else:
            ref_tokens.append(tokenizer.convert_tokens_to_ids(MutBPE(tokenizer.convert_ids_to_tokens(ref), examples, batch_idx)))
            alt_tokens.append(tokenizer.convert_tokens_to_ids(MutBPE(tokenizer.convert_ids_to_tokens(alt), examples, batch_idx)))
    ref_input_ids, alt_input_ids = tokenize_with_special_tokens_and_padding(
        ref_tokens=ref_tokens, alt_tokens=alt_tokens, tokenizer=tokenizer, pad_length=pad_len
    )
    if args.reverse_strand == "True":
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
            if args.mutbpe_method == 'mutbpe':
                ref_rc_tokens.append(tokenizer.convert_tokens_to_ids(MutBPE(tokenizer.convert_ids_to_tokens(ref), examples, batch_idx)["ref_tokens"]))
                alt_rc_tokens.append(tokenizer.convert_tokens_to_ids(MutBPE(tokenizer.convert_ids_to_tokens(ref), examples, batch_idx)["alt_tokens"]))
            else:
                ref_rc_tokens.append(tokenizer.convert_tokens_to_ids(MutBPE(tokenizer.convert_ids_to_tokens(ref), examples, batch_idx)))
                alt_rc_tokens.append(tokenizer.convert_tokens_to_ids(MutBPE(tokenizer.convert_ids_to_tokens(alt), examples, batch_idx)))
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


def tokenize_variants(examples, tokenizer, max_length: int):
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

    if args.reverse_strand == "True":  # if reverse strand is used, get the reverse complement seq
        # Reverse complement sequences
        ref_rc_tokenized = tokenizer.batch_encode_plus(
            [string_reverse_complement(seq) for seq in examples["ref_forward_sequence"]],
            add_special_tokens=False,
            return_attention_mask=False,
            max_length=max_length,
            truncation=True,
            padding='max_length',
        )
        alt_rc_tokenized = tokenizer.batch_encode_plus(
            [string_reverse_complement(seq) for seq in examples["alt_forward_sequence"]],
            add_special_tokens=False,
            return_attention_mask=False,
            max_length=max_length,
            truncation=True,
            padding='max_length',
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
        if "MutBERT" in args.model_name_or_path:
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
        if "MutBERT" in args.model_name_or_path:
            return {
                "ref_input_ids": ref_tokenized_mutbert["input_ids"],
                "alt_input_ids": alt_tokenized_mutbert["input_ids"],
            }
        return {
            "ref_input_ids": ref_tokenized["input_ids"],
            "alt_input_ids": alt_tokenized["input_ids"],
        }


def find_bpe_variant_idx(examples):
    """Find token location that differs between reference and variant sequence.

    Args:
        examples: items from the dataset (not batched).
    Returns:
        dict with values index of difference.
    """
    # Guess that variant is at halfway point
    idx = args.seq_len // 2 if args.seq_len % 2 == 0 else (args.seq_len - 1) // 2
    if examples["ref_forward_sequence"][idx] == examples["alt_forward_sequence"][idx]:
        # If no, loop through sequence and find variant location
        idx = -1
        for i, (ref, alt) in enumerate(zip(examples["ref_forward_sequence"], examples["alt_forward_sequence"])):
            if ref != alt:
                idx = i
    # Same as above, but for reverse complement
    if args.reverse_strand == "True":
        rc_idx = args.seq_len // 2 if args.seq_len % 2 == 0 else (args.seq_len - 1) // 2
        if examples["ref_forward_sequence"][rc_idx] == examples["alt_forward_sequence"][rc_idx]:
            rc_idx = -1
            for i, (ref, alt) in enumerate(zip(examples["ref_forward_sequence"], examples["alt_forward_sequence"])):
                if ref != alt:
                    rc_idx = i
        return {"variant_idx": idx, "rc_variant_idx": rc_idx}
    else:
        return {"variant_idx": idx}


def find_variant_idx(examples):
    """Find token location that differs between reference and variant sequence.

    Args:
        examples: items from the dataset (not batched).
    Returns:
        dict with values index of difference.
    """
    # Guess that variant is at halfway point
    idx = len(examples["ref_input_ids"]) // 2
    if examples["ref_input_ids"][idx] == examples["alt_input_ids"][idx]:
        # If no, loop through sequence and find variant location
        idx = -1
        for i, (ref, alt) in enumerate(zip(examples["ref_input_ids"], examples["alt_input_ids"])):
            if ref != alt:
                idx = i
    # Same as above, but for reverse complement
    if args.reverse_strand == "True":
         #Guess that variant is at halfway point
        rc_idx = len(examples["ref_rc_input_ids"]) // 2 - 1
        if examples["ref_rc_input_ids"][rc_idx] == examples["alt_rc_input_ids"][rc_idx]:
             rc_idx = -1
             for i, (ref, alt) in enumerate(zip(examples["ref_rc_input_ids"], examples["alt_rc_input_ids"])):
                 if ref != alt:
                     rc_idx = i
        return {"variant_idx": idx, "rc_variant_idx": rc_idx}
    else:
        return {"variant_idx": idx}


def prepare_dataset(args, tokenizer): # get the preprocessed dataset with ref and alt ids, bio info and variant ids
    """Prepare or load the tokenized dataset."""
    # Data Preprocessing
    num_tokens = args.seq_len // args.bp_per_token

    # Load data
    cache_dir = osp.join(
        "data", f"{args.experiment}", f"seqlen={args.seq_len}"
        # "InstaDeepAI_genomics-long-range-benchmark"
    )
    if "nucleotide-transformer" in args.model_name_or_path:  # NT uses 6-mers, so tokenization is different
        preprocessed_cache_file = osp.join(cache_dir, "6mer_token_preprocessed", f'{args.model}', f'rs={args.reverse_strand}')

    elif "hyena" in args.model_name_or_path:
        preprocessed_cache_file = osp.join(cache_dir, "char_token_preprocessed", f'{args.model}', f'rs={args.reverse_strand}')
    
    elif "MutBERT" in args.model_name_or_path:
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
        if args.experiment == 'variant_effect_causal_eqtl':
            dataset = load_dataset(
                "InstaDeepAI/genomics-long-range-benchmark",
                task_name="variant_effect_causal_eqtl",  # variant_effect_causal_eqtl
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
            print("Tokenizing variants with max_length: ")
            dataset = dataset.map(
                partial(tokenize_variants, tokenizer=tokenizer, max_length=num_tokens),
                batch_size=1000,
                batched=True,
                #remove_columns=["ref_forward_sequence", "alt_forward_sequence"],
                desc="Tokenize"
            )
            dataset = dataset.map(find_variant_idx, desc="Find variant idx") # find the variant index in the sequence
        
        else: # if BPE model
            dataset = dataset.map(find_bpe_variant_idx, desc="Find variant idx") # find the variant index in the sequence
            dataset = dataset.map(partial(extract_window_seq), batched=True, desc="Extract window sequences") # extract the window sequence
            dataset = dataset.map(
                partial(MutBPE_tokenize, tokenizer=tokenizer),
                batch_size=1000,
                batched=True,
                #remove_columns=["ref_forward_sequence", "alt_forward_sequence"],
                desc="Tokenize with MutBPE"
            )
            if args.model_name_or_path == "PoetschLab/GROVER":
                # For GROVER model, directly delete samples containing None values to avoid modifying token ID
                print("GROVER model detected, will delete samples containing None values...")
                dataset = dataset.filter(
                    lambda example: not any(
                        example.get(key, []) is not None and 
                        any(x is None for x in example.get(key, []))
                        for key in ["ref_input_ids", "alt_input_ids", "ref_rc_input_ids", "alt_rc_input_ids"]
                    ),
                    desc="Filter samples with None values for GROVER"
                )
                print(f"Dataset size after filtering: {len(dataset)}")
        
        dataset.save_to_disk(preprocessed_cache_file)

    dataset = load_from_disk(preprocessed_cache_file)
    print("Final dataset columns:", dataset.column_names)
    return dataset


def concat_storage_dict_values(storage_dict):
    """Helper method that combines lists of tensors in storage_dict into a single torch.Tensor."""
    return {key: torch.cat(storage_dict[key], dim=0) for key in storage_dict.keys()}


def generate_embeddings(args, dataset, model, device, tokenizer):
    """Dump embeddings to disk."""
    def extract_embeddings(item_ref, item_alt, variant_idx):
        """Extract embedding representation from last layer outputs

        Args:
            item_ref: torch.Tensor, shape (batch_size, seq_len, hidden_size) Ref embedding
            item_alt: torch.Tensor, shape (batch_size, seq_len, hidden_size) Alt embedding
            variant_idx: torch.Tensor, shape (batch_size,) Index of variant
        Returns:
            layer_metrics: dict, with values to save to disk
        """
        layer_metrics = {}

        # Compute windowed statistics
        if "enformer" in args.model_name_or_path.lower():
            window_size = args.window_size // 128  # Enformer's receptive field is 128
            # We also need to override variant_idx since Enformer model reduces to target_length of 896
            variant_idx = torch.ones_like(variant_idx) * item_ref.size(1) // 2
        else:
            window_size = args.window_size // args.bp_per_token

        # Add 1 so that window is: [window // 2 - SNP - window // 2]
        start, end = -window_size // 2, window_size // 2 + 1
        expanded_indices = torch.arange(start, end, device=item_ref.device).unsqueeze(0) + \
                           variant_idx.unsqueeze(1).to(item_ref.device) # [1, window_size + 1] + [batch_size, 1] = [batch_size, window_size + 1]
        expanded_indices = torch.clamp(expanded_indices, 0, item_ref.size(1) - 1)  # Handle boundary conditions [batch_size, window_size + 1]
        tokens_window_ref = torch.gather(
            item_ref, 1,
            expanded_indices.unsqueeze(-1).expand(-1, -1, item_ref.size(2))
        ).mean(dim=1) # [batch_size, seq_len, hidden_size] -> [batch_size, window_size, hidden_size] -> [batch_size, hidden_size]
        tokens_window_alt = torch.gather(
            item_alt, 1,
            expanded_indices.unsqueeze(-1).expand(-1, -1, item_ref.size(2))
        ).mean(dim=1)
        layer_metrics["concat_avg_ws"] = torch.cat([tokens_window_ref, tokens_window_alt], dim=-1) # [batch_size, hidden_size * 2]
        return layer_metrics # [batch_size, hidden_size * 2]

    def get_bpe_embedding(item_ref, item_alt, input_ids_ref, input_ids_alt, tokenizer, use_cls):
        """
        Extract BPE embedding representation from last layer outputs.
        It can either take the [CLS] token embedding or the average of all other tokens.
        """
        layer_metrics = {}
        if use_cls == "True":
            # Use the embedding of the [CLS] token.
            # Shape: [batch_size, hidden_size]
            cls_ref = item_ref[:, 0, :]
            cls_alt = item_alt[:, 0, :]
            # Concatenate to get [batch_size, hidden_size * 2]
            layer_metrics["concat_cls"] = torch.cat([cls_ref, cls_alt], dim=-1)
        else:
            # Average the embeddings of all tokens except [CLS], [SEP], and [PAD].
            cls_id = tokenizer.cls_token_id
            sep_id = tokenizer.sep_token_id
            pad_id = tokenizer.pad_token_id

            def mask_and_avg(item, input_ids):
                # Create a mask to exclude special tokens.
                mask = (input_ids != cls_id) & (input_ids != sep_id) & (input_ids != pad_id)
                mask = mask.to(item.device)  # Ensure mask and item are on the same device
                mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1)
                mask_expanded = mask.unsqueeze(-1).expand_as(item)
                item_masked = item * mask_expanded
                summed = item_masked.sum(dim=1)
                avg = summed / mask_sum
                return avg

            ref_avg = mask_and_avg(item_ref, input_ids_ref)
            alt_avg = mask_and_avg(item_alt, input_ids_alt)

            # Concatenate to get [batch_size, hidden_size * 2]
            layer_metrics["concat_avg_ws"] = torch.cat([ref_avg, alt_avg], dim=-1)
        return layer_metrics


    embeds_path = osp.join(args.downstream_save_dir, args.model, f'method={args.mutbpe_method}', '_'.join([f'rs={args.reverse_strand}', f'use_CLS={args.use_CLS}']))
    os.makedirs(embeds_path, exist_ok=True)

    dataloader_params = {
        "batch_size": args.embed_dump_batch_size,
        "collate_fn": DefaultDataCollator(return_tensors="pt"),
        "num_workers": 0,
        "pin_memory": False,
        "shuffle": False,
        "drop_last": False  # True
    }

    # Process label_encoder = preprocessing.LabelEncoder()
    if all(col in dataset.column_names for col in ["tissue"]):  
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(dataset["test"]["tissue"])
        train_tissue_embed = label_encoder.transform(dataset["train"]["tissue"])
        dataset["train"] = dataset["train"].add_column("tissue_embed", train_tissue_embed)
        test_tissue_embed = label_encoder.transform(dataset["test"]["tissue"])
        dataset["test"] = dataset["test"].add_column("tissue_embed", test_tissue_embed) # add the embedded tissue to dataset

    
    for split_name, split in dataset.items():
        #print(split["ref_input_ids"][0], split["alt_input_ids"][0])
        
        dl = DataLoader(split, **dataloader_params)

        # Dynamically initialize storage_dict to ensure keys match get_bpe_embedding return
        if args.use_CLS == "True" and args.reverse_strand == "True":    # If using CLS and reverse strand
            storage_dict = {
                "concat_cls": [],
                "rc_concat_cls": [],
                "chromosome": [],
                "labels": [],
                "distance_to_nearest_tss": [],
                "tissue_embed": [],
            }
        elif args.use_CLS == "True" and args.reverse_strand == "False":    # If using CLS but not reverse strand
            storage_dict = {
                "concat_cls": [],
                "chromosome": [],
                "labels": [],
                "distance_to_nearest_tss": [],
                "tissue_embed": [],
            }
        elif args.use_CLS == "False" and args.reverse_strand == "True":    # If using avg and reverse strand
            storage_dict = {
                "concat_avg_ws": [],
                "rc_concat_avg_ws": [],
                "chromosome": [],
                "labels": [],
                "distance_to_nearest_tss": [],
                "tissue_embed": [],
            }
        else:
            storage_dict = {
                "concat_avg_ws": [],
                "chromosome": [],
                "labels": [],
                "distance_to_nearest_tss": [],
                "tissue_embed": [],
            }
        print(f'max_length: {args.mutbpe_pad_length}')

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dl), total=len(dl), desc=f"Embedding {split_name}"
            ):
                # Add tissue_embed to storage_dict
                for key in ["chromosome", "labels", "distance_to_nearest_tss", "tissue_embed"]:
                    if key in batch:
                        storage_dict[key].append(batch[key].to("cpu", non_blocking=True))

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    if "MutBERT" in args.model_name_or_path:
                        bs_alt_inputs = F.one_hot(batch["alt_input_ids"], num_classes=9).float().to(device)
                        bs_ref_inputs = F.one_hot(batch["ref_input_ids"], num_classes=9).float().to(device)
                        if args.reverse_strand == "True":
                            batch["alt_rc_input_ids"] = F.one_hot(batch["alt_rc_input_ids"], num_classes=9).float().to(device)
                            batch["ref_rc_input_ids"] = F.one_hot(batch["ref_rc_input_ids"], num_classes=9).float().to(device)
                    else:
                        bs_alt_inputs = batch["alt_input_ids"].to(device)
                        bs_ref_inputs = batch["ref_input_ids"].to(device)
                        # output_alt = model(batch["alt_input_ids"].to(device))
                        # output_ref = model(batch["ref_input_ids"].to(device))
                    output_alt = model(bs_alt_inputs)
                    output_ref = model(bs_ref_inputs)

                    if args.reverse_strand == "True":
                        output_alt_rc = model(batch["alt_rc_input_ids"].to(device)).contiguous().flip(dims=[1])
                        output_ref_rc = model(batch["ref_rc_input_ids"].to(device)).contiguous().flip(dims=[1])

                if args.mutbpe_method == None: # if not BPE model
                    metrics = extract_embeddings(
                        item_ref=output_ref,
                        item_alt=output_alt,
                        variant_idx=batch["variant_idx"],
                        )
                    for key, value in metrics.items():
                        storage_dict[key].append(value.to("cpu", non_blocking=True))
                    if args.reverse_strand == "True":
                        metrics_rc = extract_embeddings(
                            item_ref=output_ref_rc,
                            item_alt=output_alt_rc,
                            variant_idx=batch["variant_idx"],
                        )
                        for key, value in metrics_rc.items(): 
                            storage_dict[f"rc_{key}"].append(metrics_rc[key].to("cpu", non_blocking=True))
                
                else:
                    metrics = get_bpe_embedding(
                        item_ref=output_ref,
                        item_alt=output_alt,
                        input_ids_ref=batch["ref_input_ids"],
                        input_ids_alt=batch["alt_input_ids"],
                        tokenizer= tokenizer,
                        use_cls=args.use_CLS,
                    )
                    for key, value in metrics.items():
                        storage_dict[key].append(value.to("cpu", non_blocking=True))
                        
                    if args.reverse_strand == "True":
                        metrics_rc = get_bpe_embedding(
                            item_ref=output_ref_rc,
                            item_alt=output_alt_rc,
                            input_ids_ref=batch["ref_rc_input_ids"],
                            input_ids_alt=batch["alt_rc_input_ids"],
                            tokenizer=tokenizer,
                            use_cls=args.use_CLS,
                        )
                        for key, value in metrics_rc.items():
                            storage_dict[f"rc_{key}"].append(metrics_rc[key].to("cpu", non_blocking=True))

            # Check if storage_dict has empty keys, skip if any
            non_empty_storage_dict = {}
            for key, value_list in storage_dict.items():
                if value_list and len(value_list) > 0:  # Check if list is not empty
                    non_empty_storage_dict[key] = value_list
            
            storage_dict_temp = concat_storage_dict_values(non_empty_storage_dict)
            # if reverse strand embedding exists, concatenate with forward strand embedding
            rc_avg_ws_exists = "rc_concat_avg_ws" in storage_dict_temp and storage_dict_temp["rc_concat_avg_ws"].numel() > 0
            rc_cls_exists = "rc_concat_cls" in storage_dict_temp and storage_dict_temp["rc_concat_cls"].numel() > 0

            if rc_avg_ws_exists or rc_cls_exists:
                storage_dict_temp["concat_avg_ws_both"] = torch.cat(
                    [storage_dict_temp["concat_avg_ws"], storage_dict_temp["rc_concat_avg_ws"]], dim=-1
                ) if args.use_CLS == "False" else torch.cat(
                    [storage_dict_temp["concat_cls"], storage_dict_temp["rc_concat_cls"]], dim=-1
                )
            torch.save(storage_dict_temp, osp.join(embeds_path, f"{split_name}_embeds.pt"))
            print(f"Saved {split_name} embeddings to {osp.join(embeds_path, f'{split_name}_embeds.pt')}")


def main(args):
    """Main entry point."""
    # Setup device
    device = torch.device("cuda")

    print("="*60)
    print(f"*** Current parameter settings: method={args.mutbpe_method}, reverse_strand={args.reverse_strand}, use_CLS={args.use_CLS} ***")
    print("="*60)

    # Init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              model_max_length=args.mutbpe_pad_length,
                                              trust_remote_code=True)
    
    # Set special tokens for MutBERT model
    if "MutBERT" in args.model_name_or_path:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.sep_token is None:
            tokenizer.sep_token = tokenizer.eos_token
        if tokenizer.cls_token is None:
            tokenizer.cls_token = tokenizer.bos_token if hasattr(tokenizer, 'bos_token') else tokenizer.eos_token
        print(f"MutBERT tokenizer special tokens: pad={tokenizer.pad_token}, sep={tokenizer.sep_token}, cls={tokenizer.cls_token}")

    # Get dataset
    dataset = prepare_dataset(args, tokenizer)
    print(dataset['train'].column_names)
    # Get model
    try:
        model = DNAEmbeddingModel(args.model_name_or_path).to(device)
        model = torch.nn.DataParallel(model)
        model.eval()
        print("Model loaded successfully.")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUBLAS_STATUS_NOT_INITIALIZED" in str(e):
            print(f"GPU error: {e}")
            print("Trying to use CPU instead...")
            device = torch.device("cpu")
            torch.cuda.empty_cache()
            model = DNAEmbeddingModel(args.model_name_or_path).to(device)
            model.eval()
            print("Model loaded on CPU.")
        else:
            raise e

    # Dump embeddings
    generate_embeddings(args, dataset, model, device, tokenizer)

if __name__ == "__main__":
    print(f'torch.cuda.is_available: {torch.cuda.is_available()}')
    args = get_args()
    main(args)

