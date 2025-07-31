import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr, spearmanr
from os import path as osp
import os
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--experiment", type=str, default="variant_effect_pathogenic_ClinVar", choices=['variant_effect_pathogenic_ClinVar', 
    'variant_effect_pathogenic_ClinVar_chr1', 'variant_effect_pathogenic_Cosmic_chr1', 'variant_effect_pathogenic_Complex_chr1', 'variant_effect_pathogenic_mendelian_chr11'], help="Experiment name.")
    parser.add_argument("--downstream_save_dir", type=str, default="output",
                        help="Directory to save downstream task.")
    parser.add_argument("--model", type=str, default=None, choices=['dnabert2', 'mutbert', 'NT', 'hyena', 'caduceus', 'gena-lm'], help="Embeddings model name.")
    parser.add_argument("--mutbpe_method", type=str, default=None, choices=['None', 'base', 'mutbpe'], help="MutBPE Methods.")
    parser.add_argument("--reverse_strand", type=str, choices=["True", "False"], default="False", help="Use reverse strand sequences.")
    parser.add_argument("--use_CLS", type=str, choices=["True", "False"], default="False", help="Use CLS token for classification.")
    parser.add_argument("--distance_metrics", type=str, default='none', choices=['none', 'cosine', 'pearson', 'spearman'], help="distance metrics for the embeddings.")
    args = parser.parse_args()
    return args

def cosine_similarity(a, b):
    return F.cosine_similarity(a, b, dim=-1)

def l1_distance(a, b):
    return torch.norm(a - b, p=1, dim=-1)

def l2_distance(a, b):
    return torch.norm(a - b, p=2, dim=-1)

def pearson_similarity(a, b):
    # a, b: [N, D] or [D]
    if a.ndim == 1:
        return pearsonr(a.cpu().numpy(), b.cpu().numpy())[0]
    else:
        return torch.tensor([
            pearsonr(a[i].cpu().numpy(), b[i].cpu().numpy())[0] for i in range(a.shape[0])
        ])

def spearman_similarity(a, b):
    # a, b: [N, D] or [D]
    if a.ndim == 1:
        return spearmanr(a.cpu().numpy(), b.cpu().numpy())[0]
    else:
        return torch.tensor([
            spearmanr(a[i].cpu().numpy(), b[i].cpu().numpy())[0] for i in range(a.shape[0])
        ])

def load_embeddings_and_calc_distance(embeds_path, split_name, args):
    """
    Load embedding file, extract ref and alt embeddings, calculate distance, compute AUROC, and save results.
    """
    embed_file = os.path.join(embeds_path, f"{split_name}_embeds.pt")
    data = torch.load(embed_file)
    
    # Load and process embeddings according to reverse_strand usage
    if args.reverse_strand == "False":
        # Case: only use forward strand
        # concat_avg_ws shape: [batch, hidden*2], first half is ref, second half is alt
        concat = data["concat_avg_ws"] if args.use_CLS == "False" else data["concat_cls"]
        ref, alt = torch.chunk(concat, 2, dim=-1)
    else:
        # Case: use both forward and reverse strands
        # concat_avg_ws_both shape: [batch, hidden*4], order: ref_f, alt_f, ref_r, alt_r
        concat = data["concat_avg_ws_both"]
        ref_f, alt_f, ref_r, alt_r = torch.chunk(concat, 4, dim=-1)
        # Concatenate forward and reverse strand embeddings
        ref = torch.cat([ref_f, ref_r], dim=-1)
        alt = torch.cat([alt_f, alt_r], dim=-1)

    # Calculate distance with progress bar
    if ref.ndim == 1:
        # Single sample, no progress bar needed
        if args.distance_metrics == "cosine":
            dist = cosine_similarity(ref, alt)
        elif args.distance_metrics == "l1":
            dist = l1_distance(ref, alt)
        elif args.distance_metrics == "l2":
            dist = l2_distance(ref, alt)
        elif args.distance_metrics == "pearson":
            dist = pearson_similarity(ref, alt)
        elif args.distance_metrics == "spearman":
            dist = spearman_similarity(ref, alt)
        else:
            raise ValueError(f"Unsupported distance_metrics: {args.distance_metrics}")
    else:
        # Multiple samples, calculate one by one with progress bar
        dist_list = []
        for i in tqdm(range(ref.shape[0]), desc=f"Calculating {args.distance_metrics} distance"):
            if args.distance_metrics == "cosine":
                d = F.cosine_similarity(ref[i].unsqueeze(0).float(), alt[i].unsqueeze(0).float()).item()
            elif args.distance_metrics == "l1":
                d = torch.norm(ref[i] - alt[i], p=1).item()
            elif args.distance_metrics == "l2":
                d = torch.norm(ref[i] - alt[i], p=2).item()
            elif args.distance_metrics == "pearson":
                d = pearsonr(ref[i].cpu().numpy(), alt[i].cpu().numpy())[0]
            elif args.distance_metrics == "spearman":
                d = spearmanr(ref[i].cpu().numpy(), alt[i].cpu().numpy())[0]
            else:
                raise ValueError(f"Unsupported distance_metrics: {args.distance_metrics}")
            dist_list.append(d)
        dist = np.array(dist_list)

    # Convert dist to numpy array and ensure it has shape attribute
    if isinstance(dist, torch.Tensor):
        dist_np = dist.cpu().numpy().flatten()
    else:
        dist_np = np.array(dist).flatten()
    print(f"{split_name} {args.distance_metrics} distance shape: {dist_np.shape}")

    # Read labels
    labels = data["labels"].cpu().numpy().flatten()
    
    # Compute AUROC
    try:
        auroc = roc_auc_score(labels, dist_np)
    except Exception as e:
        print(f"AUROC calculation failed: {e}")
        auroc = None
    print(f"{split_name} AUROC: {auroc}")

    # Compute AUPRC
    try:
        auprc = average_precision_score(labels, dist_np)
    except Exception as e:
        print(f"AUPRC calculation failed: {e}")
        auprc = None
    print(f"{split_name} AUPRC: {auprc}")

    # Save results
    save_dict = {
        "distance": dist_np,
        "labels": labels,
        "auroc": auroc,
        "auprc": auprc if auprc is not None else 0
    }
    zero_shot_path = osp.join(args.downstream_save_dir, args.experiment, args.model, f'method={args.mutbpe_method}', '_'.join([f'rs={args.reverse_strand}', f'use_CLS={args.use_CLS}']), 'zero_shot')
    save_path = osp.join(zero_shot_path, f"{split_name}_{args.distance_metrics}_metrics.npz")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, **save_dict)
    print(f"Saved distance, AUROC and AUPRC results to: {save_path}")
    return dist_np, auroc, auprc

def main(args):
    if args.distance_metrics != 'none':
        # Calculate distance
        embeds_path = osp.join(
            'output',
            args.experiment,
            args.model,
            f'method={args.mutbpe_method}',
            '_'.join([
                f'rs={args.reverse_strand}',
                f'use_CLS={args.use_CLS}'
            ])
        )
        for split_name in tqdm(["test"], desc="Calculating AUROC"):
            if os.path.exists(os.path.join(embeds_path, f"{split_name}_embeds.pt")):
                load_embeddings_and_calc_distance(embeds_path, split_name, args)

        # Print AUROC and AUPRC
        for split_name in tqdm(["test"], desc="Print metrics"):
            if os.path.exists(os.path.join(embeds_path, f"{split_name}_embeds.pt")):
                dist, auroc, auprc = load_embeddings_and_calc_distance(embeds_path, split_name, args)
                print(f"{split_name} AUROC: {auroc}, AUPRC: {auprc}")


if __name__ == "__main__":
    args = get_args()
    main(args)