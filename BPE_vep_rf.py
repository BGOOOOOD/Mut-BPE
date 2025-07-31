import random
import time
from os import path as osp
import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from tqdm.auto import tqdm

USE_TISSUE = [False]  # Whether to use tissue embedding
N_ESTIMATORS = [5, 20, 60]  # Number of random forest trees
SEEDS = [1, 2, 3, 4, 5]


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

def get_dist_to_tss(experiment):
    if experiment == "variant_effect_pathogenic_mendelian_chr11":
        DIST_TO_TSS = [[0, 50], [50, 600], [600, np.infty]]
    elif experiment == "variant_effect_pathogenic_Complex_chr1":
        DIST_TO_TSS = [[0, 6000], [6000, 30_000], [30_000, np.infty]]
    else:
        DIST_TO_TSS = [[0, 30_000], [30_000, 100_000], [100_000, np.infty]]
    return DIST_TO_TSS

def dataset_nan_filter(data: dict, data_key: str) -> dict:
    mask_out = torch.logical_or(
        torch.any(data[data_key].isnan(), dim=1),
        torch.any(data[f"{data_key}"].isnan(), dim=1)
    )
    new_data = dict()
    for k in data.keys():
        new_data[k] = data[k][~mask_out]
    return new_data

def dataset_tss_filter(data: dict, min_distance: int, max_distance: int) -> dict:
    distance_mask = ((data["distance_to_nearest_tss"] >= min_distance) 
                     & (data["distance_to_nearest_tss"] <= max_distance))
    new_data = dict()
    for k in data.keys():
        new_data[k] = data[k][distance_mask]
    return new_data

def main(args):
    DIST_TO_TSS = get_dist_to_tss(args.experiment)
    embeds_path = osp.join(
        'output', args.experiment, args.model,
        f'method={args.mutbpe_method}',
        f'rs={args.reverse_strand}_use_CLS={args.use_CLS}'
    )
    train_path = osp.join(embeds_path, "train_embeds.pt")
    test_path = osp.join(embeds_path, "test_embeds.pt")
    print(f"Embedding path: {embeds_path}")
    if not (osp.exists(train_path) and osp.exists(test_path)):
        print("Embedding files not found!")
        return
    if args.reverse_strand == "True":
        key = "concat_avg_ws_both"
    else:
        key = "concat_cls" if args.use_CLS == "True" else "concat_avg_ws"
    train_val_ds_raw = torch.load(train_path, map_location="cpu")
    train_val_ds_raw = dataset_nan_filter(train_val_ds_raw, data_key=key)
    test_ds_raw = torch.load(test_path, map_location="cpu")
    test_ds_raw = dataset_nan_filter(test_ds_raw, data_key=key)
    print(f"Total Train size: {len(train_val_ds_raw[key])},", end=" ")
    print(f"Total Test size: {len(test_ds_raw[key])},", end=" ")
    print(f"Shape: {test_ds_raw[key].shape[1:]}")

    metrics = {
        "bucket_id": [],
        "use_tissue": [],
        "n_estimators": [],
        "seed": [],
        "AUROC": [],
        "AUPRC": [],
    }

    has_tss = "distance_to_nearest_tss" in train_val_ds_raw and "distance_to_nearest_tss" in test_ds_raw
    has_tss = False
    total_jobs = 0
    if has_tss:
        total_jobs = len(DIST_TO_TSS) * len(USE_TISSUE) * len(N_ESTIMATORS) * len(SEEDS)
    else:
        total_jobs = 1 * len(USE_TISSUE) * len(N_ESTIMATORS) * len(SEEDS)
    job_count = 0
    pbar = tqdm(total=total_jobs, desc="RF jobs")

    if has_tss:
        for bucket_id, (min_dist, max_dist) in enumerate(DIST_TO_TSS):
            train_val_ds_filter = dataset_tss_filter(train_val_ds_raw, min_dist, max_dist)
            test_ds_filter = dataset_tss_filter(test_ds_raw, min_dist, max_dist)
            print(f"- TSS bucket: [{min_dist}, {max_dist}],", end=" ")
            print(f"Train size: {len(train_val_ds_filter[key])},", end=" ")
            print(f"Test size: {len(test_ds_filter[key])}")
            for use_tissue in USE_TISSUE:
                for n_estimators in N_ESTIMATORS:
                    for seed in SEEDS:
                        random = np.random
                        random.seed(seed)
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        rf_clf = make_pipeline(
                            StandardScaler(),
                            RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
                        )
                        X = np.array(train_val_ds_filter[key])
                        X_with_tissue = np.concatenate(
                            [X, np.array(train_val_ds_filter["tissue_embed"])[..., None]],
                            axis=-1
                        ) if use_tissue else X
                        y = train_val_ds_filter["labels"]
                        X_test = np.array(test_ds_filter[key])
                        X_test_with_tissue = np.concatenate(
                            [X_test, np.array(test_ds_filter["tissue_embed"])[..., None]],
                            axis=-1
                        ) if use_tissue else X_test
                        y_test = test_ds_filter["labels"]
                        print(f"\tFitting RF (use_tissue={use_tissue}, n_estimators={n_estimators}, seed={seed})...", end=" ")
                        mask = np.random.choice(len(X), size=5000, replace=5000 > len(X))
                        if use_tissue:
                            X_train = X_with_tissue[mask]
                            X_test_final = X_test_with_tissue
                        else:
                            X_train = X[mask]
                            X_test_final = X_test
                        y_train = y[mask]
                        rf_clf.fit(X_train, y_train)
                        rf_y_pred = rf_clf.predict(X_test_final)
                        rf_y_pred_proba = rf_clf.predict_proba(X_test_final)[:, 1]  # Get probability of positive class
                        rf_aucroc = roc_auc_score(y_test, rf_y_pred_proba)
                        rf_auprc = average_precision_score(y_test, rf_y_pred_proba)
                        print(f"AUROC: {rf_aucroc}, AUPRC: {rf_auprc}")
                        metrics["bucket_id"].append(bucket_id)
                        metrics["use_tissue"].append(use_tissue)
                        metrics["n_estimators"].append(n_estimators)
                        metrics["seed"].append(seed)
                        metrics["AUROC"].append(rf_aucroc)
                        metrics["AUPRC"].append(rf_auprc)
                        job_count += 1
                        pbar.update(1)
    else:
        bucket_id = 0
        train_val_ds_filter = train_val_ds_raw
        test_ds_filter = test_ds_raw
        print(f"No distance_to_nearest_tss key, using all data.")
        print(f"Train size: {len(train_val_ds_filter[key])},", end=" ")
        print(f"Test size: {len(test_ds_filter[key])}")
        for use_tissue in USE_TISSUE:
            for n_estimators in N_ESTIMATORS:
                for seed in SEEDS:
                    random = np.random
                    random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    rf_clf = make_pipeline(
                        StandardScaler(),
                        RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
                    )
                    X = np.array(train_val_ds_filter[key])
                    X_with_tissue = np.concatenate(
                        [X, np.array(train_val_ds_filter["tissue_embed"])[..., None]],
                        axis=-1
                    ) if use_tissue else X
                    y = train_val_ds_filter["labels"]
                    X_test = np.array(test_ds_filter[key])
                    X_test_with_tissue = np.concatenate(
                        [X_test, np.array(test_ds_filter["tissue_embed"])[..., None]],
                        axis=-1
                    ) if use_tissue else X_test
                    y_test = test_ds_filter["labels"]
                    print(f"\tFitting RF (use_tissue={use_tissue}, n_estimators={n_estimators}, seed={seed})...", end=" ")
                    mask = np.random.choice(len(X), size=5000, replace=5000 > len(X))
                    if use_tissue:
                        X_train = X_with_tissue[mask]
                        X_test_final = X_test_with_tissue
                    else:
                        X_train = X[mask]
                        X_test_final = X_test
                    y_train = y[mask]
                    rf_clf.fit(X_train, y_train)
                    rf_y_pred = rf_clf.predict(X_test_final)
                    rf_y_pred_proba = rf_clf.predict_proba(X_test_final)[:, 1]  # Get probability of positive class
                    rf_aucroc = roc_auc_score(y_test, rf_y_pred_proba)
                    rf_auprc = average_precision_score(y_test, rf_y_pred_proba)
                    print(f"AUROC: {rf_aucroc}, AUPRC: {rf_auprc}")
                    metrics["bucket_id"].append(bucket_id)
                    metrics["use_tissue"].append(use_tissue)
                    metrics["n_estimators"].append(n_estimators)
                    metrics["seed"].append(seed)
                    metrics["AUROC"].append(rf_aucroc)
                    metrics["AUPRC"].append(rf_auprc)
                    job_count += 1
                    pbar.update(1)
    pbar.close()
    out_name = osp.join(args.downstream_save_dir, args.experiment, args.model, f'method={args.mutbpe_method}', '_'.join([f'rs={args.reverse_strand}', f'use_CLS={args.use_CLS}']), 'vep_rf')
    out_path = osp.join(out_name, 'rf_results.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_metrics = pd.DataFrame.from_dict(metrics)
    df_metrics.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mutbpe_method", required=True)
    parser.add_argument("--reverse_strand", required=True)
    parser.add_argument("--use_CLS", required=True)
    parser.add_argument("--downstream_save_dir", default="output")
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    main(args)
