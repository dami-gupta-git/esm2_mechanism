"""
Nonlinear classifier probes on ESM-2 delta embeddings.

Tests whether mechanism signal (GOF/DN/LOF) is nonlinearly separable in delta
space where the linear probe was at chance. Runs MLP, GBM, RF, and kNN under
both gene-split and family-split CV.

Features:
  delta_mean — (mutant - wildtype) embedding averaged across all residues in the
               protein. Captures how the mutation shifts the global representation
               of the protein.
  delta_pos  — (mutant - wildtype) embedding at the specific mutated residue position.
               Captures the local perturbation at the mutation site.

  Input : data/valid_variants.json
          data/embeddings/esm2_t33_650M_UR50D/embeddings_*.npy
          data/pfam_families.json
  Output: results/<run_name>/nonlinear_results_seed{seed}.json
"""

import argparse
import functools
import json
import os
from pathlib import Path

import numpy as np

from esm2_mech.utils.paths import (
    EMB_MUT_MEAN, EMB_MUT_POS, EMB_WT_MEAN, EMB_WT_POS,
    PFAM_JSON, RESULTS_DIR, VALID_VARIANTS_JSON,
)
from esm2_mech.utils.probes import run_mlp_probe_cv, run_sklearn_probe_pca, run_sklearn_probe
from esm2_mech.utils.splits import gene_split_cv, family_split_cv

print = functools.partial(print, flush=True)

OUT_DIR = RESULTS_DIR


def load_data():
    with open(VALID_VARIANTS_JSON) as f:
        variants = json.load(f)
    labels = np.array([v["label_3class"] for v in variants])
    genes = np.array([v["gene"] for v in variants])
    print(f"Loaded {len(variants):,} variants, {len(set(genes))} genes")

    for path in [EMB_WT_MEAN, EMB_MUT_MEAN, EMB_WT_POS, EMB_MUT_POS]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embedding file missing: {path}")
    delta_mean = np.load(EMB_MUT_MEAN) - np.load(EMB_WT_MEAN)
    delta_pos = np.load(EMB_MUT_POS) - np.load(EMB_WT_POS)
    print(f"delta_mean: {delta_mean.shape}  delta_pos: {delta_pos.shape}")
    if delta_mean.shape[0] != len(variants):
        raise ValueError(
            f"Embedding row count {delta_mean.shape[0]} != variant count {len(variants)}."
        )

    with open(PFAM_JSON) as f:
        pfam_map = json.load(f)

    return labels, genes, delta_mean, delta_pos, pfam_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default=str(OUT_DIR),
                        help="Output directory for result JSON (default: RESULTS_DIR).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    labels, genes, delta_mean, delta_pos, pfam_map = load_data()

    gene_splits = gene_split_cv(genes, seed=args.seed)
    family_splits = family_split_cv(genes, pfam_map, seed=args.seed)
    print(f"Gene-split folds: {len(gene_splits)}  Family-split folds: {len(family_splits)}")

    def gbm_fn(seed):
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=seed)

    def rf_fn(seed):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=50, max_depth=8, random_state=seed, n_jobs=-1)

    def knn_fn(seed):
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=10, metric="cosine")

    results = {}
    for feat_name, X in [("delta_mean", delta_mean), ("delta_pos", delta_pos)]:
        for split_name, splits in [("gene", gene_splits), ("family", family_splits)]:
            key = f"mlp_{feat_name}_{split_name}"
            print(f"\n=== MLP {split_name}-split: {feat_name} ===")
            results[key] = run_mlp_probe_cv(
                X, labels, splits, seed=args.seed, genes=genes,
                max_epochs=args.max_epochs, patience=args.patience, label=key,
            )
            print(f"  macro_f1={results[key].get('macro_f1_mean', float('nan')):.3f}")

        # GBM/RF/kNN under both gene-split and family-split. The gene-split keys
        # keep their historical names (gbm_<feat>, no suffix); family-split adds a
        # _family suffix, mirroring the MLP keys above.
        for split_name, splits in [("gene", gene_splits), ("family", family_splits)]:
            gene_split = split_name == "gene"
            suffix = "" if gene_split else "_family"

            print(f"\n=== GBM {split_name}-split: {feat_name} (PCA-50) ===")
            key = f"gbm_{feat_name}{suffix}"
            results[key] = run_sklearn_probe_pca(gbm_fn, X, labels, genes, seed=args.seed, splits=splits)
            print(f"  macro_f1={results[key].get('macro_f1_mean', float('nan')):.3f}")

            print(f"\n=== RF {split_name}-split: {feat_name} (PCA-50) ===")
            key = f"rf_{feat_name}{suffix}"
            results[key] = run_sklearn_probe_pca(rf_fn, X, labels, genes, seed=args.seed, splits=splits)
            print(f"  macro_f1={results[key].get('macro_f1_mean', float('nan')):.3f}")

            print(f"\n=== kNN {split_name}-split: {feat_name} ===")
            key = f"knn_{feat_name}{suffix}"
            results[key] = run_sklearn_probe(knn_fn, X, labels, genes, seed=args.seed, normalize=True, splits=splits)
            print(f"  macro_f1={results[key].get('macro_f1_mean', float('nan')):.3f}")

    print("\n=== Summary ===")
    for feat, res in results.items():
        mf1 = res.get("macro_f1_mean", float("nan"))
        auroc_gof = res.get("auroc_GOF_mean", float("nan"))
        print(f"  {feat}: macro_f1={mf1:.3f}  auroc_GOF={auroc_gof:.3f}")

    out_path = out_dir / f"nonlinear_results_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
