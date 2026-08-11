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
from pathlib import Path

import numpy as np

from esm2_mech.utils.constants import (
    BOOTSTRAP_N_RESAMPLES,
    DELTA_MEAN_FEATURE,
    DELTA_POS_FEATURE,
    N_SEEDS,
    SPLIT_FAMILY,
    SPLIT_GENE,
    nonlinear_key,
)
from esm2_mech.utils.paths import (
    EMB_MUT_MEAN, EMB_MUT_POS, EMB_WT_MEAN, EMB_WT_POS,
    PFAM_JSON, RESULTS_DIR, VALID_VARIANTS_JSON,
)
from esm2_mech.utils.bootstrap import bootstrap_mechanism_metrics, family_or_gene_clusters
from esm2_mech.utils.io import atomic_write_json, load_variants_and_delta
from esm2_mech.utils.probes import run_mlp_probe_cv, run_sklearn_probe_pca, run_sklearn_probe
from esm2_mech.utils.splits import gene_split_cv, family_split_cv

print = functools.partial(print, flush=True)

OUT_DIR = RESULTS_DIR


def load_data():
    _variants, labels, genes, delta_mean, delta_pos = load_variants_and_delta(
        VALID_VARIANTS_JSON, EMB_WT_MEAN, EMB_MUT_MEAN, EMB_WT_POS, EMB_MUT_POS
    )

    with open(PFAM_JSON) as f:
        pfam_map = json.load(f)

    return labels, genes, delta_mean, delta_pos, pfam_map


def run_seed(seed, args, labels, genes, delta_mean, delta_pos, pfam_map):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    compute_ci = not args.no_ci
    n_boot = args.n_boot

    def _attach_ci(agg, oof, split_name):
        if compute_ci and oof is not None:
            clusters = family_or_gene_clusters(
                oof["genes"], pfam_map, is_family_split=(split_name == SPLIT_FAMILY)
            )
            agg["ci"] = bootstrap_mechanism_metrics(
                oof["y_true"], oof["proba"], clusters,
                n_resamples=n_boot, seed=seed,
            )
        return agg

    gene_splits = gene_split_cv(genes, seed=seed)
    family_splits = family_split_cv(genes, pfam_map, seed=seed)
    print(f"Gene-split folds: {len(gene_splits)}  Family-split folds: {len(family_splits)}")

    def gbm_fn(random_state):
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=random_state)

    def rf_fn(random_state):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=50, max_depth=8, random_state=random_state, n_jobs=-1)

    def knn_fn(random_state):
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=10, metric="cosine")

    TREE_KNN_MODELS = [
        ("gbm", "GBM", gbm_fn, run_sklearn_probe_pca, {}),
        ("rf", "RF", rf_fn, run_sklearn_probe_pca, {}),
        ("knn", "kNN", knn_fn, run_sklearn_probe, {"normalize": True}),
    ]

    def run_tree_knn(feat_name, X, split_name, splits, results):
        """Run GBM/RF/kNN for one feature under one split, storing into results.
        Keys are symmetric with the MLP keys: <model>_<feat>_<split>."""
        for model_key, model_label, clf_fn, probe_fn, extra_kwargs in TREE_KNN_MODELS:
            print(f"\n=== {model_label} {split_name}-split: {feat_name} ===")
            key = nonlinear_key(model_key, feat_name, split_name)
            agg, oof = probe_fn(
                clf_fn, X, labels, genes, seed=seed, splits=splits, return_oof=True,
                **extra_kwargs,
            )
            results[key] = _attach_ci(agg, oof, split_name)
            print(f"  macro_f1={results[key].get('macro_f1_mean', float('nan')):.3f}")

    out_path = out_dir / f"nonlinear_results_seed{seed}.json"

    feature_arrays = [(DELTA_MEAN_FEATURE, delta_mean), (DELTA_POS_FEATURE, delta_pos)]
    splits_by_name = [(SPLIT_GENE, gene_splits), (SPLIT_FAMILY, family_splits)]

    # Merge mode: compute only the new GBM/RF/kNN family-split arms and fold them
    # into the existing result file, leaving every existing key untouched.
    if args.only_new_family_arms:
        with open(out_path) as f:
            existing = json.load(f)
        new_arms = {}
        for feat_name, X in feature_arrays:
            run_tree_knn(feat_name, X, SPLIT_FAMILY, family_splits, new_arms)
        overwritten = sorted(set(existing) & set(new_arms))
        if overwritten:
            print(f"\nOverwriting existing keys with fresh values: {overwritten}")
        existing.update(new_arms)
        atomic_write_json(out_path, existing)
        print(f"\nMerged {len(new_arms)} family-split arms into {out_path}")
        for key, res in new_arms.items():
            print(f"  {key}: macro_f1={res.get('macro_f1_mean', float('nan')):.3f}")
        return

    results = {}
    for feat_name, X in feature_arrays:
        for split_name, splits in splits_by_name:
            key = nonlinear_key("mlp", feat_name, split_name)
            print(f"\n=== MLP {split_name}-split: {feat_name} ===")
            agg, oof = run_mlp_probe_cv(
                X, labels, splits, seed=seed, genes=genes,
                max_epochs=args.max_epochs, patience=args.patience, label=key,
                return_oof=True,
            )
            results[key] = _attach_ci(agg, oof, split_name)
            print(f"  macro_f1={results[key].get('macro_f1_mean', float('nan')):.3f}")

        # GBM/RF/kNN under both gene-split and family-split.
        for split_name, splits in splits_by_name:
            run_tree_knn(feat_name, X, split_name, splits, results)

    print("\n=== Summary ===")
    for feat, res in results.items():
        mf1 = res.get("macro_f1_mean", float("nan"))
        auroc_gof = res.get("auroc_GOF_mean", float("nan"))
        print(f"  {feat}: macro_f1={mf1:.3f}  auroc_GOF={auroc_gof:.3f}")

    atomic_write_json(out_path, results)
    print(f"\nResults written to {out_path}")


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--seeds", type=int, default=N_SEEDS,
                        help="number of seeds to run; runs 0..seeds-1 (>=1)")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default=str(OUT_DIR),
                        help="Output directory for result JSON (default: RESULTS_DIR).")
    parser.add_argument("--only_new_family_arms", action="store_true",
                        help="Compute ONLY the GBM/RF/kNN family-split arms (gbm/rf/knn_<feat>_family) "
                             "and merge them into the existing nonlinear_results_seed{seed}.json in "
                             "out_dir, preserving all existing keys. Skips MLP and all gene-split work.")
    parser.add_argument("--no_ci", action="store_true", help="skip cluster-bootstrap CIs")
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")

    labels, genes, delta_mean, delta_pos, pfam_map = load_data()

    for seed in range(args.seeds):
        print("\n\n" + "#" * 60)
        print(f"# SEED {seed}")
        print("#" * 60)
        run_seed(seed, args, labels, genes, delta_mean, delta_pos, pfam_map)


if __name__ == "__main__":
    main()
