"""
Run GBM / Random Forest / kNN on the delta features under FAMILY-split CV.

The mechanism MLP experiment (experiments/mechanism/mlp.py) runs the MLP under
both gene-split and family-split, but runs GBM, RF, and kNN under gene-split
only. As a result the family-split column for those three models is empty in
reports/run6/report_classifier.md, which makes statements like "best nonlinear
family-split score" best-among-available rather than best overall.

This script fills that gap: it runs GBM, RF, and kNN on `delta_mean` and
`delta_pos` under family-split, reusing the exact estimator factories and probe
helpers from mlp.py / utils.probes (so the numbers are directly comparable to
the existing gene-split cells), averaged over the same 5 seeds.

  Input : data/valid_variants.json, data/pfam_families.json, embeddings_*.npy
  Output: stdout table (not written to disk)

Usage:
    python -m scripts.nonlinear_family_split
    python scripts/nonlinear_family_split.py
"""

from __future__ import annotations

import functools

import numpy as np

from esm2_mech.experiments.mechanism.mlp import load_data
from esm2_mech.utils.probes import run_sklearn_probe, run_sklearn_probe_pca
from esm2_mech.utils.splits import family_split_cv

print = functools.partial(print, flush=True)

N_FOLDS = 5
N_SEEDS = 5


def gbm_fn(seed):
    from sklearn.ensemble import GradientBoostingClassifier

    return GradientBoostingClassifier(
        n_estimators=50, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=seed
    )


def rf_fn(seed):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=50, max_depth=8, random_state=seed, n_jobs=-1
    )


def knn_fn(seed):
    from sklearn.neighbors import KNeighborsClassifier

    return KNeighborsClassifier(n_neighbors=10, metric="cosine")


def _mean_std(per_seed_macro_f1):
    vals = [v for v in per_seed_macro_f1 if not np.isnan(v)]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def main() -> None:
    labels, genes, delta_mean, delta_pos, pfam_map = load_data()

    feature_sets = [("delta_mean", delta_mean), ("delta_pos", delta_pos)]
    # (label, runner, extra-kwargs) — mirrors how mlp.py invokes each on gene-split.
    models = [
        ("gbm", lambda X, splits, seed: run_sklearn_probe_pca(
            gbm_fn, X, labels, genes, seed=seed, splits=splits)),
        ("rf", lambda X, splits, seed: run_sklearn_probe_pca(
            rf_fn, X, labels, genes, seed=seed, splits=splits)),
        ("knn", lambda X, splits, seed: run_sklearn_probe(
            knn_fn, X, labels, genes, seed=seed, normalize=True, splits=splits)),
    ]

    print(f"n = {len(labels)}  averaging over {N_SEEDS} seeds, {N_FOLDS}-fold FAMILY-split CV\n")
    print(f"{'model':22} {'family macro_f1':>16}")

    for feat_name, X in feature_sets:
        for model_name, runner in models:
            per_seed = []
            for seed in range(N_SEEDS):
                splits = family_split_cv(genes, pfam_map, n_folds=N_FOLDS, seed=seed)
                res = runner(X, splits, seed)
                per_seed.append(res.get("macro_f1_mean", float("nan")))
            mean, std = _mean_std(per_seed)
            print(f"{model_name + '_' + feat_name:22} {mean:8.3f} ± {std:.3f}")


if __name__ == "__main__":
    main()
