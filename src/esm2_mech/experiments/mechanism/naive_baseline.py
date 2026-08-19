"""
Measure the naive-classifier baseline for 3-class mechanism prediction.

Computes the macro-F1 and per-class AUROC that a classifier with no learned
signal achieves on the GOF/DN/LOF task, so the floor can be reported alongside
the real feature scores in reports/run6/report_classifier.md (the `naive baseline` row).

To stay consistent with the experiment, this reuses the project's own
cross-validation split functions (gene_split_cv / family_split_cv), the same
3-class labels (label_3class), and the same metrics (macro-F1 via f1_score,
per-class one-vs-rest AUROC via roc_auc_score), averaged over the same 5 seeds.
Only the estimator differs: sklearn's DummyClassifier in place of the probe.

Three dummy strategies are reported:
  - most_frequent : always predict the majority class (LOF). This is the
                    stricter reference and the value reported in the tables.
  - prior         : same predictions as most_frequent for a single-label argmax.
  - stratified    : predict randomly in proportion to class frequencies.

Reported values are mean ± std across the 5 seeds (per-seed value = mean over
that seed's folds), matching how the experiment aggregates seeds.

  Input : data/valid_variants.json   (label_3class, gene per variant)
          data/pfam_families.json    (gene -> Pfam family, for family-split)
  Output: results/<run>/naive_baseline.json (path from paths.NAIVE_BASELINE_JSON)
          plus a summary table to stdout

Usage:
    python -m esm2_mech.experiments.mechanism.naive_baseline
"""

from __future__ import annotations

import functools
import json
from collections import Counter

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, roc_auc_score

from esm2_mech.utils.bootstrap import cluster_bootstrap_ci
from esm2_mech.utils.constants import (
    BOOTSTRAP_N_RESAMPLES,
    MECHANISM_CLASSES,
    N_FOLDS,
    N_SEEDS,
)
from esm2_mech.utils.data import labeled_variant_fingerprint, load_pfam_map, pfam_fingerprint
from esm2_mech.utils.io import write_result_json
from esm2_mech.utils.paths import NAIVE_BASELINE_JSON, PFAM_JSON, VALID_VARIANTS_JSON
from esm2_mech.utils.splits import family_split_cv, gene_split_cv

print = functools.partial(print, flush=True)

STRATEGIES = ["most_frequent", "prior", "stratified"]


def _eval_dummy_one_seed(strategy, labels, splits, seed):
    """Run a DummyClassifier across one seed's (train, test) folds.

    Returns (mean macro-F1 over folds, dict class -> mean AUROC over folds).
    DummyClassifier ignores X, so a zero placeholder is passed. NaN is returned
    for a metric with no contributing fold.
    """
    placeholder_x = np.zeros((len(labels), 1))
    macro_f1_folds: list[float] = []
    auroc_folds: dict[str, list[float]] = {cls: [] for cls in MECHANISM_CLASSES}

    for train_idx, test_idx in splits:
        y_train, y_test = labels[train_idx], labels[test_idx]
        if set(y_train.tolist()) != set(MECHANISM_CLASSES):
            continue
        if set(y_test.tolist()) != set(MECHANISM_CLASSES):
            continue
        clf = DummyClassifier(strategy=strategy, random_state=seed)
        clf.fit(placeholder_x[train_idx], y_train)
        pred = clf.predict(placeholder_x[test_idx])
        proba = clf.predict_proba(placeholder_x[test_idx])

        macro_f1_folds.append(
            float(
                f1_score(
                    y_test,
                    pred,
                    labels=list(MECHANISM_CLASSES),
                    average="macro",
                    zero_division=0,
                )
            )
        )
        for class_idx, cls in enumerate(clf.classes_):
            y_bin = (y_test == cls).astype(int)
            if y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
                auroc_folds[cls].append(
                    float(roc_auc_score(y_bin, proba[:, class_idx]))
                )

    macro_mean = float(np.mean(macro_f1_folds)) if macro_f1_folds else float("nan")
    auroc_means = {
        cls: (float(np.mean(auroc_folds[cls])) if auroc_folds[cls] else float("nan"))
        for cls in MECHANISM_CLASSES
    }
    return macro_mean, auroc_means


def _mean_std(values):
    vals = [v for v in values if not np.isnan(v)]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def evaluate(strategy, split_name, labels, genes, pfam_map, n_seeds=N_SEEDS, n_folds=N_FOLDS):
    """Mean ± std across n_seeds for one (strategy, split) cell."""
    per_seed_macro: list[float] = []
    per_seed_auroc: dict[str, list[float]] = {c: [] for c in MECHANISM_CLASSES}
    for seed in range(n_seeds):
        if split_name == "gene":
            splits = gene_split_cv(genes, n_folds=n_folds, seed=seed)
        else:
            splits = family_split_cv(genes, pfam_map, n_folds=n_folds, seed=seed)
        macro, auroc = _eval_dummy_one_seed(strategy, labels, splits, seed)
        per_seed_macro.append(macro)
        for cls in MECHANISM_CLASSES:
            per_seed_auroc[cls].append(auroc[cls])

    macro_mean, macro_std = _mean_std(per_seed_macro)
    result = {
        "macro_f1_mean": macro_mean,
        "macro_f1_std": macro_std,
        "n_seeds": n_seeds,
    }
    for cls in MECHANISM_CLASSES:
        mean, std = _mean_std(per_seed_auroc[cls])
        result[f"auroc_{cls}_mean"] = mean
        result[f"auroc_{cls}_std"] = std
    return result


def floor_macro_f1_ci(labels, genes, pfam_map, seed=0, n_boot=BOOTSTRAP_N_RESAMPLES):
    """Cluster-bootstrap CIs for the most_frequent floor macro-F1.

    The majority class is refitted inside each fold and bootstrap draw. The gene CI
    resamples whole genes; the family CI resamples whole Pfam families (unannotated
    genes excluded, matching family-split CV). Both metrics are scored within fold
    and discard a draw when a fold loses a mechanism class.
    """
    classes = list(MECHANISM_CLASSES)

    def _one_split(split_name):
        splits = (
            gene_split_cv(genes, n_folds=N_FOLDS, seed=seed)
            if split_name == "gene"
            else family_split_cv(genes, pfam_map, n_folds=N_FOLDS, seed=seed)
        )
        if len(splits) != N_FOLDS:
            raise ValueError(
                f"{split_name} majority-floor CI has {len(splits)} folds, expected {N_FOLDS}"
            )
        row_ids = np.concatenate([test_idx for _train_idx, test_idx in splits])
        oof_labels = labels[row_ids]
        oof_genes = genes[row_ids]
        oof_folds = np.concatenate([
            np.full(len(test_idx), fold_index, dtype=int)
            for fold_index, (_train_idx, test_idx) in enumerate(splits)
        ])
        clusters = (
            oof_genes
            if split_name == "gene"
            else np.array([pfam_map[gene] for gene in oof_genes], dtype=object)
        )

        def _fold_mean_floor(resampled_rows):
            resampled_rows = np.asarray(resampled_rows, dtype=int)
            sampled_folds = oof_folds[resampled_rows]
            values = []
            for fold_index in range(N_FOLDS):
                test_rows = resampled_rows[sampled_folds == fold_index]
                train_rows = resampled_rows[sampled_folds != fold_index]
                if len(test_rows) == 0 or len(train_rows) == 0:
                    return None
                y_test = oof_labels[test_rows]
                if set(y_test.tolist()) != set(classes):
                    return None
                y_train = oof_labels[train_rows]
                train_classes, counts = np.unique(y_train, return_counts=True)
                if set(train_classes.tolist()) != set(classes):
                    return None
                majority = train_classes[int(np.argmax(counts))]
                pred = np.full(len(test_rows), majority)
                values.append(float(f1_score(
                    y_test,
                    pred,
                    labels=classes,
                    average="macro",
                    zero_division=0,
                )))
            return float(np.mean(values))

        return cluster_bootstrap_ci(
            clusters,
            _fold_mean_floor,
            n_resamples=n_boot,
            seed=seed,
            discard_reason="a fold's resampled rows lost a mechanism class",
            metric_name=f"{split_name}_majority_floor_macro_f1",
        )

    return {"gene": _one_split("gene"), "family": _one_split("family")}


def main() -> None:
    with open(VALID_VARIANTS_JSON) as fh:
        variants = json.load(fh)
    pfam_map = load_pfam_map(PFAM_JSON)

    labels = np.array([v["label_3class"] for v in variants])
    genes = np.array([v["gene"] for v in variants])

    class_distribution = dict(Counter(labels))
    print(f"n = {len(labels)}  class distribution = {class_distribution}")
    print(f"Averaging over {N_SEEDS} seeds, {N_FOLDS}-fold CV\n")

    results = {
        "input_fingerprints": {
            "labeled_variants": labeled_variant_fingerprint(variants, labels),
            "pfam_assignments": pfam_fingerprint(pfam_map, genes.tolist()),
        },
        "analysis_parameters": {
            "n_seeds": N_SEEDS,
            "n_folds": N_FOLDS,
            "n_bootstrap_resamples": BOOTSTRAP_N_RESAMPLES,
        },
        "n_variants": int(len(labels)),
        "class_distribution": {k: int(v) for k, v in class_distribution.items()},
        "n_seeds": N_SEEDS,
        "n_folds": N_FOLDS,
        "by_strategy": {},
    }

    header = f"{'strategy':14} {'split':7} {'macro_f1':>15}  " + "  ".join(
        f"{c:>5}" for c in MECHANISM_CLASSES
    )
    print(header)

    for strategy in STRATEGIES:
        results["by_strategy"][strategy] = {}
        for split_name in ("gene", "family"):
            cell = evaluate(strategy, split_name, labels, genes, pfam_map)
            results["by_strategy"][strategy][split_name] = cell
            macro = f"{cell['macro_f1_mean']:.3f} ± {cell['macro_f1_std']:.3f}"
            auroc_str = "  ".join(
                f"{cell[f'auroc_{c}_mean']:.3f}" for c in MECHANISM_CLASSES
            )
            print(f"{strategy:14} {split_name:7} {macro:>15}  {auroc_str}")

    print("\nMost_frequent floor cluster-bootstrap CIs (macro-F1)...")
    floor_ci = floor_macro_f1_ci(labels, genes, pfam_map)
    results["most_frequent_floor_ci"] = floor_ci
    for split_name in ("gene", "family"):
        cell = floor_ci[split_name]
        point_str = f"{cell['point']:.3f}" if cell["point"] is not None else "n/a"
        if cell["ci_low"] is not None and cell["ci_high"] is not None:
            ci_str = f"95% CI [{cell['ci_low']:.3f}, {cell['ci_high']:.3f}]"
        else:
            ci_str = "95% CI suppressed (too few valid resamples)"
        print(
            f"  {split_name:7} point {point_str}  {ci_str}  "
            f"(n_clusters {cell['n_clusters']})"
        )

    NAIVE_BASELINE_JSON.parent.mkdir(parents=True, exist_ok=True)
    write_result_json(NAIVE_BASELINE_JSON, results, seeds=list(range(N_SEEDS)), indent=2)
    print(f"\nResults written to {NAIVE_BASELINE_JSON}")


if __name__ == "__main__":
    main()
