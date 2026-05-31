"""
Measure the naive-classifier baseline for 3-class mechanism prediction.

Computes the macro-F1 and per-class AUROC that a classifier with no learned
signal achieves on the GOF/DN/LOF task, so the floor can be reported alongside
the real feature scores in reports/run6/report_1.md (the `naive baseline` row).

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

  Input : data/valid_variants.json   (label_3class, gene per variant)
          data/pfam_families.json    (gene -> Pfam family, for family-split)
  Output: stdout table (not written to disk)

Usage:
    python -m scripts.naive_baseline
    python scripts/naive_baseline.py
"""

from __future__ import annotations

import functools
import json

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, roc_auc_score

from esm2_mech.utils.constants import MECHANISM_CLASSES
from esm2_mech.utils.paths import PFAM_JSON, VALID_VARIANTS_JSON
from esm2_mech.utils.splits import family_split_cv, gene_split_cv

print = functools.partial(print, flush=True)

N_FOLDS = 5
N_SEEDS = 5
STRATEGIES = ["most_frequent", "prior", "stratified"]


def _eval_dummy(strategy, labels, splits, seed):
    """Run a DummyClassifier across pre-computed (train, test) splits.

    Returns (list of per-fold macro-F1, dict class -> list of per-fold AUROC).
    DummyClassifier ignores X, so a zero placeholder is passed.
    """
    placeholder_x = np.zeros((len(labels), 1))
    macro_f1_folds: list[float] = []
    auroc_folds: dict[str, list[float]] = {cls: [] for cls in MECHANISM_CLASSES}

    for train_idx, test_idx in splits:
        y_train, y_test = labels[train_idx], labels[test_idx]
        if len(set(y_train)) < 2:
            continue
        clf = DummyClassifier(strategy=strategy, random_state=seed)
        clf.fit(placeholder_x[train_idx], y_train)
        pred = clf.predict(placeholder_x[test_idx])
        proba = clf.predict_proba(placeholder_x[test_idx])

        macro_f1_folds.append(
            float(f1_score(y_test, pred, average="macro", zero_division=0))
        )
        for class_idx, cls in enumerate(clf.classes_):
            y_bin = (y_test == cls).astype(int)
            if y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
                auroc_folds[cls].append(
                    float(roc_auc_score(y_bin, proba[:, class_idx]))
                )

    return macro_f1_folds, auroc_folds


def main() -> None:
    if not VALID_VARIANTS_JSON.exists():
        raise FileNotFoundError(
            f"{VALID_VARIANTS_JSON} not found — run fetch_data/build_valid_variants first"
        )
    if not PFAM_JSON.exists():
        raise FileNotFoundError(
            f"{PFAM_JSON} not found — run fetch_data/fetch_annotations --step pfam first"
        )

    with open(VALID_VARIANTS_JSON) as fh:
        variants = json.load(fh)
    with open(PFAM_JSON) as fh:
        pfam_map = json.load(fh)

    labels = np.array([v["label_3class"] for v in variants])
    genes = np.array([v["gene"] for v in variants])

    from collections import Counter

    print(f"n = {len(labels)}  class distribution = {dict(Counter(labels))}")
    print(f"Averaging over {N_SEEDS} seeds, {N_FOLDS}-fold CV\n")

    header = f"{'strategy':14} {'split':7} {'macro_f1':>9}  " + "  ".join(
        f"{c:>5}" for c in MECHANISM_CLASSES
    )
    print(header)

    for strategy in STRATEGIES:
        for split_name in ("gene", "family"):
            all_macro: list[float] = []
            all_auroc: dict[str, list[float]] = {c: [] for c in MECHANISM_CLASSES}
            for seed in range(N_SEEDS):
                if split_name == "gene":
                    splits = gene_split_cv(genes, n_folds=N_FOLDS, seed=seed)
                else:
                    splits = family_split_cv(
                        genes, pfam_map, n_folds=N_FOLDS, seed=seed
                    )
                macro_folds, auroc_folds = _eval_dummy(
                    strategy, labels, splits, seed
                )
                all_macro += macro_folds
                for cls in MECHANISM_CLASSES:
                    all_auroc[cls] += auroc_folds[cls]

            macro_mean = float(np.mean(all_macro)) if all_macro else float("nan")
            auroc_means = {
                cls: (float(np.mean(all_auroc[cls])) if all_auroc[cls] else float("nan"))
                for cls in MECHANISM_CLASSES
            }
            auroc_str = "  ".join(f"{auroc_means[c]:.3f}" for c in MECHANISM_CLASSES)
            print(f"{strategy:14} {split_name:7} {macro_mean:9.3f}  {auroc_str}")


if __name__ == "__main__":
    main()
