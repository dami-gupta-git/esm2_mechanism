"""Evaluation metrics: per-fold computation, fold aggregation, probability alignment."""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from esm2_mech.utils.constants import MECHANISM_CLASSES


def mean_std_n(values) -> tuple[float, float, int]:
    """NaN/None-safe (mean, std, n) of a list of scalars.

    Filters out None and NaN, then returns the mean, population std, and count of
    the surviving values. Empty -> (nan, nan, 0). Shared by the geometry probes,
    which each previously inlined this reducer.
    """
    clean = [v for v in values if v is not None and not np.isnan(v)]
    if not clean:
        return float("nan"), float("nan"), 0
    return float(np.mean(clean)), float(np.std(clean)), len(clean)


def majority_baseline_f1(
    y_train: np.ndarray, y_test: np.ndarray
) -> tuple[float, object]:
    """Macro-F1 of always predicting y_train's most common class on y_test.

    Returns (macro_f1, majority_class). The majority class is taken from y_train
    (which may be the same array as y_test for an in-sample floor), and macro-F1 is
    sklearn's standard `f1_score(average="macro", zero_division=0)` over all label
    classes present in y_test. Empty y_test -> (nan, None).

    Note: this uses sklearn's macro average over the classes appearing in y_test.
    Callers that need a macro average restricted to a fixed class list (e.g. only
    classes present in a family) must compute that separately — this is the
    standard floor used by the holdout probes.
    """
    if len(y_test) == 0:
        return float("nan"), None
    majority_class = Counter(np.asarray(y_train).tolist()).most_common(1)[0][0]
    pred = np.full(len(y_test), majority_class)
    macro_f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
    return macro_f1, majority_class


def standardize(X_fit: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """Standardize feature columns using stats fit on X_fit; transform every array.

    Computes per-column mean and std (with a 1e-8 floor to avoid divide-by-zero on a
    constant column) on X_fit, then applies (X - mean) / std to X_fit and each array in
    `others`. Returns the transformed arrays in the same order, so:

        X_tr, X_te = standardize(X_tr, X_te)
        X_fit, X_val, X_te = standardize(X_fit, X_val, X_te)

    Replaces the manual mean(0) / (std(0) + 1e-8) blocks the probe runners inlined.
    """
    mean = X_fit.mean(0)
    std = X_fit.std(0) + 1e-8
    return tuple((arr - mean) / std for arr in (X_fit, *others))


def auroc_at_median(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary AUROC for a continuous target: above-median = positive class.

    Binarises a continuous ground truth (e.g. ΔΔG) at its own median and scores
    the continuous predictions against that. Returns NaN if the median split
    leaves only one class. Shared by the megascale stability regression probes.
    """
    med = np.median(y_true)
    binary = (y_true >= med).astype(int)
    if binary.sum() == 0 or (1 - binary).sum() == 0:
        return float("nan")
    return float(roc_auc_score(binary, y_pred))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: list[str] = MECHANISM_CLASSES,
) -> dict:
    """Compute macro-F1 and per-class AUROC.

    y_true and y_pred must contain string class labels (e.g. "GOF", "DN", "LOF").
    y_proba columns must be aligned to classes in the same order as classes.

    Returns {"macro_f1": float, "per_class_auroc": {cls: float|None}}.
    """
    if len(y_true) == 0:
        return {"macro_f1": None, "per_class_auroc": {c: None for c in classes}, "n": 0}
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    auroc: dict[str, float | None] = {}
    for col_idx, cls in enumerate(classes):
        y_bin = (y_true == cls).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            auroc[cls] = None
        else:
            auroc[cls] = float(roc_auc_score(y_bin, y_proba[:, col_idx]))
    out: dict = {"macro_f1": macro_f1, "per_class_auroc": auroc}
    if len(y_true) > 0:
        out["n"] = int(len(y_true))
    return out


def aggregate_folds(
    fold_list: list[dict], classes: list[str] = MECHANISM_CLASSES
) -> dict:
    """Aggregate per-fold dicts from compute_metrics into mean ± std."""
    if not fold_list:
        return {"error": "no folds"}
    out: dict = {}
    f1_vals = [f["macro_f1"] for f in fold_list if f.get("macro_f1") is not None]
    out["macro_f1_mean"] = float(np.mean(f1_vals)) if f1_vals else None
    out["macro_f1_std"] = float(np.std(f1_vals)) if f1_vals else None
    for cls in classes:
        vals = [
            f["per_class_auroc"][cls]
            for f in fold_list
            if f.get("per_class_auroc", {}).get(cls) is not None
        ]
        out[f"auroc_{cls}_mean"] = float(np.mean(vals)) if vals else None
        out[f"auroc_{cls}_std"] = float(np.std(vals)) if vals else None
    out["n_folds"] = len(fold_list)
    return out


def align_proba(
    proba: np.ndarray, clf_classes: np.ndarray, classes: list[str]
) -> np.ndarray:
    """Reorder classifier probability columns to match the canonical classes order."""
    cls_to_col = {cls: idx for idx, cls in enumerate(classes)}
    aligned = np.zeros((len(proba), len(classes)), dtype=np.float32)
    for clf_col, cls in enumerate(clf_classes):
        canonical_col = cls_to_col.get(cls)
        if canonical_col is not None:
            aligned[:, canonical_col] = proba[:, clf_col]
    return aligned
