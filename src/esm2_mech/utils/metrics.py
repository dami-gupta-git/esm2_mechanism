"""Evaluation metrics: per-fold computation, fold aggregation, probability alignment."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from esm2_mech.utils.constants import MECHANISM_CLASSES


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
