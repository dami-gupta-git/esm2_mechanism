"""Evaluation metrics: per-fold computation, fold aggregation, probability alignment."""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

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


def auroc_at_median(
    y_true: np.ndarray, y_pred: np.ndarray, median: float | None = None
) -> float:
    """Binary AUROC for a continuous target: above-median = positive class.

    Binarises a continuous ground truth (e.g. ΔΔG) at a fixed median and scores
    the continuous predictions against that. When `median` is None (legacy
    behaviour), the median is computed from y_true, which gives each fold a
    different binary threshold. Pass a precomputed global median so every fold
    predicts the same binary outcome.
    """
    med = np.median(y_true) if median is None else median
    binary = (y_true >= med).astype(int)
    if binary.sum() == 0 or (1 - binary).sum() == 0:
        return float("nan")
    return float(roc_auc_score(binary, y_pred))


def binary_class_target(y_true: np.ndarray, cls: str) -> np.ndarray | None:
    """One-vs-rest 0/1 target for `cls`, or None when the class is absent or universal.

    AUROC, AUPRC, PPV and NPV are all undefined on a degenerate target, so every
    per-class scorer needs this same guard. It lives here so the four call sites
    (compute_metrics, and the AUROC/AUPRC/macro-AUROC scorers in utils.bootstrap)
    cannot drift apart on what counts as scorable.
    """
    y_bin = (np.asarray(y_true) == cls).astype(int)
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return None
    return y_bin


def fold_macro_f1(
    y_true: np.ndarray,
    block: np.ndarray,
    arm_pred: np.ndarray,
    classes: list[str] = MECHANISM_CLASSES,
) -> float | None:
    """Macro-F1 on one fold's block, or None when the block cannot score every class.

    Pins `labels=classes` so the average always runs over the same class set rather
    than whichever classes happen to appear in the block, and refuses (returns None)
    a block missing one of them rather than silently zero-filling it via
    zero_division. This gives macro-F1 the same discard contract as the per-class
    scorers built on binary_class_target: a resample that loses a class is a
    different statistic, not a noisier draw of the same one, and the caller
    (score_within_folds) drops it rather than mixing it into the average.
    """
    present = set(np.unique(np.asarray(y_true)[block]))
    if not present.issuperset(classes):
        return None
    return float(
        f1_score(
            np.asarray(y_true)[block], np.asarray(arm_pred)[block],
            labels=list(classes), average="macro", zero_division=0,
        )
    )


def imbalance_metrics(y_bin: np.ndarray, scores: np.ndarray) -> dict | None:
    """AUPRC, its prevalence baseline, and PPV/NPV at the class-prevalence operating point.

    AUROC's no-signal value is 0.5 regardless of class balance, so it reads the same
    for a 9%-prevalence class as for a balanced one. AUPRC's no-signal value is the
    class prevalence itself, which is why the baseline is returned alongside it — the
    pair is what makes the number readable.

    PPV/NPV need an operating point. This uses the prevalence-matched one: label the
    top `round(prevalence * n)` scores positive, so the predicted positive rate equals
    the observed one. That is threshold-free in the sense that matters here — it does
    not assume the scores are calibrated probabilities, only that they rank.

    Returns None when y_bin is degenerate (all one class), where none of these are
    defined; callers store that as a missing value rather than a number.
    """
    n = len(y_bin)
    n_pos = int(y_bin.sum())
    if n_pos == 0 or n_pos == n:
        return None
    prevalence = n_pos / n
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    k = int(round(prevalence * n))
    if k == 0 or k == n:
        ppv = npv = None
    else:
        predicted_positive = order[:k]
        predicted_negative = order[k:]
        true_positives = int(y_bin[predicted_positive].sum())
        true_negatives = int((1 - y_bin[predicted_negative]).sum())
        ppv = true_positives / k
        npv = true_negatives / (n - k)
    return {
        "auprc": float(average_precision_score(y_bin, scores)),
        "prevalence": float(prevalence),
        "ppv": None if ppv is None else float(ppv),
        "npv": None if npv is None else float(npv),
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: list[str] = MECHANISM_CLASSES,
) -> dict:
    """Compute macro-F1, per-class AUROC, and the imbalance metrics.

    y_true and y_pred must contain string class labels (e.g. "GOF", "DN", "LOF").
    y_proba columns must be aligned to classes in the same order as classes.

    Returns macro_f1 plus per-class AUROC, AUPRC, prevalence (the AUPRC baseline),
    PPV and NPV — see `imbalance_metrics` for what the last four mean and why AUROC
    alone is not enough at 9–15% prevalence.
    """
    empty_per_class = {c: None for c in classes}
    if len(y_true) == 0:
        return {
            "macro_f1": None,
            "per_class_auroc": dict(empty_per_class),
            "per_class_auprc": dict(empty_per_class),
            "per_class_prevalence": dict(empty_per_class),
            "per_class_ppv": dict(empty_per_class),
            "per_class_npv": dict(empty_per_class),
            "n": 0,
        }
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    auroc: dict[str, float | None] = {}
    auprc: dict[str, float | None] = {}
    prevalence: dict[str, float | None] = {}
    ppv: dict[str, float | None] = {}
    npv: dict[str, float | None] = {}
    for col_idx, cls in enumerate(classes):
        y_bin = binary_class_target(y_true, cls)
        if y_bin is None:
            auroc[cls] = None
            y_bin = (y_true == cls).astype(int)  # imbalance_metrics returns None on it
        else:
            auroc[cls] = float(roc_auc_score(y_bin, y_proba[:, col_idx]))
        imbalance = imbalance_metrics(y_bin, y_proba[:, col_idx])
        auprc[cls] = None if imbalance is None else imbalance["auprc"]
        prevalence[cls] = None if imbalance is None else imbalance["prevalence"]
        ppv[cls] = None if imbalance is None else imbalance["ppv"]
        npv[cls] = None if imbalance is None else imbalance["npv"]
    return {
        "macro_f1": macro_f1,
        "per_class_auroc": auroc,
        "per_class_auprc": auprc,
        "per_class_prevalence": prevalence,
        "per_class_ppv": ppv,
        "per_class_npv": npv,
        "n": int(len(y_true)),
    }


def aggregate_folds(
    fold_list: list[dict], classes: list[str] = MECHANISM_CLASSES
) -> dict:
    """Aggregate per-fold dicts from compute_metrics into mean ± std."""
    if not fold_list:
        return {"error": "no folds"}
    out: dict = {}
    f1_mean, f1_std, f1_n = mean_std_n(
        [f["macro_f1"] for f in fold_list if f.get("macro_f1") is not None]
    )
    out["macro_f1_mean"] = f1_mean if f1_n else None
    out["macro_f1_std"] = f1_std if f1_n else None
    # nested key in the fold dict -> flat prefix in the aggregate
    per_class_keys = {
        "per_class_auroc": "auroc",
        "per_class_auprc": "auprc",
        "per_class_prevalence": "prevalence",
        "per_class_ppv": "ppv",
        "per_class_npv": "npv",
    }
    for cls in classes:
        for nested_key, prefix in per_class_keys.items():
            vals = [
                f[nested_key][cls]
                for f in fold_list
                if f.get(nested_key, {}).get(cls) is not None
            ]
            mean, std, count = mean_std_n(vals)
            out[f"{prefix}_{cls}_mean"] = mean if count else None
            out[f"{prefix}_{cls}_std"] = std if count else None
    out["n_folds"] = len(fold_list)
    return out


def add_flat_class_metrics(
    fold_metrics: dict,
    y_true: np.ndarray,
    proba: np.ndarray,
    classes,
) -> None:
    """Write per-class AUROC/AUPRC/prevalence/PPV/NPV into a flat per-fold dict.

    In place, as `auroc_<cls>`, `auprc_<cls>`, `prevalence_<cls>`, `ppv_<cls>`,
    `npv_<cls>`. This is the flat-key counterpart of `compute_metrics` for the probe
    runners in `utils/probes.py`, which build `{"macro_f1": ..., "auroc_GOF": ...}`
    dicts for `aggregate_fold_dicts`. Degenerate classes (absent from the fold, or
    the only class in it) write no key at all, which is what that aggregator treats
    as missing.

    `y_true` must be comparable to the entries of `classes` by equality — string
    labels against string classes, or integer codes against the integer positions
    a LabelEncoder assigned them. `proba` columns must be aligned to `classes`.
    """
    for col_idx, cls in enumerate(classes):
        y_bin = (y_true == cls).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        fold_metrics[f"auroc_{cls}"] = float(roc_auc_score(y_bin, proba[:, col_idx]))
        imbalance = imbalance_metrics(y_bin, proba[:, col_idx])
        if imbalance is None:
            continue
        fold_metrics[f"auprc_{cls}"] = imbalance["auprc"]
        fold_metrics[f"prevalence_{cls}"] = imbalance["prevalence"]
        for name in ("ppv", "npv"):
            if imbalance[name] is not None:
                fold_metrics[f"{name}_{cls}"] = imbalance[name]


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
