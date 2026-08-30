"""Evaluation metrics, fold aggregation, and probability alignment."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

from esm2_mech.utils.classification import validate_classes, validate_observed_labels

# The `<metric>_mean` / `<metric>_std` pairs an aggregated arm carries summarize its
# cross-validation folds, not its model seeds. A result file states this alongside
# them so a reader never has to infer a spread's sampling unit from the key name.
FOLD_SAMPLING_UNIT = "cv_fold"


def mean_std_n(values) -> tuple[float, float, int]:
    """Return the mean, population standard deviation, and count of finite values."""
    clean = [value for value in values if value is not None and not np.isnan(value)]
    if not clean:
        return float("nan"), float("nan"), 0
    return float(np.mean(clean)), float(np.std(clean)), len(clean)


def null_standard_score(observed, null_values) -> dict:
    """Standardize an observation against complete null draws using sample spread."""
    values = np.asarray(null_values, dtype=float)
    if (
        observed is None
        or not np.isfinite(observed)
        or len(values) < 3
        or not np.isfinite(values).all()
    ):
        return {
            "state": "unavailable",
            "reason": "invalid_or_insufficient_null_draws",
            "z_score": None,
            "null_mean": None,
            "null_draw_std": None,
            "n_null_draws": int(len(values)),
            "sampling_unit": "null_draw",
        }
    null_mean = float(np.mean(values))
    null_spread = float(np.std(values, ddof=1))
    if null_spread == 0:
        return {
            "state": "unavailable",
            "reason": "zero_null_draw_spread",
            "z_score": None,
            "null_mean": null_mean,
            "null_draw_std": 0.0,
            "n_null_draws": int(len(values)),
            "sampling_unit": "null_draw",
        }
    return {
        "state": "available",
        "reason": None,
        "z_score": float((observed - null_mean) / null_spread),
        "null_mean": null_mean,
        "null_draw_std": null_spread,
        "n_null_draws": int(len(values)),
        "sampling_unit": "null_draw",
    }


def _metric_availability(available: bool, reason: str | None = None) -> dict:
    return {"available": available, "missing": not available, "reason": reason}


def majority_baseline_f1(
    y_train: np.ndarray,
    y_test: np.ndarray,
    classes: Sequence[object],
) -> tuple[float, object]:
    """Score a training-fold majority prediction against an explicit class list."""
    declared = validate_classes(classes)
    train = np.asarray(y_train)
    test = np.asarray(y_test)
    if len(train) == 0 or len(test) == 0:
        raise ValueError("majority baseline requires non-empty training and test labels")
    validate_observed_labels(train, declared, "y_train")
    validate_observed_labels(test, declared, "y_test")
    counts = Counter(train.tolist())
    largest = max(counts.values())
    tied = [class_name for class_name in declared if counts.get(class_name, 0) == largest]
    if len(tied) != 1:
        raise ValueError(f"training-fold majority class is tied: {tied!r}")
    majority_class = tied[0]
    predictions = np.full(len(test), majority_class, dtype=object)
    macro_f1 = float(
        f1_score(
            test,
            predictions,
            labels=declared,
            average="macro",
            zero_division=0,
        )
    )
    return macro_f1, majority_class


def training_frequency_reference(
    y_train: np.ndarray,
    n_test: int,
    classes: Sequence[object],
) -> tuple[np.ndarray, np.ndarray, object]:
    """Build majority predictions and class-prior probabilities from training labels."""
    declared, probabilities = _training_class_probabilities(y_train, n_test, classes)
    counts = probabilities * len(np.asarray(y_train))
    largest = float(counts.max())
    tied_indices = np.where(counts == largest)[0]
    if len(tied_indices) != 1:
        tied = [declared[index] for index in tied_indices]
        raise ValueError(f"training-fold majority class is tied: {tied!r}")
    probability_rows = np.repeat(probabilities[None, :], n_test, axis=0)
    majority_class = declared[int(tied_indices[0])]
    predictions = np.full(n_test, majority_class, dtype=object)
    return predictions, probability_rows, majority_class


def _training_class_probabilities(
    y_train: np.ndarray,
    n_test: int,
    classes: Sequence[object],
) -> tuple[list[object], np.ndarray]:
    """Return declared classes and their training-fold frequencies."""
    declared = validate_classes(classes)
    train = np.asarray(y_train)
    if len(train) == 0:
        raise ValueError("training-frequency reference requires training labels")
    if n_test <= 0:
        raise ValueError(f"training-frequency reference requires test rows, got {n_test}")
    validate_observed_labels(train, declared, "y_train")
    counts = np.array([(train == class_name).sum() for class_name in declared], dtype=float)
    return declared, counts / counts.sum()


def featureless_reference(
    y_train: np.ndarray,
    n_test: int,
    classes: Sequence[object],
    strategy: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a class-only prediction rule fitted to one training fold."""
    if strategy in ("most_frequent", "prior"):
        predictions, probabilities, _majority_class = training_frequency_reference(
            y_train, n_test, classes
        )
        return predictions, probabilities
    if strategy != "stratified":
        raise ValueError(f"unknown featureless-reference strategy: {strategy!r}")
    declared, class_probabilities = _training_class_probabilities(
        y_train, n_test, classes
    )
    random_generator = np.random.RandomState(seed)
    sampled_columns = random_generator.choice(
        len(declared), size=n_test, p=class_probabilities
    )
    sampled_predictions = np.array(
        [declared[column] for column in sampled_columns], dtype=object
    )
    sampled_probabilities = np.eye(len(declared), dtype=float)[sampled_columns]
    return sampled_predictions, sampled_probabilities


def family_frequency_reference(
    y_train: np.ndarray,
    train_families: np.ndarray,
    test_families: np.ndarray,
    classes: Sequence[object],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit family-specific class frequencies on training rows only.

    A test family absent from the training partition receives the global training
    frequencies. A tied argmax is unscorable because the analysis has no declared
    tie-breaking rule.
    """
    declared = validate_classes(classes)
    train = np.asarray(y_train)
    train_groups = np.asarray(train_families, dtype=object)
    test_groups = np.asarray(test_families, dtype=object)
    if len(train) == 0 or len(test_groups) == 0:
        raise ValueError("family-frequency reference requires training and test rows")
    if len(train_groups) != len(train):
        raise ValueError(
            f"train_families has {len(train_groups)} rows for {len(train)} labels"
        )
    validate_observed_labels(train, declared, "y_train")

    def _frequencies(mask: np.ndarray, family: object) -> tuple[np.ndarray, object]:
        counts = np.array(
            [(train[mask] == class_name).sum() for class_name in declared],
            dtype=float,
        )
        if counts.sum() == 0:
            raise RuntimeError("family-frequency reference selected no training rows")
        tied_columns = np.where(counts == counts.max())[0]
        if len(tied_columns) != 1:
            tied = [declared[column] for column in tied_columns]
            raise ValueError(
                f"training-fold family {family!r} has tied majority classes: {tied!r}"
            )
        return counts / counts.sum(), declared[int(tied_columns[0])]

    global_mask = np.ones(len(train), dtype=bool)
    global_frequencies, global_prediction = _frequencies(global_mask, "<global>")
    probabilities = np.empty((len(test_groups), len(declared)), dtype=float)
    predictions = np.empty(len(test_groups), dtype=object)
    fitted: dict[object, tuple[np.ndarray, object]] = {}
    training_family_set = set(train_groups.tolist())
    for row_index, family in enumerate(test_groups.tolist()):
        if family not in training_family_set:
            probabilities[row_index] = global_frequencies
            predictions[row_index] = global_prediction
            continue
        if family not in fitted:
            fitted[family] = _frequencies(train_groups == family, family)
        probabilities[row_index], predictions[row_index] = fitted[family]
    return predictions, probabilities


def standardize(X_fit: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """Standardize columns using statistics fitted only on ``X_fit``."""
    mean = X_fit.mean(0)
    std = X_fit.std(0) + 1e-8
    return tuple((array - mean) / std for array in (X_fit, *others))


def auroc_at_median(
    y_true: np.ndarray, y_pred: np.ndarray, median: float | None = None
) -> float:
    """Return binary AUROC after thresholding a continuous target at its median."""
    threshold = np.median(y_true) if median is None else median
    binary = (y_true >= threshold).astype(int)
    if binary.sum() == 0 or (1 - binary).sum() == 0:
        return float("nan")
    return float(roc_auc_score(binary, y_pred))


def binary_class_target(y_true: np.ndarray, class_name: object) -> np.ndarray | None:
    """Return a one-vs-rest target, or ``None`` when it is constant."""
    binary = (np.asarray(y_true) == class_name).astype(int)
    if binary.sum() == 0 or binary.sum() == len(binary):
        return None
    return binary


def fold_macro_f1(
    y_true: np.ndarray,
    block: np.ndarray,
    arm_pred: np.ndarray,
    classes: Sequence[object],
) -> float:
    """Return fixed-class macro-F1 for one fold or resampled fold block."""
    declared = validate_classes(classes)
    true_block = np.asarray(y_true)[block]
    prediction_block = np.asarray(arm_pred)[block]
    validate_observed_labels(true_block, declared, "y_true")
    validate_observed_labels(prediction_block, declared, "y_pred")
    return float(
        f1_score(
            true_block,
            prediction_block,
            labels=declared,
            average="macro",
            zero_division=0,
        )
    )


def imbalance_metrics(y_binary: np.ndarray, scores: np.ndarray) -> dict | None:
    """Return AUPRC and prevalence-matched PPV/NPV for a nonconstant target."""
    num_rows = len(y_binary)
    num_positive = int(y_binary.sum())
    if num_positive == 0 or num_positive == num_rows:
        return None
    prevalence = num_positive / num_rows
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    predicted_positive_count = int(round(prevalence * num_rows))
    if predicted_positive_count == 0 or predicted_positive_count == num_rows:
        ppv = None
        npv = None
    else:
        predicted_positive = order[:predicted_positive_count]
        predicted_negative = order[predicted_positive_count:]
        true_positives = int(y_binary[predicted_positive].sum())
        true_negatives = int((1 - y_binary[predicted_negative]).sum())
        ppv = true_positives / predicted_positive_count
        npv = true_negatives / (num_rows - predicted_positive_count)
    return {
        "auprc": float(average_precision_score(y_binary, scores)),
        "prevalence": float(prevalence),
        "ppv": None if ppv is None else float(ppv),
        "npv": None if npv is None else float(npv),
    }


def _empty_fold_metrics(classes: Sequence[object], reason: str) -> dict:
    declared = validate_classes(classes)
    per_class_null = {class_name: None for class_name in declared}
    unavailable = _metric_availability(False, reason)
    return {
        "macro_f1": None,
        "per_class_f1": dict(per_class_null),
        "balanced_accuracy": None,
        "confusion_matrix": None,
        "per_class_auroc": dict(per_class_null),
        "macro_auroc": None,
        "per_class_auprc": dict(per_class_null),
        "per_class_prevalence": dict(per_class_null),
        "per_class_ppv": dict(per_class_null),
        "per_class_npv": dict(per_class_null),
        "class_order": declared,
        "availability": {
            "macro_f1": dict(unavailable),
            "per_class_f1": {class_name: dict(unavailable) for class_name in declared},
            "balanced_accuracy": dict(unavailable),
            "confusion_matrix": dict(unavailable),
            "per_class_auroc": {class_name: dict(unavailable) for class_name in declared},
            "macro_auroc": dict(unavailable),
            "per_class_auprc": {class_name: dict(unavailable) for class_name in declared},
        },
        "n": 0,
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: Sequence[object],
) -> dict:
    """Compute every classification outcome against the caller's class list."""
    declared = validate_classes(classes)
    true = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    probabilities = np.asarray(y_proba, dtype=float)
    if len(true) == 0:
        return _empty_fold_metrics(declared, "empty_test_partition")
    if len(predicted) != len(true):
        raise ValueError(
            f"y_pred has {len(predicted)} rows for {len(true)} observed labels"
        )
    if probabilities.shape != (len(true), len(declared)):
        raise ValueError(
            "y_proba shape does not match labels and declared classes: "
            f"{probabilities.shape} vs ({len(true)}, {len(declared)})"
        )
    if not np.isfinite(probabilities).all():
        bad_rows = np.where(~np.isfinite(probabilities).all(axis=1))[0].tolist()
        raise ValueError(f"y_proba contains non-finite values in rows {bad_rows[:20]}")
    validate_observed_labels(true, declared, "y_true")
    validate_observed_labels(predicted, declared, "y_pred")

    per_class_f1_values = f1_score(
        true,
        predicted,
        labels=declared,
        average=None,
        zero_division=0,
    )
    per_class_f1 = {
        class_name: float(per_class_f1_values[index])
        for index, class_name in enumerate(declared)
    }
    macro_f1 = float(np.mean(per_class_f1_values))
    missing_test_classes = [
        class_name for class_name in declared if not np.any(true == class_name)
    ]
    if missing_test_classes:
        balanced_accuracy = None
        balanced_accuracy_availability = _metric_availability(
            False, "declared class absent from test block"
        )
        balanced_accuracy_availability["unavailable_classes"] = missing_test_classes
    else:
        per_class_recall = recall_score(
            true,
            predicted,
            labels=declared,
            average=None,
            zero_division=0,
        )
        balanced_accuracy = float(np.mean(per_class_recall))
        balanced_accuracy_availability = _metric_availability(True)
    matrix = confusion_matrix(true, predicted, labels=declared).astype(int).tolist()

    per_class_auroc: dict[object, float | None] = {}
    per_class_auprc: dict[object, float | None] = {}
    prevalence: dict[object, float | None] = {}
    ppv: dict[object, float | None] = {}
    npv: dict[object, float | None] = {}
    ranking_availability: dict[object, dict] = {}
    auprc_availability: dict[object, dict] = {}
    for column_index, class_name in enumerate(declared):
        binary = binary_class_target(true, class_name)
        if binary is None:
            per_class_auroc[class_name] = None
            per_class_auprc[class_name] = None
            prevalence[class_name] = None
            ppv[class_name] = None
            npv[class_name] = None
            reason = "constant_one_vs_rest_target"
            ranking_availability[class_name] = _metric_availability(False, reason)
            auprc_availability[class_name] = _metric_availability(False, reason)
            continue
        scores = probabilities[:, column_index]
        per_class_auroc[class_name] = float(roc_auc_score(binary, scores))
        imbalance = imbalance_metrics(binary, scores)
        if imbalance is None:
            raise RuntimeError("nonconstant one-vs-rest target produced no imbalance metrics")
        per_class_auprc[class_name] = imbalance["auprc"]
        prevalence[class_name] = imbalance["prevalence"]
        ppv[class_name] = imbalance["ppv"]
        npv[class_name] = imbalance["npv"]
        ranking_availability[class_name] = _metric_availability(True)
        auprc_availability[class_name] = _metric_availability(True)

    unavailable_auroc = [
        class_name for class_name in declared if per_class_auroc[class_name] is None
    ]
    macro_auroc = (
        None
        if unavailable_auroc
        else float(np.mean([per_class_auroc[class_name] for class_name in declared]))
    )
    return {
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "balanced_accuracy": balanced_accuracy,
        "confusion_matrix": matrix,
        "per_class_auroc": per_class_auroc,
        "macro_auroc": macro_auroc,
        "per_class_auprc": per_class_auprc,
        "per_class_prevalence": prevalence,
        "per_class_ppv": ppv,
        "per_class_npv": npv,
        "class_order": declared,
        "availability": {
            "macro_f1": _metric_availability(True),
            "per_class_f1": {
                class_name: _metric_availability(True) for class_name in declared
            },
            "balanced_accuracy": balanced_accuracy_availability,
            "confusion_matrix": _metric_availability(True),
            "per_class_auroc": ranking_availability,
            "macro_auroc": _metric_availability(
                not unavailable_auroc,
                None
                if not unavailable_auroc
                else f"unavailable classes: {unavailable_auroc!r}",
            ),
            "per_class_auprc": auprc_availability,
        },
        "n": int(len(true)),
    }


def _aggregate_scalar(values: list[float | None], requested_folds: int) -> dict:
    contributors = sum(value is not None for value in values)
    unavailable_folds = [
        fold_index for fold_index, value in enumerate(values) if value is None
    ]
    if contributors != requested_folds:
        return {
            "mean": None,
            "std": None,
            "contributing_folds": contributors,
            "missing": True,
            "reason": "metric unavailable in one or more required folds",
            "unavailable_folds": unavailable_folds,
        }
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "contributing_folds": contributors,
        "missing": False,
        "reason": None,
        "unavailable_folds": [],
    }


def empty_aggregate_metrics(
    classes: Sequence[object], requested_folds: int, reason: str
) -> dict:
    """Return explicit null aggregate fields for an unscorable or failed arm."""
    declared = validate_classes(classes)
    output = {
        "spread_sampling_unit": FOLD_SAMPLING_UNIT,
        "macro_f1_mean": None,
        "macro_f1_std": None,
        "balanced_accuracy_mean": None,
        "balanced_accuracy_std": None,
        "macro_auroc_mean": None,
        "macro_auroc_std": None,
        "confusion_matrix": None,
        "confusion_matrix_class_order": declared,
        "requested_folds": int(requested_folds),
        "completed_folds": 0,
        "n_folds": 0,
        "metric_availability": {},
    }
    scalar_names = ["macro_f1", "balanced_accuracy", "macro_auroc"]
    for metric_name in scalar_names:
        output["metric_availability"][metric_name] = {
            "available": False,
            "missing": True,
            "reason": reason,
            "contributing_folds": 0,
        }
    for class_name in declared:
        for prefix in ("f1", "auroc", "auprc", "prevalence", "ppv", "npv"):
            output[f"{prefix}_{class_name}_mean"] = None
            output[f"{prefix}_{class_name}_std"] = None
            output["metric_availability"][f"{prefix}_{class_name}"] = {
                "available": False,
                "missing": True,
                "reason": reason,
                "contributing_folds": 0,
            }
    return output


def aggregate_folds(
    fold_list: list[dict],
    classes: Sequence[object],
    requested_folds: int,
) -> dict:
    """Aggregate fold metrics without changing the fold or class denominator."""
    declared = validate_classes(classes)
    if len(fold_list) != requested_folds:
        raise ValueError(
            f"cannot aggregate {len(fold_list)} completed folds; expected {requested_folds}"
        )
    output = {
        "spread_sampling_unit": FOLD_SAMPLING_UNIT,
        "requested_folds": int(requested_folds),
        "completed_folds": int(len(fold_list)),
        "n_folds": int(len(fold_list)),
        "class_order": declared,
        "metric_availability": {},
    }
    scalar_sources = {
        "macro_f1": "macro_f1",
        "balanced_accuracy": "balanced_accuracy",
        "macro_auroc": "macro_auroc",
    }
    for output_name, source_name in scalar_sources.items():
        aggregate = _aggregate_scalar(
            [fold[source_name] for fold in fold_list], requested_folds
        )
        output[f"{output_name}_mean"] = aggregate["mean"]
        output[f"{output_name}_std"] = aggregate["std"]
        output["metric_availability"][output_name] = {
            "available": not aggregate["missing"],
            **aggregate,
        }

    nested_metrics = {
        "per_class_f1": "f1",
        "per_class_auroc": "auroc",
        "per_class_auprc": "auprc",
        "per_class_prevalence": "prevalence",
        "per_class_ppv": "ppv",
        "per_class_npv": "npv",
    }
    for class_name in declared:
        for nested_name, prefix in nested_metrics.items():
            aggregate = _aggregate_scalar(
                [fold[nested_name][class_name] for fold in fold_list], requested_folds
            )
            output[f"{prefix}_{class_name}_mean"] = aggregate["mean"]
            output[f"{prefix}_{class_name}_std"] = aggregate["std"]
            output["metric_availability"][f"{prefix}_{class_name}"] = {
                "available": not aggregate["missing"],
                **aggregate,
            }

    matrices = [np.asarray(fold["confusion_matrix"], dtype=int) for fold in fold_list]
    expected_shape = (len(declared), len(declared))
    if any(matrix.shape != expected_shape for matrix in matrices):
        raise ValueError("fold confusion matrix shape does not match declared classes")
    output["confusion_matrix"] = np.sum(matrices, axis=0).astype(int).tolist()
    output["confusion_matrix_class_order"] = declared
    output["metric_availability"]["confusion_matrix"] = {
        "available": True,
        "missing": False,
        "reason": None,
        "contributing_folds": requested_folds,
    }
    return output


def add_flat_class_metrics(
    fold_metrics: dict,
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: Sequence[object],
) -> None:
    """Add explicit flat per-class ranking metrics to an existing fold record."""
    declared = validate_classes(classes)
    true = np.asarray(y_true)
    probabilities = np.asarray(proba)
    if probabilities.shape != (len(true), len(declared)):
        raise ValueError("probability shape does not match labels and classes")
    validate_observed_labels(true, declared, "y_true")
    for column_index, class_name in enumerate(declared):
        binary = binary_class_target(true, class_name)
        if binary is None:
            for prefix in ("auroc", "auprc", "prevalence", "ppv", "npv"):
                fold_metrics[f"{prefix}_{class_name}"] = None
            continue
        scores = probabilities[:, column_index]
        fold_metrics[f"auroc_{class_name}"] = float(roc_auc_score(binary, scores))
        imbalance = imbalance_metrics(binary, scores)
        if imbalance is None:
            raise RuntimeError("nonconstant one-vs-rest target produced no imbalance metrics")
        for name in ("auprc", "prevalence", "ppv", "npv"):
            fold_metrics[f"{name}_{class_name}"] = imbalance[name]


def align_proba(
    proba: np.ndarray,
    classifier_classes: Sequence[object],
    classes: Sequence[object],
    *,
    allow_missing_classes: bool,
) -> np.ndarray:
    """Validate and align classifier probability columns to the declared classes."""
    declared = validate_classes(classes)
    fitted = list(classifier_classes)
    if len(set(fitted)) != len(fitted):
        raise ValueError(f"classifier classes contain duplicates: {fitted!r}")
    unexpected = sorted(set(fitted) - set(declared), key=repr)
    if unexpected:
        raise ValueError(f"classifier returned undeclared classes: {unexpected!r}")
    missing = [class_name for class_name in declared if class_name not in fitted]
    if missing and not allow_missing_classes:
        raise ValueError(f"classifier is missing required classes: {missing!r}")
    probabilities = np.asarray(proba, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(fitted):
        raise ValueError(
            "classifier probability width does not match classifier classes: "
            f"shape={probabilities.shape}, classes={fitted!r}"
        )
    if not np.isfinite(probabilities).all():
        bad_rows = np.where(~np.isfinite(probabilities).all(axis=1))[0].tolist()
        raise ValueError(
            f"classifier probabilities contain non-finite values in rows {bad_rows[:20]}"
        )
    aligned = np.zeros((len(probabilities), len(declared)), dtype=np.float32)
    declared_column = {class_name: index for index, class_name in enumerate(declared)}
    for classifier_column, class_name in enumerate(fitted):
        aligned[:, declared_column[class_name]] = probabilities[:, classifier_column]
    return aligned
