"""Shared contracts for classification classes, splits, and result status."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def validate_classes(classes: Sequence[object]) -> list[object]:
    """Return an ordered class list after checking the shared class contract."""
    declared = list(classes)
    if not declared:
        raise ValueError("classification classes must be non-empty")
    duplicates = [value for index, value in enumerate(declared) if value in declared[:index]]
    if duplicates:
        raise ValueError(f"classification classes contain duplicates: {duplicates!r}")
    return declared


def validate_observed_labels(
    values: Sequence[object], classes: Sequence[object], label: str
) -> None:
    """Raise when observed or predicted labels fall outside the declared classes."""
    declared = validate_classes(classes)
    observed = np.asarray(values)
    if observed.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional, got shape {observed.shape}")
    unexpected = sorted(set(observed.tolist()) - set(declared), key=repr)
    if unexpected:
        raise ValueError(f"{label} contains labels outside declared classes: {unexpected!r}")


def validate_classification_splits(
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    requested_folds: int,
    eligible_rows: Sequence[int],
    labels: Sequence[object],
    classes: Sequence[object],
    required_train_classes: Sequence[object] | None,
    required_test_classes: Sequence[object] | None,
    allow_missing_classifier_classes: bool,
    minimum_train_classes: int | None = None,
    minimum_test_classes: int | None = None,
    groups: Sequence[object] | None = None,
    held_out_unit: str | None = None,
) -> dict:
    """Validate a complete classification split set before any model is fitted."""
    declared = validate_classes(classes)
    labels_array = np.asarray(labels)
    num_rows = len(labels_array)
    validate_observed_labels(labels_array, declared, "labels")
    if requested_folds <= 0:
        raise ValueError(f"requested_folds must be positive, got {requested_folds}")

    eligible = np.asarray(eligible_rows, dtype=int)
    if eligible.ndim != 1:
        raise ValueError("eligible_rows must be one-dimensional")
    if len(np.unique(eligible)) != len(eligible):
        raise ValueError("eligible_rows contains duplicate indices")
    if np.any(eligible < 0) or np.any(eligible >= num_rows):
        raise ValueError("eligible_rows contains out-of-bounds indices")
    eligible_set = set(eligible.tolist())

    train_required = [] if required_train_classes is None else list(required_train_classes)
    test_required = [] if required_test_classes is None else list(required_test_classes)
    for requirement_name, required in (
        ("required_train_classes", train_required),
        ("required_test_classes", test_required),
    ):
        unexpected = sorted(set(required) - set(declared), key=repr)
        if unexpected:
            raise ValueError(
                f"{requirement_name} contains undeclared classes: {unexpected!r}"
            )

    groups_array = None if groups is None else np.asarray(groups, dtype=object)
    if groups_array is not None and len(groups_array) != num_rows:
        raise ValueError(
            f"groups has {len(groups_array)} rows for {num_rows} classification labels"
        )
    if groups_array is not None and held_out_unit is None:
        raise ValueError("held_out_unit is required when groups are supplied")
    group_count = (
        None
        if groups_array is None
        else int(len(set(groups_array[eligible].tolist())))
    )

    failures: list[dict] = []
    fold_records: list[dict] = []
    test_counts = np.zeros(num_rows, dtype=int)

    if len(splits) != requested_folds:
        failures.append(
            {
                "scope": "split_set",
                "reason": "wrong_fold_count",
                "requested_folds": int(requested_folds),
                "supplied_folds": int(len(splits)),
            }
        )

    for fold_index, (train_rows_raw, test_rows_raw) in enumerate(splits):
        record = {
            "fold": int(fold_index),
            "status": "valid",
            "failures": [],
            "missing_train_classes": [],
            "missing_test_classes": [],
            "crossing_groups": [],
        }
        try:
            train_rows = np.asarray(train_rows_raw, dtype=int)
            test_rows = np.asarray(test_rows_raw, dtype=int)
        except (TypeError, ValueError) as error:
            record["failures"].append(
                {"reason": "invalid_indices", "message": str(error)}
            )
            fold_records.append(record)
            failures.append({"scope": "fold", "fold": fold_index, **record["failures"][-1]})
            continue

        if train_rows.ndim != 1 or test_rows.ndim != 1:
            record["failures"].append({"reason": "indices_not_one_dimensional"})
            record["status"] = "unscorable"
            fold_records.append(record)
            failures.append(
                {
                    "scope": "fold",
                    "fold": fold_index,
                    "reason": "indices_not_one_dimensional",
                }
            )
            continue
        record["train_rows"] = int(len(train_rows))
        record["test_rows"] = int(len(test_rows))
        if len(np.unique(train_rows)) != len(train_rows):
            record["failures"].append({"reason": "duplicate_train_indices"})
        if len(np.unique(test_rows)) != len(test_rows):
            record["failures"].append({"reason": "duplicate_test_indices"})
        if (
            np.any(train_rows < 0)
            or np.any(train_rows >= num_rows)
            or np.any(test_rows < 0)
            or np.any(test_rows >= num_rows)
        ):
            record["failures"].append({"reason": "indices_out_of_bounds"})
            fold_records.append(record)
            for failure in record["failures"]:
                failures.append({"scope": "fold", "fold": fold_index, **failure})
            continue

        overlap = sorted(set(train_rows.tolist()) & set(test_rows.tolist()))
        if overlap:
            record["failures"].append(
                {"reason": "train_test_overlap", "rows": overlap[:20]}
            )
        if len(test_rows) == 0:
            record["failures"].append({"reason": "empty_test_partition"})

        ineligible_test = sorted(set(test_rows.tolist()) - eligible_set)
        if ineligible_test:
            record["failures"].append(
                {"reason": "ineligible_test_rows", "rows": ineligible_test[:20]}
            )
        test_counts[test_rows] += 1

        train_present = set(labels_array[train_rows].tolist())
        test_present = set(labels_array[test_rows].tolist())
        missing_train = [value for value in train_required if value not in train_present]
        missing_test = [value for value in test_required if value not in test_present]
        record["missing_train_classes"] = missing_train
        record["missing_test_classes"] = missing_test
        if missing_train:
            record["failures"].append(
                {"reason": "missing_train_classes", "classes": missing_train}
            )
        if missing_test:
            record["failures"].append(
                {"reason": "missing_test_classes", "classes": missing_test}
            )
        if minimum_train_classes is not None and len(train_present) < minimum_train_classes:
            record["failures"].append(
                {
                    "reason": "insufficient_train_classes",
                    "required": int(minimum_train_classes),
                    "observed": int(len(train_present)),
                }
            )
        if minimum_test_classes is not None and len(test_present) < minimum_test_classes:
            record["failures"].append(
                {
                    "reason": "insufficient_test_classes",
                    "required": int(minimum_test_classes),
                    "observed": int(len(test_present)),
                }
            )

        if groups_array is not None:
            crossing = sorted(
                set(groups_array[train_rows].tolist())
                & set(groups_array[test_rows].tolist()),
                key=repr,
            )
            if crossing:
                record["crossing_groups"] = crossing[:20]
                record["failures"].append(
                    {
                        "reason": "held_out_group_crossing",
                        "held_out_unit": held_out_unit,
                        "groups": crossing[:20],
                    }
                )

        if record["failures"]:
            record["status"] = "unscorable"
            for failure in record["failures"]:
                failures.append({"scope": "fold", "fold": fold_index, **failure})
        fold_records.append(record)

    missing_test_rows = eligible[test_counts[eligible] == 0].tolist()
    repeated_test_rows = eligible[test_counts[eligible] > 1].tolist()
    if missing_test_rows:
        failures.append(
            {
                "scope": "split_set",
                "reason": "eligible_rows_missing_from_test_folds",
                "rows": missing_test_rows[:50],
                "count": len(missing_test_rows),
            }
        )
    if repeated_test_rows:
        failures.append(
            {
                "scope": "split_set",
                "reason": "eligible_rows_repeated_in_test_folds",
                "rows": repeated_test_rows[:50],
                "count": len(repeated_test_rows),
            }
        )

    return {
        "status": "valid" if not failures else "unscorable",
        "requested_folds": int(requested_folds),
        "supplied_folds": int(len(splits)),
        "eligible_rows": int(len(eligible)),
        "eligible_row_ids": eligible.tolist(),
        "classes": declared,
        "held_out_unit": held_out_unit,
        "group_count": group_count,
        "required_train_classes": train_required,
        "required_test_classes": test_required,
        "minimum_train_classes": minimum_train_classes,
        "minimum_test_classes": minimum_test_classes,
        "allow_missing_classifier_classes": bool(allow_missing_classifier_classes),
        "folds": fold_records,
        "failures": failures,
    }


def require_valid_split_contract(contract: dict) -> None:
    """Raise when a caller attempts fitting without a successful preflight."""
    if contract.get("status") != "valid":
        raise ValueError("classification split contract is not valid")


def validate_complete_classification_splits(
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    requested_folds: int,
    eligible_rows: Sequence[int],
    labels: Sequence[object],
    classes: Sequence[object],
    groups: Sequence[object] | None,
    held_out_unit: str | None,
) -> dict:
    """Validate a split policy requiring every declared class in train and test."""
    declared = validate_classes(classes)
    return validate_classification_splits(
        splits,
        requested_folds=requested_folds,
        eligible_rows=eligible_rows,
        labels=labels,
        classes=declared,
        required_train_classes=declared,
        required_test_classes=declared,
        allow_missing_classifier_classes=False,
        groups=groups,
        held_out_unit=held_out_unit,
    )


def blocked_interval(point: float | None, reason: str) -> dict:
    """Return the result representation for an interval blocked by audit item 1.4."""
    return {
        "point": point,
        "ci_low": None,
        "ci_high": None,
        "missing": True,
        "reason": reason,
    }
