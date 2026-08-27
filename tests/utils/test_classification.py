"""Tests for classification class, split, and result-status contracts."""

import numpy as np
import pytest

from esm2_mech.utils.classification import (
    validate_classes,
    validate_classification_splits,
    validate_complete_classification_splits,
)
from esm2_mech.utils.constants import DN, GOF, LOF, MECHANISM_CLASSES
from esm2_mech.utils.splits import gene_split_cv


def _complete_contract(labels, genes, splits):
    return validate_complete_classification_splits(
        splits,
        requested_folds=5,
        eligible_rows=np.arange(len(labels)),
        labels=labels,
        classes=MECHANISM_CLASSES,
        groups=genes,
        held_out_unit="gene",
    )


def test_classes_must_be_nonempty_and_unique():
    with pytest.raises(ValueError, match="non-empty"):
        validate_classes([])
    with pytest.raises(ValueError, match="duplicates"):
        validate_classes([GOF, GOF])


def test_valid_split_set_preserves_all_folds_and_rows():
    genes = np.repeat([f"G{index}" for index in range(15)], 3)
    labels = np.tile(np.array([GOF, DN, LOF]), 15)
    splits = gene_split_cv(genes, n_folds=5, seed=3)
    contract = _complete_contract(labels, genes, splits)
    assert contract["status"] == "valid"
    assert contract["supplied_folds"] == 5
    assert contract["held_out_unit"] == "gene"
    assert contract["group_count"] == 15
    assert all(fold["status"] == "valid" for fold in contract["folds"])


def test_wrong_fold_count_makes_complete_arm_unscorable():
    genes = np.repeat([f"G{index}" for index in range(15)], 3)
    labels = np.tile(np.array([GOF, DN, LOF]), 15)
    contract = _complete_contract(labels, genes, gene_split_cv(genes)[:4])
    assert contract["status"] == "unscorable"
    assert any(failure["reason"] == "wrong_fold_count" for failure in contract["failures"])


def test_missing_classes_are_reported_for_every_invalid_fold():
    labels = np.array([GOF, GOF, DN, DN, LOF, LOF])
    splits = [
        (np.array([0, 1, 2, 3]), np.array([4, 5])),
        (np.array([2, 3, 4, 5]), np.array([0, 1])),
        (np.array([0, 1, 4, 5]), np.array([2, 3])),
    ]
    contract = validate_complete_classification_splits(
        splits,
        requested_folds=3,
        eligible_rows=np.arange(len(labels)),
        labels=labels,
        classes=MECHANISM_CLASSES,
        groups=None,
        held_out_unit=None,
    )
    assert contract["status"] == "unscorable"
    assert all(fold["missing_test_classes"] for fold in contract["folds"])


def test_within_family_policy_allows_class_incomplete_test_folds():
    labels = np.array([GOF, GOF, DN, DN, LOF, LOF])
    splits = [
        (np.array([0, 1, 2, 3]), np.array([4, 5])),
        (np.array([2, 3, 4, 5]), np.array([0, 1])),
        (np.array([0, 1, 4, 5]), np.array([2, 3])),
    ]
    contract = validate_classification_splits(
        splits,
        requested_folds=3,
        eligible_rows=np.arange(len(labels)),
        labels=labels,
        classes=MECHANISM_CLASSES,
        required_train_classes=None,
        required_test_classes=None,
        allow_missing_classifier_classes=True,
        minimum_train_classes=2,
        groups=None,
        held_out_unit=None,
    )
    assert contract["status"] == "valid"


def test_group_crossing_is_reported_before_fitting():
    labels = np.array([GOF, DN, LOF, GOF, DN, LOF])
    groups = np.array(["shared", "A", "B", "shared", "C", "D"])
    splits = [(np.array([0, 1, 2]), np.array([3, 4, 5]))]
    contract = validate_complete_classification_splits(
        splits,
        requested_folds=1,
        eligible_rows=np.arange(len(labels)),
        labels=labels,
        classes=MECHANISM_CLASSES,
        groups=groups,
        held_out_unit="gene",
    )
    assert contract["status"] == "unscorable"
    assert any(
        failure["reason"] == "held_out_group_crossing"
        for failure in contract["failures"]
    )


def test_non_vector_fold_indices_are_recorded_as_unscorable():
    labels = np.array([GOF, DN, LOF, GOF, DN, LOF])
    splits = [
        (
            np.array([[0, 1, 2]]),
            np.array([[3, 4, 5]]),
        )
    ]
    contract = validate_complete_classification_splits(
        splits,
        requested_folds=1,
        eligible_rows=np.arange(len(labels)),
        labels=labels,
        classes=MECHANISM_CLASSES,
        groups=None,
        held_out_unit=None,
    )
    assert contract["status"] == "unscorable"
    assert contract["folds"][0]["status"] == "unscorable"
    assert contract["folds"][0]["failures"] == [
        {"reason": "indices_not_one_dimensional"}
    ]
