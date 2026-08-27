"""Regression tests for proteome callers of the shared metric-layer runners."""

import numpy as np
from sklearn.preprocessing import LabelEncoder

from esm2_mech.experiments.proteome_features import clinical_utility
from esm2_mech.experiments.proteome_features import proteome_mechanism
from esm2_mech.utils.constants import MECHANISM_CLASSES


def test_clinical_utility_passes_a_valid_split_contract(monkeypatch):
    labels = np.array(["DN"] * 5 + ["GOF"] * 5 + ["LOF"] * 5)
    encoder = LabelEncoder().fit(labels)
    encoded = encoder.transform(labels)
    test_folds = [np.array([fold, fold + 5, fold + 10]) for fold in range(5)]
    captured = {}

    monkeypatch.setattr(
        clinical_utility,
        "build_family_folds",
        lambda families, n_folds, rng: test_folds,
    )

    def fake_runner(X, y, splits, classes, split_contract, **kwargs):
        captured["classes"] = classes
        captured["split_contract"] = split_contract
        return {}, {
            "row_ids": np.arange(len(y)),
            "proba": np.full((len(y), len(classes)), 1 / len(classes)),
        }

    monkeypatch.setattr(clinical_utility, "_shared_run_mlp_cv", fake_runner)
    probabilities = clinical_utility.run_mlp_cv(
        np.ones((15, 2)),
        encoded,
        np.array([f"family-{index}" for index in range(15)]),
        [0, 1],
        np.array([f"gene-{index}" for index in range(15)]),
        encoder,
    )

    assert probabilities.shape == (15, 3)
    assert captured["classes"] == list(encoder.classes_)
    assert captured["split_contract"]["status"] == "valid"


def _valid_contract(classes):
    return {
        "status": "valid",
        "classes": list(classes),
        "eligible_rows": 12,
        "eligible_row_ids": list(range(12)),
        "requested_folds": 3,
        "allow_missing_classifier_classes": False,
    }


def test_proteome_family_wrappers_disable_per_gene_scoring(monkeypatch):
    features = np.ones((12, 2))
    labels = np.array(MECHANISM_CLASSES * 4)
    groups = np.array([f"family-{index}" for index in range(12)])
    calls = []

    monkeypatch.setattr(
        proteome_mechanism,
        "validate_complete_classification_splits",
        lambda *args, **kwargs: _valid_contract(MECHANISM_CLASSES),
    )

    def fake_runner(*args, **kwargs):
        calls.append(kwargs)
        return {"status": "success"}, None

    monkeypatch.setattr(proteome_mechanism, "run_mlp_cv", fake_runner)
    monkeypatch.setattr(proteome_mechanism, "run_logreg_cv", fake_runner)
    monkeypatch.setattr(proteome_mechanism, "run_histgb_cv", fake_runner)

    proteome_mechanism.run_family_split_mlp(
        features, labels, groups, (8,), 3, 0, "mlp", compute_ci=False
    )
    proteome_mechanism.run_family_split_logreg(
        features, labels, groups, 3, 0, "logreg", compute_ci=False
    )
    proteome_mechanism.run_family_split_histgb(
        features, labels, groups, 3, 0, "histgb", compute_ci=False
    )
    proteome_mechanism.run_observed_subset_arm(
        features,
        labels,
        groups,
        3,
        0,
        "observed",
        fake_runner,
        compute_ci=False,
    )

    assert len(calls) == 4
    assert all(call["compute_per_gene"] is False for call in calls)
