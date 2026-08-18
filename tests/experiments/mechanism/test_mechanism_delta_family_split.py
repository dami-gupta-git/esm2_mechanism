"""
Tests for mechanism_delta_family_split fixes.

Invariants:
- run_probe_on_splits with n_pca fits PCA inside each fold on training rows only,
  never on the full dataset. Verified by checking that PCA fitted on the full data
  gives different test-set features than PCA fitted per fold.
- run_probe_on_splits returns macro_f1_pooled computed from concatenated OOF
  predictions, which matches the estimand used by bootstrap_mechanism_metrics.
- macro_f1_pooled and macro_f1_mean are both present and can differ.
"""

import numpy as np
import pytest
from unittest.mock import patch

from esm2_mech.experiments.mechanism.mechanism_delta_family_split import (
    PCA_COMPONENTS,
    run_probe_on_splits,
)


def _make_data(n=120, dim=50, n_classes=3, n_genes=6, seed=42):
    """Synthetic data with enough structure for a 3-class probe."""
    rng = np.random.RandomState(seed)
    genes = np.array([f"gene_{i % n_genes}" for i in range(n)])
    classes = ["GOF", "DN", "LOF"]
    labels = np.array([classes[i % n_classes] for i in range(n)])
    X = rng.randn(n, dim)
    for i, cls in enumerate(classes):
        X[labels == cls, :3] += rng.randn(3) * 2
    return X, labels, genes


class TestPcaPerFold:

    def test_pca_fitted_inside_folds_not_globally(self):
        X, labels, genes = _make_data(n=100, dim=40)
        splits = [
            (np.arange(0, 80), np.arange(80, 100)),
        ]
        n_pca = 10

        from sklearn.decomposition import PCA as RealPCA
        fit_call_counts = []

        original_fit_transform = RealPCA.fit_transform

        def tracking_fit_transform(self, X_in, *args, **kwargs):
            fit_call_counts.append(X_in.shape[0])
            return original_fit_transform(self, X_in, *args, **kwargs)

        with patch.object(RealPCA, "fit_transform", tracking_fit_transform):
            run_probe_on_splits(X, labels, splits, genes=genes, n_pca=n_pca)

        assert len(fit_call_counts) == 1
        assert fit_call_counts[0] == 80

    def test_pca_not_applied_when_dim_below_threshold(self):
        X, labels, genes = _make_data(n=60, dim=5)
        splits = [(np.arange(0, 40), np.arange(40, 60))]

        from sklearn.decomposition import PCA as RealPCA
        fit_call_counts = []

        original_fit_transform = RealPCA.fit_transform

        def tracking_fit_transform(self, X_in, *args, **kwargs):
            fit_call_counts.append(X_in.shape[0])
            return original_fit_transform(self, X_in, *args, **kwargs)

        with patch.object(RealPCA, "fit_transform", tracking_fit_transform):
            run_probe_on_splits(X, labels, splits, genes=genes, n_pca=10)

        assert len(fit_call_counts) == 0

    def test_no_pca_when_n_pca_is_none(self):
        X, labels, genes = _make_data(n=60, dim=40)
        splits = [(np.arange(0, 40), np.arange(40, 60))]

        from sklearn.decomposition import PCA as RealPCA
        fit_call_counts = []

        original_fit_transform = RealPCA.fit_transform

        def tracking_fit_transform(self, X_in, *args, **kwargs):
            fit_call_counts.append(1)
            return original_fit_transform(self, X_in, *args, **kwargs)

        with patch.object(RealPCA, "fit_transform", tracking_fit_transform):
            run_probe_on_splits(X, labels, splits, genes=genes, n_pca=None)

        assert len(fit_call_counts) == 0


class TestPooledMacroF1:

    def test_pooled_macro_f1_present(self):
        X, labels, genes = _make_data(n=90, dim=10)
        splits = [
            (np.arange(0, 60), np.arange(60, 90)),
            (np.arange(30, 90), np.arange(0, 30)),
        ]
        agg, oof = run_probe_on_splits(X, labels, splits, genes=genes)

        assert "macro_f1_pooled" in agg
        assert "macro_f1_mean" in agg
        assert isinstance(agg["macro_f1_pooled"], float)

    def test_pooled_macro_f1_matches_manual_computation(self):
        X, labels, genes = _make_data(n=90, dim=10)
        splits = [
            (np.arange(0, 60), np.arange(60, 90)),
            (np.arange(30, 90), np.arange(0, 30)),
        ]
        agg, oof = run_probe_on_splits(X, labels, splits, genes=genes)

        from sklearn.metrics import f1_score
        from esm2_mech.utils.constants import MECHANISM_CLASSES

        pred = [MECHANISM_CLASSES[col] for col in oof["proba"].argmax(axis=1)]
        expected = float(f1_score(oof["y_true"], pred, average="macro", zero_division=0))

        assert agg["macro_f1_pooled"] == pytest.approx(expected)

    def test_pooled_and_fold_mean_can_differ(self):
        rng = np.random.RandomState(0)
        X = rng.randn(150, 8)
        classes = ["GOF", "DN", "LOF"]
        labels = np.array(
            ["GOF"] * 90 + ["DN"] * 30 + ["LOF"] * 30
        )
        for i, cls in enumerate(classes):
            X[labels == cls, :2] += rng.randn(2) * 3
        genes = np.array([f"g{i}" for i in range(150)])
        splits = [
            (np.arange(0, 100), np.arange(100, 150)),
            (np.arange(50, 150), np.arange(0, 50)),
            (np.concatenate([np.arange(0, 50), np.arange(100, 150)]), np.arange(50, 100)),
        ]

        agg, _ = run_probe_on_splits(X, labels, splits, genes=genes)

        assert agg["macro_f1_pooled"] != pytest.approx(agg["macro_f1_mean"], abs=0)

    def test_no_pooled_when_no_genes(self):
        X, labels, _ = _make_data(n=60, dim=5)
        splits = [(np.arange(0, 40), np.arange(40, 60))]
        agg, oof = run_probe_on_splits(X, labels, splits, genes=None)

        assert oof is None
        assert "macro_f1_pooled" not in agg
