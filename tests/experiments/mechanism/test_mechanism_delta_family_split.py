"""
Tests for the mechanism experiment's probe.

Invariants:
- run_logreg_pca_cv fits PCA inside each fold on training rows only, never on the
  full dataset, and skips PCA when the feature is already narrower than n_pca.
- The out-of-fold predictions carry the fold each row was scored in, so the metrics
  can be computed within fold instead of over the pooled concatenation.
- The reported macro-F1 is the mean over folds, and no pooled macro-F1 is emitted.
"""

import numpy as np
import pytest
from unittest.mock import patch

from esm2_mech.utils.probes import run_logreg_pca_cv


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


def _count_pca_fits(call):
    """Run `call` and return the training-row counts PCA was fitted on."""
    from sklearn.decomposition import PCA as RealPCA

    fit_call_counts = []
    original_fit_transform = RealPCA.fit_transform

    def tracking_fit_transform(self, X_in, *args, **kwargs):
        fit_call_counts.append(X_in.shape[0])
        return original_fit_transform(self, X_in, *args, **kwargs)

    with patch.object(RealPCA, "fit_transform", tracking_fit_transform):
        call()
    return fit_call_counts


class TestPcaPerFold:

    def test_pca_fitted_inside_folds_not_globally(self):
        X, labels, genes = _make_data(n=100, dim=40)
        splits = [(np.arange(0, 80), np.arange(80, 100))]
        counts = _count_pca_fits(
            lambda: run_logreg_pca_cv(X, labels, splits, genes=genes, n_pca=10)
        )
        assert counts == [80]

    def test_pca_not_applied_when_dim_below_threshold(self):
        X, labels, genes = _make_data(n=60, dim=5)
        splits = [(np.arange(0, 40), np.arange(40, 60))]
        counts = _count_pca_fits(
            lambda: run_logreg_pca_cv(X, labels, splits, genes=genes, n_pca=10)
        )
        assert counts == []

    def test_no_pca_when_n_pca_is_none(self):
        X, labels, genes = _make_data(n=60, dim=40)
        splits = [(np.arange(0, 40), np.arange(40, 60))]
        counts = _count_pca_fits(
            lambda: run_logreg_pca_cv(X, labels, splits, genes=genes, n_pca=None)
        )
        assert counts == []


class TestOutOfFoldCarriesItsFold:

    def _two_fold_run(self):
        X, labels, genes = _make_data(n=90, dim=10)
        splits = [
            (np.arange(0, 60), np.arange(60, 90)),
            (np.arange(30, 90), np.arange(0, 30)),
        ]
        return run_logreg_pca_cv(
            X, labels, splits, genes=genes, return_oof=True
        )

    def test_every_oof_row_records_its_fold(self):
        agg, oof = self._two_fold_run()
        assert len(oof["folds"]) == len(oof["row_ids"])
        assert sorted(set(oof["folds"].tolist())) == [0, 1]

    def test_each_row_appears_once_under_one_fold(self):
        _, oof = self._two_fold_run()
        assert len(set(oof["row_ids"].tolist())) == len(oof["row_ids"])
        for fold in (0, 1):
            rows = oof["row_ids"][oof["folds"] == fold]
            assert len(rows) == 30

    def test_macro_f1_is_the_fold_mean_and_nothing_is_pooled(self):
        agg, _ = self._two_fold_run()
        assert "macro_f1_mean" in agg
        assert "macro_f1_pooled" not in agg

    def test_no_oof_when_genes_are_absent(self):
        X, labels, _ = _make_data(n=60, dim=5)
        splits = [(np.arange(0, 40), np.arange(40, 60))]
        agg, oof = run_logreg_pca_cv(X, labels, splits, genes=None, return_oof=True)
        assert oof is None
        assert "macro_f1_mean" in agg
