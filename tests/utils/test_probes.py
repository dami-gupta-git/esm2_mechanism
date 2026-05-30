"""
Tests for esm2_mech.utils.probes (probe runners).

Invariants:
- run_logreg_cv: returns dict with macro_f1_mean and per-class auroc keys
- run_logreg_cv: empty splits returns error key
- run_logreg_cv: recovers signal on separable data
- run_logreg_cv: per_gene_f1 keys present when genes provided
- run_logreg_binary_cv: returns auroc_mean > 0.5 on separable data
- run_logreg_binary_cv: empty splits returns empty dict
- run_mlp_binary_cv: returns auroc_mean on separable binary data
- run_mlp_binary_cv: empty splits returns empty dict
"""

import numpy as np
import pytest

from esm2_mech.utils.probes import (
    run_logreg_cv,
    run_logreg_binary_cv,
    run_mlp_binary_cv,
)
from esm2_mech.utils.splits import gene_split_cv
from esm2_mech.utils.constants import MECHANISM_CLASSES, GOF, DN, LOF


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _multiclass_data(seed=0):
    rng = np.random.RandomState(seed)
    n = 300
    y = np.array([GOF, DN, LOF] * 100)
    X = rng.randn(n, 20) + np.array([MECHANISM_CLASSES.index(c) for c in y])[:, None] * 2.0
    genes = np.array([f"G{i % 30}" for i in range(n)])
    splits = gene_split_cv(genes, n_folds=5, seed=seed)
    return X, y, splits, genes


def _binary_data(seed=0):
    rng = np.random.RandomState(seed)
    n = 200
    y = rng.randint(0, 2, n)
    X = rng.randn(n, 10) + y[:, None] * 2.0
    genes = np.array([f"G{i % 20}" for i in range(n)])
    splits = gene_split_cv(genes, n_folds=5, seed=seed)
    return X, y, splits


# ---------------------------------------------------------------------------
# run_logreg_cv
# ---------------------------------------------------------------------------

class TestRunLogregCv:

    def test_returns_macro_f1_mean(self):
        X, y, splits, _ = _multiclass_data()
        r = run_logreg_cv(X, y, splits)
        assert "macro_f1_mean" in r

    def test_auroc_keys_present(self):
        X, y, splits, _ = _multiclass_data()
        r = run_logreg_cv(X, y, splits)
        for cls in MECHANISM_CLASSES:
            assert f"auroc_{cls}_mean" in r

    def test_recovers_signal_on_separable_data(self):
        X, y, splits, _ = _multiclass_data()
        r = run_logreg_cv(X, y, splits)
        assert r["macro_f1_mean"] > 0.5

    def test_empty_splits_returns_error(self):
        X = np.random.randn(10, 5)
        y = np.array([GOF] * 10)
        assert "error" in run_logreg_cv(X, y, [])

    def test_per_gene_f1_keys_when_genes_provided(self):
        X, y, splits, genes = _multiclass_data()
        r = run_logreg_cv(X, y, splits, genes=genes)
        assert "per_gene_f1_mean" in r
        assert "per_gene_f1_std" in r

    def test_no_per_gene_keys_without_genes(self):
        X, y, splits, _ = _multiclass_data()
        r = run_logreg_cv(X, y, splits)
        assert "per_gene_f1_mean" not in r


# ---------------------------------------------------------------------------
# run_logreg_binary_cv
# ---------------------------------------------------------------------------

class TestRunLogregBinaryCv:

    def test_returns_auroc_mean(self):
        X, y, splits = _binary_data()
        r = run_logreg_binary_cv(X, y, splits)
        assert "auroc_mean" in r
        assert "auroc_std" in r

    def test_above_chance_on_separable(self):
        X, y, splits = _binary_data()
        r = run_logreg_binary_cv(X, y, splits)
        assert r["auroc_mean"] > 0.5

    def test_empty_splits_returns_empty(self):
        assert run_logreg_binary_cv(np.zeros((10, 5)), np.zeros(10, dtype=int), []) == {}

    def test_n_folds_present(self):
        X, y, splits = _binary_data()
        r = run_logreg_binary_cv(X, y, splits)
        assert "n_folds" in r


# ---------------------------------------------------------------------------
# run_mlp_binary_cv
# ---------------------------------------------------------------------------

class TestRunMlpBinaryCv:

    def test_returns_auroc(self):
        X, y, splits = _binary_data()
        r = run_mlp_binary_cv(X, y, splits)
        assert "auroc_mean" in r

    def test_above_chance_on_separable(self):
        X, y, splits = _binary_data()
        r = run_mlp_binary_cv(X, y, splits)
        assert r["auroc_mean"] > 0.5

    def test_empty_splits_returns_empty(self):
        assert run_mlp_binary_cv(np.zeros((10, 5)), np.zeros(10, dtype=int), []) == {}
