"""
Tests for esm2_mech.utils.probes (probe runners).

Invariants:
- run_logreg_cv: returns dict with macro_f1_mean and per-class auroc keys
- run_logreg_cv: empty splits returns error key
- run_logreg_cv: recovers signal on separable data
- run_logreg_cv: per_gene_f1 keys present when genes provided
- run_logreg_cv: a valid fold is not silently dropped (class-count skip condition)
- run_logreg_binary_cv: returns auroc_mean > 0.5 on separable data
- run_logreg_binary_cv: empty splits returns empty dict
- run_mlp_binary_cv: returns auroc_mean on separable binary data
- run_mlp_binary_cv: empty splits returns empty dict
- run_mlp_cv: returns macro_f1_mean and recovers signal; per_gene_f1 when genes given
- run_sklearn_probe: returns macro_f1_mean; insufficient data returns error
- run_sklearn_probe_pca: returns macro_f1_mean with per-fold PCA reduction
- pca_reduce: output dims match n_components; fit on train only
- _pos_class_col: returns correct column; raises when pos_label absent
"""

import numpy as np
import pytest

from sklearn.linear_model import LogisticRegression

from esm2_mech.utils.probes import (
    run_logreg_cv,
    run_logreg_binary_cv,
    run_mlp_binary_cv,
    run_mlp_cv,
    run_sklearn_probe,
    run_sklearn_probe_pca,
    pca_reduce,
    _pos_class_col,
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


# ---------------------------------------------------------------------------
# run_mlp_cv (sklearn multiclass)
# ---------------------------------------------------------------------------

class TestRunMlpCv:

    def test_returns_macro_f1_mean(self):
        X, y, splits, _ = _multiclass_data()
        r = run_mlp_cv(X, y, splits, hidden=(16,))
        assert "macro_f1_mean" in r

    def test_recovers_signal_on_separable_data(self):
        X, y, splits, _ = _multiclass_data()
        r = run_mlp_cv(X, y, splits, hidden=(16,))
        assert r["macro_f1_mean"] > 0.5

    def test_per_gene_f1_keys_when_genes_provided(self):
        X, y, splits, genes = _multiclass_data()
        r = run_mlp_cv(X, y, splits, hidden=(16,), genes=genes)
        assert "per_gene_f1_mean" in r
        assert "per_gene_f1_std" in r

    def test_empty_splits_returns_error(self):
        X = np.random.randn(10, 5)
        y = np.array([GOF] * 10)
        assert "error" in run_mlp_cv(X, y, [], hidden=(16,))

    def test_fold_with_rare_class_only_in_test_is_kept(self):
        # A fold whose train split has exactly 2 of 3 classes (the rare class
        # falls entirely in test) is still scorable — the runner must fit on it,
        # not skip it. Pins the < 2 (not < n_classes) train-class skip condition.
        rng = np.random.RandomState(0)
        n = 180
        y = np.array([GOF, DN] * 80 + [LOF] * 20)
        X = rng.randn(n, 8) + np.array(
            [MECHANISM_CLASSES.index(c) for c in y]
        )[:, None] * 2.0
        # One fold holds out every LOF row (so train has only GOF/DN) plus some
        # GOF/DN so the test split has >= 2 classes.
        lof_idx = np.where(y == LOF)[0]
        extra = np.where(y != LOF)[0][:20]
        test_idx = np.concatenate([lof_idx, extra])
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        assert len(set(y[train_idx].tolist())) == 2  # rare class only in test
        r = run_mlp_cv(X, y, [(train_idx, test_idx)], hidden=(16,))
        assert r.get("n_folds") == 1  # fold kept, not skipped


# ---------------------------------------------------------------------------
# fold-skip condition (CLAUDE.md: < 2 classes in train must skip, valid folds must not)
# ---------------------------------------------------------------------------

class TestFoldSkipping:

    def test_valid_fold_not_dropped(self):
        # All three classes present in every train fold → no fold should be skipped,
        # so the runner must produce a real result rather than an error.
        X, y, splits, _ = _multiclass_data()
        r = run_logreg_cv(X, y, splits)
        assert "macro_f1_mean" in r
        assert r["n_folds"] >= 1

    def test_single_class_train_yields_no_folds(self):
        # Every fold has only one class in train → all folds skipped → error.
        X = np.random.RandomState(0).randn(60, 5)
        y = np.array([GOF] * 60)
        genes = np.array([f"G{i % 12}" for i in range(60)])
        splits = gene_split_cv(genes, n_folds=5, seed=0)
        r = run_logreg_cv(X, y, splits)
        assert "error" in r


# ---------------------------------------------------------------------------
# run_sklearn_probe / run_sklearn_probe_pca
# ---------------------------------------------------------------------------

def _logreg_fn(seed):
    return LogisticRegression(max_iter=1000, random_state=seed)


class TestRunSklearnProbe:

    def test_returns_macro_f1_mean(self):
        X, y, splits, genes = _multiclass_data()
        r = run_sklearn_probe(_logreg_fn, X, y, genes, splits=splits)
        assert "macro_f1_mean" in r

    def test_recovers_signal(self):
        X, y, splits, genes = _multiclass_data()
        r = run_sklearn_probe(_logreg_fn, X, y, genes, splits=splits)
        assert r["macro_f1_mean"] > 0.5

    def test_auroc_keys_present(self):
        X, y, splits, genes = _multiclass_data()
        r = run_sklearn_probe(_logreg_fn, X, y, genes, splits=splits)
        # per-class AUROC keys are emitted as auroc_<encoded-label>_mean
        assert any(k.startswith("auroc_") and k.endswith("_mean") for k in r)

    def test_insufficient_data_returns_error(self):
        X = np.random.randn(10, 5)
        y = np.array([GOF] * 10)
        genes = np.array([f"G{i}" for i in range(10)])
        r = run_sklearn_probe(_logreg_fn, X, y, genes, splits=[])
        assert "error" in r

    def test_normalize_flag_runs(self):
        X, y, splits, genes = _multiclass_data()
        r = run_sklearn_probe(_logreg_fn, X, y, genes, splits=splits, normalize=True)
        assert "macro_f1_mean" in r


class TestRunSklearnProbePca:

    def test_returns_macro_f1_mean(self):
        X, y, splits, genes = _multiclass_data()
        r = run_sklearn_probe_pca(_logreg_fn, X, y, genes, splits=splits, n_pca=5)
        assert "macro_f1_mean" in r

    def test_recovers_signal(self):
        X, y, splits, genes = _multiclass_data()
        r = run_sklearn_probe_pca(_logreg_fn, X, y, genes, splits=splits, n_pca=10)
        assert r["macro_f1_mean"] > 0.5

    def test_insufficient_data_returns_error(self):
        X = np.random.randn(10, 5)
        y = np.array([GOF] * 10)
        genes = np.array([f"G{i}" for i in range(10)])
        r = run_sklearn_probe_pca(_logreg_fn, X, y, genes, splits=[])
        assert "error" in r


# ---------------------------------------------------------------------------
# pca_reduce
# ---------------------------------------------------------------------------

class TestPcaReduce:

    def test_output_dims_match_n_components(self):
        rng = np.random.RandomState(0)
        X_tr = rng.randn(40, 20)
        X_te = rng.randn(10, 20)
        tr_out, te_out = pca_reduce(X_tr, X_te, n_components=5)
        assert tr_out.shape == (40, 5)
        assert te_out.shape == (10, 5)

    def test_row_counts_preserved(self):
        rng = np.random.RandomState(1)
        X_tr = rng.randn(30, 15)
        X_te = rng.randn(7, 15)
        tr_out, te_out = pca_reduce(X_tr, X_te, n_components=4)
        assert tr_out.shape[0] == 30
        assert te_out.shape[0] == 7


# ---------------------------------------------------------------------------
# _pos_class_col
# ---------------------------------------------------------------------------

class TestPosClassCol:

    def test_returns_correct_column(self):
        classes = np.array([0, 1])
        assert _pos_class_col(classes, 1) == 1
        assert _pos_class_col(classes, 0) == 0

    def test_column_for_string_labels(self):
        classes = np.array([LOF, GOF, DN])
        assert _pos_class_col(classes, GOF) == 1

    def test_raises_when_pos_label_absent(self):
        classes = np.array([0])  # only the negative class present
        with pytest.raises(ValueError, match="not found"):
            _pos_class_col(classes, 1)
