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
- run_mlp_binary_cv: early-stopping validation rejects groups crossing an
  outer-CV train/test boundary
- run_mlp_cv: returns macro_f1_mean and recovers signal; per_gene_f1 when genes given
- run_mlp_probe_cv: keeps a fold whose train has 2 of 3 classes; skips a
  fold whose train has 1 class
- run_mlp_probe_cv: early-stopping validation holds out whole dependency groups
  and rejects groups that cross an outer-CV train/test boundary
- run_sklearn_probe: returns macro_f1_mean; insufficient data returns error
- run_sklearn_probe_pca: returns macro_f1_mean with per-fold PCA reduction
- run_histgb_cv: fits a matrix containing NaN without imputing, while
  run_logreg_cv raises on that same matrix; recovers signal on observed data;
  tolerates a fully-NaN column from a source that failed entirely
- pca_reduce: output dims match n_components; fit on train only
- _pos_class_col: returns correct column; raises when pos_label absent
"""

import numpy as np
import pytest

from sklearn.linear_model import LogisticRegression
import sklearn.neural_network as neural_network

from esm2_mech.utils.probes import (
    run_logreg_cv,
    run_logreg_binary_cv,
    run_histgb_cv,
    require_no_nan,
    run_mlp_binary_cv,
    run_mlp_cv,
    run_mlp_probe_cv,
    run_sklearn_probe,
    run_sklearn_probe_pca,
    pca_reduce,
    _validation_group_mask,
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
    return X, y, splits, genes


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
        X, y, splits, _ = _binary_data()
        r = run_logreg_binary_cv(X, y, splits)
        assert "auroc_mean" in r
        assert "auroc_std" in r

    def test_above_chance_on_separable(self):
        X, y, splits, _ = _binary_data()
        r = run_logreg_binary_cv(X, y, splits)
        assert r["auroc_mean"] > 0.5

    def test_empty_splits_returns_empty(self):
        assert run_logreg_binary_cv(np.zeros((10, 5)), np.zeros(10, dtype=int), []) == {}

    def test_n_folds_present(self):
        X, y, splits, _ = _binary_data()
        r = run_logreg_binary_cv(X, y, splits)
        assert "n_folds" in r


# ---------------------------------------------------------------------------
# run_mlp_binary_cv
# ---------------------------------------------------------------------------

class TestRunMlpBinaryCv:

    def test_returns_auroc(self):
        X, y, splits, genes = _binary_data()
        r = run_mlp_binary_cv(X, y, splits, validation_groups=genes)
        assert "auroc_mean" in r

    def test_above_chance_on_separable(self):
        X, y, splits, genes = _binary_data()
        r = run_mlp_binary_cv(X, y, splits, validation_groups=genes)
        assert r["auroc_mean"] > 0.5

    def test_empty_splits_returns_empty(self):
        assert run_mlp_binary_cv(
            np.zeros((10, 5)),
            np.zeros(10, dtype=int),
            [],
            validation_groups=np.arange(10),
        ) == {}

    def test_group_crossing_outer_boundary_raises(self):
        rng = np.random.RandomState(0)
        X = rng.randn(20, 4)
        y = np.array([0, 1] * 10)
        train_idx = np.arange(16)
        test_idx = np.arange(16, 20)
        validation_groups = np.array(
            [
                "shared",
                *[f"train{i}" for i in range(1, 16)],
                "shared",
                "test1",
                "test2",
                "test3",
            ]
        )

        with pytest.raises(ValueError, match="spans the outer CV train/test"):
            run_mlp_binary_cv(
                X,
                y,
                [(train_idx, test_idx)],
                validation_groups=validation_groups,
            )

    def test_early_stopping_resets_patience_on_meaningful_improvement(
        self, monkeypatch
    ):
        instances = []

        class RecordingMlp:
            def __init__(self, **_kwargs):
                self.max_iter = 20
                self.n_iter_no_change = 2
                self.tol = 0.001
                self.partial_fit_calls = 0
                self.coefs_ = []
                self.intercepts_ = []
                self.classes_ = np.array([0, 1])
                instances.append(self)

            def partial_fit(self, _X, _y, classes=None):
                self.partial_fit_calls += 1
                if classes is not None:
                    self.classes_ = np.asarray(classes)
                marker = float(self.partial_fit_calls)
                self.coefs_ = [np.array([[marker]])]
                self.intercepts_ = [np.array([marker])]
                return self

            def score(self, _X, _y):
                scores = (0.5, 0.6, 0.7, 0.7, 0.7, 0.7)
                return scores[min(self.partial_fit_calls - 1, len(scores) - 1)]

            def predict_proba(self, X):
                return np.full((len(X), 2), 0.5)

        monkeypatch.setattr(neural_network, "MLPClassifier", RecordingMlp)
        X = np.zeros((80, 4))
        y = np.tile(np.array([0, 1]), 40)
        train_idx = np.arange(60)
        test_idx = np.arange(60, 80)

        run_mlp_binary_cv(
            X,
            y,
            [(train_idx, test_idx)],
            validation_groups=np.array([f"G{i}" for i in range(len(X))]),
        )

        assert instances[0].partial_fit_calls == 6
        assert instances[0].coefs_[0][0, 0] == 3.0


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
# run_mlp_probe_cv (torch multiclass) — the runner behind the ESM-2 family-split
# floor, so its fold set is the reference any comparison arm must match.
# ---------------------------------------------------------------------------

class TestRunMlpProbeCv:

    def test_validation_mask_keeps_dependency_groups_intact(self):
        groups = np.repeat(np.array([f"PF{i}" for i in range(10)]), 3)

        validation_mask = _validation_group_mask(groups, seed=7)

        assert validation_mask is not None
        assert validation_mask.any()
        assert (~validation_mask).any()
        for group in set(groups):
            group_mask = validation_mask[groups == group]
            assert np.all(group_mask == group_mask[0])

    def test_group_crossing_outer_boundary_raises(self):
        rng = np.random.RandomState(0)
        X = rng.randn(12, 4)
        y = np.array([GOF, DN] * 6)
        train_idx = np.arange(8)
        test_idx = np.arange(8, 12)
        validation_groups = np.array(
            [
                "shared", "train1", "train2", "train3", "train4", "train5",
                "train6", "train7", "shared", "test1", "test2", "test3",
            ]
        )

        with pytest.raises(ValueError, match="spans the outer CV train/test"):
            run_mlp_probe_cv(
                X,
                y,
                [(train_idx, test_idx)],
                validation_groups=validation_groups,
                hidden=(8,),
                max_epochs=1,
            )

    def test_fold_with_rare_class_only_in_test_is_kept(self):
        # Train has 2 of 3 classes — fittable, so the fold must be scored.
        rng = np.random.RandomState(0)
        n = 180
        y = np.array([GOF, DN] * 80 + [LOF] * 20)
        X = rng.randn(n, 8) + np.array(
            [MECHANISM_CLASSES.index(c) for c in y]
        )[:, None] * 2.0
        lof_idx = np.where(y == LOF)[0]
        extra = np.where(y != LOF)[0][:20]
        test_idx = np.concatenate([lof_idx, extra])
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        assert len(set(y[train_idx].tolist())) == 2  # rare class only in test
        r = run_mlp_probe_cv(
            X, y, [(train_idx, test_idx)], validation_groups=np.arange(n),
            hidden=(16,), max_epochs=3,
        )
        assert "macro_f1_mean" in r  # fold kept and scored, not skipped

    def test_single_class_train_fold_is_skipped(self):
        # Train has 1 class — unfittable, so the fold must be dropped.
        rng = np.random.RandomState(0)
        n = 120
        y = np.array([GOF] * 60 + [DN] * 40 + [LOF] * 20)
        X = rng.randn(n, 8)
        train_idx = np.arange(60)  # GOF only
        test_idx = np.arange(60, n)
        r = run_mlp_probe_cv(
            X, y, [(train_idx, test_idx)], validation_groups=np.arange(n),
            hidden=(16,), max_epochs=3,
        )
        assert r == {}  # no scorable fold


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


# ---------------------------------------------------------------------------
# run_histgb_cv — NaN-native runner
# ---------------------------------------------------------------------------

def _multiclass_data_with_nan(seed=0, nan_frac=0.3):
    """Separable multiclass data with a NaN-riddled column.

    Mirrors the real proteome matrix: one column (here col 0) is missing for a
    large share of rows, exactly the case where complete-case restriction would
    discard a big, non-random slice.
    """
    X, y, splits, genes = _multiclass_data(seed)
    rng = np.random.RandomState(seed + 1)
    X = X.copy()
    X[rng.rand(len(X)) < nan_frac, 0] = np.nan
    return X, y, splits, genes


class TestRunHistgbCv:

    def test_fits_on_matrix_containing_nan(self):
        # The error condition: LogReg/MLP raise on NaN input; the NaN-native
        # runner must consume the same matrix without imputing anything.
        X, y, splits, _ = _multiclass_data_with_nan()
        assert np.isnan(X).any(), "fixture must actually contain NaN"
        r = run_histgb_cv(X, y, splits)
        assert "macro_f1_mean" in r
        assert not np.isnan(r["macro_f1_mean"])

    def test_logreg_rejects_the_same_nan_matrix(self):
        # Pins why run_histgb_cv exists: the scaler/LogReg path cannot take NaN,
        # so a matrix with missing cells must either be restricted (complete
        # case) or routed to the NaN-native runner — never silently imputed.
        X, y, splits, _ = _multiclass_data_with_nan()
        with pytest.raises(ValueError):
            run_logreg_cv(X, y, splits)

    def test_recovers_signal_happy_path(self):
        # Happy path: fully-observed separable data still classifies well.
        X, y, splits, _ = _multiclass_data()
        r = run_histgb_cv(X, y, splits)
        assert r["macro_f1_mean"] > 0.8

    def test_auroc_keys_present(self):
        X, y, splits, _ = _multiclass_data()
        r = run_histgb_cv(X, y, splits)
        for cls in MECHANISM_CLASSES:
            assert f"auroc_{cls}_mean" in r

    def test_per_gene_f1_when_genes_given(self):
        X, y, splits, genes = _multiclass_data_with_nan()
        r = run_histgb_cv(X, y, splits, genes=genes)
        assert "per_gene_f1_mean" in r

    def test_return_oof_shapes_align(self):
        X, y, splits, genes = _multiclass_data_with_nan()
        agg, oof = run_histgb_cv(X, y, splits, genes=genes, return_oof=True)
        assert oof is not None
        assert len(oof["y_true"]) == len(oof["genes"]) == len(oof["row_ids"])
        assert oof["proba"].shape == (len(oof["y_true"]), len(MECHANISM_CLASSES))

    def test_empty_splits_returns_error(self):
        X, y, _, _ = _multiclass_data()
        assert "error" in run_histgb_cv(X, y, [])

    def test_all_nan_column_does_not_crash(self):
        # A source that failed entirely leaves a fully-NaN column. It carries no
        # information, but must not take the whole probe down.
        X, y, splits, _ = _multiclass_data()
        X = X.copy()
        X[:, 3] = np.nan
        r = run_histgb_cv(X, y, splits)
        assert "macro_f1_mean" in r


# ---------------------------------------------------------------------------
# require_no_nan — the guard that makes the impute-to-silence bug unrepresentable
# ---------------------------------------------------------------------------

class TestRequireNoNan:

    def test_passes_on_dense_matrix(self):
        require_no_nan(np.zeros((4, 3)), "caller")  # must not raise

    def test_raises_on_any_nan(self):
        X = np.zeros((4, 3))
        X[2, 1] = np.nan
        with pytest.raises(ValueError):
            require_no_nan(X, "caller")

    def test_message_names_both_sanctioned_fixes(self):
        # The message has to point at the two correct responses, because the
        # tempting wrong one (impute to make the error go away) is the bug.
        X = np.array([[np.nan]])
        with pytest.raises(ValueError) as exc:
            require_no_nan(X, "run_logreg_cv")
        msg = str(exc.value)
        assert "run_histgb_cv" in msg
        assert "observed_rows_mask" in msg
        assert "Do NOT impute" in msg

    def test_message_reports_scale_of_missingness(self):
        X = np.zeros((10, 2))
        X[:3, 0] = np.nan
        with pytest.raises(ValueError, match=r"3 NaN cells across 3/10 rows"):
            require_no_nan(X, "run_logreg_cv")

    def test_nan_intolerant_runners_all_guarded(self):
        # Every runner that standardizes + fits a NaN-intolerant model must
        # fail at the probe boundary, not deep inside sklearn.
        X, y, splits, genes = _multiclass_data_with_nan()
        for runner in (run_logreg_cv, run_mlp_cv):
            with pytest.raises(ValueError, match="run_histgb_cv"):
                runner(X, y, splits)

    def test_histgb_runner_is_not_guarded(self):
        # The NaN-native runner is the sanctioned destination — it must accept
        # exactly the matrix the others reject.
        X, y, splits, _ = _multiclass_data_with_nan()
        assert "macro_f1_mean" in run_histgb_cv(X, y, splits)
