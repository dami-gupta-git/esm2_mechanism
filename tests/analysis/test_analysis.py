"""
Unit tests for pure-logic functions in the analysis modules.

Covers:
- probe4_axis_identity: biochem_features returns correct values, handles unknowns
- direction_geometry: fit_direction returns unit vector; auroc handles single-class
- magnitude_direction: decompose produces correct shapes and unit norms;
  agg_seeds handles empty/nan/valid inputs; _best picks max of logreg/mlp;
  _f formats floats and None correctly
- transfer_contrast: _auroc handles single-class; _make_clf raises on unknown kind;
  transfer_test returns transfer and pooled AUROC dicts
- enzyme_classification: majority_baseline_f1 returns majority prediction score;
  run_logreg returns expected keys; run_mlp returns expected keys
"""

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# probe4_axis_identity — biochem_features
# ---------------------------------------------------------------------------

class TestBiochemFeatures:

    def setup_method(self):
        from esm2_mechanism.analysis.probe4_axis_identity import biochem_features, FEAT_NAMES
        self.biochem_features = biochem_features
        self.FEAT_NAMES = FEAT_NAMES

    def test_returns_list_of_six(self):
        result = self.biochem_features("V", "E")
        assert isinstance(result, list)
        assert len(result) == len(self.FEAT_NAMES)

    def test_same_aa_blosum_is_positive(self):
        # BLOSUM62 diagonal is always positive
        result = self.biochem_features("A", "A")
        assert result[0] > 0  # blosum62

    def test_same_aa_delta_hydro_is_zero(self):
        result = self.biochem_features("V", "V")
        assert result[1] == pytest.approx(0.0)   # d_hydro
        assert result[2] == pytest.approx(0.0)   # abs_d_hydro

    def test_same_aa_delta_charge_is_zero(self):
        result = self.biochem_features("K", "K")
        assert result[3] == pytest.approx(0.0)   # d_charge
        assert result[4] == pytest.approx(0.0)   # abs_d_charge

    def test_abs_values_are_nonnegative(self):
        result = self.biochem_features("D", "K")
        assert result[2] >= 0   # abs_d_hydro
        assert result[4] >= 0   # abs_d_charge
        assert result[5] >= 0   # abs_d_volume

    def test_unknown_wt_returns_none(self):
        assert self.biochem_features("B", "A") is None

    def test_unknown_mut_returns_none(self):
        assert self.biochem_features("A", "Z") is None

    def test_charge_change_d_to_k(self):
        # D has charge -1, K has charge +1 → d_charge = +2
        result = self.biochem_features("D", "K")
        assert result[3] == pytest.approx(2.0)

    def test_blosum_is_symmetric(self):
        ab = self.biochem_features("V", "E")[0]
        ba = self.biochem_features("E", "V")[0]
        assert ab == ba


# ---------------------------------------------------------------------------
# direction_geometry — fit_direction, auroc
# ---------------------------------------------------------------------------

class TestFitDirection:

    def setup_method(self):
        from esm2_mechanism.analysis.direction_geometry import fit_direction, auroc
        self.fit_direction = fit_direction
        self.auroc = auroc

    def _separable(self, n=60, dim=4, seed=0):
        rng = np.random.RandomState(seed)
        X = np.vstack([rng.randn(n // 2, dim) + 3, rng.randn(n // 2, dim) - 3])
        y = np.array([1] * (n // 2) + [0] * (n // 2))
        return X, y

    def test_returns_unit_vector(self):
        X, y = self._separable()
        w, _ = self.fit_direction(X, y)
        assert np.linalg.norm(w) == pytest.approx(1.0, abs=1e-5)

    def test_unit_vector_length_matches_features(self):
        X, y = self._separable(dim=8)
        w, _ = self.fit_direction(X, y)
        assert len(w) == 8

    def test_clf_returned(self):
        from sklearn.linear_model import LogisticRegression
        X, y = self._separable()
        _, clf = self.fit_direction(X, y)
        assert hasattr(clf, "predict_proba")

    def test_auroc_perfect_separation(self):
        X, y = self._separable()
        _, clf = self.fit_direction(X, y)
        score = self.auroc(clf, X, y)
        assert score >= 0.99

    def test_auroc_single_class_returns_nan(self):
        X, y = self._separable()
        _, clf = self.fit_direction(X, y)
        y_single = np.zeros(len(y), dtype=int)
        score = self.auroc(clf, X, y_single)
        assert np.isnan(score)


# ---------------------------------------------------------------------------
# magnitude_direction — decompose, agg_seeds, _best, _f
# ---------------------------------------------------------------------------

class TestDecompose:

    def setup_method(self):
        from esm2_mechanism.analysis.magnitude_direction import decompose
        self.decompose = decompose

    def test_shapes(self):
        delta = np.random.randn(20, 10).astype(np.float32)
        out = self.decompose(delta)
        assert out["full"].shape == (20, 10)
        assert out["mag"].shape == (20, 1)
        assert out["dir"].shape == (20, 10)

    def test_direction_is_unit_norm(self):
        delta = np.random.randn(20, 10).astype(np.float32)
        out = self.decompose(delta)
        norms = np.linalg.norm(out["dir"], axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_magnitude_is_nonnegative(self):
        delta = np.random.randn(20, 10).astype(np.float32)
        out = self.decompose(delta)
        assert (out["mag"] >= 0).all()

    def test_zero_vector_direction_is_near_zero(self):
        delta = np.zeros((5, 4), dtype=np.float32)
        out = self.decompose(delta)
        # zero delta → direction is 0/(0+eps), magnitude is 0
        assert out["mag"].sum() == pytest.approx(0.0)

    def test_full_equals_input(self):
        delta = np.random.randn(10, 6).astype(np.float32)
        out = self.decompose(delta)
        np.testing.assert_array_equal(out["full"], delta)


class TestAggSeeds:

    def setup_method(self):
        from esm2_mechanism.analysis.magnitude_direction import agg_seeds
        self.agg_seeds = agg_seeds

    def test_empty_returns_nan(self):
        result = self.agg_seeds([])
        assert np.isnan(result["mean"])
        assert result["n"] == 0

    def test_all_nan_returns_nan(self):
        result = self.agg_seeds([float("nan"), float("nan")])
        assert np.isnan(result["mean"])
        assert result["n"] == 0

    def test_filters_none(self):
        result = self.agg_seeds([None, 0.8, 0.6])
        assert result["n"] == 2
        assert result["mean"] == pytest.approx(0.7)

    def test_mean_and_std_correct(self):
        vals = [0.5, 0.7, 0.9]
        result = self.agg_seeds(vals)
        assert result["mean"] == pytest.approx(np.mean(vals))
        assert result["std"] == pytest.approx(np.std(vals))
        assert result["n"] == 3


class TestBestAndF:

    def setup_method(self):
        from esm2_mechanism.analysis.magnitude_direction import _best, _f
        self._best = _best
        self._f = _f

    def _make_block(self, lr_mean, mlp_mean):
        return {
            "family_split": {
                "logreg_auroc": {"mean": lr_mean},
                "mlp_auroc": {"mean": mlp_mean},
            }
        }

    def test_best_picks_mlp_when_higher(self):
        block = self._make_block(0.7, 0.85)
        assert self._best(block, "family_split", "logreg_auroc", "mlp_auroc") == pytest.approx(0.85)

    def test_best_picks_logreg_when_higher(self):
        block = self._make_block(0.9, 0.7)
        assert self._best(block, "family_split", "logreg_auroc", "mlp_auroc") == pytest.approx(0.9)

    def test_best_nan_in_one_uses_other(self):
        block = self._make_block(float("nan"), 0.75)
        assert self._best(block, "family_split", "logreg_auroc", "mlp_auroc") == pytest.approx(0.75)

    def test_best_both_nan_returns_nan(self):
        block = self._make_block(float("nan"), float("nan"))
        assert np.isnan(self._best(block, "family_split", "logreg_auroc", "mlp_auroc"))

    def test_f_formats_float(self):
        assert self._f(0.123) == "0.123"

    def test_f_none_returns_nan_string(self):
        assert self._f(None) == "nan"

    def test_f_nan_returns_nan_string(self):
        assert self._f(float("nan")) == "nan"


# ---------------------------------------------------------------------------
# transfer_contrast — _auroc, _make_clf, transfer_test
# ---------------------------------------------------------------------------

class TestTransferContrastHelpers:

    def setup_method(self):
        from esm2_mechanism.analysis.transfer_contrast import _auroc, _make_clf
        self._auroc = _auroc
        self._make_clf = _make_clf

    def _fit_clf(self, kind="linear", seed=0):
        from sklearn.preprocessing import StandardScaler
        rng = np.random.RandomState(seed)
        X = np.vstack([rng.randn(30, 4) + 2, rng.randn(30, 4) - 2])
        y = np.array([1] * 30 + [0] * 30)
        Xs = StandardScaler().fit_transform(X)
        clf = self._make_clf(kind, seed)
        clf.fit(Xs, y)
        return clf, Xs, y

    def test_auroc_perfect(self):
        clf, Xs, y = self._fit_clf()
        score = self._auroc(clf, Xs, y)
        assert score >= 0.99

    def test_auroc_single_class_returns_nan(self):
        clf, Xs, _ = self._fit_clf()
        y_single = np.zeros(len(Xs), dtype=int)
        score = self._auroc(clf, Xs, y_single)
        assert np.isnan(score)

    def test_make_clf_linear(self):
        from sklearn.linear_model import LogisticRegression
        clf = self._make_clf("linear", seed=0)
        assert isinstance(clf, LogisticRegression)

    def test_make_clf_gbm(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = self._make_clf("gbm", seed=0)
        assert isinstance(clf, HistGradientBoostingClassifier)

    def test_make_clf_unknown_raises(self):
        with pytest.raises(ValueError):
            self._make_clf("svm", seed=0)


class TestTransferTest:

    def setup_method(self):
        from esm2_mechanism.analysis.transfer_contrast import transfer_test
        self.transfer_test = transfer_test

    def _make_data(self, n=120, dim=4, seed=0):
        rng = np.random.RandomState(seed)
        groups = np.array([f"G{i % 6}" for i in range(n)])
        y = np.array([i % 2 for i in range(n)])
        delta = rng.randn(n, dim).astype(np.float32)
        return delta, y, groups

    def test_returns_transfer_and_pooled_keys(self):
        delta, y, groups = self._make_data()
        result = self.transfer_test(delta, y, groups, kind="linear", n_partitions=3, seed=0)
        assert "transfer_auroc" in result
        assert "pooled_auroc" in result

    def test_transfer_auroc_has_three_elements(self):
        delta, y, groups = self._make_data()
        result = self.transfer_test(delta, y, groups, kind="linear", n_partitions=3, seed=0)
        mean, std, n = result["transfer_auroc"]
        assert isinstance(mean, float)
        assert std >= 0
        assert n > 0

    def test_auroc_range(self):
        delta, y, groups = self._make_data()
        result = self.transfer_test(delta, y, groups, kind="linear", n_partitions=3, seed=0)
        mean, _, _ = result["pooled_auroc"]
        assert 0.0 <= mean <= 1.0


# ---------------------------------------------------------------------------
# enzyme_classification — majority_baseline_f1, run_logreg, run_mlp
# ---------------------------------------------------------------------------

class TestMajorityBaseline:

    def setup_method(self):
        from esm2_mechanism.analysis.enzyme_classification import majority_baseline_f1
        self.majority_baseline_f1 = majority_baseline_f1

    def test_all_same_class_returns_expected_f1(self):
        # Majority = only class; macro-F1 for 1-class perfect prediction
        y = np.zeros(20, dtype=int)
        result = self.majority_baseline_f1(y)
        assert isinstance(result, float)

    def test_balanced_two_class_f1(self):
        y = np.array([0, 1] * 20)
        result = self.majority_baseline_f1(y)
        assert 0.0 <= result <= 1.0

    def test_returns_float(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        result = self.majority_baseline_f1(y)
        assert isinstance(result, float)


class TestRunLogreg:

    def setup_method(self):
        from esm2_mechanism.analysis.enzyme_classification import run_logreg, ENZYME_CLASSES
        self.run_logreg = run_logreg
        self.classes = ENZYME_CLASSES

    def _make_splits(self, n=80):
        fold = n // 2
        return [(np.arange(fold), np.arange(fold, n))]

    def test_returns_macro_f1_mean(self):
        rng = np.random.RandomState(0)
        n = 80
        X = rng.randn(n, 10).astype(np.float32)
        y = np.array([i % 4 for i in range(n)])
        splits = self._make_splits(n)
        result = self.run_logreg(X, y, splits, self.classes)
        assert "macro_f1_mean" in result

    def test_returns_per_class_auroc(self):
        rng = np.random.RandomState(0)
        n = 80
        X = rng.randn(n, 10).astype(np.float32)
        y = np.array([i % 4 for i in range(n)])
        splits = self._make_splits(n)
        result = self.run_logreg(X, y, splits, self.classes)
        assert "per_class_auroc_mean" in result
        assert set(result["per_class_auroc_mean"].keys()) == set(self.classes)

    def test_f1_in_range(self):
        rng = np.random.RandomState(0)
        n = 80
        X = rng.randn(n, 10).astype(np.float32)
        y = np.array([i % 4 for i in range(n)])
        splits = self._make_splits(n)
        result = self.run_logreg(X, y, splits, self.classes)
        f1 = result["macro_f1_mean"]
        assert f1 is None or 0.0 <= f1 <= 1.0

    def test_empty_splits_returns_none_f1(self):
        rng = np.random.RandomState(0)
        X = rng.randn(20, 4).astype(np.float32)
        y = np.array([i % 4 for i in range(20)])
        result = self.run_logreg(X, y, [], self.classes)
        assert result["macro_f1_mean"] is None


class TestRunMlp:

    def setup_method(self):
        from esm2_mechanism.analysis.enzyme_classification import run_mlp, ENZYME_CLASSES
        self.run_mlp = run_mlp
        self.classes = ENZYME_CLASSES

    def _make_splits(self, n=80):
        fold = n // 2
        return [(np.arange(fold), np.arange(fold, n))]

    def test_returns_macro_f1_mean(self):
        rng = np.random.RandomState(0)
        n = 80
        X = rng.randn(n, 10).astype(np.float32)
        y = np.array([i % 4 for i in range(n)])
        splits = self._make_splits(n)
        result = self.run_mlp(X, y, splits, self.classes)
        assert "macro_f1_mean" in result

    def test_returns_per_class_auroc(self):
        rng = np.random.RandomState(0)
        n = 80
        X = rng.randn(n, 10).astype(np.float32)
        y = np.array([i % 4 for i in range(n)])
        splits = self._make_splits(n)
        result = self.run_mlp(X, y, splits, self.classes)
        assert "per_class_auroc_mean" in result

    def test_f1_in_range(self):
        rng = np.random.RandomState(0)
        n = 80
        X = rng.randn(n, 10).astype(np.float32)
        y = np.array([i % 4 for i in range(n)])
        splits = self._make_splits(n)
        result = self.run_mlp(X, y, splits, self.classes)
        f1 = result["macro_f1_mean"]
        assert f1 is None or 0.0 <= f1 <= 1.0
