"""
Tests for esm2_mech.utils.metrics.

Invariants:
- compute_metrics: perfect predictions → macro_f1=1.0, all AUROCs=1.0
- compute_metrics: empty input → all None
- compute_metrics: single-class test → AUROC is None for that class
- compute_metrics: n field matches input length
- aggregate_folds: empty list → error key
- aggregate_folds: correct mean/std across folds
- aggregate_folds: None AUROCs excluded from aggregation
- aggregate_folds: n_folds matches input length
- align_proba: identity mapping preserved
- align_proba: column reordering correct
- align_proba: output shape matches (n_samples, n_classes)
- align_proba: missing classes filled with zero
- per_gene_f1: aggregates variants to gene level before scoring
- per_gene_f1: single gene with unanimous label scores correctly
- auroc_at_median: perfectly-ordered predictions -> 1.0, reversed -> 0.0
- auroc_at_median: median split leaving one class -> NaN
- auroc_at_median: binarisation uses y_true's median, not y_pred's
- auroc_at_median: median value (ties) lands in the positive class
"""

import numpy as np
import pytest

from esm2_mech.utils.metrics import (
    compute_metrics,
    aggregate_folds,
    align_proba,
    auroc_at_median,
)
from esm2_mech.utils.probes import _per_gene_f1 as per_gene_f1
from esm2_mech.utils.constants import MECHANISM_CLASSES, GOF, DN, LOF


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def _perfect(self):
        n = 30
        y_true = np.array([GOF, DN, LOF] * 10)
        y_proba = np.zeros((n, 3))
        for i, cls in enumerate(y_true):
            y_proba[i, MECHANISM_CLASSES.index(cls)] = 1.0
        return y_true, y_true.copy(), y_proba

    def test_perfect_macro_f1(self):
        y_true, y_pred, y_proba = self._perfect()
        r = compute_metrics(y_true, y_pred, y_proba)
        assert r["macro_f1"] == pytest.approx(1.0)

    def test_perfect_auroc(self):
        y_true, y_pred, y_proba = self._perfect()
        r = compute_metrics(y_true, y_pred, y_proba)
        for cls in MECHANISM_CLASSES:
            assert r["per_class_auroc"][cls] == pytest.approx(1.0)

    def test_empty_returns_none(self):
        r = compute_metrics(np.array([]), np.array([]), np.zeros((0, 3)))
        assert r["macro_f1"] is None
        assert all(v is None for v in r["per_class_auroc"].values())

    def test_single_class_auroc_is_none(self):
        y = np.array([GOF] * 20)
        proba = np.zeros((20, 3))
        proba[:, MECHANISM_CLASSES.index(GOF)] = 1.0
        r = compute_metrics(y, y, proba)
        assert r["per_class_auroc"][GOF] is None

    def test_n_field(self):
        y_true, y_pred, y_proba = self._perfect()
        r = compute_metrics(y_true, y_pred, y_proba)
        assert r["n"] == 30

    def test_all_classes_present_in_output(self):
        y_true, y_pred, y_proba = self._perfect()
        r = compute_metrics(y_true, y_pred, y_proba)
        for cls in MECHANISM_CLASSES:
            assert cls in r["per_class_auroc"]

    def test_chance_predictions_low_f1(self):
        rng = np.random.RandomState(0)
        y_true = np.array([GOF, DN, LOF] * 30)
        y_pred = rng.choice([GOF, DN, LOF], 90)
        y_proba = rng.dirichlet([1, 1, 1], 90)
        r = compute_metrics(y_true, y_pred, y_proba)
        assert r["macro_f1"] < 0.5


# ---------------------------------------------------------------------------
# aggregate_folds
# ---------------------------------------------------------------------------

class TestAggregateFolds:

    def _fold(self, f1, auroc_val):
        return {"macro_f1": f1, "per_class_auroc": {c: auroc_val for c in MECHANISM_CLASSES}}

    def test_empty_returns_error(self):
        assert "error" in aggregate_folds([])

    def test_mean_and_std(self):
        folds = [self._fold(0.4, 0.7), self._fold(0.6, 0.9)]
        r = aggregate_folds(folds)
        assert r["macro_f1_mean"] == pytest.approx(0.5)
        assert r["macro_f1_std"] == pytest.approx(0.1)

    def test_n_folds(self):
        folds = [self._fold(0.5, 0.8)] * 4
        assert aggregate_folds(folds)["n_folds"] == 4

    def test_none_auroc_excluded(self):
        folds = [
            {"macro_f1": 0.5, "per_class_auroc": {"GOF": 0.8, "DN": None, "LOF": 0.7}},
            {"macro_f1": 0.6, "per_class_auroc": {"GOF": 0.9, "DN": 0.75, "LOF": 0.8}},
        ]
        r = aggregate_folds(folds)
        assert r["auroc_GOF_mean"] == pytest.approx(0.85)
        assert r["auroc_DN_mean"] == pytest.approx(0.75)

    def test_single_fold(self):
        r = aggregate_folds([self._fold(0.6, 0.8)])
        assert r["macro_f1_mean"] == pytest.approx(0.6)
        assert r["macro_f1_std"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# align_proba
# ---------------------------------------------------------------------------

class TestAlignProba:

    def test_identity(self):
        proba = np.array([[0.1, 0.5, 0.4], [0.3, 0.3, 0.4]])
        result = align_proba(proba, np.array([GOF, DN, LOF]), MECHANISM_CLASSES)
        np.testing.assert_array_almost_equal(result, proba)

    def test_reorder_columns(self):
        # clf saw [LOF, GOF], canonical order is [GOF, DN, LOF]
        proba = np.array([[0.6, 0.4]])
        result = align_proba(proba, np.array([LOF, GOF]), MECHANISM_CLASSES)
        assert result[0, MECHANISM_CLASSES.index(LOF)] == pytest.approx(0.6)
        assert result[0, MECHANISM_CLASSES.index(GOF)] == pytest.approx(0.4)
        assert result[0, MECHANISM_CLASSES.index(DN)] == pytest.approx(0.0)

    def test_output_shape(self):
        proba = np.random.rand(10, 2)
        result = align_proba(proba, np.array([GOF, LOF]), MECHANISM_CLASSES)
        assert result.shape == (10, 3)

    def test_missing_class_filled_with_zero(self):
        proba = np.array([[1.0]])
        result = align_proba(proba, np.array([DN]), MECHANISM_CLASSES)
        assert result[0, MECHANISM_CLASSES.index(GOF)] == pytest.approx(0.0)
        assert result[0, MECHANISM_CLASSES.index(DN)] == pytest.approx(1.0)
        assert result[0, MECHANISM_CLASSES.index(LOF)] == pytest.approx(0.0)

    def test_output_dtype_float32(self):
        proba = np.array([[0.5, 0.5]])
        result = align_proba(proba, np.array([GOF, DN]), [GOF, DN])
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# per_gene_f1
# ---------------------------------------------------------------------------

class TestPerGeneF1:

    def test_unanimous_gene_labels(self):
        # Two genes, each with unanimous class — should score perfectly.
        genes = np.array(["A"] * 5 + ["B"] * 5)
        y_true = np.array([GOF] * 5 + [DN] * 5)
        proba = np.zeros((10, 3))
        proba[:5, MECHANISM_CLASSES.index(GOF)] = 1.0
        proba[5:, MECHANISM_CLASSES.index(DN)] = 1.0
        score = per_gene_f1(y_true, proba, genes)
        assert score == pytest.approx(1.0)

    def test_all_wrong_predictions(self):
        genes = np.array(["A"] * 5 + ["B"] * 5)
        y_true = np.array([GOF] * 5 + [DN] * 5)
        proba = np.zeros((10, 3))
        proba[:5, MECHANISM_CLASSES.index(DN)] = 1.0   # gene A predicted as DN (wrong)
        proba[5:, MECHANISM_CLASSES.index(GOF)] = 1.0  # gene B predicted as GOF (wrong)
        score = per_gene_f1(y_true, proba, genes)
        assert score == pytest.approx(0.0)

    def test_single_gene(self):
        genes = np.array(["A"] * 10)
        y_true = np.array([LOF] * 10)
        proba = np.zeros((10, 3))
        proba[:, MECHANISM_CLASSES.index(LOF)] = 1.0
        score = per_gene_f1(y_true, proba, genes)
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# auroc_at_median
# ---------------------------------------------------------------------------

class TestAurocAtMedian:

    def test_perfect_ordering(self):
        # Predictions rank-agree with the ground truth -> above-median rows all
        # outscore below-median rows -> AUROC = 1.0.
        y_true = np.arange(10, dtype=float)
        y_pred = np.arange(10, dtype=float)
        assert auroc_at_median(y_true, y_pred) == pytest.approx(1.0)

    def test_reversed_ordering(self):
        y_true = np.arange(10, dtype=float)
        y_pred = np.arange(10, dtype=float)[::-1].copy()
        assert auroc_at_median(y_true, y_pred) == pytest.approx(0.0)

    def test_single_class_returns_nan(self):
        # All-equal ground truth: median split leaves every row in the positive
        # class, so there is no negative class to score against.
        y_true = np.full(10, 3.0)
        y_pred = np.arange(10, dtype=float)
        assert np.isnan(auroc_at_median(y_true, y_pred))

    def test_binarises_on_y_true_not_y_pred(self):
        # y_true cleanly splits low/high; y_pred is ordered the OPPOSITE way and
        # has a different median. If binarisation used y_pred the score would
        # flip. Correct behaviour binarises y_true -> AUROC = 0.0.
        y_true = np.array([0.0, 1.0, 2.0, 3.0])
        y_pred = np.array([100.0, 90.0, 10.0, 0.0])
        assert auroc_at_median(y_true, y_pred) == pytest.approx(0.0)

    def test_median_tie_goes_to_positive_class(self):
        # Odd length: the median value itself exists in y_true. The `>=` binarises
        # it into the positive class, so 3 of 5 rows are positive. A monotone
        # y_pred still cleanly separates them -> AUROC = 1.0.
        y_true = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # median = 2.0
        y_pred = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        med = np.median(y_true)
        n_positive = int((y_true >= med).sum())
        assert n_positive == 3
        assert auroc_at_median(y_true, y_pred) == pytest.approx(1.0)
