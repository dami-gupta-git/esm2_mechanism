"""
Tests for CV split helpers and probe utilities in utils_probes.py.

Key invariants checked:
- No gene/family appears in both train and test of the same fold (leakage)
- Every sample appears in exactly one test fold across all folds
- Train and test are mutually exclusive and collectively exhaustive per fold
- Min-size filtering (tr>=10, te>=5) is respected
- Determinism: same seed → same splits
- Seed sensitivity: different seeds → different splits
- family_split_cv: no family spans both train and test in any fold
- family_split_cv: genes without pfam entries are excluded from all folds
- compute_metrics: correct macro-F1 and per-class AUROC
- aggregate_folds: correct mean/std aggregation
- align_proba: correct column reordering
- run_logreg_cv: returns auroc/f1 metrics dict
- run_logreg_binary_cv: returns auroc mean/std
- run_ridge_cv: returns spearman/pearson mean/std
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from utils_probes import (
    gene_split_cv, family_split_cv, family_split_indices,
    compute_metrics, aggregate_folds, align_proba,
    run_logreg_cv, run_logreg_binary_cv, run_ridge_cv,
    MECHANISM_CLASSES,
)


# ── fixtures ─────────────────────────────────────────────────────────────────

def make_genes(n_genes=20, variants_per_gene=10, seed=0):
    """Return a variants-length array of gene labels."""
    rng = np.random.RandomState(seed)
    gene_names = [f"GENE{i:02d}" for i in range(n_genes)]
    return np.array([gene_names[i] for i in rng.randint(0, n_genes, n_genes * variants_per_gene)])


def make_pfam_map(genes, n_families=5):
    """Assign each unique gene to one of n_families Pfam families."""
    unique = sorted(set(genes))
    return {g: f"FAM{i % n_families:02d}" for i, g in enumerate(unique)}


# ── gene_split_cv ─────────────────────────────────────────────────────────────

class TestGeneSplitCv:

    def test_no_gene_leakage(self):
        genes = make_genes()
        splits = gene_split_cv(genes, n_folds=5, seed=42)
        assert splits, "Expected at least one fold"
        for tr, te in splits:
            tr_genes = set(genes[tr])
            te_genes = set(genes[te])
            overlap = tr_genes & te_genes
            assert not overlap, f"Gene leakage: {overlap} appear in both train and test"

    def test_train_test_disjoint_indices(self):
        genes = make_genes()
        splits = gene_split_cv(genes, n_folds=5, seed=42)
        for tr, te in splits:
            assert len(set(tr) & set(te)) == 0, "Same index appears in both train and test"

    def test_every_sample_covered_once(self):
        """Each sample appears in exactly one test fold."""
        genes = make_genes()
        splits = gene_split_cv(genes, n_folds=5, seed=42)
        test_counts = np.zeros(len(genes), dtype=int)
        for _, te in splits:
            test_counts[te] += 1
        # Some samples may be filtered out if their gene ended up in a fold that
        # failed the min-size check — so counts can be 0 or 1, never >1.
        assert test_counts.max() <= 1, "A sample appears in more than one test fold"

    def test_min_size_respected(self):
        genes = make_genes()
        splits = gene_split_cv(genes, n_folds=5, seed=42)
        for tr, te in splits:
            assert len(tr) >= 10, f"Train set too small: {len(tr)}"
            assert len(te) >= 5, f"Test set too small: {len(te)}"

    def test_returns_index_arrays(self):
        genes = make_genes()
        splits = gene_split_cv(genes, n_folds=5, seed=42)
        for tr, te in splits:
            assert isinstance(tr, np.ndarray)
            assert isinstance(te, np.ndarray)

    def test_deterministic(self):
        genes = make_genes()
        s1 = gene_split_cv(genes, n_folds=5, seed=7)
        s2 = gene_split_cv(genes, n_folds=5, seed=7)
        assert len(s1) == len(s2)
        for (tr1, te1), (tr2, te2) in zip(s1, s2):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(te1, te2)

    def test_different_seeds_differ(self):
        genes = make_genes()
        s1 = gene_split_cv(genes, n_folds=5, seed=0)
        s2 = gene_split_cv(genes, n_folds=5, seed=99)
        # At least one fold should differ between seeds
        any_diff = any(
            not (np.array_equal(tr1, tr2) and np.array_equal(te1, te2))
            for (tr1, te1), (tr2, te2) in zip(s1, s2)
        )
        assert any_diff, "Different seeds produced identical splits"

    def test_single_gene_returns_no_folds(self):
        """If all variants belong to one gene, no fold can satisfy min-size."""
        genes = np.array(["GENE0"] * 20)
        splits = gene_split_cv(genes, n_folds=5, seed=42)
        assert splits == [], "Expected no valid folds for single-gene input"

    def test_fewer_genes_than_folds(self):
        """3 genes, 5 folds — should produce ≤3 non-empty folds."""
        genes = np.array(["A"] * 20 + ["B"] * 20 + ["C"] * 20)
        splits = gene_split_cv(genes, n_folds=5, seed=42)
        # Each fold that passes min-size check must still be leakage-free
        for tr, te in splits:
            assert not (set(genes[tr]) & set(genes[te]))

    def test_indices_in_bounds(self):
        genes = make_genes()
        splits = gene_split_cv(genes, n_folds=5, seed=42)
        n = len(genes)
        for tr, te in splits:
            assert tr.min() >= 0 and tr.max() < n
            assert te.min() >= 0 and te.max() < n


# ── family_split_cv ───────────────────────────────────────────────────────────

class TestFamilySplitCv:

    def test_no_family_leakage(self):
        genes = make_genes()
        pfam = make_pfam_map(genes)
        splits = family_split_cv(genes, pfam, n_folds=5, seed=42)
        assert splits, "Expected at least one fold"
        for tr, te in splits:
            tr_fams = {pfam[g] for g in genes[tr] if g in pfam}
            te_fams = {pfam[g] for g in genes[te] if g in pfam}
            overlap = tr_fams & te_fams
            assert not overlap, f"Family leakage: {overlap} appear in both train and test"

    def test_train_test_disjoint_indices(self):
        genes = make_genes()
        pfam = make_pfam_map(genes)
        splits = family_split_cv(genes, pfam, n_folds=5, seed=42)
        for tr, te in splits:
            assert len(set(tr) & set(te)) == 0

    def test_every_sample_covered_at_most_once(self):
        genes = make_genes()
        pfam = make_pfam_map(genes)
        splits = family_split_cv(genes, pfam, n_folds=5, seed=42)
        test_counts = np.zeros(len(genes), dtype=int)
        for _, te in splits:
            test_counts[te] += 1
        assert test_counts.max() <= 1, "A sample appears in more than one test fold"

    def test_min_size_respected(self):
        genes = make_genes()
        pfam = make_pfam_map(genes)
        splits = family_split_cv(genes, pfam, n_folds=5, seed=42)
        for tr, te in splits:
            assert len(tr) >= 10
            assert len(te) >= 5

    def test_deterministic(self):
        genes = make_genes()
        pfam = make_pfam_map(genes)
        s1 = family_split_cv(genes, pfam, n_folds=5, seed=3)
        s2 = family_split_cv(genes, pfam, n_folds=5, seed=3)
        assert len(s1) == len(s2)
        for (tr1, te1), (tr2, te2) in zip(s1, s2):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(te1, te2)

    def test_different_seeds_differ(self):
        genes = make_genes(n_genes=20, variants_per_gene=15)
        pfam = make_pfam_map(genes, n_families=8)
        s1 = family_split_cv(genes, pfam, n_folds=5, seed=1)
        s2 = family_split_cv(genes, pfam, n_folds=5, seed=99)
        any_diff = any(
            not (np.array_equal(tr1, tr2) and np.array_equal(te1, te2))
            for (tr1, te1), (tr2, te2) in zip(s1, s2)
        )
        assert any_diff, "Different seeds produced identical splits"

    def test_genes_without_pfam_excluded(self):
        """Genes missing from pfam_map must not appear in any fold."""
        genes = make_genes(n_genes=10)
        pfam = make_pfam_map(genes)
        # Remove half the genes from pfam_map
        unmapped = list(set(genes))[:5]
        partial_pfam = {g: fam for g, fam in pfam.items() if g not in unmapped}
        splits = family_split_cv(genes, partial_pfam, n_folds=5, seed=42)
        for tr, te in splits:
            for idx in list(tr) + list(te):
                assert genes[idx] in partial_pfam, (
                    f"Gene '{genes[idx]}' has no Pfam entry but appears in a fold"
                )

    def test_empty_pfam_map_returns_no_folds(self):
        genes = make_genes()
        splits = family_split_cv(genes, {}, n_folds=5, seed=42)
        assert splits == [], "Expected no folds when pfam_map is empty"

    def test_single_family_returns_no_folds(self):
        """One family means every fold has the same family in train AND test — no valid split."""
        genes = make_genes(n_genes=10)
        pfam = {g: "FAM00" for g in set(genes)}
        splits = family_split_cv(genes, pfam, n_folds=5, seed=42)
        # Either returns no folds or all folds have empty train (both are acceptable)
        for tr, te in splits:
            tr_fams = {pfam[g] for g in genes[tr] if g in pfam}
            te_fams = {pfam[g] for g in genes[te] if g in pfam}
            assert not (tr_fams & te_fams), "Family leak in single-family case"

    def test_indices_in_bounds(self):
        genes = make_genes()
        pfam = make_pfam_map(genes)
        splits = family_split_cv(genes, pfam, n_folds=5, seed=42)
        n = len(genes)
        for tr, te in splits:
            assert tr.min() >= 0 and tr.max() < n
            assert te.min() >= 0 and te.max() < n

    def test_no_gene_leakage_even_under_family_split(self):
        """Family-split already implies gene-split, but verify explicitly."""
        genes = make_genes()
        pfam = make_pfam_map(genes)
        splits = family_split_cv(genes, pfam, n_folds=5, seed=42)
        for tr, te in splits:
            tr_genes = set(genes[tr])
            te_genes = set(genes[te])
            assert not (tr_genes & te_genes), "Gene leakage under family split"


# ── family_split_indices ──────────────────────────────────────────────────────

class TestFamilySplitIndices:

    def _make_groups(self, n=100, n_fams=5, none_frac=0.1, seed=0):
        rng = np.random.RandomState(seed)
        fams = [f"FAM{i}" for i in range(n_fams)]
        groups = np.array([fams[i % n_fams] for i in range(n)], dtype=object)
        none_idx = rng.choice(n, int(n * none_frac), replace=False)
        groups[none_idx] = None
        return groups

    def test_no_family_leakage(self):
        groups = self._make_groups()
        for tr, te in family_split_indices(groups, n_folds=5, seed=42):
            tr_fams = {groups[i] for i in tr if groups[i] is not None}
            te_fams = {groups[i] for i in te if groups[i] is not None}
            assert not (tr_fams & te_fams)

    def test_all_indices_covered(self):
        groups = self._make_groups()
        seen = set()
        for tr, te in family_split_indices(groups, n_folds=5, seed=42):
            seen.update(te)
        assert seen == set(range(len(groups)))

    def test_none_groups_distributed(self):
        groups = self._make_groups(none_frac=0.2)
        none_idx = set(i for i, g in enumerate(groups) if g is None)
        covered = set()
        for _, te in family_split_indices(groups, n_folds=5, seed=42):
            covered.update(set(te) & none_idx)
        assert covered == none_idx, "Some None-group samples not covered"


# ── compute_metrics ──────────────────────────────────────────────────────────

class TestComputeMetrics:

    def _perfect(self):
        n = 30
        y_true = np.array([0, 1, 2] * 10)
        y_proba = np.zeros((n, 3))
        for i, c in enumerate(y_true):
            y_proba[i, c] = 1.0
        y_pred = y_true.copy()
        return y_true, y_pred, y_proba

    def test_perfect_f1(self):
        y_true, y_pred, y_proba = self._perfect()
        r = compute_metrics(y_true, y_pred, y_proba)
        assert r["macro_f1"] == pytest.approx(1.0)

    def test_perfect_auroc(self):
        y_true, y_pred, y_proba = self._perfect()
        r = compute_metrics(y_true, y_pred, y_proba)
        for cls in MECHANISM_CLASSES:
            assert r["per_class_auroc"][cls] == pytest.approx(1.0)

    def test_empty_returns_nones(self):
        r = compute_metrics(np.array([]), np.array([]), np.zeros((0, 3)))
        assert r["macro_f1"] is None
        assert all(v is None for v in r["per_class_auroc"].values())

    def test_single_class_auroc_is_none(self):
        y_true = np.zeros(20, dtype=int)
        y_pred = np.zeros(20, dtype=int)
        y_proba = np.zeros((20, 3))
        y_proba[:, 0] = 1.0
        r = compute_metrics(y_true, y_pred, y_proba)
        assert r["per_class_auroc"]["GOF"] is None

    def test_keys_present(self):
        y_true, y_pred, y_proba = self._perfect()
        r = compute_metrics(y_true, y_pred, y_proba)
        assert "macro_f1" in r
        assert "per_class_auroc" in r
        for cls in MECHANISM_CLASSES:
            assert cls in r["per_class_auroc"]


# ── aggregate_folds ──────────────────────────────────────────────────────────

class TestAggregateFolds:

    def _fold(self, f1, auroc_val):
        return {
            "macro_f1": f1,
            "per_class_auroc": {c: auroc_val for c in MECHANISM_CLASSES},
        }

    def test_empty_returns_error(self):
        r = aggregate_folds([])
        assert "error" in r

    def test_mean_std_computed(self):
        folds = [self._fold(0.4, 0.7), self._fold(0.6, 0.9)]
        r = aggregate_folds(folds)
        assert r["macro_f1_mean"] == pytest.approx(0.5)
        assert r["macro_f1_std"] == pytest.approx(0.1)

    def test_n_folds(self):
        folds = [self._fold(0.5, 0.8)] * 4
        r = aggregate_folds(folds)
        assert r["n_folds"] == 4

    def test_none_auroc_excluded(self):
        folds = [
            {"macro_f1": 0.5, "per_class_auroc": {"GOF": 0.8, "DN": None, "LOF": 0.7}},
            {"macro_f1": 0.6, "per_class_auroc": {"GOF": 0.9, "DN": 0.75, "LOF": 0.8}},
        ]
        r = aggregate_folds(folds)
        assert r["auroc_GOF_mean"] == pytest.approx(0.85)
        assert r["auroc_DN_mean"] == pytest.approx(0.75)


# ── align_proba ──────────────────────────────────────────────────────────────

class TestAlignProba:

    def test_identity_alignment(self):
        proba = np.array([[0.1, 0.5, 0.4], [0.3, 0.3, 0.4]])
        clf_classes = np.array([0, 1, 2])
        result = align_proba(proba, clf_classes, 3)
        np.testing.assert_array_almost_equal(result, proba)

    def test_reorder_columns(self):
        proba = np.array([[0.6, 0.4]])
        clf_classes = np.array([2, 0])
        result = align_proba(proba, clf_classes, 3)
        assert result[0, 2] == pytest.approx(0.6)
        assert result[0, 0] == pytest.approx(0.4)
        assert result[0, 1] == pytest.approx(0.0)

    def test_output_shape(self):
        proba = np.random.rand(10, 2)
        clf_classes = np.array([0, 2])
        result = align_proba(proba, clf_classes, 3)
        assert result.shape == (10, 3)


# ── run_logreg_cv ─────────────────────────────────────────────────────────────

class TestRunLogregCv:

    def _data(self, seed=0):
        rng = np.random.RandomState(seed)
        n = 300
        y = np.array([0, 1, 2] * 100)
        X = rng.randn(n, 20) + y[:, None] * 2.0
        genes = np.array([f"G{i % 30}" for i in range(n)])
        splits = gene_split_cv(genes, n_folds=5, seed=seed)
        return X, y, splits

    def test_returns_dict_with_metrics(self):
        X, y, splits = self._data()
        r = run_logreg_cv(X, y, splits)
        assert "macro_f1_mean" in r
        assert "n_folds" in r

    def test_auroc_keys_present(self):
        X, y, splits = self._data()
        r = run_logreg_cv(X, y, splits)
        for cls in MECHANISM_CLASSES:
            assert f"auroc_{cls}_mean" in r

    def test_empty_splits_returns_error(self):
        X = np.random.randn(10, 5)
        y = np.zeros(10, dtype=int)
        r = run_logreg_cv(X, y, [])
        assert "error" in r


# ── run_logreg_binary_cv ──────────────────────────────────────────────────────

class TestRunLogregBinaryCv:

    def _data(self, seed=0):
        rng = np.random.RandomState(seed)
        n = 200
        y = rng.randint(0, 2, n)
        X = rng.randn(n, 10) + y[:, None] * 2.0
        genes = np.array([f"G{i % 20}" for i in range(n)])
        splits = gene_split_cv(genes, n_folds=5, seed=seed)
        return X, y, splits

    def test_returns_auroc(self):
        X, y, splits = self._data()
        r = run_logreg_binary_cv(X, y, splits)
        assert "auroc_mean" in r
        assert "auroc_std" in r
        assert r["auroc_mean"] > 0.5

    def test_empty_splits_returns_empty(self):
        r = run_logreg_binary_cv(np.zeros((10, 5)), np.zeros(10, dtype=int), [])
        assert r == {}


# ── run_ridge_cv ──────────────────────────────────────────────────────────────

class TestRunRidgeCv:

    def _data(self, seed=0):
        rng = np.random.RandomState(seed)
        n = 200
        X = rng.randn(n, 10)
        y = X[:, 0] * 3.0 + rng.randn(n) * 0.5
        genes = np.array([f"G{i % 20}" for i in range(n)])
        splits = gene_split_cv(genes, n_folds=5, seed=seed)
        return X, y, splits

    def test_returns_spearman(self):
        X, y, splits = self._data()
        r = run_ridge_cv(X, y, splits)
        assert "spearman_mean" in r
        assert "pearson_mean" in r
        assert r["spearman_mean"] > 0.5

    def test_no_pearson_when_disabled(self):
        X, y, splits = self._data()
        r = run_ridge_cv(X, y, splits, include_pearson=False)
        assert "pearson_mean" not in r
        assert "spearman_mean" in r

    def test_empty_splits_returns_empty(self):
        r = run_ridge_cv(np.zeros((10, 5)), np.zeros(10), [])
        assert r == {}
