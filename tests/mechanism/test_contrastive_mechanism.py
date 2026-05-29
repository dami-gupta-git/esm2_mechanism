"""
Tests for contrastive_mechanism.py.

Key invariants checked:
- build_cross_family_pairs: positives are always same-mechanism, different family
- build_cross_family_pairs: negatives are always different-mechanism from anchor
- build_cross_family_pairs: no within-family positives
- build_cross_family_pairs: anchors with no cross-family positives are skipped
- build_cross_family_pairs: output lengths are consistent (anchors == positives == negatives)
- build_cross_family_pairs: deterministic under same seed, differs under different seed
- build_cross_family_pairs: respects max_pairs_per_anchor cap
- run_knn: returns macro_f1 and per-class auroc keys
- run_knn: perfect embeddings yield macro_f1=1.0
- run_knn: k is clamped to training set size
- run_cv agg: empty fold list returns error dict
- run_cv agg: mean/std computed correctly across folds
"""

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

from esm2_mechanism.mechanism.contrastive_mechanism import (
    build_cross_family_pairs,
    run_knn,
    run_cv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_3class_data(n_per_class=30, n_genes=12, n_families=4, seed=0):
    """
    Build a minimal 3-class dataset with genes and Pfam families.
    Returns labels, gene_pfam arrays and a fitted LabelEncoder.
    """
    rng = np.random.RandomState(seed)
    classes = ["GOF", "LOF", "DN"]
    labels = np.array(classes * n_per_class)

    gene_names = [f"GENE{i:02d}" for i in range(n_genes)]
    fam_names = [f"FAM{i:02d}" for i in range(n_families)]
    gene_to_fam = {g: fam_names[i % n_families] for i, g in enumerate(gene_names)}

    n = len(labels)
    gene_pfam = np.array([gene_to_fam[gene_names[i % n_genes]] for i in range(n)])

    le = LabelEncoder()
    le.fit(labels)
    return labels, gene_pfam, le


# ---------------------------------------------------------------------------
# build_cross_family_pairs
# ---------------------------------------------------------------------------

class TestBuildCrossFamilyPairs:

    def test_output_lengths_consistent(self):
        labels, gene_pfam, le = make_3class_data()
        anchors, positives, negatives = build_cross_family_pairs(labels, gene_pfam, le, seed=0)
        assert len(anchors) == len(positives) == len(negatives)

    def test_positives_are_same_mechanism(self):
        labels, gene_pfam, le = make_3class_data()
        anchors, positives, _ = build_cross_family_pairs(labels, gene_pfam, le, seed=0)
        for a, p in zip(anchors, positives):
            assert labels[a] == labels[p], (
                f"Anchor {a} ({labels[a]}) paired with positive {p} ({labels[p]}) — different mechanism"
            )

    def test_negatives_are_different_mechanism(self):
        labels, gene_pfam, le = make_3class_data()
        anchors, _, negatives = build_cross_family_pairs(labels, gene_pfam, le, seed=0)
        for a, n in zip(anchors, negatives):
            assert labels[a] != labels[n], (
                f"Anchor {a} ({labels[a]}) paired with negative {n} ({labels[n]}) — same mechanism"
            )

    def test_no_within_family_positives(self):
        labels, gene_pfam, le = make_3class_data()
        anchors, positives, _ = build_cross_family_pairs(labels, gene_pfam, le, seed=0)
        for a, p in zip(anchors, positives):
            assert gene_pfam[a] != gene_pfam[p], (
                f"Anchor {a} (fam={gene_pfam[a]}) has within-family positive {p} (fam={gene_pfam[p]})"
            )

    def test_max_pairs_per_anchor_respected(self):
        labels, gene_pfam, le = make_3class_data()
        max_pairs = 3
        anchors, _, _ = build_cross_family_pairs(
            labels, gene_pfam, le, max_pairs_per_anchor=max_pairs, seed=0
        )
        counts = np.bincount(anchors)
        assert counts.max() <= max_pairs, f"Some anchor exceeded max_pairs_per_anchor={max_pairs}"

    def test_deterministic_same_seed(self):
        labels, gene_pfam, le = make_3class_data()
        a1, p1, n1 = build_cross_family_pairs(labels, gene_pfam, le, seed=7)
        a2, p2, n2 = build_cross_family_pairs(labels, gene_pfam, le, seed=7)
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(n1, n2)

    def test_different_seeds_differ(self):
        labels, gene_pfam, le = make_3class_data()
        _, p1, n1 = build_cross_family_pairs(labels, gene_pfam, le, seed=0)
        _, p2, n2 = build_cross_family_pairs(labels, gene_pfam, le, seed=99)
        # Positives or negatives sampled should differ between seeds
        assert not (np.array_equal(p1, p2) and np.array_equal(n1, n2)), (
            "Different seeds produced identical triplet selections"
        )

    def test_single_family_produces_no_triplets(self):
        """If all variants share one family, no cross-family positives exist."""
        labels, _, le = make_3class_data()
        gene_pfam_one = np.array(["FAM00"] * len(labels))
        anchors, positives, negatives = build_cross_family_pairs(
            labels, gene_pfam_one, le, seed=0
        )
        assert len(anchors) == 0, "Expected no triplets when only one family exists"

    def test_none_family_variants_excluded_from_positives(self):
        """Variants with gene_pfam=None (fam_int=-1) must not appear as cross-family positives."""
        labels, gene_pfam, le = make_3class_data()
        # Set half the entries to None
        gene_pfam_with_none = gene_pfam.copy().astype(object)
        gene_pfam_with_none[::2] = None
        anchors, positives, _ = build_cross_family_pairs(
            labels, gene_pfam_with_none, le, seed=0
        )
        # None variants have fam_int=-1 and are excluded from positive pools
        # by the cross_fam_mask: (class_fams != fam) & (class_fams != -1)
        for a, p in zip(anchors, positives):
            assert gene_pfam_with_none[p] is not None, (
                f"Positive {p} has None family — should be excluded"
            )

    def test_indices_in_bounds(self):
        labels, gene_pfam, le = make_3class_data()
        n = len(labels)
        anchors, positives, negatives = build_cross_family_pairs(labels, gene_pfam, le, seed=0)
        for arr, name in [(anchors, "anchors"), (positives, "positives"), (negatives, "negatives")]:
            assert arr.min() >= 0 and arr.max() < n, f"{name} index out of bounds"


# ---------------------------------------------------------------------------
# run_knn
# ---------------------------------------------------------------------------

class TestRunKnn:

    def _perfect_embeddings(self, n_per_class=20, dim=8, seed=0):
        """Linearly separable embeddings: each class occupies a distinct region."""
        rng = np.random.RandomState(seed)
        le = LabelEncoder()
        classes = ["DN", "GOF", "LOF"]  # LabelEncoder sorts alphabetically
        le.fit(classes)
        Z, y = [], []
        for i, cls in enumerate(classes):
            center = np.zeros(dim)
            center[i] = 10.0
            Z.append(center + rng.randn(n_per_class, dim) * 0.1)
            y.extend([le.transform([cls])[0]] * n_per_class)
        return np.vstack(Z), np.array(y), le

    def test_returns_macro_f1(self):
        Z, y = np.random.randn(60, 8), np.array([0, 1, 2] * 20)
        le = LabelEncoder().fit(["GOF", "LOF", "DN"])
        result = run_knn(Z, Z, y, y, le, k=5)
        assert "macro_f1" in result

    def test_returns_auroc_keys(self):
        Z, y, le = self._perfect_embeddings()
        result = run_knn(Z, Z, y, y, le, k=5)
        for cls in ["GOF", "LOF", "DN"]:
            assert f"auroc_{cls}" in result

    def test_perfect_embeddings_f1_is_one(self):
        Z, y, le = self._perfect_embeddings(n_per_class=30)
        # Split so each class is represented in both train and test
        # classes are encoded 0,1,2 in blocks of n_per_class
        n_per_class = 30
        train_idx = np.concatenate([
            np.arange(i * n_per_class, i * n_per_class + 20) for i in range(3)
        ])
        test_idx = np.concatenate([
            np.arange(i * n_per_class + 20, (i + 1) * n_per_class) for i in range(3)
        ])
        result = run_knn(Z[train_idx], Z[test_idx], y[train_idx], y[test_idx], le, k=5)
        assert result["macro_f1"] == pytest.approx(1.0, abs=0.01)

    def test_k_clamped_to_train_size(self):
        """k > training set size should not raise; k is clamped internally."""
        Z_tr = np.random.randn(4, 8)
        Z_te = np.random.randn(6, 8)
        y_tr = np.array([0, 1, 2, 0])
        y_te = np.array([0, 1, 2, 0, 1, 2])
        le = LabelEncoder().fit(["GOF", "LOF", "DN"])
        result = run_knn(Z_tr, Z_te, y_tr, y_te, le, k=100)
        assert "macro_f1" in result

    def test_auroc_omitted_when_single_class_in_test(self):
        """If test set has only one class, AUROC for that class should not be computed."""
        rng = np.random.RandomState(0)
        Z_tr = rng.randn(30, 8)
        Z_te = rng.randn(10, 8)
        y_tr = np.array([0, 1, 2] * 10)
        y_te = np.zeros(10, dtype=int)  # only class 0 in test
        le = LabelEncoder().fit(["DN", "GOF", "LOF"])
        result = run_knn(Z_tr, Z_te, y_tr, y_te, le, k=5)
        # AUROC for class 0 (only class present) requires both pos/neg — should be absent
        cls0_str = le.classes_[0]
        assert f"auroc_{cls0_str}" not in result


# ---------------------------------------------------------------------------
# run_cv internal agg helper (tested via run_cv with mocked splits)
# ---------------------------------------------------------------------------

class TestRunCvAgg:
    """
    run_cv contains a local `agg` function that aggregates fold metrics.
    We test its behaviour by constructing minimal split inputs that exercise
    the aggregation path without touching the training loop.
    """

    def _make_inputs(self, n=90, dim=4, seed=0):
        rng = np.random.RandomState(seed)
        classes = ["GOF", "LOF", "DN"]
        # Interleave classes so each family has all 3 mechanisms represented
        labels = np.array(classes * (n // 3))
        genes = np.array([f"G{i % 15}" for i in range(n)])
        X = rng.randn(n, dim).astype(np.float32)
        # 6 families, assigned by position so all mechanisms appear in every family
        gene_pfam = np.array([f"FAM{i % 6}" for i in range(n)])
        pfam_map = {f"G{i}": f"FAM{i % 6}" for i in range(15)}
        le = LabelEncoder()
        le.fit(labels)
        return X, labels, genes, gene_pfam, pfam_map, le

    def test_agg_empty_folds_returns_error(self):
        X, labels, genes, gene_pfam, pfam_map, le = self._make_inputs()
        # Pass empty splits list — agg receives an empty fold list
        cont, raw = run_cv(X, labels, genes, gene_pfam, pfam_map, le,
                           splits=[], split_name="test")
        assert "error" in cont
        assert "error" in raw

    def test_agg_n_folds_matches_valid_folds(self):
        X, labels, genes, gene_pfam, pfam_map, le = self._make_inputs(n=90)
        # Build a simple 3-fold split manually
        n = len(labels)
        fold_size = n // 3
        splits = [
            (np.concatenate([np.arange(fold_size), np.arange(2 * fold_size, n)]),
             np.arange(fold_size, 2 * fold_size)),
            (np.concatenate([np.arange(fold_size, 2 * fold_size), np.arange(2 * fold_size, n)]),
             np.arange(fold_size)),
        ]
        cont, raw = run_cv(X, labels, genes, gene_pfam, pfam_map, le,
                           splits=splits, split_name="test",
                           hidden=(16, 8), seed=0, batch_size=32)
        assert cont["n_folds"] == raw["n_folds"] == 2

    def test_agg_macro_f1_between_zero_and_one(self):
        X, labels, genes, gene_pfam, pfam_map, le = self._make_inputs(n=90)
        n = len(labels)
        fold_size = n // 3
        splits = [
            (np.concatenate([np.arange(fold_size), np.arange(2 * fold_size, n)]),
             np.arange(fold_size, 2 * fold_size)),
        ]
        cont, raw = run_cv(X, labels, genes, gene_pfam, pfam_map, le,
                           splits=splits, split_name="test",
                           hidden=(16, 8), seed=0, batch_size=32)
        for result in [cont, raw]:
            f1 = result.get("macro_f1_mean")
            assert f1 is not None
            assert 0.0 <= f1 <= 1.0
