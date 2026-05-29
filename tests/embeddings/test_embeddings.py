"""
Unit tests for pure-logic functions in the embeddings modules.

Covers:
- utils_sequences.apply_missense: correct substitution, mismatch → None, OOB → None
- utils_sequences.window_sequence: short seq unchanged, long seq windowed correctly,
  new_pos always within window, boundary positions
- megascale_stability.random_split_cv: covers all indices, no duplicates, n_folds output
- megascale_stability.protein_split_cv: no protein leakage, min-size respected
- megascale_stability.cluster_split_cv: no cluster leakage, min-size respected
- megascale_stability.auroc_at_median: above-median positive, single-class → nan
- megascale_stability.apply_decision_rule: all five outcomes
- family_clustering.knn_family_purity: single-family → nan, purity in [0,1]
- family_clustering.within_between_ratio: too few pairs → nan, ratio correct sign
- family_clustering.gene_level_embeddings: shape, averaging
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# utils_sequences.apply_missense
# ---------------------------------------------------------------------------

class TestApplyMissense:

    def setup_method(self):
        from esm2_mechanism.utils_sequences import apply_missense
        self.apply_missense = apply_missense

    def test_correct_substitution(self):
        result = self.apply_missense("MKTAY", aa_pos=1, aa_wt="M", aa_mut="A")
        assert result == "AKTAY"

    def test_middle_position(self):
        result = self.apply_missense("MKTAY", aa_pos=3, aa_wt="T", aa_mut="S")
        assert result == "MKSAY"

    def test_last_position(self):
        result = self.apply_missense("MKTAY", aa_pos=5, aa_wt="Y", aa_mut="F")
        assert result == "MKTAF"

    def test_wt_mismatch_returns_none(self):
        result = self.apply_missense("MKTAY", aa_pos=1, aa_wt="A", aa_mut="V")
        assert result is None

    def test_position_zero_returns_none(self):
        result = self.apply_missense("MKTAY", aa_pos=0, aa_wt="M", aa_mut="A")
        assert result is None

    def test_position_beyond_length_returns_none(self):
        result = self.apply_missense("MKTAY", aa_pos=6, aa_wt="X", aa_mut="A")
        assert result is None

    def test_same_aa_identity(self):
        result = self.apply_missense("MKTAY", aa_pos=2, aa_wt="K", aa_mut="K")
        assert result == "MKTAY"

    def test_length_unchanged(self):
        seq = "ACDEFGHIKLM"
        result = self.apply_missense(seq, aa_pos=4, aa_wt="E", aa_mut="Q")
        assert len(result) == len(seq)


# ---------------------------------------------------------------------------
# utils_sequences.window_sequence
# ---------------------------------------------------------------------------

class TestWindowSequence:

    def setup_method(self):
        from esm2_mechanism.utils_sequences import window_sequence, MAX_SEQ_LEN
        self.window_sequence = window_sequence
        self.MAX_SEQ_LEN = MAX_SEQ_LEN

    def test_short_seq_returned_unchanged(self):
        seq = "ACDEFGHIKLM"
        windowed, new_pos = self.window_sequence(seq, aa_pos=5)
        assert windowed == seq
        assert new_pos == 5

    def test_long_seq_truncated_to_max_len(self):
        seq = "A" * (self.MAX_SEQ_LEN + 100)
        windowed, _ = self.window_sequence(seq, aa_pos=self.MAX_SEQ_LEN // 2)
        assert len(windowed) <= self.MAX_SEQ_LEN

    def test_new_pos_points_to_correct_residue(self):
        seq = "A" * 200 + "M" + "A" * 200  # M at position 201 (1-indexed)
        # Use a small max_len to force windowing
        windowed, new_pos = self.window_sequence(seq, aa_pos=201, max_len=100)
        assert windowed[new_pos - 1] == "M"

    def test_new_pos_one_indexed_in_range(self):
        seq = "A" * 2000
        windowed, new_pos = self.window_sequence(seq, aa_pos=1000)
        assert 1 <= new_pos <= len(windowed)

    def test_position_at_start(self):
        seq = "M" + "A" * (self.MAX_SEQ_LEN + 100)
        windowed, new_pos = self.window_sequence(seq, aa_pos=1)
        assert windowed[new_pos - 1] == "M"

    def test_position_at_end(self):
        n = self.MAX_SEQ_LEN + 50
        seq = "A" * (n - 1) + "M"
        windowed, new_pos = self.window_sequence(seq, aa_pos=n)
        assert windowed[new_pos - 1] == "M"

    def test_exact_max_len_not_windowed(self):
        seq = "A" * self.MAX_SEQ_LEN
        windowed, new_pos = self.window_sequence(seq, aa_pos=500)
        assert windowed == seq
        assert new_pos == 500


# ---------------------------------------------------------------------------
# megascale_stability CV splits
# ---------------------------------------------------------------------------

class TestRandomSplitCv:

    def setup_method(self):
        from esm2_mechanism.embeddings.megascale_stability import random_split_cv
        self.random_split_cv = random_split_cv

    def test_covers_all_indices(self):
        n = 50
        splits = self.random_split_cv(n, n_folds=5, seed=0)
        seen = set()
        for _, te in splits:
            seen.update(te.tolist())
        assert seen == set(range(n))

    def test_no_duplicates_in_test(self):
        n = 50
        splits = self.random_split_cv(n, n_folds=5, seed=0)
        test_counts = np.zeros(n, dtype=int)
        for _, te in splits:
            test_counts[te] += 1
        assert test_counts.max() == 1

    def test_n_splits_equals_n_folds(self):
        splits = self.random_split_cv(30, n_folds=5, seed=0)
        assert len(splits) == 5

    def test_train_test_disjoint(self):
        splits = self.random_split_cv(30, n_folds=5, seed=0)
        for tr, te in splits:
            assert len(set(tr.tolist()) & set(te.tolist())) == 0


class TestProteinSplitCv:

    def setup_method(self):
        from esm2_mechanism.embeddings.megascale_stability import protein_split_cv
        self.protein_split_cv = protein_split_cv

    def _proteins(self, n=60, n_prots=10):
        return np.array([f"P{i % n_prots}" for i in range(n)])

    def test_no_protein_leakage(self):
        proteins = self._proteins()
        for tr, te in self.protein_split_cv(proteins, n_folds=5, seed=0):
            tr_prots = set(proteins[tr])
            te_prots = set(proteins[te])
            assert not (tr_prots & te_prots)

    def test_min_size_respected(self):
        proteins = self._proteins()
        for tr, te in self.protein_split_cv(proteins, n_folds=5, seed=0):
            assert len(tr) >= 10
            assert len(te) >= 5


class TestClusterSplitCv:

    def setup_method(self):
        from esm2_mechanism.embeddings.megascale_stability import cluster_split_cv
        self.cluster_split_cv = cluster_split_cv

    def _data(self, n=60, n_prots=10):
        proteins = np.array([f"P{i % n_prots}" for i in range(n)])
        # Two proteins per cluster
        cluster_map = {f"P{i}": f"C{i // 2}" for i in range(n_prots)}
        return proteins, cluster_map

    def test_no_cluster_leakage(self):
        proteins, cluster_map = self._data()
        for tr, te in self.cluster_split_cv(proteins, cluster_map, n_folds=5, seed=0):
            tr_clusters = {cluster_map.get(p, p) for p in proteins[tr]}
            te_clusters = {cluster_map.get(p, p) for p in proteins[te]}
            assert not (tr_clusters & te_clusters)

    def test_min_size_respected(self):
        proteins, cluster_map = self._data()
        for tr, te in self.cluster_split_cv(proteins, cluster_map, n_folds=5, seed=0):
            assert len(tr) >= 10
            assert len(te) >= 5


# ---------------------------------------------------------------------------
# megascale_stability.auroc_at_median
# ---------------------------------------------------------------------------

class TestAurocAtMedian:

    def setup_method(self):
        from esm2_mechanism.embeddings.megascale_stability import auroc_at_median
        self.auroc_at_median = auroc_at_median

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([0.1, 0.2, 0.9, 0.95])  # above-median get high scores
        result = self.auroc_at_median(y, pred)
        assert result == pytest.approx(1.0)

    def test_single_class_returns_nan(self):
        # All above median → binary all-1
        y = np.array([5.0, 5.0, 5.0, 5.0])
        pred = np.array([0.5, 0.6, 0.7, 0.8])
        result = self.auroc_at_median(y, pred)
        assert np.isnan(result)

    def test_result_in_range(self):
        rng = np.random.RandomState(0)
        y = rng.randn(50)
        pred = rng.randn(50)
        result = self.auroc_at_median(y, pred)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# megascale_stability.apply_decision_rule
# ---------------------------------------------------------------------------

class TestApplyDecisionRule:

    def setup_method(self):
        from esm2_mechanism.embeddings.megascale_stability import apply_decision_rule
        self.apply_decision_rule = apply_decision_rule

    def test_leaky(self):
        assert self.apply_decision_rule(0.6, 0.4, 0.08) == "LEAKY"

    def test_heterogeneous(self):
        assert self.apply_decision_rule(0.6, 0.58, 0.20) == "HETEROGENEOUS"

    def test_robust(self):
        assert self.apply_decision_rule(0.6, 0.58, 0.08) == "ROBUST"

    def test_weak(self):
        assert self.apply_decision_rule(0.4, 0.38, 0.05) == "WEAK"

    def test_null(self):
        assert self.apply_decision_rule(0.2, 0.18, 0.05) == "NULL"


# ---------------------------------------------------------------------------
# family_clustering.knn_family_purity
# ---------------------------------------------------------------------------

class TestKnnFamilyPurity:

    def setup_method(self):
        from esm2_mechanism.mechanism.family_clustering import knn_family_purity
        self.knn_family_purity = knn_family_purity

    def test_too_few_points_returns_nan(self):
        emb = np.random.randn(3, 4).astype(np.float32)
        families = np.array(["A", "B", "A"])
        real, null, z = self.knn_family_purity(emb, families, k=5)
        assert np.isnan(real)

    def test_purity_in_range(self):
        rng = np.random.RandomState(0)
        emb = rng.randn(30, 8).astype(np.float32)
        families = np.array(["A", "B"] * 15)
        real, null, z = self.knn_family_purity(emb, families, k=3, n_shuffles=5)
        assert 0.0 <= real <= 1.0

    def test_single_family_purity_is_one(self):
        rng = np.random.RandomState(0)
        emb = rng.randn(20, 8).astype(np.float32)
        families = np.array(["A"] * 20)
        real, null, z = self.knn_family_purity(emb, families, k=5, n_shuffles=5)
        assert real == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# family_clustering.within_between_ratio
# ---------------------------------------------------------------------------

class TestWithinBetweenRatio:

    def setup_method(self):
        from esm2_mechanism.mechanism.family_clustering import within_between_ratio
        self.within_between_ratio = within_between_ratio

    def test_too_few_pairs_returns_nan(self):
        emb = np.random.randn(4, 4).astype(np.float32)
        families = np.array(["A", "B", "C", "D"])  # 0 within-family pairs
        ratio, null, z = self.within_between_ratio(emb, families, n_shuffles=3)
        assert np.isnan(ratio)

    def test_tight_clusters_ratio_below_one(self):
        rng = np.random.RandomState(0)
        # Tight clusters — within-family distance should be small
        emb = np.vstack([
            rng.randn(10, 8) * 0.01 + np.array([10.0] + [0.0]*7),
            rng.randn(10, 8) * 0.01 + np.array([-10.0] + [0.0]*7),
        ]).astype(np.float32)
        families = np.array(["A"] * 10 + ["B"] * 10)
        ratio, null, z = self.within_between_ratio(emb, families, n_shuffles=5)
        assert ratio < 1.0


# ---------------------------------------------------------------------------
# family_clustering.gene_level_embeddings
# ---------------------------------------------------------------------------

class TestGeneLevelEmbeddings:

    def setup_method(self):
        from esm2_mechanism.mechanism.family_clustering import gene_level_embeddings
        self.gene_level_embeddings = gene_level_embeddings

    def test_output_shape(self):
        emb = np.random.randn(10, 8).astype(np.float32)
        genes_arr = np.array(["G1"] * 5 + ["G2"] * 5)
        unique_genes, gene_emb = self.gene_level_embeddings(emb, genes_arr)
        assert gene_emb.shape == (2, 8)
        assert len(unique_genes) == 2

    def test_averaging_correct(self):
        emb = np.array([[1.0, 0.0], [3.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        genes_arr = np.array(["G1", "G1", "G2"])
        unique_genes, gene_emb = self.gene_level_embeddings(emb, genes_arr)
        g1_idx = list(unique_genes).index("G1")
        np.testing.assert_allclose(gene_emb[g1_idx], [2.0, 0.0], atol=1e-5)

    def test_single_variant_gene(self):
        emb = np.array([[5.0, 3.0]], dtype=np.float32)
        genes_arr = np.array(["SOLO"])
        unique_genes, gene_emb = self.gene_level_embeddings(emb, genes_arr)
        np.testing.assert_allclose(gene_emb[0], [5.0, 3.0], atol=1e-5)
