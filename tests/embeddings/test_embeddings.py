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
- family_clustering.knn_family_purity: too-few-points → nan, purity in [0,1], single-family → 1.0
- family_clustering.within_between_ratio: too few pairs → nan, ratio correct sign
- family_clustering.gene_level_embeddings: shape, averaging
- esm2_mechanism._load_alphamissense_scores: key lookup, missing file, corrupt JSON
- esm2_mechanism._load_data: missing file raises, filtering drops invalid rows
- embed_variants._build_valid_pairs: filters no-uid, missing seq, bad mutation
- utils_sequences.fetch_uniprot_sequence: returns None on 404, raises TransientFetchError on network error
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
        emb = np.vstack(
            [
                rng.randn(10, 8) * 0.01 + np.array([10.0] + [0.0] * 7),
                rng.randn(10, 8) * 0.01 + np.array([-10.0] + [0.0] * 7),
            ]
        ).astype(np.float32)
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


# ---------------------------------------------------------------------------
# esm2_mechanism._load_alphamissense_scores
# ---------------------------------------------------------------------------


class TestLoadAlphaMissenseScores:

    def setup_method(self):
        from esm2_mechanism.embeddings.esm2_mechanism import _load_alphamissense_scores

        self._load = _load_alphamissense_scores

    def _variant(self, gene="TP53", aa_pos=100, aa_wt="A", aa_mut="V"):
        return {"gene": gene, "aa_pos": aa_pos, "aa_wt": aa_wt, "aa_mut": aa_mut}

    def test_scores_loaded_for_matching_keys(self, tmp_path):
        v = self._variant()
        key = f"{v['gene']}_{v['aa_pos']}_{v['aa_wt']}_{v['aa_mut']}"
        (tmp_path / "alphamissense_scores_full.json").write_text(f'{{"{key}": 0.87}}')
        scores = self._load([v], str(tmp_path))
        assert scores.shape == (1,)
        assert scores[0] == pytest.approx(0.87)

    def test_missing_key_gives_nan(self, tmp_path):
        (tmp_path / "alphamissense_scores_full.json").write_text("{}")
        scores = self._load([self._variant()], str(tmp_path))
        assert np.isnan(scores[0])

    def test_missing_file_returns_all_nan(self, tmp_path):
        scores = self._load([self._variant()], str(tmp_path))
        assert scores.shape == (1,)
        assert np.isnan(scores[0])

    def test_corrupt_json_returns_all_nan(self, tmp_path):
        (tmp_path / "alphamissense_scores_full.json").write_text("{not valid json")
        scores = self._load([self._variant()], str(tmp_path))
        assert np.all(np.isnan(scores))

    def test_multiple_variants_partial_coverage(self, tmp_path):
        v1 = self._variant("BRCA1", 50, "M", "K")
        v2 = self._variant("BRCA2", 80, "G", "D")
        key1 = f"BRCA1_50_M_K"
        (tmp_path / "alphamissense_scores_full.json").write_text(f'{{"{key1}": 0.55}}')
        scores = self._load([v1, v2], str(tmp_path))
        assert scores[0] == pytest.approx(0.55)
        assert np.isnan(scores[1])


# ---------------------------------------------------------------------------
# esm2_mechanism._load_data
# ---------------------------------------------------------------------------


class TestLoadData:

    def setup_method(self):
        from esm2_mechanism.embeddings.esm2_mechanism import _load_data

        self._load_data = _load_data

    def _write_variants(self, tmp_path, variants):
        import json

        (tmp_path / "merged_variants.json").write_text(json.dumps(variants))

    def _valid_variant(self, **overrides):
        v = {
            "gene": "TP53",
            "uniprot_id": "P04637",
            "aa_pos": 100,
            "aa_wt": "A",
            "aa_mut": "V",
            "mechanism": "GOF",
            "label_3class": "GOF",
        }
        v.update(overrides)
        return v

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            self._load_data(str(tmp_path))

    def test_returns_valid_variants(self, tmp_path):
        self._write_variants(tmp_path, [self._valid_variant()])
        result = self._load_data(str(tmp_path))
        assert len(result) == 1

    def test_drops_missing_uniprot_id(self, tmp_path):
        self._write_variants(
            tmp_path,
            [
                self._valid_variant(),
                self._valid_variant(uniprot_id=""),
                self._valid_variant(uniprot_id=None),
            ],
        )
        result = self._load_data(str(tmp_path))
        assert len(result) == 1

    def test_drops_zero_aa_pos(self, tmp_path):
        self._write_variants(
            tmp_path,
            [
                self._valid_variant(),
                self._valid_variant(aa_pos=0),
            ],
        )
        result = self._load_data(str(tmp_path))
        assert len(result) == 1

    def test_drops_missing_aa_wt_or_mut(self, tmp_path):
        self._write_variants(
            tmp_path,
            [
                self._valid_variant(),
                self._valid_variant(aa_wt=""),
                self._valid_variant(aa_mut=None),
            ],
        )
        result = self._load_data(str(tmp_path))
        assert len(result) == 1


# ---------------------------------------------------------------------------
# embed_variants._build_valid_pairs
# ---------------------------------------------------------------------------


class TestBuildValidPairs:

    def setup_method(self):
        from esm2_mechanism.embeddings.embed_variants import _build_valid_pairs

        self._build = _build_valid_pairs

    def _variant(self, uniprot_id="P04637", aa_pos=2, aa_wt="K", aa_mut="R"):
        return {
            "uniprot_id": uniprot_id,
            "aa_pos": aa_pos,
            "aa_wt": aa_wt,
            "aa_mut": aa_mut,
        }

    def test_valid_variant_included(self):
        seq_cache = {"P04637": "MKACD"}
        valid, wt_seqs, mut_seqs, positions = self._build([self._variant()], seq_cache)
        assert len(valid) == 1
        assert len(wt_seqs) == len(mut_seqs) == len(positions) == 1

    def test_missing_uniprot_id_skipped(self):
        seq_cache = {"P04637": "MKACD"}
        valid, *_ = self._build([self._variant(uniprot_id="")], seq_cache)
        assert len(valid) == 0

    def test_uniprot_id_not_in_cache_skipped(self):
        seq_cache = {}
        valid, *_ = self._build([self._variant()], seq_cache)
        assert len(valid) == 0

    def test_wt_mismatch_skipped(self):
        # aa_wt="K" but position 2 in "MXACD" is "X"
        seq_cache = {"P04637": "MXACD"}
        valid, *_ = self._build([self._variant()], seq_cache)
        assert len(valid) == 0

    def test_position_returned_is_valid_index(self):
        seq_cache = {"P04637": "MKACD"}
        valid, wt_seqs, _, positions = self._build([self._variant()], seq_cache)
        assert len(valid) == 1
        assert 1 <= positions[0] <= len(wt_seqs[0])

    def test_mutant_sequence_differs_at_position(self):
        seq_cache = {"P04637": "MKACD"}
        _, wt_seqs, mut_seqs, positions = self._build([self._variant()], seq_cache)
        pos = positions[0] - 1  # 0-indexed
        assert wt_seqs[0][pos] != mut_seqs[0][pos]
        assert mut_seqs[0][pos] == "R"

    def test_multiple_variants_mixed(self):
        seq_cache = {"P04637": "MKACD", "P99999": "ACDEF"}
        variants = [
            self._variant(),  # valid
            self._variant(uniprot_id="MISSING"),  # no sequence
            self._variant(
                uniprot_id="P99999", aa_pos=1, aa_wt="A", aa_mut="G"
            ),  # valid
        ]
        valid, *_ = self._build(variants, seq_cache)
        assert len(valid) == 2


# ---------------------------------------------------------------------------
# utils_sequences.fetch_uniprot_sequence
# ---------------------------------------------------------------------------


class TestFetchUniprotSequence:

    def setup_method(self):
        from esm2_mechanism.utils_sequences import (
            fetch_uniprot_sequence,
            TransientFetchError,
        )

        self.fetch = fetch_uniprot_sequence
        self.TransientFetchError = TransientFetchError

    def _make_handler(self, status, body=b""):
        """Return a urllib opener that always responds with the given HTTP status."""
        import urllib.request
        import urllib.error
        import io

        class _FakeResponse:
            def __init__(self):
                self.status = status
                self._body = body

            def read(self):
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

        class _Handler(urllib.request.BaseHandler):
            def http_open(self, req):
                if status == 404:
                    raise urllib.error.HTTPError(
                        req.full_url, 404, "Not Found", {}, io.BytesIO(b"")
                    )
                return _FakeResponse()

            https_open = http_open

        opener = urllib.request.OpenerDirector()
        opener.add_handler(_Handler())
        return opener

    def test_404_returns_none(self, monkeypatch):
        import urllib.request

        opener = self._make_handler(404)
        monkeypatch.setattr(urllib.request, "urlopen", opener.open)
        result = self.fetch("P99999", retries=1)
        assert result is None

    def test_network_error_raises_transient(self, monkeypatch):
        import urllib.request

        def _fail(url, timeout=None):
            raise OSError("network unreachable")

        monkeypatch.setattr(urllib.request, "urlopen", _fail)
        with pytest.raises(self.TransientFetchError):
            self.fetch("P99999", retries=1, delay=0)

    def test_success_returns_uppercase_sequence(self, monkeypatch):
        import urllib.request
        import io

        fasta = b">sp|P12345|GENE_HUMAN Some protein\nMKTAY\nIAKQR\n"

        class _FakeResp:
            def read(self):
                return fasta

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: _FakeResp())
        result = self.fetch("P12345", retries=1)
        assert result == "MKTAYIAKQR"

    def test_empty_fasta_body_returns_none(self, monkeypatch):
        import urllib.request

        class _FakeResp:
            def read(self):
                return b">sp|P12345|GENE_HUMAN\n"

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: _FakeResp())
        result = self.fetch("P12345", retries=1)
        assert result is None
