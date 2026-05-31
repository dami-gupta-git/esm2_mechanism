"""
Tests for pure-logic functions in embed_variants.py.

Covers:
- _build_valid_pairs: skips variants with no uniprot_id
- _build_valid_pairs: skips variants whose uid is absent from seq_cache
- _build_valid_pairs: skips variants where apply_missense returns None (wt mismatch)
- _build_valid_pairs: passes variants through with correct windowed sequence and position
- _build_valid_pairs: handles sequences that need windowing (len > 1022)
- _build_valid_pairs: returns empty lists when no variants pass
- _flush_checkpoint: writes four .npy files atomically (no .tmp left)
- _flush_checkpoint: array contents roundtrip correctly
"""

import os

import numpy as np
import pytest

from esm2_mech.embeddings.embed_variants import _build_valid_pairs
from esm2_mech.utils.embed import _flush_checkpoint


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _variant(gene="BRCA1", uid="P38398", aa_pos=6, aa_wt="M", aa_mut="V", label="GOF"):
    return {
        "gene": gene,
        "uniprot_id": uid,
        "aa_pos": aa_pos,
        "aa_wt": aa_wt,
        "aa_mut": aa_mut,
        "label_3class": label,
    }


SEQ = "ACDEFMKTAY"  # len=10, pos 6 (1-indexed) = "M"


# ---------------------------------------------------------------------------
# _build_valid_pairs
# ---------------------------------------------------------------------------

class TestBuildValidPairs:

    def test_valid_variant_passes_through(self):
        variants = [_variant(uid="P1", aa_pos=6, aa_wt="M", aa_mut="V")]
        seq_cache = {"P1": SEQ}
        valid, wt_seqs, mut_seqs, positions = _build_valid_pairs(variants, seq_cache)
        assert len(valid) == 1
        assert len(wt_seqs) == 1
        assert len(mut_seqs) == 1
        assert len(positions) == 1

    def test_wt_seq_is_windowed_sequence(self):
        variants = [_variant(uid="P1", aa_pos=6, aa_wt="M", aa_mut="V")]
        seq_cache = {"P1": SEQ}
        _, wt_seqs, _, _ = _build_valid_pairs(variants, seq_cache)
        # Short sequence: window_sequence returns it unchanged
        assert wt_seqs[0] == SEQ

    def test_mut_seq_has_substitution_applied(self):
        variants = [_variant(uid="P1", aa_pos=6, aa_wt="M", aa_mut="V")]
        seq_cache = {"P1": SEQ}
        _, wt_seqs, mut_seqs, _ = _build_valid_pairs(variants, seq_cache)
        assert wt_seqs[0][5] == "M"   # 0-indexed position 5 = aa_pos 6
        assert mut_seqs[0][5] == "V"

    def test_position_returned_is_one_indexed(self):
        variants = [_variant(uid="P1", aa_pos=6, aa_wt="M", aa_mut="V")]
        seq_cache = {"P1": SEQ}
        _, wt_seqs, _, positions = _build_valid_pairs(variants, seq_cache)
        assert positions[0] == 6
        assert wt_seqs[0][positions[0] - 1] == "M"

    def test_skips_variant_with_no_uniprot_id(self):
        variants = [_variant(uid="", aa_pos=6, aa_wt="M", aa_mut="V")]
        seq_cache = {"P1": SEQ}
        valid, wt_seqs, mut_seqs, positions = _build_valid_pairs(variants, seq_cache)
        assert valid == []
        assert wt_seqs == []

    def test_skips_variant_with_none_uniprot_id(self):
        v = _variant(uid="P1", aa_pos=5, aa_wt="M", aa_mut="V")
        v["uniprot_id"] = None
        seq_cache = {"P1": SEQ}
        valid, _, _, _ = _build_valid_pairs([v], seq_cache)
        assert valid == []

    def test_skips_variant_uid_not_in_cache(self):
        variants = [_variant(uid="MISSING", aa_pos=6, aa_wt="M", aa_mut="V")]
        seq_cache = {"P1": SEQ}
        valid, _, _, _ = _build_valid_pairs(variants, seq_cache)
        assert valid == []

    def test_skips_variant_with_wt_mismatch(self):
        # aa_wt="K" but position 6 in SEQ is "M" → apply_missense returns None
        variants = [_variant(uid="P1", aa_pos=6, aa_wt="K", aa_mut="V")]
        seq_cache = {"P1": SEQ}
        valid, _, _, _ = _build_valid_pairs(variants, seq_cache)
        assert valid == []

    def test_returns_empty_when_no_variants(self):
        valid, wt_seqs, mut_seqs, positions = _build_valid_pairs([], {})
        assert valid == wt_seqs == mut_seqs == positions == []

    def test_multiple_variants_all_valid(self):
        seq_a = "ACDEFMKTAY"  # pos 6 (1-indexed) = M
        seq_b = "GHIKLMNPQR"  # pos 3 (1-indexed) = I
        variants = [
            _variant(uid="P1", aa_pos=6, aa_wt="M", aa_mut="V"),
            _variant(uid="P2", aa_pos=3, aa_wt="I", aa_mut="A"),
        ]
        seq_cache = {"P1": seq_a, "P2": seq_b}
        valid, wt_seqs, mut_seqs, positions = _build_valid_pairs(variants, seq_cache)
        assert len(valid) == 2
        assert mut_seqs[0][5] == "V"
        assert mut_seqs[1][2] == "A"

    def test_mixed_valid_and_invalid(self):
        variants = [
            _variant(uid="P1", aa_pos=6, aa_wt="M", aa_mut="V"),          # valid
            _variant(uid="MISSING", aa_pos=6, aa_wt="M", aa_mut="V"),     # no cache entry
            _variant(uid="P1", aa_pos=6, aa_wt="K", aa_mut="V"),          # wt mismatch
        ]
        seq_cache = {"P1": SEQ}
        valid, wt_seqs, _, _ = _build_valid_pairs(variants, seq_cache)
        assert len(valid) == 1
        assert valid[0]["uniprot_id"] == "P1"
        assert valid[0]["aa_wt"] == "M"

    def test_long_sequence_windowed(self):
        # Build a 2000-residue sequence with mutation at position 1000
        long_seq = "A" * 999 + "M" + "A" * 1000  # pos 1000 = M
        variants = [_variant(uid="P1", aa_pos=1000, aa_wt="M", aa_mut="V")]
        seq_cache = {"P1": long_seq}
        valid, wt_seqs, mut_seqs, positions = _build_valid_pairs(variants, seq_cache)
        assert len(valid) == 1
        assert len(wt_seqs[0]) <= 1022
        # Mutation site must still be correct in windowed sequence
        pos = positions[0]
        assert wt_seqs[0][pos - 1] == "M"
        assert mut_seqs[0][pos - 1] == "V"


# ---------------------------------------------------------------------------
# _flush_checkpoint
# ---------------------------------------------------------------------------

class TestFlushCheckpoint:

    def _arrays(self, n=3, d=4):
        return (
            [np.random.randn(d).astype(np.float32) for _ in range(n)],
            [np.random.randn(d).astype(np.float32) for _ in range(n)],
            [np.random.randn(d).astype(np.float32) for _ in range(n)],
            [np.random.randn(d).astype(np.float32) for _ in range(n)],
        )

    def test_writes_four_npy_files(self, tmp_path):
        wt_m, mut_m, wt_p, mut_p = self._arrays()
        valid_slice = [_variant()]
        _flush_checkpoint(str(tmp_path), wt_m, mut_m, wt_p, mut_p, valid_slice, n_done=1)
        for name in ("embeddings_wt_mean", "embeddings_mut_mean", "embeddings_wt_pos", "embeddings_mut_pos"):
            assert (tmp_path / f"{name}.npy").exists()

    def test_no_tmp_files_left(self, tmp_path):
        wt_m, mut_m, wt_p, mut_p = self._arrays()
        _flush_checkpoint(str(tmp_path), wt_m, mut_m, wt_p, mut_p, [], n_done=0)
        assert not any(str(f).endswith(".tmp") for f in tmp_path.iterdir())

    def test_npy_shapes_correct(self, tmp_path):
        n, d = 5, 8
        wt_m, mut_m, wt_p, mut_p = self._arrays(n=n, d=d)
        _flush_checkpoint(str(tmp_path), wt_m, mut_m, wt_p, mut_p, [], n_done=n)
        arr = np.load(tmp_path / "embeddings_wt_mean.npy")
        assert arr.shape == (n, d)

    def test_arrays_roundtrip_correctly(self, tmp_path):
        wt_m = [np.array([1.0, 2.0], dtype=np.float32)]
        mut_m = [np.array([3.0, 4.0], dtype=np.float32)]
        wt_p = [np.array([5.0, 6.0], dtype=np.float32)]
        mut_p = [np.array([7.0, 8.0], dtype=np.float32)]
        _flush_checkpoint(str(tmp_path), wt_m, mut_m, wt_p, mut_p, [], n_done=1)
        assert np.allclose(np.load(tmp_path / "embeddings_wt_mean.npy"), [[1.0, 2.0]])
        assert np.allclose(np.load(tmp_path / "embeddings_mut_mean.npy"), [[3.0, 4.0]])
        assert np.allclose(np.load(tmp_path / "embeddings_wt_pos.npy"), [[5.0, 6.0]])
        assert np.allclose(np.load(tmp_path / "embeddings_mut_pos.npy"), [[7.0, 8.0]])
