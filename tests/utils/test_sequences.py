"""
Tests for esm2_mech.utils.sequences (pure functions only).

Invariants:
- apply_missense: correct substitution at 1-indexed position
- apply_missense: returns None on wt mismatch, out-of-bounds pos, zero pos
- apply_missense: original sequence is not mutated
- window_sequence: short sequences returned unchanged with same pos
- window_sequence: long sequences windowed to <= max_len
- window_sequence: variant position is preserved in windowed coordinates
- window_sequence: windowed position is 1-indexed and within bounds
- window_sequence: position at sequence start/end handled without clipping error
"""

import pytest
from esm2_mech.utils.sequences import apply_missense, window_sequence


class TestApplyMissense:

    def test_correct_substitution(self):
        assert apply_missense("ACDEF", 2, "C", "G") == "AGDEF"

    def test_first_position(self):
        assert apply_missense("ACDEF", 1, "A", "M") == "MCDEF"

    def test_last_position(self):
        assert apply_missense("ACDEF", 5, "F", "W") == "ACDEW"

    def test_wt_mismatch_returns_none(self):
        assert apply_missense("ACDEF", 2, "X", "G") is None

    def test_pos_zero_returns_none(self):
        assert apply_missense("ACDEF", 0, "A", "M") is None

    def test_pos_beyond_end_returns_none(self):
        assert apply_missense("ACDEF", 6, "F", "W") is None

    def test_negative_pos_returns_none(self):
        assert apply_missense("ACDEF", -1, "A", "M") is None

    def test_single_residue_sequence(self):
        assert apply_missense("A", 1, "A", "G") == "G"

    def test_original_not_mutated(self):
        seq = "ACDEF"
        apply_missense(seq, 1, "A", "M")
        assert seq == "ACDEF"

    def test_synonymous_substitution(self):
        # wt == mut is valid; should return sequence unchanged
        assert apply_missense("ACDEF", 2, "C", "C") == "ACDEF"


class TestWindowSequence:

    def test_short_sequence_unchanged(self):
        seq = "A" * 100
        result_seq, result_pos, _ = window_sequence(seq, 50, window_half=500, max_len=1022)
        assert result_seq == seq
        assert result_pos == 50

    def test_exact_max_len_unchanged(self):
        seq = "A" * 1022
        result_seq, result_pos, _ = window_sequence(seq, 511, window_half=500, max_len=1022)
        assert result_seq == seq
        assert result_pos == 511

    def test_long_sequence_windowed(self):
        seq = "A" * 2000
        result_seq, result_pos, _ = window_sequence(seq, 1000, window_half=500, max_len=1022)
        assert len(result_seq) <= 1022

    def test_variant_position_preserved(self):
        # Build a sequence where the variant residue is unique so we can verify it.
        seq = "A" * 2000
        seq = seq[:999] + "M" + seq[1000:]  # position 1000 (1-indexed) is M
        result_seq, result_pos, _ = window_sequence(seq, 1000, window_half=500, max_len=1022)
        assert result_seq[result_pos - 1] == "M"

    def test_new_pos_is_one_indexed_and_in_bounds(self):
        seq = "A" * 2000
        result_seq, result_pos, _ = window_sequence(seq, 1000, window_half=500, max_len=1022)
        assert 1 <= result_pos <= len(result_seq)

    def test_position_near_start(self):
        seq = "A" * 2000
        seq = seq[:4] + "M" + seq[5:]  # position 5 (1-indexed)
        result_seq, result_pos, _ = window_sequence(seq, 5, window_half=500, max_len=1022)
        assert result_seq[result_pos - 1] == "M"
        assert len(result_seq) <= 1022

    def test_position_near_end(self):
        seq = "A" * 2000
        seq = seq[:1994] + "M" + seq[1995:]  # position 1995
        result_seq, result_pos, _ = window_sequence(seq, 1995, window_half=500, max_len=1022)
        assert result_seq[result_pos - 1] == "M"
        assert len(result_seq) <= 1022

    def test_windowed_length_does_not_exceed_max_len(self):
        seq = "A" * 5000
        for pos in [1, 100, 2500, 4900, 5000]:
            result_seq, _, _ = window_sequence(seq, pos, window_half=500, max_len=1022)
            assert len(result_seq) <= 1022, f"Window too long at pos={pos}"

    def test_start_offset_short_sequence_is_zero(self):
        seq = "A" * 100
        _, _, start = window_sequence(seq, 50, window_half=500, max_len=1022)
        assert start == 0

    def test_start_offset_slices_full_sequence_to_window(self):
        # The returned start must be the exact offset such that
        # full_seq[start:start+len(window)] == window. This is the contract
        # callers rely on to slice parallel per-residue arrays.
        seq = "".join("ACDEFGHIKLMNPQRSTVWY"[i % 20] for i in range(5000))
        for pos in [1, 100, 2500, 4900, 5000]:
            window, _, start = window_sequence(seq, pos, window_half=511, max_len=1022)
            assert seq[start : start + len(window)] == window, f"pos={pos}"
