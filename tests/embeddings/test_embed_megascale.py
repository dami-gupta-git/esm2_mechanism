"""
Tests for the pure helpers in the Megascale embedding step.

The embedding itself is GPU-bound, but pair construction and checkpoint
promotion are plain functions guarding row alignment and sequence length.

Covers:
- _build_pairs: returns wild-type and mutant sequences plus positions, in input order
- _build_pairs: the three returned lists stay row-aligned with the input
- _build_pairs: a domain longer than the model's token limit raises, naming the domain
- _build_pairs: a domain exactly at the token limit is accepted
- _build_pairs: an empty variant list returns three empty lists
- _promote_checkpoint: copies every checkpoint array to its target filename
- _promote_checkpoint: promoted arrays keep their contents
- _promote_checkpoint: a checkpoint with too few rows raises instead of promoting
- _promote_checkpoint: a short array is detected even when it is not the first one
- _promote_checkpoint: nothing is written when a row count is wrong
"""

import numpy as np
import pytest

from esm2_mech.embeddings.embed_megascale import (
    _CKPT_TO_TARGET_FILENAME,
    _build_pairs,
    _promote_checkpoint,
)
from esm2_mech.utils.constants import MAX_SEQ_LEN


def _variant(protein, wt_seq, mut_seq=None, var_pos=1):
    return {
        "protein": protein,
        "wt_seq": wt_seq,
        "mut_seq": mut_seq if mut_seq is not None else wt_seq.replace("A", "V", 1),
        "var_pos": var_pos,
    }


# ---------------------------------------------------------------------------
# _build_pairs
# ---------------------------------------------------------------------------


def test_build_pairs_returns_sequences_and_positions_in_input_order():
    variants = [
        _variant("D1", "AAAA", "VAAA", var_pos=1),
        _variant("D2", "CCCC", "CCDC", var_pos=3),
    ]
    wt_seqs, mut_seqs, positions = _build_pairs(variants)
    assert wt_seqs == ["AAAA", "CCCC"]
    assert mut_seqs == ["VAAA", "CCDC"]
    assert positions == [1, 3]


def test_build_pairs_keeps_the_three_lists_row_aligned():
    variants = [_variant(f"D{i}", "AAAA", var_pos=i + 1) for i in range(5)]
    wt_seqs, mut_seqs, positions = _build_pairs(variants)
    assert len(wt_seqs) == len(mut_seqs) == len(positions) == len(variants)


def test_build_pairs_rejects_a_domain_over_the_token_limit():
    """Windowing is not implemented here, so an over-long domain must not be
    truncated."""
    variants = [
        _variant("SHORT", "AAAA"),
        _variant("TOO_LONG", "A" * (MAX_SEQ_LEN + 1)),
    ]
    with pytest.raises(ValueError, match="TOO_LONG"):
        _build_pairs(variants)


def test_build_pairs_accepts_a_domain_at_the_token_limit():
    variants = [_variant("EXACT", "A" * MAX_SEQ_LEN)]
    wt_seqs, _, _ = _build_pairs(variants)
    assert len(wt_seqs[0]) == MAX_SEQ_LEN


def test_build_pairs_on_no_variants_returns_empty_lists():
    assert _build_pairs([]) == ([], [], [])


# ---------------------------------------------------------------------------
# _promote_checkpoint
# ---------------------------------------------------------------------------


def _write_checkpoint(ckpt_dir, n_rows, short_name=None):
    """One array per checkpoint name; `short_name` gets one row fewer."""
    for index, name in enumerate(_CKPT_TO_TARGET_FILENAME):
        rows = n_rows - 1 if name == short_name else n_rows
        array = np.full((rows, 2), float(index))
        np.save(ckpt_dir / name, array, allow_pickle=False)


def test_promote_checkpoint_writes_every_target_array(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    target_dir = tmp_path / "target"
    ckpt_dir.mkdir()
    target_dir.mkdir()
    _write_checkpoint(ckpt_dir, 3)

    _promote_checkpoint(str(ckpt_dir), 3, str(target_dir))

    for target_filename in _CKPT_TO_TARGET_FILENAME.values():
        assert (target_dir / target_filename).exists()


def test_promote_checkpoint_preserves_array_contents(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    target_dir = tmp_path / "target"
    ckpt_dir.mkdir()
    target_dir.mkdir()
    _write_checkpoint(ckpt_dir, 3)

    _promote_checkpoint(str(ckpt_dir), 3, str(target_dir))

    for name, target_filename in _CKPT_TO_TARGET_FILENAME.items():
        source = np.load(ckpt_dir / name)
        promoted = np.load(target_dir / target_filename)
        assert np.array_equal(source, promoted)


def test_promote_checkpoint_rejects_an_incomplete_array(tmp_path):
    """A short array means extraction stopped early; promoting it would ship a
    silently truncated embedding matrix."""
    ckpt_dir = tmp_path / "ckpt"
    target_dir = tmp_path / "target"
    ckpt_dir.mkdir()
    target_dir.mkdir()
    first_name = next(iter(_CKPT_TO_TARGET_FILENAME))
    _write_checkpoint(ckpt_dir, 3, short_name=first_name)

    with pytest.raises(ValueError, match="extraction incomplete"):
        _promote_checkpoint(str(ckpt_dir), 3, str(target_dir))


def test_promote_checkpoint_detects_a_short_array_that_is_not_the_first(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    target_dir = tmp_path / "target"
    ckpt_dir.mkdir()
    target_dir.mkdir()
    last_name = list(_CKPT_TO_TARGET_FILENAME)[-1]
    _write_checkpoint(ckpt_dir, 3, short_name=last_name)

    with pytest.raises(ValueError, match="extraction incomplete"):
        _promote_checkpoint(str(ckpt_dir), 3, str(target_dir))


def test_promote_checkpoint_writes_nothing_when_the_first_array_is_short(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    target_dir = tmp_path / "target"
    ckpt_dir.mkdir()
    target_dir.mkdir()
    first_name = next(iter(_CKPT_TO_TARGET_FILENAME))
    _write_checkpoint(ckpt_dir, 3, short_name=first_name)

    with pytest.raises(ValueError):
        _promote_checkpoint(str(ckpt_dir), 3, str(target_dir))
    assert list(target_dir.iterdir()) == []
