"""Pure sequence utilities: missense application and position windowing."""

from __future__ import annotations

from esm2_mech.utils.constants import MAX_SEQ_LEN, WINDOW_HALF


def apply_missense(sequence: str, aa_pos: int, aa_wt: str, aa_mut: str) -> str | None:
    """Apply a missense mutation (1-indexed aa_pos). Returns None on mismatch or OOB."""
    idx = aa_pos - 1
    if idx < 0 or idx >= len(sequence):
        return None
    if sequence[idx] != aa_wt:
        return None
    seq_list = list(sequence)
    seq_list[idx] = aa_mut
    return "".join(seq_list)


def window_sequence(
    sequence: str,
    aa_pos: int,
    window_half: int = WINDOW_HALF,
    max_len: int = MAX_SEQ_LEN,
) -> tuple[str, int]:
    """Extract a window of at most max_len residues centred on aa_pos.

    Returns (windowed_seq, new_aa_pos) where new_aa_pos is 1-indexed in the
    windowed sequence. Sequences already within max_len are returned unchanged.
    """
    if len(sequence) <= max_len:
        return sequence, aa_pos

    idx = aa_pos - 1  # 0-indexed
    start = max(0, idx - window_half)
    end = min(len(sequence), idx + window_half)
    if end - start > max_len:
        half = max_len // 2
        start = max(0, idx - half)
        end = min(len(sequence), start + max_len)

    windowed = sequence[start:end]
    new_pos = idx - start + 1  # back to 1-indexed
    return windowed, new_pos
