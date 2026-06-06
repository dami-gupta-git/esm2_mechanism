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


def build_windowed_pair(
    sequence: str, aa_pos: int, aa_wt: str, aa_mut: str
) -> tuple[str, str, int] | None:
    """Window `sequence` around aa_pos and apply the missense substitution.

    Returns (wt_window, mut_window, new_pos) where new_pos is the 1-indexed
    mutation position within the window, or None if the WT residue does not match
    the reference (apply_missense returns None) — i.e. the variant must be dropped.
    """
    wt_win, new_pos, _ = window_sequence(sequence, aa_pos)
    mut_win = apply_missense(wt_win, new_pos, aa_wt, aa_mut)
    if mut_win is None:
        return None
    return wt_win, mut_win, new_pos


def window_sequence(
    sequence: str,
    aa_pos: int,
    window_half: int = WINDOW_HALF,
    max_len: int = MAX_SEQ_LEN,
) -> tuple[str, int, int]:
    """Extract a window of at most max_len residues centred on aa_pos.

    Returns (windowed_seq, new_aa_pos, start) where new_aa_pos is 1-indexed in
    the windowed sequence and start is the 0-indexed offset of the window in the
    full sequence (0 when the sequence is returned unchanged). Callers that need
    to slice a parallel per-residue array (e.g. structure coordinates) to the
    same window MUST use this start rather than re-deriving it.
    """
    if len(sequence) <= max_len:
        return sequence, aa_pos, 0

    idx = aa_pos - 1  # 0-indexed
    start = max(0, idx - window_half)
    end = min(len(sequence), idx + window_half)
    if end - start > max_len:
        half = max_len // 2
        start = max(0, idx - half)
        end = min(len(sequence), start + max_len)

    windowed = sequence[start:end]
    new_pos = idx - start + 1  # back to 1-indexed
    return windowed, new_pos, start
