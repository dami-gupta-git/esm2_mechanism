"""Atomic file I/O utilities shared across the project."""

import functools
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)


def load_variants_and_delta(
    variants_path,
    wt_mean_path,
    mut_mean_path,
    wt_pos_path=None,
    mut_pos_path=None,
    verbose: bool = True,
):
    """Load the mechanism variant list and its row-aligned ESM-2 delta embeddings.

    Reads the variant JSON at ``variants_path`` (the list the embeddings were
    extracted from, in the same identity and order as the .npy rows), extracts the
    GOF/DN/LOF labels and gene symbols, and computes the mean-pooled delta
    (mut - wt). When ``wt_pos_path``/``mut_pos_path`` are given it also computes the
    per-residue delta_pos; otherwise that element of the returned tuple is None.

    Embedding rows must match the variant count exactly. A mismatch raises
    ValueError rather than silently slicing to the shorter length — a row-count
    mismatch means the .npy is not aligned to the variant list and any downstream
    label assignment would be wrong.

    Returns ``(variants, labels, genes, delta_mean, delta_pos)`` where labels and
    genes are string ndarrays, delta_mean is float32 (n, d), and delta_pos is
    float32 (n, d) or None.
    """
    with open(variants_path) as f:
        variants = json.load(f)

    labels = np.array([v["label_3class"] for v in variants])
    genes = np.array([v["gene"] for v in variants])
    num_variants = len(variants)

    wt_mean = np.load(wt_mean_path)
    mut_mean = np.load(mut_mean_path)
    if wt_mean.shape[0] != num_variants or mut_mean.shape[0] != num_variants:
        raise ValueError(
            f"embedding/variant row mismatch: wt_mean {wt_mean.shape[0]}, "
            f"mut_mean {mut_mean.shape[0]} vs {num_variants} variants — "
            f"{variants_path} is not row-aligned with the embeddings."
        )
    delta_mean = (mut_mean - wt_mean).astype(np.float32)

    delta_pos = None
    if wt_pos_path is not None and mut_pos_path is not None:
        wt_pos = np.load(wt_pos_path)
        mut_pos = np.load(mut_pos_path)
        if wt_pos.shape[0] != num_variants or mut_pos.shape[0] != num_variants:
            raise ValueError(
                f"position-embedding/variant row mismatch: wt_pos {wt_pos.shape[0]}, "
                f"mut_pos {mut_pos.shape[0]} vs {num_variants} variants — "
                f"{variants_path} is not row-aligned with the embeddings."
            )
        delta_pos = (mut_pos - wt_pos).astype(np.float32)

    if verbose:
        print(f"Loaded {num_variants} variants, {len(set(genes.tolist()))} genes")
        print(f"Class distribution: {dict(Counter(labels.tolist()))}")
        print(f"Delta shape: {delta_mean.shape}")

    return variants, labels, genes, delta_mean, delta_pos


def save_npy(path, arr: np.ndarray) -> None:
    """Atomically write a numpy array: write to .tmp, fsync, then rename."""
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, str(path))


def atomic_write_json(path, data, **json_kwargs) -> None:
    """Atomically write JSON: write to .tmp, fsync, then rename.

    Extra keyword args (e.g. indent, sort_keys) are forwarded to json.dump.
    """
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, **json_kwargs)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, str(path))


def load_json_or_discard(path):
    """Load JSON from path, returning None if it is missing or corrupt.

    A corrupt cache file (partial write on interrupt) is deleted so the caller's
    next run re-fetches rather than crashing on the bad file. Returns the parsed
    object on success, or None when the file is absent or could not be parsed.
    Shared by the fetch_data cache loaders, which each inlined this try/except +
    unlink pattern. For a required input that must fail loud on corruption, do NOT
    use this — load directly so a JSONDecodeError surfaces.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        # A truncated/partial write can leave invalid UTF-8 bytes, which raise
        # UnicodeDecodeError during the read (a ValueError subclass, but NOT a
        # JSONDecodeError) before parsing. Treat both as corruption: discard and
        # report a miss so the caller re-fetches. Other errors (e.g. permission)
        # still propagate so good data is never silently discarded.
        print(f"WARNING: corrupt JSON cache at {path} — discarding", flush=True)
        path.unlink()
        return None


def atomic_write_text(path, text: str) -> None:
    """Atomically write a text string: write to .tmp, fsync, then rename."""
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, str(path))
