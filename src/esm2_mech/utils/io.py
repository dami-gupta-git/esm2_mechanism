"""Atomic file I/O utilities shared across the project."""

import functools
import json
import os
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)


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
    except json.JSONDecodeError:
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
