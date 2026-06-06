"""Atomic file I/O utilities shared across the project."""

import json
import os

import numpy as np


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
