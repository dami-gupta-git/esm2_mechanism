"""Atomic file I/O utilities shared across the project."""

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
