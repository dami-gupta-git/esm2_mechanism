"""
Tests for conservation cache validation in conservation_axis.

Invariants:
- A cache with matching row count AND fingerprint is accepted.
- A cache with matching row count but no fingerprint is rejected (ValueError),
  not silently loaded — a row-count match alone cannot verify variant ordering.
- A cache with mismatched row count is discarded regardless of fingerprint.

These test the validation logic by calling extract_conservation with mocked
GPU imports, so the tests run on CPU without the ESM model.
"""

import json
import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest

from esm2_mech.experiments.geometry import conservation_axis


def _write_cache(tmp_path, n_rows, fingerprint=None):
    """Write a fake conservation cache (npy + meta json)."""
    npy_path = tmp_path / "conservation_pathogenicity.npy"
    meta_path = tmp_path / "conservation_pathogenicity_meta.json"

    arr = np.full((n_rows, 3), 1.0, dtype=np.float32)
    np.save(npy_path, arr)

    meta = {"n": n_rows}
    if fingerprint is not None:
        meta["fingerprint"] = fingerprint
    with open(meta_path, "w") as fh:
        json.dump(meta, fh)

    return npy_path, meta_path


def _make_variants(n):
    return [
        {"gene": f"GENE{i}", "aa_pos": i + 1, "aa_wt": "A", "aa_mut": "G"}
        for i in range(n)
    ]


class TestLegacyCacheRejection:

    def test_no_fingerprint_raises(self, tmp_path, monkeypatch):
        """A legacy cache with correct row count but no fingerprint must be
        rejected, not silently loaded."""
        n = 50
        variants = _make_variants(n)
        npy_path, meta_path = _write_cache(tmp_path, n, fingerprint=None)

        monkeypatch.setattr(conservation_axis, "CONS_CACHE", npy_path)
        monkeypatch.setattr(conservation_axis, "CONS_META", meta_path)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        monkeypatch.setitem(sys.modules, "torch", mock_torch)

        mock_esm = MagicMock()
        monkeypatch.setitem(sys.modules, "esm", mock_esm)

        with pytest.raises(ValueError, match="no fingerprint"):
            conservation_axis.extract_conservation(variants, seqs={})

    def test_wrong_fingerprint_discards_and_continues(self, tmp_path, monkeypatch):
        """A cache with a non-matching fingerprint should be discarded (not loaded),
        and extraction continues (hitting the model load)."""
        n = 50
        variants = _make_variants(n)
        npy_path, meta_path = _write_cache(tmp_path, n, fingerprint="wrong_fp")

        monkeypatch.setattr(conservation_axis, "CONS_CACHE", npy_path)
        monkeypatch.setattr(conservation_axis, "CONS_META", meta_path)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        monkeypatch.setitem(sys.modules, "torch", mock_torch)

        sentinel = Exception("model load reached — cache was correctly discarded")
        mock_esm = MagicMock()
        mock_esm.pretrained.load_model_and_alphabet.side_effect = sentinel
        monkeypatch.setitem(sys.modules, "esm", mock_esm)

        with pytest.raises(Exception, match="model load reached"):
            conservation_axis.extract_conservation(variants, seqs={})

    def test_matching_fingerprint_accepted(self, tmp_path, monkeypatch):
        """A cache with the correct fingerprint should be loaded and extraction
        should continue past the cache check."""
        n = 50
        variants = _make_variants(n)
        fp = conservation_axis._variant_fingerprint(variants)
        npy_path, meta_path = _write_cache(tmp_path, n, fingerprint=fp)

        monkeypatch.setattr(conservation_axis, "CONS_CACHE", npy_path)
        monkeypatch.setattr(conservation_axis, "CONS_META", meta_path)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        monkeypatch.setitem(sys.modules, "torch", mock_torch)

        sentinel = Exception("model load reached — cache was accepted and extraction continues")
        mock_esm = MagicMock()
        mock_esm.pretrained.load_model_and_alphabet.side_effect = sentinel
        monkeypatch.setitem(sys.modules, "esm", mock_esm)

        with pytest.raises(Exception, match="model load reached"):
            conservation_axis.extract_conservation(variants, seqs={})
