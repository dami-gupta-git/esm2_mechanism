"""
Tests for conservation cache validation in conservation_axis.

Invariants:
- A cache with matching variants, sequences, model, and amino-acid order is accepted.
- A cache with matching row count but no fingerprint is rejected (ValueError),
  not silently loaded — a row-count match alone cannot verify variant ordering.
- A sequence change invalidates a same-length cache.
- A windowing implementation change invalidates an otherwise matching cache.

These test the validation logic by calling extract_conservation with mocked
GPU imports, so the tests run on CPU without the ESM model.
"""

import json
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from esm2_mech.experiments.geometry import conservation_axis
from esm2_mech.utils.data import embedding_fingerprint


def _write_cache(tmp_path, n_rows, metadata=None):
    """Write a fake conservation cache (npy + meta json)."""
    npy_path = tmp_path / "conservation_pathogenicity.npy"
    meta_path = tmp_path / "conservation_pathogenicity_meta.json"

    arr = np.full((n_rows, 3), 1.0, dtype=np.float32)
    np.save(npy_path, arr)

    if metadata is None:
        meta = {"n": n_rows}
    else:
        meta = dict(metadata)
        meta.setdefault("coverage", n_rows)
        meta.setdefault("conservation_array_fingerprint", embedding_fingerprint(arr))
    with open(meta_path, "w") as fh:
        json.dump(meta, fh)

    return npy_path, meta_path


def _make_variants(n):
    return [
        {
            "gene": f"GENE{i}",
            "uniprot_id": f"P{i}",
            "aa_pos": 1,
            "aa_wt": "A",
            "aa_mut": "G",
            "label": "pathogenic" if i % 2 else "benign",
        }
        for i in range(n)
    ]


def _make_sequences(n):
    return {f"P{i}": "AAAA" for i in range(n)}


class TestLegacyCacheRejection:
    def test_no_fingerprint_raises(self, tmp_path, monkeypatch):
        """A legacy cache with correct row count but no fingerprint must be
        rejected, not silently loaded."""
        n = 50
        variants = _make_variants(n)
        npy_path, meta_path = _write_cache(tmp_path, n)

        monkeypatch.setattr(conservation_axis, "CONS_CACHE", npy_path)
        monkeypatch.setattr(conservation_axis, "CONS_META", meta_path)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        monkeypatch.setitem(sys.modules, "torch", mock_torch)

        mock_esm = MagicMock()
        monkeypatch.setitem(sys.modules, "esm", mock_esm)

        with pytest.raises(ValueError, match="no fingerprint"):
            conservation_axis.extract_conservation(variants, seqs=_make_sequences(n))

    def test_wrong_fingerprint_discards_and_continues(self, tmp_path, monkeypatch):
        """A cache with a non-matching fingerprint should be discarded (not loaded),
        and extraction continues (hitting the model load)."""
        n = 50
        variants = _make_variants(n)
        npy_path, meta_path = _write_cache(
            tmp_path, n, metadata={"n": n, "fingerprint": "wrong_fp"}
        )

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
            conservation_axis.extract_conservation(variants, seqs=_make_sequences(n))

    def test_matching_fingerprint_accepted(self, tmp_path, monkeypatch):
        """A cache with the correct fingerprint should be loaded and extraction
        should continue past the cache check."""
        n = 50
        variants = _make_variants(n)
        seqs = _make_sequences(n)
        metadata = conservation_axis.conservation_cache_identity(variants, seqs)
        npy_path, meta_path = _write_cache(tmp_path, n, metadata=metadata)

        monkeypatch.setattr(conservation_axis, "CONS_CACHE", npy_path)
        monkeypatch.setattr(conservation_axis, "CONS_META", meta_path)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        monkeypatch.setitem(sys.modules, "torch", mock_torch)

        sentinel = Exception(
            "model load reached — cache was accepted and extraction continues"
        )
        mock_esm = MagicMock()
        mock_esm.pretrained.load_model_and_alphabet.side_effect = sentinel
        monkeypatch.setitem(sys.modules, "esm", mock_esm)

        with pytest.raises(Exception, match="model load reached"):
            conservation_axis.extract_conservation(variants, seqs=seqs)


class TestFullCacheProvenance:
    def test_changed_windowing_rejects_cache(self, tmp_path, monkeypatch):
        variants = _make_variants(4)
        seqs = _make_sequences(4)
        metadata = conservation_axis.conservation_cache_identity(variants, seqs)
        npy_path, meta_path = _write_cache(tmp_path, 4, metadata=metadata)
        monkeypatch.setattr(conservation_axis, "CONS_CACHE", npy_path)
        monkeypatch.setattr(conservation_axis, "CONS_META", meta_path)

        def changed_window_sequence(sequence, aa_pos, window_half=500, max_len=1000):
            return sequence, aa_pos, 0

        monkeypatch.setattr(
            conservation_axis, "window_sequence", changed_window_sequence
        )

        with pytest.raises(ValueError, match="provenance does not match"):
            conservation_axis.load_validated_conservation_cache(variants, seqs)

    def test_changed_sequence_rejects_cache(self, tmp_path, monkeypatch):
        variants = _make_variants(4)
        seqs = _make_sequences(4)
        metadata = conservation_axis.conservation_cache_identity(variants, seqs)
        npy_path, meta_path = _write_cache(tmp_path, 4, metadata=metadata)
        monkeypatch.setattr(conservation_axis, "CONS_CACHE", npy_path)
        monkeypatch.setattr(conservation_axis, "CONS_META", meta_path)

        changed = dict(seqs)
        changed["P2"] = "AGAA"
        with pytest.raises(ValueError, match="provenance does not match"):
            conservation_axis.load_validated_conservation_cache(variants, changed)

    def test_matching_full_provenance_loads(self, tmp_path, monkeypatch):
        variants = _make_variants(4)
        seqs = _make_sequences(4)
        metadata = conservation_axis.conservation_cache_identity(variants, seqs)
        npy_path, meta_path = _write_cache(tmp_path, 4, metadata=metadata)
        monkeypatch.setattr(conservation_axis, "CONS_CACHE", npy_path)
        monkeypatch.setattr(conservation_axis, "CONS_META", meta_path)

        values, loaded_metadata = conservation_axis.load_validated_conservation_cache(
            variants, seqs
        )
        assert values.shape == (4, 3)
        assert loaded_metadata["fingerprint"] == metadata["fingerprint"]

    def test_changed_array_content_rejects_cache(self, tmp_path, monkeypatch):
        variants = _make_variants(4)
        seqs = _make_sequences(4)
        metadata = conservation_axis.conservation_cache_identity(variants, seqs)
        npy_path, meta_path = _write_cache(tmp_path, 4, metadata=metadata)
        changed = np.load(npy_path)
        changed[0, 0] = 2.0
        np.save(npy_path, changed)
        monkeypatch.setattr(conservation_axis, "CONS_CACHE", npy_path)
        monkeypatch.setattr(conservation_axis, "CONS_META", meta_path)

        with pytest.raises(ValueError, match="content fingerprint"):
            conservation_axis.load_validated_conservation_cache(variants, seqs)

    def test_coverage_mismatch_rejects_cache(self, tmp_path, monkeypatch):
        variants = _make_variants(4)
        seqs = _make_sequences(4)
        metadata = conservation_axis.conservation_cache_identity(variants, seqs)
        metadata["coverage"] = 3
        npy_path, meta_path = _write_cache(tmp_path, 4, metadata=metadata)
        monkeypatch.setattr(conservation_axis, "CONS_CACHE", npy_path)
        monkeypatch.setattr(conservation_axis, "CONS_META", meta_path)

        with pytest.raises(ValueError, match="records coverage"):
            conservation_axis.load_validated_conservation_cache(variants, seqs)
