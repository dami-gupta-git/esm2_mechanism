"""
Tests for esm2_mech.experiments.mechanism.classify_by_mechanism.load_data.

Invariants:
- valid_variants is read straight from VALID_VARIANTS_JSON (no re-derived filter)
- when every embedding array's row count matches len(valid_variants), load_data
  returns arrays/lists all aligned to that length
- a row-count mismatch on ANY of the four embedding arrays raises ValueError
  naming that array, rather than silently indexing past a misaligned pairing
"""

import json

import numpy as np
import pytest

from esm2_mech.experiments.mechanism import classify_by_mechanism
from esm2_mech.utils.constants import GOF, LOF


def _write_variants(path, n):
    variants = [
        {
            "uniprot_id": f"P{i:05d}",
            "aa_pos": i + 1,
            "aa_wt": "A",
            "aa_mut": "G",
            "gene": f"GENE{i}",
            "label_3class": GOF if i % 2 == 0 else LOF,
            "mechanism": "GOF" if i % 2 == 0 else "LOF",
            "foldx_ddg": None,
        }
        for i in range(n)
    ]
    with open(path, "w") as f:
        json.dump(variants, f)
    return variants


def _write_embeddings(path, n, dim=4):
    np.save(path, np.zeros((n, dim), dtype=np.float32))


@pytest.fixture
def patched_paths(tmp_path, monkeypatch):
    """Point classify_by_mechanism's path constants at a scratch directory."""
    valid_variants_json = tmp_path / "valid_variants.json"
    emb_paths = {
        name: tmp_path / f"{name}.npy"
        for name in ("EMB_WT_MEAN", "EMB_MUT_MEAN", "EMB_WT_POS", "EMB_MUT_POS")
    }
    monkeypatch.setattr(classify_by_mechanism, "VALID_VARIANTS_JSON", valid_variants_json)
    for name, path in emb_paths.items():
        monkeypatch.setattr(classify_by_mechanism, name, path)
    monkeypatch.setattr(
        classify_by_mechanism, "_load_alphamissense_scores",
        lambda variants: np.full(len(variants), np.nan),
    )
    return valid_variants_json, emb_paths


class TestLoadData:

    def test_aligned_embeddings_loads_successfully(self, patched_paths):
        valid_variants_json, emb_paths = patched_paths
        n = 6
        variants = _write_variants(valid_variants_json, n)
        for path in emb_paths.values():
            _write_embeddings(path, n)

        data = classify_by_mechanism.load_data()

        assert len(data["valid_variants"]) == n
        assert data["emb_wt_mean"].shape == (n, 4)
        assert data["emb_mut_mean"].shape == (n, 4)
        assert data["emb_wt_pos"].shape == (n, 4)
        assert data["emb_mut_pos"].shape == (n, 4)
        assert data["labels_3class"].shape == (n,)
        assert list(data["labels_3class"]) == [v["label_3class"] for v in variants]
        assert list(data["genes_arr"]) == [v["gene"] for v in variants]

    @pytest.mark.parametrize(
        "mismatched_name", ["EMB_WT_MEAN", "EMB_MUT_MEAN", "EMB_WT_POS", "EMB_MUT_POS"]
    )
    def test_row_mismatch_raises(self, patched_paths, mismatched_name):
        valid_variants_json, emb_paths = patched_paths
        n = 6
        _write_variants(valid_variants_json, n)
        for name, path in emb_paths.items():
            # One array gets fewer rows than the variant list -> misaligned.
            rows = n - 1 if name == mismatched_name else n
            _write_embeddings(path, rows)

        with pytest.raises(ValueError, match=mismatched_name):
            classify_by_mechanism.load_data()
