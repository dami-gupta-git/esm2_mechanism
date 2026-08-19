"""Tests for validated pathogenicity geometry inputs."""

import json

import numpy as np
import pytest

from esm2_mech.experiments.geometry import data as geometry_data
from esm2_mech.utils.data import variants_fingerprint


def _write_inputs(tmp_path, monkeypatch, fingerprint=None):
    variants = [
        {
            "gene": "G1",
            "uniprot_id": "P1",
            "aa_pos": 1,
            "aa_wt": "A",
            "aa_mut": "G",
            "label": "benign",
        },
        {
            "gene": "G2",
            "uniprot_id": "P2",
            "aa_pos": 2,
            "aa_wt": "L",
            "aa_mut": "V",
            "label": "pathogenic",
        },
    ]
    variants_path = tmp_path / "variants.json"
    wt_path = tmp_path / "wt.npy"
    mut_path = tmp_path / "mut.npy"
    meta_path = tmp_path / "meta.json"
    variants_path.write_text(json.dumps(variants))
    np.save(wt_path, np.zeros((2, 3), dtype=np.float32))
    np.save(mut_path, np.ones((2, 3), dtype=np.float32))
    meta_path.write_text(
        json.dumps(
            {
                "fingerprint": (
                    variants_fingerprint(variants)
                    if fingerprint is None
                    else fingerprint
                ),
                "n_valid": 2,
                "model": geometry_data.ESM2_MODEL_650M,
            }
        )
    )
    monkeypatch.setattr(
        geometry_data, "PATHOGENICITY_CANONICAL_VARIANTS_JSON", variants_path
    )
    monkeypatch.setattr(geometry_data, "PATH_EMB_WT_MEAN", wt_path)
    monkeypatch.setattr(geometry_data, "PATH_EMB_MUT_MEAN", mut_path)
    monkeypatch.setattr(geometry_data, "PATH_EMB_META", meta_path)
    return variants


def test_validated_loader_returns_aligned_delta(tmp_path, monkeypatch):
    variants = _write_inputs(tmp_path, monkeypatch)
    inputs = geometry_data.load_pathogenicity_geometry_inputs()

    assert inputs.variants == variants
    assert np.array_equal(inputs.labels, [0, 1])
    assert np.array_equal(inputs.delta, np.ones((2, 3), dtype=np.float32))


def test_validated_loader_rejects_stale_embedding_metadata(tmp_path, monkeypatch):
    _write_inputs(tmp_path, monkeypatch, fingerprint="stale")

    with pytest.raises(ValueError, match="do not match the embedding metadata"):
        geometry_data.load_pathogenicity_geometry_inputs()


def test_mechanism_provenance_rejects_misaligned_rows():
    with pytest.raises(ValueError, match="not row-aligned"):
        geometry_data.mechanism_geometry_provenance(
            np.zeros((2, 3)),
            np.array(["GOF"]),
            np.array(["G1", "G2"]),
            {"G1": "F1", "G2": "F2"},
        )
