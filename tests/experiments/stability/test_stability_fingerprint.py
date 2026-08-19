"""Tests for strict Tsuboyama embedding provenance."""

import json

import numpy as np
import pytest

from esm2_mech.experiments.stability import stability_data


VARIANTS = [
    {
        "protein": "1ABC.pdb",
        "mutation_code": "A1G",
        "wt_seq": "AA",
        "mut_seq": "GA",
        "var_pos": 1,
    },
    {
        "protein": "2DEF.pdb",
        "mutation_code": "L2V",
        "wt_seq": "LL",
        "mut_seq": "LV",
        "var_pos": 2,
    },
]


def _metadata(n_variants=2):
    return {
        "metadata_version": stability_data.STABILITY_EMBEDDING_METADATA_VERSION,
        "sha256": stability_data.variant_fingerprint(VARIANTS),
        "embedding_input_fingerprint": stability_data.embedding_input_fingerprint(
            VARIANTS
        ),
        "n_variants": n_variants,
        "model": stability_data.ESM2_MODEL,
    }


def test_missing_embedding_fingerprint_is_an_error(tmp_path, monkeypatch):
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(stability_data, "MEGASCALE_EMB_FINGERPRINT", missing)

    with pytest.raises(FileNotFoundError, match="no embedding fingerprint"):
        stability_data._check_fingerprint(VARIANTS)


def test_fingerprint_count_must_match(tmp_path, monkeypatch):
    path = tmp_path / "fingerprint.json"
    path.write_text(json.dumps(_metadata(n_variants=99)))
    monkeypatch.setattr(stability_data, "MEGASCALE_EMB_FINGERPRINT", path)

    with pytest.raises(ValueError, match="does not match"):
        stability_data._check_fingerprint(VARIANTS)


def test_embedding_input_fingerprint_must_match(tmp_path, monkeypatch):
    path = tmp_path / "fingerprint.json"
    metadata = _metadata()
    metadata["embedding_input_fingerprint"] = "stale"
    path.write_text(json.dumps(metadata))
    monkeypatch.setattr(stability_data, "MEGASCALE_EMB_FINGERPRINT", path)

    with pytest.raises(ValueError, match="does not match"):
        stability_data._check_fingerprint(VARIANTS)


def test_mean_embedding_content_must_match():
    wt = np.zeros((2, 3), dtype=np.float32)
    mut = np.ones((2, 3), dtype=np.float32)
    metadata = {"mean_embedding_fingerprint": "stale"}

    with pytest.raises(ValueError, match="content fingerprint"):
        stability_data._check_embedding_content(metadata, wt, mut)
