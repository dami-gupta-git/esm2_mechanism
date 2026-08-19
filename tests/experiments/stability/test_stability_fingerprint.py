"""Tests for strict Tsuboyama embedding provenance."""

import json

import pytest

from esm2_mech.experiments.stability import stability_data


VARIANTS = [
    {"protein": "1ABC.pdb", "mutation_code": "A1G"},
    {"protein": "2DEF.pdb", "mutation_code": "L2V"},
]


def test_missing_embedding_fingerprint_is_an_error(tmp_path, monkeypatch):
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(stability_data, "MEGASCALE_EMB_FINGERPRINT", missing)

    with pytest.raises(FileNotFoundError, match="no embedding fingerprint"):
        stability_data._check_fingerprint(VARIANTS)


def test_fingerprint_count_must_match(tmp_path, monkeypatch):
    path = tmp_path / "fingerprint.json"
    path.write_text(
        json.dumps(
            {
                "sha256": stability_data.variant_fingerprint(VARIANTS),
                "n_variants": 99,
            }
        )
    )
    monkeypatch.setattr(stability_data, "MEGASCALE_EMB_FINGERPRINT", path)

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        stability_data._check_fingerprint(VARIANTS)
