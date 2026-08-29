"""Tests for the exploratory magnitude/direction decomposition."""

import numpy as np
import pytest

from esm2_mech.experiments.geometry import magnitude_direction


def test_decompose_preserves_full_magnitude_and_direction():
    delta = np.array([[3.0, 4.0], [0.0, -2.0]], dtype=np.float32)
    result = magnitude_direction.decompose(delta)

    assert np.array_equal(result["full"], delta)
    assert np.allclose(result["mag"].ravel(), [5.0, 2.0])
    assert np.allclose(np.linalg.norm(result["dir"], axis=1), [1.0, 1.0])
    assert np.allclose(result["dir"] * result["mag"], delta)


def test_family_split_scored_rows_exclude_unannotated_genes():
    genes = np.array(["A", "B", "C", "A"])
    pfam_map = {"A": "PF1", "B": None}

    assert magnitude_direction.scored_rows("gene_split", genes, pfam_map).tolist() == [
        0,
        1,
        2,
        3,
    ]
    assert magnitude_direction.scored_rows(
        "family_split", genes, pfam_map
    ).tolist() == [0, 3]


def test_selected_stability_dataset_does_not_silently_skip(monkeypatch):
    def missing_inputs():
        raise FileNotFoundError("missing stability fingerprint")

    monkeypatch.setattr(magnitude_direction, "load_stability_inputs", missing_inputs)
    with pytest.raises(FileNotFoundError, match="missing stability fingerprint"):
        magnitude_direction.run_biophysical_direction(
            seeds=[0], stability_dataset="tsuboyama"
        )


def test_none_is_the_only_explicit_stability_skip():
    assert (
        magnitude_direction.run_biophysical_direction(
            seeds=[0], stability_dataset="none"
        )
        is None
    )
    with pytest.raises(ValueError, match="unknown stability dataset"):
        magnitude_direction.run_biophysical_direction(
            seeds=[0], stability_dataset="unknown"
        )
