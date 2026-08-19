"""Tests for family-held-out pathogenicity-axis summaries."""

import numpy as np
import pytest

from esm2_mech.experiments.geometry.axis_analysis import (
    family_held_out_axis_analysis,
)


def test_axis_associations_are_scored_across_held_out_family_folds():
    rng = np.random.RandomState(11)
    n_families = 10
    rows_per_family = 10
    genes = np.repeat([f"G{i}" for i in range(n_families)], rows_per_family)
    pfam = {f"G{i}": f"F{i}" for i in range(n_families)}
    labels = np.tile([0, 1] * (rows_per_family // 2), n_families)
    signal = (2 * labels - 1) + rng.normal(scale=0.2, size=len(labels))
    delta = np.column_stack([signal, rng.normal(size=(len(labels), 3))])
    biochem = np.column_stack([signal, rng.normal(size=len(labels))])

    result = family_held_out_axis_analysis(
        delta,
        labels,
        genes,
        pfam,
        {"known_signal": signal},
        regression_features=biochem,
        seeds=[0, 1],
    )

    correlation = result["correlations"]["known_signal"]
    assert correlation["n"] == 10
    assert correlation["mean"] > 0.8
    assert result["regression_r2"]["n"] == 10


def test_axis_analysis_rejects_misaligned_features():
    with pytest.raises(ValueError, match="expected 4"):
        family_held_out_axis_analysis(
            np.zeros((4, 2)),
            np.array([0, 1, 0, 1]),
            np.array(["G1", "G1", "G2", "G2"]),
            {"G1": "F1", "G2": "F2"},
            {"bad": np.zeros(3)},
            seeds=[0],
        )
