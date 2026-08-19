"""Tests for full-delta group transfer result semantics."""

import numpy as np

from esm2_mech.experiments.geometry.transfer_contrast import transfer_test


def test_transfer_uses_explicit_group_cv_name_and_counts():
    rng = np.random.RandomState(7)
    n_groups = 20
    rows_per_group = 10
    groups = np.repeat([f"F{i}" for i in range(n_groups)], rows_per_group)
    delta = rng.normal(size=(len(groups), 6))
    labels = np.tile([0, 1] * (rows_per_group // 2), n_groups)

    result = transfer_test(
        delta,
        labels,
        groups,
        kind="linear",
        n_partitions=2,
        seed=0,
        min_pos=2,
    )

    assert "pooled_auroc" not in result
    assert result["group_cv_auroc"]["n"] == 5
    assert result["transfer_auroc"]["n"] == 4
