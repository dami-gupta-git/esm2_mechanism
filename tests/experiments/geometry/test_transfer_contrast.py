"""Tests for full-delta group transfer result semantics.

Invariants:
- transfer_test: reports partition and fold counts within one seed, each spread
  naming its own sampling unit
- the across-seed stage reduces one within-seed estimate per requested seed, not
  one per partition or fold
"""

import numpy as np

from esm2_mech.experiments.geometry.transfer_contrast import transfer_test
from esm2_mech.utils.seed_aggregation import (
    SEED_SAMPLING_UNIT,
    aggregate_seed_values,
    make_seed_record,
)


def _synthetic_task(n_groups=20, rows_per_group=10, seed=7):
    rng = np.random.RandomState(seed)
    groups = np.repeat([f"F{i}" for i in range(n_groups)], rows_per_group)
    delta = rng.normal(size=(len(groups), 6))
    labels = np.tile([0, 1] * (rows_per_group // 2), n_groups)
    return delta, labels, groups


def test_within_seed_spreads_name_their_own_sampling_unit():
    delta, labels, groups = _synthetic_task()

    result = transfer_test(
        delta, labels, groups, kind="linear", n_partitions=2, seed=0, min_pos=2
    )

    assert result["group_cv_auroc"]["sampling_unit"] == "held_out_fold"
    assert "fold_std" in result["group_cv_auroc"]
    assert (
        result["transfer_auroc"]["sampling_unit"] == "random_family_partition_direction"
    )
    assert "partition_direction_std" in result["transfer_auroc"]
    # Neither within-seed spread may claim to be a spread across seeds.
    assert SEED_SAMPLING_UNIT not in (
        result["group_cv_auroc"]["sampling_unit"],
        result["transfer_auroc"]["sampling_unit"],
    )


def test_across_seed_stage_takes_one_estimate_per_requested_seed():
    delta, labels, groups = _synthetic_task()
    requested_seeds = (0, 1, 2)

    per_seed = {
        seed: transfer_test(
            delta, labels, groups, kind="linear", n_partitions=2, seed=seed, min_pos=2
        )
        for seed in requested_seeds
    }
    aggregate = aggregate_seed_values(
        requested_seeds,
        [
            make_seed_record(seed, per_seed[seed]["group_cv_auroc"]["mean"])
            for seed in requested_seeds
        ],
    )

    # One contribution per seed, not one per fold: the within-seed fold count is
    # larger than the seed count, so a flattened reduction would show up here.
    assert aggregate.available
    assert list(aggregate.contributing_seeds) == list(requested_seeds)
    assert aggregate.sampling_unit == SEED_SAMPLING_UNIT
    assert per_seed[0]["group_cv_auroc"]["n"] > len(requested_seeds)


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
