"""
Tests for combining ESM-1v checkpoint scores into an ensemble.

A variant scored by only one checkpoint was previously written as the mean of
that single value, indistinguishable in the output file from a genuine
two-model ensemble score. It is now reported as having no ensemble score.

Covers:
- average_across_checkpoints: a variant scored by every checkpoint is averaged
- average_across_checkpoints: a variant missing from one checkpoint has no score
- average_across_checkpoints: a variant scored NaN by one checkpoint has no score
- average_across_checkpoints: partially scored variants are counted
- average_across_checkpoints: a variant scored by none is not counted as partial
- average_across_checkpoints: every variant seen by any checkpoint appears
- average_across_checkpoints: no variants yields an empty result
"""

import numpy as np

from esm2_mech.embeddings.score_esm1v import CHECKPOINTS, average_across_checkpoints


def _per_ckpt(*score_maps):
    """One score map per checkpoint, in CHECKPOINTS order."""
    assert len(score_maps) == len(CHECKPOINTS)
    return dict(zip(CHECKPOINTS, score_maps))


def test_a_variant_scored_by_every_checkpoint_is_averaged():
    averaged, _ = average_across_checkpoints(_per_ckpt({"v1": 1.0}, {"v1": 3.0}))
    assert averaged["v1"] == 2.0


def test_a_variant_missing_from_one_checkpoint_has_no_ensemble_score():
    """Returning 1.0 here would look exactly like a real two-model average."""
    averaged, _ = average_across_checkpoints(_per_ckpt({"v1": 1.0}, {}))
    assert np.isnan(averaged["v1"])


def test_a_variant_scored_nan_by_one_checkpoint_has_no_ensemble_score():
    averaged, _ = average_across_checkpoints(
        _per_ckpt({"v1": 1.0}, {"v1": float("nan")})
    )
    assert np.isnan(averaged["v1"])


def test_partially_scored_variants_are_counted():
    _averaged, n_incomplete = average_across_checkpoints(
        _per_ckpt({"v1": 1.0, "v2": 2.0}, {"v2": 4.0})
    )
    assert n_incomplete == 1


def test_a_variant_scored_by_no_checkpoint_is_not_counted_as_partial():
    """Nothing was lost to a partial ensemble; it simply was never scored."""
    _averaged, n_incomplete = average_across_checkpoints(
        _per_ckpt({"v1": float("nan")}, {"v1": float("nan")})
    )
    assert n_incomplete == 0


def test_every_variant_any_checkpoint_saw_appears_in_the_result():
    averaged, _ = average_across_checkpoints(_per_ckpt({"v1": 1.0}, {"v2": 2.0}))
    assert set(averaged) == {"v1", "v2"}


def test_no_variants_yields_an_empty_result():
    averaged, n_incomplete = average_across_checkpoints(_per_ckpt({}, {}))
    assert averaged == {}
    assert n_incomplete == 0
