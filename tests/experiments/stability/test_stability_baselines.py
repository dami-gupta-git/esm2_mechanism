"""
Tests for the Megascale stability baselines.

These execute the real scoring paths on a small synthetic dataset, and pin the
shared-contract key the baselines read out of the stability probe.

Covers:
- run_ridge_with_auroc: still publishes the spearman_mean key the baselines read
- standardize: scaling statistics come from the fit array only
- standardize: the fit array is returned standardized alongside the others
- _aggregate_over_seeds: builds an aggregate for all three split schemes
- _aggregate_over_seeds: a seed whose split scored None is not counted as success
- delta_norm_baseline: scores every split from the single delta-norm feature
- delta_norm_baseline: recovers signal when ddG is a function of the norm
- nested_alpha_ridge: reports the alpha grid and a chosen-alpha median
- nested_alpha_ridge: the chosen alpha always comes from the declared grid
- label_shuffle_null: permuted labels collapse the correlation toward zero
- label_shuffle_null: the caller's ddG array is not permuted in place
- pls_component_sweep: reports one value per requested component count
"""

import numpy as np
import pytest

from esm2_mech.experiments.stability.stability_baselines import (
    ALPHA_GRID,
    PLS_COMPONENTS,
    _aggregate_over_seeds,
    delta_norm_baseline,
    label_shuffle_null,
    nested_alpha_ridge,
    pls_component_sweep,
)
from esm2_mech.experiments.stability.megascale_stability import run_ridge_with_auroc
from esm2_mech.experiments.stability.stability_data import stability_splits
from esm2_mech.utils.metrics import standardize


N_DOMAINS = 10
PER_DOMAIN = 12
N_ROWS = N_DOMAINS * PER_DOMAIN
# Wide enough for the largest PLS component count in the sweep.
N_FEATURES = 64


@pytest.fixture
def dataset():
    """Synthetic stability data where ddG is a noisy linear function of the
    embedding."""
    rng = np.random.RandomState(0)
    delta_mean = rng.normal(size=(N_ROWS, N_FEATURES))
    weights = rng.normal(size=N_FEATURES)
    ddg = delta_mean @ weights + rng.normal(scale=0.1, size=N_ROWS)
    proteins = np.array([f"D{i // PER_DOMAIN}" for i in range(N_ROWS)])
    family_map = {f"D{i}": f"F{i % 5}" for i in range(N_DOMAINS)}
    return delta_mean, ddg, proteins, family_map


# ---------------------------------------------------------------------------
# shared contract the baselines depend on
# ---------------------------------------------------------------------------


def test_ridge_probe_still_publishes_spearman_mean(dataset):
    """The baselines read this key with .get; a rename would silently return None."""
    delta_mean, ddg, proteins, family_map = dataset
    splits = stability_splits(0, len(ddg), proteins, family_map)
    result = run_ridge_with_auroc(delta_mean, ddg, splits["random"])
    assert result["spearman_status"] == "success"
    assert "spearman_mean" in result
    assert np.isfinite(result["spearman_mean"])


def test_standardize_fits_statistics_on_the_fit_array_only():
    fit = np.array([[0.0], [2.0]])
    other = np.array([[10.0]])
    scaled_fit, scaled_other = standardize(fit, other)
    assert scaled_fit.mean() == pytest.approx(0.0, abs=1e-6)
    # (10 - 1) / 1 using the fit array's mean of 1 and standard deviation of 1.
    assert scaled_other[0, 0] == pytest.approx(9.0, abs=1e-5)


def test_standardize_returns_the_fit_array_first():
    fit = np.array([[1.0], [3.0]])
    returned = standardize(fit, np.array([[5.0]]), np.array([[7.0]]))
    assert len(returned) == 3
    assert returned[0].shape == fit.shape


# ---------------------------------------------------------------------------
# _aggregate_over_seeds
# ---------------------------------------------------------------------------


def test_aggregate_over_seeds_covers_every_split_scheme():
    per_seed = [
        {"seed": 0, "random": 0.5, "domain": 0.4, "family": 0.3},
        {"seed": 1, "random": 0.7, "domain": 0.6, "family": 0.5},
    ]
    aggregate = _aggregate_over_seeds((0, 1), per_seed)
    assert set(aggregate) == {"random", "domain", "family"}
    assert aggregate["random"]["state"] == "available"
    assert aggregate["random"]["mean"] == pytest.approx(0.6)


def test_aggregate_over_seeds_does_not_count_an_unscored_split():
    """A split that returned None must not be averaged in as a real observation."""
    per_seed = [
        {"seed": 0, "random": 0.5, "domain": None, "family": 0.3},
        {"seed": 1, "random": 0.7, "domain": 0.6, "family": 0.5},
    ]
    aggregate = _aggregate_over_seeds((0, 1), per_seed)
    assert aggregate["domain"]["state"] == "unavailable"
    assert aggregate["domain"]["mean"] is None
    assert aggregate["domain"]["affected_seeds"] == [0]


# ---------------------------------------------------------------------------
# scoring paths
# ---------------------------------------------------------------------------


def test_delta_norm_baseline_scores_every_split(dataset):
    delta_mean, ddg, proteins, family_map = dataset
    result = delta_norm_baseline(delta_mean, ddg, proteins, family_map, 1, (0, 1))
    assert set(result) == {"random", "domain", "family"}
    for split in result.values():
        assert split["requested_seeds"] == [0, 1]


def test_delta_norm_baseline_recovers_a_norm_driven_signal(dataset):
    """When ddG is the embedding norm, the one-feature baseline must correlate
    strongly."""
    delta_mean, _, proteins, family_map = dataset
    ddg = np.linalg.norm(delta_mean, axis=1)
    result = delta_norm_baseline(delta_mean, ddg, proteins, family_map, 1, (0,))
    assert result["random"]["mean"] > 0.9


def test_nested_alpha_ridge_reports_grid_and_chosen_alpha(dataset):
    delta_mean, ddg, proteins, family_map = dataset
    result = nested_alpha_ridge(delta_mean, ddg, proteins, family_map, 1, (0,))
    assert result["alpha_grid"] == list(ALPHA_GRID)
    assert result["chosen_alpha_median"] is not None


def test_nested_alpha_ridge_chooses_from_the_declared_grid(dataset):
    delta_mean, ddg, proteins, family_map = dataset
    result = nested_alpha_ridge(delta_mean, ddg, proteins, family_map, 1, (0,))
    assert result["chosen_alpha_median"] in ALPHA_GRID


def test_label_shuffle_null_collapses_the_correlation(dataset):
    """Permuting ddG must destroy the signal the same probe finds on real labels.

    The comparison uses the full-embedding ridge probe, which is what the null
    itself fits — the one-feature delta-norm baseline is a different arm.
    """
    delta_mean, ddg, proteins, family_map = dataset
    splits = stability_splits(0, len(ddg), proteins, family_map)
    real = run_ridge_with_auroc(delta_mean, ddg, splits["random"])["spearman_mean"]
    null = label_shuffle_null(delta_mean, ddg, proteins, family_map, 1, (0, 1, 2))
    assert abs(null["random"]["mean"]) < 0.3
    assert null["random"]["mean"] < real


def test_label_shuffle_null_does_not_permute_the_callers_array(dataset):
    delta_mean, ddg, proteins, family_map = dataset
    original = ddg.copy()
    label_shuffle_null(delta_mean, ddg, proteins, family_map, 1, (0,))
    assert np.array_equal(ddg, original)


def test_pls_component_sweep_reports_every_requested_component_count(dataset):
    delta_mean, ddg, proteins, family_map = dataset
    result = pls_component_sweep(delta_mean, ddg, proteins, family_map, 1)
    assert set(result) == {"random", "family"}
    for split in result.values():
        assert set(split) == {str(n) for n in PLS_COMPONENTS}
