import numpy as np
import pytest

from esm2_mech.experiments.stability.megascale_mlp import run_mlp_regression
from esm2_mech.experiments.stability.megascale_stability import (
    apply_decision_rule,
    paired_spearman_gap_ci,
    run_stability_projection_3c,
)
from esm2_mech.utils.constants import BOOTSTRAP_MAX_DISCARD_FRAC

N_FOLDS = 5
N_FAMILIES = 50
VARIANTS_PER_GENE = 3


def test_3b_leaky_verdict_requires_ci_to_clear_threshold():
    control_3a_ci = {"point": 0.69, "ci_low": 0.61, "ci_high": 0.75}

    established = apply_decision_rule(
        control_3a_ci,
        {"point_diff": 0.14, "ci_low": 0.11, "ci_high": 0.17},
        None,
        None,
    )
    assert established["overall"] == "LEAKY"
    assert established["gates"]["3B"]["verdict"] == "failed"

    underpowered = apply_decision_rule(
        control_3a_ci,
        {"point_diff": 0.14, "ci_low": 0.08, "ci_high": 0.17},
        None,
        None,
    )
    assert underpowered["overall"] == "NOT FULLY ADJUDICATED"
    assert underpowered["gates"]["3B"]["verdict"] == "underpowered"

    unavailable = apply_decision_rule(control_3a_ci, None, None, None)
    assert unavailable["overall"] == "NOT FULLY ADJUDICATED"
    assert unavailable["gates"]["3B"]["verdict"].startswith("not adjudicated")


def test_stability_verdict_requires_all_four_intervals_to_clear_thresholds():
    control_3c = {
        "seed0_inference": {
            "point_estimate": -0.01,
            "difference_ci": {
                "point_diff": -0.01,
                "ci_low": -0.02,
                "ci_high": 0.0,
            },
        },
    }

    adjudication = apply_decision_rule(
        {"point": 0.60, "ci_low": 0.55, "ci_high": 0.65},
        {"point_diff": 0.05, "ci_low": 0.01, "ci_high": 0.08},
        control_3c,
        {"point": 0.07, "ci_low": 0.04, "ci_high": 0.09},
    )

    assert adjudication["overall"] == "ROBUST"
    assert {gate["verdict"] for gate in adjudication["gates"].values()} == {
        "affirmed"
    }


def test_failed_3c_gate_is_not_reported_as_unadjudicated():
    control_3c = {
        "seed0_inference": {
            "point_estimate": 0.03,
            "difference_ci": {
                "point_diff": 0.03,
                "ci_low": 0.02,
                "ci_high": 0.04,
            },
        },
    }
    adjudication = apply_decision_rule(
        {"point": 0.60, "ci_low": 0.55, "ci_high": 0.65},
        {"point_diff": 0.05, "ci_low": 0.01, "ci_high": 0.08},
        control_3c,
        {"point": 0.07, "ci_low": 0.04, "ci_high": 0.09},
    )

    assert adjudication["overall"] == "3C FAILED"
    assert adjudication["gates"]["3C"]["verdict"] == "failed"


def test_3b_gap_point_is_mean_of_within_fold_spearman_scores():
    y_true = np.array([0, 1, 2, 0, 1, 2], dtype=float)
    folds = np.array([0, 0, 0, 1, 1, 1])
    indices = np.arange(len(y_true))
    proteins = np.array(["P0", "P1", "P2", "P0", "P1", "P2"])
    family_map = {"P0": "F0", "P1": "F1", "P2": "F2"}
    common = {"y_true": y_true, "indices": indices, "folds": folds}
    arm_a = {**common, "pred": np.array([100, 101, 102, 0, 1, 2])}
    arm_b = {**common, "pred": np.array([2, 1, 0, 102, 101, 100])}

    result = paired_spearman_gap_ci(
        arm_a, arm_b, proteins, family_map, n_resamples=20, seed=0
    )

    assert result["point_a"] == pytest.approx(1.0)
    assert result["point_b"] == pytest.approx(-1.0)
    assert result["point_diff"] == pytest.approx(2.0)
    assert "domain_resampled_sensitivity" in result


def test_mlp_rejects_validation_groups_crossing_outer_split():
    X = np.arange(12, dtype=float).reshape(4, 3)
    y = np.arange(4, dtype=float)
    splits = [(np.array([0, 1]), np.array([2, 3]))]
    groups = np.array(["A", "B", "A", "C"])

    with pytest.raises(ValueError, match="outer CV train/test boundary"):
        run_mlp_regression(
            X, y, splits, groups, median=float(np.median(y)), max_epochs=1
        )


def _merged_set(rng, n_features):
    """One gene per family, several variants per gene, labels cycling over classes.

    The family count has to exceed the fold count by enough that a bootstrap draw
    still puts at least one family in every fold. The paired difference is scored
    within each fold, so a draw that empties a fold is discarded, and with one
    family per fold almost every draw would be.
    """
    genes = np.array(
        [f"G{i}" for i in range(N_FAMILIES) for _ in range(VARIANTS_PER_GENE)]
    )
    pfam_map = {f"G{i}": f"F{i}" for i in range(N_FAMILIES)}
    labels = np.tile(np.array(["DN", "GOF", "LOF"]), N_FAMILIES)
    delta = rng.normal(size=(len(labels), n_features))
    return delta, labels, genes, pfam_map


def test_3c_returns_paired_family_bootstrap_ci():
    rng = np.random.RandomState(3)
    n_features = 6
    stability_delta = rng.normal(size=(60, n_features))
    stability_ddg = stability_delta[:, 0] + 0.1 * rng.normal(size=60)
    mechanism_delta, labels, genes, pfam_map = _merged_set(rng, n_features)

    result = run_stability_projection_3c(
        mechanism_delta,
        labels,
        genes,
        pfam_map,
        [],
        stability_delta,
        stability_ddg,
        n_folds=N_FOLDS,
        n_seeds=1,
        n_boot=100,
    )

    ci = result["seed0_inference"]["difference_ci"]
    assert ci is not None
    assert ci["n_clusters"] == N_FAMILIES
    assert ci["point_diff"] is not None
    point = ci["point_diff"]
    if point <= 0.01:
        expected_verdict = (
            "affirmed" if ci["ci_high"] < 0.01 else "not distinguishable"
        )
    else:
        expected_verdict = "underpowered" if ci["ci_low"] <= 0.01 else "failed"
    assert result["3C_verdict"] == expected_verdict


def test_3c_discards_few_resamples_when_every_fold_holds_many_families():
    # A high discard rate means folds are losing whole classes, which makes the
    # surviving draws a different statistic from the point estimate. With ten
    # families per fold it should be near zero, so a regression that empties or
    # unbalances folds shows up here rather than in the interval.
    rng = np.random.RandomState(3)
    n_features = 6
    stability_delta = rng.normal(size=(60, n_features))
    stability_ddg = stability_delta[:, 0] + 0.1 * rng.normal(size=60)
    mechanism_delta, labels, genes, pfam_map = _merged_set(rng, n_features)

    result = run_stability_projection_3c(
        mechanism_delta,
        labels,
        genes,
        pfam_map,
        [],
        stability_delta,
        stability_ddg,
        n_folds=N_FOLDS,
        n_seeds=1,
        n_boot=200,
    )

    assert (
        result["seed0_inference"]["difference_ci"]["discard_frac"]
        <= BOOTSTRAP_MAX_DISCARD_FRAC
    )
