import numpy as np
import pytest

from esm2_mech.experiments.stability.megascale_mlp import run_mlp_regression
from esm2_mech.experiments.stability.megascale_stability import (
    apply_decision_rule,
    run_ridge_with_auroc,
    paired_spearman_gap_ci,
    run_stability_projection_3c,
)
from esm2_mech.utils.constants import BOOTSTRAP_MAX_DISCARD_FRAC

N_FOLDS = 5
N_FAMILIES = 50
VARIANTS_PER_GENE = 3


def test_each_gate_reads_the_point_the_interval_beside_it_was_built_around():
    adjudication = apply_decision_rule(
        {"point": 0.60, "ci_low": 0.55, "ci_high": 0.65},
        {"point_diff": 0.05, "ci_low": 0.01, "ci_high": 0.08},
        None,
        {"point": 0.07, "ci_low": 0.04, "ci_high": 0.09},
    )

    gates = adjudication["gates"]
    assert gates["3A"]["point_estimate"] == 0.60
    assert gates["3A"]["ci"]["ci_low"] == 0.55
    assert gates["3B"]["point_estimate"] == 0.05
    assert gates["3D"]["point_estimate"] == 0.07
    # 3C was not supplied, so it alone stays unadjudicated.
    assert gates["3C"]["point_estimate"] is None
    assert gates["3C"]["ci"] is None


def test_a_gate_without_its_interval_is_not_adjudicated():
    adjudication = apply_decision_rule(None, None, None, None)

    assert adjudication["overall"] != "PASS"
    assert all(gate["ci"] is None for gate in adjudication["gates"].values())


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


def test_3c_returns_a_paired_seed_point_with_its_interval():
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
    )

    difference = result["projected_minus_baseline_f1"]
    assert difference["requested_seeds"] == [0]
    assert difference["mean"] is not None
    assert difference["seed_std"] is None
    interval = result["difference_ci"]
    assert interval["point_diff"] is not None
    if not interval.get("ci_suppressed"):
        assert interval["ci_low"] <= interval["point_diff"] <= interval["ci_high"]


def test_3c_interval_does_not_discard_a_material_share_of_its_draws():
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
    )

    # A high discard rate means folds are losing whole classes, so the surviving
    # draws would be a different statistic from the point estimate.
    interval = result["difference_ci"]
    assert interval["discard_frac"] <= BOOTSTRAP_MAX_DISCARD_FRAC


def test_an_undefined_auroc_fold_does_not_withhold_the_rank_correlation():
    """Each metric is judged on its own folds.

    A fold whose held-out variants all sit on one side of the median has no
    AUROC, but its Spearman correlation is perfectly well defined, so the two
    statuses are reported separately.
    """
    rng = np.random.RandomState(0)
    n_rows = 60
    features = rng.randn(n_rows, 4)
    ddg = features[:, 0] + 0.1 * rng.randn(n_rows)
    splits = [
        (np.setdiff1d(np.arange(n_rows), test), test)
        for test in np.array_split(np.arange(n_rows), 3)
    ]
    # A median above every value leaves each fold's binarised labels all-negative.
    result = run_ridge_with_auroc(
        features, ddg, splits, median=float(ddg.max() + 1.0)
    )

    assert result["spearman_status"] == "success"
    assert result["spearman_mean"] is not None
    assert result["auroc_status"] == "unscorable"
    assert "auroc_mean" not in result
