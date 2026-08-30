"""Tests for the shared perturbation seed-summary adapter."""

import pytest

from esm2_mech.experiments.perturbation.seed_summary import aggregate_probe_results
from esm2_mech.utils.seed_aggregation import seed_result_contract


def _seed(seed, value):
    return {
        **seed_result_contract(seed),
        "results": {
            "probe_family_split": {
                "status": "success",
                "macro_f1_mean": value,
                "auroc_GOF_mean": value + 0.1,
            }
        },
    }


def test_probe_summary_uses_complete_requested_seed_set():
    per_seed = {seed: _seed(seed, value) for seed, value in enumerate((0.2, 0.4, 0.6))}

    summary = aggregate_probe_results((0, 1, 2), per_seed, ["probe_family_split"])

    macro_f1 = summary["probe_family_split"]["macro_f1"]
    assert macro_f1["mean"] == pytest.approx(0.4)
    assert macro_f1["contributing_seeds"] == [0, 1, 2]
    assert macro_f1["seed_std"] == pytest.approx(0.2)


def test_probe_summary_is_unavailable_when_requested_seed_is_missing():
    summary = aggregate_probe_results(
        (0, 1, 2), {0: _seed(0, 0.2), 2: _seed(2, 0.6)}, ["probe_family_split"]
    )

    macro_f1 = summary["probe_family_split"]["macro_f1"]
    assert macro_f1["state"] == "unavailable"
    assert macro_f1["reason"] == "missing_seed"
    assert macro_f1["affected_seeds"] == [1]


def test_declared_arm_no_seed_produced_is_reported_unavailable():
    per_seed = {seed: _seed(seed, value) for seed, value in enumerate((0.2, 0.4, 0.6))}

    summary = aggregate_probe_results(
        (0, 1, 2), per_seed, ["probe_family_split", "arm_no_seed_ran"]
    )

    assert set(summary) == {"probe_family_split", "arm_no_seed_ran"}
    missing = summary["arm_no_seed_ran"]["macro_f1"]
    assert missing["state"] == "unavailable"
    assert missing["mean"] is None


def test_one_undefined_metric_does_not_withhold_the_other():
    per_seed = {seed: _seed(seed, value) for seed, value in enumerate((0.2, 0.4, 0.6))}
    per_seed[1]["results"]["probe_family_split"]["auroc_GOF_mean"] = None

    summary = aggregate_probe_results((0, 1, 2), per_seed, ["probe_family_split"])

    assert summary["probe_family_split"]["macro_f1"]["state"] == "available"
    assert summary["probe_family_split"]["auroc_GOF"]["state"] == "unavailable"
