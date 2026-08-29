"""Tests for the unwindowed-protein WT identity sensitivity analysis."""

import pytest

from esm2_mech.experiments.mechanism import wt_identity_sensitivity as sensitivity


def test_short_protein_mask_includes_length_boundary():
    variants = [
        {"uniprot_id": "short"},
        {"uniprot_id": "boundary"},
        {"uniprot_id": "long"},
    ]
    sequences = {"short": "A" * 10, "boundary": "A" * 12, "long": "A" * 13}

    mask = sensitivity.build_short_protein_mask(variants, sequences, max_seq_len=12)

    assert mask.tolist() == [True, True, False]


def test_short_protein_mask_raises_for_missing_sequence():
    with pytest.raises(ValueError, match="sequence cache lacks"):
        sensitivity.build_short_protein_mask([{"uniprot_id": "missing"}], {})


def _seed_result(seed, low, high):
    gap = {"point_diff": 0.1, "ci_low": low, "ci_high": high, "n_clusters": 20}
    return (
        seed,
        f"seed{seed}.json",
        {"family_split": {sensitivity.WT_ONLY_FEATURE: {"split_gap_paired": gap}}},
    )


def test_split_gap_summary_applies_three_of_five_rule():
    seed_results = [
        _seed_result(0, 0.01, 0.20),
        _seed_result(1, 0.02, 0.18),
        _seed_result(2, 0.03, 0.15),
        _seed_result(3, -0.04, 0.12),
        _seed_result(4, -0.02, 0.10),
    ]

    summary = sensitivity.summarize_split_gap(seed_results)

    assert summary["seed_vote"]["payload"]["supporting_seeds"] == [0, 1, 2]
    assert summary["preregistered_rule_evaluable"] is True
    assert summary["meets_claim_2b_interval_rule"] is True


def test_split_gap_summary_does_not_adjudicate_incomplete_run():
    summary = sensitivity.summarize_split_gap(
        [_seed_result(seed, 0.01, 0.20) for seed in range(4)]
    )

    assert summary["preregistered_rule_evaluable"] is False
    assert summary["meets_claim_2b_interval_rule"] is None


def test_negative_interval_does_not_support_positive_leakage_gap():
    seed_results = [
        _seed_result(0, -0.20, -0.01),
        _seed_result(1, 0.01, 0.20),
        _seed_result(2, 0.02, 0.18),
        _seed_result(3, -0.04, 0.12),
        _seed_result(4, -0.02, 0.10),
    ]

    summary = sensitivity.summarize_split_gap(seed_results)

    assert summary["seed_vote"]["payload"]["supporting_seeds"] == [1, 2]
    assert summary["contradictory_seeds"] == [0]
    assert summary["meets_claim_2b_interval_rule"] is False
