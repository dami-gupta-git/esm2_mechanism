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


def _seed_result(seed, gene=0.5, family=0.4, paired_gap=None):
    if paired_gap is None:
        paired_gap = gene - family
    return (
        seed,
        f"seed{seed}.json",
        {
            "gene_split": {
                sensitivity.WT_ONLY_FEATURE: {
                    "status": "success",
                    "macro_f1_mean": gene,
                }
            },
            "family_split": {
                sensitivity.WT_ONLY_FEATURE: {
                    "status": "success",
                    "macro_f1_mean": family,
                    "split_gap_paired": {"point_diff": paired_gap},
                }
            },
        },
    )


def test_split_gap_summary_pairs_values_within_seed():
    seed_results = [_seed_result(seed, 0.5 + seed / 100, 0.4) for seed in range(5)]

    summary = sensitivity.summarize_split_gap(seed_results, range(5))

    aggregate = summary["gene_minus_family_seed_aggregate"]
    assert aggregate["mean"] == pytest.approx(0.12)
    assert aggregate["state"] == "available"


def test_split_gap_summary_is_unavailable_for_incomplete_run():
    summary = sensitivity.summarize_split_gap(
        [_seed_result(seed) for seed in range(4)], range(5)
    )

    assert summary["gene_minus_family_seed_aggregate"]["state"] == "unavailable"


def test_split_gap_preserves_negative_within_seed_difference():
    seed_results = [
        _seed_result(0, gene=0.3, family=0.4),
        *[_seed_result(seed, gene=0.5, family=0.4) for seed in range(1, 5)],
    ]

    summary = sensitivity.summarize_split_gap(seed_results, range(5))

    assert summary["gene_minus_family_seed_aggregate"]["mean"] == pytest.approx(0.06)


def test_split_gap_uses_stored_row_aligned_estimate():
    seed_results = [
        _seed_result(seed, gene=0.9, family=0.1, paired_gap=0.2)
        for seed in range(5)
    ]

    summary = sensitivity.summarize_split_gap(seed_results, range(5))

    assert summary["gene_minus_family_seed_aggregate"]["mean"] == pytest.approx(0.2)
