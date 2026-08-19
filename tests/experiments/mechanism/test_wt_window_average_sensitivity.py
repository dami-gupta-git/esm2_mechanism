"""Tests for the WT window-average sensitivity analysis."""

import numpy as np
import pytest

from esm2_mech.experiments.mechanism import wt_window_average_sensitivity as sensitivity
from esm2_mech.utils.constants import MECHANISM_CLASSES


def test_unique_windows_receive_equal_weight_not_variant_count():
    variants = [
        {"uniprot_id": "P1", "aa_pos": 10},
        {"uniprot_id": "P1", "aa_pos": 20},
        {"uniprot_id": "P1", "aa_pos": 1500},
    ]
    sequences = {"P1": "A" * 2000}
    embeddings = np.array([[0.0, 0.0], [2.0, 2.0], [10.0, 10.0]])

    averaged, metadata = sensitivity.build_protein_window_average(
        variants, sequences, embeddings
    )

    # Positions 10 and 20 share start 0, so that window contributes (0+2)/2=1;
    # the second window contributes 10. Equal window weighting gives (1+10)/2.
    assert np.allclose(averaged, 5.5)
    assert metadata["n_proteins_with_multiple_observed_windows"] == 1
    assert metadata["unique_window_count_distribution"] == {"2": 1}


def test_missing_sequence_raises():
    with pytest.raises(ValueError, match="sequence cache lacks"):
        sensitivity.build_protein_window_average(
            [{"uniprot_id": "missing", "aa_pos": 1}],
            {},
            np.ones((1, 2)),
        )


def _cached_arm(predictions, *, folds, n_rows=60):
    labels = np.array(
        [MECHANISM_CLASSES[row % len(MECHANISM_CLASSES)] for row in range(n_rows)]
    )
    genes = np.array([f"G{row % 12}" for row in range(n_rows)], dtype=object)
    return {
        "row_ids": np.arange(n_rows),
        "y_true": labels,
        "pred": np.asarray(predictions(labels)),
        "genes": genes,
        "folds": np.asarray(folds),
    }


def test_paired_comparison_reports_change_in_split_gap():
    n_rows = 60
    folds = np.repeat(np.arange(5), n_rows // 5)

    def correct(labels):
        return labels

    def wrong(labels):
        return np.array(
            [MECHANISM_CLASSES[(MECHANISM_CLASSES.index(label) + 1) % 3] for label in labels]
        )

    original = {
        "gene_split": _cached_arm(correct, folds=folds),
        "family_split": _cached_arm(wrong, folds=folds),
    }
    averaged = {
        "gene_split": _cached_arm(correct, folds=folds),
        "family_split": _cached_arm(correct, folds=folds),
    }
    pfam_map = {f"G{gene}": f"PF{gene % 4}" for gene in range(12)}

    result = sensitivity.compare_conditions_for_seed(
        original,
        averaged,
        pfam_map,
        n_resamples=50,
        seed=0,
        n_jobs=1,
    )

    assert result["gene_split_method_difference"]["point_diff"] == pytest.approx(0.0)
    assert result["family_split_method_difference"]["point_diff"] == pytest.approx(1.0)
    assert result["split_gap_method_difference"]["point_diff"] == pytest.approx(-1.0)
