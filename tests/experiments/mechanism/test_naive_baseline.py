"""
Tests for floor_macro_f1_ci in
esm2_mech.experiments.mechanism.naive_baseline.

This computes the majority-class baseline floor that every real result is
compared against. The risky part is the family CI: it must exclude genes with
no Pfam family and remap bootstrap row indices back into the full label array
via fam_rows[local_rows]. A misalignment there silently corrupts the reported
floor without crashing.
"""

import numpy as np

from esm2_mech.experiments.mechanism.naive_baseline import floor_macro_f1_ci
from esm2_mech.utils.constants import GOF, DN, LOF


def _make_dataset():
    # 6 variants across 3 genes; LOF is the global majority class.
    labels = np.array([LOF, LOF, LOF, GOF, DN, LOF])
    genes = np.array(["g1", "g1", "g2", "g2", "g3", "g3"])
    # g3 has no Pfam family -> must be dropped from the family floor.
    pfam_map = {"g1": "PF001", "g2": "PF002"}
    return labels, genes, pfam_map


def test_floor_returns_both_splits_with_expected_keys():
    labels, genes, pfam_map = _make_dataset()
    out = floor_macro_f1_ci(labels, genes, pfam_map, seed=0, n_boot=200)

    assert set(out) == {"gene", "family"}
    for cell in out.values():
        assert {"point", "ci_low", "ci_high", "n_resamples", "n_clusters"} <= set(cell)


def test_family_floor_excludes_unannotated_genes():
    labels, genes, pfam_map = _make_dataset()
    out = floor_macro_f1_ci(labels, genes, pfam_map, seed=0, n_boot=200)

    # Only g1 (PF001) and g2 (PF002) are annotated -> 2 family clusters.
    assert out["family"]["n_clusters"] == 2
    # Gene split keeps all 3 genes.
    assert out["gene"]["n_clusters"] == 3


def test_family_point_matches_macro_f1_on_annotated_subset_only():
    labels, genes, pfam_map = _make_dataset()
    out = floor_macro_f1_ci(labels, genes, pfam_map, seed=0, n_boot=50)

    # Reconstruct the expected family point estimate independently: majority is
    # LOF; macro-F1 over ONLY the annotated rows (g1,g1,g2,g2 -> LOF,LOF,LOF,GOF).
    from sklearn.metrics import f1_score

    annotated_mask = np.array([g in pfam_map for g in genes])
    majority = LOF  # LOF appears most across all labels
    pred = np.full(annotated_mask.sum(), majority)
    expected = f1_score(
        labels[annotated_mask], pred, average="macro", zero_division=0
    )
    assert out["family"]["point"] == expected
    # And it must differ from the all-rows gene point (g3 rows change the balance),
    # proving the family floor really used a different (annotated) subset.
    assert out["family"]["point"] != out["gene"]["point"]


def test_floor_is_deterministic_for_fixed_seed():
    labels, genes, pfam_map = _make_dataset()
    a = floor_macro_f1_ci(labels, genes, pfam_map, seed=7, n_boot=100)
    b = floor_macro_f1_ci(labels, genes, pfam_map, seed=7, n_boot=100)
    assert a == b
