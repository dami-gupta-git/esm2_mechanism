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
    # 8 variants across 4 genes; LOF is the global majority class and every
    # mechanism class remains in the annotated subset.
    labels = np.array([LOF, LOF, GOF, LOF, DN, LOF, GOF, GOF])
    genes = np.array(["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"])
    # g4 has no Pfam family -> must be dropped from the family floor.
    pfam_map = {"g1": "PF001", "g2": "PF002", "g3": "PF003"}
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

    # Only g1, g2 and g3 are annotated -> 3 family clusters.
    assert out["family"]["n_clusters"] == 3
    # Gene split keeps all 4 genes.
    assert out["gene"]["n_clusters"] == 4


def test_family_point_matches_macro_f1_on_annotated_subset_only():
    labels, genes, pfam_map = _make_dataset()
    out = floor_macro_f1_ci(labels, genes, pfam_map, seed=0, n_boot=50)

    # Reconstruct the expected family point estimate independently: majority is
    # LOF; macro-F1 over only the annotated rows.
    from sklearn.metrics import f1_score

    annotated_mask = np.array([pfam_map.get(g) is not None for g in genes])
    majority = LOF  # LOF appears most across all labels
    pred = np.full(annotated_mask.sum(), majority)
    expected = f1_score(
        labels[annotated_mask], pred, labels=[GOF, DN, LOF],
        average="macro", zero_division=0
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


def test_floor_suppresses_interval_when_required_class_is_absent():
    labels = np.array([LOF, LOF, GOF, GOF])
    genes = np.array(["g1", "g1", "g2", "g2"])
    pfam_map = {"g1": "PF001", "g2": "PF002"}

    out = floor_macro_f1_ci(labels, genes, pfam_map, seed=0, n_boot=20)

    assert out["gene"]["point"] is None
    assert out["gene"]["ci_suppressed"] is True
    assert out["family"]["point"] is None
    assert out["family"]["ci_suppressed"] is True
