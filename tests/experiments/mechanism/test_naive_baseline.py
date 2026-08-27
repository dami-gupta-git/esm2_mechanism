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
import pytest

from esm2_mech.experiments.mechanism.naive_baseline import floor_macro_f1_ci
from esm2_mech.utils.constants import GOF, DN, LOF


def _make_dataset():
    labels = []
    genes = []
    pfam_map = {}
    # Every annotated gene supplies every class with LOF as the unique majority.
    for gene_index in range(15):
        gene = f"g{gene_index}"
        genes.extend([gene] * 4)
        labels.extend([GOF, DN, LOF, LOF])
        pfam_map[gene] = f"PF{gene_index:03d}"
    # One unannotated LOF-only gene changes the full-cohort floor and is excluded
    # from the family-split floor.
    genes.extend(["g_unannotated"] * 6)
    labels.extend([LOF] * 6)
    return np.array(labels), np.array(genes), pfam_map


def test_floor_returns_both_splits_with_expected_keys():
    labels, genes, pfam_map = _make_dataset()
    out = floor_macro_f1_ci(labels, genes, pfam_map, seed=0, n_boot=200)

    assert set(out) == {"gene", "family"}
    for cell in out.values():
        assert {"point", "ci_low", "ci_high", "n_resamples", "n_clusters"} <= set(cell)


def test_family_floor_excludes_unannotated_genes():
    labels, genes, pfam_map = _make_dataset()
    out = floor_macro_f1_ci(labels, genes, pfam_map, seed=0, n_boot=200)

    assert out["family"]["n_clusters"] == 15
    assert out["gene"]["n_clusters"] == 16


def test_family_point_matches_macro_f1_on_annotated_subset_only():
    labels, genes, pfam_map = _make_dataset()
    out = floor_macro_f1_ci(labels, genes, pfam_map, seed=0, n_boot=50)

    # LOF has prevalence one half in the annotated cohort, so its F1 is two thirds;
    # fixed-class macro-F1 divides that by three.
    assert out["family"]["point"] == pytest.approx(2.0 / 9.0)
    assert out["family"]["point"] != out["gene"]["point"]


def test_floor_is_deterministic_for_fixed_seed():
    labels, genes, pfam_map = _make_dataset()
    a = floor_macro_f1_ci(labels, genes, pfam_map, seed=7, n_boot=100)
    b = floor_macro_f1_ci(labels, genes, pfam_map, seed=7, n_boot=100)
    assert a == b


def test_floor_suppresses_interval_when_required_class_is_absent():
    labels = []
    genes = []
    pfam_map = {}
    for gene_index in range(10):
        gene = f"g{gene_index}"
        genes.extend([gene] * 3)
        labels.extend([LOF, GOF, LOF])
        pfam_map[gene] = f"PF{gene_index:03d}"
    labels = np.array(labels)
    genes = np.array(genes)

    out = floor_macro_f1_ci(labels, genes, pfam_map, seed=0, n_boot=20)

    assert out["gene"]["point"] is None
    assert out["gene"]["ci_suppressed"] is True
    assert out["family"]["point"] is None
    assert out["family"]["ci_suppressed"] is True
