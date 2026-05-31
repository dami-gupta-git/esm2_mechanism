"""
Tests for experiments/mechanism/family_clustering.py pure-logic functions.

Covers (the I/O-free functions only; main() is excluded):
- gene_level_embeddings: averages per-variant rows to one row per gene
- gene_level_embeddings: returns genes in sorted-unique order
- knn_family_purity: ~1.0 purity and large positive z for separable clusters
- knn_family_purity: returns nan triple when n <= k
- within_between_ratio: ratio < 1 when within-family is tighter than between
- within_between_ratio: returns nan triple when too few same/diff pairs
- family_probe: high accuracy on separable families meeting size thresholds
- family_probe: "not enough families with min size" note below thresholds
- family_probe: "smallest kept family too small for CV" note when min size 2
"""

import numpy as np
import pytest

from esm2_mech.experiments.mechanism.family_clustering import (
    family_probe,
    gene_level_embeddings,
    knn_family_purity,
    within_between_ratio,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _clustered_points(centers, per_cluster, dim, jitter=0.01, seed=0):
    """Build points and family labels tightly grouped around given centers.

    Returns (emb, families) where each center contributes `per_cluster` points
    near it, labelled "fam{center_index}".
    """
    rng = np.random.RandomState(seed)
    emb = []
    families = []
    for ci, center in enumerate(centers):
        base = np.zeros(dim, dtype=np.float32)
        base[: len(center)] = center
        for _ in range(per_cluster):
            emb.append(base + rng.normal(0, jitter, dim).astype(np.float32))
            families.append(f"fam{ci}")
    return np.array(emb, dtype=np.float32), families


# ---------------------------------------------------------------------------
# gene_level_embeddings
# ---------------------------------------------------------------------------

def test_gene_level_embeddings_averages_per_gene():
    emb = np.array(
        [
            [0.0, 0.0],
            [2.0, 4.0],  # gene A: mean (1, 2)
            [10.0, 10.0],  # gene B: single row, mean = itself
        ],
        dtype=np.float32,
    )
    genes_arr = np.array(["A", "A", "B"])

    gene_names, gene_emb = gene_level_embeddings(emb, genes_arr)

    assert list(gene_names) == ["A", "B"]
    np.testing.assert_allclose(gene_emb[0], [1.0, 2.0])
    np.testing.assert_allclose(gene_emb[1], [10.0, 10.0])


def test_gene_level_embeddings_sorted_unique_order():
    emb = np.ones((4, 3), dtype=np.float32)
    genes_arr = np.array(["Z", "A", "M", "A"])

    gene_names, gene_emb = gene_level_embeddings(emb, genes_arr)

    assert list(gene_names) == ["A", "M", "Z"]
    assert gene_emb.shape == (3, 3)


# ---------------------------------------------------------------------------
# knn_family_purity
# ---------------------------------------------------------------------------

def test_knn_family_purity_high_for_separable_clusters():
    # 4 well-separated families, 8 members each, in distinct corners of 8-D space.
    centers = [
        [10, 0, 0, 0],
        [0, 10, 0, 0],
        [0, 0, 10, 0],
        [0, 0, 0, 10],
    ]
    emb, families = _clustered_points(centers, per_cluster=8, dim=8, seed=1)

    purity, null, z = knn_family_purity(emb, families, k=5, seed=42)

    assert purity == pytest.approx(1.0, abs=1e-6)
    assert null < purity
    assert z > 5.0


def test_knn_family_purity_nan_when_n_le_k():
    emb = np.random.RandomState(0).normal(size=(5, 4)).astype(np.float32)
    families = ["a", "a", "b", "b", "c"]

    purity, null, z = knn_family_purity(emb, families, k=5)

    assert np.isnan(purity)
    assert np.isnan(null)
    assert np.isnan(z)


# ---------------------------------------------------------------------------
# within_between_ratio
# ---------------------------------------------------------------------------

def test_within_between_ratio_below_one_for_tight_families():
    centers = [
        [10, 0, 0, 0],
        [0, 10, 0, 0],
        [0, 0, 10, 0],
    ]
    emb, families = _clustered_points(centers, per_cluster=6, dim=8, seed=2)

    ratio, null, z = within_between_ratio(emb, families, seed=42)

    # Within-family cosine distance is tiny, between-family is large.
    assert ratio < 1.0
    assert ratio < null
    # Real ratio is far below the shuffled null (which sits near 1).
    assert z < -2.0


def test_within_between_ratio_nan_when_too_few_pairs():
    # Only 4 points, each a singleton family -> 0 same-family pairs (< 5).
    emb = np.eye(4, dtype=np.float32)
    families = ["a", "b", "c", "d"]

    ratio, null, z = within_between_ratio(emb, families)

    assert np.isnan(ratio)
    assert np.isnan(null)
    assert np.isnan(z)


# ---------------------------------------------------------------------------
# family_probe
# ---------------------------------------------------------------------------

def test_family_probe_high_accuracy_on_separable_families():
    # 6 families x 6 genes each = 36 genes (>= 30), >= 5 families, smallest size 6.
    centers = [
        [20, 0, 0, 0, 0, 0],
        [0, 20, 0, 0, 0, 0],
        [0, 0, 20, 0, 0, 0],
        [0, 0, 0, 20, 0, 0],
        [0, 0, 0, 0, 20, 0],
        [0, 0, 0, 0, 0, 20],
    ]
    gene_emb, gene_families = _clustered_points(
        centers, per_cluster=6, dim=12, jitter=0.1, seed=3
    )

    result = family_probe(gene_emb, gene_families, seed=42)

    assert "accuracy" in result
    assert result["accuracy"] > 0.9
    assert result["n_families"] == 6
    assert result["n_genes"] == 36


def test_family_probe_note_when_not_enough_families():
    # Below the 30-gene / 5-family threshold.
    centers = [[10, 0], [0, 10]]
    gene_emb, gene_families = _clustered_points(centers, per_cluster=4, dim=4, seed=4)

    result = family_probe(gene_emb, gene_families, seed=42)

    assert result == {"note": "not enough families with min size"}


def test_family_probe_note_when_smallest_kept_family_too_small():
    # >= 30 genes across >= 5 families, but every kept family has exactly 2 members,
    # so n_splits collapses to < 2 and CV cannot run.
    centers = [[c, 0, 0, 0, 0] for c in range(20)]
    # 20 families x 2 members = 40 genes; all kept (min size 2), smallest size 2.
    gene_emb, gene_families = _clustered_points(
        centers, per_cluster=2, dim=10, jitter=0.05, seed=5
    )

    result = family_probe(gene_emb, gene_families, seed=42, min_family_size=2)

    assert result == {"note": "smallest kept family too small for CV"}
