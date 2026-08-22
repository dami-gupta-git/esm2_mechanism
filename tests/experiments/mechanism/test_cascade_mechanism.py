"""
Tests for experiments/mechanism/cascade_mechanism.py training-fold resampling.

The resampling is what the cascade's stage-A claim rests on, so these cover the
invariants a reader has to be able to assume:
- _round_robin_by_key: a small bucket still contributes when the pool is cut
- _round_robin_by_key: returns every row when n_keep meets or exceeds the pool
- family_matched_training_rows: every kept family holds equal LOF and non-LOF
- family_matched_training_rows: a family with only one class is dropped
- family_matched_training_rows: both GOF and DN survive the non-LOF downsample
- family_matched_training_rows: a ratio above 1 tops LOF up from dropped families
- family_matched_training_rows: selection never reaches outside the training rows
- lof_cluster_assignment: recovers separated groups and reports the realised design
- lof_cluster_assignment: reports one cluster rather than failing on a single row
"""

import numpy as np
import pytest

from esm2_mech.experiments.mechanism.cascade_mechanism import (
    _round_robin_by_key,
    family_matched_training_rows,
    lof_cluster_assignment,
)
from esm2_mech.utils.constants import DN, GOF, LOF


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_fold(family_labels: dict[str, list[str]]):
    """Build (labels, families, train_rows, is_lof, cluster_of) from {family: labels}."""
    labels, families = [], []
    for family, family_label_list in family_labels.items():
        labels.extend(family_label_list)
        families.extend([family] * len(family_label_list))
    labels = np.array(labels)
    families = np.array(families)
    is_lof = labels == LOF
    train_rows = np.arange(len(labels))
    cluster_of = {int(row): 0 for row in np.where(is_lof)[0]}
    return labels, families, train_rows, is_lof, cluster_of


# ---------------------------------------------------------------------------
# _round_robin_by_key
# ---------------------------------------------------------------------------


def test_round_robin_draws_from_a_small_bucket():
    rows = list(range(20))
    key_of = {row: ("big" if row < 17 else "small") for row in rows}
    picked = _round_robin_by_key(rows, key_of, 6, np.random.RandomState(0))
    assert len(picked) == 6
    assert sum(key_of[row] == "small" for row in picked) == 3


def test_round_robin_returns_everything_when_nothing_is_cut():
    rows = list(range(5))
    key_of = {row: 0 for row in rows}
    picked = _round_robin_by_key(rows, key_of, 9, np.random.RandomState(0))
    assert sorted(picked) == rows


# ---------------------------------------------------------------------------
# family_matched_training_rows
# ---------------------------------------------------------------------------


def test_matched_families_carry_equal_class_counts():
    labels, families, train_rows, is_lof, cluster_of = make_fold({
        "PF1": [LOF] * 8 + [GOF] * 3,
        "PF2": [LOF] * 2 + [DN] * 5,
    })
    selected, design = family_matched_training_rows(
        train_rows, is_lof, labels, families, cluster_of,
        target_ratio=1.0, rng=np.random.RandomState(0),
    )
    for family in ("PF1", "PF2"):
        in_family = selected[families[selected] == family]
        n_lof = int((labels[in_family] == LOF).sum())
        assert n_lof == len(in_family) - n_lof
    assert design["n_mixed_families"] == 2
    assert design["realised_lof_to_non_lof_ratio"] == pytest.approx(1.0)


def test_single_class_families_are_dropped():
    labels, families, train_rows, is_lof, cluster_of = make_fold({
        "PF1": [LOF] * 4 + [GOF] * 4,
        "PF_lof_only": [LOF] * 30,
        "PF_gof_only": [GOF] * 6,
    })
    selected, design = family_matched_training_rows(
        train_rows, is_lof, labels, families, cluster_of,
        target_ratio=1.0, rng=np.random.RandomState(0),
    )
    assert set(families[selected].tolist()) == {"PF1"}
    assert design["n_lof_only_families_dropped"] == 1
    assert design["n_non_lof_only_families_dropped"] == 1


def test_non_lof_downsample_keeps_both_gof_and_dn():
    labels, families, train_rows, is_lof, cluster_of = make_fold({
        "PF1": [LOF] * 4 + [GOF] * 20 + [DN] * 4,
    })
    selected, _design = family_matched_training_rows(
        train_rows, is_lof, labels, families, cluster_of,
        target_ratio=1.0, rng=np.random.RandomState(0),
    )
    kept = labels[selected]
    assert int((kept == LOF).sum()) == 4
    assert int((kept == GOF).sum()) == 2
    assert int((kept == DN).sum()) == 2


def test_ratio_above_one_tops_up_from_dropped_families():
    labels, families, train_rows, is_lof, cluster_of = make_fold({
        "PF1": [LOF] * 4 + [GOF] * 4,
        "PF_lof_only": [LOF] * 30,
    })
    selected, design = family_matched_training_rows(
        train_rows, is_lof, labels, families, cluster_of,
        target_ratio=2.0, rng=np.random.RandomState(0),
    )
    assert design["n_lof_topped_up_from_single_class_families"] == 4
    assert int((labels[selected] == LOF).sum()) == 8
    assert design["realised_lof_to_non_lof_ratio"] == pytest.approx(2.0)


def test_selection_stays_inside_the_training_rows():
    labels, families, all_rows, is_lof, cluster_of = make_fold({
        "PF1": [LOF] * 6 + [GOF] * 6,
        "PF2": [LOF] * 6 + [DN] * 6,
    })
    train_rows = all_rows[all_rows < 12]
    cluster_of = {row: 0 for row in cluster_of if row < 12}
    selected, _design = family_matched_training_rows(
        train_rows, is_lof, labels, families, cluster_of,
        target_ratio=1.0, rng=np.random.RandomState(0),
    )
    assert set(selected.tolist()) <= set(train_rows.tolist())


# ---------------------------------------------------------------------------
# lof_cluster_assignment
# ---------------------------------------------------------------------------


def test_clustering_separates_two_distant_groups():
    rng = np.random.RandomState(0)
    delta = np.vstack([
        rng.normal(-20.0, 0.1, size=(15, 6)),
        rng.normal(20.0, 0.1, size=(15, 6)),
    ]).astype(np.float32)
    cluster_of, design = lof_cluster_assignment(
        delta, np.arange(30), n_clusters=2, n_pca=4, seed=0
    )
    first_group = {cluster_of[row] for row in range(15)}
    second_group = {cluster_of[row] for row in range(15, 30)}
    assert len(first_group) == 1 and len(second_group) == 1
    assert first_group != second_group
    assert design["n_clusters_fitted"] == 2
    assert sorted(design["cluster_sizes"]) == [15, 15]


def test_clustering_reports_one_cluster_for_a_single_row():
    delta = np.zeros((1, 6), dtype=np.float32)
    cluster_of, design = lof_cluster_assignment(
        delta, np.arange(1), n_clusters=8, n_pca=4, seed=0
    )
    assert cluster_of == {0: 0}
    assert design["n_clusters_fitted"] == 1
    assert design["n_lof_rows"] == 1
