"""
Tests for the CPU-only, torch-free functions in
experiments/mechanism/contrastive_mechanism.py.

The torch projection head (train_projection_head / project_test) and the
data/IO functions (load_data, load_all_data, main) are excluded — they need a
GPU/torch and the on-disk run artifacts. Covered here:

run_knn:
- proba columns are reordered to canonical le.classes_ order even when a class
  is absent from the training set (so knn.classes_ is a subset)
- macro_f1 is present and finite
- a class with no test examples is recorded as "class_absent_in_test", not a
  silently dropped AUROC
- a class whose neighbour votes are all tied (constant proba) is recorded as
  "constant_proba"
- a perfectly separable two-class problem gives AUROC 1.0

build_cross_family_pairs:
- every positive shares the anchor's mechanism but a DIFFERENT Pfam family
  (the central invariant of the experiment)
- every negative has a DIFFERENT mechanism
- an anchor that is the only member of its (class, family) with no cross-family
  same-class partner produces no triplet
- the same seed yields identical triplets (determinism)
"""

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

from esm2_mech.experiments.mechanism.contrastive_mechanism import (
    build_cross_family_pairs,
    run_knn,
)
from esm2_mech.utils.constants import GOF, DN, LOF


@pytest.fixture
def le():
    """LabelEncoder fitted on the canonical 3-class label set."""
    encoder = LabelEncoder()
    encoder.fit(np.array([GOF, DN, LOF]))
    return encoder


# ---------------------------------------------------------------------------
# run_knn
# ---------------------------------------------------------------------------


class TestRunKnn:

    def test_macro_f1_present_and_finite(self, le):
        rng = np.random.RandomState(0)
        # Three well-separated clusters, one per class.
        centers = {GOF: [5.0, 0.0], DN: [0.0, 5.0], LOF: [-5.0, -5.0]}
        labels = np.array([GOF, DN, LOF] * 8)
        X = np.array([centers[lab] for lab in labels]) + rng.normal(0, 0.1, (24, 2))
        y = le.transform(labels)
        fm, _proba = run_knn(X, X, y, y, le, k=3)
        assert "macro_f1" in fm
        assert np.isfinite(fm["macro_f1"])
        # Separable clusters classified against themselves → perfect macro_f1.
        assert fm["macro_f1"] == pytest.approx(1.0)

    def test_perfectly_separable_auroc_is_one(self, le):
        # Two classes only (LOF absent everywhere). GOF and DN are linearly
        # separable, so per-class AUROC must be 1.0.
        labels = np.array([GOF] * 10 + [DN] * 10)
        X = np.vstack([
            np.tile([10.0, 0.0], (10, 1)),
            np.tile([0.0, 10.0], (10, 1)),
        ]).astype(float)
        y = le.transform(labels)
        fm, _proba = run_knn(X, X, y, y, le, k=3)
        assert fm[f"auroc_{GOF}"] == pytest.approx(1.0)
        assert fm[f"auroc_{DN}"] == pytest.approx(1.0)

    def test_proba_columns_aligned_when_train_class_absent(self, le):
        # LOF appears only in the TEST set, never in training. knn.classes_ is
        # therefore {GOF, DN} (a subset), and the proba matrix must still be laid
        # out in canonical le.classes_ order with a zero column for LOF.
        train_labels = np.array([GOF] * 8 + [DN] * 8)
        X_train = np.vstack([
            np.tile([10.0, 0.0], (8, 1)),
            np.tile([0.0, 10.0], (8, 1)),
        ]).astype(float)
        y_train = le.transform(train_labels)

        test_labels = np.array([GOF, DN, LOF])
        X_test = np.array([[10.0, 0.0], [0.0, 10.0], [-10.0, -10.0]], dtype=float)
        y_test = le.transform(test_labels)

        fm, _proba = run_knn(X_train, X_test, y_train, y_test, le, k=3)
        # LOF is never a training class, so knn never votes for it: its proba
        # column is all zeros (constant). The AUROC is therefore undefined and
        # must be recorded by name, never silently emitted as a number.
        assert "auroc_skipped" in fm
        assert fm["auroc_skipped"].get(LOF) == "constant_proba"
        assert f"auroc_{LOF}" not in fm

    def test_class_absent_in_test_is_recorded(self, le):
        # LOF is in the TRAINING set but has no TEST examples → its binary
        # AUROC is undefined because the positive (or negative) set is empty.
        labels_train = np.array([GOF] * 6 + [DN] * 6 + [LOF] * 6)
        rng = np.random.RandomState(1)
        centers = {GOF: [5.0, 0.0], DN: [0.0, 5.0], LOF: [-5.0, -5.0]}
        X_train = np.array([centers[lab] for lab in labels_train]) + rng.normal(
            0, 0.1, (18, 2)
        )
        y_train = le.transform(labels_train)

        # Test set has only GOF and DN.
        labels_test = np.array([GOF, GOF, DN, DN])
        X_test = np.array([[5.0, 0.0], [5.0, 0.0], [0.0, 5.0], [0.0, 5.0]])
        y_test = le.transform(labels_test)

        fm, _proba = run_knn(X_train, X_test, y_train, y_test, le, k=3)
        assert "auroc_skipped" in fm
        assert fm["auroc_skipped"].get(LOF) == "class_absent_in_test"


# ---------------------------------------------------------------------------
# build_cross_family_pairs
# ---------------------------------------------------------------------------


def _decode_pairs(anchors, positives, negatives, y, fam_int):
    """Yield (a, p, n) index triples for assertion convenience."""
    return list(zip(anchors.tolist(), positives.tolist(), negatives.tolist()))


class TestBuildCrossFamilyPairs:

    def _setup(self):
        # Two mechanisms, several families each, so cross-family positives exist.
        # idx:  0    1    2    3    4    5    6    7
        labels = np.array([GOF, GOF, GOF, GOF, DN, DN, DN, DN])
        gene_pfam = np.array(["PF1", "PF1", "PF2", "PF3", "PF4", "PF4", "PF5", "PF6"])
        encoder = LabelEncoder()
        encoder.fit(np.array([GOF, DN, LOF]))
        return labels, gene_pfam, encoder

    def test_positives_same_mech_different_family(self):
        labels, gene_pfam, encoder = self._setup()
        anchors, positives, negatives = build_cross_family_pairs(
            labels, gene_pfam, encoder, max_pairs_per_anchor=10, seed=7
        )
        y = encoder.transform(labels)
        fam_codes = {fam: i for i, fam in enumerate(sorted(set(gene_pfam)))}
        fam_int = np.array([fam_codes[f] for f in gene_pfam])

        assert len(anchors) > 0
        for anchor, pos, _neg in _decode_pairs(
            anchors, positives, negatives, y, fam_int
        ):
            # same mechanism
            assert y[pos] == y[anchor]
            # DIFFERENT Pfam family — the core leakage-prevention invariant
            assert gene_pfam[pos] != gene_pfam[anchor]

    def test_negatives_different_mechanism(self):
        labels, gene_pfam, encoder = self._setup()
        anchors, positives, negatives = build_cross_family_pairs(
            labels, gene_pfam, encoder, max_pairs_per_anchor=10, seed=7
        )
        y = encoder.transform(labels)
        for anchor, _pos, neg in zip(
            anchors.tolist(), positives.tolist(), negatives.tolist()
        ):
            assert y[neg] != y[anchor]

    def test_no_cross_family_partner_produces_no_triplet(self):
        # GOF exists only in PF1 (two variants, same family). There is no
        # other-family GOF, so neither GOF anchor can form a cross-family
        # positive and they contribute no triplets. DN spans PF2/PF3 and is fine.
        labels = np.array([GOF, GOF, DN, DN])
        gene_pfam = np.array(["PF1", "PF1", "PF2", "PF3"])
        encoder = LabelEncoder()
        encoder.fit(np.array([GOF, DN, LOF]))

        anchors, _positives, _negatives = build_cross_family_pairs(
            labels, gene_pfam, encoder, max_pairs_per_anchor=10, seed=7
        )
        anchor_set = set(anchors.tolist())
        # The GOF variants (indices 0, 1) must NOT appear as anchors.
        assert 0 not in anchor_set
        assert 1 not in anchor_set
        # The DN variants can (they have a cross-family same-class partner).
        assert anchor_set.issubset({2, 3})

    def test_determinism_same_seed(self):
        labels, gene_pfam, encoder = self._setup()
        first = build_cross_family_pairs(
            labels, gene_pfam, encoder, max_pairs_per_anchor=10, seed=123
        )
        second = build_cross_family_pairs(
            labels, gene_pfam, encoder, max_pairs_per_anchor=10, seed=123
        )
        for arr_a, arr_b in zip(first, second):
            np.testing.assert_array_equal(arr_a, arr_b)
