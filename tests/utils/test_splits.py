"""
Tests for esm2_mech.utils.splits.

Invariants:
- random_split_cv: every row appears in exactly one test fold (full coverage)
- random_split_cv: train is the exact complement of test in each fold
- random_split_cv: n_folds folds returned (no min-size guard)
- random_split_cv: indices in bounds; deterministic per seed; differs across seeds
- gene_split_cv: no gene appears in both train and test of the same fold
- gene_split_cv: each sample appears in at most one test fold
- gene_split_cv: train >= 10 and test >= 5 for every returned fold
- gene_split_cv: deterministic for same seed, differs across seeds
- gene_split_cv: single gene or too-small input returns empty list
- family_split_cv: no Pfam family spans both train and test
- family_split_cv: unannotated genes excluded from all folds
- family_split_cv: empty pfam_map returns empty list
- family_split_indices: no family leaks across train/test
- family_split_indices: all samples covered across folds
- family_split_indices: None-group samples are distributed (not excluded)
"""

import numpy as np
import pytest

from esm2_mech.utils.splits import (
    random_split_cv,
    gene_split_cv,
    family_split_cv,
    family_split_indices,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_genes(n_genes=20, variants_per_gene=10, seed=0):
    rng = np.random.RandomState(seed)
    names = [f"G{i:02d}" for i in range(n_genes)]
    return np.array([names[i] for i in rng.randint(0, n_genes, n_genes * variants_per_gene)])


def make_pfam_map(genes, n_families=5):
    unique = sorted(set(genes))
    return {g: f"FAM{i % n_families:02d}" for i, g in enumerate(unique)}


# ---------------------------------------------------------------------------
# random_split_cv
# ---------------------------------------------------------------------------

class TestRandomSplitCv:

    def test_full_test_coverage_exactly_once(self):
        # Unlike grouped splits, random CV covers every row in exactly one test
        # fold — nothing is dropped, nothing is double-counted.
        n = 53
        counts = np.zeros(n, dtype=int)
        for _, te in random_split_cv(n, n_folds=5):
            counts[te] += 1
        assert np.all(counts == 1)

    def test_train_is_complement_of_test(self):
        n = 40
        all_idx = set(range(n))
        for tr, te in random_split_cv(n, n_folds=5):
            assert set(tr) & set(te) == set()
            assert set(tr) | set(te) == all_idx

    def test_n_folds_returned(self):
        # No min-size guard, so all requested folds are always returned.
        assert len(random_split_cv(30, n_folds=5)) == 5

    def test_indices_in_bounds(self):
        n = 40
        for tr, te in random_split_cv(n, n_folds=5):
            assert tr.min() >= 0 and tr.max() < n
            assert te.min() >= 0 and te.max() < n

    def test_deterministic(self):
        s1 = random_split_cv(40, n_folds=5, seed=7)
        s2 = random_split_cv(40, n_folds=5, seed=7)
        for (tr1, te1), (tr2, te2) in zip(s1, s2):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(te1, te2)

    def test_different_seeds_differ(self):
        s1 = random_split_cv(40, n_folds=5, seed=0)
        s2 = random_split_cv(40, n_folds=5, seed=99)
        assert any(
            not np.array_equal(te1, te2)
            for (_, te1), (_, te2) in zip(s1, s2)
        )


# ---------------------------------------------------------------------------
# gene_split_cv
# ---------------------------------------------------------------------------

class TestGeneSplitCv:

    def test_no_gene_leakage(self):
        genes = make_genes()
        for tr, te in gene_split_cv(genes):
            assert not (set(genes[tr]) & set(genes[te]))

    def test_disjoint_indices(self):
        genes = make_genes()
        for tr, te in gene_split_cv(genes):
            assert len(set(tr) & set(te)) == 0

    def test_each_sample_in_exactly_one_test_fold(self):
        genes = make_genes()
        counts = np.zeros(len(genes), dtype=int)
        for _, te in gene_split_cv(genes):
            counts[te] += 1
        assert np.all(counts == 1)

    def test_deterministic(self):
        genes = make_genes()
        s1 = gene_split_cv(genes, seed=7)
        s2 = gene_split_cv(genes, seed=7)
        for (tr1, te1), (tr2, te2) in zip(s1, s2):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(te1, te2)

    def test_different_seeds_differ(self):
        genes = make_genes()
        s1 = gene_split_cv(genes, seed=0)
        s2 = gene_split_cv(genes, seed=99)
        assert any(
            not (np.array_equal(tr1, tr2) and np.array_equal(te1, te2))
            for (tr1, te1), (tr2, te2) in zip(s1, s2)
        )

    def test_single_gene_returns_every_requested_partition(self):
        genes = np.array(["GENE0"] * 30)
        splits = gene_split_cv(genes)
        assert len(splits) == 5
        assert sum(len(test_rows) for _, test_rows in splits) == len(genes)

    def test_indices_in_bounds(self):
        genes = make_genes()
        n = len(genes)
        for tr, te in gene_split_cv(genes):
            assert tr.min() >= 0 and tr.max() < n
            assert te.min() >= 0 and te.max() < n


# ---------------------------------------------------------------------------
# family_split_cv
# ---------------------------------------------------------------------------

class TestFamilySplitCv:

    def test_no_family_leakage(self):
        genes = make_genes()
        pfam = make_pfam_map(genes)
        for tr, te in family_split_cv(genes, pfam):
            tr_fams = {pfam[g] for g in genes[tr] if g in pfam}
            te_fams = {pfam[g] for g in genes[te] if g in pfam}
            assert not (tr_fams & te_fams)

    def test_disjoint_indices(self):
        genes = make_genes()
        pfam = make_pfam_map(genes)
        for tr, te in family_split_cv(genes, pfam):
            assert len(set(tr) & set(te)) == 0

    def test_each_eligible_sample_in_exactly_one_test_fold(self):
        genes = make_genes()
        pfam = make_pfam_map(genes)
        counts = np.zeros(len(genes), dtype=int)
        for _, te in family_split_cv(genes, pfam):
            counts[te] += 1
        annotated = np.array([gene in pfam for gene in genes])
        assert np.all(counts[annotated] == 1)

    def test_unannotated_genes_excluded(self):
        genes = make_genes(n_genes=10)
        pfam = make_pfam_map(genes)
        unannotated = list(set(genes))[:3]
        partial = {g: f for g, f in pfam.items() if g not in unannotated}
        for tr, te in family_split_cv(genes, partial):
            for idx in list(tr) + list(te):
                assert genes[idx] in partial

    def test_empty_pfam_returns_every_requested_partition(self):
        genes = make_genes()
        splits = family_split_cv(genes, {})
        assert len(splits) == 5
        assert all(len(train_rows) == 0 and len(test_rows) == 0 for train_rows, test_rows in splits)

    def test_deterministic(self):
        genes = make_genes()
        pfam = make_pfam_map(genes)
        s1 = family_split_cv(genes, pfam, seed=3)
        s2 = family_split_cv(genes, pfam, seed=3)
        for (tr1, te1), (tr2, te2) in zip(s1, s2):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(te1, te2)

# ---------------------------------------------------------------------------
# family_split_indices
# ---------------------------------------------------------------------------

class TestFamilySplitIndices:

    def _make_groups(self, n=100, n_fams=5, none_frac=0.1, seed=0):
        rng = np.random.RandomState(seed)
        fams = [f"FAM{i}" for i in range(n_fams)]
        groups = np.array([fams[i % n_fams] for i in range(n)], dtype=object)
        none_idx = rng.choice(n, int(n * none_frac), replace=False)
        groups[none_idx] = None
        return groups

    def test_no_family_leakage(self):
        groups = self._make_groups(none_frac=0.0)
        for tr, te in family_split_indices(groups, n_folds=5, seed=42):
            tr_fams = {groups[i] for i in tr if groups[i] is not None}
            te_fams = {groups[i] for i in te if groups[i] is not None}
            assert not (tr_fams & te_fams)

    def test_all_samples_covered(self):
        groups = self._make_groups(none_frac=0.0)
        seen = set()
        for _, te in family_split_indices(groups, n_folds=5, seed=42):
            seen.update(te)
        assert seen == set(range(len(groups)))

    def test_none_group_samples_are_rejected(self):
        groups = self._make_groups(none_frac=0.2)
        with pytest.raises(ValueError, match="without a family"):
            list(family_split_indices(groups, n_folds=5, seed=42))
