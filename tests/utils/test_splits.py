"""
Tests for esm2_mech.utils.splits.

Invariants:
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

from esm2_mech.utils.splits import gene_split_cv, family_split_cv, family_split_indices


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

    def test_each_sample_in_at_most_one_test_fold(self):
        genes = make_genes()
        counts = np.zeros(len(genes), dtype=int)
        for _, te in gene_split_cv(genes):
            counts[te] += 1
        assert counts.max() <= 1

    def test_min_size_respected(self):
        genes = make_genes()
        for tr, te in gene_split_cv(genes):
            assert len(tr) >= 10
            assert len(te) >= 5

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

    def test_single_gene_returns_empty(self):
        genes = np.array(["GENE0"] * 30)
        assert gene_split_cv(genes) == []

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

    def test_each_sample_in_at_most_one_test_fold(self):
        genes = make_genes()
        pfam = make_pfam_map(genes)
        counts = np.zeros(len(genes), dtype=int)
        for _, te in family_split_cv(genes, pfam):
            counts[te] += 1
        assert counts.max() <= 1

    def test_unannotated_genes_excluded(self):
        genes = make_genes(n_genes=10)
        pfam = make_pfam_map(genes)
        unannotated = list(set(genes))[:3]
        partial = {g: f for g, f in pfam.items() if g not in unannotated}
        for tr, te in family_split_cv(genes, partial):
            for idx in list(tr) + list(te):
                assert genes[idx] in partial

    def test_empty_pfam_returns_empty(self):
        genes = make_genes()
        assert family_split_cv(genes, {}) == []

    def test_deterministic(self):
        genes = make_genes()
        pfam = make_pfam_map(genes)
        s1 = family_split_cv(genes, pfam, seed=3)
        s2 = family_split_cv(genes, pfam, seed=3)
        for (tr1, te1), (tr2, te2) in zip(s1, s2):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(te1, te2)

    def test_min_size_respected(self):
        genes = make_genes()
        pfam = make_pfam_map(genes)
        for tr, te in family_split_cv(genes, pfam):
            assert len(tr) >= 10
            assert len(te) >= 5


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
        groups = self._make_groups()
        for tr, te in family_split_indices(groups, n_folds=5, seed=42):
            tr_fams = {groups[i] for i in tr if groups[i] is not None}
            te_fams = {groups[i] for i in te if groups[i] is not None}
            assert not (tr_fams & te_fams)

    def test_all_samples_covered(self):
        groups = self._make_groups()
        seen = set()
        for _, te in family_split_indices(groups, n_folds=5, seed=42):
            seen.update(te)
        assert seen == set(range(len(groups)))

    def test_none_group_samples_covered(self):
        groups = self._make_groups(none_frac=0.2)
        none_idx = {i for i, g in enumerate(groups) if g is None}
        covered = set()
        for _, te in family_split_indices(groups, n_folds=5, seed=42):
            covered.update(set(te) & none_idx)
        assert covered == none_idx
