"""
Tests for esm2_mech.experiments.perturbation.perturbation_scan numerics.

Invariants:
- _top_explained_variance_ratios: matches sklearn PCA's explained_variance_ratio_
  (the torch-SVD path that replaced the per-gene sklearn PCA fit).
- _top_explained_variance_ratios: returns at most k ratios, each in [0, 1].
- _top_explained_variance_ratios: a rank-1 (single-direction) matrix puts ~all
  variance in PC1.
"""

import numpy as np
import pytest

from esm2_mech.experiments.perturbation.perturbation_scan import (
    _top_explained_variance_ratios,
)


class TestTopExplainedVarianceRatios:
    def test_matches_sklearn_pca(self):
        # The torch-SVD ratios must equal sklearn's explained_variance_ratio_ for the
        # same matrix. sklearn's default solver is `auto`, which selects the
        # *approximate* randomized SVD for small n_components and disagrees at the 3rd
        # decimal; the torch path computes the exact SVD, so compare against the exact
        # `full` solver to assert true equivalence rather than chase an approximation.
        from sklearn.decomposition import PCA

        rng = np.random.RandomState(0)
        # Tall matrix (n_samples > n_features) so the components are well-separated and
        # the comparison is not dominated by near-tied singular values.
        mat = rng.randn(200, 16).astype(np.float32)

        ratios = _top_explained_variance_ratios(mat, k=2)

        pca = PCA(n_components=2, svd_solver="full")
        pca.fit(mat)
        expected = pca.explained_variance_ratio_[:2]

        assert ratios.shape == (2,)
        np.testing.assert_allclose(ratios, expected, rtol=1e-4, atol=1e-5)

    def test_ratios_are_valid_fractions(self):
        rng = np.random.RandomState(1)
        mat = rng.randn(12, 50).astype(np.float32)
        ratios = _top_explained_variance_ratios(mat, k=2)
        assert len(ratios) <= 2
        assert np.all(ratios >= 0.0)
        assert np.all(ratios <= 1.0 + 1e-6)

    def test_rank_one_matrix_concentrates_in_pc1(self):
        # All rows are scalar multiples of one direction: PC1 should explain ~100%.
        rng = np.random.RandomState(2)
        direction = rng.randn(1280).astype(np.float32)
        scales = rng.randn(30, 1).astype(np.float32)
        mat = scales * direction  # rank-1 up to centering

        ratios = _top_explained_variance_ratios(mat, k=2)
        assert ratios[0] == pytest.approx(1.0, abs=1e-3)
