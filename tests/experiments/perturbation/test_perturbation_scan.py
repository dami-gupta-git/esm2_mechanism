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
    compute_scan_features,
)


def _probe(gene, aa_pos, probe_name, seq_len=200):
    return {
        "gene": gene,
        "uniprot_id": "P00001",
        "aa_pos": aa_pos,
        "aa_wt": "G",
        "aa_mut": probe_name[0].upper(),
        "probe_name": probe_name,
        "seq_len": seq_len,
    }


def _embeddings(n_probes, dim=32, seed=0):
    rng = np.random.RandomState(seed)
    wt = rng.randn(n_probes, dim).astype(np.float32)
    mut = wt + rng.randn(n_probes, dim).astype(np.float32)
    return wt, mut


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


class TestUnmeasurableScanFeaturesRefuse:
    """The probe sampler always gives a gene enough probes for the PCA and the
    per-position substitution variance. If a sampling change breaks that, the run
    must stop rather than record a fabricated zero."""

    def test_too_few_probes_for_pca_raises(self):
        probes = [_probe("TINY", pos, "ala") for pos in (10, 20, 30)]
        wt, mut = _embeddings(len(probes))

        with pytest.raises(ValueError, match="PC1/PC2 variance needs"):
            compute_scan_features(probes, wt, mut, ["TINY"])

    def test_no_position_with_two_probes_raises(self):
        # Enough probes for the PCA, but each sits at its own position, so there is
        # no within-position spread to measure.
        probes = [_probe("SPREAD", pos, "ala") for pos in (10, 20, 30, 40, 50)]
        wt, mut = _embeddings(len(probes))

        with pytest.raises(ValueError, match="substitution"):
            compute_scan_features(probes, wt, mut, ["SPREAD"])

    def test_well_sampled_gene_gives_finite_features(self):
        probes = [
            _probe("GOOD", pos, name)
            for pos in range(10, 110, 10)
            for name in ("ala", "asp", "trp")
        ]
        wt, mut = _embeddings(len(probes))

        gene_list, X, feature_names = compute_scan_features(probes, wt, mut, ["GOOD"])

        assert list(gene_list) == ["GOOD"]
        assert np.all(np.isfinite(X))
        assert X.shape == (1, len(feature_names))


class TestHotspotSpacingDropped:
    def test_spacing_feature_is_not_declared(self):
        probes = [
            _probe("GOOD", pos, name)
            for pos in range(10, 110, 10)
            for name in ("ala", "asp", "trp")
        ]
        wt, mut = _embeddings(len(probes))

        _genes, X, feature_names = compute_scan_features(
            probes, wt, mut, ["GOOD"], ablation=True
        )

        assert "scan_hotspot_spacing_cv" not in feature_names
        assert not any("spacing" in name for name in feature_names)
        assert X.shape[1] == len(feature_names)

    def test_gene_with_under_two_hotspots_is_kept(self):
        # Fewer than two hotspots is a property of the result, not of sample size,
        # so the gene stays in the cohort and simply has no spacing feature.
        n_pos = 10
        probes = [
            _probe("FLAT", pos, name)
            for pos in range(10, 10 * n_pos + 1, 10)
            for name in ("ala", "asp", "trp")
        ]
        rng = np.random.RandomState(3)
        wt = rng.randn(len(probes), 32).astype(np.float32)
        # Near-identical delta magnitudes, so no probe exceeds mean + one SD by much.
        mut = wt + np.ones((len(probes), 32), dtype=np.float32)
        mut += rng.randn(len(probes), 32).astype(np.float32) * 1e-3

        gene_list, X, feature_names = compute_scan_features(
            probes, wt, mut, ["FLAT"], ablation=True
        )

        assert list(gene_list) == ["FLAT"]
        assert np.all(np.isfinite(X))
