"""
Tests for esm2_mech.experiments.perturbation.perturbation_pattern cohort filtering.

Invariants:
- A gene with one variant is excluded, not given zero magnitude spread, zero
  coefficient of variation, zero position spread, or zero PCA features.
- A gene with two variants is excluded, not given zero PCA features.
- A gene with three or more variants is kept and every scalar feature is a real
  number computed from its variants.
- The cohort record carries the gene counts and the class balance before and after.
- Feeding an under-sized gene straight into the feature builder raises rather
  than producing a row.
"""

import numpy as np
import pytest

from esm2_mech.experiments.perturbation import perturbation_pattern as pattern


def _variant(gene, aa_pos, label="LOF"):
    return {"gene": gene, "aa_pos": aa_pos, "label_3class": label}


def _make_inputs(variants, dim=6, seed=0):
    rng = np.random.RandomState(seed)
    n = len(variants)
    delta_pos = rng.randn(n, dim).astype(np.float32)
    delta_mean = rng.randn(n, dim).astype(np.float32)
    return delta_pos, delta_mean


class TestCohortFilter:
    def test_single_variant_gene_is_excluded(self):
        variants = [
            _variant("SOLO", 10, "GOF"),
            _variant("BIG", 5),
            _variant("BIG", 20),
            _variant("BIG", 60),
        ]
        delta_pos, delta_mean = _make_inputs(variants)

        gene_list, X, labels, n_scalar, cohort = pattern.build_gene_features(
            variants, delta_pos, delta_mean
        )

        assert "SOLO" not in set(gene_list)
        assert cohort["excluded_genes"] == ["SOLO"]
        assert cohort["n_genes_before"] == 2
        assert cohort["n_genes_excluded"] == 1
        assert cohort["n_genes_after"] == 1

    def test_two_variant_gene_is_excluded(self):
        variants = [
            _variant("PAIR", 4, "DN"),
            _variant("PAIR", 30, "DN"),
            _variant("BIG", 5),
            _variant("BIG", 20),
            _variant("BIG", 60),
        ]
        delta_pos, delta_mean = _make_inputs(variants)

        gene_list, X, labels, n_scalar, cohort = pattern.build_gene_features(
            variants, delta_pos, delta_mean
        )

        assert set(gene_list) == {"BIG"}
        assert cohort["excluded_genes"] == ["PAIR"]

    def test_three_variant_gene_has_all_features_measured(self):
        variants = [
            _variant("BIG", 5),
            _variant("BIG", 20),
            _variant("BIG", 60),
        ]
        delta_pos, delta_mean = _make_inputs(variants)

        gene_list, X, labels, n_scalar, cohort = pattern.build_gene_features(
            variants, delta_pos, delta_mean
        )

        assert list(gene_list) == ["BIG"]
        assert cohort["n_genes_excluded"] == 0
        scalars = X[0, :n_scalar]
        assert np.all(np.isfinite(scalars))
        # None of the five features that a smaller gene would have fabricated as
        # zero: magnitude spread, its CV, position spread, and the two PCA features.
        for index in (1, 2, 4, 5, 6):
            assert scalars[index] != 0.0

    def test_cohort_records_class_balance_before_and_after(self):
        variants = [
            _variant("SOLO", 10, "GOF"),
            _variant("PAIR", 4, "DN"),
            _variant("PAIR", 30, "DN"),
            _variant("BIG", 5, "LOF"),
            _variant("BIG", 20, "LOF"),
            _variant("BIG", 60, "LOF"),
        ]
        delta_pos, delta_mean = _make_inputs(variants)

        _genes, _X, _labels, _n_scalar, cohort = pattern.build_gene_features(
            variants, delta_pos, delta_mean
        )

        assert cohort["class_balance_before"] == {"GOF": 1, "DN": 1, "LOF": 1}
        assert cohort["class_balance_after"] == {"LOF": 1}
        assert cohort["class_balance_excluded"] == {"GOF": 1, "DN": 1}
        assert cohort["min_variants_per_gene"] == pattern.MIN_VARIANTS_PER_GENE


class TestUnderSizedGeneRaises:
    @pytest.mark.parametrize("n_variants", [1, 2])
    def test_feature_loop_refuses_under_sized_gene(self, monkeypatch, n_variants):
        # Lower the cohort threshold so an under-sized gene reaches the feature loop.
        # The loop must refuse it rather than emit zeros for the features that a gene
        # this small cannot support.
        monkeypatch.setattr(pattern, "MIN_VARIANTS_PER_GENE", 1)
        variants = [_variant("SMALL", 10 * (i + 1)) for i in range(n_variants)]
        delta_pos, delta_mean = _make_inputs(variants)

        with pytest.raises(ValueError, match="not measurable"):
            pattern.build_gene_features(variants, delta_pos, delta_mean)
