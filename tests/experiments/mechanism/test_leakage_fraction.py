"""
Tests for esm2_mech.experiments.mechanism.leakage_fraction.leakage_fraction_per_feature.

Invariants:
- a seed whose macro_f1_mean is None is filtered out, not crashed on (np.mean
  over a list containing None would raise / poison the average).
- leakage_fraction is None ("undefined") when the gene-split score is not
  meaningfully above chance, and when no seed is scorable.
- on clean above-chance inputs, leakage_fraction is the across-seed-mean ratio.
"""

import pytest

from esm2_mech.experiments.mechanism.leakage_fraction import (
    leakage_fraction_per_feature,
    MIN_ABOVE_CHANCE,
)


def _seed(gene_f1, family_f1, feature="delta_mean"):
    return {
        "gene_split": {feature: {"macro_f1_mean": gene_f1}},
        "family_split": {feature: {"macro_f1_mean": family_f1}},
    }


class TestLeakageFractionPerFeature:

    def test_none_seed_is_filtered_not_crashed(self):
        # One seed has no scorable gene-split fold (None) — it must be dropped,
        # and the result computed from the remaining seeds without raising.
        seeds = [
            _seed(0.60, 0.40),
            _seed(None, None),
            _seed(0.62, 0.42),
        ]
        result = leakage_fraction_per_feature(seeds, "delta_mean", chance=0.36)
        assert result["leakage_fraction"] is not None
        # gene mean over the two finite seeds = 0.61, family = 0.41
        assert result["gene_macro_f1_mean"] == pytest.approx(0.61)
        assert result["family_macro_f1_mean"] == pytest.approx(0.41)

    def test_no_scorable_seed_yields_undefined(self):
        seeds = [_seed(None, None), _seed(None, None)]
        result = leakage_fraction_per_feature(seeds, "delta_mean", chance=0.36)
        assert result["leakage_fraction"] is None

    def test_at_chance_yields_undefined(self):
        # Gene-split barely above chance (< MIN_ABOVE_CHANCE) → LF undefined.
        chance = 0.36
        gene = chance + MIN_ABOVE_CHANCE / 2
        seeds = [_seed(gene, gene)]
        result = leakage_fraction_per_feature(seeds, "delta_mean", chance=chance)
        assert result["leakage_fraction"] is None

    def test_above_chance_ratio(self):
        # gene=0.60, family=0.40, chance=0.30 → (0.60-0.40)/(0.60-0.30) = 0.6667
        seeds = [_seed(0.60, 0.40)]
        result = leakage_fraction_per_feature(seeds, "delta_mean", chance=0.30)
        assert result["leakage_fraction"] == pytest.approx(0.20 / 0.30)
