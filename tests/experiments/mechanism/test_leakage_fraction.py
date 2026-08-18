"""
Tests for esm2_mech.experiments.mechanism.leakage_fraction.

Invariants for leakage_fraction_per_feature:
- a seed whose macro_f1_mean is None is filtered out, not crashed on (np.mean
  over a list containing None would raise / poison the average).
- leakage_fraction is None ("undefined") when the gene-split score is not
  meaningfully above chance, and when no seed is scorable.
- on clean above-chance inputs, leakage_fraction is the across-seed-mean ratio.
- prefers macro_f1_pooled over macro_f1_mean when both are present.

Invariants for leakage_fraction_ci:
- the chance floor is recomputed from the resampled gene-split labels on each
  bootstrap replicate, not held fixed. This matters because resampling families
  shifts class proportions.
"""

import numpy as np
import pytest

from esm2_mech.experiments.mechanism.leakage_fraction import (
    leakage_fraction_ci,
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

    def test_prefers_pooled_over_fold_mean(self):
        seeds = [{
            "gene_split": {"delta_mean": {"macro_f1_mean": 0.50, "macro_f1_pooled": 0.55}},
            "family_split": {"delta_mean": {"macro_f1_mean": 0.40, "macro_f1_pooled": 0.42}},
        }]
        result = leakage_fraction_per_feature(seeds, "delta_mean", chance=0.30)
        assert result["gene_macro_f1_mean"] == pytest.approx(0.55)
        assert result["family_macro_f1_mean"] == pytest.approx(0.42)

    def test_falls_back_to_fold_mean_when_no_pooled(self):
        seeds = [_seed(0.60, 0.40)]
        result = leakage_fraction_per_feature(seeds, "delta_mean", chance=0.30)
        assert result["gene_macro_f1_mean"] == pytest.approx(0.60)


class TestLeakageFractionCi:

    @staticmethod
    def _make_oof_cache_entry(n=60, seed=42):
        """Synthetic OOF cache with gene-split predicting well and family-split
        predicting worse, so the leakage fraction is positive."""
        from esm2_mech.utils.constants import MECHANISM_CLASSES
        rng = np.random.RandomState(seed)
        classes = MECHANISM_CLASSES
        y_true = rng.choice(classes, size=n)
        row_ids = list(range(n))
        genes = [f"gene_{i % 6}" for i in range(n)]

        gene_pred = y_true.copy()
        gene_pred[n // 2:] = rng.choice(classes, size=n - n // 2)

        family_pred = rng.choice(classes, size=n)

        return {
            "gene_split": {
                "row_ids": row_ids,
                "y_true": y_true.tolist(),
                "pred": gene_pred.tolist(),
                "genes": genes,
            },
            "family_split": {
                "row_ids": row_ids,
                "y_true": y_true.tolist(),
                "pred": family_pred.tolist(),
                "genes": genes,
            },
        }

    def test_ci_runs_and_returns_result(self):
        entry = self._make_oof_cache_entry()
        pfam_map = {f"gene_{i}": f"PF{i % 3}" for i in range(6)}
        ci = leakage_fraction_ci(entry, pfam_map, n_resamples=50, seed=0)
        assert ci is not None
        assert "point" in ci

    def test_chance_recomputed_per_resample(self):
        """Verify the chance floor varies across resamples by checking that
        the CI is different from what a fixed-chance version would give."""
        from sklearn.metrics import f1_score
        from esm2_mech.utils.bootstrap import cluster_bootstrap_ci

        rng = np.random.RandomState(7)
        n = 80
        classes = ["GOF", "DN", "LOF"]
        y_true = np.array(["LOF"] * 50 + ["GOF"] * 20 + ["DN"] * 10)
        row_ids = list(range(n))
        genes = [f"gene_{i % 8}" for i in range(n)]

        gene_pred = y_true.copy()
        family_pred = y_true.copy()
        family_pred[40:] = rng.choice(classes, size=40)

        entry = {
            "gene_split": {
                "row_ids": row_ids,
                "y_true": y_true.tolist(),
                "pred": gene_pred.tolist(),
                "genes": genes,
            },
            "family_split": {
                "row_ids": row_ids,
                "y_true": y_true.tolist(),
                "pred": family_pred.tolist(),
                "genes": genes,
            },
        }
        pfam_map = {f"gene_{i}": f"PF{i % 4}" for i in range(8)}

        ci_resampled = leakage_fraction_ci(entry, pfam_map, n_resamples=200, seed=0)

        gene_y = np.array(entry["gene_split"]["y_true"])
        gene_p = np.array(entry["gene_split"]["pred"])
        family_y = np.array(entry["family_split"]["y_true"])
        family_p = np.array(entry["family_split"]["pred"])
        gene_positions = np.arange(n)
        family_positions = np.arange(n)
        clusters = np.array([pfam_map.get(g, g) for g in genes])

        from esm2_mech.utils.metrics import majority_baseline_f1
        fixed_chance, _ = majority_baseline_f1(gene_y, gene_y)

        def _ratio_fixed(rows):
            gf1 = float(f1_score(gene_y[rows], gene_p[rows], average="macro", zero_division=0))
            ff1 = float(f1_score(family_y[rows], family_p[rows], average="macro", zero_division=0))
            denom = gf1 - fixed_chance
            if denom <= MIN_ABOVE_CHANCE:
                return None
            return (gf1 - ff1) / denom

        ci_fixed = cluster_bootstrap_ci(clusters, _ratio_fixed, n_resamples=200, seed=0)

        assert ci_resampled is not None
        assert ci_fixed is not None
        if ci_resampled.get("ci_low") is not None and ci_fixed.get("ci_low") is not None:
            assert ci_resampled["ci_low"] != pytest.approx(ci_fixed["ci_low"], abs=1e-6)
