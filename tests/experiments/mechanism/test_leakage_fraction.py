"""
Tests for esm2_mech.experiments.mechanism.leakage_fraction.

Invariants for leakage_fraction_per_feature:
- OOF caches are mandatory; the function never falls back to scores on different rows.
- both arms are rescored within folds on their shared rows.
- leakage_fraction is undefined when the aligned gene score is at the aligned floor.

Invariants for leakage_fraction_ci:
- computed on the same basis as the headline: every seed the headline averages
  (not one seed alone), fold-averaged macro-F1 (not pooled), and the chance floor
  recomputed on the aligned feature rows and then held fixed across resamples.
- an at-floor point estimate returns None before resampling, rather than turning
  an undefined ratio into a misleading high-discard bootstrap warning.
"""

import numpy as np
import pytest

from esm2_mech.experiments.mechanism.leakage_fraction import (
    leakage_fraction_ci,
    leakage_fraction_per_feature,
    MIN_ABOVE_CHANCE,
)


class TestLeakageFractionPerFeature:

    def test_missing_cache_is_an_error(self):
        with pytest.raises(ValueError, match="OOF caches are required"):
            leakage_fraction_per_feature(
                "delta_mean", chance=0.30
            )

    def test_headline_is_rescored_from_aligned_oof_rows(self):
        entry = TestLeakageFractionCi._make_oof_cache_entry(seed=4)
        result = leakage_fraction_per_feature(
            "delta_mean",
            chance=0.20,
            oof_cache_entries=[entry],
        )
        assert result["gene_macro_f1_mean"] != pytest.approx(0.99)
        assert result["family_macro_f1_mean"] != pytest.approx(0.01)
        assert result["chance_macro_f1"] == pytest.approx(0.20)


class TestLeakageFractionCi:

    @staticmethod
    def _make_oof_cache_entry(n=60, seed=42, n_folds=3):
        """Synthetic per-seed OOF cache: gene-split predicts well, family-split
        predicts worse, so the leakage fraction is positive. n rows are split evenly
        across n_folds fold ids, as the real fold-aware cache does."""
        from esm2_mech.utils.constants import MECHANISM_CLASSES
        rng = np.random.RandomState(seed)
        classes = MECHANISM_CLASSES
        if n % n_folds != 0:
            raise ValueError("test fixture requires equal fold sizes")
        rows_per_fold = n // n_folds
        fold_labels = np.resize(classes, rows_per_fold)
        y_true = np.tile(fold_labels, n_folds)
        row_ids = list(range(n))
        genes = [f"gene_{i % 6}" for i in range(n)]
        folds = np.repeat(np.arange(n_folds), rows_per_fold).tolist()

        gene_pred = y_true.copy()
        gene_pred[n // 2:] = rng.choice(classes, size=n - n // 2)

        family_pred = rng.choice(classes, size=n)

        return {
            "gene_split": {
                "row_ids": row_ids,
                "y_true": y_true.tolist(),
                "pred": gene_pred.tolist(),
                "genes": genes,
                "folds": folds,
            },
            "family_split": {
                "row_ids": row_ids,
                "y_true": y_true.tolist(),
                "pred": family_pred.tolist(),
                "genes": genes,
                "folds": folds,
            },
        }

    def test_ci_runs_and_returns_result(self):
        entries = [self._make_oof_cache_entry(seed=s) for s in range(3)]
        pfam_map = {f"gene_{i}": f"PF{i % 3}" for i in range(6)}
        ci = leakage_fraction_ci(entries, pfam_map, chance=0.28, n_resamples=50, seed=0)
        assert ci is not None
        assert "point" in ci

    def test_at_floor_point_does_not_start_bootstrap(self, monkeypatch):
        """An undefined leakage fraction is not a failed bootstrap interval."""
        from esm2_mech.experiments.mechanism import leakage_fraction as module
        from esm2_mech.utils.constants import LOF, MECHANISM_CLASSES

        n_folds = 3
        rows_per_fold = 6
        n_rows = n_folds * rows_per_fold
        y_true = np.tile(MECHANISM_CLASSES, n_rows // len(MECHANISM_CLASSES))
        predictions = np.full(n_rows, LOF, dtype=object)
        folds = np.repeat(np.arange(n_folds), rows_per_fold)
        row_ids = list(range(n_rows))
        genes = [f"gene_{row}" for row in row_ids]
        arm = {
            "row_ids": row_ids,
            "y_true": y_true.tolist(),
            "pred": predictions.tolist(),
            "genes": genes,
            "folds": folds.tolist(),
        }
        entries = [{"gene_split": arm, "family_split": arm}]
        pfam_map = {gene: f"PF_{gene}" for gene in genes}

        def unexpected_bootstrap(*args, **kwargs):
            raise AssertionError(
                "an at-floor leakage fraction must return before bootstrap resampling"
            )

        monkeypatch.setattr(module, "cluster_bootstrap_ci", unexpected_bootstrap)

        # In each balanced fold, always predicting LOF gives macro-F1 = 1/6.
        ci = module.leakage_fraction_ci(
            entries,
            pfam_map,
            chance=1.0 / 6.0,
            n_resamples=1000,
            seed=0,
        )

        assert ci is None

    def test_missing_seed_variants_are_excluded_from_the_shared_row_space(self):
        # Seed 1's cache only scored a subset of the rows seed 0 and seed 2 scored.
        # The CI must restrict to the rows every seed scored, not error or silently
        # use seed-specific rows that would misalign the fold-average across seeds.
        full = [self._make_oof_cache_entry(seed=s) for s in (0, 2)]
        partial = self._make_oof_cache_entry(seed=1)
        for arm in ("gene_split", "family_split"):
            for key in ("row_ids", "y_true", "pred", "genes", "folds"):
                partial[arm][key] = partial[arm][key][:40]
        entries = [full[0], partial, full[1]]
        pfam_map = {f"gene_{i}": f"PF{i % 3}" for i in range(6)}
        ci = leakage_fraction_ci(entries, pfam_map, chance=0.28, n_resamples=50, seed=0)
        assert ci is not None

    def test_matches_a_hand_computed_fixed_basis_ci(self):
        # The interval must be the same quantity as the headline: fold-averaged
        # macro-F1, averaged over these seeds, divided by the distance from a chance
        # floor that is fixed (not recomputed per resample).
        from sklearn.metrics import f1_score
        from esm2_mech.utils.bootstrap import cluster_bootstrap_ci

        entries = [self._make_oof_cache_entry(seed=s, n_folds=4) for s in range(2)]
        pfam_map = {f"gene_{i}": f"PF{i % 3}" for i in range(6)}
        chance = 0.25

        ci = leakage_fraction_ci(entries, pfam_map, chance=chance, n_resamples=100, seed=0)

        def _fold_mean_f1(y_true, pred, folds, rows):
            values = []
            row_folds = np.asarray(folds)[rows]
            for fold in np.unique(row_folds):
                block = rows[row_folds == fold]
                if len(block) == 0:
                    return None
                values.append(
                    f1_score(y_true[block], pred[block], average="macro", zero_division=0)
                )
            return float(np.mean(values))

        genes = np.array(entries[0]["gene_split"]["genes"])
        clusters = np.array([pfam_map.get(g) or f"__orphan__{g}" for g in genes])

        def _ratio_fixed(rows):
            gene_vals, family_vals = [], []
            for entry in entries:
                gy = np.array(entry["gene_split"]["y_true"])
                gp = np.array(entry["gene_split"]["pred"])
                gf = np.array(entry["gene_split"]["folds"])
                fy = np.array(entry["family_split"]["y_true"])
                fp = np.array(entry["family_split"]["pred"])
                ff = np.array(entry["family_split"]["folds"])
                g_val = _fold_mean_f1(gy, gp, gf, rows)
                f_val = _fold_mean_f1(fy, fp, ff, rows)
                if g_val is None or f_val is None:
                    return None
                gene_vals.append(g_val)
                family_vals.append(f_val)
            gene_mean = float(np.mean(gene_vals))
            family_mean = float(np.mean(family_vals))
            denom = gene_mean - chance
            if denom <= MIN_ABOVE_CHANCE:
                return None
            return (gene_mean - family_mean) / denom

        ci_hand = cluster_bootstrap_ci(clusters, _ratio_fixed, n_resamples=100, seed=0)

        assert ci is not None
        assert ci["point"] == pytest.approx(ci_hand["point"], abs=1e-9)

    def test_discard_reason_counts_survive_multiprocess_workers(self, monkeypatch):
        """discard_reason_counts is a dict closed over by `_ratio` and mutated inside
        it. joblib's default backend runs each resample's `_ratio` call in a worker
        *process*, so with more than one job every worker gets its own pickled copy
        of that dict — a mutation there never reaches the parent's copy. If this
        regresses, discard_reason_counts (and the denominator-collapse suppression
        gate that reads it) silently reports zero discards on real multi-core runs,
        which use the n_jobs=-1 default.

        Forces n_jobs=2 so the bug is exercised even on a machine that would
        otherwise run everything in one process.
        """
        import functools
        from esm2_mech.experiments.mechanism import leakage_fraction as module
        from esm2_mech.utils import bootstrap as bootstrap_module
        from esm2_mech.utils.constants import GOF, DN, LOF

        monkeypatch.setattr(
            module,
            "cluster_bootstrap_ci",
            functools.partial(bootstrap_module.cluster_bootstrap_ci, n_jobs=2),
        )

        # Two genes, each supplying all three classes to its own fold. Resampling
        # the 2 gene-clusters with replacement empties one fold on exactly the
        # "AA" and "BB" draws (half the time), which discards for "fold_lost_class".
        y_true = [GOF, DN, LOF, GOF, DN, LOF]
        genes = ["gene_A", "gene_A", "gene_A", "gene_B", "gene_B", "gene_B"]
        folds = [0, 0, 0, 1, 1, 1]
        row_ids = list(range(6))
        gene_arm = {
            "row_ids": row_ids, "y_true": y_true, "pred": y_true,
            "genes": genes, "folds": folds,
        }
        family_arm = {
            "row_ids": row_ids, "y_true": y_true,
            "pred": [DN, LOF, GOF, DN, LOF, GOF],  # wrong but still all 3 classes
            "genes": genes, "folds": folds,
        }
        entries = [{"gene_split": gene_arm, "family_split": family_arm}]
        pfam_map = {}  # both genes fall back to their own orphan cluster

        ci = module.leakage_fraction_ci(
            entries, pfam_map, chance=0.1, n_resamples=400, seed=0,
        )

        assert ci is not None
        n_discarded = ci["n_resamples_total"] - ci["n_resamples"]
        assert n_discarded > 0  # the AA/BB draws must actually happen
        assert sum(ci["discard_reason_counts"].values()) == n_discarded
