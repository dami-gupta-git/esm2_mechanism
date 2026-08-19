"""
Tests for esm2_mech.utils.bootstrap (dependency-aware inference).

Invariants:
- average_oof_over_seeds: de-duplicates variants across seeds (one row per variant)
- average_oof_over_seeds: averages proba across seeds; rows stay simplex (sum to 1)
- average_oof_over_seeds: skips None entries; all-None / empty -> None
- average_oof_over_seeds: y_true and gene are carried through per row
- cluster_bootstrap_ci: point matches metric_fn on all rows; CI brackets the point
- cluster_bootstrap_ci: resamples whole clusters (n_clusters == #unique), not rows
- cluster_bootstrap_ci: a few undefined resamples are dropped, not imputed
- cluster_bootstrap_ci: too many undefined resamples -> CI suppressed (not thinned)
- cluster_bootstrap_ci: all-undefined metric -> ci_low/ci_high None
- bootstrap_mechanism_metrics: returns macro_f1 + one CI per class; recovers GOF signal
- paired_cluster_bootstrap_diff: shares one resample across both arms (same drawn
  rows handed to both metric_fn_a and metric_fn_b on every replicate)
- paired_cluster_bootstrap_diff: point_diff == point_a - point_b over all rows
- paired_cluster_bootstrap_diff: planted non-zero difference -> CI excludes zero
- paired_cluster_bootstrap_diff: no true difference -> CI spans zero
- paired_cluster_bootstrap_diff: an arm undefined on a resample drops that
  replicate; too many undefined -> ci_suppressed
- paired_cluster_bootstrap_diff_shared_clusters: one shared cluster draw selects
  each task's own rows when the row spaces differ
- paired_cluster_bootstrap_diff_cross_partition: resamples the given (coarser)
  unit, not the two arms' own fold structure
- paired_cluster_bootstrap_diff_cross_partition: still shares one resample across
  both arms, each scored under its own partition on the identical drawn rows
- paired_cluster_bootstrap_diff_cross_partition: sensitivity_clusters resamples at
  its own (finer) granularity, reported separately under gene_resampled_sensitivity
- label_permutation_pvalue: null centers on chance; planted signal -> small p
- label_permutation_pvalue: gene-level shuffle keeps one label per gene
- oof_permutation_pvalue: null centers on 0.5 (macro AUROC); planted signal -> small p
- oof_permutation_pvalue: uninformative predictions -> p far from significance
- oof_permutation_pvalue: each shuffle unit widens the null relative to the finer one
  below it (family blocks > genes > variants); the finer unit is anticonservative
- _permute_labels_by_cluster: swaps whole clusters' label blocks between same-size
  clusters, preserving the label multiset and each cluster's internal label mixing
- _permute_labels_by_cluster: a cluster with a unique size cannot move; counted
- oof_permutation_pvalue: refits nothing — predictions are never recomputed
- oof_permutation_pvalue: a draw scoring a different class set is dropped, not
  averaged into the null, and the drop is counted
- score_within_folds (via macro_ovr_auroc, bootstrap_mechanism_metrics and
  paired_oof_diff): folds whose probability scales do not compare are scored
  separately and averaged; ranking the concatenation instead drops a pair of
  perfectly ranked folds to 0.36, below chance
- macro_ovr_auroc: drops classes absent from y_true and reports which it scored;
  all-absent -> (None, ())
- bootstrap_mechanism_metrics: emits prevalence and AUPRC-minus-prevalence intervals
  beside each AUPRC, so the lift is read against a resampled baseline
- _permute_labels: with groups, every gene's rows share one (permuted) label
"""

import numpy as np
import pytest

from esm2_mech.utils.bootstrap import (
    adjudicate_diff,
    adjudicate_equivalence,
    adjudicate_level,
    attach_mechanism_ci,
    average_oof_over_seeds,
    binary_auroc_cluster_bootstrap_ci,
    bootstrap_mechanism_metrics,
    cluster_bootstrap_ci,
    cluster_bootstrap_ci_multi,
    cluster_subsample_ci,
    independent_cluster_bootstrap_diff,
    count_immovable_clusters,
    label_permutation_pvalue,
    macro_ovr_auroc,
    oof_permutation_pvalue,
    paired_cluster_bootstrap_diff,
    paired_cluster_bootstrap_diff_shared_clusters,
    paired_cluster_bootstrap_diff_cross_partition,
    paired_oof_diff,
    stack_oof_over_seeds,
    within_stratum_bootstrap_ci,
    _permute_labels,
    _permute_labels_by_cluster,
    _DEFAULT_DISCARD_REASON,
)
from esm2_mech.utils.constants import MECHANISM_CLASSES, GOF, DN, LOF


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _simplex_proba(rng, n, n_classes=3):
    """n rows of random probabilities that sum to 1 across columns."""
    raw = rng.rand(n, n_classes)
    return raw / raw.sum(axis=1, keepdims=True)


def _oof_for_seed(row_ids, y_true, genes, rng):
    """Build one seed's OOF dict over the given row ids, in a shuffled fold order."""
    order = rng.permutation(len(row_ids))
    rid = np.asarray(row_ids)[order]
    return {
        "row_ids": rid,
        "y_true": np.asarray(y_true)[order],
        "genes": np.asarray(genes)[order],
        "proba": _simplex_proba(rng, len(row_ids)),
    }


# ---------------------------------------------------------------------------
# average_oof_over_seeds
# ---------------------------------------------------------------------------

class TestAverageOofOverSeeds:
    def _setup(self, n_variants=12, n_seeds=5, seed=0):
        rng = np.random.RandomState(seed)
        row_ids = np.arange(n_variants)
        genes = np.array([f"G{r % 4}" for r in row_ids])
        y_true = np.array([[GOF, DN, LOF][r % 3] for r in row_ids])
        oofs = [_oof_for_seed(row_ids, y_true, genes, rng) for _ in range(n_seeds)]
        return oofs, row_ids, y_true, genes

    def test_dedups_to_one_row_per_variant(self):
        oofs, row_ids, _, _ = self._setup()
        avg = average_oof_over_seeds(oofs)
        # Each variant appears once per seed in OOF; averaging collapses to one row.
        assert len(avg["row_ids"]) == len(row_ids)
        assert sorted(avg["row_ids"].tolist()) == sorted(row_ids.tolist())
        assert len(set(avg["row_ids"].tolist())) == len(row_ids)

    def test_rows_sorted_by_row_id(self):
        oofs, _, _, _ = self._setup()
        avg = average_oof_over_seeds(oofs)
        assert list(avg["row_ids"]) == sorted(avg["row_ids"])

    def test_proba_is_average_and_simplex(self):
        oofs, row_ids, _, _ = self._setup()
        avg = average_oof_over_seeds(oofs)
        assert np.allclose(avg["proba"].sum(axis=1), 1.0)
        # Spot-check: avg proba for a row equals the mean of that row across seeds.
        target = row_ids[0]
        per_seed_vecs = []
        for oof in oofs:
            pos = int(np.where(oof["row_ids"] == target)[0][0])
            per_seed_vecs.append(oof["proba"][pos])
        out_pos = int(np.where(avg["row_ids"] == target)[0][0])
        assert np.allclose(avg["proba"][out_pos], np.mean(per_seed_vecs, axis=0))

    def test_y_and_gene_carried_per_row(self):
        oofs, row_ids, y_true, genes = self._setup()
        avg = average_oof_over_seeds(oofs)
        for pos, row in enumerate(avg["row_ids"]):
            assert avg["y_true"][pos] == y_true[row]
            assert avg["genes"][pos] == genes[row]

    def test_skips_none_entries(self):
        oofs, row_ids, _, _ = self._setup()
        with_none = [oofs[0], None, oofs[1], None]
        avg = average_oof_over_seeds(with_none)
        assert len(avg["row_ids"]) == len(row_ids)

    def test_all_none_returns_none(self):
        assert average_oof_over_seeds([None, None]) is None

    def test_empty_list_returns_none(self):
        assert average_oof_over_seeds([]) is None

    def test_partial_coverage_across_seeds(self):
        # A variant scored in only some seeds is still emitted, averaged over the
        # seeds that covered it (count differs per row).
        rng = np.random.RandomState(1)
        genes = np.array(["G0", "G0", "G1", "G1"])
        y_true = np.array([GOF, DN, LOF, GOF])
        full = _oof_for_seed([0, 1, 2, 3], y_true, genes, rng)
        partial = _oof_for_seed([0, 2], y_true[[0, 2]], genes[[0, 2]], rng)
        avg = average_oof_over_seeds([full, partial])
        assert sorted(avg["row_ids"].tolist()) == [0, 1, 2, 3]
        # row 1 seen once (only `full`); equals full's vector for that row.
        pos_full = int(np.where(full["row_ids"] == 1)[0][0])
        out_pos = int(np.where(avg["row_ids"] == 1)[0][0])
        assert np.allclose(avg["proba"][out_pos], full["proba"][pos_full])


# ---------------------------------------------------------------------------
# cluster_bootstrap_ci
# ---------------------------------------------------------------------------

class TestClusterBootstrapCI:
    def test_point_matches_metric_on_all_rows(self):
        clusters = np.array([f"G{i % 5}" for i in range(50)])
        values = np.arange(50, dtype=float)
        out = cluster_bootstrap_ci(
            clusters, lambda rows: float(values[rows].mean()), n_resamples=200
        )
        assert out["point"] == pytest.approx(values.mean())

    def test_ci_brackets_point(self):
        clusters = np.array([f"G{i % 5}" for i in range(50)])
        values = np.arange(50, dtype=float)
        out = cluster_bootstrap_ci(
            clusters, lambda rows: float(values[rows].mean()), n_resamples=300
        )
        assert out["ci_low"] <= out["point"] <= out["ci_high"]

    def test_n_clusters_is_unique_cluster_count(self):
        clusters = np.array([f"G{i % 7}" for i in range(40)])
        out = cluster_bootstrap_ci(clusters, lambda rows: float(len(rows)), n_resamples=10)
        assert out["n_clusters"] == 7

    def test_resamples_clusters_not_rows(self):
        # If whole clusters are resampled, the row count per resample is a sum of
        # whole-cluster sizes — here all clusters have the same size, so every
        # resample yields exactly n_rows. A row-level bootstrap would too, so make
        # cluster sizes unequal and check the metric only ever sees whole clusters.
        clusters = np.array(["A", "A", "A", "B", "C"])  # sizes 3,1,1
        seen_sizes = set()

        def metric(rows):
            seen_sizes.add(len(rows))
            return float(len(rows))

        cluster_bootstrap_ci(clusters, metric, n_resamples=200, seed=0)
        # Every resample draws 3 clusters with replacement; achievable totals are
        # sums of {3,1,1} taken 3×. A size of 2 (split cluster A) must never occur.
        assert 2 not in seen_sizes
        assert seen_sizes  # metric was actually called

    def test_few_undefined_resamples_dropped_ci_still_built(self):
        # "Undefined" must be a pure function of the drawn rows (like a rare class
        # being absent from a resample in the real metric functions) rather than an
        # external call counter: joblib's process-based backend batches and
        # repickles closures across workers, so a counter captured by reference
        # does not reliably track a global call count across replicates.
        clusters = np.array([f"G{i}" for i in range(30) for _ in range(4)])
        # Undefined only when NEITHER of two sentinel genes' rows are drawn —
        # empirically ~12% of resamples, comfortably above the 0.8 valid threshold.
        sentinel_rows = set(
            np.where((clusters == "G0") | (clusters == "G1"))[0].tolist()
        )

        def metric(rows):
            return 0.7 if set(rows.tolist()) & sentinel_rows else None

        out = cluster_bootstrap_ci(clusters, metric, n_resamples=200, seed=0)
        # n_resamples counts only the contributing (non-None) draws.
        assert out["n_resamples"] < 200
        assert out["valid_frac"] >= 0.8
        assert out["ci_suppressed"] is False
        assert out["ci_low"] is not None

    def test_ci_suppressed_when_too_many_undefined(self):
        # Same rows-content-based approach as above, but with a single sentinel
        # gene: undefined on ~36% of resamples, below the 0.8 valid threshold.
        clusters = np.array([f"G{i}" for i in range(30) for _ in range(4)])
        sentinel_rows = set(np.where(clusters == "G0")[0].tolist())

        def metric(rows):
            return 0.7 if set(rows.tolist()) & sentinel_rows else None

        out = cluster_bootstrap_ci(clusters, metric, n_resamples=200, seed=0)
        assert out["valid_frac"] < 0.8
        assert out["ci_suppressed"] is True
        assert out["ci_low"] is None and out["ci_high"] is None
        # The point estimate (over all rows, where the sentinel gene IS present)
        # is still reported.
        assert out["point"] == 0.7

    def test_all_undefined_gives_none_ci(self):
        clusters = np.array([f"G{i % 4}" for i in range(20)])
        out = cluster_bootstrap_ci(clusters, lambda rows: None, n_resamples=50)
        assert out["ci_low"] is None and out["ci_high"] is None
        assert out["ci_suppressed"] is True
        assert out["n_resamples"] == 0
        assert out["valid_frac"] == 0.0

    def test_deterministic_for_fixed_seed(self):
        clusters = np.array([f"G{i % 5}" for i in range(50)])
        values = np.arange(50, dtype=float)
        fn = lambda rows: float(values[rows].mean())
        a = cluster_bootstrap_ci(clusters, fn, n_resamples=100, seed=3)
        b = cluster_bootstrap_ci(clusters, fn, n_resamples=100, seed=3)
        assert a == b


# ---------------------------------------------------------------------------
# discard_reason plumbing — the discard-tolerance warning must name a cause
# only the caller can know (or fall back to an honest placeholder), never
# assert "a fold lost a class" for a metric_fn that fails for another reason.
# ---------------------------------------------------------------------------


def _high_discard_clusters_and_metric():
    # Same construction as test_ci_suppressed_when_too_many_undefined: a single
    # sentinel gene drives ~36% of resamples undefined, comfortably above the
    # 1% BOOTSTRAP_MAX_DISCARD_FRAC tolerance so the warning always fires.
    clusters = np.array([f"G{i}" for i in range(30) for _ in range(4)])
    sentinel_rows = set(np.where(clusters == "G0")[0].tolist())

    def metric(rows):
        return 0.7 if set(rows.tolist()) & sentinel_rows else None

    return clusters, metric


class TestDiscardReasonWarning:
    def test_custom_reason_is_printed(self, capsys):
        clusters, metric = _high_discard_clusters_and_metric()
        cluster_bootstrap_ci(
            clusters,
            metric,
            n_resamples=200,
            seed=0,
            discard_reason="a widget lost its gizmo",
        )
        out = capsys.readouterr().out
        assert "a widget lost its gizmo" in out

    def test_missing_reason_falls_back_to_default_placeholder(self, capsys):
        clusters, metric = _high_discard_clusters_and_metric()
        cluster_bootstrap_ci(clusters, metric, n_resamples=200, seed=0)
        out = capsys.readouterr().out
        assert _DEFAULT_DISCARD_REASON in out
        # The placeholder must not claim a specific cause it cannot verify.
        assert "lost a class" not in out

    def test_callable_reason_is_invoked_after_resampling(self, capsys):
        clusters, metric = _high_discard_clusters_and_metric()
        calls = []

        def reason():
            calls.append(1)
            return "computed after the fact"

        cluster_bootstrap_ci(clusters, metric, n_resamples=200, seed=0, discard_reason=reason)
        out = capsys.readouterr().out
        assert "computed after the fact" in out
        # Called exactly once, for the warning — not once per resample.
        assert calls == [1]

    def test_no_warning_below_tolerance(self, capsys):
        clusters = np.array([f"G{i % 5}" for i in range(50)])
        values = np.arange(50, dtype=float)
        cluster_bootstrap_ci(
            clusters,
            lambda rows: float(values[rows].mean()),
            n_resamples=200,
            seed=0,
            discard_reason="should never be printed",
        )
        assert "should never be printed" not in capsys.readouterr().out

    def test_within_stratum_reason_is_printed(self, capsys):
        # Within-stratum resampling draws only from each stratum's own rows, so
        # every stratum is present in every draw — a sentinel *stratum* can never
        # be absent the way a sentinel *cluster* can be in the cluster bootstrap.
        # Make one row within a stratum rare instead: S0 has 4 rows, row 0 is the
        # only one of a kind, and each draw resamples S0's 4 rows with replacement
        # from S0's own pool, so missing row 0 in all 4 draws is plausible
        # ((3/4)^4 ≈ 32%), comfortably above the 1% tolerance.
        strata = np.array([f"S{i}" for i in range(30) for _ in range(4)])
        rare_row = 0

        def metric(rows):
            return 0.7 if rare_row in rows else None

        within_stratum_bootstrap_ci(
            strata,
            metric,
            n_resamples=200,
            seed=0,
            discard_reason="a stratum-specific cause",
        )
        assert "a stratum-specific cause" in capsys.readouterr().out

    def test_multi_reasons_are_per_metric(self, capsys):
        clusters, failing_metric = _high_discard_clusters_and_metric()

        def metric_fns_a(rows):
            return {"a": failing_metric(rows)}

        def metric_fns_b(rows):
            return {"b": failing_metric(rows)}

        cluster_bootstrap_ci_multi(
            clusters,
            [metric_fns_a, metric_fns_b],
            n_resamples=200,
            seed=0,
            discard_reasons={"a": "reason for a only"},
        )
        out = capsys.readouterr().out
        assert "reason for a only" in out
        # Metric "b" got no reason and must not borrow "a"'s.
        b_line = next(line for line in out.splitlines() if "] b:" in line)
        assert "reason for a only" not in b_line
        assert _DEFAULT_DISCARD_REASON in b_line


# ---------------------------------------------------------------------------
# cluster_subsample_ci (m-out-of-n, without replacement)
# ---------------------------------------------------------------------------

class TestClusterSubsampleCI:
    def _genes_with_rows(self, n_genes=30, rows_per_gene=4):
        clusters = np.array(
            [f"G{i}" for i in range(n_genes) for _ in range(rows_per_gene)]
        )
        return clusters, n_genes

    def test_no_cluster_repeated_within_a_replicate(self):
        # The point of the subsample: every drawn cluster contributes its rows
        # exactly once per replicate, so no row is ever duplicated.
        clusters, _ = self._genes_with_rows()

        def metric(rows):
            values, counts = np.unique(rows, return_counts=True)
            assert counts.max() == 1, "a row was drawn more than once in one replicate"
            return float(len(rows))

        cluster_subsample_ci(clusters, metric, n_resamples=100, seed=0, n_jobs=1)

    def test_subsample_size_matches_fraction(self):
        clusters, n_genes = self._genes_with_rows()
        rows_per_gene = 4
        seen_sizes = set()

        def metric(rows):
            seen_sizes.add(len(rows))
            return float(len(rows))

        out = cluster_subsample_ci(
            clusters, metric, n_resamples=50, subsample_frac=0.632, seed=0, n_jobs=1
        )
        expected_clusters = round(0.632 * n_genes)
        assert out["subsample_size"] == expected_clusters
        # One call sees all rows (the point estimate); every other call is a
        # replicate, and every replicate draws the same NUMBER of clusters
        # (without replacement), so every replicate has exactly the same row
        # count.
        assert seen_sizes == {len(clusters), expected_clusters * rows_per_gene}

    def test_point_uses_all_rows_not_a_subsample(self):
        clusters, _ = self._genes_with_rows()
        values = np.arange(len(clusters), dtype=float)
        out = cluster_subsample_ci(
            clusters, lambda rows: float(values[rows].mean()), n_resamples=20
        )
        assert out["point"] == pytest.approx(values.mean())

    def test_removes_duplicate_point_bias_a_bootstrap_would_have(self):
        # Construct exactly the failure mode this function exists to fix: a
        # "purity-like" metric that is inflated whenever a cluster's rows are
        # duplicated within a replicate (as a WITH-replacement bootstrap does).
        # Two genes per "family": one point per gene, both genes of a family
        # placed at identical coordinates in a 1-D "embedding" so a family
        # drawn twice under a with-replacement bootstrap creates an exact
        # distance-0 duplicate pair that inflates same-family adjacency.
        n_families = 20
        clusters = np.array([f"F{i}" for i in range(n_families) for _ in range(2)])

        def same_family_adjacent_fraction(rows):
            # Fraction of rows whose immediate duplicate (by row content, not
            # position) is also present in this replicate — a stand-in for
            # "does this replicate contain an exact-duplicate pair," which is
            # exactly what corrupts a kNN/pairwise-distance statistic.
            values, counts = np.unique(rows, return_counts=True)
            return float((counts >= 2).sum()) / len(values)

        subsample_out = cluster_subsample_ci(
            clusters, same_family_adjacent_fraction, n_resamples=200, seed=0
        )
        bootstrap_out = cluster_bootstrap_ci(
            clusters, same_family_adjacent_fraction, n_resamples=200, seed=0
        )
        # The subsample never duplicates a row, so the metric is always 0.
        assert subsample_out["point"] == pytest.approx(0.0)
        assert subsample_out["ci_low"] == pytest.approx(0.0)
        assert subsample_out["ci_high"] == pytest.approx(0.0)
        # The ordinary with-replacement bootstrap does duplicate rows (a
        # cluster drawn >=2 times), so its interval sits strictly above zero —
        # this is the artifact cluster_subsample_ci exists to remove.
        assert bootstrap_out["ci_high"] > 0.0

    def test_ci_suppressed_when_too_many_undefined(self):
        clusters, _ = self._genes_with_rows()
        sentinel_rows = set(np.where(clusters == "G0")[0].tolist())

        def metric(rows):
            return 0.7 if set(rows.tolist()) & sentinel_rows else None

        out = cluster_subsample_ci(clusters, metric, n_resamples=200, seed=0)
        assert out["valid_frac"] < 0.8
        assert out["ci_suppressed"] is True
        assert out["ci_low"] is None and out["ci_high"] is None

    def test_deterministic_for_fixed_seed(self):
        clusters, _ = self._genes_with_rows()
        values = np.arange(len(clusters), dtype=float)
        fn = lambda rows: float(values[rows].mean())
        a = cluster_subsample_ci(clusters, fn, n_resamples=100, seed=5)
        b = cluster_subsample_ci(clusters, fn, n_resamples=100, seed=5)
        assert a == b


# ---------------------------------------------------------------------------
# bootstrap_mechanism_metrics
# ---------------------------------------------------------------------------

class TestBootstrapMechanismMetrics:
    def _signal_data(self, seed=0):
        rng = np.random.RandomState(seed)
        n_genes, rows_per_gene = 60, 6
        n = n_genes * rows_per_gene
        # One class per gene, 20 genes per fold, so every class has several genes in
        # every fold — as the real splits do. A fixture thin enough that one draw can
        # empty a class out of a fold would discard most resamples by design.
        y = np.array([GOF, DN, LOF] * (n // 3))
        genes = np.array([f"G{i % n_genes}" for i in range(n)])
        # Build proba that ranks the true class high (clear, recoverable signal).
        proba = np.full((n, 3), 0.1)
        for i, cls in enumerate(y):
            proba[i, MECHANISM_CLASSES.index(cls)] = 0.8
        proba += rng.rand(n, 3) * 0.05
        proba /= proba.sum(axis=1, keepdims=True)
        # Each gene carries one class, so folds are built to hold consecutive genes
        # and therefore all three classes — the real family-split folds do too.
        folds = np.array([(i % n_genes) // (n_genes // 3) for i in range(n)])
        return y, proba, genes, folds

    def test_keys_macro_f1_and_per_class_auroc(self):
        y, proba, genes, folds = self._signal_data()
        out = bootstrap_mechanism_metrics(y, proba, genes, folds, n_resamples=100)
        assert "macro_f1" in out
        for cls in MECHANISM_CLASSES:
            assert f"auroc_{cls}" in out
            assert {
                "point", "ci_low", "ci_high", "n_resamples", "n_resamples_total",
                "valid_frac", "ci_suppressed", "n_clusters",
            } <= set(out[f"auroc_{cls}"])

    def test_auprc_carries_a_resampled_baseline_and_lift(self):
        # AUPRC's no-signal value is the prevalence, which moves with each resample.
        # The lift must be bootstrapped as one quantity (both terms from the same
        # draw), not reconstructed by subtracting two separately-computed points.
        y, proba, genes, folds = self._signal_data()
        out = bootstrap_mechanism_metrics(y, proba, genes, folds, n_resamples=300)
        for cls in MECHANISM_CLASSES:
            assert f"prevalence_{cls}" in out
            assert f"auprc_lift_{cls}" in out
        lift = out["auprc_lift_GOF"]
        assert lift["point"] == pytest.approx(
            out["auprc_GOF"]["point"] - out["prevalence_GOF"]["point"], abs=1e-9
        )
        assert lift["ci_low"] > 0  # planted signal beats its own prevalence baseline

    def test_recovers_gof_signal_above_chance(self):
        y, proba, genes, folds = self._signal_data()
        out = bootstrap_mechanism_metrics(y, proba, genes, folds, n_resamples=300)
        gof = out["auroc_GOF"]
        assert gof["point"] > 0.9
        assert gof["ci_low"] > 0.5  # CI excludes chance

    def test_clusters_are_genes(self):
        y, proba, genes, folds = self._signal_data()
        out = bootstrap_mechanism_metrics(y, proba, genes, folds, n_resamples=10)
        assert out["macro_f1"]["n_clusters"] == len(set(genes.tolist()))

    def test_attach_helper_uses_oof_folds(self):
        y, proba, genes, folds = self._signal_data()
        result = {"macro_f1_mean": 1.0}
        oof = {"y_true": y, "proba": proba, "folds": folds}

        returned = attach_mechanism_ci(
            result,
            oof,
            genes,
            compute_ci=True,
            n_resamples=10,
        )

        assert returned is result
        assert result["ci"]["macro_f1"]["n_clusters"] == len(set(genes.tolist()))

    def test_attach_helper_leaves_result_unchanged_when_disabled(self):
        result = {"macro_f1_mean": 1.0}

        returned = attach_mechanism_ci(
            result,
            None,
            None,
            compute_ci=False,
        )

        assert returned is result
        assert "ci" not in result

    def test_attach_helper_rejects_misaligned_clusters(self):
        y, proba, genes, folds = self._signal_data()
        oof = {"y_true": y, "proba": proba, "folds": folds}

        with pytest.raises(ValueError, match="not aligned"):
            attach_mechanism_ci(
                {},
                oof,
                genes[:-1],
                compute_ci=True,
                n_resamples=10,
            )


# ---------------------------------------------------------------------------
# paired_cluster_bootstrap_diff (same-fold pairing mode)
# ---------------------------------------------------------------------------

class TestPairedClusterBootstrapDiffSameFold:
    def _genes_with_rows(self, n_genes=30, rows_per_gene=4):
        clusters = np.array(
            [f"G{i}" for i in range(n_genes) for _ in range(rows_per_gene)]
        )
        return clusters, n_genes

    def test_shares_one_resample_across_both_arms(self):
        # The pairing property itself: both metric_fn_a and metric_fn_b must be
        # called with the IDENTICAL drawn row-index array on a given replicate.
        # Independent per-arm resampling would still produce a plausible-looking
        # CI, so this checks the drawn rows directly rather than the output CI.
        clusters, n_genes = self._genes_with_rows()
        calls_a: list[np.ndarray] = []
        calls_b: list[np.ndarray] = []

        def metric_a(rows):
            calls_a.append(np.array(rows))
            return float(len(rows))

        def metric_b(rows):
            calls_b.append(np.array(rows))
            return float(len(rows)) * 2.0

        paired_cluster_bootstrap_diff(
            clusters, metric_a, metric_b, n_resamples=25, seed=0, n_jobs=1
        )
        # One extra leading call each for the point estimate (over all rows).
        assert len(calls_a) == len(calls_b) == 26
        for rows_a, rows_b in zip(calls_a, calls_b):
            assert np.array_equal(rows_a, rows_b)

    def test_point_diff_is_point_a_minus_point_b(self):
        clusters, _ = self._genes_with_rows()
        values_a = np.arange(len(clusters), dtype=float)
        values_b = np.full(len(clusters), 1.0)
        out = paired_cluster_bootstrap_diff(
            clusters,
            lambda rows: float(values_a[rows].mean()),
            lambda rows: float(values_b[rows].mean()),
            n_resamples=20,
        )
        assert out["point_a"] == pytest.approx(values_a.mean())
        assert out["point_b"] == pytest.approx(1.0)
        assert out["point_diff"] == pytest.approx(out["point_a"] - out["point_b"])

    def test_planted_difference_recovers_ci_excluding_zero(self):
        # Both arms share one per-gene base value (so the true difference reduces
        # to the planted shift plus small iid row-level noise), keeping the arms
        # genuinely paired rather than independently noisy.
        clusters, n_genes = self._genes_with_rows()
        rng = np.random.RandomState(1)
        gene_base = {f"G{i}": rng.normal(scale=1.0) for i in range(n_genes)}
        base = np.array([gene_base[g] for g in clusters])
        noise_a = rng.normal(scale=0.1, size=len(clusters))
        noise_b = rng.normal(scale=0.1, size=len(clusters))
        value_a = base + noise_a + 1.0  # planted shift
        value_b = base + noise_b

        out = paired_cluster_bootstrap_diff(
            clusters,
            lambda rows: float(value_a[rows].mean()),
            lambda rows: float(value_b[rows].mean()),
            n_resamples=500,
            seed=0,
        )
        assert out["point_diff"] == pytest.approx(1.0, abs=0.2)
        assert out["ci_low"] > 0.0  # CI excludes zero

    def test_no_true_difference_gives_ci_spanning_zero(self):
        # Same per-gene base for both arms, independent small row-level noise only
        # — the true difference is zero, so the CI should bracket zero.
        clusters, n_genes = self._genes_with_rows()
        rng = np.random.RandomState(0)
        gene_base = {f"G{i}": rng.normal(scale=1.0) for i in range(n_genes)}
        base = np.array([gene_base[g] for g in clusters])
        noise_a = rng.normal(scale=0.1, size=len(clusters))
        noise_b = rng.normal(scale=0.1, size=len(clusters))
        value_a = base + noise_a
        value_b = base + noise_b

        out = paired_cluster_bootstrap_diff(
            clusters,
            lambda rows: float(value_a[rows].mean()),
            lambda rows: float(value_b[rows].mean()),
            n_resamples=500,
            seed=0,
        )
        assert out["ci_low"] <= 0.0 <= out["ci_high"]

    def test_replicate_dropped_when_one_arm_undefined(self):
        # "Undefined" must be a pure function of the drawn rows (like a rare class
        # being absent from a resample in the real metric functions) rather than an
        # external call counter: joblib's process-based backend batches and
        # repickles closures across workers, so a counter captured by reference
        # does not reliably track a global call count across replicates.
        clusters, _ = self._genes_with_rows()
        # Undefined only when NEITHER of two sentinel genes' rows are drawn —
        # empirically ~12% of resamples, comfortably above the 0.8 valid threshold.
        sentinel_rows = set(
            np.where((clusters == "G0") | (clusters == "G1"))[0].tolist()
        )

        def metric_a(rows):
            return 0.7 if set(rows.tolist()) & sentinel_rows else None

        def metric_b(rows):
            return 0.2

        out = paired_cluster_bootstrap_diff(
            clusters, metric_a, metric_b, n_resamples=200, seed=0
        )
        assert out["n_resamples"] < 200
        assert out["valid_frac"] >= 0.8
        assert out["ci_suppressed"] is False

    def test_ci_suppressed_when_too_many_undefined(self):
        # Same rows-content-based approach as above, but with a single sentinel
        # gene: undefined on ~36% of resamples, below the 0.8 valid threshold.
        clusters, _ = self._genes_with_rows()
        sentinel_rows = set(np.where(clusters == "G0")[0].tolist())

        def metric_a(rows):
            return 0.7 if set(rows.tolist()) & sentinel_rows else None

        def metric_b(rows):
            return 0.2

        out = paired_cluster_bootstrap_diff(
            clusters, metric_a, metric_b, n_resamples=200, seed=0
        )
        assert out["valid_frac"] < 0.8
        assert out["ci_suppressed"] is True
        assert out["ci_low"] is None and out["ci_high"] is None
        # The point diff (over all rows, where the sentinel gene IS present) is
        # still reported.
        assert out["point_diff"] == pytest.approx(0.5)

    def test_deterministic_for_fixed_seed(self):
        clusters, _ = self._genes_with_rows()
        values_a = np.arange(len(clusters), dtype=float)
        values_b = np.arange(len(clusters), dtype=float) * 0.5
        fn_a = lambda rows: float(values_a[rows].mean())
        fn_b = lambda rows: float(values_b[rows].mean())
        first = paired_cluster_bootstrap_diff(
            clusters, fn_a, fn_b, n_resamples=100, seed=5
        )
        second = paired_cluster_bootstrap_diff(
            clusters, fn_a, fn_b, n_resamples=100, seed=5
        )
        assert first == second


# ---------------------------------------------------------------------------
# paired_cluster_bootstrap_diff_cross_partition (cross-partition pairing mode)
# ---------------------------------------------------------------------------

class TestPairedClusterBootstrapDiffCrossPartition:
    def _family_and_gene_clusters(self, n_families=6, genes_per_family=2, rows_per_gene=2):
        family_clusters = []
        gene_clusters = []
        for family_idx in range(n_families):
            for gene_idx in range(genes_per_family):
                for _ in range(rows_per_gene):
                    family_clusters.append(f"F{family_idx}")
                    gene_clusters.append(f"F{family_idx}_G{gene_idx}")
        return np.array(family_clusters), np.array(gene_clusters)

    def test_resamples_the_given_unit_not_the_arms_own_partition(self):
        # The resampling unit is whatever `resample_clusters` says — here families
        # — regardless of what finer structure the arms' own metric closures use
        # internally. n_clusters must reflect the family count, not the gene count.
        family_clusters, gene_clusters = self._family_and_gene_clusters()
        n_families = len(set(family_clusters.tolist()))
        n_genes = len(set(gene_clusters.tolist()))
        assert n_families != n_genes  # the two units must genuinely differ

        out = paired_cluster_bootstrap_diff_cross_partition(
            family_clusters,
            lambda rows: float(len(rows)),
            lambda rows: float(len(rows)),
            n_resamples=10,
        )
        assert out["n_clusters"] == n_families

    def test_family_resample_never_splits_a_family(self):
        # Resampling at the family level must draw whole families: if any row of a
        # family is present in a replicate's row set, every row of that family
        # (both its genes) must be present too.
        family_clusters, gene_clusters = self._family_and_gene_clusters()
        seen_rows: list[np.ndarray] = []

        def metric(rows):
            seen_rows.append(np.array(rows))
            return float(len(rows))

        paired_cluster_bootstrap_diff_cross_partition(
            family_clusters, metric, metric, n_resamples=50, seed=0, n_jobs=1
        )
        for rows in seen_rows:
            present_families = set(family_clusters[rows].tolist())
            for family in present_families:
                expected = set(np.where(family_clusters == family)[0].tolist())
                present = set(rows[family_clusters[rows] == family].tolist())
                # Every row of a present family appears once per draw of that
                # family; the set of distinct row positions must equal the full
                # family, not a subset (which would mean a gene was split off).
                assert present == expected

    def test_shares_one_resample_across_both_arms(self):
        family_clusters, _ = self._family_and_gene_clusters()
        calls_a: list[np.ndarray] = []
        calls_b: list[np.ndarray] = []

        def metric_a(rows):
            calls_a.append(np.array(rows))
            return float(len(rows))

        def metric_b(rows):
            calls_b.append(np.array(rows))
            return float(len(rows)) * 3.0

        paired_cluster_bootstrap_diff_cross_partition(
            family_clusters, metric_a, metric_b, n_resamples=15, seed=0, n_jobs=1
        )
        assert len(calls_a) == len(calls_b) == 16
        for rows_a, rows_b in zip(calls_a, calls_b):
            assert np.array_equal(rows_a, rows_b)

    def test_each_arm_scored_under_its_own_partition(self):
        # Arm A and arm B use unrelated per-row scores (standing in for
        # gene-split vs family-split predictions computed under different CV
        # partitions) — the function must not conflate them: point_a/point_b
        # must reflect each arm's own values, not each other's.
        family_clusters, _ = self._family_and_gene_clusters()
        rng = np.random.RandomState(2)
        value_a = rng.rand(len(family_clusters))
        value_b = rng.rand(len(family_clusters))

        out = paired_cluster_bootstrap_diff_cross_partition(
            family_clusters,
            lambda rows: float(value_a[rows].mean()),
            lambda rows: float(value_b[rows].mean()),
            n_resamples=10,
            seed=0,
        )
        assert out["point_a"] == pytest.approx(value_a.mean())
        assert out["point_b"] == pytest.approx(value_b.mean())

    def test_sensitivity_clusters_resample_at_finer_granularity(self):
        family_clusters, gene_clusters = self._family_and_gene_clusters()
        n_families = len(set(family_clusters.tolist()))
        n_genes = len(set(gene_clusters.tolist()))

        out = paired_cluster_bootstrap_diff_cross_partition(
            family_clusters,
            lambda rows: float(len(rows)),
            lambda rows: float(len(rows)),
            sensitivity_clusters=gene_clusters,
            n_resamples=10,
            seed=0,
        )
        assert out["n_clusters"] == n_families
        assert "gene_resampled_sensitivity" in out
        assert out["gene_resampled_sensitivity"]["n_clusters"] == n_genes

    def test_no_sensitivity_clusters_omits_the_key(self):
        family_clusters, _ = self._family_and_gene_clusters()
        out = paired_cluster_bootstrap_diff_cross_partition(
            family_clusters,
            lambda rows: float(len(rows)),
            lambda rows: float(len(rows)),
            n_resamples=10,
        )
        assert "gene_resampled_sensitivity" not in out


# ---------------------------------------------------------------------------
# label permutation
# ---------------------------------------------------------------------------

def _one_fold(rows):
    """All rows in one fold, for the tests that are about the shuffle unit alone."""
    return np.zeros(len(rows), dtype=int)


class TestPermuteLabels:
    def test_grouped_shuffle_keeps_one_label_per_gene(self):
        rng = np.random.RandomState(0)
        genes = np.array(["G0", "G0", "G0", "G1", "G1", "G2"])
        labels = np.array([GOF, GOF, GOF, DN, DN, LOF])
        permuted = _permute_labels(labels, genes, _one_fold(labels), rng)
        # Every gene's rows must still carry a single label.
        for gene in set(genes.tolist()):
            mask = genes == gene
            assert len(set(permuted[mask].tolist())) == 1

    def test_grouped_shuffle_is_a_permutation_of_gene_labels(self):
        rng = np.random.RandomState(0)
        genes = np.array(["G0", "G0", "G1", "G2", "G2"])
        labels = np.array([GOF, GOF, DN, LOF, LOF])
        permuted = _permute_labels(labels, genes, _one_fold(labels), rng)
        gene_labels_in = sorted([GOF, DN, LOF])
        gene_labels_out = sorted(
            {g: permuted[genes == g][0] for g in set(genes.tolist())}.values()
        )
        assert gene_labels_out == gene_labels_in

    def test_labels_stay_inside_their_own_fold(self):
        # Moving a label across folds changes each fold's class composition, which is
        # the confound the within-fold scoring removes. The shuffle must not undo it.
        rng = np.random.RandomState(0)
        genes = np.array(["G0", "G0", "G1", "G1", "G2", "G2", "G3", "G3"])
        labels = np.array([GOF, GOF, DN, DN, LOF, LOF, GOF, GOF])
        folds = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        permuted = _permute_labels(labels, genes, folds, rng)
        for fold in (0, 1):
            mask = folds == fold
            assert sorted(permuted[mask].tolist()) == sorted(labels[mask].tolist())

    def test_ungrouped_shuffle_permutes_all(self):
        rng = np.random.RandomState(0)
        labels = np.array([GOF, DN, LOF, GOF, DN, LOF])
        permuted = _permute_labels(labels, None, _one_fold(labels), rng)
        assert sorted(permuted.tolist()) == sorted(labels.tolist())


class TestLabelPermutationPvalue:
    def test_resolution_floor_is_flagged_as_unresolved(self):
        labels = np.arange(10)

        def exact_original_order(permuted_labels):
            return float(np.array_equal(permuted_labels, labels))

        out = label_permutation_pvalue(
            exact_original_order,
            labels,
            statistic="exact_original_order",
            folds=_one_fold(labels),
            n_permutations=1,
            seed=0,
            n_jobs=1,
        )
        assert out["p_value"] == pytest.approx(0.5)
        assert out["p_value_resolution"] == pytest.approx(0.5)
        assert out["resolution_limited"] is True

    def test_refit_permutation_reports_immovable_cluster_count(self):
        labels = np.array([GOF, LOF, DN, DN, GOF, GOF, LOF])
        genes = np.array(["a", "b", "c", "d", "e", "f", "g"])
        clusters = np.array(["F0", "F0", "F1", "F1", "F2", "F2", "F3"])

        out = label_permutation_pvalue(
            lambda permuted_labels: 0.5,
            labels,
            statistic="constant",
            folds=_one_fold(labels),
            groups=genes,
            clusters=clusters,
            n_permutations=5,
            seed=0,
            n_jobs=1,
        )
        assert out["n_clusters_immovable"] == 1

    def test_null_centers_on_chance_and_signal_is_significant(self):
        # A metric that perfectly separates by the *current* labels: the observed
        # value is high, but under a gene-level shuffle it collapses toward chance.
        rng = np.random.RandomState(0)
        genes = np.array([f"G{i // 4}" for i in range(40)])  # 10 genes, 4 rows each
        gene_score = {g: rng.rand() for g in set(genes.tolist())}
        scores = np.array([gene_score[g] for g in genes])
        # True labels: top-half genes by score are GOF, rest LOF.
        thresh = np.median(list(gene_score.values()))
        labels = np.array([GOF if gene_score[g] >= thresh else LOF for g in genes])

        from sklearn.metrics import roc_auc_score

        def run_metric(lab):
            y_bin = (lab == GOF).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                return None
            return float(roc_auc_score(y_bin, scores))

        out = label_permutation_pvalue(
            run_metric, labels, statistic="auroc_GOF", folds=_one_fold(labels),
            groups=genes, n_permutations=200,
            alternative="greater",
        )
        assert out["observed"] == pytest.approx(1.0)  # perfect by construction
        assert out["null_mean"] == pytest.approx(0.5, abs=0.1)
        assert out["p_value"] < 0.05

    def test_pvalue_formula_bounds(self):
        rng = np.random.RandomState(2)
        genes = np.array([f"G{i // 3}" for i in range(30)])
        labels = np.array([GOF if i % 2 else LOF for i in range(30)])

        out = label_permutation_pvalue(
            lambda lab: 0.5, labels, statistic="constant", folds=_one_fold(labels), groups=genes, n_permutations=50
        )
        # (1 + #{null >= observed}) / (1 + n); with a constant metric all null >=
        # observed, so p == 1.0, and p is always in (0, 1].
        assert 0.0 < out["p_value"] <= 1.0
        assert out["p_value"] == pytest.approx(1.0)

    def test_deterministic_across_runs_for_fixed_seed(self):
        # Each permutation is seeded from a SeedSequence spawned off `seed`, so the
        # whole null distribution must reproduce exactly run-to-run — independent of
        # how joblib schedules the parallel refits.
        rng = np.random.RandomState(3)
        genes = np.array([f"G{i // 4}" for i in range(40)])
        labels = np.array([GOF if i % 2 else LOF for i in range(40)])
        feature = rng.rand(40)

        def run_metric(lab):
            y_bin = (lab == GOF).astype(int)
            if y_bin.sum() in (0, len(y_bin)):
                return None
            return float(np.corrcoef(feature, y_bin)[0, 1])

        first = label_permutation_pvalue(
            run_metric, labels, statistic="corr", folds=_one_fold(labels), groups=genes, n_permutations=64, seed=7
        )
        second = label_permutation_pvalue(
            run_metric, labels, statistic="corr", folds=_one_fold(labels), groups=genes, n_permutations=64, seed=7
        )
        assert first["p_value"] == second["p_value"]
        assert first["null_mean"] == second["null_mean"]
        assert first["n_permutations"] == second["n_permutations"]

    def test_serial_and_parallel_agree(self):
        # n_jobs must not change the result: n_jobs=1 (serial) and n_jobs=2 (parallel)
        # consume the same per-permutation seeds, so the null is identical.
        rng = np.random.RandomState(4)
        genes = np.array([f"G{i // 4}" for i in range(40)])
        labels = np.array([GOF if i % 2 else LOF for i in range(40)])
        feature = rng.rand(40)

        def run_metric(lab):
            y_bin = (lab == GOF).astype(int)
            if y_bin.sum() in (0, len(y_bin)):
                return None
            return float(np.corrcoef(feature, y_bin)[0, 1])

        serial = label_permutation_pvalue(
            run_metric, labels, statistic="corr", folds=_one_fold(labels), groups=genes, n_permutations=48, seed=1, n_jobs=1
        )
        parallel = label_permutation_pvalue(
            run_metric, labels, statistic="corr", folds=_one_fold(labels), groups=genes, n_permutations=48, seed=1, n_jobs=2
        )
        assert serial["p_value"] == parallel["p_value"]
        assert serial["null_mean"] == parallel["null_mean"]

    def test_nested_closure_capturing_array_is_picklable(self):
        # The real call sites pass a nested closure that captures a large feature
        # matrix via a default arg (e.g. `_family_macro_f1(perm, _X=X)`). joblib must
        # ship that closure to workers, so this guards against a pickling regression.
        def make_metric():
            captured_X = np.random.RandomState(5).randn(200, 8)

            def metric(lab, _X=captured_X):
                y = (lab == GOF).astype(float)
                return float(np.corrcoef(_X[:, 0], y)[0, 1] ** 2)

            return metric

        genes = np.array([f"G{i // 4}" for i in range(200)])
        labels = np.array([GOF if i % 3 == 0 else LOF for i in range(200)])

        out = label_permutation_pvalue(
            make_metric(), labels, statistic="r2", folds=_one_fold(labels), groups=genes, n_permutations=32, seed=0, n_jobs=2
        )
        assert out["n_permutations"] == 32
        assert 0.0 < out["p_value"] <= 1.0

    def test_non_finite_metric_values_dropped_from_null(self):
        # A metric that sometimes returns None/NaN must not poison the null: those
        # draws are dropped and n_permutations reflects only the finite ones.
        genes = np.array([f"G{i // 2}" for i in range(20)])
        labels = np.array([GOF if i % 2 else LOF for i in range(20)])

        calls = {"n": 0}

        def flaky_metric(lab):
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                return float("nan")
            return 0.5

        out = label_permutation_pvalue(
            flaky_metric, labels, statistic="flaky", folds=_one_fold(labels), groups=genes, n_permutations=10, seed=0, n_jobs=1
        )
        # Half the draws are NaN and dropped; the kept null has only finite values.
        assert out["n_permutations"] < 10
        assert np.isfinite(out["null_mean"])


# ---------------------------------------------------------------------------
# oof_permutation_pvalue / macro_ovr_auroc
# ---------------------------------------------------------------------------

def _folds_by_unit(units, n_folds=3):
    """Assign each unit (gene or family) to a fold, the way the CV splitters do.

    Whole units stay inside one fold, which is what the within-fold shuffle needs.
    """
    unit_fold = {unit: i % n_folds for i, unit in enumerate(sorted(set(np.asarray(units).tolist())))}
    return np.array([unit_fold[u] for u in np.asarray(units)], dtype=int)


def _gene_level_labels(rng, n_genes, rows_per_gene):
    genes = np.repeat([f"G{i}" for i in range(n_genes)], rows_per_gene)
    gene_labels = rng.choice([GOF, DN, LOF], size=n_genes)
    return genes, np.repeat(gene_labels, rows_per_gene)


def _proba_matching(labels, strength=0.8):
    """Predictions that rank the true class highest, aligned to MECHANISM_CLASSES."""
    proba = np.full((len(labels), len(MECHANISM_CLASSES)), (1.0 - strength) / 2.0)
    for col_idx, cls in enumerate(MECHANISM_CLASSES):
        proba[labels == cls, col_idx] = strength
    return proba / proba.sum(axis=1, keepdims=True)


class TestPermuteLabelsByCluster:
    def _families(self):
        # F0: 2 genes mixed, F1: 2 genes homogeneous, F2: 2 genes mixed, F3: 1 gene.
        genes = np.array(["a", "b", "c", "d", "e", "f", "g"])
        clusters = np.array(["F0", "F0", "F1", "F1", "F2", "F2", "F3"])
        labels = np.array([GOF, LOF, DN, DN, GOF, GOF, LOF])
        return labels, genes, clusters

    def test_whole_blocks_move_together(self):
        labels, genes, clusters = self._families()
        rng = np.random.RandomState(0)
        permuted = _permute_labels_by_cluster(labels, genes, clusters, _one_fold(labels), rng)
        # Every size-2 family must now hold some size-2 family's original block.
        original_blocks = {("F0", (GOF, LOF)), ("F1", (DN, DN)), ("F2", (GOF, GOF))}
        blocks_in = {tuple(sorted(b)) for _, b in original_blocks}
        for fam in ["F0", "F1", "F2"]:
            got = tuple(sorted(permuted[clusters == fam].tolist()))
            assert got in blocks_in

    def test_label_multiset_is_preserved(self):
        labels, genes, clusters = self._families()
        rng = np.random.RandomState(1)
        permuted = _permute_labels_by_cluster(labels, genes, clusters, _one_fold(labels), rng)
        assert sorted(permuted.tolist()) == sorted(labels.tolist())

    def test_unique_size_cluster_cannot_move_and_is_counted(self):
        # F3 is the only single-gene family, so it has no same-size partner to swap
        # with and keeps its own label. That has to be visible, not silent.
        labels, genes, clusters = self._families()
        rng = np.random.RandomState(2)
        permuted = _permute_labels_by_cluster(labels, genes, clusters, _one_fold(labels), rng)
        immovable = count_immovable_clusters(genes, clusters, _one_fold(labels))
        assert immovable == 1
        assert permuted[clusters == "F3"][0] == LOF

    def test_within_family_mixing_is_preserved(self):
        # A family that was label-homogeneous stays homogeneous and a mixed one stays
        # mixed: forcing one label per family would make the null wider than reality.
        labels, genes, clusters = self._families()
        rng = np.random.RandomState(3)
        homogeneous_before = sum(
            len(set(labels[clusters == fam].tolist())) == 1 for fam in ["F0", "F1", "F2"]
        )
        permuted = _permute_labels_by_cluster(labels, genes, clusters, _one_fold(labels), rng)
        homogeneous_after = sum(
            len(set(permuted[clusters == fam].tolist())) == 1 for fam in ["F0", "F1", "F2"]
        )
        assert homogeneous_after == homogeneous_before

    def test_blocks_only_swap_with_partners_in_the_same_fold(self):
        # Two same-size families in different folds are not swap partners: a swap
        # between them would move labels across the fold boundary.
        genes = np.array(["a", "b", "c", "d"])
        clusters = np.array(["F0", "F0", "F1", "F1"])
        labels = np.array([GOF, GOF, LOF, LOF])
        folds = np.array([0, 0, 1, 1])
        rng = np.random.RandomState(0)
        permuted = _permute_labels_by_cluster(labels, genes, clusters, folds, rng)
        immovable = count_immovable_clusters(genes, clusters, folds)
        assert immovable == 2
        assert np.array_equal(permuted, labels)

    def test_a_cluster_spanning_two_folds_is_rejected(self):
        genes = np.array(["a", "b"])
        clusters = np.array(["F0", "F0"])
        labels = np.array([GOF, LOF])
        folds = np.array([0, 1])
        with pytest.raises(ValueError, match="within-fold shuffle"):
            _permute_labels_by_cluster(
                labels, genes, clusters, folds, np.random.RandomState(0)
            )

    def test_every_gene_keeps_one_label_across_its_rows(self):
        rows_per_gene = 3
        genes = np.repeat(["a", "b", "c", "d"], rows_per_gene)
        clusters = np.repeat(["F0", "F0", "F1", "F1"], rows_per_gene)
        labels = np.repeat([GOF, LOF, DN, GOF], rows_per_gene)
        rng = np.random.RandomState(4)
        permuted = _permute_labels_by_cluster(labels, genes, clusters, _one_fold(labels), rng)
        for gene in np.unique(genes):
            assert len(set(permuted[genes == gene].tolist())) == 1


def _offset_fold_scales():
    """Two folds, each ranked perfectly, on probability scales that do not compare.

    Every fold is fitted independently, so its probabilities carry their own scale.
    Here fold 1 sits entirely above fold 0: within either fold the positives outrank
    the negatives, but fold 1's negatives outrank fold 0's positives. The fold sizes
    are lopsided so that ranking the concatenation lands below chance rather than
    merely below one, which is the symptom the real run showed — per-fold AUROCs of
    0.55 to 0.61 reported as a pooled 0.40 to 0.48.

    Returns the labels, a one-column probability matrix, the fold index and the
    gene of each row.
    """
    fold_0 = [(GOF, 0.10)] * 8 + [(LOF, 0.00)] * 2
    fold_1 = [(GOF, 1.00)] * 2 + [(LOF, 0.90)] * 8
    labels = np.array([cls for cls, _ in fold_0 + fold_1])
    proba = np.array([[score] for _, score in fold_0 + fold_1])
    folds = np.array([0] * len(fold_0) + [1] * len(fold_1))
    genes = np.array([f"G{i}" for i in range(len(labels))])
    return labels, proba, folds, genes


POOLED_AUROC_UNDER_OFFSET_SCALES = 0.36


class TestFoldScalesAreNotPooled:
    """The defect this whole change exists to remove, as one assertion.

    Ranking the concatenation of independently fitted folds compares scores that
    were never on a common scale. Scoring within each fold and averaging is immune
    to it, so the fold-aware value is the perfect 1.0 that each fold actually
    achieved while the pooled value falls below chance.
    """

    def test_pooled_ranking_really_does_fall_below_chance_here(self):
        # The counterfactual, so the test below is anchored to a number rather than
        # to an inequality that a broken implementation could also satisfy.
        from sklearn.metrics import roc_auc_score

        labels, proba, _, _ = _offset_fold_scales()
        pooled = roc_auc_score((labels == GOF).astype(int), proba[:, 0])
        assert pooled == pytest.approx(POOLED_AUROC_UNDER_OFFSET_SCALES)

    def test_macro_ovr_auroc_scores_within_fold(self):
        labels, proba, folds, _ = _offset_fold_scales()
        value, scored = macro_ovr_auroc(labels, proba, folds, classes=[GOF])
        assert scored == (GOF,)
        assert value == pytest.approx(1.0)

    def test_mechanism_metrics_point_estimate_scores_within_fold(self):
        labels, proba, folds, genes = _offset_fold_scales()
        out = bootstrap_mechanism_metrics(
            labels, proba, genes, folds, classes=[GOF], n_resamples=50
        )
        assert out["auroc_GOF"]["point"] == pytest.approx(1.0)

    def test_paired_diff_scores_within_fold(self):
        # A paired difference carries the distortion on both sides, so it is the
        # most exposed of the three. Both arms rank perfectly within fold, so their
        # difference is zero; pooling either side would move it off zero.
        labels, proba, folds, genes = _offset_fold_scales()
        arm = {
            "y_true": labels, "proba": proba, "folds": folds,
            "genes": genes, "row_ids": np.arange(len(labels)),
        }
        out = paired_oof_diff(
            arm, arm, {gene: gene for gene in genes}, "same-arm",
            classes=[GOF], metric="auroc_one_vs_rest", pos_class=GOF,
            n_resamples=50,
        )
        assert out["point_a"] == pytest.approx(1.0)
        assert out["point_diff"] == pytest.approx(0.0)


class TestMacroOvrAuroc:
    def test_perfect_ranking_is_one_and_chance_is_half(self):
        rng = np.random.RandomState(0)
        genes, labels = _gene_level_labels(rng, 30, 4)
        folds = _folds_by_unit(genes)
        value, scored = macro_ovr_auroc(labels, _proba_matching(labels), folds)
        assert value == pytest.approx(1.0)
        assert set(scored) == set(MECHANISM_CLASSES)
        chance, _ = macro_ovr_auroc(labels, _simplex_proba(rng, len(labels)), folds)
        assert chance == pytest.approx(0.5, abs=0.15)

    def test_absent_class_is_dropped_and_reported(self):
        # DN never appears, so it has no defined one-vs-rest AUROC. Averaging it in
        # as 0.5 would drag a perfect two-class result away from 1.0 — and the caller
        # has to know only two classes were averaged, or it will compare this against
        # a three-class value.
        labels = np.array([GOF, GOF, LOF, LOF])
        folds = np.array([0, 1, 0, 1])
        value, scored = macro_ovr_auroc(labels, _proba_matching(labels), folds)
        assert value == pytest.approx(1.0)
        assert set(scored) == {GOF, LOF}

    def test_class_missing_from_one_fold_is_dropped(self):
        # LOF is in both folds. GOF is only in fold 0 and DN only in fold 1, so each
        # of those would be an average over one fold while LOF averages over two —
        # different statistics, and only the one both folds can score survives.
        labels = np.array([GOF, LOF, LOF, LOF, LOF, DN])
        folds = np.array([0, 0, 0, 1, 1, 1])
        _, scored = macro_ovr_auroc(labels, _proba_matching(labels), folds)
        assert set(scored) == {LOF}

    def test_single_class_everywhere_returns_none(self):
        labels = np.array([LOF, LOF, LOF])
        folds = np.array([0, 0, 1])
        assert macro_ovr_auroc(labels, _proba_matching(labels), folds) == (None, ())


class TestOofPermutationPvalue:
    def test_resolution_floor_is_flagged_as_unresolved(self):
        labels = np.array([GOF, GOF, DN, DN, LOF, LOF])
        genes = np.array(["G0", "G1", "G2", "G3", "G4", "G5"])
        out = oof_permutation_pvalue(
            labels,
            _proba_matching(labels),
            _one_fold(labels),
            groups=genes,
            n_permutations=1,
            seed=0,
        )
        assert out["p_value"] == pytest.approx(0.5)
        assert out["p_value_resolution"] == pytest.approx(0.5)
        assert out["resolution_limited"] is True

    def test_planted_signal_is_significant_and_null_centers_on_chance(self):
        rng = np.random.RandomState(0)
        genes, labels = _gene_level_labels(rng, 60, 4)
        out = oof_permutation_pvalue(
            labels, _proba_matching(labels), _folds_by_unit(genes),
            groups=genes, n_permutations=200, seed=0,
        )
        assert out["observed"] == pytest.approx(1.0)
        assert out["null_mean"] == pytest.approx(0.5, abs=0.05)
        assert out["p_value"] < 0.05
        assert out["statistic"] == "macro_ovr_auroc"
        assert out["null_type"] == "oof_fixed_predictions"

    def test_uninformative_predictions_are_not_significant(self):
        rng = np.random.RandomState(1)
        genes, labels = _gene_level_labels(rng, 60, 4)
        out = oof_permutation_pvalue(
            labels, _simplex_proba(rng, len(labels)), _folds_by_unit(genes),
            groups=genes, n_permutations=200, seed=0,
        )
        assert out["p_value"] > 0.1

    def test_family_block_shuffle_gives_a_wider_null_than_gene_level(self):
        # Families whose genes share a mechanism are the real exchangeable unit. A
        # gene-level shuffle breaks that shared structure and builds too tight a null,
        # which would make a chance-level score look significant.
        rng = np.random.RandomState(6)
        n_families, genes_per_family, rows_per_gene = 30, 3, 2
        genes, clusters, labels = [], [], []
        for fam in range(n_families):
            family_label = [GOF, DN, LOF][fam % 3]
            for gene_idx in range(genes_per_family):
                gene = f"F{fam}g{gene_idx}"
                genes += [gene] * rows_per_gene
                clusters += [f"F{fam}"] * rows_per_gene
                labels += [family_label] * rows_per_gene
        genes, clusters, labels = np.array(genes), np.array(clusters), np.array(labels)
        proba = _simplex_proba(rng, len(labels))

        folds = _folds_by_unit(clusters)
        block = oof_permutation_pvalue(
            labels, proba, folds, groups=genes, clusters=clusters,
            n_permutations=300, seed=0,
        )
        gene_level = oof_permutation_pvalue(
            labels, proba, folds, groups=genes, n_permutations=300, seed=0
        )
        assert block["permutation_unit"] == "cluster_block"
        assert block["n_clusters_immovable"] == 0
        assert gene_level["permutation_unit"] == "gene"
        assert block["null_std"] > gene_level["null_std"]

    def test_gene_level_shuffle_gives_a_wider_null_than_variant_level(self):
        # Labels are constant within a gene, so shuffling individual variants breaks
        # that structure and builds an unrealistically tight null — which would make
        # a chance-level score look significant. The gene-level null must be wider.
        rng = np.random.RandomState(2)
        genes, labels = _gene_level_labels(rng, 40, 8)
        proba = _simplex_proba(rng, len(labels))

        folds = _folds_by_unit(genes)
        gene_level = oof_permutation_pvalue(
            labels, proba, folds, groups=genes, n_permutations=400, seed=0
        )
        variant_level = oof_permutation_pvalue(
            labels, proba, folds, groups=None, n_permutations=400, seed=0
        )
        # Both nulls sit on chance; only their widths differ. 40 genes is the real
        # sample size, 320 variants is not, so the variant-level null is too tight.
        assert gene_level["null_mean"] == pytest.approx(0.5, abs=0.05)
        assert variant_level["null_mean"] == pytest.approx(0.5, abs=0.05)
        assert gene_level["null_std"] > variant_level["null_std"]

    def test_predictions_are_never_recomputed(self):
        # The whole point of this path is that it costs no refits: the proba matrix
        # handed in must be the one scored on every permutation.
        rng = np.random.RandomState(3)
        genes, labels = _gene_level_labels(rng, 20, 3)
        proba = _proba_matching(labels)
        before = proba.copy()
        oof_permutation_pvalue(
            labels, proba, _folds_by_unit(genes), groups=genes, n_permutations=50, seed=0
        )
        assert np.array_equal(proba, before)

    def test_deterministic_for_fixed_seed(self):
        rng = np.random.RandomState(4)
        genes, labels = _gene_level_labels(rng, 30, 4)
        proba = _simplex_proba(rng, len(labels))
        folds = _folds_by_unit(genes)
        first = oof_permutation_pvalue(labels, proba, folds, groups=genes, n_permutations=64, seed=7)
        second = oof_permutation_pvalue(labels, proba, folds, groups=genes, n_permutations=64, seed=7)
        assert first["p_value"] == second["p_value"]
        assert first["null_mean"] == second["null_mean"]

    def test_draws_scoring_a_different_class_set_are_dropped_and_counted(self):
        # One gene carries the only DN, so most shuffles land DN somewhere scorable
        # but some leave a class degenerate. Those draws average a different number
        # of classes and must not enter the null.
        rng = np.random.RandomState(5)
        genes = np.array(["G0", "G0", "G1", "G1", "G2", "G2"])
        labels = np.array([GOF, GOF, LOF, LOF, DN, DN])
        proba = _simplex_proba(rng, len(labels))
        out = oof_permutation_pvalue(
            labels, proba, np.zeros(len(labels), dtype=int),
            groups=genes, n_permutations=50, seed=0,
        )
        assert set(out["classes_scored"]) == set(MECHANISM_CLASSES)
        assert out["n_permutations"] + out["n_dropped_class_mismatch"] <= 50
        assert out["n_dropped_class_mismatch"] >= 0


# ---------------------------------------------------------------------------
# paired_oof_diff
# ---------------------------------------------------------------------------

def _mechanism_oof(row_ids, genes, y_true, proba, folds=None):
    return {
        "row_ids": np.asarray(row_ids),
        "genes": np.asarray(genes, dtype=object),
        "y_true": np.asarray(y_true),
        "proba": np.asarray(proba, dtype=float),
        "folds": _folds_by_unit(genes) if folds is None else np.asarray(folds),
    }


def _confident_proba(labels, classes, correct_mask):
    """Proba that predicts the true label where correct_mask, else the next class."""
    rows = []
    for label, correct in zip(labels, correct_mask):
        target = classes.index(label)
        if not correct:
            target = (target + 1) % len(classes)
        vec = np.full(len(classes), 0.01)
        vec[target] = 1.0
        rows.append(vec / vec.sum())
    return np.array(rows)


class TestPairedOofDiff:
    """paired_oof_diff aligns two arms by row_ids and pairs one resample across both."""

    def _arms(self, n=120, n_genes=20, n_families=5, a_correct=1.0, b_correct=0.5,
              seed=0, correctness_by_family=False):
        rng = np.random.RandomState(seed)
        row_ids = np.arange(n)
        genes = np.array([f"G{i % n_genes}" for i in row_ids], dtype=object)
        pfam_map = {f"G{i}": f"F{i % n_families}" for i in range(n_genes)}
        classes = list(MECHANISM_CLASSES)
        y_true = np.array([classes[i % len(classes)] for i in row_ids])
        if correctness_by_family:
            # Rows in a family succeed or fail together, which is what makes the
            # family the real independent unit. With row-independent errors the two
            # resampling units carry almost the same variance and the contrast
            # between them is not there to measure.
            families = np.array([pfam_map[g] for g in genes])
            unique_families = np.unique(families)
            fam_a = {f: rng.rand() < a_correct for f in unique_families}
            fam_b = {f: rng.rand() < b_correct for f in unique_families}
            mask_a = np.array([fam_a[f] for f in families])
            mask_b = np.array([fam_b[f] for f in families])
        else:
            mask_a = rng.rand(n) < a_correct
            mask_b = rng.rand(n) < b_correct
        arm_a = _mechanism_oof(row_ids, genes, y_true, _confident_proba(y_true, classes, mask_a))
        arm_b = _mechanism_oof(row_ids, genes, y_true, _confident_proba(y_true, classes, mask_b))
        return arm_a, arm_b, pfam_map, classes

    def test_planted_difference_excludes_zero(self):
        arm_a, arm_b, pfam_map, classes = self._arms()
        out = paired_oof_diff(
            arm_a, arm_b, pfam_map, "planted", classes=classes, n_resamples=200
        )
        assert out["point_diff"] > 0
        assert out["ci_low"] > 0, "a large planted difference must exclude zero"
        assert out["n_shared"] == 120

    def test_identical_arms_give_exactly_zero_difference(self):
        arm_a, _, pfam_map, classes = self._arms()
        out = paired_oof_diff(
            arm_a, arm_a, pfam_map, "identical", classes=classes, n_resamples=100
        )
        # The pairing is what makes this exact: both arms see the same drawn rows on
        # every replicate, so an identical arm cancels to zero in each one.
        assert out["point_diff"] == pytest.approx(0.0)
        assert out["ci_low"] == pytest.approx(0.0)
        assert out["ci_high"] == pytest.approx(0.0)

    def test_family_split_resamples_families_not_genes(self):
        arm_a, arm_b, pfam_map, classes = self._arms(n_genes=20, n_families=5)
        family = paired_oof_diff(
            arm_a, arm_b, pfam_map, "fam", classes=classes,
            is_family_split=True, n_resamples=50,
        )
        gene = paired_oof_diff(
            arm_a, arm_b, pfam_map, "gene", classes=classes,
            is_family_split=False, n_resamples=50,
        )
        assert family["n_clusters"] == 5
        assert gene["n_clusters"] == 20

    def test_arms_aligned_by_row_id_not_position(self):
        arm_a, arm_b, pfam_map, classes = self._arms()
        shuffled = np.random.RandomState(7).permutation(len(arm_b["row_ids"]))
        reordered = {key: np.asarray(value)[shuffled] for key, value in arm_b.items()}
        straight = paired_oof_diff(
            arm_a, arm_b, pfam_map, "ordered", classes=classes, n_resamples=50
        )
        shuffled_out = paired_oof_diff(
            arm_a, reordered, pfam_map, "shuffled", classes=classes, n_resamples=50
        )
        assert shuffled_out["point_diff"] == pytest.approx(straight["point_diff"])

    def test_non_overlapping_rows_are_dropped(self):
        arm_a, arm_b, pfam_map, classes = self._arms(n=120)
        keep = slice(0, 80)
        trimmed = {key: np.asarray(value)[keep] for key, value in arm_b.items()}
        out = paired_oof_diff(
            arm_a, trimmed, pfam_map, "partial", classes=classes, n_resamples=50
        )
        assert out["n_shared"] == 80

    def test_missing_row_ids_raises(self):
        arm_a, arm_b, pfam_map, classes = self._arms()
        no_rows = {k: v for k, v in arm_b.items() if k != "row_ids"}
        # Equal-length arrays are not evidence of alignment: two arms can drop
        # different folds and still match in length, so this must not fall back to
        # positional pairing.
        with pytest.raises(KeyError, match="row_ids"):
            paired_oof_diff(arm_a, no_rows, pfam_map, "no-rows", classes=classes)

    def test_disagreeing_labels_on_a_shared_row_raises(self):
        arm_a, arm_b, pfam_map, classes = self._arms()
        corrupted = {key: np.array(value, copy=True) for key, value in arm_b.items()}
        corrupted["y_true"][0] = "GOF" if corrupted["y_true"][0] != "GOF" else "LOF"
        with pytest.raises(RuntimeError, match="disagree on the label"):
            paired_oof_diff(arm_a, corrupted, pfam_map, "corrupt", classes=classes)

    def test_missing_arm_returns_none(self):
        arm_a, _, pfam_map, classes = self._arms()
        assert paired_oof_diff(arm_a, None, pfam_map, "none", classes=classes) is None
        assert paired_oof_diff(None, arm_a, pfam_map, "none", classes=classes) is None

    def test_macro_f1_requires_classes(self):
        arm_a, arm_b, pfam_map, _ = self._arms()
        with pytest.raises(ValueError, match="requires the `classes` list"):
            paired_oof_diff(arm_a, arm_b, pfam_map, "no-classes")

    def test_one_vs_rest_isolates_the_named_class(self):
        # An arm that is perfect on DN and random elsewhere must show a large DN
        # difference and near-zero differences on the other classes.
        n, n_genes = 300, 40
        rng = np.random.RandomState(11)
        classes = list(MECHANISM_CLASSES)
        row_ids = np.arange(n)
        genes = np.array([f"G{i % n_genes}" for i in row_ids], dtype=object)
        pfam_map = {f"G{i}": f"F{i % 8}" for i in range(n_genes)}
        y_true = np.array([classes[i % len(classes)] for i in row_ids])

        dn_col = classes.index(DN)
        strong = rng.rand(n, len(classes))
        strong[:, dn_col] = np.where(y_true == DN, 0.9, 0.1)
        weak = rng.rand(n, len(classes))

        arm_a = _mechanism_oof(row_ids, genes, y_true, strong)
        arm_b = _mechanism_oof(row_ids, genes, y_true, weak)

        dn = paired_oof_diff(
            arm_a, arm_b, pfam_map, "dn", classes=classes,
            metric="auroc_one_vs_rest", pos_class=DN, n_resamples=200,
        )
        assert dn["point_a"] > 0.95
        assert dn["ci_low"] > 0

    def test_one_vs_rest_requires_a_pos_class_in_classes(self):
        arm_a, arm_b, pfam_map, classes = self._arms()
        # Selecting the proba column by a name not in `classes` would silently read
        # the wrong class's column.
        with pytest.raises(ValueError, match="not in classes"):
            paired_oof_diff(
                arm_a, arm_b, pfam_map, "bad-class", classes=classes,
                metric="auroc_one_vs_rest", pos_class="NOT_A_CLASS",
            )
        with pytest.raises(ValueError, match="requires the `classes` list"):
            paired_oof_diff(
                arm_a, arm_b, pfam_map, "no-classes",
                metric="auroc_one_vs_rest", pos_class=DN,
            )

    def test_unknown_metric_raises(self):
        arm_a, arm_b, pfam_map, classes = self._arms()
        with pytest.raises(ValueError, match="unknown metric"):
            paired_oof_diff(
                arm_a, arm_b, pfam_map, "bad", classes=classes, metric="rmse"
            )

    def test_classes_parameter_is_used_not_a_module_constant(self):
        # A caller with a 2-class arm must not be silently scored against the
        # 3-class MECHANISM_CLASSES.
        n, n_genes = 60, 10
        row_ids = np.arange(n)
        genes = np.array([f"G{i % n_genes}" for i in row_ids], dtype=object)
        pfam_map = {f"G{i}": f"F{i % 3}" for i in range(n_genes)}
        classes = ["benign", "pathogenic"]
        y_true = np.array([classes[i % 2] for i in row_ids])
        arm_a = _mechanism_oof(
            row_ids, genes, y_true, _confident_proba(y_true, classes, np.ones(n, bool))
        )
        arm_b = _mechanism_oof(
            row_ids, genes, y_true,
            _confident_proba(y_true, classes, np.zeros(n, bool)),
        )
        out = paired_oof_diff(
            arm_a, arm_b, pfam_map, "2class", classes=classes, n_resamples=50
        )
        assert out["point_a"] == pytest.approx(1.0)
        assert out["point_b"] == pytest.approx(0.0)

    def test_auroc_binary_metric(self):
        n, n_genes = 100, 20
        rng = np.random.RandomState(2)
        row_ids = np.arange(n)
        genes = np.array([f"G{i % n_genes}" for i in row_ids], dtype=object)
        pfam_map = {f"G{i}": f"F{i % 4}" for i in range(n_genes)}
        y_true = np.array([0] * (n // 2) + [1] * (n // 2))
        perfect = _mechanism_oof(row_ids, genes, y_true, y_true.astype(float))
        noise = _mechanism_oof(row_ids, genes, y_true, rng.rand(n))
        out = paired_oof_diff(
            perfect, noise, pfam_map, "auroc", metric="auroc_binary", n_resamples=200
        )
        assert out["point_a"] == pytest.approx(1.0)
        assert out["point_diff"] > 0
        assert out["ci_low"] > 0

    def test_cross_partition_resamples_families_and_adds_gene_sensitivity(self):
        arm_a, arm_b, pfam_map, classes = self._arms(n_genes=20, n_families=5)
        out = paired_oof_diff(
            arm_a, arm_b, pfam_map, "gap", classes=classes,
            cross_partition=True, n_resamples=50,
        )
        # Pre-registration §1.2: the primary interval resamples the coarser unit (families); the
        # gene-resampled one rides alongside as a labelled sensitivity check.
        assert out["n_clusters"] == 5
        assert out["gene_resampled_sensitivity"]["n_clusters"] == 20

    def test_cross_partition_family_interval_is_wider_than_gene(self):
        arm_a, arm_b, pfam_map, classes = self._arms(
            n=400, n_genes=40, n_families=8, a_correct=0.8, b_correct=0.6,
            correctness_by_family=True,
        )
        out = paired_oof_diff(
            arm_a, arm_b, pfam_map, "gap", classes=classes,
            cross_partition=True, n_resamples=400,
        )
        family_width = out["ci_high"] - out["ci_low"]
        gene = out["gene_resampled_sensitivity"]
        gene_width = gene["ci_high"] - gene["ci_low"]
        # Fewer effective clusters means a wider interval. A gene-resampled gap
        # understates the family-split arm's variance, which is why it is only ever
        # reported as a sensitivity check.
        assert family_width > gene_width


# ---------------------------------------------------------------------------
# adjudicate_diff / adjudicate_level (pre-registration §1.1 verdicts)
# ---------------------------------------------------------------------------

class TestAdjudicateDiff:
    def test_pass_with_ci_above_zero_is_established(self):
        ci = {"ci_low": 0.01, "ci_high": 0.05}
        assert "established" in adjudicate_diff(True, ci, 0.03)

    def test_pass_with_ci_spanning_zero_is_not_distinguishable(self):
        ci = {"ci_low": -0.01, "ci_high": 0.05}
        assert adjudicate_diff(True, ci, 0.03) == (
            "pass on point estimate, not distinguishable (CI spans zero)"
        )

    def test_fail_with_ci_spanning_threshold_is_underpowered(self):
        # A failing gate whose CI still reaches the pre-registered effect size is
        # underpowered, never evidence of no effect.
        ci = {"ci_low": -0.02, "ci_high": 0.04}
        assert "underpowered" in adjudicate_diff(False, ci, 0.03)

    def test_fail_with_ci_ending_at_threshold_is_underpowered(self):
        ci = {"ci_low": -0.02, "ci_high": 0.03}
        assert "underpowered" in adjudicate_diff(False, ci, 0.03)

    def test_fail_with_ci_below_threshold_is_established(self):
        ci = {"ci_low": -0.02, "ci_high": 0.01}
        assert adjudicate_diff(False, ci, 0.03) == (
            "fail, established (CI excludes the pre-registered threshold)"
        )

    def test_suppressed_or_absent_ci_reports_no_ci(self):
        assert adjudicate_diff(True, None, 0.03) == "pass, no CI"
        assert adjudicate_diff(False, {"ci_suppressed": True}, 0.03) == "fail, no CI"

    def test_no_point_estimate_is_not_adjudicated(self):
        assert adjudicate_diff(None, {"ci_low": 0.1, "ci_high": 0.2}, 0.03) == (
            "not adjudicated (no point estimate)"
        )


class TestAdjudicateLevel:
    def test_clearing_threshold_with_clear_interval_is_established(self):
        assert "established" in adjudicate_level(0.891, {"ci_low": 0.86, "ci_high": 0.92}, 0.85)

    def test_clearing_threshold_with_covering_interval_is_not_distinguishable(self):
        # An interval that still covers 0.85 has not established the level, however
        # far the point estimate sits above it.
        assert adjudicate_level(0.891, {"ci_low": 0.84, "ci_high": 0.93}, 0.85) == (
            "pass on point estimate, not distinguishable (CI covers the threshold)"
        )

    def test_below_threshold_with_covering_interval_is_underpowered(self):
        assert "underpowered" in adjudicate_level(0.83, {"ci_low": 0.79, "ci_high": 0.88}, 0.85)

    def test_interval_ending_at_threshold_still_covers_it(self):
        assert adjudicate_level(0.83, {"ci_low": 0.79, "ci_high": 0.85}, 0.85) == (
            "fail, underpowered (CI covers the threshold)"
        )

    def test_below_threshold_with_clear_interval_is_established(self):
        assert adjudicate_level(0.80, {"ci_low": 0.77, "ci_high": 0.83}, 0.85) == (
            "fail, established (CI excludes the threshold)"
        )

    def test_nan_or_missing_value_is_not_adjudicated(self):
        assert adjudicate_level(None, {"ci_low": 0.8, "ci_high": 0.9}, 0.85) == (
            "not adjudicated (no point estimate)"
        )
        assert adjudicate_level(float("nan"), {"ci_low": 0.8, "ci_high": 0.9}, 0.85) == (
            "not adjudicated (no point estimate)"
        )

    def test_suppressed_ci_reports_no_ci(self):
        assert adjudicate_level(0.9, {"ci_suppressed": True}, 0.85) == "pass, no CI"


class TestAdjudicateEquivalence:
    def test_pass_ci_within_band_is_established(self):
        ci = {"ci_low": -0.02, "ci_high": 0.03}
        assert "established" in adjudicate_equivalence(True, ci, 0.05)

    def test_pass_ci_exceeding_band_is_not_established(self):
        # Point estimate within ±0.05 but CI extends past the band.
        ci = {"ci_low": 0.01, "ci_high": 0.08}
        result = adjudicate_equivalence(True, ci, 0.05)
        assert "not established" in result

    def test_fail_ci_outside_band_is_established(self):
        ci = {"ci_low": 0.06, "ci_high": 0.10}
        assert "established" in adjudicate_equivalence(False, ci, 0.05)

    def test_fail_ci_overlapping_band_is_underpowered(self):
        ci = {"ci_low": -0.02, "ci_high": 0.02}
        # Point estimate outside ±0.05 but CI overlaps.
        result = adjudicate_equivalence(False, ci, 0.05)
        assert "underpowered" in result

    def test_suppressed_or_absent_ci_reports_no_ci(self):
        assert adjudicate_equivalence(True, None, 0.05) == "pass, no CI"
        assert adjudicate_equivalence(False, {"ci_suppressed": True}, 0.05) == "fail, no CI"

    def test_no_point_estimate_is_not_adjudicated(self):
        assert adjudicate_equivalence(None, {"ci_low": -0.01, "ci_high": 0.01}, 0.05) == (
            "not adjudicated (no point estimate)"
        )


class TestIndependentClusterBootstrapDiff:
    def test_resamples_each_dataset_at_its_own_cluster_level(self):
        clusters_a = np.repeat(["A1", "A2", "A3", "A4"], [2, 3, 4, 5])
        clusters_b = np.repeat(["B1", "B2", "B3"], [3, 4, 5])
        values_a = np.linspace(0.6, 1.0, len(clusters_a))
        values_b = np.linspace(0.0, 0.4, len(clusters_b))

        out = independent_cluster_bootstrap_diff(
            clusters_a,
            clusters_b,
            lambda rows: float(values_a[rows].mean()),
            lambda rows: float(values_b[rows].mean()),
            n_resamples=300,
            seed=7,
        )

        assert out["n_clusters_a"] == 4
        assert out["n_clusters_b"] == 3
        assert out["point_diff"] == pytest.approx(values_a.mean() - values_b.mean())
        assert out["ci_low"] > 0

    def test_is_deterministic_for_a_fixed_seed(self):
        clusters_a = np.repeat(np.arange(5), 2)
        clusters_b = np.repeat(np.arange(4), 3)
        values_a = np.arange(len(clusters_a), dtype=float)
        values_b = np.arange(len(clusters_b), dtype=float)
        kwargs = dict(n_resamples=100, seed=11)

        first = independent_cluster_bootstrap_diff(
            clusters_a,
            clusters_b,
            lambda rows: float(values_a[rows].mean()),
            lambda rows: float(values_b[rows].mean()),
            **kwargs,
        )
        second = independent_cluster_bootstrap_diff(
            clusters_a,
            clusters_b,
            lambda rows: float(values_a[rows].mean()),
            lambda rows: float(values_b[rows].mean()),
            **kwargs,
        )
        assert first == second


class TestPairedClusterBootstrapDiffSharedClusters:
    def test_pairs_shared_clusters_across_different_row_spaces(self):
        clusters_a = np.array(["F1", "F2", "F2", "F3", "F3"])
        clusters_b = np.array(["F2", "F3", "F4"])
        family_value = {"F1": 1.0, "F2": 2.0, "F3": 3.0, "F4": 4.0}
        values_a = np.array([family_value[family] + 1.0 for family in clusters_a])
        values_b = np.array([family_value[family] for family in clusters_b])

        out = paired_cluster_bootstrap_diff_shared_clusters(
            clusters_a,
            clusters_b,
            lambda rows: float(values_a[rows].mean()),
            lambda rows: float(values_b[rows].mean()),
            n_resamples=100,
            seed=7,
        )

        assert out["n_clusters_a_total"] == 3
        assert out["n_clusters_b_total"] == 3
        assert out["n_clusters_shared"] == 2
        assert out["n_rows_a_shared"] == 4
        assert out["n_rows_b_shared"] == 2
        assert out["point_diff"] == pytest.approx(1.0)
        assert out["ci_low"] == pytest.approx(1.0)
        assert out["ci_high"] == pytest.approx(1.0)

    def test_rejects_disjoint_cluster_populations(self):
        with pytest.raises(ValueError, match="no clusters shared"):
            paired_cluster_bootstrap_diff_shared_clusters(
                np.array(["F1"]),
                np.array(["F2"]),
                lambda rows: 1.0,
                lambda rows: 0.0,
                n_resamples=10,
            )


# ---------------------------------------------------------------------------
# fold-aware ranking (the defect this module was fixed for)
# ---------------------------------------------------------------------------

class TestRankingIsFoldAware:
    """Per-fold probability scales must not leak into a ranking metric.

    Each fold is fitted on its own, so its scores carry their own offset. Within any
    one fold the ranking here is perfect; ranking the concatenation instead makes
    fold 1's worst case outscore fold 0's best. A pooled implementation returns
    roughly 0.5 on this data and a fold-aware one returns 1.0.
    """

    def _offset_folds(self):
        n_per_fold = 40
        folds = np.array([0] * n_per_fold + [1] * n_per_fold)
        y_bin = np.array(([0] * (n_per_fold // 2) + [1] * (n_per_fold // 2)) * 2)
        # Fold 0 scores live in [0.0, 0.2]; fold 1's in [0.8, 1.0]. Positives rank
        # above negatives inside each fold and nowhere near it across the two.
        scores = np.where(y_bin == 1, 0.15, 0.05).astype(float)
        scores[folds == 1] = np.where(y_bin[folds == 1] == 1, 0.95, 0.85)
        genes = np.array([f"G{i}" for i in range(len(y_bin))], dtype=object)
        return y_bin, scores, folds, genes

    def test_binary_auroc_scores_within_fold(self):
        y_bin, scores, folds, genes = self._offset_folds()
        # Fold 1's negatives outrank fold 0's positives, so pooling inverts half the
        # comparisons; scoring inside each fold keeps the planted order.
        out = binary_auroc_cluster_bootstrap_ci(
            {"y_true": y_bin, "proba": scores, "genes": genes, "folds": folds},
            n_resamples=50,
        )
        assert out["point"] == pytest.approx(1.0)

    def test_an_oof_without_folds_is_refused(self):
        y_bin, scores, _, genes = self._offset_folds()
        with pytest.raises(KeyError, match="folds"):
            binary_auroc_cluster_bootstrap_ci(
                {"y_true": y_bin, "proba": scores, "genes": genes}, n_resamples=10
            )

    def test_multiclass_auroc_scores_within_fold(self):
        y_bin, scores, folds, genes = self._offset_folds()
        y_true = np.where(y_bin == 1, GOF, LOF)
        proba = np.zeros((len(y_true), len(MECHANISM_CLASSES)))
        proba[:, MECHANISM_CLASSES.index(GOF)] = scores
        proba[:, MECHANISM_CLASSES.index(LOF)] = 1.0 - scores
        out = bootstrap_mechanism_metrics(y_true, proba, genes, folds, n_resamples=50)
        assert out["auroc_GOF"]["point"] == pytest.approx(1.0)
