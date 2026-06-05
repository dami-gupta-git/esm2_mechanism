"""
Tests for esm2_mech.utils.bootstrap (dependency-aware inference).

Invariants:
- average_oof_over_seeds: de-duplicates variants across seeds (one row per variant)
- average_oof_over_seeds: averages proba across seeds; rows stay simplex (sum to 1)
- average_oof_over_seeds: skips None entries; all-None / empty -> None
- average_oof_over_seeds: y_true and gene are carried through per row
- cluster_bootstrap_ci: point matches metric_fn on all rows; CI brackets the point
- cluster_bootstrap_ci: resamples whole clusters (n_clusters == #unique), not rows
- cluster_bootstrap_ci: undefined resamples are dropped, not imputed
- cluster_bootstrap_ci: all-undefined metric -> ci_low/ci_high None
- bootstrap_mechanism_metrics: returns macro_f1 + one CI per class; recovers GOF signal
- label_permutation_pvalue: null centers on chance; planted signal -> small p
- label_permutation_pvalue: gene-level shuffle keeps one label per gene
- _permute_labels: with groups, every gene's rows share one (permuted) label
"""

import numpy as np
import pytest

from esm2_mech.utils.bootstrap import (
    average_oof_over_seeds,
    bootstrap_mechanism_metrics,
    cluster_bootstrap_ci,
    label_permutation_pvalue,
    _permute_labels,
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

    def test_undefined_resamples_dropped(self):
        clusters = np.array([f"G{i % 4}" for i in range(20)])
        calls = {"n": 0}

        def metric(rows):
            calls["n"] += 1
            # Return None on every other call to simulate an undefined resample.
            return None if calls["n"] % 2 == 0 else 0.7

        out = cluster_bootstrap_ci(clusters, metric, n_resamples=100)
        # n_resamples in the result counts only the contributing (non-None) draws.
        assert out["n_resamples"] < 100
        assert out["ci_low"] is not None

    def test_all_undefined_gives_none_ci(self):
        clusters = np.array([f"G{i % 4}" for i in range(20)])
        out = cluster_bootstrap_ci(clusters, lambda rows: None, n_resamples=50)
        assert out["ci_low"] is None and out["ci_high"] is None
        assert out["n_resamples"] == 0

    def test_deterministic_for_fixed_seed(self):
        clusters = np.array([f"G{i % 5}" for i in range(50)])
        values = np.arange(50, dtype=float)
        fn = lambda rows: float(values[rows].mean())
        a = cluster_bootstrap_ci(clusters, fn, n_resamples=100, seed=3)
        b = cluster_bootstrap_ci(clusters, fn, n_resamples=100, seed=3)
        assert a == b


# ---------------------------------------------------------------------------
# bootstrap_mechanism_metrics
# ---------------------------------------------------------------------------

class TestBootstrapMechanismMetrics:
    def _signal_data(self, seed=0):
        rng = np.random.RandomState(seed)
        n = 120
        y = np.array([GOF, DN, LOF] * (n // 3))
        genes = np.array([f"G{i % 12}" for i in range(n)])
        # Build proba that ranks the true class high (clear, recoverable signal).
        proba = np.full((n, 3), 0.1)
        for i, cls in enumerate(y):
            proba[i, MECHANISM_CLASSES.index(cls)] = 0.8
        proba += rng.rand(n, 3) * 0.05
        proba /= proba.sum(axis=1, keepdims=True)
        return y, proba, genes

    def test_keys_macro_f1_and_per_class_auroc(self):
        y, proba, genes = self._signal_data()
        out = bootstrap_mechanism_metrics(y, proba, genes, n_resamples=100)
        assert "macro_f1" in out
        for cls in MECHANISM_CLASSES:
            assert f"auroc_{cls}" in out
            assert {"point", "ci_low", "ci_high", "n_resamples", "n_clusters"} <= set(
                out[f"auroc_{cls}"]
            )

    def test_recovers_gof_signal_above_chance(self):
        y, proba, genes = self._signal_data()
        out = bootstrap_mechanism_metrics(y, proba, genes, n_resamples=300)
        gof = out["auroc_GOF"]
        assert gof["point"] > 0.9
        assert gof["ci_low"] > 0.5  # CI excludes chance

    def test_clusters_are_genes(self):
        y, proba, genes = self._signal_data()
        out = bootstrap_mechanism_metrics(y, proba, genes, n_resamples=10)
        assert out["macro_f1"]["n_clusters"] == len(set(genes.tolist()))


# ---------------------------------------------------------------------------
# label permutation
# ---------------------------------------------------------------------------

class TestPermuteLabels:
    def test_grouped_shuffle_keeps_one_label_per_gene(self):
        rng = np.random.RandomState(0)
        genes = np.array(["G0", "G0", "G0", "G1", "G1", "G2"])
        labels = np.array([GOF, GOF, GOF, DN, DN, LOF])
        permuted = _permute_labels(labels, genes, rng)
        # Every gene's rows must still carry a single label.
        for gene in set(genes.tolist()):
            mask = genes == gene
            assert len(set(permuted[mask].tolist())) == 1

    def test_grouped_shuffle_is_a_permutation_of_gene_labels(self):
        rng = np.random.RandomState(0)
        genes = np.array(["G0", "G0", "G1", "G2", "G2"])
        labels = np.array([GOF, GOF, DN, LOF, LOF])
        permuted = _permute_labels(labels, genes, rng)
        gene_labels_in = sorted([GOF, DN, LOF])
        gene_labels_out = sorted(
            {g: permuted[genes == g][0] for g in set(genes.tolist())}.values()
        )
        assert gene_labels_out == gene_labels_in

    def test_ungrouped_shuffle_permutes_all(self):
        rng = np.random.RandomState(0)
        labels = np.array([GOF, DN, LOF, GOF, DN, LOF])
        permuted = _permute_labels(labels, None, rng)
        assert sorted(permuted.tolist()) == sorted(labels.tolist())


class TestLabelPermutationPvalue:
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
            run_metric, labels, groups=genes, n_permutations=200, alternative="greater"
        )
        assert out["observed"] == pytest.approx(1.0)  # perfect by construction
        assert out["null_mean"] == pytest.approx(0.5, abs=0.1)
        assert out["p_value"] < 0.05

    def test_pvalue_formula_bounds(self):
        rng = np.random.RandomState(2)
        genes = np.array([f"G{i // 3}" for i in range(30)])
        labels = np.array([GOF if i % 2 else LOF for i in range(30)])

        out = label_permutation_pvalue(
            lambda lab: 0.5, labels, groups=genes, n_permutations=50
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
            run_metric, labels, groups=genes, n_permutations=64, seed=7
        )
        second = label_permutation_pvalue(
            run_metric, labels, groups=genes, n_permutations=64, seed=7
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
            run_metric, labels, groups=genes, n_permutations=48, seed=1, n_jobs=1
        )
        parallel = label_permutation_pvalue(
            run_metric, labels, groups=genes, n_permutations=48, seed=1, n_jobs=2
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
            make_metric(), labels, groups=genes, n_permutations=32, seed=0, n_jobs=2
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
            flaky_metric, labels, groups=genes, n_permutations=10, seed=0, n_jobs=1
        )
        # Half the draws are NaN and dropped; the kept null has only finite values.
        assert out["n_permutations"] < 10
        assert np.isfinite(out["null_mean"])
