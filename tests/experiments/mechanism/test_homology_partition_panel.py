"""
Tests for esm2_mech.experiments.mechanism.homology_partition_panel (Task 2b).

Covers:
- leakage_fraction_ci_for_partition: the LF ratio formula matches a hand-computed
  value on shared rows, records the partition's own cluster count, and returns
  None when the two arms have no shared rows.
- _partition_row: n_clusters in the returned mechanism-null CI matches the number
  of distinct partition-unit ids passed in, so a clan row reports the clan count
  and an mmseqs row reports the cluster count, never a gene/family count.
- _measured_chance_floors / _live_mlp_py_family_reference: read live from a result
  file rather than a hardcoded literal, and return None (no fallback) when the
  source file is absent.
- An end-to-end (marked slow) run of the three-row pipeline on a tiny synthetic
  dataset, asserting all three rows are present with gated intervals and the
  correct effective cluster count per row.

Also: a regression test for the pre-existing mmseqs_cluster_holdout.py bug this
task fixed (int-encoded y fed to run_mlp_cv, which expects labels matching the
MECHANISM_CLASSES strings — crashed with "need at least one array to concatenate"
the moment run_mlp_cv tried to build the oversampling index).

Classification intervals remain suppressed under audit item 1.4.
"""

import json

import numpy as np
import pytest
from sklearn.metrics import f1_score

from esm2_mech.experiments.mechanism import clan_holdout, homology_partition_panel as panel
from esm2_mech.utils.constants import DN, GOF, LOF, MECHANISM_CLASSES
from esm2_mech.utils.metrics import majority_baseline_f1


def _one_hot_proba(labels, classes=MECHANISM_CLASSES):
    """Proba matrix that predicts each row's own label with probability 1."""
    col = {cls: i for i, cls in enumerate(classes)}
    proba = np.zeros((len(labels), len(classes)), dtype=np.float64)
    for row, label in enumerate(labels):
        proba[row, col[label]] = 1.0
    return proba


class TestLeakageFractionCiForPartition:

    def test_ratio_matches_hand_computed_value(self):
        # 10 shared rows. Gene arm predicts perfectly (gene_f1 = 1.0). Partition
        # arm gets the last 4 rows wrong (predicts LOF for a GOF/DN true label).
        y_true = np.array([LOF] * 6 + [GOF, GOF, DN, DN])
        row_ids = np.arange(10)
        folds = np.zeros(len(y_true), dtype=int)
        oof_gene = {
            "y_true": y_true,
            "proba": _one_hot_proba(y_true),
            "row_ids": row_ids,
            "folds": folds,
        }

        part_pred_labels = y_true.copy()
        part_pred_labels[6:] = LOF  # last 4 rows wrong
        oof_partition = {
            "y_true": y_true,
            "proba": _one_hot_proba(part_pred_labels),
            "row_ids": row_ids,
            "folds": folds,
        }
        partition_clusters = np.array(["clanA"] * 5 + ["clanB"] * 5)

        chance, _ = majority_baseline_f1(y_true, y_true, MECHANISM_CLASSES)
        ci = panel.leakage_fraction_ci_for_partition(
            oof_gene,
            oof_partition,
            partition_clusters,
            gene_chance=chance,
            n_boot=30,
            seed=0,
        )

        gene_f1 = f1_score(y_true, y_true, average="macro", zero_division=0)  # 1.0
        part_f1 = f1_score(y_true, part_pred_labels, average="macro", zero_division=0)
        expected = (gene_f1 - part_f1) / (gene_f1 - chance)

        assert ci is not None
        assert ci["point"] == pytest.approx(expected)

    def test_resamples_partition_cluster_not_row_count(self):
        # 12 rows collapsed into exactly 3 partition clusters (4 rows each) —
        # n_clusters in the returned CI must be 3, not 12 (rows) and not
        # whatever a gene id count would have been.
        y_true = np.array([LOF, GOF, DN] * 4)
        row_ids = np.arange(12)
        folds = np.zeros(len(y_true), dtype=int)
        oof_gene = {"y_true": y_true, "proba": _one_hot_proba(y_true), "row_ids": row_ids, "folds": folds}
        oof_partition = {"y_true": y_true, "proba": _one_hot_proba(y_true), "row_ids": row_ids, "folds": folds}
        partition_clusters = np.array(["c0", "c1", "c2"] * 4)

        ci = panel.leakage_fraction_ci_for_partition(
            oof_gene, oof_partition, partition_clusters,
            gene_chance=0.30, n_boot=20, seed=0
        )
        # gene_f1 == part_f1 == 1.0 here so LF is 0 everywhere but n_clusters
        # must still reflect the distinct partition ids, not the row count.
        assert ci["n_clusters"] == 3

    def test_undefined_when_gene_arm_not_above_chance(self):
        y_true = np.array([LOF] * 8 + [GOF] * 2)
        row_ids = np.arange(10)
        # Gene arm predicts everything as LOF -> low macro-F1, at/near chance.
        gene_pred = np.full(10, LOF)
        folds = np.zeros(len(y_true), dtype=int)
        oof_gene = {"y_true": y_true, "proba": _one_hot_proba(gene_pred), "row_ids": row_ids, "folds": folds}
        oof_partition = {"y_true": y_true, "proba": _one_hot_proba(y_true), "row_ids": row_ids, "folds": folds}
        partition_clusters = np.array(["c0"] * 5 + ["c1"] * 5)

        # Gene arm predicts everything as LOF. The majority class IS LOF (8/10),
        # so gene_f1 equals the chance floor and denom <= MIN_ABOVE_CHANCE.
        ci = panel.leakage_fraction_ci_for_partition(
            oof_gene, oof_partition, partition_clusters,
            gene_chance=0.2962962962962963, n_boot=20, seed=0
        )
        assert ci["ci_suppressed"] is True

    def test_no_shared_rows_returns_none(self):
        y_true = np.array([LOF, GOF, DN])
        oof_gene = {"y_true": y_true, "proba": _one_hot_proba(y_true), "row_ids": np.array([0, 1, 2])}
        oof_partition = {"y_true": y_true, "proba": _one_hot_proba(y_true), "row_ids": np.array([10, 11, 12])}
        result = panel.leakage_fraction_ci_for_partition(
            oof_gene, oof_partition, np.array(["c0"] * 3),
            gene_chance=0.2, n_boot=10, seed=0
        )
        assert result is None


class TestPartitionRow:

    def test_n_clusters_matches_distinct_partition_units(self):
        # 20 rows spread over 4 distinct clan ids (5 rows each) — the row's own
        # n_clusters must be 4 (clans), regardless of the fact there are 20 rows.
        rng = np.random.RandomState(0)
        y_true = np.array(rng.choice(MECHANISM_CLASSES, size=20))
        row_ids = np.arange(20)
        proba = _one_hot_proba(y_true)
        folds = np.zeros(len(y_true), dtype=int)
        oof_gene = {"y_true": y_true, "proba": proba, "row_ids": row_ids, "folds": folds}
        oof_partition = {"y_true": y_true, "proba": proba, "row_ids": row_ids, "folds": folds}
        clan_ids = np.array([f"clan{i % 4}" for i in range(20)])

        row = panel._partition_row(
            "pfam_clan", oof_gene, oof_partition, clan_ids,
            gene_chance=0.3, family_chance=0.288, n_boot=20, seed=0,
        )
        assert row["partition"] == "pfam_clan"
        assert row["n_clusters"] == 4
        assert row["measured_floor"] == 0.288
        assert row["mechanism_null_macro_f1"]["ci_suppressed"] is True
        assert row["mechanism_null_macro_f1"]["reason"] == "blocked_by_audit_1_4"


class TestMeasuredChanceFloors:

    def test_reads_live_values_not_hardcoded(self, tmp_path, monkeypatch):
        naive_baseline = tmp_path / "naive_baseline.json"
        with open(naive_baseline, "w") as f:
            json.dump(
                {
                    "by_strategy": {
                        "most_frequent": {
                            "gene": {"macro_f1_mean": 0.1234},
                            "family": {"macro_f1_mean": 0.5678},
                        }
                    }
                },
                f,
            )
        monkeypatch.setattr(panel, "NAIVE_BASELINE_JSON", naive_baseline)

        gene_chance, family_chance = panel._measured_chance_floors()
        assert gene_chance == pytest.approx(0.1234)
        assert family_chance == pytest.approx(0.5678)


class TestLiveMlpPyFamilyReference:

    def test_reads_live_value(self, tmp_path, monkeypatch):
        seed_path = tmp_path / "nonlinear_results_seed{seed}.json"
        monkeypatch.setattr(panel, "NONLINEAR_RESULTS_SEED_JSON", str(seed_path))
        with open(str(seed_path).format(seed=0), "w") as f:
            json.dump({"mlp_delta_mean_family": {"macro_f1_mean": 0.4242}}, f)

        assert panel._live_mlp_py_family_reference(0) == pytest.approx(0.4242)

    def test_missing_file_returns_none_no_fallback(self, tmp_path, monkeypatch):
        seed_path = tmp_path / "nonlinear_results_seed{seed}.json"
        monkeypatch.setattr(panel, "NONLINEAR_RESULTS_SEED_JSON", str(seed_path))
        assert panel._live_mlp_py_family_reference(0) is None


class TestClanHoldoutLiveFamilySplitRefs:
    """clan_holdout.py's stale-literal fix (Task 2b item 3): the old
    family_split_mlp_f1_result7=0.352 / family_split_contrastive_f1_result9=0.387
    hardcoded numbers are replaced by a live read from the result files the rest
    of the project cites."""

    def test_reads_live_mlp_and_contrastive_values(self, tmp_path, monkeypatch):
        mlp_seed_path = tmp_path / "nonlinear_results_seed{seed}.json"
        with open(str(mlp_seed_path).format(seed=0), "w") as f:
            json.dump({"mlp_delta_mean_family": {"macro_f1_mean": 0.4111}}, f)
        monkeypatch.setattr(clan_holdout, "NONLINEAR_RESULTS_SEED_JSON", str(mlp_seed_path))

        contrastive_path = tmp_path / "contrastive_aggregate.json"
        with open(contrastive_path, "w") as f:
            json.dump(
                {"across_seed": {"family_split": {"contrastive_knn": {"macro_f1_seed_mean": 0.4222}}}},
                f,
            )
        monkeypatch.setattr(clan_holdout, "CONTRASTIVE_AGGREGATE_JSON", contrastive_path)

        mlp_f1, contrastive_f1 = clan_holdout._read_live_family_split_refs()
        assert mlp_f1 == pytest.approx(0.4111)
        assert contrastive_f1 == pytest.approx(0.4222)

    def test_missing_files_return_none_no_hardcoded_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            clan_holdout, "NONLINEAR_RESULTS_SEED_JSON", str(tmp_path / "missing_seed{seed}.json")
        )
        monkeypatch.setattr(clan_holdout, "CONTRASTIVE_AGGREGATE_JSON", tmp_path / "missing.json")

        mlp_f1, contrastive_f1 = clan_holdout._read_live_family_split_refs()
        # No fallback/default literal (not 0.352 / 0.387) — genuinely absent.
        assert mlp_f1 is None
        assert contrastive_f1 is None
        assert mlp_f1 != 0.352
        assert contrastive_f1 != 0.387


class TestMmseqsClusterHoldoutLabelEncodingRegression:
    """Regression test for the bug this task fixed: mmseqs_cluster_holdout.py
    used to int-encode y before calling run_mlp (a thin wrapper over
    utils.probes.run_mlp_cv), but run_mlp_cv compares y against the string
    `classes` (default MECHANISM_CLASSES) to build its oversampling index —
    an int-encoded y produced zero rows for every class and crashed with
    "need at least one array to concatenate" on the first fold. y must be the
    original string labels."""

    def test_run_mlp_does_not_crash_on_string_labels(self):
        from esm2_mech.experiments.mechanism.mmseqs_cluster_holdout import run_mlp

        rng = np.random.RandomState(0)
        n = 60
        delta = rng.normal(size=(n, 8)).astype(np.float32)
        labels = np.array(rng.choice(MECHANISM_CLASSES, size=n))
        genes = np.array([f"g{i}" for i in range(n)])
        # 6 clusters of 10 rows each.
        groups = np.array([f"cluster{i % 6}" for i in range(n)])

        agg, oof = run_mlp(
            delta, labels, genes, groups, hidden=(8, 4), n_folds=3, seed=0,
            label="test", return_oof=True,
        )
        assert "error" not in agg
        assert oof is not None
        assert set(np.unique(oof["y_true"])) <= set(MECHANISM_CLASSES)


@pytest.mark.slow
class TestPanelEndToEnd:
    """Runs the full family/clan/mmseqs pipeline on a tiny synthetic dataset —
    small enough to fit fast, large enough to give clan_holdout.py's
    leave-one-clan-out at least one qualifying clan (its hardcoded threshold:
    the 2nd-most-frequent class in the held-out clan has >= 20 variants and the
    clan spans >= 3 genes)."""

    @staticmethod
    def _build_dataset():
        rng = np.random.RandomState(0)
        genes, labels = [], []

        # Qualifying clan "CLA" = families PF1 (g1,g2,g3) + PF2 (g4,g5,g6).
        # class dist across the clan: LOF=40, GOF=25, DN=10 (2nd-most, GOF=25 >= 20).
        cla_genes = ["g1", "g2", "g3", "g4", "g5", "g6"]
        cla_dist = {LOF: 40, GOF: 25, DN: 10}
        for label, count in cla_dist.items():
            for i in range(count):
                genes.append(cla_genes[i % len(cla_genes)])
                labels.append(label)

        # Non-qualifying clan "CLB" = families PF3 (g7,g8) + PF4 (g9,g10): small,
        # mostly LOF, well under the 20-variant second-class threshold.
        clb_genes = ["g7", "g8", "g9", "g10"]
        clb_dist = {LOF: 16, GOF: 3, DN: 1}
        for label, count in clb_dist.items():
            for i in range(count):
                genes.append(clb_genes[i % len(clb_genes)])
                labels.append(label)

        genes = np.array(genes)
        labels = np.array(labels)
        n = len(genes)
        delta = rng.normal(size=(n, 6)).astype(np.float32)
        # Make the delta feature weakly informative so folds aren't degenerate.
        for row in range(n):
            col = MECHANISM_CLASSES.index(labels[row])
            delta[row, col] += 1.5

        pfam_map = {
            "g1": "PF1", "g2": "PF1", "g3": "PF1",
            "g4": "PF2", "g5": "PF2", "g6": "PF2",
            "g7": "PF3", "g8": "PF3",
            "g9": "PF4", "g10": "PF4",
        }
        clan_map = {"PF1": "CLA", "PF2": "CLA", "PF3": "CLB", "PF4": "CLB"}
        clan_names = {"CLA": "Clan A", "CLB": "Clan B"}
        gene_clan = {g: clan_map[acc] for g, acc in pfam_map.items()}
        # MMseqs clusters: a partition independent of family/clan — pairs of
        # genes grouped differently than the Pfam families above.
        mmseqs_pairs = [
            ("g1", "g4"), ("g2", "g5"), ("g3", "g6"),
            ("g7", "g9"), ("g8", "g10"),
        ]
        gene_to_cluster = {}
        for cluster_id, (a, b) in enumerate(mmseqs_pairs):
            gene_to_cluster[a] = f"mc{cluster_id}"
            gene_to_cluster[b] = f"mc{cluster_id}"

        return labels, genes, delta, pfam_map, gene_clan, clan_names, gene_to_cluster

    def test_all_three_rows_present_with_correct_cluster_counts(self, tmp_path, monkeypatch):
        (labels, genes, delta, pfam_map, gene_clan, clan_names, gene_to_cluster) = self._build_dataset()

        naive_baseline = tmp_path / "naive_baseline.json"
        with open(naive_baseline, "w") as f:
            json.dump(
                {
                    "by_strategy": {
                        "most_frequent": {
                            "gene": {"macro_f1_mean": 0.30},
                            "family": {"macro_f1_mean": 0.28},
                        }
                    }
                },
                f,
            )
        monkeypatch.setattr(panel, "NAIVE_BASELINE_JSON", naive_baseline)
        monkeypatch.setattr(
            panel, "NONLINEAR_RESULTS_SEED_JSON", str(tmp_path / "missing_seed{seed}.json")
        )

        seed, n_boot, n_folds = 0, 30, 2

        gene_chance, family_chance = panel._measured_chance_floors()
        oof_gene, family_row = panel.run_family_row(
            labels, genes, delta, pfam_map, seed, n_boot, n_folds=n_folds
        )
        clan_row = panel.run_clan_row(
            labels, genes, delta, gene_clan, clan_names, oof_gene,
            gene_chance, family_chance, seed, n_boot,
        )
        mmseqs_row = panel.run_mmseqs_row(
            labels, genes, delta, gene_to_cluster, oof_gene,
            gene_chance, family_chance, seed, n_boot, n_folds=n_folds,
        )

        rows = {row["partition"]: row for row in (family_row, clan_row, mmseqs_row) if row is not None}
        assert panel.PARTITION_FAMILY in rows
        assert panel.PARTITION_CLAN in rows
        assert panel.PARTITION_MMSEQS in rows

        # Effective cluster counts must match each partition's OWN unit: 4
        # distinct Pfam families, 1 qualifying clan (CLB doesn't clear the
        # leave-one-clan-out threshold), 5 distinct MMseqs2 clusters — never
        # the gene count (10) reused across rows.
        assert rows[panel.PARTITION_FAMILY]["n_clusters"] == 4
        assert rows[panel.PARTITION_CLAN]["n_clusters"] == 1
        assert rows[panel.PARTITION_MMSEQS]["n_clusters"] == 5

        for name, row in rows.items():
            null_ci = row["mechanism_null_macro_f1"]
            assert null_ci["ci_suppressed"] is True, f"{name} CI was not gated"
            assert null_ci["ci_low"] is None and null_ci["ci_high"] is None
            assert null_ci["reason"] == "blocked_by_audit_1_4"

        # Write and reload the consolidated JSON, matching main()'s shape.
        from esm2_mech.utils.io import atomic_write_json

        out_path = tmp_path / "panel.json"
        atomic_write_json(out_path, {"rows": list(rows.values())}, indent=2)
        with open(out_path) as f:
            reloaded = json.load(f)
        assert len(reloaded["rows"]) == 3
