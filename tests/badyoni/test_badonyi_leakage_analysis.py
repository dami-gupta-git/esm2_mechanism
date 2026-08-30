"""
Tests for the Badonyi training-set leakage analysis (badonyi_leakage_analysis.py).

Invariants:
- A gene with no feature row gets NaN, never 0.0, which is a plausible real
  value for these features.
- The S3 training flags parse per classifier, and a blank cell means "not in any
  training set" rather than a dropped row.
- The IN and OUT regimes partition the family-annotated variants exactly: no
  variant is counted in both, none is silently lost, and genes without a Pfam
  family are excluded from all three regimes.
- A regime with too few variants is reported as skipped, with no metrics.
- Across-seed aggregation averages only the seeds that scored. A metric no seed
  reported is absent rather than zero, and a suppressed CI bound never enters
  the pooled interval.
- The summary table prints without error when a regime is skipped or a metric
  is unavailable.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from esm2_mech.experiments.badonyi import badonyi_leakage_analysis as leakage
from esm2_mech.experiments.badonyi.badonyi_leakage_analysis import (
    CLASSES,
    aggregate_seeds,
    broadcast,
    load_badonyi_train_flags,
    print_table,
    run_regime,
    run_seed,
)
from esm2_mech.utils.seed_aggregation import seed_result_contract


class TestBroadcast:

    def test_maps_each_gene_to_its_feature_row(self):
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        genes = np.array(["GENE_B", "GENE_A", "GENE_B"])

        broadcast_matrix = broadcast(genes, matrix, {"GENE_A": 0, "GENE_B": 1})

        assert list(broadcast_matrix[0]) == [3.0, 4.0]
        assert list(broadcast_matrix[1]) == [1.0, 2.0]
        assert list(broadcast_matrix[2]) == [3.0, 4.0]

    def test_gene_with_no_row_is_nan_not_zero(self):
        matrix = np.array([[1.0, 2.0]], dtype=np.float32)
        genes = np.array(["GENE_A", "GENE_UNKNOWN"])

        broadcast_matrix = broadcast(genes, matrix, {"GENE_A": 0})

        assert np.isnan(broadcast_matrix[1]).all()
        assert not np.isnan(broadcast_matrix[0]).any()

    def test_row_index_past_the_matrix_is_nan_not_an_index_error(self):
        matrix = np.array([[1.0, 2.0]], dtype=np.float32)
        genes = np.array(["GENE_A", "GENE_STALE"])

        broadcast_matrix = broadcast(genes, matrix, {"GENE_A": 0, "GENE_STALE": 99})

        assert np.isnan(broadcast_matrix[1]).all()


class TestLoadBadonyiTrainFlags:

    def _patch_s3(self, monkeypatch, frame):
        monkeypatch.setattr(
            leakage.pd, "read_excel", lambda *args, **kwargs: frame.copy()
        )

    def test_parses_per_classifier_flags(self, monkeypatch):
        frame = pd.DataFrame(
            {
                "gene": ["GENE_DN", "GENE_GOF", "GENE_LOF", "GENE_NONE"],
                "train_dn_gof_lof": ["1|0|0", "0|1|0", "0|0|1", "0|0|0"],
            }
        )
        self._patch_s3(monkeypatch, frame)

        any_train, per_class = load_badonyi_train_flags()

        assert per_class["DN"]["GENE_DN"] == 1
        assert per_class["GOF"]["GENE_GOF"] == 1
        assert per_class["LOF"]["GENE_LOF"] == 1
        assert per_class["DN"]["GENE_GOF"] == 0
        assert any_train == {
            "GENE_DN": 1,
            "GENE_GOF": 1,
            "GENE_LOF": 1,
            "GENE_NONE": 0,
        }

    def test_blank_cell_means_not_in_any_training_set(self, monkeypatch):
        frame = pd.DataFrame(
            {"gene": ["GENE_A", "GENE_B"], "train_dn_gof_lof": ["1|1|0", None]}
        )
        self._patch_s3(monkeypatch, frame)

        any_train, per_class = load_badonyi_train_flags()

        assert any_train["GENE_B"] == 0
        assert all(per_class[class_name]["GENE_B"] == 0 for class_name in CLASSES)
        assert any_train["GENE_A"] == 1

    def test_a_gene_in_two_training_sets_counts_once_as_in_any(self, monkeypatch):
        frame = pd.DataFrame(
            {"gene": ["GENE_A"], "train_dn_gof_lof": ["1|0|1"]}
        )
        self._patch_s3(monkeypatch, frame)

        any_train, per_class = load_badonyi_train_flags()

        assert any_train["GENE_A"] == 1
        assert per_class["DN"]["GENE_A"] == 1
        assert per_class["LOF"]["GENE_A"] == 1
        assert per_class["GOF"]["GENE_A"] == 0


class TestRunRegime:

    def test_small_regime_is_skipped_without_metrics(self):
        n = 40
        result = run_regime(
            "IN",
            np.ones(n, dtype=bool),
            np.zeros((n, 3), dtype=np.float32),
            np.zeros((n, 3), dtype=np.float32),
            np.array(["LOF"] * n, dtype=object),
            np.array([f"GENE_{i}" for i in range(n)]),
            np.array(["PF0001"] * n),
            n_folds=5,
            seed=0,
            pfam_map={},
        )

        assert result == {
            "skipped": True,
            "reason": "fewer_than_30_variants",
            "n_variants": n,
        }
        assert "V_bad" not in result

    def test_regime_reports_the_masked_cohort_size(self, monkeypatch):
        n = 300
        monkeypatch.setattr(
            leakage, "run_probe", lambda *args, **kwargs: {"macro_f1_mean": 0.5}
        )
        mask = np.zeros(n, dtype=bool)
        mask[:150] = True

        result = run_regime(
            "IN",
            mask,
            np.zeros((n, 3), dtype=np.float32),
            np.zeros((n, 3), dtype=np.float32),
            np.array(["LOF"] * n, dtype=object),
            np.array([f"GENE_{i % 50}" for i in range(n)]),
            np.array([f"PF{i % 10:04d}" for i in range(n)]),
            n_folds=5,
            seed=0,
            pfam_map={},
        )

        assert result["n_variants"] == 150
        assert result["n_genes"] == 50
        assert set(result["V2"]) == {"macro_f1_mean"}


class TestRunSeedRegimes:

    def _capture_masks(self, monkeypatch):
        captured = {}

        def _fake_run_regime(regime_name, mask, *args, **kwargs):
            captured[regime_name] = np.asarray(mask).copy()
            return {"skipped": True, "n_variants": int(mask.sum())}

        monkeypatch.setattr(leakage, "run_regime", _fake_run_regime)
        return captured

    def test_in_and_out_partition_the_family_annotated_variants(self, monkeypatch):
        captured = self._capture_masks(monkeypatch)
        genes = np.array(["GENE_A", "GENE_B", "GENE_C", "GENE_A"])
        pfam_map = {"GENE_A": "PF0001", "GENE_B": "PF0002", "GENE_C": "PF0003"}
        train_flags = {"GENE_A": 1, "GENE_B": 0}

        run_seed(
            seed=0,
            n_folds=5,
            y=np.array(["GOF", "DN", "LOF", "GOF"], dtype=object),
            genes=genes,
            pfam_map=pfam_map,
            X_prot=np.zeros((4, 3), dtype=np.float32),
            X_bad=np.zeros((4, 3), dtype=np.float32),
            train_flag_any=train_flags,
            compute_ci=False,
        )

        assert not (captured["IN"] & captured["OUT"]).any()
        assert ((captured["IN"] | captured["OUT"]) == captured["ALL"]).all()
        assert list(captured["IN"]) == [True, False, False, True]

    def test_gene_without_a_family_is_excluded_from_every_regime(self, monkeypatch):
        captured = self._capture_masks(monkeypatch)
        genes = np.array(["GENE_A", "GENE_NOFAM", "GENE_C"])
        pfam_map = {"GENE_A": "PF0001", "GENE_C": "PF0003"}

        run_seed(
            seed=0,
            n_folds=5,
            y=np.array(["GOF", "DN", "LOF"], dtype=object),
            genes=genes,
            pfam_map=pfam_map,
            X_prot=np.zeros((3, 3), dtype=np.float32),
            X_bad=np.zeros((3, 3), dtype=np.float32),
            train_flag_any={"GENE_A": 1, "GENE_NOFAM": 1, "GENE_C": 0},
            compute_ci=False,
        )

        for regime_name in ("ALL", "IN", "OUT"):
            assert captured[regime_name][1] == False  # noqa: E712
        assert list(captured["ALL"]) == [True, False, True]

    def test_gene_absent_from_the_training_flags_counts_as_out(self, monkeypatch):
        captured = self._capture_masks(monkeypatch)
        genes = np.array(["GENE_A", "GENE_UNLISTED"])
        pfam_map = {"GENE_A": "PF0001", "GENE_UNLISTED": "PF0002"}

        run_seed(
            seed=0,
            n_folds=5,
            y=np.array(["GOF", "DN"], dtype=object),
            genes=genes,
            pfam_map=pfam_map,
            X_prot=np.zeros((2, 3), dtype=np.float32),
            X_bad=np.zeros((2, 3), dtype=np.float32),
            train_flag_any={"GENE_A": 1},
            compute_ci=False,
        )

        assert list(captured["OUT"]) == [False, True]


def _seed_result(
    seed,
    macro_f1=0.5,
    per_gene_f1=0.4,
    auroc_dn=0.6,
    ci=(0.3, 0.7),
    regimes=("ALL", "IN", "OUT"),
):
    arm = {
        "macro_f1_mean": macro_f1,
        "per_gene_f1_mean": per_gene_f1,
        f"auroc_{CLASSES[1]}_mean": auroc_dn,
    }
    if ci is not None:
        arm["ci"] = {"macro_f1": {"ci_low": ci[0], "ci_high": ci[1]}}
    else:
        arm["ci"] = {"macro_f1": {"ci_suppressed": True}}
    def _regime_block():
        return {
            "n_variants": 500,
            "n_genes": 100,
            "class_dist_variants": {class_name: 10 for class_name in CLASSES},
            "V2": copy.deepcopy(arm),
            "V_bad": copy.deepcopy(arm),
            "V2_bad": copy.deepcopy(arm),
        }

    return {
        **seed_result_contract(seed),
        "regimes": {name: _regime_block() for name in regimes},
    }


class TestAggregateSeeds:

    def test_averages_across_seeds(self):
        seeds = [
            _seed_result(0, macro_f1=0.4),
            _seed_result(1, macro_f1=0.5),
            _seed_result(2, macro_f1=0.6),
        ]

        summary = aggregate_seeds(seeds, range(3))

        all_regime = summary["regimes"]["ALL"]
        aggregate = all_regime["V2_macro_f1_seed_aggregate"]
        assert aggregate["mean"] == pytest.approx(0.5)
        assert aggregate["seed_std"] == pytest.approx(0.1)

    def test_skipped_regime_stays_skipped_with_no_numbers(self):
        seeds = [_seed_result(0)]
        seeds[0]["regimes"]["IN"] = {
            "skipped": True,
            "reason": "fewer_than_30_variants",
            "n_variants": 12,
        }

        summary = aggregate_seeds(seeds, [0])

        assert summary["regimes"]["IN"]["status"] == "excluded"
        assert summary["regimes"]["OUT"]["V2_macro_f1_seed_aggregate"]["mean"] is not None

    def test_regime_eligibility_cannot_change_across_seeds(self):
        skipped = _seed_result(0)
        skipped["regimes"]["IN"] = {
            "skipped": True,
            "reason": "fewer_than_30_variants",
            "n_variants": 12,
        }
        seeds = [skipped, _seed_result(1, macro_f1=0.8)]

        with pytest.raises(ValueError, match="eligibility changed"):
            aggregate_seeds(seeds, range(2))

    def test_metric_no_seed_reported_is_absent_not_zero(self):
        seeds = [_seed_result(0)]
        for regime in seeds[0]["regimes"].values():
            for arm in ("V2", "V_bad", "V2_bad"):
                regime[arm].pop("per_gene_f1_mean")

        summary = aggregate_seeds(seeds, [0])

        assert summary["regimes"]["ALL"]["V2_per_gene_f1_seed_aggregate"]["state"] == "unavailable"
        assert summary["regimes"]["ALL"]["V2_macro_f1_seed_aggregate"]["state"] == "available"

    def test_none_valued_metric_is_not_averaged_as_a_number(self):
        seeds = [_seed_result(0), _seed_result(1, macro_f1=None)]

        summary = aggregate_seeds(seeds, range(2))

        assert summary["regimes"]["ALL"]["V2_macro_f1_seed_aggregate"]["state"] == "unavailable"

    def test_suppressed_ci_bounds_are_left_out_of_the_pooled_interval(self):
        seeds = [_seed_result(0, ci=(0.2, 0.8)), _seed_result(1, ci=None)]

        summary = aggregate_seeds(seeds, range(2))

        all_regime = summary["regimes"]["ALL"]
        assert "V2_macro_f1_ci_low_seed_mean" not in all_regime
        assert "V2_macro_f1_ci_high_seed_mean" not in all_regime

    def test_all_ci_suppressed_leaves_no_pooled_interval(self):
        seeds = [_seed_result(0, ci=None)]

        summary = aggregate_seeds(seeds, [0])

        assert "V2_macro_f1_ci_low_seed_mean" not in summary["regimes"]["ALL"]


class TestPrintTable:

    def test_prints_a_skipped_regime_without_error(self, capsys):
        seeds = [_seed_result(0)]
        seeds[0]["regimes"]["IN"] = {
            "skipped": True,
            "reason": "fewer_than_30_variants",
            "n_variants": 12,
        }

        print_table(aggregate_seeds(seeds, [0]))

        printed = capsys.readouterr().out
        assert "IN" in printed
        assert "SKIPPED" in printed

    def test_prints_unavailable_metrics_as_not_available(self, capsys):
        seeds = [_seed_result(0)]
        for regime in seeds[0]["regimes"].values():
            for arm in ("V2", "V_bad", "V2_bad"):
                regime[arm].pop("per_gene_f1_mean")

        print_table(aggregate_seeds(seeds, [0]))

        assert "N/A" in capsys.readouterr().out
