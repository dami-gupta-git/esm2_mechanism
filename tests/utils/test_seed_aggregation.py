"""
Tests for esm2_mech.utils.seed_aggregation.

Invariants:
- load_seed_files: loads every matching seed file as (seed, basename, dict), seed
  parsed from the glob's `*` position
- load_seed_files: corrupt JSON file is skipped (not silently dropped, not crashing)
- load_seed_files: non-matching glob returns empty list
- load_seed_files: a filename whose `*` position is not a plain integer raises
- load_seed_files: two files claiming the same seed number raises
- load_seed_files: expected_seeds catches a missing seed and an unexpected extra seed
- aggregate_across_seeds: mean/std/n_seeds computed across seeds per split→feature→metric
- aggregate_across_seeds: only <metric>_mean keys are aggregated (others ignored)
- aggregate_across_seeds: None and NaN per-seed values are skipped, not fabricated
- aggregate_across_seeds: n_seeds counts only the seeds that contributed a value
- aggregate_across_seeds: a feature present in only some seeds still aggregates
- print_table: does not crash when a feature is missing from one split
"""

import json

import numpy as np
import pytest

from esm2_mech.utils.seed_aggregation import (
    SPLITS,
    aggregate_across_seeds,
    load_seed_files,
    print_table,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _seed_result(macro_f1_mean, *, split="gene_split", feature="esm2"):
    """Build a minimal per-seed result dict for one split/feature."""
    return {split: {feature: {"macro_f1_mean": macro_f1_mean, "macro_f1_std": 0.01}}}


def _write_seed_file(path, data):
    path.write_text(json.dumps(data))
    return path


def _with_seeds(named_results):
    """[(filename, dict), ...] -> [(seed, filename, dict), ...] for aggregate_across_seeds tests."""
    return [(i, filename, data) for i, (filename, data) in enumerate(named_results)]


# ---------------------------------------------------------------------------
# load_seed_files
# ---------------------------------------------------------------------------

class TestLoadSeedFiles:

    def test_loads_matching_files(self, tmp_path):
        _write_seed_file(tmp_path / "res_seed0.json", _seed_result(0.5))
        _write_seed_file(tmp_path / "res_seed1.json", _seed_result(0.6))
        loaded = load_seed_files(str(tmp_path), "res_seed*.json")
        assert len(loaded) == 2
        names = [name for _seed, name, _data in loaded]
        assert "res_seed0.json" in names
        assert "res_seed1.json" in names

    def test_returns_seed_basename_and_parsed_dict(self, tmp_path):
        _write_seed_file(tmp_path / "res_seed0.json", _seed_result(0.5))
        loaded = load_seed_files(str(tmp_path), "res_seed*.json")
        seed, name, data = loaded[0]
        assert seed == 0
        assert name == "res_seed0.json"
        assert data["gene_split"]["esm2"]["macro_f1_mean"] == 0.5

    def test_corrupt_file_skipped(self, tmp_path):
        _write_seed_file(tmp_path / "res_seed0.json", _seed_result(0.5))
        (tmp_path / "res_seed1.json").write_text("{not valid json")
        loaded = load_seed_files(str(tmp_path), "res_seed*.json")
        # The good file loads; the corrupt one is skipped rather than crashing.
        assert len(loaded) == 1
        assert loaded[0][1] == "res_seed0.json"

    def test_no_match_returns_empty(self, tmp_path):
        _write_seed_file(tmp_path / "res_seed0.json", _seed_result(0.5))
        assert load_seed_files(str(tmp_path), "nomatch*.json") == []

    def test_sorted_order(self, tmp_path):
        _write_seed_file(tmp_path / "res_seed2.json", _seed_result(0.5))
        _write_seed_file(tmp_path / "res_seed0.json", _seed_result(0.5))
        _write_seed_file(tmp_path / "res_seed1.json", _seed_result(0.5))
        loaded = load_seed_files(str(tmp_path), "res_seed*.json")
        names = [name for _seed, name, _data in loaded]
        assert names == sorted(names)

    def test_non_integer_seed_token_raises(self, tmp_path):
        _write_seed_file(tmp_path / "res_seedfinal.json", _seed_result(0.5))
        with pytest.raises(ValueError, match="does not encode an integer seed"):
            load_seed_files(str(tmp_path), "res_seed*.json")

    def test_duplicate_seed_raises(self, tmp_path):
        # Two distinct filenames that both parse to seed 0 under this glob.
        _write_seed_file(tmp_path / "res_seed0.json", _seed_result(0.5))
        _write_seed_file(tmp_path / "res_seed00.json", _seed_result(0.6))
        with pytest.raises(ValueError, match="duplicate seed"):
            load_seed_files(str(tmp_path), "res_seed*.json")

    def test_expected_seeds_satisfied_passes(self, tmp_path):
        _write_seed_file(tmp_path / "res_seed0.json", _seed_result(0.5))
        _write_seed_file(tmp_path / "res_seed1.json", _seed_result(0.6))
        loaded = load_seed_files(str(tmp_path), "res_seed*.json", expected_seeds=range(2))
        assert len(loaded) == 2

    def test_expected_seeds_missing_raises(self, tmp_path):
        _write_seed_file(tmp_path / "res_seed0.json", _seed_result(0.5))
        with pytest.raises(ValueError, match="missing"):
            load_seed_files(str(tmp_path), "res_seed*.json", expected_seeds=range(2))

    def test_expected_seeds_unexpected_raises(self, tmp_path):
        _write_seed_file(tmp_path / "res_seed0.json", _seed_result(0.5))
        _write_seed_file(tmp_path / "res_seed5.json", _seed_result(0.6))
        with pytest.raises(ValueError, match="unexpected"):
            load_seed_files(str(tmp_path), "res_seed*.json", expected_seeds=range(2))


# ---------------------------------------------------------------------------
# aggregate_across_seeds
# ---------------------------------------------------------------------------

class TestAggregateAcrossSeeds:

    def test_mean_and_std_across_seeds(self):
        seed_results = [
            ("seed0.json", _seed_result(0.4)),
            ("seed1.json", _seed_result(0.6)),
        ]
        agg = aggregate_across_seeds(_with_seeds(seed_results))
        feature = agg["gene_split"]["esm2"]
        assert feature["macro_f1_seed_mean"] == pytest.approx(0.5)
        assert feature["macro_f1_seed_std"] == pytest.approx(0.1)
        assert feature["macro_f1_n_seeds"] == 2

    def test_only_mean_keys_aggregated(self):
        # macro_f1_std is present per seed but must NOT be aggregated as a metric.
        seed_results = [("seed0.json", _seed_result(0.5))]
        agg = aggregate_across_seeds(_with_seeds(seed_results))
        feature = agg["gene_split"]["esm2"]
        assert "macro_f1_seed_mean" in feature
        assert "macro_f1_std_seed_mean" not in feature

    def test_none_value_skipped(self):
        seed_results = [
            ("seed0.json", _seed_result(0.6)),
            ("seed1.json", _seed_result(None)),
        ]
        agg = aggregate_across_seeds(_with_seeds(seed_results))
        feature = agg["gene_split"]["esm2"]
        # Only the one real value contributes; None is not fabricated to 0.
        assert feature["macro_f1_seed_mean"] == pytest.approx(0.6)
        assert feature["macro_f1_n_seeds"] == 1

    def test_nan_value_skipped(self):
        seed_results = [
            ("seed0.json", _seed_result(0.6)),
            ("seed1.json", _seed_result(float("nan"))),
        ]
        agg = aggregate_across_seeds(_with_seeds(seed_results))
        feature = agg["gene_split"]["esm2"]
        assert feature["macro_f1_seed_mean"] == pytest.approx(0.6)
        assert feature["macro_f1_n_seeds"] == 1

    def test_n_seeds_counts_contributors_only(self):
        seed_results = [
            ("seed0.json", _seed_result(0.5)),
            ("seed1.json", _seed_result(0.5)),
            ("seed2.json", _seed_result(None)),
        ]
        agg = aggregate_across_seeds(_with_seeds(seed_results))
        assert agg["gene_split"]["esm2"]["macro_f1_n_seeds"] == 2

    def test_feature_present_in_some_seeds(self):
        # esm2 in both seeds, esm3 only in seed1.
        seed_results = [
            ("seed0.json", _seed_result(0.5, feature="esm2")),
            (
                "seed1.json",
                {
                    "gene_split": {
                        "esm2": {"macro_f1_mean": 0.7},
                        "esm3": {"macro_f1_mean": 0.9},
                    }
                },
            ),
        ]
        agg = aggregate_across_seeds(_with_seeds(seed_results))
        assert agg["gene_split"]["esm2"]["macro_f1_n_seeds"] == 2
        assert agg["gene_split"]["esm3"]["macro_f1_n_seeds"] == 1
        assert agg["gene_split"]["esm3"]["macro_f1_seed_mean"] == pytest.approx(0.9)

    def test_both_splits_in_output(self):
        seed_results = [("seed0.json", _seed_result(0.5))]
        agg = aggregate_across_seeds(_with_seeds(seed_results))
        for split in SPLITS:
            assert split in agg

    def test_empty_seed_results(self):
        agg = aggregate_across_seeds([])
        for split in SPLITS:
            assert agg[split] == {}

    def test_multiple_metrics_aggregated_independently(self):
        seed_results = [
            (
                "seed0.json",
                {"gene_split": {"esm2": {"macro_f1_mean": 0.4, "auroc_GOF_mean": 0.8}}},
            ),
            (
                "seed1.json",
                {"gene_split": {"esm2": {"macro_f1_mean": 0.6, "auroc_GOF_mean": 0.9}}},
            ),
        ]
        agg = aggregate_across_seeds(_with_seeds(seed_results))
        feature = agg["gene_split"]["esm2"]
        assert feature["macro_f1_seed_mean"] == pytest.approx(0.5)
        assert feature["auroc_GOF_seed_mean"] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# print_table
# ---------------------------------------------------------------------------

class TestPrintTable:

    def test_does_not_crash_on_full_aggregate(self, capsys):
        seed_results = [
            ("seed0.json", _seed_result(0.4)),
            ("seed1.json", _seed_result(0.6)),
        ]
        agg = aggregate_across_seeds(_with_seeds(seed_results))
        print_table(agg)
        assert "macro_f1 across seeds" in capsys.readouterr().out

    def test_does_not_crash_when_feature_missing_from_one_split(self, capsys):
        # esm2 only present in gene_split, esm3 only in family_split.
        agg = {
            "gene_split": {"esm2": {"macro_f1_seed_mean": 0.5, "macro_f1_seed_std": 0.0, "macro_f1_n_seeds": 2}},
            "family_split": {"esm3": {"macro_f1_seed_mean": 0.4, "macro_f1_seed_std": 0.0, "macro_f1_n_seeds": 2}},
        }
        print_table(agg)
        out = capsys.readouterr().out
        assert "esm2" in out
        assert "esm3" in out

    def test_does_not_crash_on_empty_aggregate(self):
        print_table({"gene_split": {}, "family_split": {}})
