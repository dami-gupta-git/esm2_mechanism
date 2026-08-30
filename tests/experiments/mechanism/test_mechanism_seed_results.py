"""
Tests for esm2_mech.experiments.mechanism.seed_results.

Invariants:
- aggregate_across_seeds: one shared aggregate is stored per split→feature→metric
- aggregate_across_seeds: only <metric>_mean keys are aggregated (others ignored)
- aggregate_across_seeds: None, NaN, failed seeds, and missing features make the aggregate unavailable
- aggregate_across_seeds: a per-seed file must declare the shared root contract
- aggregate_across_seeds: a non-success root status makes every feature unavailable
- aggregate_across_seeds: confusion matrices are row-normalized per seed before averaging
- print_table: does not crash when a feature is missing from one split
- aggregate_across_seeds: the caller declares the requested seeds; a missing or
  unrequested seed makes the aggregate unavailable
- read_across_seed_metric: an unavailable metric is read back as None
- read_across_seed_metric: a file without the current schema version raises
"""

import json

import numpy as np
import pytest

from esm2_mech.experiments.mechanism.seed_results import (
    SPLITS,
    aggregate_across_seeds,
    aggregate_result_contract,
    print_table,
    read_across_seed_metric,
)
from esm2_mech.utils.seed_aggregation import SEED_SCHEMA_KEY, seed_result_contract
from tests.helpers import (
    available_seed_aggregate,
    seed_result,
    unavailable_seed_aggregate,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _with_seeds(named_results, statuses=None):
    """[(filename, dict), ...] -> the (seed, filename, dict) triples the walker takes."""
    declared = statuses or {}
    triples = []
    for seed, (filename, data) in enumerate(named_results):
        data.update(seed_result_contract(seed, status=declared.get(seed, "success")))
        triples.append((seed, filename, data))
    return triples


def _aggregate(named_results, statuses=None, requested=None, **kwargs):
    """Aggregate the given results, requesting exactly the seeds they carry."""
    triples = _with_seeds(named_results, statuses)
    if requested is None:
        requested = range(len(triples))
    return aggregate_across_seeds(triples, requested, **kwargs)


# ---------------------------------------------------------------------------
# aggregate_across_seeds
# ---------------------------------------------------------------------------


class TestAggregateAcrossSeeds:

    def test_mean_and_std_across_seeds(self):
        seed_results = [
            ("seed0.json", seed_result(0.4)),
            ("seed1.json", seed_result(0.5)),
            ("seed2.json", seed_result(0.6)),
        ]
        agg = _aggregate(seed_results)
        feature = agg["gene_split"]["esm2"]
        metric = feature["macro_f1_seed_aggregate"]
        assert metric["mean"] == pytest.approx(0.5)
        assert metric["seed_std"] == pytest.approx(0.1)
        assert metric["contributing_seeds"] == [0, 1, 2]
        assert metric["sampling_unit"] == "model_seed"

    def test_only_mean_keys_aggregated(self):
        # macro_f1_std is present per seed but must NOT be aggregated as a metric.
        seed_results = [("seed0.json", seed_result(0.5))]
        agg = _aggregate(seed_results)
        feature = agg["gene_split"]["esm2"]
        assert set(feature) == {"macro_f1_seed_aggregate"}

    def test_none_value_makes_metric_unavailable(self):
        seed_results = [
            ("seed0.json", seed_result(0.6)),
            ("seed1.json", seed_result(None)),
        ]
        agg = _aggregate(seed_results)
        feature = agg["gene_split"]["esm2"]
        metric = feature["macro_f1_seed_aggregate"]
        assert metric["mean"] is None
        assert metric["contributing_seeds"] == [0]
        assert metric["affected_seeds"] == [1]

    def test_nan_value_makes_metric_unavailable(self):
        seed_results = [
            ("seed0.json", seed_result(0.6)),
            ("seed1.json", seed_result(float("nan"))),
        ]
        agg = _aggregate(seed_results)
        feature = agg["gene_split"]["esm2"]
        metric = feature["macro_f1_seed_aggregate"]
        assert metric["mean"] is None
        assert metric["contributing_seeds"] == [0]

    def test_n_seeds_counts_contributors_only(self):
        seed_results = [
            ("seed0.json", seed_result(0.5)),
            ("seed1.json", seed_result(0.5)),
            ("seed2.json", seed_result(None)),
        ]
        agg = _aggregate(seed_results)
        metric = agg["gene_split"]["esm2"]["macro_f1_seed_aggregate"]
        assert metric["contributing_seeds"] == [0, 1]

    def test_feature_present_in_some_seeds_is_unavailable(self):
        # esm2 in both seeds, esm3 only in seed1.
        seed_results = [
            ("seed0.json", seed_result(0.5, feature="esm2")),
            (
                "seed1.json",
                {
                    "gene_split": {
                        "esm2": {"status": "success", "macro_f1_mean": 0.7},
                        "esm3": {"status": "success", "macro_f1_mean": 0.9},
                    }
                },
            ),
        ]
        agg = _aggregate(seed_results)
        assert (
            agg["gene_split"]["esm2"]["macro_f1_seed_aggregate"]["state"] == "available"
        )
        esm3 = agg["gene_split"]["esm3"]["macro_f1_seed_aggregate"]
        assert esm3["contributing_seeds"] == [1]
        assert esm3["mean"] is None
        assert esm3["state"] == "unavailable"

    def test_unknown_seed_status_is_not_relabelled_as_failed(self):
        result = seed_result(0.5)
        result["gene_split"]["esm2"]["status"] = "unknown"
        with pytest.raises(ValueError, match="unsupported seed status"):
            _aggregate([("seed0.json", result)])

    def test_missing_seed_status_is_not_filled_in(self):
        result = seed_result(0.5)
        del result["gene_split"]["esm2"]["status"]
        with pytest.raises(ValueError, match="has no status"):
            _aggregate([("seed0.json", result)])

    def test_result_without_the_root_contract_raises(self):
        result = seed_result(0.5)
        triples = _with_seeds([("seed0.json", result)])
        del triples[0][2][SEED_SCHEMA_KEY]
        with pytest.raises(ValueError, match="seed schema version"):
            aggregate_across_seeds(triples, range(1))

    def test_root_status_overrides_a_successful_block(self):
        seed_results = [
            ("seed0.json", seed_result(0.5)),
            ("seed1.json", seed_result(0.6)),
        ]
        agg = _aggregate(seed_results, statuses={1: "failed"})
        feature = agg["gene_split"]["esm2"]
        metric = feature["macro_f1_seed_aggregate"]
        assert metric["affected_seeds"] == [1]
        assert metric["mean"] is None
        assert metric["reason"] == "failed_seed"

    def test_a_requested_seed_with_no_result_is_unavailable(self):
        seed_results = [
            ("seed0.json", seed_result(0.4)),
            ("seed1.json", seed_result(0.6)),
        ]
        agg = _aggregate(seed_results, requested=range(3))
        feature = agg["gene_split"]["esm2"]
        metric = feature["macro_f1_seed_aggregate"]
        assert metric["requested_seeds"] == [0, 1, 2]
        assert metric["mean"] is None
        assert metric["reason"] == "missing_seed"
        assert metric["affected_seeds"] == [2]

    def test_a_result_for_an_unrequested_seed_is_unavailable(self):
        seed_results = [
            ("seed0.json", seed_result(0.4)),
            ("seed1.json", seed_result(0.6)),
        ]
        agg = _aggregate(seed_results, requested=[0])
        feature = agg["gene_split"]["esm2"]
        metric = feature["macro_f1_seed_aggregate"]
        assert metric["mean"] is None
        assert metric["reason"] == "unexpected_seed"

    def test_both_splits_in_output(self):
        seed_results = [("seed0.json", seed_result(0.5))]
        agg = _aggregate(seed_results)
        for split in SPLITS:
            assert split in agg

    def test_empty_seed_results(self):
        agg = aggregate_across_seeds([], [])
        for split in SPLITS:
            assert agg[split] == {}

    def test_multiple_metrics_aggregated_independently(self):
        seed_results = [
            (
                "seed0.json",
                {
                    "gene_split": {
                        "esm2": {
                            "status": "success",
                            "macro_f1_mean": 0.4,
                            "auroc_GOF_mean": 0.8,
                        }
                    }
                },
            ),
            (
                "seed1.json",
                {
                    "gene_split": {
                        "esm2": {
                            "status": "success",
                            "macro_f1_mean": 0.6,
                            "auroc_GOF_mean": 0.9,
                        }
                    }
                },
            ),
        ]
        agg = _aggregate(seed_results)
        feature = agg["gene_split"]["esm2"]
        assert feature["macro_f1_seed_aggregate"]["mean"] == pytest.approx(0.5)
        assert feature["auroc_GOF_seed_aggregate"]["mean"] == pytest.approx(0.85)

    def test_failed_seed_prevents_reduced_seed_aggregate(self):
        successful = seed_result(0.6)
        failed = seed_result(None)
        failed["gene_split"]["esm2"]["status"] = "unscorable"
        aggregate = _aggregate([("seed0.json", successful), ("seed1.json", failed)])
        feature = aggregate["gene_split"]["esm2"]
        metric = feature["macro_f1_seed_aggregate"]
        assert metric["state"] == "unavailable"
        assert metric["mean"] is None
        assert metric["affected_seeds"] == [1]

    def test_confusion_matrix_is_normalized_per_seed_before_average(self):
        first = seed_result(0.5)
        second = seed_result(0.5)
        first_block = first["gene_split"]["esm2"]
        second_block = second["gene_split"]["esm2"]
        first_block.update(
            {
                "confusion_matrix": [[8, 2], [1, 9]],
                "confusion_matrix_class_order": ["A", "B"],
            }
        )
        second_block.update(
            {
                "confusion_matrix": [[1, 9], [4, 6]],
                "confusion_matrix_class_order": ["A", "B"],
            }
        )
        aggregate = _aggregate(
            [("seed0.json", first), ("seed1.json", second)],
            confusion_matrix_class_order=["A", "B"],
        )["gene_split"]["esm2"]
        assert np.allclose(
            aggregate["confusion_matrix_seed_aggregate"]["payload"][
                "normalized_seed_mean"
            ],
            [[0.45, 0.55], [0.25, 0.75]],
        )


# ---------------------------------------------------------------------------
# print_table
# ---------------------------------------------------------------------------


class TestPrintTable:

    def test_two_seeds_print_the_estimate_without_a_spread(self, capsys):
        """Too few seeds for a spread is not the same as an unscorable metric.

        The estimate satisfies its own contract, so it is printed and labelled as
        carrying no spread rather than hidden.
        """
        seed_results = [
            ("seed0.json", seed_result(0.4)),
            ("seed1.json", seed_result(0.6)),
        ]
        agg = _aggregate(seed_results)
        print_table(agg)
        out = capsys.readouterr().out
        assert "mean ± SD across model seeds" in out
        # The gene-split estimate is shown; the family column is genuinely absent
        # from this fixture, so it alone reads as unavailable.
        esm2_row = next(line for line in out.splitlines() if line.startswith("esm2"))
        assert "0.500 (no spread)" in esm2_row

    def test_does_not_crash_when_feature_missing_from_one_split(self, capsys):
        # esm2 only present in gene_split, esm3 only in family_split.
        agg = {
            "gene_split": {
                "esm2": {
                    "macro_f1_seed_aggregate": available_seed_aggregate(
                        0.5, seeds=(0, 1, 2)
                    ),
                }
            },
            "family_split": {
                "esm3": {
                    "macro_f1_seed_aggregate": available_seed_aggregate(
                        0.4, seeds=(0, 1, 2)
                    ),
                }
            },
        }
        print_table(agg)
        out = capsys.readouterr().out
        assert "esm2" in out
        assert "esm3" in out

    def test_does_not_crash_on_empty_aggregate(self):
        print_table({"gene_split": {}, "family_split": {}})


def _aggregate_file(tmp_path, seed_aggregate, **root):
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(
        json.dumps(
            {
                **aggregate_result_contract(),
                **root,
                "across_seed": {
                    "family_split": {
                        "delta_mean": {"macro_f1_seed_aggregate": seed_aggregate}
                    }
                },
            }
        )
    )
    return aggregate_path


def test_read_across_seed_metric_preserves_unavailable_value(tmp_path):
    path = _aggregate_file(tmp_path, unavailable_seed_aggregate())

    assert read_across_seed_metric(str(path), "family_split", "delta_mean") is None


def test_read_across_seed_metric_reads_the_stored_aggregate(tmp_path):
    stored = available_seed_aggregate(0.36)
    path = _aggregate_file(tmp_path, stored)

    assert read_across_seed_metric(
        str(path), "family_split", "delta_mean"
    ) == pytest.approx(0.36)


def test_read_across_seed_metric_rejects_a_file_without_the_schema_version(tmp_path):
    path = _aggregate_file(tmp_path, available_seed_aggregate(0.36))
    content = json.loads(path.read_text())
    del content[SEED_SCHEMA_KEY]
    path.write_text(json.dumps(content))

    with pytest.raises(ValueError, match="seed schema version"):
        read_across_seed_metric(str(path), "family_split", "delta_mean")
