"""
Integration tests for the per-seed-file path: files on disk through the loader
and into the shared across-seed core.

These exercise the seam where the real defects live — seed identity taken from a
filename, stray files in a results directory, and a run that stopped partway.
Each test writes real JSON to a temporary directory rather than mocking the load.

Some tests pin behaviour the current loader already has, so that migrating to the
shared core does not quietly lose it. Others require the new contract and fail
until it exists.

Invariants:
- the expected seed set is supplied by the caller, never inferred from the files
  that happen to be present
- a seed's identity comes from its filename and must agree with the identity
  recorded inside the file
- a results directory containing an unrelated file that matches the glob is a run
  in error, not a seed to aggregate
- a run that wrote only some of its seed files yields a refusal, not a partial mean
- fold spread and seed spread never share a field name in a written result
"""

import json

import pytest

from esm2_mech.utils.seed_aggregation import (
    SeedUnavailableReason,
    aggregate_seed_values,
    load_seed_files,
    make_seed_record,
)
from tests.helpers import (
    FIVE_SEEDS,
    FIVE_VALUES_BY_SEED as MACRO_F1_BY_SEED,
)

SEED_GLOB = "results_seed*.json"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _seed_file_contents(seed, macro_f1_mean, *, status="success"):
    """A per-seed result carrying its own identity and one fold-reduced metric.

    `macro_f1_mean` is the within-seed value: the fold reduction has already
    happened, and `macro_f1_fold_std` is the spread over folds inside this seed.
    """
    return {
        "seed": seed,
        "status": status,
        "gene_split": {
            "delta_mean": {
                "status": status,
                "macro_f1_mean": macro_f1_mean,
                "macro_f1_fold_std": 0.01,
            }
        },
    }


def _write_run(directory, values_by_seed, **kwargs):
    """Write one seed file per entry and return the directory path as a string."""
    for seed, macro_f1 in values_by_seed.items():
        contents = _seed_file_contents(seed, macro_f1, **kwargs)
        (directory / f"results_seed{seed}.json").write_text(json.dumps(contents))
    return str(directory)


def _assert_refused_for(aggregate, seeds):
    """A refusal names the seeds responsible and carries no number."""
    assert aggregate.available is False
    assert aggregate.mean is None
    assert sorted(aggregate.affected_seeds) == sorted(seeds)


def _macro_f1_by_seed(loaded):
    """Pull one within-seed point estimate per seed out of loaded files.

    This is the experiment-specific traversal the plan keeps outside the shared
    core: the core receives a plain mapping.
    """
    records = []
    for seed, _filename, result in loaded:
        block = result.get("gene_split", {}).get("delta_mean", {})
        if "status" not in block:
            raise ValueError(f"seed {seed} has no metric status")
        records.append(
            make_seed_record(
                seed,
                block.get("macro_f1_mean"),
                status=block["status"],
            )
        )
    return records


# ---------------------------------------------------------------------------
# a complete run
# ---------------------------------------------------------------------------


class TestCompleteRun:

    def test_five_files_produce_one_aggregate(self, tmp_path):
        run_dir = _write_run(tmp_path, MACRO_F1_BY_SEED)
        loaded = load_seed_files(run_dir, SEED_GLOB, expected_seeds=FIVE_SEEDS)
        aggregate = aggregate_seed_values(FIVE_SEEDS, _macro_f1_by_seed(loaded))
        assert aggregate.available is True
        assert aggregate.mean == pytest.approx(0.3)
        assert aggregate.spread == pytest.approx(0.15811388300841897)

    def test_seed_identity_comes_from_the_file_not_the_load_order(self, tmp_path):
        """Files are globbed in string order, so seed 10 sorts before seed 2. The
        aggregate must not depend on that."""
        values = {2: 0.2, 10: 0.4}
        for seed, macro_f1 in values.items():
            contents = _seed_file_contents(seed, macro_f1)
            (tmp_path / f"results_seed{seed}.json").write_text(json.dumps(contents))
        loaded = load_seed_files(str(tmp_path), SEED_GLOB, expected_seeds=(2, 10))
        by_seed = {record.seed: record.value for record in _macro_f1_by_seed(loaded)}
        assert by_seed[2] == pytest.approx(0.2)
        assert by_seed[10] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# an incomplete run
# ---------------------------------------------------------------------------


class TestIncompleteRun:

    def test_a_run_that_stopped_partway_refuses(self, tmp_path):
        """Seeds are written as each finishes, so an interrupted run leaves a
        directory that looks complete to anything that globs it."""
        partial = {seed: MACRO_F1_BY_SEED[seed] for seed in (0, 1, 2)}
        run_dir = _write_run(tmp_path, partial)
        with pytest.raises(ValueError):
            load_seed_files(run_dir, SEED_GLOB, expected_seeds=FIVE_SEEDS)

    def test_the_expected_set_is_not_inferred_from_the_files_present(self, tmp_path):
        """Three files must never be read as a complete three-seed run."""
        partial = {seed: MACRO_F1_BY_SEED[seed] for seed in (0, 1, 2)}
        run_dir = _write_run(tmp_path, partial)
        loaded = load_seed_files(run_dir, SEED_GLOB, expected_seeds=(0, 1, 2))
        aggregate = aggregate_seed_values(FIVE_SEEDS, _macro_f1_by_seed(loaded))
        assert aggregate.available is False
        assert sorted(aggregate.affected_seeds) == [3, 4]

    def test_caller_must_supply_the_expected_seeds(self, tmp_path):
        """Loading without an expected set is how a short run passes unnoticed."""
        run_dir = _write_run(tmp_path, MACRO_F1_BY_SEED)
        with pytest.raises(TypeError):
            load_seed_files(run_dir, SEED_GLOB)

    def test_a_failed_seed_refuses_the_aggregate(self, tmp_path):
        run_dir = _write_run(tmp_path, MACRO_F1_BY_SEED)
        broken = _seed_file_contents(3, None, status="failed")
        (tmp_path / "results_seed3.json").write_text(json.dumps(broken))
        loaded = load_seed_files(run_dir, SEED_GLOB, expected_seeds=FIVE_SEEDS)
        aggregate = aggregate_seed_values(FIVE_SEEDS, _macro_f1_by_seed(loaded))
        _assert_refused_for(aggregate, [3])


# ---------------------------------------------------------------------------
# a results directory with strays in it
# ---------------------------------------------------------------------------


class TestStrayFiles:

    def test_a_backup_matching_the_glob_is_an_error_not_a_seed(self, tmp_path):
        """A real hazard: a backup named with the result prefix sits in the same
        directory and is picked up by the glob."""
        run_dir = _write_run(tmp_path, MACRO_F1_BY_SEED)
        stray = _seed_file_contents(0, 0.99)
        (tmp_path / "results_seed0_step2.json").write_text(json.dumps(stray))
        with pytest.raises(ValueError):
            load_seed_files(run_dir, SEED_GLOB, expected_seeds=FIVE_SEEDS)

    def test_an_extra_seed_beyond_the_expected_set_is_an_error(self, tmp_path):
        """A sixth seed file means the directory holds two different runs."""
        run_dir = _write_run(tmp_path, {**MACRO_F1_BY_SEED, 5: 0.6})
        with pytest.raises(ValueError):
            load_seed_files(run_dir, SEED_GLOB, expected_seeds=FIVE_SEEDS)

    def test_two_files_claiming_the_same_seed_is_an_error(self, tmp_path):
        run_dir = _write_run(tmp_path, MACRO_F1_BY_SEED)
        duplicate = _seed_file_contents(1, 0.7)
        (tmp_path / "results_seed01.json").write_text(json.dumps(duplicate))
        with pytest.raises(ValueError):
            load_seed_files(run_dir, SEED_GLOB, expected_seeds=FIVE_SEEDS)


# ---------------------------------------------------------------------------
# identity recorded inside the file
# ---------------------------------------------------------------------------


class TestRecordedIdentity:

    def test_filename_and_recorded_seed_must_agree(self, tmp_path):
        """Backfilling a missing seed by copying another seed's file is caught
        here and nowhere else."""
        run_dir = _write_run(tmp_path, MACRO_F1_BY_SEED)
        copied = _seed_file_contents(2, MACRO_F1_BY_SEED[2])
        (tmp_path / "results_seed4.json").write_text(json.dumps(copied))
        with pytest.raises(ValueError):
            load_seed_files(run_dir, SEED_GLOB, expected_seeds=FIVE_SEEDS)

    def test_a_file_with_no_recorded_seed_is_refused(self, tmp_path):
        """Identity has to be in the file; a filename alone can be renamed."""
        run_dir = _write_run(tmp_path, MACRO_F1_BY_SEED)
        anonymous = _seed_file_contents(3, 0.4)
        del anonymous["seed"]
        (tmp_path / "results_seed3.json").write_text(json.dumps(anonymous))
        with pytest.raises(ValueError):
            load_seed_files(run_dir, SEED_GLOB, expected_seeds=FIVE_SEEDS)

    def test_boolean_recorded_seed_is_not_treated_as_integer_one(self, tmp_path):
        run_dir = _write_run(tmp_path, MACRO_F1_BY_SEED)
        invalid = _seed_file_contents(True, MACRO_F1_BY_SEED[1])
        (tmp_path / "results_seed1.json").write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match="must be an integer"):
            load_seed_files(run_dir, SEED_GLOB, expected_seeds=FIVE_SEEDS)


# ---------------------------------------------------------------------------
# a corrupt file
# ---------------------------------------------------------------------------


class TestCorruptFile:

    def test_unreadable_json_refuses_rather_than_skipping(self, tmp_path):
        """Skipping a corrupt file and averaging the rest is the same defect as
        dropping a failed seed."""
        run_dir = _write_run(tmp_path, MACRO_F1_BY_SEED)
        (tmp_path / "results_seed2.json").write_text("{not json")
        with pytest.raises(ValueError):
            load_seed_files(run_dir, SEED_GLOB, expected_seeds=FIVE_SEEDS)

    def test_an_empty_file_refuses(self, tmp_path):
        run_dir = _write_run(tmp_path, MACRO_F1_BY_SEED)
        (tmp_path / "results_seed2.json").write_text("")
        with pytest.raises(ValueError):
            load_seed_files(run_dir, SEED_GLOB, expected_seeds=FIVE_SEEDS)

    def test_non_object_json_refuses(self, tmp_path):
        run_dir = _write_run(tmp_path, MACRO_F1_BY_SEED)
        (tmp_path / "results_seed2.json").write_text(json.dumps([2, 0.3]))
        with pytest.raises(ValueError, match="must be a JSON object"):
            load_seed_files(run_dir, SEED_GLOB, expected_seeds=FIVE_SEEDS)


# ---------------------------------------------------------------------------
# spread names in written results
# ---------------------------------------------------------------------------


class TestSpreadNaming:

    def test_per_seed_file_carries_a_fold_spread_not_a_bare_std(self, tmp_path):
        """The within-seed file must not write a field a reader could mistake
        for a seed spread."""
        run_dir = _write_run(tmp_path, MACRO_F1_BY_SEED)
        loaded = load_seed_files(run_dir, SEED_GLOB, expected_seeds=FIVE_SEEDS)
        for _seed, filename, result in loaded:
            block = result["gene_split"]["delta_mean"]
            assert (
                "macro_f1_std" not in block
            ), f"{filename} writes an unlabelled spread"
            assert "macro_f1_fold_std" in block

    def test_a_fold_spread_is_never_read_as_a_seed_spread(self, tmp_path):
        """Five seeds each with a small fold spread must not produce a small seed
        spread; the two are computed from different numbers."""
        run_dir = _write_run(tmp_path, MACRO_F1_BY_SEED)
        loaded = load_seed_files(run_dir, SEED_GLOB, expected_seeds=FIVE_SEEDS)
        aggregate = aggregate_seed_values(FIVE_SEEDS, _macro_f1_by_seed(loaded))
        fold_spread = 0.01
        assert aggregate.spread > fold_spread * 10
