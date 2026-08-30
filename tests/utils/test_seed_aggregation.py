"""
Tests for esm2_mech.utils.seed_aggregation.

Invariants:
- load_seed_files: loads every matching seed file as (seed, basename, dict), seed
  parsed from the glob's `*` position
- load_seed_files: corrupt JSON and missing recorded identity raise
- load_seed_files: non-matching glob returns empty list
- load_seed_files: a filename whose `*` position is not a plain integer raises
- load_seed_files: two files claiming the same seed number raises
- load_seed_files: expected_seeds catches a missing seed and an unexpected extra seed
- seed_result_contract: declares the schema version, seed, and root seed status
- read_seed_result_contract: a wrong schema version, seed, or status raises
"""

import json

import pytest

from esm2_mech.utils.constants import SEED_AGGREGATION_SCHEMA_VERSION
from esm2_mech.utils.seed_aggregation import (
    SEED_SCHEMA_KEY,
    SEED_STATUS_KEY,
    load_seed_files,
    read_seed_result_contract,
    seed_result_contract,
)
from tests.helpers import seed_result

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_seed_file(path, data):
    token = path.stem.rsplit("seed", 1)[-1]
    data["seed"] = int(token) if token.isdigit() else 0
    path.write_text(json.dumps(data))
    return path


# ---------------------------------------------------------------------------
# load_seed_files
# ---------------------------------------------------------------------------


class TestLoadSeedFiles:

    def test_loads_matching_files(self, tmp_path):
        _write_seed_file(tmp_path / "res_seed0.json", seed_result(0.5))
        _write_seed_file(tmp_path / "res_seed1.json", seed_result(0.6))
        loaded = load_seed_files(str(tmp_path), "res_seed*.json", expected_seeds=(0, 1))
        assert len(loaded) == 2
        names = [name for _seed, name, _data in loaded]
        assert "res_seed0.json" in names
        assert "res_seed1.json" in names

    def test_returns_seed_basename_and_parsed_dict(self, tmp_path):
        _write_seed_file(tmp_path / "res_seed0.json", seed_result(0.5))
        loaded = load_seed_files(str(tmp_path), "res_seed*.json", expected_seeds=(0,))
        seed, name, data = loaded[0]
        assert seed == 0
        assert name == "res_seed0.json"
        assert data["gene_split"]["esm2"]["macro_f1_mean"] == 0.5

    def test_corrupt_file_raises(self, tmp_path):
        _write_seed_file(tmp_path / "res_seed0.json", seed_result(0.5))
        (tmp_path / "res_seed1.json").write_text("{not valid json")
        with pytest.raises(ValueError, match="invalid seed-result JSON"):
            load_seed_files(str(tmp_path), "res_seed*.json", expected_seeds=(0, 1))

    def test_no_match_returns_empty(self, tmp_path):
        _write_seed_file(tmp_path / "res_seed0.json", seed_result(0.5))
        assert load_seed_files(str(tmp_path), "nomatch*.json", expected_seeds=()) == []

    def test_sorted_order(self, tmp_path):
        _write_seed_file(tmp_path / "res_seed2.json", seed_result(0.5))
        _write_seed_file(tmp_path / "res_seed0.json", seed_result(0.5))
        _write_seed_file(tmp_path / "res_seed1.json", seed_result(0.5))
        loaded = load_seed_files(
            str(tmp_path), "res_seed*.json", expected_seeds=(0, 1, 2)
        )
        names = [name for _seed, name, _data in loaded]
        assert names == sorted(names)

    def test_non_integer_seed_token_raises(self, tmp_path):
        _write_seed_file(tmp_path / "res_seedfinal.json", seed_result(0.5))
        with pytest.raises(ValueError, match="does not encode an integer seed"):
            load_seed_files(str(tmp_path), "res_seed*.json", expected_seeds=(0,))

    def test_duplicate_seed_raises(self, tmp_path):
        # Two distinct filenames that both parse to seed 0 under this glob.
        _write_seed_file(tmp_path / "res_seed0.json", seed_result(0.5))
        _write_seed_file(tmp_path / "res_seed00.json", seed_result(0.6))
        with pytest.raises(ValueError, match="duplicate seed"):
            load_seed_files(str(tmp_path), "res_seed*.json", expected_seeds=(0,))

    def test_expected_seeds_satisfied_passes(self, tmp_path):
        _write_seed_file(tmp_path / "res_seed0.json", seed_result(0.5))
        _write_seed_file(tmp_path / "res_seed1.json", seed_result(0.6))
        loaded = load_seed_files(
            str(tmp_path), "res_seed*.json", expected_seeds=range(2)
        )
        assert len(loaded) == 2

    def test_expected_seeds_missing_raises(self, tmp_path):
        _write_seed_file(tmp_path / "res_seed0.json", seed_result(0.5))
        with pytest.raises(ValueError, match="missing"):
            load_seed_files(str(tmp_path), "res_seed*.json", expected_seeds=range(2))

    def test_expected_seeds_unexpected_raises(self, tmp_path):
        _write_seed_file(tmp_path / "res_seed0.json", seed_result(0.5))
        _write_seed_file(tmp_path / "res_seed5.json", seed_result(0.6))
        with pytest.raises(ValueError, match="unexpected"):
            load_seed_files(str(tmp_path), "res_seed*.json", expected_seeds=range(2))


# ---------------------------------------------------------------------------
# the per-seed file root contract
# ---------------------------------------------------------------------------


class TestSeedResultContract:

    def test_declares_schema_seed_and_status(self):
        contract = seed_result_contract(3)
        assert contract[SEED_SCHEMA_KEY] == SEED_AGGREGATION_SCHEMA_VERSION
        assert contract["seed"] == 3
        assert contract[SEED_STATUS_KEY] == "success"

    def test_unsupported_status_raises(self):
        with pytest.raises(ValueError, match="unsupported seed status"):
            seed_result_contract(0, status="partial")

    def test_reads_back_the_declared_status(self):
        contract = seed_result_contract(1, status="unscorable")
        assert read_seed_result_contract(1, "seed1.json", contract) == "unscorable"

    def test_missing_schema_version_raises(self):
        contract = seed_result_contract(0)
        del contract[SEED_SCHEMA_KEY]
        with pytest.raises(ValueError, match="seed schema version"):
            read_seed_result_contract(0, "seed0.json", contract)

    def test_seed_disagreement_raises(self):
        contract = seed_result_contract(0)
        with pytest.raises(ValueError, match="declares seed"):
            read_seed_result_contract(1, "seed1.json", contract)

    def test_unknown_root_status_raises(self):
        contract = seed_result_contract(0)
        contract[SEED_STATUS_KEY] = "probably fine"
        with pytest.raises(ValueError, match="is not a seed status"):
            read_seed_result_contract(0, "seed0.json", contract)
