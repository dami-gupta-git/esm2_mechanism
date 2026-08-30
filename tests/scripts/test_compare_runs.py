"""
Tests for scripts/compare_runs.py (the run6 -> run_biorxiv numeric regression).

Invariants:
- load_run: flattens nested dicts/lists to dotted paths keyed by file
- load_run: a corrupt JSON is skipped and reported, not fatal
- compare: a run against itself reports no movement, no additions, no removals
- compare: a shared `mean` is judged against its declared `seed_std`
- compare: a leaf with no valid seed spread falls back to the absolute threshold and is
  counted as such
- compare: a zero or non-finite `_std` falls back rather than flagging everything
- compare: NaN -> NaN is unchanged; NaN appearing or disappearing is incomparable
- compare: string changes (gate verdicts) are reported separately from numeric moves
- compare: added keys (the new CI keys) are not counted as movement
- format_report: names every bucket even when empty
"""

import json

import pytest

from scripts.compare_runs import compare, format_report, load_run
from esm2_mech.utils.seed_aggregation import SEED_AGGREGATION_SCHEMA_VERSION


def _write(run_dir, name, payload):
    path = run_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle)
    return path


@pytest.fixture
def run_dir(tmp_path):
    directory = tmp_path / "run6"
    directory.mkdir()
    return directory


class TestLoadRun:
    def test_flattens_nested_structures_to_dotted_paths(self, run_dir):
        _write(run_dir, "a.json", {"gates": {"some_gate": {"value": 0.891}}, "seeds": [1, 2]})
        leaves = load_run(run_dir)
        assert leaves["a.json.gates.some_gate.value"] == 0.891
        assert leaves["a.json.seeds[0]"] == 1
        assert leaves["a.json.seeds[1]"] == 2

    def test_keys_are_relative_to_the_run_directory(self, run_dir):
        _write(run_dir, "nested/b.json", {"x": 1})
        assert "nested/b.json.x" in load_run(run_dir)

    def test_corrupt_file_is_skipped_not_fatal(self, run_dir, capsys):
        _write(run_dir, "good.json", {"x": 1})
        (run_dir / "bad.json").write_text("{not json")
        leaves = load_run(run_dir)
        # One corrupt file must not hide movement in every other file.
        assert leaves == {"good.json.x": 1}
        assert "could not read" in capsys.readouterr().out


class TestCompare:
    @staticmethod
    def _seed_aggregate(prefix, mean, spread):
        return {
            f"{prefix}.schema_version": SEED_AGGREGATION_SCHEMA_VERSION,
            f"{prefix}.state": "available",
            f"{prefix}.sampling_unit": "model_seed",
            f"{prefix}.mean": mean,
            f"{prefix}.seed_std": spread,
        }

    def test_self_diff_reports_nothing(self, run_dir):
        _write(run_dir, "a.json", {
            "macro_f1_mean": 0.418, "macro_f1_std": 0.02,
            "verdict": "pass", "n": 17826, "missing": None,
        })
        leaves = load_run(run_dir)
        result = compare(leaves, leaves, abs_threshold=0.005)
        assert result["moved"] == []
        assert result["changed"] == []
        assert result["added"] == []
        assert result["removed"] == []
        assert result["incomparable"] == []

    def test_mean_is_judged_against_its_seed_std(self):
        old = self._seed_aggregate("f.json.macro_f1", 0.400, 0.05)
        # +0.03 is inside one seed-std (0.05) but far outside the flat threshold.
        new = self._seed_aggregate("f.json.macro_f1", 0.430, 0.05)
        result = compare(old, new, abs_threshold=0.005)
        assert result["moved"] == []

    def test_movement_beyond_one_seed_std_is_flagged(self):
        old = self._seed_aggregate("f.json.macro_f1", 0.400, 0.01)
        new = self._seed_aggregate("f.json.macro_f1", 0.430, 0.01)
        result = compare(old, new, abs_threshold=0.005)
        assert len(result["moved"]) == 1
        key, old_value, new_value, delta, threshold, used_std = result["moved"][0]
        assert key == "f.json.macro_f1.mean"
        assert delta == pytest.approx(0.03)
        assert threshold == 0.01
        assert used_std is True

    def test_leaf_without_sibling_std_uses_absolute_threshold(self):
        old = {"f.json.leakage_fraction": 0.40}
        new = {"f.json.leakage_fraction": 0.41}
        result = compare(old, new, abs_threshold=0.005)
        assert len(result["moved"]) == 1
        assert result["moved"][0][5] is False
        assert result["threshold_fallbacks"] == 1

    def test_zero_seed_std_falls_back_rather_than_flagging_everything(self):
        # A zero std carries no information about spread; using it as a threshold
        # would flag every floating-point wobble as material movement.
        old = self._seed_aggregate("f.json.auroc", 0.894, 0.0)
        new = self._seed_aggregate("f.json.auroc", 0.8941, 0.0)
        result = compare(old, new, abs_threshold=0.005)
        assert result["moved"] == []
        assert result["threshold_fallbacks"] >= 2

    def test_fold_spread_is_not_used_as_a_seed_threshold(self):
        old = {
            "f.json.metric.mean": 0.4,
            "f.json.metric.fold_std": 0.2,
            "f.json.metric.sampling_unit": "cv_fold",
        }
        new = dict(old, **{"f.json.metric.mean": 0.41})
        result = compare(old, new, abs_threshold=0.005)
        assert len(result["moved"]) == 1
        assert result["moved"][0][5] is False

    def test_nan_to_nan_is_unchanged(self):
        old = {"f.json.rho_mean": float("nan")}
        new = {"f.json.rho_mean": float("nan")}
        result = compare(old, new, abs_threshold=0.005)
        assert result["moved"] == []
        assert result["incomparable"] == []
        assert result["unchanged"] == 1

    def test_nan_appearing_is_incomparable_not_movement(self):
        old = {"f.json.rho_mean": 0.5}
        new = {"f.json.rho_mean": float("nan")}
        result = compare(old, new, abs_threshold=0.005)
        # Becoming unmeasurable is a different event from moving, and reporting it as
        # a delta would produce a meaningless NaN in the table.
        assert result["moved"] == []
        assert len(result["incomparable"]) == 1

    def test_gate_verdict_change_is_reported(self):
        old = {"f.json.gates.M2.verdict": "pass, established (CI excludes zero)"}
        new = {"f.json.gates.M2.verdict": "pass on point estimate, not distinguishable"}
        result = compare(old, new, abs_threshold=0.005)
        assert len(result["changed"]) == 1
        assert result["moved"] == []

    def test_boolean_change_is_categorical_not_numeric(self):
        old = {"f.json.gates.some_gate.passed": True}
        new = {"f.json.gates.some_gate.passed": False}
        result = compare(old, new, abs_threshold=0.005)
        assert len(result["changed"]) == 1
        assert result["moved"] == []

    def test_null_becoming_a_number_is_reported_as_a_change(self):
        old = {"f.json.ci_low": None}
        new = {"f.json.ci_low": 0.31}
        result = compare(old, new, abs_threshold=0.005)
        assert len(result["changed"]) == 1

    def test_added_keys_are_not_counted_as_movement(self):
        old = self._seed_aggregate("f.json.macro_f1", 0.4, 0.01)
        new = dict(old, **{"f.json.ci.ci_low": 0.3, "f.json.ci.ci_high": 0.5})
        result = compare(old, new, abs_threshold=0.005)
        # run_biorxiv adds CI keys throughout; folding them into movement would bury
        # the signal this script exists to surface.
        assert result["moved"] == []
        assert result["added"] == ["f.json.ci.ci_high", "f.json.ci.ci_low"]

    def test_removed_keys_are_reported(self):
        old = {"f.json.a": 1, "f.json.b": 2}
        new = {"f.json.a": 1}
        result = compare(old, new, abs_threshold=0.005)
        assert result["removed"] == ["f.json.b"]


class TestFormatReport:
    def test_empty_diff_names_every_bucket(self):
        result = compare({"f.json.x": 1}, {"f.json.x": 1}, abs_threshold=0.005)
        report = format_report("run6", "run_biorxiv", result, 0.005)
        assert "# Delta note — run6 to run_biorxiv" in report
        assert "## Material movement" in report
        assert "## Keys added (0)" in report
        assert "## Keys removed (0)" in report

    def test_moved_rows_show_old_new_delta_and_basis(self):
        old = TestCompare._seed_aggregate("f.json.macro_f1", 0.400, 0.01)
        new = TestCompare._seed_aggregate("f.json.macro_f1", 0.430, 0.01)
        report = format_report(
            "run6", "run_biorxiv", compare(old, new, abs_threshold=0.005), 0.005
        )
        assert "`f.json.macro_f1.mean`" in report
        assert "+0.03" in report
        assert "seed SD" in report
