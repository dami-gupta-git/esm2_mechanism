"""
Tests for esm2_mech.utils.eval.

Invariants:
- run_family_split_eval: returns 1 on empty rows
- run_family_split_eval: returns 0 and writes output files on valid input
- run_family_split_eval: overall.json contains correct n, n_pos, n_neg, auroc
- run_family_split_eval: per_family.json excludes families below min_pos/min_neg
- run_family_split_eval: summary.json written when any families pass threshold
- run_family_split_eval: summary auroc distribution stats are self-consistent
- run_family_split_eval: skips families with too few positives or negatives
- vkey: produces gene_pos_wt_mut format
"""

import json
from pathlib import Path

import pytest

from esm2_mech.utils.eval import run_family_split_eval, vkey


# ---------------------------------------------------------------------------
# vkey
# ---------------------------------------------------------------------------

def test_vkey_format():
    v = {"gene": "BRCA1", "aa_pos": 100, "aa_wt": "A", "aa_mut": "V"}
    assert vkey(v) == "BRCA1_100_A_V"


def test_vkey_numeric_pos():
    v = {"gene": "TP53", "aa_pos": 1, "aa_wt": "M", "aa_mut": "K"}
    assert vkey(v) == "TP53_1_M_K"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _paths(tmp_path):
    return (
        tmp_path / "overall.json",
        tmp_path / "per_family.json",
        tmp_path / "summary.json",
    )


def _call(rows, tmp_path, **kwargs):
    overall, per_family, summary = _paths(tmp_path)
    return run_family_split_eval(
        rows,
        overall_path=overall,
        per_family_path=per_family,
        summary_path=summary,
        **kwargs,
    )


def _make_rows(n_pos, n_neg, family, perfect=True):
    """Build (label, score, family) rows. perfect=True means scores separate perfectly."""
    rows = []
    for _ in range(n_pos):
        rows.append((1, 0.9 if perfect else 0.5, family))
    for _ in range(n_neg):
        rows.append((0, 0.1 if perfect else 0.5, family))
    return rows


# ---------------------------------------------------------------------------
# run_family_split_eval
# ---------------------------------------------------------------------------

class TestRunFamilySplitEval:

    def test_empty_rows_returns_1(self, tmp_path):
        assert _call([], tmp_path) == 1

    def test_valid_input_returns_0(self, tmp_path):
        assert _call(_make_rows(15, 15, "FAM1"), tmp_path) == 0

    def test_overall_json_written(self, tmp_path):
        _call(_make_rows(15, 15, "FAM1"), tmp_path)
        assert (tmp_path / "overall.json").exists()

    def test_per_family_json_written(self, tmp_path):
        _call(_make_rows(15, 15, "FAM1"), tmp_path)
        assert (tmp_path / "per_family.json").exists()

    def test_overall_counts_correct(self, tmp_path):
        _call(_make_rows(15, 20, "FAM1"), tmp_path)
        overall = json.loads((tmp_path / "overall.json").read_text())
        assert overall["n"] == 35
        assert overall["n_pos"] == 15
        assert overall["n_neg"] == 20

    def test_perfect_auroc(self, tmp_path):
        _call(_make_rows(15, 15, "FAM1", perfect=True), tmp_path)
        overall = json.loads((tmp_path / "overall.json").read_text())
        assert overall["auroc"] == pytest.approx(1.0)

    def test_family_below_min_excluded(self, tmp_path):
        rows = _make_rows(15, 15, "FAM1") + _make_rows(2, 2, "FAM2")
        _call(rows, tmp_path, min_pos=10, min_neg=10)
        per_family = json.loads((tmp_path / "per_family.json").read_text())
        assert "FAM1" in per_family
        assert "FAM2" not in per_family

    def test_summary_written_when_families_pass(self, tmp_path):
        _call(_make_rows(15, 15, "FAM1"), tmp_path, min_pos=10, min_neg=10)
        assert (tmp_path / "summary.json").exists()

    def test_summary_n_families(self, tmp_path):
        rows = _make_rows(15, 15, "FAM1") + _make_rows(12, 12, "FAM2")
        _call(rows, tmp_path, min_pos=10, min_neg=10)
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["n_families"] == 2

    def test_summary_stats_self_consistent(self, tmp_path):
        rows = _make_rows(15, 15, "FAM1") + _make_rows(12, 12, "FAM2")
        _call(rows, tmp_path, min_pos=10, min_neg=10)
        s = json.loads((tmp_path / "summary.json").read_text())
        assert s["per_family_min"] <= s["per_family_median"] <= s["per_family_max"]
        assert s["per_family_iqr"] == pytest.approx(s["per_family_q75"] - s["per_family_q25"])

    def test_no_summary_when_no_families_pass(self, tmp_path):
        _call(_make_rows(2, 2, "FAM1"), tmp_path, min_pos=10, min_neg=10)
        assert not (tmp_path / "summary.json").exists()

    def test_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "a" / "b"
        rows = _make_rows(15, 15, "FAM1")
        run_family_split_eval(
            rows,
            overall_path=nested / "overall.json",
            per_family_path=nested / "per_family.json",
            summary_path=nested / "summary.json",
        )
        assert nested.exists()
