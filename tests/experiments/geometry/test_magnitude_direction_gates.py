"""
Tests for esm2_mech.experiments.geometry.magnitude_direction gate logic.

Invariants:
- _is_missing flags None and NaN (the two "no real observation" cases) and
  nothing else.
- evaluate_gates returns passed=None (SKIP) when a gate's input is NaN — an
  all-NaN probe is missing data, never a genuine pass/fail.
- evaluate_gates returns passed=None and does not crash when the P3 chance
  floor is None (None + margin would otherwise raise TypeError).
- a fully-populated, clearly-passing input yields passed=True for every gate.
"""

import numpy as np

from esm2_mech.experiments.geometry.magnitude_direction import (
    _is_missing,
    evaluate_gates,
)


def _path_block(mag_auroc, dir_auroc):
    """Minimal path_res with the family_split logreg/mlp AUROCs the gates read."""
    def block(auroc):
        return {
            "family_split": {
                "logreg_auroc": {"mean": auroc},
                "mlp_auroc": {"mean": auroc},
            }
        }
    return {"mag": block(mag_auroc), "dir": block(dir_auroc)}


def _mech_res(floor, dir_f1):
    return {
        "chance_floor": {"family_split": {"mean": floor}},
        "dir": {"family_split": {"mlp_macro_f1": {"mean": dir_f1}}},
    }


class TestIsMissing:

    def test_none_is_missing(self):
        assert _is_missing(None)

    def test_nan_is_missing(self):
        assert _is_missing(float("nan"))

    def test_real_value_not_missing(self):
        assert not _is_missing(0.0)
        assert not _is_missing(0.85)


class TestEvaluateGates:

    def test_all_pass_when_populated(self):
        # mag high (>=0.85), dir low (<=0.70), dir-F1 at/below floor+margin.
        path_res = _path_block(mag_auroc=0.90, dir_auroc=0.60)
        mech_res = _mech_res(floor=0.36, dir_f1=0.36)
        gates = evaluate_gates(path_res, mech_res, bio_res=None)
        assert gates["P1"]["passed"] is True
        assert gates["P2"]["passed"] is True
        assert gates["P3"]["passed"] is True
        # bio_res None → P4 explicitly skipped
        assert gates["P4"]["passed"] is None

    def test_nan_inputs_skip_not_fail(self):
        # Every family-split mean is NaN (no fold scored) → SKIP, never False.
        nan = float("nan")
        path_res = _path_block(mag_auroc=nan, dir_auroc=nan)
        mech_res = _mech_res(floor=0.36, dir_f1=nan)
        gates = evaluate_gates(path_res, mech_res, bio_res=None)
        assert gates["P1"]["passed"] is None
        assert gates["P2"]["passed"] is None
        assert gates["P3"]["passed"] is None

    def test_none_chance_floor_skips_without_crash(self):
        # A None P3 chance floor must not raise (None + margin) — it is SKIP.
        path_res = _path_block(mag_auroc=0.90, dir_auroc=0.60)
        mech_res = _mech_res(floor=None, dir_f1=0.36)
        gates = evaluate_gates(path_res, mech_res, bio_res=None)
        assert gates["P3"]["passed"] is None
        assert gates["P3"]["threshold"] is None

    def test_p4_nan_means_skip(self):
        path_res = _path_block(mag_auroc=0.90, dir_auroc=0.60)
        mech_res = _mech_res(floor=0.36, dir_f1=0.36)
        bio_res = {
            "c2_sign_auroc": {
                "full": {"mean": float("nan")},
                "dir": {"mean": float("nan")},
            },
            "c1_spearman_mag_absddg": float("nan"),
        }
        gates = evaluate_gates(path_res, mech_res, bio_res)
        assert gates["P4"]["passed"] is None
