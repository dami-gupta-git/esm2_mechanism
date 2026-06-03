"""
Tests for esm2_mech.experiments.mechanism.loaders._label_3class.

Invariants:
- HI and AR collapse to LOF; GOF/DN/LOF pass through unchanged
- an explicit label_3class field short-circuits the mechanism lookup
- an unexpected or missing mechanism raises ValueError (no LOF fallback)
"""

import pytest

from esm2_mech.experiments.mechanism.loaders import _label_3class
from esm2_mech.utils.constants import GOF, DN, LOF


class TestLabel3Class:

    @pytest.mark.parametrize(
        "mechanism, expected",
        [
            ("HI", LOF),
            ("AR", LOF),
            ("LOF", LOF),
            ("GOF", GOF),
            ("DN", DN),
        ],
    )
    def test_mechanism_mapping(self, mechanism, expected):
        assert _label_3class({"mechanism": mechanism}) == expected

    def test_explicit_label_short_circuits(self):
        # label_3class takes precedence and bypasses the mechanism collapse,
        # even when mechanism would map elsewhere.
        variant = {"label_3class": GOF, "mechanism": "HI"}
        assert _label_3class(variant) == GOF

    def test_unexpected_mechanism_raises(self):
        # No silent LOF fallback — an unknown mechanism is a data error.
        with pytest.raises(ValueError):
            _label_3class({"mechanism": "XYZ", "gene": "BRCA1", "aa_pos": 42})

    def test_missing_mechanism_raises(self):
        with pytest.raises(ValueError):
            _label_3class({"gene": "BRCA1", "aa_pos": 42})
