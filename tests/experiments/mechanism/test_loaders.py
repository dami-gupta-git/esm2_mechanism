"""
Tests for esm2_mech.experiments.mechanism.loaders._label_3class.

Invariants:
- HI and AR collapse to LOF; GOF/DN/LOF pass through unchanged
- an explicit label_3class field short-circuits the mechanism lookup
- an unexpected or missing mechanism raises ValueError (no LOF fallback)
"""

import numpy as np
import pytest

from esm2_mech.experiments.mechanism import loaders
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


def test_load_merged_does_not_require_position_embeddings(monkeypatch):
    calls = []

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        variants = [{"gene": "G1", "label_3class": GOF}]
        return (
            variants,
            np.array([GOF]),
            np.array(["G1"]),
            np.ones((1, 3), dtype=np.float32),
            None,
        )

    monkeypatch.setattr(loaders, "load_variants_and_delta", fake_load)
    delta, labels, genes = loaders.load_merged()

    assert len(calls[0][0]) == 4
    assert calls[0][0][1] == loaders.EMB_VALID_VARIANTS_JSON
    assert calls[0][1] == {"verbose": False}
    assert delta.shape == (1, 3)
    assert labels.tolist() == [GOF]
    assert genes.tolist() == ["G1"]
