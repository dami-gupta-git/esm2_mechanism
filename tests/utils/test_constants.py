"""
Tests for esm2_mech.utils.constants.

Invariants:
- Model ID strings are non-empty and match the expected ESM naming convention
- MECHANISM_CLASSES has exactly 3 entries in canonical order
- Numeric limits are positive and self-consistent
- UNIPROT_REST is a well-formed HTTPS URL
"""

from esm2_mech.utils.constants import (
    ESM2_MODEL,
    ESM3_MODEL,
    MECHANISM_CLASSES,
    MAX_SEQ_LEN,
    WINDOW_HALF,
    UNIPROT_REST,
)


def test_esm2_model_nonempty():
    assert isinstance(ESM2_MODEL, str) and ESM2_MODEL


def test_esm3_model_nonempty():
    assert isinstance(ESM3_MODEL, str) and ESM3_MODEL


def test_mechanism_classes_canonical_order():
    assert MECHANISM_CLASSES == ["GOF", "DN", "LOF"]


def test_mechanism_classes_length():
    assert len(MECHANISM_CLASSES) == 3


def test_max_seq_len_positive():
    assert MAX_SEQ_LEN > 0


def test_window_half_positive():
    assert WINDOW_HALF > 0


def test_window_half_fits_in_max_seq_len():
    # Two half-windows must not exceed the full allowed length.
    assert WINDOW_HALF * 2 <= MAX_SEQ_LEN


def test_uniprot_rest_is_https():
    assert UNIPROT_REST.startswith("https://")


def test_uniprot_rest_contains_uniprot():
    assert "uniprot" in UNIPROT_REST.lower()
