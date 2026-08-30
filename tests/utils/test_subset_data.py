"""
Tests for the row-alignment helpers behind the single-source mechanism arm.

Covers:
- build_source_mask: selects only rows whose source matches
- build_source_mask: a source string that matches nothing yields an all-False mask
- build_source_mask: a variant with no source field is excluded, not defaulted
- build_source_mask: mask length always equals the variant count
- subset_data: arrays and lists stay row-aligned after subsetting
- subset_data: an array whose row count differs from valid_variants raises
- subset_data: a list whose row count differs from valid_variants raises
- subset_data: a mask shorter than the data raises instead of truncating lists
- subset_data: a mask longer than the data raises
- subset_data: a non-row-aligned value type raises
- subset_data: an all-False mask yields empty, still-aligned outputs
"""

import numpy as np
import pytest

from esm2_mech.utils.constants import SOURCE_GERASIMAVICIUS
from esm2_mech.utils.data import build_source_mask, subset_data


def _variants(sources):
    """One variant dict per source; None means the source field is absent."""
    out = []
    for index, source in enumerate(sources):
        variant = {"gene": f"G{index}", "aa_pos": index + 1}
        if source is not None:
            variant["source"] = source
        out.append(variant)
    return out


# ---------------------------------------------------------------------------
# build_source_mask
# ---------------------------------------------------------------------------


def test_build_source_mask_selects_matching_rows():
    variants = _variants([SOURCE_GERASIMAVICIUS, "other", SOURCE_GERASIMAVICIUS])
    mask = build_source_mask(variants, SOURCE_GERASIMAVICIUS)
    assert mask.dtype == bool
    assert mask.tolist() == [True, False, True]


def test_build_source_mask_unknown_source_selects_nothing():
    """A renamed or typo'd source must produce an empty subset, not a plausible one."""
    variants = _variants([SOURCE_GERASIMAVICIUS, SOURCE_GERASIMAVICIUS])
    mask = build_source_mask(variants, "gerasimavicious")
    assert not mask.any()


def test_build_source_mask_missing_source_field_is_excluded():
    """Absent provenance is excluded rather than assigned to the requested source."""
    variants = _variants([SOURCE_GERASIMAVICIUS, None, "other"])
    mask = build_source_mask(variants, SOURCE_GERASIMAVICIUS)
    assert mask.tolist() == [True, False, False]


def test_build_source_mask_length_matches_variant_count():
    variants = _variants([None, "other", SOURCE_GERASIMAVICIUS, None])
    mask = build_source_mask(variants, SOURCE_GERASIMAVICIUS)
    assert len(mask) == len(variants)


# ---------------------------------------------------------------------------
# subset_data
# ---------------------------------------------------------------------------


def _data(n_rows=4):
    return {
        "valid_variants": [{"gene": f"G{i}"} for i in range(n_rows)],
        "labels_3class": np.array([f"L{i}" for i in range(n_rows)]),
        "features": np.arange(n_rows * 2, dtype=float).reshape(n_rows, 2),
        "aa_wt_list": [f"W{i}" for i in range(n_rows)],
    }


def test_subset_data_keeps_arrays_and_lists_aligned():
    data = _data()
    mask = np.array([True, False, True, False])
    subset = subset_data(data, mask)

    assert [v["gene"] for v in subset["valid_variants"]] == ["G0", "G2"]
    assert subset["labels_3class"].tolist() == ["L0", "L2"]
    assert subset["aa_wt_list"] == ["W0", "W2"]
    assert subset["features"].tolist() == [[0.0, 1.0], [4.0, 5.0]]
    assert all(len(value) == int(mask.sum()) for value in subset.values())


def test_subset_data_rejects_misaligned_array():
    data = _data()
    data["features"] = np.zeros((3, 2))
    with pytest.raises(ValueError, match="not row-aligned"):
        subset_data(data, np.array([True, False, True, False]))


def test_subset_data_rejects_misaligned_list():
    data = _data()
    data["aa_wt_list"] = ["W0", "W1"]
    with pytest.raises(ValueError, match="not row-aligned"):
        subset_data(data, np.array([True, False, True, False]))


def test_subset_data_rejects_short_mask():
    """A short mask silently truncates a list comprehension; it must raise instead."""
    data = _data()
    with pytest.raises(ValueError, match="mask"):
        subset_data(data, np.array([True, False]))


def test_subset_data_rejects_long_mask():
    data = _data()
    with pytest.raises(ValueError, match="mask"):
        subset_data(data, np.array([True, False, True, False, True]))


def test_subset_data_rejects_unexpected_value_type():
    data = _data()
    data["alphamissense_scores"] = {"G0": 0.9}
    with pytest.raises(TypeError):
        subset_data(data, np.array([True, False, True, False]))


def test_subset_data_all_false_mask_yields_empty_aligned_outputs():
    data = _data()
    subset = subset_data(data, np.zeros(4, dtype=bool))
    assert subset["valid_variants"] == []
    assert subset["aa_wt_list"] == []
    assert subset["labels_3class"].shape[0] == 0
    assert subset["features"].shape == (0, 2)
