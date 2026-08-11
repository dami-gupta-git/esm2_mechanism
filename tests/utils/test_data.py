"""
Tests for utils/data.py pure-logic functions.

Covers:
- load_variants: returns all variants when all fields are present
- load_variants: filters out variants with missing uniprot_id
- load_variants: filters out variants with missing aa_wt
- load_variants: filters out variants with missing aa_mut
- load_variants: filters out variants with aa_pos == 0
- load_variants: filters out variants with aa_pos < 0 (negative)
- load_variants: handles empty input list
- build_gene_to_row: assigns row indices in order of first appearance
- build_gene_to_row: deduplicates repeated genes, keeps first occurrence index
- build_gene_to_row: strips whitespace from gene names
- build_gene_to_row: returns empty dict for header-only TSV
"""

import json

import numpy as np
import pytest

from esm2_mech.utils.data import build_gene_to_row, load_variants, observed_rows_mask


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _variant(uid="P1", aa_wt="M", aa_mut="V", aa_pos=5, **extra):
    return {"uniprot_id": uid, "aa_wt": aa_wt, "aa_mut": aa_mut, "aa_pos": aa_pos, **extra}


def _write_json(path, data):
    path.write_text(json.dumps(data))
    return path


def _write_tsv(path, rows):
    """Write a TSV with a 'gene' column. rows is a list of gene name strings."""
    lines = ["gene\tmechanism"] + [f"{g}\tGOF" for g in rows]
    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# load_variants
# ---------------------------------------------------------------------------

class TestLoadVariants:

    def test_valid_variant_passes_through(self, tmp_path):
        data = [_variant()]
        path = _write_json(tmp_path / "v.json", data)
        result = load_variants(path)
        assert len(result) == 1

    def test_filters_missing_uniprot_id(self, tmp_path):
        data = [_variant(uid=""), _variant(uid=None), _variant()]
        path = _write_json(tmp_path / "v.json", data)
        result = load_variants(path)
        assert len(result) == 1

    def test_filters_missing_aa_wt(self, tmp_path):
        data = [_variant(aa_wt=""), _variant(aa_wt=None), _variant()]
        path = _write_json(tmp_path / "v.json", data)
        result = load_variants(path)
        assert len(result) == 1

    def test_filters_missing_aa_mut(self, tmp_path):
        data = [_variant(aa_mut=""), _variant(aa_mut=None), _variant()]
        path = _write_json(tmp_path / "v.json", data)
        result = load_variants(path)
        assert len(result) == 1

    def test_filters_aa_pos_zero(self, tmp_path):
        data = [_variant(aa_pos=0), _variant()]
        path = _write_json(tmp_path / "v.json", data)
        result = load_variants(path)
        assert len(result) == 1

    def test_filters_negative_aa_pos(self, tmp_path):
        data = [_variant(aa_pos=-1), _variant()]
        path = _write_json(tmp_path / "v.json", data)
        result = load_variants(path)
        assert len(result) == 1

    def test_empty_input_returns_empty(self, tmp_path):
        path = _write_json(tmp_path / "v.json", [])
        result = load_variants(path)
        assert result == []

    def test_all_invalid_returns_empty(self, tmp_path):
        data = [_variant(uid=""), _variant(aa_pos=0), _variant(aa_wt=None)]
        path = _write_json(tmp_path / "v.json", data)
        result = load_variants(path)
        assert result == []

    def test_preserves_extra_fields(self, tmp_path):
        data = [_variant(gene="BRCA1", label_3class="GOF")]
        path = _write_json(tmp_path / "v.json", data)
        result = load_variants(path)
        assert result[0]["gene"] == "BRCA1"
        assert result[0]["label_3class"] == "GOF"

    def test_multiple_valid_variants(self, tmp_path):
        data = [_variant(uid=f"P{i}") for i in range(5)]
        path = _write_json(tmp_path / "v.json", data)
        result = load_variants(path)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# build_gene_to_row
# ---------------------------------------------------------------------------

class TestBuildGeneToRow:

    def test_assigns_indices_in_order(self, tmp_path):
        path = _write_tsv(tmp_path / "genes.tsv", ["BRCA1", "TP53", "MYC"])
        result = build_gene_to_row(path)
        assert result == {"BRCA1": 0, "TP53": 1, "MYC": 2}

    def test_deduplicates_repeated_gene(self, tmp_path):
        path = _write_tsv(tmp_path / "genes.tsv", ["BRCA1", "TP53", "BRCA1"])
        result = build_gene_to_row(path)
        assert result["BRCA1"] == 0
        assert result["TP53"] == 1
        assert len(result) == 2

    def test_second_occurrence_does_not_shift_indices(self, tmp_path):
        # BRCA1 appears at rows 0 and 2; TP53 should still be index 1
        path = _write_tsv(tmp_path / "genes.tsv", ["BRCA1", "TP53", "BRCA1", "MYC"])
        result = build_gene_to_row(path)
        assert result["TP53"] == 1
        assert result["MYC"] == 2

    def test_strips_whitespace_from_gene_names(self, tmp_path):
        path = tmp_path / "genes.tsv"
        path.write_text("gene\tmechanism\n BRCA1 \tGOF\n")
        result = build_gene_to_row(path)
        assert "BRCA1" in result
        assert " BRCA1 " not in result

    def test_returns_empty_for_header_only(self, tmp_path):
        path = tmp_path / "genes.tsv"
        path.write_text("gene\tmechanism\n")
        result = build_gene_to_row(path)
        assert result == {}

    def test_single_gene(self, tmp_path):
        path = _write_tsv(tmp_path / "genes.tsv", ["BRCA1"])
        result = build_gene_to_row(path)
        assert result == {"BRCA1": 0}


# ---------------------------------------------------------------------------
# observed_rows_mask — complete-case restriction for models that reject NaN
# ---------------------------------------------------------------------------

class TestObservedRowsMask:

    def test_all_observed_keeps_every_row(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert observed_rows_mask(X).all()

    def test_drops_row_with_any_nan(self):
        X = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, np.nan]])
        assert observed_rows_mask(X).tolist() == [True, False, False]

    def test_only_named_columns_decide(self):
        # A NaN in a column this probe does not use must not drop the row —
        # otherwise a single sparse feature silently shrinks every other arm.
        X = np.array([[1.0, np.nan], [2.0, np.nan]])
        assert observed_rows_mask(X, col_idx=[0]).all()
        assert not observed_rows_mask(X, col_idx=[1]).any()

    def test_fully_nan_column_drops_everything(self):
        X = np.array([[1.0, np.nan], [2.0, np.nan]])
        assert not observed_rows_mask(X).any()

    def test_mask_is_boolean_and_row_aligned(self):
        X = np.array([[1.0], [np.nan], [3.0]])
        mask = observed_rows_mask(X)
        assert mask.dtype == bool
        assert len(mask) == len(X)
