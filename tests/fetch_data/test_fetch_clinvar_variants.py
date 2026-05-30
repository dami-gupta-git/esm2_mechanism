"""
Tests for fetch_clinvar_variants.py.

Covers:
- _parse_hgvsp: 3-letter HGVSp formats
- _parse_hgvsp: 1-letter HGVSp formats
- _parse_hgvsp: synonymous variants are excluded
- _parse_hgvsp: stop-gain variants are excluded
- _parse_hgvsp: unparseable strings return None
- validate_wt: correct WT amino acid passes
- validate_wt: wrong WT amino acid fails
- validate_wt: out-of-bounds positions fail
- fetch_clinvar_variants: cache hit returns cached data without HTTP
- fetch_uniprot_id: prefilled value is written to cache and returned
- fetch_uniprot_id: cache hit returns without hitting REST API
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from esm2_mech.fetch_data.fetch_variants import (
    _parse_hgvsp,
    validate_wt,
    fetch_clinvar_variants,
    fetch_uniprot_id,
)

# ---------------------------------------------------------------------------
# _parse_hgvsp
# ---------------------------------------------------------------------------


class TestParseHgvsp:

    def test_three_letter_missense(self):
        result = _parse_hgvsp("NM_004333.6(BRAF):c.1799T>A (p.Val600Glu)")
        assert result == ("V", 600, "E")

    def test_one_letter_missense(self):
        result = _parse_hgvsp("p.V600E")
        assert result == ("V", 600, "E")

    def test_three_letter_bare(self):
        result = _parse_hgvsp("p.Arg117His")
        assert result == ("R", 117, "H")

    def test_three_letter_synonymous_returns_none(self):
        # wt == mut after mapping should return None; construct a synthetic case
        # p.Ala1Ala — same amino acid
        result = _parse_hgvsp("p.Ala1Ala")
        assert result is None

    def test_stop_gain_returns_none(self):
        result = _parse_hgvsp("p.Arg117Ter")
        assert result is None

    def test_wt_stop_returns_none(self):
        # p.Ter → some AA — excluded
        result = _parse_hgvsp("p.Ter500Val")
        assert result is None

    def test_frameshift_returns_none(self):
        result = _parse_hgvsp("p.Arg117fs")
        assert result is None

    def test_empty_string_returns_none(self):
        assert _parse_hgvsp("") is None

    def test_cdna_only_no_protein_returns_none(self):
        assert _parse_hgvsp("NM_004333.6:c.1799T>A") is None

    def test_unknown_three_letter_code_returns_none(self):
        # Xaa is mapped to "X" which is a valid code — use an invalid abbreviation
        result = _parse_hgvsp("p.Zzz100Val")
        assert result is None

    def test_position_extracted_correctly(self):
        wt, pos, mut = _parse_hgvsp("p.Lys1000Arg")
        assert pos == 1000

    def test_one_letter_synonymous_returns_none(self):
        result = _parse_hgvsp("p.V600V")
        assert result is None


# ---------------------------------------------------------------------------
# validate_wt
# ---------------------------------------------------------------------------


class TestValidateWt:

    SEQ = "MKTAYIAKQR"  # 10 AA, 1-indexed

    def test_correct_wt_passes(self):
        variant = {"pos": 1, "wt_aa": "M"}
        assert validate_wt(variant, self.SEQ) is True

    def test_wrong_wt_fails(self):
        variant = {"pos": 1, "wt_aa": "A"}
        assert validate_wt(variant, self.SEQ) is False

    def test_position_zero_is_out_of_bounds(self):
        variant = {"pos": 0, "wt_aa": "M"}
        assert validate_wt(variant, self.SEQ) is False

    def test_position_beyond_length_is_out_of_bounds(self):
        variant = {"pos": 11, "wt_aa": "X"}
        assert validate_wt(variant, self.SEQ) is False

    def test_last_position_passes(self):
        variant = {"pos": 10, "wt_aa": "R"}
        assert validate_wt(variant, self.SEQ) is True

    def test_mid_sequence_correct(self):
        variant = {"pos": 5, "wt_aa": "Y"}  # SEQ[4] == "Y"
        assert validate_wt(variant, self.SEQ) is True


# ---------------------------------------------------------------------------
# fetch_clinvar_variants – cache hit (no HTTP)
# ---------------------------------------------------------------------------


class TestFetchClinvarVariantsCache:

    def test_cache_hit_returns_cached_data(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "esm2_mech.fetch_data.fetch_variants.CLINVAR_CACHE",
            tmp_path,
        )
        cached = [
            {
                "hgvs_p": "p.Val600Glu",
                "wt_aa": "V",
                "pos": 600,
                "mut_aa": "E",
                "clinsig": "pathogenic",
            }
        ]
        (tmp_path / "BRAF.json").write_text(json.dumps(cached))

        result = fetch_clinvar_variants("BRAF")
        assert result == cached

    def test_cache_hit_does_not_call_network(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "esm2_mech.fetch_data.fetch_variants.CLINVAR_CACHE",
            tmp_path,
        )
        (tmp_path / "BRAF.json").write_text(json.dumps([]))

        with patch("esm2_mech.fetch_data.fetch_variants._get_json") as mock_get:
            fetch_clinvar_variants("BRAF")
            mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# fetch_uniprot_id
# ---------------------------------------------------------------------------


class TestFetchUniprotId:

    def test_prefilled_value_returned_and_cached(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "esm2_mech.fetch_data.fetch_variants.UNIPROT_CACHE",
            tmp_path,
        )
        result = fetch_uniprot_id("BRAF", "P15056")
        assert result == "P15056"
        cached = json.loads((tmp_path / "BRAF.json").read_text())
        assert cached["uniprot_id"] == "P15056"

    def test_cache_hit_returns_without_api_call(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "esm2_mech.fetch_data.fetch_variants.UNIPROT_CACHE",
            tmp_path,
        )
        (tmp_path / "BRAF.json").write_text(json.dumps({"uniprot_id": "P15056"}))

        with patch("esm2_mech.fetch_data.fetch_variants._get_json") as mock_get:
            result = fetch_uniprot_id("BRAF", None)
            mock_get.assert_not_called()
        assert result == "P15056"

    def test_network_failure_not_cached(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "esm2_mech.fetch_data.fetch_variants.UNIPROT_CACHE",
            tmp_path,
        )
        with patch(
            "esm2_mech.fetch_data.fetch_variants._get_json",
            return_value=None,
        ):
            result = fetch_uniprot_id("UNKNOWN_GENE", None)
        assert result is None
        assert not (tmp_path / "UNKNOWN_GENE.json").exists()

    def test_exact_gene_match_preferred_over_first_result(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "esm2_mech.fetch_data.fetch_variants.UNIPROT_CACHE",
            tmp_path,
        )
        api_response = {
            "results": [
                {
                    "primaryAccession": "Q99999",
                    "genes": [{"geneName": {"value": "OTHER"}}],
                },
                {
                    "primaryAccession": "P15056",
                    "genes": [{"geneName": {"value": "BRAF"}}],
                },
            ]
        }
        with patch(
            "esm2_mech.fetch_data.fetch_variants._get_json",
            return_value=api_response,
        ):
            result = fetch_uniprot_id("BRAF", None)
        assert result == "P15056"

    def test_no_exact_match_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "esm2_mech.fetch_data.fetch_variants.UNIPROT_CACHE",
            tmp_path,
        )
        api_response = {
            "results": [
                {
                    "primaryAccession": "Q11111",
                    "genes": [{"geneName": {"value": "BRAF_ALIAS"}}],
                },
            ]
        }
        with patch(
            "esm2_mech.fetch_data.fetch_variants._get_json",
            return_value=api_response,
        ):
            result = fetch_uniprot_id("BRAF", None)
        assert result is None
        cached = json.loads((tmp_path / "BRAF.json").read_text())
        assert cached["uniprot_id"] is None
