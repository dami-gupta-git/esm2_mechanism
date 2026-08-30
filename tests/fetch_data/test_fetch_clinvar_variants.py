"""
Tests for fetch_clinvar_variants.py.

Covers:
- _parse_hgvsp: missense is parsed from 1-letter, 3-letter and full HGVS forms
- _parse_hgvsp: synonymous, stop and frameshift variants return None
- _parse_hgvsp: empty, cDNA-only and unparseable strings return None
- validate_wt: the wild-type residue is checked against the sequence
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

    @pytest.mark.parametrize(
        ("hgvsp", "expected"),
        [
            ("NM_004333.6(BRAF):c.1799T>A (p.Val600Glu)", ("V", 600, "E")),
            ("p.V600E", ("V", 600, "E")),
            ("p.Arg117His", ("R", 117, "H")),
            ("p.Lys1000Arg", ("K", 1000, "R")),
        ],
        ids=["three_letter_in_full_hgvs", "one_letter", "three_letter_bare",
             "four_digit_position"],
    )
    def test_missense_is_parsed(self, hgvsp, expected):
        assert _parse_hgvsp(hgvsp) == expected

    @pytest.mark.parametrize(
        "hgvsp",
        [
            # Same amino acid either side, in both notations.
            "p.Ala1Ala",
            "p.V600V",
            # A stop codon on either side is not a missense substitution.
            "p.Arg117Ter",
            "p.Ter500Val",
            "p.Arg117fs",
            "",
            # cDNA with no protein consequence.
            "NM_004333.6:c.1799T>A",
            # Zzz is not a real abbreviation. Xaa would map to the valid code "X".
            "p.Zzz100Val",
        ],
        ids=["synonymous_three_letter", "synonymous_one_letter", "stop_gain",
             "stop_as_wild_type", "frameshift", "empty_string", "cdna_only",
             "unknown_three_letter_code"],
    )
    def test_non_missense_returns_none(self, hgvsp):
        assert _parse_hgvsp(hgvsp) is None


# ---------------------------------------------------------------------------
# validate_wt
# ---------------------------------------------------------------------------


class TestValidateWt:

    SEQ = "MKTAYIAKQR"  # 10 AA, 1-indexed

    @pytest.mark.parametrize(
        ("pos", "wt_aa", "expected"),
        [
            (1, "M", True),
            (10, "R", True),   # last position
            (5, "Y", True),    # SEQ[4] == "Y"
            (1, "A", False),   # wrong residue at a valid position
            (0, "M", False),   # 1-indexed, so 0 is out of bounds
            (11, "X", False),  # one past the end
        ],
        ids=["first_position", "last_position", "mid_sequence", "wrong_residue",
             "position_zero", "position_beyond_length"],
    )
    def test_wt_residue_is_checked_against_the_sequence(self, pos, wt_aa, expected):
        assert validate_wt({"pos": pos, "wt_aa": wt_aa}, self.SEQ) is expected


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

        # A cache hit is always complete by construction.
        result, complete = fetch_clinvar_variants("BRAF")
        assert result == cached
        assert complete is True

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
