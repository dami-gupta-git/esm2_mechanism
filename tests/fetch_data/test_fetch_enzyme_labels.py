"""
Tests for enzyme annotation parsing, classification, and fetch failure handling.
"""

import json

import pytest
from esm2_mech.fetch_data import fetch_annotations
from esm2_mech.fetch_data.fetch_annotations import (
    parse_ec_and_keywords,
    assign_4class,
)
from esm2_mech.fetch_data.uniprot_fetch import TransientFetchError

# ---------------------------------------------------------------------------
# parse_ec_and_keywords
# ---------------------------------------------------------------------------


class TestParseEcAndKeywords:

    def test_ec_from_recommended_name(self):
        entry = {
            "proteinDescription": {
                "recommendedName": {"ecNumbers": [{"value": "2.7.11.1"}]}
            }
        }
        ec, kw = parse_ec_and_keywords(entry)
        assert ec == ["2.7.11.1"]
        assert kw == []

    def test_ec_from_alternative_names(self):
        entry = {
            "proteinDescription": {
                "recommendedName": {},
                "alternativeNames": [{"ecNumbers": [{"value": "3.4.21.4"}]}],
            }
        }
        ec, kw = parse_ec_and_keywords(entry)
        assert "3.4.21.4" in ec

    def test_ec_from_catalytic_activity_string(self):
        entry = {
            "comments": [
                {
                    "commentType": "CATALYTIC ACTIVITY",
                    "reaction": {"ecNumber": "1.1.1.1"},
                }
            ]
        }
        ec, kw = parse_ec_and_keywords(entry)
        assert "1.1.1.1" in ec

    def test_ec_from_catalytic_activity_list_of_strings(self):
        entry = {
            "comments": [
                {
                    "commentType": "CATALYTIC ACTIVITY",
                    "reaction": {"ecNumber": ["1.1.1.1", "1.1.1.2"]},
                }
            ]
        }
        ec, kw = parse_ec_and_keywords(entry)
        assert "1.1.1.1" in ec
        assert "1.1.1.2" in ec

    def test_ec_deduplicated_across_sources(self):
        entry = {
            "proteinDescription": {
                "recommendedName": {"ecNumbers": [{"value": "2.7.11.1"}]}
            },
            "comments": [
                {
                    "commentType": "CATALYTIC ACTIVITY",
                    "reaction": {"ecNumber": "2.7.11.1"},
                }
            ],
        }
        ec, _ = parse_ec_and_keywords(entry)
        assert ec.count("2.7.11.1") == 1

    def test_keywords_extracted(self):
        entry = {
            "keywords": [
                {"id": "KW-0418", "name": "Kinase"},
                {"id": "KW-0067", "name": "ATP-binding"},
            ]
        }
        _, kw = parse_ec_and_keywords(entry)
        assert "KW-0418" in kw
        assert "KW-0067" in kw

    def test_empty_entry_returns_empty_lists(self):
        ec, kw = parse_ec_and_keywords({})
        assert ec == []
        assert kw == []

    def test_recommended_name_null_does_not_drop_alternative_names(self):
        """recommendedName: null (not missing) must not cause alternative name ECs to be lost."""
        entry = {
            "proteinDescription": {
                "recommendedName": None,
                "alternativeNames": [{"ecNumbers": [{"value": "3.4.21.4"}]}],
            }
        }
        ec, _ = parse_ec_and_keywords(entry)
        assert "3.4.21.4" in ec

    def test_non_catalytic_comments_ignored(self):
        entry = {
            "comments": [
                {"commentType": "FUNCTION", "reaction": {"ecNumber": "2.7.11.1"}}
            ]
        }
        ec, _ = parse_ec_and_keywords(entry)
        assert ec == []

    def test_all_keyword_ids_returned(self):
        """All keyword IDs are returned — no silent truncation."""
        entry = {"keywords": [{"id": f"KW-{i:04d}"} for i in range(20)]}
        _, kw = parse_ec_and_keywords(entry)
        assert len(kw) == 20


# ---------------------------------------------------------------------------
# assign_4class
# ---------------------------------------------------------------------------


class TestAssign4Class:

    def test_kinase_by_keyword(self):
        assert assign_4class([], ["KW-0418"]) == "kinase"

    def test_kinase_by_ec(self):
        assert assign_4class(["2.7.11.1"], []) == "kinase"

    def test_kinase_keyword_takes_priority_over_ec(self):
        # KW-0418 present alongside a protease EC — kinase wins
        assert assign_4class(["3.4.21.4"], ["KW-0418"]) == "kinase"

    def test_protease_by_ec(self):
        assert assign_4class(["3.4.21.4"], []) == "protease"

    def test_protease_not_triggered_by_other_ec3(self):
        # EC 3.x but not 3.4 is outside the simplified four-class cohort.
        assert assign_4class(["3.1.1.1"], []) is None

    def test_oxidoreductase_by_ec(self):
        assert assign_4class(["1.1.1.1"], []) == "oxidoreductase"

    def test_oxidoreductase_not_triggered_by_non_ec1(self):
        assert assign_4class(["2.1.1.1"], []) is None

    def test_non_enzyme_empty(self):
        assert assign_4class([], []) == "non-enzyme"

    def test_non_enzyme_unrecognised_ec(self):
        assert assign_4class(["4.2.1.1"], []) is None

    def test_kinase_beats_protease_priority(self):
        # Both kinase and protease EC present — kinase wins
        assert assign_4class(["2.7.11.1", "3.4.21.4"], []) == "kinase"

    def test_protease_beats_oxidoreductase_priority(self):
        assert assign_4class(["3.4.21.4", "1.1.1.1"], []) == "protease"

    def test_ec_27_not_kinase_without_27_prefix(self):
        # EC 2.6 is a transferase but not a kinase
        assert assign_4class(["2.6.1.1"], []) is None


def test_transient_fetch_failure_does_not_replace_enzyme_labels(tmp_path, monkeypatch):
    variants_path = tmp_path / "variants.json"
    genes_path = tmp_path / "gene_list.tsv"
    output_path = tmp_path / "enzyme_labels.tsv"
    cache_dir = tmp_path / "enzyme_cache"
    variants_path.write_text(json.dumps([{"gene": "GENE1", "uniprot_id": "P00001"}]))
    genes_path.write_text("gene\nGENE1\n")
    output_path.write_text("existing artifact\n")

    monkeypatch.setattr(fetch_annotations, "MERGED_VARIANTS", variants_path)
    monkeypatch.setattr(fetch_annotations, "MERGED_GENE_LIST", genes_path)
    monkeypatch.setattr(fetch_annotations, "ENZYME_OUT", output_path)
    monkeypatch.setattr(fetch_annotations, "ENZYME_CACHE_DIR", cache_dir)
    monkeypatch.setattr(fetch_annotations.time, "sleep", lambda _: None)

    def fail_fetch(_accession):
        raise TransientFetchError("temporary failure")

    monkeypatch.setattr(fetch_annotations, "_fetch_uniprot_entry", fail_fetch)

    with pytest.raises(RuntimeError, match="was not written"):
        fetch_annotations.main_enzyme()

    assert output_path.read_text() == "existing artifact\n"
