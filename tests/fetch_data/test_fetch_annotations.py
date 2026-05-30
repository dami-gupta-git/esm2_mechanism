"""
Tests for fetch_annotations.py pure-logic functions.

Covers:
- _stream_am_filter: header row is skipped, comment lines are skipped,
  only indexed (uniprot, variant) pairs are matched.
- _build_am_gene_uniprot_map: most-frequent UniProt ID wins on conflict,
  genes with no uniprot_id are skipped.
"""

import gzip
import json
import tempfile
from pathlib import Path

import pytest

from esm2_mechanism.fetch_data.fetch_annotations import (
    _stream_am_filter,
    _build_am_gene_uniprot_map,
)


# ---------------------------------------------------------------------------
# _stream_am_filter
# ---------------------------------------------------------------------------

def _write_am_gz(path: Path, lines: list[str]) -> None:
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")


def test_stream_am_filter_skips_header(tmp_path):
    am_gz = tmp_path / "am.tsv.gz"
    _write_am_gz(am_gz, [
        "uniprot_id\tprotein_variant\tam_pathogenicity\tam_class",
        "P12345\tA1V\t0.9\tpathogenic",
    ])
    index = {("P12345", "A1V"): "GENE_1_A_V"}
    scores = _stream_am_filter(am_gz, index)
    assert scores == {"GENE_1_A_V": pytest.approx(0.9)}


def test_stream_am_filter_skips_comment_lines(tmp_path):
    am_gz = tmp_path / "am.tsv.gz"
    _write_am_gz(am_gz, [
        "# this is a comment",
        "uniprot_id\tprotein_variant\tam_pathogenicity\tam_class",
        "P12345\tA1V\t0.9\tpathogenic",
    ])
    index = {("P12345", "A1V"): "GENE_1_A_V"}
    scores = _stream_am_filter(am_gz, index)
    assert scores == {"GENE_1_A_V": pytest.approx(0.9)}


def test_stream_am_filter_only_matches_indexed_pairs(tmp_path):
    am_gz = tmp_path / "am.tsv.gz"
    _write_am_gz(am_gz, [
        "uniprot_id\tprotein_variant\tam_pathogenicity\tam_class",
        "P12345\tA1V\t0.9\tpathogenic",
        "P99999\tR5K\t0.2\tbenign",
    ])
    index = {("P12345", "A1V"): "GENE_1_A_V"}
    scores = _stream_am_filter(am_gz, index)
    assert "GENE_1_A_V" in scores
    assert len(scores) == 1


def test_stream_am_filter_empty_index(tmp_path):
    am_gz = tmp_path / "am.tsv.gz"
    _write_am_gz(am_gz, [
        "uniprot_id\tprotein_variant\tam_pathogenicity\tam_class",
        "P12345\tA1V\t0.9\tpathogenic",
    ])
    scores = _stream_am_filter(am_gz, {})
    assert scores == {}


def test_stream_am_filter_malformed_score_skipped(tmp_path):
    am_gz = tmp_path / "am.tsv.gz"
    _write_am_gz(am_gz, [
        "uniprot_id\tprotein_variant\tam_pathogenicity\tam_class",
        "P12345\tA1V\tNOT_A_FLOAT\tpathogenic",
    ])
    index = {("P12345", "A1V"): "GENE_1_A_V"}
    scores = _stream_am_filter(am_gz, index)
    assert scores == {}


# ---------------------------------------------------------------------------
# _build_am_gene_uniprot_map
# ---------------------------------------------------------------------------

def _write_variants(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows))


def test_build_am_gene_uniprot_map_single_id(tmp_path, monkeypatch):
    import esm2_mechanism.fetch_data.fetch_annotations as fa
    vf = tmp_path / "merged_valid_variants.json"
    _write_variants(vf, [
        {"gene": "BRCA1", "uniprot_id": "P38398"},
        {"gene": "BRCA1", "uniprot_id": "P38398"},
    ])
    monkeypatch.setattr(fa, "AM_MERGED_VALID_VARIANTS", vf)
    result = fa._build_am_gene_uniprot_map()
    assert result == {"BRCA1": "P38398"}


def test_build_am_gene_uniprot_map_most_frequent_wins(tmp_path, monkeypatch):
    import esm2_mechanism.fetch_data.fetch_annotations as fa
    vf = tmp_path / "merged_valid_variants.json"
    _write_variants(vf, [
        {"gene": "TP53", "uniprot_id": "P04637"},
        {"gene": "TP53", "uniprot_id": "P04637"},
        {"gene": "TP53", "uniprot_id": "WRONG99"},
    ])
    monkeypatch.setattr(fa, "AM_MERGED_VALID_VARIANTS", vf)
    result = fa._build_am_gene_uniprot_map()
    assert result["TP53"] == "P04637"


def test_build_am_gene_uniprot_map_skips_empty_uniprot(tmp_path, monkeypatch):
    import esm2_mechanism.fetch_data.fetch_annotations as fa
    vf = tmp_path / "merged_valid_variants.json"
    _write_variants(vf, [
        {"gene": "MYC", "uniprot_id": ""},
        {"gene": "MYC", "uniprot_id": None},
    ])
    monkeypatch.setattr(fa, "AM_MERGED_VALID_VARIANTS", vf)
    result = fa._build_am_gene_uniprot_map()
    assert "MYC" not in result


def test_build_am_gene_uniprot_map_multiple_genes(tmp_path, monkeypatch):
    import esm2_mechanism.fetch_data.fetch_annotations as fa
    vf = tmp_path / "merged_valid_variants.json"
    _write_variants(vf, [
        {"gene": "BRCA1", "uniprot_id": "P38398"},
        {"gene": "TP53", "uniprot_id": "P04637"},
    ])
    monkeypatch.setattr(fa, "AM_MERGED_VALID_VARIANTS", vf)
    result = fa._build_am_gene_uniprot_map()
    assert result == {"BRCA1": "P38398", "TP53": "P04637"}
