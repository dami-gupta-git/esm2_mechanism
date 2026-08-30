"""
Tests for fetch_annotations.py pure-logic functions.

Covers (canonical helpers in alphamissense_common):
- stream_am_filter: header row is skipped, comment lines are skipped,
  only indexed (uniprot, variant) pairs are matched.
- build_gene_uniprot_map: most-frequent UniProt ID wins on conflict,
  genes with no uniprot_id are skipped.
"""

import gzip

from pathlib import Path

import pytest

from esm2_mech.fetch_data.alphamissense_common import (
    build_gene_uniprot_map,
    stream_am_filter,
)

# ---------------------------------------------------------------------------
# _stream_am_filter
# ---------------------------------------------------------------------------


AM_HEADER = "uniprot_id\tprotein_variant\tam_pathogenicity\tam_class"
SCORED_ROW = "P12345\tA1V\t0.9\tpathogenic"
INDEX = {("P12345", "A1V"): "GENE_1_A_V"}


def _write_am_gz(path: Path, lines: list[str]) -> None:
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")


def _am_file(tmp_path, *rows, preamble=()):
    """An AlphaMissense TSV.gz: any preamble lines, then the header, then rows."""
    path = tmp_path / "am.tsv.gz"
    _write_am_gz(path, [*preamble, AM_HEADER, *rows])
    return path


def test_stream_am_filter_skips_header(tmp_path):
    scores = stream_am_filter(_am_file(tmp_path, SCORED_ROW), INDEX)
    assert scores == {"GENE_1_A_V": pytest.approx(0.9)}


def test_stream_am_filter_skips_comment_lines(tmp_path):
    am_gz = _am_file(tmp_path, SCORED_ROW, preamble=["# this is a comment"])
    assert stream_am_filter(am_gz, INDEX) == {"GENE_1_A_V": pytest.approx(0.9)}


def test_stream_am_filter_only_matches_indexed_pairs(tmp_path):
    am_gz = _am_file(tmp_path, SCORED_ROW, "P99999\tR5K\t0.2\tbenign")
    scores = stream_am_filter(am_gz, INDEX)
    assert "GENE_1_A_V" in scores
    assert len(scores) == 1


def test_stream_am_filter_empty_index(tmp_path):
    assert stream_am_filter(_am_file(tmp_path, SCORED_ROW), {}) == {}


def test_stream_am_filter_malformed_score_skipped(tmp_path):
    am_gz = _am_file(tmp_path, "P12345\tA1V\tNOT_A_FLOAT\tpathogenic")
    assert stream_am_filter(am_gz, INDEX) == {}


# ---------------------------------------------------------------------------
# build_gene_uniprot_map
# ---------------------------------------------------------------------------


def test_build_gene_uniprot_map_single_id():
    result = build_gene_uniprot_map(
        [
            {"gene": "BRCA1", "uniprot_id": "P38398"},
            {"gene": "BRCA1", "uniprot_id": "P38398"},
        ]
    )
    assert result == {"BRCA1": "P38398"}


def test_build_gene_uniprot_map_most_frequent_wins():
    result = build_gene_uniprot_map(
        [
            {"gene": "TP53", "uniprot_id": "P04637"},
            {"gene": "TP53", "uniprot_id": "P04637"},
            {"gene": "TP53", "uniprot_id": "WRONG99"},
        ]
    )
    assert result["TP53"] == "P04637"


def test_build_gene_uniprot_map_skips_empty_uniprot():
    result = build_gene_uniprot_map(
        [
            {"gene": "MYC", "uniprot_id": ""},
            {"gene": "MYC", "uniprot_id": None},
        ]
    )
    assert "MYC" not in result


def test_build_gene_uniprot_map_multiple_genes():
    result = build_gene_uniprot_map(
        [
            {"gene": "BRCA1", "uniprot_id": "P38398"},
            {"gene": "TP53", "uniprot_id": "P04637"},
        ]
    )
    assert result == {"BRCA1": "P38398", "TP53": "P04637"}
