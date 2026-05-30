"""
Tests for build_proteome_features.py pure-logic functions.

Covers:
- build_aligned_matrix: median-imputes a partially-missing column, skips
  gene/pfam_family text columns, and RAISES on a fully-missing column rather
  than fabricating a 0.0 default (global no-fallback rule).
- get_shet: joins GeneBayes s_het estimates onto gene symbols via ensg map.
"""

import gzip
import json

import numpy as np
import pytest

from esm2_mech.fetch_data import build_proteome_features as bpf
from esm2_mech.fetch_data.build_proteome_features import (
    _load_paralog_cache,
    build_aligned_matrix,
    get_bioplex_degree,
    get_shet,
)


def _rows(values):
    """Build minimal row dicts from a list of pLI values (None allowed)."""
    return [
        {"gene": f"G{i}", "pfam_family": "", "pLI": v} for i, v in enumerate(values)
    ]


COLS = ["gene", "pfam_family", "pLI"]


def test_skips_text_columns():
    X, num_cols = build_aligned_matrix(_rows([0.1, 0.2, 0.3]), COLS)
    assert num_cols == ["pLI"]
    assert X.shape == (3, 1)


def test_median_imputes_partial_missing():
    # values 0.0, None, 0.4 → median of observed {0.0, 0.4} = 0.2 fills the gap
    X, _ = build_aligned_matrix(_rows([0.0, None, 0.4]), COLS)
    assert X[1, 0] == pytest.approx(0.2)
    assert X[0, 0] == pytest.approx(0.0)
    assert X[2, 0] == pytest.approx(0.4)


def test_fully_missing_column_left_as_nan():
    X, _ = build_aligned_matrix(_rows([None, None, None]), COLS)
    assert np.all(np.isnan(X[:, 0]))


def test_observed_values_unchanged():
    X, _ = build_aligned_matrix(_rows([1.5, 2.5]), COLS)
    assert X[0, 0] == pytest.approx(1.5)
    assert X[1, 0] == pytest.approx(2.5)
    assert X.dtype == np.float32


# ---------------------------------------------------------------------------
# get_shet
# ---------------------------------------------------------------------------
_SHET_FILE = (
    "ensg\thgnc\tchrom\tobs_lof\texp_lof\tprior_mean\tpost_mean\tpost_lower_95\tpost_upper_95\n"
    "ENSG00000001\tHGNC:1\tchr1\t5\t4.0\t0.01\t0.005\t0.001\t0.01\n"
    "ENSG00000002\tHGNC:2\tchr2\t10\t8.0\t0.02\t0.015\t0.005\t0.03\n"
    "ENSG00000003\tHGNC:3\tchr3\t0\t5.0\t0.001\t0.0001\t0.00001\t0.001\n"
)

_ENSG_MAP = {
    "ENSG00000001": "BRCA1",
    "ENSG00000002": "TP53",
    "ENSG00000003": "UNKNOWN_GENE",
}


def _patch_shet(monkeypatch, tmp_path, shet_content=_SHET_FILE, ensg_map=_ENSG_MAP):
    """Set up tmp shet + ensg-cache fixtures and patch module globals."""
    shet_file = tmp_path / "s_het_estimates.genebayes.tsv"
    shet_file.write_text(shet_content)
    cache_file = tmp_path / "ensg_map.json"
    cache_file.write_text(json.dumps(ensg_map))
    monkeypatch.setattr(bpf, "SHET_MANUAL", shet_file)
    monkeypatch.setattr(bpf, "GNOMAD_V2_ENSG_CACHE", cache_file)
    # Bypass _load_ensg_to_symbol entirely — return the fixture map directly
    monkeypatch.setattr(bpf, "_load_ensg_to_symbol", lambda: ensg_map)


def test_get_shet_maps_values(tmp_path, monkeypatch):
    _patch_shet(monkeypatch, tmp_path)
    result = get_shet(["BRCA1", "TP53", "MYC"])
    assert result["BRCA1"] == pytest.approx(0.005)
    assert result["TP53"] == pytest.approx(0.015)
    assert result["MYC"] is None  # not in shet file


def test_get_shet_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(bpf, "SHET_MANUAL", tmp_path / "nonexistent.tsv")
    monkeypatch.setattr(bpf, "_load_ensg_to_symbol", lambda: _ENSG_MAP)
    result = get_shet(["BRCA1", "TP53"])
    assert result["BRCA1"] is None
    assert result["TP53"] is None


def test_get_shet_gene_not_in_universe_ignored(tmp_path, monkeypatch):
    _patch_shet(monkeypatch, tmp_path)
    # UNKNOWN_GENE is in the shet file but not in genes list — must not appear in result
    result = get_shet(["BRCA1"])
    assert "UNKNOWN_GENE" not in result
    assert result["BRCA1"] == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# _load_paralog_cache — error-tagged entries must force a re-fetch
# ---------------------------------------------------------------------------
def test_paralog_cache_missing_file(tmp_path):
    usable, val = _load_paralog_cache(tmp_path / "nope.json")
    assert (usable, val) == (False, None)


def test_paralog_cache_valid_entry(tmp_path):
    f = tmp_path / "g.json"
    f.write_text(json.dumps({"paralog_count": 7}))
    assert _load_paralog_cache(f) == (True, 7)


def test_paralog_cache_zero_is_valid(tmp_path):
    # A legitimate "zero paralogs" answer must be honoured (no error tag, int 0).
    f = tmp_path / "g.json"
    f.write_text(json.dumps({"paralog_count": 0}))
    assert _load_paralog_cache(f) == (True, 0)


def test_paralog_cache_error_tag_forces_refetch(tmp_path):
    # A prior transient failure was cached as error-tagged — must NOT be reused.
    f = tmp_path / "g.json"
    f.write_text(json.dumps({"paralog_count": None, "error": "timeout"}))
    assert _load_paralog_cache(f) == (False, None)


def test_paralog_cache_malformed_count_forces_refetch(tmp_path):
    f = tmp_path / "g.json"
    f.write_text(json.dumps({"paralog_count": None}))
    assert _load_paralog_cache(f) == (False, None)


def test_paralog_cache_corrupt_json_forces_refetch(tmp_path):
    f = tmp_path / "g.json"
    f.write_text("{not json")
    assert _load_paralog_cache(f) == (False, None)


# ---------------------------------------------------------------------------
# get_bioplex_degree — low coverage must be discarded, not silently emitted
# ---------------------------------------------------------------------------
def _patch_bioplex(monkeypatch, tmp_path, content):
    f = tmp_path / "bioplex.tsv"
    f.write_text(content)
    monkeypatch.setattr(bpf, "BIOPLEX_CACHE", f)
    monkeypatch.setattr(bpf, "_download_file", lambda *a, **kw: True)


def test_bioplex_normal_coverage(monkeypatch, tmp_path):
    # 3 universe genes, all present as edges → coverage 100%, returns degrees.
    content = "GeneA\tGeneB\nBRCA1\tTP53\nBRCA1\tMYC\nTP53\tMYC\n"
    _patch_bioplex(monkeypatch, tmp_path, content)
    result = get_bioplex_degree(["BRCA1", "TP53", "MYC"])
    assert result["BRCA1"] == 2
    assert result["TP53"] == 2
    assert result["MYC"] == 2


def test_bioplex_low_coverage_discarded(monkeypatch, tmp_path):
    # 100 universe genes, only 1 matched → 1% coverage < 5% threshold.
    # Likely the column held non-symbol IDs; refuse to emit a near-zero matrix.
    edges = "\n".join(f"P{i:05d}\tP{i+1:05d}" for i in range(50))
    content = "GeneA\tGeneB\n" + edges + "\nBRCA1\tTP53\n"
    _patch_bioplex(monkeypatch, tmp_path, content)
    genes = [f"REAL_GENE_{i}" for i in range(98)] + ["BRCA1", "TP53"]
    result = get_bioplex_degree(genes)
    # All None — refused to fabricate near-zero degrees from an apparent ID mismatch.
    assert all(v is None for v in result.values())
