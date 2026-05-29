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

from esm2_mechanism.fetch_data import build_proteome_features as bpf
from esm2_mechanism.fetch_data.build_proteome_features import (
    build_aligned_matrix,
    get_shet,
)


def _rows(values):
    """Build minimal row dicts from a list of pLI values (None allowed)."""
    return [{"gene": f"G{i}", "pfam_family": "", "pLI": v} for i, v in enumerate(values)]


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


def test_fully_missing_column_raises():
    with pytest.raises(ValueError, match="no observed values"):
        build_aligned_matrix(_rows([None, None, None]), COLS)


def test_observed_values_unchanged():
    X, _ = build_aligned_matrix(_rows([1.5, 2.5]), COLS)
    assert X[0, 0] == pytest.approx(1.5)
    assert X[1, 0] == pytest.approx(2.5)
    assert X.dtype == np.float32


# ---------------------------------------------------------------------------
# get_shet
# ---------------------------------------------------------------------------
_SHET_FILE = "ensg\thgnc\tchrom\tobs_lof\texp_lof\tprior_mean\tpost_mean\tpost_lower_95\tpost_upper_95\n" \
             "ENSG00000001\tHGNC:1\tchr1\t5\t4.0\t0.01\t0.005\t0.001\t0.01\n" \
             "ENSG00000002\tHGNC:2\tchr2\t10\t8.0\t0.02\t0.015\t0.005\t0.03\n" \
             "ENSG00000003\tHGNC:3\tchr3\t0\t5.0\t0.001\t0.0001\t0.00001\t0.001\n"

_ENSG_MAP = {"ENSG00000001": "BRCA1", "ENSG00000002": "TP53", "ENSG00000003": "UNKNOWN_GENE"}


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
