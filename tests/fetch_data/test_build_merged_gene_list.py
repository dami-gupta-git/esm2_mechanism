"""
Tests for build_merged_gene_list.py.

Covers:
- _load_functional_protein_class: normal, blank mechanism cell, multiple genes
- _load_clinvar_gene_level: uid extraction, mechanism majority vote, unknown fallback
- _load_g2p: confidence filtering, mechanism mapping, majority vote per gene
- build(): gerasimavicius-only gene, g2p-only gene, unknown-gerasimavicius overridden by g2p,
           g2p_disagrees populated, unknown-in-both excluded, legacy alias injection,
           output sorted by gene, missing input file raises FileNotFoundError
"""

import csv
from pathlib import Path

import openpyxl
import pandas as pd
import pytest

from esm2_mechanism.fetch_data.build_merged_gene_list import (
    LEGACY_GENE_ALIASES,
    _load_functional_protein_class,
    _load_clinvar_gene_level,
    _load_g2p,
    build,
)

_LEGACY_GENES = {a["gene"] for a in LEGACY_GENE_ALIASES}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_workbook(func_rows: list[tuple], cv_rows: list[tuple]) -> openpyxl.Workbook:
    """
    Build an in-memory workbook with Functional_protein_class and
    ClinVar_gene_level sheets.

    func_rows: list of (gene, inheritance, mechanism)
    cv_rows:   list of (gene, uniprot_id, ?, ?, ?, ?, ?, ?, ?, mechanism)
               columns 2-8 (0-indexed) are unused; pass None for them.
    """
    wb = openpyxl.Workbook()

    # Functional_protein_class sheet
    ws_func = wb.active
    ws_func.title = "Functional_protein_class"
    ws_func.append(["gene", "inheritance", "mechanism"])  # header row
    for row in func_rows:
        ws_func.append(list(row))

    # ClinVar_gene_level sheet
    ws_cv = wb.create_sheet("ClinVar_gene_level")
    ws_cv.append(["gene", "uniprot_id", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "mechanism"])
    for row in cv_rows:
        # pad to 10 columns
        padded = [row[0], row[1]] + [None] * 7 + [row[2]]
        ws_cv.append(padded)

    return wb


def _make_g2p_csv(rows: list[dict], tmp_path: Path) -> Path:
    """Write a minimal AllG2P.csv to tmp_path and return its path."""
    path = tmp_path / "AllG2P.csv"
    df = pd.DataFrame(rows, columns=["gene symbol", "confidence", "molecular mechanism"])
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# _load_functional_protein_class
# ---------------------------------------------------------------------------

class TestLoadFunctionalProteinClass:

    def test_basic_extraction(self):
        wb = _make_workbook(
            func_rows=[("BRCA1", "AD", "HI"), ("KRAS", "AD", "GOF")],
            cv_rows=[],
        )
        result = _load_functional_protein_class(wb)
        assert result == {"BRCA1": "HI", "KRAS": "GOF"}

    def test_blank_mechanism_normalised_to_unknown(self):
        wb = _make_workbook(
            func_rows=[("GENE1", "AD", None)],
            cv_rows=[],
        )
        result = _load_functional_protein_class(wb)
        assert result["GENE1"] == "Unknown"

    def test_empty_sheet_returns_empty_dict(self):
        wb = _make_workbook(func_rows=[], cv_rows=[])
        result = _load_functional_protein_class(wb)
        assert result == {}

    def test_gene_without_mechanism_row_skipped_when_gene_is_none(self):
        wb = _make_workbook(
            func_rows=[(None, "AD", "HI"), ("BRCA2", "AD", "LOF")],
            cv_rows=[],
        )
        result = _load_functional_protein_class(wb)
        assert None not in result
        assert "BRCA2" in result


# ---------------------------------------------------------------------------
# _load_clinvar_gene_level
# ---------------------------------------------------------------------------

class TestLoadClinvarGeneLevel:

    def test_uid_extracted(self):
        wb = _make_workbook(
            func_rows=[],
            cv_rows=[("BRCA1", "P38398", "HI")],
        )
        uid_map, _ = _load_clinvar_gene_level(wb)
        assert uid_map["BRCA1"] == "P38398"

    def test_missing_uid_defaults_to_empty_string(self):
        wb = _make_workbook(func_rows=[], cv_rows=[("BRCA1", None, "HI")])
        uid_map, _ = _load_clinvar_gene_level(wb)
        assert uid_map["BRCA1"] == ""

    def test_mechanism_majority_vote(self):
        wb = _make_workbook(
            func_rows=[],
            cv_rows=[
                ("GENE1", "P00001", "HI"),
                ("GENE1", "P00001", "HI"),
                ("GENE1", "P00001", "GOF"),
            ],
        )
        _, mech_map = _load_clinvar_gene_level(wb)
        assert mech_map["GENE1"] == "HI"

    def test_unknown_mechanism_resolved(self):
        wb = _make_workbook(func_rows=[], cv_rows=[("GENE1", "P00001", "Unknown")])
        _, mech_map = _load_clinvar_gene_level(wb)
        assert mech_map["GENE1"] == "Unknown"

    def test_unmapped_mechanism_falls_back_to_unknown(self):
        wb = _make_workbook(func_rows=[], cv_rows=[("GENE1", "P00001", "BIZARRE_VALUE")])
        _, mech_map = _load_clinvar_gene_level(wb)
        assert mech_map["GENE1"] == "Unknown"

    def test_none_mechanism_treated_as_unknown(self):
        wb = _make_workbook(func_rows=[], cv_rows=[("GENE1", "P00001", None)])
        _, mech_map = _load_clinvar_gene_level(wb)
        assert mech_map["GENE1"] == "Unknown"


# ---------------------------------------------------------------------------
# _load_g2p
# ---------------------------------------------------------------------------

class TestLoadG2p:

    def _g2p_row(self, gene, confidence, mechanism):
        return {"gene symbol": gene, "confidence": confidence, "molecular mechanism": mechanism}

    def test_lof_mapped_correctly(self, tmp_path):
        path = _make_g2p_csv([self._g2p_row("GENE1", "definitive", "loss of function")], tmp_path)
        result = _load_g2p(path)
        assert result["GENE1"] == "LOF"

    def test_gof_mapped_correctly(self, tmp_path):
        path = _make_g2p_csv([self._g2p_row("GENE1", "strong", "gain of function")], tmp_path)
        result = _load_g2p(path)
        assert result["GENE1"] == "GOF"

    def test_dn_mapped_correctly(self, tmp_path):
        path = _make_g2p_csv([self._g2p_row("GENE1", "definitive", "dominant negative")], tmp_path)
        result = _load_g2p(path)
        assert result["GENE1"] == "DN"

    def test_low_confidence_excluded(self, tmp_path):
        path = _make_g2p_csv([self._g2p_row("GENE1", "moderate", "loss of function")], tmp_path)
        result = _load_g2p(path)
        assert "GENE1" not in result

    def test_undetermined_mechanism_excluded(self, tmp_path):
        path = _make_g2p_csv([self._g2p_row("GENE1", "definitive", "undetermined")], tmp_path)
        result = _load_g2p(path)
        assert "GENE1" not in result

    def test_majority_vote_across_entries(self, tmp_path):
        rows = [
            self._g2p_row("GENE1", "definitive", "loss of function"),
            self._g2p_row("GENE1", "strong", "loss of function"),
            self._g2p_row("GENE1", "definitive", "gain of function"),
        ]
        path = _make_g2p_csv(rows, tmp_path)
        result = _load_g2p(path)
        assert result["GENE1"] == "LOF"

    def test_empty_csv_returns_empty_dict(self, tmp_path):
        path = _make_g2p_csv([], tmp_path)
        # empty df has no columns — _load_g2p will raise KeyError; patch columns
        df = pd.DataFrame(columns=["gene symbol", "confidence", "molecular mechanism"])
        df.to_csv(path, index=False)
        result = _load_g2p(path)
        assert result == {}


# ---------------------------------------------------------------------------
# build() — integration tests using tmp_path fixtures
# ---------------------------------------------------------------------------

def _read_tsv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


class TestBuild:

    def _run(self, tmp_path, func_rows, cv_rows, g2p_rows):
        """Helper: build workbook + g2p csv, run build(), return rows excluding legacy aliases."""
        wb = _make_workbook(func_rows, cv_rows)
        xlsx_path = tmp_path / "test.xlsx"
        wb.save(xlsx_path)

        g2p_path = _make_g2p_csv(g2p_rows, tmp_path)
        out_path = tmp_path / "out.tsv"
        build(xlsx_path, g2p_path, out_path)
        return [r for r in _read_tsv(out_path) if r["gene"] not in _LEGACY_GENES]

    def test_gerasimavicius_gene_emitted_with_source(self, tmp_path):
        rows = self._run(
            tmp_path,
            func_rows=[("BRCA1", "AD", "HI")],
            cv_rows=[("BRCA1", "P38398", "HI")],
            g2p_rows=[],
        )
        assert len(rows) == 1
        assert rows[0]["gene"] == "BRCA1"
        assert rows[0]["mechanism"] == "HI"
        assert rows[0]["source"] == "gerasimavicius"
        assert rows[0]["uniprot_id"] == "P38398"

    def test_g2p_only_gene_emitted(self, tmp_path):
        rows = self._run(
            tmp_path,
            func_rows=[],
            cv_rows=[],
            g2p_rows=[{"gene symbol": "KRAS", "confidence": "definitive", "molecular mechanism": "gain of function"}],
        )
        assert len(rows) == 1
        assert rows[0]["gene"] == "KRAS"
        assert rows[0]["mechanism"] == "GOF"
        assert rows[0]["source"] == "g2p"

    def test_unknown_gerasimavicius_overridden_by_g2p(self, tmp_path):
        rows = self._run(
            tmp_path,
            func_rows=[("GENE1", "AD", "Unknown")],
            cv_rows=[("GENE1", "P00001", "Unknown")],
            g2p_rows=[{"gene symbol": "GENE1", "confidence": "definitive", "molecular mechanism": "loss of function"}],
        )
        assert len(rows) == 1
        assert rows[0]["mechanism"] == "LOF"
        assert rows[0]["source"] == "g2p"

    def test_unknown_in_both_sources_excluded(self, tmp_path):
        rows = self._run(
            tmp_path,
            func_rows=[("GENE1", "AD", "Unknown")],
            cv_rows=[("GENE1", "P00001", "Unknown")],
            g2p_rows=[],
        )
        assert len(rows) == 0

    def test_g2p_disagrees_populated(self, tmp_path):
        rows = self._run(
            tmp_path,
            func_rows=[("BRCA1", "AD", "HI")],
            cv_rows=[("BRCA1", "P38398", "HI")],
            g2p_rows=[{"gene symbol": "BRCA1", "confidence": "definitive", "molecular mechanism": "loss of function"}],
        )
        assert rows[0]["source"] == "gerasimavicius"
        assert rows[0]["g2p_disagrees"] == "LOF"

    def test_g2p_agrees_disagrees_is_empty(self, tmp_path):
        rows = self._run(
            tmp_path,
            func_rows=[("GENE1", "AD", "GOF")],
            cv_rows=[("GENE1", "P00001", "GOF")],
            g2p_rows=[{"gene symbol": "GENE1", "confidence": "definitive", "molecular mechanism": "gain of function"}],
        )
        assert rows[0]["g2p_disagrees"] == ""

    def test_output_sorted_by_gene(self, tmp_path):
        rows = self._run(
            tmp_path,
            func_rows=[("ZEBRA", "AD", "HI"), ("ALPHA", "AD", "GOF")],
            cv_rows=[("ZEBRA", "P00002", "HI"), ("ALPHA", "P00001", "GOF")],
            g2p_rows=[],
        )
        genes = [r["gene"] for r in rows]
        assert genes == sorted(genes)

    def test_functional_protein_class_takes_priority_over_clinvar(self, tmp_path):
        # func says GOF, clinvar says HI — func should win
        rows = self._run(
            tmp_path,
            func_rows=[("GENE1", "AD", "GOF")],
            cv_rows=[("GENE1", "P00001", "HI")],
            g2p_rows=[],
        )
        assert rows[0]["mechanism"] == "GOF"

    def test_clinvar_fallback_when_func_is_unknown(self, tmp_path):
        rows = self._run(
            tmp_path,
            func_rows=[("GENE1", "AD", "Unknown")],
            cv_rows=[("GENE1", "P00001", "HI")],
            g2p_rows=[],
        )
        assert rows[0]["mechanism"] == "HI"
        assert rows[0]["source"] == "gerasimavicius"

    def test_gene_in_both_sources_not_duplicated(self, tmp_path):
        rows = self._run(
            tmp_path,
            func_rows=[("GENE1", "AD", "HI")],
            cv_rows=[("GENE1", "P00001", "HI")],
            g2p_rows=[{"gene symbol": "GENE1", "confidence": "definitive", "molecular mechanism": "loss of function"}],
        )
        assert len(rows) == 1

    def test_output_file_written(self, tmp_path):
        wb = _make_workbook(
            func_rows=[("BRCA1", "AD", "HI")],
            cv_rows=[("BRCA1", "P38398", "HI")],
        )
        xlsx_path = tmp_path / "test.xlsx"
        wb.save(xlsx_path)
        g2p_path = _make_g2p_csv([], tmp_path)
        out_path = tmp_path / "subdir" / "out.tsv"
        build(xlsx_path, g2p_path, out_path)
        assert out_path.exists()

    def test_missing_xlsx_raises(self, tmp_path):
        g2p_path = _make_g2p_csv([], tmp_path)
        with pytest.raises(Exception):
            build(tmp_path / "nonexistent.xlsx", g2p_path, tmp_path / "out.tsv")

    def test_clinvar_gene_not_in_func_still_emitted(self, tmp_path):
        # Gene only appears in ClinVar_gene_level, not Functional_protein_class
        rows = self._run(
            tmp_path,
            func_rows=[],
            cv_rows=[("GENE1", "P00001", "HI")],
            g2p_rows=[],
        )
        assert len(rows) == 1
        assert rows[0]["gene"] == "GENE1"
        assert rows[0]["mechanism"] == "HI"
