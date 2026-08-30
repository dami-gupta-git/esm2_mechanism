"""
Tests for the gene-list builder's merge and label-mapping logic.

The sheet loaders take an openpyxl-like workbook, so these tests feed a stub
workbook rather than writing a real xlsx.

Covers:
- _load_functional_protein_class: reads gene -> mechanism, skips blank gene rows
- _load_functional_protein_class: a blank mechanism becomes the Unknown marker
- _load_clinvar_gene_level: reads the UniProt ID and mechanism columns
- _load_clinvar_gene_level: rows with too few columns are skipped
- _load_clinvar_gene_level: a gene with no UniProt ID gets an empty string
- _load_clinvar_gene_level: a later UniProt ID backfills an empty first one
- _load_clinvar_gene_level: conflicting UniProt IDs keep the first and are reported
- _load_clinvar_gene_level: a mechanism outside the map becomes Unknown
- _load_clinvar_gene_level: the most frequent mechanism wins
- _load_g2p: keeps only definitive/strong confidence rows
- _load_g2p: keeps only mechanisms present in the mechanism map
- _load_g2p: a gene with one mechanism across rows is kept
- _load_g2p: a conflict resolved by a single definitive mechanism is kept
- _load_g2p: an unresolvable mechanism conflict excludes the gene
"""

import pandas as pd
import pytest

from esm2_mech.fetch_data.build_gene_list import (
    _load_clinvar_gene_level,
    _load_functional_protein_class,
    _load_g2p,
)


class _Sheet:
    def __init__(self, rows):
        self._rows = rows

    def iter_rows(self, min_row=1, values_only=True):
        return iter(self._rows[min_row - 1:])


class _Workbook:
    """Minimal stand-in for the openpyxl workbook the loaders index by sheet name."""

    def __init__(self, sheets):
        self._sheets = {name: _Sheet(rows) for name, rows in sheets.items()}

    def __getitem__(self, name):
        return self._sheets[name]


def _functional_wb(data_rows):
    header = [("gene", "inheritance", "mechanism")]
    return _Workbook({"Functional_protein_class": header + data_rows})


def _clinvar_wb(data_rows):
    header = [tuple(f"col{i}" for i in range(10))]
    return _Workbook({"ClinVar_gene_level": header + data_rows})


def _clinvar_row(gene, uid, mech):
    """A ClinVar row: gene in column 0, UniProt ID in column 1, mechanism in column 9."""
    return (gene, uid, None, None, None, None, None, None, None, mech)


# ---------------------------------------------------------------------------
# _load_functional_protein_class
# ---------------------------------------------------------------------------


def test_functional_protein_class_reads_gene_mechanism_pairs():
    wb = _functional_wb([("BRCA1", "AD", "HI"), ("KRAS", "AD", "GOF")])
    assert _load_functional_protein_class(wb) == {"BRCA1": "HI", "KRAS": "GOF"}


def test_functional_protein_class_skips_blank_gene_rows():
    wb = _functional_wb([("BRCA1", "AD", "HI"), (None, "AD", "GOF"), ("", "AD", "DN")])
    assert _load_functional_protein_class(wb) == {"BRCA1": "HI"}


def test_functional_protein_class_blank_mechanism_becomes_unknown_marker():
    """A blank mechanism must stay distinguishable from a real label; main() drops it."""
    wb = _functional_wb([("BRCA1", "AD", None), ("KRAS", "AD", "")])
    assert _load_functional_protein_class(wb) == {"BRCA1": "Unknown", "KRAS": "Unknown"}


# ---------------------------------------------------------------------------
# _load_clinvar_gene_level
# ---------------------------------------------------------------------------


def test_clinvar_reads_uniprot_and_mechanism():
    wb = _clinvar_wb([_clinvar_row("BRCA1", "P38398", "HI")])
    uid_map, mech_map = _load_clinvar_gene_level(wb)
    assert uid_map == {"BRCA1": "P38398"}
    assert mech_map == {"BRCA1": "HI"}


def test_clinvar_skips_short_rows():
    """A row without the mechanism column must be skipped, not read at a shifted index."""
    wb = _clinvar_wb([("BRCA1", "P38398", "HI"), _clinvar_row("KRAS", "P01116", "GOF")])
    uid_map, mech_map = _load_clinvar_gene_level(wb)
    assert "BRCA1" not in uid_map
    assert mech_map == {"KRAS": "GOF"}


def test_clinvar_missing_uniprot_id_is_empty_string():
    wb = _clinvar_wb([_clinvar_row("BRCA1", None, "HI")])
    uid_map, _ = _load_clinvar_gene_level(wb)
    assert uid_map == {"BRCA1": ""}


def test_clinvar_later_uniprot_id_backfills_empty_first():
    wb = _clinvar_wb([
        _clinvar_row("BRCA1", None, "HI"),
        _clinvar_row("BRCA1", "P38398", "HI"),
    ])
    uid_map, _ = _load_clinvar_gene_level(wb)
    assert uid_map == {"BRCA1": "P38398"}


def test_clinvar_conflicting_uniprot_ids_keep_first_and_are_reported(capsys):
    wb = _clinvar_wb([
        _clinvar_row("BRCA1", "P38398", "HI"),
        _clinvar_row("BRCA1", "Q99999", "HI"),
    ])
    uid_map, _ = _load_clinvar_gene_level(wb)
    assert uid_map == {"BRCA1": "P38398"}
    output = capsys.readouterr().out
    assert "multiple UniProt IDs" in output
    assert "BRCA1" in output


def test_clinvar_unmapped_mechanism_becomes_unknown():
    """A mechanism string outside the map must not be emitted as a real label."""
    wb = _clinvar_wb([_clinvar_row("BRCA1", "P38398", "not-a-mechanism")])
    _, mech_map = _load_clinvar_gene_level(wb)
    assert mech_map == {"BRCA1": "Unknown"}


def test_clinvar_most_frequent_mechanism_wins():
    wb = _clinvar_wb([
        _clinvar_row("BRCA1", "P38398", "HI"),
        _clinvar_row("BRCA1", "P38398", "GOF"),
        _clinvar_row("BRCA1", "P38398", "HI"),
    ])
    _, mech_map = _load_clinvar_gene_level(wb)
    assert mech_map == {"BRCA1": "HI"}


# ---------------------------------------------------------------------------
# _load_g2p
# ---------------------------------------------------------------------------


def _write_g2p(tmp_path, rows):
    path = tmp_path / "g2p.csv"
    pd.DataFrame(
        rows, columns=["gene symbol", "confidence", "molecular mechanism"]
    ).to_csv(path, index=False)
    return path


def test_g2p_keeps_only_definitive_and_strong(tmp_path):
    path = _write_g2p(tmp_path, [
        ("BRCA1", "definitive", "loss of function"),
        ("KRAS", "strong", "gain of function"),
        ("TP53", "limited", "loss of function"),
        ("EGFR", "moderate", "gain of function"),
    ])
    assert _load_g2p(path) == {"BRCA1": "LOF", "KRAS": "GOF"}


def test_g2p_keeps_only_mapped_mechanisms(tmp_path):
    path = _write_g2p(tmp_path, [
        ("BRCA1", "definitive", "loss of function"),
        ("KRAS", "definitive", "undetermined"),
    ])
    assert _load_g2p(path) == {"BRCA1": "LOF"}


def test_g2p_single_mechanism_across_rows_is_kept(tmp_path):
    path = _write_g2p(tmp_path, [
        ("BRCA1", "definitive", "loss of function"),
        ("BRCA1", "strong", "loss of function"),
    ])
    assert _load_g2p(path) == {"BRCA1": "LOF"}


def test_g2p_conflict_resolved_by_single_definitive(tmp_path):
    path = _write_g2p(tmp_path, [
        ("BRCA1", "definitive", "loss of function"),
        ("BRCA1", "strong", "gain of function"),
    ])
    assert _load_g2p(path) == {"BRCA1": "LOF"}


def test_g2p_unresolvable_conflict_excludes_gene(tmp_path, capsys):
    """Two definitive mechanisms for one gene must drop it rather than pick one."""
    path = _write_g2p(tmp_path, [
        ("BRCA1", "definitive", "loss of function"),
        ("BRCA1", "definitive", "gain of function"),
    ])
    assert _load_g2p(path) == {}
    assert "conflicting mechanisms" in capsys.readouterr().out
