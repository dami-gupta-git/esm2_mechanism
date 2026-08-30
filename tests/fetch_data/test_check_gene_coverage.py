"""
Tests for the gene-coverage checker.

This tool guards the positional alignment between the gene universe and the
feature matrices built from it, which is the misalignment the project has been
bitten by before. Each check must return False on a real disagreement rather
than passing quietly.

Covers:
- load_tsv_genes: reads the named column and strips whitespace
- load_tsv_genes: blank cells are skipped, not stored as empty strings
- load_tsv_genes: a missing column raises, naming the header it found
- load_tsv_genes: a non-default column can be read
- load_json_keys: returns stripped top-level keys
- load_json_keys: a JSON array raises rather than being read as genes
- count_tsv_rows: counts data rows, excluding the header
- count_tsv_rows: a header-only file counts zero
- count_tsv_rows: blank trailing lines are not counted
- check_subset: passes when every gene is present, fails when any is missing
- check_subset: an empty subset passes
- check_equal: passes only when both sets match exactly
- check_equal: a difference in either direction fails and is reported
- check_row_count: passes when the matrix rows match the expected count
- check_row_count: a row-count difference fails
- main: a run with no files reports nothing checked
- main: a partial run does not claim every expectation holds
- main: a partial run names the files that stopped a check running
- main: a complete, consistent run passes
"""

import json

import numpy as np
import pytest

from esm2_mech.fetch_data.check_gene_coverage import (
    check_equal,
    check_row_count,
    check_subset,
    count_tsv_rows,
    load_json_keys,
    load_tsv_genes,
)


def _write_tsv(path, header, rows):
    lines = ["\t".join(header)] + ["\t".join(row) for row in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def test_load_tsv_genes_reads_the_column_and_strips_whitespace(tmp_path):
    path = _write_tsv(
        tmp_path / "genes.tsv",
        ["gene", "mechanism"],
        [[" BRCA1 ", "HI"], ["TP53", "DN"]],
    )
    assert load_tsv_genes(path) == {"BRCA1", "TP53"}


def test_load_tsv_genes_skips_blank_cells(tmp_path):
    """A blank cell is an absent gene, not a gene named the empty string."""
    path = _write_tsv(
        tmp_path / "genes.tsv",
        ["gene", "mechanism"],
        [["BRCA1", "HI"], ["", "DN"], ["   ", "GOF"]],
    )
    assert load_tsv_genes(path) == {"BRCA1"}


def test_load_tsv_genes_missing_column_raises_naming_the_header(tmp_path):
    path = _write_tsv(tmp_path / "genes.tsv", ["symbol"], [["BRCA1"]])
    with pytest.raises(ValueError, match="no 'gene' column"):
        load_tsv_genes(path)


def test_load_tsv_genes_reads_a_non_default_column(tmp_path):
    path = _write_tsv(tmp_path / "genes.tsv", ["symbol"], [["BRCA1"]])
    assert load_tsv_genes(path, column="symbol") == {"BRCA1"}


def test_load_json_keys_returns_stripped_keys(tmp_path):
    path = tmp_path / "map.json"
    path.write_text(json.dumps({" BRCA1 ": "PF1", "TP53": "PF2"}))
    assert load_json_keys(path) == {"BRCA1", "TP53"}


def test_load_json_keys_rejects_a_json_array(tmp_path):
    """A list would silently yield positional indices instead of gene names."""
    path = tmp_path / "map.json"
    path.write_text(json.dumps(["BRCA1", "TP53"]))
    with pytest.raises(ValueError, match="expected a JSON object"):
        load_json_keys(path)


def test_count_tsv_rows_excludes_the_header(tmp_path):
    path = _write_tsv(tmp_path / "g.tsv", ["gene"], [["A"], ["B"], ["C"]])
    assert count_tsv_rows(path) == 3


def test_count_tsv_rows_header_only_is_zero(tmp_path):
    path = _write_tsv(tmp_path / "g.tsv", ["gene"], [])
    assert count_tsv_rows(path) == 0


def test_count_tsv_rows_ignores_blank_trailing_lines(tmp_path):
    path = tmp_path / "g.tsv"
    path.write_text("gene\nA\nB\n\n\n")
    assert count_tsv_rows(path) == 2


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def test_check_subset_passes_when_every_gene_is_present():
    assert check_subset("t", {"A", "B"}, {"A", "B", "C"}, "sub", "sup") is True


def test_check_subset_fails_when_a_gene_is_missing(capsys):
    assert check_subset("t", {"A", "Z"}, {"A", "B"}, "sub", "sup") is False
    assert "Z" in capsys.readouterr().out


def test_check_subset_empty_subset_passes():
    assert check_subset("t", set(), {"A"}, "sub", "sup") is True


def test_check_equal_passes_only_on_an_exact_match():
    assert check_equal("t", {"A", "B"}, {"B", "A"}, "left", "right") is True


@pytest.mark.parametrize(
    ("left", "right", "expected_in_output"),
    [
        ({"A", "B"}, {"A"}, "B"),          # extra on the left
        ({"A"}, {"A", "B"}, "B"),          # extra on the right
        ({"A"}, {"B"}, "A"),               # disjoint
    ],
    ids=["extra_on_left", "extra_on_right", "disjoint"],
)
def test_check_equal_fails_on_any_difference(left, right, expected_in_output, capsys):
    assert check_equal("t", left, right, "left", "right") is False
    assert expected_in_output in capsys.readouterr().out


def test_check_row_count_passes_when_rows_match(tmp_path):
    path = tmp_path / "m.npy"
    np.save(path, np.zeros((5, 3)))
    assert check_row_count("t", path, 5, "gene_universe") is True


def test_check_row_count_fails_on_a_row_mismatch(capsys, tmp_path):
    """A matrix with the wrong number of rows is the misalignment this catches."""
    path = tmp_path / "m.npy"
    np.save(path, np.zeros((4, 3)))
    assert check_row_count("t", path, 5, "gene_universe") is False
    assert "positional alignment" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# main — a skipped check is not a passed check
# ---------------------------------------------------------------------------


def _run_main(tmp_path, monkeypatch):
    import sys

    from esm2_mech.fetch_data import check_gene_coverage

    monkeypatch.setattr(
        sys, "argv", ["check_gene_coverage", "--data-dir", str(tmp_path)]
    )
    return check_gene_coverage.main()


def test_main_with_no_files_reports_nothing_checked(tmp_path, monkeypatch, capsys):
    assert _run_main(tmp_path, monkeypatch) == 1
    assert "nothing checked" in capsys.readouterr().out


def test_main_does_not_claim_all_expectations_hold_on_a_partial_run(
    tmp_path, monkeypatch, capsys
):
    """Only some inputs exist, so most checks cannot run. Reporting PASS here
    would say the gene sets are consistent on the strength of one check."""
    _write_tsv(tmp_path / "gene_list.tsv", ["gene"], [["A"], ["B"]])
    _write_tsv(tmp_path / "gene_universe.tsv", ["gene"], [["A"]])

    exit_code = _run_main(tmp_path, monkeypatch)
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "all gene-coverage expectations hold" not in output
    assert "INCOMPLETE" in output


def test_a_partial_run_names_the_missing_files(tmp_path, monkeypatch, capsys):
    _write_tsv(tmp_path / "gene_list.tsv", ["gene"], [["A"]])
    _write_tsv(tmp_path / "gene_universe.tsv", ["gene"], [["A"]])

    _run_main(tmp_path, monkeypatch)
    output = capsys.readouterr().out
    assert "pfam_families.json" in output


def test_main_passes_when_every_check_runs_and_agrees(tmp_path, monkeypatch, capsys):
    genes = [["A"], ["B"]]
    for name in (
        "gene_list.tsv",
        "gene_universe.tsv",
        "gene_proteome_features.tsv",
        "badonyi_features.tsv",
        "enzyme_labels.tsv",
    ):
        _write_tsv(tmp_path / name, ["gene"], genes)
    (tmp_path / "pfam_families.json").write_text(json.dumps({"A": "PF1", "B": "PF2"}))
    for name in ("proteome_features_aligned.npy", "badonyi_features_aligned.npy"):
        np.save(tmp_path / name, np.zeros((2, 3)))

    exit_code = _run_main(tmp_path, monkeypatch)
    output = capsys.readouterr().out
    assert exit_code == 0, output
    assert "all gene-coverage expectations hold" in output
