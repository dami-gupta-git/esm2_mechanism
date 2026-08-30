"""
Tests for esm2_mech.fetch_data.alphamissense_common.

stream_am_filter invariants:
- the header row is skipped (not treated as data).
- only indexed (uniprot, variant) keys are returned.
- a non-finite score ("nan"/"inf") is dropped, never stored — float() accepts
  it, so a bare parse would silently let it into the scientific output.
- an unparseable score is dropped.
- a clean finite score for a matched key is returned.

build_lookup invariants:
- one index entry per mappable variant, keyed by UniProt and protein variant.
- a variant whose gene has no UniProt mapping is dropped and counted.
- a genuine duplicate key is dropped and counted, keeping the first.
- the same variant repeated is not counted as a collision.
- every dropped variant is reported with a total, not just an example.
"""

import gzip

from esm2_mech.fetch_data.alphamissense_common import build_lookup, stream_am_filter


def _write_am_gz(tmp_path, rows):
    """Write a minimal AlphaMissense-style TSV.gz: header + (uniprot, pv, score)."""
    path = tmp_path / "am.tsv.gz"
    with gzip.open(path, "wt") as f:
        f.write("uniprot_id\tprotein_variant\tam_pathogenicity\n")
        for uniprot, pv, score in rows:
            f.write(f"{uniprot}\t{pv}\t{score}\n")
    return path


def test_returns_only_finite_matched_scores(tmp_path):
    index = {
        ("P1", "A1B"): "v_good",
        ("P1", "C2D"): "v_nan",
        ("P1", "E3F"): "v_inf",
        ("P1", "G4H"): "v_bad",
        # ("P9", ...) intentionally absent from the file
        ("P9", "Z9Z"): "v_missing",
    }
    am_gz = _write_am_gz(
        tmp_path,
        [
            ("P1", "A1B", "0.873"),    # valid → kept
            ("P1", "C2D", "nan"),      # non-finite → dropped
            ("P1", "E3F", "inf"),      # non-finite → dropped
            ("P1", "G4H", "not_a_num"),  # unparseable → dropped
            ("P2", "X1Y", "0.5"),      # not in index → ignored
        ],
    )
    scores = stream_am_filter(am_gz, index)
    assert scores == {"v_good": 0.873}


def test_header_not_parsed_as_data(tmp_path):
    # If the header were treated as data, ("uniprot_id","protein_variant") would
    # be looked up — it isn't in index, so this just confirms no crash and that a
    # real first data row is still matched.
    index = {("P1", "A1B"): "v_good"}
    am_gz = _write_am_gz(tmp_path, [("P1", "A1B", "0.42")])
    scores = stream_am_filter(am_gz, index)
    assert scores == {"v_good": 0.42}


# ---------------------------------------------------------------------------
# build_lookup
# ---------------------------------------------------------------------------


def _variant(gene, aa_pos, aa_wt="A", aa_mut="V"):
    return {"gene": gene, "aa_pos": aa_pos, "aa_wt": aa_wt, "aa_mut": aa_mut}


def test_build_lookup_indexes_every_mappable_variant():
    variants = [_variant("BRCA1", 1), _variant("KRAS", 2)]
    index = build_lookup(variants, {"BRCA1": "P38398", "KRAS": "P01116"})
    assert index == {
        ("P38398", "A1V"): "BRCA1_1_A_V",
        ("P01116", "A2V"): "KRAS_2_A_V",
    }


def test_build_lookup_drops_and_counts_variants_with_no_uniprot(capsys):
    """A gene with no UniProt mapping must be dropped and its total reported."""
    variants = [_variant("BRCA1", 1), _variant("NOMAP", 2), _variant("NOMAP", 3)]
    index = build_lookup(variants, {"BRCA1": "P38398"})
    assert list(index) == [("P38398", "A1V")]
    output = capsys.readouterr().out
    assert "2 variants dropped with no UniProt mapping" in output
    assert "NOMAP" in output


def test_build_lookup_drops_and_counts_a_duplicate_key(capsys):
    """Two genes sharing a UniProt ID collide on one protein variant."""
    variants = [_variant("GENE_A", 1), _variant("GENE_B", 1)]
    index = build_lookup(variants, {"GENE_A": "P1", "GENE_B": "P1"})
    assert index == {("P1", "A1V"): "GENE_A_1_A_V"}
    output = capsys.readouterr().out
    assert "1 variants dropped on a duplicate" in output
    assert "GENE_B" in output


def test_build_lookup_repeated_identical_variant_is_not_a_collision(capsys):
    variants = [_variant("BRCA1", 1), _variant("BRCA1", 1)]
    index = build_lookup(variants, {"BRCA1": "P38398"})
    assert index == {("P38398", "A1V"): "BRCA1_1_A_V"}
    assert "duplicate" not in capsys.readouterr().out


def test_build_lookup_reports_the_mapped_total(capsys):
    variants = [_variant("BRCA1", 1), _variant("NOMAP", 2)]
    build_lookup(variants, {"BRCA1": "P38398"})
    assert "variants with UniProt mapping: 1 / 2" in capsys.readouterr().out
