"""
Tests for esm2_mech.fetch_data.alphamissense_common.stream_am_filter.

Invariants:
- the header row is skipped (not treated as data).
- only indexed (uniprot, variant) keys are returned.
- a non-finite score ("nan"/"inf") is dropped, never stored — float() accepts
  it, so a bare parse would silently let it into the scientific output.
- an unparseable score is dropped.
- a clean finite score for a matched key is returned.
"""

import gzip

from esm2_mech.fetch_data.alphamissense_common import stream_am_filter


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
