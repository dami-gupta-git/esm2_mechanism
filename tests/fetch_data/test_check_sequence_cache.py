"""
Tests for the cached-sequence validator.

This tool exists to catch an isoform mismatch: a cached sequence that is the
right length but the wrong protein, so a variant's wild-type residue does not
match the position it names. Every count it reports must be a real count, since
a silently dropped variant is what the check is meant to reveal.

Covers:
- load_json: a missing file raises rather than returning an empty result
- load_json: a corrupt file raises naming the path
- load_sequences: the extended overlay takes precedence over the base file
- load_sequences: a missing overlay file leaves the base sequences unchanged
- check_sequence_sanity: an empty sequence is reported
- check_sequence_sanity: a sequence with a letter outside the alphabet is reported
- check_sequence_sanity: selenocysteine is accepted as a real residue
- check_sequence_sanity: clean sequences report nothing
- check_wt_agreement: a matching residue counts as checked, not mismatched
- check_wt_agreement: a differing residue is counted and its protein named
- check_wt_agreement: a variant with no cached sequence is counted separately
- check_wt_agreement: positions past the end and below one are out of bounds
- check_wt_agreement: a variant with no UniProt ID is skipped entirely
- check_wt_agreement: only proteins with a mismatch are reported, with their totals
- check_wt_agreement: the four counts and the skips account for every variant
"""

import json

import pytest

from esm2_mech.fetch_data.check_sequence_cache import (
    check_sequence_sanity,
    check_wt_agreement,
    load_json,
    load_sequences,
)

SEQ = "MKTAYIAKQR"  # 10 residues, 1-indexed


def _variant(uniprot_id="P1", aa_pos=1, aa_wt="M"):
    return {"uniprot_id": uniprot_id, "aa_pos": aa_pos, "aa_wt": aa_wt}


def _write_json(path, data):
    path.write_text(json.dumps(data))
    return path


# ---------------------------------------------------------------------------
# load_json / load_sequences
# ---------------------------------------------------------------------------


def test_load_json_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="required input not found"):
        load_json(tmp_path / "absent.json")


def test_load_json_corrupt_file_raises_naming_the_path(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    with pytest.raises(ValueError, match="broken.json"):
        load_json(path)


def test_extended_sequences_override_the_base_file(tmp_path):
    """The overlay exists to correct a base sequence, so it must win."""
    base = _write_json(tmp_path / "seq.json", {"P1": "AAAA", "P2": "CCCC"})
    extended = _write_json(tmp_path / "ext.json", {"P1": "WWWW"})
    assert load_sequences(base, extended) == {"P1": "WWWW", "P2": "CCCC"}


def test_absent_overlay_leaves_base_sequences_unchanged(tmp_path):
    base = _write_json(tmp_path / "seq.json", {"P1": "AAAA"})
    assert load_sequences(base, tmp_path / "absent.json") == {"P1": "AAAA"}


# ---------------------------------------------------------------------------
# check_sequence_sanity
# ---------------------------------------------------------------------------


def test_empty_sequence_is_reported():
    assert check_sequence_sanity({"P1": "", "P2": SEQ}) == ["P1"]


def test_sequence_with_an_invalid_letter_is_reported():
    """A residue letter outside the alphabet means the cache holds something
    that is not a protein sequence, such as an error page or a gap character."""
    assert check_sequence_sanity({"P1": "AAXA"}) == ["P1"]
    assert check_sequence_sanity({"P1": "AA-A"}) == ["P1"]


def test_selenocysteine_is_accepted():
    """U is a real residue and must not be reported as invalid."""
    assert check_sequence_sanity({"P1": "AAUA"}) == []


def test_clean_sequences_report_nothing():
    assert check_sequence_sanity({"P1": SEQ, "P2": "ACDEFG"}) == []


# ---------------------------------------------------------------------------
# check_wt_agreement
# ---------------------------------------------------------------------------


def test_a_matching_residue_counts_as_checked():
    checked, missing, out_of_bounds, mismatched, proteins = check_wt_agreement(
        {"P1": SEQ}, [_variant(aa_pos=1, aa_wt="M")]
    )
    assert (checked, missing, out_of_bounds, mismatched) == (1, 0, 0, 0)
    assert proteins == {}


def test_a_differing_residue_is_counted_and_its_protein_named():
    checked, _, _, mismatched, proteins = check_wt_agreement(
        {"P1": SEQ}, [_variant(aa_pos=1, aa_wt="A")]
    )
    assert (checked, mismatched) == (1, 1)
    assert proteins == {"P1": [1, 1]}


def test_a_variant_with_no_cached_sequence_is_counted_separately():
    """No sequence is not a mismatch; conflating them would hide a fetch failure."""
    _, missing, _, mismatched, _ = check_wt_agreement(
        {}, [_variant(uniprot_id="P_ABSENT")]
    )
    assert (missing, mismatched) == (1, 0)


@pytest.mark.parametrize("aa_pos", [0, -1, 11, 999])
def test_positions_outside_the_sequence_are_out_of_bounds(aa_pos):
    _, _, out_of_bounds, mismatched, _ = check_wt_agreement(
        {"P1": SEQ}, [_variant(aa_pos=aa_pos)]
    )
    assert (out_of_bounds, mismatched) == (1, 0)


def test_last_position_is_in_bounds():
    checked, _, out_of_bounds, mismatched, _ = check_wt_agreement(
        {"P1": SEQ}, [_variant(aa_pos=10, aa_wt="R")]
    )
    assert (checked, out_of_bounds, mismatched) == (1, 0, 0)


@pytest.mark.parametrize("uniprot_id", ["", None])
def test_a_variant_with_no_uniprot_id_is_skipped(uniprot_id):
    counts = check_wt_agreement({"P1": SEQ}, [_variant(uniprot_id=uniprot_id)])
    assert counts[:4] == (0, 0, 0, 0)


def test_only_proteins_with_a_mismatch_are_reported_with_their_totals():
    sequences = {"P1": SEQ, "P2": SEQ}
    variants = [
        _variant("P1", 1, "M"),   # matches
        _variant("P1", 2, "Z"),   # mismatches
        _variant("P2", 1, "M"),   # matches
    ]
    _, _, _, mismatched, proteins = check_wt_agreement(sequences, variants)
    assert mismatched == 1
    assert proteins == {"P1": [2, 1]}


def test_every_variant_lands_in_exactly_one_bucket():
    """A variant that is silently dropped from all four counts would hide the
    very disagreement this tool is looking for."""
    sequences = {"P1": SEQ}
    variants = [
        _variant("P1", 1, "M"),        # checked, matches
        _variant("P1", 2, "Z"),        # checked, mismatches
        _variant("P_ABSENT", 1, "M"),  # no cached sequence
        _variant("P1", 99, "M"),       # out of bounds
        _variant("", 1, "M"),          # skipped: no UniProt ID
    ]
    checked, missing, out_of_bounds, _, _ = check_wt_agreement(sequences, variants)
    skipped = sum(1 for variant in variants if not variant["uniprot_id"])
    assert checked + missing + out_of_bounds + skipped == len(variants)
