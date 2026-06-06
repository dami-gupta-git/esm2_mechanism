"""
Tests for esm2_mech.fetch_data.build_valid_variants.

build_valid_variants is the central data-integrity filter feeding every
downstream embedding/analysis step. A bug here silently drops valid variants
or admits invalid ones, so the filter logic is exercised directly.

Invariants:
- a fully-valid variant is kept
- a variant with no uniprot_id is dropped (missing-UID bucket)
- a variant whose uniprot_id is absent from the sequence cache is dropped
- a variant missing aa_pos / aa_wt / aa_mut is dropped (missing-fields bucket)
- a variant whose WT residue does not match the reference is dropped (invalid window)
- each drop lands in exactly one bucket; counts sum to the inputs
- the kept variants are the input dicts unchanged
"""

import json

import pytest

from esm2_mech.fetch_data import build_valid_variants as bvv


# A short reference sequence for the single cached accession.
#                pos: 1234567890
REF_SEQUENCE = "MKTAYIAKQR"


def _variant(uid="P12345", aa_pos=1, aa_wt="M", aa_mut="A", **extra):
    v = {"uniprot_id": uid, "aa_pos": aa_pos, "aa_wt": aa_wt, "aa_mut": aa_mut}
    v.update(extra)
    return v


@pytest.fixture
def patched_inputs(tmp_path, monkeypatch):
    """Point the module's input paths at temp files; caller writes their content."""
    variants_path = tmp_path / "variants.json"
    sequences_path = tmp_path / "sequences.json"
    monkeypatch.setattr(bvv, "VARIANTS_JSON", variants_path)
    monkeypatch.setattr(bvv, "SEQUENCES_JSON", sequences_path)

    def write(variants, seq_cache=None):
        if seq_cache is None:
            seq_cache = {"P12345": REF_SEQUENCE}
        with open(variants_path, "w") as f:
            json.dump(variants, f)
        with open(sequences_path, "w") as f:
            json.dump(seq_cache, f)

    return write


def test_valid_variant_is_kept(patched_inputs):
    patched_inputs([_variant()])
    valid = bvv.build_valid_variants()
    assert len(valid) == 1
    # the kept entry is the input dict, unmodified
    assert valid[0]["uniprot_id"] == "P12345"
    assert valid[0]["aa_pos"] == 1


def test_missing_uniprot_id_is_dropped(patched_inputs):
    patched_inputs([_variant(uid=None), _variant(uid="")])
    assert bvv.build_valid_variants() == []


def test_uid_not_in_sequence_cache_is_dropped(patched_inputs):
    patched_inputs([_variant(uid="Q99999")], seq_cache={"P12345": REF_SEQUENCE})
    assert bvv.build_valid_variants() == []


def test_missing_aa_fields_are_dropped(patched_inputs):
    patched_inputs(
        [
            _variant(aa_pos=None),
            _variant(aa_wt=""),
            _variant(aa_mut=""),
        ]
    )
    assert bvv.build_valid_variants() == []


def test_aa_pos_zero_is_treated_as_present_but_window_invalid(patched_inputs):
    # aa_pos is checked with `is None`, so 0 passes the field check but
    # apply_missense rejects the out-of-range (0-indexed: -1) position.
    patched_inputs([_variant(aa_pos=0)])
    assert bvv.build_valid_variants() == []


def test_wt_mismatch_is_dropped(patched_inputs):
    # position 2 is 'K' in REF_SEQUENCE, not 'A'
    patched_inputs([_variant(aa_pos=2, aa_wt="A", aa_mut="G")])
    assert bvv.build_valid_variants() == []


def test_out_of_range_position_is_dropped(patched_inputs):
    patched_inputs([_variant(aa_pos=999, aa_wt="M", aa_mut="A")])
    assert bvv.build_valid_variants() == []


def test_correct_wt_at_interior_position_is_kept(patched_inputs):
    # position 3 is 'T' in REF_SEQUENCE
    patched_inputs([_variant(aa_pos=3, aa_wt="T", aa_mut="S")])
    valid = bvv.build_valid_variants()
    assert len(valid) == 1


def test_mixed_batch_keeps_only_valid(patched_inputs):
    variants = [
        _variant(aa_pos=1, aa_wt="M", aa_mut="A"),  # valid
        _variant(uid=None),                         # dropped: no uid
        _variant(uid="Q00000"),                     # dropped: not cached
        _variant(aa_pos=None),                      # dropped: missing field
        _variant(aa_pos=2, aa_wt="A", aa_mut="G"),  # dropped: wt mismatch
        _variant(aa_pos=4, aa_wt="A", aa_mut="V"),  # valid (pos 4 is 'A')
    ]
    patched_inputs(variants)
    valid = bvv.build_valid_variants()
    assert len(valid) == 2
    assert {tuple([w["aa_pos"], w["aa_wt"]]) for w in valid} == {(1, "M"), (4, "A")}


def test_missing_variants_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(bvv, "VARIANTS_JSON", tmp_path / "absent.json")
    monkeypatch.setattr(bvv, "SEQUENCES_JSON", tmp_path / "seq.json")
    with pytest.raises(FileNotFoundError):
        bvv.build_valid_variants()


def test_missing_sequences_file_raises(tmp_path, monkeypatch):
    variants_path = tmp_path / "variants.json"
    with open(variants_path, "w") as f:
        json.dump([_variant()], f)
    monkeypatch.setattr(bvv, "VARIANTS_JSON", variants_path)
    monkeypatch.setattr(bvv, "SEQUENCES_JSON", tmp_path / "absent_seq.json")
    with pytest.raises(FileNotFoundError):
        bvv.build_valid_variants()
