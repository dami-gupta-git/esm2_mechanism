"""
Tests for fetch_uniprot_sequences.py.

Covers:
- parse_fasta: standard UniProt sp| format
- parse_fasta: plain (non-pipe) header
- parse_fasta: multiple records
- parse_fasta: empty input
- parse_fasta: multi-line sequences are concatenated
- select_extended: excludes base, excludes unneeded IDs, new overrides existing
"""

from esm2_mechanism.fetch_data.fetch_uniprot_sequences import parse_fasta, select_extended


ISOFORM_FASTA = """\
>sp|P12345-2|GENE_HUMAN Isoform 2 OS=Homo sapiens
ACDEFGHIKLM
"""

UNIPROT_FASTA = """\
>sp|P12345|GENE_HUMAN Some protein OS=Homo sapiens
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL
>sp|Q99999|PROT_MOUSE Another protein OS=Mus musculus
ACDEFGHIKLM
"""

PLAIN_FASTA = """\
>P12345
MKTAYIAKQRQISFVK
"""

MULTILINE_FASTA = """\
>sp|A00001|MLTI_HUMAN Multiline protein
ACDEFG
HIKLMN
PQRST
"""


def test_uniprot_format_extracts_accession():
    result = parse_fasta(UNIPROT_FASTA)
    assert "P12345" in result
    assert "Q99999" in result


def test_uniprot_format_sequences_correct():
    result = parse_fasta(UNIPROT_FASTA)
    assert result["Q99999"] == "ACDEFGHIKLM"


def test_plain_header_uses_full_id():
    result = parse_fasta(PLAIN_FASTA)
    assert "P12345" in result
    assert result["P12345"] == "MKTAYIAKQRQISFVK"


def test_multiline_sequence_concatenated():
    result = parse_fasta(MULTILINE_FASTA)
    assert result["A00001"] == "ACDEFGHIKLMNPQRST"


def test_empty_input_returns_empty_dict():
    assert parse_fasta("") == {}


def test_multiple_records_all_returned():
    result = parse_fasta(UNIPROT_FASTA)
    assert len(result) == 2


def test_isoform_suffix_stripped():
    result = parse_fasta(ISOFORM_FASTA)
    assert "P12345" in result
    assert "P12345-2" not in result


def test_select_extended_excludes_base_sequences():
    # An ID covered by sequences.json must not be duplicated into the extended cache.
    base = {"P1": "AAA"}
    out = select_extended(base, {}, {"P1": "AAA", "P2": "CCC"}, needed={"P1", "P2"})
    assert out == {"P2": "CCC"}


def test_select_extended_excludes_unneeded_ids():
    # Sequences no variant references must not bloat the extended cache.
    out = select_extended({}, {"P9": "GGG"}, {"P2": "CCC"}, needed={"P2"})
    assert out == {"P2": "CCC"}


def test_select_extended_new_overrides_existing():
    out = select_extended({}, {"P2": "OLD"}, {"P2": "NEW"}, needed={"P2"})
    assert out == {"P2": "NEW"}


def test_select_extended_empty_when_all_in_base():
    out = select_extended({"P1": "AAA"}, {}, {}, needed={"P1"})
    assert out == {}
