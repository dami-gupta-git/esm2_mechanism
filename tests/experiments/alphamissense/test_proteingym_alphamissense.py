"""
Tests for the ProteinGym AlphaMissense module's lookup cache and streaming.

A failed network lookup was previously cached as an empty string, identical to a
genuine "no such record". That permanently dropped the assay from every later
run with no way to tell a transient outage from a real absence.

Covers:
- load_mnemonic_map: a successful lookup is cached as its accession
- load_mnemonic_map: a definitive empty response is cached as null
- load_mnemonic_map: a 404 is cached as null, since it is a real answer
- load_mnemonic_map: a transient network error is not cached, so it is retried
- load_mnemonic_map: a transient failure is reported, not silent
- load_mnemonic_map: an already-cached mnemonic is not queried again
- load_mnemonic_map: a corrupt cache file is discarded and refetched
- stream_am: matching rows are returned, non-matching ignored
- stream_am: the header and comment lines are skipped
- stream_am: a short row is skipped
- stream_am: an unparseable score is counted and reported, not silently dropped
"""

import gzip
import json
from urllib.error import HTTPError, URLError

import pytest

from esm2_mech.experiments.alphamissense import proteingym_alphamissense as pga


class _Response:
    """Minimal stand-in for the urlopen context manager."""

    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body.encode()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _patch_uniprot(monkeypatch, tmp_path, responses):
    """Serve `responses` by mnemonic; a value that is an exception is raised."""
    monkeypatch.setattr(pga, "MAP_CACHE", tmp_path / "mnemonics.json")
    monkeypatch.setattr(pga.time, "sleep", lambda _s: None)

    def fake_urlopen(request, timeout=None):
        mnemonic = request.full_url.rsplit("/", 1)[1].split(".")[0]
        result = responses[mnemonic]
        if isinstance(result, Exception):
            raise result
        return _Response(result)

    monkeypatch.setattr(pga, "urlopen", fake_urlopen)


def _cache_contents(tmp_path):
    return json.loads((tmp_path / "mnemonics.json").read_text())


# ---------------------------------------------------------------------------
# load_mnemonic_map
# ---------------------------------------------------------------------------


def test_a_successful_lookup_is_cached_as_its_accession(monkeypatch, tmp_path):
    _patch_uniprot(monkeypatch, tmp_path, {"BRCA1_HUMAN": "Entry\nP38398\n"})
    assert pga.load_mnemonic_map(["BRCA1_HUMAN"]) == {"BRCA1_HUMAN": "P38398"}
    assert _cache_contents(tmp_path) == {"BRCA1_HUMAN": "P38398"}


def test_a_definitive_empty_response_is_cached_as_null(monkeypatch, tmp_path):
    """A 2xx with no data row means the mnemonic genuinely has no entry."""
    _patch_uniprot(monkeypatch, tmp_path, {"NOPE_HUMAN": "Entry\n"})
    assert pga.load_mnemonic_map(["NOPE_HUMAN"]) == {"NOPE_HUMAN": None}
    assert _cache_contents(tmp_path) == {"NOPE_HUMAN": None}


def test_a_404_is_cached_as_null(monkeypatch, tmp_path):
    error = HTTPError("url", 404, "Not Found", {}, None)
    _patch_uniprot(monkeypatch, tmp_path, {"GONE_HUMAN": error})
    assert pga.load_mnemonic_map(["GONE_HUMAN"]) == {"GONE_HUMAN": None}
    assert _cache_contents(tmp_path) == {"GONE_HUMAN": None}


@pytest.mark.parametrize(
    "error",
    [URLError("connection reset"), HTTPError("url", 503, "Unavailable", {}, None)],
    ids=["network_error", "server_error"],
)
def test_a_transient_failure_is_not_cached(monkeypatch, tmp_path, error):
    """Caching it would drop the assay from every later run permanently."""
    _patch_uniprot(monkeypatch, tmp_path, {"FLAKY_HUMAN": error})
    result = pga.load_mnemonic_map(["FLAKY_HUMAN"])
    assert "FLAKY_HUMAN" not in result
    assert _cache_contents(tmp_path) == {}


def test_a_transient_failure_is_reported(monkeypatch, tmp_path, capsys):
    _patch_uniprot(monkeypatch, tmp_path, {"FLAKY_HUMAN": URLError("reset")})
    pga.load_mnemonic_map(["FLAKY_HUMAN"])
    assert "not cached" in capsys.readouterr().err


def test_an_already_cached_mnemonic_is_not_queried_again(monkeypatch, tmp_path):
    cache = tmp_path / "mnemonics.json"
    cache.write_text(json.dumps({"BRCA1_HUMAN": "P38398"}))
    monkeypatch.setattr(pga, "MAP_CACHE", cache)

    def fail(*_args, **_kwargs):
        raise AssertionError("should not query UniProt for a cached mnemonic")

    monkeypatch.setattr(pga, "urlopen", fail)
    assert pga.load_mnemonic_map(["BRCA1_HUMAN"]) == {"BRCA1_HUMAN": "P38398"}


def test_a_corrupt_cache_is_discarded_and_refetched(monkeypatch, tmp_path):
    cache = tmp_path / "mnemonics.json"
    cache.write_text("{not json")
    _patch_uniprot(monkeypatch, tmp_path, {"BRCA1_HUMAN": "Entry\nP38398\n"})
    assert pga.load_mnemonic_map(["BRCA1_HUMAN"]) == {"BRCA1_HUMAN": "P38398"}


# ---------------------------------------------------------------------------
# stream_am
# ---------------------------------------------------------------------------


def _write_am(tmp_path, monkeypatch, lines):
    path = tmp_path / "am.tsv.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("\n".join(lines) + "\n")
    monkeypatch.setattr(pga, "AM_FILE", path)
    return path


def test_only_indexed_rows_are_returned(tmp_path, monkeypatch):
    _write_am(
        tmp_path,
        monkeypatch,
        [
            "uniprot_id\tprotein_variant\tam_pathogenicity",
            "P1\tA1V\t0.9",
            "P2\tR5K\t0.2",
        ],
    )
    scores = pga.stream_am({("P1", "A1V"): []})
    assert scores == {("P1", "A1V"): pytest.approx(0.9)}


def test_header_and_comment_lines_are_skipped(tmp_path, monkeypatch):
    _write_am(
        tmp_path,
        monkeypatch,
        [
            "# a comment",
            "uniprot_id\tprotein_variant\tam_pathogenicity",
            "P1\tA1V\t0.9",
        ],
    )
    assert pga.stream_am({("P1", "A1V"): []}) == {("P1", "A1V"): pytest.approx(0.9)}


def test_a_short_row_is_skipped(tmp_path, monkeypatch):
    _write_am(
        tmp_path,
        monkeypatch,
        ["uniprot_id\tprotein_variant\tam_pathogenicity", "P1\tA1V"],
    )
    assert pga.stream_am({("P1", "A1V"): []}) == {}


def test_an_unparseable_score_is_counted_and_reported(tmp_path, monkeypatch, capsys):
    """A dropped row that nothing counts is how a coverage loss goes unnoticed."""
    _write_am(
        tmp_path,
        monkeypatch,
        [
            "uniprot_id\tprotein_variant\tam_pathogenicity",
            "P1\tA1V\tnot_a_number",
        ],
    )
    assert pga.stream_am({("P1", "A1V"): []}) == {}
    assert "unparseable" in capsys.readouterr().err
