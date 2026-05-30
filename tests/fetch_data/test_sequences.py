"""
Tests for esm2_mech.fetch_data.sequences.

Network calls are mocked throughout — no real HTTP requests.

Invariants:
- fetch_uniprot_sequence: returns uppercase sequence on success
- fetch_uniprot_sequence: returns None on HTTP 404
- fetch_uniprot_sequence: raises TransientFetchError after all retries on non-404 HTTP error
- fetch_uniprot_sequence: raises TransientFetchError on network exception
- fetch_uniprot_sequence: strips FASTA header lines, concatenates sequence lines
- build_sequence_cache: skips already-cached IDs
- build_sequence_cache: writes cache atomically (no .tmp left)
- build_sequence_cache: corrupted cache file is deleted and re-fetched
- build_sequence_cache: transient failures do not write to cache
- fetch_pfam_families: returns gene -> pfam_id dict
- fetch_pfam_families: returns None for genes with no Pfam entry
- fetch_pfam_families: HTTP 404 maps gene to None
- fetch_pfam_families: corrupted cache deleted and re-fetched
- fetch_pfam_families: transient failures skip cache write
"""

import json
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from esm2_mech.fetch_data.sequences import (
    TransientFetchError,
    build_sequence_cache,
    fetch_pfam_families,
    fetch_uniprot_sequence,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fasta_response(seq, accession="P12345"):
    body = f">sp|{accession}|GENE_HUMAN Some protein\n{seq}\n"
    mock = MagicMock()
    mock.read.return_value = body.encode()
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    return mock


def _http_error(code):
    return urllib.error.HTTPError(url="", code=code, msg="", hdrs=None, fp=None)


# ---------------------------------------------------------------------------
# fetch_uniprot_sequence
# ---------------------------------------------------------------------------

class TestFetchUniprotSequence:

    def test_returns_uppercase_sequence(self):
        with patch("urllib.request.urlopen", return_value=_fasta_response("acdef")):
            result = fetch_uniprot_sequence("P12345", retries=1)
        assert result == "ACDEF"

    def test_multiline_fasta_concatenated(self):
        body = ">sp|P12345|GENE\nACDE\nFGHI\n"
        mock = MagicMock()
        mock.read.return_value = body.encode()
        mock.__enter__ = lambda s: s
        mock.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock):
            result = fetch_uniprot_sequence("P12345", retries=1)
        assert result == "ACDEFGHI"

    def test_returns_none_on_404(self):
        with patch("urllib.request.urlopen", side_effect=_http_error(404)):
            result = fetch_uniprot_sequence("P99999", retries=1)
        assert result is None

    def test_raises_transient_on_non_404_http_error(self):
        with patch("urllib.request.urlopen", side_effect=_http_error(500)):
            with pytest.raises(TransientFetchError):
                fetch_uniprot_sequence("P12345", retries=1, delay=0)

    def test_raises_transient_on_network_exception(self):
        with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
            with pytest.raises(TransientFetchError):
                fetch_uniprot_sequence("P12345", retries=1, delay=0)

    def test_retries_before_raising(self):
        with patch("urllib.request.urlopen", side_effect=_http_error(503)) as mock_open:
            with pytest.raises(TransientFetchError):
                fetch_uniprot_sequence("P12345", retries=3, delay=0)
        assert mock_open.call_count == 3

    def test_succeeds_after_transient_failure(self):
        responses = [_http_error(503), _fasta_response("MKTAY")]
        with patch("urllib.request.urlopen", side_effect=responses):
            result = fetch_uniprot_sequence("P12345", retries=2, delay=0)
        assert result == "MKTAY"


# ---------------------------------------------------------------------------
# build_sequence_cache
# ---------------------------------------------------------------------------

class TestBuildSequenceCache:

    def _variants(self, ids):
        return [{"uniprot_id": uid} for uid in ids]

    def test_skips_already_cached(self, tmp_path):
        existing = {"P1": "AAAA"}
        (tmp_path / "sequences.json").write_text(json.dumps(existing))
        with patch("esm2_mech.fetch_data.sequences.fetch_uniprot_sequence") as mock_fetch:
            result = build_sequence_cache(self._variants(["P1"]), str(tmp_path))
        mock_fetch.assert_not_called()
        assert result["P1"] == "AAAA"

    def test_fetches_missing_ids(self, tmp_path):
        with patch("esm2_mech.fetch_data.sequences.fetch_uniprot_sequence", return_value="ACDEF"):
            result = build_sequence_cache(self._variants(["P1"]), str(tmp_path))
        assert result["P1"] == "ACDEF"

    def test_no_tmp_file_left(self, tmp_path):
        with patch("esm2_mech.fetch_data.sequences.fetch_uniprot_sequence", return_value="ACDEF"):
            build_sequence_cache(self._variants(["P1"]), str(tmp_path))
        assert not (tmp_path / "sequences.json.tmp").exists()

    def test_corrupted_cache_deleted_and_refetched(self, tmp_path):
        cache = tmp_path / "sequences.json"
        cache.write_text("not valid json{{{")
        with patch("esm2_mech.fetch_data.sequences.fetch_uniprot_sequence", return_value="FRESH"):
            result = build_sequence_cache(self._variants(["P1"]), str(tmp_path))
        assert result["P1"] == "FRESH"

    def test_transient_failure_not_cached(self, tmp_path):
        with patch(
            "esm2_mech.fetch_data.sequences.fetch_uniprot_sequence",
            side_effect=TransientFetchError("P1"),
        ):
            result = build_sequence_cache(self._variants(["P1"]), str(tmp_path))
        assert "P1" not in result


# ---------------------------------------------------------------------------
# fetch_pfam_families
# ---------------------------------------------------------------------------

class TestFetchPfamFamilies:

    def _variants(self, pairs):
        return [{"gene": g, "uniprot_id": uid} for g, uid in pairs]

    def _uniprot_json_response(self, pfam_id):
        data = {}
        if pfam_id:
            data["uniProtKBCrossReferences"] = [{"database": "Pfam", "id": pfam_id}]
        body = json.dumps(data).encode()
        mock = MagicMock()
        mock.read.return_value = body
        mock.__enter__ = lambda s: s
        mock.__exit__ = MagicMock(return_value=False)
        return mock

    def test_returns_pfam_id(self, tmp_path):
        variants = self._variants([("BRCA1", "P38398")])
        with patch("urllib.request.urlopen", return_value=self._uniprot_json_response("PF00001")):
            result = fetch_pfam_families(variants, str(tmp_path))
        assert result["BRCA1"] == "PF00001"

    def test_returns_none_when_no_pfam(self, tmp_path):
        variants = self._variants([("TP53", "P04637")])
        with patch("urllib.request.urlopen", return_value=self._uniprot_json_response(None)):
            result = fetch_pfam_families(variants, str(tmp_path))
        assert result["TP53"] is None

    def test_404_maps_to_none(self, tmp_path):
        variants = self._variants([("GENE1", "P99999")])
        with patch("urllib.request.urlopen", side_effect=_http_error(404)):
            result = fetch_pfam_families(variants, str(tmp_path))
        assert result["GENE1"] is None

    def test_uses_cached_result(self, tmp_path):
        cached = {"BRCA1": "PF00001"}
        (tmp_path / "pfam_families.json").write_text(json.dumps(cached))
        with patch("urllib.request.urlopen") as mock_open:
            result = fetch_pfam_families(self._variants([("BRCA1", "P38398")]), str(tmp_path))
        mock_open.assert_not_called()
        assert result == cached

    def test_corrupted_cache_deleted_and_refetched(self, tmp_path):
        (tmp_path / "pfam_families.json").write_text("not json{{{")
        variants = self._variants([("BRCA1", "P38398")])
        with patch("urllib.request.urlopen", return_value=self._uniprot_json_response("PF00002")):
            result = fetch_pfam_families(variants, str(tmp_path))
        assert result["BRCA1"] == "PF00002"

    def test_transient_failure_skips_cache_write(self, tmp_path):
        variants = self._variants([("GENE1", "P12345")])
        with patch("urllib.request.urlopen", side_effect=_http_error(503)):
            fetch_pfam_families(variants, str(tmp_path))
        assert not (tmp_path / "pfam_families.json").exists()
