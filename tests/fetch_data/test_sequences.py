"""
Tests for esm2_mech.fetch_data.sequences.

Network calls are mocked throughout — no real HTTP requests.

Invariants:
- fetch_uniprot_sequence: returns uppercase sequence on success
- fetch_uniprot_sequence: returns None on HTTP 404
- fetch_uniprot_sequence: raises TransientFetchError after all retries on non-404 HTTP error
- fetch_uniprot_sequence: raises TransientFetchError on network exception
- fetch_uniprot_sequence: strips FASTA header lines, concatenates sequence lines
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

    def test_returns_pfam_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr("esm2_mech.fetch_data.sequences.PFAM_JSON", tmp_path / "pfam_families.json")
        variants = self._variants([("BRCA1", "P38398")])
        with patch("urllib.request.urlopen", return_value=self._uniprot_json_response("PF00001")):
            result = fetch_pfam_families(variants)
        assert result["BRCA1"] == "PF00001"

    def test_returns_none_when_no_pfam(self, tmp_path, monkeypatch):
        monkeypatch.setattr("esm2_mech.fetch_data.sequences.PFAM_JSON", tmp_path / "pfam_families.json")
        variants = self._variants([("TP53", "P04637")])
        with patch("urllib.request.urlopen", return_value=self._uniprot_json_response(None)):
            result = fetch_pfam_families(variants)
        assert result["TP53"] is None

    def test_404_maps_to_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr("esm2_mech.fetch_data.sequences.PFAM_JSON", tmp_path / "pfam_families.json")
        variants = self._variants([("GENE1", "P99999")])
        with patch("urllib.request.urlopen", side_effect=_http_error(404)):
            result = fetch_pfam_families(variants)
        assert result["GENE1"] is None

    def test_uses_cached_result(self, tmp_path, monkeypatch):
        cache_path = tmp_path / "pfam_families.json"
        monkeypatch.setattr("esm2_mech.fetch_data.sequences.PFAM_JSON", cache_path)
        cached = {"BRCA1": "PF00001"}
        cache_path.write_text(json.dumps(cached))
        with patch("urllib.request.urlopen") as mock_open:
            result = fetch_pfam_families(self._variants([("BRCA1", "P38398")]))
        mock_open.assert_not_called()
        assert result == cached

    def test_corrupted_cache_deleted_and_refetched(self, tmp_path, monkeypatch):
        cache_path = tmp_path / "pfam_families.json"
        monkeypatch.setattr("esm2_mech.fetch_data.sequences.PFAM_JSON", cache_path)
        cache_path.write_text("not json{{{")
        variants = self._variants([("BRCA1", "P38398")])
        with patch("urllib.request.urlopen", return_value=self._uniprot_json_response("PF00002")):
            result = fetch_pfam_families(variants)
        assert result["BRCA1"] == "PF00002"

    def test_transient_failure_skips_cache_write(self, tmp_path, monkeypatch):
        cache_path = tmp_path / "pfam_families.json"
        monkeypatch.setattr("esm2_mech.fetch_data.sequences.PFAM_JSON", cache_path)
        variants = self._variants([("GENE1", "P12345")])
        with patch("urllib.request.urlopen", side_effect=_http_error(503)):
            fetch_pfam_families(variants)
        assert not cache_path.exists()
