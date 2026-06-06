"""
Tests for esm2_mech.fetch_data.fetch_sequences.prefetch_sequences.

The sequence prefetcher must honour the project's cache invariants:
  1. Atomic writes (verified indirectly: no .tmp left behind after a run).
  2. Resume — accessions already cached or already known-absent are not re-fetched.
  3. A definitive 404 is recorded in the not-found cache (so it is not re-fetched),
     while a transient failure is NOT written to either cache and is left to retry.

Network calls are mocked throughout — no real HTTP requests, and time.sleep is
patched out so the test does not wait.
"""

import json

import pytest

from esm2_mech.fetch_data import fetch_sequences as fs
from esm2_mech.fetch_data.uniprot_fetch import TransientFetchError


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Redirect all module paths into tmp_path and disable the inter-fetch sleep."""
    variants_path = tmp_path / "variants.json"
    seq_path = tmp_path / "sequences.json"
    not_found_path = tmp_path / "sequences_not_found.json"

    monkeypatch.setattr(fs, "VARIANTS_JSON", variants_path)
    monkeypatch.setattr(fs, "SEQUENCES_JSON", seq_path)
    monkeypatch.setattr(fs, "SEQUENCES_NOT_FOUND_JSON", not_found_path)
    monkeypatch.setattr(fs, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(fs.time, "sleep", lambda *_: None)

    def write_variants(uids):
        with open(variants_path, "w") as f:
            json.dump([{"uniprot_id": uid} for uid in uids], f)

    def read_json(path):
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    class Env:
        pass

    e = Env()
    e.tmp = tmp_path
    e.variants_path = variants_path
    e.seq_path = seq_path
    e.not_found_path = not_found_path
    e.write_variants = write_variants
    e.seqs = lambda: read_json(seq_path)
    e.not_found = lambda: read_json(not_found_path)
    return e


def _patch_fetch(monkeypatch, mapping):
    """Patch fetch_uniprot_sequence with a dict {uid: result-or-exception-instance}.

    A value that is an exception instance is raised; otherwise it is returned.
    Records the call order so callers can assert what was (not) fetched.
    """
    calls = []

    def fake_fetch(uid):
        calls.append(uid)
        outcome = mapping[uid]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(fs, "fetch_uniprot_sequence", fake_fetch)
    return calls


def test_successful_fetch_is_cached(env, monkeypatch):
    env.write_variants(["P1", "P2"])
    _patch_fetch(monkeypatch, {"P1": "ACDEF", "P2": "GHIKL"})
    fs.prefetch_sequences()
    assert env.seqs() == {"P1": "ACDEF", "P2": "GHIKL"}
    assert env.not_found() == []


def test_404_recorded_in_not_found(env, monkeypatch):
    # fetch_uniprot_sequence returns None for a definitive 404.
    env.write_variants(["P1", "PX"])
    _patch_fetch(monkeypatch, {"P1": "ACDEF", "PX": None})
    fs.prefetch_sequences()
    assert env.seqs() == {"P1": "ACDEF"}
    assert env.not_found() == ["PX"]


def test_transient_failure_not_cached(env, monkeypatch):
    # A transient failure must not land in either cache — it stays retryable.
    env.write_variants(["P1", "PT"])
    _patch_fetch(
        monkeypatch,
        {"P1": "ACDEF", "PT": TransientFetchError("boom")},
    )
    fs.prefetch_sequences()
    assert env.seqs() == {"P1": "ACDEF"}
    assert env.not_found() == []
    assert "PT" not in (env.seqs() or {})


def test_resume_skips_already_cached(env, monkeypatch):
    env.write_variants(["P1", "P2"])
    with open(env.seq_path, "w") as f:
        json.dump({"P1": "EXISTING"}, f)
    calls = _patch_fetch(monkeypatch, {"P2": "NEW"})
    fs.prefetch_sequences()
    assert calls == ["P2"]  # P1 not re-fetched
    assert env.seqs() == {"P1": "EXISTING", "P2": "NEW"}


def test_resume_skips_known_absent(env, monkeypatch):
    env.write_variants(["P1", "PX"])
    with open(env.not_found_path, "w") as f:
        json.dump(["PX"], f)
    calls = _patch_fetch(monkeypatch, {"P1": "ACDEF"})
    fs.prefetch_sequences()
    assert calls == ["P1"]  # PX not re-fetched
    assert env.seqs() == {"P1": "ACDEF"}


def test_nothing_to_fetch_writes_nothing(env, monkeypatch):
    env.write_variants(["P1"])
    with open(env.seq_path, "w") as f:
        json.dump({"P1": "ACDEF"}, f)

    def boom(uid):
        raise AssertionError("should not fetch when cache is complete")

    monkeypatch.setattr(fs, "fetch_uniprot_sequence", boom)
    fs.prefetch_sequences()  # must not raise
    assert env.seqs() == {"P1": "ACDEF"}


def test_corrupt_seq_cache_is_discarded_and_refetched(env, monkeypatch):
    # load_json_or_discard deletes a corrupt cache and returns None, so the
    # accession is re-fetched rather than crashing the run.
    env.write_variants(["P1"])
    with open(env.seq_path, "w") as f:
        f.write("not json{{{")
    _patch_fetch(monkeypatch, {"P1": "ACDEF"})
    fs.prefetch_sequences()
    assert env.seqs() == {"P1": "ACDEF"}


def test_no_tmp_file_left_behind(env, monkeypatch):
    # Atomic writes go through a .tmp then os.replace; none should survive.
    env.write_variants(["P1", "P2"])
    _patch_fetch(monkeypatch, {"P1": "ACDEF", "P2": "GHIKL"})
    fs.prefetch_sequences()
    leftover = list(env.tmp.glob("*.tmp"))
    assert leftover == []


def test_uids_deduplicated_across_variants(env, monkeypatch):
    # The same accession appearing on multiple variant rows is fetched once.
    env.write_variants(["P1", "P1", "P1"])
    calls = _patch_fetch(monkeypatch, {"P1": "ACDEF"})
    fs.prefetch_sequences()
    assert calls == ["P1"]


def test_missing_variants_file_raises(env, monkeypatch):
    # variants.json absent → FileNotFoundError, no fetch attempted.
    env.variants_path.unlink(missing_ok=True)
    with pytest.raises(FileNotFoundError):
        fs.prefetch_sequences()
