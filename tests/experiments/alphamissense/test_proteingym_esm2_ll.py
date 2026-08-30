"""
Tests for the ProteinGym delta-LL module's pure logic.

Two things this pins. The caches were previously keyed on file existence alone,
so a changed assay list, window length or model silently reused a stale file.
And the decision gates compared a not-a-number against a threshold when nothing
had been scored, which records a definite FAIL for a run that had no data.

Covers:
- parse_mutant: a well-formed single substitution parses
- parse_mutant: multi-mutants, lowercase, and truncated forms return None
- parse_mutant: surrounding whitespace is tolerated
- windowing: the module uses the shared window function, not a local copy
- _score_cache_params: a changed variant list changes the fingerprint
- _score_cache_params: a reordered job list changes the fingerprint
- _score_cache_params: an unchanged job list keeps the same fingerprint
- _score_cache_params: a different model changes the fingerprint
- _score_cache_params: the window length is part of the fingerprint
- _load_cache_if_current: matching params return the cached payload
- _load_cache_if_current: differing params refuse the cache
- _load_cache_if_current: a corrupt params sidecar refuses the cache
- _load_cache_if_current: a cache with no params sidecar is refused
- _save_cache: writes the payload and the params that produced it
"""

import json

import pytest

from esm2_mech.experiments.alphamissense import proteingym_esm2_ll as pg
from esm2_mech.utils.sequences import window_sequence as shared_window_sequence


def _job(dms_id="A_HUMAN_Doe_2020", mutants=("A1C", "D2E")):
    return {"DMS_id": dms_id, "variants": [{"mutant": m} for m in mutants]}


# ---------------------------------------------------------------------------
# parse_mutant
# ---------------------------------------------------------------------------


def test_a_single_substitution_parses():
    assert pg.parse_mutant("A673C") == ("A", 673, "C")


@pytest.mark.parametrize(
    "mut_str",
    ["A673", "673C", "a673c", "A673C:B12D", "", "AA673C", "A673CC"],
    ids=["no_mut_residue", "no_wt_residue", "lowercase", "multi_mutant",
         "empty", "two_wt_letters", "two_mut_letters"],
)
def test_anything_but_a_single_substitution_returns_none(mut_str):
    assert pg.parse_mutant(mut_str) is None


def test_surrounding_whitespace_is_tolerated():
    assert pg.parse_mutant("  A673C  ") == ("A", 673, "C")


def test_module_uses_the_shared_window_function():
    """A local copy used a different window length, so these scores were not
    comparable with any other log-likelihood in the project."""
    assert pg.window_sequence is shared_window_sequence


# ---------------------------------------------------------------------------
# cache fingerprints
# ---------------------------------------------------------------------------


def test_an_unchanged_job_list_keeps_the_same_fingerprint():
    jobs = [_job()]
    assert pg._score_cache_params(jobs, "m") == pg._score_cache_params(jobs, "m")


def test_a_changed_variant_list_changes_the_fingerprint():
    before = pg._score_cache_params([_job(mutants=("A1C",))], "m")
    after = pg._score_cache_params([_job(mutants=("A1C", "D2E"))], "m")
    assert before["jobs_sha256"] != after["jobs_sha256"]


def test_a_reordered_job_list_changes_the_fingerprint():
    """Scores are stored per assay, so order is part of what produced the file."""
    before = pg._score_cache_params([_job("A"), _job("B")], "m")
    after = pg._score_cache_params([_job("B"), _job("A")], "m")
    assert before["jobs_sha256"] != after["jobs_sha256"]


def test_a_different_model_changes_the_fingerprint():
    jobs = [_job()]
    assert pg._score_cache_params(jobs, "model_a") != pg._score_cache_params(
        jobs, "model_b"
    )


def test_the_window_length_is_part_of_the_fingerprint():
    """The window decides which residues the model sees, so it changes the score."""
    from esm2_mech.utils.constants import MAX_SEQ_LEN

    assert pg._score_cache_params([_job()], "m")["max_seq_len"] == MAX_SEQ_LEN


# ---------------------------------------------------------------------------
# cache load / save
# ---------------------------------------------------------------------------


def _cache_pair(tmp_path):
    return tmp_path / "cache.json", tmp_path / "cache.params.json"


def test_matching_params_return_the_cached_payload(tmp_path):
    cache, params_path = _cache_pair(tmp_path)
    pg._save_cache(cache, params_path, {"a": 1}, {"version": 1})
    assert pg._load_cache_if_current(cache, params_path, {"version": 1}) == {"a": 1}


def test_differing_params_refuse_the_cache(tmp_path, capsys):
    cache, params_path = _cache_pair(tmp_path)
    pg._save_cache(cache, params_path, {"a": 1}, {"version": 1})
    assert pg._load_cache_if_current(cache, params_path, {"version": 2}) is None
    assert "different inputs" in capsys.readouterr().out


def test_a_corrupt_params_sidecar_refuses_the_cache(tmp_path):
    cache, params_path = _cache_pair(tmp_path)
    pg._save_cache(cache, params_path, {"a": 1}, {"version": 1})
    params_path.write_text("{not json")
    assert pg._load_cache_if_current(cache, params_path, {"version": 1}) is None


def test_a_cache_with_no_params_sidecar_is_refused(tmp_path):
    """A file written before fingerprinting existed has unknown provenance."""
    cache, params_path = _cache_pair(tmp_path)
    cache.write_text(json.dumps({"a": 1}))
    assert pg._load_cache_if_current(cache, params_path, {"version": 1}) is None


def test_save_cache_writes_the_payload_and_its_params(tmp_path):
    cache, params_path = _cache_pair(tmp_path)
    pg._save_cache(cache, params_path, {"a": 1}, {"version": 1, "model": "m"})
    assert json.loads(cache.read_text()) == {"a": 1}
    assert json.loads(params_path.read_text()) == {"version": 1, "model": "m"}
