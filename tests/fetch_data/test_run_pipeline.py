"""
Tests for run_pipeline.py.

Covers:
- _load_state / _save_state: reads, writes, handles corrupt/missing file
- Auto-resume: correct resume_from derived from state (no state, partial, all done)
- --from-step override: ignores saved state
- Step execution: runs correct steps, skips earlier ones, saves state after each success
- Failure behaviour: exits on error, does not update state for the failed step
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from esm2_mech.fetch_data.run_fetch_pipeline import (
    FIRST_STEP,
    LAST_STEP,
    _load_state,
    _save_state,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_main(argv: list[str], step_side_effects: dict[int, Exception | None] = None):
    """
    Invoke run_pipeline.main() with the given argv, mocking out all step functions.

    step_side_effects: maps step number to an exception to raise (or None for success).
    Returns (mock_calls, saved_states) where saved_states is a list of state dicts
    written by _save_state in order.
    """
    side_effects = step_side_effects or {}
    saved_states = []

    def make_step_mock(step_num):
        exc = side_effects.get(step_num)
        mock = MagicMock(name=f"step_{step_num}")
        if exc is not None:
            mock.side_effect = exc
        return mock

    step_mocks = {n: make_step_mock(n) for n in range(FIRST_STEP, LAST_STEP + 1)}

    def fake_make_steps(pathogenic_only, from_scratch):
        labels = {
            1:  "build_gene_universe gene-list",
            2:  "fetch_variants gerasimavicius",
            3:  "fetch_variants clinvar",
            4:  "fetch_variants merge",
            5:  "fetch_sequences",
            6:  "fetch_annotations pfam",
            7:  "build_gene_universe universe",
            8:  "fetch_annotations uniprot",
            9:  "fetch_annotations enzyme",
            10: "build_proteome_features",
            11: "build_badonyi_features",
            12: "fetch_annotations alphamissense",
        }
        return [(n, labels[n], step_mocks[n]) for n in range(FIRST_STEP, LAST_STEP + 1)]

    def fake_save_state(state):
        saved_states.append(json.loads(json.dumps(state)))

    with (
        patch("sys.argv", ["run_pipeline"] + argv),
        patch(
            "esm2_mech.fetch_data.run_fetch_pipeline._make_steps",
            side_effect=fake_make_steps,
        ),
        patch(
            "esm2_mech.fetch_data.run_fetch_pipeline._save_state",
            side_effect=fake_save_state,
        ),
    ):
        from esm2_mech.fetch_data import run_fetch_pipeline

        try:
            run_fetch_pipeline.main()
        except SystemExit as exc:
            return step_mocks, saved_states, exc.code
    return step_mocks, saved_states, 0


# ---------------------------------------------------------------------------
# _load_state / _save_state
# ---------------------------------------------------------------------------


def test_load_state_missing_file(tmp_path):
    state_path = tmp_path / ".pipeline_state.json"
    with patch("esm2_mech.fetch_data.run_fetch_pipeline.STATE_FILE", state_path):
        assert _load_state() == {}


def test_save_and_load_state_roundtrip(tmp_path):
    state_path = tmp_path / ".pipeline_state.json"
    data = {"last_completed_step": 5, "last_completed_at": "2024-01-01T00:00:00"}
    with patch("esm2_mech.fetch_data.run_fetch_pipeline.STATE_FILE", state_path):
        _save_state(data)
        assert state_path.exists()
        loaded = _load_state()
    assert loaded == data


def test_save_state_is_atomic(tmp_path):
    state_path = tmp_path / ".pipeline_state.json"
    with patch("esm2_mech.fetch_data.run_fetch_pipeline.STATE_FILE", state_path):
        _save_state({"last_completed_step": 3})
    # tmp file must not remain
    assert not state_path.with_suffix(".json.tmp").exists()
    assert state_path.exists()


def test_load_state_corrupt_file(tmp_path):
    state_path = tmp_path / ".pipeline_state.json"
    state_path.write_text("{ not valid json")
    with patch("esm2_mech.fetch_data.run_fetch_pipeline.STATE_FILE", state_path):
        state = _load_state()
    assert state == {}


# ---------------------------------------------------------------------------
# Auto-resume logic
# ---------------------------------------------------------------------------


def test_auto_resume_no_prior_state(tmp_path):
    state_path = tmp_path / ".pipeline_state.json"
    with (
        patch("esm2_mech.fetch_data.run_fetch_pipeline.STATE_FILE", state_path),
        patch(
            "esm2_mech.fetch_data.run_fetch_pipeline._load_state", return_value={}
        ),
    ):
        mocks, saved, exit_code = _run_main([])
    assert exit_code == 0
    # all steps should have been called
    for n in range(FIRST_STEP, LAST_STEP + 1):
        mocks[n].assert_called_once()


def test_auto_resume_from_last_completed(tmp_path):
    state_path = tmp_path / ".pipeline_state.json"
    prior_state = {"last_completed_step": 5, "last_completed_at": "2024-01-01T00:00:00"}
    with (
        patch("esm2_mech.fetch_data.run_fetch_pipeline.STATE_FILE", state_path),
        patch(
            "esm2_mech.fetch_data.run_fetch_pipeline._load_state",
            return_value=prior_state,
        ),
    ):
        mocks, saved, exit_code = _run_main([])
    assert exit_code == 0
    for n in range(1, 6):
        mocks[n].assert_not_called()
    for n in range(6, LAST_STEP + 1):
        mocks[n].assert_called_once()


def test_auto_resume_all_done(tmp_path, capsys):
    prior_state = {
        "last_completed_step": LAST_STEP,
        "last_completed_at": "2024-01-01T00:00:00",
    }
    with patch(
        "esm2_mech.fetch_data.run_fetch_pipeline._load_state",
        return_value=prior_state,
    ):
        mocks, saved, exit_code = _run_main([])
    assert exit_code == 0
    for n in range(FIRST_STEP, LAST_STEP + 1):
        mocks[n].assert_not_called()
    captured = capsys.readouterr()
    assert "already completed" in captured.out


# ---------------------------------------------------------------------------
# --from-step override
# ---------------------------------------------------------------------------


def test_from_step_overrides_saved_state(tmp_path):
    prior_state = {"last_completed_step": 8}
    with patch(
        "esm2_mech.fetch_data.run_fetch_pipeline._load_state",
        return_value=prior_state,
    ):
        mocks, saved, exit_code = _run_main(["--from-step", "3"])
    assert exit_code == 0
    for n in range(1, 3):
        mocks[n].assert_not_called()
    for n in range(3, LAST_STEP + 1):
        mocks[n].assert_called_once()


def test_from_step_invalid_value():
    with patch(
        "esm2_mech.fetch_data.run_fetch_pipeline._load_state", return_value={}
    ):
        _, _, exit_code = _run_main(["--from-step", "99"])
    assert exit_code != 0


def test_from_step_boundary_first(tmp_path):
    with patch(
        "esm2_mech.fetch_data.run_fetch_pipeline._load_state", return_value={}
    ):
        mocks, saved, exit_code = _run_main(["--from-step", str(FIRST_STEP)])
    assert exit_code == 0
    for n in range(FIRST_STEP, LAST_STEP + 1):
        mocks[n].assert_called_once()


def test_from_step_boundary_last(tmp_path):
    with patch(
        "esm2_mech.fetch_data.run_fetch_pipeline._load_state", return_value={}
    ):
        mocks, saved, exit_code = _run_main(["--from-step", str(LAST_STEP)])
    assert exit_code == 0
    for n in range(FIRST_STEP, LAST_STEP):
        mocks[n].assert_not_called()
    mocks[LAST_STEP].assert_called_once()


# ---------------------------------------------------------------------------
# State saved after each successful step
# ---------------------------------------------------------------------------


def test_state_saved_after_each_step():
    with patch(
        "esm2_mech.fetch_data.run_fetch_pipeline._load_state", return_value={}
    ):
        mocks, saved, exit_code = _run_main(["--from-step", "1"])
    assert exit_code == 0
    assert len(saved) == LAST_STEP
    for idx, state in enumerate(saved):
        expected_step = idx + 1
        assert state["last_completed_step"] == expected_step
        assert f"step_{expected_step}_completed_at" in state


def test_state_not_updated_before_step_runs():
    # Only steps 1 and 2 run; step 3 fails — state should show last_completed = 2
    with patch(
        "esm2_mech.fetch_data.run_fetch_pipeline._load_state", return_value={}
    ):
        mocks, saved, exit_code = _run_main(
            ["--from-step", "1"],
            step_side_effects={3: RuntimeError("boom")},
        )
    assert exit_code == 1
    assert saved[-1]["last_completed_step"] == 2


# ---------------------------------------------------------------------------
# Failure behaviour
# ---------------------------------------------------------------------------


def test_pipeline_exits_on_step_failure():
    with patch(
        "esm2_mech.fetch_data.run_fetch_pipeline._load_state", return_value={}
    ):
        mocks, saved, exit_code = _run_main(
            ["--from-step", "1"],
            step_side_effects={5: ValueError("fetch failed")},
        )
    assert exit_code == 1


def test_steps_after_failure_not_called():
    with patch(
        "esm2_mech.fetch_data.run_fetch_pipeline._load_state", return_value={}
    ):
        mocks, saved, exit_code = _run_main(
            ["--from-step", "1"],
            step_side_effects={4: RuntimeError("bad merge")},
        )
    assert exit_code == 1
    for n in range(5, LAST_STEP + 1):
        mocks[n].assert_not_called()


def test_failed_step_state_not_saved():
    with patch(
        "esm2_mech.fetch_data.run_fetch_pipeline._load_state", return_value={}
    ):
        mocks, saved, exit_code = _run_main(
            ["--from-step", "1"],
            step_side_effects={6: RuntimeError("universe build failed")},
        )
    assert exit_code == 1
    completed_steps = [s["last_completed_step"] for s in saved]
    assert 6 not in completed_steps
    assert 5 in completed_steps


def test_rerun_after_failure_resumes_at_failed_step():
    # Simulate: steps 1-4 completed previously, step 5 failed last run.
    # State file shows last_completed = 4. Next run should start at 5.
    prior_state = {"last_completed_step": 4}
    with patch(
        "esm2_mech.fetch_data.run_fetch_pipeline._load_state",
        return_value=prior_state,
    ):
        mocks, saved, exit_code = _run_main([])
    assert exit_code == 0
    for n in range(1, 5):
        mocks[n].assert_not_called()
    for n in range(5, LAST_STEP + 1):
        mocks[n].assert_called_once()
