"""
Tests for the nested-block status reader.

Invariants:
- A block that failed keeps saying failed; it is never rewritten as unscorable,
  which would report a crash as a property of the data.
- A skipped block keeps saying skipped.
- A block that is absent never ran, which is a failure.
- A block declaring something that is not a seed status is refused rather than
  quietly treated as a success or a data problem.
"""

import pytest

from esm2_mech.utils.seed_aggregation import (
    SEED_STATUS_FAILED,
    SEED_STATUS_SKIPPED,
    SEED_STATUS_SUCCESS,
    SEED_STATUS_UNSCORABLE,
    block_seed_status,
)


def test_each_declared_status_passes_through_unchanged():
    for status in (
        SEED_STATUS_SUCCESS,
        SEED_STATUS_FAILED,
        SEED_STATUS_SKIPPED,
        SEED_STATUS_UNSCORABLE,
    ):
        assert block_seed_status({"status": status}) == status


def test_a_failure_is_not_reported_as_a_data_problem():
    assert block_seed_status({"status": SEED_STATUS_FAILED}) != SEED_STATUS_UNSCORABLE


def test_a_missing_block_is_a_failure():
    assert block_seed_status(None) == SEED_STATUS_FAILED


def test_a_block_without_a_status_is_refused():
    with pytest.raises(ValueError, match="is not a seed status"):
        block_seed_status({"macro_f1_mean": 0.4})


def test_an_unknown_status_string_is_refused():
    with pytest.raises(ValueError, match="is not a seed status"):
        block_seed_status({"status": "unavailable"})
