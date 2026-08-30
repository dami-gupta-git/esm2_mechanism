"""
Tests for the shared across-seed scalar core in esm2_mech.utils.seed_aggregation.

These tests define the contract before the implementation exists. They fail until
`aggregate_seed_values` and the two shared readers are written.

The core takes the requested seed identifiers and one value per seed, and returns
either a point estimate or an explicit refusal. It knows nothing about datasets,
splits, features, arms, regimes, or file layouts.

Invariants:
- spread is the SAMPLE standard deviation, not the population one
- a missing, failed, null, or non-finite value for a requested seed refuses the
  whole aggregate; survivors are never averaged
- seed identity comes from the mapping key; a duplicate, unexpected, or missing
  identifier refuses the aggregate
- refusal carries a fixed reason code and the seeds responsible, never a bare None
- fewer than three contributors yields a point estimate with a null spread
- every aggregate names its sampling unit, so a fold spread can never be read as
  a seed spread
- failed, skipped, and unscorable statuses remain distinct refusal reasons
"""

import math

import pytest

from esm2_mech.utils.seed_aggregation import (
    SEED_SAMPLING_UNIT,
    SEED_STATUS_FAILED,
    SeedUnavailableReason,
    aggregate_seed_results,
    aggregate_seed_values as aggregate_seed_records,
    make_seed_record,
    read_seed_inference,
    read_seed_point_estimate,
    seed_result_contract,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

FIVE_SEEDS = (0, 1, 2, 3, 4)

# mean 0.3; sample spread 0.158113...; population spread 0.141421...
FIVE_VALUES = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5}

SAMPLE_SPREAD_OF_FIVE_VALUES = 0.15811388300841897
POPULATION_SPREAD_OF_FIVE_VALUES = 0.1414213562373095


def aggregate_seed_values(requested_seeds, values):
    """Keep the test cases compact while exercising the required constructor."""
    if isinstance(values, dict):
        records = [make_seed_record(seed, value) for seed, value in values.items()]
    else:
        records = values
    return aggregate_seed_records(requested_seeds, records)


def _assert_refused(aggregate, reason, affected):
    """A refusal carries no number, the expected reason code, and the blamed seeds."""
    assert aggregate.available is False
    assert aggregate.mean is None
    assert aggregate.spread is None
    assert aggregate.reason is reason
    assert sorted(aggregate.affected_seeds) == sorted(affected)


def test_result_adapter_uses_declared_seed_identity_and_root_status():
    results = [
        {**seed_result_contract(0), "metric": 0.2},
        seed_result_contract(1, status=SEED_STATUS_FAILED),
    ]
    aggregate = aggregate_seed_results(
        (0, 1), results, lambda result: result["metric"]
    )
    _assert_refused(aggregate, SeedUnavailableReason.FAILED_SEED, [1])


# ---------------------------------------------------------------------------
# the spread formula
# ---------------------------------------------------------------------------


class TestSpreadFormula:

    def test_spread_is_the_sample_standard_deviation(self):
        """Pinned against a hand-calculation, since the population value passes
        every other assertion in this file."""
        aggregate = aggregate_seed_values(FIVE_SEEDS, FIVE_VALUES)
        assert aggregate.available is True
        assert aggregate.mean == pytest.approx(0.3)
        assert aggregate.spread == pytest.approx(SAMPLE_SPREAD_OF_FIVE_VALUES)

    def test_spread_is_not_the_population_standard_deviation(self):
        """The regression this file exists to catch: reverting to np.std()."""
        aggregate = aggregate_seed_values(FIVE_SEEDS, FIVE_VALUES)
        assert aggregate.spread != pytest.approx(POPULATION_SPREAD_OF_FIVE_VALUES)

    def test_three_contributors_use_n_minus_one(self):
        aggregate = aggregate_seed_values((0, 1, 2), {0: 0.1, 1: 0.2, 2: 0.3})
        assert aggregate.mean == pytest.approx(0.2)
        assert aggregate.spread == pytest.approx(0.1)

    def test_identical_values_give_zero_spread_not_a_refusal(self):
        """Zero spread is a real observation about the seeds, not a failure."""
        aggregate = aggregate_seed_values((0, 1, 2), {0: 0.4, 1: 0.4, 2: 0.4})
        assert aggregate.available is True
        assert aggregate.spread == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# survivor averaging
# ---------------------------------------------------------------------------


class TestNoSurvivorAveraging:

    def test_missing_seed_refuses_rather_than_averaging_the_rest(self):
        """The headline defect: four seeds must not be reported as a five-seed mean."""
        values = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
        aggregate = aggregate_seed_values(FIVE_SEEDS, values)
        _assert_refused(aggregate, SeedUnavailableReason.MISSING_SEED, [4])

    def test_none_value_refuses(self):
        values = {**FIVE_VALUES, 2: None}
        aggregate = aggregate_seed_values(FIVE_SEEDS, values)
        _assert_refused(aggregate, SeedUnavailableReason.INVALID_VALUE, [2])

    def test_nan_value_refuses(self):
        values = {**FIVE_VALUES, 3: float("nan")}
        aggregate = aggregate_seed_values(FIVE_SEEDS, values)
        _assert_refused(aggregate, SeedUnavailableReason.INVALID_VALUE, [3])

    @pytest.mark.parametrize("infinity", [float("inf"), float("-inf")])
    def test_infinite_value_refuses(self, infinity):
        """Infinity passes an isnan() guard, which is how it survives today."""
        values = {**FIVE_VALUES, 1: infinity}
        aggregate = aggregate_seed_values(FIVE_SEEDS, values)
        _assert_refused(aggregate, SeedUnavailableReason.INVALID_VALUE, [1])

    def test_every_bad_seed_is_named_not_just_the_first(self):
        values = {**FIVE_VALUES, 1: None, 3: float("nan")}
        aggregate = aggregate_seed_values(FIVE_SEEDS, values)
        assert sorted(aggregate.affected_seeds) == [1, 3]

    def test_refusal_reports_which_seeds_did_contribute(self):
        """A refusal still has to say what was found, so the run can be diagnosed."""
        values = {0: 0.1, 1: 0.2}
        aggregate = aggregate_seed_values(FIVE_SEEDS, values)
        assert sorted(aggregate.contributing_seeds) == [0, 1]
        assert sorted(aggregate.requested_seeds) == list(FIVE_SEEDS)

    @pytest.mark.parametrize(
        ("status", "reason"),
        [
            ("failed", SeedUnavailableReason.FAILED_SEED),
            ("skipped", SeedUnavailableReason.SKIPPED_SEED),
            ("unscorable", SeedUnavailableReason.UNSCORABLE_SEED),
        ],
    )
    def test_seed_status_refuses_with_its_own_reason(self, status, reason):
        records = [
            make_seed_record(
                seed,
                FIVE_VALUES[seed],
                status=status if seed == 2 else "success",
            )
            for seed in FIVE_SEEDS
        ]
        aggregate = aggregate_seed_values(FIVE_SEEDS, records)
        _assert_refused(aggregate, reason, [2])

    def test_mixed_failures_name_every_affected_seed(self):
        records = [
            make_seed_record(0, 0.1),
            make_seed_record(1, None, status="failed"),
            make_seed_record(2, None, status="skipped"),
        ]
        aggregate = aggregate_seed_records((0, 1, 2, 3), records)
        _assert_refused(
            aggregate,
            SeedUnavailableReason.MISSING_SEED,
            [1, 2, 3],
        )
        assert aggregate.contributing_seeds == (0,)


# ---------------------------------------------------------------------------
# seed identity
# ---------------------------------------------------------------------------


class TestSeedIdentity:

    def test_unexpected_seed_refuses(self):
        values = {**FIVE_VALUES, 9: 0.6}
        aggregate = aggregate_seed_values(FIVE_SEEDS, values)
        _assert_refused(aggregate, SeedUnavailableReason.UNEXPECTED_SEED, [9])

    def test_duplicate_requested_identifier_refuses(self):
        """A caller that asks for the same seed twice has a bug; averaging it
        would double-weight that seed."""
        aggregate = aggregate_seed_values((0, 1, 1, 2), {0: 0.1, 1: 0.2, 2: 0.3})
        _assert_refused(aggregate, SeedUnavailableReason.DUPLICATE_SEED, [1])

    def test_empty_requested_set_refuses(self):
        aggregate = aggregate_seed_values((), {})
        assert aggregate.available is False

    def test_record_for_empty_requested_set_is_unexpected(self):
        aggregate = aggregate_seed_values((), {4: 0.2})
        _assert_refused(aggregate, SeedUnavailableReason.UNEXPECTED_SEED, [4])

    def test_values_are_matched_by_identifier_not_by_order(self):
        """Two mappings with the same values in different insertion order must
        give the same answer, and a shifted mapping must not."""
        forward = aggregate_seed_values((0, 1, 2), {0: 0.1, 1: 0.2, 2: 0.9})
        shuffled = aggregate_seed_values((0, 1, 2), {2: 0.9, 0: 0.1, 1: 0.2})
        assert forward.mean == pytest.approx(shuffled.mean)
        assert forward.spread == pytest.approx(shuffled.spread)


# ---------------------------------------------------------------------------
# contributor count and spread suppression
# ---------------------------------------------------------------------------


class TestContributorCount:

    def test_two_contributors_give_a_mean_with_no_spread(self):
        """Two points cannot support a spread, but the mean is still real."""
        aggregate = aggregate_seed_values((0, 1), {0: 0.2, 1: 0.4})
        assert aggregate.available is True
        assert aggregate.mean == pytest.approx(0.3)
        assert aggregate.spread is None

    def test_one_contributor_gives_a_mean_with_no_spread(self):
        aggregate = aggregate_seed_values((0,), {0: 0.25})
        assert aggregate.available is True
        assert aggregate.mean == pytest.approx(0.25)
        assert aggregate.spread is None

    def test_contributor_count_is_recorded(self):
        aggregate = aggregate_seed_values(FIVE_SEEDS, FIVE_VALUES)
        assert sorted(aggregate.contributing_seeds) == list(FIVE_SEEDS)


# ---------------------------------------------------------------------------
# sampling unit
# ---------------------------------------------------------------------------


class TestSamplingUnit:

    def test_available_aggregate_names_its_sampling_unit(self):
        """Fold spread and seed spread are not interchangeable; the unit travels
        with the number so a figure cannot pick up the wrong one."""
        aggregate = aggregate_seed_values(FIVE_SEEDS, FIVE_VALUES)
        assert aggregate.sampling_unit == SEED_SAMPLING_UNIT

    def test_refused_aggregate_also_names_its_sampling_unit(self):
        aggregate = aggregate_seed_values(FIVE_SEEDS, {0: 0.1})
        assert aggregate.sampling_unit == SEED_SAMPLING_UNIT


# ---------------------------------------------------------------------------
# reason codes
# ---------------------------------------------------------------------------


class TestReasonCodes:

    def test_reason_is_a_code_not_a_sentence(self):
        """Consumers branch on the code; prose is for the report only."""
        aggregate = aggregate_seed_values(FIVE_SEEDS, {0: 0.1})
        assert isinstance(aggregate.reason, SeedUnavailableReason)

    def test_refusal_carries_a_human_readable_message(self):
        aggregate = aggregate_seed_values(FIVE_SEEDS, {0: 0.1})
        assert isinstance(aggregate.message, str)
        assert aggregate.message

    def test_available_aggregate_has_no_reason(self):
        aggregate = aggregate_seed_values(FIVE_SEEDS, FIVE_VALUES)
        assert aggregate.reason is None


# ---------------------------------------------------------------------------
# the core stays generic
# ---------------------------------------------------------------------------


class TestCoreIsShapeAgnostic:

    def test_core_accepts_a_plain_mapping_with_no_result_structure(self):
        """The core must not require split/feature/metric nesting; thirty-three
        producers do not have that shape."""
        aggregate = aggregate_seed_values(FIVE_SEEDS, FIVE_VALUES)
        assert aggregate.available is True

    def test_core_accepts_non_contiguous_seed_identifiers(self):
        """Requested seeds are an explicit set, not range(n)."""
        aggregate = aggregate_seed_values((7, 11, 13), {7: 0.1, 11: 0.2, 13: 0.3})
        assert aggregate.available is True
        assert aggregate.mean == pytest.approx(0.2)

    def test_integer_values_are_accepted_and_returned_as_floats(self):
        aggregate = aggregate_seed_values((0, 1, 2), {0: 1, 1: 2, 2: 3})
        assert isinstance(aggregate.mean, float)
        assert aggregate.mean == pytest.approx(2.0)

    def test_a_bool_is_not_a_metric_value(self):
        """bool is a subclass of int in Python; a True that reached a metric
        field is a bug upstream, not a 1.0."""
        with pytest.raises((TypeError, ValueError)):
            aggregate_seed_values((0, 1, 2), {0: 0.1, 1: True, 2: 0.3})

    def test_a_numeric_string_is_not_converted_to_a_metric_value(self):
        with pytest.raises(TypeError):
            make_seed_record(0, "0.5")

    def test_core_rejects_values_that_bypass_the_record_constructor(self):
        with pytest.raises(TypeError):
            aggregate_seed_records((0, 1, 2), {0: 0.1, 1: 0.2, 2: 0.3})


# ---------------------------------------------------------------------------
# regression guards on the old behaviour
# ---------------------------------------------------------------------------


class TestOldBehaviourIsGone:

    def test_reduced_count_is_never_reported_as_a_mean(self):
        """The old helper returned (mean_of_survivors, spread, reduced_count).
        No combination of inputs may produce a mean over fewer seeds than were
        requested without the aggregate being refused."""
        for dropped in FIVE_SEEDS:
            values = {seed: FIVE_VALUES[seed] for seed in FIVE_SEEDS if seed != dropped}
            aggregate = aggregate_seed_values(FIVE_SEEDS, values)
            assert aggregate.available is False, f"seed {dropped} was silently dropped"

    def test_empty_input_does_not_return_nan(self):
        """The old helper returned NaN for an empty list, which reads as a value."""
        aggregate = aggregate_seed_values(FIVE_SEEDS, {})
        assert aggregate.mean is None
        assert not (aggregate.mean is not None and math.isnan(aggregate.mean))


class TestSharedReaders:

    def test_point_reader_accepts_complete_current_schema_aggregate(self):
        aggregate = aggregate_seed_values(FIVE_SEEDS, FIVE_VALUES)
        result = read_seed_point_estimate(aggregate.to_dict())
        assert result.available is True
        assert result.value == pytest.approx(0.3)
        assert result.spread == pytest.approx(SAMPLE_SPREAD_OF_FIVE_VALUES)

    def test_point_reader_preserves_unavailable_reason(self):
        aggregate = aggregate_seed_values(FIVE_SEEDS, {0: 0.1})
        result = read_seed_point_estimate(aggregate.to_dict())
        assert result.available is False
        assert result.reason is SeedUnavailableReason.MISSING_SEED

    def test_reader_rejects_wrong_schema(self):
        stored = aggregate_seed_values(FIVE_SEEDS, FIVE_VALUES).to_dict()
        stored["schema_version"] += 1
        result = read_seed_point_estimate(stored)
        assert result.reason is SeedUnavailableReason.SCHEMA_MISMATCH

    def test_reader_rejects_boolean_schema_version(self):
        stored = aggregate_seed_values(FIVE_SEEDS, FIVE_VALUES).to_dict()
        stored["schema_version"] = True
        result = read_seed_point_estimate(stored)
        assert result.reason is SeedUnavailableReason.SCHEMA_MISMATCH

    def test_reader_rejects_wrong_sampling_unit(self):
        stored = aggregate_seed_values(FIVE_SEEDS, FIVE_VALUES).to_dict()
        stored["sampling_unit"] = "cross_validation_fold"
        result = read_seed_point_estimate(stored)
        assert result.reason is SeedUnavailableReason.SAMPLING_UNIT_MISMATCH

    def test_inference_reader_requires_three_seeds_and_a_spread(self):
        aggregate = aggregate_seed_values((0, 1), {0: 0.2, 1: 0.4})
        result = read_seed_inference(aggregate)
        assert result.available is False
        assert result.reason is SeedUnavailableReason.INSUFFICIENT_SEEDS

    @pytest.mark.parametrize("field", ["reason", "message", "affected_seeds"])
    def test_available_reader_rejects_unavailable_metadata(self, field):
        stored = aggregate_seed_values(FIVE_SEEDS, FIVE_VALUES).to_dict()
        replacements = {
            "reason": SeedUnavailableReason.FAILED_SEED.value,
            "message": "a requested seed failed",
            "affected_seeds": [2],
        }
        stored[field] = replacements[field]
        result = read_seed_point_estimate(stored)
        assert result.reason is SeedUnavailableReason.INVALID_AGGREGATE

    def test_reader_rejects_missing_spread_for_three_or_more_seeds(self):
        stored = aggregate_seed_values(FIVE_SEEDS, FIVE_VALUES).to_dict()
        stored["seed_std"] = None
        result = read_seed_point_estimate(stored)
        assert result.reason is SeedUnavailableReason.INVALID_AGGREGATE

    def test_reader_rejects_spread_for_fewer_than_three_seeds(self):
        stored = aggregate_seed_values((0, 1), {0: 0.2, 1: 0.4}).to_dict()
        stored["seed_std"] = 0.1
        result = read_seed_point_estimate(stored)
        assert result.reason is SeedUnavailableReason.INVALID_AGGREGATE

    def test_unavailable_reader_rejects_a_stored_mean(self):
        stored = aggregate_seed_values(FIVE_SEEDS, {0: 0.1}).to_dict()
        stored["mean"] = 0.1
        result = read_seed_point_estimate(stored)
        assert result.reason is SeedUnavailableReason.INVALID_AGGREGATE

    def test_empty_request_refusal_remains_readable(self):
        aggregate = aggregate_seed_values((), {})
        result = read_seed_point_estimate(aggregate)
        assert result.reason is SeedUnavailableReason.EMPTY_REQUESTED_SEEDS

    def test_empty_request_reason_is_invalid_for_nonempty_request(self):
        stored = aggregate_seed_values(FIVE_SEEDS, {0: 0.1}).to_dict()
        stored["reason"] = SeedUnavailableReason.EMPTY_REQUESTED_SEEDS.value
        stored["affected_seeds"] = []
        result = read_seed_point_estimate(stored)
        assert result.reason is SeedUnavailableReason.INVALID_AGGREGATE

    def test_reader_rejects_a_reader_only_reason_as_stored_state(self):
        stored = aggregate_seed_values(FIVE_SEEDS, {0: 0.1}).to_dict()
        stored["reason"] = SeedUnavailableReason.SCHEMA_MISMATCH.value
        result = read_seed_point_estimate(stored)
        assert result.reason is SeedUnavailableReason.INVALID_AGGREGATE
