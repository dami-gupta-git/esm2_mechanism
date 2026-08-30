"""Shared validation, aggregation, and reading for model-seed results.

The scalar core requires one explicit record for every requested seed. Experiment
modules traverse their own result structures and call these reducers; this module
holds no experiment-specific layout.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from esm2_mech.utils.constants import SEED_AGGREGATION_SCHEMA_VERSION

SEED_SAMPLING_UNIT = "model_seed"

# Root keys every per-seed result file declares, alongside its "seed".
SEED_SCHEMA_KEY = "seed_schema_version"
SEED_STATUS_KEY = "seed_status"

SEED_STATUS_SUCCESS = "success"
SEED_STATUS_FAILED = "failed"
SEED_STATUS_SKIPPED = "skipped"
SEED_STATUS_UNSCORABLE = "unscorable"
SEED_STATUSES = frozenset(
    {
        SEED_STATUS_SUCCESS,
        SEED_STATUS_FAILED,
        SEED_STATUS_SKIPPED,
        SEED_STATUS_UNSCORABLE,
    }
)


class SeedUnavailableReason(str, Enum):
    DUPLICATE_SEED = "duplicate_seed"
    UNEXPECTED_SEED = "unexpected_seed"
    MISSING_SEED = "missing_seed"
    FAILED_SEED = "failed_seed"
    SKIPPED_SEED = "skipped_seed"
    UNSCORABLE_SEED = "unscorable_seed"
    INVALID_VALUE = "invalid_value"
    SCHEMA_MISMATCH = "schema_mismatch"
    SAMPLING_UNIT_MISMATCH = "sampling_unit_mismatch"
    INVALID_AGGREGATE = "invalid_aggregate"
    INSUFFICIENT_SEEDS = "insufficient_seeds"
    INVALID_ROW_SET = "invalid_row_set"
    METADATA_MISMATCH = "metadata_mismatch"
    CLASS_ORDER_MISMATCH = "class_order_mismatch"
    INVALID_SHAPE = "invalid_shape"
    ZERO_SUPPORT = "zero_support"


_REASON_MESSAGES = {
    SeedUnavailableReason.DUPLICATE_SEED: "a seed identifier is duplicated",
    SeedUnavailableReason.UNEXPECTED_SEED: "an unrequested seed record is present",
    SeedUnavailableReason.MISSING_SEED: "a requested seed record is missing",
    SeedUnavailableReason.FAILED_SEED: "a requested seed failed",
    SeedUnavailableReason.SKIPPED_SEED: "a requested seed was skipped",
    SeedUnavailableReason.UNSCORABLE_SEED: "a requested seed was unscorable",
    SeedUnavailableReason.INVALID_VALUE: "a requested seed has no finite metric value",
    SeedUnavailableReason.SCHEMA_MISMATCH: "the seed aggregate schema version does not match",
    SeedUnavailableReason.SAMPLING_UNIT_MISMATCH: "the aggregate has the wrong sampling unit",
    SeedUnavailableReason.INVALID_AGGREGATE: "the stored seed aggregate is internally inconsistent",
    SeedUnavailableReason.INSUFFICIENT_SEEDS: "at least three requested seeds are required for inference",
    SeedUnavailableReason.INVALID_ROW_SET: "a requested seed does not cover the declared row set",
    SeedUnavailableReason.METADATA_MISMATCH: "seed records disagree with declared metadata",
    SeedUnavailableReason.CLASS_ORDER_MISMATCH: "a seed record has the wrong class order",
    SeedUnavailableReason.INVALID_SHAPE: "a seed payload has an invalid shape or value",
    SeedUnavailableReason.ZERO_SUPPORT: "a confusion-matrix row has zero observed support",
}

_STORED_UNAVAILABLE_REASONS = frozenset(
    {
        SeedUnavailableReason.DUPLICATE_SEED,
        SeedUnavailableReason.UNEXPECTED_SEED,
        SeedUnavailableReason.MISSING_SEED,
        SeedUnavailableReason.FAILED_SEED,
        SeedUnavailableReason.SKIPPED_SEED,
        SeedUnavailableReason.UNSCORABLE_SEED,
        SeedUnavailableReason.INVALID_VALUE,
    }
)

def _check_seed(seed) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed identifiers must be integers")


@dataclass(frozen=True)
class SeedValueRecord:
    seed: int
    status: str
    value: float | None

    def __post_init__(self):
        """Checked here so a record is validated once, however it was built."""
        _check_seed(self.seed)
        if self.status not in SEED_STATUSES:
            raise ValueError(f"unsupported seed status {self.status!r}")
        if isinstance(self.value, bool):
            raise TypeError("a boolean is not a scientific metric value")
        if self.value is not None and not isinstance(self.value, Real):
            raise TypeError("seed metric values must be numeric or None")
        if self.value is not None:
            object.__setattr__(self, "value", float(self.value))


@dataclass(frozen=True)
class SeedPayloadRecord:
    seed: int
    status: str
    payload: Any


@dataclass(frozen=True)
class SeedPayloadAggregate:
    state: str
    reason: SeedUnavailableReason | None
    requested_seeds: tuple[int, ...]
    contributing_seeds: tuple[int, ...]
    affected_seeds: tuple[int, ...]
    payload: Any
    sampling_unit: str
    message: str | None
    schema_version: int = SEED_AGGREGATION_SCHEMA_VERSION

    @property
    def available(self) -> bool:
        return self.state == "available"

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "state": self.state,
            "reason": None if self.reason is None else self.reason.value,
            "requested_seeds": list(self.requested_seeds),
            "contributing_seeds": list(self.contributing_seeds),
            "affected_seeds": list(self.affected_seeds),
            "payload": self.payload,
            "sampling_unit": self.sampling_unit,
            "message": self.message,
        }


@dataclass(frozen=True)
class SeedAggregate:
    state: str
    reason: SeedUnavailableReason | None
    requested_seeds: tuple[int, ...]
    contributing_seeds: tuple[int, ...]
    affected_seeds: tuple[int, ...]
    mean: float | None
    spread: float | None
    sampling_unit: str
    message: str | None
    schema_version: int = SEED_AGGREGATION_SCHEMA_VERSION

    @property
    def available(self) -> bool:
        return self.state == "available"

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "state": self.state,
            "reason": None if self.reason is None else self.reason.value,
            "requested_seeds": list(self.requested_seeds),
            "contributing_seeds": list(self.contributing_seeds),
            "affected_seeds": list(self.affected_seeds),
            "mean": self.mean,
            "seed_std": self.spread,
            "sampling_unit": self.sampling_unit,
            "message": self.message,
        }


def seed_count(value: str) -> int:
    """argparse type for `--seeds`, which is a count of seeds 0..n-1."""
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("--seeds must be >= 1")
    return count


@dataclass(frozen=True)
class SeedMetricRead:
    value: float | None
    spread: float | None
    reason: SeedUnavailableReason | None
    message: str | None

    @property
    def available(self) -> bool:
        return self.value is not None and self.reason is None


def make_seed_record(
    seed: int,
    value: float | int | None,
    *,
    status: str = SEED_STATUS_SUCCESS,
) -> SeedValueRecord:
    """Construct one seed record without changing a missing scientific value."""
    return SeedValueRecord(seed=seed, status=status, value=value)


def seed_result_contract(seed: int, *, status: str = SEED_STATUS_SUCCESS) -> dict:
    """Root fields every per-seed result file declares for the shared contract.

    A per-seed file states its own seed and status once, at the root, so an
    aggregator reads what the run declared rather than inferring a seed's fate
    from whichever inner block it happens to look at first.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed identifiers must be integers")
    if status not in SEED_STATUSES:
        raise ValueError(f"unsupported seed status {status!r}")
    return {
        SEED_SCHEMA_KEY: SEED_AGGREGATION_SCHEMA_VERSION,
        "seed": seed,
        SEED_STATUS_KEY: status,
    }


def aggregate_result_contract() -> dict:
    """Root field every across-seed aggregate result file declares."""
    return {SEED_SCHEMA_KEY: SEED_AGGREGATION_SCHEMA_VERSION}


def read_seed_result_contract(seed: int, source: str, result: Mapping) -> str:
    """Return the declared root status of one per-seed result file.

    Raises where the readers below return unavailable: a bad per-seed file means
    this run is wrong, while a bad aggregate may be another experiment's and must
    suppress only the dependent output.
    """
    version = result.get(SEED_SCHEMA_KEY)
    if version != SEED_AGGREGATION_SCHEMA_VERSION:
        raise ValueError(
            f"{source}: seed schema version {version!r} does not match the "
            f"expected {SEED_AGGREGATION_SCHEMA_VERSION}"
        )
    declared_seed = result.get("seed")
    if declared_seed != seed:
        raise ValueError(
            f"{source}: declares seed {declared_seed!r} but was loaded as seed {seed}"
        )
    status = result.get(SEED_STATUS_KEY)
    if status not in SEED_STATUSES:
        raise ValueError(f"{source}: root seed status {status!r} is not a seed status")
    return status


def block_seed_status(block: Any) -> str:
    """The status a nested result block declares, preserving how it ended.

    A block that broke declares 'failed' and a block whose data could not support
    the metric declares 'unscorable'. Rewriting the first as the second reports a
    crashed arm as a property of the data, so the declared status passes through
    unchanged. A block that is absent entirely never ran, which is a failure.
    """
    if not isinstance(block, dict):
        return SEED_STATUS_FAILED
    status = block.get("status")
    if status not in SEED_STATUSES:
        raise ValueError(f"result block status {status!r} is not a seed status")
    return status


def make_seed_payload_record(
    seed: int,
    payload: Any,
    *,
    status: str = SEED_STATUS_SUCCESS,
) -> SeedPayloadRecord:
    """Construct one non-scalar seed record under the shared seed contract."""
    make_seed_record(seed, None, status=status)
    return SeedPayloadRecord(seed=seed, status=status, payload=payload)


def _payload_unavailable(
    reason: SeedUnavailableReason,
    requested_seeds: Iterable[int],
    contributing_seeds: Iterable[int],
    affected_seeds: Iterable[int],
) -> SeedPayloadAggregate:
    return SeedPayloadAggregate(
        state="unavailable",
        reason=reason,
        requested_seeds=tuple(requested_seeds),
        contributing_seeds=tuple(contributing_seeds),
        affected_seeds=tuple(affected_seeds),
        payload=None,
        sampling_unit=SEED_SAMPLING_UNIT,
        message=_REASON_MESSAGES[reason],
    )


def _payload_available(
    requested_seeds: Iterable[int], payload: Any
) -> SeedPayloadAggregate:
    requested = tuple(requested_seeds)
    return SeedPayloadAggregate(
        state="available",
        reason=None,
        requested_seeds=requested,
        contributing_seeds=requested,
        affected_seeds=(),
        payload=payload,
        sampling_unit=SEED_SAMPLING_UNIT,
        message=None,
    )


def _validate_payload_seed_contract(
    requested_seeds: Iterable[int],
    records: Iterable[SeedPayloadRecord],
) -> tuple[tuple[int, ...], dict[int, SeedPayloadRecord], SeedPayloadAggregate | None]:
    """Apply the scalar core's identity and status rules to payload records."""
    requested = tuple(requested_seeds)
    payload_records = tuple(records)
    for record in payload_records:
        if not isinstance(record, SeedPayloadRecord):
            raise TypeError(
                "payload records must be created with make_seed_payload_record"
            )
    identity = aggregate_seed_values(
        requested,
        [
            make_seed_record(
                record.seed,
                0.0 if record.status == SEED_STATUS_SUCCESS else None,
                status=record.status,
            )
            for record in payload_records
        ],
    )
    if not identity.available:
        return (
            requested,
            {},
            _payload_unavailable(
                identity.reason,
                identity.requested_seeds,
                identity.contributing_seeds,
                identity.affected_seeds,
            ),
        )
    return requested, {record.seed: record for record in payload_records}, None


def _unavailable(
    reason: SeedUnavailableReason,
    requested_seeds: Iterable[int],
    contributing_seeds: Iterable[int],
    affected_seeds: Iterable[int],
) -> SeedAggregate:
    return SeedAggregate(
        state="unavailable",
        reason=reason,
        requested_seeds=tuple(requested_seeds),
        contributing_seeds=tuple(contributing_seeds),
        affected_seeds=tuple(affected_seeds),
        mean=None,
        spread=None,
        sampling_unit=SEED_SAMPLING_UNIT,
        message=_REASON_MESSAGES[reason],
    )


def aggregate_seed_values(
    requested_seeds: Iterable[int],
    values: Iterable[SeedValueRecord],
) -> SeedAggregate:
    """Aggregate one finite point estimate from every explicitly requested seed."""
    requested = tuple(requested_seeds)
    for seed in requested:
        _check_seed(seed)
    if not requested:
        raise ValueError("at least one seed must be requested")

    duplicate_requested = sorted(
        {seed for seed in requested if requested.count(seed) > 1}
    )
    if duplicate_requested:
        return _unavailable(
            SeedUnavailableReason.DUPLICATE_SEED,
            requested,
            (),
            duplicate_requested,
        )

    records = tuple(values)
    for record in records:
        if not isinstance(record, SeedValueRecord):
            raise TypeError("seed records must be created with make_seed_record")

    record_seeds = [record.seed for record in records]
    duplicate_records = sorted(
        {seed for seed in record_seeds if record_seeds.count(seed) > 1}
    )
    if duplicate_records:
        return _unavailable(
            SeedUnavailableReason.DUPLICATE_SEED,
            requested,
            (),
            duplicate_records,
        )

    records_by_seed = {record.seed: record for record in records}
    requested_set = set(requested)
    unexpected = sorted(set(record_seeds) - requested_set)
    missing = [seed for seed in requested if seed not in records_by_seed]
    contributing = [
        seed
        for seed in requested
        if seed in records_by_seed
        and records_by_seed[seed].status == SEED_STATUS_SUCCESS
        and records_by_seed[seed].value is not None
        and math.isfinite(records_by_seed[seed].value)
    ]

    status_reasons = (
        (SEED_STATUS_FAILED, SeedUnavailableReason.FAILED_SEED),
        (SEED_STATUS_SKIPPED, SeedUnavailableReason.SKIPPED_SEED),
        (SEED_STATUS_UNSCORABLE, SeedUnavailableReason.UNSCORABLE_SEED),
    )
    affected_by_status = {}
    for status, reason in status_reasons:
        affected_by_status[reason] = [
            seed
            for seed in requested
            if seed in records_by_seed and records_by_seed[seed].status == status
        ]

    invalid = [
        seed
        for seed in requested
        if seed in records_by_seed
        and records_by_seed[seed].status == SEED_STATUS_SUCCESS
        and (
            records_by_seed[seed].value is None
            or not math.isfinite(records_by_seed[seed].value)
        )
    ]
    affected = sorted(
        set(unexpected)
        | set(missing)
        | set(invalid)
        | {
            seed
            for status_affected in affected_by_status.values()
            for seed in status_affected
        }
    )
    reason = None
    if unexpected:
        reason = SeedUnavailableReason.UNEXPECTED_SEED
    elif missing:
        reason = SeedUnavailableReason.MISSING_SEED
    else:
        for status_reason in (
            SeedUnavailableReason.FAILED_SEED,
            SeedUnavailableReason.SKIPPED_SEED,
            SeedUnavailableReason.UNSCORABLE_SEED,
        ):
            if affected_by_status[status_reason]:
                reason = status_reason
                break
    if reason is None and invalid:
        reason = SeedUnavailableReason.INVALID_VALUE
    if reason is not None:
        return _unavailable(reason, requested, contributing, affected)

    numeric_values = [float(records_by_seed[seed].value) for seed in requested]
    spread = float(np.std(numeric_values, ddof=1)) if len(numeric_values) >= 3 else None
    return SeedAggregate(
        state="available",
        reason=None,
        requested_seeds=requested,
        contributing_seeds=requested,
        affected_seeds=(),
        mean=float(np.mean(numeric_values)),
        spread=spread,
        sampling_unit=SEED_SAMPLING_UNIT,
        message=None,
    )


def aggregate_seed_results(
    requested_seeds: Iterable[int],
    results: Iterable[Mapping],
    value: Callable[[Mapping], float | int | None],
    *,
    status: Callable[[Mapping], str] | None = None,
) -> SeedAggregate:
    """Aggregate one value from each current-schema per-seed result.

    Experiment modules supply only their local value and optional nested-status
    lookups. Seed identity, root status, validation, and arithmetic stay here.
    """
    records = []
    for result in results:
        seed = result.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("a per-seed result has no integer seed identifier")
        root_status = read_seed_result_contract(seed, f"seed {seed}", result)
        metric_status = root_status
        if root_status == SEED_STATUS_SUCCESS and status is not None:
            metric_status = status(result)
        metric_value = value(result) if metric_status == SEED_STATUS_SUCCESS else None
        records.append(make_seed_record(seed, metric_value, status=metric_status))
    return aggregate_seed_values(requested_seeds, records)


def aggregate_seed_vote(
    requested_seeds: Iterable[int],
    values: Iterable[SeedValueRecord],
    *,
    threshold: float,
    minimum_supporting_seeds: int,
    comparison: str = "less_than",
) -> SeedPayloadAggregate:
    """Apply one complete-seed threshold voting rule."""
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise TypeError("vote threshold must be numeric")
    if not math.isfinite(threshold):
        raise ValueError("vote threshold must be finite")
    if comparison not in {"less_than", "greater_than"}:
        raise ValueError("comparison must be 'less_than' or 'greater_than'")
    requested = tuple(requested_seeds)
    if (
        isinstance(minimum_supporting_seeds, bool)
        or not isinstance(minimum_supporting_seeds, int)
        or minimum_supporting_seeds < 1
        or minimum_supporting_seeds > len(requested)
    ):
        raise ValueError(
            "minimum_supporting_seeds must be between one and the requested seed count"
        )
    records = tuple(values)
    scalar_contract = aggregate_seed_values(requested, records)
    if not scalar_contract.available:
        return _payload_unavailable(
            scalar_contract.reason,
            scalar_contract.requested_seeds,
            scalar_contract.contributing_seeds,
            scalar_contract.affected_seeds,
        )
    by_seed = {record.seed: float(record.value) for record in records}
    if comparison == "less_than":
        supporting = [seed for seed in requested if by_seed[seed] < threshold]
    else:
        supporting = [seed for seed in requested if by_seed[seed] > threshold]
    return _payload_available(
        requested,
        {
            "decision": len(supporting) >= minimum_supporting_seeds,
            "threshold": float(threshold),
            "comparison": comparison,
            "minimum_supporting_seeds": minimum_supporting_seeds,
            "supporting_seeds": supporting,
            "n_supporting_seeds": len(supporting),
            "values_by_seed": {seed: by_seed[seed] for seed in requested},
        },
    )


def aggregate_paired_seed_difference(
    requested_seeds: Iterable[int],
    arm_a: Iterable[SeedValueRecord],
    arm_b: Iterable[SeedValueRecord],
) -> SeedAggregate:
    """Subtract arm B from arm A within seed, then aggregate the differences."""
    requested = tuple(requested_seeds)
    records_a = tuple(arm_a)
    records_b = tuple(arm_b)
    aggregate_a = aggregate_seed_values(requested, records_a)
    aggregate_b = aggregate_seed_values(requested, records_b)
    if not aggregate_a.available or not aggregate_b.available:
        failures = [
            aggregate
            for aggregate in (aggregate_a, aggregate_b)
            if not aggregate.available
        ]
        reason = failures[0].reason
        assert reason is not None
        affected = sorted(
            {seed for aggregate in failures for seed in aggregate.affected_seeds}
        )
        contributing = [
            seed
            for seed in requested
            if all(seed in aggregate.contributing_seeds for aggregate in failures)
        ]
        return _unavailable(reason, requested, contributing, affected)
    values_a = {record.seed: float(record.value) for record in records_a}
    values_b = {record.seed: float(record.value) for record in records_b}
    differences = [
        make_seed_record(seed, values_a[seed] - values_b[seed]) for seed in requested
    ]
    return aggregate_seed_values(requested, differences)


def aggregate_seed_confusion_matrices(
    requested_seeds: Iterable[int],
    class_order: Iterable,
    records: Iterable[SeedPayloadRecord],
) -> SeedPayloadAggregate:
    """Aggregate complete per-seed confusion matrices under a declared class order."""
    requested, by_seed, failure = _validate_payload_seed_contract(
        requested_seeds, records
    )
    if failure is not None:
        return failure
    declared_classes = tuple(class_order)
    if not declared_classes or len(set(declared_classes)) != len(declared_classes):
        raise ValueError("class_order must contain unique declared classes")

    raw_by_seed = {}
    normalized_by_seed = {}
    normalized = []
    defects = {
        SeedUnavailableReason.CLASS_ORDER_MISMATCH: [],
        SeedUnavailableReason.INVALID_SHAPE: [],
        SeedUnavailableReason.ZERO_SUPPORT: [],
    }
    expected_shape = (len(declared_classes), len(declared_classes))
    for seed in requested:
        payload = by_seed[seed].payload
        if not isinstance(payload, Mapping):
            defects[SeedUnavailableReason.INVALID_SHAPE].append(seed)
            continue
        seed_order = payload.get("class_order")
        if (
            not isinstance(seed_order, (list, tuple))
            or tuple(seed_order) != declared_classes
        ):
            defects[SeedUnavailableReason.CLASS_ORDER_MISMATCH].append(seed)
            continue
        matrix = np.asarray(payload.get("matrix"))
        if (
            matrix.shape != expected_shape
            or not np.issubdtype(matrix.dtype, np.number)
            or not np.isfinite(matrix).all()
            or np.any(matrix < 0)
            or np.any(matrix != np.floor(matrix))
        ):
            defects[SeedUnavailableReason.INVALID_SHAPE].append(seed)
            continue
        row_totals = matrix.sum(axis=1, keepdims=True)
        if np.any(row_totals == 0):
            defects[SeedUnavailableReason.ZERO_SUPPORT].append(seed)
            continue
        matrix_float = matrix.astype(float)
        seed_normalized = matrix_float / row_totals
        raw_by_seed[seed] = matrix.tolist()
        normalized_by_seed[seed] = seed_normalized.tolist()
        normalized.append(seed_normalized)

    affected = sorted({seed for seeds in defects.values() for seed in seeds})
    if affected:
        reason = next(reason for reason, seeds in defects.items() if seeds)
        contributing = [seed for seed in requested if seed not in affected]
        return _payload_unavailable(reason, requested, contributing, affected)

    pooled_raw = np.sum(
        [np.asarray(raw_by_seed[seed], dtype=float) for seed in requested], axis=0
    )
    return _payload_available(
        requested,
        {
            "class_order": list(declared_classes),
            "raw_by_seed": raw_by_seed,
            "normalized_by_seed": normalized_by_seed,
            "pooled_raw": pooled_raw.tolist(),
            "normalized_seed_mean": np.mean(normalized, axis=0).tolist(),
        },
    )


def aggregate_seed_oof(
    requested_seeds: Iterable[int],
    declared_row_ids: Iterable[int],
    declared_labels: Iterable,
    declared_clusters: Iterable,
    class_order: Iterable,
    declared_fold_ids: Iterable[int],
    records: Iterable[SeedPayloadRecord],
) -> SeedPayloadAggregate:
    """Align complete OOF predictions without dropping seeds or intersecting rows."""
    requested, by_seed, failure = _validate_payload_seed_contract(
        requested_seeds, records
    )
    if failure is not None:
        return failure

    row_ids = np.asarray(tuple(declared_row_ids))
    labels = np.asarray(tuple(declared_labels))
    clusters = np.asarray(tuple(declared_clusters), dtype=object)
    classes = tuple(class_order)
    fold_ids = tuple(declared_fold_ids)
    if (
        row_ids.ndim != 1
        or labels.ndim != 1
        or clusters.ndim != 1
        or len(row_ids) == 0
        or not (len(row_ids) == len(labels) == len(clusters))
        or not np.issubdtype(row_ids.dtype, np.integer)
        or len(np.unique(row_ids)) != len(row_ids)
        or not classes
        or len(set(classes)) != len(classes)
        or not set(labels.tolist()).issubset(set(classes))
        or not fold_ids
        or any(isinstance(fold, bool) or not isinstance(fold, int) for fold in fold_ids)
        or len(set(fold_ids)) != len(fold_ids)
    ):
        raise ValueError(
            "declared OOF rows, labels, clusters, classes, or folds are invalid"
        )

    seed_payloads = {}
    defects = {
        SeedUnavailableReason.INVALID_ROW_SET: [],
        SeedUnavailableReason.CLASS_ORDER_MISMATCH: [],
        SeedUnavailableReason.INVALID_SHAPE: [],
        SeedUnavailableReason.METADATA_MISMATCH: [],
    }
    for seed in requested:
        payload = by_seed[seed].payload
        if not isinstance(payload, Mapping):
            defects[SeedUnavailableReason.INVALID_SHAPE].append(seed)
            continue
        seed_rows = np.asarray(payload.get("row_ids"))
        if seed_rows.ndim != 1 or len(np.unique(seed_rows)) != len(seed_rows):
            defects[SeedUnavailableReason.INVALID_ROW_SET].append(seed)
            continue
        if set(seed_rows.tolist()) != set(row_ids.tolist()):
            defects[SeedUnavailableReason.INVALID_ROW_SET].append(seed)
            continue
        seed_order = payload.get("classes")
        if not isinstance(seed_order, (list, tuple)) or tuple(seed_order) != classes:
            defects[SeedUnavailableReason.CLASS_ORDER_MISMATCH].append(seed)
            continue
        seed_position = {
            row_id: position for position, row_id in enumerate(seed_rows.tolist())
        }
        order = np.array(
            [seed_position[row_id] for row_id in row_ids.tolist()], dtype=int
        )
        seed_labels = np.asarray(payload.get("y_true"))
        seed_clusters = np.asarray(payload.get("genes"), dtype=object)
        probabilities = np.asarray(payload.get("proba"))
        folds = np.asarray(payload.get("folds"))
        if (
            seed_labels.shape != labels.shape
            or seed_clusters.shape != clusters.shape
            or probabilities.shape != (len(row_ids), len(classes))
            or folds.shape != row_ids.shape
            or not np.issubdtype(probabilities.dtype, np.number)
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1.0)
            or not np.issubdtype(folds.dtype, np.integer)
            or set(folds.tolist()) != set(fold_ids)
        ):
            defects[SeedUnavailableReason.INVALID_SHAPE].append(seed)
            continue
        if not np.array_equal(seed_labels[order], labels) or not np.array_equal(
            seed_clusters[order], clusters
        ):
            defects[SeedUnavailableReason.METADATA_MISMATCH].append(seed)
            continue
        seed_payloads[seed] = {
            "seed": seed,
            "proba": probabilities[order],
            "folds": folds[order],
        }

    affected = sorted({seed for seeds in defects.values() for seed in seeds})
    if affected:
        reason = next(reason for reason, seeds in defects.items() if seeds)
        contributing = [seed for seed in requested if seed not in affected]
        return _payload_unavailable(reason, requested, contributing, affected)

    return _payload_available(
        requested,
        {
            "requested_seeds": list(requested),
            "row_ids": row_ids,
            "y_true": labels,
            "genes": clusters,
            "classes": list(classes),
            "oof_by_seed": seed_payloads,
        },
    )


def aggregate_oof_dicts(
    requested_seeds: Iterable[int],
    oof_by_seed: Mapping[int, dict | None],
    statuses_by_seed: Mapping[int, str],
    *,
    declared_row_ids: Iterable[int],
    declared_labels: Iterable,
    declared_clusters: Iterable,
    class_order: Iterable,
    declared_fold_ids: Iterable[int],
) -> SeedPayloadAggregate:
    """Adapt probe OOF dictionaries to the strict shared OOF reducer.

    The caller declares each seed's status. A seed whose probe crashed and a seed
    whose data could not support the metric both arrive with no predictions, and
    only the producer knows which happened, so reading the status off the missing
    predictions would record every crash as a property of the data.
    """
    requested = tuple(requested_seeds)
    if set(statuses_by_seed) != set(oof_by_seed):
        raise ValueError(
            "every seed of out-of-fold predictions must declare a status: "
            f"predictions for {sorted(oof_by_seed)}, "
            f"statuses for {sorted(statuses_by_seed)}"
        )
    records = [
        make_seed_payload_record(seed, oof, status=statuses_by_seed[seed])
        for seed, oof in oof_by_seed.items()
    ]
    return aggregate_seed_oof(
        requested,
        declared_row_ids,
        declared_labels,
        declared_clusters,
        class_order,
        declared_fold_ids,
        records,
    )


def _read_failure(reason: SeedUnavailableReason) -> SeedMetricRead:
    return SeedMetricRead(
        value=None,
        spread=None,
        reason=reason,
        message=_REASON_MESSAGES[reason],
    )


def _aggregate_fields(
    aggregate: SeedAggregate | Mapping,
) -> tuple[Mapping, SeedUnavailableReason | None]:
    if isinstance(aggregate, SeedAggregate):
        return aggregate.to_dict(), None
    if not isinstance(aggregate, Mapping):
        return {}, SeedUnavailableReason.INVALID_AGGREGATE
    schema_version = aggregate.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SEED_AGGREGATION_SCHEMA_VERSION
    ):
        return aggregate, SeedUnavailableReason.SCHEMA_MISMATCH
    return aggregate, None


def _seed_id_list(value) -> list[int] | None:
    if not isinstance(value, (list, tuple)):
        return None
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in value):
        return None
    if len(set(value)) != len(value):
        return None
    return list(value)


def read_seed_point_estimate(
    aggregate: SeedAggregate | Mapping,
    *,
    expected_sampling_unit: str = SEED_SAMPLING_UNIT,
) -> SeedMetricRead:
    """Read a complete current-schema seed mean without inventing a fallback."""
    fields, failure = _aggregate_fields(aggregate)
    if failure is not None:
        return _read_failure(failure)
    if fields.get("sampling_unit") != expected_sampling_unit:
        return _read_failure(SeedUnavailableReason.SAMPLING_UNIT_MISMATCH)

    requested = _seed_id_list(fields.get("requested_seeds"))
    contributing = _seed_id_list(fields.get("contributing_seeds"))
    affected = _seed_id_list(fields.get("affected_seeds"))
    if requested is None or contributing is None or affected is None:
        return _read_failure(SeedUnavailableReason.INVALID_AGGREGATE)

    state = fields.get("state")
    if state == "unavailable":
        try:
            reason = SeedUnavailableReason(fields.get("reason"))
        except (TypeError, ValueError):
            return _read_failure(SeedUnavailableReason.INVALID_AGGREGATE)
        if reason not in _STORED_UNAVAILABLE_REASONS or fields.get("mean") is not None:
            return _read_failure(SeedUnavailableReason.INVALID_AGGREGATE)
        message = fields.get("message")
        return SeedMetricRead(
            value=None,
            spread=None,
            reason=reason,
            message=message if isinstance(message, str) else _REASON_MESSAGES[reason],
        )
    if state != "available":
        return _read_failure(SeedUnavailableReason.INVALID_AGGREGATE)

    if (
        not requested
        or contributing != requested
        or affected
        or fields.get("reason") is not None
        or fields.get("message") is not None
    ):
        return _read_failure(SeedUnavailableReason.INVALID_AGGREGATE)

    mean = fields.get("mean")
    if isinstance(mean, bool) or not isinstance(mean, Real) or not math.isfinite(mean):
        return _read_failure(SeedUnavailableReason.INVALID_AGGREGATE)
    spread = fields.get("seed_std")
    if len(requested) < 3:
        if spread is not None:
            return _read_failure(SeedUnavailableReason.INVALID_AGGREGATE)
    elif (
        isinstance(spread, bool)
        or not isinstance(spread, Real)
        or not math.isfinite(spread)
        or spread < 0
    ):
        return _read_failure(SeedUnavailableReason.INVALID_AGGREGATE)
    return SeedMetricRead(
        value=float(mean),
        spread=None if spread is None else float(spread),
        reason=None,
        message=None,
    )


def read_seed_inference(
    aggregate: SeedAggregate | Mapping,
    *,
    expected_sampling_unit: str = SEED_SAMPLING_UNIT,
) -> SeedMetricRead:
    """Read a seed mean only when a finite seed spread supports inference."""
    result = read_seed_point_estimate(
        aggregate, expected_sampling_unit=expected_sampling_unit
    )
    if not result.available:
        return result
    fields, _failure = _aggregate_fields(aggregate)
    requested = fields["requested_seeds"]
    if len(requested) < 3 or result.spread is None:
        return _read_failure(SeedUnavailableReason.INSUFFICIENT_SEEDS)
    return result


def load_seed_files(
    run_dir: str,
    seed_glob: str,
    *,
    expected_seeds: Iterable[int],
) -> list[tuple[int, str, dict]]:
    """Return [(seed, filename, parsed_json), ...] for every seed file in run_dir.

    `seed_glob` is the filename pattern of the per-seed result files, with exactly
    one `*` standing in for the seed number (e.g.
    "family_split_baselines_seed*.json"); the caller supplies it so this helper
    stays generic. Each match's seed number is parsed from that `*` position — a
    filename where that position is not a plain integer, or a seed number that
    repeats across two files, is a run in error and raises rather than silently
    averaging over an unknown or double-counted seed.

    The loaded seed set must equal `expected_seeds` exactly:
    a seed present on disk but outside `expected_seeds`, or a seed in
    `expected_seeds` with no loadable file, raises.

    A corrupt or empty seed file raises. It is never omitted from aggregation.

    A result that itself records which seed produced it (a top-level "seed" key)
    must agree with the seed parsed from its filename — a renamed or misfiled
    copy (e.g. backfilling a missing seed by copying another seed's file) would
    otherwise be silently aggregated under the wrong seed number.
    """
    expected_sequence = tuple(expected_seeds)
    if any(
        isinstance(seed, bool) or not isinstance(seed, int)
        for seed in expected_sequence
    ):
        raise TypeError("expected seed identifiers must be integers")
    if len(set(expected_sequence)) != len(expected_sequence):
        raise ValueError("expected_seeds contains duplicate identifiers")

    if seed_glob.count("*") != 1:
        raise ValueError(f"seed_glob must contain exactly one '*': {seed_glob!r}")
    prefix, suffix = seed_glob.split("*")

    paths = sorted(glob.glob(os.path.join(run_dir, seed_glob)))
    loaded: list[tuple[int, str, dict]] = []
    seed_to_filename: dict[int, str] = {}
    for path in paths:
        filename = os.path.basename(path)
        token = filename[len(prefix) : len(filename) - len(suffix)]
        try:
            seed = int(token)
        except ValueError:
            raise ValueError(
                f"{path}: filename does not encode an integer seed between "
                f"{prefix!r} and {suffix!r} (got {token!r})"
            )
        if seed in seed_to_filename:
            raise ValueError(
                f"duplicate seed {seed} in {run_dir}: "
                f"{seed_to_filename[seed]} and {filename}"
            )
        seed_to_filename[seed] = filename
        try:
            with open(path) as handle:
                result = json.load(handle)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: invalid seed-result JSON") from error
        if not isinstance(result, dict):
            raise ValueError(f"{path}: seed result must be a JSON object")
        recorded_seed = result.get("seed")
        if recorded_seed is None:
            raise ValueError(f"{path}: seed result does not record its seed identifier")
        if isinstance(recorded_seed, bool) or not isinstance(recorded_seed, int):
            raise ValueError(f"{path}: recorded seed identifier must be an integer")
        if recorded_seed != seed:
            raise ValueError(
                f"{path}: filename encodes seed {seed} but the result records "
                f"seed {recorded_seed!r} — file was renamed or copied to the wrong seed"
            )
        loaded.append((seed, filename, result))

    expected = set(expected_sequence)
    found = {seed for seed, _filename, _result in loaded}
    missing = sorted(expected - found)
    unexpected = sorted(found - expected)
    if missing or unexpected:
        raise ValueError(
            f"{run_dir}: seed files for {seed_glob!r} do not match the "
            f"expected seeds {sorted(expected)} "
            f"(missing={missing}, unexpected={unexpected})"
        )

    return loaded
