"""Shared validation, aggregation, and reading for model-seed results.

The scalar core requires one explicit record for every requested seed. The
mechanism-specific traversal below remains as a delegating compatibility layer
until its callers migrate to the scalar core.
"""

from __future__ import annotations

import functools
import glob
import json
import math
import os
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Iterable, Mapping

import numpy as np

from esm2_mech.utils.constants import SEED_AGGREGATION_SCHEMA_VERSION

print = functools.partial(print, flush=True)

GENE_SPLIT = "gene_split"
FAMILY_SPLIT = "family_split"
SPLITS = [GENE_SPLIT, FAMILY_SPLIT]
HEADLINE_METRIC = "macro_f1"

# Per-feature mean computed across seeds is stored under "<metric>_seed_mean".
SEED_MEAN_SUFFIX = "_seed_mean"

# Top-level key under which the across-seed aggregate is nested in the run's
# aggregate result file (written by classify_by_mechanism).
ACROSS_SEED_KEY = "across_seed"

SEED_SAMPLING_UNIT = "model_seed"
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
    EMPTY_REQUESTED_SEEDS = "empty_requested_seeds"
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


_REASON_MESSAGES = {
    SeedUnavailableReason.EMPTY_REQUESTED_SEEDS: "no model seeds were requested",
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
}

_STORED_UNAVAILABLE_REASONS = frozenset(
    {
        SeedUnavailableReason.EMPTY_REQUESTED_SEEDS,
        SeedUnavailableReason.DUPLICATE_SEED,
        SeedUnavailableReason.UNEXPECTED_SEED,
        SeedUnavailableReason.MISSING_SEED,
        SeedUnavailableReason.FAILED_SEED,
        SeedUnavailableReason.SKIPPED_SEED,
        SeedUnavailableReason.UNSCORABLE_SEED,
        SeedUnavailableReason.INVALID_VALUE,
    }
)


@dataclass(frozen=True)
class SeedValueRecord:
    seed: int
    status: str
    value: float | None


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
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed identifiers must be integers")
    if status not in SEED_STATUSES:
        raise ValueError(f"unsupported seed status {status!r}")
    if isinstance(value, bool):
        raise TypeError("a boolean is not a scientific metric value")
    if value is not None and not isinstance(value, Real):
        raise TypeError("seed metric values must be numeric or None")
    numeric_value = None if value is None else float(value)
    return SeedValueRecord(seed=seed, status=status, value=numeric_value)


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


def validate_seed_records(
    values: Iterable[SeedValueRecord],
) -> tuple[SeedValueRecord, ...]:
    """Validate records made in memory or reconstructed from stored results."""
    records = tuple(values)
    for record in records:
        if not isinstance(record, SeedValueRecord):
            raise TypeError("seed records must be created with make_seed_record")
        if isinstance(record.seed, bool) or not isinstance(record.seed, int):
            raise TypeError("seed identifiers must be integers")
        if record.status not in SEED_STATUSES:
            raise ValueError(f"unsupported seed status {record.status!r}")
        if isinstance(record.value, bool):
            raise TypeError("a boolean is not a scientific metric value")
        if record.value is not None and not isinstance(record.value, Real):
            raise TypeError("seed metric values must be numeric or None")
    return records


def aggregate_seed_values(
    requested_seeds: Iterable[int],
    values: Iterable[SeedValueRecord],
) -> SeedAggregate:
    """Aggregate one finite point estimate from every explicitly requested seed."""
    requested = tuple(requested_seeds)
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in requested):
        raise TypeError("requested seed identifiers must be integers")

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

    records = validate_seed_records(values)

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

    if not requested:
        if record_seeds:
            return _unavailable(
                SeedUnavailableReason.UNEXPECTED_SEED,
                requested,
                (),
                sorted(record_seeds),
            )
        return _unavailable(
            SeedUnavailableReason.EMPTY_REQUESTED_SEEDS, requested, (), ()
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
        if reason not in _STORED_UNAVAILABLE_REASONS:
            return _read_failure(SeedUnavailableReason.INVALID_AGGREGATE)
        message = fields.get("message")
        empty_request = not requested
        if (
            fields.get("mean") is not None
            or fields.get("seed_std") is not None
            or (message is not None and not isinstance(message, str))
            or any(seed not in requested for seed in contributing)
            or any(seed in contributing for seed in affected)
            or (reason is SeedUnavailableReason.EMPTY_REQUESTED_SEEDS) != empty_request
            or (
                not affected
                and reason is not SeedUnavailableReason.EMPTY_REQUESTED_SEEDS
            )
        ):
            return _read_failure(SeedUnavailableReason.INVALID_AGGREGATE)
        return SeedMetricRead(
            value=None,
            spread=None,
            reason=reason,
            message=message or _REASON_MESSAGES[reason],
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


def aggregate_across_seeds(
    seed_results: list[tuple[int, str, dict]],
) -> dict[str, dict[str, dict]]:
    """Aggregate only complete, successful seed sets for each feature and metric.

    Reads each per-seed `<metric>_mean` value. Returns nested dict:
        {split: {feature: {<metric>_seed_mean, <metric>_seed_std, n_seeds}}}
    """
    required_seeds = [seed for seed, _filename, _result in seed_results]
    if len(set(required_seeds)) != len(required_seeds):
        raise ValueError("seed_results contains duplicate seed identifiers")
    aggregated: dict[str, dict[str, dict]] = {}
    for split in SPLITS:
        features = sorted(
            {
                feature
                for _seed, _filename, result in seed_results
                for feature in result.get(split, {})
            }
        )
        split_out: dict[str, dict] = {}
        for feature in features:
            blocks = {
                seed: result.get(split, {}).get(feature)
                for seed, _filename, result in seed_results
            }
            metric_names = sorted(
                {
                    key[: -len("_mean")]
                    for block in blocks.values()
                    if isinstance(block, dict)
                    for key in block
                    if key.endswith("_mean")
                }
            )
            feature_out: dict = {
                "required_seeds": required_seeds,
                "status": "success",
            }
            failed_seeds = [
                seed
                for seed, block in blocks.items()
                if not isinstance(block, dict) or block.get("status") != "success"
            ]
            if failed_seeds:
                feature_out["status"] = "unavailable"
                feature_out["unavailable_seeds"] = failed_seeds

            for base_metric in metric_names:
                records = []
                for seed in required_seeds:
                    block = blocks[seed]
                    if not isinstance(block, dict):
                        records.append(make_seed_record(seed, None))
                        continue
                    if "status" not in block:
                        raise ValueError(
                            f"seed {seed} feature {feature!r} has no status"
                        )
                    value = block.get(f"{base_metric}_mean")
                    status = block["status"]
                    records.append(make_seed_record(seed, value, status=status))
                aggregate = aggregate_seed_values(required_seeds, records)
                feature_out[f"{base_metric}_seed_aggregate"] = aggregate.to_dict()
                feature_out[f"{base_metric}_n_seeds"] = len(
                    aggregate.contributing_seeds
                )
                feature_out[f"{base_metric}_missing"] = not aggregate.available
                feature_out[f"{base_metric}_missing_seeds"] = list(
                    aggregate.affected_seeds
                )
                feature_out[f"{base_metric}{SEED_MEAN_SUFFIX}"] = aggregate.mean
                feature_out[f"{base_metric}_seed_std"] = aggregate.spread
                feature_out[f"{base_metric}_reason"] = aggregate.message
                if not aggregate.available:
                    feature_out["status"] = "unavailable"

            matrix_blocks = [
                (seed, block)
                for seed, block in blocks.items()
                if isinstance(block, dict) and "confusion_matrix" in block
            ]
            if matrix_blocks:
                _aggregate_confusion_matrices(feature_out, blocks, required_seeds)
            split_out[feature] = feature_out
        aggregated[split] = split_out
    return aggregated


def _aggregate_confusion_matrices(
    output: dict,
    blocks: dict[int, dict | None],
    required_seeds: list[int],
) -> None:
    """Store raw matrices and their equal-seed mean after row normalization."""
    raw_by_seed: dict[str, list] = {}
    normalized_by_seed: dict[str, list] = {}
    class_order = None
    missing_seeds = []
    normalized_matrices = []
    for seed in required_seeds:
        block = blocks[seed]
        if not isinstance(block, dict) or block.get("status") != "success":
            missing_seeds.append(seed)
            continue
        matrix_value = block.get("confusion_matrix")
        seed_order = block.get("confusion_matrix_class_order")
        if matrix_value is None or seed_order is None:
            missing_seeds.append(seed)
            continue
        if class_order is None:
            class_order = list(seed_order)
        elif list(seed_order) != class_order:
            raise ValueError("confusion matrix class order differs across seeds")
        matrix = np.asarray(matrix_value, dtype=float)
        expected_shape = (len(class_order), len(class_order))
        if matrix.shape != expected_shape:
            raise ValueError(
                f"confusion matrix for seed {seed} has shape {matrix.shape}, "
                f"expected {expected_shape}"
            )
        row_totals = matrix.sum(axis=1, keepdims=True)
        if np.any(row_totals == 0):
            raise ValueError(
                f"confusion matrix for seed {seed} has an empty observed-class row"
            )
        normalized = matrix / row_totals
        raw_by_seed[str(seed)] = matrix.astype(int).tolist()
        normalized_by_seed[str(seed)] = normalized.tolist()
        normalized_matrices.append(normalized)

    output["confusion_matrix_raw_by_seed"] = raw_by_seed
    output["confusion_matrix_normalized_by_seed"] = normalized_by_seed
    output["confusion_matrix_class_order"] = class_order
    output["confusion_matrix_n_seeds"] = len(normalized_matrices)
    output["confusion_matrix_missing_seeds"] = missing_seeds
    output["confusion_matrix_seed_mean"] = (
        None if missing_seeds else np.mean(normalized_matrices, axis=0).tolist()
    )


def read_across_seed_metric(
    aggregate_path: str,
    split: str,
    feature: str,
    metric: str = HEADLINE_METRIC,
) -> float | None:
    """Read one across-seed metric mean from a run's aggregate result file.

    Returns the `<metric>_seed_mean` value for the given split and feature
    (e.g. family_split / delta_mean / macro_f1). The caller supplies the path so
    this helper stays generic. No fallback: if the file or the requested
    split/feature/metric is absent, the underlying KeyError/FileNotFoundError
    propagates so the caller knows that baseline has not been produced. A present
    but unavailable metric remains ``None``.
    """
    with open(aggregate_path) as handle:
        aggregate = json.load(handle)
    block = aggregate[ACROSS_SEED_KEY][split][feature]
    value = block[f"{metric}{SEED_MEAN_SUFFIX}"]
    return None if value is None else float(value)


def print_table(aggregated: dict[str, dict[str, dict]]) -> None:
    """Print the headline-metric table: per-feature gene vs family, across seeds."""
    gene = aggregated.get("gene_split", {})
    family = aggregated.get("family_split", {})
    features = sorted(set(gene) | set(family))

    mean_key = f"{HEADLINE_METRIC}_seed_mean"
    std_key = f"{HEADLINE_METRIC}_seed_std"
    n_key = f"{HEADLINE_METRIC}_n_seeds"

    print(f"\n=== {HEADLINE_METRIC} across seeds (mean ± std) ===")
    print(
        f"{'feature':<20} {'gene-split':>18} {'family-split':>18} {'Δ(gene−fam)':>14}"
    )
    for feature in features:
        gene_metrics = gene.get(feature, {})
        family_metrics = family.get(feature, {})
        gene_mean = gene_metrics.get(mean_key)
        gene_std = gene_metrics.get(std_key)
        family_mean = family_metrics.get(mean_key)
        family_std = family_metrics.get(std_key)
        n_seeds = gene_metrics.get(n_key, family_metrics.get(n_key, 0))
        if None in (gene_mean, gene_std, family_mean, family_std):
            print(
                f"{feature:<20} {'Unscorable':>18} {'Unscorable':>18} {'NA':>14}  (n_seeds={n_seeds})"
            )
            continue
        delta = gene_mean - family_mean
        print(
            f"{feature:<20} "
            f"{gene_mean:>8.3f} ± {gene_std:<6.3f} "
            f"{family_mean:>8.3f} ± {family_std:<6.3f} "
            f"{delta:>+13.3f}  (n_seeds={n_seeds})"
        )
