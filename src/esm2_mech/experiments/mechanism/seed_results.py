"""Traversal of the mechanism per-seed result layout onto the shared seed contract.

The mechanism experiments write one result file per model seed, each holding a
split, then a feature, then that feature's within-seed metrics. This module walks
that layout and hands every value to the shared reducers in
`utils/seed_aggregation.py`, which remains the only implementation of scientific
aggregation across model seeds.
"""

from __future__ import annotations

import functools
import json
from typing import Iterable

from esm2_mech.utils.constants import SEED_AGGREGATION_SCHEMA_VERSION
from esm2_mech.utils.seed_aggregation import (
    SEED_SCHEMA_KEY,
    SEED_STATUS_SUCCESS,
    SEED_STATUS_UNSCORABLE,
    SeedMetricRead,
    aggregate_seed_confusion_matrices,
    aggregate_seed_values,
    make_seed_payload_record,
    make_seed_record,
    read_seed_inference,
    read_seed_point_estimate,
    read_seed_result_contract,
)

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


def _feature_seed_status(root_status: str, block, seed: int, feature: str) -> str:
    """Combine the seed's declared root status with one feature block's status.

    A seed that declares itself failed, skipped, or unscorable at the root is that
    status for every feature, whatever an inner block happens to say.
    """
    if root_status != SEED_STATUS_SUCCESS:
        return root_status
    if not isinstance(block, dict):
        return SEED_STATUS_UNSCORABLE
    if "status" not in block:
        raise ValueError(f"seed {seed} feature {feature!r} has no status")
    return block["status"]


def aggregate_across_seeds(
    seed_results: list[tuple[int, str, dict]],
    requested_seeds: Iterable[int],
    *,
    confusion_matrix_class_order: Iterable | None = None,
) -> dict[str, dict[str, dict]]:
    """Aggregate only complete, successful seed sets for each feature and metric.

    The caller declares which seeds the run asked for. A requested seed with no
    result, or a result for a seed nobody asked for, makes the affected aggregate
    unavailable; the seed set is never taken from whichever files happened to load.

    Reads each per-seed `<metric>_mean` value. Returns nested dict:
        {split: {feature: {<metric>_seed_mean, <metric>_seed_std, n_seeds}}}
    """
    requested = tuple(requested_seeds)
    if len(set(requested)) != len(requested):
        raise ValueError("requested_seeds contains duplicate identifiers")
    received_seeds = [seed for seed, _filename, _result in seed_results]
    if len(set(received_seeds)) != len(received_seeds):
        raise ValueError("seed_results contains duplicate seed identifiers")
    declared_confusion_order = (
        None
        if confusion_matrix_class_order is None
        else tuple(confusion_matrix_class_order)
    )
    root_status = {
        seed: read_seed_result_contract(seed, filename, result)
        for seed, filename, result in seed_results
    }
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
            statuses = {
                seed: _feature_seed_status(root_status[seed], block, seed, feature)
                for seed, block in blocks.items()
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
                "required_seeds": list(requested),
                "status": "success",
            }
            unavailable_seeds = sorted(
                set(requested).symmetric_difference(received_seeds)
                | {
                    seed
                    for seed, status in statuses.items()
                    if status != SEED_STATUS_SUCCESS
                }
            )
            if unavailable_seeds:
                feature_out["status"] = "unavailable"
                feature_out["unavailable_seeds"] = unavailable_seeds

            for base_metric in metric_names:
                records = []
                for seed in received_seeds:
                    block = blocks[seed]
                    value = (
                        block.get(f"{base_metric}_mean")
                        if isinstance(block, dict)
                        else None
                    )
                    records.append(
                        make_seed_record(seed, value, status=statuses[seed])
                    )
                aggregate = aggregate_seed_values(requested, records)
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

            has_matrix = any(
                isinstance(block, dict) and "confusion_matrix" in block
                for block in blocks.values()
            )
            if has_matrix:
                if declared_confusion_order is None:
                    raise ValueError(
                        "confusion_matrix_class_order must be declared by the caller"
                    )
                _aggregate_confusion_matrices(
                    feature_out,
                    blocks,
                    statuses,
                    requested,
                    declared_confusion_order,
                )
            split_out[feature] = feature_out
        aggregated[split] = split_out
    return aggregated


def _aggregate_confusion_matrices(
    output: dict,
    blocks: dict[int, dict | None],
    statuses: dict[int, str],
    requested_seeds: tuple[int, ...],
    class_order: Iterable,
) -> None:
    """Adapt mechanism result blocks to the shared matrix reducer."""
    declared_classes = tuple(class_order)
    records = []
    for seed, block in blocks.items():
        payload = (
            {
                "matrix": block.get("confusion_matrix"),
                "class_order": block.get("confusion_matrix_class_order"),
            }
            if isinstance(block, dict)
            else None
        )
        records.append(
            make_seed_payload_record(seed, payload, status=statuses[seed])
        )
    aggregate = aggregate_seed_confusion_matrices(
        requested_seeds, declared_classes, records
    )
    output["confusion_matrix_seed_aggregate"] = aggregate.to_dict()
    output["confusion_matrix_class_order"] = list(declared_classes)
    output["confusion_matrix_n_seeds"] = len(aggregate.contributing_seeds)
    output["confusion_matrix_missing_seeds"] = list(aggregate.affected_seeds)
    if not aggregate.available:
        output["confusion_matrix_raw_by_seed"] = {}
        output["confusion_matrix_normalized_by_seed"] = {}
        output["confusion_matrix_pooled_raw"] = None
        output["confusion_matrix_seed_mean"] = None
        return
    output["confusion_matrix_raw_by_seed"] = aggregate.payload["raw_by_seed"]
    output["confusion_matrix_normalized_by_seed"] = aggregate.payload[
        "normalized_by_seed"
    ]
    output["confusion_matrix_pooled_raw"] = aggregate.payload["pooled_raw"]
    output["confusion_matrix_seed_mean"] = aggregate.payload["normalized_seed_mean"]


def aggregate_result_contract() -> dict:
    """Root field every across-seed aggregate result file declares.

    The file states the schema its aggregates were written under, so a reader
    rejects a stale file outright rather than trusting individual numbers in it.
    """
    return {SEED_SCHEMA_KEY: SEED_AGGREGATION_SCHEMA_VERSION}


def read_across_seed_metric(
    aggregate_path: str,
    split: str,
    feature: str,
    metric: str = HEADLINE_METRIC,
) -> float | None:
    """Read one across-seed metric mean from a run's aggregate result file.

    Reads the stored seed aggregate through the shared reader rather than the
    convenience copy of its mean. The caller supplies the path so this helper
    stays generic. No fallback: if the file or the requested split/feature/metric
    is absent, the underlying KeyError/FileNotFoundError propagates so the caller
    knows that baseline has not been produced. A present but unavailable metric
    remains ``None``.
    """
    with open(aggregate_path) as handle:
        aggregate = json.load(handle)
    version = aggregate.get(SEED_SCHEMA_KEY)
    if version != SEED_AGGREGATION_SCHEMA_VERSION:
        raise ValueError(
            f"{aggregate_path}: seed schema version {version!r} does not match the "
            f"expected {SEED_AGGREGATION_SCHEMA_VERSION}"
        )
    return read_feature_metric(
        aggregate[ACROSS_SEED_KEY], split, feature, metric
    ).value


def read_feature_metric(
    across_seed: dict,
    split: str,
    feature: str,
    metric: str = HEADLINE_METRIC,
    *,
    require_spread: bool = False,
) -> SeedMetricRead:
    """Read one across-seed feature metric through the shared seed reader.

    `require_spread` is for a display that draws an error bar: it reports the
    metric as unavailable unless a seed spread supports one.
    """
    block = across_seed[split][feature]
    stored = block[f"{metric}_seed_aggregate"]
    if require_spread:
        return read_seed_inference(stored)
    return read_seed_point_estimate(stored)


def print_table(aggregated: dict[str, dict[str, dict]]) -> None:
    """Print the headline-metric table: per-feature gene vs family, across seeds."""
    gene = aggregated.get(GENE_SPLIT, {})
    family = aggregated.get(FAMILY_SPLIT, {})
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
