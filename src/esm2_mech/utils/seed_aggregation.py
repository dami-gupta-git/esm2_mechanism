"""
Pool per-seed probe-result JSONs into one across-seed headline figure.

A multi-seed experiment writes one JSON per seed, and a run.log typically only
prints the last seed's summary — so any single number read off is one seed, and
its ± is across-FOLD variation within that seed. That under-reports the true
uncertainty, which is the spread ACROSS seeds.

These helpers pool all seed files in a run directory and, for every
feature × split × metric, report mean ± std ACROSS seeds (the honest headline).

Each per-seed file stores, per split, metrics as `<metric>_mean` / `<metric>_std`
(the per-fold aggregate for that seed). We aggregate the per-seed `<metric>_mean`
values only when every required seed produced that metric successfully.

This is a reusable utility: callers pass the run directory and the seed-file glob
pattern in — no experiment-specific path is hardcoded here.
"""

from __future__ import annotations

import functools
import glob
import json
import os

import numpy as np

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


def load_seed_files(
    run_dir: str, seed_glob: str, expected_seeds: "list[int] | range | None" = None
) -> list[tuple[int, str, dict]]:
    """Return [(seed, filename, parsed_json), ...] for every seed file in run_dir.

    `seed_glob` is the filename pattern of the per-seed result files, with exactly
    one `*` standing in for the seed number (e.g.
    "family_split_baselines_seed*.json"); the caller supplies it so this helper
    stays generic. Each match's seed number is parsed from that `*` position — a
    filename where that position is not a plain integer, or a seed number that
    repeats across two files, is a run in error and raises rather than silently
    averaging over an unknown or double-counted seed.

    If `expected_seeds` is given, the loaded seed set must equal it exactly:
    a seed present on disk but outside `expected_seeds`, or a seed in
    `expected_seeds` with no loadable file, raises. Leave it None when the caller
    has no fixed expected set (e.g. an aggregator run separately from whatever
    produced the seed files).

    A corrupt (unparseable JSON) seed file is skipped with a warning rather than
    silently dropped or fabricated; that seed is then treated as absent for the
    `expected_seeds` completeness check.

    A result that itself records which seed produced it (a top-level "seed" key)
    must agree with the seed parsed from its filename — a renamed or misfiled
    copy (e.g. backfilling a missing seed by copying another seed's file) would
    otherwise be silently aggregated under the wrong seed number.
    """
    if seed_glob.count("*") != 1:
        raise ValueError(f"seed_glob must contain exactly one '*': {seed_glob!r}")
    prefix, suffix = seed_glob.split("*")

    paths = sorted(glob.glob(os.path.join(run_dir, seed_glob)))
    loaded: list[tuple[int, str, dict]] = []
    seed_to_filename: dict[int, str] = {}
    for path in paths:
        filename = os.path.basename(path)
        token = filename[len(prefix): len(filename) - len(suffix)]
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
        except json.JSONDecodeError:
            print(f"  WARNING: corrupt seed file {path} — skipping")
            continue
        recorded_seed = result.get("seed")
        if recorded_seed is not None and recorded_seed != seed:
            raise ValueError(
                f"{path}: filename encodes seed {seed} but the result records "
                f"seed {recorded_seed!r} — file was renamed or copied to the wrong seed"
            )
        loaded.append((seed, filename, result))

    if expected_seeds is not None:
        expected = set(expected_seeds)
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
    seed_results: list[tuple[int, str, dict]]
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
                values = []
                missing_seeds = []
                for seed in required_seeds:
                    block = blocks[seed]
                    value = None if not isinstance(block, dict) else block.get(
                        f"{base_metric}_mean"
                    )
                    if (
                        not isinstance(block, dict)
                        or block.get("status") != "success"
                        or value is None
                        or (isinstance(value, float) and np.isnan(value))
                    ):
                        missing_seeds.append(seed)
                    else:
                        values.append(float(value))
                feature_out[f"{base_metric}_n_seeds"] = len(values)
                feature_out[f"{base_metric}_missing"] = bool(missing_seeds)
                feature_out[f"{base_metric}_missing_seeds"] = missing_seeds
                if missing_seeds:
                    feature_out[f"{base_metric}{SEED_MEAN_SUFFIX}"] = None
                    feature_out[f"{base_metric}_seed_std"] = None
                    feature_out[f"{base_metric}_reason"] = (
                        "metric unavailable in one or more required seeds"
                    )
                else:
                    feature_out[f"{base_metric}{SEED_MEAN_SUFFIX}"] = float(
                        np.mean(values)
                    )
                    feature_out[f"{base_metric}_seed_std"] = float(np.std(values))
                    feature_out[f"{base_metric}_reason"] = None

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
        None
        if missing_seeds
        else np.mean(normalized_matrices, axis=0).tolist()
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
    print(f"{'feature':<20} {'gene-split':>18} {'family-split':>18} {'Δ(gene−fam)':>14}")
    for feature in features:
        gene_metrics = gene.get(feature, {})
        family_metrics = family.get(feature, {})
        gene_mean = gene_metrics.get(mean_key)
        gene_std = gene_metrics.get(std_key)
        family_mean = family_metrics.get(mean_key)
        family_std = family_metrics.get(std_key)
        n_seeds = gene_metrics.get(n_key, family_metrics.get(n_key, 0))
        if None in (gene_mean, gene_std, family_mean, family_std):
            print(f"{feature:<20} {'Unscorable':>18} {'Unscorable':>18} {'NA':>14}  (n_seeds={n_seeds})")
            continue
        delta = gene_mean - family_mean
        print(
            f"{feature:<20} "
            f"{gene_mean:>8.3f} ± {gene_std:<6.3f} "
            f"{family_mean:>8.3f} ± {family_std:<6.3f} "
            f"{delta:>+13.3f}  (n_seeds={n_seeds})"
        )
