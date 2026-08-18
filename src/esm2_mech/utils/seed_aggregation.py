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
values; a seed missing a metric is skipped for that metric only and counted, so
nothing is fabricated.

This is a reusable utility: callers pass the run directory and the seed-file glob
pattern in — no experiment-specific path is hardcoded here.
"""

from __future__ import annotations

import functools
import glob
import json
import os
from collections import defaultdict

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
                loaded.append((seed, filename, json.load(handle)))
        except json.JSONDecodeError:
            print(f"  WARNING: corrupt seed file {path} — skipping")

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
) -> dict[str, dict[str, dict[str, float]]]:
    """For each split → feature → metric, compute mean/std across seeds.

    Reads each per-seed `<metric>_mean` value. Returns nested dict:
        {split: {feature: {<metric>_seed_mean, <metric>_seed_std, n_seeds}}}
    """
    aggregated: dict[str, dict[str, dict[str, float]]] = {}
    for split in SPLITS:
        # feature -> base_metric -> [per-seed mean values]
        collected: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for _seed, _filename, result in seed_results:
            split_block = result.get(split, {})
            for feature, metrics in split_block.items():
                for key, value in metrics.items():
                    if not key.endswith("_mean"):
                        continue
                    base_metric = key[: -len("_mean")]
                    if value is not None and not (
                        isinstance(value, float) and np.isnan(value)
                    ):
                        collected[feature][base_metric].append(float(value))

        split_out: dict[str, dict[str, float]] = {}
        for feature, metric_values in collected.items():
            feature_out: dict[str, float] = {}
            for base_metric, values in metric_values.items():
                feature_out[f"{base_metric}{SEED_MEAN_SUFFIX}"] = float(np.mean(values))
                feature_out[f"{base_metric}_seed_std"] = float(np.std(values))
                feature_out[f"{base_metric}_n_seeds"] = len(values)
            split_out[feature] = feature_out
        aggregated[split] = split_out
    return aggregated


def read_across_seed_metric(
    aggregate_path: str,
    split: str,
    feature: str,
    metric: str = HEADLINE_METRIC,
) -> float:
    """Read one across-seed metric mean from a run's aggregate result file.

    Returns the `<metric>_seed_mean` value for the given split and feature
    (e.g. family_split / delta_mean / macro_f1). The caller supplies the path so
    this helper stays generic. No fallback: if the file or the requested
    split/feature/metric is absent, the underlying KeyError/FileNotFoundError
    propagates so the caller knows that baseline has not been produced.
    """
    with open(aggregate_path) as handle:
        aggregate = json.load(handle)
    block = aggregate[ACROSS_SEED_KEY][split][feature]
    return float(block[f"{metric}{SEED_MEAN_SUFFIX}"])


def print_table(aggregated: dict[str, dict[str, dict[str, float]]]) -> None:
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
        gene_mean = gene_metrics.get(mean_key, float("nan"))
        gene_std = gene_metrics.get(std_key, float("nan"))
        family_mean = family_metrics.get(mean_key, float("nan"))
        family_std = family_metrics.get(std_key, float("nan"))
        delta = gene_mean - family_mean
        n_seeds = gene_metrics.get(n_key, family_metrics.get(n_key, 0))
        print(
            f"{feature:<20} "
            f"{gene_mean:>8.3f} ± {gene_std:<6.3f} "
            f"{family_mean:>8.3f} ± {family_std:<6.3f} "
            f"{delta:>+13.3f}  (n_seeds={n_seeds})"
        )
