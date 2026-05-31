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

SPLITS = ["gene_split", "family_split"]
HEADLINE_METRIC = "macro_f1"


def load_seed_files(run_dir: str, seed_glob: str) -> list[tuple[str, dict]]:
    """Return [(filename, parsed_json), ...] for every seed file in run_dir.

    `seed_glob` is the filename pattern of the per-seed result files (e.g.
    "family_split_baselines_seed*.json"); the caller supplies it so this helper
    stays generic. A corrupt seed file is skipped with a warning rather than
    silently dropped or fabricated.
    """
    paths = sorted(glob.glob(os.path.join(run_dir, seed_glob)))
    loaded: list[tuple[str, dict]] = []
    for path in paths:
        try:
            with open(path) as handle:
                loaded.append((os.path.basename(path), json.load(handle)))
        except json.JSONDecodeError:
            print(f"  WARNING: corrupt seed file {path} — skipping")
    return loaded


def aggregate_across_seeds(
    seed_results: list[tuple[str, dict]]
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
        for _filename, result in seed_results:
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
                feature_out[f"{base_metric}_seed_mean"] = float(np.mean(values))
                feature_out[f"{base_metric}_seed_std"] = float(np.std(values))
                feature_out[f"{base_metric}_n_seeds"] = len(values)
            split_out[feature] = feature_out
        aggregated[split] = split_out
    return aggregated


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
