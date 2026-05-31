"""
Aggregate the per-seed family_split_baselines results into one headline figure.

`mechanism_delta_family_split.py` writes one JSON per seed
(family_split_baselines_seed{N}.json) and its run.log only prints the last
seed's summary — so any single number you read off is one seed, and its ± is
across-FOLD variation within that seed. That under-reports the true uncertainty,
which is the spread ACROSS seeds.

This script pools all seed files in a run directory and, for every
feature × split × metric, reports mean ± std ACROSS seeds (the honest headline),
plus the gene→family macro-F1 drop. It writes aggregate.json and prints a table.

Each per-seed file stores, per split, metrics as `<metric>_mean` / `<metric>_std`
(the per-fold aggregate for that seed). We aggregate the per-seed `<metric>_mean`
values; a seed missing a metric is skipped for that metric only and counted, so
nothing is fabricated.

Usage:
    python -m esm2_mech.experiments.mechanism.aggregate_seeds --run-dir results/run1
"""

from __future__ import annotations

import argparse
import functools
import glob
import json
import os
from collections import defaultdict

import numpy as np

from esm2_mech.utils.paths import RESULTS_DIR
from esm2_mech.utils.io import atomic_write_json

print = functools.partial(print, flush=True)

SEED_GLOB = "family_split_baselines_seed*.json"
SPLITS = ["gene_split", "family_split"]
HEADLINE_METRIC = "macro_f1"


def load_seed_files(run_dir: str) -> list[tuple[str, dict]]:
    """Return [(filename, parsed_json), ...] for every seed file in run_dir.

    A corrupt seed file is skipped with a warning rather than silently dropped
    or fabricated.
    """
    paths = sorted(glob.glob(os.path.join(run_dir, SEED_GLOB)))
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-dir",
        default=str(RESULTS_DIR / "run1"),
        help="Directory holding family_split_baselines_seed*.json files.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path for aggregate.json (default: <run-dir>/aggregate.json).",
    )
    args = parser.parse_args()

    seed_results = load_seed_files(args.run_dir)
    if not seed_results:
        print(f"ERROR: no seed files matching {SEED_GLOB} in {args.run_dir}")
        return 1
    print(f"Loaded {len(seed_results)} seed files from {args.run_dir}:")
    for filename, _result in seed_results:
        print(f"  {filename}")

    aggregated = aggregate_across_seeds(seed_results)

    # Carry forward dataset-level metadata only if identical across all seeds —
    # otherwise the seeds are not comparable and we surface that rather than
    # picking one arbitrarily.
    meta: dict = {}
    for meta_key in ["n_variants", "n_genes", "n_families"]:
        values = {result.get(meta_key) for _filename, result in seed_results}
        if len(values) == 1:
            meta[meta_key] = next(iter(values))
        else:
            print(f"  WARNING: {meta_key} differs across seeds: {sorted(values)}")
            meta[meta_key] = sorted(values)

    output = {
        "n_seeds": len(seed_results),
        "seed_files": [filename for filename, _result in seed_results],
        "metadata": meta,
        "across_seed": aggregated,
    }

    out_path = args.out or os.path.join(args.run_dir, "aggregate.json")
    atomic_write_json(out_path, output)

    print_table(aggregated)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
