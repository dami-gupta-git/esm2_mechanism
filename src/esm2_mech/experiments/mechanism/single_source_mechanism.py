"""
Single-source robustness check for the mechanism null.

The merged dataset confounds mechanism class with curation source: the AR loss-of-
function subtype comes entirely from Gerasimavicius, HI is mostly Gene2Phenotype,
while GOF and DN are predominantly Gerasimavicius. A reviewer could therefore argue
that any class-level difference (or its absence) reflects source/curation effects
rather than biology.

This script removes that confound by re-running the exact Section 2 gene-split vs
family-split mechanism probe on the Gerasimavicius-only subset, which contains all
three classes from a single curation pipeline. It reuses load_data() and
run_family_split() unchanged — only the row set is filtered — and recomputes the
majority-class floor on the subset (the floor shifts because the subset's class
balance differs from the merged set).

If the merged-dataset finding holds here (delta at the subset floor; wt_only's
gene-split lift collapsing under family-split), the mechanism null is not a source
artifact.

  Output: results/<run>/single_source_gerasimavicius/
            family_split_baselines_seed{0..4}.json  (per-seed probe results)
            aggregate.json                          (mean ± std across seeds)
            naive_baseline.json                     (subset majority-class floor)

Usage:
    python -m esm2_mech.experiments.mechanism.single_source_mechanism
"""

import argparse
import functools
import json
from collections import Counter

import numpy as np

from esm2_mech.experiments.mechanism.classify_by_mechanism import load_data
from esm2_mech.experiments.mechanism.mechanism_delta_family_split import run as run_family_split
from esm2_mech.experiments.mechanism.naive_baseline import evaluate as eval_naive
from esm2_mech.utils.constants import SEED_RESULT_GLOB, SOURCE_GERASIMAVICIUS
from esm2_mech.utils.data import build_source_mask, subset_data
from esm2_mech.utils.io import atomic_write_json
from esm2_mech.utils.paths import (
    PFAM_JSON,
    SINGLE_SOURCE_AGGREGATE_JSON,
    SINGLE_SOURCE_DIR,
    SINGLE_SOURCE_NAIVE_BASELINE_JSON,
)
from esm2_mech.utils.seed_aggregation import (
    FAMILY_SPLIT,
    GENE_SPLIT,
    HEADLINE_METRIC,
    SEED_MEAN_SUFFIX,
    aggregate_across_seeds,
    load_seed_files,
    print_table,
)

print = functools.partial(print, flush=True)

DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def compute_subset_floor(labels: np.ndarray, genes: np.ndarray) -> dict:
    """Majority-class (most_frequent DummyClassifier) macro-F1 under gene- and family-split,
    recomputed on the subset because its class balance — and hence the floor — differs from
    the merged set. Reuses naive_baseline.evaluate so the floor matches the report's method."""
    with open(PFAM_JSON) as handle:
        pfam_map = json.load(handle)
    floor = {
        "class_distribution": {k: int(v) for k, v in Counter(labels).items()},
        "gene_split": eval_naive("most_frequent", "gene", labels, genes, pfam_map),
        "family_split": eval_naive("most_frequent", "family", labels, genes, pfam_map),
    }
    print(
        f"Subset majority-class floor: "
        f"gene-split {floor['gene_split']['macro_f1_mean']:.3f}, "
        f"family-split {floor['family_split']['macro_f1_mean']:.3f}"
    )
    return floor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help="Random seeds to run (default: 0 1 2 3 4)",
    )
    args = parser.parse_args()

    SINGLE_SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()

    print(f"\n=== Filtering to source == '{SOURCE_GERASIMAVICIUS}' ===")
    mask = build_source_mask(data["valid_variants"], SOURCE_GERASIMAVICIUS)
    print(
        f"Subset: {int(mask.sum())} / {mask.size} variants  "
        f"(class distribution {dict(Counter(data['labels_3class'][mask]))})"
    )
    if int(mask.sum()) < 50:
        raise ValueError(f"Subset has only {int(mask.sum())} variants — too few to run.")

    subset = subset_data(data, mask)

    print("\n=== Subset majority-class floor ===")
    floor = compute_subset_floor(subset["labels_3class"], subset["genes_arr"])
    atomic_write_json(SINGLE_SOURCE_NAIVE_BASELINE_JSON, floor)
    print(f"Wrote {SINGLE_SOURCE_NAIVE_BASELINE_JSON}")

    print(f"\n=== Mechanism probe on {SOURCE_GERASIMAVICIUS}-only subset ===")
    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        run_family_split(data=subset, out_dir=str(SINGLE_SOURCE_DIR), seed=seed, n_folds=args.n_folds)

    print("\n=== Aggregating across seeds ===")
    seed_results = load_seed_files(str(SINGLE_SOURCE_DIR), SEED_RESULT_GLOB)
    if not seed_results:
        print(f"WARNING: no seed files to aggregate in {SINGLE_SOURCE_DIR}")
        return
    aggregated = aggregate_across_seeds(seed_results)
    atomic_write_json(
        SINGLE_SOURCE_AGGREGATE_JSON,
        {
            "source": SOURCE_GERASIMAVICIUS,
            "n_seeds": len(seed_results),
            "seed_files": [filename for filename, _ in seed_results],
            "subset_floor": floor,
            "across_seed": aggregated,
        },
    )
    print_table(aggregated)
    print(f"\nWrote {SINGLE_SOURCE_AGGREGATE_JSON}")

    print("\n=== Robustness read (compare to merged-dataset Section 2) ===")
    mean_key = f"{HEADLINE_METRIC}{SEED_MEAN_SUFFIX}"
    family_floor = floor["family_split"]["macro_f1_mean"]
    for feature in ["wt_only_mean", "delta_mean"]:
        gene = aggregated.get(GENE_SPLIT, {}).get(feature, {}).get(mean_key, float("nan"))
        family = aggregated.get(FAMILY_SPLIT, {}).get(feature, {}).get(mean_key, float("nan"))
        print(
            f"  {feature:14} gene-split {gene:.3f} → family-split {family:.3f} "
            f"(Δ {gene - family:+.3f}); subset family-split floor {family_floor:.3f}"
        )


if __name__ == "__main__":
    main()
