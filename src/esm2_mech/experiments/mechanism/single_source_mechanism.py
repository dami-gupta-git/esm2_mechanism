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
import glob
import json
import os
from collections import Counter

import numpy as np

from esm2_mech.experiments.mechanism.classify_by_mechanism import load_data
from esm2_mech.experiments.mechanism.mechanism_delta_family_split import (
    mechanism_input_fingerprints,
    run as run_family_split,
)
from esm2_mech.experiments.mechanism.naive_baseline import evaluate as eval_naive
from esm2_mech.utils.constants import (
    BOOTSTRAP_N_RESAMPLES,
    MECHANISM_CLASSES,
    N_SEEDS,
    MECHANISM_OOF_CACHE_GLOB,
    SEED_RESULT_GLOB,
    SOURCE_GERASIMAVICIUS,
)
from esm2_mech.utils.data import build_source_mask, load_pfam_map, subset_data
from esm2_mech.utils.io import write_result_json
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


def compute_subset_floor(
    labels: np.ndarray, genes: np.ndarray, n_seeds: int, n_folds: int
) -> dict:
    """Majority-class (most_frequent DummyClassifier) macro-F1 under gene- and family-split,
    recomputed on the subset because its class balance — and hence the floor — differs from
    the merged set. Reuses naive_baseline.evaluate so the floor matches the report's method.
    n_seeds/n_folds must match the probe's --seeds/--n_folds so the floor and the probe it's
    compared against use the same CV setup."""
    pfam_map = load_pfam_map(PFAM_JSON)
    floor = {
        "class_distribution": {k: int(v) for k, v in Counter(labels).items()},
        "gene_split": eval_naive(
            "most_frequent", "gene", labels, genes, pfam_map, n_seeds=n_seeds, n_folds=n_folds
        ),
        "family_split": eval_naive(
            "most_frequent", "family", labels, genes, pfam_map, n_seeds=n_seeds, n_folds=n_folds
        ),
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
        "--seeds", type=int, default=N_SEEDS,
        help="number of seeds to run; runs 0..seeds-1 (>=1)",
    )
    parser.add_argument("--no_ci", action="store_true", help="skip cluster-bootstrap CIs")
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    parser.add_argument(
        "--n_permutations", type=int, default=0,
        help="label-permutation reps for headline features (0 = skip; slow, refits per rep)",
    )
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")

    SINGLE_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    stale = []
    for output_glob in (SEED_RESULT_GLOB, MECHANISM_OOF_CACHE_GLOB):
        stale.extend(glob.glob(os.path.join(str(SINGLE_SOURCE_DIR), output_glob)))
    for path in stale:
        os.remove(path)
    if stale:
        print(f"Removed {len(stale)} stale seed file(s) from a prior run in {SINGLE_SOURCE_DIR}")

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
    pfam_map = load_pfam_map(PFAM_JSON)
    input_fingerprints = mechanism_input_fingerprints(subset, pfam_map)

    print("\n=== Subset majority-class floor ===")
    floor = compute_subset_floor(
        subset["labels_3class"], subset["genes_arr"], n_seeds=args.seeds, n_folds=args.n_folds
    )
    floor["input_fingerprints"] = {
        "labeled_variants": input_fingerprints["labeled_variants"],
        "pfam_assignments": input_fingerprints["pfam_assignments"],
    }
    write_result_json(SINGLE_SOURCE_NAIVE_BASELINE_JSON, floor, seeds=list(range(args.seeds)))
    print(f"Wrote {SINGLE_SOURCE_NAIVE_BASELINE_JSON}")

    print(f"\n=== Mechanism probe on {SOURCE_GERASIMAVICIUS}-only subset ===")
    for seed in range(args.seeds):
        print(f"\n--- Seed {seed} ---")
        run_family_split(
            data=subset, out_dir=str(SINGLE_SOURCE_DIR), seed=seed, n_folds=args.n_folds,
            compute_ci=not args.no_ci, n_boot=args.n_boot,
            n_permutations=args.n_permutations,
            input_fingerprints=input_fingerprints,
        )

    print("\n=== Aggregating across seeds ===")
    seed_results = load_seed_files(
        str(SINGLE_SOURCE_DIR), SEED_RESULT_GLOB, expected_seeds=range(args.seeds)
    )
    aggregated = aggregate_across_seeds(
        seed_results, confusion_matrix_class_order=MECHANISM_CLASSES
    )
    write_result_json(
        SINGLE_SOURCE_AGGREGATE_JSON,
        {
            "source": SOURCE_GERASIMAVICIUS,
            "n_seeds": len(seed_results),
            "seed_files": [filename for _seed, filename, _result in seed_results],
            "input_fingerprints": input_fingerprints,
            "subset_floor": floor,
            "across_seed": aggregated,
        },
        seeds=list(range(args.seeds)),
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
