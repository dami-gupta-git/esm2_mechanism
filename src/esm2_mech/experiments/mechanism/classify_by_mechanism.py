"""Classify variants by mechanism (GOF/DN/LOF) using ESM-2 embeddings under gene-split and family-split CV."""

import argparse
import functools
import os

import numpy as np

from esm2_mech.experiments.mechanism.mechanism_delta_family_split import run as run_family_split
from esm2_mech.experiments.mechanism.mechanism_delta_probe import _load_alphamissense_scores
from esm2_mech.utils.data import load_variants
from esm2_mech.utils.io import atomic_write_json
from esm2_mech.utils.seed_aggregation import (
    aggregate_across_seeds,
    load_seed_files,
    print_table,
)
from esm2_mech.utils.constants import BOOTSTRAP_N_RESAMPLES, N_SEEDS, SEED_RESULT_GLOB
from esm2_mech.utils.paths import (
    EMB_WT_MEAN,
    EMB_MUT_MEAN,
    EMB_WT_POS,
    EMB_MUT_POS,
    MECHANISM_AGGREGATE_JSON,
    RESULTS_DIR,
    VALID_VARIANTS_JSON,
)

print = functools.partial(print, flush=True)

OUT_DIR = RESULTS_DIR


def load_data() -> dict:
    """Load variants, embeddings, labels, and auxiliary features."""
    print("\n=== Loading valid variants ===")
    valid_variants = load_variants(VALID_VARIANTS_JSON)
    print(f"Valid variant pairs: {len(valid_variants)}")
    if len(valid_variants) < 50:
        print("WARNING: Very few valid variants. Results may not be reliable.")

    print("\n=== Loading embeddings ===")
    for path in [EMB_WT_MEAN, EMB_MUT_MEAN, EMB_WT_POS, EMB_MUT_POS]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embedding file missing: {path}")
    emb_wt_mean = np.load(EMB_WT_MEAN)
    emb_mut_mean = np.load(EMB_MUT_MEAN)
    emb_wt_pos = np.load(EMB_WT_POS)
    emb_mut_pos = np.load(EMB_MUT_POS)

    num_variants = len(valid_variants)
    for name, arr in [
        ("EMB_WT_MEAN", emb_wt_mean),
        ("EMB_MUT_MEAN", emb_mut_mean),
        ("EMB_WT_POS", emb_wt_pos),
        ("EMB_MUT_POS", emb_mut_pos),
    ]:
        if arr.shape[0] != num_variants:
            raise ValueError(
                f"embedding/variant row mismatch: {name} has {arr.shape[0]} rows "
                f"vs {num_variants} valid_variants — {VALID_VARIANTS_JSON} is not "
                f"row-aligned with the embeddings."
            )

    labels_3class = np.array([v["label_3class"] for v in valid_variants])
    labels_4class = np.array([v["mechanism"] for v in valid_variants])
    genes_arr = np.array([v["gene"] for v in valid_variants])
    foldx_ddg = np.array(
        [v["foldx_ddg"] if v["foldx_ddg"] is not None else np.nan for v in valid_variants]
    )
    aa_wt_list = [v["aa_wt"] for v in valid_variants]
    aa_mut_list = [v["aa_mut"] for v in valid_variants]

    print("\n=== Loading AlphaMissense scores ===")
    alphamissense_scores = _load_alphamissense_scores(valid_variants)

    return {
        "valid_variants": valid_variants,
        "emb_wt_mean": emb_wt_mean,
        "emb_mut_mean": emb_mut_mean,
        "emb_wt_pos": emb_wt_pos,
        "emb_mut_pos": emb_mut_pos,
        "labels_3class": labels_3class,
        "labels_4class": labels_4class,
        "genes_arr": genes_arr,
        "foldx_ddg": foldx_ddg,
        "aa_wt_list": aa_wt_list,
        "aa_mut_list": aa_mut_list,
        "alphamissense_scores": alphamissense_scores,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()

    print("\n=== Result 2: Gene-split vs family-split baselines ===")
    for seed in range(args.seeds):
        print(f"\n--- Seed {seed} ---")
        run_family_split(
            data=data, out_dir=str(OUT_DIR), seed=seed,
            compute_ci=not args.no_ci, n_boot=args.n_boot,
            n_permutations=args.n_permutations,
        )

    print("\n=== Aggregating across seeds ===")
    seed_results = load_seed_files(str(OUT_DIR), SEED_RESULT_GLOB)
    if not seed_results:
        print(f"WARNING: no seed files to aggregate in {OUT_DIR}")
        return
    print(f"Loaded {len(seed_results)} seed files:")
    for filename, _result in seed_results:
        print(f"  {filename}")

    aggregated = aggregate_across_seeds(seed_results)
    atomic_write_json(
        MECHANISM_AGGREGATE_JSON,
        {
            "n_seeds": len(seed_results),
            "seed_files": [filename for filename, _result in seed_results],
            "across_seed": aggregated,
        },
    )
    print_table(aggregated)
    print(f"\nWrote {MECHANISM_AGGREGATE_JSON}")


if __name__ == "__main__":
    main()
