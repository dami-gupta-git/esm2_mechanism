"""Classify variants by mechanism (GOF/DN/LOF) using ESM-2 embeddings under gene-split and family-split CV."""

import argparse
import functools
import os

import numpy as np

from esm2_mech.experiments.mechanism.mechanism_delta_family_split import (
    PERMUTATION_FEATURES,
    run as run_family_split,
)
from esm2_mech.experiments.mechanism.mechanism_delta_probe import _load_alphamissense_scores
from esm2_mech.utils.data import load_variants, validate_embedding_variant_identity
from esm2_mech.utils.io import write_result_json
from esm2_mech.utils.seed_aggregation import (
    aggregate_across_seeds,
    load_seed_files,
    print_table,
)
from esm2_mech.utils.constants import (
    BOOTSTRAP_N_RESAMPLES,
    N_SEEDS,
    PERMUTATION_MIN_SIGNIFICANT_SEEDS,
    PERMUTATION_SIGNIFICANCE_THRESHOLD,
    SEED_RESULT_GLOB,
)
from esm2_mech.utils.paths import (
    EMB_WT_MEAN,
    EMB_MUT_MEAN,
    EMB_WT_POS,
    EMB_MUT_POS,
    EMB_VALID_VARIANTS_JSON,
    MECHANISM_AGGREGATE_JSON,
    RESULTS_DIR,
    VALID_VARIANTS_JSON,
)

print = functools.partial(print, flush=True)

OUT_DIR = RESULTS_DIR


def aggregate_permutation_results(
    seed_results: list[tuple[int, str, dict]],
) -> dict[str, dict]:
    """Collect the preregistered permutation distribution across seeds.

    The ordinary across-seed metric aggregator only reads flat ``*_mean`` values,
    so nested permutation results need an explicit path. A three-of-five decision
    is emitted only when all five seeds have a finite p-value; incomplete results
    remain visible and the decision is ``None`` rather than being treated as a
    negative result.
    """
    summaries = {}
    for feature in PERMUTATION_FEATURES:
        per_seed = []
        missing_seeds = []
        seeds_without_valid_p_value = []
        for seed, filename, result in seed_results:
            permutation = (
                result.get("family_split", {})
                .get(feature, {})
                .get("permutation")
            )
            if permutation is None:
                missing_seeds.append(seed)
                continue

            p_value = permutation.get("p_value")
            if p_value is None or not np.isfinite(p_value):
                seeds_without_valid_p_value.append(seed)
            per_seed.append({
                "seed": seed,
                "source_file": filename,
                **permutation,
            })

        finite_p_values = [
            row["p_value"]
            for row in per_seed
            if row.get("p_value") is not None and np.isfinite(row["p_value"])
        ]
        n_below_threshold = sum(
            p_value < PERMUTATION_SIGNIFICANCE_THRESHOLD
            for p_value in finite_p_values
        )
        complete = (
            len(seed_results) == N_SEEDS
            and len(per_seed) == N_SEEDS
            and len(finite_p_values) == N_SEEDS
        )
        summaries[feature] = {
            "per_seed": per_seed,
            "n_seed_results": len(per_seed),
            "n_valid_p_values": len(finite_p_values),
            "missing_seeds": missing_seeds,
            "seeds_without_valid_p_value": seeds_without_valid_p_value,
            "resolution_limited_seeds": [
                row["seed"] for row in per_seed if row.get("resolution_limited") is True
            ],
            "significance_threshold": PERMUTATION_SIGNIFICANCE_THRESHOLD,
            "n_below_significance_threshold": n_below_threshold,
            "required_seed_count": N_SEEDS,
            "required_significant_seed_count": PERMUTATION_MIN_SIGNIFICANT_SEEDS,
            "preregistered_rule_evaluable": complete,
            "meets_preregistered_three_of_five_rule": (
                n_below_threshold >= PERMUTATION_MIN_SIGNIFICANT_SEEDS
                if complete else None
            ),
        }
    return summaries


def print_permutation_summary(summary: dict[str, dict]) -> None:
    """Print the full-seed permutation decision without hiding incomplete runs."""
    print("\n=== Permutation results across seeds ===")
    for feature in PERMUTATION_FEATURES:
        feature_summary = summary[feature]
        count = feature_summary["n_below_significance_threshold"]
        if feature_summary["preregistered_rule_evaluable"]:
            verdict = (
                "criterion met"
                if feature_summary["meets_preregistered_three_of_five_rule"]
                else "criterion not met"
            )
            print(
                f"  {feature}: {count}/{N_SEEDS} p-values below "
                f"{PERMUTATION_SIGNIFICANCE_THRESHOLD} ({verdict}); "
                f"resolution-limited seeds "
                f"{feature_summary['resolution_limited_seeds']}"
            )
        else:
            print(
                f"  {feature}: incomplete, {feature_summary['n_valid_p_values']}/"
                f"{N_SEEDS} valid p-values; missing permutation seeds "
                f"{feature_summary['missing_seeds']}; invalid p-value seeds "
                f"{feature_summary['seeds_without_valid_p_value']}"
            )


def load_data() -> dict:
    """Load variants, embeddings, labels, and auxiliary features."""
    print("\n=== Loading valid variants ===")
    valid_variants = load_variants(VALID_VARIANTS_JSON)
    validate_embedding_variant_identity(valid_variants, EMB_VALID_VARIANTS_JSON)
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
    seed_results = load_seed_files(
        str(OUT_DIR), SEED_RESULT_GLOB, expected_seeds=range(args.seeds)
    )
    print(f"Loaded {len(seed_results)} seed files:")
    for _seed, filename, _result in seed_results:
        print(f"  {filename}")

    aggregated = aggregate_across_seeds(seed_results)
    permutation_summary = (
        aggregate_permutation_results(seed_results)
        if args.n_permutations > 0 else None
    )
    aggregate_payload = {
        "n_seeds": len(seed_results),
        "seed_files": [filename for _seed, filename, _result in seed_results],
        "across_seed": aggregated,
    }
    if permutation_summary is not None:
        aggregate_payload["permutation_summary"] = permutation_summary
    write_result_json(
        MECHANISM_AGGREGATE_JSON,
        aggregate_payload,
        seeds=list(range(args.seeds)),
    )
    print_table(aggregated)
    if permutation_summary is not None:
        print_permutation_summary(permutation_summary)
    print(f"\nWrote {MECHANISM_AGGREGATE_JSON}")


if __name__ == "__main__":
    main()
