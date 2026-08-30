"""Classify variants by mechanism (GOF/DN/LOF) using ESM-2 embeddings under gene-split and family-split CV."""

import argparse
import functools
import json
import os
from typing import Iterable

import numpy as np

from esm2_mech.experiments.mechanism.mechanism_delta_family_split import (
    PERMUTATION_FEATURES,
    run as run_family_split,
)
from esm2_mech.experiments.mechanism.mechanism_delta_probe import (
    _load_alphamissense_scores,
)
from esm2_mech.utils.data import load_variants, validate_embedding_variant_identity
from esm2_mech.utils.io import write_result_json
from esm2_mech.utils.seed_aggregation import (
    SEED_STATUS_SUCCESS,
    SEED_STATUS_UNSCORABLE,
    aggregate_seed_values,
    aggregate_seed_vote,
    block_seed_status,
    load_seed_files,
    make_seed_record,
    read_seed_point_estimate,
    seed_count,
)
from esm2_mech.experiments.mechanism.seed_results import (
    aggregate_across_seeds,
    aggregate_result_contract,
    print_table,
)
from esm2_mech.utils.constants import (
    BOOTSTRAP_N_RESAMPLES,
    MECHANISM_NULL_FLOOR_MARGIN,
    MECHANISM_CLASSES,
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
    NAIVE_BASELINE_JSON,
    RESULTS_DIR,
    VALID_VARIANTS_JSON,
)

print = functools.partial(print, flush=True)

OUT_DIR = RESULTS_DIR


def summarize_split_gap(
    seed_results: list[tuple[int, str, dict]],
    requested_seeds,
    feature: str = "wt_only_mean",
) -> dict:
    """Aggregate the row-aligned gene-minus-family point estimate by model seed."""
    gap_records = []
    for seed, filename, result in seed_results:
        family_result = result.get("family_split", {}).get(feature, {})
        paired_gap = family_result.get("split_gap_paired", {})
        point_difference = paired_gap.get("point_diff")
        # A feature block that failed keeps saying so; only a block that ran and
        # still produced no difference is unscorable.
        status = block_seed_status(family_result)
        if status == SEED_STATUS_SUCCESS and point_difference is None:
            status = SEED_STATUS_UNSCORABLE
        gap_records.append(make_seed_record(seed, point_difference, status=status))
    difference = aggregate_seed_values(requested_seeds, gap_records)
    return {
        "feature": feature,
        "gene_minus_family_seed_aggregate": difference.to_dict(),
        # Each seed's own paired bootstrap interval, kept per seed. Averaging
        # interval bounds across seeds would describe neither the within-seed
        # resampling uncertainty nor the spread between seeds.
        "per_seed_interval": [
            {
                "seed": seed,
                "source_file": filename,
                **{
                    key: result.get("family_split", {})
                    .get(feature, {})
                    .get("split_gap_paired", {})
                    .get(key)
                    for key in ("point_diff", "ci_low", "ci_high", "n_clusters")
                },
            }
            for seed, filename, result in seed_results
        ],
    }


def mechanism_null_assessment(family_chance_floor: float) -> dict:
    """Report the family-held-out chance floor and threshold with no verdict.

    The only interval available here is one seed's family-resampled bootstrap. It
    describes that seed, not the across-seed macro-F1 reported here, so it is
    neither attached to that estimate nor used to adjudicate it. The floor and
    threshold stay reportable because each satisfies its own contract.
    """
    return {
        "feature": "delta_mean",
        "family_chance_floor": family_chance_floor,
        "floor_margin": MECHANISM_NULL_FLOOR_MARGIN,
        "threshold": family_chance_floor + MECHANISM_NULL_FLOOR_MARGIN,
        "interval": None,
        "interval_reason": (
            "an interval for the across-seed macro-F1 is unavailable pending audit "
            "item 1.4; a single-seed bootstrap is not a substitute"
        ),
        "interval_dependent_verdict": None,
    }


def aggregate_permutation_results(
    seed_results: list[tuple[int, str, dict]],
    requested_seeds: Iterable[int],
) -> dict[str, dict]:
    """Collect the permutation distribution across seeds.

    The ordinary across-seed metric aggregator only reads flat ``*_mean`` values,
    so nested permutation results need an explicit path. A seed-vote decision
    is emitted only when every requested seed has a finite p-value; incomplete
    results remain visible and the decision is ``None`` rather than being treated
    as a negative result. The requested seeds come from the entry point, so a run
    of fewer or more seeds votes on the seeds it actually asked for.
    """
    summaries = {}
    requested_seeds = tuple(requested_seeds)
    for feature in PERMUTATION_FEATURES:
        per_seed = []
        vote_records = []
        for seed, filename, result in seed_results:
            permutation = (
                result.get("family_split", {}).get(feature, {}).get("permutation")
            )
            p_value = None if permutation is None else permutation.get("p_value")
            vote_records.append(make_seed_record(seed, p_value))
            record = {"seed": seed, "source_file": filename}
            if permutation is not None:
                record.update(permutation)
            per_seed.append(record)

        vote = aggregate_seed_vote(
            requested_seeds,
            vote_records,
            threshold=PERMUTATION_SIGNIFICANCE_THRESHOLD,
            minimum_supporting_seeds=PERMUTATION_MIN_SIGNIFICANT_SEEDS,
        )
        summaries[feature] = {
            "seed_vote": vote.to_dict(),
            "feature": feature,
            "per_seed": per_seed,
            "resolution_limited_seeds": [
                row["seed"] for row in per_seed if row.get("resolution_limited") is True
            ],
            "significance_threshold": PERMUTATION_SIGNIFICANCE_THRESHOLD,
            "required_seed_count": len(requested_seeds),
            "required_significant_seed_count": PERMUTATION_MIN_SIGNIFICANT_SEEDS,
            "permutation_rule_evaluable": vote.available,
            "meets_permutation_seed_vote_rule": (
                vote.payload["decision"] if vote.available else None
            ),
        }
    return summaries


def print_permutation_summary(summary: dict[str, dict]) -> None:
    """Print the full-seed permutation decision without hiding incomplete runs."""
    print("\n=== Permutation results across seeds ===")
    for feature in PERMUTATION_FEATURES:
        vote = summary[feature]["seed_vote"]
        if vote["state"] == "available":
            count = vote["payload"]["n_supporting_seeds"]
            verdict = (
                "criterion met" if vote["payload"]["decision"] else "criterion not met"
            )
            print(
                f"  {feature}: {count}/{len(vote['requested_seeds'])} p-values below "
                f"{PERMUTATION_SIGNIFICANCE_THRESHOLD} ({verdict}); "
                f"resolution-limited seeds "
                f"{summary[feature]['resolution_limited_seeds']}"
            )
        else:
            print(
                f"  {feature}: unavailable ({vote['reason']}), affected "
                f"seeds {vote['affected_seeds']}"
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
        [
            v["foldx_ddg"] if v["foldx_ddg"] is not None else np.nan
            for v in valid_variants
        ]
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
        "--seeds",
        type=seed_count,
        default=N_SEEDS,
        help="number of seeds to run; runs 0..seeds-1 (>=1)",
    )
    parser.add_argument(
        "--no_ci", action="store_true", help="skip cluster-bootstrap CIs"
    )
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    parser.add_argument(
        "--n_permutations",
        type=int,
        default=0,
        help="label-permutation reps for headline features (0 = skip; slow, refits per rep)",
    )
    args = parser.parse_args()
    requested_seeds = range(args.seeds)
    # The vote needs enough requested seeds to be satisfiable at all. Refuse here
    # rather than after the permutation refits have already run.
    if args.n_permutations > 0 and args.seeds < PERMUTATION_MIN_SIGNIFICANT_SEEDS:
        parser.error(
            f"--n_permutations needs at least {PERMUTATION_MIN_SIGNIFICANT_SEEDS} "
            f"seeds, because the vote requires that many to agree; got {args.seeds}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()

    print("\n=== Result 2: Gene-split vs family-split baselines ===")
    for seed in range(args.seeds):
        print(f"\n--- Seed {seed} ---")
        run_family_split(
            data=data,
            out_dir=str(OUT_DIR),
            seed=seed,
            compute_ci=not args.no_ci,
            n_boot=args.n_boot,
            n_permutations=args.n_permutations,
        )

    print("\n=== Aggregating across seeds ===")
    seed_results = load_seed_files(
        str(OUT_DIR), SEED_RESULT_GLOB, expected_seeds=requested_seeds
    )
    print(f"Loaded {len(seed_results)} seed files:")
    for _seed, filename, _result in seed_results:
        print(f"  {filename}")

    input_fingerprints = seed_results[0][2].get("input_fingerprints")
    analysis_parameters = seed_results[0][2].get("analysis_parameters")
    if input_fingerprints is None:
        raise ValueError("seed results lack mechanism input fingerprints")
    if analysis_parameters is None:
        raise ValueError("seed results lack mechanism analysis parameters")
    for seed, filename, result in seed_results:
        if result.get("input_fingerprints") != input_fingerprints:
            raise ValueError(
                f"{filename}: seed {seed} was produced from different inputs"
            )
        if result.get("analysis_parameters") != analysis_parameters:
            raise ValueError(
                f"{filename}: seed {seed} used different analysis parameters"
            )

    aggregated = aggregate_across_seeds(
        seed_results,
        requested_seeds,
        confusion_matrix_class_order=MECHANISM_CLASSES,
    )
    split_gap_summary = summarize_split_gap(seed_results, requested_seeds)
    permutation_summary = (
        aggregate_permutation_results(seed_results, requested_seeds)
        if args.n_permutations > 0
        else None
    )
    aggregate_payload = {
        **aggregate_result_contract(),
        "n_seeds": len(seed_results),
        "seed_files": [filename for _seed, filename, _result in seed_results],
        "input_fingerprints": input_fingerprints,
        "analysis_parameters": analysis_parameters,
        "across_seed": aggregated,
        "gene_minus_family_split_gap": split_gap_summary,
    }
    if NAIVE_BASELINE_JSON.exists():
        with open(NAIVE_BASELINE_JSON) as handle:
            naive_result = json.load(handle)
        naive_fingerprints = naive_result.get("input_fingerprints")
        matching_floor = naive_fingerprints is not None and all(
            naive_fingerprints.get(key) == input_fingerprints.get(key)
            for key in ("labeled_variants", "pfam_assignments")
        )
        if matching_floor:
            family_floor_read = read_seed_point_estimate(
                naive_result["by_strategy"]["most_frequent"]["family"][
                    "macro_f1_seed_aggregate"
                ]
            )
            if family_floor_read.available:
                aggregate_payload["mechanism_above_chance_family_held_out"] = (
                    mechanism_null_assessment(family_floor_read.value)
                )
            else:
                aggregate_payload["mechanism_above_chance_family_held_out"] = {
                    "interval": None,
                    "interval_reason": family_floor_read.message,
                    "interval_dependent_verdict": None,
                }
        else:
            aggregate_payload["mechanism_above_chance_family_held_out"] = {
                "interval": None,
                "interval_reason": (
                    "the available naive baseline was produced from different or "
                    "unfingerprinted inputs"
                ),
                "interval_dependent_verdict": None,
            }
    else:
        aggregate_payload["mechanism_above_chance_family_held_out"] = {
            "interval": None,
            "interval_reason": "the naive baseline result does not exist",
            "interval_dependent_verdict": None,
        }
    if permutation_summary is not None:
        aggregate_payload["permutation_summary"] = permutation_summary
    write_result_json(
        MECHANISM_AGGREGATE_JSON,
        aggregate_payload,
        seeds=list(range(args.seeds)),
    )
    print_table(aggregated)
    split_gap = read_seed_point_estimate(
        split_gap_summary["gene_minus_family_seed_aggregate"]
    )
    if split_gap.available:
        print(f"\nRow-aligned gene-minus-family macro-F1: " f"{split_gap.value:+.3f}")
    else:
        print(f"\nRow-aligned split gap is unavailable ({split_gap.message}).")
    if permutation_summary is not None:
        print_permutation_summary(permutation_summary)
    print(f"\nWrote {MECHANISM_AGGREGATE_JSON}")


if __name__ == "__main__":
    main()
