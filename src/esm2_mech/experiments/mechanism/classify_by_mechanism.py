"""Classify variants by mechanism (GOF/DN/LOF) using ESM-2 embeddings under gene-split and family-split CV."""

import argparse
import functools
import json
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
    aggregate_seed_vote,
    load_seed_files,
    make_seed_record,
)
from esm2_mech.experiments.mechanism.seed_results import (
    aggregate_across_seeds,
    aggregate_result_contract,
    print_table,
)
from esm2_mech.utils.constants import (
    BOOTSTRAP_N_RESAMPLES,
    MECHANISM_NULL_FLOOR_MARGIN,
    MECHANISM_NULL_MIN_AFFIRMING_SEEDS,
    MECHANISM_CLASSES,
    N_SEEDS,
    PERMUTATION_MIN_SIGNIFICANT_SEEDS,
    PERMUTATION_SIGNIFICANCE_THRESHOLD,
    SEED_RESULT_GLOB,
    SPLIT_GAP_MIN_SUPPORTING_SEEDS,
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
    seed_results: list[tuple[int, str, dict]], feature: str = "wt_only_mean"
) -> dict:
    """Apply claim 2B's positive-gap interval rule across all five seeds."""
    per_seed = []
    contradictory_seeds = []
    vote_records = []
    for seed, filename, result in seed_results:
        gap = result.get("family_split", {}).get(feature, {}).get("split_gap_paired")
        if gap is None or gap.get("ci_low") is None or gap.get("ci_high") is None:
            vote_records.append(make_seed_record(seed, None))
            per_seed.append({"seed": seed, "source_file": filename, "split_gap_paired": gap})
            continue
        contradicts_positive_gap = gap["ci_high"] < 0
        vote_records.append(make_seed_record(seed, gap["ci_low"]))
        if contradicts_positive_gap:
            contradictory_seeds.append(seed)
        per_seed.append({
            "seed": seed,
            "source_file": filename,
            "point_diff": gap.get("point_diff"),
            "ci_low": gap["ci_low"],
            "ci_high": gap["ci_high"],
            "n_clusters": gap.get("n_clusters"),
            "contradicts_positive_gene_minus_family_gap": contradicts_positive_gap,
        })

    vote = aggregate_seed_vote(
        range(N_SEEDS),
        vote_records,
        threshold=0.0,
        minimum_supporting_seeds=SPLIT_GAP_MIN_SUPPORTING_SEEDS,
        comparison="greater_than",
    )
    return {
        "seed_vote": vote.to_dict(),
        "feature": feature,
        "per_seed": per_seed,
        "contradictory_seeds": contradictory_seeds,
        "required_seed_count": N_SEEDS,
        "required_supporting_seed_count": SPLIT_GAP_MIN_SUPPORTING_SEEDS,
        "preregistered_rule_evaluable": vote.available,
        "meets_claim_2b_interval_rule": (
            vote.payload["decision"] if vote.available else None
        ),
    }


def summarize_mechanism_null_ci(
    seed_results: list[tuple[int, str, dict]], family_chance_floor: float
) -> dict:
    """Apply claim 2A's three-of-five rule to its seed-specific intervals."""
    threshold = family_chance_floor + MECHANISM_NULL_FLOOR_MARGIN
    per_seed = []
    vote_records = []
    for seed, filename, result in seed_results:
        ci = (
            result.get("family_split", {})
            .get("delta_mean", {})
            .get("ci", {})
            .get("macro_f1")
        )
        if ci is None or ci.get("ci_low") is None or ci.get("ci_high") is None:
            vote_records.append(make_seed_record(seed, None))
            per_seed.append({"seed": seed, "source_file": filename, "ci": ci})
            continue
        vote_records.append(make_seed_record(seed, ci["ci_high"]))
        per_seed.append({
            "seed": seed,
            "source_file": filename,
            "point": ci.get("point"),
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            "n_clusters": ci.get("n_clusters"),
        })
    vote = aggregate_seed_vote(
        range(N_SEEDS),
        vote_records,
        threshold=threshold,
        minimum_supporting_seeds=MECHANISM_NULL_MIN_AFFIRMING_SEEDS,
    )
    return {
        "seed_vote": vote.to_dict(),
        "feature": "delta_mean",
        "family_chance_floor": family_chance_floor,
        "floor_margin": MECHANISM_NULL_FLOOR_MARGIN,
        "threshold": threshold,
        "per_seed": per_seed,
        "required_seed_count": N_SEEDS,
        "required_affirming_seed_count": MECHANISM_NULL_MIN_AFFIRMING_SEEDS,
        "preregistered_rule_evaluable": vote.available,
        "meets_claim_2a_interval_rule": (
            vote.payload["decision"] if vote.available else None
        ),
    }


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
    requested_seeds = tuple(range(N_SEEDS))
    for feature in PERMUTATION_FEATURES:
        per_seed = []
        vote_records = []
        for seed, filename, result in seed_results:
            permutation = (
                result.get("family_split", {})
                .get(feature, {})
                .get("permutation")
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
            "required_seed_count": N_SEEDS,
            "required_significant_seed_count": PERMUTATION_MIN_SIGNIFICANT_SEEDS,
            "preregistered_rule_evaluable": vote.available,
            "meets_preregistered_three_of_five_rule": (
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
                "criterion met"
                if vote["payload"]["decision"]
                else "criterion not met"
            )
            print(
                f"  {feature}: {count}/{N_SEEDS} p-values below "
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
    requested_seeds = range(args.seeds)

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
        str(OUT_DIR), SEED_RESULT_GLOB, expected_seeds=requested_seeds
    )
    print(f"Loaded {len(seed_results)} seed files:")
    for _seed, filename, _result in seed_results:
        print(f"  {filename}")

    input_fingerprints = seed_results[0][2].get("input_fingerprints")
    analysis_parameters = seed_results[0][2].get("analysis_parameters")
    if input_fingerprints is None:
        raise ValueError("seed results lack Section 4 input fingerprints")
    if analysis_parameters is None:
        raise ValueError("seed results lack Section 4 analysis parameters")
    for seed, filename, result in seed_results:
        if result.get("input_fingerprints") != input_fingerprints:
            raise ValueError(f"{filename}: seed {seed} was produced from different inputs")
        if result.get("analysis_parameters") != analysis_parameters:
            raise ValueError(f"{filename}: seed {seed} used different analysis parameters")

    aggregated = aggregate_across_seeds(
        seed_results,
        requested_seeds,
        confusion_matrix_class_order=MECHANISM_CLASSES,
    )
    split_gap_summary = summarize_split_gap(seed_results)
    permutation_summary = (
        aggregate_permutation_results(seed_results)
        if args.n_permutations > 0 else None
    )
    aggregate_payload = {
        **aggregate_result_contract(),
        "n_seeds": len(seed_results),
        "seed_files": [filename for _seed, filename, _result in seed_results],
        "input_fingerprints": input_fingerprints,
        "analysis_parameters": analysis_parameters,
        "across_seed": aggregated,
        "claim_2b_split_gap_summary": split_gap_summary,
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
            family_floor = float(
                naive_result["by_strategy"]["most_frequent"]["family"]["macro_f1_mean"]
            )
            aggregate_payload["claim_2a_ci_summary"] = summarize_mechanism_null_ci(
                seed_results, family_floor
            )
            aggregate_payload["claim_2a_ci_summary_missing"] = False
            aggregate_payload["claim_2a_ci_summary_missing_reason"] = None
        else:
            aggregate_payload["claim_2a_ci_summary"] = None
            aggregate_payload["claim_2a_ci_summary_missing"] = True
            aggregate_payload["claim_2a_ci_summary_missing_reason"] = (
                "the available naive baseline was produced from different or "
                "unfingerprinted inputs"
            )
    else:
        aggregate_payload["claim_2a_ci_summary"] = None
        aggregate_payload["claim_2a_ci_summary_missing"] = True
        aggregate_payload["claim_2a_ci_summary_missing_reason"] = (
            "the naive baseline result does not exist"
        )
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
