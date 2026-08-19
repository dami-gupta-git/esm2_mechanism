"""Test whether variant-centered WT windows change the Section 4 result.

For each UniProt protein, this analysis reconstructs every variant window start,
averages embeddings within identical windows, then averages the unique-window
vectors once each. The resulting protein vector is assigned to every variant from
that protein. It runs the original and window-averaged representations through the
same WT-only five-seed gene/family-split probe and compares their OOF macro-F1
scores with paired cluster bootstraps.

This uses existing embeddings. The window average is an observed-window
approximation, not a full-protein embedding for proteins longer than ESM-2's limit.

Usage:
    python -m esm2_mech.experiments.mechanism.wt_window_average_sensitivity \
        --seeds 5 --n_boot 1000
"""

from __future__ import annotations

import argparse
import functools
import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from esm2_mech.experiments.mechanism.classify_by_mechanism import (
    load_data,
    summarize_split_gap,
)
from esm2_mech.experiments.mechanism.mechanism_delta_family_split import (
    mechanism_input_fingerprints,
)
from esm2_mech.experiments.mechanism.mechanism_delta_family_split import (
    run as run_family_split,
)
from esm2_mech.utils.bootstrap import (
    family_or_gene_clusters,
    folds_to_arms,
    paired_cluster_bootstrap_diff,
    paired_cluster_bootstrap_diff_cross_partition,
    score_within_folds,
)
from esm2_mech.utils.constants import (
    BOOTSTRAP_N_RESAMPLES,
    MECHANISM_CLASSES,
    MECHANISM_OOF_CACHE_GLOB,
    MECHANISM_OOF_CACHE_SCHEMA_VERSION,
    N_FOLDS,
    N_SEEDS,
    SEED_RESULT_GLOB,
    mechanism_oof_cache_filename,
    seed_result_filename,
)
from esm2_mech.utils.data import embedding_fingerprint, load_pfam_map
from esm2_mech.utils.io import write_result_json
from esm2_mech.utils.metrics import fold_macro_f1
from esm2_mech.utils.paths import (
    EMB_WT_MEAN,
    PFAM_JSON,
    SEQUENCES_JSON,
    VALID_VARIANTS_JSON,
    WT_WINDOW_AVERAGE_AGGREGATE_JSON,
    WT_WINDOW_AVERAGE_CANONICAL_DIR,
    WT_WINDOW_AVERAGE_DIR,
    WT_WINDOW_AVERAGE_ORIGINAL_DIR,
)
from esm2_mech.utils.seed_aggregation import (
    aggregate_across_seeds,
    load_seed_files,
    print_table,
)
from esm2_mech.utils.sequences import window_sequence

print = functools.partial(print, flush=True)

WT_ONLY_FEATURE = "wt_only_mean"
CONDITION_ORIGINAL = "variant_centered"
CONDITION_AVERAGED = "protein_window_average"
CACHED_ARM_FIELDS = ("row_ids", "y_true", "pred", "genes", "folds")


def build_protein_window_average(
    variants: list[dict], sequences: dict[str, str], wt_embeddings: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Assign every UniProt protein the equal-weight mean of its unique windows."""
    if wt_embeddings.ndim != 2 or wt_embeddings.shape[0] != len(variants):
        raise ValueError(
            "WT embedding rows must align to variants: "
            f"{wt_embeddings.shape} for {len(variants)} variants"
        )
    if not np.isfinite(wt_embeddings).all():
        raise ValueError("WT embeddings contain non-finite values")

    rows_by_protein_window: dict[str, dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    rows_by_protein: dict[str, list[int]] = defaultdict(list)
    for row_index, variant in enumerate(variants):
        uniprot_id = variant.get("uniprot_id")
        if not uniprot_id:
            raise ValueError(f"variant row {row_index} has no uniprot_id")
        sequence = sequences.get(uniprot_id)
        if sequence is None:
            raise ValueError(
                f"sequence cache lacks {uniprot_id!r} from valid variant row {row_index}"
            )
        aa_pos = int(variant["aa_pos"])
        if aa_pos < 1 or aa_pos > len(sequence):
            raise ValueError(
                f"variant row {row_index} position {aa_pos} is outside "
                f"{uniprot_id} length {len(sequence)}"
            )
        _window, _new_pos, window_start = window_sequence(sequence, aa_pos)
        rows_by_protein_window[uniprot_id][window_start].append(row_index)
        rows_by_protein[uniprot_id].append(row_index)

    averaged = np.empty_like(wt_embeddings)
    unique_window_counts = Counter()
    rows_with_multiple_windows = 0
    for uniprot_id, window_rows in rows_by_protein_window.items():
        per_window_vectors = [
            wt_embeddings[row_indices].mean(axis=0)
            for _start, row_indices in sorted(window_rows.items())
        ]
        protein_vector = np.stack(per_window_vectors).mean(axis=0)
        protein_rows = rows_by_protein[uniprot_id]
        averaged[protein_rows] = protein_vector
        n_unique_windows = len(window_rows)
        unique_window_counts[n_unique_windows] += 1
        if n_unique_windows > 1:
            rows_with_multiple_windows += len(protein_rows)

    metadata = {
        "n_variants": len(variants),
        "n_uniprot_proteins": len(rows_by_protein),
        "n_proteins_with_multiple_observed_windows": sum(
            count for n_windows, count in unique_window_counts.items() if n_windows > 1
        ),
        "n_variants_in_multiwindow_proteins": rows_with_multiple_windows,
        "unique_window_count_distribution": {
            str(n_windows): count
            for n_windows, count in sorted(unique_window_counts.items())
        },
        "original_wt_embedding_fingerprint": embedding_fingerprint(wt_embeddings),
        "averaged_wt_embedding_fingerprint": embedding_fingerprint(averaged),
    }
    return averaged, metadata


def _clear_condition_outputs(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    stale = []
    for output_glob in (SEED_RESULT_GLOB, MECHANISM_OOF_CACHE_GLOB):
        stale.extend(glob.glob(os.path.join(str(directory), output_glob)))
    for path in stale:
        os.remove(path)
    if stale:
        print(f"Removed {len(stale)} stale file(s) from {directory}")


def _load_validated_condition(
    directory: Path, expected_seeds: range
) -> tuple[list[tuple[int, str, dict]], dict[int, dict]]:
    seed_results = load_seed_files(
        str(directory), SEED_RESULT_GLOB, expected_seeds=expected_seeds
    )
    oof_by_seed = {}
    for seed, _filename, result in seed_results:
        cache_path = directory / mechanism_oof_cache_filename(seed)
        if not cache_path.exists():
            raise FileNotFoundError(f"missing required OOF cache {cache_path}")
        with open(cache_path) as handle:
            cache = json.load(handle)
        expected = {
            "cache_schema_version": MECHANISM_OOF_CACHE_SCHEMA_VERSION,
            "seed": seed,
            "analysis_run_id": result.get("analysis_run_id"),
            "input_fingerprints": result.get("input_fingerprints"),
            "analysis_parameters": result.get("analysis_parameters"),
        }
        for key, expected_value in expected.items():
            if expected_value is None or cache.get(key) != expected_value:
                raise ValueError(
                    f"{cache_path}: cache {key} does not match "
                    f"{directory / seed_result_filename(seed)}"
                )
        feature = cache.get("features", {}).get(WT_ONLY_FEATURE)
        if not isinstance(feature, dict):
            raise TypeError(f"{cache_path} lacks {WT_ONLY_FEATURE} OOF predictions")
        for split in ("gene_split", "family_split"):
            arm = feature.get(split)
            if arm is None or not set(CACHED_ARM_FIELDS).issubset(arm):
                raise ValueError(
                    f"{cache_path} lacks {WT_ONLY_FEATURE}/{split} fields "
                    f"{list(CACHED_ARM_FIELDS)}"
                )
            lengths = {field: len(arm[field]) for field in CACHED_ARM_FIELDS}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"{cache_path} has misaligned {split} fields: {lengths}")
            if len({int(row) for row in arm["row_ids"]}) != lengths["row_ids"]:
                raise ValueError(f"{cache_path} has duplicate {split} row ids")
        oof_by_seed[seed] = feature
    return seed_results, oof_by_seed


def _align_cached_arms(arms: dict[str, dict], label: str) -> dict[str, dict]:
    """Restrict cached prediction arms to one validated row-id intersection."""
    row_maps = {
        name: {int(row): pos for pos, row in enumerate(arm["row_ids"])}
        for name, arm in arms.items()
    }
    shared_rows = sorted(set.intersection(*(set(rows) for rows in row_maps.values())))
    if not shared_rows:
        raise ValueError(f"{label}: cached arms share no OOF rows")

    aligned = {}
    for name, arm in arms.items():
        indices = np.array([row_maps[name][row] for row in shared_rows], dtype=int)
        aligned[name] = {
            field: np.asarray(arm[field])[indices] for field in CACHED_ARM_FIELDS
        }

    reference = next(iter(aligned.values()))
    for name, arm in aligned.items():
        if not np.array_equal(arm["y_true"], reference["y_true"]):
            raise ValueError(f"{label}: arm {name} disagrees on labels")
        if not np.array_equal(arm["genes"], reference["genes"]):
            raise ValueError(f"{label}: arm {name} disagrees on genes")
    return aligned


def _macro_f1_scorer(arm: dict):
    y_true = np.asarray(arm["y_true"])
    prediction_arms = folds_to_arms(np.asarray(arm["pred"]), np.asarray(arm["folds"]))

    def _fold_score(block: np.ndarray, predictions: np.ndarray) -> float | None:
        return fold_macro_f1(y_true, block, predictions, MECHANISM_CLASSES)

    def _score(rows: np.ndarray) -> float | None:
        return score_within_folds(rows, prediction_arms, _fold_score)

    return _score


def compare_conditions_for_seed(
    original: dict,
    averaged: dict,
    pfam_map: dict,
    *,
    n_resamples: int,
    seed: int,
    n_jobs: int = -1,
) -> dict:
    """Paired OOF comparisons for score levels and the split-gap change."""
    output = {}
    for split, is_family_split in (("gene_split", False), ("family_split", True)):
        aligned = _align_cached_arms(
            {"averaged": averaged[split], "original": original[split]},
            f"{split} averaged versus original",
        )
        genes = np.asarray(aligned["averaged"]["genes"], dtype=object)
        clusters = family_or_gene_clusters(genes, pfam_map, is_family_split)
        comparison = paired_cluster_bootstrap_diff(
            clusters,
            _macro_f1_scorer(aligned["averaged"]),
            _macro_f1_scorer(aligned["original"]),
            n_resamples=n_resamples,
            seed=seed,
            n_jobs=n_jobs,
            discard_reason="a fold lost a mechanism class in either representation",
        )
        comparison["n_shared"] = len(genes)
        comparison["contrast"] = "protein_window_average_minus_variant_centered"
        output[f"{split}_method_difference"] = comparison

    gap_arms = _align_cached_arms(
        {
            "averaged_gene": averaged["gene_split"],
            "averaged_family": averaged["family_split"],
            "original_gene": original["gene_split"],
            "original_family": original["family_split"],
        },
        "change in gene-minus-family gap",
    )
    genes = np.asarray(gap_arms["averaged_gene"]["genes"], dtype=object)
    scorers = {name: _macro_f1_scorer(arm) for name, arm in gap_arms.items()}

    def _averaged_gap(rows: np.ndarray) -> float | None:
        gene_value = scorers["averaged_gene"](rows)
        family_value = scorers["averaged_family"](rows)
        if gene_value is None or family_value is None:
            return None
        return gene_value - family_value

    def _original_gap(rows: np.ndarray) -> float | None:
        gene_value = scorers["original_gene"](rows)
        family_value = scorers["original_family"](rows)
        if gene_value is None or family_value is None:
            return None
        return gene_value - family_value

    gap_comparison = paired_cluster_bootstrap_diff_cross_partition(
        family_or_gene_clusters(genes, pfam_map, is_family_split=True),
        _averaged_gap,
        _original_gap,
        sensitivity_clusters=genes,
        n_resamples=n_resamples,
        seed=seed,
        n_jobs=n_jobs,
        discard_reason=(
            "a gene- or family-split fold lost a mechanism class in either representation"
        ),
    )
    gap_comparison["n_shared"] = len(genes)
    gap_comparison["contrast"] = (
        "(protein_window_average_gene_minus_family)_minus_"
        "(variant_centered_gene_minus_family)"
    )
    output["split_gap_method_difference"] = gap_comparison
    return output


def _comparison_summary(per_seed: list[dict]) -> dict:
    summary = {}
    for comparison_name in (
        "gene_split_method_difference",
        "family_split_method_difference",
        "split_gap_method_difference",
    ):
        valid = []
        positive = []
        negative = []
        for row in per_seed:
            result = row[comparison_name]
            if result.get("ci_low") is None or result.get("ci_high") is None:
                continue
            valid.append(row["seed"])
            if result["ci_low"] > 0:
                positive.append(row["seed"])
            if result["ci_high"] < 0:
                negative.append(row["seed"])
        summary[comparison_name] = {
            "valid_ci_seeds": valid,
            "positive_difference_seeds": positive,
            "negative_difference_seeds": negative,
            "note": "Post hoc sensitivity; no across-seed decision rule was preregistered.",
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", type=int, default=N_SEEDS,
        help="number of seeds to run; runs 0..seeds-1 (>=1)",
    )
    parser.add_argument("--n_folds", type=int, default=N_FOLDS)
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")
    if args.n_folds < 2:
        parser.error("--n_folds must be >= 2")
    if args.n_boot < 1:
        parser.error("--n_boot must be >= 1")

    with open(SEQUENCES_JSON) as handle:
        sequences = json.load(handle)
    data = load_data()
    pfam_map = load_pfam_map(PFAM_JSON)
    averaged_wt, averaging_metadata = build_protein_window_average(
        data["valid_variants"], sequences, data["emb_wt_mean"]
    )
    averaged_data = {**data, "emb_wt_mean": averaged_wt}
    original_fingerprints = mechanism_input_fingerprints(data, pfam_map)
    averaged_fingerprints = mechanism_input_fingerprints(averaged_data, pfam_map)

    WT_WINDOW_AVERAGE_DIR.mkdir(parents=True, exist_ok=True)
    for directory in (WT_WINDOW_AVERAGE_ORIGINAL_DIR, WT_WINDOW_AVERAGE_CANONICAL_DIR):
        _clear_condition_outputs(directory)

    conditions = (
        (
            CONDITION_ORIGINAL,
            data,
            WT_WINDOW_AVERAGE_ORIGINAL_DIR,
            original_fingerprints,
        ),
        (
            CONDITION_AVERAGED,
            averaged_data,
            WT_WINDOW_AVERAGE_CANONICAL_DIR,
            averaged_fingerprints,
        ),
    )
    for condition_name, condition_data, output_dir, fingerprints in conditions:
        print(f"\n=== {condition_name} ===")
        for seed in range(args.seeds):
            print(f"\n--- Seed {seed} ---")
            run_family_split(
                data=condition_data,
                out_dir=str(output_dir),
                seed=seed,
                n_folds=args.n_folds,
                compute_ci=True,
                n_boot=args.n_boot,
                n_permutations=0,
                feature_names=(WT_ONLY_FEATURE,),
                input_fingerprints=fingerprints,
            )

    expected_seeds = range(args.seeds)
    original_results, original_oof = _load_validated_condition(
        WT_WINDOW_AVERAGE_ORIGINAL_DIR, expected_seeds
    )
    averaged_results, averaged_oof = _load_validated_condition(
        WT_WINDOW_AVERAGE_CANONICAL_DIR, expected_seeds
    )
    original_aggregate = aggregate_across_seeds(original_results)
    averaged_aggregate = aggregate_across_seeds(averaged_results)
    original_gap = summarize_split_gap(original_results)
    averaged_gap = summarize_split_gap(averaged_results)
    original_result_by_seed = {
        seed: result for seed, _filename, result in original_results
    }
    averaged_result_by_seed = {
        seed: result for seed, _filename, result in averaged_results
    }

    per_seed_comparisons = []
    for seed in expected_seeds:
        comparison = compare_conditions_for_seed(
            original_oof[seed],
            averaged_oof[seed],
            pfam_map,
            n_resamples=args.n_boot,
            seed=seed,
        )
        comparison["seed"] = seed
        per_seed_comparisons.append(comparison)

        stored_original_gap = (
            original_result_by_seed[seed]["family_split"][WT_ONLY_FEATURE][
                "split_gap_paired"
            ]["point_diff"]
        )
        stored_averaged_gap = (
            averaged_result_by_seed[seed]["family_split"][WT_ONLY_FEATURE][
                "split_gap_paired"
            ]["point_diff"]
        )
        gap_comparison = comparison["split_gap_method_difference"]
        if not np.isclose(gap_comparison["point_a"], stored_averaged_gap):
            raise RuntimeError(f"seed {seed}: averaged split-gap estimands disagree")
        if not np.isclose(gap_comparison["point_b"], stored_original_gap):
            raise RuntimeError(f"seed {seed}: original split-gap estimands disagree")

    result = {
        "analysis": "WT window-average sensitivity on the full Section 4 row set",
        "interpretation_limit": (
            "The protein vector averages only variant-observed windows and is not "
            "a full-protein embedding for sequences longer than ESM-2's limit."
        ),
        "method": {
            "window_weighting": "equal weight per unique reconstructed window start",
            "duplicate_window_weighting": (
                "rows with the same UniProt ID and window start are averaged first"
            ),
            **averaging_metadata,
        },
        "inputs": {
            "valid_variants_path": str(VALID_VARIANTS_JSON),
            "sequences_path": str(SEQUENCES_JSON),
            "wt_embedding_path": str(EMB_WT_MEAN),
            "original_input_fingerprints": original_fingerprints,
            "averaged_input_fingerprints": averaged_fingerprints,
        },
        "n_seeds": args.seeds,
        "n_folds": args.n_folds,
        "n_bootstrap_resamples": args.n_boot,
        "class_distribution": dict(Counter(data["labels_3class"])),
        "conditions": {
            CONDITION_ORIGINAL: {
                "output_dir": str(WT_WINDOW_AVERAGE_ORIGINAL_DIR),
                "across_seed": original_aggregate,
                "split_gap_summary": original_gap,
            },
            CONDITION_AVERAGED: {
                "output_dir": str(WT_WINDOW_AVERAGE_CANONICAL_DIR),
                "across_seed": averaged_aggregate,
                "split_gap_summary": averaged_gap,
            },
        },
        "paired_method_comparisons": {
            "contrast_direction": "protein_window_average minus variant_centered",
            "per_seed": per_seed_comparisons,
            "summary": _comparison_summary(per_seed_comparisons),
        },
        "section_4_conclusion_changed": (
            original_gap["meets_claim_2b_interval_rule"]
            != averaged_gap["meets_claim_2b_interval_rule"]
            if original_gap["preregistered_rule_evaluable"]
            and averaged_gap["preregistered_rule_evaluable"]
            else None
        ),
    }
    write_result_json(
        WT_WINDOW_AVERAGE_AGGREGATE_JSON,
        result,
        seeds=list(expected_seeds),
        indent=2,
    )

    print("\n=== Variant-centered WT ===")
    print_table(original_aggregate)
    print("\n=== Protein window-average WT ===")
    print_table(averaged_aggregate)
    print(f"\nWrote {WT_WINDOW_AVERAGE_AGGREGATE_JSON}")


if __name__ == "__main__":
    main()
