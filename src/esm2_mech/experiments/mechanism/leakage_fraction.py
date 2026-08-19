"""Compute the leakage fraction: how much of each feature's gene-split signal
is family recognition rather than transferable mechanism signal.
"""

from __future__ import annotations

import argparse
import functools
import json
import os

import numpy as np

from esm2_mech.utils.bootstrap import (
    BootstrapMetricResult,
    cluster_bootstrap_ci,
    folds_to_arms,
    score_within_folds,
)
from esm2_mech.utils.constants import (
    BOOTSTRAP_MAX_DISCARD_FRAC,
    BOOTSTRAP_N_RESAMPLES,
    MECHANISM_CLASSES,
    MECHANISM_OOF_CACHE_SCHEMA_VERSION,
    N_SEEDS,
    SEED_RESULT_GLOB,
    mechanism_oof_cache_filename,
)
from esm2_mech.utils.data import load_pfam_map
from esm2_mech.utils.io import write_result_json
from esm2_mech.utils.metrics import fold_macro_f1, mean_std_n
from esm2_mech.utils.paths import (
    FAMILY_CLUSTERING_JSON,
    LEAKAGE_FRACTION_JSON,
    NAIVE_BASELINE_JSON,
    PFAM_JSON,
    RESULTS_DIR,
)
from esm2_mech.utils.seed_aggregation import load_seed_files

print = functools.partial(print, flush=True)

# Below this margin the denominator is ~0 and the ratio is noise.
MIN_ABOVE_CHANCE = 0.02


def _measured_chance_result():
    """Load the full-cohort majority-class result for provenance and comparison."""
    with open(NAIVE_BASELINE_JSON) as fh:
        return json.load(fh)


def leakage_fraction_per_feature(feature, chance, oof_cache_entries=None):
    """Leakage fraction for one feature, from seed-averaged macro-F1.

    The gene-split arm's macro-F1 in the per-seed baseline files is scored over
    every variant, while the family-split arm's is scored only over the rows whose
    gene has a Pfam annotation (family_split_cv excludes the rest). Reading both
    straight from the baseline files therefore compares two arms over different row
    sets. When this feature's OOF cache is available, both arms are instead
    rescored on the rows they share — the same row-alignment leakage_fraction_ci
    already uses — so the headline and the interval describe the same quantity.
    A cache is required. Falling back to the per-seed files would compare arms
    scored on different row sets and would therefore be a different estimand.
    """
    if oof_cache_entries is None:
        raise ValueError(f"{feature}: OOF caches are required for leakage analysis")
    aligned = _align_seed_arms(oof_cache_entries)
    if aligned is None:
        raise ValueError(f"{feature}: seed OOF caches have no shared scored rows")
    per_seed, _ = aligned
    n_shared = len(per_seed[0]["gene"]["y_true"])
    n_total = len(oof_cache_entries[0]["gene_split"]["row_ids"])
    n_excluded = n_total - n_shared
    rows = np.arange(n_shared)
    gene_f1 = [_arm_macro_f1(seed_arms["gene"], rows) for seed_arms in per_seed]
    family_f1 = [_arm_macro_f1(seed_arms["family"], rows) for seed_arms in per_seed]

    gene_mean, gene_std, gene_n = mean_std_n(gene_f1)
    family_mean, family_std, _ = mean_std_n(family_f1)

    result = {
        "gene_macro_f1_mean": gene_mean,
        "gene_macro_f1_std": gene_std,
        "family_macro_f1_mean": family_mean,
        "family_macro_f1_std": family_std,
        "drop_mean": gene_mean - family_mean,
    }
    result["n_excluded_unannotated"] = n_excluded
    result["chance_macro_f1"] = chance
    if gene_n == 0:
        result["leakage_fraction"] = None
        result["note"] = "no scorable gene-split seed; LF undefined"
        return result
    denom = gene_mean - chance
    if denom > MIN_ABOVE_CHANCE:
        result["leakage_fraction"] = (gene_mean - family_mean) / denom
    else:
        result["leakage_fraction"] = None
        result["note"] = "gene-split score not meaningfully above chance; LF undefined"
    return result


def aligned_majority_chance(oof_cache_entries: list[dict]) -> float:
    """Five-seed majority floor on the same rows and folds as one feature."""
    aligned = _align_seed_arms(oof_cache_entries)
    if aligned is None:
        raise ValueError("cannot compute an aligned chance floor without shared OOF rows")
    per_seed, _ = aligned
    per_seed_floor = []
    for seed_arms in per_seed:
        gene_arm = seed_arms["gene"]
        y_true = np.asarray(gene_arm["y_true"])
        folds = np.asarray(gene_arm["folds"])
        predictions = np.empty(len(y_true), dtype=object)
        for fold in np.unique(folds):
            test_mask = folds == fold
            train_labels = y_true[~test_mask]
            classes, counts = np.unique(train_labels, return_counts=True)
            if set(classes.tolist()) != set(MECHANISM_CLASSES):
                raise RuntimeError(
                    "aligned majority floor training rows lost a mechanism class"
                )
            predictions[test_mask] = classes[int(np.argmax(counts))]
        floor_arm = {"y_true": y_true, "pred": predictions, "folds": folds}
        value = _arm_macro_f1(floor_arm, np.arange(len(y_true)))
        if value is None:
            raise RuntimeError(
                "aligned majority floor cannot score every mechanism class in every fold"
            )
        per_seed_floor.append(value)
    return float(np.mean(per_seed_floor))


def _load_validated_oof_caches(seed_results: list[tuple[int, str, dict]]) -> list[dict]:
    """Load caches that belong to the exact executions producing the seed results."""
    caches = []
    for seed, _filename, result in seed_results:
        path = os.path.join(RESULTS_DIR, mechanism_oof_cache_filename(seed))
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"missing required OOF cache {path}; rerun Section 4.1 with CIs enabled"
            )
        with open(path) as handle:
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
                    f"{path}: cache {key} does not match seed {seed}'s result file"
                )
        features = cache.get("features")
        if not isinstance(features, dict) or not features:
            raise ValueError(f"{path}: cache has no feature OOF predictions")
        caches.append(features)
    return caches


def _arm_macro_f1(arm: dict, rows: np.ndarray) -> float | None:
    """Fold-averaged macro-F1 for one cached arm on a set of resampled rows."""
    y_true = np.asarray(arm["y_true"])
    pred = np.asarray(arm["pred"])
    arms = folds_to_arms(pred, np.asarray(arm["folds"]))

    def _fold_f1(block: np.ndarray, arm_pred: np.ndarray) -> float | None:
        return fold_macro_f1(y_true, block, arm_pred, MECHANISM_CLASSES)

    return score_within_folds(rows, arms, _fold_f1)


def _align_seed_arms(oof_cache_entries: list[dict]) -> tuple[list[dict], np.ndarray] | None:
    """Restrict every seed's two arms to the variants all of them scored.

    Each seed reshuffles the folds, so a variant can be scored by one seed's split and
    dropped by another's. Resampling has to index one shared row space, and the gene
    names used for clustering have to come from that same space.
    """
    per_seed = []
    for seed_index, entry in enumerate(oof_cache_entries):
        gene_arm, family_arm = entry["gene_split"], entry["family_split"]
        for arm_name, arm in (("gene", gene_arm), ("family", family_arm)):
            lengths = {
                key: len(arm[key])
                for key in ("row_ids", "y_true", "pred", "genes", "folds")
            }
            if len(set(lengths.values())) != 1:
                raise ValueError(
                    f"seed cache {seed_index} {arm_name} arm has misaligned fields {lengths}"
                )
            if len(set(int(row) for row in arm["row_ids"])) != lengths["row_ids"]:
                raise ValueError(
                    f"seed cache {seed_index} {arm_name} arm has duplicate row ids"
                )
        per_seed.append((
            {int(row): pos for pos, row in enumerate(gene_arm["row_ids"])},
            {int(row): pos for pos, row in enumerate(family_arm["row_ids"])},
            gene_arm,
            family_arm,
        ))
    shared = sorted(
        set.intersection(*[set(g) & set(f) for g, f, _, _ in per_seed])
    )
    if not shared:
        return None

    aligned = []
    reference_labels = None
    reference_genes = None
    for seed_index, (gene_pos, family_pos, gene_arm, family_arm) in enumerate(per_seed):
        gene_idx = np.array([gene_pos[row] for row in shared], dtype=int)
        family_idx = np.array([family_pos[row] for row in shared], dtype=int)
        gene_labels = np.asarray(gene_arm["y_true"])[gene_idx]
        family_labels = np.asarray(family_arm["y_true"])[family_idx]
        gene_names = np.asarray(gene_arm["genes"])[gene_idx]
        family_genes = np.asarray(family_arm["genes"])[family_idx]
        if not np.array_equal(gene_labels, family_labels):
            raise ValueError(f"seed cache {seed_index} arms disagree on shared labels")
        if not np.array_equal(gene_names, family_genes):
            raise ValueError(f"seed cache {seed_index} arms disagree on shared genes")
        if reference_labels is None:
            reference_labels = gene_labels
            reference_genes = gene_names
        elif not np.array_equal(reference_labels, gene_labels) or not np.array_equal(
            reference_genes, gene_names
        ):
            raise ValueError(
                f"seed cache {seed_index} does not describe the same shared variants"
            )
        aligned.append({
            "gene": {key: np.asarray(gene_arm[key])[gene_idx] for key in
                     ("y_true", "pred", "folds")},
            "family": {key: np.asarray(family_arm[key])[family_idx] for key in
                       ("y_true", "pred", "folds")},
        })
    return aligned, reference_genes


def leakage_fraction_ci(
    oof_cache_entries,
    pfam_map,
    chance,
    n_resamples,
    seed=0,
    metric_name="leakage_fraction",
):
    """Cluster-bootstrap CI on the leakage fraction, matching the headline exactly.

    The headline averages the per-fold macro-F1 of both arms over every seed and
    divides by the distance from a floor taken once from the naive baseline. This
    computes the same expression on each resample: same seeds, same per-fold basis,
    same fixed floor. Previously the headline used five seeds and a fixed floor while
    the interval used one seed and recomputed the floor on every draw, so the interval
    did not bracket the number it was printed beside.
    """
    aligned = _align_seed_arms(oof_cache_entries)
    if aligned is None:
        return None
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    per_seed, gene_names = aligned
    # Cluster on Pfam family; orphan genes become singleton clusters.
    clusters = np.array([pfam_map.get(g) or f"__orphan__{g}" for g in gene_names])
    all_rows = np.arange(len(gene_names))

    def _ratio(rows):
        gene_values, family_values = [], []
        for arms in per_seed:
            gene_f1 = _arm_macro_f1(arms["gene"], rows)
            family_f1 = _arm_macro_f1(arms["family"], rows)
            if gene_f1 is None or family_f1 is None:
                return BootstrapMetricResult(None, "fold_lost_class")
            gene_values.append(gene_f1)
            family_values.append(family_f1)
        gene_mean = float(np.mean(gene_values))
        family_mean = float(np.mean(family_values))
        denom = gene_mean - chance
        if denom <= MIN_ABOVE_CHANCE:
            return BootstrapMetricResult(None, "denominator_below_threshold")
        return BootstrapMetricResult((gene_mean - family_mean) / denom)

    # The headline ratio (full data, no resampling) is undefined for a feature
    # at the chance floor. Bootstrapping it anyway produces a near-100% discard
    # rate that the shared warning then blames on class loss, when the real
    # cause is that almost every resample's own denominator is undefined too —
    # a fault report about a quantity that was never going to exist.
    point = _ratio(all_rows)
    if point.discard_reason == "denominator_below_threshold":
        return None
    if point.value is None:
        raise RuntimeError(
            "the full leakage-fraction dataset cannot score every mechanism class "
            "in every seed/fold"
        )

    ci = cluster_bootstrap_ci(
        clusters,
        _ratio,
        n_resamples=n_resamples,
        seed=seed,
        metric_name=metric_name,
    )

    # A denominator-driven discard is not exchangeable noise: it removes exactly
    # the resamples whose gene-split score dipped near the floor, so the draws
    # that survive are a biased-upward subset and any interval built only from
    # them misstates its own uncertainty. A fold-losing-a-class discard has no
    # such direction (exp4_fixes.md's own rare-class-rule analysis), so it alone
    # does not force suppression.
    reason_counts = ci.get("discard_reason_counts") or {}
    denom_discard_frac = reason_counts.get("denominator_below_threshold", 0) / n_resamples
    if denom_discard_frac > BOOTSTRAP_MAX_DISCARD_FRAC and not ci["ci_suppressed"]:
        ci["ci_low"] = None
        ci["ci_high"] = None
        ci["ci_suppressed"] = True
        ci["ci_suppressed_reason"] = (
            f"{denom_discard_frac:.1%} of resamples were discarded for a "
            "collapsed denominator, which biases the surviving draws rather than "
            "just narrowing them"
        )
    return ci


def main(compute_ci: bool = True, n_boot: int = BOOTSTRAP_N_RESAMPLES) -> None:
    seed_results = load_seed_files(
        str(RESULTS_DIR), SEED_RESULT_GLOB, expected_seeds=range(N_SEEDS)
    )
    seed_numbers = [seed for seed, _filename, _result in seed_results]
    seeds = [result for _seed, _filename, result in seed_results]
    naive_result = _measured_chance_result()
    reference_chance = float(
        naive_result["by_strategy"]["most_frequent"]["gene"]["macro_f1_mean"]
    )
    common_fingerprints = seeds[0].get("input_fingerprints")
    common_parameters = seeds[0].get("analysis_parameters")
    if common_fingerprints is None:
        raise ValueError("seed results lack Section 4 input fingerprints")
    if common_parameters is None:
        raise ValueError("seed results lack Section 4 analysis parameters")
    for seed_number, seed_result in zip(seed_numbers, seeds):
        if seed_result.get("input_fingerprints") != common_fingerprints:
            raise ValueError(f"seed {seed_number} was produced from different Section 4 inputs")
        if seed_result.get("analysis_parameters") != common_parameters:
            raise ValueError(f"seed {seed_number} used different Section 4 parameters")
    naive_fingerprints = naive_result.get("input_fingerprints")
    for key in ("labeled_variants", "pfam_assignments"):
        if naive_fingerprints is None or naive_fingerprints.get(key) != common_fingerprints.get(key):
            raise ValueError(f"naive baseline {key} does not match the probe seed results")

    feature_sets = [set(seed["gene_split"]) & set(seed["family_split"]) for seed in seeds]
    if any(feature_set != feature_sets[0] for feature_set in feature_sets[1:]):
        raise ValueError("Section 4 seed files do not contain the same feature set")
    features = sorted(feature_sets[0])
    oof_caches = _load_validated_oof_caches(seed_results)

    meta = seeds[0]
    results = {
        "n_variants": int(meta["n_variants"]),
        "n_genes": int(meta["n_genes"]),
        "n_families": int(meta["n_families"]),
        "n_seeds": len(seeds),
        "seed_analysis_run_ids": {
            str(seed_number): seed_result["analysis_run_id"]
            for seed_number, seed_result in zip(seed_numbers, seeds)
        },
        "reference_full_cohort_chance_macro_f1": reference_chance,
        "input_fingerprints": common_fingerprints,
        "analysis_parameters": {
            "source_probe_parameters": common_parameters,
            "n_bootstrap_resamples": n_boot if compute_ci else None,
            "ci_enabled": compute_ci,
        },
        "class_distribution": meta.get("class_distribution"),
        "by_feature": {},
    }

    if not FAMILY_CLUSTERING_JSON.exists():
        raise FileNotFoundError(
            f"missing required family-clustering result {FAMILY_CLUSTERING_JSON}"
        )
    with open(FAMILY_CLUSTERING_JSON) as fh:
        fc = json.load(fh)
    family_fingerprints = fc.get("input_fingerprints")
    for key in (
        "labeled_variants",
        "wt_mean_embedding",
        "mut_mean_embedding",
        "pfam_assignments",
    ):
        if (
            family_fingerprints is None
            or family_fingerprints.get(key) != common_fingerprints.get(key)
        ):
            raise ValueError(
                f"family clustering {key} does not match the probe seed results"
            )
    results["within_family_mechanism_agreement"] = (
        fc["by_view"]["wt_mean"].get("frac_gene_mech_matches_family_majority")
    )
    pfam_map = load_pfam_map(PFAM_JSON)

    print(f"n={results['n_variants']} variants, {results['n_genes']} genes, "
          f"{results['n_families']} families, {results['n_seeds']} seeds")
    print(
        "full-cohort reference chance macro-F1 "
        f"(feature rows are recomputed below) = {reference_chance:.3f}\n"
    )
    print(f"{'feature':18} {'gene':>6} {'family':>7} {'drop':>6} {'leakage_fraction':>20}")

    for feature in features:
        if not all(feature in cache for cache in oof_caches):
            missing_seeds = [
                seed_numbers[index]
                for index, cache in enumerate(oof_caches)
                if feature not in cache
            ]
            raise ValueError(f"{feature}: OOF cache missing for seeds {missing_seeds}")
        feature_caches = [cache[feature] for cache in oof_caches]
        chance = aligned_majority_chance(feature_caches)
        cell = leakage_fraction_per_feature(feature, chance, feature_caches)
        if compute_ci:
            ci = leakage_fraction_ci(
                feature_caches,
                pfam_map,
                chance,
                n_boot,
                seed=0,
                metric_name=f"{feature}_leakage_fraction",
            )
            if ci is not None:
                cell["ci"] = ci
        results["by_feature"][feature] = cell
        lf = cell["leakage_fraction"]
        lf_str = f"{lf:.1%}" if lf is not None else "undefined (at floor)"
        ci = cell.get("ci") or {}
        ci_str = ""
        if ci.get("ci_low") is not None:
            ci_str = f"  [{ci['ci_low']:+.1%}, {ci['ci_high']:+.1%}]"
            if ci["ci_low"] <= 0.0 <= ci["ci_high"]:
                ci_str += " (includes zero)"
        elif ci:
            ci_str = "  CI suppressed"
        print(
            f"{feature:18} {cell['gene_macro_f1_mean']:6.3f} "
            f"{cell['family_macro_f1_mean']:7.3f} {cell['drop_mean']:6.3f} "
            f"{lf_str:>20}{ci_str}"
        )

    LEAKAGE_FRACTION_JSON.parent.mkdir(parents=True, exist_ok=True)
    write_result_json(LEAKAGE_FRACTION_JSON, results, seeds=seed_numbers, indent=2)
    print(f"\nResults written to {LEAKAGE_FRACTION_JSON}")


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_ci", action="store_true", help="skip the leakage-fraction ratio CI")
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    args = parser.parse_args()
    main(compute_ci=not args.no_ci, n_boot=args.n_boot)


if __name__ == "__main__":
    _cli()
