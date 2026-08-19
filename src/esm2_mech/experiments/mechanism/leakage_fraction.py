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


def _load_seed_baselines():
    """Per-seed (seed number, gene/family macro-F1) for every feature.

    No expected seed count is known here — leakage_fraction runs as a downstream
    aggregator over whatever family_split_baselines_seed*.json files
    classify_by_mechanism already wrote — but load_seed_files still rejects a
    filename with no parseable seed number or two files claiming the same seed.
    """
    seed_results = load_seed_files(str(RESULTS_DIR), SEED_RESULT_GLOB)
    if not seed_results:
        raise FileNotFoundError(f"no seed files matching {SEED_RESULT_GLOB!r} in {RESULTS_DIR}")
    seed_numbers = [seed for seed, _filename, _result in seed_results]
    seeds = [result for _seed, _filename, result in seed_results]
    return seed_numbers, seeds


def _measured_chance():
    """Majority-class macro-F1 floor (gene-split)."""
    with open(NAIVE_BASELINE_JSON) as fh:
        nb = json.load(fh)
    return float(nb["by_strategy"]["most_frequent"]["gene"]["macro_f1_mean"])


def _pick_macro_f1(arm_result):
    """The per-fold macro-F1, the basis every other metric in this run uses."""
    return arm_result.get("macro_f1_mean")


def leakage_fraction_per_feature(seeds, feature, chance, oof_cache_entries=None):
    """Leakage fraction for one feature, from seed-averaged macro-F1.

    The gene-split arm's macro-F1 in the per-seed baseline files is scored over
    every variant, while the family-split arm's is scored only over the rows whose
    gene has a Pfam annotation (family_split_cv excludes the rest). Reading both
    straight from the baseline files therefore compares two arms over different row
    sets. When this feature's OOF cache is available, both arms are instead
    rescored on the rows they share — the same row-alignment leakage_fraction_ci
    already uses — so the headline and the interval describe the same quantity.
    Falls back to the unrestricted baseline-file reading when no cache exists
    (foldx_ddg / alphamissense, whose row space is feature-local to begin with).
    """
    n_excluded = None
    if oof_cache_entries is not None:
        aligned = _align_seed_arms(oof_cache_entries)
        if aligned is not None:
            per_seed, _ = aligned
            n_shared = len(per_seed[0]["gene"]["y_true"])
            n_total = len(oof_cache_entries[0]["gene_split"]["row_ids"])
            n_excluded = n_total - n_shared
            rows = np.arange(n_shared)
            gene_f1 = [_arm_macro_f1(s["gene"], rows) for s in per_seed]
            family_f1 = [_arm_macro_f1(s["family"], rows) for s in per_seed]
        else:
            gene_f1 = family_f1 = []
    else:
        gene_f1 = [_pick_macro_f1(s["gene_split"][feature]) for s in seeds]
        family_f1 = [_pick_macro_f1(s["family_split"][feature]) for s in seeds]

    gene_mean, gene_std, gene_n = mean_std_n(gene_f1)
    family_mean, family_std, _ = mean_std_n(family_f1)

    result = {
        "gene_macro_f1_mean": gene_mean,
        "gene_macro_f1_std": gene_std,
        "family_macro_f1_mean": family_mean,
        "family_macro_f1_std": family_std,
        "drop_mean": gene_mean - family_mean,
    }
    if n_excluded is not None:
        result["n_excluded_unannotated"] = n_excluded
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
    for entry in oof_cache_entries:
        gene_arm, family_arm = entry["gene_split"], entry["family_split"]
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
    for gene_pos, family_pos, gene_arm, family_arm in per_seed:
        gene_idx = np.array([gene_pos[row] for row in shared], dtype=int)
        family_idx = np.array([family_pos[row] for row in shared], dtype=int)
        aligned.append({
            "gene": {key: np.asarray(gene_arm[key])[gene_idx] for key in
                     ("y_true", "pred", "folds")},
            "family": {key: np.asarray(family_arm[key])[family_idx] for key in
                       ("y_true", "pred", "folds")},
        })
    first_gene_pos, _, first_gene_arm, _ = per_seed[0]
    gene_names = np.asarray(first_gene_arm["genes"])[
        np.array([first_gene_pos[row] for row in shared], dtype=int)
    ]
    return aligned, gene_names


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
    seed_numbers, seeds = _load_seed_baselines()
    chance = _measured_chance()
    features = list(seeds[0]["gene_split"].keys())

    meta = seeds[0]
    results = {
        "n_variants": int(meta["n_variants"]),
        "n_genes": int(meta["n_genes"]),
        "n_families": int(meta["n_families"]),
        "n_seeds": len(seeds),
        "chance_macro_f1": chance,
        "class_distribution": meta.get("class_distribution"),
        "by_feature": {},
    }

    if FAMILY_CLUSTERING_JSON.exists():
        with open(FAMILY_CLUSTERING_JSON) as fh:
            fc = json.load(fh)
        results["within_family_mechanism_agreement"] = (
            fc["by_view"]["wt_mean"].get("frac_gene_mech_matches_family_majority")
        )

    # One cache per seed, and every seed the headline averages must be present: a
    # headline or interval built from a subset of the seeds is not the reported
    # quantity. The cache is also what lets the headline restrict the gene-split
    # arm to the rows the family-split arm actually scored (see
    # leakage_fraction_per_feature), so it is loaded whenever available, not only
    # when compute_ci is set — compute_ci gates the interval, not the row alignment.
    oof_caches = None
    pfam_map = None
    cache_paths = [
        os.path.join(RESULTS_DIR, mechanism_oof_cache_filename(seed_i))
        for seed_i in seed_numbers
    ]
    missing = [path for path in cache_paths if not os.path.exists(path)]
    if missing:
        print(
            f"  NOTE: {len(missing)} of {len(cache_paths)} seed OOF caches not "
            "found — leakage-fraction headline falls back to the unrestricted "
            "gene-split score, and CI is skipped for all features (re-run "
            "mechanism_delta_family_split for every seed with CIs on)."
        )
    else:
        oof_caches = []
        for path in cache_paths:
            with open(path) as fh:
                oof_caches.append(json.load(fh))
        pfam_map = load_pfam_map(PFAM_JSON)

    print(f"n={results['n_variants']} variants, {results['n_genes']} genes, "
          f"{results['n_families']} families, {results['n_seeds']} seeds")
    print(f"chance macro-F1 (measured majority-class floor) = {chance:.3f}\n")
    print(f"{'feature':18} {'gene':>6} {'family':>7} {'drop':>6} {'leakage_fraction':>20}")

    for feature in features:
        feature_caches = None
        if oof_caches is not None and all(feature in cache for cache in oof_caches):
            feature_caches = [cache[feature] for cache in oof_caches]
        cell = leakage_fraction_per_feature(seeds, feature, chance, feature_caches)
        if compute_ci and feature_caches is not None:
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
