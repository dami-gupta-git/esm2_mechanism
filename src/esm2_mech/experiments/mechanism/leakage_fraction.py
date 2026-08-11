"""
Compute the leakage fraction for the run6 mechanism dataset.

The leakage fraction (LF) measures what proportion of a feature's above-chance
gene-split signal disappears under family-split CV — i.e. how much of the
gene-split score is family recognition rather than transferable mechanism signal:

    LF = (gene_split_macro_f1 - family_split_macro_f1) / (gene_split_macro_f1 - chance)

A high LF means the gene-split number is mostly homology leakage; a low (or
negative/undefined) LF means the feature carries little family-mediated signal
to begin with. LF is only meaningful for features that score above chance on
gene-split — for a feature already at the floor the denominator is ~0 and the
ratio is noise, so it is reported as undefined.

`chance` is the measured majority-class floor from naive_baseline.json (NOT a
hardcoded 1/3), so the denominator matches the actual task floor.

LF is computed from the across-seed mean gene/family macro-F1 (average the F1s,
then take the ratio) — not by averaging per-seed ratios, which are unstable
because each divides by the small, noisy (gene_f1 - chance). Per-seed F1 spread
is reported separately. Structural quantities the LF derives from (within-family
mechanism agreement, class balance) are recorded alongside for context.

  Input : results/run6/family_split_baselines_seed{0..4}.json (gene/family macro-F1 per feature)
          results/run6/naive_baseline.json                    (measured chance floor)
          results/run6/family_clustering.json                 (within-family agreement; optional)
  Output: results/run6/leakage_fraction.json + stdout table

Usage:
    python -m esm2_mech.experiments.mechanism.leakage_fraction
"""

from __future__ import annotations

import argparse
import functools
import glob
import json
import os

import numpy as np
from sklearn.metrics import f1_score

from esm2_mech.utils.bootstrap import cluster_bootstrap_ci
from esm2_mech.utils.constants import BOOTSTRAP_N_RESAMPLES, mechanism_oof_cache_filename
from esm2_mech.utils.metrics import mean_std_n
from esm2_mech.utils.paths import (
    FAMILY_CLUSTERING_JSON,
    LEAKAGE_FRACTION_JSON,
    NAIVE_BASELINE_JSON,
    PFAM_JSON,
    RESULTS_DIR,
)

print = functools.partial(print, flush=True)

BASELINES_GLOB = str(RESULTS_DIR / "family_split_baselines_seed*.json")

# A feature must clear chance by at least this margin on gene-split for its LF to
# be defined; below it the denominator is ~0 and the ratio is meaningless noise.
MIN_ABOVE_CHANCE = 0.02


def _load_seed_baselines():
    """Per-seed gene/family macro-F1 for every feature, from the baseline files."""
    files = sorted(glob.glob(BASELINES_GLOB))
    if not files:
        raise FileNotFoundError(BASELINES_GLOB)
    seeds = []
    for path in files:
        with open(path) as fh:
            seeds.append(json.load(fh))
    return seeds


def _measured_chance():
    """Majority-class macro-F1 floor (gene-split) from the naive baseline."""
    with open(NAIVE_BASELINE_JSON) as fh:
        nb = json.load(fh)
    return float(nb["by_strategy"]["most_frequent"]["gene"]["macro_f1_mean"])


def leakage_fraction_per_feature(seeds, feature, chance):
    """Leakage fraction for one feature, from seed-averaged macro-F1.

    LF is computed from the across-seed mean gene/family macro-F1, not by
    averaging per-seed ratios. Per-seed ratios are unstable: each divides by
    (gene_f1 - chance), a small noisy quantity, so a single lucky-high gene seed
    inflates that seed's ratio. Averaging the F1s first removes that artefact.
    Per-seed F1 spread is reported separately for context.
    """
    # macro_f1_mean can be None (a seed where no fold scored) — use the
    # None/NaN-filtering reducer rather than plain np.mean, which would crash
    # (object dtype) or silently poison the average.
    gene_f1 = [s["gene_split"][feature]["macro_f1_mean"] for s in seeds]
    family_f1 = [s["family_split"][feature]["macro_f1_mean"] for s in seeds]
    gene_mean, gene_std, gene_n = mean_std_n(gene_f1)
    family_mean, family_std, _ = mean_std_n(family_f1)

    result = {
        "gene_macro_f1_mean": gene_mean,
        "gene_macro_f1_std": gene_std,
        "family_macro_f1_mean": family_mean,
        "family_macro_f1_std": family_std,
        "drop_mean": gene_mean - family_mean,
    }
    if gene_n == 0:
        # No scorable gene-split seed for this feature; LF undefined.
        result["leakage_fraction"] = None
        result["note"] = "no scorable gene-split seed; LF undefined"
        return result
    denom = gene_mean - chance
    if denom > MIN_ABOVE_CHANCE:
        result["leakage_fraction"] = (gene_mean - family_mean) / denom
    else:
        # Feature is at/near the chance floor on gene-split; LF undefined
        # (no above-chance signal to attribute to leakage).
        result["leakage_fraction"] = None
        result["note"] = "gene-split score not meaningfully above chance; LF undefined"
    return result


def leakage_fraction_ci(oof_cache_entry, pfam_map, chance, n_resamples, seed=0):
    """Cluster-bootstrap CI on the leakage-fraction RATIO, resampled once per
    replicate and shared across both arms.

    The ratio's numerator and denominator both depend on the gene-split macro-F1,
    so combining two separately-computed CIs (one on gene_f1, one on family_f1)
    would ignore that dependence. Instead this draws ONE resample of the shared
    row set per replicate and recomputes gene_f1, family_f1, and the ratio
    together on that resample — gene-split and family-split are different CV
    partitions of the same underlying variants, so this is the cross-partition
    case (Task 1 / R7.3): the resampling unit is family, the coarser of the two
    arms' units, not gene.

    `oof_cache_entry` is {"gene_split": {...}, "family_split": {...}} as written
    by mechanism_delta_family_split.run's seed-0 OOF cache (row_ids, y_true,
    pred, genes for each arm). Restricted to the intersection of row_ids present
    in both arms before resampling, per Task 1's shared-subset requirement.
    """
    gene_arm = oof_cache_entry["gene_split"]
    family_arm = oof_cache_entry["family_split"]

    gene_pos_by_row = {row: pos for pos, row in enumerate(gene_arm["row_ids"])}
    family_pos_by_row = {row: pos for pos, row in enumerate(family_arm["row_ids"])}
    shared_rows = sorted(set(gene_pos_by_row) & set(family_pos_by_row))
    if not shared_rows:
        return None

    gene_y_true = np.array(gene_arm["y_true"])
    gene_pred = np.array(gene_arm["pred"])
    family_y_true = np.array(family_arm["y_true"])
    family_pred = np.array(family_arm["pred"])
    gene_names = np.array(gene_arm["genes"])

    gene_positions = np.array([gene_pos_by_row[row] for row in shared_rows])
    family_positions = np.array([family_pos_by_row[row] for row in shared_rows])
    # Cluster on Pfam family (the coarser unit) per shared row; a gene with no
    # Pfam annotation falls back to its own gene id as a singleton cluster.
    row_genes = gene_names[gene_positions]
    clusters = np.array([pfam_map.get(g) or f"__orphan__{g}" for g in row_genes])

    def _ratio(rows):
        g_idx = gene_positions[rows]
        f_idx = family_positions[rows]
        gene_f1 = float(
            f1_score(gene_y_true[g_idx], gene_pred[g_idx], average="macro", zero_division=0)
        )
        family_f1 = float(
            f1_score(family_y_true[f_idx], family_pred[f_idx], average="macro", zero_division=0)
        )
        denom = gene_f1 - chance
        if denom <= MIN_ABOVE_CHANCE:
            return None
        return (gene_f1 - family_f1) / denom

    return cluster_bootstrap_ci(clusters, _ratio, n_resamples=n_resamples, seed=seed)


def main(compute_ci: bool = True, n_boot: int = BOOTSTRAP_N_RESAMPLES) -> None:
    seeds = _load_seed_baselines()
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

    # Structural context (optional — present once family_clustering has run).
    if FAMILY_CLUSTERING_JSON.exists():
        with open(FAMILY_CLUSTERING_JSON) as fh:
            fc = json.load(fh)
        results["within_family_mechanism_agreement"] = (
            fc["by_view"]["wt_mean"].get("frac_gene_mech_matches_family_majority")
        )

    # Seed-0 OOF cache (delta_mean, wt_only_mean) for the joint ratio bootstrap.
    # Optional: absent when mechanism_delta_family_split was run with --no_ci or
    # before this cache existed, in which case the LF ratio CI is simply omitted.
    oof_cache_path = os.path.join(RESULTS_DIR, mechanism_oof_cache_filename(0))
    oof_cache = None
    pfam_map = None
    if compute_ci and os.path.exists(oof_cache_path):
        with open(oof_cache_path) as fh:
            oof_cache = json.load(fh)
        with open(PFAM_JSON) as fh:
            pfam_map = json.load(fh)
    elif compute_ci:
        print(
            f"  NOTE: {oof_cache_path} not found — leakage-fraction CI skipped "
            "for all features (re-run mechanism_delta_family_split seed 0 with CIs on)."
        )

    print(f"n={results['n_variants']} variants, {results['n_genes']} genes, "
          f"{results['n_families']} families, {results['n_seeds']} seeds")
    print(f"chance macro-F1 (measured majority-class floor) = {chance:.3f}\n")
    print(f"{'feature':18} {'gene':>6} {'family':>7} {'drop':>6} {'leakage_fraction':>20}")

    for feature in features:
        cell = leakage_fraction_per_feature(seeds, feature, chance)
        if oof_cache is not None and feature in oof_cache:
            ci = leakage_fraction_ci(oof_cache[feature], pfam_map, chance, n_boot, seed=0)
            if ci is not None:
                cell["ci"] = ci
        results["by_feature"][feature] = cell
        lf = cell["leakage_fraction"]
        lf_str = f"{lf:.1%}" if lf is not None else "undefined (at floor)"
        print(
            f"{feature:18} {cell['gene_macro_f1_mean']:6.3f} "
            f"{cell['family_macro_f1_mean']:7.3f} {cell['drop_mean']:6.3f} {lf_str:>20}"
        )

    LEAKAGE_FRACTION_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(LEAKAGE_FRACTION_JSON, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults written to {LEAKAGE_FRACTION_JSON}")


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_ci", action="store_true", help="skip the leakage-fraction ratio CI")
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    args = parser.parse_args()
    main(compute_ci=not args.no_ci, n_boot=args.n_boot)


if __name__ == "__main__":
    _cli()
