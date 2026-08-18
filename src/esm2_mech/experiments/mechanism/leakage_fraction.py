"""Compute the leakage fraction: how much of each feature's gene-split signal
is family recognition rather than transferable mechanism signal.
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
from esm2_mech.utils.metrics import majority_baseline_f1, mean_std_n
from esm2_mech.utils.paths import (
    FAMILY_CLUSTERING_JSON,
    LEAKAGE_FRACTION_JSON,
    NAIVE_BASELINE_JSON,
    PFAM_JSON,
    RESULTS_DIR,
)

print = functools.partial(print, flush=True)

BASELINES_GLOB = str(RESULTS_DIR / "family_split_baselines_seed*.json")

# Below this margin the denominator is ~0 and the ratio is noise.
MIN_ABOVE_CHANCE = 0.02


def _load_seed_baselines():
    """Per-seed gene/family macro-F1 for every feature."""
    files = sorted(glob.glob(BASELINES_GLOB))
    if not files:
        raise FileNotFoundError(BASELINES_GLOB)
    seeds = []
    for path in files:
        with open(path) as fh:
            seeds.append(json.load(fh))
    return seeds


def _measured_chance():
    """Majority-class macro-F1 floor (gene-split)."""
    with open(NAIVE_BASELINE_JSON) as fh:
        nb = json.load(fh)
    return float(nb["by_strategy"]["most_frequent"]["gene"]["macro_f1_mean"])


def _pick_macro_f1(arm_result):
    """Prefer pooled OOF macro-F1 (consistent with bootstrap CI); fall back to fold mean."""
    return arm_result.get("macro_f1_pooled", arm_result.get("macro_f1_mean"))


def leakage_fraction_per_feature(seeds, feature, chance):
    """Leakage fraction for one feature, from seed-averaged macro-F1."""
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


def leakage_fraction_ci(oof_cache_entry, pfam_map, n_resamples, seed=0):
    """Cluster-bootstrap CI on the LF ratio, resampling family clusters shared across both arms."""
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
    # Cluster on Pfam family; orphan genes become singleton clusters.
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
        resample_chance, _ = majority_baseline_f1(gene_y_true[g_idx], gene_y_true[g_idx])
        denom = gene_f1 - resample_chance
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

    if FAMILY_CLUSTERING_JSON.exists():
        with open(FAMILY_CLUSTERING_JSON) as fh:
            fc = json.load(fh)
        results["within_family_mechanism_agreement"] = (
            fc["by_view"]["wt_mean"].get("frac_gene_mech_matches_family_majority")
        )

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
            ci = leakage_fraction_ci(oof_cache[feature], pfam_map, n_boot, seed=0)
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
