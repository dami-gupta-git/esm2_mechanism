"""
Probe runs for perturbation scan features.

Loads scan_features.npy (output of perturbation_scan.py phase 3) and runs
logistic regression under gene-split and family-split CV, comparing:

  - Baseline: mean-pooled delta (result_7 / result_18 baseline)
  - Scan features only (5 pre-registered)
  - Scan + mean-pooled delta
  - Scan + proteome features (result_13)
  - Scan + Badonyi features (result_15)

Decision rules (pre-registered in plan_perturb.md):
  G1: scan-only family-split F1 > 0.368  (ClinVar-pattern + 0.02)
  G2: scan+delta family-split F1 > 0.419  (result_18 combined + 0.02)
  G3: scan+proteome family-split F1 > 0.405  (proteome-only + 0.02)

Usage:
  cd esm2_mechanism
  python3 scripts/perturbation_probe.py
"""

import functools
import json, os, sys, numpy as np
from pathlib import Path

from esm2_mech.utils.data import build_gene_to_row
from esm2_mech.utils.paths import (
    BADONYI_FEATURES_ALIGNED,
    EMB_MUT_MEAN,
    EMB_WT_MEAN,
    GENE_UNIVERSE,
    PFAM_JSON,
    PROTEOME_FEATURES_ALIGNED,
    RESULTS_DIR as _RESULTS_DIR,
    SCAN_FEATURES_META_JSON,
    SCAN_FEATURES_NPY,
    VALID_VARIANTS_JSON,
)
from esm2_mech.utils.splits import gene_split_cv, family_split_cv
from esm2_mech.utils.embed import load_gene_delta
from esm2_mech.utils.probes import run_logreg_cv, run_histgb_cv

print = functools.partial(print, flush=True)
OUT = _RESULTS_DIR / "perturbation_scan"
OUT.mkdir(parents=True, exist_ok=True)


DECISION_RULES = {
    "G1": ("scan_only_family_split", "macro_f1_mean", 0.368),
    "G2": ("scan_delta_family_split", "macro_f1_mean", 0.419),
    "G3": ("scan_proteome_family_split", "macro_f1_mean", 0.405),
}


def load_all_features(gene_list):
    """Load all feature matrices aligned to gene_list.

    All returned arrays are row-aligned to the same filtered gene list.
    The second return value (gene_mask) maps rows back to gene_list.
    """
    features = {}

    # 1. Scan features (phase 3 output). Row order is pinned by the meta file's
    # own gene list, not by gene_universe.
    scan_X = np.load(SCAN_FEATURES_NPY)
    with open(SCAN_FEATURES_META_JSON) as f:
        scan_meta = json.load(f)
    scan_idx = {g: i for i, g in enumerate(scan_meta["genes"])}

    # 2. Mean-pooled delta index
    gene_delta = load_gene_delta(VALID_VARIANTS_JSON, EMB_WT_MEAN, EMB_MUT_MEAN)

    # 3. Proteome gene→row index. The matrix rows are in gene_universe.tsv order
    # (see paths.GENE_UNIVERSE); gene_list.tsv is a longer, differently-ordered
    # superset and indexing by it reads the wrong gene's features.
    pg_idx: dict = {}
    if PROTEOME_FEATURES_ALIGNED.exists():
        pg_idx = build_gene_to_row(GENE_UNIVERSE)

    # Determine which genes have ALL required features so every array has the
    # same row count (previously each source filtered independently, making
    # np.hstack silently produce misaligned concatenations).
    def _has_all(g):
        if g not in scan_idx:
            return False
        if g not in gene_delta:
            return False
        if PROTEOME_FEATURES_ALIGNED.exists() and g not in pg_idx:
            return False
        return True

    gene_mask = np.array([_has_all(g) for g in gene_list])
    scan_gene_list = gene_list[gene_mask]
    print(f"  Genes with all features: {len(scan_gene_list)}/{len(gene_list)}")

    features["scan"] = np.array(
        [scan_X[scan_idx[g]] for g in scan_gene_list], dtype=np.float32
    )
    features["delta"] = np.array(
        [np.mean(gene_delta[g], axis=0) for g in scan_gene_list], dtype=np.float32
    )

    if PROTEOME_FEATURES_ALIGNED.exists():
        proteome_X = np.load(PROTEOME_FEATURES_ALIGNED)
        if proteome_X.shape[0] != len(pg_idx):
            raise ValueError(
                f"{PROTEOME_FEATURES_ALIGNED} has {proteome_X.shape[0]} rows but "
                f"{GENE_UNIVERSE} lists {len(pg_idx)} genes — not row-aligned."
            )
        features["proteome"] = np.array(
            [proteome_X[pg_idx[g]] for g in scan_gene_list], dtype=np.float32
        )
    else:
        print(f"  {PROTEOME_FEATURES_ALIGNED} not found — skipping proteome features")

    # Sanity check: all feature arrays must have the same number of rows
    row_counts = {name: X.shape[0] for name, X in features.items()}
    if len(set(row_counts.values())) > 1:
        raise RuntimeError(f"Feature row count mismatch after alignment: {row_counts}")

    return features, gene_mask


# Combos whose matrix includes the proteome block, which carries real NaN.
# These route to the NaN-native runner; the dense scan/delta combos keep LogReg
# so the linear read on them is unchanged.
NAN_BEARING_COMBOS = {"scan_proteome"}


def run_probe(X, labels, splits, seed=42, combo_name=""):
    """Route by whether this combo's matrix can contain missing cells.

    The proteome block is sparse, and hstacking it onto the dense scan block
    means a few missing proteome columns would otherwise make the whole row
    unusable. Nothing is imputed and no gene is dropped.
    """
    if combo_name in NAN_BEARING_COMBOS:
        return run_histgb_cv(X, labels, splits, seed=seed)
    return run_logreg_cv(X, labels, splits, seed=seed)


def main():
    print("=== Loading data ===")

    # Gene list and labels from merged dataset
    with open(VALID_VARIANTS_JSON) as f:
        variants = json.load(f)
    for v in variants:
        if "label_3class" not in v:
            v["label_3class"] = (
                "LOF"
                if v.get("mechanism") in ("HI", "AR")
                else v.get("mechanism", "LOF")
            )

    # Gene-level: one label per gene (majority vote)
    from collections import Counter, defaultdict

    gene_labels = defaultdict(list)
    for v in variants:
        gene_labels[v["gene"].upper()].append(v["label_3class"])
    gene_list = np.array(sorted(gene_labels.keys()))
    labels = np.array([Counter(gene_labels[g]).most_common(1)[0][0] for g in gene_list])

    print(f"Genes: {len(gene_list)}  Classes: {dict(Counter(labels))}")

    # Load features
    features, gene_mask = load_all_features(gene_list)

    # Filter to genes with scan features
    gene_list_scan = gene_list[gene_mask]
    labels_scan = labels[gene_mask]
    print(f"Genes with scan features: {len(gene_list_scan)}")

    # Pfam map
    with open(PFAM_JSON) as f:
        pfam_map = json.load(f)

    # Feature combinations to test
    scan_X = features["scan"]
    delta_X = features["delta"] if "delta" in features else None
    proteome_X = features.get("proteome")

    combos = {
        "baseline_delta": delta_X,
        "scan_only": scan_X,
        "scan_delta": np.hstack([scan_X, delta_X]) if delta_X is not None else None,
        "scan_proteome": (
            np.hstack([scan_X, proteome_X]) if proteome_X is not None else None
        ),
    }
    combos = {k: v for k, v in combos.items() if v is not None}

    # 5-seed runs
    all_results = {}
    for seed in range(5):
        print(f"\n=== Seed {seed} ===")
        gs = gene_split_cv(gene_list_scan, seed=seed)
        fs = family_split_cv(gene_list_scan, pfam_map, seed=seed)
        seed_res = {}
        for combo_name, X in combos.items():
            for split_name, splits in [("gene_split", gs), ("family_split", fs)]:
                key = f"{combo_name}_{split_name}"
                r = run_probe(X, labels_scan, splits, seed=seed, combo_name=combo_name)
                seed_res[key] = r
                f1 = r.get("macro_f1_mean", float("nan"))
                gof = r.get("auroc_GOF_mean", float("nan"))
                print(f"  {key}: F1={f1:.3f}  GOF={gof:.3f}")
        all_results[seed] = seed_res

    # Summary
    print("\n=== 5-SEED SUMMARY ===")
    summary = {}
    for key in all_results[0].keys():
        f1_vals = [
            all_results[s][key].get("macro_f1_mean", float("nan")) for s in range(5)
        ]
        gof_vals = [
            all_results[s][key].get("auroc_GOF_mean", float("nan")) for s in range(5)
        ]
        summary[key] = {
            "macro_f1_mean": float(np.nanmean(f1_vals)),
            "macro_f1_std": float(np.nanstd(f1_vals)),
            "auroc_GOF_mean": float(np.nanmean(gof_vals)),
            "auroc_GOF_std": float(np.nanstd(gof_vals)),
        }
        print(
            f"  {key}: F1={summary[key]['macro_f1_mean']:.3f}±{summary[key]['macro_f1_std']:.3f}"
            f"  GOF={summary[key]['auroc_GOF_mean']:.3f}±{summary[key]['auroc_GOF_std']:.3f}"
        )

    # Decision rules
    print("\n=== DECISION RULES ===")
    gate_results = {}
    for gate, (key, metric, threshold) in DECISION_RULES.items():
        val = summary.get(key, {}).get(metric, float("nan"))
        passed = val > threshold
        gate_results[gate] = {"value": val, "threshold": threshold, "passed": passed}
        status = "PASS ✓" if passed else "FAIL ✗"
        print(
            f"  {gate}: {key} {metric} = {val:.3f} (threshold {threshold:.3f}) → {status}"
        )

    if not gate_results.get("G1", {}).get("passed"):
        print(
            "\n  G1 failed — scan features do not improve on ClinVar-pattern baseline."
        )
        print("  Do not proceed to G2/G3 analysis.")

    # Save
    out = {
        "summary": summary,
        "gate_results": gate_results,
        "per_seed": {str(s): all_results[s] for s in range(5)},
        "n_genes": int(len(gene_list_scan)),
        "feature_combos": list(combos.keys()),
    }
    out_path = OUT / "probe_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
