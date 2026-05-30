"""
Badonyi leakage analysis — quantify the label-leakage contribution to result_15.

Result_15 used Badonyi 2024's SVM output probabilities (pDN, pGOF, pLOF) as input
features for V_bad and V2+bad. Many genes in the merged dataset were in Badonyi's
training set, so their pDN/pGOF/pLOF values are partly fit-to-label predictions
from a pre-trained model. This script quantifies the magnitude.

Approach: re-run V_bad and V2+bad family-split CV on three subsets:
    ALL  — every labeled+family-annotated variant (reproduces result_15)
    IN   — variants whose gene was in Badonyi's training set (any classifier)
    OUT  — variants whose gene was NOT in Badonyi's training set

V2 (proteome only) is run on all three regimes as a leakage-free comparator.

Outputs:
    results/badonyi_leakage/leakage_seed{0..4}.json
    results/badonyi_leakage/leakage_summary.json

Usage:
    python3 scripts/badonyi_leakage_analysis.py
    python3 scripts/badonyi_leakage_analysis.py --seed 0 --n-folds 5
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from esm2_mechanism.utils_probes import family_split_indices, run_logreg_cv
from esm2_mechanism.utils_paths import DATA_DIR, RESULTS_DIR
import functools

print = functools.partial(print, flush=True)

warnings.filterwarnings("ignore")

EMB_DIR = DATA_DIR / "embeddings"

MERGED_VALID_VARIANTS = EMB_DIR / "merged_valid_variants.json"
PROTEOME_FEATURES = DATA_DIR / "proteome_features_aligned.npy"
BADONYI_FEATURES = DATA_DIR / "badonyi_features_aligned.npy"
BADONYI_S3 = DATA_DIR / "cache" / "badonyi" / "table_S3.xlsx"
BADONYI_RAW_COLS = [0, 1, 2]  # pDN, pGOF, pLOF

MERGED_GENE_LIST = DATA_DIR / "merged_gene_list.tsv"
PFAM_FAMILIES = DATA_DIR / "pfam_families.json"

CLASSES = ["GOF", "DN", "LOF"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_data():
    with open(MERGED_VALID_VARIANTS) as f:
        variants = json.load(f)
    labels = np.array([v["label_3class"] for v in variants])
    genes = np.array([v["gene"] for v in variants])
    print(f"Loaded {len(variants)} variants, {len(set(genes.tolist()))} genes")
    return labels, genes


def load_pfam():
    with open(PFAM_FAMILIES) as f:
        return json.load(f)


def build_gene_to_row():
    seen, idx = {}, 0
    with open(MERGED_GENE_LIST) as f:
        f.readline()
        for line in f:
            g = line.strip().split("\t")[0]
            if g and g not in seen:
                seen[g] = idx
                idx += 1
    return seen


def broadcast(genes, matrix, gene_to_row):
    n, d = len(genes), matrix.shape[1]
    X = np.zeros((n, d), dtype=np.float32)
    for i, g in enumerate(genes):
        r = gene_to_row.get(g)
        if r is not None and r < matrix.shape[0]:
            X[i] = matrix[r]
    return X


def load_badonyi_train_flags():
    """Return dict[gene] -> bool: True if gene was in any of Badonyi's three
    training sets (DN, GOF, or LOF classifier).

    Also returns per-classifier dict for stratified diagnostics.
    """
    print(f"Loading Badonyi S3 train flags from {BADONYI_S3.name}")
    s3 = pd.read_excel(BADONYI_S3, sheet_name="table_S3")

    def parse(s):
        if pd.isna(s):
            return (0, 0, 0)
        return tuple(int(b) for b in str(s).split("|"))

    parts = s3["train_dn_gof_lof"].map(parse)
    s3["tr_DN"] = [p[0] for p in parts]
    s3["tr_GOF"] = [p[1] for p in parts]
    s3["tr_LOF"] = [p[2] for p in parts]
    s3["in_any"] = (s3[["tr_DN", "tr_GOF", "tr_LOF"]].sum(axis=1) > 0).astype(int)

    any_train = dict(zip(s3["gene"], s3["in_any"].astype(int)))
    per_class = {
        "DN": dict(zip(s3["gene"], s3["tr_DN"].astype(int))),
        "GOF": dict(zip(s3["gene"], s3["tr_GOF"].astype(int))),
        "LOF": dict(zip(s3["gene"], s3["tr_LOF"].astype(int))),
    }
    print(f"  S3 rows: {len(s3)}.  In any train set: {sum(any_train.values())}")
    return any_train, per_class


# ---------------------------------------------------------------------------
# CV
# ---------------------------------------------------------------------------


def run_logreg(X, y, genes, groups, n_folds, seed, label):
    splits = list(family_split_indices(groups, n_folds, seed))
    return run_logreg_cv(X, y, splits, seed=seed, genes=genes, label=label)


# ---------------------------------------------------------------------------
# Run one regime (ALL / IN / OUT) for one seed
# ---------------------------------------------------------------------------


def run_regime(regime_name, mask, X_prot, X_bad, y, genes, groups, n_folds, seed):
    """Apply mask, run V2, V_bad, V2+bad. Skip if not enough variants."""
    n_var = int(mask.sum())
    if n_var < 100:
        print(f"  [{regime_name}] skipped: only {n_var} variants")
        return {"skipped": True, "n_variants": n_var}

    X_p = X_prot[mask]
    X_b = X_bad[mask]
    X_2b = np.concatenate([X_p, X_b], axis=1)
    y_m = y[mask]
    genes_m = genes[mask]
    groups_m = groups[mask]

    class_counts = Counter(y_m.tolist())
    cd = {CLASSES[k]: int(v) for k, v in class_counts.items()}
    n_fams = len(set(groups_m.tolist()))
    print(
        f"\n  [{regime_name}] n_variants={n_var}, n_genes={len(set(genes_m.tolist()))}, "
        f"n_families={n_fams}, classes={cd}"
    )

    res = {
        "n_variants": n_var,
        "n_genes": len(set(genes_m.tolist())),
        "n_families": n_fams,
        "class_dist_variants": cd,
    }

    print(f"    V2 (proteome 37):")
    res["V2"] = run_logreg(X_p, y_m, genes_m, groups_m, n_folds, seed, "V2")
    print(f"    V_bad (pDN/pGOF/pLOF):")
    res["V_bad"] = run_logreg(X_b, y_m, genes_m, groups_m, n_folds, seed, "V_bad")
    print(f"    V2+bad (40):")
    res["V2_bad"] = run_logreg(X_2b, y_m, genes_m, groups_m, n_folds, seed, "V2+bad")
    return res


def run_seed(seed, n_folds, y, genes, pfam_map, X_prot, X_bad, train_flag_any):
    print(f"\n{'='*72}\nSEED {seed}\n{'='*72}")

    le = LabelEncoder()
    le.fit(CLASSES)
    y_enc = le.transform(y)

    # Restrict to variants whose gene has a Pfam family (same as result_15)
    gene_pfam = np.array([pfam_map.get(g) for g in genes])
    has_fam = gene_pfam != None  # noqa
    fam_mask = has_fam

    # Build per-variant in-Badonyi flag
    in_bad = np.array([train_flag_any.get(g, 0) for g in genes], dtype=int)

    # Three regimes (all also restricted to family-annotated)
    mask_all = fam_mask
    mask_in = fam_mask & (in_bad == 1)
    mask_out = fam_mask & (in_bad == 0)

    print(f"\nVariant counts (family-annotated):")
    print(f"  ALL : {int(mask_all.sum())}")
    print(f"  IN  : {int(mask_in.sum())}  (gene was in Badonyi's training set)")
    print(f"  OUT : {int(mask_out.sum())}  (gene was NOT in Badonyi's training)")

    seed_results = {"seed": seed, "regimes": {}}
    for name, mask in [("ALL", mask_all), ("IN", mask_in), ("OUT", mask_out)]:
        seed_results["regimes"][name] = run_regime(
            name, mask, X_prot, X_bad, y_enc, genes, gene_pfam, n_folds, seed
        )
    return seed_results


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------


def aggregate_seeds(all_seed):
    """Aggregate macro-F1 / per-gene-F1 / AUROC means across seeds, per regime
    and per variant."""
    summary = {"n_seeds": len(all_seed), "regimes": {}}
    regimes = ["ALL", "IN", "OUT"]
    variants = ["V2", "V_bad", "V2_bad"]

    for r in regimes:
        rs = []
        for s in all_seed:
            if s["regimes"][r].get("skipped"):
                continue
            rs.append(s["regimes"][r])
        if not rs:
            summary["regimes"][r] = {"skipped": True}
            continue
        out = {
            "n_seeds_present": len(rs),
            "n_variants_first": rs[0]["n_variants"],
            "n_genes_first": rs[0]["n_genes"],
            "class_dist_first": rs[0]["class_dist_variants"],
        }
        for v in variants:
            for metric in ["macro_f1_mean", "per_gene_f1_mean"]:
                vals = [
                    x[v].get(metric)
                    for x in rs
                    if v in x and x[v].get(metric) is not None
                ]
                stem = f"{v}_{metric.replace('_mean','')}"
                if vals:
                    out[f"{stem}_mean"] = float(np.mean(vals))
                    out[f"{stem}_std"] = float(np.std(vals))
            for cls in CLASSES:
                vals = [
                    x[v].get(f"auroc_{cls}_mean")
                    for x in rs
                    if v in x and x[v].get(f"auroc_{cls}_mean") is not None
                ]
                if vals:
                    out[f"{v}_auroc_{cls}_mean"] = float(np.mean(vals))
                    out[f"{v}_auroc_{cls}_std"] = float(np.std(vals))
        summary["regimes"][r] = out
    return summary


def print_table(summary):
    print("\n" + "=" * 84)
    print("LEAKAGE ANALYSIS — macro-F1 (per-gene) and DN/GOF AUROC by regime")
    print("=" * 84)
    print(
        f"{'Regime':<6} {'N_var':>6} {'N_gene':>7}  {'V2 F1':>14} {'V_bad F1':>14} {'V2+bad F1':>14}"
    )
    print("-" * 84)
    for r in ["ALL", "IN", "OUT"]:
        if "skipped" in summary["regimes"].get(r, {}):
            print(f"{r:<6} SKIPPED")
            continue
        x = summary["regimes"][r]

        def f(k):
            m = x.get(f"{k}_per_gene_f1_mean")
            s = x.get(f"{k}_per_gene_f1_std")
            if m is None:
                return "   N/A   "
            return f"{m:.3f}±{s:.3f}"

        print(
            f"{r:<6} {x['n_variants_first']:>6} {x['n_genes_first']:>7}  "
            f"{f('V2'):>14} {f('V_bad'):>14} {f('V2_bad'):>14}"
        )

    print("\nDN AUROC")
    print(f"{'Regime':<6}  {'V2':>14} {'V_bad':>14} {'V2+bad':>14}")
    print("-" * 60)
    for r in ["ALL", "IN", "OUT"]:
        if "skipped" in summary["regimes"].get(r, {}):
            continue
        x = summary["regimes"][r]

        def f(k):
            m = x.get(f"{k}_auroc_DN_mean")
            s = x.get(f"{k}_auroc_DN_std")
            if m is None:
                return "   N/A   "
            return f"{m:.3f}±{s:.3f}"

        print(f"{r:<6}  {f('V2'):>14} {f('V_bad'):>14} {f('V2_bad'):>14}")

    print("\nGOF AUROC")
    print(f"{'Regime':<6}  {'V2':>14} {'V_bad':>14} {'V2+bad':>14}")
    print("-" * 60)
    for r in ["ALL", "IN", "OUT"]:
        if "skipped" in summary["regimes"].get(r, {}):
            continue
        x = summary["regimes"][r]

        def f(k):
            m = x.get(f"{k}_auroc_GOF_mean")
            s = x.get(f"{k}_auroc_GOF_std")
            if m is None:
                return "   N/A   "
            return f"{m:.3f}±{s:.3f}"

        print(f"{r:<6}  {f('V2'):>14} {f('V_bad'):>14} {f('V2_bad'):>14}")

    # Leakage gauge
    print("\nLeakage gauge (IN − OUT, V_bad):")
    if not summary["regimes"].get("IN", {}).get("skipped") and not summary[
        "regimes"
    ].get("OUT", {}).get("skipped"):
        i = summary["regimes"]["IN"]
        o = summary["regimes"]["OUT"]
        for metric_name, key in [
            ("per-gene F1", "V_bad_per_gene_f1_mean"),
            ("macro-F1", "V_bad_macro_f1_mean"),
            ("DN AUROC", "V_bad_auroc_DN_mean"),
            ("GOF AUROC", "V_bad_auroc_GOF_mean"),
            ("LOF AUROC", "V_bad_auroc_LOF_mean"),
        ]:
            iv = i.get(key)
            ov = o.get(key)
            if iv is not None and ov is not None:
                print(
                    f"  {metric_name:<14}: IN={iv:.3f}  OUT={ov:.3f}  "
                    f"Δ(IN−OUT)={iv-ov:+.3f}"
                )
    print("=" * 84)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None, help="Single seed; default: 0..4")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    seeds = [args.seed] if args.seed is not None else list(range(5))
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR / "badonyi_leakage"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Loading data ===")
    labels, genes = load_data()
    pfam_map = load_pfam()

    print("\n=== Broadcasting feature matrices ===")
    gene_to_row = build_gene_to_row()
    prot = np.load(PROTEOME_FEATURES).astype(np.float32)
    bad_full = np.load(BADONYI_FEATURES).astype(np.float32)
    X_prot = broadcast(genes, prot, gene_to_row)
    X_bad = broadcast(genes, bad_full[:, BADONYI_RAW_COLS], gene_to_row)
    print(f"  X_prot {X_prot.shape}  X_bad {X_bad.shape}")

    print("\n=== Badonyi training flags ===")
    train_any, train_per_class = load_badonyi_train_flags()

    # Per-variant in-training summary
    in_bad = np.array([train_any.get(g, 0) for g in genes], dtype=int)
    print(f"\nPer-variant 'gene in Badonyi train':")
    print(f"  Total variants: {len(in_bad)}")
    print(f"  In Badonyi train: {int(in_bad.sum())} " f"({100*in_bad.mean():.1f}%)")

    # Stratified by class
    print("\nClass-stratified gene overlap with Badonyi train:")
    unique_genes_labels = {}
    for g, lab in zip(genes, labels):
        unique_genes_labels[g] = lab
    counts_in = Counter()
    counts_total = Counter()
    for g, lab in unique_genes_labels.items():
        counts_total[lab] += 1
        if train_any.get(g, 0):
            counts_in[lab] += 1
    for cls in CLASSES:
        tot = counts_total[cls]
        ib = counts_in[cls]
        print(
            f"  {cls}: {ib}/{tot} ({100*ib/tot:.1f}%) of labeled genes in Badonyi train"
        )

    all_seed_results = []
    for s in seeds:
        sr = run_seed(
            s, args.n_folds, labels, genes, pfam_map, X_prot, X_bad, train_any
        )
        all_seed_results.append(sr)
        path = out_dir / f"leakage_seed{s}.json"
        path.write_text(json.dumps(sr, indent=2))
        print(f"  Saved per-seed: {path}")

    summary = aggregate_seeds(all_seed_results)
    summary_path = out_dir / "leakage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary: {summary_path}")
    print_table(summary)


if __name__ == "__main__":
    main()
