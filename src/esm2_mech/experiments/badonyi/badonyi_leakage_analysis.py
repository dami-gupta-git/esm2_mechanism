"""Quantify label-leakage from Badonyi training-set overlap by comparing V_bad/V2+bad on IN vs OUT genes."""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from esm2_mech.utils.constants import MECHANISM_CLASSES, BOOTSTRAP_N_RESAMPLES
from esm2_mech.utils.data import build_gene_to_row as _build_gene_to_row, load_pfam_map
from esm2_mech.utils.splits import family_split_indices
from esm2_mech.utils.probes import run_histgb_cv
from esm2_mech.utils.bootstrap import attach_mechanism_ci, family_or_gene_clusters
from esm2_mech.utils.classification import validate_complete_classification_splits
from esm2_mech.utils.paths import (
    BADONYI_CACHE_DIR,
    GENE_UNIVERSE,
    PFAM_JSON,
    PROTEOME_FEATURES_ALIGNED,
    PROTEOME_FEATURE_COLUMNS_JSON,
    PROTEOME_FEATURES_TSV,
    BADONYI_FEATURES_ALIGNED,
    BADONYI_FEATURE_COLUMNS_JSON,
    BADONYI_FEATURES_TSV,
    RESULTS_DIR,
    VALID_VARIANTS_JSON,
)
import functools

print = functools.partial(print, flush=True)

OUT_DIR = RESULTS_DIR

warnings.filterwarnings("ignore")

MERGED_VALID_VARIANTS = VALID_VARIANTS_JSON
PROTEOME_FEATURES = PROTEOME_FEATURES_ALIGNED
BADONYI_FEATURES = BADONYI_FEATURES_ALIGNED
BADONYI_S3 = BADONYI_CACHE_DIR / "table_S3.xlsx"
BADONYI_RAW_COLS = [0, 1, 2]  # pDN, pGOF, pLOF

# MUST be GENE_UNIVERSE, not GENE_LIST_TSV: .npy rows are in gene_universe.tsv order.
MERGED_GENE_LIST = GENE_UNIVERSE
PFAM_FAMILIES = PFAM_JSON

CLASSES = MECHANISM_CLASSES


def load_data():
    with open(MERGED_VALID_VARIANTS) as f:
        variants = json.load(f)
    labels = np.array([v["label_3class"] for v in variants])
    genes = np.array([v["gene"] for v in variants])
    print(f"Loaded {len(variants)} variants, {len(set(genes.tolist()))} genes")
    return labels, genes


def build_gene_to_row():
    return _build_gene_to_row(MERGED_GENE_LIST)


# NaN not 0.0 for missing genes: 0.0 is a plausible real observation.
def broadcast(genes, matrix, gene_to_row):
    """Broadcast per-gene features to variant rows, NaN where the gene has no row."""
    n, d = len(genes), matrix.shape[1]
    X = np.full((n, d), np.nan, dtype=np.float32)
    n_missing = 0
    for i, g in enumerate(genes):
        r = gene_to_row.get(g)
        if r is not None and r < matrix.shape[0]:
            X[i] = matrix[r]
        else:
            n_missing += 1
    if n_missing:
        print(f"  {n_missing}/{n} rows have no feature row for their gene (NaN)")
    return X


def load_badonyi_train_flags():
    """Return (any-train dict, per-classifier dict) from Badonyi S3."""
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


# Nothing is imputed: filling before the split would leak test-fold statistics into training.
# CI resamples families not genes; cluster array is derived via pfam_map since `genes` are real gene ids.
def run_probe(
    X, y, genes, groups, n_folds, seed, label, pfam_map,
    compute_ci=True, n_boot=BOOTSTRAP_N_RESAMPLES,
):
    """NaN-native family-split CV."""
    splits = list(family_split_indices(groups, n_folds, seed))
    contract = validate_complete_classification_splits(
        splits, requested_folds=n_folds,
        eligible_rows=np.concatenate([test for _train, test in splits]),
        labels=y, classes=MECHANISM_CLASSES, groups=groups, held_out_unit="family",
    )
    agg, oof = run_histgb_cv(
        X, y, splits, MECHANISM_CLASSES, contract, seed=seed,
        genes=genes, label=label, return_oof=True
    )
    clusters = (
        family_or_gene_clusters(oof["genes"], pfam_map, is_family_split=True)
        if oof is not None
        else None
    )
    return attach_mechanism_ci(
        agg,
        oof,
        clusters,
        compute_ci=compute_ci,
        n_resamples=n_boot,
        seed=seed,
    )


def run_regime(
    regime_name, mask, X_prot, X_bad, y, genes, groups, n_folds, seed, pfam_map,
    compute_ci=True, n_boot=BOOTSTRAP_N_RESAMPLES,
):
    """Apply mask, run V2/V_bad/V2+bad; skip if fewer than 100 variants."""
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
    cd = {c: int(class_counts.get(c, 0)) for c in CLASSES}
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

    def _log_ci(agg):
        ci = agg.get("ci", {}).get("macro_f1")
        if not ci or ci.get("ci_suppressed"):
            return ""
        return f"  CI=[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}] ({ci['n_clusters']} families)"

    kwargs = dict(pfam_map=pfam_map, compute_ci=compute_ci, n_boot=n_boot)
    print(f"    V2 (proteome 37):")
    res["V2"] = run_probe(X_p, y_m, genes_m, groups_m, n_folds, seed, "V2", **kwargs)
    print(f"    macro_f1={res['V2'].get('macro_f1_mean', float('nan')):.4f}" + _log_ci(res["V2"]))
    print(f"    V_bad (pDN/pGOF/pLOF):")
    res["V_bad"] = run_probe(X_b, y_m, genes_m, groups_m, n_folds, seed, "V_bad", **kwargs)
    print(f"    macro_f1={res['V_bad'].get('macro_f1_mean', float('nan')):.4f}" + _log_ci(res["V_bad"]))
    print(f"    V2+bad (40):")
    res["V2_bad"] = run_probe(X_2b, y_m, genes_m, groups_m, n_folds, seed, "V2+bad", **kwargs)
    print(f"    macro_f1={res['V2_bad'].get('macro_f1_mean', float('nan')):.4f}" + _log_ci(res["V2_bad"]))
    return res


def run_seed(
    seed, n_folds, y, genes, pfam_map, X_prot, X_bad, train_flag_any,
    compute_ci=True, n_boot=BOOTSTRAP_N_RESAMPLES,
):
    print(f"\n{'='*72}\nSEED {seed}\n{'='*72}")

    # y stays string labels: LabelEncoder alphabetical order mismatches MECHANISM_CLASSES order.

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
            name, mask, X_prot, X_bad, y, genes, gene_pfam, n_folds, seed, pfam_map,
            compute_ci=compute_ci, n_boot=n_boot,
        )
    return seed_results


def aggregate_seeds(all_seed):
    """Aggregate macro-F1, per-gene-F1, and AUROC across seeds per regime and variant."""
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

            # CI bounds pooled across seeds — separate from seed-to-seed std.
            for bound in ("ci_low", "ci_high"):
                vals = [
                    x[v]["ci"]["macro_f1"].get(bound)
                    for x in rs
                    if v in x
                    and x[v].get("ci", {}).get("macro_f1")
                    and not x[v]["ci"]["macro_f1"].get("ci_suppressed")
                ]
                if vals:
                    out[f"{v}_macro_f1_{bound}_seed_mean"] = float(np.mean(vals))
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
    ap.add_argument("--no_ci", action="store_true", help="skip cluster-bootstrap CIs")
    ap.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    args = ap.parse_args()

    seeds = [args.seed] if args.seed is not None else list(range(5))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading data ===")
    labels, genes = load_data()
    pfam_map = load_pfam_map(PFAM_FAMILIES)

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
            s, args.n_folds, labels, genes, pfam_map, X_prot, X_bad, train_any,
            compute_ci=not args.no_ci, n_boot=args.n_boot,
        )
        all_seed_results.append(sr)
        path = OUT_DIR / f"leakage_seed{s}.json"
        path.write_text(json.dumps(sr, indent=2))
        print(f"  Saved per-seed: {path}")

    summary = aggregate_seeds(all_seed_results)
    summary_path = OUT_DIR / "leakage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary: {summary_path}")
    print_table(summary)


if __name__ == "__main__":
    main()
