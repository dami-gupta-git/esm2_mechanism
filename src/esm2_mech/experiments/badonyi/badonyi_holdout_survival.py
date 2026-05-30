"""
Badonyi-survives-holdouts (plan_badonyi.md)

Tests whether Badonyi & Marsh 2024's published gene-level mechanism predictor
(pDN, pGOF, pLOF) maintains its reported AUROCs under the project's strict
holdout protocols.

Distinct from result_15's V_bad — we use Badonyi's raw published predictions
directly, no retrained LogReg. The published model is the model on test.

For each holdout protocol H ∈ {none-as-baseline, Pfam family, MMseqs2-20}:
  1. Split labeled genes into 5 folds holding out entire groups (families/clusters)
  2. For each held-out gene, take its raw pDN/pGOF/pLOF as the prediction
  3. Concatenate across folds → one prediction per gene
  4. Compute three binary AUROCs as in Badonyi 2024:
       DN-vs-LOF:        among DN+LOF genes, AUROC(pDN,  true=DN)
       GOF-vs-LOF:       among GOF+LOF genes, AUROC(pGOF, true=GOF)
       LOF-vs-non-LOF:   among all labeled,   AUROC(pLOF, true=LOF)
  5. 5 seeds, report mean ± std

Also: stratified by Badonyi training-set membership (IN vs OUT) under
each holdout, to separate generalisation from train-leakage.

Outputs:
  results/badonyi_survival/badonyi_survival_summary.json
  results/badonyi_survival/badonyi_survival_seed{0..4}.json

Usage:
  python3 scripts/badonyi_holdout_survival.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter


import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import functools

print = functools.partial(print, flush=True)

from esm2_mech.utils.paths import DATA_DIR, RESULTS_DIR, GENE_LIST_TSV

warnings.filterwarnings("ignore")

MERGED_GENE_LIST = GENE_LIST_TSV
PFAM_FAMILIES = DATA_DIR / "pfam_families.json"
MMSEQS_CLUSTERS = DATA_DIR / "mmseqs_clusters.json"
BADONYI_S3 = DATA_DIR / "cache" / "badonyi" / "table_S3.xlsx"

OUT_DIR = RESULTS_DIR / "badonyi_survival"

CLASSES_3 = ["GOF", "DN", "LOF"]
N_FOLDS = 5


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_gene_table():
    """Per-gene table with raw Badonyi pDN/pGOF/pLOF, family, cluster,
    training-set membership, 3-class label."""
    print(f"Loading merged gene list...")
    ml = pd.read_csv(MERGED_GENE_LIST, sep="\t", dtype=str)
    print(f"  {len(ml)} merged genes")

    print(f"Loading Badonyi S3...")
    s3 = pd.read_excel(BADONYI_S3, sheet_name="table_S3")

    # Parse train flag "DN|GOF|LOF"
    def parse(s):
        if pd.isna(s):
            return (0, 0, 0)
        return tuple(int(b) for b in str(s).split("|"))

    parts = s3["train_dn_gof_lof"].map(parse)
    s3["tr_DN"] = [p[0] for p in parts]
    s3["tr_GOF"] = [p[1] for p in parts]
    s3["tr_LOF"] = [p[2] for p in parts]
    s3["in_any"] = (s3[["tr_DN", "tr_GOF", "tr_LOF"]].sum(axis=1) > 0).astype(int)
    print(f"  {len(s3)} S3 rows.  In any train: {int(s3['in_any'].sum())}")

    print(f"Loading Pfam families and MMseqs2 clusters...")
    with open(PFAM_FAMILIES) as f:
        pfam = json.load(f)
    with open(MMSEQS_CLUSTERS) as f:
        gene_to_cluster = json.load(f)["gene_to_cluster"]

    # Build per-gene table
    df = ml[["gene", "mechanism"]].drop_duplicates(subset=["gene"]).copy()

    def collapse(m):
        if m in ("GOF", "DN"):
            return m
        if m in ("HI", "AR", "LOF"):
            return "LOF"
        return None

    df["label3"] = df["mechanism"].map(collapse)

    s3_lookup = s3.set_index("gene")
    for col in ["pDN", "pGOF", "pLOF", "tr_DN", "tr_GOF", "tr_LOF", "in_any"]:
        df[col] = df["gene"].map(
            lambda g, c=col: s3_lookup[c].get(g) if g in s3_lookup.index else None
        )

    df["pfam"] = df["gene"].map(lambda g: pfam.get(g))
    df["cluster"] = df["gene"].map(lambda g: gene_to_cluster.get(g))

    print(f"  Labeled (3-class): {df['label3'].notna().sum()}/{len(df)}")
    print(f"  With Badonyi predictions: {df['pDN'].notna().sum()}/{len(df)}")
    print(f"  With Pfam family: {df['pfam'].notna().sum()}/{len(df)}")
    print(f"  With MMseqs2 cluster: {df['cluster'].notna().sum()}/{len(df)}")

    return df


# ---------------------------------------------------------------------------
# Group-based holdout
# ---------------------------------------------------------------------------


def assign_folds(df, group_col, n_folds, seed):
    """Return per-row fold assignment. Rows where group_col is None get
    fold=-1 (excluded from CV)."""
    groups_present = sorted([g for g in df[group_col].dropna().unique()])
    rng = np.random.RandomState(seed)
    rng.shuffle(groups_present)
    fold_of_group = {g: i % n_folds for i, g in enumerate(groups_present)}
    return df[group_col].map(lambda g: fold_of_group.get(g, -1)).values.astype(int)


# ---------------------------------------------------------------------------
# AUROC evaluation
# ---------------------------------------------------------------------------


def compute_aurocs(df, mask=None):
    """Three binary AUROCs from raw Badonyi predictions.

    mask is a boolean Series subsetting df. If None, use all rows with a label
    and a Badonyi prediction.
    """
    base = df.copy()
    if mask is not None:
        base = base[mask].copy()
    base = base[base["label3"].notna() & base["pDN"].notna()].copy()

    out = {
        "n_total": int(len(base)),
        "class_dist": dict(Counter(base["label3"].tolist())),
    }

    # DN-vs-LOF
    sub = base[base["label3"].isin(["DN", "LOF"])].copy()
    if sub["label3"].nunique() == 2 and len(sub) >= 5:
        y = (sub["label3"] == "DN").astype(int)
        out["DN_vs_LOF"] = float(roc_auc_score(y, sub["pDN"]))
        out["DN_vs_LOF_n_pos"] = int(y.sum())
        out["DN_vs_LOF_n_neg"] = int((1 - y).sum())
    else:
        out["DN_vs_LOF"] = None

    # GOF-vs-LOF
    sub = base[base["label3"].isin(["GOF", "LOF"])].copy()
    if sub["label3"].nunique() == 2 and len(sub) >= 5:
        y = (sub["label3"] == "GOF").astype(int)
        out["GOF_vs_LOF"] = float(roc_auc_score(y, sub["pGOF"]))
        out["GOF_vs_LOF_n_pos"] = int(y.sum())
        out["GOF_vs_LOF_n_neg"] = int((1 - y).sum())
    else:
        out["GOF_vs_LOF"] = None

    # LOF-vs-non-LOF
    if base["label3"].nunique() >= 2:
        y = (base["label3"] == "LOF").astype(int)
        if 0 < y.sum() < len(y):
            out["LOF_vs_nonLOF"] = float(roc_auc_score(y, base["pLOF"]))
            out["LOF_vs_nonLOF_n_pos"] = int(y.sum())
            out["LOF_vs_nonLOF_n_neg"] = int((1 - y).sum())
        else:
            out["LOF_vs_nonLOF"] = None
    else:
        out["LOF_vs_nonLOF"] = None

    return out


def run_holdout(df, group_col, n_folds, seed):
    """Compute AUROCs after fold assignment by group_col. Held-out predictions
    are concatenated across folds — but since Badonyi's predictions are
    fixed (no retraining), this is equivalent to computing AUROC on the full
    labeled set restricted to rows that have a group assignment. The CV
    structure matters only because it tests whether the held-out cluster's
    predictions are correlated within-cluster vs between-cluster — in this
    setup, that doesn't actually change the AUROC since we never retrain.

    However, we keep the CV-style aggregation for two reasons:
      1. To match the project's other CV evaluations in framing.
      2. To allow fold-level variance estimation (each fold's AUROC).
    """
    folds = assign_folds(df, group_col, n_folds, seed)
    # Rows without a group are excluded
    mask = folds != -1
    sub = df[mask].copy().reset_index(drop=True)
    sub["fold"] = folds[mask]

    fold_results = []
    for k in range(n_folds):
        fold_mask = sub["fold"] == k
        if fold_mask.sum() == 0:
            continue
        fm = compute_aurocs(sub, mask=fold_mask)
        fm["fold"] = k
        fold_results.append(fm)

    # Aggregate held-out predictions across folds = full set, since no
    # retraining
    agg = compute_aurocs(sub)
    # Mean across folds (gives a fold-variance estimate)
    fold_mean = {}
    for key in ["DN_vs_LOF", "GOF_vs_LOF", "LOF_vs_nonLOF"]:
        vals = [f[key] for f in fold_results if f.get(key) is not None]
        if vals:
            fold_mean[key + "_fold_mean"] = float(np.mean(vals))
            fold_mean[key + "_fold_std"] = float(np.std(vals))
            fold_mean[key + "_n_folds_valid"] = len(vals)

    return {
        "all_holdout": agg,
        "fold_mean_aurocs": fold_mean,
        "folds": fold_results,
        "n_rows_with_group": int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# Seed runner
# ---------------------------------------------------------------------------


def _fmt(v):
    return f"{v:.3f}" if v is not None else "N/A"


def run_seed(df, seed, n_folds):
    print(f"\n{'='*72}\nSEED {seed}\n{'='*72}")
    out = {"seed": seed}

    # Baseline: no holdout (whole-set AUROC with Badonyi predictions)
    print("  Baseline (no holdout) — Badonyi raw on whole labeled set")
    out["baseline_no_holdout"] = compute_aurocs(df)
    print(f"    DN-vs-LOF: {_fmt(out['baseline_no_holdout']['DN_vs_LOF'])}")
    print(f"    GOF-vs-LOF: {_fmt(out['baseline_no_holdout']['GOF_vs_LOF'])}")
    print(f"    LOF-vs-nonLOF: {_fmt(out['baseline_no_holdout']['LOF_vs_nonLOF'])}")

    # Family-split holdout
    print("\n  Pfam family-split holdout")
    out["family_holdout"] = run_holdout(df, "pfam", n_folds, seed)
    fa = out["family_holdout"]["all_holdout"]
    print(
        f"    All-rows (held-out): DN={_fmt(fa['DN_vs_LOF'])}  "
        f"GOF={_fmt(fa['GOF_vs_LOF'])}  LOF={_fmt(fa['LOF_vs_nonLOF'])}"
    )

    # MMseqs2-20 holdout
    print("\n  MMseqs2-20 cluster-split holdout")
    out["mmseqs_holdout"] = run_holdout(df, "cluster", n_folds, seed)
    ma = out["mmseqs_holdout"]["all_holdout"]
    print(
        f"    All-rows (held-out): DN={_fmt(ma['DN_vs_LOF'])}  "
        f"GOF={_fmt(ma['GOF_vs_LOF'])}  LOF={_fmt(ma['LOF_vs_nonLOF'])}"
    )

    # Stratified: IN vs OUT of Badonyi training
    print("\n  Stratified by Badonyi training-set membership (in any classifier)")
    in_mask = df["in_any"] == 1
    out_mask = df["in_any"] == 0
    out["badonyi_in_train"] = compute_aurocs(df, mask=in_mask)
    out["badonyi_out_train"] = compute_aurocs(df, mask=out_mask)
    print(
        f"    IN-Badonyi: n={out['badonyi_in_train']['n_total']}, "
        f"DN={out['badonyi_in_train'].get('DN_vs_LOF', 'NA')}, "
        f"GOF={out['badonyi_in_train'].get('GOF_vs_LOF', 'NA')}, "
        f"LOF={out['badonyi_in_train'].get('LOF_vs_nonLOF', 'NA')}"
    )
    print(
        f"    OUT-Badonyi: n={out['badonyi_out_train']['n_total']}, "
        f"DN={out['badonyi_out_train'].get('DN_vs_LOF', 'NA')}, "
        f"GOF={out['badonyi_out_train'].get('GOF_vs_LOF', 'NA')}, "
        f"LOF={out['badonyi_out_train'].get('LOF_vs_nonLOF', 'NA')}"
    )

    return out


def aggregate_seeds(all_seed):
    """Mean ± std of the 'all_holdout' AUROC across seeds, per holdout."""
    summary = {"n_seeds": len(all_seed)}

    for key, label in [
        ("baseline_no_holdout", "baseline_no_holdout"),
        ("family_holdout", "family_holdout"),
        ("mmseqs_holdout", "mmseqs_holdout"),
        ("badonyi_in_train", "badonyi_in_train"),
        ("badonyi_out_train", "badonyi_out_train"),
    ]:
        cur = {}
        for metric in ["DN_vs_LOF", "GOF_vs_LOF", "LOF_vs_nonLOF"]:
            vals = []
            for s in all_seed:
                payload = s[key]
                # Family/MMseqs go through "all_holdout"
                if "all_holdout" in payload:
                    payload = payload["all_holdout"]
                v = payload.get(metric)
                if v is not None:
                    vals.append(v)
            if vals:
                cur[f"{metric}_mean"] = float(np.mean(vals))
                cur[f"{metric}_std"] = float(np.std(vals))
                cur[f"{metric}_n_seeds"] = len(vals)
        # Also keep ns for context (from first seed only)
        first_payload = all_seed[0][key]
        if "all_holdout" in first_payload:
            first_payload = first_payload["all_holdout"]
        cur["n_total_first"] = first_payload.get("n_total")
        summary[label] = cur

    # Deltas vs baseline
    base = summary["baseline_no_holdout"]
    deltas = {}
    for h_key in ["family_holdout", "mmseqs_holdout"]:
        d = {}
        for metric in ["DN_vs_LOF", "GOF_vs_LOF", "LOF_vs_nonLOF"]:
            bm = base.get(f"{metric}_mean")
            hm = summary[h_key].get(f"{metric}_mean")
            if bm is not None and hm is not None:
                d[f"delta_{metric}"] = float(hm - bm)
        deltas[h_key] = d
    summary["deltas_vs_baseline"] = deltas

    return summary


def print_table(summary):
    print("\n" + "=" * 100)
    print("BADONYI SURVIVAL — 5-seed mean ± std AUROC under different holdouts")
    print("=" * 100)
    print(f"{'Holdout':<28} {'DN-vs-LOF':<22} {'GOF-vs-LOF':<22} {'LOF-vs-nonLOF':<22}")
    print("-" * 100)

    def fmt(d, key):
        m = d.get(f"{key}_mean")
        s = d.get(f"{key}_std")
        if m is None:
            return "      N/A      "
        return f"{m:.3f} ± {s:.3f}"

    for h, label in [
        ("baseline_no_holdout", "none (whole labeled set)"),
        ("family_holdout", "Pfam family-split"),
        ("mmseqs_holdout", "MMseqs2-20 cluster-split"),
        ("badonyi_in_train", "Badonyi IN-train (no h-out)"),
        ("badonyi_out_train", "Badonyi OUT-train (no h-out)"),
    ]:
        d = summary[h]
        n = d.get("n_total_first", "—")
        print(
            f"{label:<28} {fmt(d,'DN_vs_LOF'):<22} {fmt(d,'GOF_vs_LOF'):<22} {fmt(d,'LOF_vs_nonLOF'):<22}  n={n}"
        )

    print(
        "\nΔ AUROC vs no-holdout baseline (pre-registered: ≥−0.03 robust, ≤−0.10 mostly leakage)"
    )
    print("-" * 100)
    for h_key, label in [
        ("family_holdout", "Pfam family-split"),
        ("mmseqs_holdout", "MMseqs2-20 cluster-split"),
    ]:
        d = summary["deltas_vs_baseline"][h_key]

        def f(k):
            v = d.get(f"delta_{k}")
            if v is None:
                return "    N/A    "
            tag = ""
            if v >= -0.03:
                tag = "  ROBUST"
            elif v <= -0.10:
                tag = "  MOSTLY-LEAKAGE"
            else:
                tag = "  PARTIAL"
            return f"{v:+.3f}{tag}"

        print(
            f"{label:<28} DN: {f('DN_vs_LOF'):<24} GOF: {f('GOF_vs_LOF'):<24} LOF: {f('LOF_vs_nonLOF')}"
        )
    print("=" * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed] if args.seed is not None else list(range(5))

    df = load_gene_table()

    all_seed_results = []
    for s in seeds:
        sr = run_seed(df, s, args.n_folds)
        all_seed_results.append(sr)
        path = OUT_DIR / f"badonyi_survival_seed{s}.json"
        path.write_text(json.dumps(sr, indent=2))
        print(f"  Saved: {path}")

    summary = aggregate_seeds(all_seed_results)
    spath = OUT_DIR / "badonyi_survival_summary.json"
    spath.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary: {spath}")
    print_table(summary)


if __name__ == "__main__":
    main()
