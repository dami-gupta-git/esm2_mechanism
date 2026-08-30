"""Test whether Badonyi 2024's raw published pDN/pGOF/pLOF AUROCs hold under family and cluster holdouts."""

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

from esm2_mech.utils.constants import MECHANISM_CLASSES, N_FOLDS, N_SEEDS, BOOTSTRAP_N_RESAMPLES
from esm2_mech.utils.bootstrap import binary_auroc_cluster_bootstrap_ci
from esm2_mech.utils.seed_aggregation import (
    aggregate_paired_seed_difference,
    aggregate_seed_results,
    make_seed_record,
    read_seed_inference,
    read_seed_point_estimate,
    seed_result_contract,
)
from esm2_mech.experiments.mechanism.seed_results import aggregate_result_contract
from esm2_mech.utils.paths import BADONYI_CACHE_DIR, DATA_DIR, GENE_LIST_TSV, RESULTS_DIR

warnings.filterwarnings("ignore")

MERGED_GENE_LIST = GENE_LIST_TSV
PFAM_FAMILIES = DATA_DIR / "pfam_families.json"
MMSEQS_CLUSTERS = DATA_DIR / "mmseqs_clusters.json"
BADONYI_S3 = BADONYI_CACHE_DIR / "table_S3.xlsx"

OUT_DIR = RESULTS_DIR / "badonyi_survival"

CLASSES_3 = MECHANISM_CLASSES


def load_gene_table():
    """Per-gene table with raw Badonyi predictions, family, cluster, train flag, and label."""
    print(f"Loading merged gene list...")
    ml = pd.read_csv(MERGED_GENE_LIST, sep="\t", dtype=str)
    print(f"  {len(ml)} merged genes")

    print(f"Loading Badonyi S3...")
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
    print(f"  {len(s3)} S3 rows.  In any train: {int(s3['in_any'].sum())}")

    print(f"Loading Pfam families and MMseqs2 clusters...")
    with open(PFAM_FAMILIES) as f:
        pfam = json.load(f)
    with open(MMSEQS_CLUSTERS) as f:
        gene_to_cluster = json.load(f)["gene_to_cluster"]

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


def assign_folds(df, group_col, n_folds, seed):
    """Return per-row fold assignment; rows where group_col is None get fold=-1."""
    groups_present = sorted([g for g in df[group_col].dropna().unique()])
    rng = np.random.RandomState(seed)
    rng.shuffle(groups_present)
    fold_of_group = {g: i % n_folds for i, g in enumerate(groups_present)}
    return df[group_col].map(lambda g: fold_of_group.get(g, -1)).values.astype(int)


# cluster_col is the resampling unit: family or cluster for holdout arms, None for baseline/IN/OUT.
def _ci_for_binary(sub, y, score_col, cluster_col, compute_ci, n_boot, seed):
    """Cluster-bootstrap CI for one binary AUROC."""
    if not compute_ci:
        return None
    genes = sub["gene"].values
    clusters = sub[cluster_col].values if cluster_col else None
    oof = {"y_true": y.values.astype(int), "proba": sub[score_col].values, "genes": genes}
    return binary_auroc_cluster_bootstrap_ci(
        oof, n_resamples=n_boot, seed=seed, clusters=clusters
    )


def compute_aurocs(
    df, mask=None, compute_ci=False, n_boot=BOOTSTRAP_N_RESAMPLES, seed=0, cluster_col=None,
):
    """Three binary AUROCs (DN-vs-LOF, GOF-vs-LOF, LOF-vs-nonLOF) from raw Badonyi predictions."""
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
        ci = _ci_for_binary(sub, y, "pDN", cluster_col, compute_ci, n_boot, seed)
        if ci is not None:
            out["DN_vs_LOF_ci"] = ci
    else:
        out["DN_vs_LOF"] = None

    # GOF-vs-LOF
    sub = base[base["label3"].isin(["GOF", "LOF"])].copy()
    if sub["label3"].nunique() == 2 and len(sub) >= 5:
        y = (sub["label3"] == "GOF").astype(int)
        out["GOF_vs_LOF"] = float(roc_auc_score(y, sub["pGOF"]))
        out["GOF_vs_LOF_n_pos"] = int(y.sum())
        out["GOF_vs_LOF_n_neg"] = int((1 - y).sum())
        ci = _ci_for_binary(sub, y, "pGOF", cluster_col, compute_ci, n_boot, seed)
        if ci is not None:
            out["GOF_vs_LOF_ci"] = ci
    else:
        out["GOF_vs_LOF"] = None

    # LOF-vs-non-LOF
    if base["label3"].nunique() >= 2:
        y = (base["label3"] == "LOF").astype(int)
        if 0 < y.sum() < len(y):
            out["LOF_vs_nonLOF"] = float(roc_auc_score(y, base["pLOF"]))
            out["LOF_vs_nonLOF_n_pos"] = int(y.sum())
            out["LOF_vs_nonLOF_n_neg"] = int((1 - y).sum())
            ci = _ci_for_binary(base, y, "pLOF", cluster_col, compute_ci, n_boot, seed)
            if ci is not None:
                out["LOF_vs_nonLOF_ci"] = ci
        else:
            out["LOF_vs_nonLOF"] = None
    else:
        out["LOF_vs_nonLOF"] = None

    return out


def run_holdout(df, group_col, n_folds, seed, compute_ci=True, n_boot=BOOTSTRAP_N_RESAMPLES):
    """Compute AUROCs under group-based fold assignment."""
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
    # retraining. CI resamples group_col (the holdout unit), not genes, so the
    # interval matches what this holdout arm actually claims robustness to.
    agg = compute_aurocs(
        sub, compute_ci=compute_ci, n_boot=n_boot, seed=seed, cluster_col=group_col,
    )
    # Mean across folds (gives a fold-variance estimate)
    fold_mean = {}
    for key in ["DN_vs_LOF", "GOF_vs_LOF", "LOF_vs_nonLOF"]:
        vals = [f[key] for f in fold_results if f.get(key) is not None]
        fold_mean[key + "_n_folds_valid"] = len(vals)
        if len(vals) == n_folds and np.isfinite(vals).all():
            fold_mean[key + "_fold_mean"] = float(np.mean(vals))
            fold_mean[key + "_fold_std"] = float(np.std(vals))
        else:
            fold_mean[key + "_fold_mean"] = None
            fold_mean[key + "_fold_std"] = None

    return {
        "all_holdout": agg,
        "fold_mean_aurocs": fold_mean,
        "folds": fold_results,
        "n_rows_with_group": int(mask.sum()),
    }


def _fmt(v):
    return f"{v:.3f}" if v is not None else "N/A"


def compute_fixed_arms(df, compute_ci=True, n_boot=BOOTSTRAP_N_RESAMPLES):
    """The arms that have no fold assignment, so no dependence on the model seed.

    Badonyi's predictions are published, not refitted here, and these three arms
    score them on a fixed set of rows. Running them once per seed would repeat
    one number five times and report its zero spread as seed stability, so they
    are computed once and carry their cluster-bootstrap interval instead.
    """
    fixed = {}
    print("\nSeed-independent arms (no holdout, no fold assignment)")
    print("  Baseline (no holdout) — Badonyi raw on whole labeled set")
    fixed["baseline_no_holdout"] = compute_aurocs(
        df, compute_ci=compute_ci, n_boot=n_boot, seed=0
    )
    print(f"    DN-vs-LOF: {_fmt(fixed['baseline_no_holdout']['DN_vs_LOF'])}")
    print(f"    GOF-vs-LOF: {_fmt(fixed['baseline_no_holdout']['GOF_vs_LOF'])}")
    print(f"    LOF-vs-nonLOF: {_fmt(fixed['baseline_no_holdout']['LOF_vs_nonLOF'])}")

    print("\n  Stratified by Badonyi training-set membership (in any classifier)")
    fixed["badonyi_in_train"] = compute_aurocs(
        df, mask=df["in_any"] == 1, compute_ci=compute_ci, n_boot=n_boot, seed=0
    )
    fixed["badonyi_out_train"] = compute_aurocs(
        df, mask=df["in_any"] == 0, compute_ci=compute_ci, n_boot=n_boot, seed=0
    )
    for name, label in (
        ("badonyi_in_train", "IN-Badonyi"),
        ("badonyi_out_train", "OUT-Badonyi"),
    ):
        arm = fixed[name]
        print(
            f"    {label}: n={arm['n_total']}, "
            f"DN={_fmt(arm['DN_vs_LOF'])}, GOF={_fmt(arm['GOF_vs_LOF'])}, "
            f"LOF={_fmt(arm['LOF_vs_nonLOF'])}"
        )
    return fixed


def run_seed(df, seed, n_folds, compute_ci=True, n_boot=BOOTSTRAP_N_RESAMPLES):
    print(f"\n{'='*72}\nSEED {seed}\n{'='*72}")
    out = {**seed_result_contract(seed)}

    # Family-split holdout
    print("\n  Pfam family-split holdout")
    out["family_holdout"] = run_holdout(df, "pfam", n_folds, seed, compute_ci=compute_ci, n_boot=n_boot)
    fa = out["family_holdout"]["all_holdout"]
    print(
        f"    All-rows (held-out): DN={_fmt(fa['DN_vs_LOF'])}  "
        f"GOF={_fmt(fa['GOF_vs_LOF'])}  LOF={_fmt(fa['LOF_vs_nonLOF'])}"
    )

    # MMseqs2-20 holdout
    print("\n  MMseqs2-20 cluster-split holdout")
    out["mmseqs_holdout"] = run_holdout(df, "cluster", n_folds, seed, compute_ci=compute_ci, n_boot=n_boot)
    ma = out["mmseqs_holdout"]["all_holdout"]
    print(
        f"    All-rows (held-out): DN={_fmt(ma['DN_vs_LOF'])}  "
        f"GOF={_fmt(ma['GOF_vs_LOF'])}  LOF={_fmt(ma['LOF_vs_nonLOF'])}"
    )

    return out


def aggregate_seeds(all_seed, requested_seeds, fixed_arms):
    """Mean +/- std across seeds for the holdout arms; fixed arms pass through.

    Only the holdout arms depend on the seed, through their fold assignment. The
    fixed arms are stored as the single value they are, with their interval.
    """
    summary = {
        **aggregate_result_contract(),
        "requested_seeds": list(requested_seeds),
        "seed_independent_arms": fixed_arms,
    }

    for key in ["family_holdout", "mmseqs_holdout"]:
        cur = {}
        for metric in ["DN_vs_LOF", "GOF_vs_LOF", "LOF_vs_nonLOF"]:
            cur[f"{metric}_seed_aggregate"] = aggregate_seed_results(
                requested_seeds,
                all_seed,
                lambda result, holdout=key, name=metric: (
                    result[holdout]["all_holdout"].get(name)
                ),
            ).to_dict()
        # Also keep ns for context (from first seed only)
        cur["n_total_first"] = all_seed[0][key]["all_holdout"].get("n_total")
        summary[key] = cur

    # Deltas vs baseline. The baseline is one number, so each seed's difference
    # is that seed's holdout value minus the same constant.
    deltas = {}
    for h_key in ["family_holdout", "mmseqs_holdout"]:
        d = {}
        for metric in ["DN_vs_LOF", "GOF_vs_LOF", "LOF_vs_nonLOF"]:
            baseline_value = fixed_arms["baseline_no_holdout"].get(metric)
            baseline_records = [
                make_seed_record(result["seed"], baseline_value)
                for result in all_seed
            ]
            holdout_records = [
                make_seed_record(
                    result["seed"], result[h_key]["all_holdout"].get(metric)
                )
                for result in all_seed
            ]
            d[f"delta_{metric}_seed_aggregate"] = aggregate_paired_seed_difference(
                requested_seeds, holdout_records, baseline_records
            ).to_dict()
        deltas[h_key] = d
    summary["deltas_vs_baseline"] = deltas

    return summary


def print_table(summary):
    print("\n" + "=" * 100)
    print(
        "BADONYI SURVIVAL — AUROC under different holdouts. The holdout rows are "
        "mean ± std across model seeds; the rows without a holdout have no fold "
        "assignment, so they carry their bootstrap interval instead."
    )
    print("=" * 100)
    print(f"{'Holdout':<28} {'DN-vs-LOF':<22} {'GOF-vs-LOF':<22} {'LOF-vs-nonLOF':<22}")
    print("-" * 100)

    def fmt(d, key):
        metric = read_seed_inference(d.get(f"{key}_seed_aggregate", {}))
        if not metric.available:
            return "      N/A      "
        return f"{metric.value:.3f} ± {metric.spread:.3f}"

    def fmt_fixed(arm, key):
        value = arm.get(key)
        if value is None:
            return "      N/A      "
        interval = arm.get(f"{key}_ci")
        if not interval or interval.get("ci_low") is None:
            return f"{value:.3f}"
        return f"{value:.3f} [{interval['ci_low']:.3f}, {interval['ci_high']:.3f}]"

    for arm_key, label in [
        ("baseline_no_holdout", "none (whole labeled set)"),
        ("badonyi_in_train", "Badonyi IN-train (no h-out)"),
        ("badonyi_out_train", "Badonyi OUT-train (no h-out)"),
    ]:
        arm = summary["seed_independent_arms"][arm_key]
        print(
            f"{label:<28} {fmt_fixed(arm,'DN_vs_LOF'):<22} "
            f"{fmt_fixed(arm,'GOF_vs_LOF'):<22} "
            f"{fmt_fixed(arm,'LOF_vs_nonLOF'):<22}  n={arm.get('n_total', '—')}"
        )

    for h, label in [
        ("family_holdout", "Pfam family-split"),
        ("mmseqs_holdout", "MMseqs2-20 cluster-split"),
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
            metric = read_seed_point_estimate(
                d.get(f"delta_{k}_seed_aggregate", {})
            )
            if not metric.available:
                return "    N/A    "
            v = metric.value
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
    ap.add_argument("--no_ci", action="store_true", help="skip cluster-bootstrap CIs")
    ap.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed] if args.seed is not None else list(range(N_SEEDS))

    df = load_gene_table()

    fixed_arms = compute_fixed_arms(
        df, compute_ci=not args.no_ci, n_boot=args.n_boot
    )

    all_seed_results = []
    for s in seeds:
        sr = run_seed(df, s, args.n_folds, compute_ci=not args.no_ci, n_boot=args.n_boot)
        all_seed_results.append(sr)
        path = OUT_DIR / f"badonyi_survival_seed{s}.json"
        path.write_text(json.dumps(sr, indent=2))
        print(f"  Saved: {path}")

    summary = aggregate_seeds(all_seed_results, seeds, fixed_arms)
    # A single-seed run is a spot check, not the multi-seed deliverable, so it
    # writes its own file rather than replacing the full summary.
    spath = OUT_DIR / (
        f"badonyi_survival_summary_seed{args.seed}.json"
        if args.seed is not None
        else "badonyi_survival_summary.json"
    )
    spath.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary: {spath}")
    print_table(summary)


if __name__ == "__main__":
    main()
