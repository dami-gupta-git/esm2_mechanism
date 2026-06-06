"""
MMseqs2-20 cluster-holdout evaluation — Saadat & Fellay 2025 comparator.

Re-runs V1, V2, V_bad, V2+bad, V_all under MMseqs2 sequence-similarity cluster
holdout (20% identity, 20% coverage) — same clustering parameters as
Saadat & Fellay 2025 (iScience). Compares directly against the family-split
numbers in result_15.

The cluster_id map is computed by `scripts/fetch_uniprot_sequences.py` followed
by `mmseqs easy-cluster ... --min-seq-id 0.20 -c 0.20 --cov-mode 0` and stored
in `data/mmseqs_clusters.json` (key: `gene_to_cluster`).

Outputs:
    results/mmseqs_cluster_holdout/cluster_seed{0..4}.json
    results/mmseqs_cluster_holdout/cluster_summary.json

Usage:
    python3 scripts/mmseqs_cluster_holdout.py
    python3 scripts/mmseqs_cluster_holdout.py --seed 0
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder
from esm2_mech.utils.data import build_gene_to_row as _build_gene_to_row
from esm2_mech.utils.metrics import mean_std_n
from esm2_mech.utils.probes import run_mlp_cv, run_logreg_cv
from esm2_mech.utils.paths import (
    DATA_DIR,
    RESULTS_DIR,
    VALID_VARIANTS_JSON,
    EMB_WT_MEAN,
    EMB_MUT_MEAN,
    GENE_LIST_TSV,
)
import functools

print = functools.partial(print, flush=True)

OUT_DIR = RESULTS_DIR

warnings.filterwarnings("ignore")

MERGED_VALID_VARIANTS = VALID_VARIANTS_JSON
MERGED_WT_MEAN = EMB_WT_MEAN
MERGED_MUT_MEAN = EMB_MUT_MEAN

PROTEOME_FEATURES = DATA_DIR / "proteome_features_aligned.npy"
BADONYI_FEATURES = DATA_DIR / "badonyi_features_aligned.npy"
BADONYI_RAW_COLS = [0, 1, 2]
MERGED_GENE_LIST = GENE_LIST_TSV
MMSEQS_CLUSTERS = DATA_DIR / "mmseqs_clusters.json"

CLASSES = ["GOF", "DN", "LOF"]


def load_data():
    with open(MERGED_VALID_VARIANTS) as f:
        variants = json.load(f)
    labels = np.array([v["label_3class"] for v in variants])
    genes = np.array([v["gene"] for v in variants])
    n = len(variants)
    wt = np.load(MERGED_WT_MEAN)[:n]
    mut = np.load(MERGED_MUT_MEAN)[:n]
    delta = (mut - wt).astype(np.float32)
    print(f"Loaded {n} variants, {len(set(genes.tolist()))} genes")
    return labels, genes, delta


def load_clusters():
    with open(MMSEQS_CLUSTERS) as f:
        d = json.load(f)
    return d["gene_to_cluster"]


def build_gene_to_row():
    return _build_gene_to_row(MERGED_GENE_LIST)


def broadcast(genes, matrix, gene_to_row):
    """Align a per-gene feature matrix to the variant list.

    Returns (X, observed) where observed[i] is True only if gene[i] had a real
    feature row. Rows for genes with no feature data are left as NaN (not 0.0):
    0.0 is a plausible real feature value, so imputing it would silently
    contaminate the probe. Callers must restrict to `observed` before fitting.
    """
    n, d = len(genes), matrix.shape[1]
    X = np.full((n, d), np.nan, dtype=np.float32)
    observed = np.zeros(n, dtype=bool)
    for i, g in enumerate(genes):
        r = gene_to_row.get(g)
        if r is not None and r < matrix.shape[0]:
            X[i] = matrix[r]
            observed[i] = True
    return X, observed


def cluster_split_indices(groups, n_folds, seed):
    """Same as family_split_indices but with cluster_id as the grouping variable."""
    rng = np.random.RandomState(seed)
    unique = np.array(sorted(g for g in set(groups) if g is not None))
    rng.shuffle(unique)
    fold_of_group = {g: i % n_folds for i, g in enumerate(unique)}
    fold_of = np.array([fold_of_group[g] if g is not None else -1 for g in groups])
    for k in range(n_folds):
        test = np.where(fold_of == k)[0]
        train = np.where((fold_of != k) & (fold_of != -1))[0]
        yield train, test


def run_logreg(X, y, genes, groups, n_folds, seed, label):
    splits = list(cluster_split_indices(groups, n_folds, seed))
    return run_logreg_cv(X, y, splits, seed=seed, genes=genes, label=label)


def run_mlp(X, y, genes, groups, hidden, n_folds, seed, label):
    splits = list(cluster_split_indices(groups, n_folds, seed))
    return run_mlp_cv(X, y, splits, hidden=hidden, seed=seed, genes=genes, label=label)


def run_seed(
    seed,
    n_folds,
    labels,
    genes,
    delta,
    X_prot,
    X_bad,
    prot_observed,
    bad_observed,
    gene_to_cluster,
):
    print(f"\n{'='*72}\nSEED {seed}\n{'='*72}")

    # Use fixed mapping matching CLASSES order: GOF=0, DN=1, LOF=2
    cls_to_idx = {c: i for i, c in enumerate(CLASSES)}
    y = np.array([cls_to_idx[lbl] for lbl in labels])

    cluster_of = np.array([gene_to_cluster.get(g) for g in genes])
    has_cluster = np.array([c is not None for c in cluster_of])
    idx = np.where(has_cluster)[0]
    y_f = y[idx]
    genes_f = genes[idx]
    groups = cluster_of[idx]
    X_delta_f = delta[idx]
    X_prot_f = X_prot[idx]
    X_bad_f = X_bad[idx]
    # Per-arm observed masks (subset of the cluster-filtered set). Proteome/Badonyi
    # genes with no feature row are NaN here; restrict each feature arm to its own
    # observed subset rather than imputing 0.0 (CLAUDE.md: no fillna on probe
    # features; recompute splits on the observed subset).
    prot_obs_f = prot_observed[idx]
    bad_obs_f = bad_observed[idx]

    n_clusters = len(set(groups.tolist()))
    cd = {CLASSES[k]: int(v) for k, v in Counter(y_f.tolist()).items()}
    print(f"  Variants with cluster: {len(idx)}/{len(y)} ({n_clusters} clusters)")
    print(f"  Class dist: {cd}")
    print(
        f"  Feature-observed (within cluster set): "
        f"proteome {int(prot_obs_f.sum())}/{len(idx)}, "
        f"Badonyi {int(bad_obs_f.sum())}/{len(idx)}"
    )

    res = {
        "seed": seed,
        "n_variants": int(len(idx)),
        "n_clusters": n_clusters,
        "class_dist": cd,
    }

    def report(key):
        res_key = res[key]
        print(
            f"  {key} F1={res_key['macro_f1_mean']:.4f}±{res_key['macro_f1_std']:.4f}  "
            f"pgF1={res_key.get('per_gene_f1_mean', float('nan')):.4f}"
        )

    # Each feature arm is restricted to the variants whose features are all
    # observed (no NaN-imputed rows), and cluster splits are recomputed on that
    # subset inside run_logreg/run_mlp. n_used records how many variants the arm
    # actually saw, so a silently shrunk arm is visible in the output.
    def subset(*masks):
        keep = np.ones(len(idx), dtype=bool)
        for mask in masks:
            keep &= mask
        return keep

    print(f"\n--- V1: ESM-2 delta MLP ---")
    # delta (ESM-2) is always observed — no feature-row dropout.
    res["V1"] = run_mlp(X_delta_f, y_f, genes_f, groups, (256, 64), n_folds, seed, "V1")
    res["V1"]["n_used"] = int(len(idx))
    report("V1")

    print(f"\n--- V2: Proteome LogReg ---")
    keep = subset(prot_obs_f)
    res["V2"] = run_logreg(
        X_prot_f[keep], y_f[keep], genes_f[keep], groups[keep], n_folds, seed, "V2"
    )
    res["V2"]["n_used"] = int(keep.sum())
    report("V2")

    print(f"\n--- V_bad: Badonyi LogReg ---")
    keep = subset(bad_obs_f)
    res["V_bad"] = run_logreg(
        X_bad_f[keep], y_f[keep], genes_f[keep], groups[keep], n_folds, seed, "V_bad"
    )
    res["V_bad"]["n_used"] = int(keep.sum())
    report("V_bad")

    print(f"\n--- V2+bad: Proteome+Badonyi LogReg ---")
    keep = subset(prot_obs_f, bad_obs_f)
    X_v2bad = np.concatenate([X_prot_f[keep], X_bad_f[keep]], axis=1)
    res["V2_bad"] = run_logreg(
        X_v2bad, y_f[keep], genes_f[keep], groups[keep], n_folds, seed, "V2+bad"
    )
    res["V2_bad"]["n_used"] = int(keep.sum())
    report("V2_bad")

    print(f"\n--- V_all: ESM-2+proteome+Badonyi MLP ---")
    keep = subset(prot_obs_f, bad_obs_f)
    X_vall = np.concatenate([X_delta_f[keep], X_prot_f[keep], X_bad_f[keep]], axis=1)
    res["V_all"] = run_mlp(
        X_vall, y_f[keep], genes_f[keep], groups[keep], (256, 64), n_folds, seed, "V_all"
    )
    res["V_all"]["n_used"] = int(keep.sum())
    report("V_all")

    return res


def aggregate_seeds(all_res):
    out = {"n_seeds": len(all_res)}
    # mean_std_n filters both None and NaN per-seed values, so a seed whose
    # metric was unscorable (None) or NaN never poisons the across-seed mean/std.
    for key in ["V1", "V2", "V_bad", "V2_bad", "V_all"]:
        for metric in ["macro_f1_mean", "per_gene_f1_mean"]:
            vals = [r[key].get(metric) for r in all_res if key in r]
            stem = f"{key}_{metric.replace('_mean','')}"
            mean, std, n = mean_std_n(vals)
            if n:
                out[f"{stem}_mean"] = mean
                out[f"{stem}_std"] = std
        for cls in CLASSES:
            vals = [r[key].get(f"auroc_{cls}_mean") for r in all_res if key in r]
            mean, std, n = mean_std_n(vals)
            if n:
                out[f"{key}_auroc_{cls}_mean"] = mean
                out[f"{key}_auroc_{cls}_std"] = std
    return out


def print_table(summary):
    print("\n" + "=" * 96)
    print("MMseqs2-20 cluster-holdout — 5-seed mean ± std")
    print("=" * 96)
    print(
        f"{'Variant':<10} {'F1(variant)':<16} {'F1(gene)':<16} {'GOF AUROC':<14} {'DN AUROC':<14} {'LOF AUROC':<14}"
    )
    print("-" * 96)

    def fmt(m, s):
        if m is None:
            return "    N/A      "
        return f"{m:.3f}±{s:.3f}"

    for key, label in [
        ("V1", "V1(seq)"),
        ("V2", "V2(prot)"),
        ("V_bad", "V_bad"),
        ("V2_bad", "V2+bad"),
        ("V_all", "V_all"),
    ]:
        f1 = fmt(
            summary.get(f"{key}_macro_f1_mean"), summary.get(f"{key}_macro_f1_std")
        )
        pg = fmt(
            summary.get(f"{key}_per_gene_f1_mean"),
            summary.get(f"{key}_per_gene_f1_std"),
        )
        gof = fmt(
            summary.get(f"{key}_auroc_GOF_mean"), summary.get(f"{key}_auroc_GOF_std")
        )
        dn = fmt(
            summary.get(f"{key}_auroc_DN_mean"), summary.get(f"{key}_auroc_DN_std")
        )
        lof = fmt(
            summary.get(f"{key}_auroc_LOF_mean"), summary.get(f"{key}_auroc_LOF_std")
        )
        print(f"{label:<10} {f1:<16} {pg:<16} {gof:<14} {dn:<14} {lof:<14}")
    print("=" * 96)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    seeds = [args.seed] if args.seed is not None else list(range(5))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading data ===")
    labels, genes, delta = load_data()

    print("\n=== Loading MMseqs2 cluster mapping ===")
    gene_to_cluster = load_clusters()
    n_with = sum(1 for g in genes if gene_to_cluster.get(g))
    print(f"  Variants with cluster mapping: {n_with}/{len(genes)}")

    print("\n=== Broadcasting feature matrices ===")
    gene_to_row = build_gene_to_row()
    prot = np.load(PROTEOME_FEATURES).astype(np.float32)
    bad_full = np.load(BADONYI_FEATURES).astype(np.float32)
    X_prot, prot_observed = broadcast(genes, prot, gene_to_row)
    X_bad, bad_observed = broadcast(genes, bad_full[:, BADONYI_RAW_COLS], gene_to_row)
    print(f"  X_prot {X_prot.shape}  X_bad {X_bad.shape}")
    print(
        f"  Feature rows observed: proteome {int(prot_observed.sum())}/{len(genes)}, "
        f"Badonyi {int(bad_observed.sum())}/{len(genes)}"
    )

    all_res = []
    for s in seeds:
        res = run_seed(
            s,
            args.n_folds,
            labels,
            genes,
            delta,
            X_prot,
            X_bad,
            prot_observed,
            bad_observed,
            gene_to_cluster,
        )
        all_res.append(res)
        path = OUT_DIR / f"cluster_seed{s}.json"
        path.write_text(json.dumps(res, indent=2))
        print(f"  Saved: {path}")

    summary = aggregate_seeds(all_res)
    spath = OUT_DIR / "cluster_summary.json"
    spath.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary: {spath}")
    print_table(summary)


if __name__ == "__main__":
    main()
