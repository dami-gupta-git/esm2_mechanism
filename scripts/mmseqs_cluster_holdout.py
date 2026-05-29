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
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from utils_probes import compute_metrics, per_gene_f1
import functools
print = functools.partial(print, flush=True)

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
EMB_DIR = DATA_DIR / "embeddings"

MERGED_VALID_VARIANTS = EMB_DIR / "merged_valid_variants.json"
MERGED_WT_MEAN = EMB_DIR / "merged_embeddings_wt_mean.npy"
MERGED_MUT_MEAN = EMB_DIR / "merged_embeddings_mut_mean.npy"

PROTEOME_FEATURES = DATA_DIR / "proteome_features_aligned.npy"
BADONYI_FEATURES = DATA_DIR / "badonyi_features_aligned.npy"
BADONYI_RAW_COLS = [0, 1, 2]
MERGED_GENE_LIST = DATA_DIR / "merged_gene_list.tsv"
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


def _align_proba(proba, clf_classes):
    out = np.zeros((len(proba), len(CLASSES)), dtype=np.float32)
    for ci, c in enumerate(list(clf_classes)):
        if c < len(CLASSES):
            out[:, c] = proba[:, ci]
    return out



def aggregate(fold_list):
    if not fold_list:
        return {"error": "no folds"}
    out = {"n_folds": len(fold_list)}
    vals = [f["macro_f1"] for f in fold_list]
    out["macro_f1_mean"] = float(np.mean(vals))
    out["macro_f1_std"] = float(np.std(vals))
    for cls in CLASSES:
        v = [f["per_class_auroc"][cls] for f in fold_list
             if f["per_class_auroc"].get(cls) is not None]
        out[f"auroc_{cls}_mean"] = float(np.mean(v)) if v else None
        out[f"auroc_{cls}_std"] = float(np.std(v)) if v else None
    return out


def run_logreg(X, y, genes, groups, n_folds, seed, label):
    fold_results, pgs = [], []
    for fi, (tr, te) in enumerate(cluster_split_indices(groups, n_folds, seed)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        if len(set(y_tr.tolist())) < len(CLASSES):
            continue
        if len(set(y_te.tolist())) < 2:
            continue
        sc = StandardScaler().fit(X_tr)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                 random_state=seed)
        clf.fit(sc.transform(X_tr), y_tr)
        proba = _align_proba(clf.predict_proba(sc.transform(X_te)), clf.classes_)
        pred = proba.argmax(axis=1)
        fm = compute_metrics(y_te, pred, proba)
        fold_results.append(fm)
        pgs.append(per_gene_f1(y_te, proba, genes[te]))
        print(f"      [{label}] Fold {fi+1}: n_te={len(te)} F1={fm['macro_f1']:.3f}"
              f" pgF1={pgs[-1]:.3f} " + " ".join(
                  f"{c}={fm['per_class_auroc'].get(c, float('nan')):.3f}"
                  if fm['per_class_auroc'].get(c) is not None else f"{c}=NA"
                  for c in CLASSES))
    agg = aggregate(fold_results)
    if pgs:
        agg["per_gene_f1_mean"] = float(np.mean(pgs))
        agg["per_gene_f1_std"] = float(np.std(pgs))
    return agg


def run_mlp(X, y, genes, groups, hidden, n_folds, seed, label):
    fold_results, pgs = [], []
    for fi, (tr, te) in enumerate(cluster_split_indices(groups, n_folds, seed)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        if len(set(y_tr.tolist())) < len(CLASSES):
            continue
        if len(set(y_te.tolist())) < 2:
            continue
        sc = StandardScaler().fit(X_tr)
        X_tr_s = sc.transform(X_tr)
        X_te_s = sc.transform(X_te)
        # Oversample to balance
        counts = np.bincount(y_tr, minlength=len(CLASSES))
        max_c = counts.max()
        rng = np.random.RandomState(seed)
        os_idx = []
        for c in range(len(CLASSES)):
            ci = np.where(y_tr == c)[0]
            if len(ci) == 0:
                continue
            os_idx.append(np.tile(ci, max_c // len(ci)))
            rem = max_c % len(ci)
            if rem:
                os_idx.append(rng.choice(ci, rem, replace=False))
        os_idx = np.concatenate(os_idx)
        rng.shuffle(os_idx)
        clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=500,
                            early_stopping=True, validation_fraction=0.15,
                            random_state=seed)
        clf.fit(X_tr_s[os_idx], y_tr[os_idx])
        proba = _align_proba(clf.predict_proba(X_te_s), clf.classes_)
        pred = proba.argmax(axis=1)
        fm = compute_metrics(y_te, pred, proba)
        fold_results.append(fm)
        pgs.append(per_gene_f1(y_te, proba, genes[te]))
        print(f"      [{label}] Fold {fi+1}: n_te={len(te)} F1={fm['macro_f1']:.3f}"
              f" pgF1={pgs[-1]:.3f} " + " ".join(
                  f"{c}={fm['per_class_auroc'].get(c, float('nan')):.3f}"
                  if fm['per_class_auroc'].get(c) is not None else f"{c}=NA"
                  for c in CLASSES))
    agg = aggregate(fold_results)
    if pgs:
        agg["per_gene_f1_mean"] = float(np.mean(pgs))
        agg["per_gene_f1_std"] = float(np.std(pgs))
    return agg


def run_seed(seed, n_folds, labels, genes, delta, X_prot, X_bad, gene_to_cluster):
    print(f"\n{'='*72}\nSEED {seed}\n{'='*72}")

    # Use fixed mapping matching CLASSES order: GOF=0, DN=1, LOF=2
    cls_to_idx = {c: i for i, c in enumerate(CLASSES)}
    y = np.array([cls_to_idx[lbl] for lbl in labels])

    cluster_of = np.array([gene_to_cluster.get(g) for g in genes])
    has_cluster = cluster_of != None  # noqa
    idx = np.where(has_cluster)[0]
    y_f = y[idx]
    genes_f = genes[idx]
    groups = cluster_of[idx]
    X_delta_f = delta[idx]
    X_prot_f = X_prot[idx]
    X_bad_f = X_bad[idx]
    X_v2bad = np.concatenate([X_prot_f, X_bad_f], axis=1)
    X_v1bad = np.concatenate([X_delta_f, X_bad_f], axis=1)
    X_vall = np.concatenate([X_delta_f, X_prot_f, X_bad_f], axis=1)

    n_clusters = len(set(groups.tolist()))
    cd = {CLASSES[k]: int(v) for k, v in Counter(y_f.tolist()).items()}
    print(f"  Variants with cluster: {len(idx)}/{len(y)} ({n_clusters} clusters)")
    print(f"  Class dist: {cd}")

    res = {"seed": seed, "n_variants": int(len(idx)),
           "n_clusters": n_clusters, "class_dist": cd}

    print(f"\n--- V1: ESM-2 delta MLP ---")
    res["V1"] = run_mlp(X_delta_f, y_f, genes_f, groups, (256, 64), n_folds, seed, "V1")
    print(f"  V1 F1={res['V1']['macro_f1_mean']:.4f}±{res['V1']['macro_f1_std']:.4f}  "
          f"pgF1={res['V1'].get('per_gene_f1_mean', float('nan')):.4f}")

    print(f"\n--- V2: Proteome LogReg ---")
    res["V2"] = run_logreg(X_prot_f, y_f, genes_f, groups, n_folds, seed, "V2")
    print(f"  V2 F1={res['V2']['macro_f1_mean']:.4f}±{res['V2']['macro_f1_std']:.4f}  "
          f"pgF1={res['V2'].get('per_gene_f1_mean', float('nan')):.4f}")

    print(f"\n--- V_bad: Badonyi LogReg ---")
    res["V_bad"] = run_logreg(X_bad_f, y_f, genes_f, groups, n_folds, seed, "V_bad")
    print(f"  V_bad F1={res['V_bad']['macro_f1_mean']:.4f}±{res['V_bad']['macro_f1_std']:.4f}  "
          f"pgF1={res['V_bad'].get('per_gene_f1_mean', float('nan')):.4f}")

    print(f"\n--- V2+bad: Proteome+Badonyi LogReg ---")
    res["V2_bad"] = run_logreg(X_v2bad, y_f, genes_f, groups, n_folds, seed, "V2+bad")
    print(f"  V2+bad F1={res['V2_bad']['macro_f1_mean']:.4f}±{res['V2_bad']['macro_f1_std']:.4f}  "
          f"pgF1={res['V2_bad'].get('per_gene_f1_mean', float('nan')):.4f}")

    print(f"\n--- V_all: ESM-2+proteome+Badonyi MLP ---")
    res["V_all"] = run_mlp(X_vall, y_f, genes_f, groups, (256, 64), n_folds, seed, "V_all")
    print(f"  V_all F1={res['V_all']['macro_f1_mean']:.4f}±{res['V_all']['macro_f1_std']:.4f}  "
          f"pgF1={res['V_all'].get('per_gene_f1_mean', float('nan')):.4f}")

    return res


def aggregate_seeds(all_res):
    out = {"n_seeds": len(all_res)}
    for key in ["V1", "V2", "V_bad", "V2_bad", "V_all"]:
        for metric in ["macro_f1_mean", "per_gene_f1_mean"]:
            vals = [r[key].get(metric) for r in all_res
                    if key in r and r[key].get(metric) is not None]
            stem = f"{key}_{metric.replace('_mean','')}"
            if vals:
                out[f"{stem}_mean"] = float(np.mean(vals))
                out[f"{stem}_std"] = float(np.std(vals))
        for cls in CLASSES:
            vals = [r[key].get(f"auroc_{cls}_mean") for r in all_res
                    if key in r and r[key].get(f"auroc_{cls}_mean") is not None]
            if vals:
                out[f"{key}_auroc_{cls}_mean"] = float(np.mean(vals))
                out[f"{key}_auroc_{cls}_std"] = float(np.std(vals))
    return out


def print_table(summary):
    print("\n" + "=" * 96)
    print("MMseqs2-20 cluster-holdout — 5-seed mean ± std")
    print("=" * 96)
    print(f"{'Variant':<10} {'F1(variant)':<16} {'F1(gene)':<16} {'GOF AUROC':<14} {'DN AUROC':<14} {'LOF AUROC':<14}")
    print("-" * 96)

    def fmt(m, s):
        if m is None:
            return "    N/A      "
        return f"{m:.3f}±{s:.3f}"

    for key, label in [("V1","V1(seq)"), ("V2","V2(prot)"),
                       ("V_bad","V_bad"), ("V2_bad","V2+bad"),
                       ("V_all","V_all")]:
        f1 = fmt(summary.get(f"{key}_macro_f1_mean"),
                 summary.get(f"{key}_macro_f1_std"))
        pg = fmt(summary.get(f"{key}_per_gene_f1_mean"),
                 summary.get(f"{key}_per_gene_f1_std"))
        gof = fmt(summary.get(f"{key}_auroc_GOF_mean"),
                  summary.get(f"{key}_auroc_GOF_std"))
        dn = fmt(summary.get(f"{key}_auroc_DN_mean"),
                 summary.get(f"{key}_auroc_DN_std"))
        lof = fmt(summary.get(f"{key}_auroc_LOF_mean"),
                  summary.get(f"{key}_auroc_LOF_std"))
        print(f"{label:<10} {f1:<16} {pg:<16} {gof:<14} {dn:<14} {lof:<14}")
    print("=" * 96)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    seeds = [args.seed] if args.seed is not None else list(range(5))
    out_dir = Path(args.out_dir) if args.out_dir \
        else PROJECT_DIR / "results" / "mmseqs_cluster_holdout"
    out_dir.mkdir(parents=True, exist_ok=True)

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
    X_prot = broadcast(genes, prot, gene_to_row)
    X_bad = broadcast(genes, bad_full[:, BADONYI_RAW_COLS], gene_to_row)
    print(f"  X_prot {X_prot.shape}  X_bad {X_bad.shape}")

    all_res = []
    for s in seeds:
        res = run_seed(s, args.n_folds, labels, genes, delta, X_prot, X_bad, gene_to_cluster)
        all_res.append(res)
        path = out_dir / f"cluster_seed{s}.json"
        path.write_text(json.dumps(res, indent=2))
        print(f"  Saved: {path}")

    summary = aggregate_seeds(all_res)
    spath = out_dir / "cluster_summary.json"
    spath.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary: {spath}")
    print_table(summary)


if __name__ == "__main__":
    main()
