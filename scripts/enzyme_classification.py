"""
Enzyme type classification from ESM-2 WT mean-pooled embeddings.

Classifies genes into 4 classes (kinase / protease / oxidoreductase / non-enzyme)
using frozen ESM-2 WT embeddings — no mutation delta needed.

This is a positive control experiment paralleling the mechanism arc (results 1–10):
  - Gene-split vs family-split CV quantifies family-recognition leakage
  - Comparison to mechanism floor (F1 ~0.38) shows whether mechanism null is task-specific

Probes run:
  1. LogReg gene-split (5-fold, 5-seed)
  2. LogReg family-split (5-fold, 5-seed)   ← primary metric
  3. MLP family-split (5-fold, 5-seed)       ← tests whether nonlinearity helps
  4. Proteome-features family-split          ← does gene biology predict enzyme class?
  5. Majority baseline

Usage:
    python enzyme_classification.py [--data_dir ../data] [--emb_dir ../data/embeddings]
                                    [--out_dir ../results/enzyme_classification]
                                    [--seeds 0 1 2 3 4]

Output: results/enzyme_classification/enzyme_classification_summary.json
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import warnings
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

print = functools.partial(print, flush=True)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent

ENZYME_CLASSES = ["kinase", "protease", "oxidoreductase", "non-enzyme"]

# Reference numbers for comparison (from docs/README.md)
MECHANISM_FAMILY_SPLIT_F1 = 0.385   # merged dataset, 5-seed (result 7)
MECHANISM_GENE_SPLIT_F1 = 0.415     # merged dataset, MLP, seed 0 (result 7)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gene_embeddings(data_dir: Path, emb_dir: Path) -> tuple:
    """
    Load per-gene WT embeddings by taking the first variant's embedding for each gene.
    Returns (X, gene_list, uniprot_list) where X has shape (n_genes, 1280).
    """
    variants_path = data_dir / "merged_valid_variants.json"
    with open(variants_path) as f:
        variants = json.load(f)

    wt_mean = np.load(emb_dir / "merged_embeddings_wt_mean.npy")
    print(f"Loaded {len(variants)} variants, wt_mean shape: {wt_mean.shape}")

    # Take the first variant index per gene to get one embedding per gene
    gene_first_idx = {}
    gene_uniprot = {}
    for i, v in enumerate(variants):
        g = v["gene"]
        if g not in gene_first_idx:
            gene_first_idx[g] = i
            gene_uniprot[g] = v.get("uniprot_id", "")

    gene_list = list(gene_first_idx.keys())
    idxs = [gene_first_idx[g] for g in gene_list]
    X = wt_mean[idxs].astype(np.float32)

    print(f"Per-gene embeddings: {X.shape} ({len(gene_list)} genes)")
    return X, gene_list, [gene_uniprot[g] for g in gene_list]


def load_enzyme_labels(data_dir: Path) -> dict:
    """Load gene -> enzyme_4class from enzyme_labels.tsv."""
    import csv
    labels: dict[str, str] = {}
    label_path = data_dir / "enzyme_labels.tsv"
    with open(label_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            labels[row["gene"]] = row["enzyme_4class"]
    print(f"Loaded enzyme labels for {len(labels)} genes")
    return labels


def load_pfam(data_dir: Path) -> dict:
    with open(data_dir / "pfam_families.json") as f:
        return json.load(f)


def load_proteome_features(data_dir: Path) -> tuple:
    """Load proteome feature matrix and aligned gene list."""
    X = np.load(data_dir / "proteome_features_aligned.npy").astype(np.float32)
    with open(data_dir / "proteome_feature_columns.json") as f:
        cols = json.load(f)
    # Load gene order from merged_gene_list.tsv
    import csv
    genes = []
    with open(data_dir / "merged_gene_list.tsv") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            genes.append(row["gene"])
    print(f"Proteome features: {X.shape}, {len(genes)} genes")
    return X, genes


# ---------------------------------------------------------------------------
# CV splits (mirroring utils_probes.py but operating on gene-level arrays)
# ---------------------------------------------------------------------------

def gene_split_cv(genes: list, n_folds: int = 5, seed: int = 42) -> list:
    u = np.array(sorted(set(genes)))
    np.random.RandomState(seed).shuffle(u)
    g = np.array(genes)
    splits = []
    for fold in np.array_split(u, n_folds):
        tr = np.where(~np.isin(g, fold))[0]
        te = np.where(np.isin(g, fold))[0]
        if len(tr) >= 10 and len(te) >= 5:
            splits.append((tr, te))
    return splits


def family_split_cv(genes: list, pfam_map: dict,
                    n_folds: int = 5, seed: int = 42) -> list:
    g = np.array(genes)
    g2p = {gene: pfam_map[gene] for gene in set(genes) if pfam_map.get(gene)}
    fams = np.array(sorted(set(g2p.values())))
    np.random.RandomState(seed).shuffle(fams)
    splits = []
    for fold_fams in np.array_split(fams, n_folds):
        fs = set(fold_fams)
        te = np.array([g[i] in g2p and g2p[g[i]] in fs for i in range(len(g))])
        tr = np.array([g[i] in g2p and g2p[g[i]] not in fs for i in range(len(g))])
        if tr.sum() >= 10 and te.sum() >= 5:
            splits.append((np.where(tr)[0], np.where(te)[0]))
    return splits


# ---------------------------------------------------------------------------
# Probe runners
# ---------------------------------------------------------------------------

def run_logreg(X: np.ndarray, y: np.ndarray, splits: list[tuple],
               classes: list[str], seed: int = 42) -> dict:
    from sklearn.linear_model import LogisticRegression
    n_cls = len(classes)
    fold_f1s, fold_aurocs = [], {c: [] for c in classes}

    for tr, te in splits:
        if len(set(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                 random_state=seed)
        clf.fit(sc.transform(X[tr]), y[tr])
        proba_raw = clf.predict_proba(sc.transform(X[te]))

        # Align proba columns to canonical class indices
        proba = np.zeros((len(te), n_cls), dtype=np.float32)
        for ci, c in enumerate(clf.classes_):
            if 0 <= c < n_cls:
                proba[:, c] = proba_raw[:, ci]

        pred = proba.argmax(axis=1)
        fold_f1s.append(float(f1_score(y[te], pred, average="macro", zero_division=0)))

        for i, cls in enumerate(classes):
            y_bin = (y[te] == i).astype(int)
            if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
                fold_aurocs[cls].append(float(roc_auc_score(y_bin, proba[:, i])))

    return {
        "macro_f1_mean": float(np.mean(fold_f1s)) if fold_f1s else None,
        "macro_f1_std": float(np.std(fold_f1s)) if fold_f1s else None,
        "per_class_auroc_mean": {c: float(np.mean(v)) if v else None
                                 for c, v in fold_aurocs.items()},
        "per_class_auroc_std": {c: float(np.std(v)) if v else None
                                for c, v in fold_aurocs.items()},
        "n_folds": len(fold_f1s),
    }


def run_mlp(X: np.ndarray, y: np.ndarray, splits: list[tuple],
            classes: list[str], seed: int = 42) -> dict:
    n_cls = len(classes)
    fold_f1s, fold_aurocs = [], {c: [] for c in classes}

    for tr, te in splits:
        if len(set(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 64),
            activation="relu",
            alpha=1e-3,
            max_iter=300,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
        )
        clf.fit(sc.transform(X[tr]), y[tr])
        proba_raw = clf.predict_proba(sc.transform(X[te]))

        proba = np.zeros((len(te), n_cls), dtype=np.float32)
        for ci, c in enumerate(clf.classes_):
            if 0 <= c < n_cls:
                proba[:, c] = proba_raw[:, ci]

        pred = proba.argmax(axis=1)
        fold_f1s.append(float(f1_score(y[te], pred, average="macro", zero_division=0)))

        for i, cls in enumerate(classes):
            y_bin = (y[te] == i).astype(int)
            if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
                fold_aurocs[cls].append(float(roc_auc_score(y_bin, proba[:, i])))

    return {
        "macro_f1_mean": float(np.mean(fold_f1s)) if fold_f1s else None,
        "macro_f1_std": float(np.std(fold_f1s)) if fold_f1s else None,
        "per_class_auroc_mean": {c: float(np.mean(v)) if v else None
                                 for c, v in fold_aurocs.items()},
        "per_class_auroc_std": {c: float(np.std(v)) if v else None
                                for c, v in fold_aurocs.items()},
        "n_folds": len(fold_f1s),
    }


def majority_baseline_f1(y: np.ndarray) -> float:
    majority = Counter(y.tolist()).most_common(1)[0][0]
    pred = np.full(len(y), majority)
    return float(f1_score(y, pred, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------

def run_multiseed(X: np.ndarray, y: np.ndarray, genes: list[str],
                  pfam_map: dict, le: LabelEncoder,
                  seeds: list[int], n_folds: int = 5) -> dict:
    classes = list(le.classes_)
    print(f"\nClasses: {classes}")
    print(f"Class distribution: {dict(Counter(y.tolist()))}")

    gs_f1s, fs_f1s, mlp_f1s = [], [], []
    gs_aurocs = {c: [] for c in classes}
    fs_aurocs = {c: [] for c in classes}
    mlp_aurocs = {c: [] for c in classes}

    for seed in seeds:
        print(f"\n  Seed {seed}:")

        # Gene-split
        gs_splits = gene_split_cv(genes, n_folds=n_folds, seed=seed)
        gs = run_logreg(X, y, gs_splits, classes, seed=seed)
        gs_f1s.append(gs["macro_f1_mean"])
        for c in classes:
            v = gs["per_class_auroc_mean"].get(c)
            if v is not None:
                gs_aurocs[c].append(v)
        print(f"    LogReg gene-split  F1={gs['macro_f1_mean']:.3f}")

        # Family-split LogReg
        fs_splits = family_split_cv(genes, pfam_map, n_folds=n_folds, seed=seed)
        fs = run_logreg(X, y, fs_splits, classes, seed=seed)
        fs_f1s.append(fs["macro_f1_mean"])
        for c in classes:
            v = fs["per_class_auroc_mean"].get(c)
            if v is not None:
                fs_aurocs[c].append(v)
        print(f"    LogReg family-split F1={fs['macro_f1_mean']:.3f}  "
              f"AUROC: " + " ".join(f"{c}={fs['per_class_auroc_mean'].get(c, float('nan')):.3f}"
                                    for c in classes))

        # Family-split MLP
        mlp = run_mlp(X, y, fs_splits, classes, seed=seed)
        mlp_f1s.append(mlp["macro_f1_mean"])
        for c in classes:
            v = mlp["per_class_auroc_mean"].get(c)
            if v is not None:
                mlp_aurocs[c].append(v)
        print(f"    MLP    family-split F1={mlp['macro_f1_mean']:.3f}")

    maj_f1 = majority_baseline_f1(y)

    def _agg(vals):
        vals = [v for v in vals if v is not None]
        return (float(np.mean(vals)) if vals else None,
                float(np.std(vals)) if vals else None)

    gs_mean, gs_std = _agg(gs_f1s)
    fs_mean, fs_std = _agg(fs_f1s)
    mlp_mean, mlp_std = _agg(mlp_f1s)

    leakage_pct = None
    if gs_mean and fs_mean and gs_mean > 0:
        leakage_pct = round(100.0 * (gs_mean - fs_mean) / gs_mean, 1)

    print(f"\n  Results ({len(seeds)} seeds):")
    print(f"    Majority baseline:       F1={maj_f1:.3f}")
    print(f"    LogReg gene-split:       F1={gs_mean:.3f} ± {gs_std:.3f}")
    print(f"    LogReg family-split:     F1={fs_mean:.3f} ± {fs_std:.3f}  ← primary metric")
    print(f"    MLP    family-split:     F1={mlp_mean:.3f} ± {mlp_std:.3f}")
    if leakage_pct is not None:
        print(f"    Leakage fraction:        {leakage_pct:.1f}%")
    print(f"\n  vs mechanism family-split floor ({MECHANISM_FAMILY_SPLIT_F1:.3f}):")
    if fs_mean is not None:
        delta = fs_mean - MECHANISM_FAMILY_SPLIT_F1
        sym = ">>>" if delta > 0.15 else (">>" if delta > 0.05 else (">" if delta > 0 else "~"))
        print(f"    {sym} enzyme {fs_mean:.3f}  vs  mechanism {MECHANISM_FAMILY_SPLIT_F1:.3f}  Δ={delta:+.3f}")

    return {
        "majority_f1": maj_f1,
        "logreg_gene_split": {
            "macro_f1_mean": gs_mean, "macro_f1_std": gs_std,
            "per_class_auroc_mean": {c: float(np.mean(v)) if v else None for c, v in gs_aurocs.items()},
            "n_seeds": len(seeds),
        },
        "logreg_family_split": {
            "macro_f1_mean": fs_mean, "macro_f1_std": fs_std,
            "per_class_auroc_mean": {c: float(np.mean(v)) if v else None for c, v in fs_aurocs.items()},
            "n_seeds": len(seeds),
        },
        "mlp_family_split": {
            "macro_f1_mean": mlp_mean, "macro_f1_std": mlp_std,
            "per_class_auroc_mean": {c: float(np.mean(v)) if v else None for c, v in mlp_aurocs.items()},
            "n_seeds": len(seeds),
        },
        "leakage_pct": leakage_pct,
        "mechanism_reference_f1": MECHANISM_FAMILY_SPLIT_F1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(ROOT / "data"))
    parser.add_argument("--emb_dir",  default=str(ROOT / "data" / "embeddings"))
    parser.add_argument("--out_dir",  default=str(ROOT / "results" / "enzyme_classification"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    emb_dir  = Path(args.emb_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Enzyme Classification from ESM-2 WT Embeddings ===")
    print(f"Seeds: {args.seeds}  Folds: {args.n_folds}")

    # Load data
    X_emb, gene_list, _ = load_gene_embeddings(data_dir, emb_dir)
    enzyme_labels = load_enzyme_labels(data_dir)
    pfam_map = load_pfam(data_dir)

    # Align labels to gene_list order
    missing = [g for g in gene_list if g not in enzyme_labels]
    if missing:
        print(f"  WARNING: {len(missing)} genes have no enzyme label and will be assigned 'non-enzyme': "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    y_str = [enzyme_labels.get(g, "non-enzyme") for g in gene_list]
    le = LabelEncoder()
    le.fit(ENZYME_CLASSES)
    y = le.transform(y_str)

    print(f"\nGenes in embedding: {len(gene_list)}")
    print(f"Label coverage: {sum(1 for g in gene_list if g in enzyme_labels)}/{len(gene_list)}")
    print(f"Class distribution: {dict(Counter(y_str))}")
    print(f"Pfam-annotated genes: {sum(1 for g in gene_list if pfam_map.get(g))}/{len(gene_list)}")

    # -----------------------------------------------------------------------
    # ESM-2 WT embedding probes
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("PART 1: ESM-2 WT embedding (1280-dim)")
    print("="*60)
    emb_results = run_multiseed(
        X_emb, y, gene_list, pfam_map, le,
        seeds=args.seeds, n_folds=args.n_folds
    )

    # -----------------------------------------------------------------------
    # Proteome feature baseline
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("PART 2: Proteome features (37-dim) — baseline comparison")
    print("="*60)

    proteome_results = None
    try:
        X_prot, prot_genes = load_proteome_features(data_dir)
        prot_gene_to_idx = {g: i for i, g in enumerate(prot_genes)}

        # Align proteome features to gene_list
        prot_aligned_idxs = [prot_gene_to_idx[g] for g in gene_list if g in prot_gene_to_idx]
        prot_aligned_genes = [g for g in gene_list if g in prot_gene_to_idx]
        prot_aligned_y = y[[gene_list.index(g) for g in prot_aligned_genes]]
        X_prot_aligned = X_prot[prot_aligned_idxs]

        # NaN check: proteome_features_aligned.npy is pre-imputed in build_proteome_features.py
        # so NaNs should not occur. Log if any are present; do not impute here
        # (imputation must be done per-fold inside CV to avoid test-set leakage).
        if np.isnan(X_prot_aligned).any():
            n_nan = int(np.isnan(X_prot_aligned).sum())
            print(f"  WARNING: {n_nan} NaN values found in proteome features — "
                  f"these will cause errors in StandardScaler. "
                  f"Re-run build_proteome_features.py to regenerate pre-imputed features.")

        print(f"Proteome-aligned genes: {len(prot_aligned_genes)}")
        proteome_results = run_multiseed(
            X_prot_aligned, prot_aligned_y, prot_aligned_genes, pfam_map, le,
            seeds=args.seeds, n_folds=args.n_folds
        )
    except Exception as e:
        print(f"Proteome baseline failed: {e}")

    # -----------------------------------------------------------------------
    # Decision rule evaluation (pre-registered)
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("PRE-REGISTERED DECISION RULES")
    print("="*60)
    fs_f1 = emb_results["logreg_family_split"]["macro_f1_mean"]
    mlp_f1 = emb_results["mlp_family_split"]["macro_f1_mean"]
    gs_f1 = emb_results["logreg_gene_split"]["macro_f1_mean"]

    h1 = fs_f1 is not None and fs_f1 >= 0.70
    h2 = fs_f1 is not None and (fs_f1 - MECHANISM_FAMILY_SPLIT_F1) > 0.10
    h4 = (mlp_f1 is not None and fs_f1 is not None and
          abs(mlp_f1 - fs_f1) < 0.05)

    print(f"\nH1 — family-split F1 ≥ 0.70:  {'PASS' if h1 else 'FAIL'}  (F1={fs_f1:.3f})")
    print(f"H2 — enzyme >> mechanism floor:  {'CONFIRMED' if h2 else 'NOT CONFIRMED'}  "
          f"(Δ={fs_f1 - MECHANISM_FAMILY_SPLIT_F1:+.3f})")
    print(f"H4 — MLP ≈ LogReg family-split:  {'CONFIRMED' if h4 else 'NOT CONFIRMED'}  "
          f"(ΔMLP-LR={mlp_f1 - fs_f1:+.3f})")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    output = {
        "description": (
            "Enzyme type classification (kinase/protease/oxidoreductase/non-enzyme) "
            "from ESM-2 WT mean-pooled embeddings. Positive control paralleling the "
            "mechanism arc (results 1-10). Primary metric: family-split LogReg macro-F1."
        ),
        "seeds": args.seeds,
        "n_folds": args.n_folds,
        "n_genes": len(gene_list),
        "class_distribution": dict(Counter(y_str)),
        "classes": list(le.classes_),
        "esm2_wt_embedding": emb_results,
        "proteome_features": proteome_results,
        "hypothesis_evaluation": {
            "H1_family_split_f1_ge_0.70": bool(h1),
            "H2_enzyme_beats_mechanism_by_0.10": bool(h2),
            "H4_mlp_approx_logreg": bool(h4),
            "fs_f1": fs_f1,
            "mlp_f1": mlp_f1,
            "gs_f1": gs_f1,
            "mechanism_reference_f1": MECHANISM_FAMILY_SPLIT_F1,
        },
        "references": {
            "mechanism_family_split_f1_result7": MECHANISM_FAMILY_SPLIT_F1,
            "mechanism_gene_split_f1_result7": MECHANISM_GENE_SPLIT_F1,
        },
    }

    out_path = out_dir / "enzyme_classification_summary.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
