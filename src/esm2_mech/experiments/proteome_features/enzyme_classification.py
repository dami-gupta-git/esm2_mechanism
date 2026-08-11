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
                                    [--seeds 5]

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
from esm2_mech.utils.constants import N_SEEDS
from esm2_mech.utils.splits import gene_split_cv, family_split_cv
from esm2_mech.utils.data import build_gene_to_row
from esm2_mech.utils.paths import (
    EMB_WT_MEAN,
    ENZYME_LABELS_TSV,
    GENE_UNIVERSE,
    PFAM_JSON,
    PROTEOME_FEATURES_ALIGNED,
    PROTEOME_FEATURE_COLUMNS_JSON,
    RESULTS_DIR,
    VALID_VARIANTS_JSON,
)

print = functools.partial(print, flush=True)

OUT_DIR = RESULTS_DIR
warnings.filterwarnings("ignore")

ENZYME_CLASSES = ["kinase", "protease", "oxidoreductase", "non-enzyme"]

# Reference numbers for comparison (from docs/README.md)
MECHANISM_FAMILY_SPLIT_F1 = 0.385  # merged dataset, 5-seed (result 7)
MECHANISM_GENE_SPLIT_F1 = 0.415  # merged dataset, MLP, seed 0 (result 7)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_gene_embeddings() -> tuple:
    """
    Load per-gene WT embeddings by taking the first variant's embedding for each gene.
    Returns (X, gene_list, uniprot_list) where X has shape (n_genes, 1280).
    """
    with open(VALID_VARIANTS_JSON) as f:
        variants = json.load(f)

    wt_mean = np.load(EMB_WT_MEAN)
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


def load_enzyme_labels() -> dict:
    """Load gene -> enzyme_4class from enzyme_labels.tsv."""
    import csv

    labels: dict[str, str] = {}
    label_path = ENZYME_LABELS_TSV
    with open(label_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            labels[row["gene"]] = row["enzyme_4class"]
    print(f"Loaded enzyme labels for {len(labels)} genes")
    return labels


def load_pfam() -> dict:
    with open(PFAM_JSON) as f:
        return json.load(f)


def load_proteome_features() -> tuple:
    """Load the proteome feature matrix and the gene list its rows are aligned to.

    The row order is gene_universe.tsv, not gene_list.tsv — the latter is a
    longer, differently-ordered superset, so indexing the matrix by it silently
    reads the wrong gene's features.
    """
    X = np.load(PROTEOME_FEATURES_ALIGNED).astype(np.float32)
    with open(PROTEOME_FEATURE_COLUMNS_JSON) as f:
        cols = json.load(f)
    genes = list(build_gene_to_row(GENE_UNIVERSE))
    if len(genes) != X.shape[0]:
        raise ValueError(
            f"{PROTEOME_FEATURES_ALIGNED} has {X.shape[0]} rows but "
            f"{GENE_UNIVERSE} lists {len(genes)} genes — not row-aligned."
        )
    print(f"Proteome features: {X.shape}, {len(genes)} genes")
    return X, genes


# ---------------------------------------------------------------------------
# Probe runners
# ---------------------------------------------------------------------------


def _run_cv(
    clf_factory,
    scale: bool,
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple],
    classes: list[str],
    seed: int = 42,
) -> dict:
    """Shared CV body for the three probe runners, which differ only in the
    estimator and whether the features need standardizing.

    scale=False is used by the NaN-native runner: gradient-boosted trees are
    invariant to monotone rescaling, and StandardScaler would need its own
    missing-value handling anyway.
    """
    n_cls = len(classes)
    fold_f1s, fold_aurocs = [], {c: [] for c in classes}

    for tr, te in splits:
        if len(set(y[tr])) < 2:
            continue
        if scale:
            sc = StandardScaler().fit(X[tr])
            X_tr, X_te = sc.transform(X[tr]), sc.transform(X[te])
        else:
            X_tr, X_te = X[tr], X[te]
        clf = clf_factory(seed)
        clf.fit(X_tr, y[tr])
        proba_raw = clf.predict_proba(X_te)

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
        "per_class_auroc_mean": {
            c: float(np.mean(v)) if v else None for c, v in fold_aurocs.items()
        },
        "per_class_auroc_std": {
            c: float(np.std(v)) if v else None for c, v in fold_aurocs.items()
        },
        "n_folds": len(fold_f1s),
    }


def run_logreg(X, y, splits, classes, seed: int = 42) -> dict:
    """Linear probe. Requires fully-observed X (see run_histgb for NaN input)."""
    from sklearn.linear_model import LogisticRegression

    return _run_cv(
        lambda s: LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=s
        ),
        True, X, y, splits, classes, seed,
    )


def run_mlp(X, y, splits, classes, seed: int = 42) -> dict:
    """Nonlinear probe. Requires fully-observed X (see run_histgb for NaN input)."""
    return _run_cv(
        lambda s: MLPClassifier(
            hidden_layer_sizes=(256, 64),
            activation="relu",
            alpha=1e-3,
            max_iter=300,
            random_state=s,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
        ),
        True, X, y, splits, classes, seed,
    )


def run_histgb(X, y, splits, classes, seed: int = 42) -> dict:
    """NaN-native probe — for the proteome matrix, which has real missing cells.

    Consumes NaN directly, so no value is fabricated and no gene is dropped.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    return _run_cv(
        lambda s: HistGradientBoostingClassifier(
            max_iter=200, class_weight="balanced", random_state=s
        ),
        False, X, y, splits, classes, seed,
    )


def majority_baseline_f1(y: np.ndarray) -> float:
    majority = Counter(y.tolist()).most_common(1)[0][0]
    pred = np.full(len(y), majority)
    return float(f1_score(y, pred, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------


def run_multiseed(
    X: np.ndarray,
    y: np.ndarray,
    genes: list[str],
    pfam_map: dict,
    le: LabelEncoder,
    seeds: list[int],
    n_folds: int = 5,
    nan_native: bool = False,
) -> dict:
    """Run the linear and nonlinear probes across seeds.

    nan_native : set for a feature matrix with missing cells (the proteome
        block). Both probes then use the NaN-native booster, which consumes
        missing values directly rather than having them imputed — imputation
        before the CV split would leak test-fold statistics into training. The
        dense embedding arm leaves this False and keeps LogReg/MLP.
    """
    linear_probe = run_histgb if nan_native else run_logreg
    nonlinear_probe = run_histgb if nan_native else run_mlp
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
        gs = linear_probe(X, y, gs_splits, classes, seed=seed)
        gs_f1s.append(gs["macro_f1_mean"])
        for c in classes:
            v = gs["per_class_auroc_mean"].get(c)
            if v is not None:
                gs_aurocs[c].append(v)
        print(f"    LogReg gene-split  F1={gs['macro_f1_mean']:.3f}")

        # Family-split LogReg
        fs_splits = family_split_cv(genes, pfam_map, n_folds=n_folds, seed=seed)
        fs = linear_probe(X, y, fs_splits, classes, seed=seed)
        fs_f1s.append(fs["macro_f1_mean"])
        for c in classes:
            v = fs["per_class_auroc_mean"].get(c)
            if v is not None:
                fs_aurocs[c].append(v)
        print(
            f"    LogReg family-split F1={fs['macro_f1_mean']:.3f}  "
            f"AUROC: "
            + " ".join(
                f"{c}={fs['per_class_auroc_mean'].get(c, float('nan')):.3f}"
                for c in classes
            )
        )

        # Family-split MLP
        mlp = nonlinear_probe(X, y, fs_splits, classes, seed=seed)
        mlp_f1s.append(mlp["macro_f1_mean"])
        for c in classes:
            v = mlp["per_class_auroc_mean"].get(c)
            if v is not None:
                mlp_aurocs[c].append(v)
        print(f"    MLP    family-split F1={mlp['macro_f1_mean']:.3f}")

    maj_f1 = majority_baseline_f1(y)

    def _agg(vals):
        vals = [v for v in vals if v is not None]
        return (
            float(np.mean(vals)) if vals else None,
            float(np.std(vals)) if vals else None,
        )

    gs_mean, gs_std = _agg(gs_f1s)
    fs_mean, fs_std = _agg(fs_f1s)
    mlp_mean, mlp_std = _agg(mlp_f1s)

    leakage_pct = None
    if gs_mean and fs_mean and gs_mean > 0:
        leakage_pct = round(100.0 * (gs_mean - fs_mean) / gs_mean, 1)

    print(f"\n  Results ({len(seeds)} seeds):")
    print(f"    Majority baseline:       F1={maj_f1:.3f}")
    print(f"    LogReg gene-split:       F1={gs_mean:.3f} ± {gs_std:.3f}")
    print(
        f"    LogReg family-split:     F1={fs_mean:.3f} ± {fs_std:.3f}  ← primary metric"
    )
    print(f"    MLP    family-split:     F1={mlp_mean:.3f} ± {mlp_std:.3f}")
    if leakage_pct is not None:
        print(f"    Leakage fraction:        {leakage_pct:.1f}%")
    print(f"\n  vs mechanism family-split floor ({MECHANISM_FAMILY_SPLIT_F1:.3f}):")
    if fs_mean is not None:
        delta = fs_mean - MECHANISM_FAMILY_SPLIT_F1
        sym = (
            ">>>"
            if delta > 0.15
            else (">>" if delta > 0.05 else (">" if delta > 0 else "~"))
        )
        print(
            f"    {sym} enzyme {fs_mean:.3f}  vs  mechanism {MECHANISM_FAMILY_SPLIT_F1:.3f}  Δ={delta:+.3f}"
        )

    return {
        "majority_f1": maj_f1,
        "logreg_gene_split": {
            "macro_f1_mean": gs_mean,
            "macro_f1_std": gs_std,
            "per_class_auroc_mean": {
                c: float(np.mean(v)) if v else None for c, v in gs_aurocs.items()
            },
            "n_seeds": len(seeds),
        },
        "logreg_family_split": {
            "macro_f1_mean": fs_mean,
            "macro_f1_std": fs_std,
            "per_class_auroc_mean": {
                c: float(np.mean(v)) if v else None for c, v in fs_aurocs.items()
            },
            "n_seeds": len(seeds),
        },
        "mlp_family_split": {
            "macro_f1_mean": mlp_mean,
            "macro_f1_std": mlp_std,
            "per_class_auroc_mean": {
                c: float(np.mean(v)) if v else None for c, v in mlp_aurocs.items()
            },
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
    parser.add_argument("--seeds", type=int, default=N_SEEDS,
                        help="number of seeds to run; runs 0..seeds-1 (>=1)")
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")

    seeds = list(range(args.seeds))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Enzyme Classification from ESM-2 WT Embeddings ===")
    print(f"Seeds: {seeds}  Folds: {args.n_folds}")

    # Load data
    X_emb, gene_list, _ = load_gene_embeddings()
    enzyme_labels = load_enzyme_labels()
    pfam_map = load_pfam()

    # Align labels to gene_list order
    missing = [g for g in gene_list if g not in enzyme_labels]
    if missing:
        print(
            f"  WARNING: {len(missing)} genes have no enzyme label and will be assigned 'non-enzyme': "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    y_str = [enzyme_labels.get(g, "non-enzyme") for g in gene_list]
    le = LabelEncoder()
    le.fit(ENZYME_CLASSES)
    y = le.transform(y_str)

    print(f"\nGenes in embedding: {len(gene_list)}")
    print(
        f"Label coverage: {sum(1 for g in gene_list if g in enzyme_labels)}/{len(gene_list)}"
    )
    print(f"Class distribution: {dict(Counter(y_str))}")
    print(
        f"Pfam-annotated genes: {sum(1 for g in gene_list if pfam_map.get(g))}/{len(gene_list)}"
    )

    # -----------------------------------------------------------------------
    # ESM-2 WT embedding probes
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PART 1: ESM-2 WT embedding (1280-dim)")
    print("=" * 60)
    emb_results = run_multiseed(
        X_emb, y, gene_list, pfam_map, le, seeds=seeds, n_folds=args.n_folds
    )

    # -----------------------------------------------------------------------
    # Proteome feature baseline
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PART 2: Proteome features (37-dim) — baseline comparison")
    print("=" * 60)

    X_prot, prot_genes = load_proteome_features()
    prot_gene_to_idx = {g: i for i, g in enumerate(prot_genes)}

    # Align proteome features to gene_list
    prot_aligned_idxs = [prot_gene_to_idx[g] for g in gene_list if g in prot_gene_to_idx]
    prot_aligned_genes = [g for g in gene_list if g in prot_gene_to_idx]
    prot_aligned_y = y[[gene_list.index(g) for g in prot_aligned_genes]]
    X_prot_aligned = X_prot[prot_aligned_idxs]

    # The proteome matrix carries real NaN — nothing is imputed at build time,
    # because a value filled in before the CV split leaks test-fold statistics
    # into training and is indistinguishable from a real measurement after.
    # run_multiseed therefore uses the NaN-native probes, which consume the
    # missing cells directly and keep every gene.
    n_nan = int(np.isnan(X_prot_aligned).sum())
    if n_nan:
        n_rows = int(np.isnan(X_prot_aligned).any(axis=1).sum())
        print(
            f"  {n_nan} missing cells across {n_rows}/{len(X_prot_aligned)} genes "
            f"— consumed natively, not imputed"
        )

    print(f"Proteome-aligned genes: {len(prot_aligned_genes)}")
    proteome_results = run_multiseed(
        X_prot_aligned,
        prot_aligned_y,
        prot_aligned_genes,
        pfam_map,
        le,
        seeds=seeds,
        n_folds=args.n_folds,
        nan_native=True,
    )

    # -----------------------------------------------------------------------
    # Decision rule evaluation (pre-registered)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PRE-REGISTERED DECISION RULES")
    print("=" * 60)
    fs_f1 = emb_results["logreg_family_split"]["macro_f1_mean"]
    mlp_f1 = emb_results["mlp_family_split"]["macro_f1_mean"]
    gs_f1 = emb_results["logreg_gene_split"]["macro_f1_mean"]

    h1 = fs_f1 is not None and fs_f1 >= 0.70
    h2 = fs_f1 is not None and (fs_f1 - MECHANISM_FAMILY_SPLIT_F1) > 0.10
    h4 = mlp_f1 is not None and fs_f1 is not None and abs(mlp_f1 - fs_f1) < 0.05

    print(
        f"\nH1 — family-split F1 ≥ 0.70:  {'PASS' if h1 else 'FAIL'}  (F1={fs_f1:.3f})"
    )
    print(
        f"H2 — enzyme >> mechanism floor:  {'CONFIRMED' if h2 else 'NOT CONFIRMED'}  "
        f"(Δ={fs_f1 - MECHANISM_FAMILY_SPLIT_F1:+.3f})"
    )
    print(
        f"H4 — MLP ≈ LogReg family-split:  {'CONFIRMED' if h4 else 'NOT CONFIRMED'}  "
        f"(ΔMLP-LR={mlp_f1 - fs_f1:+.3f})"
    )

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    output = {
        "description": (
            "Enzyme type classification (kinase/protease/oxidoreductase/non-enzyme) "
            "from ESM-2 WT mean-pooled embeddings. Positive control paralleling the "
            "mechanism arc (results 1-10). Primary metric: family-split LogReg macro-F1."
        ),
        "seeds": seeds,
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

    out_path = OUT_DIR / "enzyme_classification_summary.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
