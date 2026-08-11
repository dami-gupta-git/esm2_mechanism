"""
Experiment 11 — Phase 3 Modelling
===================================

Four model variants under 5-fold family-split CV with 5 seeds (seeds 0–4):

  V1  ESM-2 delta only (1280-dim)              MLP 1280→256→64→3
  V2  Proteome features only (41-dim)          LightGBM (primary)
  V3  ESM-2 delta + proteome (1321-dim)        NaN-native gradient boosting
  V4  Contrastive head on V3 inputs (1321-dim) projection 1321→256→64,
                                               TripletMarginLoss + kNN

Missing-data policy. The proteome block carries real NaN (its family-residual
columns are ~44-61% missing by design); nothing is imputed. V1 is delta-only
and fully observed, so it keeps its MLP. Every arm that touches the proteome
block — including V3, where it is concatenated onto the dense delta — uses a
model that consumes NaN directly, which keeps all genes without fabricating a
value. V2 also reports a LogReg restricted to fully-observed genes to answer
whether the signal is linear, paired with a NaN-native run on those same rows.
V4's projection head and k-NN cannot take NaN at all, so V4 is complete-case
and V3 is re-run on V4's exact rows as its comparator.

Pre-registered decision gates:
  Gate 1 (V2): macro-F1 ≥ 0.35              → proceed to V3
  Gate 2 (V3): macro-F1 ≥ max(V1,V2)+0.02  → proceed to V4
  Gate 3 (V4): report only — against V3 on V4's own gene subset, so the gate
               reads a method difference rather than a change in coverage

Also runs gene-split CV for V3 to compute leakage delta.

Usage:
    python scripts/proteome_mechanism.py                    # all 5 seeds
    python scripts/proteome_mechanism.py --seed 0           # single seed
    python scripts/proteome_mechanism.py --variants-only    # skip V4
    python scripts/proteome_mechanism.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from esm2_mech.utils.splits import family_split_indices
from esm2_mech.utils.constants import MECHANISM_CLASSES
from esm2_mech.utils.metrics import compute_metrics, mean_std_n, align_proba
from esm2_mech.utils.probes import run_mlp_cv, run_logreg_cv, run_histgb_cv
from esm2_mech.utils.data import build_gene_to_row, observed_rows_mask
from esm2_mech.utils.io import load_variants_and_delta
from esm2_mech.utils.paths import (
    RESULTS_DIR,
    VALID_VARIANTS_JSON,
    EMB_WT_MEAN,
    EMB_MUT_MEAN,
    GENE_UNIVERSE,
    PFAM_JSON,
    PROTEOME_FEATURES_ALIGNED,
    PROTEOME_FEATURE_COLUMNS_JSON,
    PROTEOME_FEATURES_TSV,
    BADONYI_FEATURES_ALIGNED,
    BADONYI_FEATURE_COLUMNS_JSON,
    BADONYI_FEATURES_TSV,
)
import functools

print = functools.partial(print, flush=True)

OUT_DIR = RESULTS_DIR

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MERGED_VALID_VARIANTS = VALID_VARIANTS_JSON
MERGED_WT_MEAN = EMB_WT_MEAN
MERGED_MUT_MEAN = EMB_MUT_MEAN

PROTEOME_FEATURES = PROTEOME_FEATURES_ALIGNED
PROTEOME_COLS = PROTEOME_FEATURE_COLUMNS_JSON
# Row index for the aligned feature matrices. MUST be GENE_UNIVERSE, not
# GENE_LIST_TSV: build_proteome_features/build_badonyi_features write their
# .npy rows in gene_universe.tsv order, and gene_list.tsv is a longer,
# differently-ordered superset (see paths.GENE_UNIVERSE).
MERGED_GENE_LIST = GENE_UNIVERSE
PFAM_FAMILIES = PFAM_JSON

CLASSES = MECHANISM_CLASSES

# ---------------------------------------------------------------------------
# Gene-split CV (from mlp.py / contrastive_mechanism.py)
# ---------------------------------------------------------------------------


def gene_split_indices(
    genes: np.ndarray,
    n_folds: int,
    seed: int,
    pfam_map: dict | None = None,
):
    """
    Yield (train_idx, test_idx) splitting by unique gene identity, while
    keeping all genes from the same Pfam family in the same fold.

    Without pfam_map: simple gene-split (original behaviour, contaminated by
    family leakage — retained for comparison only).

    With pfam_map: family-aware gene-split. Genes are grouped by their Pfam
    family first; families are shuffled and assigned to folds; then within
    each family, individual genes are the unit of test holdout. This ensures
    no family straddles a train/test boundary, making the leakage delta
    (gene-split F1 − family-split F1) a clean measure of within-family
    positional signal rather than a contaminated lower bound.
    """
    rng = np.random.RandomState(seed)

    if pfam_map is None:
        # Original behaviour — kept for reference
        unique_genes = np.array(sorted(set(genes)))
        rng.shuffle(unique_genes)
        gene_folds = np.array_split(unique_genes, n_folds)
        for fold_genes in gene_folds:
            fold_set = set(fold_genes.tolist())
            test_mask = np.array([g in fold_set for g in genes])
            train_mask = ~test_mask
            if train_mask.sum() < 10 or test_mask.sum() < 5:
                continue
            yield np.where(train_mask)[0], np.where(test_mask)[0]
        return

    # Family-aware gene-split
    # 1. Group unique genes by Pfam family
    unique_genes = sorted(set(genes.tolist()))
    fam_to_genes: dict[str, list[str]] = {}
    no_fam: list[str] = []
    for g in unique_genes:
        fam = pfam_map.get(g)
        if fam:
            fam_to_genes.setdefault(fam, []).append(g)
        else:
            no_fam.append(g)

    # 2. Shuffle families, assign each family to a fold
    fam_list = np.array(sorted(fam_to_genes.keys()))
    rng.shuffle(fam_list)
    fam_fold = {f: i % n_folds for i, f in enumerate(fam_list)}

    # 3. For each fold k: test = all variants whose gene is in a family
    #    assigned to fold k; train = everything else (no family overlap)
    gene_fold = {}
    for fam, fold_k in fam_fold.items():
        for g in fam_to_genes[fam]:
            gene_fold[g] = fold_k
    # Genes with no family: distribute across folds (they never leak)
    rng.shuffle(no_fam)
    for i, g in enumerate(no_fam):
        gene_fold[g] = i % n_folds

    fold_of = np.array([gene_fold.get(g, 0) for g in genes])
    for k in range(n_folds):
        test = np.where(fold_of == k)[0]
        train = np.where(fold_of != k)[0]
        if len(train) < 10 or len(test) < 5:
            continue
        yield train, test


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data() -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    """
    Load merged variants, labels, genes, and ESM-2 delta embeddings.

    Returns:
        variants  : list of variant dicts
        labels    : (n,) string array of 3-class labels
        genes     : (n,) string array of gene symbols
        delta     : (n, 1280) float32 ESM-2 delta embeddings
    """
    variants, labels, genes, delta, _ = load_variants_and_delta(
        MERGED_VALID_VARIANTS, MERGED_WT_MEAN, MERGED_MUT_MEAN
    )
    return variants, labels, genes, delta


def load_pfam() -> dict[str, str]:
    with open(PFAM_FAMILIES) as f:
        return json.load(f)


def build_gene_to_proteome_row() -> dict[str, int]:
    """{gene_symbol: row_index} for indexing into proteome_features_aligned.npy."""
    return build_gene_to_row(MERGED_GENE_LIST)


def load_proteome_features(genes: np.ndarray) -> np.ndarray:
    """
    Broadcast gene-level proteome features (2424 × 41) to variant level.

    For each variant, look up its gene's row in the proteome matrix.
    Variants whose gene is absent from the gene-order list are given a
    zero-filled row (should not occur given aligned construction, but
    this is defensive).

    Returns (n_variants, 41) float32 array.
    """
    prot_matrix = np.load(PROTEOME_FEATURES).astype(np.float32)  # (2424, 41)
    gene_to_row = build_gene_to_proteome_row()

    n = len(genes)
    n_feats = prot_matrix.shape[1]
    # NaN, not 0.0, for a variant whose gene has no proteome row: 0.0 is a
    # plausible real value for pLI/LOEUF/PPI_degree and a zero-filled row would
    # be indistinguishable from a real measurement. The proteome arms consume
    # NaN natively.
    X_prot = np.full((n, n_feats), np.nan, dtype=np.float32)
    n_missing = 0
    for i, g in enumerate(genes):
        row = gene_to_row.get(g)
        if row is not None and row < prot_matrix.shape[0]:
            X_prot[i] = prot_matrix[row]
        else:
            n_missing += 1
    if n_missing > 0:
        print(f"  {n_missing} variants have no proteome row for their gene (NaN)")
    print(f"Proteome features broadcast: {X_prot.shape}")
    return X_prot


def aggregate_fold_results(fold_list: list[dict]) -> dict:
    """Aggregate a list of per-fold metric dicts into mean ± std."""
    if not fold_list:
        return {"error": "no folds"}

    out: dict = {}
    # macro_f1 — NaN-filtered so a single NaN fold (e.g. a degenerate split)
    # does not poison the mean, matching the per-class AUROC guard below.
    macro_mean, macro_std, _ = mean_std_n([f["macro_f1"] for f in fold_list])
    out["macro_f1_mean"] = macro_mean
    out["macro_f1_std"] = macro_std

    # per-class AUROC
    for cls in CLASSES:
        vals_cls = [
            f["per_class_auroc"][cls]
            for f in fold_list
            if f.get("per_class_auroc", {}).get(cls) is not None
        ]
        if vals_cls:
            out[f"auroc_{cls}_mean"] = float(np.mean(vals_cls))
            out[f"auroc_{cls}_std"] = float(np.std(vals_cls))
        else:
            out[f"auroc_{cls}_mean"] = None
            out[f"auroc_{cls}_std"] = None

    out["n_folds"] = len(fold_list)
    return out


# ---------------------------------------------------------------------------
# V1 / V2 / V3 — sklearn MLPClassifier runners
# ---------------------------------------------------------------------------


def run_family_split_mlp(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    hidden_layer_sizes: tuple,
    n_folds: int,
    seed: int,
    label: str,
) -> dict:
    splits = list(family_split_indices(groups, n_folds, seed))
    return run_mlp_cv(X, y, splits, hidden=hidden_layer_sizes, seed=seed, label=label)


def run_family_split_logreg(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    seed: int,
    label: str,
) -> dict:
    splits = list(family_split_indices(groups, n_folds, seed))
    return run_logreg_cv(X, y, splits, seed=seed, label=label)


def run_family_split_histgb(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    seed: int,
    label: str,
) -> dict:
    """NaN-native family-split CV — for any arm whose matrix includes the
    proteome block, alone or concatenated with the ESM-2 delta."""
    splits = list(family_split_indices(groups, n_folds, seed))
    return run_histgb_cv(X, y, splits, seed=seed, label=label)


def run_observed_subset_arm(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    seed: int,
    label: str,
    runner,
    **runner_kwargs,
) -> dict:
    """Run `runner` on the fully-observed rows of X, with folds recomputed there.

    For a model that cannot consume NaN. The returned metrics describe only the
    complete-case subset, which is smaller than and not a random sample of the
    full gene set (genes drop out for being singletons or sitting in small
    families), so `n_observed` / `frac_observed` are reported alongside and any
    comparison must be against an arm run on this same subset.
    """
    observed = observed_rows_mask(X, label=label)
    n_obs = int(observed.sum())
    if n_obs < n_folds:
        return {
            "error": "too few fully-observed rows",
            "n_observed": n_obs,
            "n_total": int(len(X)),
        }
    splits = list(family_split_indices(groups[observed], n_folds, seed))
    result = runner(
        X[observed], y[observed], splits, seed=seed, label=label, **runner_kwargs
    )
    result["n_observed"] = n_obs
    result["n_total"] = int(len(X))
    result["frac_observed"] = float(n_obs / len(X))
    return result


# ---------------------------------------------------------------------------
# V2 — LightGBM (gradient boosting, handles tabular/sparse features well)
# ---------------------------------------------------------------------------


def run_family_split_lgbm(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    seed: int,
    label: str,
) -> dict:
    """5-fold family-split CV with LightGBM multiclass classifier."""
    import lightgbm as lgb

    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(
        family_split_indices(groups, n_folds, seed)
    ):
        X_tr, X_te = X[train_idx].astype(np.float32), X[test_idx].astype(np.float32)
        y_tr, y_te = y[train_idx], y[test_idx]

        if len(set(y_tr.tolist())) < len(CLASSES):
            print(f"    [{label}] Fold {fold_i+1}: skipped (missing class in train)")
            continue
        if len(set(y_te.tolist())) < 2:
            print(f"    [{label}] Fold {fold_i+1}: skipped (< 2 test classes)")
            continue

        # Class weights — inverse frequency
        counts = {c: int((y_tr == c).sum()) for c in CLASSES}
        class_weight = {c: 1.0 / max(counts[c], 1) for c in CLASSES}
        sample_weight = np.array([class_weight[yi] for yi in y_tr], dtype=np.float32)

        clf = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbose=-1,
            n_jobs=4,
        )
        clf.fit(X_tr, y_tr, sample_weight=sample_weight)

        pred = clf.predict(X_te)
        proba_aligned = align_proba(clf.predict_proba(X_te), clf.classes_, CLASSES)

        fm = compute_metrics(y_te, pred, proba_aligned)
        fold_results.append(fm)
        print(
            f"    [{label}] Fold {fold_i+1}: macro_f1={fm['macro_f1']:.3f}  "
            + "  ".join(
                (
                    f"{cls}={fm['per_class_auroc'].get(cls, float('nan')):.3f}"
                    if fm["per_class_auroc"].get(cls) is not None
                    else f"{cls}=NA"
                )
                for cls in CLASSES
            )
        )

    return aggregate_fold_results(fold_results)


# ---------------------------------------------------------------------------
# Gene-split CV for V3 (leakage diagnostic)
# ---------------------------------------------------------------------------


def run_gene_split_histgb(
    X: np.ndarray,
    y: np.ndarray,
    genes: np.ndarray,
    n_folds: int,
    seed: int,
    label: str,
    pfam_map: dict | None = None,
) -> dict:
    """NaN-native gene-split CV — the leakage-diagnostic counterpart to
    run_family_split_histgb, for matrices that include the proteome block."""
    splits = list(gene_split_indices(genes, n_folds, seed, pfam_map=pfam_map))
    return run_histgb_cv(X, y, splits, seed=seed, label=label)


# ---------------------------------------------------------------------------
# V4 — Contrastive projection head (PyTorch) + k-NN
# ---------------------------------------------------------------------------


def build_triplets_v4(
    y: np.ndarray,
    gene_pfam: np.ndarray,
    max_triplets: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mine triplets for V4:
      anchor: any variant
      positive: same mechanism, different Pfam family
      negative: different mechanism (hard negative)

    Capped at max_triplets total.
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    n_classes = len(CLASSES)

    # Local int encoding for this function's dict-of-int-keys bookkeeping only;
    # y itself stays string-typed for every other caller (compute_metrics,
    # run_logreg_cv, run_mlp_cv all key on the string labels in CLASSES).
    cls_to_idx = {c: i for i, c in enumerate(CLASSES)}
    y_int = np.array([cls_to_idx[lbl] for lbl in y])

    # Encode family strings to ints
    unique_fams = [f for f in set(gene_pfam.tolist()) if f is not None]
    fam_to_int = {f: i for i, f in enumerate(unique_fams)}
    fam_int = np.array([fam_to_int.get(f, -1) for f in gene_pfam], dtype=np.int32)

    by_mech = {c: np.where(y_int == c)[0] for c in range(n_classes)}
    neg_by_class = {
        c: np.concatenate([by_mech[o] for o in range(n_classes) if o != c])
        for c in range(n_classes)
    }

    # For each (class, family) build cross-family positive pool
    by_mech_fam_arr = {c: np.array(by_mech[c]) for c in range(n_classes)}
    unique_combos = set((int(y_int[i]), int(fam_int[i])) for i in range(n))
    combo_pos_pool: dict[tuple[int, int], np.ndarray] = {}
    for c, fam in unique_combos:
        idxs = by_mech_fam_arr[c]
        cross = idxs[fam_int[idxs] != fam]
        # also exclude -1 (no family) from positives
        cross = cross[fam_int[cross] != -1]
        combo_pos_pool[(c, fam)] = cross

    anchor_list, pos_list, neg_list = [], [], []
    triplets_per_anchor = max(1, max_triplets // n)

    for i in range(n):
        c = int(y_int[i])
        fam = int(fam_int[i])
        pos_pool = combo_pos_pool.get((c, fam), np.array([], dtype=np.int64))
        neg_pool = neg_by_class[c]

        if len(pos_pool) == 0 or len(neg_pool) == 0:
            continue

        k = min(triplets_per_anchor, len(pos_pool), len(neg_pool))
        ps = pos_pool[rng.randint(0, len(pos_pool), k)]
        ns = neg_pool[rng.randint(0, len(neg_pool), k)]
        anchor_list.append(np.full(k, i, dtype=np.int64))
        pos_list.append(ps.astype(np.int64))
        neg_list.append(ns.astype(np.int64))

    if not anchor_list:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )

    anchors = np.concatenate(anchor_list)
    positives = np.concatenate(pos_list)
    negatives = np.concatenate(neg_list)

    # Cap total
    if len(anchors) > max_triplets:
        idx = rng.choice(len(anchors), max_triplets, replace=False)
        anchors, positives, negatives = anchors[idx], positives[idx], negatives[idx]

    return anchors, positives, negatives


def train_projection_head_v4(
    X_train: np.ndarray,
    y_train: np.ndarray,
    gene_pfam_train: np.ndarray,
    n_epochs: int,
    lr: float,
    max_triplets: int,
    seed: int,
) -> tuple:
    """
    Train projection head: Linear(1321,256) → ReLU → Linear(256,64)
    with TripletMarginLoss. Returns (proj_model, mu, std).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    mu = X_train.mean(0).astype(np.float32)
    std = (X_train.std(0) + 1e-8).astype(np.float32)
    X_norm = ((X_train - mu) / std).astype(np.float32)

    anchors, positives, negatives = build_triplets_v4(
        y_train, gene_pfam_train, max_triplets=max_triplets, seed=seed
    )

    if len(anchors) < 20:
        print(f"      WARNING: only {len(anchors)} triplets; V4 may be unreliable")

    in_dim = X_norm.shape[1]
    proj = nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
    ).to(device)
    optimizer = torch.optim.Adam(proj.parameters(), lr=lr)
    triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

    X_t = torch.tensor(X_norm, dtype=torch.float32).to(device)

    anc_t = torch.tensor(anchors, dtype=torch.long)
    pos_t = torch.tensor(positives, dtype=torch.long)
    neg_t = torch.tensor(negatives, dtype=torch.long)

    ds = TensorDataset(anc_t, pos_t, neg_t)
    loader = DataLoader(ds, batch_size=512, shuffle=True)

    proj.train()
    for epoch in range(n_epochs):
        for anc_b, pos_b, neg_b in loader:
            anc_b, pos_b, neg_b = anc_b.to(device), pos_b.to(device), neg_b.to(device)
            optimizer.zero_grad()
            z_a = proj(X_t[anc_b])
            z_p = proj(X_t[pos_b])
            z_n = proj(X_t[neg_b])
            loss = triplet_loss_fn(z_a, z_p, z_n)
            loss.backward()
            optimizer.step()

    proj.eval()
    with torch.no_grad():
        Z_train = proj(X_t).cpu().numpy()

    return proj, mu, std, Z_train


def project_v4(proj, X: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    import torch

    device = next(proj.parameters()).device
    X_norm = (X.astype(np.float32) - mu) / std
    proj.eval()
    with torch.no_grad():
        Z = proj(torch.tensor(X_norm, dtype=torch.float32).to(device)).cpu().numpy()
    return Z


def run_knn_v4(
    Z_train: np.ndarray,
    Z_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    k: int = 10,
) -> dict:
    k_eff = min(k, len(Z_train) - 1)
    knn = KNeighborsClassifier(n_neighbors=k_eff, metric="cosine")
    knn.fit(Z_train, y_train)
    pred = knn.predict(Z_test)

    proba_aligned = align_proba(knn.predict_proba(Z_test), knn.classes_, CLASSES)

    return compute_metrics(y_test, pred, proba_aligned)


def run_v4_family_split(
    X: np.ndarray,
    y: np.ndarray,
    genes: np.ndarray,
    gene_pfam: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    seed: int,
    n_epochs: int = 30,
    lr: float = 1e-3,
    max_triplets: int = 2000,
) -> dict:
    """
    V4: contrastive projection head + k-NN under family-split CV.
    Uses the same fold splits as V1–V3 (family_split_indices).
    """
    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(
        family_split_indices(groups, n_folds, seed)
    ):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        gene_pfam_tr = gene_pfam[train_idx]

        if len(set(y_tr.tolist())) < len(CLASSES):
            print(f"    [V4] Fold {fold_i+1}: skipped (missing class in train)")
            continue
        if len(set(y_te.tolist())) < 2:
            print(f"    [V4] Fold {fold_i+1}: skipped (< 2 test classes)")
            continue

        print(f"    [V4] Fold {fold_i+1}: training projection head ...")
        proj, mu, std, Z_tr = train_projection_head_v4(
            X_tr,
            y_tr,
            gene_pfam_tr,
            n_epochs=n_epochs,
            lr=lr,
            max_triplets=max_triplets,
            seed=seed + fold_i,
        )
        Z_te = project_v4(proj, X_te, mu, std)

        fm = run_knn_v4(Z_tr, Z_te, y_tr, y_te, k=10)
        fold_results.append(fm)
        print(
            f"    [V4] Fold {fold_i+1}: macro_f1={fm['macro_f1']:.3f}  "
            + "  ".join(
                (
                    f"{cls}={fm['per_class_auroc'].get(cls, float('nan')):.3f}"
                    if fm["per_class_auroc"].get(cls) is not None
                    else f"{cls}=NA"
                )
                for cls in CLASSES
            )
        )

    return aggregate_fold_results(fold_results)


# ---------------------------------------------------------------------------
# Single-seed runner
# ---------------------------------------------------------------------------


def run_seed(
    seed: int,
    n_folds: int,
    variants_only: bool,
    labels: np.ndarray,
    genes: np.ndarray,
    delta: np.ndarray,
    X_prot: np.ndarray,
    pfam_map: dict[str, str],
) -> dict:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")

    # y stays the string labels themselves (matching CLASSES/MECHANISM_CLASSES).
    # run_mlp_cv/run_logreg_cv/compute_metrics compare y against `classes`
    # (string labels) internally; an int-encoded y silently produces zero
    # counts for every class (LogReg/MLP) or zero AUROC for every class
    # (compute_metrics, since `y_true == "GOF"` is always False against ints).
    y = np.asarray(labels)

    # Build groups (variant-level Pfam family for family-split CV)
    # Variants with no Pfam family for their gene are excluded
    gene_pfam = np.array([pfam_map.get(g) for g in genes])
    has_family = np.array([p is not None for p in gene_pfam])

    X_delta_all = delta
    X_prot_all = X_prot
    X_concat_all = np.concatenate([delta, X_prot], axis=1)  # (n, 1321)

    # Restrict to variants with a Pfam family assignment (same as pilot)
    fam_idx = np.where(has_family)[0]
    X_delta = X_delta_all[fam_idx]
    X_prot_fam = X_prot_all[fam_idx]
    X_concat = X_concat_all[fam_idx]
    y_fam = y[fam_idx]
    genes_fam = genes[fam_idx]
    gene_pfam_fam = gene_pfam[fam_idx]
    groups = gene_pfam_fam  # groups = Pfam family strings

    print(f"  Variants with Pfam family: {len(fam_idx)}/{len(y)}")
    print(f"  Unique families: {len(set(groups.tolist()))}")
    print(
        f"  Class distribution: "
        + ", ".join(f"{c}={int((y_fam==c).sum())}" for c in CLASSES)
    )

    results: dict = {
        "seed": seed,
        "n_variants_with_family": int(len(fam_idx)),
        "n_total_variants": int(len(y)),
        "n_families": int(len(set(groups.tolist()))),
    }

    # ------------------------------------------------------------------
    # V1 — ESM-2 delta only, MLP 1280→256→64→3
    # ------------------------------------------------------------------
    print(f"\n--- V1: ESM-2 delta only (MLP 1280→256→64→3) ---")
    v1_res = run_family_split_mlp(
        X_delta,
        y_fam,
        groups,
        hidden_layer_sizes=(256, 64),
        n_folds=n_folds,
        seed=seed,
        label="V1",
    )
    results["V1_family_split"] = v1_res
    v1_f1 = v1_res.get("macro_f1_mean", float("nan"))
    print(
        f"  V1 family-split macro-F1 = {v1_f1:.4f} ± {v1_res.get('macro_f1_std', float('nan')):.4f}"
    )

    # ------------------------------------------------------------------
    # V2 — Proteome features only
    #
    # The proteome block has missing cells in every feature (the family-residual
    # columns are ~44-61% missing by design), so LightGBM — which consumes NaN
    # natively — is the primary arm: it keeps every gene without fabricating a
    # value. The linear question ("is the signal linear?") is answered by a
    # LogReg restricted to fully-observed genes, paired with a LightGBM run on
    # that same restricted subset so the linear/nonlinear comparison comes from
    # one population rather than confounding model class with gene coverage.
    # ------------------------------------------------------------------
    print(f"\n--- V2: Proteome features only (LightGBM primary; linear check on observed subset) ---")

    v2_lgbm_res = run_family_split_lgbm(
        X_prot_fam,
        y_fam,
        groups,
        n_folds=n_folds,
        seed=seed,
        label="V2-LGBM",
    )
    results["V2_lgbm_family_split"] = v2_lgbm_res

    v2_logreg_obs_res = run_observed_subset_arm(
        X_prot_fam,
        y_fam,
        groups,
        n_folds=n_folds,
        seed=seed,
        label="V2-LR-observed",
        runner=run_logreg_cv,
    )
    results["V2_logreg_observed_subset"] = v2_logreg_obs_res

    # Matched comparator: same rows as the LogReg arm, NaN-native model.
    v2_histgb_obs_res = run_observed_subset_arm(
        X_prot_fam,
        y_fam,
        groups,
        n_folds=n_folds,
        seed=seed,
        label="V2-HistGB-observed",
        runner=run_histgb_cv,
    )
    results["V2_histgb_observed_subset"] = v2_histgb_obs_res

    # Gate 1 is decided on the full-data primary (LightGBM), not on the
    # restricted linear arm, which describes a different gene population.
    # A run with no folds returns {"error": ...} and no "macro_f1_mean"; treating
    # that as F1=0.0 would fabricate a metric and corrupt the Gate-1 decision.
    v2_f1_raw = v2_lgbm_res.get("macro_f1_mean")
    if v2_f1_raw is not None and not np.isnan(v2_f1_raw):
        v2_label, v2_f1 = "LGBM", v2_f1_raw
        print(f"  V2 family-split macro-F1 = {v2_f1:.4f}  ({v2_label}, all genes)")
    else:
        v2_label, v2_f1 = None, float("nan")
        print("  V2 family-split macro-F1 = N/A (primary model produced no folds)")
    results["V2_best_macro_f1_mean"] = v2_f1
    results["V2_best_model"] = v2_label

    # Gate 1
    gate1_pass = v2_f1 >= 0.35
    gate1_msg = (
        f"GATE_PASS (Gate 1): V2 macro-F1 {v2_f1:.4f} >= 0.35"
        if gate1_pass
        else f"GATE_FAIL (Gate 1): V2 macro-F1 {v2_f1:.4f} < 0.35"
    )
    print(f"  {gate1_msg}")
    results["gate_1"] = {"passed": gate1_pass, "v2_f1": v2_f1, "threshold": 0.35}

    if not gate1_pass:
        print("  Stopping after Gate 1 failure (V3/V4 not run for this seed).")
        return results

    # ------------------------------------------------------------------
    # V3 — Concatenated: ESM-2 delta + proteome
    #
    # NaN-native: the concat is a dense 1280-dim delta block glued to the sparse
    # 41-col proteome block, so a handful of missing proteome cells would make
    # the whole 1321-dim row unusable to an MLP. Restricting instead would throw
    # away ~65% of genes whose embeddings are perfectly well observed.
    # ------------------------------------------------------------------
    print(f"\n--- V3: ESM-2 delta + proteome concat (NaN-native gradient boosting) ---")
    v3_family_res = run_family_split_histgb(
        X_concat,
        y_fam,
        groups,
        n_folds=n_folds,
        seed=seed,
        label="V3",
    )
    results["V3_family_split"] = v3_family_res
    v3_f1 = v3_family_res.get("macro_f1_mean", float("nan"))
    print(
        f"  V3 family-split macro-F1 = {v3_f1:.4f} ± {v3_family_res.get('macro_f1_std', float('nan')):.4f}"
    )

    # Gene-split for V3 (leakage diagnostic — uses all variants, not just family-annotated)
    print(f"\n--- V3 gene-split (leakage diagnostic) ---")
    v3_gene_res = run_gene_split_histgb(
        X_concat_all,
        y,
        genes,
        n_folds=n_folds,
        seed=seed,
        label="V3-GS",
        pfam_map=pfam_map,
    )
    results["V3_gene_split"] = v3_gene_res
    v3_gs_f1 = v3_gene_res.get("macro_f1_mean", float("nan"))
    leakage_delta = v3_gs_f1 - v3_f1
    results["V3_leakage_delta"] = (
        float(leakage_delta) if not (np.isnan(v3_gs_f1) or np.isnan(v3_f1)) else None
    )
    print(
        f"  V3 gene-split  macro-F1 = {v3_gs_f1:.4f}   "
        f"leakage delta (gene-split − family-split) = {leakage_delta:+.4f}"
    )

    # Gate 2
    max_v1_v2 = max(v1_f1, v2_f1)
    gate2_threshold = max_v1_v2 + 0.02
    gate2_pass = v3_f1 >= gate2_threshold
    gate2_msg = (
        f"GATE_PASS (Gate 2): V3 macro-F1 {v3_f1:.4f} >= max(V1,V2)+0.02 ({gate2_threshold:.4f})"
        if gate2_pass
        else f"GATE_FAIL (Gate 2): V3 macro-F1 {v3_f1:.4f} < max(V1,V2)+0.02 ({gate2_threshold:.4f})"
    )
    print(f"  {gate2_msg}")
    results["gate_2"] = {
        "passed": gate2_pass,
        "v3_f1": v3_f1,
        "max_v1_v2": max_v1_v2,
        "threshold": gate2_threshold,
    }

    if variants_only:
        print("  --variants-only: skipping V4.")
        results["V4_family_split"] = {"skipped": True, "reason": "--variants-only"}
        return results

    if not gate2_pass:
        print("  Stopping after Gate 2 failure (V4 not run for this seed).")
        results["V4_family_split"] = {"skipped": True, "reason": "Gate 2 failed"}
        return results

    # ------------------------------------------------------------------
    # V4 — Contrastive head on V3 inputs (1321-dim)
    #
    # The triplet-trained projection head and the k-NN that scores it both
    # require real numbers at every coordinate, and there is no NaN-native
    # substitute for a learned embedding. So V4 runs complete-case, on genes
    # whose proteome block is fully observed, with folds recomputed there.
    # V3 is re-run on those exact rows as the comparator: Gate 3 otherwise
    # measures V4-on-a-subset against V3-on-everything, which reads a change in
    # gene population as a method effect.
    # ------------------------------------------------------------------
    print(f"\n--- V4: Contrastive projection head (1321→256→64) + k-NN, observed subset ---")
    v4_observed = observed_rows_mask(X_concat, label="V4")
    n_v4 = int(v4_observed.sum())

    if n_v4 < n_folds:
        print(f"  V4 skipped: only {n_v4} fully-observed genes")
        results["V4_family_split"] = {
            "skipped": True,
            "reason": "too few fully-observed genes",
            "n_observed": n_v4,
            "n_total": int(len(X_concat)),
        }
        return results

    v4_res = run_v4_family_split(
        X_concat[v4_observed],
        y_fam[v4_observed],
        genes_fam[v4_observed],
        gene_pfam_fam[v4_observed],
        groups[v4_observed],
        n_folds=n_folds,
        seed=seed,
        n_epochs=30,
        lr=1e-3,
        max_triplets=2000,
    )
    v4_res["n_observed"] = n_v4
    v4_res["n_total"] = int(len(X_concat))
    v4_res["frac_observed"] = float(n_v4 / len(X_concat))
    results["V4_family_split"] = v4_res
    v4_f1 = v4_res.get("macro_f1_mean", float("nan"))
    print(
        f"  V4 family-split macro-F1 = {v4_f1:.4f} ± {v4_res.get('macro_f1_std', float('nan')):.4f}"
        f"  (n={n_v4}/{len(X_concat)} fully-observed genes)"
    )

    # V3 on V4's exact rows — the like-for-like comparator for Gate 3.
    v3_matched_res = run_family_split_histgb(
        X_concat[v4_observed],
        y_fam[v4_observed],
        groups[v4_observed],
        n_folds=n_folds,
        seed=seed,
        label="V3-matched-to-V4",
    )
    v3_matched_res["n_observed"] = n_v4
    results["V3_family_split_matched_to_V4"] = v3_matched_res
    v3_matched_f1 = v3_matched_res.get("macro_f1_mean", float("nan"))
    print(f"  V3 on the same {n_v4} genes = {v3_matched_f1:.4f} (Gate 3 comparator)")

    # Gate 3 — report only (no hard stop). Compared against V3 on the identical
    # gene subset, not the historical all-genes reference, which was computed on
    # a different population and a different model class.
    gate3_target = v3_matched_f1 + 0.03
    gate3_pass = v4_f1 >= gate3_target
    gate3_msg = (
        f"GATE_PASS (Gate 3): V4 macro-F1 {v4_f1:.4f} >= matched V3 + 0.03 ({gate3_target:.4f})"
        if gate3_pass
        else f"GATE_FAIL (Gate 3): V4 macro-F1 {v4_f1:.4f} < matched V3 + 0.03 ({gate3_target:.4f})"
    )
    print(f"  {gate3_msg}")
    results["gate_3"] = {
        "passed": bool(gate3_pass),
        "v4_f1": v4_f1,
        "v3_matched_f1": v3_matched_f1,
        "target": gate3_target,
        "n_observed": n_v4,
        "comparator": "V3 (NaN-native) on the same fully-observed gene subset",
    }

    return results


# ---------------------------------------------------------------------------
# Summary across seeds
# ---------------------------------------------------------------------------


def aggregate_seeds(all_seed_results: list[dict]) -> dict:
    """Aggregate per-seed results into mean ± std across seeds."""
    summary: dict = {"n_seeds": len(all_seed_results)}

    def get_mean_std(key_mean: str) -> tuple[float | None, float | None]:
        vals = []
        for r in all_seed_results:
            v = nested_get(r, key_mean)
            if v is not None and not np.isnan(float(v)):
                vals.append(float(v))
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    # V1
    m, s = get_mean_std("V1_family_split.macro_f1_mean")
    summary["V1_family_split_macro_f1_mean"] = m
    summary["V1_family_split_macro_f1_std"] = s

    # V2 (flat key, not nested)
    v2_vals = [
        r["V2_best_macro_f1_mean"]
        for r in all_seed_results
        if "V2_best_macro_f1_mean" in r
        and r["V2_best_macro_f1_mean"] is not None
        and not np.isnan(r["V2_best_macro_f1_mean"])
    ]
    summary["V2_best_macro_f1_mean"] = float(np.mean(v2_vals)) if v2_vals else None
    summary["V2_best_macro_f1_std"] = float(np.std(v2_vals)) if v2_vals else None

    m, s = get_mean_std("V2_lgbm_family_split.macro_f1_mean")
    summary["V2_lgbm_macro_f1_mean"] = m
    summary["V2_lgbm_macro_f1_std"] = s

    # V2 linear check and its matched comparator, both on the fully-observed
    # subset only. Reported as a pair because the linear-vs-nonlinear read is
    # only valid between these two; neither is comparable to the all-genes
    # LightGBM arm above.
    for key, out_prefix in [
        ("V2_logreg_observed_subset", "V2_logreg_observed"),
        ("V2_histgb_observed_subset", "V2_histgb_observed"),
    ]:
        m, s = get_mean_std(f"{key}.macro_f1_mean")
        summary[f"{out_prefix}_macro_f1_mean"] = m
        summary[f"{out_prefix}_macro_f1_std"] = s
    obs_fracs = [
        float(r["V2_logreg_observed_subset"]["frac_observed"])
        for r in all_seed_results
        if r.get("V2_logreg_observed_subset", {}).get("frac_observed") is not None
    ]
    summary["V2_observed_subset_frac"] = (
        float(np.mean(obs_fracs)) if obs_fracs else None
    )

    # V3
    m, s = get_mean_std("V3_family_split.macro_f1_mean")
    summary["V3_family_split_macro_f1_mean"] = m
    summary["V3_family_split_macro_f1_std"] = s

    m, s = get_mean_std("V3_gene_split.macro_f1_mean")
    summary["V3_gene_split_macro_f1_mean"] = m
    summary["V3_gene_split_macro_f1_std"] = s

    leakage_vals = [
        float(r["V3_leakage_delta"])
        for r in all_seed_results
        if r.get("V3_leakage_delta") is not None
    ]
    if leakage_vals:
        summary["V3_leakage_delta_mean"] = float(np.mean(leakage_vals))
        summary["V3_leakage_delta_std"] = float(np.std(leakage_vals))

    # V4 (fully-observed subset) and the V3 run on those identical rows. Gate 3
    # is read from this pair; V3_family_split above is on all genes and would
    # confound gene coverage with method.
    m, s = get_mean_std("V4_family_split.macro_f1_mean")
    summary["V4_family_split_macro_f1_mean"] = m
    summary["V4_family_split_macro_f1_std"] = s

    m, s = get_mean_std("V3_family_split_matched_to_V4.macro_f1_mean")
    summary["V3_matched_to_V4_macro_f1_mean"] = m
    summary["V3_matched_to_V4_macro_f1_std"] = s

    v4_fracs = [
        float(r["V4_family_split"]["frac_observed"])
        for r in all_seed_results
        if r.get("V4_family_split", {}).get("frac_observed") is not None
    ]
    summary["V4_observed_subset_frac"] = float(np.mean(v4_fracs)) if v4_fracs else None

    # Per-class AUROC across seeds (V1, V3)
    for variant_key in ["V1_family_split", "V3_family_split", "V4_family_split"]:
        for cls in CLASSES:
            m, s = get_mean_std(f"{variant_key}.auroc_{cls}_mean")
            summary[f"{variant_key}_auroc_{cls}_mean"] = m
            summary[f"{variant_key}_auroc_{cls}_std"] = s

    return summary


def nested_get(d: dict, key: str):
    """Get nested dict value using dot-notation key like 'V1_family_split.macro_f1_mean'."""
    parts = key.split(".", 1)
    if parts[0] not in d:
        return None
    v = d[parts[0]]
    if len(parts) == 1:
        return v
    if isinstance(v, dict):
        return nested_get(v, parts[1])
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 11 Phase 3: ESM-2 + proteome features, V1–V4"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single seed to run (default: run all seeds 0–4)",
    )
    parser.add_argument(
        "--variants-only", action="store_true", help="Skip V4 contrastive head (faster)"
    )
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else list(range(5))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load shared data once
    # ------------------------------------------------------------------
    print("=== Loading data ===")
    variants, labels, genes, delta = load_data()

    print("\n=== Loading Pfam families ===")
    pfam_map = load_pfam()
    n_annotated = sum(1 for g in genes if pfam_map.get(g) is not None)
    print(
        f"Pfam coverage: {n_annotated}/{len(genes)} variants have a family annotation"
    )

    print("\n=== Loading proteome features ===")
    X_prot = load_proteome_features(genes)
    print(f"Proteome feature matrix: {X_prot.shape}, expected (n_variants, 41)")

    # Sanity: concat shape
    X_concat_check = np.concatenate([delta, X_prot], axis=1)
    print(f"Concatenated shape: {X_concat_check.shape}  (expected ~19k × 1313)")

    # ------------------------------------------------------------------
    # Per-seed runs
    # ------------------------------------------------------------------
    all_seed_results = []

    for seed in seeds:
        seed_results = run_seed(
            seed=seed,
            n_folds=args.n_folds,
            variants_only=args.variants_only,
            labels=labels,
            genes=genes,
            delta=delta,
            X_prot=X_prot,
            pfam_map=pfam_map,
        )
        all_seed_results.append(seed_results)

        out_path = OUT_DIR / f"proteome_mechanism_seed{seed}.json"
        out_path.write_text(json.dumps(seed_results, indent=2))
        print(f"\n  Saved: {out_path}")

    # ------------------------------------------------------------------
    # Summary across seeds
    # ------------------------------------------------------------------
    summary = aggregate_seeds(all_seed_results)

    summary_path = OUT_DIR / "proteome_mechanism_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary: {summary_path}")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY (mean ± std across seeds)")
    print("=" * 70)

    def fmt(m, s):
        if m is None:
            return "    N/A   "
        s_str = f"{s:.4f}" if s is not None else "  N/A"
        return f"{m:.4f} ± {s_str}"

    rows = [
        (
            "V1 family-split macro-F1",
            summary.get("V1_family_split_macro_f1_mean"),
            summary.get("V1_family_split_macro_f1_std"),
        ),
        (
            "V2 best family-split macro-F1",
            summary.get("V2_best_macro_f1_mean"),
            summary.get("V2_best_macro_f1_std"),
        ),
        (
            "V3 family-split macro-F1",
            summary.get("V3_family_split_macro_f1_mean"),
            summary.get("V3_family_split_macro_f1_std"),
        ),
        (
            "V3 gene-split macro-F1",
            summary.get("V3_gene_split_macro_f1_mean"),
            summary.get("V3_gene_split_macro_f1_std"),
        ),
        (
            "V4 family-split macro-F1",
            summary.get("V4_family_split_macro_f1_mean"),
            summary.get("V4_family_split_macro_f1_std"),
        ),
    ]
    for name, m, s in rows:
        print(f"  {name:<38}  {fmt(m, s)}")

    if summary.get("V3_leakage_delta_mean") is not None:
        ld = summary["V3_leakage_delta_mean"]
        ld_s = summary.get("V3_leakage_delta_std", 0.0)
        print(f"\n  V3 leakage delta (gene−family)         {ld:+.4f} ± {ld_s:.4f}")

    print("\nPer-class AUROC (family-split, mean across seeds):")
    for vk, label in [
        ("V1_family_split", "V1"),
        ("V3_family_split", "V3"),
        ("V4_family_split", "V4"),
    ]:
        parts = []
        for cls in CLASSES:
            mv = summary.get(f"{vk}_auroc_{cls}_mean")
            sv = summary.get(f"{vk}_auroc_{cls}_std")
            if mv is not None:
                parts.append(f"{cls}={mv:.3f}±{sv:.3f}")
            else:
                parts.append(f"{cls}=N/A")
        print(f"  {label}: " + "  ".join(parts))

    # Gate summary across seeds
    print("\nDecision gates (per seed):")
    for sr in all_seed_results:
        seed = sr["seed"]
        g1 = sr.get("gate_1", {})
        g2 = sr.get("gate_2", {})
        g3 = sr.get("gate_3", {})
        g1_str = ("PASS" if g1.get("passed") else "FAIL") if g1 else "N/A"
        g2_str = ("PASS" if g2.get("passed") else "FAIL") if g2 else "N/A"
        g3_str = ("PASS" if g3.get("passed") else "FAIL") if g3 else "N/A"
        v1_f1 = nested_get(sr, "V1_family_split.macro_f1_mean")
        v2_f1 = sr.get("V2_best_macro_f1_mean")
        v3_f1 = nested_get(sr, "V3_family_split.macro_f1_mean")
        v4_f1 = nested_get(sr, "V4_family_split.macro_f1_mean")
        v1_s = f"{v1_f1:.3f}" if v1_f1 is not None else "N/A"
        v2_s = f"{v2_f1:.3f}" if v2_f1 is not None else "N/A"
        v3_s = f"{v3_f1:.3f}" if v3_f1 is not None else "N/A"
        v4_s = f"{v4_f1:.3f}" if v4_f1 is not None else "N/A"
        print(
            f"  Seed {seed}: V1={v1_s} V2={v2_s} V3={v3_s} V4={v4_s}  "
            f"G1={g1_str} G2={g2_str} G3={g3_str}"
        )

    print("=" * 70)
    print(f"Results directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
