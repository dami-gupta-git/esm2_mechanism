"""Phase 3 modelling: V1-V4 model variants under family-split CV.

V1 ESM-2 delta MLP, V2 proteome LightGBM, V3 concat NaN-native,
V4 contrastive head + kNN. Decision gates between stages.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from esm2_mech.utils.splits import family_split_indices
from esm2_mech.utils.constants import MECHANISM_CLASSES, BOOTSTRAP_N_RESAMPLES, N_SEEDS
from esm2_mech.utils.metrics import (
    aggregate_folds,
    align_proba,
    compute_metrics,
    empty_aggregate_metrics,
)
from esm2_mech.utils.probes import run_mlp_cv, run_logreg_cv, run_histgb_cv
from esm2_mech.utils.bootstrap import attach_mechanism_ci
from esm2_mech.utils.classification import validate_complete_classification_splits
from esm2_mech.utils.data import build_gene_to_row, observed_rows_mask, load_pfam_map
from esm2_mech.utils.seed_aggregation import (
    SEED_STATUS_SKIPPED,
    aggregate_seed_results,
    read_seed_inference,
    seed_result_contract,
)
from esm2_mech.experiments.mechanism.seed_results import aggregate_result_contract
from esm2_mech.utils.io import load_variants_and_delta, write_result_json
from esm2_mech.utils.paths import (
    RESULTS_DIR,
    VALID_VARIANTS_JSON,
    EMB_VALID_VARIANTS_JSON,
    EMB_WT_MEAN,
    EMB_MUT_MEAN,
    GENE_UNIVERSE,
    PFAM_JSON,
    PROTEOME_FEATURES_ALIGNED,
    PROTEOME_FEATURE_COLUMNS_JSON,
)
import functools

print = functools.partial(print, flush=True)

OUT_DIR = RESULTS_DIR

warnings.filterwarnings("ignore")

MERGED_VALID_VARIANTS = VALID_VARIANTS_JSON
MERGED_WT_MEAN = EMB_WT_MEAN
MERGED_MUT_MEAN = EMB_MUT_MEAN

PROTEOME_FEATURES = PROTEOME_FEATURES_ALIGNED
PROTEOME_COLS = PROTEOME_FEATURE_COLUMNS_JSON
# MUST be GENE_UNIVERSE (aligned .npy row order), not GENE_LIST_TSV (longer, differently-ordered superset).
MERGED_GENE_LIST = GENE_UNIVERSE
PFAM_FAMILIES = PFAM_JSON

CLASSES = MECHANISM_CLASSES


def gene_split_indices(
    genes: np.ndarray,
    n_folds: int,
    seed: int,
    pfam_map: dict | None = None,
):
    """Yield (train_idx, test_idx) splitting by gene, optionally family-aware."""
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



def load_data() -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    """Load merged variants, labels, genes, and ESM-2 delta embeddings."""
    variants, labels, genes, delta, _ = load_variants_and_delta(
        MERGED_VALID_VARIANTS, EMB_VALID_VARIANTS_JSON,
        MERGED_WT_MEAN, MERGED_MUT_MEAN
    )
    return variants, labels, genes, delta


def build_gene_to_proteome_row() -> dict[str, int]:
    """{gene_symbol: row_index} for indexing into proteome_features_aligned.npy."""
    return build_gene_to_row(MERGED_GENE_LIST)


def load_proteome_features(genes: np.ndarray) -> np.ndarray:
    """Broadcast gene-level proteome features to variant level."""
    prot_matrix = np.load(PROTEOME_FEATURES).astype(np.float32)  # (2424, 41)
    gene_to_row = build_gene_to_proteome_row()

    n = len(genes)
    n_feats = prot_matrix.shape[1]
    # NaN not 0.0: zero is a plausible real value for pLI/LOEUF/PPI_degree.
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


def _attach_ci(agg: dict, oof: dict | None, compute_ci: bool, n_boot: int, seed: int) -> dict:
    """Attach a cluster-bootstrap CI, resampling whatever unit oof["genes"] holds.

    oof["genes"] is already the cluster array (family id or gene id depending on arm).
    """
    return attach_mechanism_ci(
        agg,
        oof,
        oof["genes"] if oof is not None else None,
        compute_ci=compute_ci,
        n_resamples=n_boot,
        seed=seed,
    )


def run_family_split_mlp(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    hidden_layer_sizes: tuple,
    n_folds: int,
    seed: int,
    label: str,
    compute_ci: bool = True,
    n_boot: int = BOOTSTRAP_N_RESAMPLES,
) -> dict:
    splits = list(family_split_indices(groups, n_folds, seed))
    contract = validate_complete_classification_splits(
        splits, requested_folds=n_folds,
        eligible_rows=np.concatenate([test for _train, test in splits]),
        labels=y, classes=CLASSES, groups=groups, held_out_unit="family",
    )
    agg, oof = run_mlp_cv(
        X, y, splits, CLASSES, contract, hidden=hidden_layer_sizes,
        seed=seed, genes=groups, label=label,
        return_oof=True, compute_per_gene=False,
    )
    return _attach_ci(agg, oof, compute_ci, n_boot, seed)


def run_family_split_logreg(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    seed: int,
    label: str,
    compute_ci: bool = True,
    n_boot: int = BOOTSTRAP_N_RESAMPLES,
) -> dict:
    splits = list(family_split_indices(groups, n_folds, seed))
    contract = validate_complete_classification_splits(
        splits, requested_folds=n_folds,
        eligible_rows=np.concatenate([test for _train, test in splits]),
        labels=y, classes=CLASSES, groups=groups, held_out_unit="family",
    )
    agg, oof = run_logreg_cv(
        X, y, splits, CLASSES, contract, seed=seed, genes=groups,
        label=label, return_oof=True, compute_per_gene=False,
    )
    return _attach_ci(agg, oof, compute_ci, n_boot, seed)


def run_family_split_histgb(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    seed: int,
    label: str,
    compute_ci: bool = True,
    n_boot: int = BOOTSTRAP_N_RESAMPLES,
) -> dict:
    """NaN-native family-split CV for arms that include the proteome block."""
    splits = list(family_split_indices(groups, n_folds, seed))
    contract = validate_complete_classification_splits(
        splits, requested_folds=n_folds,
        eligible_rows=np.concatenate([test for _train, test in splits]),
        labels=y, classes=CLASSES, groups=groups, held_out_unit="family",
    )
    agg, oof = run_histgb_cv(
        X, y, splits, CLASSES, contract, seed=seed, genes=groups,
        label=label, return_oof=True, compute_per_gene=False,
    )
    return _attach_ci(agg, oof, compute_ci, n_boot, seed)


def run_observed_subset_arm(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    seed: int,
    label: str,
    runner,
    compute_ci: bool = True,
    n_boot: int = BOOTSTRAP_N_RESAMPLES,
    **runner_kwargs,
) -> dict:
    """Run runner on fully-observed rows of X, with folds recomputed there.

    # Observed subset is not a random sample; compare only against arms on the same rows.
    """
    observed = observed_rows_mask(X, label=label)
    n_obs = int(observed.sum())
    if n_obs < n_folds:
        return {
            "error": "too few fully-observed rows",
            "n_observed": n_obs,
            "n_total": int(len(X)),
        }
    groups_obs = groups[observed]
    splits = list(family_split_indices(groups_obs, n_folds, seed))
    contract = validate_complete_classification_splits(
        splits, requested_folds=n_folds,
        eligible_rows=np.concatenate([test for _train, test in splits]),
        labels=y[observed], classes=CLASSES, groups=groups_obs,
        held_out_unit="family",
    )
    agg, oof = runner(
        X[observed], y[observed], splits, CLASSES, contract,
        seed=seed, genes=groups_obs, label=label,
        return_oof=True, compute_per_gene=False, **runner_kwargs,
    )
    agg = _attach_ci(agg, oof, compute_ci, n_boot, seed)
    agg["n_observed"] = n_obs
    agg["n_total"] = int(len(X))
    agg["frac_observed"] = float(n_obs / len(X))
    return agg


def run_family_split_lgbm(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    seed: int,
    label: str,
    compute_ci: bool = True,
    n_boot: int = BOOTSTRAP_N_RESAMPLES,
) -> dict:
    """Family-split CV with LightGBM multiclass classifier."""
    import lightgbm as lgb

    splits = list(family_split_indices(groups, n_folds, seed))
    contract = validate_complete_classification_splits(
        splits, requested_folds=n_folds,
        eligible_rows=np.concatenate([test for _train, test in splits]),
        labels=y, classes=CLASSES, groups=groups, held_out_unit="family",
    )
    if contract["status"] != "valid":
        result = empty_aggregate_metrics(
            CLASSES, n_folds, "split_validation_failed"
        )
        result.update(
            {
                "status": "unscorable",
                "classes": list(CLASSES),
                "eligible_rows": contract["eligible_rows"],
                "out_of_fold_rows": 0,
                "split_validation": contract,
            }
        )
        return _attach_ci(result, None, compute_ci, n_boot, seed)

    fold_results = []
    oof_y, oof_proba, oof_groups, oof_rows, oof_folds = [], [], [], [], []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_tr, X_te = X[train_idx].astype(np.float32), X[test_idx].astype(np.float32)
        y_tr, y_te = y[train_idx], y[test_idx]

        counts = {c: int((y_tr == c).sum()) for c in CLASSES}
        class_weight = {c: 1.0 / counts[c] for c in CLASSES}
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
        proba_aligned = align_proba(
            clf.predict_proba(X_te),
            clf.classes_,
            CLASSES,
            allow_missing_classes=False,
        )

        fm = compute_metrics(y_te, pred, proba_aligned, CLASSES)
        fold_results.append(fm)
        oof_y.append(y_te)
        oof_proba.append(proba_aligned)
        oof_groups.append(groups[test_idx])
        oof_rows.append(test_idx)
        oof_folds.append(np.full(len(test_idx), fold_i, dtype=int))
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

    agg = aggregate_folds(fold_results, CLASSES, n_folds)
    agg.update(
        {
            "status": "success",
            "classes": list(CLASSES),
            "eligible_rows": contract["eligible_rows"],
            "out_of_fold_rows": contract["eligible_rows"],
            "split_validation": contract,
        }
    )
    oof = {
        "y_true": np.concatenate(oof_y),
        "proba": np.concatenate(oof_proba),
        "genes": np.concatenate(oof_groups),
        "row_ids": np.concatenate(oof_rows),
        "folds": np.concatenate(oof_folds),
        "classes": list(CLASSES),
    }
    return _attach_ci(agg, oof, compute_ci, n_boot, seed)


def run_gene_split_histgb(
    X: np.ndarray,
    y: np.ndarray,
    genes: np.ndarray,
    n_folds: int,
    seed: int,
    label: str,
    pfam_map: dict | None = None,
    compute_ci: bool = True,
    n_boot: int = BOOTSTRAP_N_RESAMPLES,
) -> dict:
    """NaN-native gene-split CV for leakage diagnostics; CI resamples genes."""
    splits = list(gene_split_indices(genes, n_folds, seed, pfam_map=pfam_map))
    contract = validate_complete_classification_splits(
        splits, requested_folds=n_folds,
        eligible_rows=np.concatenate([test for _train, test in splits]),
        labels=y, classes=CLASSES, groups=genes, held_out_unit="gene",
    )
    agg, oof = run_histgb_cv(
        X, y, splits, CLASSES, contract, seed=seed, genes=genes,
        label=label, return_oof=True
    )
    return _attach_ci(agg, oof, compute_ci, n_boot, seed)


def build_triplets_v4(
    y: np.ndarray,
    gene_pfam: np.ndarray,
    max_triplets: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mine cross-family triplets for V4, capped at max_triplets."""
    rng = np.random.RandomState(seed)
    n = len(y)
    n_classes = len(CLASSES)

    # Local int encoding for dict-of-int-keys bookkeeping; y stays string-typed elsewhere.
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
    """Train projection head 1321->256->64 with TripletMarginLoss."""
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

    # Seed torch before head build: weight init and DataLoader shuffle use global RNG.
    torch.manual_seed(seed)

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
    shuffle_gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=512, shuffle=True, generator=shuffle_gen)

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
) -> tuple[dict, np.ndarray]:
    k_eff = min(k, len(Z_train) - 1)
    knn = KNeighborsClassifier(n_neighbors=k_eff, metric="cosine")
    knn.fit(Z_train, y_train)
    pred = knn.predict(Z_test)

    proba_aligned = align_proba(
        knn.predict_proba(Z_test),
        knn.classes_,
        CLASSES,
        allow_missing_classes=False,
    )

    return compute_metrics(y_test, pred, proba_aligned, CLASSES), proba_aligned


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
    compute_ci: bool = True,
    n_boot: int = BOOTSTRAP_N_RESAMPLES,
) -> dict:
    """V4 contrastive projection head + kNN under family-split CV."""
    splits = list(family_split_indices(groups, n_folds, seed))
    contract = validate_complete_classification_splits(
        splits, requested_folds=n_folds,
        eligible_rows=np.concatenate([test for _train, test in splits]),
        labels=y, classes=CLASSES, groups=groups, held_out_unit="family",
    )
    if contract["status"] != "valid":
        result = empty_aggregate_metrics(
            CLASSES, n_folds, "split_validation_failed"
        )
        result.update(
            {
                "status": "unscorable",
                "classes": list(CLASSES),
                "eligible_rows": contract["eligible_rows"],
                "out_of_fold_rows": 0,
                "split_validation": contract,
            }
        )
        return _attach_ci(result, None, compute_ci, n_boot, seed)
    fold_results = []
    oof_y, oof_proba, oof_groups, oof_rows, oof_folds = [], [], [], [], []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        gene_pfam_tr = gene_pfam[train_idx]

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

        fm, proba_aligned = run_knn_v4(Z_tr, Z_te, y_tr, y_te, k=10)
        fold_results.append(fm)
        oof_y.append(y_te)
        oof_proba.append(proba_aligned)
        oof_groups.append(groups[test_idx])
        oof_rows.append(test_idx)
        oof_folds.append(np.full(len(test_idx), fold_i, dtype=int))
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

    agg = aggregate_folds(fold_results, CLASSES, n_folds)
    agg.update(
        {
            "status": "success",
            "classes": list(CLASSES),
            "eligible_rows": contract["eligible_rows"],
            "out_of_fold_rows": contract["eligible_rows"],
            "split_validation": contract,
        }
    )
    oof = {
        "y_true": np.concatenate(oof_y),
        "proba": np.concatenate(oof_proba),
        "genes": np.concatenate(oof_groups),
        "row_ids": np.concatenate(oof_rows),
        "folds": np.concatenate(oof_folds),
        "classes": list(CLASSES),
    }
    return _attach_ci(agg, oof, compute_ci, n_boot, seed)


def _fmt_ci(agg: dict) -> str:
    ci = agg.get("ci", {}).get("macro_f1")
    if not ci or ci.get("ci_suppressed"):
        return ""
    return f"  CI=[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}] ({ci['n_clusters']} clusters)"


def run_seed(
    seed: int,
    n_folds: int,
    variants_only: bool,
    labels: np.ndarray,
    genes: np.ndarray,
    delta: np.ndarray,
    X_prot: np.ndarray,
    pfam_map: dict[str, str],
    compute_ci: bool = True,
    n_boot: int = BOOTSTRAP_N_RESAMPLES,
) -> dict:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")

    # y must stay string-typed: int-encoded y silently zeroes all class counts and AUROCs.
    y = np.asarray(labels)

    gene_pfam = np.array([pfam_map.get(g) for g in genes])
    has_family = np.array([p is not None for p in gene_pfam])

    X_delta_all = delta
    X_prot_all = X_prot
    X_concat_all = np.concatenate([delta, X_prot], axis=1)  # (n, 1321)

    fam_idx = np.where(has_family)[0]
    X_delta = X_delta_all[fam_idx]
    X_prot_fam = X_prot_all[fam_idx]
    X_concat = X_concat_all[fam_idx]
    y_fam = y[fam_idx]
    genes_fam = genes[fam_idx]
    gene_pfam_fam = gene_pfam[fam_idx]
    groups = gene_pfam_fam

    print(f"  Variants with Pfam family: {len(fam_idx)}/{len(y)}")
    print(f"  Unique families: {len(set(groups.tolist()))}")
    print(
        f"  Class distribution: "
        + ", ".join(f"{c}={int((y_fam==c).sum())}" for c in CLASSES)
    )

    results: dict = {
        **seed_result_contract(seed),
        "n_variants_with_family": int(len(fam_idx)),
        "n_total_variants": int(len(y)),
        "n_families": int(len(set(groups.tolist()))),
    }

    print(f"\n--- V1: ESM-2 delta only (MLP 1280→256→64→3) ---")
    v1_res = run_family_split_mlp(
        X_delta,
        y_fam,
        groups,
        hidden_layer_sizes=(256, 64),
        n_folds=n_folds,
        seed=seed,
        label="V1",
        compute_ci=compute_ci,
        n_boot=n_boot,
    )
    results["V1_family_split"] = v1_res
    v1_f1 = v1_res.get("macro_f1_mean")
    if v1_f1 is None:
        print("  V1 family-split macro-F1 = N/A")
    else:
        print(
            f"  V1 family-split macro-F1 = {v1_f1:.4f} ± {v1_res['macro_f1_std']:.4f}"
            + _fmt_ci(v1_res)
        )

    # LightGBM is primary (NaN-native); LogReg + matched HistGB on observed subset for linearity check.
    print(f"\n--- V2: Proteome features only (LightGBM primary; linear check on observed subset) ---")

    v2_lgbm_res = run_family_split_lgbm(
        X_prot_fam,
        y_fam,
        groups,
        n_folds=n_folds,
        seed=seed,
        label="V2-LGBM",
        compute_ci=compute_ci,
        n_boot=n_boot,
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
        compute_ci=compute_ci,
        n_boot=n_boot,
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
        compute_ci=compute_ci,
        n_boot=n_boot,
    )
    results["V2_histgb_observed_subset"] = v2_histgb_obs_res

    # Gate 1 is decided on the full-data primary (LightGBM), not on the
    # restricted linear arm, which describes a different gene population.
    # A run with no folds returns {"error": ...} and no "macro_f1_mean"; treating
    # that as F1=0.0 would fabricate a metric and corrupt the Gate-1 decision.
    v2_f1_raw = v2_lgbm_res.get("macro_f1_mean")
    if v2_f1_raw is not None and not np.isnan(v2_f1_raw):
        v2_label, v2_f1 = "LGBM", v2_f1_raw
        print(
            f"  V2 family-split macro-F1 = {v2_f1:.4f}  ({v2_label}, all genes)"
            + _fmt_ci(v2_lgbm_res)
        )
    else:
        v2_label, v2_f1 = None, None
        print("  V2 family-split macro-F1 = N/A (primary model produced no folds)")
    results["V2_best_macro_f1_mean"] = v2_f1
    results["V2_best_model"] = v2_label

    # Gate 1
    gate1_pass = v2_f1 >= 0.35 if v2_f1 is not None else None
    gate1_msg = (
        "GATE_UNAVAILABLE (Gate 1): V2 macro-F1 unavailable"
        if gate1_pass is None
        else (
            f"GATE_PASS (Gate 1): V2 macro-F1 {v2_f1:.4f} >= 0.35"
            if gate1_pass
            else f"GATE_FAIL (Gate 1): V2 macro-F1 {v2_f1:.4f} < 0.35"
        )
    )
    print(f"  {gate1_msg}")
    results["gate_1"] = {"passed": gate1_pass, "v2_f1": v2_f1, "threshold": 0.35}

    if not gate1_pass:
        gate1_stop = "unavailable" if gate1_pass is None else "failed"
        print(f"  Stopping after Gate 1 {gate1_stop} (V3/V4 not run for this seed).")
        skipped_block = {
            "status": SEED_STATUS_SKIPPED,
            "skipped": True,
            "reason": (
                "Gate 1 unavailable" if gate1_pass is None else "Gate 1 failed"
            ),
        }
        results["V3_family_split"] = skipped_block
        results["V3_gene_split"] = dict(skipped_block)
        results["V4_family_split"] = dict(skipped_block)
        results["V3_family_split_matched_to_V4"] = dict(skipped_block)
        return results

    # NaN-native: restricting to complete cases would discard ~65% of genes with good embeddings.
    print(f"\n--- V3: ESM-2 delta + proteome concat (NaN-native gradient boosting) ---")
    v3_family_res = run_family_split_histgb(
        X_concat,
        y_fam,
        groups,
        n_folds=n_folds,
        seed=seed,
        label="V3",
        compute_ci=compute_ci,
        n_boot=n_boot,
    )
    results["V3_family_split"] = v3_family_res
    v3_f1 = v3_family_res.get("macro_f1_mean")
    if v3_f1 is None:
        print("  V3 family-split macro-F1 = N/A")
    else:
        print(
            f"  V3 family-split macro-F1 = {v3_f1:.4f} ± {v3_family_res['macro_f1_std']:.4f}"
            + _fmt_ci(v3_family_res)
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
        compute_ci=compute_ci,
        n_boot=n_boot,
    )
    results["V3_gene_split"] = v3_gene_res
    v3_gs_f1 = v3_gene_res.get("macro_f1_mean")
    leakage_delta = (
        v3_gs_f1 - v3_f1
        if v3_gs_f1 is not None and v3_f1 is not None
        else None
    )
    results["V3_leakage_delta"] = leakage_delta
    if leakage_delta is None:
        print("  V3 gene-split macro-F1 or leakage delta = N/A")
    else:
        print(
            f"  V3 gene-split  macro-F1 = {v3_gs_f1:.4f}   "
            f"leakage delta (gene-split − family-split) = {leakage_delta:+.4f}"
        )

    # Gate 2
    gate2_inputs = (v1_f1, v2_f1, v3_f1)
    max_v1_v2 = (
        max(v1_f1, v2_f1) if all(value is not None for value in gate2_inputs) else None
    )
    gate2_threshold = None if max_v1_v2 is None else max_v1_v2 + 0.02
    gate2_pass = None if gate2_threshold is None else v3_f1 >= gate2_threshold
    gate2_msg = (
        "GATE_UNAVAILABLE (Gate 2): a required macro-F1 is unavailable"
        if gate2_pass is None
        else (
            f"GATE_PASS (Gate 2): V3 macro-F1 {v3_f1:.4f} >= max(V1,V2)+0.02 ({gate2_threshold:.4f})"
            if gate2_pass
            else f"GATE_FAIL (Gate 2): V3 macro-F1 {v3_f1:.4f} < max(V1,V2)+0.02 ({gate2_threshold:.4f})"
        )
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
        return results

    if not gate2_pass:
        print("  Stopping after Gate 2 failure (V4 not run for this seed).")
        skipped_block = {
            "status": SEED_STATUS_SKIPPED,
            "skipped": True,
            "reason": (
                "Gate 2 unavailable" if gate2_pass is None else "Gate 2 failed"
            ),
        }
        # V3's matched re-run exists only to compare with V4 on V4's rows, so it
        # is skipped for the same reason and says so rather than going missing.
        results["V4_family_split"] = skipped_block
        results["V3_family_split_matched_to_V4"] = dict(skipped_block)
        return results

    # V4 needs complete-case (no NaN-native substitute for learned embeddings).
    # V3 re-run on V4's rows as comparator so Gate 3 reads method, not coverage.
    print(f"\n--- V4: Contrastive projection head (1321→256→64) + k-NN, observed subset ---")
    v4_observed = observed_rows_mask(X_concat, label="V4")
    n_v4 = int(v4_observed.sum())

    if n_v4 < n_folds:
        print(f"  V4 skipped: only {n_v4} fully-observed genes")
        skipped_block = {
            "status": SEED_STATUS_SKIPPED,
            "skipped": True,
            "reason": "too few fully-observed genes",
            "n_observed": n_v4,
            "n_total": int(len(X_concat)),
        }
        results["V4_family_split"] = skipped_block
        results["V3_family_split_matched_to_V4"] = dict(skipped_block)
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
        compute_ci=compute_ci,
        n_boot=n_boot,
    )
    v4_res["n_observed"] = n_v4
    v4_res["n_total"] = int(len(X_concat))
    v4_res["frac_observed"] = float(n_v4 / len(X_concat))
    results["V4_family_split"] = v4_res
    v4_f1 = v4_res.get("macro_f1_mean")
    if v4_f1 is None:
        print("  V4 family-split macro-F1 = N/A")
    else:
        print(
            f"  V4 family-split macro-F1 = {v4_f1:.4f} ± {v4_res['macro_f1_std']:.4f}"
            f"  (n={n_v4}/{len(X_concat)} fully-observed genes)"
            + _fmt_ci(v4_res)
        )

    # V3 on V4's exact rows for like-for-like Gate 3 comparison.
    v3_matched_res = run_family_split_histgb(
        X_concat[v4_observed],
        y_fam[v4_observed],
        groups[v4_observed],
        n_folds=n_folds,
        seed=seed,
        label="V3-matched-to-V4",
        compute_ci=compute_ci,
        n_boot=n_boot,
    )
    v3_matched_res["n_observed"] = n_v4
    results["V3_family_split_matched_to_V4"] = v3_matched_res
    v3_matched_f1 = v3_matched_res.get("macro_f1_mean")
    if v3_matched_f1 is None:
        print(f"  V3 on the same {n_v4} genes = N/A (Gate 3 comparator)")
    else:
        print(
            f"  V3 on the same {n_v4} genes = {v3_matched_f1:.4f} (Gate 3 comparator)"
            + _fmt_ci(v3_matched_res)
        )

    # Gate 3 — report only; compared against V3 on the identical gene subset.
    gate3_target = None if v3_matched_f1 is None else v3_matched_f1 + 0.03
    gate3_pass = (
        v4_f1 >= gate3_target
        if v4_f1 is not None and gate3_target is not None
        else None
    )
    gate3_msg = (
        "GATE_UNAVAILABLE (Gate 3): a required macro-F1 is unavailable"
        if gate3_pass is None
        else (
            f"GATE_PASS (Gate 3): V4 macro-F1 {v4_f1:.4f} >= matched V3 + 0.03 ({gate3_target:.4f})"
            if gate3_pass
            else f"GATE_FAIL (Gate 3): V4 macro-F1 {v4_f1:.4f} < matched V3 + 0.03 ({gate3_target:.4f})"
        )
    )
    print(f"  {gate3_msg}")
    results["gate_3"] = {
        "passed": gate3_pass,
        "v4_f1": v4_f1,
        "v3_matched_f1": v3_matched_f1,
        "target": gate3_target,
        "n_observed": n_v4,
        "comparator": "V3 (NaN-native) on the same fully-observed gene subset",
    }

    return results


def aggregate_seeds(
    all_seed_results: list[dict], requested_seeds, *, include_v4: bool = True
) -> dict:
    """Aggregate every retained per-seed point estimate through the shared contract."""
    summary: dict = {
        **aggregate_result_contract(),
        "requested_seeds": list(requested_seeds),
    }

    scalar_paths = {
        "V1_family_split_macro_f1": "V1_family_split.macro_f1_mean",
        "V2_best_macro_f1": "V2_best_macro_f1_mean",
        "V2_lgbm_macro_f1": "V2_lgbm_family_split.macro_f1_mean",
        "V2_logreg_observed_macro_f1": "V2_logreg_observed_subset.macro_f1_mean",
        "V2_histgb_observed_macro_f1": "V2_histgb_observed_subset.macro_f1_mean",
        "V2_observed_subset_frac": "V2_logreg_observed_subset.frac_observed",
        "V3_family_split_macro_f1": "V3_family_split.macro_f1_mean",
        "V3_gene_split_macro_f1": "V3_gene_split.macro_f1_mean",
        "V3_leakage_delta": "V3_leakage_delta",
    }
    if include_v4:
        scalar_paths.update(
            {
                "V4_family_split_macro_f1": "V4_family_split.macro_f1_mean",
                "V3_matched_to_V4_macro_f1": (
                    "V3_family_split_matched_to_V4.macro_f1_mean"
                ),
                "V4_observed_subset_frac": "V4_family_split.frac_observed",
            }
        )
    else:
        summary["arm_exclusions"] = {
            "V4": {"reason": "excluded_by_variants_only_option"}
        }

    def metric_status(result, metric_path):
        first = metric_path.split(".", 1)[0]
        if first == "V2_best_macro_f1_mean":
            block = result["V2_lgbm_family_split"]
        elif first == "V3_leakage_delta":
            block = result["V3_family_split"]
        else:
            block = result[first]
        return block["status"]

    for output_name, path in scalar_paths.items():
        summary[f"{output_name}_seed_aggregate"] = aggregate_seed_results(
            requested_seeds,
            all_seed_results,
            lambda result, metric_path=path: nested_get(result, metric_path),
            status=lambda result, metric_path=path: metric_status(
                result, metric_path
            ),
        ).to_dict()

    variant_keys = ["V1_family_split", "V3_family_split"]
    if include_v4:
        variant_keys.append("V4_family_split")
    for variant_key in variant_keys:
        for cls in CLASSES:
            path = f"{variant_key}.auroc_{cls}_mean"
            summary[f"{variant_key}_auroc_{cls}_seed_aggregate"] = (
                aggregate_seed_results(
                    requested_seeds,
                    all_seed_results,
                    lambda result, metric_path=path: nested_get(result, metric_path),
                    status=lambda result, metric_path=path: metric_status(
                        result, metric_path
                    ),
                ).to_dict()
            )

    return summary


def nested_get(d: dict, key: str):
    """Get nested dict value using dot-notation key."""
    parts = key.split(".", 1)
    if parts[0] not in d:
        return None
    v = d[parts[0]]
    if len(parts) == 1:
        return v
    if isinstance(v, dict):
        return nested_get(v, parts[1])
    return None


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
    parser.add_argument("--no_ci", action="store_true", help="skip cluster-bootstrap CIs")
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else list(range(N_SEEDS))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading data ===")
    variants, labels, genes, delta = load_data()

    print("\n=== Loading Pfam families ===")
    pfam_map = load_pfam_map(PFAM_FAMILIES)
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
            compute_ci=not args.no_ci,
            n_boot=args.n_boot,
        )
        all_seed_results.append(seed_results)

        out_path = OUT_DIR / f"proteome_mechanism_seed{seed}.json"
        write_result_json(out_path, seed_results, seeds=[seed], indent=2)
        print(f"\n  Saved: {out_path}")

    summary = aggregate_seeds(
        all_seed_results, seeds, include_v4=not args.variants_only
    )

    summary_path = OUT_DIR / "proteome_mechanism_summary.json"
    write_result_json(summary_path, summary, seeds=seeds, indent=2)
    print(f"\nSaved summary: {summary_path}")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY (mean ± std across seeds)")
    print("=" * 70)

    def fmt(aggregate):
        metric = read_seed_inference(aggregate)
        if not metric.available:
            return "    N/A   "
        return f"{metric.value:.4f} ± {metric.spread:.4f}"

    rows = [
        (
            "V1 family-split macro-F1",
            summary.get("V1_family_split_macro_f1_seed_aggregate", {}),
        ),
        (
            "V2 best family-split macro-F1",
            summary.get("V2_best_macro_f1_seed_aggregate", {}),
        ),
        (
            "V3 family-split macro-F1",
            summary.get("V3_family_split_macro_f1_seed_aggregate", {}),
        ),
        (
            "V3 gene-split macro-F1",
            summary.get("V3_gene_split_macro_f1_seed_aggregate", {}),
        ),
    ]
    if not args.variants_only:
        rows.append(
            (
                "V4 family-split macro-F1",
                summary.get("V4_family_split_macro_f1_seed_aggregate", {}),
            )
        )
    for name, aggregate in rows:
        print(f"  {name:<38}  {fmt(aggregate)}")

    leakage = read_seed_inference(summary.get("V3_leakage_delta_seed_aggregate", {}))
    if leakage.available:
        print(
            f"\n  V3 leakage delta (gene−family)         "
            f"{leakage.value:+.4f} ± {leakage.spread:.4f}"
        )

    print("\nPer-class AUROC (family-split, mean across seeds):")
    auroc_rows = [
        ("V1_family_split", "V1"),
        ("V3_family_split", "V3"),
    ]
    if not args.variants_only:
        auroc_rows.append(("V4_family_split", "V4"))
    for vk, label in auroc_rows:
        parts = []
        for cls in CLASSES:
            metric = read_seed_inference(
                summary.get(f"{vk}_auroc_{cls}_seed_aggregate", {})
            )
            if metric.available:
                parts.append(f"{cls}={metric.value:.3f}±{metric.spread:.3f}")
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
