"""
Contrastive metric learning on ESM-2 delta embeddings for mechanism classification.

Trains a small MLP projection head with supervised contrastive (SupCon) loss where:
  - Positives: same-mechanism variants from DIFFERENT Pfam families
  - Within-family pairs excluded from positives (prevents leaking family identity)
  - Negatives: different-mechanism variants

Then evaluates k-NN classification in the learned 64-d space under family-split CV,
compared to k-NN in raw 1280-d delta space.

Key question: if we explicitly force the projection to be family-invariant, does
mechanism signal emerge that the standard MLP (which has no such constraint) could not find?

Runs the merged dataset (VALID_VARIANTS_JSON) across seeds 0-4 and pools the
per-seed files into an across-seed headline (mean ± std ACROSS seeds), mirroring
classify_by_mechanism. Per-seed files: contrastive_results_seed{seed}.json under
RESULTS_DIR; pooled into contrastive_aggregate.json.

Usage:
    python contrastive_mechanism.py            # all 5 seeds + aggregate
    python contrastive_mechanism.py --seed 2   # single seed, no aggregation
"""

import argparse
import json
import os
import warnings
from collections import Counter

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from esm2_mech.utils.io import atomic_write_json
from esm2_mech.utils.splits import gene_split_cv, family_split_cv
from esm2_mech.utils.paths import (
    CONTRASTIVE_AGGREGATE_JSON,
    CONTRASTIVE_RESULTS_DIR,
    EMB_MUT_MEAN,
    EMB_WT_MEAN,
    MECHANISM_AGGREGATE_JSON,
    PFAM_JSON,
    VALID_VARIANTS_JSON,
)
from esm2_mech.utils.constants import (
    CONTRASTIVE_SEED_RESULT_GLOB,
    DELTA_MEAN_FEATURE,
    DN,
    GOF,
    MECHANISM_CLASSES,
    contrastive_seed_result_filename,
)
from esm2_mech.utils.seed_aggregation import (
    FAMILY_SPLIT,
    aggregate_across_seeds,
    load_seed_files,
    print_table,
    read_across_seed_metric,
)
import functools

print = functools.partial(print, flush=True)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data():
    """Load the merged Gerasimavicius + G2P variants and their delta embeddings."""
    with open(VALID_VARIANTS_JSON) as f:
        variants = json.load(f)
    print(f"Loaded {len(variants)} merged variants (Gerasimavicius + G2P)")

    labels = np.array([v["label_3class"] for v in variants])
    genes = np.array([v["gene"] for v in variants])
    n = len(variants)

    print(f"Class distribution: {dict(Counter(labels))}")
    print(f"Unique genes: {len(set(genes))}")

    wt_mean = np.load(EMB_WT_MEAN)[:n]
    mut_mean = np.load(EMB_MUT_MEAN)[:n]
    delta_mean = (mut_mean - wt_mean).astype(np.float32)

    print(f"Delta embeddings: {delta_mean.shape}")
    return variants, labels, genes, delta_mean


def load_pfam(genes):
    with open(PFAM_JSON) as f:
        pfam_map = json.load(f)
    gene_pfam = np.array([pfam_map.get(g) for g in genes])
    n_annotated = sum(1 for p in gene_pfam if p is not None)
    print(f"Pfam coverage: {n_annotated}/{len(genes)} genes")
    return gene_pfam, pfam_map


# ---------------------------------------------------------------------------
# Supervised contrastive loss
# ---------------------------------------------------------------------------


def build_cross_family_pairs(labels, gene_pfam, le, max_pairs_per_anchor=10, seed=42):
    """
    Vectorised triplet construction. For each anchor, sample positives from
    same mechanism + different Pfam family; negatives from different mechanism.
    Within-family pairs are excluded from positives.
    """
    rng = np.random.RandomState(seed)
    y = le.transform(labels)
    n = len(labels)
    n_classes = len(le.classes_)

    # Encode family strings to integers for fast comparison
    unique_fams = list({f for f in gene_pfam if f is not None})
    fam_to_int = {f: i for i, f in enumerate(unique_fams)}
    fam_int = np.array([fam_to_int.get(f, -1) for f in gene_pfam], dtype=np.int32)

    # Pre-build index arrays per (class, family) — avoids repeated scanning
    by_mech = {c: np.where(y == c)[0] for c in range(n_classes)}
    # For each class, indices grouped by family: dict[(class, fam_int)] -> array
    by_mech_fam = {}
    for c in range(n_classes):
        for idx in by_mech[c]:
            key = (c, int(fam_int[idx]))
            by_mech_fam.setdefault(key, []).append(idx)
    by_mech_fam = {k: np.array(v) for k, v in by_mech_fam.items()}

    # Pre-build negative pool per class (all variants of other classes)
    neg_by_class = {}
    for c in range(n_classes):
        neg_by_class[c] = np.concatenate(
            [by_mech[o] for o in range(n_classes) if o != c]
        )

    # For each class, build a flat positive pool excluding each family —
    # do this per unique (class, family) combination rather than per anchor
    # so O(unique_combos) not O(n_variants)
    by_class_arr = {c: np.array(by_mech[c]) for c in range(n_classes)}

    # Map each variant to its per-class positive pool (same mech, diff family)
    # We group anchors by (class, family) and assign the same pool to all in group
    anchor_list, pos_list, neg_list = [], [], []

    unique_combos = set()
    for i in range(n):
        unique_combos.add((int(y[i]), int(fam_int[i])))

    # For each unique (class, fam) build the cross-family positive pool once
    combo_pos_pool = {}
    for c, fam in unique_combos:
        # All variants of class c whose family != fam
        class_idxs = by_class_arr[c]
        class_fams = fam_int[class_idxs]
        cross_fam_mask = (class_fams != fam) & (class_fams != -1)
        pool = class_idxs[cross_fam_mask]
        combo_pos_pool[(c, fam)] = pool

    for anchor_i in range(n):
        c = int(y[anchor_i])
        fam = int(fam_int[anchor_i])
        pos_pool = combo_pos_pool.get((c, fam), np.array([], dtype=np.int64))
        neg_pool = neg_by_class[c]

        if len(pos_pool) == 0 or len(neg_pool) == 0:
            continue

        n_pairs = min(max_pairs_per_anchor, len(pos_pool), len(neg_pool))
        pos_sample = pos_pool[rng.randint(0, len(pos_pool), n_pairs)]
        neg_sample = neg_pool[rng.randint(0, len(neg_pool), n_pairs)]

        anchor_list.append(np.full(n_pairs, anchor_i, dtype=np.int64))
        pos_list.append(pos_sample)
        neg_list.append(neg_sample)

    if not anchor_list:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )

    anchors = np.concatenate(anchor_list)
    positives = np.concatenate(pos_list)
    negatives = np.concatenate(neg_list)

    print(
        f"Built {len(anchors)} triplets "
        f"({len(set(anchors.tolist()))} unique anchors, cross-family positives only)"
    )
    return anchors, positives, negatives


# ---------------------------------------------------------------------------
# Contrastive projection head (PyTorch)
# ---------------------------------------------------------------------------


def train_projection_head(
    X_train,
    labels_train,
    gene_pfam_train,
    le,
    hidden=(256, 64),
    lr=1e-3,
    max_epochs=80,
    patience=12,
    batch_size=512,
    margin=1.0,
    seed=42,
    progress_label="",
    log_every=10,
):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # Normalize
    mu = X_train.mean(0)
    std = X_train.std(0) + 1e-8
    X_norm = ((X_train - mu) / std).astype(np.float32)

    # Build triplets from training data
    anchors, positives, negatives = build_cross_family_pairs(
        labels_train, gene_pfam_train, le, max_pairs_per_anchor=8, seed=seed
    )

    if len(anchors) < 50:
        # The experiment is defined by cross-family-only positives (see module
        # docstring). Silently substituting all-family positives would change
        # what is being measured, so refuse rather than degrade quietly.
        raise ValueError(
            f"Only {len(anchors)} cross-family triplets available (need >= 50); "
            "cannot train the family-invariant projection head on this fold."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build MLP projection head
    layers = []
    prev = X_norm.shape[1]
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
        prev = h
    proj = nn.Sequential(*layers).to(device)

    optimizer = torch.optim.Adam(proj.parameters(), lr=lr, weight_decay=1e-4)
    triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    X_t = torch.tensor(X_norm, dtype=torch.float32)

    # Validation: hold out ~15% of triplets
    n_val = min(max(10, len(anchors) // 7), len(anchors) - 1)
    rng_val = np.random.RandomState(seed + 99)
    val_idx = rng_val.choice(len(anchors), size=n_val, replace=False)
    train_idx_mask = np.ones(len(anchors), dtype=bool)
    train_idx_mask[val_idx] = False

    anc_tr = anchors[train_idx_mask]
    pos_tr = positives[train_idx_mask]
    neg_tr = negatives[train_idx_mask]
    anc_val = anchors[val_idx]
    pos_val = positives[val_idx]
    neg_val = negatives[val_idx]

    ds = TensorDataset(torch.tensor(anc_tr), torch.tensor(pos_tr), torch.tensor(neg_tr))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_loss = float("inf")
    patience_count = 0
    best_state = None

    print(
        f"    training projection head{(' ' + progress_label) if progress_label else ''}: "
        f"{len(anc_tr)} train / {len(anc_val)} val triplets, "
        f"max_epochs={max_epochs} patience={patience} batch_size={batch_size} "
        f"on {device.type}"
    )

    # Count of epochs actually run (0 if max_epochs == 0), so the return below
    # is well-defined even when the loop body never executes.
    epochs_run = 0
    for epoch in range(max_epochs):
        epochs_run = epoch + 1
        proj.train()
        for anc_b, pos_b, neg_b in loader:
            optimizer.zero_grad()
            z_a = proj(X_t[anc_b].to(device))
            z_p = proj(X_t[pos_b].to(device))
            z_n = proj(X_t[neg_b].to(device))
            loss = triplet_loss(z_a, z_p, z_n)
            loss.backward()
            optimizer.step()

        proj.eval()
        with torch.no_grad():
            z_a = proj(X_t[anc_val].to(device))
            z_p = proj(X_t[pos_val].to(device))
            z_n = proj(X_t[neg_val].to(device))
            val_loss = triplet_loss(z_a, z_p, z_n).item()

        improved = val_loss < best_loss - 1e-4
        if improved:
            best_loss = val_loss
            patience_count = 0
            best_state = {k: v.clone() for k, v in proj.state_dict().items()}
        else:
            patience_count += 1

        # Heartbeat so the training loop is not silent for the bulk of each fold:
        # log on the first epoch, every log_every epochs, and the last epoch.
        if epoch == 0 or epochs_run % log_every == 0 or patience_count >= patience:
            print(
                f"      epoch {epochs_run}/{max_epochs}  "
                f"val_loss={val_loss:.4f}  best={best_loss:.4f}  "
                f"patience={patience_count}/{patience}"
            )

        if patience_count >= patience:
            print(f"      early stop at epoch {epochs_run} (best val_loss={best_loss:.4f})")
            break

    if best_state is not None:
        proj.load_state_dict(best_state)

    proj.eval()
    with torch.no_grad():
        Z = proj(X_t.to(device)).cpu().numpy()

    return proj, Z, mu, std, epochs_run


def project_test(proj, X_test, mu, std):
    import torch

    X_norm = ((X_test - mu) / std).astype(np.float32)
    device = next(proj.parameters()).device
    proj.eval()
    with torch.no_grad():
        Z = proj(torch.tensor(X_norm).to(device)).cpu().numpy()
    return Z


# ---------------------------------------------------------------------------
# k-NN evaluation
# ---------------------------------------------------------------------------


def run_knn(Z_train, Z_test, y_train, y_test, le, k=10):
    # Clamp k to training set size
    k_eff = min(k, len(Z_train) - 1)
    knn = KNeighborsClassifier(n_neighbors=k_eff, metric="cosine")
    knn.fit(Z_train, y_train)
    pred = knn.predict(Z_test)

    # Build full probability matrix aligned to le.classes_ (string labels)
    raw_proba = knn.predict_proba(Z_test)  # shape (n_test, n_train_classes)
    # knn.classes_ are integer encoded; decode back to string class names
    train_cls_int = list(knn.classes_)
    all_classes = list(le.classes_)  # string names in canonical order
    proba = np.zeros((len(Z_test), len(all_classes)), dtype=np.float32)
    for train_i, cls_int in enumerate(train_cls_int):
        cls_str = le.classes_[cls_int]
        all_i = all_classes.index(cls_str)
        proba[:, all_i] = raw_proba[:, train_i]

    fm = {"macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0))}
    # A class whose AUROC is undefined on this fold (absent from test, or all-equal
    # neighbour votes) is recorded by name so the caller can see that this fold's
    # per-class metric rests on fewer folds than n_folds — never silently dropped.
    auroc_skipped = {}
    for all_i, cls_str in enumerate(all_classes):
        cls_int = le.transform([cls_str])[0]
        y_bin = (y_test == cls_int).astype(int)
        if y_bin.sum() == 0 or (1 - y_bin).sum() == 0:
            auroc_skipped[cls_str] = "class_absent_in_test"
        elif proba[:, all_i].std() == 0:
            auroc_skipped[cls_str] = "constant_proba"
        else:
            fm[f"auroc_{cls_str}"] = float(roc_auc_score(y_bin, proba[:, all_i]))
    if auroc_skipped:
        fm["auroc_skipped"] = auroc_skipped
    return fm


# ---------------------------------------------------------------------------
# Main CV loop
# ---------------------------------------------------------------------------


def run_cv(
    X,
    labels,
    genes,
    gene_pfam,
    pfam_map,
    le,
    splits,
    split_name,
    hidden=(256, 64),
    seed=42,
    batch_size=512,
):
    y = le.transform(labels)
    fold_results_contrastive = []
    fold_results_raw_knn = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        labels_tr = labels[train_idx]
        gene_pfam_tr = gene_pfam[train_idx]

        # fold_i is the split index (position in `splits`); it advances even when
        # a split is skipped, so the seed offset (seed + fold_i) stays stable. The
        # number of folds that actually contribute is len(fold_results_*), which
        # can be < len(splits) — reported as n_folds in the aggregate.
        if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
            print(f"  Split {fold_i+1}/{len(splits)}: skipped (missing class)")
            continue

        print(
            f"\n  Split {fold_i+1}/{len(splits)} [{split_name}]  "
            f"(completed so far: {len(fold_results_contrastive)})  "
            f"train={len(train_idx)} test={len(test_idx)}"
        )
        print(f"    train classes: {dict(Counter(labels_tr))}")

        # --- Raw k-NN baseline (no learning) ---
        # Normalize on train stats
        mu_raw = X_tr.mean(0)
        std_raw = X_tr.std(0) + 1e-8
        Z_tr_raw = (X_tr - mu_raw) / std_raw
        Z_te_raw = (X_te - mu_raw) / std_raw
        raw_fm = run_knn(Z_tr_raw, Z_te_raw, y_tr, y_te, le, k=10)
        fold_results_raw_knn.append(raw_fm)
        print(
            f"    raw k-NN:      macro_f1={raw_fm['macro_f1']:.3f}  "
            f"{GOF}={raw_fm.get(f'auroc_{GOF}', float('nan')):.3f}  "
            f"{DN}={raw_fm.get(f'auroc_{DN}', float('nan')):.3f}"
        )

        # --- Contrastive projection head ---
        proj, Z_tr_proj, mu, std, epochs = train_projection_head(
            X_tr,
            labels_tr,
            gene_pfam_tr,
            le,
            hidden=hidden,
            seed=seed + fold_i,
            batch_size=batch_size,
            progress_label=f"[{split_name} split {fold_i+1}/{len(splits)}]",
        )
        Z_te_proj = project_test(proj, X_te, mu, std)

        cont_fm = run_knn(Z_tr_proj, Z_te_proj, y_tr, y_te, le, k=10)
        fold_results_contrastive.append(cont_fm)
        print(
            f"    contrastive:   macro_f1={cont_fm['macro_f1']:.3f}  "
            f"{GOF}={cont_fm.get(f'auroc_{GOF}', float('nan')):.3f}  "
            f"{DN}={cont_fm.get(f'auroc_{DN}', float('nan')):.3f}  "
            f"(epochs={epochs})"
        )

    def agg(fold_list):
        if not fold_list:
            return {"error": "no folds"}
        # Only aggregate float-valued per-fold metrics (macro_f1, per-class AUROC).
        # Restricting to float (not int/bool) means a future count or boolean
        # per-fold field is never meaned into a _mean/_std, and the np.isnan guard
        # below only ever sees floats. The "auroc_skipped" bookkeeping dict is
        # pooled separately so the caller can see how many folds each per-class
        # AUROC actually rests on.
        metric_keys = set()
        for fold in fold_list:
            for key, value in fold.items():
                if type(value) is float:
                    metric_keys.add(key)
        out = {}
        for key in metric_keys:
            vals = [
                fold[key]
                for fold in fold_list
                if key in fold and not np.isnan(fold[key])
            ]
            if vals:
                out[f"{key}_mean"] = float(np.mean(vals))
                out[f"{key}_std"] = float(np.std(vals))
                # Count of folds contributing to this metric (may be < n_folds
                # when a per-class AUROC was undefined on some folds).
                out[f"{key}_n_folds"] = len(vals)
        # Pool per-class AUROC skip reasons across folds, counted by reason.
        skip_counts = Counter()
        for fold in fold_list:
            for cls_str, reason in fold.get("auroc_skipped", {}).items():
                skip_counts[f"auroc_{cls_str}:{reason}"] += 1
        if skip_counts:
            out["auroc_skipped_counts"] = dict(skip_counts)
            print(f"    auroc skipped (by class:reason): {dict(skip_counts)}")
        out["n_folds"] = len(fold_list)
        return out

    return agg(fold_results_contrastive), agg(fold_results_raw_knn)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(data, out_dir, seed, n_folds=5, proj_dim=64, batch_size=512):
    """Run gene-split and family-split contrastive CV for one seed.

    `data` is the preloaded dict from load_all_data() (loaded once and shared
    across seeds so the embeddings are not re-read five times). Writes one
    per-seed JSON to out_dir and returns the results dict.
    """
    labels = data["labels"]
    genes = data["genes"]
    delta_mean = data["delta_mean"]
    gene_pfam = data["gene_pfam"]
    pfam_map = data["pfam_map"]
    le = data["le"]

    np.random.seed(seed)

    print("\n=== Building CV splits ===")
    gene_splits = gene_split_cv(genes, n_folds=n_folds, seed=seed)
    fam_splits = family_split_cv(genes, pfam_map, n_folds=n_folds, seed=seed)
    print(f"Gene-split: {len(gene_splits)} folds | Family-split: {len(fam_splits)} folds")

    hidden = (256, proj_dim)

    print("\n\n" + "=" * 60)
    print(f"GENE-SPLIT CV (seed {seed})")
    print("=" * 60)
    gene_cont, gene_raw = run_cv(
        delta_mean, labels, genes, gene_pfam, pfam_map, le, gene_splits,
        "gene-split", hidden=hidden, seed=seed, batch_size=batch_size,
    )

    print("\n\n" + "=" * 60)
    print(f"FAMILY-SPLIT CV (seed {seed})")
    print("=" * 60)
    fam_cont, fam_raw = run_cv(
        delta_mean, labels, genes, gene_pfam, pfam_map, le, fam_splits,
        "family-split", hidden=hidden, seed=seed, batch_size=batch_size,
    )

    results = {
        "description": (
            "Contrastive metric learning with cross-family-only positives. "
            "Positives: same mechanism, different Pfam family. "
            "Negatives: different mechanism. "
            "Within-family pairs excluded from positives. "
            f"Evaluated by k-NN (k=10, cosine) in projected {proj_dim}-d space."
        ),
        "architecture": f"1280 -> 256 -> {proj_dim} (TripletMarginLoss)",
        "seed": seed,
        "n_folds": n_folds,
        "gene_split": {
            "contrastive_knn": gene_cont,
            "raw_knn_baseline": gene_raw,
        },
        "family_split": {
            "contrastive_knn": fam_cont,
            "raw_knn_baseline": fam_raw,
        },
    }

    # Per-seed headline summary
    print("\n\n" + "=" * 60)
    print(f"SEED {seed} HEADLINE SUMMARY")
    print("=" * 60)
    for split_name, split_key in [
        ("Gene-split", "gene_split"),
        ("Family-split", "family_split"),
    ]:
        cont = results[split_key]["contrastive_knn"]
        raw = results[split_key]["raw_knn_baseline"]

        def auroc_str(metrics):
            return "".join(
                f"  {cls}={metrics.get(f'auroc_{cls}_mean', float('nan')):.3f}"
                for cls in MECHANISM_CLASSES
            )

        print(f"\n{split_name}:")
        print(
            f"  Contrastive k-NN:  macro_f1={cont.get('macro_f1_mean', float('nan')):.3f} "
            f"± {cont.get('macro_f1_std', float('nan')):.3f}" + auroc_str(cont)
        )
        print(
            f"  Raw k-NN baseline: macro_f1={raw.get('macro_f1_mean', float('nan')):.3f} "
            f"± {raw.get('macro_f1_std', float('nan')):.3f}" + auroc_str(raw)
        )
        delta_f1 = cont.get("macro_f1_mean", float("nan")) - raw.get(
            "macro_f1_mean", float("nan")
        )
        print(f"  Δ contrastive − raw: {delta_f1:+.3f}")

    # Per-seed file written as each seed completes (resume + progress).
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, contrastive_seed_result_filename(seed))
    atomic_write_json(out_path, results)
    print(f"\nSeed {seed} results written to {out_path}")
    return results


def load_all_data():
    """Load variants, Pfam map, deltas, and a fitted label encoder once."""
    print("=== Loading data ===")
    variants, labels, genes, delta_mean = load_data()

    print("\n=== Loading Pfam map ===")
    gene_pfam, pfam_map = load_pfam(genes)

    le = LabelEncoder()
    le.fit(labels)
    print(f"Classes: {list(le.classes_)}")

    return {
        "variants": variants,
        "labels": labels,
        "genes": genes,
        "delta_mean": delta_mean,
        "gene_pfam": gene_pfam,
        "pfam_map": pfam_map,
        "le": le,
    }


def print_interpretation(aggregated):
    """Print the across-seed verdict against the MLP delta_mean family floor.

    The floor is read live from the run's mechanism aggregate.json (never
    hardcoded), so it tracks whatever the current run's MLP baseline is.
    """
    fam = aggregated.get(FAMILY_SPLIT, {})
    cont = fam.get("contrastive_knn", {})
    raw = fam.get("raw_knn_baseline", {})
    cont_f1 = cont.get("macro_f1_seed_mean", float("nan"))
    raw_f1 = raw.get("macro_f1_seed_mean", float("nan"))
    floor = read_across_seed_metric(
        MECHANISM_AGGREGATE_JSON, FAMILY_SPLIT, DELTA_MEAN_FEATURE
    )

    print("\n=== Across-seed interpretation (family-split) ===")
    print(f"  MLP delta_mean floor (from aggregate.json): {floor:.3f}")
    # Distinguish "data missing" from a genuine negative: a NaN in any operand
    # makes every > comparison False, which would otherwise be misreported as the
    # "no signal" verdict. Report the missing data explicitly instead.
    if not all(np.isfinite(value) for value in (cont_f1, raw_f1, floor)):
        print(
            "  ? Verdict undefined — a required value is missing/NaN: "
            f"contrastive_f1={cont_f1}, raw_f1={raw_f1}, floor={floor}."
        )
        print("    → Check that all seeds produced family-split macro_f1 and that "
              "the MLP aggregate.json contains the delta_mean floor.")
        return
    if cont_f1 > floor + 0.03:
        print(f"  ✓ Contrastive family-split F1 ({cont_f1:.3f}) > MLP floor ({floor:.3f}) + 0.03")
        print("    → Cross-family mechanism signal CAN be improved by explicit family-invariance pressure.")
        print("    → ESM-2 deltas DO encode mechanism beyond family, accessible via metric learning.")
    elif cont_f1 > raw_f1 + 0.02:
        print(f"  ~ Contrastive ({cont_f1:.3f}) > raw k-NN ({raw_f1:.3f}) but below MLP floor ({floor:.3f}) + 0.03")
        print("    → Small improvement from contrastive training; mechanism is weakly present but hard to extract.")
    else:
        print(f"  ✗ Contrastive ({cont_f1:.3f}) ≈ raw k-NN ({raw_f1:.3f})")
        print("    → Family-invariance pressure doesn't recover additional mechanism signal.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run a single seed (no across-seed aggregation). Omit to run seeds 0-4.",
    )
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument(
        "--proj_dim", type=int, default=64,
        help="Output dimension of projection head (default 64)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=512,
        help="Batch size for triplet training (default 512, use 4096+ on GPU)",
    )
    args = parser.parse_args()

    out_dir = str(CONTRASTIVE_RESULTS_DIR)
    data = load_all_data()

    seeds = [args.seed] if args.seed is not None else [0, 1, 2, 3, 4]
    for seed in seeds:
        print("\n\n" + "#" * 60)
        print(f"# SEED {seed}")
        print("#" * 60)
        run(
            data, out_dir, seed,
            n_folds=args.n_folds, proj_dim=args.proj_dim, batch_size=args.batch_size,
        )

    if args.seed is not None:
        # Single-seed run: skip aggregation (would pool only one seed).
        return

    # Pool the per-seed files into one across-seed headline (mean ± std ACROSS
    # seeds), mirroring classify_by_mechanism.main.
    print("\n=== Aggregating across seeds ===")
    seed_results = load_seed_files(out_dir, CONTRASTIVE_SEED_RESULT_GLOB)
    if not seed_results:
        print(f"WARNING: no seed files to aggregate in {out_dir}")
        return
    print(f"Loaded {len(seed_results)} seed files:")
    for filename, _result in seed_results:
        print(f"  {filename}")

    aggregated = aggregate_across_seeds(seed_results)
    atomic_write_json(
        CONTRASTIVE_AGGREGATE_JSON,
        {
            "n_seeds": len(seed_results),
            "seed_files": [filename for filename, _result in seed_results],
            "across_seed": aggregated,
        },
    )
    print_table(aggregated)
    print_interpretation(aggregated)
    print(f"\nWrote {CONTRASTIVE_AGGREGATE_JSON}")


if __name__ == "__main__":
    main()
