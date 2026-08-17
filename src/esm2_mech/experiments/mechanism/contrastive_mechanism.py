"""Contrastive metric learning on ESM-2 delta embeddings for mechanism classification.

Trains a projection head with cross-family-only positives (SupCon triplet loss), then
evaluates k-NN in the projected space under family-split CV vs raw-delta k-NN.
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
from esm2_mech.utils.io import atomic_write_json, load_variants_and_delta
from esm2_mech.utils.metrics import align_proba
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
from esm2_mech.utils.bootstrap import (
    adjudicate_diff,
    bootstrap_mechanism_metrics,
    family_or_gene_clusters,
    paired_oof_diff,
)
from esm2_mech.utils.constants import (
    BOOTSTRAP_N_RESAMPLES,
    CONTRASTIVE_SEED_RESULT_GLOB,
    DELTA_MEAN_FEATURE,
    DN,
    GOF,
    MECHANISM_CLASSES,
    N_SEEDS,
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

def load_data():
    variants, labels, genes, delta_mean, _ = load_variants_and_delta(
        VALID_VARIANTS_JSON, EMB_WT_MEAN, EMB_MUT_MEAN
    )
    return variants, labels, genes, delta_mean


def load_pfam(genes):
    with open(PFAM_JSON) as f:
        pfam_map = json.load(f)
    gene_pfam = np.array([pfam_map.get(g) for g in genes])
    n_annotated = sum(1 for p in gene_pfam if p is not None)
    print(f"Pfam coverage: {n_annotated}/{len(genes)} genes")
    return gene_pfam, pfam_map


def build_cross_family_pairs(labels, gene_pfam, le, max_pairs_per_anchor=10, seed=42):
    """Build triplets with cross-family-only positives and different-mechanism negatives."""
    rng = np.random.RandomState(seed)
    y = le.transform(labels)
    n = len(labels)
    n_classes = len(le.classes_)

    unique_fams = list({f for f in gene_pfam if f is not None})
    fam_to_int = {f: i for i, f in enumerate(unique_fams)}
    fam_int = np.array([fam_to_int.get(f, -1) for f in gene_pfam], dtype=np.int32)

    by_mech = {c: np.where(y == c)[0] for c in range(n_classes)}
    by_mech_fam = {}
    for c in range(n_classes):
        for idx in by_mech[c]:
            key = (c, int(fam_int[idx]))
            by_mech_fam.setdefault(key, []).append(idx)
    by_mech_fam = {k: np.array(v) for k, v in by_mech_fam.items()}

    neg_by_class = {}
    for c in range(n_classes):
        neg_by_class[c] = np.concatenate(
            [by_mech[o] for o in range(n_classes) if o != c]
        )

    by_class_arr = {c: np.array(by_mech[c]) for c in range(n_classes)}

    anchor_list, pos_list, neg_list = [], [], []

    unique_combos = set()
    for i in range(n):
        unique_combos.add((int(y[i]), int(fam_int[i])))

    combo_pos_pool = {}
    for c, fam in unique_combos:
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

    mu = X_train.mean(0)
    std = X_train.std(0) + 1e-8
    X_norm = ((X_train - mu) / std).astype(np.float32)

    anchors, positives, negatives = build_cross_family_pairs(
        labels_train, gene_pfam_train, le, max_pairs_per_anchor=8, seed=seed
    )

    if len(anchors) < 50:
        # Substituting all-family positives would change what this experiment measures.
        raise ValueError(
            f"Only {len(anchors)} cross-family triplets available (need >= 50); "
            "cannot train the family-invariant projection head on this fold."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Weight init and dropout draw from the global RNG, so seed before building the head.
    torch.manual_seed(seed)

    layers = []
    prev = X_norm.shape[1]
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
        prev = h
    proj = nn.Sequential(*layers).to(device)

    optimizer = torch.optim.Adam(proj.parameters(), lr=lr, weight_decay=1e-4)
    triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    # Full matrix on-device avoids per-batch host→device copies.
    X_t = torch.tensor(X_norm, dtype=torch.float32, device=device)

    n_val = min(max(10, len(anchors) // 7), len(anchors) - 1)
    rng_val = np.random.RandomState(seed + 99)
    val_idx = rng_val.choice(len(anchors), size=n_val, replace=False)
    train_idx_mask = np.ones(len(anchors), dtype=bool)
    train_idx_mask[val_idx] = False

    anc_tr = torch.as_tensor(anchors[train_idx_mask], dtype=torch.long, device=device)
    pos_tr = torch.as_tensor(positives[train_idx_mask], dtype=torch.long, device=device)
    neg_tr = torch.as_tensor(negatives[train_idx_mask], dtype=torch.long, device=device)
    anc_val = torch.as_tensor(anchors[val_idx], dtype=torch.long, device=device)
    pos_val = torch.as_tensor(positives[val_idx], dtype=torch.long, device=device)
    neg_val = torch.as_tensor(negatives[val_idx], dtype=torch.long, device=device)
    n_train_triplets = anc_tr.shape[0]

    gen = torch.Generator(device=device).manual_seed(seed + 7)

    best_loss = float("inf")
    patience_count = 0
    best_state = None

    print(
        f"    training projection head{(' ' + progress_label) if progress_label else ''}: "
        f"{len(anc_tr)} train / {len(anc_val)} val triplets, "
        f"max_epochs={max_epochs} patience={patience} batch_size={batch_size} "
        f"on {device.type}"
    )

    epochs_run = 0
    for epoch in range(max_epochs):
        epochs_run = epoch + 1
        proj.train()
        perm = torch.randperm(n_train_triplets, generator=gen, device=device)
        for start in range(0, n_train_triplets, batch_size):
            batch_idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            z_a = proj(X_t[anc_tr[batch_idx]])
            z_p = proj(X_t[pos_tr[batch_idx]])
            z_n = proj(X_t[neg_tr[batch_idx]])
            loss = triplet_loss(z_a, z_p, z_n)
            loss.backward()
            optimizer.step()

        proj.eval()
        with torch.no_grad():
            z_a = proj(X_t[anc_val])
            z_p = proj(X_t[pos_val])
            z_n = proj(X_t[neg_val])
            val_loss = triplet_loss(z_a, z_p, z_n).item()

        improved = val_loss < best_loss - 1e-4
        if improved:
            best_loss = val_loss
            patience_count = 0
            best_state = {k: v.clone() for k, v in proj.state_dict().items()}
        else:
            patience_count += 1

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
        Z = proj(X_t).cpu().numpy()

    return proj, Z, mu, std, epochs_run


def project_test(proj, X_test, mu, std):
    import torch

    X_norm = ((X_test - mu) / std).astype(np.float32)
    device = next(proj.parameters()).device
    proj.eval()
    with torch.no_grad():
        Z = proj(torch.tensor(X_norm).to(device)).cpu().numpy()
    return Z


def run_knn(Z_train, Z_test, y_train, y_test, le, k=10):
    """Returns (metrics_dict, proba aligned to MECHANISM_CLASSES order)."""
    k_eff = min(k, len(Z_train) - 1)
    knn = KNeighborsClassifier(n_neighbors=k_eff, metric="cosine")
    knn.fit(Z_train, y_train)
    pred = knn.predict(Z_test)

    raw_proba = knn.predict_proba(Z_test)
    all_classes = list(le.classes_)
    train_cls_str = le.classes_[np.asarray(knn.classes_)]
    proba = align_proba(raw_proba, train_cls_str, all_classes)
    proba_mechanism_order = align_proba(raw_proba, train_cls_str, MECHANISM_CLASSES)

    fm = {"macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0))}
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
    return fm, proba_mechanism_order


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
    oof_contrastive = {"y_true": [], "proba": [], "genes": [], "row_ids": []}
    oof_raw_knn = {"y_true": [], "proba": [], "genes": [], "row_ids": []}

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        labels_tr, labels_te = labels[train_idx], labels[test_idx]
        gene_pfam_tr = gene_pfam[train_idx]
        genes_te = genes[test_idx]

        # fold_i advances even when skipped, keeping seed offset (seed + fold_i) stable.
        if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
            print(f"  Split {fold_i+1}/{len(splits)}: skipped (missing class)")
            continue

        print(
            f"\n  Split {fold_i+1}/{len(splits)} [{split_name}]  "
            f"(completed so far: {len(fold_results_contrastive)})  "
            f"train={len(train_idx)} test={len(test_idx)}"
        )
        print(f"    train classes: {dict(Counter(labels_tr))}")

        mu_raw = X_tr.mean(0)
        std_raw = X_tr.std(0) + 1e-8
        Z_tr_raw = (X_tr - mu_raw) / std_raw
        Z_te_raw = (X_te - mu_raw) / std_raw
        raw_fm, raw_proba = run_knn(Z_tr_raw, Z_te_raw, y_tr, y_te, le, k=10)
        fold_results_raw_knn.append(raw_fm)
        oof_raw_knn["y_true"].append(labels_te)
        oof_raw_knn["proba"].append(raw_proba)
        oof_raw_knn["genes"].append(genes_te)
        oof_raw_knn["row_ids"].append(test_idx)
        print(
            f"    raw k-NN:      macro_f1={raw_fm['macro_f1']:.3f}  "
            f"{GOF}={raw_fm.get(f'auroc_{GOF}', float('nan')):.3f}  "
            f"{DN}={raw_fm.get(f'auroc_{DN}', float('nan')):.3f}"
        )

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

        cont_fm, cont_proba = run_knn(Z_tr_proj, Z_te_proj, y_tr, y_te, le, k=10)
        fold_results_contrastive.append(cont_fm)
        oof_contrastive["y_true"].append(labels_te)
        oof_contrastive["proba"].append(cont_proba)
        oof_contrastive["genes"].append(genes_te)
        oof_contrastive["row_ids"].append(test_idx)
        print(
            f"    contrastive:   macro_f1={cont_fm['macro_f1']:.3f}  "
            f"{GOF}={cont_fm.get(f'auroc_{GOF}', float('nan')):.3f}  "
            f"{DN}={cont_fm.get(f'auroc_{DN}', float('nan')):.3f}  "
            f"(epochs={epochs})"
        )

    def agg(fold_list):
        if not fold_list:
            return {"error": "no folds"}
        # Only aggregate float-valued per-fold metrics, so int/bool fields are never meaned.
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
                out[f"{key}_n_folds"] = len(vals)
        skip_counts = Counter()
        for fold in fold_list:
            for cls_str, reason in fold.get("auroc_skipped", {}).items():
                skip_counts[f"auroc_{cls_str}:{reason}"] += 1
        if skip_counts:
            out["auroc_skipped_counts"] = dict(skip_counts)
            print(f"    auroc skipped (by class:reason): {dict(skip_counts)}")
        out["n_folds"] = len(fold_list)
        return out

    def _finalize_oof(oof):
        if not oof["y_true"]:
            return None
        return {
            "y_true": np.concatenate(oof["y_true"]),
            "proba": np.concatenate(oof["proba"]),
            "genes": np.concatenate(oof["genes"]),
            "row_ids": np.concatenate(oof["row_ids"]),
        }

    return (
        agg(fold_results_contrastive),
        agg(fold_results_raw_knn),
        _finalize_oof(oof_contrastive),
        _finalize_oof(oof_raw_knn),
    )


def run(
    data, out_dir, seed, n_folds=5, proj_dim=64, batch_size=512,
    compute_ci=True, n_boot=BOOTSTRAP_N_RESAMPLES,
):
    """Run gene-split and family-split contrastive CV for one seed."""
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
    gene_cont, gene_raw, gene_cont_oof, gene_raw_oof = run_cv(
        delta_mean, labels, genes, gene_pfam, pfam_map, le, gene_splits,
        "gene-split", hidden=hidden, seed=seed, batch_size=batch_size,
    )

    print("\n\n" + "=" * 60)
    print(f"FAMILY-SPLIT CV (seed {seed})")
    print("=" * 60)
    fam_cont, fam_raw, fam_cont_oof, fam_raw_oof = run_cv(
        delta_mean, labels, genes, gene_pfam, pfam_map, le, fam_splits,
        "family-split", hidden=hidden, seed=seed, batch_size=batch_size,
    )

    if compute_ci:
        for agg, oof, is_family_split in (
            (gene_cont, gene_cont_oof, False), (gene_raw, gene_raw_oof, False),
            (fam_cont, fam_cont_oof, True), (fam_raw, fam_raw_oof, True),
        ):
            if oof is not None:
                clusters = family_or_gene_clusters(
                    oof["genes"], pfam_map, is_family_split=is_family_split
                )
                agg["ci"] = bootstrap_mechanism_metrics(
                    oof["y_true"], oof["proba"], clusters,
                    n_resamples=n_boot, seed=seed,
                )

    paired = {}
    if compute_ci:
        print("\n=== PAIRED DIFFERENCES (contrastive − raw k-NN) ===")
        for split_name, cont_oof, raw_oof, is_family_split in (
            ("gene_split", gene_cont_oof, gene_raw_oof, False),
            ("family_split", fam_cont_oof, fam_raw_oof, True),
        ):
            diff = paired_oof_diff(
                cont_oof, raw_oof, pfam_map,
                f"{split_name}: contrastive − raw_knn",
                classes=list(MECHANISM_CLASSES),
                is_family_split=is_family_split,
                n_resamples=n_boot,
                seed=seed,
            )
            if diff is None:
                continue
            paired[split_name] = {"macro_f1": diff}
            if diff.get("ci_low") is None:
                print(f"  {split_name} macro_f1: diff={diff['point_diff']:+.4f}  CI suppressed")
            else:
                print(
                    f"  {split_name} macro_f1: diff={diff['point_diff']:+.4f}  "
                    f"[{diff['ci_low']:+.4f}, {diff['ci_high']:+.4f}]  "
                    f"({diff['n_clusters']} clusters) — "
                    f"{adjudicate_diff(diff['point_diff'] > 0, diff, 0.0)}"
                )

            for cls in MECHANISM_CLASSES:
                cls_diff = paired_oof_diff(
                    cont_oof, raw_oof, pfam_map,
                    f"{split_name}: contrastive − raw_knn ({cls} AUROC)",
                    classes=list(MECHANISM_CLASSES),
                    metric="auroc_one_vs_rest",
                    pos_class=cls,
                    is_family_split=is_family_split,
                    n_resamples=n_boot,
                    seed=seed,
                )
                if cls_diff is None:
                    continue
                paired[split_name][f"auroc_{cls}"] = cls_diff
                if cls_diff.get("ci_low") is None:
                    print(
                        f"  {split_name} auroc_{cls}: "
                        f"diff={cls_diff['point_diff']:+.4f}  CI suppressed"
                    )
                else:
                    print(
                        f"  {split_name} auroc_{cls}: diff={cls_diff['point_diff']:+.4f}  "
                        f"[{cls_diff['ci_low']:+.4f}, {cls_diff['ci_high']:+.4f}]  "
                        f"— {adjudicate_diff(cls_diff['point_diff'] > 0, cls_diff, 0.0)}"
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
            "paired_diff_vs_raw_knn": paired.get("gene_split"),
        },
        "family_split": {
            "contrastive_knn": fam_cont,
            "raw_knn_baseline": fam_raw,
            "paired_diff_vs_raw_knn": paired.get("family_split"),
        },
    }

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

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, contrastive_seed_result_filename(seed))
    atomic_write_json(out_path, results)
    print(f"\nSeed {seed} results written to {out_path}")
    return results


def load_all_data():
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
    """Print across-seed verdict against the live MLP delta_mean family floor."""
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
    # NaN makes > False, which would silently report "no signal" — check explicitly.
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
        help="Run a single seed (no across-seed aggregation). Omit to run seeds 0..N_SEEDS-1.",
    )
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument(
        "--proj_dim", type=int, default=64,
        help="Output dimension of projection head (default 64)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16384,
        help="Batch size for triplet training. With the feature matrix resident "
        "on-device, large batches are cheap; default 16384 (lower for tiny GPUs).",
    )
    parser.add_argument("--no_ci", action="store_true", help="skip cluster-bootstrap CIs")
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    args = parser.parse_args()

    out_dir = str(CONTRASTIVE_RESULTS_DIR)
    data = load_all_data()

    seeds = [args.seed] if args.seed is not None else list(range(N_SEEDS))
    for seed in seeds:
        print("\n\n" + "#" * 60)
        print(f"# SEED {seed}")
        print("#" * 60)
        run(
            data, out_dir, seed,
            n_folds=args.n_folds, proj_dim=args.proj_dim, batch_size=args.batch_size,
            compute_ci=not args.no_ci, n_boot=args.n_boot,
        )

    if args.seed is not None:
        return

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
