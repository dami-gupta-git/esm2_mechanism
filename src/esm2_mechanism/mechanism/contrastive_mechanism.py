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

Usage:
    python contrastive_mechanism.py \
        --data_dir ../data \
        --emb_dir ../data/embeddings \
        --out_dir ../results/20260524_baseline_run/run_0 \
        --seed 0

Outputs: contrastive_results_seed{seed}.json
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
from esm2_mechanism.utils_probes import gene_split_cv, family_split_cv
from esm2_mechanism.utils_paths import VALID_VARIANTS_JSON, EMB_WT_MEAN, EMB_MUT_MEAN
import functools

print = functools.partial(print, flush=True)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(data_dir, emb_dir, merged=False):
    """Load variants and embeddings. Use merged=True for the full 19100-variant dataset."""
    with open(VALID_VARIANTS_JSON) as f:
        mv = json.load(f)

    if merged:
        variants = mv  # all 19100
        print(f"Loaded {len(variants)} merged variants (Gerasimavicius + G2P)")
    else:
        variants = [v for v in mv if v.get("source") == "gerasimavicius"]
        assert (
            len(variants) == 10231
        ), f"Expected 10231 Gerasimavicius variants, got {len(variants)}"
        print(f"Loaded {len(variants)} Gerasimavicius variants")

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


def load_pfam(data_dir, genes):
    pfam_path = os.path.join(data_dir, "pfam_families.json")
    with open(pfam_path) as f:
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
        print(
            "  WARNING: too few cross-family triplets, falling back to all-family positives"
        )
        # Fallback: positives = same mechanism regardless of family
        y = le.transform(labels_train)
        by_mech = {c: np.where(y == c)[0] for c in range(len(le.classes_))}
        rng = np.random.RandomState(seed)
        anchors_fb, pos_fb, neg_fb = [], [], []
        for i in range(len(X_norm)):
            c = y[i]
            pool_pos = [j for j in by_mech[c] if j != i]
            pool_neg = [j for j in range(len(X_norm)) if y[j] != c]
            if not pool_pos or not pool_neg:
                continue
            for _ in range(4):
                anchors_fb.append(i)
                pos_fb.append(rng.choice(pool_pos))
                neg_fb.append(rng.choice(pool_neg))
        anchors, positives, negatives = (
            np.array(anchors_fb),
            np.array(pos_fb),
            np.array(neg_fb),
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

    for epoch in range(max_epochs):
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

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            patience_count = 0
            best_state = {k: v.clone() for k, v in proj.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    if best_state is not None:
        proj.load_state_dict(best_state)

    proj.eval()
    with torch.no_grad():
        Z = proj(X_t.to(device)).cpu().numpy()

    return proj, Z, mu, std, epoch + 1


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
    for all_i, cls_str in enumerate(all_classes):
        cls_int = le.transform([cls_str])[0]
        y_bin = (y_test == cls_int).astype(int)
        if y_bin.sum() > 0 and (1 - y_bin).sum() > 0 and proba[:, all_i].std() > 0:
            fm[f"auroc_{cls_str}"] = float(roc_auc_score(y_bin, proba[:, all_i]))
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

        if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
            print(f"  Fold {fold_i+1}: skipped (missing class)")
            continue

        print(
            f"\n  Fold {fold_i+1}/{len(splits)} [{split_name}]  "
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
            f"GOF={raw_fm.get('auroc_GOF', float('nan')):.3f}  "
            f"DN={raw_fm.get('auroc_DN', float('nan')):.3f}"
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
        )
        Z_te_proj = project_test(proj, X_te, mu, std)

        cont_fm = run_knn(Z_tr_proj, Z_te_proj, y_tr, y_te, le, k=10)
        fold_results_contrastive.append(cont_fm)
        print(
            f"    contrastive:   macro_f1={cont_fm['macro_f1']:.3f}  "
            f"GOF={cont_fm.get('auroc_GOF', float('nan')):.3f}  "
            f"DN={cont_fm.get('auroc_DN', float('nan')):.3f}  "
            f"(epochs={epochs})"
        )

    def agg(fold_list):
        if not fold_list:
            return {"error": "no folds"}
        all_keys = set().union(*[set(f.keys()) for f in fold_list])
        out = {}
        for k in all_keys:
            vals = [f[k] for f in fold_list if k in f and not np.isnan(f[k])]
            if vals:
                out[f"{k}_mean"] = float(np.mean(vals))
                out[f"{k}_std"] = float(np.std(vals))
        out["n_folds"] = len(fold_list)
        return out

    return agg(fold_results_contrastive), agg(fold_results_raw_knn)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--emb_dir", default="../data/embeddings")
    parser.add_argument("--out_dir", default="../results/20260524_baseline_run/run_0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument(
        "--proj_dim",
        type=int,
        default=64,
        help="Output dimension of projection head (default 64)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for triplet training (default 512, use 4096+ on GPU)",
    )
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Use full merged dataset (19100 variants, 1985 genes) instead of Gerasimavicius only",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=== Loading data ===")
    geras, labels, genes, delta_mean = load_data(
        args.data_dir, args.emb_dir, merged=args.merged
    )

    print("\n=== Loading Pfam map ===")
    gene_pfam, pfam_map = load_pfam(args.data_dir, genes)

    le = LabelEncoder()
    le.fit(labels)
    print(f"Classes: {list(le.classes_)}")

    print("\n=== Building CV splits ===")
    gene_splits = gene_split_cv(genes, n_folds=args.n_folds, seed=args.seed)
    fam_splits = family_split_cv(genes, pfam_map, n_folds=args.n_folds, seed=args.seed)
    print(
        f"Gene-split: {len(gene_splits)} folds | Family-split: {len(fam_splits)} folds"
    )

    hidden = (256, args.proj_dim)

    print("\n\n" + "=" * 60)
    print("GENE-SPLIT CV")
    print("=" * 60)
    gene_cont, gene_raw = run_cv(
        delta_mean,
        labels,
        genes,
        gene_pfam,
        pfam_map,
        le,
        gene_splits,
        "gene-split",
        hidden=hidden,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    print("\n\n" + "=" * 60)
    print("FAMILY-SPLIT CV")
    print("=" * 60)
    fam_cont, fam_raw = run_cv(
        delta_mean,
        labels,
        genes,
        gene_pfam,
        pfam_map,
        le,
        fam_splits,
        "family-split",
        hidden=hidden,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    results = {
        "description": (
            "Contrastive metric learning with cross-family-only positives. "
            "Positives: same mechanism, different Pfam family. "
            "Negatives: different mechanism. "
            "Within-family pairs excluded from positives. "
            "Evaluated by k-NN (k=10, cosine) in projected 64-d space."
        ),
        "architecture": f"1280 -> 256 -> {args.proj_dim} (TripletMarginLoss)",
        "seed": args.seed,
        "n_folds": args.n_folds,
        "gene_split": {
            "contrastive_knn": gene_cont,
            "raw_knn_baseline": gene_raw,
        },
        "family_split": {
            "contrastive_knn": fam_cont,
            "raw_knn_baseline": fam_raw,
        },
    }

    # Headline summary
    print("\n\n" + "=" * 60)
    print("HEADLINE SUMMARY")
    print("=" * 60)
    for split_name, split_key in [
        ("Gene-split", "gene_split"),
        ("Family-split", "family_split"),
    ]:
        cont = results[split_key]["contrastive_knn"]
        raw = results[split_key]["raw_knn_baseline"]
        print(f"\n{split_name}:")
        print(
            f"  Contrastive k-NN:  macro_f1={cont.get('macro_f1_mean', float('nan')):.3f} ± {cont.get('macro_f1_std', float('nan')):.3f}"
            f"  GOF={cont.get('auroc_GOF_mean', float('nan')):.3f}"
            f"  DN={cont.get('auroc_DN_mean', float('nan')):.3f}"
            f"  LOF={cont.get('auroc_LOF_mean', float('nan')):.3f}"
        )
        print(
            f"  Raw k-NN baseline: macro_f1={raw.get('macro_f1_mean', float('nan')):.3f} ± {raw.get('macro_f1_std', float('nan')):.3f}"
            f"  GOF={raw.get('auroc_GOF_mean', float('nan')):.3f}"
            f"  DN={raw.get('auroc_DN_mean', float('nan')):.3f}"
            f"  LOF={raw.get('auroc_LOF_mean', float('nan')):.3f}"
        )
        delta_f1 = cont.get("macro_f1_mean", float("nan")) - raw.get(
            "macro_f1_mean", float("nan")
        )
        print(f"  Δ contrastive − raw: {delta_f1:+.3f}")

    print("\nInterpretation:")
    fam_cont_f1 = fam_cont.get("macro_f1_mean", float("nan"))
    raw_fam_f1 = fam_raw.get("macro_f1_mean", float("nan"))
    floor = 0.364  # known MLP family-split F1 from result_7
    if fam_cont_f1 > floor + 0.03:
        print(
            f"  ✓ Contrastive family-split F1 ({fam_cont_f1:.3f}) > MLP floor ({floor:.3f}) + 0.03"
        )
        print(
            "    → Cross-family mechanism signal CAN be improved by explicit family-invariance pressure."
        )
        print(
            "    → ESM-2 deltas DO encode mechanism beyond family, accessible via metric learning."
        )
    elif fam_cont_f1 > raw_fam_f1 + 0.02:
        print(
            f"  ~ Contrastive ({fam_cont_f1:.3f}) > raw k-NN ({raw_fam_f1:.3f}) but below MLP floor ({floor:.3f})"
        )
        print(
            "    → Small improvement from contrastive training; mechanism is weakly present but hard to extract."
        )
    else:
        print(f"  ✗ Contrastive ({fam_cont_f1:.3f}) ≈ raw k-NN ({raw_fam_f1:.3f})")
        print(
            "    → Family-invariance pressure doesn't recover additional mechanism signal."
        )
        print(
            "    → Strong negative: even with explicit supervision to ignore family, ESM-2 deltas"
        )
        print("      cannot separate mechanism across families.")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = "merged" if args.merged else "geras"
    out_path = os.path.join(
        args.out_dir, f"contrastive_results_{tag}_seed{args.seed}.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
