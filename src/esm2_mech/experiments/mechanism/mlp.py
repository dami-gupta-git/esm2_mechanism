"""
MLP nonlinearity probe for ESM-2 delta embeddings.

Tests whether mechanism signal (GOF/DN/LOF) is nonlinearly separable in delta space
where the linear probe (experiment.py) was at chance.

Loads cached embeddings from the canonical paths in utils_paths (written by embed_variants.py).

Probe: 2-layer MLP (1280 -> 256 -> 64 -> 3), dropout 0.3, early stopping.
CV: same 5-fold gene-split as experiment.py.
Features tested: delta_mean, delta_pos (per-residue at variant position).
No hyperparameter tuning — fixed architecture, signal-detection only.
"""

import argparse
import json
import os
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder
from esm2_mech.utils.splits import gene_split_cv
from esm2_mech.utils.paths import EMB_MUT_MEAN, EMB_MUT_POS, EMB_WT_MEAN, EMB_WT_POS, RESULTS_DIR, SEQUENCES_JSON, VARIANTS_JSON
import functools

print = functools.partial(print, flush=True)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Reuse data-loading helpers from experiment.py
# ---------------------------------------------------------------------------


def load_variants_and_labels(variants_file=None):
    if variants_file:
        # Pre-filtered variant list (e.g. valid_variants.json) — skip sequence filtering
        with open(variants_file) as f:
            valid_variants = json.load(f)
        for v in valid_variants:
            if "label_3class" not in v:
                v["label_3class"] = (
                    "LOF" if v["mechanism"] in ("HI", "AR") else v["mechanism"]
                )
    else:
        cache_path = VARIANTS_JSON
        with open(cache_path) as f:
            variants = json.load(f)
        for v in variants:
            v["label_3class"] = (
                "LOF" if v["mechanism"] in ("HI", "AR") else v["mechanism"]
            )
        variants = [
            v
            for v in variants
            if v["uniprot_id"] and v["aa_wt"] and v["aa_mut"] and v["aa_pos"] > 0
        ]

        with open(SEQUENCES_JSON) as f:
            seq_cache = json.load(f)

        from esm2_mech.utils.sequences import apply_missense, window_sequence

        valid_variants = []
        for v in variants:
            uid = v["uniprot_id"]
            if uid not in seq_cache:
                continue
            wt_full = seq_cache[uid]
            wt_win, new_pos = window_sequence(wt_full, v["aa_pos"])
            mut_win = apply_missense(wt_win, new_pos, v["aa_wt"], v["aa_mut"])
            if mut_win is None:
                continue
            valid_variants.append(v)

    labels = np.array([v["label_3class"] for v in valid_variants])
    genes = np.array([v["gene"] for v in valid_variants])
    print(f"Loaded {len(valid_variants)} variants, {len(set(genes))} genes")
    from collections import Counter

    print(f"Class distribution: {dict(Counter(labels))}")
    return valid_variants, labels, genes


def load_embeddings():
    """Load Gerasimavicius embeddings from canonical paths."""
    for path in [EMB_WT_MEAN, EMB_MUT_MEAN, EMB_WT_POS, EMB_MUT_POS]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Embedding file missing: {path}\n"
                f"Run: python -m esm2_mechanism.embeddings.embed_variants"
            )
    delta_mean = np.load(EMB_MUT_MEAN) - np.load(EMB_WT_MEAN)
    delta_pos = np.load(EMB_MUT_POS) - np.load(EMB_WT_POS)
    print(f"Embeddings loaded: delta_mean {delta_mean.shape}, delta_pos {delta_pos.shape}")
    return delta_mean, delta_pos


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# MLP probe (PyTorch)
# ---------------------------------------------------------------------------


def make_family_splits(genes, pfam_map, n_folds=5, seed=42):
    """Build family-split CV using Pfam annotations."""
    n = len(genes)
    gene_to_pfam = {g: pfam_map.get(g) for g in np.unique(genes) if pfam_map.get(g)}
    unique_fams = sorted(set(gene_to_pfam.values()))
    rng = np.random.RandomState(seed)
    fam_arr = np.array(unique_fams)
    rng.shuffle(fam_arr)
    splits = []
    for fold_fams in np.array_split(fam_arr, n_folds):
        fold_set = set(fold_fams)
        te = np.array(
            [
                genes[i] in gene_to_pfam and gene_to_pfam[genes[i]] in fold_set
                for i in range(n)
            ]
        )
        tr = np.array(
            [
                genes[i] in gene_to_pfam and gene_to_pfam[genes[i]] not in fold_set
                for i in range(n)
            ]
        )
        if tr.sum() >= 10 and te.sum() >= 5:
            splits.append((np.where(tr)[0], np.where(te)[0]))
    print(f"  Family-split: {len(splits)} folds, {len(unique_fams)} families")
    return splits


def run_mlp_probe(
    X,
    labels,
    genes,
    n_folds=5,
    seed=42,
    hidden=(256, 64),
    dropout=0.3,
    lr=1e-3,
    max_epochs=100,
    patience=10,
    batch_size=256,
    splits=None,
):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    le = LabelEncoder()
    y = le.fit_transform(labels)
    classes = le.classes_
    n_classes = len(classes)
    if splits is None:
        splits = gene_split_cv(genes, n_folds=n_folds, seed=seed)

    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_tr, X_te = X[train_idx].astype(np.float32), X[test_idx].astype(np.float32)
        y_tr, y_te = y[train_idx], y[test_idx]

        if len(set(y_tr)) < 2:
            continue

        # Hold out 15% of training genes as validation for early stopping
        train_genes = genes[train_idx]
        unique_tr_genes = np.array(sorted(set(train_genes)))
        rng = np.random.RandomState(seed + fold_i)
        rng.shuffle(unique_tr_genes)
        n_val_genes = max(1, int(0.15 * len(unique_tr_genes)))
        val_gene_set = set(unique_tr_genes[:n_val_genes])
        val_mask = np.array([g in val_gene_set for g in train_genes])
        fit_mask = ~val_mask

        X_fit = X_tr[fit_mask]
        y_fit = y_tr[fit_mask]
        X_val = X_tr[val_mask]
        y_val = y_tr[val_mask]

        if len(X_fit) < 10 or len(X_val) < 5:
            continue

        # Normalize using fit-set stats
        mu = X_fit.mean(0)
        std = X_fit.std(0) + 1e-8
        X_fit = (X_fit - mu) / std
        X_val = (X_val - mu) / std
        X_te_norm = (X_te - mu) / std

        # Class weights from full training fold (y_tr), not the fit subset,
        # to avoid weight explosion when a rare class is absent from y_fit.
        class_counts = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
        class_weights = torch.tensor(1.0 / (class_counts + 1e-8))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _build_mlp(X_fit.shape[1], hidden, dropout, n_classes).to(device)
        class_weights = class_weights.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        fit_ds = TensorDataset(
            torch.tensor(X_fit), torch.tensor(y_fit, dtype=torch.long)
        )
        fit_loader = DataLoader(fit_ds, batch_size=batch_size, shuffle=True)

        best_val_loss = float("inf")
        patience_count = 0
        best_state = None

        for epoch in range(max_epochs):
            model.train()
            for xb, yb in fit_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(
                    model(torch.tensor(X_val).to(device)),
                    torch.tensor(y_val, dtype=torch.long).to(device),
                ).item()

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                patience_count = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_te_norm).to(device))
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        pred = proba.argmax(1)

        fm = {"macro_f1": float(f1_score(y_te, pred, average="macro", zero_division=0))}
        for i, cls in enumerate(classes):
            y_bin = (y_te == i).astype(int)
            if y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
                fm[f"auroc_{cls}"] = float(roc_auc_score(y_bin, proba[:, i]))
        fm["epochs_run"] = epoch + 1
        fold_results.append(fm)
        print(
            f"  Fold {fold_i+1}: macro_f1={fm['macro_f1']:.3f}, epochs={fm['epochs_run']}"
        )

    if not fold_results:
        return {"error": "insufficient data"}

    agg = {}
    all_keys = set().union(*[set(f.keys()) for f in fold_results]) - {"epochs_run"}
    for key in all_keys:
        vals = [f[key] for f in fold_results if key in f and not np.isnan(f[key])]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
    return agg


def _build_mlp(in_dim, hidden, dropout, n_classes):
    import torch.nn as nn

    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, n_classes))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# GBM, RF, kNN probes (sklearn)
# ---------------------------------------------------------------------------


def run_sklearn_probe(clf_fn, X, labels, genes, n_folds=5, seed=42, normalize=False):
    """Generic gene-split CV runner for any sklearn classifier."""
    le = LabelEncoder()
    y = le.fit_transform(labels)
    classes = le.classes_
    splits = gene_split_cv(genes, n_folds=n_folds, seed=seed)

    fold_results = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_tr, X_te = X[train_idx].astype(np.float32), X[test_idx].astype(np.float32)
        y_tr, y_te = y[train_idx], y[test_idx]

        if len(set(y_tr)) < 2:
            continue

        if normalize:
            mu = X_tr.mean(0)
            std = X_tr.std(0) + 1e-8
            X_tr = (X_tr - mu) / std
            X_te = (X_te - mu) / std

        clf = clf_fn(seed)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)

        fm = {"macro_f1": float(f1_score(y_te, pred, average="macro", zero_division=0))}
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_te)
            for i, cls in enumerate(classes):
                y_bin = (y_te == i).astype(int)
                if y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
                    fm[f"auroc_{cls}"] = float(roc_auc_score(y_bin, proba[:, i]))
        fold_results.append(fm)
        print(f"  Fold {fold_i+1}: macro_f1={fm['macro_f1']:.3f}")

    if not fold_results:
        return {"error": "insufficient data"}

    agg = {}
    for key in fold_results[0]:
        vals = [f[key] for f in fold_results if key in f and not np.isnan(f[key])]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
    return agg


# ---------------------------------------------------------------------------
# PCA reduction helper (GBM/RF are slow on 1280-dim; reduce first)
# ---------------------------------------------------------------------------


def pca_reduce(X_tr, X_te, n_components=50):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=0)
    return pca.fit_transform(X_tr), pca.transform(X_te)


def run_sklearn_probe_pca(clf_fn, X, labels, genes, n_folds=5, seed=42, n_pca=50):
    """Gene-split CV with PCA reduction applied per fold."""
    from sklearn.decomposition import PCA

    le = LabelEncoder()
    y = le.fit_transform(labels)
    classes = le.classes_
    splits = gene_split_cv(genes, n_folds=n_folds, seed=seed)

    fold_results = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_tr, X_te = X[train_idx].astype(np.float32), X[test_idx].astype(np.float32)
        y_tr, y_te = y[train_idx], y[test_idx]

        if len(set(y_tr)) < 2:
            continue

        # Normalize then PCA
        mu = X_tr.mean(0)
        std = X_tr.std(0) + 1e-8
        X_tr = (X_tr - mu) / std
        X_te = (X_te - mu) / std
        pca = PCA(
            n_components=min(n_pca, X_tr.shape[1], X_tr.shape[0] - 1), random_state=seed
        )
        X_tr = pca.fit_transform(X_tr)
        X_te = pca.transform(X_te)

        clf = clf_fn(seed)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)

        fm = {"macro_f1": float(f1_score(y_te, pred, average="macro", zero_division=0))}
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_te)
            for i, cls in enumerate(classes):
                y_bin = (y_te == i).astype(int)
                if y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
                    fm[f"auroc_{cls}"] = float(roc_auc_score(y_bin, proba[:, i]))
        fold_results.append(fm)
        print(f"  Fold {fold_i+1}: macro_f1={fm['macro_f1']:.3f}")

    if not fold_results:
        return {"error": "insufficient data"}

    agg = {}
    for key in fold_results[0]:
        vals = [f[key] for f in fold_results if key in f and not np.isnan(f[key])]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Unused; variants and sequences are loaded from canonical paths.py paths.",
    )
    parser.add_argument("--out_dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--family_split",
        action="store_true",
        help="Run family-split CV in addition to gene-split",
    )
    parser.add_argument(
        "--pfam_map",
        type=str,
        default=None,
        help="Path to pfam_families.json (required for --family_split)",
    )
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--variants_file",
        type=str,
        default=None,
        help="Pre-filtered variants JSON (e.g. valid_variants.json). "
        "Skips sequence filtering. Embeddings must be aligned.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=== Loading variants and labels ===")
    valid_variants, labels, genes = load_variants_and_labels(args.variants_file)

    print("\n=== Loading embeddings ===")
    delta_mean, delta_pos = load_embeddings()

    assert len(delta_mean) == len(labels), (
        f"Embedding count {len(delta_mean)} != variant count {len(labels)}. "
        "Ensure embeddings were generated from the same filtered variant list."
    )

    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier

    def gbm_fn(seed):
        return GradientBoostingClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=seed,
        )

    def rf_fn(seed):
        return RandomForestClassifier(
            n_estimators=50, max_depth=8, random_state=seed, n_jobs=-1
        )

    def knn_fn(seed):
        return KNeighborsClassifier(n_neighbors=10, metric="cosine")

    results = {}

    # Build splits
    gene_splits = gene_split_cv(genes, seed=args.seed)
    family_splits = None
    if args.family_split:
        pfam_path = args.pfam_map or os.path.join(args.data_dir, "pfam_families.json")
        with open(pfam_path) as f:
            pfam_map = json.load(f)
        family_splits = make_family_splits(genes, pfam_map, seed=args.seed)

    for feat_name, X in [("delta_mean", delta_mean), ("delta_pos", delta_pos)]:
        print(f"\n=== MLP gene-split: {feat_name} ===")
        results[f"mlp_{feat_name}_gene"] = run_mlp_probe(
            X,
            labels,
            genes,
            seed=args.seed,
            splits=gene_splits,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        print(
            f"  macro_f1={results[f'mlp_{feat_name}_gene'].get('macro_f1_mean', float('nan')):.3f}"
        )

        if family_splits:
            print(f"\n=== MLP family-split: {feat_name} ===")
            results[f"mlp_{feat_name}_family"] = run_mlp_probe(
                X,
                labels,
                genes,
                seed=args.seed,
                splits=family_splits,
                max_epochs=args.max_epochs,
                patience=args.patience,
            )
            print(
                f"  macro_f1={results[f'mlp_{feat_name}_family'].get('macro_f1_mean', float('nan')):.3f}"
            )
            delta = results[f"mlp_{feat_name}_gene"].get(
                "macro_f1_mean", float("nan")
            ) - results[f"mlp_{feat_name}_family"].get("macro_f1_mean", float("nan"))
            print(f"  Δ(gene − family) = {delta:+.3f}  ← positive ⇒ homology leakage")

        print(f"\n=== GBM gene-split: {feat_name} (PCA-50) ===")
        results[f"gbm_{feat_name}"] = run_sklearn_probe_pca(
            gbm_fn, X, labels, genes, seed=args.seed, n_pca=50
        )
        print(
            f"  macro_f1={results[f'gbm_{feat_name}'].get('macro_f1_mean', float('nan')):.3f}"
        )

        print(f"\n=== RF gene-split: {feat_name} (PCA-50) ===")
        results[f"rf_{feat_name}"] = run_sklearn_probe_pca(
            rf_fn, X, labels, genes, seed=args.seed, n_pca=50
        )
        print(
            f"  macro_f1={results[f'rf_{feat_name}'].get('macro_f1_mean', float('nan')):.3f}"
        )

        print(f"\n=== kNN gene-split: {feat_name} ===")
        results[f"knn_{feat_name}"] = run_sklearn_probe(
            knn_fn, X, labels, genes, seed=args.seed, normalize=True
        )
        print(
            f"  macro_f1={results[f'knn_{feat_name}'].get('macro_f1_mean', float('nan')):.3f}"
        )

    print("\n=== Summary ===")
    for feat, res in results.items():
        mf1 = res.get("macro_f1_mean", float("nan"))
        auroc_gof = res.get("auroc_GOF_mean", float("nan"))
        print(f"  {feat}: macro_f1={mf1:.3f}  auroc_GOF={auroc_gof:.3f}")

    out_path = os.path.join(args.out_dir, f"mlp_results_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
