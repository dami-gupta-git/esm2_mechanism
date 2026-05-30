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
from esm2_mech.utils.probes import run_mlp_probe_cv
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
        results[f"mlp_{feat_name}_gene"] = run_mlp_probe_cv(
            X,
            labels,
            gene_splits,
            seed=args.seed,
            genes=genes,
            max_epochs=args.max_epochs,
            patience=args.patience,
            label=f"{feat_name}_gene",
        )
        print(
            f"  macro_f1={results[f'mlp_{feat_name}_gene'].get('macro_f1_mean', float('nan')):.3f}"
        )

        if family_splits:
            print(f"\n=== MLP family-split: {feat_name} ===")
            results[f"mlp_{feat_name}_family"] = run_mlp_probe_cv(
                X,
                labels,
                family_splits,
                seed=args.seed,
                genes=genes,
                max_epochs=args.max_epochs,
                patience=args.patience,
                label=f"{feat_name}_family",
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
