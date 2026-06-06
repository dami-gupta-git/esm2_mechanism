"""
Follow-up analysis: run all baselines (WT-only, one-hot AA, FoldX ΔΔG,
AlphaMissense) AND the delta probe under BOTH gene-split and family-split CV,
using the cached embeddings and Pfam map from a prior `experiment.py` run.

Headline question: does the WT-only baseline's macro-F1 (~0.58 under gene-split)
survive family-split CV? If not, the apparent mechanism signal is paralog/
homology leakage from gene-split alone.

Usage:
    python family_split_baselines.py --run_dir run_0 --model esm2_t33_650M_UR50D
"""

import argparse
import json
import os
from collections import Counter

import numpy as np

from esm2_mech.utils.paths import (
    EMB_WT_MEAN, EMB_MUT_MEAN, EMB_WT_POS, EMB_MUT_POS,
    SEQUENCES_JSON, VARIANTS_JSON, RESULTS_DIR, PFAM_JSON,
)

OUT_DIR = RESULTS_DIR
from esm2_mech.utils.sequences import window_sequence, apply_missense, build_wt_mut_onehot
from esm2_mech.utils.constants import (
    seed_result_filename,
    MECHANISM_CLASSES,
    BOOTSTRAP_N_RESAMPLES,
)
from esm2_mech.utils.splits import gene_split_cv, family_split_cv
from esm2_mech.utils.embed import unpack_run_data
from esm2_mech.utils.metrics import align_proba
from esm2_mech.utils.bootstrap import bootstrap_mechanism_metrics, label_permutation_pvalue

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
import functools

PCA_COMPONENTS = 256  # applied to embedding features (dim > PCA_COMPONENTS)

print = functools.partial(print, flush=True)


def run_probe_on_splits(X, y, splits, genes=None, seed=42):
    """Run logistic regression across a pre-computed list of (train, test) splits.

    Returns (agg, oof). `agg` is the per-fold mean ± std metric dict (unchanged).
    `oof` collects the out-of-fold test predictions for dependency-aware inference:
    {"y_true", "proba" (aligned to MECHANISM_CLASSES), "genes"}, or None when `genes`
    is not supplied or no fold ran. Each variant appears once (one held-out fold).
    """
    fold_results = []
    oof_y, oof_proba, oof_genes = [], [], []
    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if len(set(y_train)) < 2:
            continue
        clf = LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", random_state=seed
        )
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        pred = clf.predict(X_test)

        fm = {
            "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0))
        }
        for i, cls in enumerate(clf.classes_):
            y_bin = (y_test == cls).astype(int)
            if y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
                fm[f"auroc_{cls}"] = float(roc_auc_score(y_bin, proba[:, i]))
                prec, rec, _ = precision_recall_curve(y_bin, proba[:, i])
                fm[f"pr_auc_{cls}"] = float(auc(rec, prec))
        fold_results.append(fm)

        if genes is not None:
            oof_y.append(y_test)
            oof_proba.append(align_proba(proba, clf.classes_, MECHANISM_CLASSES))
            oof_genes.append(genes[test_idx])

    if not fold_results:
        return {"error": "insufficient data"}, None

    agg = {}
    all_keys = set().union(*[set(f.keys()) for f in fold_results])
    for key in all_keys:
        vals = [f[key] for f in fold_results if key in f and not np.isnan(f[key])]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
    agg["n_folds"] = len(fold_results)

    oof = None
    if oof_y:
        oof = {
            "y_true": np.concatenate(oof_y),
            "proba": np.concatenate(oof_proba),
            "genes": np.concatenate(oof_genes),
        }
    return agg, oof


PERMUTATION_FEATURES = ("delta_mean", "wt_only_mean")  # headline features for the slow permutation test


def run(
    data: dict,
    out_dir: str,
    seed: int = 0,
    n_folds: int = 5,
    compute_ci: bool = True,
    n_boot: int = BOOTSTRAP_N_RESAMPLES,
    n_permutations: int = 0,
) -> dict:
    # ------------------------------------------------------------------
    # 1–3. Extract pre-loaded data
    # ------------------------------------------------------------------
    data = unpack_run_data(data)
    valid_variants = data["valid_variants"]
    emb_wt_mean = data["emb_wt_mean"]
    emb_mut_mean = data["emb_mut_mean"]
    emb_wt_pos = data["emb_wt_pos"]
    emb_mut_pos = data["emb_mut_pos"]
    labels_3class = data["labels_3class"]
    genes_arr = data["genes_arr"]
    foldx_ddg = data["foldx_ddg"]
    aa_wt_list = data["aa_wt_list"]
    aa_mut_list = data["aa_mut_list"]
    alphamissense_scores = data["alphamissense_scores"]
    deltas_mean = data["deltas_mean"]
    deltas_pos = data["deltas_pos"]
    print(f"Embedding shape: {emb_wt_mean.shape}")
    print(f"Valid variants: {len(valid_variants)}")
    print(f"Class distribution: {dict(Counter(labels_3class))}")
    print(f"Unique genes: {len(set(genes_arr))}")

    # ------------------------------------------------------------------
    # 4. Build feature matrices
    # ------------------------------------------------------------------
    # FoldX and AlphaMissense: restrict to variants with observed values only.
    # 0.0 is a plausible real ΔΔG — imputing NaN with 0.0 fabricates observations.
    foldx_mask = ~np.isnan(foldx_ddg)
    am_mask = ~np.isnan(alphamissense_scores)

    # Reduce high-dimensional embeddings with PCA before probing.
    # Fit PCA on the full set (no label leakage — PCA is unsupervised).
    print(f"\n=== PCA reduction to {PCA_COMPONENTS} components ===")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=seed)
    emb_wt_mean_pca = pca.fit_transform(emb_wt_mean)
    emb_mut_mean_pca = pca.transform(emb_mut_mean)
    deltas_mean_pca = pca.transform(deltas_mean)
    deltas_pos_pca = pca.transform(deltas_pos)
    print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    features: dict = {
        "wt_only_mean": emb_wt_mean_pca,
        "mut_only_mean": emb_mut_mean_pca,
        "delta_mean": deltas_mean_pca,
        "delta_per_residue": deltas_pos_pca,
        "wt_concat_mut": np.concatenate([emb_wt_mean_pca, emb_mut_mean_pca], axis=1),
        "onehot_aa": build_wt_mut_onehot(aa_wt_list, aa_mut_list),
    }

    # Masked baselines — rebuild splits on the observed subset so CV is valid
    if foldx_mask.sum() >= 20:
        features["foldx_ddg"] = (foldx_ddg[foldx_mask].reshape(-1, 1),
                                  labels_3class[foldx_mask], genes_arr[foldx_mask])
    else:
        print(f"  [skip foldx_ddg: only {foldx_mask.sum()} observed values]")

    if am_mask.sum() >= 20:
        features["alphamissense"] = (alphamissense_scores[am_mask].reshape(-1, 1),
                                     labels_3class[am_mask], genes_arr[am_mask])
    else:
        print(f"  [skip alphamissense: only {am_mask.sum()} observed values]")

    # ------------------------------------------------------------------
    # 5. Build splits — gene-split AND family-split
    # ------------------------------------------------------------------
    print("\n=== Building CV splits ===")
    if not PFAM_JSON.exists():
        raise FileNotFoundError(f"{PFAM_JSON} not found — run fetch_data/fetch_annotations --step pfam first")
    with open(PFAM_JSON) as f:
        pfam_map = json.load(f)
    gene_splits = gene_split_cv(genes_arr, n_folds=n_folds, seed=seed)
    family_splits = family_split_cv(
        genes_arr, pfam_map, n_folds=n_folds, seed=seed
    )

    print(f"Gene-split folds:   {len(gene_splits)}")
    print(f"Family-split folds: {len(family_splits)}")
    # Count families actually represented in the dataset genes, not every family
    # in pfam_map (which includes families with no gene in this variant set).
    n_families = len({pfam_map.get(g) for g in genes_arr} - {None})
    print(f"Unique Pfam families: {n_families}")

    # ------------------------------------------------------------------
    # 6. Run every feature under both split schemes
    # ------------------------------------------------------------------
    results = {
        "n_variants": len(valid_variants),
        "n_genes": int(len(set(genes_arr))),
        "n_families": n_families,
        "class_distribution": dict(Counter(labels_3class)),
        "gene_split": {},
        "family_split": {},
    }

    for name, entry in features.items():
        # Masked baselines carry their own subset labels/genes; others use full arrays
        if isinstance(entry, tuple):
            X, feat_labels, feat_genes = entry
            feat_gene_splits = gene_split_cv(feat_genes, n_folds=n_folds, seed=seed)
            feat_family_splits = family_split_cv(feat_genes, pfam_map, n_folds=n_folds, seed=seed)
        else:
            X, feat_labels, feat_genes = entry, labels_3class, genes_arr
            feat_gene_splits, feat_family_splits = gene_splits, family_splits

        if X.shape[1] > 0 and np.allclose(X.std(), 0):
            print(f"  [skip {name}: zero variance]")
            continue
        print(f"\n--- {name} (dim={X.shape[1]}, n={len(feat_labels)}) ---")

        gs, gs_oof = run_probe_on_splits(X, feat_labels, feat_gene_splits, feat_genes, seed=seed)
        if compute_ci and gs_oof is not None:
            gs["ci"] = bootstrap_mechanism_metrics(
                gs_oof["y_true"], gs_oof["proba"], gs_oof["genes"],
                n_resamples=n_boot, seed=seed,
            )
        results["gene_split"][name] = gs
        print(
            f"  gene-split   macro-F1 = {gs.get('macro_f1_mean', float('nan')):.3f} "
            f"± {gs.get('macro_f1_std', float('nan')):.3f}  "
            f"(GOF AUROC {gs.get('auroc_GOF_mean', float('nan')):.3f}, "
            f"DN {gs.get('auroc_DN_mean', float('nan')):.3f}, "
            f"LOF {gs.get('auroc_LOF_mean', float('nan')):.3f})"
        )

        if feat_family_splits:
            fs, fs_oof = run_probe_on_splits(X, feat_labels, feat_family_splits, feat_genes, seed=seed)
            if compute_ci and fs_oof is not None:
                fs["ci"] = bootstrap_mechanism_metrics(
                    fs_oof["y_true"], fs_oof["proba"], fs_oof["genes"],
                    n_resamples=n_boot, seed=seed,
                )
            if n_permutations > 0 and name in PERMUTATION_FEATURES:
                def _family_macro_f1(perm_labels, _X=X, _splits=feat_family_splits, _genes=feat_genes):
                    agg_perm, _ = run_probe_on_splits(_X, perm_labels, _splits, _genes, seed=seed)
                    return agg_perm.get("macro_f1_mean")

                fs["permutation"] = label_permutation_pvalue(
                    _family_macro_f1, feat_labels, groups=feat_genes,
                    n_permutations=n_permutations, seed=seed,
                )
                print(
                    f"  family-split permutation p = {fs['permutation'].get('p_value')} "
                    f"(observed {fs['permutation'].get('observed')}, "
                    f"null mean {fs['permutation'].get('null_mean')})"
                )
            results["family_split"][name] = fs
            delta_macro = gs.get("macro_f1_mean", float("nan")) - fs.get(
                "macro_f1_mean", float("nan")
            )
            print(
                f"  family-split macro-F1 = {fs.get('macro_f1_mean', float('nan')):.3f} "
                f"± {fs.get('macro_f1_std', float('nan')):.3f}  "
                f"(GOF AUROC {fs.get('auroc_GOF_mean', float('nan')):.3f}, "
                f"DN {fs.get('auroc_DN_mean', float('nan')):.3f}, "
                f"LOF {fs.get('auroc_LOF_mean', float('nan')):.3f})"
            )
            print(
                f"  Δ(gene − family) macro-F1 = {delta_macro:+.3f}  "
                f"← positive ⇒ homology leakage"
            )

    # ------------------------------------------------------------------
    # 7. Write results
    # ------------------------------------------------------------------
    out_path = os.path.join(out_dir, seed_result_filename(seed))
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")
    return results


def _load_data_inline() -> dict:
    """Load all data needed by run(). Used by main() for standalone execution."""
    from esm2_mech.utils.data import load_variants as _load_variants
    from esm2_mech.experiments.mechanism.mechanism_delta_probe import _load_alphamissense_scores

    if not VARIANTS_JSON.exists():
        raise FileNotFoundError(
            f"{VARIANTS_JSON} not found — run fetch_data/fetch_variants.py --step merge first"
        )
    variants = _load_variants(VARIANTS_JSON)

    if not SEQUENCES_JSON.exists():
        raise FileNotFoundError(f"{SEQUENCES_JSON} not found — run fetch_data/fetch_sequences first")
    with open(SEQUENCES_JSON) as f:
        seq_cache = json.load(f)

    valid_variants = []
    for v in variants:
        uid = v["uniprot_id"]
        if uid not in seq_cache:
            continue
        wt_win, new_pos, _ = window_sequence(seq_cache[uid], v["aa_pos"])
        mut_win = apply_missense(wt_win, new_pos, v["aa_wt"], v["aa_mut"])
        if mut_win is None:
            continue
        valid_variants.append(v)

    print(f"Valid variant pairs: {len(valid_variants)}")

    for path in [EMB_WT_MEAN, EMB_MUT_MEAN, EMB_WT_POS, EMB_MUT_POS]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing cached embedding: {path}")

    emb_wt_mean = np.load(EMB_WT_MEAN)
    emb_mut_mean = np.load(EMB_MUT_MEAN)
    emb_wt_pos = np.load(EMB_WT_POS)
    emb_mut_pos = np.load(EMB_MUT_POS)

    labels_3class = np.array([v["label_3class"] for v in valid_variants])
    labels_4class = np.array([v["mechanism"] for v in valid_variants])
    genes_arr = np.array([v["gene"] for v in valid_variants])
    foldx_ddg = np.array(
        [v["foldx_ddg"] if v["foldx_ddg"] is not None else np.nan for v in valid_variants]
    )
    aa_wt_list = [v["aa_wt"] for v in valid_variants]
    aa_mut_list = [v["aa_mut"] for v in valid_variants]

    alphamissense_scores = _load_alphamissense_scores(valid_variants)

    return {
        "valid_variants": valid_variants,
        "emb_wt_mean": emb_wt_mean,
        "emb_mut_mean": emb_mut_mean,
        "emb_wt_pos": emb_wt_pos,
        "emb_mut_pos": emb_mut_pos,
        "labels_3class": labels_3class,
        "labels_4class": labels_4class,
        "genes_arr": genes_arr,
        "foldx_ddg": foldx_ddg,
        "aa_wt_list": aa_wt_list,
        "aa_mut_list": aa_mut_list,
        "alphamissense_scores": alphamissense_scores,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--no_ci", action="store_true", help="skip cluster-bootstrap CIs")
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    parser.add_argument(
        "--n_permutations", type=int, default=0,
        help="label-permutation reps for headline features (0 = skip; slow, refits per rep)",
    )
    args = parser.parse_args()
    data = _load_data_inline()
    run(
        data=data, out_dir=str(OUT_DIR), seed=args.seed, n_folds=args.n_folds,
        compute_ci=not args.no_ci, n_boot=args.n_boot, n_permutations=args.n_permutations,
    )


if __name__ == "__main__":
    main()
