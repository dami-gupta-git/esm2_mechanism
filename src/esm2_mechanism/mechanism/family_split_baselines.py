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

from esm2_mechanism.embeddings.esm2_mechanism import _load_data, _load_alphamissense_scores, ESM2_MODEL_650M
from esm2_mechanism.utils_sequences import build_sequence_cache, window_sequence, apply_missense, fetch_pfam_families
from esm2_mechanism.utils_probes import gene_split_cv, family_split_cv

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
import functools
print = functools.partial(print, flush=True)


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {a: i for i, a in enumerate(AA_ORDER)}


def run_probe_on_splits(X, y, splits, seed=42):
    """Run logistic regression across a pre-computed list of (train, test) splits."""
    fold_results = []
    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if len(set(y_train)) < 2:
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                  random_state=seed)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        pred = clf.predict(X_test)

        fm = {"macro_f1": float(f1_score(y_test, pred, average="macro",
                                          zero_division=0))}
        for i, cls in enumerate(clf.classes_):
            y_bin = (y_test == cls).astype(int)
            if y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
                fm[f"auroc_{cls}"] = float(roc_auc_score(y_bin, proba[:, i]))
                prec, rec, _ = precision_recall_curve(y_bin, proba[:, i])
                fm[f"pr_auc_{cls}"] = float(auc(rec, prec))
        fold_results.append(fm)

    if not fold_results:
        return {"error": "insufficient data"}

    agg = {}
    all_keys = set().union(*[set(f.keys()) for f in fold_results])
    for key in all_keys:
        vals = [f[key] for f in fold_results if key in f and not np.isnan(f[key])]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
    agg["n_folds"] = len(fold_results)
    return agg


def build_onehot(aa_wt_list, aa_mut_list):
    n = len(aa_wt_list)
    onehot = np.zeros((n, 40), dtype=np.float32)
    for i, (wt, mut) in enumerate(zip(aa_wt_list, aa_mut_list)):
        if wt.upper() in AA_INDEX:
            onehot[i, AA_INDEX[wt.upper()]] = 1.0
        if mut.upper() in AA_INDEX:
            onehot[i, 20 + AA_INDEX[mut.upper()]] = 1.0
    return onehot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="run_0",
                        help="Directory containing data/ with cached embeddings")
    parser.add_argument("--model", type=str, default=ESM2_MODEL_650M)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--out", type=str, default="family_split_baselines.json")
    args = parser.parse_args()

    data_dir = os.path.join(args.run_dir, "data")

    # ------------------------------------------------------------------
    # 1. Rebuild the valid_variants list IDENTICALLY to experiment.py
    # ------------------------------------------------------------------
    print("=== Loading cached dataset and sequences ===")
    variants = _load_data(data_dir)

    seq_cache = build_sequence_cache(variants, data_dir)

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

    print(f"Valid variants: {len(valid_variants)}")

    # ------------------------------------------------------------------
    # 2. Load cached embeddings
    # ------------------------------------------------------------------
    emb_dir = os.path.join(data_dir, "embeddings", args.model)
    emb_cache_wt     = os.path.join(emb_dir, "embeddings_wt_mean.npy")
    emb_cache_mut    = os.path.join(emb_dir, "embeddings_mut_mean.npy")
    emb_cache_wt_pos = os.path.join(emb_dir, "embeddings_wt_pos.npy")
    emb_cache_mut_pos = os.path.join(emb_dir, "embeddings_mut_pos.npy")

    for p in [emb_cache_wt, emb_cache_mut]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing cached embedding: {p}\n"
                f"Run experiment.py first to populate the cache."
            )

    emb_wt_mean = np.load(emb_cache_wt)
    emb_mut_mean = np.load(emb_cache_mut)
    emb_wt_pos = (np.load(emb_cache_wt_pos)
                  if os.path.exists(emb_cache_wt_pos) else emb_wt_mean)
    emb_mut_pos = (np.load(emb_cache_mut_pos)
                   if os.path.exists(emb_cache_mut_pos) else emb_mut_mean)

    deltas_mean = emb_mut_mean - emb_wt_mean
    deltas_pos = emb_mut_pos - emb_wt_pos
    print(f"Embedding shape: {emb_wt_mean.shape}")

    assert len(valid_variants) == emb_wt_mean.shape[0], (
        f"Variant count mismatch: {len(valid_variants)} variants vs "
        f"{emb_wt_mean.shape[0]} embeddings. The cached embeddings were built "
        f"from a different filtering — re-run experiment.py to align."
    )

    # ------------------------------------------------------------------
    # 3. Build label / gene / aux arrays
    # ------------------------------------------------------------------
    labels_3class = np.array([v["label_3class"] for v in valid_variants])
    genes_arr = np.array([v["gene"] for v in valid_variants])
    foldx_ddg = np.array([v["foldx_ddg"] if v["foldx_ddg"] is not None else np.nan
                          for v in valid_variants])
    aa_wt_list = [v["aa_wt"] for v in valid_variants]
    aa_mut_list = [v["aa_mut"] for v in valid_variants]

    print(f"Class distribution: {dict(Counter(labels_3class))}")
    print(f"Unique genes: {len(set(genes_arr))}")

    alphamissense_scores = _load_alphamissense_scores(valid_variants, data_dir)

    # ------------------------------------------------------------------
    # 4. Build feature matrices
    # ------------------------------------------------------------------
    features = {
        "wt_only_mean":        emb_wt_mean,
        "mut_only_mean":       emb_mut_mean,
        "delta_mean":          deltas_mean,
        "delta_per_residue":   deltas_pos,
        "wt_concat_mut":       np.concatenate([emb_wt_mean, emb_mut_mean], axis=1),
        "onehot_aa":           build_onehot(aa_wt_list, aa_mut_list),
        "foldx_ddg":           np.nan_to_num(foldx_ddg, nan=0.0).reshape(-1, 1),
        "alphamissense":       np.nan_to_num(alphamissense_scores, nan=0.0).reshape(-1, 1),
    }

    # ------------------------------------------------------------------
    # 5. Build splits — gene-split AND family-split
    # ------------------------------------------------------------------
    print("\n=== Building CV splits ===")
    pfam_map = fetch_pfam_families(valid_variants, data_dir)
    gene_splits = gene_split_cv(genes_arr, n_folds=args.n_folds, seed=args.seed)
    family_splits = family_split_cv(genes_arr, pfam_map,
                                     n_folds=args.n_folds, seed=args.seed)

    print(f"Gene-split folds:   {len(gene_splits)}")
    print(f"Family-split folds: {len(family_splits)}")
    n_families = len(set(v for v in pfam_map.values() if v is not None))
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

    for name, X in features.items():
        if X.shape[1] > 0 and np.allclose(X.std(), 0):
            print(f"  [skip {name}: zero variance]")
            continue
        print(f"\n--- {name} (dim={X.shape[1]}) ---")

        gs = run_probe_on_splits(X, labels_3class, gene_splits, seed=args.seed)
        results["gene_split"][name] = gs
        print(f"  gene-split   macro-F1 = {gs.get('macro_f1_mean', float('nan')):.3f} "
              f"± {gs.get('macro_f1_std', float('nan')):.3f}  "
              f"(GOF AUROC {gs.get('auroc_GOF_mean', float('nan')):.3f}, "
              f"DN {gs.get('auroc_DN_mean', float('nan')):.3f}, "
              f"LOF {gs.get('auroc_LOF_mean', float('nan')):.3f})")

        if family_splits:
            fs = run_probe_on_splits(X, labels_3class, family_splits, seed=args.seed)
            results["family_split"][name] = fs
            delta_macro = (gs.get("macro_f1_mean", float("nan"))
                           - fs.get("macro_f1_mean", float("nan")))
            print(f"  family-split macro-F1 = {fs.get('macro_f1_mean', float('nan')):.3f} "
                  f"± {fs.get('macro_f1_std', float('nan')):.3f}  "
                  f"(GOF AUROC {fs.get('auroc_GOF_mean', float('nan')):.3f}, "
                  f"DN {fs.get('auroc_DN_mean', float('nan')):.3f}, "
                  f"LOF {fs.get('auroc_LOF_mean', float('nan')):.3f})")
            print(f"  Δ(gene − family) macro-F1 = {delta_macro:+.3f}  "
                  f"← positive ⇒ homology leakage")

    # ------------------------------------------------------------------
    # 7. Write results
    # ------------------------------------------------------------------
    out_path = os.path.join(args.run_dir, args.out)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")

    # ------------------------------------------------------------------
    # 8. Headline interpretation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("HEADLINE")
    print("=" * 60)
    wt_gene   = results["gene_split"].get("wt_only_mean", {}).get("macro_f1_mean", float("nan"))
    wt_family = results["family_split"].get("wt_only_mean", {}).get("macro_f1_mean", float("nan"))
    delta_gene   = results["gene_split"].get("delta_mean", {}).get("macro_f1_mean", float("nan"))
    delta_family = results["family_split"].get("delta_mean", {}).get("macro_f1_mean", float("nan"))

    def _fmt(val):
        return f"{val:.3f}" if not np.isnan(val) else "n/a"

    def _fmt_delta(a, b):
        return f"{a - b:+.3f}" if not (np.isnan(a) or np.isnan(b)) else "n/a"

    print(f"WT-only macro-F1:  gene-split {_fmt(wt_gene)}  →  family-split {_fmt(wt_family)}  "
          f"(Δ = {_fmt_delta(wt_gene, wt_family)})")
    print(f"Delta   macro-F1:  gene-split {_fmt(delta_gene)}  →  family-split {_fmt(delta_family)}  "
          f"(Δ = {_fmt_delta(delta_gene, delta_family)})")
    print(f"Chance (3-class):  0.333\n")

    if not family_splits:
        print("  ⇒ Family-split CV was infeasible (< 10 Pfam families annotated).")
        print("    Gene-split results only — homology leakage cannot be assessed.")
    elif not np.isnan(wt_family):
        if wt_family > 0.50:
            print("  ⇒ WT-only signal SURVIVES family-split — real protein-family→mechanism")
            print("    association in ESM-2 embeddings.")
        elif wt_family > 0.40:
            print("  ⇒ WT-only signal PARTIALLY survives — some real signal + some leakage.")
        else:
            print("  ⇒ WT-only signal COLLAPSES under family-split — apparent mechanism")
            print("    classification was largely paralog/homology leakage.")


if __name__ == "__main__":
    main()
