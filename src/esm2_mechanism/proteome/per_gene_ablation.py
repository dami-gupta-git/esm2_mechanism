"""
T2: Per-gene scoring for V1, V2, V3
T4: V2 feature-class ablation

T2 — Per-gene scoring
---------------------
Existing results 11-13 score per-variant: each of the ~19k variants votes
independently, so high-variant genes (KCNQ2 with 240 DN variants, SCN1A with
374 GOF variants) dominate the macro-F1. Re-scoring per-gene gives each gene
one vote regardless of how many variants it has.

Method:
  V1  ESM-2 delta: for each gene, mean the variant-level predicted probability
      vectors across all variants of that gene, then argmax.
  V2  Proteome features: one feature vector per gene → one prediction per gene.
      No aggregation needed.
  V3  Concat: mean the variant-level predicted probability vectors.

All under 5-fold family-split CV, 5 seeds. Reports macro-F1 per-gene.

T4 — V2 Feature-class ablation
-------------------------------
Drop one feature class at a time from V2 (LogReg + MLP on proteome features).
Feature classes:
  constraint   pLI, LOEUF, mis_z  (+ all _missing, _familyresid, _familyresid_missing)
  paralogs     paralog_count      (+ derived)
  expression   tissue_specificity_tau (+ derived)
  abundance    log_abundance_ppm  (+ derived)
  interactome  PPI_degree         (+ derived)
  dosage       HI_score, TS_score (+ derived)

Reports delta-F1 = V2_full - V2_minus_class for each class.
Per DN class separately (most scientifically interesting).

Usage
-----
    python scripts/per_gene_ablation.py
    python scripts/per_gene_ablation.py --seed 0
    python scripts/per_gene_ablation.py --skip-t2   # only T4
    python scripts/per_gene_ablation.py --skip-t4   # only T2
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from esm2_mechanism.utils_probes import (
    family_split_indices,
    compute_metrics,
    aggregate_folds,
    align_proba,
)
from esm2_mechanism.utils_paths import DATA_DIR, RESULTS_DIR, PROJECT_ROOT
import functools

print = functools.partial(print, flush=True)

warnings.filterwarnings("ignore")

EMB_DIR = DATA_DIR / "embeddings"

MERGED_VALID_VARIANTS = EMB_DIR / "merged_valid_variants.json"
MERGED_WT_MEAN = EMB_DIR / "merged_embeddings_wt_mean.npy"
MERGED_MUT_MEAN = EMB_DIR / "merged_embeddings_mut_mean.npy"
PROTEOME_FEATURES = DATA_DIR / "proteome_features_aligned.npy"
PROTEOME_COLS = DATA_DIR / "proteome_feature_columns.json"
MERGED_GENE_LIST = DATA_DIR / "merged_gene_list.tsv"
PFAM_FAMILIES = DATA_DIR / "pfam_families.json"

CLASSES = ["GOF", "DN", "LOF"]
N_FOLDS = 5

# ---------------------------------------------------------------------------
# Feature class definitions for T4
# ---------------------------------------------------------------------------
FEATURE_CLASSES = {
    "constraint": ["pLI", "LOEUF", "mis_z"],
    "paralogs": ["paralog_count"],
    "expression": ["tissue_specificity_tau"],
    "abundance": ["log_abundance_ppm"],
    "interactome": ["PPI_degree"],
    "dosage": ["HI_score", "TS_score"],
}


def get_drop_indices(feature_names: list[str], base_features: list[str]) -> list[int]:
    """Return column indices to drop for a given set of base feature names.
    Drops the raw column plus all derived variants (_missing, _familyresid, etc.)
    """
    drop = []
    for i, name in enumerate(feature_names):
        for bf in base_features:
            if name == bf or name.startswith(bf + "_"):
                drop.append(i)
                break
    return drop


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all():
    with open(MERGED_VALID_VARIANTS) as f:
        variants = json.load(f)
    labels = np.array([v["label_3class"] for v in variants])
    genes = np.array([v["gene"] for v in variants])
    n = len(variants)
    wt = np.load(MERGED_WT_MEAN)[:n]
    mut = np.load(MERGED_MUT_MEAN)[:n]
    delta = (mut - wt).astype(np.float32)

    with open(PFAM_FAMILIES) as f:
        pfam_map = json.load(f)

    # Build gene→proteome row index
    gene_to_row: dict[str, int] = {}
    idx = 0
    with open(MERGED_GENE_LIST) as f:
        f.readline()
        for line in f:
            g = line.strip().split("\t")[0]
            if g and g not in gene_to_row:
                gene_to_row[g] = idx
                idx += 1

    prot_matrix = np.load(PROTEOME_FEATURES).astype(np.float32)
    X_prot_var = np.zeros((n, prot_matrix.shape[1]), dtype=np.float32)
    for i, g in enumerate(genes):
        row = gene_to_row.get(g)
        if row is not None:
            X_prot_var[i] = prot_matrix[row]

    with open(PROTEOME_COLS) as f:
        col_meta = json.load(f)
    feature_names = col_meta["numerical_columns"]

    # Also load gene-level data (for V2 per-gene)
    gene_list_df = pd.read_csv(MERGED_GENE_LIST, sep="\t")
    mech_map = {"LOF": "LOF", "HI": "LOF", "GOF": "GOF", "DN": "DN", "AR": None}
    gene_list_df["mech3"] = gene_list_df["mechanism"].map(mech_map)
    gene_level = gene_list_df[gene_list_df["mech3"].notna()].copy()
    gene_level["pfam_family"] = gene_level["gene"].map(pfam_map)
    gene_level_idx = np.array([gene_to_row.get(g, -1) for g in gene_level["gene"]])
    valid_gene_mask = gene_level_idx >= 0
    gene_level = gene_level[valid_gene_mask].copy()
    gene_level_idx = gene_level_idx[valid_gene_mask]
    X_prot_gene = prot_matrix[gene_level_idx]

    return (
        variants,
        labels,
        genes,
        delta,
        X_prot_var,
        pfam_map,
        gene_level,
        X_prot_gene,
        prot_matrix,
        gene_to_row,
        feature_names,
    )


# ---------------------------------------------------------------------------
# T2 — Per-gene scoring
# ---------------------------------------------------------------------------
def run_per_gene_cv(
    *,
    # variant-level arrays (for V1 and V3 aggregation)
    delta: np.ndarray,
    X_prot_var: np.ndarray,
    var_labels: np.ndarray,
    var_genes: np.ndarray,
    # gene-level arrays (for V2)
    gene_level_df: pd.DataFrame,
    X_prot_gene: np.ndarray,
    pfam_map: dict,
    seed: int,
    le: LabelEncoder,
) -> dict:
    """
    For each CV fold:
      - Train V1 MLP on variant-level delta (train variants)
      - Train V2 LogReg on gene-level proteome (train genes)
      - Train V3 MLP on variant-level concat (train variants)
      - For test: aggregate per-variant probabilities to per-gene by mean
      - Compute macro-F1 per-gene (one vote per gene)
    """
    y_var = le.transform(var_labels)
    gene_pfam_var = np.array([pfam_map.get(g) for g in var_genes])
    has_fam_var = np.array([p is not None for p in gene_pfam_var])
    fam_idx = np.where(has_fam_var)[0]

    # Restrict variant arrays to those with Pfam (same as proteome_mechanism.py)
    delta_f = delta[fam_idx]
    X_prot_f = X_prot_var[fam_idx]
    X_concat_f = np.concatenate([delta_f, X_prot_f], axis=1)
    y_f = y_var[fam_idx]
    genes_f = var_genes[fam_idx]
    groups_f = gene_pfam_var[fam_idx]

    # Gene-level: restrict to genes with Pfam annotation
    gene_pfam_g = np.array([pfam_map.get(g) for g in gene_level_df["gene"].values])
    has_fam_g = np.array([p is not None for p in gene_pfam_g])
    gene_df_f = gene_level_df[has_fam_g].copy()
    X_prot_g_f = X_prot_gene[has_fam_g]
    y_g_f = le.transform(gene_df_f["mech3"].values)
    groups_g = gene_pfam_g[has_fam_g]

    v1_folds, v2_folds, v3_folds = [], [], []

    for fold_i, (tr_g, te_g) in enumerate(
        family_split_indices(groups_g, N_FOLDS, seed)
    ):
        # Gene-level train/test split
        train_genes_set = set(gene_df_f.iloc[tr_g]["gene"].values)
        test_genes_set = set(gene_df_f.iloc[te_g]["gene"].values)

        if not test_genes_set:
            continue

        # --- V2: proteome features, gene-level ---
        X_tr_g, y_tr_g = X_prot_g_f[tr_g], y_g_f[tr_g]
        X_te_g, y_te_g = X_prot_g_f[te_g], y_g_f[te_g]

        if len(set(y_tr_g.tolist())) < len(CLASSES) or len(set(y_te_g.tolist())) < 2:
            continue

        sc2 = StandardScaler().fit(X_tr_g)
        lr2 = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed
        )
        lr2.fit(sc2.transform(X_tr_g), y_tr_g)
        pr2 = lr2.predict_proba(sc2.transform(X_te_g))
        # Align to CLASSES order
        pr2_al = np.zeros((len(te_g), len(CLASSES)))
        for ci, c in enumerate(lr2.classes_):
            pr2_al[:, c] = pr2[:, ci]
        pd2 = pr2_al.argmax(axis=1)
        # --- V1 and V3: variant-level train, gene-level test ---
        # Train: all variants from train genes
        tr_var_mask = np.array([g in train_genes_set for g in genes_f])
        # Test: variants from test genes — aggregate per gene
        te_var_mask = np.array([g in test_genes_set for g in genes_f])

        if tr_var_mask.sum() < 10 or te_var_mask.sum() < 5:
            continue

        v2_folds.append(compute_metrics(y_te_g, pd2, pr2_al))

        X_tr_d, y_tr_d = delta_f[tr_var_mask], y_f[tr_var_mask]
        X_tr_c, y_tr_c = X_concat_f[tr_var_mask], y_f[tr_var_mask]

        # Oversample for MLP
        def oversample(X, y, seed):
            counts = np.bincount(y, minlength=len(CLASSES))
            mc = counts.max()
            rng = np.random.RandomState(seed)
            idx = []
            for c in range(len(CLASSES)):
                ci = np.where(y == c)[0]
                if len(ci) == 0:
                    continue
                rep = mc // len(ci)
                rem = mc % len(ci)
                idx.append(np.tile(ci, rep))
                if rem > 0:
                    idx.append(rng.choice(ci, rem, replace=False))
            idx = np.concatenate(idx)
            rng.shuffle(idx)
            return X[idx], y[idx]

        sc1 = StandardScaler().fit(X_tr_d)
        X_tr_d_s = sc1.transform(X_tr_d)
        X_bal_d, y_bal_d = oversample(X_tr_d_s, y_tr_d, seed)
        mlp1 = MLPClassifier(
            (256, 64),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed,
        )
        mlp1.fit(X_bal_d, y_bal_d)

        sc3 = StandardScaler().fit(X_tr_c)
        X_tr_c_s = sc3.transform(X_tr_c)
        X_bal_c, y_bal_c = oversample(X_tr_c_s, y_tr_c, seed)
        mlp3 = MLPClassifier(
            (256, 64),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed,
        )
        mlp3.fit(X_bal_c, y_bal_c)

        # Aggregate per-gene predictions for test genes
        test_genes_list = sorted(test_genes_set)
        y_gene_true, y_gene_pred1, y_gene_pred3 = [], [], []
        pr_gene1 = []
        pr_gene3 = []

        for gene in test_genes_list:
            gene_mask = genes_f[te_var_mask] == gene
            if gene_mask.sum() == 0:
                continue
            # True label: gene-level (take first, all same within gene)
            gene_var_idx = np.where(te_var_mask)[0][gene_mask]
            y_gene_true.append(y_f[gene_var_idx[0]])

            # V1
            X_g_d = sc1.transform(delta_f[gene_var_idx])
            pr1_g = align_proba(
                mlp1.predict_proba(X_g_d), mlp1.classes_, len(CLASSES)
            ).mean(0)
            pr_gene1.append(pr1_g)
            y_gene_pred1.append(pr1_g.argmax())

            # V3
            X_g_c = sc3.transform(X_concat_f[gene_var_idx])
            pr3_g = align_proba(
                mlp3.predict_proba(X_g_c), mlp3.classes_, len(CLASSES)
            ).mean(0)
            pr_gene3.append(pr3_g)
            y_gene_pred3.append(pr3_g.argmax())

        y_gene_true = np.array(y_gene_true)
        y_gene_pred1 = np.array(y_gene_pred1)
        y_gene_pred3 = np.array(y_gene_pred3)
        pr_gene1 = np.array(pr_gene1)
        pr_gene3 = np.array(pr_gene3)

        if len(set(y_gene_true.tolist())) < 2:
            continue

        v1_folds.append(compute_metrics(y_gene_true, y_gene_pred1, pr_gene1))
        v3_folds.append(compute_metrics(y_gene_true, y_gene_pred3, pr_gene3))

        print(
            f"  [T2 seed={seed}] Fold {fold_i+1}: "
            f"V1={v1_folds[-1]['macro_f1']:.3f}  "
            f"V2={v2_folds[-1]['macro_f1']:.3f}  "
            f"V3={v3_folds[-1]['macro_f1']:.3f}"
        )

    return {
        "V1_per_gene": aggregate_folds(v1_folds) if v1_folds else {},
        "V2_per_gene": aggregate_folds(v2_folds) if v2_folds else {},
        "V3_per_gene": aggregate_folds(v3_folds) if v3_folds else {},
    }


# ---------------------------------------------------------------------------
# T4 — V2 Feature-class ablation
# ---------------------------------------------------------------------------
def run_v2_ablation(
    X_prot_gene: np.ndarray,
    y_gene: np.ndarray,
    groups_gene: np.ndarray,
    feature_names: list[str],
    seed: int,
    le: LabelEncoder,
) -> dict:
    """Run V2 LogReg with and without each feature class. Return delta-F1."""

    def run_logreg_cv(X, y, groups):
        folds = []
        for tr, te in family_split_indices(groups, N_FOLDS, seed):
            X_tr, y_tr = X[tr], y[tr]
            X_te, y_te = X[te], y[te]
            if len(set(y_tr.tolist())) < len(CLASSES) or len(set(y_te.tolist())) < 2:
                continue
            sc = StandardScaler().fit(X_tr)
            lr = LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=seed
            )
            lr.fit(sc.transform(X_tr), y_tr)
            pr_al = align_proba(
                lr.predict_proba(sc.transform(X_te)), lr.classes_, len(CLASSES)
            )
            pd_ = pr_al.argmax(axis=1)
            folds.append(compute_metrics(y_te, pd_, pr_al))
        return aggregate_folds(folds) if folds else {"macro_f1_mean": float("nan")}

    full_result = run_logreg_cv(X_prot_gene, y_gene, groups_gene)
    full_f1 = full_result["macro_f1_mean"]
    print(f"  [T4 seed={seed}] V2 FULL: macro_f1={full_f1:.4f}")

    ablation_results = {"FULL": full_result}

    for cls_name, base_feats in FEATURE_CLASSES.items():
        drop_idx = get_drop_indices(feature_names, base_feats)
        keep_idx = [i for i in range(X_prot_gene.shape[1]) if i not in drop_idx]
        X_abl = X_prot_gene[:, keep_idx]
        res = run_logreg_cv(X_abl, y_gene, groups_gene)
        delta_f1 = full_f1 - res["macro_f1_mean"]
        # DN AUROC delta — missing AUROC must NOT be coerced to 0.0
        # (a 0.0 default would read as "no change" and silently mask the failure).
        dn_full = full_result.get("auroc_DN_mean")
        dn_abl = res.get("auroc_DN_mean")
        if dn_full is None or dn_abl is None:
            delta_dn = float("nan")
        else:
            delta_dn = dn_full - dn_abl
        ablation_results[cls_name] = {
            **res,
            "n_features_dropped": len(drop_idx),
            "n_features_kept": len(keep_idx),
            "delta_f1": float(delta_f1),
            "delta_auroc_DN": float(delta_dn),
        }
        print(
            f"  [T4 seed={seed}] minus {cls_name:12s}: "
            f"f1={res['macro_f1_mean']:.4f}  ΔF1={delta_f1:+.4f}  "
            f"ΔDN_AUROC={delta_dn:+.4f}"
        )

    return ablation_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-t2", action="store_true")
    parser.add_argument("--skip-t4", action="store_true")
    parser.add_argument("--out-dir", default="results/proteome_mechanism")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    (
        variants,
        var_labels,
        var_genes,
        delta,
        X_prot_var,
        pfam_map,
        gene_level_df,
        X_prot_gene,
        prot_matrix,
        gene_to_row,
        feature_names,
    ) = load_all()

    le = LabelEncoder()
    le.fit(CLASSES)

    # Gene-level arrays for T4
    gene_pfam_g = np.array([pfam_map.get(g) for g in gene_level_df["gene"].values])
    has_fam_g = np.array([p is not None for p in gene_pfam_g])
    gene_df_f = gene_level_df[has_fam_g].copy()
    X_prot_g_f = X_prot_gene[has_fam_g]
    y_g_f = le.transform(gene_df_f["mech3"].values)
    groups_g = gene_pfam_g[has_fam_g]

    seeds = [args.seed] if args.seed is not None else list(range(5))

    # -----------------------------------------------------------------------
    # T2 — Per-gene scoring
    # -----------------------------------------------------------------------
    if not args.skip_t2:
        print("\n=== T2: Per-gene scoring ===")
        t2_seed_results = []
        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            res = run_per_gene_cv(
                delta=delta,
                X_prot_var=X_prot_var,
                var_labels=var_labels,
                var_genes=var_genes,
                gene_level_df=gene_level_df,
                X_prot_gene=X_prot_gene,
                pfam_map=pfam_map,
                seed=seed,
                le=le,
            )
            res["seed"] = seed
            t2_seed_results.append(res)
            out_path = out_dir / f"per_gene_seed{seed}.json"
            with open(out_path, "w") as f:
                json.dump(res, f, indent=2)
            print(f"  Saved {out_path.name}")
            for v in ["V1_per_gene", "V2_per_gene", "V3_per_gene"]:
                f1 = res[v].get("macro_f1_mean", float("nan"))
                print(f"  {v}: macro_f1={f1:.4f}")

        # Aggregate T2
        t2_summary = {}
        for v in ["V1_per_gene", "V2_per_gene", "V3_per_gene"]:
            f1s = [
                r[v].get("macro_f1_mean")
                for r in t2_seed_results
                if r[v].get("macro_f1_mean") is not None
            ]
            t2_summary[v] = {
                "macro_f1_mean": float(np.mean(f1s)) if f1s else None,
                "macro_f1_std": float(np.std(f1s)) if f1s else None,
            }
            for cls in CLASSES:
                vals = [
                    r[v].get(f"auroc_{cls}_mean")
                    for r in t2_seed_results
                    if r[v].get(f"auroc_{cls}_mean") is not None
                ]
                t2_summary[v][f"auroc_{cls}_mean"] = (
                    float(np.mean(vals)) if vals else None
                )
                t2_summary[v][f"auroc_{cls}_std"] = (
                    float(np.std(vals)) if vals else None
                )

        t2_sum_path = out_dir / "per_gene_summary.json"
        with open(t2_sum_path, "w") as f:
            json.dump(t2_summary, f, indent=2)

        print("\n=== T2 SUMMARY (per-gene, mean ± std across seeds) ===")
        for v in ["V1_per_gene", "V2_per_gene", "V3_per_gene"]:
            m = t2_summary[v].get("macro_f1_mean")
            s = t2_summary[v].get("macro_f1_std")
            dn = t2_summary[v].get("auroc_DN_mean")
            print(
                f"  {v}: macro_f1={m:.4f}±{s:.4f}  DN_AUROC={dn:.3f}"
                if m
                else f"  {v}: N/A"
            )

    # -----------------------------------------------------------------------
    # T4 — V2 Feature-class ablation
    # -----------------------------------------------------------------------
    if not args.skip_t4:
        print("\n=== T4: V2 Feature-class ablation ===")
        t4_seed_results = []
        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            res = run_v2_ablation(X_prot_g_f, y_g_f, groups_g, feature_names, seed, le)
            res["seed"] = seed
            t4_seed_results.append(res)
            out_path = out_dir / f"v2_ablation_seed{seed}.json"
            with open(out_path, "w") as f:
                json.dump(res, f, indent=2)
            print(f"  Saved {out_path.name}")

        # Aggregate T4
        t4_summary = {"FULL": {}}
        full_f1s = [
            r["FULL"].get("macro_f1_mean")
            for r in t4_seed_results
            if r["FULL"].get("macro_f1_mean") is not None
        ]
        t4_summary["FULL"]["macro_f1_mean"] = (
            float(np.mean(full_f1s)) if full_f1s else None
        )
        t4_summary["FULL"]["macro_f1_std"] = (
            float(np.std(full_f1s)) if full_f1s else None
        )

        for cls_name in FEATURE_CLASSES:
            delta_f1s = [
                r[cls_name]["delta_f1"]
                for r in t4_seed_results
                if cls_name in r and "delta_f1" in r[cls_name]
            ]
            delta_dns = [
                r[cls_name]["delta_auroc_DN"]
                for r in t4_seed_results
                if cls_name in r and "delta_auroc_DN" in r[cls_name]
            ]
            abl_f1s = [
                r[cls_name].get("macro_f1_mean")
                for r in t4_seed_results
                if cls_name in r and r[cls_name].get("macro_f1_mean") is not None
            ]
            t4_summary[cls_name] = {
                "macro_f1_mean": float(np.mean(abl_f1s)) if abl_f1s else None,
                "macro_f1_std": float(np.std(abl_f1s)) if abl_f1s else None,
                "delta_f1_mean": float(np.mean(delta_f1s)) if delta_f1s else None,
                "delta_f1_std": float(np.std(delta_f1s)) if delta_f1s else None,
                "delta_auroc_DN_mean": float(np.mean(delta_dns)) if delta_dns else None,
                "delta_auroc_DN_std": float(np.std(delta_dns)) if delta_dns else None,
            }

        t4_sum_path = out_dir / "v2_ablation_summary.json"
        with open(t4_sum_path, "w") as f:
            json.dump(t4_summary, f, indent=2)

        print("\n=== T4 SUMMARY (V2 ablation, mean ± std across seeds) ===")
        full_m = t4_summary["FULL"].get("macro_f1_mean")
        full_s = t4_summary["FULL"].get("macro_f1_std")
        print(f"  V2 FULL:  {full_m:.4f} ± {full_s:.4f}")
        print(f"  {'Class':<14}  {'Abl F1':>8}  {'ΔF1':>8}  {'ΔDN AUROC':>10}")
        for cls_name in FEATURE_CLASSES:
            m = t4_summary[cls_name].get("macro_f1_mean")
            df = t4_summary[cls_name].get("delta_f1_mean")
            ddn = t4_summary[cls_name].get("delta_auroc_DN_mean")
            ds = t4_summary[cls_name].get("delta_f1_std")
            m_s = f"{m:.4f}" if m is not None else "  N/A  "
            df_s = f"{df:+.4f}±{ds:.4f}" if df is not None else "  N/A  "
            ddn_s = f"{ddn:+.4f}" if ddn is not None else "  N/A  "
            print(f"  minus {cls_name:<12}  {m_s}  {df_s}  {ddn_s}")


if __name__ == "__main__":
    main()
