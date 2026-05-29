"""
Result 15 — Badonyi 2024 predictions as a modality comparison

Adds Badonyi & Marsh 2024 PLOS One per-gene SVM probabilities (pDN, pGOF, pLOF)
as a new modality and tests them alongside ESM-2 delta and proteome features.

Model variants (all under 5-fold family-split CV, 5 seeds, per-gene T2 scoring):

  V1      ESM-2 delta only (1280-dim)                   MLP 1280→256→64→3
  V2      Proteome only (37-dim)                        LogReg (best from result_13)
  V_bad   Badonyi priors only (3-dim: pDN,pGOF,pLOF)   LogReg
  V2+bad  Proteome + Badonyi (50-dim)                   LogReg
  V1+bad  ESM-2 delta + Badonyi (1293-dim)              MLP 1280→256→64→3
  V_all   ESM-2 delta + proteome + Badonyi (1330-dim)   MLP 1330→256→64→3

Usage:
    python scripts/badonyi_mechanism.py               # all 5 seeds
    python scripts/badonyi_mechanism.py --seed 0      # single seed
    python scripts/badonyi_mechanism.py --out-dir results/badonyi_mechanism/
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from esm2_mechanism.utils_probes import family_split_indices, run_mlp_cv, run_logreg_cv
from esm2_mechanism.utils_paths import DATA_DIR, RESULTS_DIR
import functools
print = functools.partial(print, flush=True)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EMB_DIR = DATA_DIR / "embeddings"

MERGED_VALID_VARIANTS = EMB_DIR / "merged_valid_variants.json"
MERGED_WT_MEAN = EMB_DIR / "merged_embeddings_wt_mean.npy"
MERGED_MUT_MEAN = EMB_DIR / "merged_embeddings_mut_mean.npy"

PROTEOME_FEATURES = DATA_DIR / "proteome_features_aligned.npy"   # (2424, 37)
BADONYI_FEATURES = DATA_DIR / "badonyi_features_aligned.npy"     # (2424, 13)
BADONYI_RAW_COLS = [0, 1, 2]                                      # pDN, pGOF, pLOF only

MERGED_GENE_LIST = DATA_DIR / "merged_gene_list.tsv"
PFAM_FAMILIES = DATA_DIR / "pfam_families.json"

CLASSES = ["GOF", "DN", "LOF"]



# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    with open(MERGED_VALID_VARIANTS) as f:
        variants = json.load(f)
    labels = np.array([v["label_3class"] for v in variants])
    genes = np.array([v["gene"] for v in variants])
    n = len(variants)
    wt = np.load(MERGED_WT_MEAN)[:n]
    mut = np.load(MERGED_MUT_MEAN)[:n]
    delta = (mut - wt).astype(np.float32)
    print(f"Loaded {n} variants, {len(set(genes.tolist()))} unique genes")
    print(f"Class dist: {dict(Counter(labels.tolist()))}")
    return variants, labels, genes, delta


def load_pfam() -> dict[str, str]:
    with open(PFAM_FAMILIES) as f:
        return json.load(f)


def build_gene_to_row() -> dict[str, int]:
    seen: dict[str, int] = {}
    idx = 0
    with open(MERGED_GENE_LIST) as f:
        f.readline()
        for line in f:
            gene = line.strip().split("\t")[0]
            if gene and gene not in seen:
                seen[gene] = idx
                idx += 1
    return seen


def broadcast_gene_features(genes: np.ndarray, matrix: np.ndarray,
                             gene_to_row: dict[str, int]) -> np.ndarray:
    n, n_feats = len(genes), matrix.shape[1]
    X = np.zeros((n, n_feats), dtype=np.float32)
    for i, g in enumerate(genes):
        row = gene_to_row.get(g)
        if row is not None and row < matrix.shape[0]:
            X[i] = matrix[row]
    return X


# ---------------------------------------------------------------------------
# Per-gene T2 scoring (aggregate variant-level proba → gene-level)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_logreg_family_split(X, y, genes, groups, n_folds, seed, label) -> dict:
    splits = list(family_split_indices(groups, n_folds, seed))
    return run_logreg_cv(X, y, splits, seed=seed, genes=genes, label=label)


def run_mlp_family_split(X, y, genes, groups, hidden, n_folds, seed, label) -> dict:
    splits = list(family_split_indices(groups, n_folds, seed))
    return run_mlp_cv(X, y, splits, hidden=hidden, seed=seed,
                      genes=genes, label=label)


# ---------------------------------------------------------------------------
# Single-seed runner
# ---------------------------------------------------------------------------

def run_seed(seed, n_folds, labels, genes, delta, X_prot, X_bad_raw,
             pfam_map) -> dict:
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")

    cls_to_idx = {c: i for i, c in enumerate(CLASSES)}
    y = np.array([cls_to_idx[lbl] for lbl in labels])

    gene_pfam = np.array([pfam_map.get(g) for g in genes])
    has_family = np.array([p is not None for p in gene_pfam])
    fam_idx = np.where(has_family)[0]

    # --- Restrict to family-annotated variants ---
    y_f = y[fam_idx]
    genes_f = genes[fam_idx]
    groups = gene_pfam[fam_idx]

    # Feature matrices at variant level
    X_delta_f = delta[fam_idx]
    X_prot_f = X_prot[fam_idx]
    X_bad_f = X_bad_raw[fam_idx]                              # (n, 3) raw pDN/pGOF/pLOF

    # Concatenations
    X_v2bad = np.concatenate([X_prot_f, X_bad_f], axis=1)    # 37+3=40
    X_v1bad = np.concatenate([X_delta_f, X_bad_f], axis=1)   # 1280+3=1283
    X_vall = np.concatenate([X_delta_f, X_prot_f, X_bad_f], axis=1)  # 1280+37+3=1320

    print(f"  Variants with family: {len(fam_idx)}/{len(y)}, "
          f"families: {len(set(groups.tolist()))}")
    print(f"  Classes: " + ", ".join(f"{c}={int((y_f==i).sum())}"
                                      for i, c in enumerate(CLASSES)))

    results: dict = {
        "seed": seed,
        "n_variants_with_family": int(len(fam_idx)),
        "n_total_variants": int(len(y)),
    }

    # V1 — ESM-2 delta MLP
    print(f"\n--- V1: ESM-2 delta only ---")
    results["V1"] = run_mlp_family_split(
        X_delta_f, y_f, genes_f, groups, (256, 64), n_folds, seed, "V1")
    print(f"  V1 macro_f1={results['V1']['macro_f1_mean']:.4f} ± "
          f"{results['V1']['macro_f1_std']:.4f}  "
          f"per_gene={results['V1'].get('per_gene_f1_mean', float('nan')):.4f}")

    # V2 — Proteome LogReg (best from result_13)
    print(f"\n--- V2: Proteome only (LogReg) ---")
    results["V2"] = run_logreg_family_split(
        X_prot_f, y_f, genes_f, groups, n_folds, seed, "V2")
    print(f"  V2 macro_f1={results['V2']['macro_f1_mean']:.4f} ± "
          f"{results['V2']['macro_f1_std']:.4f}  "
          f"per_gene={results['V2'].get('per_gene_f1_mean', float('nan')):.4f}")

    # V_bad — Badonyi priors only (3 features: pDN, pGOF, pLOF)
    print(f"\n--- V_bad: Badonyi priors only (3-dim LogReg) ---")
    results["V_bad"] = run_logreg_family_split(
        X_bad_f, y_f, genes_f, groups, n_folds, seed, "V_bad")
    print(f"  V_bad macro_f1={results['V_bad']['macro_f1_mean']:.4f} ± "
          f"{results['V_bad']['macro_f1_std']:.4f}  "
          f"per_gene={results['V_bad'].get('per_gene_f1_mean', float('nan')):.4f}")

    # V2+bad — Proteome + Badonyi
    print(f"\n--- V2+bad: Proteome + Badonyi (40-dim LogReg) ---")
    results["V2_bad"] = run_logreg_family_split(
        X_v2bad, y_f, genes_f, groups, n_folds, seed, "V2+bad")
    print(f"  V2+bad macro_f1={results['V2_bad']['macro_f1_mean']:.4f} ± "
          f"{results['V2_bad']['macro_f1_std']:.4f}  "
          f"per_gene={results['V2_bad'].get('per_gene_f1_mean', float('nan')):.4f}")

    # V1+bad — ESM-2 delta + Badonyi
    print(f"\n--- V1+bad: ESM-2 delta + Badonyi (1283-dim MLP) ---")
    results["V1_bad"] = run_mlp_family_split(
        X_v1bad, y_f, genes_f, groups, (256, 64), n_folds, seed, "V1+bad")
    print(f"  V1+bad macro_f1={results['V1_bad']['macro_f1_mean']:.4f} ± "
          f"{results['V1_bad']['macro_f1_std']:.4f}  "
          f"per_gene={results['V1_bad'].get('per_gene_f1_mean', float('nan')):.4f}")

    # V_all — ESM-2 + proteome + Badonyi
    print(f"\n--- V_all: ESM-2 + proteome + Badonyi (1320-dim MLP) ---")
    results["V_all"] = run_mlp_family_split(
        X_vall, y_f, genes_f, groups, (256, 64), n_folds, seed, "V_all")
    print(f"  V_all macro_f1={results['V_all']['macro_f1_mean']:.4f} ± "
          f"{results['V_all']['macro_f1_std']:.4f}  "
          f"per_gene={results['V_all'].get('per_gene_f1_mean', float('nan')):.4f}")

    return results


# ---------------------------------------------------------------------------
# Aggregate across seeds
# ---------------------------------------------------------------------------

def aggregate_seeds(all_results: list[dict]) -> dict:
    summary: dict = {"n_seeds": len(all_results)}
    for key in ["V1", "V2", "V_bad", "V2_bad", "V1_bad", "V_all"]:
        for metric in ["macro_f1_mean", "per_gene_f1_mean"]:
            vals = [r[key][metric] for r in all_results
                    if key in r and r[key].get(metric) is not None]
            stem = f"{key}_{metric.replace('_mean','')}"
            if vals:
                summary[f"{stem}_mean"] = float(np.mean(vals))
                summary[f"{stem}_std"] = float(np.std(vals))
            else:
                summary[f"{stem}_mean"] = None
                summary[f"{stem}_std"] = None
        for cls in CLASSES:
            vals = [r[key].get(f"auroc_{cls}_mean") for r in all_results
                    if key in r and r[key].get(f"auroc_{cls}_mean") is not None]
            if vals:
                summary[f"{key}_auroc_{cls}_mean"] = float(np.mean(vals))
                summary[f"{key}_auroc_{cls}_std"] = float(np.std(vals))
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Result 15: Badonyi priors as a modality")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else list(range(5))
    out_dir = Path(args.out_dir) if args.out_dir else \
        RESULTS_DIR / "badonyi_mechanism"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Loading data ===")
    variants, labels, genes, delta = load_data()

    print("\n=== Loading Pfam families ===")
    pfam_map = load_pfam()

    print("\n=== Broadcasting proteome features ===")
    prot_matrix = np.load(PROTEOME_FEATURES).astype(np.float32)
    gene_to_row = build_gene_to_row()
    X_prot = broadcast_gene_features(genes, prot_matrix, gene_to_row)
    print(f"  Proteome: {X_prot.shape}")

    print("\n=== Broadcasting Badonyi features (pDN, pGOF, pLOF only) ===")
    bad_matrix_full = np.load(BADONYI_FEATURES).astype(np.float32)
    # Only use the 3 raw probability columns (indices 0,1,2 = pDN, pGOF, pLOF)
    bad_matrix = bad_matrix_full[:, BADONYI_RAW_COLS]
    X_bad = broadcast_gene_features(genes, bad_matrix, gene_to_row)
    print(f"  Badonyi: {X_bad.shape}  (pDN, pGOF, pLOF)")

    all_results = []
    for seed in seeds:
        res = run_seed(seed, args.n_folds, labels, genes, delta,
                       X_prot, X_bad, pfam_map)
        all_results.append(res)
        path = out_dir / f"badonyi_mechanism_seed{seed}.json"
        path.write_text(json.dumps(res, indent=2))
        print(f"\n  Saved: {path}")

    summary = aggregate_seeds(all_results)
    summary_path = out_dir / "badonyi_mechanism_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary: {summary_path}")

    # --- Print summary table ---
    print("\n" + "="*72)
    print("SUMMARY — macro-F1 (mean ± std across 5 seeds, family-split CV)")
    print("="*72)
    header = f"{'Variant':<12} {'F1(var)':<16} {'F1(gene)':<16} {'GOF AUROC':<14} {'DN AUROC':<14} {'LOF AUROC':<14}"
    print(header)
    print("-"*72)

    def fmt(m, s):
        if m is None:
            return "    N/A       "
        return f"{m:.4f}±{s:.4f}"

    for key, label in [("V1","V1(seq)"), ("V2","V2(prot)"),
                        ("V_bad","V_bad"), ("V2_bad","V2+bad"),
                        ("V1_bad","V1+bad"), ("V_all","V_all")]:
        f1 = fmt(summary.get(f"{key}_macro_f1_mean"),
                 summary.get(f"{key}_macro_f1_std"))
        pg = fmt(summary.get(f"{key}_per_gene_f1_mean"),
                 summary.get(f"{key}_per_gene_f1_std"))
        gof = fmt(summary.get(f"{key}_auroc_GOF_mean"),
                  summary.get(f"{key}_auroc_GOF_std"))
        dn = fmt(summary.get(f"{key}_auroc_DN_mean"),
                 summary.get(f"{key}_auroc_DN_std"))
        lof = fmt(summary.get(f"{key}_auroc_LOF_mean"),
                  summary.get(f"{key}_auroc_LOF_std"))
        print(f"{label:<12} {f1:<16} {pg:<16} {gof:<14} {dn:<14} {lof:<14}")

    print("="*72)
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
