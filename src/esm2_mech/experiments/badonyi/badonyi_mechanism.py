"""
Result 15 — Badonyi 2024 predictions as a modality comparison

Adds Badonyi & Marsh 2024 PLOS One per-gene SVM probabilities (pDN, pGOF, pLOF)
as a new modality and tests them alongside ESM-2 delta and proteome features.

Model variants (all under 5-fold family-split CV, 5 seeds, per-gene T2 scoring):

  V1      ESM-2 delta only (1280-dim)                   MLP 1280→256→64→3
  V2      Proteome only (37-dim)                        NaN-native boosting
  V_bad   Badonyi priors only (3-dim: pDN,pGOF,pLOF)    NaN-native boosting
  V2+bad  Proteome + Badonyi (50-dim)                   NaN-native boosting
  V1+bad  ESM-2 delta + Badonyi (1293-dim)              NaN-native boosting
  V_all   ESM-2 delta + proteome + Badonyi (1330-dim)   NaN-native boosting

Missing-data policy. The proteome and Badonyi blocks carry real NaN — a gene
with no Badonyi row, or a proteome family-residual undefined for a singleton
family — and nothing is imputed. V1 is delta-only and fully observed, so it
keeps its MLP. Every other arm includes one of those blocks (V1+bad and V_all
concatenate them onto the dense delta, where a few missing columns would
otherwise make the whole row unusable), so all of them use a model that
consumes NaN directly and keeps every gene.

Usage:
    python scripts/badonyi_mechanism.py               # all 5 seeds
    python scripts/badonyi_mechanism.py --seed 0      # single seed
    python scripts/badonyi_mechanism.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from esm2_mech.utils.constants import MECHANISM_CLASSES
from esm2_mech.utils.data import build_gene_to_row as _build_gene_to_row
from esm2_mech.utils.splits import family_split_indices
from esm2_mech.utils.probes import run_mlp_cv, run_logreg_cv, run_histgb_cv
from esm2_mech.utils.paths import (
    RESULTS_DIR,
    VALID_VARIANTS_JSON,
    EMB_WT_MEAN,
    EMB_MUT_MEAN,
    GENE_UNIVERSE,
    PFAM_JSON,
    PROTEOME_FEATURES_ALIGNED,
    PROTEOME_FEATURE_COLUMNS_JSON,
    PROTEOME_FEATURES_TSV,
    BADONYI_FEATURES_ALIGNED,
    BADONYI_FEATURE_COLUMNS_JSON,
    BADONYI_FEATURES_TSV,
)
import functools

print = functools.partial(print, flush=True)

OUT_DIR = RESULTS_DIR

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MERGED_VALID_VARIANTS = VALID_VARIANTS_JSON
MERGED_WT_MEAN = EMB_WT_MEAN
MERGED_MUT_MEAN = EMB_MUT_MEAN

PROTEOME_FEATURES = PROTEOME_FEATURES_ALIGNED
BADONYI_FEATURES = BADONYI_FEATURES_ALIGNED
BADONYI_RAW_COLS = [0, 1, 2]  # pDN, pGOF, pLOF only

# Row index for the aligned feature matrices. MUST be GENE_UNIVERSE, not
# GENE_LIST_TSV: build_proteome_features/build_badonyi_features write their
# .npy rows in gene_universe.tsv order, and gene_list.tsv is a longer,
# differently-ordered superset (see paths.GENE_UNIVERSE).
MERGED_GENE_LIST = GENE_UNIVERSE
PFAM_FAMILIES = PFAM_JSON

CLASSES = MECHANISM_CLASSES


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
    return _build_gene_to_row(MERGED_GENE_LIST)


def broadcast_gene_features(
    genes: np.ndarray, matrix: np.ndarray, gene_to_row: dict[str, int]
) -> np.ndarray:
    """Broadcast per-gene features to variant rows, NaN where the gene has no row.

    A gene absent from the feature matrix gets NaN, not 0.0: these are
    probability scores and constraint metrics where 0.0 is a plausible real
    observation, so a zero-filled row would be indistinguishable from a real
    measurement of zero. Downstream arms either consume the NaN natively or
    restrict to the observed rows.
    """
    n, n_feats = len(genes), matrix.shape[1]
    X = np.full((n, n_feats), np.nan, dtype=np.float32)
    n_missing = 0
    for i, g in enumerate(genes):
        row = gene_to_row.get(g)
        if row is not None and row < matrix.shape[0]:
            X[i] = matrix[row]
        else:
            n_missing += 1
    if n_missing:
        print(
            f"  {n_missing}/{n} variant rows have no feature row for their gene "
            f"(left as NaN)"
        )
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
    return run_mlp_cv(X, y, splits, hidden=hidden, seed=seed, genes=genes, label=label)


def run_histgb_family_split(X, y, genes, groups, n_folds, seed, label) -> dict:
    """NaN-native family-split CV — for every arm whose matrix includes the
    proteome or Badonyi block, alone or concatenated onto the ESM-2 delta.

    Those blocks carry real missing cells (a gene with no Badonyi row, or a
    proteome family-residual that is undefined for a singleton family). Nothing
    is imputed: a filled-in value computed before the CV split would leak
    test-fold statistics into training and be indistinguishable from a real
    score afterwards. Restricting instead would discard every gene missing any
    one feature, including genes whose ESM-2 embedding is fully observed.
    """
    splits = list(family_split_indices(groups, n_folds, seed))
    return run_histgb_cv(X, y, splits, seed=seed, genes=genes, label=label)


# ---------------------------------------------------------------------------
# Single-seed runner
# ---------------------------------------------------------------------------


def run_seed(seed, n_folds, labels, genes, delta, X_prot, X_bad_raw, pfam_map) -> dict:
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")

    # y stays string labels — run_logreg_cv/run_mlp_cv key on `classes` (strings).
    y = np.asarray(labels)

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
    X_bad_f = X_bad_raw[fam_idx]  # (n, 3) raw pDN/pGOF/pLOF

    # Concatenations
    X_v2bad = np.concatenate([X_prot_f, X_bad_f], axis=1)  # 37+3=40
    X_v1bad = np.concatenate([X_delta_f, X_bad_f], axis=1)  # 1280+3=1283
    X_vall = np.concatenate([X_delta_f, X_prot_f, X_bad_f], axis=1)  # 1280+37+3=1320

    print(
        f"  Variants with family: {len(fam_idx)}/{len(y)}, "
        f"families: {len(set(groups.tolist()))}"
    )
    print(
        f"  Classes: "
        + ", ".join(f"{c}={int((y_f==c).sum())}" for c in CLASSES)
    )

    results: dict = {
        "seed": seed,
        "n_variants_with_family": int(len(fam_idx)),
        "n_total_variants": int(len(y)),
    }

    # V1 — ESM-2 delta MLP
    print(f"\n--- V1: ESM-2 delta only ---")
    results["V1"] = run_mlp_family_split(
        X_delta_f, y_f, genes_f, groups, (256, 64), n_folds, seed, "V1"
    )
    print(
        f"  V1 macro_f1={results['V1']['macro_f1_mean']:.4f} ± "
        f"{results['V1']['macro_f1_std']:.4f}  "
        f"per_gene={results['V1'].get('per_gene_f1_mean', float('nan')):.4f}"
    )

    # V2 — Proteome only (NaN-native: the proteome block has missing cells)
    print(f"\n--- V2: Proteome only (NaN-native gradient boosting) ---")
    results["V2"] = run_histgb_family_split(
        X_prot_f, y_f, genes_f, groups, n_folds, seed, "V2"
    )
    print(
        f"  V2 macro_f1={results['V2']['macro_f1_mean']:.4f} ± "
        f"{results['V2']['macro_f1_std']:.4f}  "
        f"per_gene={results['V2'].get('per_gene_f1_mean', float('nan')):.4f}"
    )

    # V_bad — Badonyi priors only (3 features: pDN, pGOF, pLOF)
    print(f"\n--- V_bad: Badonyi priors only (3-dim, NaN-native) ---")
    results["V_bad"] = run_histgb_family_split(
        X_bad_f, y_f, genes_f, groups, n_folds, seed, "V_bad"
    )
    print(
        f"  V_bad macro_f1={results['V_bad']['macro_f1_mean']:.4f} ± "
        f"{results['V_bad']['macro_f1_std']:.4f}  "
        f"per_gene={results['V_bad'].get('per_gene_f1_mean', float('nan')):.4f}"
    )

    # V2+bad — Proteome + Badonyi
    print(f"\n--- V2+bad: Proteome + Badonyi (40-dim, NaN-native) ---")
    results["V2_bad"] = run_histgb_family_split(
        X_v2bad, y_f, genes_f, groups, n_folds, seed, "V2+bad"
    )
    print(
        f"  V2+bad macro_f1={results['V2_bad']['macro_f1_mean']:.4f} ± "
        f"{results['V2_bad']['macro_f1_std']:.4f}  "
        f"per_gene={results['V2_bad'].get('per_gene_f1_mean', float('nan')):.4f}"
    )

    # V1+bad — ESM-2 delta + Badonyi
    print(f"\n--- V1+bad: ESM-2 delta + Badonyi (1283-dim, NaN-native) ---")
    results["V1_bad"] = run_histgb_family_split(
        X_v1bad, y_f, genes_f, groups, n_folds, seed, "V1+bad"
    )
    print(
        f"  V1+bad macro_f1={results['V1_bad']['macro_f1_mean']:.4f} ± "
        f"{results['V1_bad']['macro_f1_std']:.4f}  "
        f"per_gene={results['V1_bad'].get('per_gene_f1_mean', float('nan')):.4f}"
    )

    # V_all — ESM-2 + proteome + Badonyi
    print(f"\n--- V_all: ESM-2 + proteome + Badonyi (1320-dim, NaN-native) ---")
    results["V_all"] = run_histgb_family_split(
        X_vall, y_f, genes_f, groups, n_folds, seed, "V_all"
    )
    print(
        f"  V_all macro_f1={results['V_all']['macro_f1_mean']:.4f} ± "
        f"{results['V_all']['macro_f1_std']:.4f}  "
        f"per_gene={results['V_all'].get('per_gene_f1_mean', float('nan')):.4f}"
    )

    return results


# ---------------------------------------------------------------------------
# Aggregate across seeds
# ---------------------------------------------------------------------------


def aggregate_seeds(all_results: list[dict]) -> dict:
    summary: dict = {"n_seeds": len(all_results)}

    def pull(key, metric):
        # Returns the per-seed metric or None; seed_metric_mean_std drops None+NaN.
        return lambda r, k=key, m=metric: r[k].get(m) if k in r else None

    for key in ["V1", "V2", "V_bad", "V2_bad", "V1_bad", "V_all"]:
        for metric in ["macro_f1_mean", "per_gene_f1_mean"]:
            stem = f"{key}_{metric.replace('_mean','')}"
            mean, std, n = seed_metric_mean_std(all_results, pull(key, metric))
            summary[f"{stem}_mean"] = mean if n else None
            summary[f"{stem}_std"] = std if n else None
        for cls in CLASSES:
            mean, std, n = seed_metric_mean_std(
                all_results, pull(key, f"auroc_{cls}_mean")
            )
            if n:
                summary[f"{key}_auroc_{cls}_mean"] = mean
                summary[f"{key}_auroc_{cls}_std"] = std
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Result 15: Badonyi priors as a modality"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else list(range(5))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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
        res = run_seed(
            seed, args.n_folds, labels, genes, delta, X_prot, X_bad, pfam_map
        )
        all_results.append(res)
        path = OUT_DIR / f"badonyi_mechanism_seed{seed}.json"
        path.write_text(json.dumps(res, indent=2))
        print(f"\n  Saved: {path}")

    summary = aggregate_seeds(all_results)
    summary_path = OUT_DIR / "badonyi_mechanism_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary: {summary_path}")

    # --- Print summary table ---
    print("\n" + "=" * 72)
    print("SUMMARY — macro-F1 (mean ± std across 5 seeds, family-split CV)")
    print("=" * 72)
    header = f"{'Variant':<12} {'F1(var)':<16} {'F1(gene)':<16} {'GOF AUROC':<14} {'DN AUROC':<14} {'LOF AUROC':<14}"
    print(header)
    print("-" * 72)

    def fmt(m, s):
        if m is None:
            return "    N/A       "
        return f"{m:.4f}±{s:.4f}"

    for key, label in [
        ("V1", "V1(seq)"),
        ("V2", "V2(prot)"),
        ("V_bad", "V_bad"),
        ("V2_bad", "V2+bad"),
        ("V1_bad", "V1+bad"),
        ("V_all", "V_all"),
    ]:
        f1 = fmt(
            summary.get(f"{key}_macro_f1_mean"), summary.get(f"{key}_macro_f1_std")
        )
        pg = fmt(
            summary.get(f"{key}_per_gene_f1_mean"),
            summary.get(f"{key}_per_gene_f1_std"),
        )
        gof = fmt(
            summary.get(f"{key}_auroc_GOF_mean"), summary.get(f"{key}_auroc_GOF_std")
        )
        dn = fmt(
            summary.get(f"{key}_auroc_DN_mean"), summary.get(f"{key}_auroc_DN_std")
        )
        lof = fmt(
            summary.get(f"{key}_auroc_LOF_mean"), summary.get(f"{key}_auroc_LOF_std")
        )
        print(f"{label:<12} {f1:<16} {pg:<16} {gof:<14} {dn:<14} {lof:<14}")

    print("=" * 72)
    print(f"Results: {OUT_DIR}")


if __name__ == "__main__":
    main()
