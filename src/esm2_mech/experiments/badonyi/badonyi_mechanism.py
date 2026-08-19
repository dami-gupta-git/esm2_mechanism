"""Compare Badonyi 2024 pDN/pGOF/pLOF as a modality against ESM-2 delta and proteome features."""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from esm2_mech.utils.constants import MECHANISM_CLASSES, BOOTSTRAP_N_RESAMPLES
from esm2_mech.utils.data import build_gene_to_row as _build_gene_to_row, load_pfam_map
from esm2_mech.utils.splits import family_split_indices
from esm2_mech.utils.probes import run_mlp_cv, run_logreg_cv, run_histgb_cv
from esm2_mech.utils.bootstrap import attach_mechanism_ci, family_or_gene_clusters
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

MERGED_VALID_VARIANTS = VALID_VARIANTS_JSON
MERGED_WT_MEAN = EMB_WT_MEAN
MERGED_MUT_MEAN = EMB_MUT_MEAN

PROTEOME_FEATURES = PROTEOME_FEATURES_ALIGNED
BADONYI_FEATURES = BADONYI_FEATURES_ALIGNED
BADONYI_RAW_COLS = [0, 1, 2]  # pDN, pGOF, pLOF only

# MUST be GENE_UNIVERSE, not GENE_LIST_TSV: .npy rows are in gene_universe.tsv order.
MERGED_GENE_LIST = GENE_UNIVERSE
PFAM_FAMILIES = PFAM_JSON

CLASSES = MECHANISM_CLASSES


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


def build_gene_to_row() -> dict[str, int]:
    return _build_gene_to_row(MERGED_GENE_LIST)


# NaN not 0.0 for missing genes: 0.0 is a plausible real observation.
def broadcast_gene_features(
    genes: np.ndarray, matrix: np.ndarray, gene_to_row: dict[str, int]
) -> np.ndarray:
    """Broadcast per-gene features to variant rows, NaN where the gene has no row."""
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


# Resamples families not genes: genes within one family are not independent draws.
def _attach_family_split_ci(
    agg: dict, oof: dict | None, pfam_map: dict, compute_ci: bool, n_boot: int, seed: int
) -> dict:
    """Attach a family-resampled cluster-bootstrap CI to a family-split CV result."""
    clusters = (
        family_or_gene_clusters(oof["genes"], pfam_map, is_family_split=True)
        if oof is not None
        else None
    )
    return attach_mechanism_ci(
        agg,
        oof,
        clusters,
        compute_ci=compute_ci,
        n_resamples=n_boot,
        seed=seed,
    )


def run_logreg_family_split(
    X, y, genes, groups, n_folds, seed, label, pfam_map, compute_ci=True, n_boot=BOOTSTRAP_N_RESAMPLES
) -> dict:
    splits = list(family_split_indices(groups, n_folds, seed))
    agg, oof = run_logreg_cv(X, y, splits, seed=seed, genes=genes, label=label, return_oof=True)
    return _attach_family_split_ci(agg, oof, pfam_map, compute_ci, n_boot, seed)


def run_mlp_family_split(
    X, y, genes, groups, hidden, n_folds, seed, label, pfam_map, compute_ci=True, n_boot=BOOTSTRAP_N_RESAMPLES
) -> dict:
    splits = list(family_split_indices(groups, n_folds, seed))
    agg, oof = run_mlp_cv(
        X, y, splits, hidden=hidden, seed=seed, genes=genes, label=label, return_oof=True
    )
    return _attach_family_split_ci(agg, oof, pfam_map, compute_ci, n_boot, seed)


# Nothing is imputed: filling before the split would leak test-fold statistics into training.
def run_histgb_family_split(
    X, y, genes, groups, n_folds, seed, label, pfam_map, compute_ci=True, n_boot=BOOTSTRAP_N_RESAMPLES
) -> dict:
    """NaN-native family-split CV for arms with proteome or Badonyi blocks."""
    splits = list(family_split_indices(groups, n_folds, seed))
    agg, oof = run_histgb_cv(X, y, splits, seed=seed, genes=genes, label=label, return_oof=True)
    return _attach_family_split_ci(agg, oof, pfam_map, compute_ci, n_boot, seed)


def run_seed(
    seed, n_folds, labels, genes, delta, X_prot, X_bad_raw, pfam_map,
    compute_ci=True, n_boot=BOOTSTRAP_N_RESAMPLES,
) -> dict:
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

    def _log_ci(agg: dict) -> str:
        ci = agg.get("ci", {}).get("macro_f1")
        if not ci or ci.get("ci_suppressed"):
            return ""
        return f"  CI=[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}] ({ci['n_clusters']} families)"

    # V1 — ESM-2 delta MLP
    print(f"\n--- V1: ESM-2 delta only ---")
    results["V1"] = run_mlp_family_split(
        X_delta_f, y_f, genes_f, groups, (256, 64), n_folds, seed, "V1",
        pfam_map, compute_ci=compute_ci, n_boot=n_boot,
    )
    print(
        f"  V1 macro_f1={results['V1']['macro_f1_mean']:.4f} ± "
        f"{results['V1']['macro_f1_std']:.4f}  "
        f"per_gene={results['V1'].get('per_gene_f1_mean', float('nan')):.4f}"
        + _log_ci(results["V1"])
    )

    # V2 — Proteome only (NaN-native: the proteome block has missing cells)
    print(f"\n--- V2: Proteome only (NaN-native gradient boosting) ---")
    results["V2"] = run_histgb_family_split(
        X_prot_f, y_f, genes_f, groups, n_folds, seed, "V2",
        pfam_map, compute_ci=compute_ci, n_boot=n_boot,
    )
    print(
        f"  V2 macro_f1={results['V2']['macro_f1_mean']:.4f} ± "
        f"{results['V2']['macro_f1_std']:.4f}  "
        f"per_gene={results['V2'].get('per_gene_f1_mean', float('nan')):.4f}"
        + _log_ci(results["V2"])
    )

    # V_bad — Badonyi priors only (3 features: pDN, pGOF, pLOF)
    print(f"\n--- V_bad: Badonyi priors only (3-dim, NaN-native) ---")
    results["V_bad"] = run_histgb_family_split(
        X_bad_f, y_f, genes_f, groups, n_folds, seed, "V_bad",
        pfam_map, compute_ci=compute_ci, n_boot=n_boot,
    )
    print(
        f"  V_bad macro_f1={results['V_bad']['macro_f1_mean']:.4f} ± "
        f"{results['V_bad']['macro_f1_std']:.4f}  "
        f"per_gene={results['V_bad'].get('per_gene_f1_mean', float('nan')):.4f}"
        + _log_ci(results["V_bad"])
    )

    # V2+bad — Proteome + Badonyi
    print(f"\n--- V2+bad: Proteome + Badonyi (40-dim, NaN-native) ---")
    results["V2_bad"] = run_histgb_family_split(
        X_v2bad, y_f, genes_f, groups, n_folds, seed, "V2+bad",
        pfam_map, compute_ci=compute_ci, n_boot=n_boot,
    )
    print(
        f"  V2+bad macro_f1={results['V2_bad']['macro_f1_mean']:.4f} ± "
        f"{results['V2_bad']['macro_f1_std']:.4f}  "
        f"per_gene={results['V2_bad'].get('per_gene_f1_mean', float('nan')):.4f}"
        + _log_ci(results["V2_bad"])
    )

    # V1+bad — ESM-2 delta + Badonyi
    print(f"\n--- V1+bad: ESM-2 delta + Badonyi (1283-dim, NaN-native) ---")
    results["V1_bad"] = run_histgb_family_split(
        X_v1bad, y_f, genes_f, groups, n_folds, seed, "V1+bad",
        pfam_map, compute_ci=compute_ci, n_boot=n_boot,
    )
    print(
        f"  V1+bad macro_f1={results['V1_bad']['macro_f1_mean']:.4f} ± "
        f"{results['V1_bad']['macro_f1_std']:.4f}  "
        f"per_gene={results['V1_bad'].get('per_gene_f1_mean', float('nan')):.4f}"
        + _log_ci(results["V1_bad"])
    )

    # V_all — ESM-2 + proteome + Badonyi
    print(f"\n--- V_all: ESM-2 + proteome + Badonyi (1320-dim, NaN-native) ---")
    results["V_all"] = run_histgb_family_split(
        X_vall, y_f, genes_f, groups, n_folds, seed, "V_all",
        pfam_map, compute_ci=compute_ci, n_boot=n_boot,
    )
    print(
        f"  V_all macro_f1={results['V_all']['macro_f1_mean']:.4f} ± "
        f"{results['V_all']['macro_f1_std']:.4f}  "
        f"per_gene={results['V_all'].get('per_gene_f1_mean', float('nan')):.4f}"
        + _log_ci(results["V_all"])
    )

    return results


def seed_metric_mean_std(all_results: list[dict], extractor) -> tuple[float | None, float | None, int]:
    """Mean/std of one metric across seeds, skipping None and NaN."""
    vals = []
    for r in all_results:
        v = extractor(r)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(float(v))
    if not vals:
        return None, None, 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


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

        # CI bounds pooled across seeds — separate from seed-to-seed std.
        for metric_name in ["macro_f1"] + [f"auroc_{cls}" for cls in CLASSES]:
            for bound in ("ci_low", "ci_high"):
                def extractor(r, k=key, m=metric_name, b=bound):
                    ci = r.get(k, {}).get("ci", {}).get(m)
                    if not ci or ci.get("ci_suppressed"):
                        return None
                    return ci.get(b)

                mean, std, n = seed_metric_mean_std(all_results, extractor)
                if n:
                    summary[f"{key}_{metric_name}_{bound}_seed_mean"] = mean
                    summary[f"{key}_{metric_name}_{bound}_n_seeds"] = n
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Result 15: Badonyi priors as a modality"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--no_ci", action="store_true", help="skip cluster-bootstrap CIs")
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else list(range(5))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading data ===")
    variants, labels, genes, delta = load_data()

    print("\n=== Loading Pfam families ===")
    pfam_map = load_pfam_map(PFAM_FAMILIES)

    print("\n=== Broadcasting proteome features ===")
    prot_matrix = np.load(PROTEOME_FEATURES).astype(np.float32)
    gene_to_row = build_gene_to_row()
    X_prot = broadcast_gene_features(genes, prot_matrix, gene_to_row)
    print(f"  Proteome: {X_prot.shape}")

    print("\n=== Broadcasting Badonyi features (pDN, pGOF, pLOF only) ===")
    bad_matrix_full = np.load(BADONYI_FEATURES).astype(np.float32)
    bad_matrix = bad_matrix_full[:, BADONYI_RAW_COLS]
    X_bad = broadcast_gene_features(genes, bad_matrix, gene_to_row)
    print(f"  Badonyi: {X_bad.shape}  (pDN, pGOF, pLOF)")

    all_results = []
    for seed in seeds:
        res = run_seed(
            seed, args.n_folds, labels, genes, delta, X_prot, X_bad, pfam_map,
            compute_ci=not args.no_ci, n_boot=args.n_boot,
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
