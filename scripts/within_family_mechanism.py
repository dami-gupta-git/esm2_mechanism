"""
Result 16 — Within-family mechanism prediction from family-residual features

Central question: is mechanism predictable from *within-family variation* in
gene-level features, independent of family identity?

Design:
- Use only the family-mean-centred residual columns from proteome_features_aligned.npy
  (pLI_familyresid, LOEUF_familyresid, mis_z_familyresid, paralog_count_familyresid,
   tissue_specificity_tau_familyresid, log_abundance_ppm_familyresid,
   PPI_degree_familyresid, HI_score_familyresid, TS_score_familyresid)
- For each qualifying Pfam family (>= 6 genes, >= 2 mechanism classes), run
  leave-one-gene-out (LOGO) CV using logistic regression and k-NN.
- Aggregate: overall macro-F1 pooling predictions across all families; per-family
  breakdown to identify which protein architectures have learnable within-family signal.
- Compare raw features vs residual features to isolate "within-family" vs
  "cross-family" contributions.

Also runs:
- Badonyi residuals (pDN_familyresid, pGOF_familyresid, pLOF_familyresid) — does
  structural prior also carry within-family signal?
- Combined residuals (proteome + Badonyi) within families.

Usage:
    python scripts/within_family_mechanism.py
    python scripts/within_family_mechanism.py --min-genes 6 --min-classes 2
    python scripts/within_family_mechanism.py --out-dir results/within_family/
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from utils_probes import compute_metrics, align_proba
import functools
print = functools.partial(print, flush=True)

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

MERGED_GENE_LIST = DATA_DIR / "merged_gene_list.tsv"
PFAM_FAMILIES = DATA_DIR / "pfam_families.json"
PROTEOME_FEATURES = DATA_DIR / "proteome_features_aligned.npy"   # (2424, 37)
BADONYI_FEATURES = DATA_DIR / "badonyi_features_aligned.npy"     # (2424, 13)

CLASSES = ["GOF", "DN", "LOF"]

# Column indices in proteome_features_aligned.npy (0-indexed into the 37-col matrix)
# TSV cols: gene(0), pfam(1), is_singleton(2), then features 3..38
# NPY cols: 0..36 (gene and pfam not in npy — it's numerical only)
# Mapping: npy_idx = tsv_idx - 2 (for numerical cols starting at tsv index 2)
# Raw continuous: pLI=1, LOEUF=5, mis_z=9, paralog_count=13, tau=17, abundance=21, PPI=25, HI=29, TS=33
# But npy only has numerical columns. Let's re-derive from gene_proteome_features.tsv header.

# From the column list above (npy indices, 0-based):
# is_singleton=0, pLI=1, pLI_missing=2, pLI_familyresid=3, pLI_familyresid_missing=4,
# LOEUF=5, LOEUF_missing=6, LOEUF_familyresid=7, ..., etc.
# Actually the npy has 37 cols = all_columns[2:] (skipping gene and pfam_family)

RAW_INDICES = [1, 5, 9, 13, 17, 21, 25, 29, 33]          # pLI,LOEUF,mis_z,paralog,tau,abund,PPI,HI,TS
RESID_INDICES = [3, 7, 11, 15, 19, 23, 27, 31, 35]        # *_familyresid cols
RAW_NAMES = ["pLI","LOEUF","mis_z","paralog_count","tissue_tau","log_abundance","PPI_degree","HI_score","TS_score"]

# Badonyi npy cols: 0=pDN, 1=pGOF, 2=pLOF, 3=pDN_missing, ..., 6=pDN_familyresid, 7=pGOF_familyresid, 8=pLOF_familyresid
BAD_RAW_INDICES = [0, 1, 2]
BAD_RESID_INDICES = [6, 7, 8]


def load_gene_list():
    """Returns list of (gene, mechanism_3class) for all labeled genes."""
    genes, mechs = [], []
    with open(MERGED_GENE_LIST) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            gene = parts[0]
            mech = parts[1] if len(parts) > 1 else ""
            genes.append(gene)
            mechs.append(mech)
    return genes, mechs


def collapse_3class(mech: str) -> str | None:
    if mech in ("GOF",): return "GOF"
    if mech in ("DN",): return "DN"
    if mech in ("HI", "LOF"): return "LOF"
    return None


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


def get_qualifying_families(genes, mechs, pfam_map, min_genes, min_classes):
    """Return dict of {pfam: {gene: label}} for families meeting criteria."""
    fam_genes: dict[str, dict[str, str]] = defaultdict(dict)
    for gene, mech in zip(genes, mechs):
        m3 = collapse_3class(mech)
        if m3 is None:
            continue
        fam = pfam_map.get(gene)
        if fam is None:
            continue
        fam_genes[fam][gene] = m3

    qualifying = {}
    for fam, gmap in fam_genes.items():
        n_classes = len(set(gmap.values()))
        if len(gmap) >= min_genes and n_classes >= min_classes:
            qualifying[fam] = gmap
    return qualifying


def logo_cv(X: np.ndarray, y: np.ndarray, gene_names: list[str],
            seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Leave-one-gene-out CV with LogReg.
    Returns (y_true, y_pred, y_proba) concatenated across all left-out genes.
    """
    n = len(y)
    all_true, all_pred, all_proba = [], [], []

    for i in range(n):
        train_idx = np.array([j for j in range(n) if j != i])
        test_idx = np.array([i])

        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Need at least 2 classes in training
        if len(set(y_tr.tolist())) < 2:
            continue

        sc = StandardScaler().fit(X_tr)
        clf = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed,
            solver="lbfgs", multi_class="auto",
        )
        try:
            clf.fit(sc.transform(X_tr), y_tr)
        except Exception:
            continue

        proba = align_proba(clf.predict_proba(sc.transform(X_te)),
                            clf.classes_, len(CLASSES))

        all_true.append(y_te[0])
        all_pred.append(int(proba.argmax()))
        all_proba.append(proba[0])

    if not all_true:
        return np.array([]), np.array([]), np.zeros((0, len(CLASSES)))

    return (np.array(all_true), np.array(all_pred),
            np.stack(all_proba))



def run_feature_set(
    qualifying: dict[str, dict[str, str]],
    gene_to_row: dict[str, int],
    feature_matrix: np.ndarray,
    col_indices: list[int],
    col_names: list[str],
    label: str,
    seed: int,
) -> dict:
    """
    Run LOGO CV within each qualifying family for a given feature set.
    Returns per-family results + aggregate.
    """
    le = LabelEncoder()
    le.fit(CLASSES)

    per_family: dict[str, dict] = {}
    all_true_pool, all_pred_pool, all_proba_pool = [], [], []
    family_names_pool = []

    for fam, gmap in qualifying.items():
        gene_list = sorted(gmap.keys())
        labels_raw = [gmap[g] for g in gene_list]

        # Build feature matrix for this family
        rows = [gene_to_row.get(g) for g in gene_list]
        missing = [g for g, r in zip(gene_list, rows) if r is None]
        if missing:
            continue

        X_fam = feature_matrix[rows][:, col_indices].astype(np.float32)
        y_fam = le.transform(labels_raw)

        # Skip if any class has 0 examples or fewer than 2 classes
        class_counts = Counter(y_fam.tolist())
        if len(class_counts) < 2:
            continue

        y_true, y_pred, y_proba = logo_cv(X_fam, y_fam, gene_list, seed)

        if len(y_true) == 0:
            continue

        fm = compute_metrics(y_true, y_pred, y_proba)
        fm["class_counts"] = dict(class_counts)
        fm["genes"] = gene_list
        fm["labels"] = labels_raw
        per_family[fam] = fm

        all_true_pool.append(y_true)
        all_pred_pool.append(y_pred)
        all_proba_pool.append(y_proba)
        family_names_pool.extend([fam] * len(y_true))

        f1_str = f"{fm['macro_f1']:.3f}" if fm['macro_f1'] is not None else "N/A"
        auroc_parts = []
        for cls in CLASSES:
            v = fm['per_class_auroc'].get(cls)
            auroc_parts.append(f"{cls}={v:.3f}" if v is not None else f"{cls}=NA")
        print(f"  [{label}] {fam:<20} n={len(gene_list):2d}  "
              f"F1={f1_str}  " + "  ".join(auroc_parts)
              + f"  classes={dict(class_counts)}")

    # Aggregate across all families
    if all_true_pool:
        y_true_all = np.concatenate(all_true_pool)
        y_pred_all = np.concatenate(all_pred_pool)
        y_proba_all = np.concatenate(all_proba_pool)
        agg = compute_metrics(y_true_all, y_pred_all, y_proba_all)
    else:
        agg = {"macro_f1": None, "per_class_auroc": {c: None for c in CLASSES}, "n": 0}

    # Majority baseline (per-family, predicting modal class)
    majority_correct = 0
    majority_total = 0
    for fam, gmap in qualifying.items():
        labels = list(gmap.values())
        modal = Counter(labels).most_common(1)[0][0]
        majority_correct += sum(1 for l in labels if l == modal)
        majority_total += len(labels)
    majority_acc = majority_correct / majority_total if majority_total > 0 else None

    return {
        "aggregate": agg,
        "per_family": per_family,
        "n_qualifying_families": len(qualifying),
        "n_families_run": len(per_family),
        "majority_accuracy": majority_acc,
        "col_names": col_names,
        "label": label,
    }


def run_seed(seed: int, qualifying: dict, gene_to_row: dict,
             prot_matrix: np.ndarray, bad_matrix: np.ndarray) -> dict:
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
    results = {"seed": seed}

    # 1. Raw proteome features (cross-family signal, baseline)
    print(f"\n--- RAW proteome features (cross-family baseline) ---")
    results["raw_proteome"] = run_feature_set(
        qualifying, gene_to_row, prot_matrix,
        RAW_INDICES, RAW_NAMES, "raw_prot", seed)

    # 2. Family-residual proteome features (within-family signal)
    print(f"\n--- RESIDUAL proteome features (within-family) ---")
    results["resid_proteome"] = run_feature_set(
        qualifying, gene_to_row, prot_matrix,
        RESID_INDICES, [n + "_resid" for n in RAW_NAMES], "resid_prot", seed)

    # 3. Raw Badonyi (cross-family structural prior)
    print(f"\n--- RAW Badonyi pDN/pGOF/pLOF ---")
    results["raw_badonyi"] = run_feature_set(
        qualifying, gene_to_row, bad_matrix,
        BAD_RAW_INDICES, ["pDN", "pGOF", "pLOF"], "raw_bad", seed)

    # 4. Residual Badonyi (within-family structural variation)
    print(f"\n--- RESIDUAL Badonyi (within-family structural) ---")
    results["resid_badonyi"] = run_feature_set(
        qualifying, gene_to_row, bad_matrix,
        BAD_RESID_INDICES, ["pDN_resid", "pGOF_resid", "pLOF_resid"], "resid_bad", seed)

    # 5. Combined residuals: proteome + Badonyi within-family
    combined_matrix = np.concatenate([prot_matrix, bad_matrix], axis=1)
    combined_resid_idx = RESID_INDICES + [prot_matrix.shape[1] + i for i in BAD_RESID_INDICES]
    combined_resid_names = [n + "_resid" for n in RAW_NAMES] + ["pDN_resid", "pGOF_resid", "pLOF_resid"]
    print(f"\n--- COMBINED residuals: proteome + Badonyi (within-family) ---")
    results["resid_combined"] = run_feature_set(
        qualifying, gene_to_row, combined_matrix,
        combined_resid_idx, combined_resid_names, "resid_comb", seed)

    # Print seed summary
    print(f"\n--- Seed {seed} summary ---")
    majority_f1 = results["raw_proteome"]["majority_accuracy"]
    print(f"  Majority accuracy (per-family modal class): {majority_f1:.3f}" if majority_f1 else "  Majority: N/A")
    for key, label in [
        ("raw_proteome", "Raw proteome    "),
        ("resid_proteome", "Resid proteome  "),
        ("raw_badonyi", "Raw Badonyi     "),
        ("resid_badonyi", "Resid Badonyi   "),
        ("resid_combined", "Resid combined  "),
    ]:
        agg = results[key]["aggregate"]
        f1 = agg.get("macro_f1")
        n = agg.get("n", 0)
        dn = agg["per_class_auroc"].get("DN")
        gof = agg["per_class_auroc"].get("GOF")
        lof = agg["per_class_auroc"].get("LOF")
        def _f(v): return f"{v:.3f}" if v is not None else "N/A"
        print(f"  {label}  F1={_f(f1)}  n={n}  "
              f"GOF={_f(gof)}  DN={_f(dn)}  LOF={_f(lof)}")

    return results


def aggregate_seeds(all_results: list[dict]) -> dict:
    summary: dict = {"n_seeds": len(all_results)}
    for key in ["raw_proteome", "resid_proteome", "raw_badonyi",
                "resid_badonyi", "resid_combined"]:
        f1_vals = [r[key]["aggregate"]["macro_f1"] for r in all_results
                   if r[key]["aggregate"].get("macro_f1") is not None]
        summary[f"{key}_macro_f1_mean"] = float(np.mean(f1_vals)) if f1_vals else None
        summary[f"{key}_macro_f1_std"] = float(np.std(f1_vals)) if f1_vals else None
        for cls in CLASSES:
            vals = [r[key]["aggregate"]["per_class_auroc"].get(cls)
                    for r in all_results
                    if r[key]["aggregate"]["per_class_auroc"].get(cls) is not None]
            summary[f"{key}_auroc_{cls}_mean"] = float(np.mean(vals)) if vals else None
            summary[f"{key}_auroc_{cls}_std"] = float(np.std(vals)) if vals else None
    return summary


def main():
    parser = argparse.ArgumentParser(description="Result 16: within-family mechanism prediction")
    parser.add_argument("--min-genes", type=int, default=6)
    parser.add_argument("--min-classes", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else \
        PROJECT_DIR / "results" / "within_family"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Loading gene list ===")
    genes, mechs = load_gene_list()
    print(f"  {len(genes)} genes")

    print("\n=== Loading Pfam families ===")
    pfam_map = load_pfam()

    print("\n=== Finding qualifying families ===")
    qualifying = get_qualifying_families(
        genes, mechs, pfam_map, args.min_genes, args.min_classes)
    print(f"  {len(qualifying)} qualifying families "
          f"(>= {args.min_genes} genes, >= {args.min_classes} classes)")
    for fam, gmap in sorted(qualifying.items(), key=lambda x: -len(x[1])):
        cc = Counter(gmap.values())
        print(f"    {fam:<20} n={len(gmap):2d}  {dict(cc)}")

    print("\n=== Loading feature matrices ===")
    prot_matrix = np.load(PROTEOME_FEATURES).astype(np.float32)
    bad_matrix = np.load(BADONYI_FEATURES).astype(np.float32)
    gene_to_row = build_gene_to_row()
    print(f"  Proteome: {prot_matrix.shape}  Badonyi: {bad_matrix.shape}")

    # Sanity check residual col indices
    assert max(RESID_INDICES) < prot_matrix.shape[1], \
        f"Residual index {max(RESID_INDICES)} out of range for matrix width {prot_matrix.shape[1]}"
    assert max(BAD_RESID_INDICES) < bad_matrix.shape[1], \
        f"Badonyi residual index out of range"

    all_results = []
    for seed in args.seeds:
        res = run_seed(seed, qualifying, gene_to_row, prot_matrix, bad_matrix)
        all_results.append(res)
        path = out_dir / f"within_family_seed{seed}.json"
        path.write_text(json.dumps(res, indent=2, default=str))
        print(f"\n  Saved: {path}")

    summary = aggregate_seeds(all_results)
    summary_path = out_dir / "within_family_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary: {summary_path}")

    # Final summary table
    print("\n" + "="*72)
    print("SUMMARY — macro-F1 (mean ± std across seeds, LOGO CV within families)")
    print("="*72)
    print(f"{'Feature set':<22} {'F1':<16} {'GOF':<12} {'DN':<12} {'LOF':<12}")
    print("-"*72)

    def fmt(m, s):
        if m is None: return "   N/A     "
        return f"{m:.3f}±{s:.3f}"

    for key, label in [
        ("raw_proteome",   "Raw proteome   "),
        ("resid_proteome", "Resid proteome "),
        ("raw_badonyi",    "Raw Badonyi    "),
        ("resid_badonyi",  "Resid Badonyi  "),
        ("resid_combined", "Resid combined "),
    ]:
        f1 = fmt(summary.get(f"{key}_macro_f1_mean"),
                 summary.get(f"{key}_macro_f1_std") or 0)
        gof = fmt(summary.get(f"{key}_auroc_GOF_mean"),
                  summary.get(f"{key}_auroc_GOF_std") or 0)
        dn = fmt(summary.get(f"{key}_auroc_DN_mean"),
                 summary.get(f"{key}_auroc_DN_std") or 0)
        lof = fmt(summary.get(f"{key}_auroc_LOF_mean"),
                  summary.get(f"{key}_auroc_LOF_std") or 0)
        print(f"{label:<22} {f1:<16} {gof:<12} {dn:<12} {lof:<12}")

    print("="*72)
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
