"""
Probe runs for perturbation scan features.

Loads scan_features.npy (output of perturbation_scan.py phase 3) and runs
logistic regression under gene-split and family-split CV, comparing:

  - Baseline: mean-pooled delta (result_7 / result_18 baseline)
  - Scan features only (5 pre-registered)
  - Scan + mean-pooled delta
  - Scan + proteome features (result_13)
  - Scan + Badonyi features (result_15)

Decision rules (pre-registered in plan_perturb.md):
  G1: scan-only family-split F1 > 0.368  (ClinVar-pattern + 0.02)
  G2: scan+delta family-split F1 > 0.419  (result_18 combined + 0.02)
  G3: scan+proteome family-split F1 > 0.405  (proteome-only + 0.02)

Usage:
  cd esm2_mechanism
  python3 scripts/perturbation_probe.py
"""

import functools
import json, os, sys, numpy as np
from pathlib import Path

from esm2_mechanism.utils_paths import DATA_DIR as DATA, RESULTS_DIR as _RESULTS_DIR, VALID_VARIANTS_JSON, EMB_WT_MEAN, EMB_MUT_MEAN, GENE_LIST_TSV
from esm2_mechanism.mechanism.multiseed_v1 import gene_split_cv, family_split_cv

print = functools.partial(print, flush=True)
EMB = DATA / "embeddings"
OUT = _RESULTS_DIR / "perturbation_scan"
OUT.mkdir(parents=True, exist_ok=True)


DECISION_RULES = {
    "G1": ("scan_only_family_split", "macro_f1_mean", 0.368),
    "G2": ("scan_delta_family_split", "macro_f1_mean", 0.419),
    "G3": ("scan_proteome_family_split", "macro_f1_mean", 0.405),
}


def load_all_features(gene_list):
    """Load all feature matrices aligned to gene_list.

    All returned arrays are row-aligned to the same filtered gene list.
    The second return value (gene_mask) maps rows back to gene_list.
    """
    features = {}

    # 1. Scan features (phase 3 output)
    scan_X = np.load(DATA / "scan_features.npy")
    with open(DATA / "scan_features_meta.json") as f:
        scan_meta = json.load(f)
    scan_genes = np.array(scan_meta["genes"])
    scan_idx = {g: i for i, g in enumerate(scan_genes)}

    # 2. Mean-pooled delta index
    with open(VALID_VARIANTS_JSON) as f:
        variants = json.load(f)
    wt_emb = np.load(EMB_WT_MEAN)
    mut_emb = np.load(EMB_MUT_MEAN)
    delta = mut_emb - wt_emb

    from collections import defaultdict

    gene_delta = defaultdict(list)
    for i, v in enumerate(variants):
        gene_delta[v["gene"].upper()].append(delta[i])

    # 3. Proteome / Badonyi gene-to-row index
    proteome_path = DATA / "proteome_features_aligned.npy"
    pg_idx: dict = {}
    if proteome_path.exists():
        with open(GENE_LIST_TSV) as f:
            merged_genes = [
                line.split("\t")[0].strip() for line in f if not line.startswith("gene")
            ]
        pg_idx = {g: i for i, g in enumerate(merged_genes)}

    badonyi_path = DATA / "badonyi_features_aligned.npy"

    # Determine which genes have ALL required features so every array has the
    # same row count (previously each source filtered independently, making
    # np.hstack silently produce misaligned concatenations).
    def _has_all(g):
        if g not in scan_idx:
            return False
        if g not in gene_delta:
            return False
        if proteome_path.exists() and g not in pg_idx:
            return False
        if badonyi_path.exists() and g not in pg_idx:
            return False
        return True

    gene_mask = np.array([_has_all(g) for g in gene_list])
    scan_gene_list = gene_list[gene_mask]
    print(f"  Genes with all features: {len(scan_gene_list)}/{len(gene_list)}")

    features["scan"] = np.array(
        [scan_X[scan_idx[g]] for g in scan_gene_list], dtype=np.float32
    )
    features["delta"] = np.array(
        [np.mean(gene_delta[g], axis=0) for g in scan_gene_list], dtype=np.float32
    )

    if proteome_path.exists():
        proteome_X = np.load(proteome_path)
        features["proteome"] = np.array(
            [proteome_X[pg_idx[g]] for g in scan_gene_list], dtype=np.float32
        )
    else:
        print("  proteome_features_aligned.npy not found — skipping proteome features")

    if badonyi_path.exists():
        badonyi_X = np.load(badonyi_path)
        features["badonyi"] = np.array(
            [badonyi_X[pg_idx[g]] for g in scan_gene_list], dtype=np.float32
        )

    # Sanity check: all feature arrays must have the same number of rows
    row_counts = {name: X.shape[0] for name, X in features.items()}
    if len(set(row_counts.values())) > 1:
        raise RuntimeError(f"Feature row count mismatch after alignment: {row_counts}")

    return features, gene_mask


def run_probe(X, labels, splits, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, f1_score
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(["GOF", "DN", "LOF"])
    y = le.transform(labels)
    classes = le.classes_
    fold_results = []

    for tr, te in splits:
        if len(set(y[tr])) < len(classes):
            continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        clf = LogisticRegression(
            max_iter=2000, C=1.0, class_weight="balanced", random_state=seed
        )
        clf.fit(Xtr, y[tr])
        raw_proba = clf.predict_proba(Xte)
        proba = np.zeros((len(Xte), len(classes)), dtype=np.float32)
        for ci, c in enumerate(clf.classes_):
            proba[:, c] = raw_proba[:, ci]
        pred = proba.argmax(axis=1)
        fm = {
            "macro_f1": float(f1_score(y[te], pred, average="macro", zero_division=0))
        }
        for i, cls in enumerate(classes):
            yb = (y[te] == i).astype(int)
            if yb.sum() > 0 and (1 - yb).sum() > 0:
                fm[f"auroc_{cls}"] = float(roc_auc_score(yb, proba[:, i]))
        fold_results.append(fm)

    if not fold_results:
        return {}
    agg = {}
    for key in set().union(*[set(f) for f in fold_results]):
        vals = [f[key] for f in fold_results if key in f]
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))
    return agg


def main():
    print("=== Loading data ===")

    # Gene list and labels from merged dataset
    with open(VALID_VARIANTS_JSON) as f:
        variants = json.load(f)
    for v in variants:
        if "label_3class" not in v:
            v["label_3class"] = (
                "LOF"
                if v.get("mechanism") in ("HI", "AR")
                else v.get("mechanism", "LOF")
            )

    # Gene-level: one label per gene (majority vote)
    from collections import Counter, defaultdict

    gene_labels = defaultdict(list)
    for v in variants:
        gene_labels[v["gene"].upper()].append(v["label_3class"])
    gene_list = np.array(sorted(gene_labels.keys()))
    labels = np.array([Counter(gene_labels[g]).most_common(1)[0][0] for g in gene_list])

    print(f"Genes: {len(gene_list)}  Classes: {dict(Counter(labels))}")

    # Load features
    features, gene_mask = load_all_features(gene_list)

    # Filter to genes with scan features
    gene_list_scan = gene_list[gene_mask]
    labels_scan = labels[gene_mask]
    print(f"Genes with scan features: {len(gene_list_scan)}")

    # Pfam map
    with open(DATA / "pfam_families.json") as f:
        pfam_map = json.load(f)

    # Feature combinations to test
    scan_X = features["scan"]
    delta_X = features["delta"] if "delta" in features else None
    proteome_X = features.get("proteome")

    combos = {
        "baseline_delta": delta_X,
        "scan_only": scan_X,
        "scan_delta": np.hstack([scan_X, delta_X]) if delta_X is not None else None,
        "scan_proteome": (
            np.hstack([scan_X, proteome_X]) if proteome_X is not None else None
        ),
    }
    combos = {k: v for k, v in combos.items() if v is not None}

    # 5-seed runs
    all_results = {}
    for seed in range(5):
        print(f"\n=== Seed {seed} ===")
        gs = gene_split_cv(gene_list_scan, seed=seed)
        fs = family_split_cv(gene_list_scan, pfam_map, seed=seed)
        seed_res = {}
        for combo_name, X in combos.items():
            for split_name, splits in [("gene_split", gs), ("family_split", fs)]:
                key = f"{combo_name}_{split_name}"
                r = run_probe(X, labels_scan, splits, seed=seed)
                seed_res[key] = r
                f1 = r.get("macro_f1_mean", float("nan"))
                gof = r.get("auroc_GOF_mean", float("nan"))
                print(f"  {key}: F1={f1:.3f}  GOF={gof:.3f}")
        all_results[seed] = seed_res

    # Summary
    print("\n=== 5-SEED SUMMARY ===")
    summary = {}
    for key in all_results[0].keys():
        f1_vals = [
            all_results[s][key].get("macro_f1_mean", float("nan")) for s in range(5)
        ]
        gof_vals = [
            all_results[s][key].get("auroc_GOF_mean", float("nan")) for s in range(5)
        ]
        summary[key] = {
            "macro_f1_mean": float(np.nanmean(f1_vals)),
            "macro_f1_std": float(np.nanstd(f1_vals)),
            "auroc_GOF_mean": float(np.nanmean(gof_vals)),
            "auroc_GOF_std": float(np.nanstd(gof_vals)),
        }
        print(
            f"  {key}: F1={summary[key]['macro_f1_mean']:.3f}±{summary[key]['macro_f1_std']:.3f}"
            f"  GOF={summary[key]['auroc_GOF_mean']:.3f}±{summary[key]['auroc_GOF_std']:.3f}"
        )

    # Decision rules
    print("\n=== DECISION RULES ===")
    gate_results = {}
    for gate, (key, metric, threshold) in DECISION_RULES.items():
        val = summary.get(key, {}).get(metric, float("nan"))
        passed = val > threshold
        gate_results[gate] = {"value": val, "threshold": threshold, "passed": passed}
        status = "PASS ✓" if passed else "FAIL ✗"
        print(
            f"  {gate}: {key} {metric} = {val:.3f} (threshold {threshold:.3f}) → {status}"
        )

    if not gate_results.get("G1", {}).get("passed"):
        print(
            "\n  G1 failed — scan features do not improve on ClinVar-pattern baseline."
        )
        print("  Do not proceed to G2/G3 analysis.")

    # Save
    out = {
        "summary": summary,
        "gate_results": gate_results,
        "per_seed": {str(s): all_results[s] for s in range(5)},
        "n_genes": int(len(gene_list_scan)),
        "feature_combos": list(combos.keys()),
    }
    out_path = OUT / "probe_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    main()
