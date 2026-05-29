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

import json, os, sys, numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
EMB  = DATA / "embeddings"
OUT  = ROOT / "results" / "perturbation_scan"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from multiseed_v1 import gene_split_cv, family_split_cv


DECISION_RULES = {
    "G1": ("scan_only_family_split",         "macro_f1_mean", 0.368),
    "G2": ("scan_delta_family_split",        "macro_f1_mean", 0.419),
    "G3": ("scan_proteome_family_split",     "macro_f1_mean", 0.405),
}


def load_all_features(gene_list):
    """Load all feature matrices aligned to gene_list."""
    features = {}

    # 1. Scan features (phase 3 output)
    scan_X = np.load(DATA / "scan_features.npy")
    with open(DATA / "scan_features_meta.json") as f:
        scan_meta = json.load(f)
    scan_genes = np.array(scan_meta["genes"])
    # Align to gene_list
    scan_idx = {g: i for i, g in enumerate(scan_genes)}
    aligned_scan = np.array([scan_X[scan_idx[g]] for g in gene_list
                              if g in scan_idx], dtype=np.float32)
    # Filter gene_list to those with scan features
    gene_mask = np.array([g in scan_idx for g in gene_list])
    scan_gene_list = gene_list[gene_mask]   # genes that have scan features
    features["scan"] = aligned_scan

    # 2. Mean-pooled delta — built over scan_gene_list so rows align to scan features
    with open(DATA / "merged_valid_variants.json") as f:
        variants = json.load(f)
    wt_emb  = np.load(EMB / "merged_embeddings_wt_mean.npy")
    mut_emb = np.load(EMB / "merged_embeddings_mut_mean.npy")
    delta   = mut_emb - wt_emb

    from collections import defaultdict
    gene_delta = defaultdict(list)
    for i, v in enumerate(variants):
        gene_delta[v["gene"].upper()].append(delta[i])
    delta_X = np.array([np.mean(gene_delta[g], axis=0) for g in scan_gene_list
                         if g in gene_delta], dtype=np.float32)
    features["delta"] = delta_X

    # 3. Proteome features — built over scan_gene_list so rows align to scan features
    proteome_path = DATA / "proteome_features_aligned.npy"
    if proteome_path.exists():
        with open(DATA / "merged_gene_list.tsv") as f:
            merged_genes = [line.split("\t")[0].strip() for line in f
                            if not line.startswith("gene")]
        proteome_X = np.load(proteome_path)
        pg_idx = {g: i for i, g in enumerate(merged_genes)}
        proteome_aligned = np.array([proteome_X[pg_idx[g]] for g in scan_gene_list
                                      if g in pg_idx], dtype=np.float32)
        features["proteome"] = proteome_aligned
    else:
        print("  proteome_features_aligned.npy not found — skipping proteome features")

    # 4. Badonyi features
    badonyi_path = DATA / "badonyi_features_aligned.npy"
    if badonyi_path.exists():
        features["badonyi"] = np.load(badonyi_path)[:len(gene_list)]

    return features, gene_mask


def run_probe(X, labels, splits, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, f1_score
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(labels)
    classes = le.classes_
    fold_results = []

    for tr, te in splits:
        if len(set(y[tr])) < 2: continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        clf = LogisticRegression(max_iter=2000, C=1.0,
                                  class_weight="balanced", random_state=seed)
        clf.fit(Xtr, y[tr])
        proba = clf.predict_proba(Xte)
        pred  = clf.predict(Xte)
        fm = {"macro_f1": float(f1_score(y[te], pred, average="macro", zero_division=0))}
        for i, cls in enumerate(classes):
            yb = (y[te] == i).astype(int)
            if yb.sum() > 0 and (1-yb).sum() > 0:
                fm[f"auroc_{cls}"] = float(roc_auc_score(yb, proba[:, i]))
        fold_results.append(fm)

    if not fold_results: return {}
    agg = {}
    for key in set().union(*[set(f) for f in fold_results]):
        vals = [f[key] for f in fold_results if key in f]
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"]  = float(np.std(vals))
    return agg


def main():
    print("=== Loading data ===")

    # Gene list and labels from merged dataset
    with open(DATA / "merged_valid_variants.json") as f:
        variants = json.load(f)
    for v in variants:
        if "label_3class" not in v:
            v["label_3class"] = "LOF" if v.get("mechanism") in ("HI","AR") else v.get("mechanism","LOF")

    # Gene-level: one label per gene (majority vote)
    from collections import Counter, defaultdict
    gene_labels = defaultdict(list)
    for v in variants:
        gene_labels[v["gene"].upper()].append(v["label_3class"])
    gene_list = np.array(sorted(gene_labels.keys()))
    labels    = np.array([Counter(gene_labels[g]).most_common(1)[0][0] for g in gene_list])

    print(f"Genes: {len(gene_list)}  Classes: {dict(Counter(labels))}")

    # Load features
    features, gene_mask = load_all_features(gene_list)

    # Filter to genes with scan features
    gene_list_scan = gene_list[gene_mask]
    labels_scan    = labels[gene_mask]
    print(f"Genes with scan features: {len(gene_list_scan)}")

    # Pfam map
    with open(DATA / "pfam_families.json") as f:
        pfam_map = json.load(f)

    # Feature combinations to test
    scan_X    = features["scan"]
    delta_X   = features["delta"] if "delta" in features else None
    proteome_X = features.get("proteome")

    combos = {
        "baseline_delta":  delta_X,
        "scan_only":       scan_X,
        "scan_delta":      np.hstack([scan_X, delta_X]) if delta_X is not None else None,
        "scan_proteome":   np.hstack([scan_X, proteome_X]) if proteome_X is not None else None,
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
                f1  = r.get("macro_f1_mean", float("nan"))
                gof = r.get("auroc_GOF_mean", float("nan"))
                print(f"  {key}: F1={f1:.3f}  GOF={gof:.3f}")
        all_results[seed] = seed_res

    # Summary
    print("\n=== 5-SEED SUMMARY ===")
    summary = {}
    for key in all_results[0].keys():
        f1_vals  = [all_results[s][key].get("macro_f1_mean", float("nan")) for s in range(5)]
        gof_vals = [all_results[s][key].get("auroc_GOF_mean", float("nan")) for s in range(5)]
        summary[key] = {
            "macro_f1_mean":  float(np.nanmean(f1_vals)),
            "macro_f1_std":   float(np.nanstd(f1_vals)),
            "auroc_GOF_mean": float(np.nanmean(gof_vals)),
            "auroc_GOF_std":  float(np.nanstd(gof_vals)),
        }
        print(f"  {key}: F1={summary[key]['macro_f1_mean']:.3f}±{summary[key]['macro_f1_std']:.3f}"
              f"  GOF={summary[key]['auroc_GOF_mean']:.3f}±{summary[key]['auroc_GOF_std']:.3f}")

    # Decision rules
    print("\n=== DECISION RULES ===")
    gate_results = {}
    for gate, (key, metric, threshold) in DECISION_RULES.items():
        val = summary.get(key, {}).get(metric, float("nan"))
        passed = val > threshold
        gate_results[gate] = {"value": val, "threshold": threshold, "passed": passed}
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {gate}: {key} {metric} = {val:.3f} (threshold {threshold:.3f}) → {status}")

    if not gate_results.get("G1", {}).get("passed"):
        print("\n  G1 failed — scan features do not improve on ClinVar-pattern baseline.")
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
