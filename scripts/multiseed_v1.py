"""
Multi-seed replication of v1 headline numbers.

Runs seeds 1–4 (seed 0 already exists) for:
  1. Mechanism MLP probe (delta_mean, delta_pos) — gene-split + family-split
     on both Gerasimavicius and merged datasets.
  2. Pathogenicity control (delta_mean MLP + logreg) — gene-split + family-split.

All embeddings are cached locally — no GPU required.

Outputs:
  results/v1_multiseed/mechanism_geras_seed{N}.json
  results/v1_multiseed/mechanism_merged_seed{N}.json
  results/v1_multiseed/pathogenicity_seed{N}.json
  results/v1_multiseed/summary.json   ← 5-seed mean ± std across all headline numbers

Usage:
  cd esm2_mechanism
  python scripts/multiseed_v1.py
  python scripts/multiseed_v1.py --seeds 0 1 2 3 4   # re-run all including seed 0
"""

import argparse
import json
import os
import sys
import numpy as np
from collections import defaultdict
import functools
print = functools.partial(print, flush=True)
from utils_probes import (
    gene_split_cv, family_split_cv, run_logreg_binary_cv,
)

# ── paths ────────────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
EMB_DIR = os.path.join(DATA_DIR, "embeddings")
OUT_DIR = os.path.join(ROOT, "results", "v1_multiseed")
SEED0_DIR = os.path.join(ROOT, "results", "20260524_baseline_run", "run_0")

PFAM_JSON = os.path.join(DATA_DIR, "pfam_families.json")
GERAS_VARIANTS = os.path.join(DATA_DIR, "gerasimavicius_variants.json")
MERGED_VARIANTS = os.path.join(DATA_DIR, "merged_valid_variants.json")

# Gerasimavicius embeddings
GERAS_EMB_WT_MEAN  = os.path.join(EMB_DIR, "embeddings_wt_esm2_t33_650M_UR50D.npy")
GERAS_EMB_MUT_MEAN = os.path.join(EMB_DIR, "embeddings_mut_esm2_t33_650M_UR50D.npy")
GERAS_EMB_WT_POS   = os.path.join(EMB_DIR, "embeddings_wt_pos_esm2_t33_650M_UR50D.npy")
GERAS_EMB_MUT_POS  = os.path.join(EMB_DIR, "embeddings_mut_pos_esm2_t33_650M_UR50D.npy")

# Merged embeddings
MERGED_EMB_WT_MEAN  = os.path.join(EMB_DIR, "merged_embeddings_wt_mean.npy")
MERGED_EMB_MUT_MEAN = os.path.join(EMB_DIR, "merged_embeddings_mut_mean.npy")

# Pathogenicity embeddings
PATH_EMB_WT  = os.path.join(EMB_DIR, "emb_wt_mean_pathogenicity_esm2_t33_650M_UR50D_n17259.npy")
PATH_EMB_MUT = os.path.join(EMB_DIR, "emb_mut_mean_pathogenicity_esm2_t33_650M_UR50D_n17259.npy")
PATH_VARIANTS = os.path.join(DATA_DIR, "clinvar_pathogenicity_variants.json")


# ── CV helpers ───────────────────────────────────────────────────────────────


# ── MLP probe (PyTorch) ──────────────────────────────────────────────────────

def run_mlp_probe(X, labels, splits, seed=42, hidden=(256, 64), dropout=0.3,
                  lr=1e-3, max_epochs=100, patience=10, batch_size=256):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import roc_auc_score, f1_score

    le = LabelEncoder()
    le.fit(["GOF", "DN", "LOF"])
    y = le.transform(labels)
    classes = le.classes_
    n_classes = len(classes)
    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_tr = X[train_idx].astype(np.float32)
        X_te = X[test_idx].astype(np.float32)
        y_tr, y_te = y[train_idx], y[test_idx]
        if len(set(y_tr)) < 3:
            continue

        tr_genes_local = np.arange(len(train_idx))
        rng = np.random.RandomState(seed + fold_i)
        rng.shuffle(tr_genes_local)
        n_val = max(1, int(0.15 * len(tr_genes_local)))
        val_idx_local = tr_genes_local[:n_val]
        fit_idx_local = tr_genes_local[n_val:]
        X_fit, y_fit = X_tr[fit_idx_local], y_tr[fit_idx_local]
        X_val, y_val = X_tr[val_idx_local], y_tr[val_idx_local]
        if len(X_fit) < 10 or len(X_val) < 5:
            continue

        mu = X_fit.mean(0); std = X_fit.std(0) + 1e-8
        X_fit = (X_fit - mu) / std
        X_val = (X_val - mu) / std
        X_te_n = (X_te - mu) / std

        class_counts = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
        cw = torch.tensor(1.0 / (class_counts + 1e-8))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layers = []
        prev = X_fit.shape[1]
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        model = nn.Sequential(*layers).to(device)
        cw = cw.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        crit = nn.CrossEntropyLoss(weight=cw)

        ds = TensorDataset(torch.tensor(X_fit), torch.tensor(y_fit, dtype=torch.long))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        best_val, patience_cnt, best_state = float("inf"), 0, None
        for epoch in range(max_epochs):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                crit(model(xb.to(device)), yb.to(device)).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = crit(model(torch.tensor(X_val).to(device)),
                          torch.tensor(y_val, dtype=torch.long).to(device)).item()
            if vl < best_val - 1e-4:
                best_val, patience_cnt = vl, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    break
        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            proba = torch.softmax(model(torch.tensor(X_te_n).to(device)), 1).cpu().numpy()
        pred = proba.argmax(1)
        fm = {"macro_f1": float(f1_score(y_te, pred, average="macro", zero_division=0))}
        for i, cls in enumerate(classes):
            yb = (y_te == i).astype(int)
            if yb.sum() > 0 and (1 - yb).sum() > 0:
                fm[f"auroc_{cls}"] = float(roc_auc_score(yb, proba[:, i]))
        fold_results.append(fm)
        print(f"    fold {fold_i+1}: macro_f1={fm['macro_f1']:.3f}")

    if not fold_results:
        return {}
    agg = {}
    for key in set().union(*[set(f) for f in fold_results]):
        vals = [f[key] for f in fold_results if key in f and not np.isnan(f[key])]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"]  = float(np.std(vals))
    return agg


# ── Logistic regression probe (binary, for pathogenicity) ────────────────────


def run_mlp_binary(X, y, splits, seed=42):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    aurocs = []
    for tr, te in splits:
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300,
                            random_state=seed, early_stopping=True,
                            validation_fraction=0.1)
        clf.fit(Xtr, y[tr])
        proba = clf.predict_proba(Xte)[:, list(clf.classes_).index(1)]
        aurocs.append(float(roc_auc_score(y[te], proba)))
    if not aurocs:
        return {}
    return {"auroc_mean": float(np.mean(aurocs)), "auroc_std": float(np.std(aurocs)),
            "n_folds": len(aurocs)}


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_geras(pfam_map):
    # Load cached valid variants if available, otherwise derive from basic filter
    geras_valid_cache = os.path.join(DATA_DIR, "geras_valid_variants.json")
    if os.path.exists(geras_valid_cache):
        with open(geras_valid_cache) as f:
            variants = json.load(f)
    else:
        raise FileNotFoundError(
            f"{geras_valid_cache} not found.\n"
            "This file must list exactly the variants whose embeddings were extracted "
            "(in the same order), so that label and embedding indices align.\n"
            "Generate it by running esm2_mechanism.py and saving the valid_variants "
            "list after apply_missense filtering, or reconstruct it by replaying the "
            "exact sequence-fetch + apply_missense pipeline used when the embeddings "
            "were produced. Truncating from the tail is incorrect because apply_missense "
            "failures are not positionally determined."
        )

    for v in variants:
        if "label_3class" not in v:
            v["label_3class"] = "LOF" if v.get("mechanism") in ("HI", "AR") else v.get("mechanism", "LOF")
    labels = np.array([v["label_3class"] for v in variants])
    genes  = np.array([v["gene"] for v in variants])
    wt  = np.load(GERAS_EMB_WT_MEAN)
    mut = np.load(GERAS_EMB_MUT_MEAN)
    delta_mean = mut - wt
    wt_p  = np.load(GERAS_EMB_WT_POS)
    mut_p = np.load(GERAS_EMB_MUT_POS)
    delta_pos = mut_p - wt_p
    assert len(delta_mean) == len(labels), f"Geras embedding/variant count mismatch: {len(delta_mean)} vs {len(labels)}"
    print(f"  Gerasimavicius: {len(variants)} variants, {len(set(genes))} genes")
    return delta_mean, delta_pos, labels, genes


def load_merged(pfam_map):
    with open(MERGED_VARIANTS) as f:
        variants = json.load(f)
    for v in variants:
        if "label_3class" not in v:
            v["label_3class"] = "LOF" if v.get("mechanism") in ("HI", "AR") else v.get("mechanism", "LOF")
    labels = np.array([v["label_3class"] for v in variants])
    genes  = np.array([v["gene"] for v in variants])
    wt  = np.load(MERGED_EMB_WT_MEAN)
    mut = np.load(MERGED_EMB_MUT_MEAN)
    delta_mean = mut - wt
    assert len(delta_mean) == len(labels), f"Merged embedding/variant count mismatch: {len(delta_mean)} vs {len(labels)}"
    print(f"  Merged: {len(variants)} variants, {len(set(genes))} genes")
    return delta_mean, labels, genes


def load_pathogenicity(pfam_map):
    path_valid_cache = os.path.join(DATA_DIR, "pathogenicity_valid_variants.json")
    if os.path.exists(path_valid_cache):
        with open(path_valid_cache) as f:
            variants = json.load(f)
    else:
        with open(PATH_VARIANTS) as f:
            variants = json.load(f)
        emb_n = np.load(PATH_EMB_WT, mmap_mode="r").shape[0]
        if len(variants) > emb_n:
            print(f"  Warning: {len(variants)} pathogenicity variants but {emb_n} embeddings; "
                  f"truncating to {emb_n} (23 dropped on RunPod)")
            variants = variants[:emb_n]
        with open(path_valid_cache, "w") as f:
            json.dump(variants, f)
        print(f"  Saved {path_valid_cache}")

    genes = np.array([v["gene"] for v in variants])
    y = np.array([1 if v["label"] == "pathogenic" else 0 for v in variants])
    wt  = np.load(PATH_EMB_WT)
    mut = np.load(PATH_EMB_MUT)
    delta = mut - wt
    assert len(delta) == len(y), f"Path embedding/variant count mismatch: {len(delta)} vs {len(y)}"
    print(f"  Pathogenicity: {len(variants)} variants, {len(set(genes))} genes, "
          f"{int(y.sum())} pathogenic / {int((1-y).sum())} benign")
    return delta, y, genes


# ── Main ──────────────────────────────────────────────────────────────────────

def run_seed(seed, pfam_map, out_dir):
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")

    # ── 1. Gerasimavicius mechanism ──────────────────────────────────────────
    geras_out = os.path.join(out_dir, f"mechanism_geras_seed{seed}.json")
    if os.path.exists(geras_out):
        print(f"  [skip] {geras_out} already exists")
        geras_results = json.load(open(geras_out))
    else:
        print("\n--- Gerasimavicius mechanism ---")
        dm, dp, labels, genes = load_geras(pfam_map)
        gs = gene_split_cv(genes, seed=seed)
        fs = family_split_cv(genes, pfam_map, seed=seed)
        geras_results = {}
        for feat_name, X in [("delta_mean", dm), ("delta_pos", dp)]:
            print(f"  MLP gene-split {feat_name}")
            geras_results[f"mlp_{feat_name}_gene"] = run_mlp_probe(X, labels, gs, seed=seed)
            print(f"  MLP family-split {feat_name}")
            geras_results[f"mlp_{feat_name}_family"] = run_mlp_probe(X, labels, fs, seed=seed)
        with open(geras_out, "w") as f:
            json.dump(geras_results, f, indent=2)
        print(f"  -> {geras_out}")

    # ── 2. Merged dataset mechanism ──────────────────────────────────────────
    merged_out = os.path.join(out_dir, f"mechanism_merged_seed{seed}.json")
    if os.path.exists(merged_out):
        print(f"  [skip] {merged_out} already exists")
        merged_results = json.load(open(merged_out))
    else:
        print("\n--- Merged dataset mechanism ---")
        dm, labels, genes = load_merged(pfam_map)
        gs = gene_split_cv(genes, seed=seed)
        fs = family_split_cv(genes, pfam_map, seed=seed)
        merged_results = {}
        print(f"  MLP gene-split delta_mean")
        merged_results["mlp_delta_mean_gene"] = run_mlp_probe(dm, labels, gs, seed=seed)
        print(f"  MLP family-split delta_mean")
        merged_results["mlp_delta_mean_family"] = run_mlp_probe(dm, labels, fs, seed=seed)
        with open(merged_out, "w") as f:
            json.dump(merged_results, f, indent=2)
        print(f"  -> {merged_out}")

    # ── 3. Pathogenicity control ─────────────────────────────────────────────
    path_out = os.path.join(out_dir, f"pathogenicity_seed{seed}.json")
    if os.path.exists(path_out):
        print(f"  [skip] {path_out} already exists")
        path_results = json.load(open(path_out))
    else:
        print("\n--- Pathogenicity control ---")
        delta, y, genes = load_pathogenicity(pfam_map)
        # Use gene-symbol based pfam lookup for pathogenicity genes
        gs = gene_split_cv(genes, seed=seed)
        fs = family_split_cv(genes, pfam_map, seed=seed)
        path_results = {}
        print(f"  logreg gene-split")
        path_results["logreg_gene"] = run_logreg_binary_cv(delta, y, gs, seed=seed)
        print(f"  logreg family-split")
        path_results["logreg_family"] = run_logreg_binary_cv(delta, y, fs, seed=seed)
        print(f"  MLP gene-split")
        path_results["mlp_gene"] = run_mlp_binary(delta, y, gs, seed=seed)
        print(f"  MLP family-split")
        path_results["mlp_family"] = run_mlp_binary(delta, y, fs, seed=seed)
        with open(path_out, "w") as f:
            json.dump(path_results, f, indent=2)
        print(f"  -> {path_out}")

    return geras_results, merged_results, path_results


def summarise(all_seeds, out_dir):
    """Aggregate across seeds. all_seeds: list of (seed, geras, merged, path) tuples."""

    def agg(values):
        v = [x for x in values if x is not None and not np.isnan(x)]
        if not v:
            return None, None
        return float(np.mean(v)), float(np.std(v))

    # headline keys
    metrics = {
        "mechanism_geras_mlp_delta_mean_family_macro_f1": [],
        "mechanism_geras_mlp_delta_mean_family_auroc_GOF": [],
        "mechanism_geras_mlp_delta_mean_family_auroc_DN": [],
        "mechanism_geras_mlp_delta_mean_family_auroc_LOF": [],
        "mechanism_geras_mlp_delta_mean_gene_macro_f1": [],
        "mechanism_merged_mlp_delta_mean_family_macro_f1": [],
        "mechanism_merged_mlp_delta_mean_family_auroc_GOF": [],
        "mechanism_merged_mlp_delta_mean_family_auroc_DN": [],
        "mechanism_merged_mlp_delta_mean_family_auroc_LOF": [],
        "mechanism_merged_mlp_delta_mean_gene_macro_f1": [],
        "pathogenicity_mlp_gene_auroc": [],
        "pathogenicity_mlp_family_auroc": [],
        "pathogenicity_logreg_gene_auroc": [],
        "pathogenicity_logreg_family_auroc": [],
    }

    per_seed = {}
    for seed, geras, merged, path in all_seeds:
        per_seed[seed] = {}

        def g(d, *keys):
            v = d
            for k in keys:
                if not isinstance(v, dict) or k not in v:
                    return None
                v = v[k]
            return v

        vals = {
            "mechanism_geras_mlp_delta_mean_family_macro_f1":   g(geras, "mlp_delta_mean_family", "macro_f1_mean"),
            "mechanism_geras_mlp_delta_mean_family_auroc_GOF":  g(geras, "mlp_delta_mean_family", "auroc_GOF_mean"),
            "mechanism_geras_mlp_delta_mean_family_auroc_DN":   g(geras, "mlp_delta_mean_family", "auroc_DN_mean"),
            "mechanism_geras_mlp_delta_mean_family_auroc_LOF":  g(geras, "mlp_delta_mean_family", "auroc_LOF_mean"),
            "mechanism_geras_mlp_delta_mean_gene_macro_f1":     g(geras, "mlp_delta_mean_gene", "macro_f1_mean"),
            "mechanism_merged_mlp_delta_mean_family_macro_f1":  g(merged, "mlp_delta_mean_family", "macro_f1_mean"),
            "mechanism_merged_mlp_delta_mean_family_auroc_GOF": g(merged, "mlp_delta_mean_family", "auroc_GOF_mean"),
            "mechanism_merged_mlp_delta_mean_family_auroc_DN":  g(merged, "mlp_delta_mean_family", "auroc_DN_mean"),
            "mechanism_merged_mlp_delta_mean_family_auroc_LOF": g(merged, "mlp_delta_mean_family", "auroc_LOF_mean"),
            "mechanism_merged_mlp_delta_mean_gene_macro_f1":    g(merged, "mlp_delta_mean_gene", "macro_f1_mean"),
            "pathogenicity_mlp_gene_auroc":     g(path, "mlp_gene", "auroc_mean"),
            "pathogenicity_mlp_family_auroc":   g(path, "mlp_family", "auroc_mean"),
            "pathogenicity_logreg_gene_auroc":  g(path, "logreg_gene", "auroc_mean"),
            "pathogenicity_logreg_family_auroc": g(path, "logreg_family", "auroc_mean"),
        }
        per_seed[seed] = vals
        for k, v in vals.items():
            metrics[k].append(v)

    summary = {"per_seed": per_seed, "aggregate": {}}
    print("\n" + "=" * 60)
    print("V1 MULTISEED SUMMARY")
    print("=" * 60)
    for k, vals in metrics.items():
        mean, std = agg(vals)
        summary["aggregate"][k] = {"mean": mean, "std": std, "n": len([v for v in vals if v is not None])}
        if mean is not None:
            print(f"  {k:<60s}  {mean:.3f} ± {std:.3f}  (n={summary['aggregate'][k]['n']})")

    # Headline leakage fraction
    gf_vals = metrics["mechanism_geras_mlp_delta_mean_gene_macro_f1"]
    fs_vals = metrics["mechanism_geras_mlp_delta_mean_family_macro_f1"]
    chance = 0.333
    leakage_fracs = []
    for gf, fs in zip(gf_vals, fs_vals):
        if gf is not None and fs is not None and (gf - chance) > 0:
            leakage_fracs.append((gf - fs) / (gf - chance))
    if leakage_fracs:
        print(f"\n  Leakage fraction (Gerasimavicius):  {np.mean(leakage_fracs):.1%} ± {np.std(leakage_fracs):.1%}")

    path_delta_vals = []
    for s in summary["per_seed"].values():
        gv = s.get("pathogenicity_mlp_gene_auroc")
        fv = s.get("pathogenicity_mlp_family_auroc")
        if gv is not None and fv is not None:
            path_delta_vals.append(gv - fv)
    if path_delta_vals:
        print(f"  Pathogenicity gene→family Δ:         {np.mean(path_delta_vals):.3f} ± {np.std(path_delta_vals):.3f}")

    out = os.path.join(out_dir, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {out}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="Seeds to run (default: 1 2 3 4; seed 0 already exists)")
    parser.add_argument("--include_seed0", action="store_true",
                        help="Also re-run seed 0 (normally skipped — already exists)")
    parser.add_argument("--summarise_only", action="store_true",
                        help="Skip computation, just re-aggregate existing JSONs")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(PFAM_JSON) as f:
        pfam_map = json.load(f)
    print(f"Pfam map loaded: {len(pfam_map)} genes")

    seeds = args.seeds
    if args.include_seed0:
        seeds = [0] + seeds

    all_seeds = []

    # Always include seed 0 from existing results
    seed0_geras = json.load(open(os.path.join(SEED0_DIR, "mlp_results_seed0.json")))
    seed0_merged = json.load(open(os.path.join(SEED0_DIR, "mlp_merged_results_seed0.json")))
    seed0_path = json.load(open(os.path.join(SEED0_DIR, "pathogenicity_control.json")))

    # Reformat seed 0 pathogenicity to match our schema
    def reformat_path_seed0(d):
        bf = d.get("by_feature", {})
        dm = bf.get("delta_mean", {})
        return {
            "logreg_gene":   {"auroc_mean": dm.get("gene_split_logreg", {}).get("auroc_mean")},
            "logreg_family": {"auroc_mean": dm.get("family_split_logreg", {}).get("auroc_mean")},
            "mlp_gene":      {"auroc_mean": dm.get("gene_split_mlp", {}).get("auroc_mean")},
            "mlp_family":    {"auroc_mean": dm.get("family_split_mlp", {}).get("auroc_mean")},
        }

    all_seeds.append((0, seed0_geras, seed0_merged, reformat_path_seed0(seed0_path)))

    if not args.summarise_only:
        for seed in seeds:
            if seed == 0 and not args.include_seed0:
                continue
            g, m, p = run_seed(seed, pfam_map, OUT_DIR)
            all_seeds.append((seed, g, m, p))
    else:
        for seed in seeds:
            gf = os.path.join(OUT_DIR, f"mechanism_geras_seed{seed}.json")
            mf = os.path.join(OUT_DIR, f"mechanism_merged_seed{seed}.json")
            pf = os.path.join(OUT_DIR, f"pathogenicity_seed{seed}.json")
            if os.path.exists(gf) and os.path.exists(mf) and os.path.exists(pf):
                all_seeds.append((seed,
                                   json.load(open(gf)),
                                   json.load(open(mf)),
                                   json.load(open(pf))))
            else:
                print(f"Warning: seed {seed} results not found, skipping")

    summarise(all_seeds, OUT_DIR)


if __name__ == "__main__":
    main()
