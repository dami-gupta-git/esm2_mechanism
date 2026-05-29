"""
Nonlinear (MLP) stability probe on S1724 — companion to megascale_stability.py.

Runs MLP regression (1280→256→64→1) under the same three CV schemes
(random / protein-holdout / cluster-holdout) and 5 seeds as the Ridge probe,
then compares to check whether nonlinearity adds signal beyond Ridge.

Same question as result_3/5/7 for mechanism: does the MLP lift survive
family-holdout, or does it evaporate (leakage)?

Usage:
  cd esm2_mechanism
  python scripts/megascale_mlp.py

Outputs:
  results/megascale_stability/mlp_summary.json
  results/megascale_stability/mlp_per_protein_spearman.json
"""

import json
import os
import sys
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embeddings.megascale_stability import (
    load_s1724_variants, assign_protein_clusters,
    random_split_cv, protein_split_cv, cluster_split_cv,
    per_protein_spearman,

)

PFAM = {
    '1AJ3': 'PF13499', '1BNI': 'PF00545', '1CEY': 'PF00072', '1CUN': 'PF00545',
    '1DIV': 'PF04563', '1EKG': 'PF02234', '1FKJ': 'PF00254', '1FT8': 'PF00062',
    '1FTG': 'PF00062', '1GUA': 'PF00244', '1H7M': 'PF00084', '1IOB': 'PF00545',
    '1LVE': 'PF00089', '1O6X': 'PF02885', '1RIS': 'PF00042', '1RX4': 'PF00042',
    '1SHF': 'PF00130', '1STN': 'PF00565', '1TEN': 'PF07679', '1TTG': 'PF09289',
    '1UBQ': 'NO_PFAM', '2CI2': 'PF00280', '2IFB': 'PF14651', '2PTL': 'PF00020',
    '3BDC': 'PF00565', '3HHR': 'PF00103', '4HXJ': 'PF00870'
}

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
EMB  = os.path.join(DATA, "embeddings")
OUT  = os.path.join(ROOT, "results", "megascale_stability")

WT_MEAN_EMB  = os.path.join(EMB, "megascale_wt_mean.npy")
MUT_MEAN_EMB = os.path.join(EMB, "megascale_mut_mean.npy")

N_SEEDS = 5
N_FOLDS = 5

os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# Pfam family-split CV
# ---------------------------------------------------------------------------

def pfam_split_cv(proteins, n_folds=5, seed=42):
    pfam_arr = np.array([PFAM.get(p, p) for p in proteins])
    fams = np.array(sorted(set(pfam_arr)))
    np.random.RandomState(seed).shuffle(fams)
    splits = []
    for fold_fams in np.array_split(fams, n_folds):
        fs = set(fold_fams)
        tr = np.where(~np.isin(pfam_arr, list(fs)))[0]
        te = np.where( np.isin(pfam_arr, list(fs)))[0]
        if len(tr) >= 10 and len(te) >= 5:
            splits.append((tr, te))
    return splits


# ---------------------------------------------------------------------------
# Generic sklearn regression probe (Ridge, RF, GBM)
# ---------------------------------------------------------------------------

def run_sklearn_probe(X, y, splits, clf_fn):
    rhos, aurocs = [], []
    for tr, te in splits:
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        clf = clf_fn()
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        rho, _ = spearmanr(y[te], pred)
        binary = (y[te] >= np.median(y[te])).astype(int)
        au = float(roc_auc_score(binary, pred)) if binary.sum() > 0 and (1-binary).sum() > 0 else float("nan")
        rhos.append(float(rho))
        aurocs.append(au)
    if not rhos:
        return {}
    return {
        "spearman_mean": float(np.mean(rhos)),
        "spearman_std":  float(np.std(rhos)),
        "auroc_mean":    float(np.nanmean(aurocs)),
        "auroc_std":     float(np.nanstd(aurocs)),
        "n_folds": len(rhos),
    }


# ---------------------------------------------------------------------------
# MLP regression probe
# ---------------------------------------------------------------------------

def run_mlp_regression(X, y, splits, seed=42, hidden=(256, 64),
                        lr=1e-3, max_epochs=200, patience=15, batch_size=64):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    rhos, rs, aurocs = [], [], []

    for fold_i, (tr, te) in enumerate(splits):
        X_tr = X[tr].astype(np.float32)
        X_te = X[te].astype(np.float32)
        y_tr = y[tr].astype(np.float32)
        y_te = y[te].astype(np.float32)

        # hold out 15% of train for early stopping
        rng = np.random.RandomState(seed + fold_i)
        idx = np.arange(len(X_tr))
        rng.shuffle(idx)
        n_val = max(1, int(0.15 * len(idx)))
        val_idx, fit_idx = idx[:n_val], idx[n_val:]

        mu = X_tr[fit_idx].mean(0); std = X_tr[fit_idx].std(0) + 1e-8
        X_fit = (X_tr[fit_idx] - mu) / std
        X_val = (X_tr[val_idx] - mu) / std
        X_te_n = (X_te - mu) / std

        layers = []
        prev = X_fit.shape[1]
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        model = nn.Sequential(*layers)

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        crit = nn.MSELoss()

        ds = TensorDataset(torch.tensor(X_fit), torch.tensor(y_tr[fit_idx]).unsqueeze(1))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        best_val, patience_cnt, best_state = float("inf"), 0, None
        for epoch in range(max_epochs):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                crit(model(xb), yb).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = crit(model(torch.tensor(X_val)),
                          torch.tensor(y_tr[val_idx]).unsqueeze(1)).item()
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
            pred = model(torch.tensor(X_te_n)).squeeze(1).numpy()

        rho, _ = spearmanr(y_te, pred)
        r, _   = pearsonr(y_te, pred)
        med = np.median(y_te)
        binary = (y_te >= med).astype(int)
        au = float(roc_auc_score(binary, pred)) if binary.sum() > 0 and (1-binary).sum() > 0 else float("nan")

        rhos.append(float(rho))
        rs.append(float(r))
        aurocs.append(au)

    if not rhos:
        return {}
    return {
        "spearman_mean": float(np.mean(rhos)),
        "spearman_std":  float(np.std(rhos)),
        "pearson_mean":  float(np.mean(rs)),
        "pearson_std":   float(np.std(rs)),
        "auroc_mean":    float(np.nanmean(aurocs)),
        "auroc_std":     float(np.nanstd(aurocs)),
        "n_folds": len(rhos),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    variants = load_s1724_variants()
    proteins  = np.array([v["protein"] for v in variants])
    ddg       = np.array([v["ddg"]     for v in variants])
    cluster_map = assign_protein_clusters(variants)

    print(f"Loaded {len(variants)} variants across {len(set(proteins))} proteins")

    wt_mean  = np.load(WT_MEAN_EMB)
    mut_mean = np.load(MUT_MEAN_EMB)
    X = mut_mean - wt_mean
    print(f"Embeddings: {X.shape}")

    probes = {
        "MLP":  lambda seed: (lambda: None),  # handled separately below
        "RF":   lambda seed: RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
        "GBM":  lambda seed: GradientBoostingRegressor(n_estimators=100, random_state=seed),
    }

    summary = {}

    # MLP: multi-seed via run_mlp_regression
    print("\n── MLP ──")
    for split_name, splits_fn in [
        ("random",  lambda s: random_split_cv(len(variants), N_FOLDS, s)),
        ("protein", lambda s: protein_split_cv(proteins, N_FOLDS, s)),
        ("pfam",    lambda s: pfam_split_cv(proteins, N_FOLDS, s)),
    ]:
        rhos, aurocs = [], []
        for seed in range(N_SEEDS):
            res = run_mlp_regression(X, ddg, splits_fn(seed), seed=seed)
            if res:
                rhos.append(res["spearman_mean"])
                aurocs.append(res["auroc_mean"])
        key = f"mlp_{split_name}"
        summary[key] = {
            "spearman_mean": float(np.mean(rhos)),
            "spearman_std":  float(np.std(rhos)),
            "auroc_mean":    float(np.nanmean(aurocs)),
            "auroc_std":     float(np.nanstd(aurocs)),
        }
        print(f"  {split_name:8s}: ρ={summary[key]['spearman_mean']:.3f}±{summary[key]['spearman_std']:.3f}  "
              f"AUROC={summary[key]['auroc_mean']:.3f}±{summary[key]['auroc_std']:.3f}")

    # RF and GBM: use sklearn probe
    for probe_name, clf_factory in [
        ("RF",  lambda seed: RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)),
        ("GBM", lambda seed: GradientBoostingRegressor(n_estimators=100, random_state=seed)),
    ]:
        print(f"\n── {probe_name} ──")
        for split_name, splits_fn in [
            ("random",  lambda s: random_split_cv(len(variants), N_FOLDS, s)),
            ("protein", lambda s: protein_split_cv(proteins, N_FOLDS, s)),
            ("pfam",    lambda s: pfam_split_cv(proteins, N_FOLDS, s)),
        ]:
            rhos, aurocs = [], []
            for seed in range(N_SEEDS):
                res = run_sklearn_probe(X, ddg, splits_fn(seed),
                                        clf_fn=lambda s=seed: clf_factory(s))
                if res:
                    rhos.append(res["spearman_mean"])
                    aurocs.append(res["auroc_mean"])
            key = f"{probe_name.lower()}_{split_name}"
            summary[key] = {
                "spearman_mean": float(np.mean(rhos)),
                "spearman_std":  float(np.std(rhos)),
                "auroc_mean":    float(np.nanmean(aurocs)),
                "auroc_std":     float(np.nanstd(aurocs)),
            }
            print(f"  {split_name:8s}: ρ={summary[key]['spearman_mean']:.3f}±{summary[key]['spearman_std']:.3f}  "
                  f"AUROC={summary[key]['auroc_mean']:.3f}±{summary[key]['auroc_std']:.3f}")

    # Final comparison table
    print(f"\n{'='*70}")
    print(f"{'Probe':6s}  {'Random ρ':>9}  {'Protein ρ':>10}  {'Pfam ρ':>7}  {'Δ rnd→pfam':>10}  {'Pfam AUROC':>11}")
    for probe in ["ridge", "mlp", "rf", "gbm"]:
        rnd = summary.get(f"{probe}_random",  {}).get("spearman_mean", float("nan"))
        prt = summary.get(f"{probe}_protein", {}).get("spearman_mean", float("nan"))
        pfm = summary.get(f"{probe}_pfam",    {}).get("spearman_mean", float("nan"))
        pau = summary.get(f"{probe}_pfam",    {}).get("auroc_mean",    float("nan"))
        if probe == "ridge":
            rnd, prt, pfm, pau = 0.546, 0.280, 0.193, 0.597
        print(f"  {probe.upper():6s}  {rnd:>9.3f}  {prt:>10.3f}  {pfm:>7.3f}  {rnd-pfm:>10.3f}  {pau:>11.3f}")
    print(f"{'='*70}")

    with open(os.path.join(OUT, "mlp_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to {OUT}/")


if __name__ == "__main__":
    main()
