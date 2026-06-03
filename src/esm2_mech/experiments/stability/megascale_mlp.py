"""
Nonlinear (MLP) stability probe on S1724 — companion to megascale_stability.py.

Runs MLP regression (1280→256→64→1) under the same three CV schemes
(random / protein-holdout / cluster-holdout) and 5 seeds as the Ridge probe,
then compares to check whether nonlinearity adds signal beyond Ridge.

Same question as result_3/5/7 for mechanism: does the MLP lift survive
family-holdout, or does it evaporate (leakage)?

Usage:
  cd esm2_mechanism
  python -m esm2_mech.experiments.stability.megascale_mlp

Outputs:
  results/megascale_stability/mlp_summary.json
"""

import functools
import json
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr

print = functools.partial(print, flush=True)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from esm2_mech.experiments.stability.megascale_stability import load_s1724_variants
from esm2_mech.utils.metrics import auroc_at_median, mean_std_n
from esm2_mech.utils.splits import random_split_cv, gene_split_cv, family_split_cv

PFAM = {
    "1AJ3": "PF13499",
    "1BNI": "PF00545",
    "1CEY": "PF00072",
    "1CUN": "PF00545",
    "1DIV": "PF04563",
    "1EKG": "PF02234",
    "1FKJ": "PF00254",
    "1FT8": "PF00062",
    "1FTG": "PF00062",
    "1GUA": "PF00244",
    "1H7M": "PF00084",
    "1IOB": "PF00545",
    "1LVE": "PF00089",
    "1O6X": "PF02885",
    "1RIS": "PF00042",
    "1RX4": "PF00042",
    "1SHF": "PF00130",
    "1STN": "PF00565",
    "1TEN": "PF07679",
    "1TTG": "PF09289",
    "1UBQ": "NO_PFAM",
    "2CI2": "PF00280",
    "2IFB": "PF14651",
    "2PTL": "PF00020",
    "3BDC": "PF00565",
    "3HHR": "PF00103",
    "4HXJ": "PF00870",
}

from esm2_mech.utils.paths import (
    DATA_DIR as _DATA_DIR,
    RESULTS_DIR as _RESULTS_DIR,
    MEGASCALE_EMB_WT_MEAN,
    MEGASCALE_EMB_MUT_MEAN,
)

DATA = str(_DATA_DIR)
OUT = str(_RESULTS_DIR / "megascale_stability")

WT_MEAN_EMB = MEGASCALE_EMB_WT_MEAN
MUT_MEAN_EMB = MEGASCALE_EMB_MUT_MEAN

N_SEEDS = 5
N_FOLDS = 5

os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# Pfam family-split CV — defers to the shared family_split_cv helper.
# PFAM (above) is the {PDB_id: Pfam} map; every S1724 protein has an entry
# (ubiquitin = "NO_PFAM", a truthy singleton family), so no protein is dropped.
# ---------------------------------------------------------------------------


def pfam_split_cv(proteins, n_folds=5, seed=42):
    return family_split_cv(proteins, PFAM, n_folds=n_folds, seed=seed)


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
        rhos.append(float(rho))
        aurocs.append(auroc_at_median(y[te], pred))
    if not rhos:
        return {}
    rho_mean, rho_std, n_rho = mean_std_n(rhos)
    au_mean, au_std, _ = mean_std_n(aurocs)
    return {
        "spearman_mean": rho_mean,
        "spearman_std": rho_std,
        "auroc_mean": au_mean,
        "auroc_std": au_std,
        "n_folds": n_rho,
    }


# ---------------------------------------------------------------------------
# MLP regression probe
# ---------------------------------------------------------------------------


def run_mlp_regression(
    X,
    y,
    splits,
    seed=42,
    hidden=(256, 64),
    lr=1e-3,
    max_epochs=200,
    patience=15,
    batch_size=64,
):
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

        mu = X_tr[fit_idx].mean(0)
        std = X_tr[fit_idx].std(0) + 1e-8
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

        ds = TensorDataset(
            torch.tensor(X_fit), torch.tensor(y_tr[fit_idx]).unsqueeze(1)
        )
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
                vl = crit(
                    model(torch.tensor(X_val)), torch.tensor(y_tr[val_idx]).unsqueeze(1)
                ).item()
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
        r, _ = pearsonr(y_te, pred)
        rhos.append(float(rho))
        rs.append(float(r))
        aurocs.append(auroc_at_median(y_te, pred))

    if not rhos:
        return {}
    rho_mean, rho_std, n_rho = mean_std_n(rhos)
    r_mean, r_std, _ = mean_std_n(rs)
    au_mean, au_std, _ = mean_std_n(aurocs)
    return {
        "spearman_mean": rho_mean,
        "spearman_std": rho_std,
        "pearson_mean": r_mean,
        "pearson_std": r_std,
        "auroc_mean": au_mean,
        "auroc_std": au_std,
        "n_folds": n_rho,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    variants = load_s1724_variants()
    proteins = np.array([v["protein"] for v in variants])
    ddg = np.array([v["ddg"] for v in variants])

    print(f"Loaded {len(variants)} variants across {len(set(proteins))} proteins")

    wt_mean = np.load(WT_MEAN_EMB)
    mut_mean = np.load(MUT_MEAN_EMB)
    X = mut_mean - wt_mean
    print(f"Embeddings: {X.shape}")

    # The three CV schemes, built per-seed. Shared by every probe.
    split_builders = [
        ("random", lambda seed: random_split_cv(len(variants), N_FOLDS, seed)),
        ("protein", lambda seed: gene_split_cv(proteins, n_folds=N_FOLDS, seed=seed)),
        ("pfam", lambda seed: pfam_split_cv(proteins, N_FOLDS, seed)),
    ]

    # Each probe maps (X, y, splits, seed) -> per-fold-aggregated dict. The MLP
    # uses its own torch runner; Ridge/RF/GBM use run_sklearn_probe with a fresh
    # estimator per seed. Ridge is the linear baseline — running it here (rather
    # than reading it from the stability summary) keeps its Pfam-split numbers
    # computed under the exact same folds as the nonlinear probes, so the
    # comparison table traces to this run with no hardcoded values.
    def _ridge(seed):
        return Ridge(alpha=1.0)

    def _rf(seed):
        return RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)

    def _gbm(seed):
        return GradientBoostingRegressor(n_estimators=100, random_state=seed)

    probe_runners = [
        ("ridge", lambda X, y, splits, seed: run_sklearn_probe(X, y, splits, lambda: _ridge(seed))),
        ("mlp", lambda X, y, splits, seed: run_mlp_regression(X, y, splits, seed=seed)),
        ("rf", lambda X, y, splits, seed: run_sklearn_probe(X, y, splits, lambda: _rf(seed))),
        ("gbm", lambda X, y, splits, seed: run_sklearn_probe(X, y, splits, lambda: _gbm(seed))),
    ]

    summary = {}
    for probe_name, run_probe in probe_runners:
        print(f"\n── {probe_name.upper()} ──")
        for split_name, build_splits in split_builders:
            rhos, aurocs = [], []
            for seed in range(N_SEEDS):
                res = run_probe(X, ddg, build_splits(seed), seed)
                if res:
                    rhos.append(res["spearman_mean"])
                    aurocs.append(res["auroc_mean"])
            key = f"{probe_name}_{split_name}"
            rho_mean, rho_std, n_seeds_used = mean_std_n(rhos)
            au_mean, au_std, _ = mean_std_n(aurocs)
            if n_seeds_used == 0:
                print(f"  {split_name:8s}: no valid folds across {N_SEEDS} seeds — skipped")
                continue
            summary[key] = {
                "spearman_mean": rho_mean,
                "spearman_std": rho_std,
                "auroc_mean": au_mean,
                "auroc_std": au_std,
                "n_seeds": n_seeds_used,
            }
            print(
                f"  {split_name:8s}: ρ={summary[key]['spearman_mean']:.3f}±{summary[key]['spearman_std']:.3f}  "
                f"AUROC={summary[key]['auroc_mean']:.3f}±{summary[key]['auroc_std']:.3f}"
            )

    # Final comparison table
    print(f"\n{'='*70}")
    print(
        f"{'Probe':6s}  {'Random ρ':>9}  {'Protein ρ':>10}  {'Pfam ρ':>7}  {'Δ rnd→pfam':>10}  {'Pfam AUROC':>11}"
    )
    for probe in ["ridge", "mlp", "rf", "gbm"]:
        rnd = summary.get(f"{probe}_random", {}).get("spearman_mean", float("nan"))
        prt = summary.get(f"{probe}_protein", {}).get("spearman_mean", float("nan"))
        pfm = summary.get(f"{probe}_pfam", {}).get("spearman_mean", float("nan"))
        pau = summary.get(f"{probe}_pfam", {}).get("auroc_mean", float("nan"))
        print(
            f"  {probe.upper():6s}  {rnd:>9.3f}  {prt:>10.3f}  {pfm:>7.3f}  {rnd-pfm:>10.3f}  {pau:>11.3f}"
        )
    print(f"{'='*70}")

    with open(os.path.join(OUT, "mlp_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to {OUT}/")


if __name__ == "__main__":
    main()
