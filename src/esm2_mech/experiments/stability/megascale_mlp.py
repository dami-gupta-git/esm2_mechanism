"""Nonlinear stability probe (MLP, RF) companion to megascale_stability.py.

RF is EXPLORATORY / post-hoc — only Ridge and MLP are pre-registered.
Uses cuML GPU random forest when available, falls back to sklearn on CPU.
"""

import functools
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr

print = functools.partial(print, flush=True)

import torch

_HAS_CUML = False
try:
    from cuml.ensemble import RandomForestRegressor as CumlRFRegressor
    _HAS_CUML = torch.cuda.is_available()
except ImportError:
    pass

if not _HAS_CUML:
    from sklearn.ensemble import RandomForestRegressor

from esm2_mech.experiments.stability.stability_data import (
    load_stability_inputs,
    stability_splits,
)
from esm2_mech.experiments.stability.megascale_stability import run_regression_cv, OUT
from esm2_mech.utils.constants import N_SEEDS
from esm2_mech.utils.io import write_result_json
from esm2_mech.utils.metrics import auroc_at_median, mean_std_n, standardize

os.makedirs(OUT, exist_ok=True)



def run_mlp_regression(
    X,
    y,
    splits,
    seed=42,
    hidden=(256, 64),
    lr=1e-3,
    max_epochs=60,
    patience=15,
    batch_size=2048,
    median=None,
):
    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        X_fit, X_val, X_te_n = standardize(X_tr[fit_idx], X_tr[val_idx], X_te)

        # Seed torch before the model is built: weight init, dropout masks and
        # the per-epoch permutation below all draw from the global RNG.
        torch.manual_seed(seed + fold_i)

        layers = []
        prev = X_fit.shape[1]
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        model = nn.Sequential(*layers).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        crit = nn.MSELoss()

        # Move all tensors to device once; per-batch .to(device) via DataLoader
        # was the bottleneck (GPU idle ~1%, host-copy bound).
        X_fit_t = torch.tensor(X_fit).to(device)
        y_fit_t = torch.tensor(y_tr[fit_idx]).unsqueeze(1).to(device)
        X_val_t = torch.tensor(X_val).to(device)
        y_val_t = torch.tensor(y_tr[val_idx]).unsqueeze(1).to(device)
        X_te_t = torch.tensor(X_te_n).to(device)
        n_fit = X_fit_t.shape[0]

        best_val, patience_cnt, best_state = float("inf"), 0, None
        for epoch in range(max_epochs):
            model.train()
            perm = torch.randperm(n_fit, device=device)
            for start in range(0, n_fit, batch_size):
                bidx = perm[start:start + batch_size]
                opt.zero_grad()
                crit(model(X_fit_t[bidx]), y_fit_t[bidx]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = crit(model(X_val_t), y_val_t).item()
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
            pred = model(X_te_t).squeeze(1).cpu().numpy()

        rho, _ = spearmanr(y_te, pred)
        r, _ = pearsonr(y_te, pred)
        rhos.append(float(rho))
        rs.append(float(r))
        aurocs.append(auroc_at_median(y_te, pred, median=median))

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



def main(use_xgboost=False):
    inputs = load_stability_inputs()
    variants = inputs.variants
    proteins = inputs.proteins
    ddg = inputs.ddg
    family_map = inputs.family_map
    X = inputs.delta_mean
    print(f"Embeddings: {X.shape}")
    print(f"RF backend: {'cuML (GPU)' if _HAS_CUML else 'sklearn (CPU)'}")

    global_median = float(np.median(ddg))

    split_names = ("random", "domain", "family")
    build_splits = lambda name, seed: stability_splits(
        seed, len(variants), proteins, family_map
    )[name]

    def _rf(seed):
        if _HAS_CUML:
            return CumlRFRegressor(n_estimators=100, random_state=seed)
        return RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)

    if use_xgboost:
        # xgboost mode runs ONLY the GPU tree booster — the MLP is already produced
        # by the default run (mlp_summary.json) so recomputing it here is wasted.
        # GPU-trained gradient-boosted trees are a fast alternative to the sklearn
        # RF/GBM (CPU-only, slow on 177k×1280). Reported as probe name 'xgb'; NOT a
        # drop-in for sklearn GBM (different library/defaults) — a fast complement.
        def _xgb(seed):
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                device="cuda",
                random_state=seed,
                n_jobs=-1,
            )

        probe_runners = [
            ("xgb", lambda X, y, splits, seed: run_regression_cv(X, y, splits, lambda: _xgb(seed), with_pearson=False, median=global_median, label=f"xgb/seed{seed}")),
        ]
    else:
        probe_runners = [
            ("mlp", lambda X, y, splits, seed: run_mlp_regression(X, y, splits, seed=seed, median=global_median)),
            ("rf", lambda X, y, splits, seed: run_regression_cv(X, y, splits, lambda: _rf(seed), with_pearson=False, median=global_median, label=f"rf/seed{seed}")),
        ]

    summary = {}
    for probe_name, run_probe in probe_runners:
        print(f"\n── {probe_name.upper()} ──")
        for split_name in split_names:
            rhos, aurocs = [], []
            for seed in range(N_SEEDS):
                res = run_probe(X, ddg, build_splits(split_name, seed), seed)
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
    print(f"\n{'='*72}")
    print(
        f"{'Probe':6s}  {'Random ρ':>9}  {'Domain ρ':>9}  {'Family ρ':>9}  "
        f"{'Δ rnd→fam':>10}  {'Family AUROC':>13}"
    )
    for probe_name, _ in probe_runners:
        probe = probe_name
        rnd = summary.get(f"{probe}_random", {}).get("spearman_mean", float("nan"))
        dom = summary.get(f"{probe}_domain", {}).get("spearman_mean", float("nan"))
        fam = summary.get(f"{probe}_family", {}).get("spearman_mean", float("nan"))
        fau = summary.get(f"{probe}_family", {}).get("auroc_mean", float("nan"))
        print(
            f"  {probe.upper():6s}  {rnd:>9.3f}  {dom:>9.3f}  {fam:>9.3f}  "
            f"{rnd-fam:>10.3f}  {fau:>13.3f}"
        )
    print(f"{'='*72}")

    # Separate output file for the xgboost variant so it never overwrites the
    # default sklearn comparison (mlp_summary.json).
    out_name = "mlp_summary_xgb.json" if use_xgboost else "mlp_summary.json"
    write_result_json(os.path.join(OUT, out_name), summary, seeds=list(range(N_SEEDS)))
    print(f"\nResults written to {os.path.join(OUT, out_name)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xgboost",
        action="store_true",
        help="Use GPU XGBoost (probe 'xgb') instead of sklearn RF/GBM. Faster on "
        "large high-dim data; writes mlp_summary_xgb.json. Requires xgboost installed.",
    )
    args = parser.parse_args()
    main(use_xgboost=args.xgboost)
