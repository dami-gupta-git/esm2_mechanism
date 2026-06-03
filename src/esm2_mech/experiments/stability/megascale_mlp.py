"""
Nonlinear stability probe — companion to megascale_stability.py.

Dataset: full Tsuboyama 2023 point-mutant set, natural domains only (see
tsuboyama_loader.py). Runs Ridge, MLP (1280→256→64→1), RF and GBM under the
same three CV schemes (random / domain-holdout / family-holdout) and 5 seeds,
then compares to check whether nonlinearity adds signal beyond the linear probe.
Family-holdout uses real Pfam families (build_domain_families.py).

NOTE on pre-registration: only Ridge and MLP are pre-registered (see the plan in
docs/plans/plan_megascale_stability.md). RF and GBM are EXPLORATORY / post-hoc —
report them as such, never as confirmed hypotheses.

Same question as result_3/5/7 for mechanism: does the nonlinear lift survive
family-holdout, or does it evaporate (leakage)?

Usage (embeddings must already be extracted on GPU):
  cd esm2_mechanism
  python -m esm2_mech.experiments.stability.megascale_mlp

Outputs:
  results/<run>/megascale_stability/mlp_summary.json
"""

import functools
import json
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr

print = functools.partial(print, flush=True)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from esm2_mech.experiments.stability.tsuboyama_loader import load_tsuboyama_variants
from esm2_mech.experiments.stability.build_domain_families import build_family_map
from esm2_mech.utils.metrics import auroc_at_median, mean_std_n
from esm2_mech.utils.splits import random_split_cv, gene_split_cv, family_split_cv
from esm2_mech.utils.paths import (
    RESULTS_DIR as _RESULTS_DIR,
    MEGASCALE_EMB_WT_MEAN,
    MEGASCALE_EMB_MUT_MEAN,
    MEGASCALE_DOMAIN_FAMILIES_JSON,
)

OUT = str(_RESULTS_DIR / "megascale_stability")

WT_MEAN_EMB = MEGASCALE_EMB_WT_MEAN
MUT_MEAN_EMB = MEGASCALE_EMB_MUT_MEAN

N_SEEDS = 5
N_FOLDS = 5

os.makedirs(OUT, exist_ok=True)


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
    max_epochs=60,
    patience=15,
    batch_size=2048,
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
        model = nn.Sequential(*layers).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        crit = nn.MSELoss()

        # Move ALL tensors to the device once. The data fits in GPU memory, so we
        # index minibatches on-device with a per-epoch permutation instead of a
        # DataLoader — a DataLoader doing torch.tensor(...).to(device) per batch
        # was the bottleneck (GPU idle ~1%, host-copy bound) on this tiny model.
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


def main(use_xgboost=False):
    variants = load_tsuboyama_variants()
    proteins = np.array([v["protein"] for v in variants])
    ddg = np.array([v["ddg"] for v in variants])
    print(f"Loaded {len(variants)} variants across {len(set(proteins))} natural domains")

    # Domain → Pfam family map (cached); orphans absent → excluded from family-split.
    if os.path.exists(MEGASCALE_DOMAIN_FAMILIES_JSON):
        with open(MEGASCALE_DOMAIN_FAMILIES_JSON) as handle:
            family_map = json.load(handle)
    else:
        family_map = build_family_map(variants=variants)

    wt_mean = np.load(WT_MEAN_EMB)
    mut_mean = np.load(MUT_MEAN_EMB)
    if len(wt_mean) != len(variants):
        raise ValueError(
            f"embedding/variant row mismatch: {len(wt_mean)} rows vs "
            f"{len(variants)} variants — {WT_MEAN_EMB} is not row-aligned."
        )
    X = mut_mean - wt_mean
    print(f"Embeddings: {X.shape}")

    # The three CV schemes, built per-seed. Shared by every probe.
    split_builders = [
        ("random", lambda seed: random_split_cv(len(variants), N_FOLDS, seed)),
        ("domain", lambda seed: gene_split_cv(proteins, n_folds=N_FOLDS, seed=seed)),
        ("family", lambda seed: family_split_cv(proteins, family_map, n_folds=N_FOLDS, seed=seed)),
    ]

    # Each probe maps (X, y, splits, seed) -> per-fold-aggregated dict. The MLP
    # uses its own torch runner; RF/GBM use run_sklearn_probe with a fresh
    # estimator per seed. The Ridge linear baseline is NOT recomputed here — it is
    # produced (identically) by megascale_stability.py; running it again was the
    # slow stage (ill-conditioned at alpha=1.0, ~75 fits) and added nothing. Read
    # the Ridge row from results/<run>/megascale_stability/summary.json when
    # building the comparison table in the report.
    def _rf(seed):
        return RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)

    def _gbm(seed):
        return GradientBoostingRegressor(n_estimators=100, random_state=seed)

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
            ("xgb", lambda X, y, splits, seed: run_sklearn_probe(X, y, splits, lambda: _xgb(seed))),
        ]
    else:
        probe_runners = [
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
    with open(os.path.join(OUT, out_name), "w") as f:
        json.dump(summary, f, indent=2)
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
