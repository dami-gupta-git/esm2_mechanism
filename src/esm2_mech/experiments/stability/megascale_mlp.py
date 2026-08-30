"""Nonlinear stability probe (MLP, RF) companion to megascale_stability.py.

RF is EXPLORATORY / post-hoc — only Ridge and MLP are primary.
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
from esm2_mech.experiments.stability.megascale_stability import (
    OUT,
    run_regression_cv,
    spearman_cluster_bootstrap_ci,
)
from esm2_mech.utils.constants import BOOTSTRAP_N_RESAMPLES, N_SEEDS
from esm2_mech.utils.io import write_result_json
from esm2_mech.utils.metrics import auroc_at_median, mean_std_n, standardize
from esm2_mech.utils.seed_aggregation import (
    SEED_STATUS_UNSCORABLE,
    aggregate_result_contract,
    aggregate_seed_results,
    read_seed_point_estimate,
    seed_result_contract,
)

os.makedirs(OUT, exist_ok=True)



def run_mlp_regression(
    X,
    y,
    splits,
    validation_groups,
    median,
    seed=42,
    hidden=(256, 64),
    lr=1e-3,
    max_epochs=60,
    patience=15,
    batch_size=2048,
    clusters=None,
    return_oof=False,
):
    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if validation_groups is not None:
        validation_groups = np.asarray(validation_groups)
        if len(validation_groups) != len(X):
            raise ValueError(
                f"validation_groups has {len(validation_groups)} rows for "
                f"{len(X)} samples"
            )
    if return_oof and clusters is None:
        raise ValueError("clusters are required when return_oof=True")

    rhos, rs, aurocs = [], [], []
    oof_y, oof_pred, oof_clusters, oof_indices, oof_folds = [], [], [], [], []

    for fold_i, (tr, te) in enumerate(splits):
        X_tr = X[tr].astype(np.float32)
        X_te = X[te].astype(np.float32)
        y_tr = y[tr].astype(np.float32)
        y_te = y[te].astype(np.float32)

        # Hold out 15% of rows for the random split, or 15% of whole dependency
        # groups for domain/family splits.
        rng = np.random.RandomState(seed + fold_i)
        if validation_groups is None:
            shuffled_rows = np.arange(len(X_tr))
            rng.shuffle(shuffled_rows)
            n_validation = max(1, int(0.15 * len(shuffled_rows)))
            val_idx = shuffled_rows[:n_validation]
            fit_idx = shuffled_rows[n_validation:]
        else:
            training_groups = validation_groups[tr]
            testing_groups = validation_groups[te]
            if any(group is None for group in training_groups) or any(
                group is None for group in testing_groups
            ):
                raise ValueError(
                    "group-disjoint early stopping requires a group for every "
                    "outer-CV row"
                )
            outer_overlap = set(training_groups.tolist()) & set(
                testing_groups.tolist()
            )
            if outer_overlap:
                examples = sorted(outer_overlap, key=str)[:5]
                raise ValueError(
                    "validation group spans the outer CV train/test boundary; "
                    f"examples: {examples}"
                )
            unique_groups = np.array(
                sorted(set(training_groups.tolist()), key=str), dtype=object
            )
            if len(unique_groups) < 2:
                raise ValueError(
                    "group-disjoint early stopping requires at least two "
                    "training groups"
                )
            rng.shuffle(unique_groups)
            n_validation_groups = max(1, int(0.15 * len(unique_groups)))
            n_validation_groups = min(
                n_validation_groups, len(unique_groups) - 1
            )
            validation_group_set = set(unique_groups[:n_validation_groups])
            validation_mask = np.array(
                [group in validation_group_set for group in training_groups]
            )
            val_idx = np.where(validation_mask)[0]
            fit_idx = np.where(~validation_mask)[0]

        if len(fit_idx) == 0 or len(val_idx) == 0:
            raise ValueError("early-stopping split produced an empty fit or validation set")

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
        if return_oof:
            oof_y.append(y_te)
            oof_pred.append(pred)
            oof_clusters.append(np.asarray(clusters)[te])
            oof_indices.append(np.asarray(te))
            oof_folds.append(np.full(len(te), fold_i, dtype=int))

    # Each metric is judged on its own folds: a fold whose AUROC is undefined
    # says nothing about the rank correlation measured on the same fold.
    def _metric_status(fold_values):
        return (
            "success"
            if fold_values and all(np.isfinite(value) for value in fold_values)
            else SEED_STATUS_UNSCORABLE
        )

    result = {
        "status": "success" if rhos else SEED_STATUS_UNSCORABLE,
        "spearman_status": _metric_status(rhos),
        "pearson_status": _metric_status(rs),
        "auroc_status": _metric_status(aurocs),
        "n_folds": len(rhos),
        "sampling_unit": "cv_fold",
    }
    if result["spearman_status"] == "success":
        rho_mean, rho_std, _ = mean_std_n(rhos)
        result["spearman_mean"] = rho_mean
        result["spearman_fold_std"] = rho_std
    if result["pearson_status"] == "success":
        r_mean, r_std, _ = mean_std_n(rs)
        result["pearson_mean"] = r_mean
        result["pearson_fold_std"] = r_std
    if result["auroc_status"] == "success":
        au_mean, au_std, _ = mean_std_n(aurocs)
        result["auroc_mean"] = au_mean
        result["auroc_fold_std"] = au_std
    if not return_oof:
        return result
    oof = None
    if oof_y:
        oof = {
            "y_true": np.concatenate(oof_y),
            "pred": np.concatenate(oof_pred),
            "clusters": np.concatenate(oof_clusters),
            "indices": np.concatenate(oof_indices),
            "folds": np.concatenate(oof_folds),
        }
    return result, oof


def _show(summary):
    metric = read_seed_point_estimate(summary)
    if not metric.available:
        return f"unavailable ({metric.message})"
    spread = metric.spread
    return (
        f"{metric.value:.3f}"
        if spread is None
        else f"{metric.value:.3f}±{spread:.3f} seed SD"
    )



def main(
    use_xgboost=False,
    n_seeds=N_SEEDS,
    compute_ci=True,
    n_boot=BOOTSTRAP_N_RESAMPLES,
):
    requested_seeds = tuple(range(n_seeds))
    inputs = load_stability_inputs()
    variants = inputs.variants
    proteins = inputs.proteins
    ddg = inputs.ddg
    family_map = inputs.family_map
    X = inputs.delta_mean
    input_fingerprints = inputs.input_fingerprints
    print(f"Embeddings: {X.shape}")
    print(f"RF backend: {'cuML (GPU)' if _HAS_CUML else 'sklearn (CPU)'}")

    global_median = float(np.median(ddg))

    split_names = ("random", "domain", "family")
    build_splits = lambda name, seed: stability_splits(
        seed, len(variants), proteins, family_map
    )[name]
    family_groups = np.array(
        [family_map.get(protein) for protein in proteins],
        dtype=object,
    )

    def _validation_groups(split_name):
        if split_name == "random":
            return None
        if split_name == "domain":
            return proteins
        if split_name == "family":
            return family_groups
        raise ValueError(f"unknown stability split {split_name!r}")

    def _ci_clusters(split_name):
        return family_groups if split_name == "family" else proteins

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
            (
                "xgb",
                lambda X, y, splits, seed: run_regression_cv(
                    X,
                    y,
                    splits,
                    lambda: _xgb(seed),
                    with_pearson=False,
                    median=global_median,
                    label=f"xgb/seed{seed}",
                ),
            ),
        ]
    else:
        probe_runners = [
            ("mlp", None),
            (
                "rf",
                lambda X, y, splits, seed: run_regression_cv(
                    X,
                    y,
                    splits,
                    lambda: _rf(seed),
                    with_pearson=False,
                    median=global_median,
                    label=f"rf/seed{seed}",
                ),
            ),
        ]

    summary = {}
    for probe_name, run_probe in probe_runners:
        print(f"\n── {probe_name.upper()} ──")
        for split_name in split_names:
            per_seed = []
            seed0_ci = None
            for seed in requested_seeds:
                splits = build_splits(split_name, seed)
                if probe_name == "mlp":
                    collect_oof = compute_ci and seed == 0
                    mlp_result = run_mlp_regression(
                        X,
                        ddg,
                        splits,
                        seed=seed,
                        median=global_median,
                        validation_groups=_validation_groups(split_name),
                        clusters=_ci_clusters(split_name) if collect_oof else None,
                        return_oof=collect_oof,
                    )
                    if collect_oof:
                        res, oof = mlp_result
                        if oof is not None:
                            seed0_ci = spearman_cluster_bootstrap_ci(
                                oof, n_resamples=n_boot, seed=0
                            )
                    else:
                        res = mlp_result
                else:
                    res = run_probe(X, ddg, splits, seed)
                status = res["status"]
                per_seed.append(
                    {
                        **seed_result_contract(seed, status=status),
                        "result": res,
                    }
                )
            key = f"{probe_name}_{split_name}"
            spearman = aggregate_seed_results(
                requested_seeds,
                per_seed,
                lambda seed_result: seed_result["result"].get("spearman_mean"),
                status=lambda seed_result: seed_result["result"]["spearman_status"],
            ).to_dict()
            auroc = aggregate_seed_results(
                requested_seeds,
                per_seed,
                lambda seed_result: seed_result["result"].get("auroc_mean"),
                status=lambda seed_result: seed_result["result"]["auroc_status"],
            ).to_dict()
            summary[key] = {
                "across_seed": {
                    "spearman": spearman,
                    "auroc": auroc,
                },
                "per_seed_fold_summaries": per_seed,
            }
            if seed0_ci is not None:
                # A within-seed interval on seed 0's estimate. It is a separate
                # field from the across-seed aggregate and is not a seed spread.
                summary[key]["seed0_inference"] = {
                    "seed": 0,
                    "point_estimate": seed0_ci.get("point"),
                    "ci": seed0_ci,
                }
            print(
                f"  {split_name:8s}: ρ={_show(spearman)}  AUROC={_show(auroc)}"
            )

    # Final comparison table
    print(f"\n{'='*72}")
    print(
        f"{'Probe':6s}  {'Random ρ':>9}  {'Domain ρ':>9}  {'Family ρ':>9}  "
        f"{'Δ rnd→fam':>10}  {'Family AUROC':>13}"
    )
    for probe_name, _ in probe_runners:
        probe = probe_name
        rnd = read_seed_point_estimate(
            summary[f"{probe}_random"]["across_seed"]["spearman"]
        ).value
        dom = read_seed_point_estimate(
            summary[f"{probe}_domain"]["across_seed"]["spearman"]
        ).value
        fam = read_seed_point_estimate(
            summary[f"{probe}_family"]["across_seed"]["spearman"]
        ).value
        fau = read_seed_point_estimate(
            summary[f"{probe}_family"]["across_seed"]["auroc"]
        ).value
        if any(value is None for value in (rnd, dom, fam, fau)):
            print(f"  {probe.upper():6s}  unavailable")
            continue
        print(
            f"  {probe.upper():6s}  {rnd:>9.3f}  {dom:>9.3f}  {fam:>9.3f}  "
            f"{rnd-fam:>10.3f}  {fau:>13.3f}"
        )
    print(f"{'='*72}")

    # Separate output file for the xgboost variant so it never overwrites the
    # default sklearn comparison (mlp_summary.json).
    out_name = "mlp_summary_xgb.json" if use_xgboost else "mlp_summary.json"
    summary.update(aggregate_result_contract())
    summary["result_version"] = 4
    summary["input_fingerprints"] = input_fingerprints
    summary["analysis_parameters"] = {
        "n_seeds": n_seeds,
        "compute_ci": compute_ci,
        "n_boot": n_boot,
        "probe_mode": "xgboost" if use_xgboost else "mlp_and_random_forest",
    }
    write_result_json(os.path.join(OUT, out_name), summary, seeds=list(requested_seeds))
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
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument(
        "--no_ci",
        action="store_true",
        help="Skip seed-0 dependency-aware CIs for the primary MLP.",
    )
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    args = parser.parse_args()
    main(
        use_xgboost=args.xgboost,
        n_seeds=args.seeds,
        compute_ci=not args.no_ci,
        n_boot=args.n_boot,
    )
