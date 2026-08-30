"""Controls and interpretation baselines for the Megascale stability probe.

All exploratory: delta-norm, nested-CV alpha, label-shuffle null, PLS sweep.
"""

import functools
import os

import numpy as np
from joblib import Parallel, delayed, parallel_config
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.cross_decomposition import PLSRegression

from esm2_mech.experiments.stability.stability_data import (
    load_stability_inputs,
    stability_splits,
)
from esm2_mech.experiments.stability.megascale_stability import (
    run_ridge_with_auroc,
    OUT,
)
from esm2_mech.utils.constants import N_FOLDS, N_SEEDS
from esm2_mech.utils.io import write_result_json
from esm2_mech.utils.metrics import mean_std_n, standardize
from esm2_mech.utils.seed_aggregation import (
    aggregate_result_contract,
    aggregate_seed_values,
    make_seed_record,
    read_seed_point_estimate,
)

print = functools.partial(print, flush=True)

# Inner-CV alpha grid for the nested Ridge.
ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
# PLS component counts to sweep (exploratory dimensionality).
PLS_COMPONENTS = (1, 2, 5, 10, 20, 50)


def _aggregate_over_seeds(requested_seeds, per_seed):
    """Aggregate one complete fold summary from each requested model seed."""
    out = {}
    for split in ("random", "domain", "family"):
        out[split] = aggregate_seed_values(
            requested_seeds,
            [make_seed_record(result["seed"], result.get(split)) for result in per_seed],
        ).to_dict()
    return out


def _delta_norm_one_seed(seed, norms, ddg, proteins, family_map):
    splits = stability_splits(seed, len(ddg), proteins, family_map)
    return {
        "seed": seed,
        **{
            name: run_ridge_with_auroc(norms, ddg, sp).get("spearman_mean")
            for name, sp in splits.items()
        },
    }


def delta_norm_baseline(delta_mean, ddg, proteins, family_map, n_jobs, requested_seeds):
    """1-feature Ridge on ||delta_mean||."""
    norms = np.linalg.norm(delta_mean, axis=1).reshape(-1, 1)
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        per_seed = Parallel()(
            delayed(_delta_norm_one_seed)(seed, norms, ddg, proteins, family_map)
            for seed in requested_seeds
        )
    return _aggregate_over_seeds(requested_seeds, per_seed)


def _nested_alpha_one_seed(seed, delta_mean, ddg, proteins, family_map):
    splits = stability_splits(seed, len(ddg), proteins, family_map)
    seed_rho = {}
    seed_alphas = []
    for name, sp in splits.items():
        rhos = []
        for tr, te in sp:
            x_tr, x_te = standardize(delta_mean[tr], delta_mean[te])
            clf = RidgeCV(alphas=ALPHA_GRID)
            clf.fit(x_tr, ddg[tr])
            seed_alphas.append(float(clf.alpha_))
            rho, _ = spearmanr(ddg[te], clf.predict(x_te))
            rhos.append(float(rho))
        seed_rho[name] = (
            float(np.mean(rhos))
            if len(rhos) == N_FOLDS and np.isfinite(rhos).all()
            else None
        )
    return {"seed": seed, **seed_rho}, seed_alphas


def nested_alpha_ridge(delta_mean, ddg, proteins, family_map, n_jobs, requested_seeds):
    """RidgeCV with inner-CV alpha selection; no leakage into test fold."""
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        results = Parallel()(
            delayed(_nested_alpha_one_seed)(seed, delta_mean, ddg, proteins, family_map)
            for seed in requested_seeds
        )
    per_seed = [r[0] for r in results]
    chosen_alphas = [a for r in results for a in r[1]]
    agg = _aggregate_over_seeds(requested_seeds, per_seed)
    agg["alpha_grid"] = list(ALPHA_GRID)
    agg["chosen_alpha_median"] = float(np.median(chosen_alphas)) if chosen_alphas else None
    return agg


def _label_shuffle_one_seed(seed, delta_mean, ddg, proteins, family_map):
    rng = np.random.RandomState(seed)
    ddg_shuf = ddg[rng.permutation(len(ddg))]
    splits = stability_splits(seed, len(ddg), proteins, family_map)
    return {
        "seed": seed,
        **{
            name: run_ridge_with_auroc(delta_mean, ddg_shuf, sp).get("spearman_mean")
            for name, sp in splits.items()
        },
    }


def label_shuffle_null(delta_mean, ddg, proteins, family_map, n_jobs, requested_seeds):
    """Ridge on permuted ddG; ρ should collapse to ~0."""
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        per_seed = Parallel()(
            delayed(_label_shuffle_one_seed)(seed, delta_mean, ddg, proteins, family_map)
            for seed in requested_seeds
        )
    return _aggregate_over_seeds(requested_seeds, per_seed)


def _pls_one_config(split_name, n_components, sp, delta_mean, ddg):
    rhos = []
    for tr, te in sp:
        x_tr, x_te = standardize(delta_mean[tr], delta_mean[te])
        pls = PLSRegression(n_components=n_components)
        pls.fit(x_tr, ddg[tr])
        pred = pls.predict(x_te).ravel()
        rho, _ = spearmanr(ddg[te], pred)
        rhos.append(float(rho))
    mean = (
        float(np.mean(rhos))
        if len(rhos) == N_FOLDS and np.isfinite(rhos).all()
        else None
    )
    return split_name, n_components, mean


def pls_component_sweep(delta_mean, ddg, proteins, family_map, n_jobs):
    """PLS ρ vs n_components on random and family split (seed 0 only)."""
    splits = stability_splits(0, len(ddg), proteins, family_map)
    configs = [
        (split_name, nc, splits[split_name])
        for split_name in ("random", "family")
        for nc in PLS_COMPONENTS
    ]
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        results = Parallel()(
            delayed(_pls_one_config)(sn, nc, sp, delta_mean, ddg)
            for sn, nc, sp in configs
        )
    out = {}
    for split_name, n_components, mean in results:
        out.setdefault(split_name, {})[str(n_components)] = mean
    return out


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


def main(n_jobs, n_seeds=N_SEEDS):
    requested_seeds = tuple(range(n_seeds))
    inputs = load_stability_inputs()
    proteins = inputs.proteins
    ddg = inputs.ddg
    family_map = inputs.family_map
    delta_mean = inputs.delta_mean

    results = {**aggregate_result_contract()}
    results["input_fingerprints"] = inputs.input_fingerprints
    results["analysis_parameters"] = {
        "n_seeds": n_seeds,
        "alpha_grid": list(ALPHA_GRID),
        "pls_components": list(PLS_COMPONENTS),
    }

    print("\n[1/4] delta-norm baseline (||delta_mean||, 1 feature)")
    results["delta_norm"] = delta_norm_baseline(
        delta_mean, ddg, proteins, family_map, n_jobs, requested_seeds
    )
    for split, stats in results["delta_norm"].items():
        print(f"  {split:7s}: ρ={_show(stats)}")

    print("\n[2/4] nested-CV alpha Ridge")
    results["nested_alpha"] = nested_alpha_ridge(
        delta_mean, ddg, proteins, family_map, n_jobs, requested_seeds
    )
    for split in ("random", "domain", "family"):
        print(f"  {split:7s}: ρ={_show(results['nested_alpha'][split])}")
    print(f"  median chosen alpha: {results['nested_alpha']['chosen_alpha_median']}")

    print("\n[3/4] label-shuffle null (ρ should be ~0)")
    results["label_shuffle"] = label_shuffle_null(
        delta_mean, ddg, proteins, family_map, n_jobs, requested_seeds
    )
    for split, stats in results["label_shuffle"].items():
        print(f"  {split:7s}: ρ={_show(stats)}")

    print("\n[4/4] PLS component sweep (exploratory)")
    results["pls_sweep"] = pls_component_sweep(delta_mean, ddg, proteins, family_map, n_jobs)
    for split, by_n_components in results["pls_sweep"].items():
        pretty = "  ".join(
            f"k={n_components}:" + ("unavailable" if rho is None else f"{rho:.3f}")
            for n_components, rho in by_n_components.items()
        )
        print(f"  {split:7s}: {pretty}")

    out_path = os.path.join(OUT, "baselines.json")
    write_result_json(out_path, results, seeds=list(requested_seeds))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n_jobs", type=int, required=True,
        help="Max concurrent worker processes for the delta-norm, nested-alpha, "
        "label-shuffle, and PLS sweep parallel loops. Set explicitly (never -1) "
        "to bound peak RAM. Start low (e.g. 4), watch peak RAM, raise only if it fits.",
    )
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()
    main(n_jobs=args.n_jobs, n_seeds=args.seeds)
