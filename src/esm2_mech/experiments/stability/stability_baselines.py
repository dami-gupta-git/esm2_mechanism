"""
Controls and interpretation for the Megascale stability probe (NOT pre-registered).

These do not feed the H1–H4 verdict; they answer "is the linear signal real and
earned by ESM-2, and how is it structured?" Kept separate from the pre-registered
probe (megascale_stability.py) so the report can present them as controls /
exploratory interpretation, not confirmed hypotheses.

Four analyses, all under the same three CV schemes (random / domain / family) and
N_SEEDS seeds as the main probe:

  1. delta-norm baseline — a 1-feature Ridge on ||delta_mean|| only. If this
     recovers most of the full-embedding ρ, "ESM-2 encodes stability" weakens to
     "ESM-2's representation moves more for drastic mutations".
  2. nested-CV alpha — RidgeCV picks alpha on an inner split of each train fold
     (no leakage), removing the arbitrary alpha=1.0 of the main probe; reports the
     chosen alphas so the linear ceiling isn't an artifact of one value.
  3. label-shuffle null — ddG permuted; ρ must collapse to ~0 under every split.
     A non-zero shuffled ρ would indicate a pipeline/CV-leakage artifact.
  4. PLS component sweep — ρ vs n_components (random + family split). Characterises
     the intrinsic dimensionality of the stability direction (exploratory).

Usage (embeddings must already be extracted):
  python -m esm2_mech.experiments.stability.stability_baselines

Output: results/<run>/megascale_stability/baselines.json
"""

import functools
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from esm2_mech.experiments.stability.tsuboyama_loader import load_tsuboyama_variants
from esm2_mech.experiments.stability.build_domain_families import build_family_map
from esm2_mech.experiments.stability.megascale_stability import (
    run_ridge_with_auroc,
    N_SEEDS,
    N_FOLDS,
    OUT,
    WT_MEAN_EMB,
    MUT_MEAN_EMB,
)
from esm2_mech.utils.io import atomic_write_json
from esm2_mech.utils.metrics import mean_std_n
from esm2_mech.utils.paths import MEGASCALE_DOMAIN_FAMILIES_JSON
from esm2_mech.utils.splits import random_split_cv, gene_split_cv, family_split_cv

print = functools.partial(print, flush=True)

# Inner-CV alpha grid for the nested Ridge.
ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
# PLS component counts to sweep (exploratory dimensionality).
PLS_COMPONENTS = (1, 2, 5, 10, 20, 50)


def _splits_for(seed, n, proteins, family_map):
    """The three CV schemes for one seed, as a name→splits dict."""
    return {
        "random": random_split_cv(n, N_FOLDS, seed),
        "domain": gene_split_cv(proteins, n_folds=N_FOLDS, seed=seed),
        "family": family_split_cv(proteins, family_map, n_folds=N_FOLDS, seed=seed),
    }


def _aggregate_over_seeds(per_seed):
    """per_seed: list of {split: rho}. Returns {split: {mean,std,n_seeds}}."""
    out = {}
    for split in ("random", "domain", "family"):
        vals = [d[split] for d in per_seed if split in d]
        mean, std, n = mean_std_n(vals)
        out[split] = {"spearman_mean": mean, "spearman_std": std, "n_seeds": n}
    return out


def delta_norm_baseline(delta_mean, ddg, proteins, family_map):
    """1-feature Ridge on ||delta_mean|| under all three splits."""
    norms = np.linalg.norm(delta_mean, axis=1).reshape(-1, 1)
    per_seed = []
    for seed in range(N_SEEDS):
        splits = _splits_for(seed, len(ddg), proteins, family_map)
        per_seed.append(
            {name: run_ridge_with_auroc(norms, ddg, sp).get("spearman_mean", float("nan"))
             for name, sp in splits.items()}
        )
    return _aggregate_over_seeds(per_seed)


def nested_alpha_ridge(delta_mean, ddg, proteins, family_map):
    """RidgeCV (inner CV chooses alpha per train fold) under all three splits.

    No leakage: alpha is selected by RidgeCV's internal CV on the training fold
    only. Records the chosen alphas so the report can show the linear ceiling does
    not hinge on a single arbitrary alpha.
    """
    per_seed, chosen_alphas = [], []
    for seed in range(N_SEEDS):
        splits = _splits_for(seed, len(ddg), proteins, family_map)
        seed_rho = {}
        for name, sp in splits.items():
            rhos = []
            for tr, te in sp:
                scaler = StandardScaler()
                x_tr = scaler.fit_transform(delta_mean[tr])
                x_te = scaler.transform(delta_mean[te])
                clf = RidgeCV(alphas=ALPHA_GRID)
                clf.fit(x_tr, ddg[tr])
                chosen_alphas.append(float(clf.alpha_))
                rho, _ = spearmanr(ddg[te], clf.predict(x_te))
                rhos.append(float(rho))
            mean, _, _ = mean_std_n(rhos)
            seed_rho[name] = mean
        per_seed.append(seed_rho)
    agg = _aggregate_over_seeds(per_seed)
    agg["alpha_grid"] = list(ALPHA_GRID)
    agg["chosen_alpha_median"] = float(np.median(chosen_alphas)) if chosen_alphas else None
    return agg


def label_shuffle_null(delta_mean, ddg, proteins, family_map):
    """Ridge on permuted ddG; ρ must collapse to ~0 under every split."""
    per_seed = []
    for seed in range(N_SEEDS):
        rng = np.random.RandomState(seed)
        ddg_shuf = ddg[rng.permutation(len(ddg))]
        splits = _splits_for(seed, len(ddg), proteins, family_map)
        per_seed.append(
            {name: run_ridge_with_auroc(delta_mean, ddg_shuf, sp).get("spearman_mean", float("nan"))
             for name, sp in splits.items()}
        )
    return _aggregate_over_seeds(per_seed)


def pls_component_sweep(delta_mean, ddg, proteins, family_map):
    """ρ vs n_components for PLS, on random and family split (exploratory).

    Single seed (seed 0) per component count — this characterises dimensionality,
    not a headline number, so a multi-seed mean is unnecessary.
    """
    splits = _splits_for(0, len(ddg), proteins, family_map)
    out = {}
    for split_name in ("random", "family"):
        sp = splits[split_name]
        by_k = {}
        for k in PLS_COMPONENTS:
            rhos = []
            for tr, te in sp:
                scaler = StandardScaler()
                x_tr = scaler.fit_transform(delta_mean[tr])
                x_te = scaler.transform(delta_mean[te])
                pls = PLSRegression(n_components=k)
                pls.fit(x_tr, ddg[tr])
                pred = pls.predict(x_te).ravel()
                rho, _ = spearmanr(ddg[te], pred)
                rhos.append(float(rho))
            mean, _, _ = mean_std_n(rhos)
            by_k[str(k)] = mean
        out[split_name] = by_k
    return out


def main():
    variants = load_tsuboyama_variants()
    proteins = np.array([v["protein"] for v in variants])
    ddg = np.array([v["ddg"] for v in variants])
    print(f"Loaded {len(variants)} variants across {len(set(proteins))} domains")

    if os.path.exists(MEGASCALE_DOMAIN_FAMILIES_JSON):
        with open(MEGASCALE_DOMAIN_FAMILIES_JSON) as handle:
            family_map = json.load(handle)
    else:
        family_map = build_family_map(variants=variants)

    wt_mean = np.load(WT_MEAN_EMB)
    mut_mean = np.load(MUT_MEAN_EMB)
    if len(wt_mean) != len(variants):
        raise ValueError(
            f"embedding/variant row mismatch: {len(wt_mean)} rows vs {len(variants)} "
            f"variants — {WT_MEAN_EMB} is not row-aligned."
        )
    delta_mean = mut_mean - wt_mean

    results = {}

    print("\n[1/4] delta-norm baseline (||delta_mean||, 1 feature)")
    results["delta_norm"] = delta_norm_baseline(delta_mean, ddg, proteins, family_map)
    for split, d in results["delta_norm"].items():
        print(f"  {split:7s}: ρ={d['spearman_mean']:.3f}")

    print("\n[2/4] nested-CV alpha Ridge")
    results["nested_alpha"] = nested_alpha_ridge(delta_mean, ddg, proteins, family_map)
    for split in ("random", "domain", "family"):
        print(f"  {split:7s}: ρ={results['nested_alpha'][split]['spearman_mean']:.3f}")
    print(f"  median chosen alpha: {results['nested_alpha']['chosen_alpha_median']}")

    print("\n[3/4] label-shuffle null (ρ should be ~0)")
    results["label_shuffle"] = label_shuffle_null(delta_mean, ddg, proteins, family_map)
    for split, d in results["label_shuffle"].items():
        print(f"  {split:7s}: ρ={d['spearman_mean']:.3f}")

    print("\n[4/4] PLS component sweep (exploratory)")
    results["pls_sweep"] = pls_component_sweep(delta_mean, ddg, proteins, family_map)
    for split, by_k in results["pls_sweep"].items():
        pretty = "  ".join(f"k={k}:{rho:.3f}" for k, rho in by_k.items())
        print(f"  {split:7s}: {pretty}")

    out_path = os.path.join(OUT, "baselines.json")
    atomic_write_json(out_path, results)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
