"""Linear (Ridge) stability probe on Tsuboyama 2023 point-mutant ΔΔG.

Pre-registered H1-H4 hypotheses; see docs/plans/plan_megascale_stability.md.
Companion nonlinear probe: megascale_mlp.py.
"""

import argparse
import functools
import json
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr

print = functools.partial(print, flush=True)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from esm2_mech.experiments.mechanism.loaders import load_merged
from esm2_mech.experiments.stability.stability_data import (
    load_stability_inputs,
    stability_splits,
)
from esm2_mech.utils.bootstrap import (
    cluster_bootstrap_ci,
    paired_cluster_bootstrap_diff,
    paired_cluster_bootstrap_diff_cross_partition,
)
from esm2_mech.utils.constants import BOOTSTRAP_N_RESAMPLES, N_SEEDS, N_FOLDS
from esm2_mech.utils.io import atomic_write_json
from esm2_mech.utils.metrics import auroc_at_median, mean_std_n, standardize
from esm2_mech.utils.splits import family_split_cv
from esm2_mech.utils.paths import (
    DATA_DIR as _DATA_DIR,
    RESULTS_DIR as _RESULTS_DIR,
    VALID_VARIANTS_JSON,
    EMB_WT_MEAN,
    EMB_MUT_MEAN,
    PFAM_JSON,
    ESM2_MODEL,
)

OUT = str(_RESULTS_DIR / "megascale_stability")

os.makedirs(OUT, exist_ok=True)
os.makedirs(str(_DATA_DIR / "embeddings" / ESM2_MODEL), exist_ok=True)



def run_regression_cv(X, y, splits, clf_fn, with_pearson=True, clusters=None,
                      return_oof=False, median=None):
    """Standardise-fit-predict a regressor over CV folds; return ρ/AUROC."""
    rhos, pearsons, aurocs = [], [], []
    oof_y, oof_pred, oof_clusters, oof_indices = [], [], [], []
    for tr, te in splits:
        Xtr, Xte = standardize(X[tr], X[te])
        clf = clf_fn()
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        rho, _ = spearmanr(y[te], pred)
        rhos.append(float(rho))
        aurocs.append(auroc_at_median(y[te], pred, median=median))
        if with_pearson:
            pearson, _ = pearsonr(y[te], pred)
            pearsons.append(float(pearson))
        if return_oof and clusters is not None:
            oof_y.append(y[te])
            oof_pred.append(pred)
            oof_clusters.append(clusters[te])
            oof_indices.append(te)
    if not rhos:
        out = {}
    else:
        rho_mean, rho_std, n_rho = mean_std_n(rhos)
        au_mean, au_std, _ = mean_std_n(aurocs)
        out = {
            "spearman_mean": rho_mean,
            "spearman_std": rho_std,
            "auroc_mean": au_mean,
            "auroc_std": au_std,
            "n_folds": n_rho,
        }
        if with_pearson:
            pearson_mean, pearson_std, _ = mean_std_n(pearsons)
            out["pearson_mean"] = pearson_mean
            out["pearson_std"] = pearson_std
    if not return_oof:
        return out
    oof = None
    if oof_y:
        all_y = np.concatenate(oof_y)
        all_pred = np.concatenate(oof_pred)
        oof = {
            "y_true": all_y,
            "pred": all_pred,
            "clusters": np.concatenate(oof_clusters),
            "indices": np.concatenate(oof_indices),
        }
        pooled_rho, _ = spearmanr(all_y, all_pred)
        if np.isfinite(pooled_rho):
            out["spearman_pooled"] = float(pooled_rho)
    return out, oof


def run_ridge_with_auroc(X, y, splits, clusters=None, return_oof=False, median=None):
    """Ridge alpha=1.0 stability probe."""
    return run_regression_cv(
        X, y, splits, lambda: Ridge(alpha=1.0), with_pearson=True,
        clusters=clusters, return_oof=return_oof, median=median,
    )


def spearman_cluster_bootstrap_ci(oof, n_resamples=BOOTSTRAP_N_RESAMPLES, seed=0):
    """Cluster-bootstrap CI on Spearman ρ from OOF predictions."""
    y_true = oof["y_true"]
    pred = oof["pred"]

    def _rho(rows):
        if len(set(rows.tolist())) < 2:
            return None
        rho, _ = spearmanr(y_true[rows], pred[rows])
        return float(rho) if np.isfinite(rho) else None

    return cluster_bootstrap_ci(oof["clusters"], _rho, n_resamples=n_resamples, seed=seed)



def per_protein_spearman(X, y, proteins):
    """Leave-one-protein-out Ridge ρ for each protein with ≥5 variants."""
    unique = sorted(set(proteins))
    results = {}
    for prot in unique:
        mask = proteins == prot
        if mask.sum() < 5:
            continue
        tr = np.where(~mask)[0]
        te = np.where(mask)[0]
        if len(tr) < 10:
            continue
        Xtr, Xte = standardize(X[tr], X[te])
        clf = Ridge(alpha=1.0)
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        rho, pval = spearmanr(y[te], pred)
        results[prot] = {
            "spearman": float(rho),
            "p_value": float(pval),
            "n_variants": int(mask.sum()),
        }
    return results



def run_h3_stability_projection(
    merged_delta_mean,
    merged_labels,
    merged_proteins,
    pfam_map,
    stability_variants,
    stability_delta_mean,
    stability_ddg,
    n_folds=5,
    n_seeds=5,
    n_boot=BOOTSTRAP_N_RESAMPLES,
):
    """Project stability out of mechanism delta_mean; compare family-split F1."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import f1_score

    # Fit stability Ridge on the stability (Tsuboyama) set
    sc_s = StandardScaler()
    X_s = sc_s.fit_transform(stability_delta_mean)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_s, stability_ddg)

    # Projection vector is in sc_s feature space (unit-normalised Ridge weights).
    stability_weights = ridge.coef_
    stability_dir = stability_weights / (np.linalg.norm(stability_weights) + 1e-12)

    # Both arms stay in sc_s space; a per-fold re-standardisation would reintroduce
    # variance along the removed direction, defeating the test.
    merged_scaled = sc_s.transform(merged_delta_mean.astype(np.float64)).astype(
        np.float32
    )
    proj = merged_scaled @ stability_dir  # (N,) scalar stability score per variant
    residuals = merged_scaled - np.outer(proj, stability_dir)

    # H3 hinges on this removal; assert var along stability_dir ≈ 0.
    var_before = float(np.var(merged_scaled.astype(np.float64) @ stability_dir))
    var_after = float(np.var(residuals.astype(np.float64) @ stability_dir))
    if var_after > 1e-6 * var_before + 1e-8:
        raise AssertionError(
            f"stability projection failed: var along stability_dir was {var_before:.3e} "
            f"before and {var_after:.3e} after projecting out — the removed direction "
            "leaked back in, so the projected arm still contains stability signal."
        )

    le = LabelEncoder()
    y = le.fit_transform(merged_labels)

    baseline_f1s, projected_f1s = [], []
    seed0_oof = {}
    for seed in range(n_seeds):
        splits = family_split_cv(merged_proteins, pfam_map, n_folds=n_folds, seed=seed)
        for X, tag in [(merged_scaled, "baseline"), (residuals, "projected")]:
            fold_f1s = []
            oof_y, oof_pred, oof_genes = [], [], []
            for tr, te in splits:
                # No per-fold StandardScaler: X is already sc_s-standardised, and
                # re-standardising would undo the projection (see note above). Both
                # arms get identical handling so the only difference is the projection.
                clf = LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    class_weight="balanced",
                    random_state=seed,
                )
                clf.fit(X[tr], y[tr])
                pred = clf.predict(X[te])
                fold_f1s.append(
                    float(f1_score(y[te], pred, average="macro", zero_division=0))
                )
                if seed == 0:
                    oof_y.append(y[te])
                    oof_pred.append(pred)
                    oof_genes.append(merged_proteins[te])
            seed_f1_mean, _, _ = mean_std_n(fold_f1s)
            if tag == "baseline":
                baseline_f1s.append(seed_f1_mean)
            else:
                projected_f1s.append(seed_f1_mean)
            if seed == 0 and oof_y:
                seed0_oof[tag] = {
                    "y_true": np.concatenate(oof_y),
                    "pred": np.concatenate(oof_pred),
                    "genes": np.concatenate(oof_genes),
                }

    baseline_f1_mean, baseline_f1_std, _ = mean_std_n(baseline_f1s)
    projected_f1_mean, projected_f1_std, _ = mean_std_n(projected_f1s)
    difference_ci = None
    if "baseline" in seed0_oof and "projected" in seed0_oof:
        baseline_oof = seed0_oof["baseline"]
        projected_oof = seed0_oof["projected"]
        if not np.array_equal(baseline_oof["genes"], projected_oof["genes"]):
            raise AssertionError("H3 baseline and projected OOF rows are not aligned")
        clusters = np.array(
            [pfam_map[gene] for gene in baseline_oof["genes"]], dtype=object
        )

        def _projected_f1(rows):
            return float(
                f1_score(
                    projected_oof["y_true"][rows],
                    projected_oof["pred"][rows],
                    average="macro",
                    zero_division=0,
                )
            )

        def _baseline_f1(rows):
            return float(
                f1_score(
                    baseline_oof["y_true"][rows],
                    baseline_oof["pred"][rows],
                    average="macro",
                    zero_division=0,
                )
            )

        difference_ci = paired_cluster_bootstrap_diff(
            clusters,
            _projected_f1,
            _baseline_f1,
            n_resamples=n_boot,
            seed=0,
        )
    if difference_ci is None or difference_ci.get("ci_high") is None:
        h3_verdict = "not adjudicated (paired family-bootstrap CI unavailable)"
    elif difference_ci["ci_high"] <= 0.01:
        h3_verdict = "pass — established"
    elif difference_ci["ci_low"] > 0.01:
        h3_verdict = "fail — established"
    else:
        h3_verdict = "underpowered — CI overlaps +0.01 threshold"
    return {
        "baseline_f1_mean": baseline_f1_mean,
        "baseline_f1_std": baseline_f1_std,
        "projected_f1_mean": projected_f1_mean,
        "projected_f1_std": projected_f1_std,
        "delta_f1": projected_f1_mean - baseline_f1_mean,
        "difference_ci": difference_ci,
        "h3_passes": h3_verdict == "pass — established",
        "h3_verdict": h3_verdict,
    }



def apply_decision_rule(random_rho, protein_rho, per_prot_std, h2_gap_ci=None):
    """Apply pre-registered decision rule, ordered by informativeness."""
    delta = random_rho - protein_rho
    # Check in order of informativeness
    if random_rho >= 0.5 and delta >= 0.10:
        if h2_gap_ci is None or h2_gap_ci.get("ci_low") is None:
            return "NOT ADJUDICATED (H2 paired CI unavailable)"
        if h2_gap_ci["ci_low"] >= 0.10:
            return "LEAKY"
        return "UNDERPOWERED (H2 point estimate is LEAKY; CI overlaps 0.10)"
    if random_rho >= 0.5 and delta <= 0.05 and per_prot_std >= 0.15:
        return "HETEROGENEOUS"
    if random_rho >= 0.5 and delta <= 0.05 and per_prot_std <= 0.10:
        return "ROBUST"
    if 0.3 <= random_rho < 0.5:
        return "WEAK"
    if random_rho < 0.3:
        return "NULL"
    return f"INTERMEDIATE (rho={random_rho:.3f}, delta={delta:.3f}, std={per_prot_std:.3f})"



def main(compute_ci=True, n_boot=BOOTSTRAP_N_RESAMPLES):
    inputs = load_stability_inputs(include_pos=True)
    variants = inputs.variants
    proteins = inputs.proteins
    ddg = inputs.ddg
    family_map = inputs.family_map
    delta_mean = inputs.delta_mean
    delta_pos = inputs.delta_pos
    n_families = inputs.n_families

    print(f"Embeddings: delta_mean {delta_mean.shape}, delta_pos {delta_pos.shape}")

    global_median = float(np.median(ddg))
    print(f"Global ΔΔG median: {global_median:.4f}")

    results_by_seed = []
    seed0_oofs = {}
    for seed in range(N_SEEDS):
        print(f"\n── Seed {seed} ──")

        splits_by_name = stability_splits(seed, len(variants), proteins, family_map)

        seed_result = {"seed": seed}

        for feat_name, X in [("delta_mean", delta_mean), ("delta_pos", delta_pos)]:
            for split_name, splits in splits_by_name.items():
                key = f"{feat_name}_{split_name}"
                if compute_ci and seed == 0:
                    ci_clusters = (
                        np.array(
                            [family_map.get(p, f"__orphan__{p}") for p in proteins]
                        )
                        if split_name == "family"
                        else proteins
                    )
                    res, oof = run_ridge_with_auroc(
                        X, ddg, splits, clusters=ci_clusters, return_oof=True,
                        median=global_median,
                    )
                    if oof is not None:
                        res["ci"] = spearman_cluster_bootstrap_ci(
                            oof, n_resamples=n_boot, seed=seed
                        )
                        seed0_oofs[key] = oof
                else:
                    res = run_ridge_with_auroc(
                        X, ddg, splits, median=global_median,
                    )
                seed_result[key] = res
                if res:
                    print(
                        f"  {key}: ρ={res['spearman_mean']:.3f}±{res['spearman_std']:.3f}  "
                        f"AUROC={res['auroc_mean']:.3f}"
                    )

        results_by_seed.append(seed_result)

    print("\nPer-protein Spearman (leave-one-protein-out)...")
    per_prot = per_protein_spearman(delta_mean, ddg, proteins)
    prot_rhos = [entry["spearman"] for entry in per_prot.values()]
    # NaN-guard: constant predictions yield NaN which must not poison per_prot_std.
    per_prot_mean, per_prot_std, n_finite_prot = mean_std_n(prot_rhos)
    finite_rhos = [rho for rho in prot_rhos if np.isfinite(rho)]
    if finite_rhos:
        print(
            f"  Per-protein ρ: mean={per_prot_mean:.3f}  std={per_prot_std:.3f}  "
            f"min={min(finite_rhos):.3f}  max={max(finite_rhos):.3f}  "
            f"n={n_finite_prot} (of {len(prot_rhos)} proteins with ≥5 variants)"
        )
    else:
        print("  Per-protein ρ: no protein yielded a finite ρ — skipped")

    atomic_write_json(os.path.join(OUT, "per_protein_spearman.json"), per_prot)

    summary = {}
    all_keys = set()
    for seed_result in results_by_seed:
        for key_name, value in seed_result.items():
            if isinstance(value, dict):
                all_keys.add(key_name)

    for key in sorted(all_keys):
        vals_rho = [
            sr[key]["spearman_mean"] for sr in results_by_seed if key in sr and sr[key]
        ]
        vals_auroc = [
            sr[key]["auroc_mean"] for sr in results_by_seed if key in sr and sr[key]
        ]
        if not vals_rho:
            continue
        rho_mean, rho_std, n_seeds_used = mean_std_n(vals_rho)
        au_mean, au_std, _ = mean_std_n(vals_auroc)
        summary[key] = {
            "spearman_mean": rho_mean,
            "spearman_std": rho_std,
            "auroc_mean": au_mean,
            "auroc_std": au_std,
            "n_seeds": n_seeds_used,
        }
        # Seed 0's cluster-bootstrap CI carries through to the summary.
        seed0_ci = results_by_seed[0].get(key, {}).get("ci") if results_by_seed else None
        if seed0_ci is not None:
            summary[key]["ci"] = seed0_ci

    summary["per_protein"] = {
        "spearman_mean": per_prot_mean,
        "spearman_std": per_prot_std,
        "n_proteins": len(prot_rhos),
        "n_proteins_finite": n_finite_prot,
    }

    h3_result = None
    if all(
        os.path.exists(path)
        for path in [VALID_VARIANTS_JSON, EMB_WT_MEAN, EMB_MUT_MEAN, PFAM_JSON]
    ):
        print("\nRunning H3 stability projection test...")
        with open(PFAM_JSON) as handle:
            pfam_map = json.load(handle)

        merged_delta, merged_labels, merged_proteins = load_merged()

        h3_result = run_h3_stability_projection(
            merged_delta,
            merged_labels,
            merged_proteins,
            pfam_map,
            variants,
            delta_mean,
            ddg,
            n_folds=N_FOLDS,
            n_seeds=N_SEEDS,
            n_boot=n_boot,
        )
        print(
            f"  H3: baseline F1={h3_result['baseline_f1_mean']:.3f}  "
            f"projected F1={h3_result['projected_f1_mean']:.3f}  "
            f"Δ={h3_result['delta_f1']:+.3f}  "
            f"passes={'YES' if h3_result['h3_passes'] else 'NO (stability direction is informative)'}"
        )
        atomic_write_json(
            os.path.join(OUT, "h3_stability_projection.json"), h3_result
        )
    else:
        print("\nSkipping H3 (merged embeddings not found — run on pod with full data)")

    h2_gap_ci = None
    oof_random = seed0_oofs.get("delta_mean_random")
    oof_family = seed0_oofs.get("delta_mean_family")
    if compute_ci and oof_random is not None and oof_family is not None:
        print("\nComputing paired CI on random-to-family Spearman gap (H2)...")
        idx_r = oof_random["indices"]
        idx_f = oof_family["indices"]
        shared = np.intersect1d(idx_r, idx_f)
        pos_r = {int(idx): i for i, idx in enumerate(idx_r)}
        pos_f = {int(idx): i for i, idx in enumerate(idx_f)}
        sel_r = np.array([pos_r[int(s)] for s in shared])
        sel_f = np.array([pos_f[int(s)] for s in shared])

        shared_y = oof_random["y_true"][sel_r]
        shared_pred_random = oof_random["pred"][sel_r]
        shared_pred_family = oof_family["pred"][sel_f]
        shared_families = np.array(
            [family_map.get(proteins[int(s)], f"__orphan__{proteins[int(s)]}")
             for s in shared]
        )

        def _rho_random(rows):
            rho, _ = spearmanr(shared_y[rows], shared_pred_random[rows])
            return float(rho) if np.isfinite(rho) else None

        def _rho_family(rows):
            rho, _ = spearmanr(shared_y[rows], shared_pred_family[rows])
            return float(rho) if np.isfinite(rho) else None

        h2_gap_ci = paired_cluster_bootstrap_diff_cross_partition(
            resample_clusters=shared_families,
            metric_fn_a=_rho_random,
            metric_fn_b=_rho_family,
            n_resamples=n_boot,
            seed=0,
        )
        print(
            f"  H2 gap: {h2_gap_ci['point_diff']:.3f} "
            f"[{h2_gap_ci.get('ci_low', '?')}, {h2_gap_ci.get('ci_high', '?')}]"
        )

    dm_random = summary.get("delta_mean_random", {}).get("spearman_mean", float("nan"))
    dm_family = summary.get("delta_mean_family", {}).get("spearman_mean", float("nan"))

    verdict = apply_decision_rule(dm_random, dm_family, per_prot_std, h2_gap_ci)
    summary["verdict"] = verdict
    if h2_gap_ci is not None:
        summary["h2_gap_ci"] = h2_gap_ci
    summary["n_variants"] = len(variants)
    summary["n_proteins"] = len(set(proteins))
    summary["n_families"] = n_families
    summary["n_seeds"] = N_SEEDS
    summary["h3"] = h3_result

    print(f"\n{'='*60}")
    print(
        f"VERDICT: {verdict}  (ordered by informativeness: LEAKY > HETEROGENEOUS > ROBUST > WEAK > NULL)"
    )
    print(f"  delta_mean random ρ  : {dm_random:.3f}  (H1 threshold ≥ 0.5)")
    print(f"  delta_mean family ρ  : {dm_family:.3f}")
    print(f"  Δ (random − family)  : {dm_random - dm_family:.3f}  (LEAKY if Δ ≥ 0.10)")
    print(f"  per-domain ρ std     : {per_prot_std:.3f}  (HETEROGENEOUS if ≥ 0.15)")
    if h3_result:
        print(
            f"  H3 Δ mechanism F1    : {h3_result['delta_f1']:+.3f}  "
            f"(passes if ≤ +0.01 — stability projection doesn't help mechanism)"
        )
    print(f"{'='*60}")

    atomic_write_json(os.path.join(OUT, "summary.json"), summary)
    print(f"\nResults written to {OUT}/")


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_ci", action="store_true", help="skip cluster-bootstrap CIs")
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    args = parser.parse_args()
    main(compute_ci=not args.no_ci, n_boot=args.n_boot)


if __name__ == "__main__":
    _cli()
