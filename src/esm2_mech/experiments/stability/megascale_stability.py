"""Linear (Ridge) stability probe on Tsuboyama 2023 point-mutant ΔΔG.

Pre-registered stability controls 3A-3D; see biorxiv/PREREGISTRATION_run_biorxiv.md.
Companion nonlinear probe: megascale_mlp.py.
"""

import argparse
import functools
import json
import os
import numpy as np
from joblib import Parallel, delayed, parallel_config
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
    folds_to_arms,
    paired_cluster_bootstrap_diff,
    paired_cluster_bootstrap_diff_cross_partition,
    score_within_folds,
)
from esm2_mech.utils.constants import (
    BOOTSTRAP_N_RESAMPLES,
    MECHANISM_CLASSES,
    N_SEEDS,
    N_FOLDS,
)
from esm2_mech.utils.data import (
    embedding_fingerprint,
    labeled_variant_fingerprint,
    load_pfam_map,
    pfam_fingerprint,
)
from esm2_mech.utils.io import write_result_json
from esm2_mech.utils.metrics import auroc_at_median, fold_macro_f1, mean_std_n, standardize
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
                      return_oof=False, median=None, label=None):
    """Standardise-fit-predict a regressor over CV folds; return ρ/AUROC."""
    rhos, pearsons, aurocs = [], [], []
    oof_y, oof_pred, oof_clusters, oof_indices, oof_folds = [], [], [], [], []
    splits = list(splits)
    n_folds = len(splits)
    for fold_i, (tr, te) in enumerate(splits):
        Xtr, Xte = standardize(X[tr], X[te])
        clf = clf_fn()
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        rho, _ = spearmanr(y[te], pred)
        rhos.append(float(rho))
        aurocs.append(auroc_at_median(y[te], pred, median=median))
        if label:
            print(f"    {label} fold {fold_i+1}/{n_folds}: ρ={float(rho):.3f}")
        if with_pearson:
            pearson, _ = pearsonr(y[te], pred)
            pearsons.append(float(pearson))
        if return_oof and clusters is not None:
            oof_y.append(y[te])
            oof_pred.append(pred)
            oof_clusters.append(clusters[te])
            oof_indices.append(te)
            oof_folds.append(np.full(len(te), fold_i, dtype=int))
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
        # No pooled rank correlation here. Each fold's regressor has its own
        # intercept and scale, so ranking one concatenated list of predictions
        # compares values that were never on a common scale. The reported figure is
        # the fold mean, and the interval below is computed the same way.
        oof = {
            "y_true": np.concatenate(oof_y),
            "pred": np.concatenate(oof_pred),
            "clusters": np.concatenate(oof_clusters),
            "indices": np.concatenate(oof_indices),
            "folds": np.concatenate(oof_folds),
        }
    return out, oof


def run_ridge_with_auroc(X, y, splits, clusters=None, return_oof=False, median=None):
    """Ridge alpha=1.0 stability probe."""
    return run_regression_cv(
        X, y, splits, lambda: Ridge(alpha=1.0), with_pearson=True,
        clusters=clusters, return_oof=return_oof, median=median,
    )


def spearman_cluster_bootstrap_ci(oof, n_resamples=BOOTSTRAP_N_RESAMPLES, seed=0):
    """Cluster-bootstrap CI on Spearman ρ, correlated within fold and averaged.

    Matches the reported fold mean. Ranking the pooled predictions would mix folds
    fitted with different intercepts into one ranking.
    """
    y_true = oof["y_true"]
    arms = folds_to_arms(oof["pred"], oof["folds"])

    def _fold_rho(block, arm_pred):
        if len(set(block.tolist())) < 2:
            return None
        rho, _ = spearmanr(y_true[block], arm_pred[block])
        return float(rho) if np.isfinite(rho) else None

    def _rho(rows):
        return score_within_folds(rows, arms, _fold_rho)

    return cluster_bootstrap_ci(
        oof["clusters"], _rho, n_resamples=n_resamples, seed=seed,
        discard_reason=(
            "a fold's resampled rows had fewer than 2 distinct values, so its "
            "rank correlation is undefined (no class to lose — this is a "
            "regression task)"
        ),
        metric_name="spearman",
    )


def paired_spearman_gap_ci(
    oof_a,
    oof_b,
    proteins,
    family_map,
    n_resamples=BOOTSTRAP_N_RESAMPLES,
    seed=0,
):
    """Paired family-bootstrap CI on a fold-aware Spearman difference."""
    indices_a = np.asarray(oof_a["indices"])
    indices_b = np.asarray(oof_b["indices"])
    shared = np.intersect1d(indices_a, indices_b)
    positions_a = {int(index): position for position, index in enumerate(indices_a)}
    positions_b = {int(index): position for position, index in enumerate(indices_b)}
    selected_a = np.array([positions_a[int(index)] for index in shared])
    selected_b = np.array([positions_b[int(index)] for index in shared])

    y_true_a = np.asarray(oof_a["y_true"])[selected_a]
    y_true_b = np.asarray(oof_b["y_true"])[selected_b]
    if not np.array_equal(y_true_a, y_true_b):
        raise ValueError("paired Spearman arms have different labels on shared rows")

    predictions_a = np.asarray(oof_a["pred"])[selected_a]
    predictions_b = np.asarray(oof_b["pred"])[selected_b]
    folds_a = np.asarray(oof_a["folds"])[selected_a]
    folds_b = np.asarray(oof_b["folds"])[selected_b]
    arms_a = folds_to_arms(predictions_a, folds_a)
    arms_b = folds_to_arms(predictions_b, folds_b)

    def _fold_rho(block, arm_predictions):
        if len(np.unique(y_true_a[block])) < 2:
            return None
        rho, _ = spearmanr(y_true_a[block], arm_predictions[block])
        return float(rho) if np.isfinite(rho) else None

    def _rho_a(rows):
        return score_within_folds(rows, arms_a, _fold_rho)

    def _rho_b(rows):
        return score_within_folds(rows, arms_b, _fold_rho)

    shared_proteins = np.array(
        [proteins[int(index)] for index in shared],
        dtype=object,
    )
    shared_families = np.array(
        [family_map[protein] for protein in shared_proteins],
        dtype=object,
    )
    result = paired_cluster_bootstrap_diff_cross_partition(
        resample_clusters=shared_families,
        metric_fn_a=_rho_a,
        metric_fn_b=_rho_b,
        sensitivity_clusters=shared_proteins,
        n_resamples=n_resamples,
        seed=seed,
        discard_reason=(
            "at least one arm had a fold with undefined or non-finite rank "
            "correlation on the shared family resample"
        ),
    )
    result["domain_resampled_sensitivity"] = result.pop(
        "gene_resampled_sensitivity"
    )
    return result


def per_protein_std_bootstrap_ci(
    per_protein_results,
    n_resamples=BOOTSTRAP_N_RESAMPLES,
    seed=0,
):
    """Protein-bootstrap CI on the spread of leave-one-protein-out Spearman rho."""
    finite = [
        (protein, result["spearman"])
        for protein, result in per_protein_results.items()
        if np.isfinite(result["spearman"])
    ]
    if not finite:
        return None
    proteins = np.array([protein for protein, _ in finite], dtype=object)
    correlations = np.array([rho for _, rho in finite], dtype=float)

    def _std(rows):
        return float(np.std(correlations[rows]))

    return cluster_bootstrap_ci(
        proteins,
        _std,
        n_resamples=n_resamples,
        seed=seed,
        metric_name="per_protein_spearman_std",
    )


def _adjudicate_lower_bound(point, ci, threshold):
    """Adjudicate a gate whose point estimate must be at least threshold."""
    if point is None or not np.isfinite(point):
        return "not adjudicated (point estimate unavailable)"
    if ci is None or ci.get("ci_low") is None or ci.get("ci_high") is None:
        return "not adjudicated (CI unavailable)"
    if point >= threshold:
        return "affirmed" if ci["ci_low"] > threshold else "not distinguishable"
    return "underpowered" if ci["ci_high"] >= threshold else "failed"


def _adjudicate_upper_bound(point, ci, threshold):
    """Adjudicate a gate whose point estimate must be at most threshold."""
    if point is None or not np.isfinite(point):
        return "not adjudicated (point estimate unavailable)"
    if ci is None or ci.get("ci_low") is None or ci.get("ci_high") is None:
        return "not adjudicated (CI unavailable)"
    if point <= threshold:
        return "affirmed" if ci["ci_high"] < threshold else "not distinguishable"
    return "underpowered" if ci["ci_low"] <= threshold else "failed"



def _fit_one_protein(prot, X, y, proteins):
    """Fit Ridge leaving out one protein; return (prot, result) or None."""
    mask = proteins == prot
    if mask.sum() < 5:
        return None
    tr = np.where(~mask)[0]
    te = np.where(mask)[0]
    if len(tr) < 10:
        return None
    Xtr, Xte = standardize(X[tr], X[te])
    clf = Ridge(alpha=1.0)
    clf.fit(Xtr, y[tr])
    pred = clf.predict(Xte)
    rho, pval = spearmanr(y[te], pred)
    return prot, {
        "spearman": float(rho),
        "p_value": float(pval),
        "n_variants": int(mask.sum()),
    }


def per_protein_spearman(X, y, proteins, n_jobs):
    """Leave-one-protein-out Ridge ρ for each protein with ≥5 variants.

    Each worker standardizes and fits against nearly the full 177k×1280 matrix,
    so n_jobs is capped explicitly (never -1) to bound peak RAM, and each worker
    is limited to one BLAS thread so its own linear algebra doesn't oversubscribe
    the cores the outer Parallel pool already claimed.
    """
    unique = sorted(set(proteins))
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        hits = Parallel()(
            delayed(_fit_one_protein)(prot, X, y, proteins) for prot in unique
        )
    return {prot: result for prot, result in (h for h in hits if h is not None)}



def run_stability_projection_3c(
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
    n_jobs=1,
    compute_ci=True,
):
    """Project stability out of mechanism delta_mean; compare family-split F1."""
    from sklearn.linear_model import LogisticRegression
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

    # Control 3C hinges on this removal; assert var along stability_dir ≈ 0.
    var_before = float(np.var(merged_scaled.astype(np.float64) @ stability_dir))
    var_after = float(np.var(residuals.astype(np.float64) @ stability_dir))
    if var_after > 1e-6 * var_before + 1e-8:
        raise AssertionError(
            f"stability projection failed: var along stability_dir was {var_before:.3e} "
            f"before and {var_after:.3e} after projecting out — the removed direction "
            "leaked back in, so the projected arm still contains stability signal."
        )

    y = np.asarray(merged_labels)

    def _run_3c_seed(seed, collect_oof):
        splits = family_split_cv(merged_proteins, pfam_map, n_folds=n_folds, seed=seed)
        seed_oof = {}
        seed_baseline_f1 = None
        seed_projected_f1 = None
        for X, tag in [(merged_scaled, "baseline"), (residuals, "projected")]:
            fold_f1s = []
            oof_y, oof_pred, oof_genes, oof_folds = [], [], [], []
            for fold_i, (tr, te) in enumerate(splits):
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
                if collect_oof:
                    oof_y.append(y[te])
                    oof_pred.append(pred)
                    oof_genes.append(merged_proteins[te])
                    oof_folds.append(np.full(len(te), fold_i, dtype=int))
            seed_f1_mean, _, _ = mean_std_n(fold_f1s)
            if tag == "baseline":
                seed_baseline_f1 = seed_f1_mean
            else:
                seed_projected_f1 = seed_f1_mean
            if collect_oof and oof_y:
                seed_oof[tag] = {
                    "y_true": np.concatenate(oof_y),
                    "pred": np.concatenate(oof_pred),
                    "genes": np.concatenate(oof_genes),
                    "folds": np.concatenate(oof_folds),
                }
        return seed_baseline_f1, seed_projected_f1, seed_oof

    seed0_bl, seed0_pr, seed0_oof = _run_3c_seed(
        0, collect_oof=compute_ci
    )
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        rest = Parallel()(
            delayed(_run_3c_seed)(seed, False) for seed in range(1, n_seeds)
        )
    baseline_f1s = [seed0_bl] + [r[0] for r in rest]
    projected_f1s = [seed0_pr] + [r[1] for r in rest]

    baseline_f1_mean, baseline_f1_std, _ = mean_std_n(baseline_f1s)
    projected_f1_mean, projected_f1_std, _ = mean_std_n(projected_f1s)
    difference_ci = None
    if compute_ci and "baseline" in seed0_oof and "projected" in seed0_oof:
        baseline_oof = seed0_oof["baseline"]
        projected_oof = seed0_oof["projected"]
        if not np.array_equal(baseline_oof["genes"], projected_oof["genes"]):
            raise AssertionError("3C baseline and projected OOF rows are not aligned")
        clusters = np.array(
            [pfam_map[gene] for gene in baseline_oof["genes"]], dtype=object
        )

        def _fold_f1(oof):
            y_true = oof["y_true"]
            arms = folds_to_arms(oof["pred"], oof["folds"])

            def _score(block, arm_pred):
                return fold_macro_f1(y_true, block, arm_pred, MECHANISM_CLASSES)
            return lambda rows: score_within_folds(rows, arms, _score)

        # Both arms are the same fold assignment (one seed, one split), so the paired
        # difference stays row-for-row aligned while each side is scored per fold.
        _projected_f1 = _fold_f1(projected_oof)
        _baseline_f1 = _fold_f1(baseline_oof)

        difference_ci = paired_cluster_bootstrap_diff(
            clusters,
            _projected_f1,
            _baseline_f1,
            n_resamples=n_boot,
            seed=0,
            discard_reason=(
                "at least one arm's fold lost a mechanism class on the shared "
                "resample"
            ),
        )
    inferential_point = (
        None if difference_ci is None else difference_ci.get("point_diff")
    )
    control_3c_verdict = _adjudicate_upper_bound(
        inferential_point, difference_ci, 0.01
    )
    return {
        "baseline_f1_mean": baseline_f1_mean,
        "baseline_f1_std": baseline_f1_std,
        "projected_f1_mean": projected_f1_mean,
        "projected_f1_std": projected_f1_std,
        "delta_f1": projected_f1_mean - baseline_f1_mean,
        "inferential_point_estimate": inferential_point,
        "difference_ci": difference_ci,
        "3C_passes": control_3c_verdict == "affirmed",
        "3C_verdict": control_3c_verdict,
    }


def apply_decision_rule(control_3a_ci, control_3b_gap_ci, control_3c, control_3d_ci):
    """Adjudicate controls 3A-3D from their registered point/CI pairs."""
    point_3a = None if control_3a_ci is None else control_3a_ci.get("point")
    point_3b = (
        None if control_3b_gap_ci is None else control_3b_gap_ci.get("point_diff")
    )
    point_3c = (
        None if control_3c is None else control_3c.get("inferential_point_estimate")
    )
    ci_3c = None if control_3c is None else control_3c.get("difference_ci")
    point_3d = None if control_3d_ci is None else control_3d_ci.get("point")

    gates = {
        "3A": {
            "criterion": "random_split_spearman_at_least_0.5",
            "threshold": 0.5,
            "point_estimate": point_3a,
            "ci": control_3a_ci,
            "verdict": _adjudicate_lower_bound(point_3a, control_3a_ci, 0.5),
        },
        "3B": {
            "criterion": "random_minus_family_spearman_at_most_0.10",
            "threshold": 0.10,
            "point_estimate": point_3b,
            "ci": control_3b_gap_ci,
            "verdict": _adjudicate_upper_bound(
                point_3b, control_3b_gap_ci, 0.10
            ),
        },
        "3C": {
            "criterion": "projected_minus_baseline_mechanism_f1_at_most_0.01",
            "threshold": 0.01,
            "point_estimate": point_3c,
            "ci": ci_3c,
            "verdict": _adjudicate_upper_bound(point_3c, ci_3c, 0.01),
        },
        "3D": {
            "criterion": "per_protein_spearman_std_at_most_0.10",
            "threshold": 0.10,
            "point_estimate": point_3d,
            "ci": control_3d_ci,
            "verdict": _adjudicate_upper_bound(point_3d, control_3d_ci, 0.10),
        },
    }

    if gates["3A"]["verdict"] == "failed":
        overall = "3A FAILED"
    elif gates["3B"]["verdict"] == "failed":
        overall = "LEAKY"
    elif gates["3C"]["verdict"] == "failed":
        overall = "3C FAILED"
    elif gates["3D"]["verdict"] == "failed":
        overall = "HETEROGENEOUS"
    elif all(gate["verdict"] == "affirmed" for gate in gates.values()):
        overall = "ROBUST"
    else:
        overall = "NOT FULLY ADJUDICATED"
    return {"overall": overall, "gates": gates}



def main(compute_ci=True, n_boot=BOOTSTRAP_N_RESAMPLES, n_jobs=1):
    inputs = load_stability_inputs(include_pos=True)
    variants = inputs.variants
    proteins = inputs.proteins
    ddg = inputs.ddg
    family_map = inputs.family_map
    delta_mean = inputs.delta_mean
    delta_pos = inputs.delta_pos
    n_families = inputs.n_families
    stability_fingerprints = inputs.input_fingerprints
    analysis_parameters = {
        "n_folds": N_FOLDS,
        "n_seeds": N_SEEDS,
        "compute_ci": compute_ci,
        "n_boot": n_boot,
        "ridge_alpha": 1.0,
    }

    print(f"Embeddings: delta_mean {delta_mean.shape}, delta_pos {delta_pos.shape}")

    global_median = float(np.median(ddg))
    print(f"Global ΔΔG median: {global_median:.4f}")

    def _run_seed0():
        """Seed 0 additionally computes cluster-bootstrap CIs (joblib-parallel
        internally), so it must run alone rather than alongside the other seeds
        — nesting Parallel inside Parallel just splits the same core pool."""
        print("\n── Seed 0 ──")
        splits_by_name = stability_splits(0, len(variants), proteins, family_map)
        seed_result = {"seed": 0}
        oofs = {}
        for feat_name, X in [("delta_mean", delta_mean), ("delta_pos", delta_pos)]:
            for split_name, splits in splits_by_name.items():
                key = f"{feat_name}_{split_name}"
                if compute_ci:
                    ci_clusters = (
                        np.array([family_map.get(p) for p in proteins], dtype=object)
                        if split_name == "family"
                        else proteins
                    )
                    res, oof = run_ridge_with_auroc(
                        X,
                        ddg,
                        splits,
                        clusters=ci_clusters,
                        return_oof=True,
                        median=global_median,
                    )
                else:
                    res = run_ridge_with_auroc(
                        X, ddg, splits, median=global_median
                    )
                    oof = None
                if compute_ci and oof is not None:
                    res["ci"] = spearman_cluster_bootstrap_ci(
                        oof, n_resamples=n_boot, seed=0
                    )
                    oofs[key] = oof
                seed_result[key] = res
                if res:
                    print(
                        f"  {key}: ρ={res['spearman_mean']:.3f}±{res['spearman_std']:.3f}  "
                        f"AUROC={res['auroc_mean']:.3f}"
                    )
        return seed_result, oofs

    def _run_seed_plain(seed):
        """Seeds 1..N-1: no CI, no OOF — independent of each other and of seed 0."""
        print(f"\n── Seed {seed} ──")
        splits_by_name = stability_splits(seed, len(variants), proteins, family_map)
        seed_result = {"seed": seed}
        for feat_name, X in [("delta_mean", delta_mean), ("delta_pos", delta_pos)]:
            for split_name, splits in splits_by_name.items():
                key = f"{feat_name}_{split_name}"
                res = run_ridge_with_auroc(X, ddg, splits, median=global_median)
                seed_result[key] = res
                if res:
                    print(
                        f"  {key}: ρ={res['spearman_mean']:.3f}±{res['spearman_std']:.3f}  "
                        f"AUROC={res['auroc_mean']:.3f}"
                    )
        return seed_result

    seed0_result, seed0_oofs = _run_seed0()
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        remaining_results = Parallel()(
            delayed(_run_seed_plain)(seed) for seed in range(1, N_SEEDS)
        )
    results_by_seed = [seed0_result] + list(remaining_results)

    print("\nPer-protein Spearman (leave-one-protein-out)...")
    per_prot = per_protein_spearman(delta_mean, ddg, proteins, n_jobs=n_jobs)
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

    per_protein_output = {
        "per_protein": per_prot,
        "input_fingerprints": stability_fingerprints,
        "analysis_parameters": {
            "probe": "leave_one_protein_out_ridge",
            "ridge_alpha": 1.0,
        },
    }
    write_result_json(
        os.path.join(OUT, "per_protein_spearman.json"),
        per_protein_output,
        seeds=None,
    )

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
            "across_seed": {
                "spearman_mean": rho_mean,
                "spearman_std": rho_std,
                "auroc_mean": au_mean,
                "auroc_std": au_std,
                "n_seeds": n_seeds_used,
            },
        }
        seed0_ci = results_by_seed[0].get(key, {}).get("ci") if results_by_seed else None
        if seed0_ci is not None:
            summary[key]["seed0_inference"] = {
                "point_estimate": seed0_ci.get("point"),
                "ci": seed0_ci,
            }

    summary["per_protein"] = {
        "spearman_mean": per_prot_mean,
        "spearman_std": per_prot_std,
        "n_proteins": len(prot_rhos),
        "n_proteins_finite": n_finite_prot,
    }
    control_3d_ci = None
    if compute_ci:
        control_3d_ci = per_protein_std_bootstrap_ci(
            per_prot, n_resamples=n_boot, seed=0
        )
        summary["per_protein"]["spearman_std_ci"] = control_3d_ci

    required_3c_paths = [VALID_VARIANTS_JSON, EMB_WT_MEAN, EMB_MUT_MEAN, PFAM_JSON]
    missing_3c_paths = [
        str(path) for path in required_3c_paths if not os.path.exists(path)
    ]
    if missing_3c_paths:
        raise FileNotFoundError(
            "registered control 3C requires missing input files: "
            + ", ".join(missing_3c_paths)
        )
    print("\nRunning 3C stability projection test...")
    pfam_map = load_pfam_map(PFAM_JSON)

    merged_delta, merged_labels, merged_proteins = load_merged()
    with open(VALID_VARIANTS_JSON) as handle:
        mechanism_variants = json.load(handle)
    mechanism_projection_fingerprints = {
        "labeled_variants": labeled_variant_fingerprint(
            mechanism_variants, merged_labels
        ),
        "delta_mean_content": embedding_fingerprint(merged_delta),
        "pfam_assignments": pfam_fingerprint(
            pfam_map, merged_proteins.tolist()
        ),
    }

    control_3c_result = run_stability_projection_3c(
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
        n_jobs=n_jobs,
        compute_ci=compute_ci,
    )
    inferential_3c = control_3c_result["inferential_point_estimate"]
    inferential_3c_text = (
        "unavailable" if inferential_3c is None else f"{inferential_3c:+.3f}"
    )
    print(
        f"  3C: five-seed baseline F1={control_3c_result['baseline_f1_mean']:.3f}  "
        f"projected F1={control_3c_result['projected_f1_mean']:.3f}  "
        f"mean Δ={control_3c_result['delta_f1']:+.3f}  "
        f"seed-0 inferential Δ={inferential_3c_text}  "
        f"verdict={control_3c_result['3C_verdict']}"
    )
    write_result_json(
        os.path.join(OUT, "stability_projection_3c.json"),
        {
            **control_3c_result,
            "input_fingerprints": {
                "stability": stability_fingerprints,
                "mechanism_projection": mechanism_projection_fingerprints,
            },
            "analysis_parameters": analysis_parameters,
        },
        seeds=list(range(N_SEEDS)),
    )

    control_3b_gap_ci = None
    oof_random = seed0_oofs.get("delta_mean_random")
    oof_family = seed0_oofs.get("delta_mean_family")
    if compute_ci and oof_random is not None and oof_family is not None:
        print("\nComputing paired CI on random-to-family Spearman gap (3B)...")
        control_3b_gap_ci = paired_spearman_gap_ci(
            oof_random,
            oof_family,
            proteins,
            family_map,
            n_resamples=n_boot,
            seed=0,
        )
        print(
            f"  3B gap: {control_3b_gap_ci['point_diff']:.3f} "
            f"[{control_3b_gap_ci.get('ci_low', '?')}, {control_3b_gap_ci.get('ci_high', '?')}]"
        )

    dm_random = summary["delta_mean_random"]["across_seed"]["spearman_mean"]
    dm_family = summary["delta_mean_family"]["across_seed"]["spearman_mean"]

    control_3a_ci = summary.get("delta_mean_random", {}).get(
        "seed0_inference", {}
    ).get("ci")
    adjudication = apply_decision_rule(
        control_3a_ci,
        control_3b_gap_ci,
        control_3c_result,
        control_3d_ci,
    )
    verdict = adjudication["overall"]
    summary["result_version"] = 3
    summary["verdict"] = verdict
    summary["gates"] = adjudication["gates"]
    if control_3b_gap_ci is not None:
        summary["3B_gap_ci"] = control_3b_gap_ci
    summary["n_variants"] = len(variants)
    summary["n_proteins"] = len(set(proteins))
    summary["n_families"] = n_families
    summary["n_seeds"] = N_SEEDS
    summary["3C"] = control_3c_result
    summary["input_fingerprints"] = {
        "stability": stability_fingerprints,
        "mechanism_projection": mechanism_projection_fingerprints,
    }
    summary["analysis_parameters"] = analysis_parameters

    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"  delta_mean random ρ  : {dm_random:.3f}  (3A threshold ≥ 0.5)")
    print(f"  delta_mean family ρ  : {dm_family:.3f}")
    print(f"  Δ (random − family)  : {dm_random - dm_family:.3f}  (LEAKY if Δ ≥ 0.10)")
    print(f"  per-domain ρ std     : {per_prot_std:.3f}  (3D threshold ≤ 0.10)")
    if control_3c_result:
        print(
            f"  3C seed-0 Δ mechanism F1: "
            f"{inferential_3c_text}  "
            f"(passes if ≤ +0.01 — stability projection doesn't help mechanism)"
        )
    for gate_name, gate in adjudication["gates"].items():
        print(f"  {gate_name} verdict          : {gate['verdict']}")
    print(f"{'='*60}")

    write_result_json(os.path.join(OUT, "summary.json"), summary, seeds=list(range(N_SEEDS)))
    print(f"\nResults written to {OUT}/")


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_ci", action="store_true", help="skip cluster-bootstrap CIs")
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    parser.add_argument(
        "--n_jobs", type=int, required=True,
        help="Max concurrent worker processes for the per-seed, per-protein, and "
        "3C parallel loops. Each worker standardizes/fits against most of the "
        "177k x 1280 matrix, so this must be set explicitly (never -1) to bound "
        "peak RAM. Start low (e.g. 4), watch peak RAM, raise only if it fits.",
    )
    args = parser.parse_args()
    main(compute_ci=not args.no_ci, n_boot=args.n_boot, n_jobs=args.n_jobs)


if __name__ == "__main__":
    _cli()
