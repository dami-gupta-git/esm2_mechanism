"""Linear (Ridge) stability probe on Tsuboyama 2023 point-mutant ΔΔG.

Evaluates the stability control gates 3A-3D.
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
from esm2_mech.utils.classification import validate_complete_classification_splits
from esm2_mech.utils.constants import (
    BOOTSTRAP_N_RESAMPLES,
    INFERENTIAL_SEED,
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
from esm2_mech.utils.metrics import (
    auroc_at_median,
    fold_macro_f1,
    mean_std_n,
    standardize,
)
from esm2_mech.utils.probes import run_logreg_cv
from esm2_mech.utils.seed_aggregation import (
    SEED_STATUS_UNSCORABLE,
    aggregate_paired_seed_difference,
    aggregate_result_contract,
    aggregate_seed_results,
    make_seed_record,
    read_seed_point_estimate,
    seed_result_contract,
)
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
    # Each metric is judged on its own folds. A fold whose AUROC is undefined —
    # every held-out variant on one side of the median, say — says nothing about
    # the Spearman correlation from the same fold, so one undefined metric no
    # longer withholds the others.
    def _metric_status(fold_values):
        return (
            "success"
            if fold_values and all(np.isfinite(value) for value in fold_values)
            else SEED_STATUS_UNSCORABLE
        )

    out = {
        "status": "success" if rhos else SEED_STATUS_UNSCORABLE,
        "spearman_status": _metric_status(rhos),
        "auroc_status": _metric_status(aurocs),
        "n_folds": len(rhos),
        "sampling_unit": "cv_fold",
    }
    if out["spearman_status"] == "success":
        rho_mean, rho_std, _ = mean_std_n(rhos)
        out["spearman_mean"] = rho_mean
        out["spearman_fold_std"] = rho_std
    if out["auroc_status"] == "success":
        au_mean, au_std, _ = mean_std_n(aurocs)
        out["auroc_mean"] = au_mean
        out["auroc_fold_std"] = au_std
    if with_pearson:
        out["pearson_status"] = _metric_status(pearsons)
        if out["pearson_status"] == "success":
            pearson_mean, pearson_std, _ = mean_std_n(pearsons)
            out["pearson_mean"] = pearson_mean
            out["pearson_fold_std"] = pearson_std
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
        family_groups = np.array(
            [pfam_map.get(gene) for gene in merged_proteins], dtype=object
        )
        split_contract = validate_complete_classification_splits(
            splits,
            requested_folds=n_folds,
            eligible_rows=np.concatenate([test for _train, test in splits]),
            labels=y,
            classes=MECHANISM_CLASSES,
            groups=family_groups,
            held_out_unit="family",
        )
        seed_result = {**seed_result_contract(seed), "results": {}}
        seed_oof = {}
        for X, tag in [(merged_scaled, "baseline"), (residuals, "projected")]:
            probe_result = run_logreg_cv(
                X,
                y,
                splits,
                MECHANISM_CLASSES,
                split_contract,
                seed=seed,
                genes=merged_proteins if collect_oof else None,
                return_oof=collect_oof,
                prescaled=True,
                compute_per_gene=False,
                label=f"control_3c_{tag}",
            )
            result, oof = probe_result if collect_oof else (probe_result, None)
            seed_result["results"][tag] = result
            if oof is not None:
                seed_oof[tag] = {
                    **oof,
                    "pred": np.array(
                        [MECHANISM_CLASSES[column] for column in oof["proba"].argmax(1)]
                    ),
                }
        # The out-of-fold arrays stay outside seed_result, which is written to the
        # result file.
        return seed_result, seed_oof

    requested_seeds = tuple(range(n_seeds))
    # Seed 0 also carries the paired bootstrap, which parallelises internally, so
    # it runs on its own rather than nested inside the per-seed pool.
    seed0_result, seed0_oof = _run_3c_seed(0, collect_oof=compute_ci)
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        remaining = Parallel()(
            delayed(_run_3c_seed)(seed, False) for seed in requested_seeds[1:]
        )
    seed_results = [seed0_result] + [result for result, _oof in remaining]
    baseline = aggregate_seed_results(
        requested_seeds,
        seed_results,
        lambda result: result["results"]["baseline"].get("macro_f1_mean"),
        status=lambda result: result["results"]["baseline"]["status"],
    )
    projected = aggregate_seed_results(
        requested_seeds,
        seed_results,
        lambda result: result["results"]["projected"].get("macro_f1_mean"),
        status=lambda result: result["results"]["projected"]["status"],
    )
    difference = aggregate_paired_seed_difference(
        requested_seeds,
        [
            make_seed_record(
                seed_result["seed"],
                seed_result["results"]["projected"].get("macro_f1_mean"),
                status=seed_result["results"]["projected"]["status"],
            )
            for seed_result in seed_results
        ],
        [
            make_seed_record(
                seed_result["seed"],
                seed_result["results"]["baseline"].get("macro_f1_mean"),
                status=seed_result["results"]["baseline"]["status"],
            )
            for seed_result in seed_results
        ],
    )

    # Seed 0's paired family bootstrap: a within-seed interval on one seed's
    # difference, reported separately from the across-seed paired difference above.
    difference_ci = None
    if "baseline" in seed0_oof and "projected" in seed0_oof:
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

        # Both arms share the fold assignment (one seed, one split), so the paired
        # difference stays row-for-row aligned while each side is scored per fold.
        # Macro-F1 has a fixed class denominator, so it is defined on every draw
        # and the plain cluster resample applies.
        difference_ci = paired_cluster_bootstrap_diff(
            clusters,
            _fold_f1(projected_oof),
            _fold_f1(baseline_oof),
            n_resamples=n_boot,
            seed=0,
            discard_reason="a fold's resampled rows lost every row",
        )
    inferential_point = (
        None if difference_ci is None else difference_ci.get("point_diff")
    )
    control_3c_verdict = _adjudicate_upper_bound(
        inferential_point, difference_ci, 0.01
    )
    return {
        **aggregate_result_contract(),
        "baseline_f1": baseline.to_dict(),
        "projected_f1": projected.to_dict(),
        "projected_minus_baseline_f1": difference.to_dict(),
        "per_seed_fold_summaries": seed_results,
        "inferential_seed": INFERENTIAL_SEED,
        "inferential_point_estimate": inferential_point,
        "difference_ci": difference_ci,
        "3C_passes": control_3c_verdict == "affirmed",
        "3C_verdict": control_3c_verdict,
    }


def apply_decision_rule(control_3a_ci, control_3b_gap_ci, control_3c, control_3d_ci):
    """Adjudicate controls 3A-3D from their registered point/CI pairs.

    Each gate reads the point estimate the interval beside it was built around, so
    a verdict never compares an interval with a different estimand.
    """
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
            "verdict": _adjudicate_upper_bound(point_3b, control_3b_gap_ci, 0.10),
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


def _show_seed_value(value, signed=False):
    if value is None or not np.isfinite(value):
        return "unavailable"
    return f"{value:+.3f}" if signed else f"{value:.3f}"



def main(n_jobs=1, n_seeds=N_SEEDS, compute_ci=True, n_boot=BOOTSTRAP_N_RESAMPLES):
    requested_seeds = tuple(range(n_seeds))
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
        "n_seeds": n_seeds,
        "compute_ci": compute_ci,
        "n_boot": n_boot,
        "ridge_alpha": 1.0,
    }

    print(f"Embeddings: delta_mean {delta_mean.shape}, delta_pos {delta_pos.shape}")

    global_median = float(np.median(ddg))
    print(f"Global ΔΔG median: {global_median:.4f}")

    def _run_seed(seed, collect_oof):
        print(f"\n── Seed {seed} ──")
        splits_by_name = stability_splits(seed, len(variants), proteins, family_map)
        seed_result = {**seed_result_contract(seed), "results": {}}
        seed_oofs = {}
        for feat_name, X in [("delta_mean", delta_mean), ("delta_pos", delta_pos)]:
            for split_name, splits in splits_by_name.items():
                key = f"{feat_name}_{split_name}"
                if collect_oof:
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
                    if oof is not None:
                        seed_oofs[key] = oof
                else:
                    res = run_ridge_with_auroc(X, ddg, splits, median=global_median)
                seed_result["results"][key] = res
                spearman_text = (
                    f"ρ={res['spearman_mean']:.3f}±"
                    f"{res['spearman_fold_std']:.3f} fold SD"
                    if res["spearman_status"] == "success"
                    else "ρ=unscorable"
                )
                auroc_text = (
                    f"AUROC={res['auroc_mean']:.3f}"
                    if res["auroc_status"] == "success"
                    else "AUROC=unscorable"
                )
                print(f"  {key}: {spearman_text}  {auroc_text}")
        return seed_result, seed_oofs

    # Seed 0 also carries the cluster bootstrap, which parallelises internally, so
    # it runs on its own rather than nested inside the per-seed pool.
    seed0_result, seed0_oofs = _run_seed(0, collect_oof=compute_ci)
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        remaining = Parallel()(
            delayed(_run_seed)(seed, False) for seed in requested_seeds[1:]
        )
    results_by_seed = [seed0_result] + [result for result, _oofs in remaining]

    seed0_intervals = {}
    if compute_ci:
        print("\nCluster-bootstrap CIs on seed 0's fold-mean Spearman...")
        for key, oof in seed0_oofs.items():
            seed0_intervals[key] = spearman_cluster_bootstrap_ci(
                oof, n_resamples=n_boot, seed=0
            )

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

    summary = {**aggregate_result_contract()}
    all_keys = sorted(results_by_seed[0]["results"])
    for key in all_keys:
        spearman = aggregate_seed_results(
            requested_seeds,
            results_by_seed,
            lambda seed_result, key=key: seed_result["results"][key].get(
                "spearman_mean"
            ),
            status=lambda seed_result, key=key: seed_result["results"][key][
                "spearman_status"
            ],
        )
        auroc = aggregate_seed_results(
            requested_seeds,
            results_by_seed,
            lambda seed_result, key=key: seed_result["results"][key].get(
                "auroc_mean"
            ),
            status=lambda seed_result, key=key: seed_result["results"][key][
                "auroc_status"
            ],
        )
        summary[key] = {
            "across_seed": {
                "spearman": spearman.to_dict(),
                "auroc": auroc.to_dict(),
            },
            "per_seed_fold_summaries": {
                str(result["seed"]): result["results"][key]
                for result in results_by_seed
            },
        }
        # A within-seed interval on seed 0's estimate, kept in its own field: it
        # is not the across-seed spread reported in the aggregate above.
        if key in seed0_intervals:
            summary[key]["seed0_inference"] = {
                "seed": 0,
                "point_estimate": seed0_intervals[key].get("point"),
                "ci": seed0_intervals[key],
            }

    summary["per_protein"] = {
        "spearman_protein_mean": per_prot_mean,
        "spearman_protein_std": per_prot_std,
        "sampling_unit": "protein",
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
        n_seeds=n_seeds,
        n_boot=n_boot,
        n_jobs=n_jobs,
        compute_ci=compute_ci,
    )
    baseline_3c = read_seed_point_estimate(control_3c_result["baseline_f1"]).value
    projected_3c = read_seed_point_estimate(control_3c_result["projected_f1"]).value
    difference_3c = read_seed_point_estimate(
        control_3c_result["projected_minus_baseline_f1"]
    ).value
    inferential_3c = control_3c_result["inferential_point_estimate"]
    print(
        f"  3C: baseline F1={_show_seed_value(baseline_3c)}  "
        f"projected F1={_show_seed_value(projected_3c)}  "
        f"paired mean Δ={_show_seed_value(difference_3c, signed=True)}  "
        f"seed-0 inferential Δ={_show_seed_value(inferential_3c, signed=True)}  "
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
        seeds=list(requested_seeds),
    )

    control_3b_gap = aggregate_paired_seed_difference(
        requested_seeds,
        [
            make_seed_record(
                result["seed"],
                result["results"]["delta_mean_random"].get("spearman_mean"),
                status=result["results"]["delta_mean_random"]["spearman_status"],
            )
            for result in results_by_seed
        ],
        [
            make_seed_record(
                result["seed"],
                result["results"]["delta_mean_family"].get("spearman_mean"),
                status=result["results"]["delta_mean_family"]["spearman_status"],
            )
            for result in results_by_seed
        ],
    )
    summary["3B_random_minus_family_spearman"] = control_3b_gap.to_dict()

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
        summary["3B_gap_ci"] = control_3b_gap_ci
        print(
            f"  3B gap: {control_3b_gap_ci['point_diff']:.3f} "
            f"[{control_3b_gap_ci.get('ci_low', '?')}, "
            f"{control_3b_gap_ci.get('ci_high', '?')}]"
        )

    dm_random = read_seed_point_estimate(
        summary["delta_mean_random"]["across_seed"]["spearman"]
    ).value
    dm_family = read_seed_point_estimate(
        summary["delta_mean_family"]["across_seed"]["spearman"]
    ).value

    control_3a_ci = (
        summary.get("delta_mean_random", {}).get("seed0_inference", {}).get("ci")
    )
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
    summary["n_variants"] = len(variants)
    summary["n_proteins"] = len(set(proteins))
    summary["n_families"] = n_families
    summary["n_seeds"] = n_seeds
    summary["3C"] = control_3c_result
    summary["input_fingerprints"] = {
        "stability": stability_fingerprints,
        "mechanism_projection": mechanism_projection_fingerprints,
    }
    summary["analysis_parameters"] = analysis_parameters

    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"  delta_mean random ρ  : {_show_seed_value(dm_random)}  (3A threshold ≥ 0.5)")
    print(f"  delta_mean family ρ  : {_show_seed_value(dm_family)}")
    print(
        "  Δ (random − family)  : "
        f"{_show_seed_value(control_3b_gap.mean, signed=True)}  "
        "(LEAKY if Δ ≥ 0.10)"
    )
    print(
        f"  per-protein ρ std    : {_show_seed_value(per_prot_std)}  "
        "(3D threshold ≤ 0.10)"
    )
    if control_3c_result:
        print(
            f"  3C paired-seed Δ mechanism F1: "
            f"{_show_seed_value(difference_3c, signed=True)}  "
            f"(seed-0 inferential Δ "
            f"{_show_seed_value(inferential_3c, signed=True)})  "
            f"(passes if ≤ +0.01 — stability projection doesn't help mechanism)"
        )
    for gate_name, gate in adjudication["gates"].items():
        print(f"  {gate_name} verdict          : {gate['verdict']}")
    print(f"{'='*60}")

    write_result_json(os.path.join(OUT, "summary.json"), summary, seeds=list(requested_seeds))
    print(f"\nResults written to {OUT}/")


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument(
        "--no_ci", action="store_true", help="skip cluster-bootstrap CIs"
    )
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    parser.add_argument(
        "--n_jobs", type=int, required=True,
        help="Max concurrent worker processes for the per-seed, per-protein, and "
        "3C parallel loops. Each worker standardizes/fits against most of the "
        "177k x 1280 matrix, so this must be set explicitly (never -1) to bound "
        "peak RAM. Start low (e.g. 4), watch peak RAM, raise only if it fits.",
    )
    args = parser.parse_args()
    main(
        n_jobs=args.n_jobs,
        n_seeds=args.seeds,
        compute_ci=not args.no_ci,
        n_boot=args.n_boot,
    )


if __name__ == "__main__":
    _cli()
