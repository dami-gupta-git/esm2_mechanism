"""Linear (Ridge) stability probe on Tsuboyama 2023 point-mutant ΔΔG.

Evaluates the the stability control gates.
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
    oof_score_arms,
    paired_cluster_bootstrap_diff,
    paired_cluster_bootstrap_diff_cross_partition,
    score_within_folds,
)
from esm2_mech.utils.classification import validate_complete_classification_splits
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
    aggregate_seed_oof,
    aggregate_seed_results,
    make_seed_payload_record,
    make_seed_record,
    read_seed_point_estimate,
    seed_count,
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

RANDOM_SPLIT_MIN_SPEARMAN = 0.50
FAMILY_TRANSFER_AFFIRMED_MAX_GAP = 0.05
FAMILY_TRANSFER_LEAKY_MIN_GAP = 0.10
STABILITY_PROJECTION_MAX_F1_CHANGE = 0.01
PER_PROTEIN_MAX_SPEARMAN_SPREAD = 0.10

os.makedirs(OUT, exist_ok=True)
os.makedirs(str(_DATA_DIR / "embeddings" / ESM2_MODEL), exist_ok=True)


def run_regression_cv(
    X,
    y,
    splits,
    clf_fn,
    with_pearson=True,
    clusters=None,
    return_oof=False,
    median=None,
    label=None,
):
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
        X,
        y,
        splits,
        lambda: Ridge(alpha=1.0),
        with_pearson=True,
        clusters=clusters,
        return_oof=return_oof,
        median=median,
    )


def _combine_regression_oof(requested_seeds, oof_by_seed, declared_indices):
    """Align one complete regression OOF block for every requested seed."""
    row_ids = np.asarray(declared_indices, dtype=int)
    combined = {}
    y_true = None
    clusters = None
    for seed in requested_seeds:
        if seed not in oof_by_seed:
            raise ValueError(f"missing regression OOF predictions for seed {seed}")
        oof = oof_by_seed[seed]
        seed_rows = np.asarray(oof["indices"], dtype=int)
        if set(seed_rows.tolist()) != set(row_ids.tolist()):
            raise ValueError(f"regression OOF row set differs for seed {seed}")
        positions = {row_id: pos for pos, row_id in enumerate(seed_rows.tolist())}
        order = np.array([positions[row_id] for row_id in row_ids.tolist()])
        seed_y = np.asarray(oof["y_true"])[order]
        seed_clusters = np.asarray(oof["clusters"], dtype=object)[order]
        if y_true is None:
            y_true = seed_y
            clusters = seed_clusters
        elif not np.array_equal(seed_y, y_true) or not np.array_equal(
            seed_clusters, clusters
        ):
            raise ValueError(f"regression OOF metadata differs for seed {seed}")
        combined[seed] = {
            "proba": np.asarray(oof["pred"])[order],
            "folds": np.asarray(oof["folds"])[order],
        }
    return {
        "requested_seeds": list(requested_seeds),
        "indices": row_ids,
        "y_true": y_true,
        "clusters": clusters,
        "oof_by_seed": combined,
    }


def spearman_cluster_bootstrap_ci(oof, n_resamples=BOOTSTRAP_N_RESAMPLES, seed=0):
    """Cluster-bootstrap CI on Spearman ρ, correlated within fold and averaged.

    Matches the reported fold mean. Ranking the pooled predictions would mix folds
    fitted with different intercepts into one ranking.
    """
    y_true = oof["y_true"]
    arms = oof_score_arms(oof, "stability Spearman")

    def _fold_rho(block, arm_pred):
        if len(set(block.tolist())) < 2:
            return None
        rho, _ = spearmanr(y_true[block], arm_pred[block])
        return float(rho) if np.isfinite(rho) else None

    def _rho(rows):
        return score_within_folds(rows, arms, _fold_rho)

    return cluster_bootstrap_ci(
        oof["clusters"],
        _rho,
        n_resamples=n_resamples,
        seed=seed,
        discard_reason=(
            "a fold's resampled rows had fewer than 2 distinct values, so its "
            "rank correlation is undefined (no class to lose — this is a "
            "regression task)"
        ),
        metric_name="spearman",
    )


def _paired_spearman_gap_inputs(oof_a, oof_b, proteins, family_map):
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

    arms_a = [
        (predictions[selected_a], folds[selected_a], np.unique(folds[selected_a]))
        for predictions, folds, _fold_ids in oof_score_arms(
            oof_a, "stability random split"
        )
    ]
    arms_b = [
        (predictions[selected_b], folds[selected_b], np.unique(folds[selected_b]))
        for predictions, folds, _fold_ids in oof_score_arms(
            oof_b, "stability family split"
        )
    ]

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
    return _rho_a, _rho_b, shared_proteins, shared_families


def paired_spearman_gap_ci(
    oof_a,
    oof_b,
    proteins,
    family_map,
    n_resamples=BOOTSTRAP_N_RESAMPLES,
    seed=0,
):
    """Paired family-bootstrap CI on a fold-aware Spearman difference."""
    rho_a, rho_b, shared_proteins, shared_families = _paired_spearman_gap_inputs(
        oof_a, oof_b, proteins, family_map
    )
    result = paired_cluster_bootstrap_diff_cross_partition(
        resample_clusters=shared_families,
        metric_fn_a=rho_a,
        metric_fn_b=rho_b,
        sensitivity_clusters=shared_proteins,
        n_resamples=n_resamples,
        seed=seed,
        discard_reason=(
            "at least one arm had a fold with undefined or non-finite rank "
            "correlation on the shared family resample"
        ),
    )
    result["domain_resampled_sensitivity"] = result.pop("gene_resampled_sensitivity")
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


def _seed_gate_unavailability(ci):
    if ci is None:
        return "not adjudicated (CI unavailable)"
    requested_seeds = ci.get("requested_seeds", [])
    seed_std = ci.get("seed_std")
    if len(requested_seeds) < 3 or seed_std is None or not np.isfinite(seed_std):
        return "not adjudicated (complete multi-seed estimate unavailable)"
    return None


def _adjudicate_seed_lower_bound(point, ci, threshold):
    unavailable = _seed_gate_unavailability(ci)
    return unavailable or _adjudicate_lower_bound(point, ci, threshold)


def _adjudicate_seed_upper_bound(point, ci, threshold):
    unavailable = _seed_gate_unavailability(ci)
    return unavailable or _adjudicate_upper_bound(point, ci, threshold)


def _adjudicate_family_transfer_gap(point, ci):
    unavailable = _seed_gate_unavailability(ci)
    if unavailable:
        return unavailable
    if point is None or not np.isfinite(point):
        return "not adjudicated (point estimate unavailable)"
    if ci.get("ci_low") is None or ci.get("ci_high") is None:
        return "not adjudicated (CI unavailable)"
    if point <= FAMILY_TRANSFER_AFFIRMED_MAX_GAP:
        return (
            "affirmed"
            if ci["ci_high"] < FAMILY_TRANSFER_AFFIRMED_MAX_GAP
            else "not distinguishable"
        )
    if point >= FAMILY_TRANSFER_LEAKY_MIN_GAP:
        return (
            "failed" if ci["ci_low"] > FAMILY_TRANSFER_LEAKY_MIN_GAP else "underpowered"
        )
    return "not adjudicated (point estimate is between the two boundaries)"


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


def run_stability_projection(
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

    # The stability-projection control hinges on this removal; assert var along
    # stability_dir ≈ 0.
    var_before = float(np.var(merged_scaled.astype(np.float64) @ stability_dir))
    var_after = float(np.var(residuals.astype(np.float64) @ stability_dir))
    if var_after > 1e-6 * var_before + 1e-8:
        raise AssertionError(
            f"stability projection failed: var along stability_dir was {var_before:.3e} "
            f"before and {var_after:.3e} after projecting out — the removed direction "
            "leaked back in, so the projected arm still contains stability signal."
        )

    y = np.asarray(merged_labels)
    eligible_rows = np.flatnonzero(
        np.array(
            [pfam_map.get(gene) is not None for gene in merged_proteins], dtype=bool
        )
    )

    def _run_projection_seed(seed, collect_oof):
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
                label=f"stability_projection_{tag}",
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
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        seed_outputs = Parallel()(
            delayed(_run_projection_seed)(seed, compute_ci) for seed in requested_seeds
        )
    seed_results = [result for result, _oof in seed_outputs]
    seed_oofs = {result["seed"]: oof for result, oof in seed_outputs}
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

    difference_ci = None
    if compute_ci:

        def _combined_oof(tag):
            combined = aggregate_seed_oof(
                requested_seeds,
                [
                    make_seed_payload_record(
                        seed,
                        seed_oofs[seed].get(tag),
                        status=seed_results[seed]["results"][tag]["status"],
                    )
                    for seed in requested_seeds
                ],
                declared_row_ids=eligible_rows,
                declared_labels=y[eligible_rows],
                declared_clusters=np.asarray(merged_proteins)[eligible_rows],
                class_order=MECHANISM_CLASSES,
                declared_fold_ids=range(n_folds),
            )
            return combined.payload if combined.available else None

        baseline_oof = _combined_oof("baseline")
        projected_oof = _combined_oof("projected")
        if baseline_oof is not None and projected_oof is not None:
            if not np.array_equal(baseline_oof["genes"], projected_oof["genes"]):
                raise AssertionError(
                    "stability-projection baseline and projected OOF rows are not aligned"
                )
            clusters = np.array(
                [pfam_map[gene] for gene in baseline_oof["genes"]], dtype=object
            )

            def _fold_f1(oof):
                y_true = oof["y_true"]
                arms = [
                    (
                        np.array(
                            [MECHANISM_CLASSES[col] for col in proba.argmax(axis=1)]
                        ),
                        folds,
                        fold_ids,
                    )
                    for proba, folds, fold_ids in oof_score_arms(
                        oof, "stability projection"
                    )
                ]

                def _score(block, arm_pred):
                    return fold_macro_f1(y_true, block, arm_pred, MECHANISM_CLASSES)

                return lambda rows: score_within_folds(rows, arms, _score)

            difference_ci = paired_cluster_bootstrap_diff(
                clusters,
                _fold_f1(projected_oof),
                _fold_f1(baseline_oof),
                n_resamples=n_boot,
                seed=0,
                discard_reason="a fold's resampled rows lost every row",
            )
            difference_ci["seed_std"] = difference.spread
            difference_ci["requested_seeds"] = list(requested_seeds)
    inferential_point = (
        None if difference_ci is None else difference_ci.get("point_diff")
    )
    stability_projection_verdict = _adjudicate_upper_bound(
        inferential_point, difference_ci, STABILITY_PROJECTION_MAX_F1_CHANGE
    )
    return {
        **aggregate_result_contract(),
        "baseline_f1": baseline.to_dict(),
        "projected_f1": projected.to_dict(),
        "projected_minus_baseline_f1": difference.to_dict(),
        "per_seed_fold_summaries": seed_results,
        "inferential_seeds": list(requested_seeds),
        "inferential_point_estimate": inferential_point,
        "difference_ci": difference_ci,
        # None, not False, when the gate was not adjudicated: an unevaluated rule
        # is not a failed one, and a bare False here would read as a real failure.
        "projected_minus_baseline_mechanism_f1_passes": (
            None
            if stability_projection_verdict.startswith("not adjudicated")
            else stability_projection_verdict == "affirmed"
        ),
        "projected_minus_baseline_mechanism_f1_verdict": stability_projection_verdict,
    }


def apply_decision_rule(
    random_split_ci, family_transfer_gap_ci, stability_projection, per_protein_spread_ci
):
    """Adjudicate the four stability controls from matched point/CI pairs.

    Each gate reads the point estimate the interval beside it was built around, so
    a verdict never compares an interval with a different estimand.
    """
    random_split_point = (
        None if random_split_ci is None else random_split_ci.get("point")
    )
    family_transfer_point = (
        None
        if family_transfer_gap_ci is None
        else family_transfer_gap_ci.get("point_diff")
    )
    projection_point = (
        None
        if stability_projection is None
        else stability_projection.get("inferential_point_estimate")
    )
    projection_ci = (
        None
        if stability_projection is None
        else stability_projection.get("difference_ci")
    )
    per_protein_point = (
        None if per_protein_spread_ci is None else per_protein_spread_ci.get("point")
    )

    gates = {
        "random_split_spearman": {
            "criterion": "random_split_spearman_at_least_0.50",
            "threshold": RANDOM_SPLIT_MIN_SPEARMAN,
            "point_estimate": random_split_point,
            "ci": random_split_ci,
            "verdict": _adjudicate_seed_lower_bound(
                random_split_point, random_split_ci, RANDOM_SPLIT_MIN_SPEARMAN
            ),
        },
        "random_minus_family_spearman_gap": {
            "criterion": "random_minus_family_spearman_gap_two_boundary_rule",
            "affirmed_maximum": FAMILY_TRANSFER_AFFIRMED_MAX_GAP,
            "leaky_minimum": FAMILY_TRANSFER_LEAKY_MIN_GAP,
            "point_estimate": family_transfer_point,
            "ci": family_transfer_gap_ci,
            "verdict": _adjudicate_family_transfer_gap(
                family_transfer_point, family_transfer_gap_ci
            ),
        },
        "projected_minus_baseline_mechanism_f1": {
            "criterion": "projected_minus_baseline_mechanism_f1_at_most_0.01",
            "threshold": STABILITY_PROJECTION_MAX_F1_CHANGE,
            "point_estimate": projection_point,
            "ci": projection_ci,
            "verdict": _adjudicate_seed_upper_bound(
                projection_point, projection_ci, STABILITY_PROJECTION_MAX_F1_CHANGE
            ),
        },
        "per_protein_spearman_spread": {
            "criterion": "per_protein_spearman_std_at_most_0.10",
            "threshold": PER_PROTEIN_MAX_SPEARMAN_SPREAD,
            "point_estimate": per_protein_point,
            "ci": per_protein_spread_ci,
            "verdict": _adjudicate_upper_bound(
                per_protein_point,
                per_protein_spread_ci,
                PER_PROTEIN_MAX_SPEARMAN_SPREAD,
            ),
        },
    }

    if gates["random_split_spearman"]["verdict"] == "failed":
        overall = "random_split_spearman FAILED"
    elif gates["random_minus_family_spearman_gap"]["verdict"] == "failed":
        overall = "LEAKY"
    elif gates["projected_minus_baseline_mechanism_f1"]["verdict"] == "failed":
        overall = "projected_minus_baseline_mechanism_f1 FAILED"
    elif gates["per_protein_spearman_spread"]["verdict"] == "failed":
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

    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        seed_outputs = Parallel()(
            delayed(_run_seed)(seed, True) for seed in requested_seeds
        )
    results_by_seed = [result for result, _oofs in seed_outputs]
    oofs_by_seed = {result["seed"]: oofs for result, oofs in seed_outputs}

    combined_oofs = {}
    bootstrap_intervals = {}
    if compute_ci:
        print("\nCluster-bootstrap CIs on all seeds' fold-mean Spearman...")
        all_rows = np.arange(len(variants))
        family_rows = np.flatnonzero(
            np.array([family_map.get(protein) is not None for protein in proteins])
        )
        for key in sorted(results_by_seed[0]["results"]):
            declared_rows = family_rows if key.endswith("_family") else all_rows
            combined_oofs[key] = _combine_regression_oof(
                requested_seeds,
                {seed: oofs_by_seed[seed][key] for seed in requested_seeds},
                declared_rows,
            )
            bootstrap_intervals[key] = spearman_cluster_bootstrap_ci(
                combined_oofs[key], n_resamples=n_boot, seed=0
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
            lambda seed_result, key=key: seed_result["results"][key].get("auroc_mean"),
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
        if key in bootstrap_intervals:
            bootstrap_intervals[key]["seed_std"] = spearman.spread
            bootstrap_intervals[key]["requested_seeds"] = list(requested_seeds)
            summary[key]["bootstrap_ci"] = bootstrap_intervals[key]

    summary["per_protein"] = {
        "spearman_protein_mean": per_prot_mean,
        "spearman_protein_std": per_prot_std,
        "sampling_unit": "protein",
        "n_proteins": len(prot_rhos),
        "n_proteins_finite": n_finite_prot,
    }
    per_protein_spread_ci = None
    if compute_ci:
        per_protein_spread_ci = per_protein_std_bootstrap_ci(
            per_prot, n_resamples=n_boot, seed=0
        )
        summary["per_protein"]["spearman_std_ci"] = per_protein_spread_ci

    required_3c_paths = [VALID_VARIANTS_JSON, EMB_WT_MEAN, EMB_MUT_MEAN, PFAM_JSON]
    missing_3c_paths = [
        str(path) for path in required_3c_paths if not os.path.exists(path)
    ]
    if missing_3c_paths:
        raise FileNotFoundError(
            "the stability-projection control requires missing input files: "
            + ", ".join(missing_3c_paths)
        )
    print("\nRunning the stability-projection control...")
    pfam_map = load_pfam_map(PFAM_JSON)

    merged_delta, merged_labels, merged_proteins = load_merged()
    with open(VALID_VARIANTS_JSON) as handle:
        mechanism_variants = json.load(handle)
    mechanism_projection_fingerprints = {
        "labeled_variants": labeled_variant_fingerprint(
            mechanism_variants, merged_labels
        ),
        "delta_mean_content": embedding_fingerprint(merged_delta),
        "pfam_assignments": pfam_fingerprint(pfam_map, merged_proteins.tolist()),
    }

    stability_projection_result = run_stability_projection(
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
    projection_baseline = read_seed_point_estimate(
        stability_projection_result["baseline_f1"]
    ).value
    projection_projected = read_seed_point_estimate(
        stability_projection_result["projected_f1"]
    ).value
    projection_difference = read_seed_point_estimate(
        stability_projection_result["projected_minus_baseline_f1"]
    ).value
    projection_inferential = stability_projection_result["inferential_point_estimate"]
    print(
        f"  stability projection: baseline F1={_show_seed_value(projection_baseline)}  "
        f"projected F1={_show_seed_value(projection_projected)}  "
        f"paired mean Δ={_show_seed_value(projection_difference, signed=True)}  "
        f"bootstrap point Δ={_show_seed_value(projection_inferential, signed=True)}  "
        f"verdict={stability_projection_result['projected_minus_baseline_mechanism_f1_verdict']}"
    )
    write_result_json(
        os.path.join(OUT, "stability_projection.json"),
        {
            **stability_projection_result,
            "input_fingerprints": {
                "stability": stability_fingerprints,
                "mechanism_projection": mechanism_projection_fingerprints,
            },
            "analysis_parameters": analysis_parameters,
        },
        seeds=list(requested_seeds),
    )

    random_records = []
    family_records = []
    for seed in requested_seeds:
        rho_random, rho_family, shared_proteins, _shared_families = (
            _paired_spearman_gap_inputs(
                oofs_by_seed[seed]["delta_mean_random"],
                oofs_by_seed[seed]["delta_mean_family"],
                proteins,
                family_map,
            )
        )
        shared_rows = np.arange(len(shared_proteins))
        random_records.append(make_seed_record(seed, rho_random(shared_rows)))
        family_records.append(make_seed_record(seed, rho_family(shared_rows)))
    family_transfer_gap = aggregate_paired_seed_difference(
        requested_seeds, random_records, family_records
    )
    summary["random_minus_family_spearman_gap_seed_aggregate"] = (
        family_transfer_gap.to_dict()
    )

    family_transfer_gap_ci = None
    if compute_ci:
        print("\nComputing paired CI on the random-to-family Spearman gap...")
        family_transfer_gap_ci = paired_spearman_gap_ci(
            combined_oofs["delta_mean_random"],
            combined_oofs["delta_mean_family"],
            proteins,
            family_map,
            n_resamples=n_boot,
            seed=0,
        )
        family_transfer_gap_ci["seed_std"] = family_transfer_gap.spread
        family_transfer_gap_ci["requested_seeds"] = list(requested_seeds)
        summary["random_minus_family_spearman_gap_ci"] = family_transfer_gap_ci
        print(
            f"  family-transfer gap: {family_transfer_gap_ci['point_diff']:.3f} "
            f"[{family_transfer_gap_ci.get('ci_low', '?')}, "
            f"{family_transfer_gap_ci.get('ci_high', '?')}]"
        )

    dm_random = read_seed_point_estimate(
        summary["delta_mean_random"]["across_seed"]["spearman"]
    ).value
    dm_family = read_seed_point_estimate(
        summary["delta_mean_family"]["across_seed"]["spearman"]
    ).value

    random_split_ci = summary.get("delta_mean_random", {}).get("bootstrap_ci")
    adjudication = apply_decision_rule(
        random_split_ci,
        family_transfer_gap_ci,
        stability_projection_result,
        per_protein_spread_ci,
    )
    verdict = adjudication["overall"]
    summary["result_version"] = 3
    summary["verdict"] = verdict
    summary["gates"] = adjudication["gates"]
    summary["n_variants"] = len(variants)
    summary["n_proteins"] = len(set(proteins))
    summary["n_families"] = n_families
    summary["n_seeds"] = n_seeds
    summary["projected_minus_baseline_mechanism_f1"] = stability_projection_result
    summary["input_fingerprints"] = {
        "stability": stability_fingerprints,
        "mechanism_projection": mechanism_projection_fingerprints,
    }
    summary["analysis_parameters"] = analysis_parameters

    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(
        f"  delta_mean random ρ  : {_show_seed_value(dm_random)}  "
        f"(random-split threshold ≥ {RANDOM_SPLIT_MIN_SPEARMAN:.2f})"
    )
    print(f"  delta_mean family ρ  : {_show_seed_value(dm_family)}")
    print(
        "  Δ (random − family)  : "
        f"{_show_seed_value(family_transfer_gap.mean, signed=True)}  "
        f"(affirmed if Δ ≤ {FAMILY_TRANSFER_AFFIRMED_MAX_GAP:.2f}; "
        f"LEAKY if Δ ≥ {FAMILY_TRANSFER_LEAKY_MIN_GAP:.2f})"
    )
    print(
        f"  per-protein ρ std    : {_show_seed_value(per_prot_std)}  "
        f"(per-protein spread threshold ≤ {PER_PROTEIN_MAX_SPEARMAN_SPREAD:.2f})"
    )
    if stability_projection_result:
        print(
            f"  stability projection, paired-seed Δ mechanism F1: "
            f"{_show_seed_value(projection_difference, signed=True)}  "
            f"(all-seed bootstrap Δ "
            f"{_show_seed_value(projection_inferential, signed=True)})  "
            f"(passes if ≤ +{STABILITY_PROJECTION_MAX_F1_CHANGE:.2f})"
        )
    for gate_name, gate in adjudication["gates"].items():
        print(f"  {gate_name} verdict          : {gate['verdict']}")
    print(f"{'='*60}")

    write_result_json(
        os.path.join(OUT, "summary.json"), summary, seeds=list(requested_seeds)
    )
    print(f"\nResults written to {OUT}/")


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=seed_count, default=N_SEEDS)
    parser.add_argument(
        "--no_ci", action="store_true", help="skip cluster-bootstrap CIs"
    )
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    parser.add_argument(
        "--n_jobs",
        type=int,
        required=True,
        help="Max concurrent worker processes for the per-seed, per-protein, and "
        "the stability-projection parallel loops. Each worker standardizes/fits "
        "against most of the "
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
