"""Enzyme type classification (kinase/protease/oxidoreductase/non-enzyme) from
ESM-2 WT mean-pooled embeddings, as a positive control for the mechanism arc.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

from esm2_mech.utils.constants import (
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_N_RESAMPLES,
    N_SEEDS,
    ENZYME_MECHANISM_MIN_F1_GAP,
    MECHANISM_CLASSES,
    MECHANISM_OOF_CACHE_SCHEMA_VERSION,
    mechanism_oof_cache_filename,
    seed_result_filename,
)
from esm2_mech.utils.metrics import fold_macro_f1, majority_baseline_f1
from esm2_mech.utils.splits import gene_split_cv, family_split_cv
from esm2_mech.utils.data import (
    build_gene_to_row,
    embedding_fingerprint,
    load_pfam_map,
    observed_rows_mask,
    pfam_fingerprint,
    validate_embedding_variant_identity,
)
from esm2_mech.utils.io import write_result_json
from esm2_mech.utils.probes import run_logreg_cv, run_mlp_cv
from esm2_mech.utils.classification import validate_complete_classification_splits
from esm2_mech.utils.bootstrap import (
    attach_mechanism_ci,
    folds_to_arms,
    score_within_folds,
    adjudicate_equivalence,
    adjudicate_diff,
    adjudicate_level,
    family_or_gene_clusters,
    oof_permutation_pvalue,
    paired_oof_diff,
)
from esm2_mech.utils.paths import (
    EMB_WT_MEAN,
    EMB_VALID_VARIANTS_JSON,
    ENZYME_CLASSIFICATION_JSON,
    ENZYME_LABELS_TSV,
    ENZYME_RESULTS_DIR,
    GENE_UNIVERSE,
    MECHANISM_AGGREGATE_JSON,
    PFAM_JSON,
    PROTEOME_FEATURES_ALIGNED,
    PROTEOME_FEATURE_COLUMNS_JSON,
    RESULTS_DIR,
    VALID_VARIANTS_JSON,
)

print = functools.partial(print, flush=True)

ENZYME_CLASSES = ["kinase", "protease", "oxidoreductase", "non-enzyme"]


def _canonical_fingerprint(value) -> str:
    """Hash a JSON-compatible scientific input with deterministic ordering."""
    content = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(content.encode()).hexdigest()


def _load_mechanism_reference_f1() -> float | None:
    """Read the mechanism family-split F1 from the mechanism aggregate result."""
    if not MECHANISM_AGGREGATE_JSON.exists():
        print(f"  WARNING: {MECHANISM_AGGREGATE_JSON} not found — mechanism reference unavailable")
        return None
    result_path = RESULTS_DIR / seed_result_filename(0)
    if not result_path.exists():
        raise FileNotFoundError(
            f"{result_path} is missing; the mechanism aggregate cannot be validated"
        )
    with open(MECHANISM_AGGREGATE_JSON) as fh:
        agg = json.load(fh)
    with open(result_path) as fh:
        result = json.load(fh)
    for key in ("input_fingerprints", "analysis_parameters"):
        if result.get(key) is None or agg.get(key) != result.get(key):
            raise ValueError(
                f"{MECHANISM_AGGREGATE_JSON}: {key} does not match {result_path}"
            )
    val = (
        agg.get("across_seed", {})
        .get("family_split", {})
        .get("delta_mean", {})
        .get("macro_f1_seed_mean")
    )
    if val is None:
        print("  WARNING: macro_f1_seed_mean not found in mechanism aggregate — reference unavailable")
    return val


def _load_mechanism_family_oof() -> dict | None:
    """Load the OOF cache bound to the exact completed seed-0 mechanism run."""
    cache_path = RESULTS_DIR / mechanism_oof_cache_filename(0)
    result_path = RESULTS_DIR / seed_result_filename(0)
    if not cache_path.exists() and not result_path.exists():
        print(
            "  WARNING: seed-0 mechanism result and OOF cache not found — "
            "enzyme/mechanism difference CI unavailable"
        )
        return None
    if not cache_path.exists() or not result_path.exists():
        missing = cache_path if not cache_path.exists() else result_path
        raise FileNotFoundError(
            f"{missing} is missing; the seed-0 result and OOF cache must be "
            "regenerated together"
        )

    with open(cache_path) as handle:
        cache = json.load(handle)
    with open(result_path) as handle:
        result = json.load(handle)

    expected = {
        "cache_schema_version": MECHANISM_OOF_CACHE_SCHEMA_VERSION,
        "seed": 0,
        "analysis_run_id": result.get("analysis_run_id"),
        "input_fingerprints": result.get("input_fingerprints"),
        "analysis_parameters": result.get("analysis_parameters"),
    }
    for key, expected_value in expected.items():
        if expected_value is None or cache.get(key) != expected_value:
            raise ValueError(
                f"{cache_path}: cache {key} does not match {result_path}"
            )

    oof = cache.get("features", {}).get("delta_mean", {}).get("family_split")
    required = {"row_ids", "y_true", "pred", "genes", "folds"}
    if oof is None or not required.issubset(oof):
        raise ValueError(
            f"{cache_path} lacks delta_mean family-split OOF fields "
            f"{sorted(required)}"
        )
    lengths = {key: len(oof[key]) for key in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"{cache_path} has misaligned mechanism OOF fields: {lengths}"
        )
    if len(set(int(row) for row in oof["row_ids"])) != lengths["row_ids"]:
        raise ValueError(f"{cache_path} has duplicate mechanism OOF row ids")
    return oof


def _mechanism_reference_fingerprints() -> dict:
    """Fingerprint the validated mechanism result and OOF values used by 2G."""
    aggregate_path = MECHANISM_AGGREGATE_JSON
    result_path = RESULTS_DIR / seed_result_filename(0)
    cache_path = RESULTS_DIR / mechanism_oof_cache_filename(0)
    missing = [
        str(path)
        for path in (aggregate_path, result_path, cache_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "enzyme claim 2G requires missing mechanism reference files: "
            + ", ".join(missing)
        )
    with open(aggregate_path) as handle:
        aggregate = json.load(handle)
    with open(result_path) as handle:
        result = json.load(handle)
    with open(cache_path) as handle:
        cache = json.load(handle)
    for key in ("input_fingerprints", "analysis_parameters"):
        if result.get(key) is None or aggregate.get(key) != result.get(key):
            raise ValueError(
                f"{aggregate_path}: {key} does not match {result_path}"
            )
    expected_cache = {
        "cache_schema_version": MECHANISM_OOF_CACHE_SCHEMA_VERSION,
        "seed": 0,
        "analysis_run_id": result.get("analysis_run_id"),
        "input_fingerprints": result.get("input_fingerprints"),
        "analysis_parameters": result.get("analysis_parameters"),
    }
    for key, expected_value in expected_cache.items():
        if expected_value is None or cache.get(key) != expected_value:
            raise ValueError(
                f"{cache_path}: cache {key} does not match {result_path}"
            )
    reference = {
        "analysis_run_id": result.get("analysis_run_id"),
        "input_fingerprints": result.get("input_fingerprints"),
        "analysis_parameters": result.get("analysis_parameters"),
        "aggregate_family_delta_mean": aggregate.get("across_seed", {})
        .get("family_split", {})
        .get("delta_mean"),
        "seed0_family_delta_mean_oof": cache.get("features", {})
        .get("delta_mean", {})
        .get("family_split"),
    }
    missing_values = [key for key, value in reference.items() if value is None]
    if missing_values:
        raise ValueError(
            "mechanism reference lacks required scientific values "
            f"{missing_values}"
        )
    return {
        "content": _canonical_fingerprint(reference),
        "analysis_run_id": reference["analysis_run_id"],
        "input_fingerprints": reference["input_fingerprints"],
        "analysis_parameters": reference["analysis_parameters"],
    }


def load_gene_embeddings() -> tuple:
    """Load per-gene WT embeddings by taking the first variant's embedding for each gene."""
    with open(VALID_VARIANTS_JSON) as fh:
        variants = json.load(fh)

    validate_embedding_variant_identity(variants, EMB_VALID_VARIANTS_JSON)
    wt_mean = np.load(EMB_WT_MEAN)
    if len(variants) != wt_mean.shape[0]:
        raise ValueError(
            f"{VALID_VARIANTS_JSON} has {len(variants)} variants but "
            f"{EMB_WT_MEAN} has {wt_mean.shape[0]} rows — files are out of sync."
        )
    print(f"Loaded {len(variants)} variants, wt_mean shape: {wt_mean.shape}")

    gene_first_idx = {}
    gene_uniprot = {}
    for i, v in enumerate(variants):
        g = v["gene"]
        if g not in gene_first_idx:
            gene_first_idx[g] = i
            gene_uniprot[g] = v.get("uniprot_id", "")

    gene_list = list(gene_first_idx.keys())
    idxs = [gene_first_idx[g] for g in gene_list]
    X = wt_mean[idxs].astype(np.float32)

    print(f"Per-gene embeddings: {X.shape} ({len(gene_list)} genes)")
    return X, gene_list, [gene_uniprot[g] for g in gene_list]


def load_enzyme_labels() -> dict:
    """Load gene -> enzyme_4class from enzyme_labels.tsv."""
    import csv

    labels: dict[str, str] = {}
    excluded = 0
    with open(ENZYME_LABELS_TSV, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = {
            "gene",
            "enzyme_4class",
            "uniprot_missing_flag",
            "enzyme_4class_excluded_flag",
        }
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"{ENZYME_LABELS_TSV} lacks columns {sorted(missing_columns)}; "
                "rerun the enzyme annotation fetch"
            )
        for row in reader:
            gene = row["gene"]
            enzyme_class = row["enzyme_4class"]
            if enzyme_class:
                if enzyme_class not in ENZYME_CLASSES:
                    raise ValueError(
                        f"{ENZYME_LABELS_TSV} has unsupported enzyme class "
                        f"{enzyme_class!r} for {gene}"
                    )
                if gene in labels:
                    raise ValueError(f"{ENZYME_LABELS_TSV} contains duplicate gene {gene}")
                labels[gene] = enzyme_class
                continue
            if (
                row["uniprot_missing_flag"] == "1"
                or row["enzyme_4class_excluded_flag"] == "1"
            ):
                excluded += 1
                continue
            raise ValueError(
                f"{ENZYME_LABELS_TSV} has a blank enzyme class for {gene} "
                "without a missing or exclusion flag"
            )
    print(f"Loaded enzyme labels for {len(labels)} genes; excluded {excluded} unlabeled genes")
    return labels


def load_proteome_features() -> tuple:
    """Load the proteome feature matrix aligned to gene_universe.tsv row order."""
    X = np.load(PROTEOME_FEATURES_ALIGNED).astype(np.float32)
    with open(PROTEOME_FEATURE_COLUMNS_JSON) as fh:
        cols = json.load(fh)
    genes = list(build_gene_to_row(GENE_UNIVERSE))
    if len(genes) != X.shape[0]:
        raise ValueError(
            f"{PROTEOME_FEATURES_ALIGNED} has {X.shape[0]} rows but "
            f"{GENE_UNIVERSE} lists {len(genes)} genes — not row-aligned."
        )
    print(f"Proteome features: {X.shape}, {len(genes)} genes")
    return X, genes, cols


def enzyme_input_fingerprints(
    X_emb,
    genes,
    uniprot_ids,
    labels,
    pfam_map,
    X_proteome,
    proteome_genes,
    proteome_labels,
    proteome_columns,
    mechanism_reference,
) -> dict:
    """Fingerprint every scientific input used by the enzyme controls."""
    lengths = {
        "embedding rows": len(X_emb),
        "genes": len(genes),
        "UniProt ids": len(uniprot_ids),
        "labels": len(labels),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"enzyme embedding inputs are misaligned: {lengths}")
    proteome_lengths = {
        "feature rows": len(X_proteome),
        "genes": len(proteome_genes),
        "labels": len(proteome_labels),
    }
    if len(set(proteome_lengths.values())) != 1:
        raise ValueError(f"enzyme proteome inputs are misaligned: {proteome_lengths}")
    labeled_genes = [
        [gene, uniprot_id, str(label)]
        for gene, uniprot_id, label in zip(genes, uniprot_ids, labels)
    ]
    proteome_cohort = [
        [gene, str(label)]
        for gene, label in zip(proteome_genes, proteome_labels)
    ]
    return {
        "enzyme_labeled_genes": _canonical_fingerprint(labeled_genes),
        "wt_embedding_content": embedding_fingerprint(X_emb),
        "pfam_assignments": pfam_fingerprint(pfam_map, genes),
        "proteome_labeled_genes": _canonical_fingerprint(proteome_cohort),
        "proteome_feature_content": embedding_fingerprint(X_proteome),
        "proteome_feature_columns": _canonical_fingerprint(proteome_columns),
        "mechanism_reference": mechanism_reference,
    }


def run_multiseed(
    X: np.ndarray,
    y: np.ndarray,
    genes: list[str],
    pfam_map: dict,
    le: LabelEncoder,
    seeds: list[int],
    n_folds: int = 5,
    compute_ci: bool = True,
    n_boot: int = BOOTSTRAP_N_RESAMPLES,
    n_permutations: int = 0,
    mechanism_family_oof: dict | None = None,
) -> dict:
    """Run shared linear and nonlinear probes across seeds.

    `y` contains the string class names listed by `le.classes_`.
    """
    classes = list(le.classes_)
    unknown_labels = sorted(set(y.tolist()) - set(classes), key=str)
    if unknown_labels:
        raise ValueError(
            f"run_multiseed received labels outside the enzyme classes: {unknown_labels}"
        )
    genes_arr = np.array(genes)
    print(f"\nClasses: {classes}")
    print(f"Class distribution: {dict(Counter(y.tolist()))}")

    gs_f1s, fs_f1s, mlp_f1s = [], [], []
    gs_reference_f1s, fs_reference_f1s = [], []
    gs_aurocs = {c: [] for c in classes}
    fs_aurocs = {c: [] for c in classes}
    mlp_aurocs = {c: [] for c in classes}

    seed0_fs_oof = None
    seed0_mlp_fs_oof = None

    for seed in seeds:
        print(f"\n  Seed {seed}:")

        gs_splits = gene_split_cv(genes_arr, n_folds=n_folds, seed=seed)
        gs_contract = validate_complete_classification_splits(
            gs_splits, requested_folds=n_folds,
            eligible_rows=np.concatenate([test for _train, test in gs_splits]),
            labels=y, classes=classes, groups=genes_arr, held_out_unit="gene",
        )
        gs, _ = run_logreg_cv(
            X,
            y,
            gs_splits,
            classes,
            gs_contract,
            genes=genes_arr,
            seed=seed,
            label="enzyme logreg gene",
            return_oof=True,
        )
        try:
            gs_reference = (
                float(np.mean([
                    majority_baseline_f1(y[train], y[test], classes)[0]
                    for train, test in gs_splits
                ]))
                if gs_contract["status"] == "valid"
                else None
            )
        except ValueError:
            gs_reference = None
        gs_reference_f1s.append(gs_reference)
        gs_f1s.append(gs["macro_f1_mean"])
        for c in classes:
            v = gs.get(f"auroc_{c}_mean")
            gs_aurocs[c].append(v)
        print(
            f"    LogReg gene-split  F1={gs['macro_f1_mean']:.3f}"
            if gs["status"] == "success"
            else f"    LogReg gene-split  {gs['status']}"
        )

        fs_splits = family_split_cv(genes_arr, pfam_map, n_folds=n_folds, seed=seed)
        family_groups = family_or_gene_clusters(
            genes_arr, pfam_map, is_family_split=True
        )
        fs_contract = validate_complete_classification_splits(
            fs_splits, requested_folds=n_folds,
            eligible_rows=np.concatenate([test for _train, test in fs_splits]),
            labels=y, classes=classes, groups=family_groups, held_out_unit="family",
        )
        fs, fs_oof = run_logreg_cv(
            X,
            y,
            fs_splits,
            classes,
            fs_contract,
            genes=genes_arr,
            seed=seed,
            label="enzyme logreg family",
            return_oof=True,
        )
        try:
            fs_reference = (
                float(np.mean([
                    majority_baseline_f1(y[train], y[test], classes)[0]
                    for train, test in fs_splits
                ]))
                if fs_contract["status"] == "valid"
                else None
            )
        except ValueError:
            fs_reference = None
        fs_reference_f1s.append(fs_reference)
        fs_f1s.append(fs["macro_f1_mean"])
        for c in classes:
            v = fs.get(f"auroc_{c}_mean")
            fs_aurocs[c].append(v)
        if fs["status"] == "success":
            print(
                f"    LogReg family-split F1={fs['macro_f1_mean']:.3f}  AUROC: "
                + " ".join(f"{c}={fs[f'auroc_{c}_mean']:.3f}" for c in classes)
            )
        else:
            print(f"    LogReg family-split {fs['status']}")

        if seed == seeds[0]:
            seed0_fs_oof = fs_oof

        mlp, mlp_oof = run_mlp_cv(
            X,
            y,
            fs_splits,
            classes,
            fs_contract,
            hidden=(256, 64),
            genes=genes_arr,
            seed=seed,
            label="enzyme mlp family",
            return_oof=True,
            max_iter=300,
            activation="relu",
            alpha=1e-3,
            validation_fraction=0.1,
            n_iter_no_change=15,
            oversample=False,
            balanced_sample_weight=True,
        )
        mlp_f1s.append(mlp["macro_f1_mean"])
        for c in classes:
            v = mlp.get(f"auroc_{c}_mean")
            mlp_aurocs[c].append(v)
        print(
            f"    MLP    family-split F1={mlp['macro_f1_mean']:.3f}"
            if mlp["status"] == "success"
            else f"    MLP    family-split {mlp['status']}"
        )

        if seed == seeds[0]:
            seed0_mlp_fs_oof = mlp_oof

    def _agg(vals):
        if len(vals) != len(seeds) or any(value is None for value in vals):
            return None, None
        arr = np.asarray(vals, dtype=float)
        return float(np.mean(arr)), float(np.std(arr))

    gs_mean, gs_std = _agg(gs_f1s)
    fs_mean, fs_std = _agg(fs_f1s)
    mlp_mean, mlp_std = _agg(mlp_f1s)
    gs_reference_mean, gs_reference_std = _agg(gs_reference_f1s)
    fs_reference_mean, fs_reference_std = _agg(fs_reference_f1s)

    leakage_pct = None
    if (
        gs_mean is not None
        and fs_mean is not None
        and gs_reference_mean is not None
        and gs_mean > gs_reference_mean
    ):
        leakage_pct = round(
            100.0 * (gs_mean - fs_mean) / (gs_mean - gs_reference_mean), 1
        )

    print(f"\n  Results ({len(seeds)} seeds):")
    for summary_label, mean, std in (
        ("Gene-split reference", gs_reference_mean, gs_reference_std),
        ("Family-split reference", fs_reference_mean, fs_reference_std),
        ("LogReg gene-split", gs_mean, gs_std),
        ("LogReg family-split", fs_mean, fs_std),
        ("MLP family-split", mlp_mean, mlp_std),
    ):
        if mean is None:
            print(f"    {summary_label}: Unscorable")
        else:
            print(f"    {summary_label}: F1={mean:.3f} +/- {std:.3f}")
    if leakage_pct is not None:
        print(f"    Leakage fraction:        {leakage_pct:.1f}%")

    ci_result = None
    permutation_result = None
    oof_fs_f1 = None
    oof_mlp_f1 = None
    paired_mlp_vs_logreg = None
    paired_logreg_vs_mechanism = None

    def _oof_macro_f1(oof):
        """Seed-0 out-of-fold macro-F1, scored per fold and averaged."""
        y_str = np.asarray(oof["y_true"])
        pred = np.array([classes[col] for col in oof["proba"].argmax(axis=1)])
        arms = folds_to_arms(pred, oof["folds"])

        def _fold_f1(block, arm_pred):
            return fold_macro_f1(y_str, block, arm_pred, classes)
        return y_str, score_within_folds(np.arange(len(y_str)), arms, _fold_f1)

    if seed0_fs_oof is not None:
        oof_y_str, oof_fs_f1 = _oof_macro_f1(seed0_fs_oof)
        print(f"\n  Seed-0 OOF LogReg family-split F1: {oof_fs_f1:.3f}")

    if seed0_mlp_fs_oof is not None:
        mlp_oof_y_str, oof_mlp_f1 = _oof_macro_f1(seed0_mlp_fs_oof)
        print(f"  Seed-0 OOF MLP family-split F1: {oof_mlp_f1:.3f}")

    if compute_ci and seed0_fs_oof is not None:
        print("\n  Computing cluster-bootstrap CIs (seed 0, family-split)...")
        clusters = family_or_gene_clusters(
            seed0_fs_oof["genes"], pfam_map, is_family_split=True
        )
        ci_container: dict = {}
        attach_mechanism_ci(
            ci_container,
            {**seed0_fs_oof, "y_true": oof_y_str},
            clusters,
            compute_ci=True,
            classes=classes,
            n_resamples=n_boot,
            ci_level=BOOTSTRAP_CI_LEVEL,
            seed=0,
        )
        ci_result = ci_container["ci"]
        for metric_name, ci in ci_result.items():
            lo = ci.get("ci_lower")
            hi = ci.get("ci_upper")
            pt = ci.get("point")
            if lo is not None and hi is not None and pt is not None:
                print(f"    {metric_name}: {pt:.3f} [{lo:.3f}, {hi:.3f}]")

        if seed0_mlp_fs_oof is not None:
            print("\n  Computing paired CI: MLP minus LogReg (family-split)...")
            mlp_oof_for_diff = {**seed0_mlp_fs_oof, "y_true": mlp_oof_y_str}
            lr_oof_for_diff = {**seed0_fs_oof, "y_true": oof_y_str}
            paired_mlp_vs_logreg = paired_oof_diff(
                oof_a=mlp_oof_for_diff,
                oof_b=lr_oof_for_diff,
                pfam_map=pfam_map,
                label="2H MLP-LogReg",
                classes=classes,
                metric="macro_f1",
                is_family_split=True,
                n_resamples=n_boot,
                seed=0,
            )
            if paired_mlp_vs_logreg is not None:
                lo = paired_mlp_vs_logreg.get("ci_low")
                hi = paired_mlp_vs_logreg.get("ci_high")
                pt = paired_mlp_vs_logreg.get("point_diff")
                if lo is not None and hi is not None and pt is not None:
                    print(f"    MLP-LogReg diff: {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]")

        if mechanism_family_oof is not None:
            print("\n  Computing paired CI: enzyme LogReg minus mechanism...")
            enzyme_pred = np.array([classes[col] for col in seed0_fs_oof["proba"].argmax(axis=1)])
            mechanism_y = np.asarray(mechanism_family_oof["y_true"])
            mechanism_pred = np.asarray(mechanism_family_oof["pred"])
            enzyme_arms = folds_to_arms(enzyme_pred, seed0_fs_oof["folds"])
            mechanism_arms = folds_to_arms(
                mechanism_pred, mechanism_family_oof["folds"]
            )
            def _enzyme_f1(rows):
                return score_within_folds(
                    rows,
                    enzyme_arms,
                    lambda block, arm_pred: fold_macro_f1(
                        oof_y_str, block, arm_pred, classes
                    ),
                )

            def _mechanism_f1(rows):
                return score_within_folds(
                    rows,
                    mechanism_arms,
                    lambda block, arm_pred: fold_macro_f1(
                        mechanism_y, block, arm_pred, MECHANISM_CLASSES
                    ),
                )

            enzyme_point = _enzyme_f1(np.arange(len(oof_y_str)))
            mechanism_point = _mechanism_f1(np.arange(len(mechanism_y)))
            point_difference = enzyme_point - mechanism_point
            paired_logreg_vs_mechanism = {
                "point_a": enzyme_point,
                "point_b": mechanism_point,
                "point_diff": point_difference,
                "point": point_difference,
                "ci_low": None,
                "ci_high": None,
                "ci_suppressed": True,
                "missing": True,
                "reason": "blocked_by_audit_1_4",
            }
            lo = paired_logreg_vs_mechanism.get("ci_low")
            hi = paired_logreg_vs_mechanism.get("ci_high")
            point = paired_logreg_vs_mechanism.get("point_diff")
            if lo is not None and hi is not None and point is not None:
                print(f"    enzyme-mechanism diff: {point:+.3f} [{lo:+.3f}, {hi:+.3f}]")

        if n_permutations > 0:
            print(f"\n  Computing permutation p-value ({n_permutations} reps)...")
            permutation_result = oof_permutation_pvalue(
                y_true=oof_y_str,
                proba=seed0_fs_oof["proba"],
                folds=seed0_fs_oof["folds"],
                groups=seed0_fs_oof["genes"],
                clusters=clusters,
                classes=classes,
                n_permutations=n_permutations,
                seed=0,
            )
            p_value_text = (
                f"unresolved at resolution {permutation_result['p_value_resolution']}"
                if permutation_result.get("resolution_limited")
                else str(permutation_result.get("p_value"))
            )
            immovable_text = (
                f"; {permutation_result['n_clusters_immovable']} immovable families"
                if permutation_result.get("n_clusters_immovable") is not None else ""
            )
            print(f"    permutation p-value: {p_value_text}{immovable_text}")

    result = {
        "majority_reference": {
            "gene_split": {
                "macro_f1_mean": gs_reference_mean,
                "macro_f1_std": gs_reference_std,
                "per_seed": gs_reference_f1s,
            },
            "family_split": {
                "macro_f1_mean": fs_reference_mean,
                "macro_f1_std": fs_reference_std,
                "per_seed": fs_reference_f1s,
            },
        },
        "logreg_gene_split": {
            "macro_f1_mean": gs_mean,
            "macro_f1_std": gs_std,
            "per_class_auroc_mean": {
                c: _agg(v)[0] for c, v in gs_aurocs.items()
            },
            "n_seeds": len(seeds),
        },
        "logreg_family_split": {
            "macro_f1_mean": fs_mean,
            "macro_f1_std": fs_std,
            "per_class_auroc_mean": {
                c: _agg(v)[0] for c, v in fs_aurocs.items()
            },
            "n_seeds": len(seeds),
        },
        "mlp_family_split": {
            "macro_f1_mean": mlp_mean,
            "macro_f1_std": mlp_std,
            "per_class_auroc_mean": {
                c: _agg(v)[0] for c, v in mlp_aurocs.items()
            },
            "n_seeds": len(seeds),
        },
        "leakage_pct": leakage_pct,
    }

    if oof_fs_f1 is not None:
        result["logreg_family_split"]["oof_macro_f1"] = oof_fs_f1
    if oof_mlp_f1 is not None:
        result["mlp_family_split"]["oof_macro_f1"] = oof_mlp_f1
    if ci_result is not None:
        result["bootstrap_ci"] = ci_result
    if paired_mlp_vs_logreg is not None:
        result["paired_ci_mlp_minus_logreg"] = paired_mlp_vs_logreg
    if paired_logreg_vs_mechanism is not None:
        result["paired_ci_logreg_minus_mechanism"] = paired_logreg_vs_mechanism
    if permutation_result is not None:
        result["permutation_test"] = permutation_result

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=N_SEEDS,
                        help="number of seeds to run; runs 0..seeds-1 (>=1)")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--no_ci", action="store_true", help="skip cluster-bootstrap CIs")
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    parser.add_argument(
        "--n_permutations", type=int, default=0,
        help="label-permutation reps for OOF macro AUROC (0 = skip)",
    )
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")

    seeds = list(range(args.seeds))
    compute_ci = not args.no_ci

    ENZYME_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Enzyme Classification from ESM-2 WT Embeddings ===")
    print(f"Seeds: {seeds}  Folds: {args.n_folds}  CI: {compute_ci}  n_boot: {args.n_boot}")

    mechanism_ref_f1 = _load_mechanism_reference_f1()
    mechanism_family_oof = _load_mechanism_family_oof()

    X_emb, gene_list, gene_uniprot_ids = load_gene_embeddings()
    enzyme_labels = load_enzyme_labels()
    pfam_map = load_pfam_map(PFAM_JSON)

    missing = [g for g in gene_list if g not in enzyme_labels]
    if missing:
        print(
            f"  Excluding {len(missing)} genes outside the four-class cohort: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    labeled_mask = np.array([g in enzyme_labels for g in gene_list])
    X_emb = X_emb[labeled_mask]
    gene_uniprot_ids = list(np.asarray(gene_uniprot_ids)[labeled_mask])
    gene_list = [g for g in gene_list if g in enzyme_labels]

    y_str = [enzyme_labels[g] for g in gene_list]
    le = LabelEncoder()
    le.fit(ENZYME_CLASSES)
    y = np.asarray(y_str)

    print(f"\nLabeled genes: {len(gene_list)}")
    print(f"Class distribution: {dict(Counter(y_str))}")
    print(
        f"Pfam-annotated genes: {sum(1 for g in gene_list if pfam_map.get(g))}/{len(gene_list)}"
    )

    print("\n" + "=" * 60)
    print("PART 1: ESM-2 WT embedding (1280-dim)")
    print("=" * 60)
    emb_results = run_multiseed(
        X_emb, y, gene_list, pfam_map, le, seeds=seeds, n_folds=args.n_folds,
        compute_ci=compute_ci, n_boot=args.n_boot, n_permutations=args.n_permutations,
        mechanism_family_oof=mechanism_family_oof,
    )

    print("\n" + "=" * 60)
    print("PART 2: Proteome features (37-dim) — baseline comparison")
    print("=" * 60)

    X_prot, prot_genes, proteome_columns = load_proteome_features()
    prot_gene_to_idx = {g: i for i, g in enumerate(prot_genes)}

    gene_to_emb_idx = {g: i for i, g in enumerate(gene_list)}
    prot_aligned_idxs = [prot_gene_to_idx[g] for g in gene_list if g in prot_gene_to_idx]
    prot_aligned_genes = [g for g in gene_list if g in prot_gene_to_idx]
    prot_aligned_y = y[np.array([gene_to_emb_idx[g] for g in prot_aligned_genes])]
    X_prot_aligned = X_prot[prot_aligned_idxs]

    observed = observed_rows_mask(X_prot_aligned, label="enzyme proteome features")
    X_prot_aligned = X_prot_aligned[observed]
    prot_aligned_y = prot_aligned_y[observed]
    prot_aligned_genes = list(np.asarray(prot_aligned_genes)[observed])

    print(f"Proteome-aligned genes: {len(prot_aligned_genes)}")
    proteome_results = run_multiseed(
        X_prot_aligned,
        prot_aligned_y,
        prot_aligned_genes,
        pfam_map,
        le,
        seeds=seeds,
        n_folds=args.n_folds,
        compute_ci=compute_ci,
        n_boot=args.n_boot,
        n_permutations=args.n_permutations,
    )

    input_fingerprints = enzyme_input_fingerprints(
        X_emb=X_emb,
        genes=gene_list,
        uniprot_ids=gene_uniprot_ids,
        labels=y,
        pfam_map=pfam_map,
        X_proteome=X_prot_aligned,
        proteome_genes=prot_aligned_genes,
        proteome_labels=prot_aligned_y,
        proteome_columns=proteome_columns,
        mechanism_reference=_mechanism_reference_fingerprints(),
    )

    print("\n" + "=" * 60)
    print("DECISION RULES (PREREGISTRATION_run_biorxiv.md, 2F-2H)")
    print("=" * 60)

    fs_f1 = emb_results["logreg_family_split"].get("oof_macro_f1")
    mlp_f1 = emb_results["mlp_family_split"].get("oof_macro_f1")
    gs_f1 = emb_results["logreg_gene_split"]["macro_f1_mean"]
    fs_ci = (emb_results.get("bootstrap_ci") or {}).get("macro_f1")
    paired_ci = emb_results.get("paired_ci_mlp_minus_logreg")
    paired_mechanism_ci = emb_results.get("paired_ci_logreg_minus_mechanism")

    gate_2f = fs_f1 is not None and fs_f1 >= 0.70
    mechanism_point = (
        paired_mechanism_ci.get("point_b")
        if paired_mechanism_ci is not None
        else None
    )
    enzyme_mechanism_diff = (
        paired_mechanism_ci.get("point_diff")
        if paired_mechanism_ci is not None
        else None
    )
    gate_2g = (
        bool(enzyme_mechanism_diff >= ENZYME_MECHANISM_MIN_F1_GAP)
        if enzyme_mechanism_diff is not None
        else None
    )
    gate_2h = mlp_f1 is not None and fs_f1 is not None and abs(mlp_f1 - fs_f1) < 0.05

    verdict_2f = adjudicate_level(fs_f1, fs_ci, 0.70)
    verdict_2g = adjudicate_diff(
        gate_2g,
        paired_mechanism_ci,
        ENZYME_MECHANISM_MIN_F1_GAP,
    )
    verdict_2h = adjudicate_equivalence(gate_2h, paired_ci, 0.05)

    print(f"\n2F — family-split F1 >= 0.70:  {verdict_2f}  (F1={fs_f1:.3f})")
    if enzyme_mechanism_diff is not None:
        print(
            f"2G — enzyme minus mechanism F1 >= "
            f"{ENZYME_MECHANISM_MIN_F1_GAP:.2f}:  {verdict_2g}  "
            f"(delta={enzyme_mechanism_diff:+.3f})"
        )
    else:
        print("2G — not adjudicated (paired enzyme-mechanism difference unavailable)")
    if mlp_f1 is not None and fs_f1 is not None:
        print(
            f"2H — MLP approx LogReg family-split:  {verdict_2h}  "
            f"(delta_MLP-LR={mlp_f1 - fs_f1:+.3f})"
        )
    else:
        print("2H — MLP approx LogReg family-split:  SKIPPED (missing OOF)")

    output = {
        "description": (
            "Enzyme type classification (kinase/protease/oxidoreductase/non-enzyme) "
            "from ESM-2 WT mean-pooled embeddings. Positive control paralleling the "
            "mechanism arc (results 1-10). Primary metric: family-split LogReg macro-F1."
        ),
        "seeds": seeds,
        "n_folds": args.n_folds,
        "compute_ci": compute_ci,
        "n_boot": args.n_boot if compute_ci else None,
        "n_permutations": args.n_permutations,
        "n_genes": len(gene_list),
        "class_distribution": dict(Counter(y_str)),
        "classes": list(le.classes_),
        "input_fingerprints": input_fingerprints,
        "analysis_parameters": {
            "seeds": seeds,
            "n_folds": args.n_folds,
            "compute_ci": compute_ci,
            "n_boot": args.n_boot if compute_ci else None,
            "n_permutations": args.n_permutations,
        },
        "esm2_wt_embedding": emb_results,
        "proteome_features": proteome_results,
        "gate_evaluation": {
            "2F_family_split_f1_ge_0.70": bool(gate_2f),
            "2F_verdict": verdict_2f,
            "2G_enzyme_beats_mechanism": gate_2g,
            "2G_verdict": verdict_2g,
            "2G_minimum_f1_gap": ENZYME_MECHANISM_MIN_F1_GAP,
            "2H_mlp_approx_logreg": bool(gate_2h),
            "2H_verdict": verdict_2h,
            "fs_f1": fs_f1,
            "mlp_f1": mlp_f1,
            "gs_f1": gs_f1,
            "mechanism_reference_f1": mechanism_ref_f1,
            "mechanism_seed0_oof_f1": mechanism_point,
            "enzyme_minus_mechanism_f1": enzyme_mechanism_diff,
            "note": "Gate point estimates and bootstrap CIs use fold-aware macro-F1",
        },
    }

    write_result_json(ENZYME_CLASSIFICATION_JSON, output, seeds=seeds, indent=2)
    print(f"\nResults written to {ENZYME_CLASSIFICATION_JSON}")


if __name__ == "__main__":
    main()
