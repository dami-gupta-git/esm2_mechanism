"""Enzyme type classification (kinase/protease/oxidoreductase/non-enzyme) from
ESM-2 WT mean-pooled embeddings, as a positive control for the mechanism arc.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
from collections import Counter

import numpy as np
from sklearn.preprocessing import LabelEncoder

from esm2_mech.utils.constants import (
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_N_RESAMPLES,
    N_FOLDS,
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
    adjudicate_diff,
    adjudicate_equivalence,
    adjudicate_level,
    attach_mechanism_ci,
    family_or_gene_clusters,
    oof_permutation_pvalue,
    oof_score_arms,
    paired_cluster_bootstrap_diff_shared_clusters,
    paired_oof_diff,
    score_within_folds,
)
from esm2_mech.utils.seed_aggregation import (
    SEED_STATUS_SUCCESS,
    SEED_STATUS_UNSCORABLE,
    aggregate_result_contract,
    aggregate_seed_oof,
    aggregate_seed_results,
    make_seed_payload_record,
    read_seed_point_estimate,
    read_seed_result_contract,
    seed_count,
    seed_result_contract,
)
from esm2_mech.experiments.mechanism.seed_results import read_across_seed_metric
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
    value = read_across_seed_metric(
        str(MECHANISM_AGGREGATE_JSON),
        "family_split",
        "delta_mean",
    )
    if value is None:
        print("  WARNING: mechanism family-split reference is unavailable")
    return value


def _load_mechanism_seed_records(seeds: list[int]) -> list:
    """Load current-schema mechanism family-split results by seed."""
    records = []
    for seed in seeds:
        result_path = RESULTS_DIR / seed_result_filename(seed)
        if not result_path.exists():
            print(
                f"  WARNING: mechanism seed {seed} result file is missing: "
                f"{result_path}"
            )
            continue
        with open(result_path) as handle:
            result = json.load(handle)
        root_status = read_seed_result_contract(seed, str(result_path), result)
        block = result.get("family_split", {}).get("delta_mean", {})
        records.append(
            {
                **seed_result_contract(seed, status=root_status),
                "mechanism": block,
            }
        )
    return records


def _load_mechanism_family_oof_for_seed(seed: int) -> dict | None:
    """Load one seed's mechanism family-split OOF, bound to that seed's result."""
    cache_path = RESULTS_DIR / mechanism_oof_cache_filename(seed)
    result_path = RESULTS_DIR / seed_result_filename(seed)
    if not cache_path.exists() and not result_path.exists():
        print(
            f"  WARNING: mechanism seed {seed} result and OOF cache not found — "
            "enzyme/mechanism difference CI unavailable"
        )
        return None
    if not cache_path.exists() or not result_path.exists():
        missing = cache_path if not cache_path.exists() else result_path
        raise FileNotFoundError(
            f"{missing} is missing; the seed {seed} result and OOF cache must be "
            "regenerated together"
        )

    with open(cache_path) as handle:
        cache = json.load(handle)
    with open(result_path) as handle:
        result = json.load(handle)

    expected = {
        "cache_schema_version": MECHANISM_OOF_CACHE_SCHEMA_VERSION,
        "seed": seed,
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


def load_mechanism_family_oof_arms(seeds: list[int]) -> tuple | None:
    """Every requested seed's mechanism OOF, aligned on one shared row order.

    The enzyme-minus-mechanism effect is averaged over model seeds inside each
    bootstrap draw, so the mechanism arm contributes one scoring block per seed,
    all indexed by the same rows. A seed whose cache is missing makes the whole
    comparison unavailable rather than reducing it to the seeds that survived.
    """
    per_seed = {}
    for seed in seeds:
        oof = _load_mechanism_family_oof_for_seed(seed)
        if oof is not None:
            predictions = np.asarray(oof["pred"], dtype=object)
            if not set(predictions).issubset(MECHANISM_CLASSES):
                raise ValueError(
                    f"mechanism seed {seed} OOF contains an undeclared class"
                )
            probabilities = np.zeros(
                (len(predictions), len(MECHANISM_CLASSES)), dtype=float
            )
            for class_index, class_name in enumerate(MECHANISM_CLASSES):
                probabilities[predictions == class_name, class_index] = 1.0
            per_seed[seed] = {
                **oof,
                "classes": list(MECHANISM_CLASSES),
                "proba": probabilities,
            }

    if not per_seed:
        return None
    first = next(iter(per_seed.values()))
    combined = aggregate_seed_oof(
        seeds,
        # Only seeds whose cache loaded are present; a seed with no cache is
        # absent here and is refused as a missing seed, which is what happened.
        [
            make_seed_payload_record(seed, oof, status=SEED_STATUS_SUCCESS)
            for seed, oof in per_seed.items()
        ],
        declared_row_ids=first["row_ids"],
        declared_labels=first["y_true"],
        declared_clusters=first["genes"],
        class_order=MECHANISM_CLASSES,
        declared_fold_ids=range(N_FOLDS),
    )
    if not combined.available:
        print(f"  WARNING: mechanism OOF unavailable — {combined.message}")
        return None
    payload = combined.payload
    observed = np.asarray(payload["y_true"], dtype=object)
    genes = np.asarray(payload["genes"], dtype=object)
    arms = [
        (
            np.asarray(MECHANISM_CLASSES, dtype=object)[proba.argmax(axis=1)],
            folds,
            fold_ids,
        )
        for proba, folds, fold_ids in oof_score_arms(
            payload, "mechanism family-split"
        )
    ]
    return observed, genes, arms


def _mechanism_reference_fingerprints() -> dict:
    """Fingerprint the mechanism aggregate when the optional mechanism comparison input exists."""
    if not MECHANISM_AGGREGATE_JSON.exists():
        return {"content": None, "content_missing": True}
    with open(MECHANISM_AGGREGATE_JSON) as handle:
        aggregate = json.load(handle)
    return {
        "content": _canonical_fingerprint(aggregate),
        "content_missing": False,
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
    mechanism_seed_records: list | None = None,
    mechanism_family_arms: tuple | None = None,
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

    seed_runs = []
    permutation_oof = None
    logreg_family_oof_records = []
    mlp_family_oof_records = []

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
        if fs["status"] == "success":
            print(
                f"    LogReg family-split F1={fs['macro_f1_mean']:.3f}  AUROC: "
                + " ".join(f"{c}={fs[f'auroc_{c}_mean']:.3f}" for c in classes)
            )
        else:
            print(f"    LogReg family-split {fs['status']}")

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
        print(
            f"    MLP    family-split F1={mlp['macro_f1_mean']:.3f}"
            if mlp["status"] == "success"
            else f"    MLP    family-split {mlp['status']}"
        )

        seed_runs.append(
            {
                **seed_result_contract(seed),
                "gene_reference": gs_reference,
                "family_reference": fs_reference,
                "logreg_gene": gs,
                "logreg_family": fs,
                "mlp_family": mlp,
            }
        )
        logreg_family_oof_records.append(
            make_seed_payload_record(seed, fs_oof, status=fs["status"])
        )
        mlp_family_oof_records.append(
            make_seed_payload_record(seed, mlp_oof, status=mlp["status"])
        )
        if seed == seeds[0]:
            permutation_oof = fs_oof

    def arm_aggregate(arm, metric):
        return aggregate_seed_results(
            seeds,
            seed_runs,
            lambda result: result[arm].get(metric),
            status=lambda result: result[arm]["status"],
        )

    def reference_aggregate(key):
        return aggregate_seed_results(
            seeds,
            seed_runs,
            lambda result: result.get(key),
            status=lambda result: (
                SEED_STATUS_SUCCESS
                if result.get(key) is not None
                else SEED_STATUS_UNSCORABLE
            ),
        )

    aggregates = {
        "gene_reference": reference_aggregate("gene_reference"),
        "family_reference": reference_aggregate("family_reference"),
        "logreg_gene": arm_aggregate("logreg_gene", "macro_f1_mean"),
        "logreg_family": arm_aggregate("logreg_family", "macro_f1_mean"),
        "mlp_family": arm_aggregate("mlp_family", "macro_f1_mean"),
    }
    metric_reads = {
        key: read_seed_point_estimate(aggregate)
        for key, aggregate in aggregates.items()
    }

    leakage_pct = None
    if (
        metric_reads["logreg_gene"].available
        and metric_reads["logreg_family"].available
        and metric_reads["gene_reference"].available
        and metric_reads["logreg_gene"].value > metric_reads["gene_reference"].value
    ):
        leakage_pct = round(
            100.0
            * (metric_reads["logreg_gene"].value - metric_reads["logreg_family"].value)
            / (metric_reads["logreg_gene"].value - metric_reads["gene_reference"].value),
            1,
        )

    print(f"\n  Results ({len(seeds)} seeds):")
    for summary_label, key in (
        ("Gene-split reference", "gene_reference"),
        ("Family-split reference", "family_reference"),
        ("LogReg gene-split", "logreg_gene"),
        ("LogReg family-split", "logreg_family"),
        ("MLP family-split", "mlp_family"),
    ):
        metric = metric_reads[key]
        if not metric.available:
            print(f"    {summary_label}: Unscorable")
        else:
            spread = "N/A" if metric.spread is None else f"{metric.spread:.3f}"
            print(f"    {summary_label}: F1={metric.value:.3f} +/- {spread}")
    if leakage_pct is not None:
        print(f"    Leakage fraction:        {leakage_pct:.1f}%")

    permutation_result = None
    paired_mlp_vs_logreg = aggregate_seed_results(
        seeds,
        seed_runs,
        lambda result: (
            result["mlp_family"]["macro_f1_mean"]
            - result["logreg_family"]["macro_f1_mean"]
        ),
        status=lambda result: (
            result["mlp_family"]["status"]
            if result["mlp_family"]["status"] != SEED_STATUS_SUCCESS
            else result["logreg_family"]["status"]
        ),
    )

    mechanism_by_seed = {
        result["seed"]: result for result in (mechanism_seed_records or [])
    }
    paired_logreg_mechanism_results = []
    for enzyme_result in seed_runs:
        seed = enzyme_result["seed"]
        mechanism_result = mechanism_by_seed.get(seed)
        if mechanism_result is None:
            continue
        paired_logreg_mechanism_results.append(
            {
                **seed_result_contract(
                    seed, status=mechanism_result["seed_status"]
                ),
                "enzyme": enzyme_result["logreg_family"],
                "mechanism": mechanism_result["mechanism"],
            }
        )
    paired_logreg_vs_mechanism = aggregate_seed_results(
        seeds,
        paired_logreg_mechanism_results,
        lambda result: (
            result["enzyme"]["macro_f1_mean"]
            - result["mechanism"]["macro_f1_mean"]
        ),
        status=lambda result: (
            result["enzyme"]["status"]
            if result["enzyme"]["status"] != SEED_STATUS_SUCCESS
            else result["mechanism"]["status"]
        ),
    )

    # Interval block. Each bootstrap draw selects families once, both arms and
    # every model seed are scored on that same draw, and the effect is the mean
    # over seeds. The intervals below are therefore across-seed quantities, kept
    # in their own keys beside the seed aggregates whose spread describes
    # variation between seeds.
    ci_result = None
    paired_mlp_vs_logreg_ci = None
    paired_logreg_vs_mechanism_ci = None
    oof_fs_f1 = None
    oof_mlp_f1 = None

    def _combined_oof(oof_records):
        combined = aggregate_seed_oof(
            seeds,
            oof_records,
            declared_row_ids=np.arange(len(y)),
            declared_labels=y,
            declared_clusters=genes_arr,
            class_order=classes,
            declared_fold_ids=range(n_folds),
        )
        return combined.payload if combined.available else None

    logreg_family_oof = _combined_oof(logreg_family_oof_records)
    mlp_family_oof = _combined_oof(mlp_family_oof_records)

    def _oof_macro_f1(oof):
        """Out-of-fold macro-F1, scored per fold and per seed, then averaged."""
        observed = np.asarray(oof["y_true"])
        arms = [
            (
                np.array([classes[col] for col in np.asarray(proba).argmax(axis=1)]),
                np.asarray(folds),
                fold_ids,
            )
            for proba, folds, fold_ids in oof_score_arms(oof, "enzyme family-split")
        ]

        def _fold_f1(block, arm_pred):
            return fold_macro_f1(observed, block, arm_pred, classes)

        return arms, score_within_folds(np.arange(len(observed)), arms, _fold_f1)

    logreg_arms = None
    if logreg_family_oof is not None:
        logreg_arms, oof_fs_f1 = _oof_macro_f1(logreg_family_oof)
        print(f"\n  Across-seed OOF LogReg family-split F1: {oof_fs_f1:.3f}")
    if mlp_family_oof is not None:
        _mlp_arms, oof_mlp_f1 = _oof_macro_f1(mlp_family_oof)
        print(f"  Across-seed OOF MLP family-split F1: {oof_mlp_f1:.3f}")

    if compute_ci and logreg_family_oof is not None:
        print("\n  Computing cluster-bootstrap CIs (all seeds, family-split)...")
        clusters = family_or_gene_clusters(
            np.asarray(logreg_family_oof["genes"]), pfam_map, is_family_split=True
        )
        ci_container: dict = {}
        # attach_mechanism_ci stratifies the ranking metrics' draws itself.
        attach_mechanism_ci(
            ci_container,
            logreg_family_oof,
            clusters,
            compute_ci=True,
            classes=classes,
            n_resamples=n_boot,
            ci_level=BOOTSTRAP_CI_LEVEL,
            seed=0,
        )
        ci_result = ci_container["ci"]
        for metric_name, interval in ci_result.items():
            low = interval.get("ci_low")
            high = interval.get("ci_high")
            point = interval.get("point")
            if low is not None and high is not None and point is not None:
                print(f"    {metric_name}: {point:.3f} [{low:.3f}, {high:.3f}]")

        if mlp_family_oof is not None:
            print("\n  Computing paired CI: MLP minus LogReg (family-split)...")
            # paired_oof_diff stratifies its own draws where the metric needs it.
            paired_mlp_vs_logreg_ci = paired_oof_diff(
                oof_a=mlp_family_oof,
                oof_b=logreg_family_oof,
                pfam_map=pfam_map,
                label="MLP minus LogReg",
                classes=classes,
                metric="macro_f1",
                is_family_split=True,
                n_resamples=n_boot,
                seed=0,
            )
            if paired_mlp_vs_logreg_ci is not None:
                low = paired_mlp_vs_logreg_ci.get("ci_low")
                high = paired_mlp_vs_logreg_ci.get("ci_high")
                point = paired_mlp_vs_logreg_ci.get("point_diff")
                if low is not None and high is not None and point is not None:
                    print(f"    MLP-LogReg diff: {point:+.3f} [{low:+.3f}, {high:+.3f}]")

        if mechanism_family_arms is not None:
            print("\n  Computing paired CI: enzyme LogReg minus mechanism...")
            # The two tasks score different rows — genes here, variants there —
            # but they share Pfam families. One draw of the families present in
            # both cohorts selects each arm's own rows, so the difference is a
            # task difference rather than a difference between two family
            # populations. Both arms are macro-F1 with a fixed class denominator,
            # scored within fold and averaged over seeds.
            mechanism_y, mechanism_genes, mechanism_arms = mechanism_family_arms
            enzyme_observed = np.asarray(logreg_family_oof["y_true"])
            enzyme_clusters = family_or_gene_clusters(
                np.asarray(logreg_family_oof["genes"]),
                pfam_map,
                is_family_split=True,
            )
            mechanism_clusters = family_or_gene_clusters(
                mechanism_genes, pfam_map, is_family_split=True
            )

            def _enzyme_f1(rows):
                return score_within_folds(
                    rows,
                    logreg_arms,
                    lambda block, arm_pred: fold_macro_f1(
                        enzyme_observed, block, arm_pred, classes
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

            paired_logreg_vs_mechanism_ci = (
                paired_cluster_bootstrap_diff_shared_clusters(
                    enzyme_clusters,
                    mechanism_clusters,
                    _enzyme_f1,
                    _mechanism_f1,
                    n_resamples=n_boot,
                    seed=0,
                )
            )
            low = paired_logreg_vs_mechanism_ci.get("ci_low")
            high = paired_logreg_vs_mechanism_ci.get("ci_high")
            point = paired_logreg_vs_mechanism_ci.get("point_diff")
            print(
                f"    shared families: "
                f"{paired_logreg_vs_mechanism_ci['n_clusters_shared']} of "
                f"{paired_logreg_vs_mechanism_ci['n_clusters_a_total']} enzyme and "
                f"{paired_logreg_vs_mechanism_ci['n_clusters_b_total']} mechanism"
            )
            if low is not None and high is not None and point is not None:
                print(f"    enzyme-mechanism diff: {point:+.3f} [{low:+.3f}, {high:+.3f}]")

    if n_permutations > 0 and permutation_oof is not None:
        print(
            f"\n  Computing permutation p-value ({n_permutations} reps) "
            f"for seed {seeds[0]}..."
        )
        clusters = family_or_gene_clusters(
            permutation_oof["genes"], pfam_map, is_family_split=True
        )
        permutation_result = oof_permutation_pvalue(
            y_true=np.asarray(permutation_oof["y_true"]),
            proba=permutation_oof["proba"],
            folds=permutation_oof["folds"],
            groups=permutation_oof["genes"],
            clusters=clusters,
            classes=classes,
            n_permutations=n_permutations,
            seed=seeds[0],
        )
        permutation_result["seed"] = seeds[0]
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
        **aggregate_result_contract(),
        "majority_reference": {
            "gene_split": aggregates["gene_reference"].to_dict(),
            "family_split": aggregates["family_reference"].to_dict(),
        },
        "logreg_gene_split": {
            "macro_f1_seed_aggregate": aggregates["logreg_gene"].to_dict(),
            "per_class_auroc_seed_aggregate": {
                class_name: arm_aggregate(
                    "logreg_gene", f"auroc_{class_name}_mean"
                ).to_dict()
                for class_name in classes
            },
        },
        "logreg_family_split": {
            "macro_f1_seed_aggregate": aggregates["logreg_family"].to_dict(),
            "per_class_auroc_seed_aggregate": {
                class_name: arm_aggregate(
                    "logreg_family", f"auroc_{class_name}_mean"
                ).to_dict()
                for class_name in classes
            },
        },
        "mlp_family_split": {
            "macro_f1_seed_aggregate": aggregates["mlp_family"].to_dict(),
            "per_class_auroc_seed_aggregate": {
                class_name: arm_aggregate(
                    "mlp_family", f"auroc_{class_name}_mean"
                ).to_dict()
                for class_name in classes
            },
        },
        "paired_mlp_minus_logreg_seed_aggregate": paired_mlp_vs_logreg.to_dict(),
        "paired_logreg_minus_mechanism_seed_aggregate": (
            paired_logreg_vs_mechanism.to_dict()
        ),
        "leakage_pct": leakage_pct,
    }

    if oof_fs_f1 is not None:
        result["logreg_family_split"]["across_seed_oof_macro_f1"] = oof_fs_f1
    if oof_mlp_f1 is not None:
        result["mlp_family_split"]["across_seed_oof_macro_f1"] = oof_mlp_f1
    if ci_result is not None:
        result["bootstrap_ci"] = ci_result
    if paired_mlp_vs_logreg_ci is not None:
        result["paired_ci_mlp_minus_logreg"] = paired_mlp_vs_logreg_ci
    if paired_logreg_vs_mechanism_ci is not None:
        result["paired_ci_logreg_minus_mechanism"] = paired_logreg_vs_mechanism_ci
    if permutation_result is not None:
        result["permutation_test"] = permutation_result

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=seed_count, default=N_SEEDS,
                        help="number of seeds to run; runs 0..seeds-1 (>=1)")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--no_ci", action="store_true", help="skip cluster-bootstrap CIs")
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    parser.add_argument(
        "--n_permutations", type=int, default=0,
        help="label-permutation reps for OOF macro AUROC (0 = skip)",
    )
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    compute_ci = not args.no_ci

    ENZYME_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Enzyme Classification from ESM-2 WT Embeddings ===")
    print(f"Seeds: {seeds}  Folds: {args.n_folds}  CI: {compute_ci}  n_boot: {args.n_boot}")

    mechanism_ref_f1 = _load_mechanism_reference_f1()
    mechanism_seed_records = _load_mechanism_seed_records(seeds)
    mechanism_family_arms = (
        load_mechanism_family_oof_arms(seeds) if compute_ci else None
    )

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
        compute_ci=compute_ci, n_boot=args.n_boot,
        n_permutations=args.n_permutations,
        mechanism_seed_records=mechanism_seed_records,
        mechanism_family_arms=mechanism_family_arms,
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
    print("DECISION RULES")
    print("=" * 60)

    fs_metric = read_seed_point_estimate(
        emb_results["logreg_family_split"]["macro_f1_seed_aggregate"]
    )
    mlp_metric = read_seed_point_estimate(
        emb_results["mlp_family_split"]["macro_f1_seed_aggregate"]
    )
    gs_metric = read_seed_point_estimate(
        emb_results["logreg_gene_split"]["macro_f1_seed_aggregate"]
    )
    mechanism_difference = read_seed_point_estimate(
        emb_results["paired_logreg_minus_mechanism_seed_aggregate"]
    )
    fs_f1 = fs_metric.value
    mlp_f1 = mlp_metric.value
    gs_f1 = gs_metric.value
    enzyme_mechanism_diff = mechanism_difference.value

    # Each gate is adjudicated on the point its own interval was built around:
    # the effect averaged over model seeds within each bootstrap draw. The seed
    # aggregates above are reported beside them and are never compared against an
    # interval bound, because a resampling interval and a seed spread are
    # different quantities.
    oof_fs_f1 = emb_results["logreg_family_split"].get("across_seed_oof_macro_f1")
    oof_mlp_f1 = emb_results["mlp_family_split"].get("across_seed_oof_macro_f1")
    fs_ci = (emb_results.get("bootstrap_ci") or {}).get("macro_f1")
    paired_ci = emb_results.get("paired_ci_mlp_minus_logreg")
    mechanism_ci = emb_results.get("paired_ci_logreg_minus_mechanism")
    oof_mechanism_diff = (
        mechanism_ci.get("point_diff") if mechanism_ci is not None else None
    )
    mechanism_point = mechanism_ci.get("point_b") if mechanism_ci is not None else None

    enzyme_f1_gate = oof_fs_f1 >= 0.70 if oof_fs_f1 is not None else None
    enzyme_beats_mechanism_gate = (
        bool(oof_mechanism_diff >= ENZYME_MECHANISM_MIN_F1_GAP)
        if oof_mechanism_diff is not None
        else None
    )
    mlp_logreg_equivalence_gate = (
        abs(oof_mlp_f1 - oof_fs_f1) < 0.05
        if oof_mlp_f1 is not None and oof_fs_f1 is not None
        else None
    )

    enzyme_f1_verdict = adjudicate_level(oof_fs_f1, fs_ci, 0.70)
    enzyme_beats_mechanism_verdict = adjudicate_diff(enzyme_beats_mechanism_gate, mechanism_ci, ENZYME_MECHANISM_MIN_F1_GAP)
    mlp_logreg_equivalence_verdict = adjudicate_equivalence(mlp_logreg_equivalence_gate, paired_ci, 0.05)

    fs_text = "N/A" if oof_fs_f1 is None else f"{oof_fs_f1:.3f}"
    print(f"\nenzyme family-held-out F1 >= 0.70:  {enzyme_f1_verdict}  "
        f"(across-seed F1={fs_text})")
    if oof_mechanism_diff is not None:
        print(
            f"enzyme minus mechanism F1 >= "
            f"{ENZYME_MECHANISM_MIN_F1_GAP:.2f}:  {enzyme_beats_mechanism_verdict}  "
            f"(across-seed delta={oof_mechanism_diff:+.3f}, paired on shared families)"
        )
    else:
        print("enzyme minus mechanism F1: not adjudicated (difference unavailable)")
    if oof_mlp_f1 is not None and oof_fs_f1 is not None:
        print(
            f"MLP equivalent to LogReg on the family split:  "
            f"{mlp_logreg_equivalence_verdict}  "
            f"(across-seed delta_MLP-LR={oof_mlp_f1 - oof_fs_f1:+.3f})"
        )
    else:
        print("MLP equivalent to LogReg on the family split: SKIPPED (missing OOF)")

    output = {
        **aggregate_result_contract(),
        "description": (
            "Enzyme type classification (kinase/protease/oxidoreductase/non-enzyme) "
            "from ESM-2 WT mean-pooled embeddings. Positive control paralleling the "
            "mechanism arc (results 1-10). Primary metric: family-split LogReg macro-F1."
        ),
        "seeds": seeds,
        "n_folds": args.n_folds,
        "n_permutations": args.n_permutations,
        "n_genes": len(gene_list),
        "class_distribution": dict(Counter(y_str)),
        "classes": list(le.classes_),
        "input_fingerprints": input_fingerprints,
        "compute_ci": compute_ci,
        "n_boot": args.n_boot if compute_ci else None,
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
            "enzyme_family_held_out_macro_f1_ge_0.70": enzyme_f1_gate,
            "enzyme_family_held_out_macro_f1_verdict": enzyme_f1_verdict,
            "enzyme_minus_mechanism_macro_f1_clears_gap": enzyme_beats_mechanism_gate,
            "enzyme_minus_mechanism_macro_f1_verdict": enzyme_beats_mechanism_verdict,
            "enzyme_minus_mechanism_minimum_f1_gap": ENZYME_MECHANISM_MIN_F1_GAP,
            "mlp_equivalent_to_logreg_family_split": mlp_logreg_equivalence_gate,
            "mlp_equivalent_to_logreg_family_split_verdict": mlp_logreg_equivalence_verdict,
            "adjudicated_on": "across_seed_out_of_fold_estimates",
            "across_seed_fs_f1": oof_fs_f1,
            "across_seed_mlp_f1": oof_mlp_f1,
            "across_seed_mechanism_oof_f1": mechanism_point,
            "across_seed_enzyme_minus_mechanism_f1": oof_mechanism_diff,
            "fs_f1": fs_f1,
            "mlp_f1": mlp_f1,
            "gs_f1": gs_f1,
            "mechanism_reference_f1": mechanism_ref_f1,
            "enzyme_minus_mechanism_f1": enzyme_mechanism_diff,
            "note": (
                "Verdicts adjudicate seed-0 point estimates against seed-0 "
                "bootstrap intervals; fs_f1, mlp_f1, gs_f1 and "
                "enzyme_minus_mechanism_f1 are across-seed means and carry no "
                "interval"
            ),
        },
    }

    write_result_json(ENZYME_CLASSIFICATION_JSON, output, seeds=seeds, indent=2)
    print(f"\nResults written to {ENZYME_CLASSIFICATION_JSON}")


if __name__ == "__main__":
    main()
