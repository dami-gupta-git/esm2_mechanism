"""Pathogenicity positive control: embed ClinVar variants and probe pathogenic-vs-benign as a pipeline check."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sklearn
from joblib import Parallel, delayed

from esm2_mech.fetch_data.fetch_pathogenicity_variants import (
    load_validated_pathogenicity_cache,
)
from esm2_mech.utils import bootstrap as bootstrap_module
from esm2_mech.utils import data as data_module
from esm2_mech.utils import metrics as metrics_module
from esm2_mech.utils import probes as probes_module
from esm2_mech.utils import sequences as sequences_module
from esm2_mech.utils import splits as splits_module
from esm2_mech.utils.bootstrap import (
    binary_auroc_cluster_bootstrap_ci,
    family_or_gene_clusters,
)
from esm2_mech.utils.constants import (
    BOOTSTRAP_N_RESAMPLES,
    INFERENTIAL_SEED,
    N_FOLDS,
    N_SEEDS,
)
from esm2_mech.utils.classification import validate_complete_classification_splits
from esm2_mech.utils.data import (
    embedding_fingerprint,
    load_pfam_map,
    pathogenicity_label,
    pfam_fingerprint,
    validate_balanced_pathogenicity_variants,
    variants_fingerprint,
)
from esm2_mech.utils.embed import get_esm2_embeddings_for_pairs
from esm2_mech.utils.io import atomic_write_json, save_npy, write_result_json
from esm2_mech.utils.paths import (
    EMB_DIR,
    PATH_EMB_META,
    PATH_EMB_MUT_MEAN,
    PATH_EMB_WT_MEAN,
    PATHOGENICITY_CONTROL_JSON,
    PATHOGENICITY_CONTROL_SEED_JSON,
    PFAM_JSON,
    RESULTS_DIR,
    SEQUENCES_JSON,
)
from esm2_mech.utils.probes import run_logreg_binary_cv, run_mlp_binary_cv
from esm2_mech.utils.sequences import apply_missense, window_sequence
from esm2_mech.utils.splits import family_split_cv, gene_split_cv
from esm2_mech.utils.seed_aggregation import (
    aggregate_result_contract,
    aggregate_seed_results,
    read_seed_point_estimate,
    read_seed_result_contract,
    seed_count,
    seed_result_contract,
)

print = functools.partial(print, flush=True)

ESM2_MODEL_650M = "esm2_t33_650M_UR50D"
PATHOGENICITY_AUROC_MIN = 0.85
_EMBEDDING_METADATA_VERSION = 2
_PROBE_RESULT_VERSION = 4
_BINARY_METRICS = ("auroc", "auprc", "prevalence", "ppv", "npv")


@dataclass(frozen=True)
class ExpectedPathogenicitySelection:
    valid_indices: list[int]
    variants: list[dict]
    wt_sequences: list[str]
    mut_sequences: list[str]
    positions: list[int]
    fingerprint: str
    embedding_input_fingerprint: str
    accounting: dict


def _source_files_fingerprint() -> str:
    """Hash every project source file that defines the pathogenicity control estimand."""
    paths = {
        Path(__file__),
        Path(bootstrap_module.__file__),
        Path(data_module.__file__),
        Path(metrics_module.__file__),
        Path(probes_module.__file__),
        Path(sequences_module.__file__),
        Path(splits_module.__file__),
    }
    digest = hashlib.sha256()
    for path in sorted(paths, key=str):
        digest.update(str(path.name).encode())
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def load_fetched_variants():
    """Load the fetched variant set and validate its complete metadata contract."""
    return load_validated_pathogenicity_cache(
        max_per_gene_per_class=20,
        seed=42,
    )


def _build_valid_pairs_indexed(variants, seq_cache):
    """Filter to embeddable variants; return original indices for re-alignment."""
    valid_indices, valid, wt_seqs, mut_seqs, positions = [], [], [], [], []
    skipped = {"no_uid": 0, "uid_not_in_seq_cache": 0, "apply_missense_none": 0}
    for idx, v in enumerate(variants):
        uid = v.get("uniprot_id")
        if not uid:
            skipped["no_uid"] += 1
            continue
        if uid not in seq_cache:
            skipped["uid_not_in_seq_cache"] += 1
            continue
        wt_win, new_pos, _ = window_sequence(seq_cache[uid], v["aa_pos"])
        mut_win = apply_missense(wt_win, new_pos, v["aa_wt"], v["aa_mut"])
        if mut_win is None:
            skipped["apply_missense_none"] += 1
            continue
        valid_indices.append(idx)
        valid.append(v)
        wt_seqs.append(wt_win)
        mut_seqs.append(mut_win)
        positions.append(new_pos)
    for bucket, count in skipped.items():
        if count:
            print(f"  WARNING: skipped {count} variants ({bucket})")
    print(f"  Valid variant pairs: {len(valid)}")
    return valid_indices, valid, wt_seqs, mut_seqs, positions, skipped


def _rebalance_after_filter(valid_indices, valid, wt_seqs, mut_seqs, positions):
    """Re-equalize pathogenic/benign counts per gene after filtering dropped variants unevenly."""
    pre_counts = Counter(v["label"] for v in valid)
    by_gene_class = defaultdict(list)
    for i, v in enumerate(valid):
        by_gene_class[(v["gene"], v["label"])].append(i)

    keep = []
    genes_with_both = {
        gene for (gene, label) in by_gene_class
        if (gene, "pathogenic") in by_gene_class and (gene, "benign") in by_gene_class
    }
    n_dropped_genes = len({g for (g, _) in by_gene_class} - genes_with_both)
    for gene in sorted(genes_with_both):
        p_idx = by_gene_class[(gene, "pathogenic")]
        b_idx = by_gene_class[(gene, "benign")]
        n = min(len(p_idx), len(b_idx))
        keep.extend(p_idx[:n])
        keep.extend(b_idx[:n])
    keep.sort()

    out_indices = [valid_indices[i] for i in keep]
    out_valid = [valid[i] for i in keep]
    out_wt = [wt_seqs[i] for i in keep]
    out_mut = [mut_seqs[i] for i in keep]
    out_pos = [positions[i] for i in keep]

    post_counts = Counter(v["label"] for v in out_valid)
    n_removed = len(valid) - len(out_valid)
    if n_removed:
        print(f"  Rebalanced after filter: {pre_counts} -> {post_counts} "
              f"(removed {n_removed} variants, dropped {n_dropped_genes} single-class genes)")
    accounting = {
        "n_embeddable_before_rebalance": len(valid),
        "n_removed_by_postfilter_balance": n_removed,
        "n_single_class_genes_dropped_postfilter": n_dropped_genes,
    }
    return out_indices, out_valid, out_wt, out_mut, out_pos, accounting


def _derive_expected_selection(variants, seq_cache):
    """Apply the current filter and balance code used to define embedding rows."""
    (
        valid_indices,
        valid,
        wt_sequences,
        mut_sequences,
        positions,
        skipped,
    ) = _build_valid_pairs_indexed(variants, seq_cache)
    (
        valid_indices,
        valid,
        wt_sequences,
        mut_sequences,
        positions,
        balance_accounting,
    ) = _rebalance_after_filter(
        valid_indices, valid, wt_sequences, mut_sequences, positions
    )
    realised_design = validate_balanced_pathogenicity_variants(
        valid, require_unique_substitutions=True
    )
    accounting = {
        "n_fetched_variants": len(variants),
        "filter_skips": skipped,
        **balance_accounting,
        "n_scored_variants": len(valid),
        "realised_design": realised_design,
    }
    input_digest = hashlib.sha256()
    for variant, wt_sequence, mut_sequence, position in zip(
        valid, wt_sequences, mut_sequences, positions
    ):
        input_digest.update(variants_fingerprint([variant]).encode())
        input_digest.update(b"\x00")
        input_digest.update(wt_sequence.encode())
        input_digest.update(b"\x00")
        input_digest.update(mut_sequence.encode())
        input_digest.update(b"\x00")
        input_digest.update(str(position).encode())
        input_digest.update(b"\x00")
    return ExpectedPathogenicitySelection(
        valid_indices=valid_indices,
        variants=valid,
        wt_sequences=wt_sequences,
        mut_sequences=mut_sequences,
        positions=positions,
        fingerprint=variants_fingerprint(valid),
        embedding_input_fingerprint=input_digest.hexdigest(),
        accounting=accounting,
    )


def _validate_embedding_cache(expected, fetch_metadata, model):
    """Load embeddings only when arrays, metadata, and current selection agree."""
    paths = [PATH_EMB_WT_MEAN, PATH_EMB_MUT_MEAN, PATH_EMB_META]
    existing = [path.exists() for path in paths]
    if not all(existing):
        missing = [str(path) for path, exists in zip(paths, existing) if not exists]
        raise FileNotFoundError(
            f"pathogenicity embedding cache is incomplete; missing {missing}. "
            "Run the embed phase to regenerate all three files."
        )
    try:
        with open(PATH_EMB_META) as f:
            meta = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{PATH_EMB_META} is corrupt; regenerate the embedding cache"
        ) from exc

    expected_metadata = {
        "metadata_version": _EMBEDDING_METADATA_VERSION,
        "n": expected.accounting["n_fetched_variants"],
        "n_valid": len(expected.variants),
        "fingerprint": expected.fingerprint,
        "embedding_input_fingerprint": expected.embedding_input_fingerprint,
        "model": model,
        "fetch_variant_fingerprint": fetch_metadata["variant_fingerprint"],
        "selection_accounting": expected.accounting,
    }
    mismatches = {
        key: {"cached": meta.get(key), "current": value}
        for key, value in expected_metadata.items()
        if meta.get(key) != value
    }
    if meta.get("valid_indices") != expected.valid_indices:
        mismatches["valid_indices"] = {
            "cached_count": len(meta.get("valid_indices", [])),
            "current_count": len(expected.valid_indices),
            "same_count": len(meta.get("valid_indices", [])) == len(expected.valid_indices),
        }
    if mismatches:
        raise ValueError(
            "pathogenicity embedding cache was not produced by the current "
            f"selection contract: {mismatches}. Regenerate the embed phase."
        )

    wt_mean = np.load(PATH_EMB_WT_MEAN)
    mut_mean = np.load(PATH_EMB_MUT_MEAN)
    expected_rows = len(expected.variants)
    if wt_mean.shape[0] != expected_rows or mut_mean.shape[0] != expected_rows:
        raise ValueError(
            f"embedding row mismatch: expected {expected_rows}, got "
            f"wt={wt_mean.shape[0]} and mut={mut_mean.shape[0]}"
        )
    actual_embedding_fingerprint = embedding_fingerprint(wt_mean, mut_mean)
    if meta.get("embedding_fingerprint") != actual_embedding_fingerprint:
        raise ValueError(
            "pathogenicity embedding arrays do not match the fingerprint stored "
            "at extraction time; regenerate the embed phase"
        )
    return wt_mean, mut_mean, meta


def embed_phase(variants, fetch_metadata, model, batch_size, force=False):
    """Phase 1. Extract and cache pathogenicity embeddings (GPU)."""
    print("\n=== Phase 1: extract ESM-2 embeddings ===")
    with open(SEQUENCES_JSON) as f:
        seq_cache = json.load(f)

    expected = _derive_expected_selection(variants, seq_cache)

    cache_paths = [PATH_EMB_WT_MEAN, PATH_EMB_MUT_MEAN, PATH_EMB_META]
    if any(path.exists() for path in cache_paths) and not force:
        _validate_embedding_cache(expected, fetch_metadata, model)
        print("  Embeddings already complete and validated; skipping extraction.")
        return
    if force and any(path.exists() for path in cache_paths):
        print("  Replacing the pathogenicity embedding cache (--force_embed).")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device} | Model: {model}")
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    wt_mean, mut_mean, _, _ = get_esm2_embeddings_for_pairs(
        expected.wt_sequences, expected.mut_sequences, expected.positions,
        valid_variants=expected.variants, out_dir=None,
        model_name=model, device=device, batch_size=batch_size,
    )

    if wt_mean.shape[0] != len(expected.variants) or mut_mean.shape[0] != len(expected.variants):
        raise ValueError(
            "embedding extractor returned a row count that does not match the "
            "derived pathogenicity selection"
        )
    extracted_embedding_fingerprint = embedding_fingerprint(wt_mean, mut_mean)

    save_npy(str(PATH_EMB_WT_MEAN), wt_mean)
    save_npy(str(PATH_EMB_MUT_MEAN), mut_mean)
    atomic_write_json(
        PATH_EMB_META,
        {
            "metadata_version": _EMBEDDING_METADATA_VERSION,
            "valid_indices": expected.valid_indices,
            "n": len(variants),
            "n_valid": len(expected.variants),
            "fingerprint": expected.fingerprint,
            "embedding_input_fingerprint": expected.embedding_input_fingerprint,
            "model": model,
            "fetch_variant_fingerprint": fetch_metadata["variant_fingerprint"],
            "selection_accounting": expected.accounting,
            "embedding_fingerprint": extracted_embedding_fingerprint,
        },
    )
    print(f"  Saved {wt_mean.shape} -> {EMB_DIR}")


# ===========================================================================
# Phase 2 — 5-seed probes
# ===========================================================================
def _json_fingerprint(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _finite_or_none(value):
    if value is None:
        return None
    return float(value) if np.isfinite(value) else None


def _seed_params(seed, compute_ci, n_boot, meta, fetch_metadata, pfam_map, genes):
    return {
        "probe_result_version": _PROBE_RESULT_VERSION,
        "seed": int(seed),
        "compute_ci": bool(compute_ci),
        "n_boot": int(n_boot) if compute_ci else None,
        "variant_fingerprint": meta["fingerprint"],
        "model": meta["model"],
        "pfam_fingerprint": pfam_fingerprint(pfam_map, genes.tolist()),
        "embedding_fingerprint": meta["embedding_fingerprint"],
        "fetch_metadata_fingerprint": _json_fingerprint(fetch_metadata),
        "analysis_source_fingerprint": _source_files_fingerprint(),
        "runtime_versions": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }


def _seed_cell(probe_result, oof, ci, split_name):
    metrics = {}
    for metric in _BINARY_METRICS:
        mean_key = f"{metric}_mean"
        std_key = f"{metric}_std"
        metrics[metric] = {
            "fold_mean": _finite_or_none(probe_result.get(mean_key)),
            "fold_std": _finite_or_none(probe_result.get(std_key)),
        }
    return {
        "status": probe_result.get("status"),
        "metrics": metrics,
        "n_folds": probe_result.get("n_folds"),
        "n_scored": None if oof is None else int(len(oof["y_true"])),
        "resampling_unit": "pfam_family" if split_name == "family" else "gene",
        "auroc_ci": ci,
    }


def _aggregate_metric(seed_results, requested_seeds, cell_key, metric):
    return aggregate_seed_results(
        requested_seeds,
        seed_results,
        lambda result: result["cells"][cell_key]["metrics"][metric]["fold_mean"],
        status=lambda result: result["cells"][cell_key]["status"],
    )


def _build_pathogenicity_auroc_assessment(single_seed_inference, across_seed_point_estimate):
    """Report the claim 2C threshold and estimates without an interval verdict.

    The only interval available here is a bootstrap over one seed's out-of-fold
    predictions. It describes that seed, not the across-seed AUROC reported in
    the same record, so it is neither carried as this claim's interval nor used
    to adjudicate it. Both point estimates stay reportable because each satisfies
    its own contract; the verdict waits for the replacement interval method under
    audit item 1.4.
    """
    return {
        "assessment": "pathogenicity_family_held_out_auroc",
        "feature": "delta_mean",
        "probe": "mlp",
        "split": "family",
        "seed": single_seed_inference["seed"],
        "threshold": PATHOGENICITY_AUROC_MIN,
        "point_estimate": single_seed_inference["point_estimate"],
        "across_seed_point_estimate": across_seed_point_estimate,
        "estimate_basis": single_seed_inference["estimate_basis"],
        "resampling_unit": single_seed_inference["resampling_unit"],
        "n_scored": single_seed_inference["n_scored"],
        "n_excluded": single_seed_inference["n_excluded"],
        "interval": None,
        "interval_reason": (
            "an interval for the across-seed AUROC is unavailable pending audit "
            "item 1.4; a single-seed bootstrap is not a substitute"
        ),
        "interval_dependent_verdict": None,
        "verdict": None,
    }


def _run_probe_with_contract(
    probe,
    features,
    labels,
    splits,
    classes,
    split_contract,
    **probe_kwargs,
):
    """Call a shared binary probe with its explicit classification contract."""
    return probe(
        features,
        labels,
        splits,
        classes=classes,
        split_contract=split_contract,
        **probe_kwargs,
    )


def probe_phase(
    variants,
    fetch_metadata,
    n_seeds,
    n_jobs=-1,
    compute_ci=True,
    n_boot=BOOTSTRAP_N_RESAMPLES,
):
    """Phase 2. 5-seed logreg + MLP probes on delta_mean and wt_only."""
    print("\n=== Phase 2: probes ===")
    with open(SEQUENCES_JSON) as handle:
        seq_cache = json.load(handle)
    expected = _derive_expected_selection(variants, seq_cache)
    wt_mean, mut_mean, meta = _validate_embedding_cache(
        expected, fetch_metadata, ESM2_MODEL_650M
    )
    valid = expected.variants

    delta = mut_mean - wt_mean
    y = np.array([pathogenicity_label(v["label"]) for v in valid])
    genes = np.array([v["gene"] for v in valid])
    pfam_map = load_pfam_map(PFAM_JSON)

    print(f"  {len(valid)} variants  pathogenic={int(y.sum())} benign={int((1 - y).sum())}  "
          f"{len(set(genes))} genes")

    features = {"delta_mean": delta, "wt_only": wt_mean}
    probes = {"logreg": run_logreg_binary_cv, "mlp": run_mlp_binary_cv}
    family_validation_groups = family_or_gene_clusters(
        genes, pfam_map, is_family_split=True
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for seed in range(n_seeds):
        seed_params = _seed_params(
            seed, compute_ci, n_boot, meta, fetch_metadata, pfam_map, genes
        )
        seed_path = Path(PATHOGENICITY_CONTROL_SEED_JSON.format(seed=seed))
        if seed_path.exists():
            cached_seed_params = None
            try:
                with open(seed_path) as handle:
                    cached_seed_result = json.load(handle)
                read_seed_result_contract(seed, str(seed_path), cached_seed_result)
                cached_seed_params = cached_seed_result.get("_params")
            except (json.JSONDecodeError, ValueError):
                pass
            if cached_seed_params == seed_params:
                print(f"  seed {seed}: cached, skipping")
                continue
            print(
                f"  seed {seed}: cache built with {cached_seed_params}, "
                f"current params {seed_params} — recomputing"
            )

        gs = gene_split_cv(genes, seed=seed)
        fs = family_split_cv(genes, pfam_map, seed=seed)
        classes = [0, 1]
        split_contracts = {
            "gene": validate_complete_classification_splits(
                gs,
                requested_folds=N_FOLDS,
                eligible_rows=np.concatenate([test for _train, test in gs]),
                labels=y,
                classes=classes,
                groups=genes,
                held_out_unit="gene",
            ),
            "family": validate_complete_classification_splits(
                fs,
                requested_folds=N_FOLDS,
                eligible_rows=np.concatenate([test for _train, test in fs]),
                labels=y,
                classes=classes,
                groups=family_validation_groups,
                held_out_unit="family",
            ),
        }
        cells = [
            (fname, pname, split_name, splits)
            for fname in features
            for pname in probes
            for split_name, splits in (("gene", gs), ("family", fs))
        ]

        def _run_cell(fname, pname, split_name, splits, seed=seed):
            probe_kwargs = {
                "seed": seed,
                "genes": genes,
                "return_oof": True,
            }
            if pname == "mlp":
                probe_kwargs["validation_groups"] = (
                    family_validation_groups if split_name == "family" else genes
                )
            probe_result, oof = _run_probe_with_contract(
                probes[pname],
                features[fname],
                y,
                splits,
                classes,
                split_contracts[split_name],
                **probe_kwargs,
            )
            if compute_ci and oof is not None and seed == 0:
                clusters = family_or_gene_clusters(
                    oof["genes"], pfam_map, is_family_split=(split_name == "family")
                )
                # binary_auroc_cluster_bootstrap_ci stratifies its own draws by
                # class presence, so the AUROC is defined on every resample.
                ci = binary_auroc_cluster_bootstrap_ci(
                    oof, n_resamples=n_boot, seed=seed, clusters=clusters
                )
            else:
                ci = None
            key = f"{fname}_{pname}_{split_name}"
            return key, _seed_cell(probe_result, oof, ci, split_name)

        outcomes = Parallel(n_jobs=n_jobs)(
            delayed(_run_cell)(*c) for c in cells
        )

        seed_result = {
            **seed_result_contract(seed),
            "_params": seed_params,
            "cells": {key: cell for key, cell in outcomes},
        }
        write_result_json(seed_path, seed_result, seeds=[seed], indent=2)
        summary = "  ".join(
            f"{key}={cell['metrics']['auroc']['fold_mean']:.3f}"
            for key, cell in seed_result["cells"].items()
            if "mlp" in key and cell["metrics"]["auroc"]["fold_mean"] is not None
        )
        print(f"  seed {seed} done -> {seed_path.name}   {summary}")

    seed_results = []
    for seed in range(n_seeds):
        seed_path = Path(PATHOGENICITY_CONTROL_SEED_JSON.format(seed=seed))
        with open(seed_path) as handle:
            seed_result = json.load(handle)
        expected_params = _seed_params(
            seed, compute_ci, n_boot, meta, fetch_metadata, pfam_map, genes
        )
        if seed_result.get("_params") != expected_params:
            raise ValueError(
                f"{seed_path} changed between cache validation and aggregation"
            )
        read_seed_result_contract(seed, str(seed_path), seed_result)
        seed_results.append(seed_result)

    results = {
        **aggregate_result_contract(),
        "result_version": _PROBE_RESULT_VERSION,
        "n_variants": int(len(valid)),
        "n_pathogenic": int(y.sum()),
        "n_benign": int((1 - y).sum()),
        "n_genes": int(len(set(genes))),
        "n_seeds": n_seeds,
        "data_accounting": {
            "fetch": fetch_metadata["accounting"],
            "embedding_selection": expected.accounting,
        },
        "data_provenance": {
            "fetch_selection": fetch_metadata["selection"],
            "clinvar_source": fetch_metadata["clinvar_source"],
            "fetch_variant_fingerprint": fetch_metadata["variant_fingerprint"],
            "scored_variant_fingerprint": expected.fingerprint,
            "embedding_fingerprint": meta["embedding_fingerprint"],
            "pfam_fingerprint": pfam_fingerprint(pfam_map, genes.tolist()),
            "model": meta["model"],
            "analysis_source_fingerprint": _source_files_fingerprint(),
            "runtime_versions": {
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
            },
        },
        "by_feature": {},
    }
    for fname in features:
        results["by_feature"][fname] = {}
        for pname in probes:
            for split_name in ("gene", "family"):
                key = f"{fname}_{pname}_{split_name}"
                metrics = {
                    metric: _aggregate_metric(
                        seed_results, range(n_seeds), key, metric
                    ).to_dict()
                    for metric in _BINARY_METRICS
                }
                # Select the bootstrapped seed by its declared identity rather
                # than by list position, so the reported seed cannot drift from
                # the numbers beside it.
                cells_by_seed = {
                    seed_result["seed"]: seed_result["cells"][key]
                    for seed_result in seed_results
                }
                if INFERENTIAL_SEED not in cells_by_seed:
                    raise ValueError(
                        f"{key} has no result for seed {INFERENTIAL_SEED}, which "
                        "carries the bootstrap"
                    )
                inferential_cell = cells_by_seed[INFERENTIAL_SEED]
                inferential_ci = inferential_cell["auroc_ci"]
                inferential_point = inferential_cell["metrics"]["auroc"]["fold_mean"]
                if inferential_ci is not None and (
                    inferential_ci["point"] is None
                    or inferential_point is None
                    or not np.isclose(inferential_ci["point"], inferential_point)
                ):
                    raise ValueError(
                        f"{key} seed-{INFERENTIAL_SEED} CI point "
                        f"{inferential_ci['point']} does not match the seed-"
                        f"{INFERENTIAL_SEED} fold-mean AUROC {inferential_point}"
                    )
                cell = {
                    "metrics": metrics,
                    # A within-seed resampling interval on one seed's own
                    # estimate. It is kept apart from the seed aggregates above,
                    # whose spread describes variation between model seeds, and
                    # it does not adjudicate anything.
                    "single_seed_inference": {
                        "seed": INFERENTIAL_SEED,
                        "point_estimate": inferential_point,
                        "ci": inferential_ci,
                        "n_scored": inferential_cell["n_scored"],
                        "n_excluded": (
                            None
                            if inferential_cell["n_scored"] is None
                            else len(valid) - inferential_cell["n_scored"]
                        ),
                        "resampling_unit": inferential_cell["resampling_unit"],
                        "estimate_basis": (
                            f"seed_{INFERENTIAL_SEED}_mean_of_fold_aurocs"
                        ),
                    },
                }
                results["by_feature"][fname][f"{pname}_{split_name}"] = cell

    claim_cell = results["by_feature"]["delta_mean"]["mlp_family"]
    claim_metric = read_seed_point_estimate(claim_cell["metrics"]["auroc"])
    results["pathogenicity_family_held_out_auroc"] = _build_pathogenicity_auroc_assessment(
        claim_cell["single_seed_inference"], claim_metric.value
    )

    write_result_json(PATHOGENICITY_CONTROL_JSON, results, seeds=list(range(n_seeds)), indent=2)
    print(f"  Aggregated results written to {PATHOGENICITY_CONTROL_JSON}")
    return results


def _print_headline(results):
    print("\n" + "=" * 60)
    print("HEADLINE — pathogenicity positive control")
    print("=" * 60)

    def cell(feature, key):
        c = results["by_feature"][feature][key]
        return read_seed_point_estimate(c["metrics"]["auroc"])

    for feature in ("delta_mean", "wt_only"):
        for key in ("logreg_gene", "logreg_family", "mlp_gene", "mlp_family"):
            metric = cell(feature, key)
            if not metric.available:
                print(f"  {feature:11s} {key:14s} AUROC = undefined (no valid fold)")
            else:
                spread = "N/A" if metric.spread is None else f"{metric.spread:.3f}"
                print(
                    f"  {feature:11s} {key:14s} AUROC = "
                    f"{metric.value:.3f} ± {spread}"
                )
        print()

    claim = results["pathogenicity_family_held_out_auroc"]
    across_seed = claim["across_seed_point_estimate"]
    single_seed = claim["point_estimate"]
    if across_seed is not None:
        print(f"  Claim 2C: across-seed family-split AUROC = {across_seed:.3f}")
    if single_seed is not None:
        print(
            f"    seed {claim['seed']} alone = {single_seed:.3f} "
            "(one seed, not the reported estimate)"
        )
    print(f"  Verdict: not adjudicated — {claim['interval_reason']}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=ESM2_MODEL_650M, choices=[ESM2_MODEL_650M])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seeds", type=seed_count, default=N_SEEDS, help="number of probe seeds (>=1)")
    parser.add_argument("--n_jobs", type=int, default=-1, help="parallel jobs for probes (-1 = all cores)")
    parser.add_argument("--no_ci", action="store_true", help="skip cluster-bootstrap CIs")
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    parser.add_argument(
        "--force_embed",
        action="store_true",
        help="replace the three pathogenicity embedding-cache files",
    )
    parser.add_argument(
        "--phase", choices=["embed", "probe", "both"], default="both",
        help="Run only 'embed' (GPU) or 'probe' (CPU), or 'both' (default)",
    )
    args = parser.parse_args()

    if args.force_embed and args.phase == "probe":
        parser.error("--force_embed requires --phase embed or --phase both")

    variants, fetch_metadata = load_fetched_variants()

    if args.phase in ("embed", "both"):
        embed_phase(
            variants,
            fetch_metadata,
            model=args.model,
            batch_size=args.batch_size,
            force=args.force_embed,
        )

    if args.phase in ("probe", "both"):
        results = probe_phase(
            variants, fetch_metadata, n_seeds=args.seeds, n_jobs=args.n_jobs,
            compute_ci=not args.no_ci, n_boot=args.n_boot,
        )
        _print_headline(results)


if __name__ == "__main__":
    main()
