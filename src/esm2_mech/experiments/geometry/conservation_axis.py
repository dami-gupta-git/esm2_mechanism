"""Conservation decider: is the pathogenicity axis just ESM-2's conservation
signal, or does the embedding carry pathogenicity beyond masked-LM likelihood?

Phase 1 (GPU, --extract): mask WT position, read log P over 20 AAs.
Phase 2 (CPU): compare conservation features to a family-held-out pathogenicity axis.
"""

import argparse
import hashlib
import inspect
import json
import numpy as np
import functools
from joblib import Parallel, delayed

from esm2_mech.utils.data import (
    embedding_fingerprint,
    load_pfam_map,
    variants_fingerprint,
)
from esm2_mech.utils.io import (
    atomic_write_json,
    load_npy_or_discard,
    save_npy,
    write_result_json,
)
from esm2_mech.utils.paths import (
    GEOMETRY_RESULTS_DIR,
    CONSERVATION_AXIS_JSON,
    CONSERVATION_PATHOGENICITY_NPY,
    CONSERVATION_PATHOGENICITY_META_JSON,
    PATHOGENICITY_CANONICAL_VARIANTS_JSON,
    PFAM_JSON,
    SEQUENCES_JSON,
)

from esm2_mech.embeddings.embed_variants import ESM2_MODEL_650M
from esm2_mech.utils.embed import load_esm2_model, masked_aa_log_probs
from esm2_mech.utils.bootstrap import (
    adjudicate_diff,
    adjudicate_level,
    binary_auroc_cluster_bootstrap_ci,
    family_or_gene_clusters,
    paired_oof_diff,
)
from esm2_mech.utils.constants import (
    AA_ORDER,
    BOOTSTRAP_N_RESAMPLES,
    N_FOLDS,
    N_SEEDS,
)
from esm2_mech.utils.seed_aggregation import (
    aggregate_seed_oof,
    aggregate_paired_seed_difference,
    aggregate_result_contract,
    aggregate_seed_values,
    make_seed_payload_record,
    make_seed_record,
    read_seed_point_estimate,
)
from esm2_mech.utils.probes import run_logreg_binary_cv
from esm2_mech.utils.sequences import window_sequence
from esm2_mech.utils.splits import annotated_gene_mask, family_split_cv
from esm2_mech.utils.classification import validate_complete_classification_splits
from esm2_mech.experiments.geometry.axis_analysis import (
    family_held_out_axis_analysis,
    format_axis_summary,
)
from esm2_mech.experiments.geometry.data import (
    load_pathogenicity_geometry_inputs,
    pathogenicity_geometry_provenance,
)

print = functools.partial(print, flush=True)

GEOMETRY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = PATHOGENICITY_CANONICAL_VARIANTS_JSON
SEQS = SEQUENCES_JSON
CONS_CACHE = CONSERVATION_PATHOGENICITY_NPY
CONS_META = CONSERVATION_PATHOGENICITY_META_JSON

# Claim thresholds
CONSERVATION_ONLY_AUROC_MIN = 0.85
DELTA_ADDED_VALUE_MIN = 0.02

PATHOGENIC = 1
LOGREG_MAX_ITER = 2000
CONSERVATION_CACHE_VERSION = 4


def _sequence_fingerprint(variants, seqs):
    """Hash every sequence entry used by the ordered canonical variant set."""
    digest = hashlib.sha256()
    for uniprot_id in sorted({variant["uniprot_id"] for variant in variants}):
        sequence = seqs.get(uniprot_id)
        marker = "<missing>" if sequence is None else sequence
        digest.update(f"{uniprot_id}|{marker}".encode())
        digest.update(b"\x00")
    return digest.hexdigest()


def _windowing_fingerprint():
    """Hash the windowing implementation and its bound configuration."""
    payload = {
        "source": inspect.getsource(window_sequence),
        "defaults": window_sequence.__defaults__,
        "keyword_defaults": window_sequence.__kwdefaults__,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def conservation_cache_identity(variants, seqs):
    """Describe all mutable inputs that determine the conservation array."""
    identity = {
        "version": CONSERVATION_CACHE_VERSION,
        "n": len(variants),
        "variant_fingerprint": variants_fingerprint(variants),
        "sequence_fingerprint": _sequence_fingerprint(variants, seqs),
        "windowing_fingerprint": _windowing_fingerprint(),
        "model": ESM2_MODEL_650M,
        "aa_order": AA_ORDER,
        "features": ["logP_wt", "logP_mut", "entropy"],
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    identity["fingerprint"] = hashlib.sha256(payload.encode()).hexdigest()
    return identity


def _save_conservation_cache(values, identity):
    save_npy(CONS_CACHE, values)
    metadata = dict(identity)
    metadata["coverage"] = int(np.isfinite(values).all(axis=1).sum())
    metadata["conservation_array_fingerprint"] = embedding_fingerprint(values)
    atomic_write_json(CONS_META, metadata)


def load_validated_conservation_cache(variants, seqs):
    """Load a conservation cache only when its full provenance matches."""
    if not CONS_CACHE.exists():
        raise FileNotFoundError(CONS_CACHE)
    if not CONS_META.exists():
        raise FileNotFoundError(
            f"conservation metadata is missing at {CONS_META}; re-run --extract"
        )
    values = load_npy_or_discard(CONS_CACHE)
    if values is None:
        raise ValueError(f"conservation cache at {CONS_CACHE} could not be loaded")
    with open(CONS_META) as handle:
        metadata = json.load(handle)
    expected = conservation_cache_identity(variants, seqs)
    if metadata.get("fingerprint") != expected["fingerprint"]:
        raise ValueError(
            "conservation cache provenance does not match the current variants, "
            "sequences, model, or amino-acid order; re-run --extract"
        )
    expected_shape = (len(variants), 3)
    if values.shape != expected_shape:
        raise ValueError(
            f"conservation cache has shape {values.shape}; expected {expected_shape}"
        )
    current_array_fingerprint = embedding_fingerprint(values)
    if metadata.get("conservation_array_fingerprint") != current_array_fingerprint:
        raise ValueError(
            "conservation array does not match the content fingerprint stored when "
            "the cache was written; re-run --extract"
        )
    current_coverage = int(np.isfinite(values).all(axis=1).sum())
    if metadata.get("coverage") != current_coverage:
        raise ValueError(
            f"conservation metadata records coverage {metadata.get('coverage')}, "
            f"but the array contains {current_coverage} complete rows; re-run --extract"
        )
    return values, metadata


def extract_conservation(variants, seqs, batch_size=64, ckpt_every=2000):
    """Returns (N,3) array [logP_wt, logP_mut, entropy]; NaN where unavailable."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError(
            "Phase 1 needs a GPU (CUDA). Run --extract on a GPU host; "
            "Phase 2 analysis runs on CPU once the cache exists."
        )

    N = len(variants)
    identity = conservation_cache_identity(variants, seqs)
    out = np.full((N, 3), np.nan, dtype=np.float32)
    done = 0
    if CONS_CACHE.exists() and CONS_META.exists():
        cached = load_npy_or_discard(CONS_CACHE)
        with open(CONS_META) as _f:
            meta = json.load(_f)
        if (
            cached is not None
            and cached.shape == (N, 3)
            and meta.get("fingerprint") == identity["fingerprint"]
            and meta.get("conservation_array_fingerprint")
            == embedding_fingerprint(cached)
            and meta.get("coverage") == int(np.isfinite(cached).all(axis=1).sum())
        ):
            out = cached
            done = int(np.isfinite(out).all(axis=1).sum())
            print(f"Resuming: {done}/{N} variants already extracted")
        elif cached is not None and len(cached) == N and "fingerprint" not in meta:
            raise ValueError(
                f"Conservation cache at {CONS_CACHE} has {N} rows but no fingerprint "
                f"in {CONS_META}. A row-count match alone cannot verify the variant "
                f"ordering. Delete the cache and re-extract."
            )
        else:
            print("Existing conservation cache has stale provenance; recomputing it")

    model, alphabet = load_esm2_model(ESM2_MODEL_650M, device=device)

    work = []
    skipped = 0
    for i, v in enumerate(variants):
        if np.isfinite(out[i]).all():
            continue
        seq = seqs.get(v.get("uniprot_id"))
        if not seq or not (1 <= v["aa_pos"] <= len(seq)):
            skipped += 1
            continue
        win, new_pos, _ = window_sequence(seq, v["aa_pos"])
        if win[new_pos - 1] != v["aa_wt"]:  # alignment / sequence mismatch
            skipped += 1
            continue
        masked = list(win)
        masked[new_pos - 1] = "<mask>"
        work.append((i, "".join(masked), new_pos, v["aa_wt"], v["aa_mut"]))
    print(
        f"To extract: {len(work)} variants ({skipped} skipped: missing seq / pos / WT mismatch)"
    )

    for bs in range(0, len(work), batch_size):
        batch = work[bs : bs + batch_size]
        items = [(idx, mseq, new_pos) for (idx, mseq, new_pos, _wt, _mut) in batch]
        log_probs_by_idx = masked_aa_log_probs(
            model, alphabet, device, items, AA_ORDER, batch_size=batch_size
        )
        for idx, _mseq, _new_pos, wt, mut in batch:
            log_probs = log_probs_by_idx[idx]
            p20 = np.exp(log_probs)
            p20 = p20 / p20.sum()
            entropy = float(-(p20 * np.log(p20 + 1e-12)).sum())
            out[idx, 0] = (
                float(log_probs[AA_ORDER.index(wt)]) if wt in AA_ORDER else np.nan
            )
            out[idx, 1] = (
                float(log_probs[AA_ORDER.index(mut)]) if mut in AA_ORDER else np.nan
            )
            out[idx, 2] = entropy
        done = int(np.isfinite(out).all(axis=1).sum())
        if (bs // batch_size) % max(1, (ckpt_every // batch_size)) == 0:
            _save_conservation_cache(out, identity)
            print(f"  {done}/{N} done (checkpointed)")

    _save_conservation_cache(out, identity)
    print(f"Saved {CONS_CACHE}: {done}/{N} variants with conservation scores")
    return out


def _unavailable_interval(combined_result):
    """Record why an interval was not computed, without inventing bounds."""
    return {
        "ci_low": None,
        "ci_high": None,
        "ci_suppressed": True,
        "missing": True,
        "reason": combined_result.reason.value,
        "message": combined_result.message,
        "affected_seeds": list(combined_result.affected_seeds),
    }


def _positive_class_oof(combined):
    """Reduce a two-column OOF to the pathogenic-class score an AUROC ranks.

    The bootstrap ranks one score per row while the probe's OOF carries one column
    per declared class, so the positive column is selected by its index in the
    declared class order.
    """
    positive_column = combined["classes"].index(PATHOGENIC)
    return {
        **combined,
        "oof_by_seed": {
            seed: {
                **payload,
                "proba": np.asarray(payload["proba"])[:, positive_column],
            }
            for seed, payload in combined["oof_by_seed"].items()
        },
    }


def _probe_one_seed(X, y, genes, pfam, seed, return_oof=False):
    """Return one mean held-out-fold AUROC for one model seed."""
    splits = list(family_split_cv(genes, pfam, seed=seed))
    family_groups = family_or_gene_clusters(genes, pfam, is_family_split=True)
    contract = validate_complete_classification_splits(
        splits,
        requested_folds=5,
        eligible_rows=np.concatenate([test for _train, test in splits]),
        labels=y,
        classes=[0, 1],
        groups=family_groups,
        held_out_unit="family",
    )
    result = run_logreg_binary_cv(
        X,
        y,
        splits,
        [0, 1],
        contract,
        seed=seed,
        pos_label=PATHOGENIC,
        genes=genes if return_oof else None,
        return_oof=return_oof,
        max_iter=LOGREG_MAX_ITER,
    )
    aggregate, oof = result if return_oof else (result, None)
    return {
        "seed": int(seed),
        "status": aggregate["status"],
        "fold_mean": aggregate.get("auroc_mean"),
        "fold_std": aggregate.get("auroc_std"),
        "sampling_unit": "held_out_fold",
    }, oof


def auroc_family_split(
    X, y, genes, pfam, seeds=range(N_SEEDS), n_jobs=-1, collect_oof=True
):
    """Aggregate one complete family-split estimate per requested model seed.

    Returns the across-seed aggregate, the per-seed fold summaries, and each seed's
    out-of-fold predictions. The out-of-fold arrays stay out of the summaries
    because the summaries are written to the result file.
    """
    seeds = tuple(seeds)
    seed_runs = Parallel(n_jobs=n_jobs)(
        delayed(_probe_one_seed)(X, y, genes, pfam, seed, return_oof=collect_oof)
        for seed in seeds
    )
    aggregate = aggregate_seed_values(
        seeds,
        [
            make_seed_record(run["seed"], run["fold_mean"], status=run["status"])
            for run, _oof in seed_runs
        ],
    )
    return (
        aggregate.to_dict(),
        {run["seed"]: run for run, _oof in seed_runs},
        {run["seed"]: oof for run, oof in seed_runs},
    )


def analyse(compute_ci=True, n_boot=BOOTSTRAP_N_RESAMPLES):
    inputs = load_pathogenicity_geometry_inputs()
    with open(SEQS) as handle:
        seqs = json.load(handle)
    cons, conservation_meta = load_validated_conservation_cache(inputs.variants, seqs)
    pfam = load_pfam_map(PFAM_JSON)

    valid = np.isfinite(cons).all(axis=1)
    print(f"Conservation coverage: {valid.sum()}/{len(valid)} variants")
    delta = inputs.delta[valid]
    cons = cons[valid]
    genes = inputs.genes[valid]
    y = inputs.labels[valid]

    logP_wt, logP_mut, entropy = cons[:, 0], cons[:, 1], cons[:, 2]
    masked_marginal = logP_wt - logP_mut
    cons_feats = np.column_stack([logP_wt, logP_mut, entropy, masked_marginal])

    axis_associations = family_held_out_axis_analysis(
        delta,
        y,
        genes,
        pfam,
        {
            "masked_marginal": masked_marginal,
            "entropy": entropy,
            "logP_wt": logP_wt,
        },
    )

    requested_seeds = tuple(range(N_SEEDS))
    print(f"\nRunning family-split AUROCs ({len(requested_seeds)} seeds)...")
    feature_sets = {
        "conservation": cons_feats,
        "delta": delta,
        "conservation_plus_delta": np.hstack([cons_feats, delta]),
        "masked_marginal_only": masked_marginal.reshape(-1, 1),
    }
    scored_rows = np.flatnonzero(annotated_gene_mask(genes, pfam))
    auroc = {}
    auroc_ci = {}
    seed_runs_by_feature = {}
    oof_by_feature = {}
    for name, feat in feature_sets.items():
        aggregate, seed_runs, oof_by_seed = auroc_family_split(
            feat, y, genes, pfam, seeds=requested_seeds, collect_oof=compute_ci
        )
        auroc[name] = aggregate
        seed_runs_by_feature[name] = seed_runs
        print(f"  {name:26s} AUROC = {_show_seed_summary(aggregate)}", flush=True)
        if not compute_ci:
            continue
        # The interval is a within-seed cluster bootstrap over the pooled
        # out-of-fold predictions of every requested seed. It is reported in its
        # own field and never mixed with the across-seed aggregate above.
        combined_result = aggregate_seed_oof(
            requested_seeds,
            [
                make_seed_payload_record(
                    seed, oof_by_seed[seed], status=seed_runs[seed]["status"]
                )
                for seed in requested_seeds
            ],
            declared_row_ids=scored_rows,
            declared_labels=y[scored_rows],
            declared_clusters=genes[scored_rows],
            class_order=[0, 1],
            declared_fold_ids=range(N_FOLDS),
        )
        if not combined_result.available:
            auroc_ci[name] = _unavailable_interval(combined_result)
            print(f"    interval unavailable: {combined_result.message}", flush=True)
            continue
        combined = _positive_class_oof(combined_result.payload)
        oof_by_feature[name] = combined
        clusters = family_or_gene_clusters(
            combined["genes"], pfam, is_family_split=True
        )
        interval = binary_auroc_cluster_bootstrap_ci(
            combined, n_resamples=n_boot, seed=0, clusters=clusters
        )
        auroc_ci[name] = interval
        if interval.get("ci_low") is None:
            print("    interval suppressed by the bootstrap", flush=True)
        else:
            print(
                f"    bootstrap {interval['point']:.3f} "
                f"[{interval['ci_low']:.3f}, {interval['ci_high']:.3f}]",
                flush=True,
            )

    print("\n=== PAIRED DIFFERENCES (within model seed, and family-cluster CI) ===")
    contrasts = [
        (
            "delta_added_value_beyond_conservation",
            "conservation_plus_delta",
            "conservation",
        ),
        (
            "descriptive_conservation_beyond_delta",
            "conservation_plus_delta",
            "delta",
        ),
        ("descriptive_conservation_vs_delta", "conservation", "delta"),
    ]
    paired = {}
    paired_ci = {}
    for key, arm_a, arm_b in contrasts:
        label = f"{key}: {arm_a} − {arm_b}"
        arm_a_records = [
            make_seed_record(
                seed,
                seed_runs_by_feature[arm_a][seed]["fold_mean"],
                status=seed_runs_by_feature[arm_a][seed]["status"],
            )
            for seed in requested_seeds
        ]
        arm_b_records = [
            make_seed_record(
                seed,
                seed_runs_by_feature[arm_b][seed]["fold_mean"],
                status=seed_runs_by_feature[arm_b][seed]["status"],
            )
            for seed in requested_seeds
        ]
        diff = aggregate_paired_seed_difference(
            requested_seeds, arm_a_records, arm_b_records
        ).to_dict()
        paired[key] = diff
        print(f"  {label}: {_show_seed_summary(diff)}")
        if not compute_ci:
            continue
        # paired_oof_diff resamples families and stratifies the ranking metric
        # itself; its interval is a within-seed quantity beside the across-seed
        # paired difference above, not a replacement for it.
        diff_ci = paired_oof_diff(
            oof_by_feature.get(arm_a),
            oof_by_feature.get(arm_b),
            pfam,
            label,
            metric="auroc_binary",
            is_family_split=True,
            n_resamples=n_boot,
        )
        if diff_ci is None:
            continue
        paired_ci[key] = diff_ci
        if diff_ci.get("ci_low") is None:
            print(f"    bootstrap diff={diff_ci['point_diff']:+.4f}  CI suppressed")
        else:
            print(
                f"    bootstrap diff={diff_ci['point_diff']:+.4f}  "
                f"[{diff_ci['ci_low']:+.4f}, {diff_ci['ci_high']:+.4f}]  "
                f"({diff_ci['n_clusters']} clusters)"
            )

    cons_a = read_seed_point_estimate(auroc["conservation"]).value
    both_a = read_seed_point_estimate(auroc["conservation_plus_delta"]).value
    delta_a = read_seed_point_estimate(auroc["delta"]).value
    conservation_only_passed = (
        bool(cons_a >= CONSERVATION_ONLY_AUROC_MIN)
        if cons_a is not None and np.isfinite(cons_a)
        else None
    )
    delta_added_diff = paired.get("delta_added_value_beyond_conservation")
    delta_added_read = read_seed_point_estimate(delta_added_diff or {})
    delta_added_passed = (
        bool(delta_added_read.value >= DELTA_ADDED_VALUE_MIN)
        if delta_added_read.available
        else None
    )
    # A verdict compares an interval with the point estimate that interval was
    # built around, so it reads the bootstrap's own point rather than the
    # across-seed mean reported beside it.
    conservation_ci = auroc_ci.get("conservation")
    conservation_ci_point = (
        None if conservation_ci is None else conservation_ci.get("point")
    )
    delta_added_ci = paired_ci.get("delta_added_value_beyond_conservation")
    delta_added_ci_point = (
        None if delta_added_ci is None else delta_added_ci.get("point_diff")
    )
    delta_added_ci_passed = (
        bool(delta_added_ci_point >= DELTA_ADDED_VALUE_MIN)
        if delta_added_ci_point is not None and np.isfinite(delta_added_ci_point)
        else None
    )
    claims = {
        "conservation_only_pathogenicity_auroc": {
            "value": cons_a,
            "threshold": CONSERVATION_ONLY_AUROC_MIN,
            "passed": conservation_only_passed,
            "interval_point_estimate": conservation_ci_point,
            "ci": conservation_ci,
            "verdict": adjudicate_level(
                conservation_ci_point, conservation_ci, CONSERVATION_ONLY_AUROC_MIN
            ),
        },
        "delta_added_value_beyond_conservation": {
            "value": delta_added_read.value,
            "threshold": DELTA_ADDED_VALUE_MIN,
            "conservation": cons_a,
            "conservation_plus_delta": both_a,
            "paired_diff": delta_added_diff,
            "passed": delta_added_passed,
            "interval_point_estimate": delta_added_ci_point,
            "paired_diff_ci": delta_added_ci,
            "verdict": adjudicate_diff(
                delta_added_ci_passed, delta_added_ci, DELTA_ADDED_VALUE_MIN
            ),
        },
        "descriptive_conservation_beyond_delta": {
            "value": read_seed_point_estimate(
                paired.get("descriptive_conservation_beyond_delta", {})
            ).value,
            "delta": delta_a,
            "conservation_plus_delta": both_a,
            "paired_diff": paired.get("descriptive_conservation_beyond_delta"),
            "paired_diff_ci": paired_ci.get("descriptive_conservation_beyond_delta"),
        },
        "descriptive_conservation_vs_delta": {
            "conservation": cons_a,
            "delta": delta_a,
            "paired_diff": paired.get("descriptive_conservation_vs_delta"),
            "paired_diff_ci": paired_ci.get("descriptive_conservation_vs_delta"),
        },
    }

    provenance = pathogenicity_geometry_provenance(inputs, pfam)
    provenance.update(
        {
            "conservation_cache_fingerprint": conservation_meta["fingerprint"],
            "conservation_array_fingerprint": conservation_meta[
                "conservation_array_fingerprint"
            ],
            "conservation_sequence_fingerprint": conservation_meta[
                "sequence_fingerprint"
            ],
        }
    )
    result = {
        **aggregate_result_contract(),
        "n_valid": int(valid.sum()),
        "axis_conservation_correlations_family_held_out": axis_associations[
            "correlations"
        ],
        "auroc_family_split": auroc,
        "auroc_family_split_ci": auroc_ci,
        "auroc_family_split_per_seed_fold_summaries": seed_runs_by_feature,
        "paired_difference_ci": paired_ci,
        "claims": claims,
        "thresholds": {
            "conservation_only_pathogenicity_auroc": CONSERVATION_ONLY_AUROC_MIN,
            "delta_added_value_beyond_conservation": DELTA_ADDED_VALUE_MIN,
        },
        "input_provenance": provenance,
        "calibration_note": (
            "The probes are uncalibrated and measure discrimination only; the "
            "reported AUROCs are not risk estimates."
        ),
        "inference": {
            "estimate_basis": "mean of one held-out-fold summary per model seed",
            "interval_basis": (
                "family-cluster bootstrap over the pooled out-of-fold predictions "
                "of every requested seed, scored within fold"
            ),
            "interval_resampling_unit": "pfam_family",
            "interval_computed": bool(compute_ci),
        },
    }
    write_result_json(CONSERVATION_AXIS_JSON, result, seeds=list(requested_seeds))

    print("\n" + "=" * 60)
    print("CONSERVATION DECIDER")
    print("=" * 60)
    print("  Family-held-out axis correlations:")
    for name, summary in axis_associations["correlations"].items():
        print(f"    {name:20s} rho = {format_axis_summary(summary)}")
    print("\n  AUROC (family-split across model seeds, and bootstrap interval):")
    for name, summary in auroc.items():
        print(
            f"    {name:26s} {_show_seed_summary(summary)}  "
            f"{_show_interval(auroc_ci.get(name))}"
        )
    print(
        f"\n  conservation alone >= {CONSERVATION_ONLY_AUROC_MIN}: "
        f"{_show_seed_summary(auroc['conservation'])} "
        f"{_show_interval(conservation_ci)} -> "
        f"{claims['2D_conservation_clears_0.85']['verdict']}"
    )
    delta_added_value = claims["delta_added_value_beyond_conservation"]["value"]
    delta_added_shown = (
        f"{delta_added_value:+.3f}"
        if delta_added_value is not None
        else "no point estimate"
    )
    print(
        f"  delta adds over conservation >= {DELTA_ADDED_VALUE_MIN}: {delta_added_shown}"
    )
    print(f"     {claims['delta_added_value_beyond_conservation']['verdict']}")
    print(f"\nResults -> {CONSERVATION_AXIS_JSON}")


def _show_seed_summary(summary):
    metric = read_seed_point_estimate(summary)
    if not metric.available:
        return f"unavailable ({metric.message})"
    if metric.spread is None:
        return f"{metric.value:.3f} (seed spread unavailable)"
    return f"{metric.value:.3f} ± {metric.spread:.3f} seed SD"


def _show_interval(interval):
    """Render one bootstrap interval, which is not the seed spread beside it."""
    if interval is None:
        return "(no interval computed)"
    if interval.get("ci_low") is None:
        return "(CI suppressed)"
    return f"[{interval['ci_low']:.3f}, {interval['ci_high']:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--extract",
        action="store_true",
        help="Phase 1 masked-LL extraction (needs GPU)",
    )
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument(
        "--no_ci", action="store_true", help="skip the family-cluster bootstrap CIs"
    )
    ap.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    args = ap.parse_args()

    if args.extract:
        with open(VARIANTS) as _f:
            variants = json.load(_f)
        with open(SEQS) as _f:
            seqs = json.load(_f)
        print(f"Variants: {len(variants)}  Sequences available: {len(seqs)}")
        extract_conservation(variants, seqs, batch_size=args.batch_size)
        return
    analyse(compute_ci=not args.no_ci, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
