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
from esm2_mech.utils.constants import AA_ORDER
from esm2_mech.utils.metrics import mean_std_n
from esm2_mech.utils.probes import run_logreg_binary_cv
from esm2_mech.utils.sequences import window_sequence
from esm2_mech.utils.splits import family_split_cv
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

# Pre-registered thresholds
CLAIM_2D_CONSERVATION_MIN = 0.85
CLAIM_2E_DELTA_ADD_MIN = 0.02

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
            and meta.get("coverage")
            == int(np.isfinite(cached).all(axis=1).sum())
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


def _oof_one_seed(X, y, genes, pfam, seed):
    """Family-split out-of-fold positive-class probabilities for one seed."""
    splits = list(family_split_cv(genes, pfam, seed=seed))
    aggregate, oof = run_logreg_binary_cv(
        X,
        y,
        splits,
        seed=seed,
        pos_label=PATHOGENIC,
        genes=genes,
        return_oof=True,
        max_iter=LOGREG_MAX_ITER,
    )
    return {
        "seed": int(seed),
        "fold_mean": aggregate.get("auroc_mean"),
        "oof": oof,
    }


def auroc_family_split(X, y, genes, pfam, seeds=range(5), n_jobs=-1):
    """Seed-0 family-split inference plus a five-seed descriptive summary."""
    seeds = tuple(seeds)
    seed_runs = Parallel(n_jobs=n_jobs)(
        delayed(_oof_one_seed)(X, y, genes, pfam, seed) for seed in seeds
    )
    seed0_run = next((run for run in seed_runs if run["seed"] == 0), None)
    if seed0_run is None or seed0_run["oof"] is None:
        raise RuntimeError("conservation inference requires seed-0 OOF predictions")

    seed0_oof = seed0_run["oof"]
    clusters = family_or_gene_clusters(
        seed0_oof["genes"], pfam, is_family_split=True
    )
    ci = binary_auroc_cluster_bootstrap_ci(seed0_oof, clusters=clusters, seed=0)
    point = ci["point"]
    fold_mean = seed0_run["fold_mean"]
    if point is None or fold_mean is None or not np.isclose(point, fold_mean):
        raise RuntimeError(
            "seed-0 conservation AUROC point and bootstrap estimand disagree"
        )

    per_seed_values = [run["fold_mean"] for run in seed_runs]
    mean, std, count = mean_std_n(per_seed_values)
    descriptive = {
        "across_seed_mean": mean,
        "across_seed_std": std,
        "n_seeds": count,
        "per_seed": [
            {"seed": run["seed"], "fold_mean": run["fold_mean"]}
            for run in seed_runs
        ],
    }
    return point, ci, seed0_oof, descriptive


def analyse():
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

    print("\nRunning family-split AUROCs (5 seeds)...")
    feature_sets = {
        "conservation": cons_feats,
        "delta": delta,
        "conservation_plus_delta": np.hstack([cons_feats, delta]),
        "masked_marginal_only": masked_marginal.reshape(-1, 1),
    }
    auroc = {}
    auroc_ci = {}
    auroc_descriptive = {}
    oof_by_feature = {}
    for name, feat in feature_sets.items():
        value, ci, oof, descriptive = auroc_family_split(feat, y, genes, pfam)
        auroc[name] = value
        auroc_ci[name] = ci
        auroc_descriptive[name] = descriptive
        oof_by_feature[name] = oof
        if ci is None or ci.get("ci_low") is None:
            print(f"  {name:26s} AUROC = {value:.3f}  (CI suppressed)", flush=True)
        else:
            print(
                f"  {name:26s} AUROC = {value:.3f} "
                f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]",
                flush=True,
            )

    print("\n=== PAIRED DIFFERENCES (family-cluster bootstrap) ===")
    contrasts = [
        (
            "2E_delta_beyond_conservation",
            "conservation_plus_delta",
            "conservation",
            CLAIM_2E_DELTA_ADD_MIN,
        ),
        (
            "descriptive_conservation_beyond_delta",
            "conservation_plus_delta",
            "delta",
            0.0,
        ),
        ("descriptive_conservation_vs_delta", "conservation", "delta", 0.0),
    ]
    paired = {}
    for key, arm_a, arm_b, threshold in contrasts:
        label = f"{key}: {arm_a} − {arm_b}"
        diff = paired_oof_diff(
            oof_by_feature.get(arm_a),
            oof_by_feature.get(arm_b),
            pfam,
            label,
            metric="auroc_binary",
            is_family_split=True,
        )
        if diff is None:
            continue
        paired[key] = diff
        if diff.get("ci_low") is None:
            print(f"  {label}: diff={diff['point_diff']:+.4f}  CI suppressed")
        else:
            print(
                f"  {label}: diff={diff['point_diff']:+.4f}  "
                f"[{diff['ci_low']:+.4f}, {diff['ci_high']:+.4f}]  "
                f"({diff['n_clusters']} clusters)"
            )

    cons_a = auroc["conservation"]
    both_a = auroc["conservation_plus_delta"]
    delta_a = auroc["delta"]
    claim_2d_passed = (
        bool(cons_a >= CLAIM_2D_CONSERVATION_MIN) if np.isfinite(cons_a) else None
    )
    claim_2e_diff = paired.get("2E_delta_beyond_conservation")
    claim_2e_passed = (
        bool(claim_2e_diff["point_diff"] >= CLAIM_2E_DELTA_ADD_MIN)
        if claim_2e_diff and claim_2e_diff.get("point_diff") is not None
        else None
    )
    claims = {
        "2D_conservation_clears_0.85": {
            "seed": 0,
            "value": cons_a,
            "threshold": CLAIM_2D_CONSERVATION_MIN,
            "ci": auroc_ci["conservation"],
            "passed": claim_2d_passed,
            "verdict": adjudicate_level(
                cons_a, auroc_ci["conservation"], CLAIM_2D_CONSERVATION_MIN
            ),
        },
        "2E_delta_beyond_conservation": {
            "seed": 0,
            "value": claim_2e_diff["point_diff"] if claim_2e_diff else None,
            "threshold": CLAIM_2E_DELTA_ADD_MIN,
            "conservation": cons_a,
            "conservation_plus_delta": both_a,
            "paired_diff": claim_2e_diff,
            "passed": claim_2e_passed,
            "verdict": adjudicate_diff(
                claim_2e_passed, claim_2e_diff, CLAIM_2E_DELTA_ADD_MIN
            ),
        },
        "descriptive_conservation_beyond_delta": {
            "value": (
                paired["descriptive_conservation_beyond_delta"]["point_diff"]
                if "descriptive_conservation_beyond_delta" in paired
                else None
            ),
            "delta": delta_a,
            "conservation_plus_delta": both_a,
            "paired_diff": paired.get("descriptive_conservation_beyond_delta"),
        },
        "descriptive_conservation_vs_delta": {
            "conservation": cons_a,
            "delta": delta_a,
            "paired_diff": paired.get("descriptive_conservation_vs_delta"),
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
        "n_valid": int(valid.sum()),
        "axis_conservation_correlations_family_held_out": axis_associations[
            "correlations"
        ],
        "auroc_family_split": auroc,
        "auroc_family_split_ci": auroc_ci,
        "auroc_family_split_five_seed_descriptive": auroc_descriptive,
        "claims": claims,
        "thresholds": {"2D": CLAIM_2D_CONSERVATION_MIN, "2E": CLAIM_2E_DELTA_ADD_MIN},
        "input_provenance": provenance,
        "calibration_note": (
            "The probes are uncalibrated and measure discrimination only; the "
            "reported AUROCs are not risk estimates (pre-registration §1.4)."
        ),
        "inference": {
            "seed": 0,
            "estimate_basis": "mean of seed-0 held-out-fold AUROCs",
            "resampling_unit": "pfam_family",
            "five_seed_role": "descriptive",
        },
    }
    write_result_json(CONSERVATION_AXIS_JSON, result, seeds=list(range(5)))

    print("\n" + "=" * 60)
    print("CONSERVATION DECIDER")
    print("=" * 60)
    print("  Family-held-out axis correlations:")
    for name, summary in axis_associations["correlations"].items():
        print(f"    {name:20s} rho = {format_axis_summary(summary)}")
    print("\n  AUROC (family-split, seed-0 inference):")
    for name, value in auroc.items():
        ci = auroc_ci.get(name)
        interval = (
            f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"
            if ci and ci.get("ci_low") is not None
            else "(CI suppressed)"
        )
        print(f"    {name:26s} {value:.3f} {interval}")
    print(
        f"\n  2D conservation-alone >= {CLAIM_2D_CONSERVATION_MIN}: {cons_a:.3f} -> "
        f"{claims['2D_conservation_clears_0.85']['verdict']}"
    )
    claim_2e_value = claims["2E_delta_beyond_conservation"]["value"]
    claim_2e_shown = (
        f"{claim_2e_value:+.3f}" if claim_2e_value is not None else "no point estimate"
    )
    print(
        f"  2E delta adds over conservation >= {CLAIM_2E_DELTA_ADD_MIN}: {claim_2e_shown}"
    )
    print(f"     {claims['2E_delta_beyond_conservation']['verdict']}")
    print(f"\nResults -> {CONSERVATION_AXIS_JSON}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--extract",
        action="store_true",
        help="Phase 1 masked-LL extraction (needs GPU)",
    )
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    if args.extract:
        with open(VARIANTS) as _f:
            variants = json.load(_f)
        with open(SEQS) as _f:
            seqs = json.load(_f)
        print(f"Variants: {len(variants)}  Sequences available: {len(seqs)}")
        extract_conservation(variants, seqs, batch_size=args.batch_size)
        return
    analyse()


if __name__ == "__main__":
    main()
