"""Pathogenicity positive control: embed ClinVar variants and probe pathogenic-vs-benign as a pipeline check."""

from __future__ import annotations

import argparse
import functools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from esm2_mech.utils.bootstrap import binary_auroc_cluster_bootstrap_ci, family_or_gene_clusters
from esm2_mech.utils.constants import BOOTSTRAP_N_RESAMPLES, N_SEEDS
from esm2_mech.utils.data import variants_fingerprint
from esm2_mech.utils.embed import get_esm2_embeddings_for_pairs
from esm2_mech.utils.io import atomic_write_json, save_npy
from esm2_mech.utils.paths import (
    CLINVAR_PATHOGENICITY_VARIANTS_JSON,
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

print = functools.partial(print, flush=True)

ESM2_MODEL_650M = "esm2_t33_650M_UR50D"


def load_fetched_variants():
    """Load the variant set written by fetch_pathogenicity_variants.py."""
    if not CLINVAR_PATHOGENICITY_VARIANTS_JSON.exists():
        raise FileNotFoundError(
            f"{CLINVAR_PATHOGENICITY_VARIANTS_JSON} not found. Run "
            "`python -m esm2_mech.fetch_data.fetch_pathogenicity_variants` first "
            "(locally — it's network-only, no GPU needed) and copy its output here."
        )
    with open(CLINVAR_PATHOGENICITY_VARIANTS_JSON) as f:
        return json.load(f)


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
    return valid_indices, valid, wt_seqs, mut_seqs, positions


def _embeddings_complete(valid_fingerprint, n_valid):
    """True only if cached embeddings match the current valid set by content fingerprint."""
    if not all(p.exists() for p in [PATH_EMB_WT_MEAN, PATH_EMB_MUT_MEAN, PATH_EMB_META]):
        return False
    try:
        with open(PATH_EMB_META) as f:
            meta = json.load(f)
    except json.JSONDecodeError:
        print(f"  WARNING: corrupt {PATH_EMB_META} — re-extracting")
        PATH_EMB_META.unlink()
        return False
    n_on_disk = np.load(PATH_EMB_WT_MEAN, mmap_mode="r").shape[0]
    return (
        n_on_disk == n_valid == meta.get("n_valid")
        and meta.get("fingerprint") == valid_fingerprint
    )


def embed_phase(variants, model, batch_size):
    """Phase 1. Extract and cache pathogenicity embeddings (GPU)."""
    print("\n=== Phase 1: extract ESM-2 embeddings ===")
    with open(SEQUENCES_JSON) as f:
        seq_cache = json.load(f)

    valid_indices, valid, wt_seqs, mut_seqs, positions = _build_valid_pairs_indexed(
        variants, seq_cache
    )
    valid_fingerprint = variants_fingerprint(valid)

    if _embeddings_complete(valid_fingerprint, len(valid)):
        print("  Embeddings already complete — skipping extraction.")
        return

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device} | Model: {model}")
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    wt_mean, mut_mean, _, _ = get_esm2_embeddings_for_pairs(
        wt_seqs, mut_seqs, positions,
        valid_variants=valid, out_dir=None,
        model_name=model, device=device, batch_size=batch_size,
    )

    save_npy(str(PATH_EMB_WT_MEAN), wt_mean)
    save_npy(str(PATH_EMB_MUT_MEAN), mut_mean)
    atomic_write_json(
        PATH_EMB_META,
        {
            "valid_indices": valid_indices,
            "n": len(variants),
            "n_valid": len(valid),
            "fingerprint": valid_fingerprint,
            "model": model,
        },
    )
    print(f"  Saved {wt_mean.shape} -> {EMB_DIR}")


# ===========================================================================
# Phase 2 — 5-seed probes
# ===========================================================================
def probe_phase(variants, n_seeds, n_jobs=-1, compute_ci=True, n_boot=BOOTSTRAP_N_RESAMPLES):
    """Phase 2. 5-seed logreg + MLP probes on delta_mean and wt_only."""
    print("\n=== Phase 2: probes ===")
    with open(PATH_EMB_META) as f:
        meta = json.load(f)
    wt_mean = np.load(PATH_EMB_WT_MEAN)
    mut_mean = np.load(PATH_EMB_MUT_MEAN)

    # valid_indices index into the variant list that produced the embeddings.
    valid = [variants[i] for i in meta["valid_indices"]]
    if not (len(valid) == wt_mean.shape[0] == mut_mean.shape[0]):
        raise ValueError(
            f"Row mismatch: {len(valid)} variants vs {wt_mean.shape[0]} embedding rows."
        )

    # Verify by content, not count: the embeddings must have been built from
    # exactly these variants in this order. A seed/cap change that yields a
    # colliding count would otherwise misalign labels/genes to embedding rows.
    if variants_fingerprint(valid) != meta.get("fingerprint"):
        raise ValueError(
            "Embedding fingerprint does not match the current variant set — the "
            f"embedding cache is stale. Delete {PATH_EMB_META.name} and the "
            "pathogenicity_*.npy files to re-extract."
        )

    delta = mut_mean - wt_mean
    y = np.array([1 if v["label"] == "pathogenic" else 0 for v in valid])
    genes = np.array([v["gene"] for v in valid])
    with open(PFAM_JSON) as f:
        pfam_map = json.load(f)

    print(f"  {len(valid)} variants  pathogenic={int(y.sum())} benign={int((1 - y).sum())}  "
          f"{len(set(genes))} genes")

    features = {"delta_mean": delta, "wt_only": wt_mean}
    probes = {"logreg": run_logreg_binary_cv, "mlp": run_mlp_binary_cv}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run one seed at a time, writing that seed's per-cell AUROCs to its own file
    # as it completes (resume + progress visibility). Within a seed the 8 cells
    # (2 features x 2 probes x 2 splits) are independent, so they run in parallel.
    # Cache freshness is keyed on more than existence: a seed file built with
    # --no_ci (or a different --n_boot) looks complete on disk but is missing
    # CIs a later CI-on run needs, so the params that produced each file are
    # stored alongside it and checked before treating it as reusable.
    seed_params = {
        "compute_ci": compute_ci,
        "n_boot": n_boot if compute_ci else None,
        "variant_fingerprint": meta["fingerprint"],
        "model": meta.get("model"),
    }
    for seed in range(n_seeds):
        seed_path = Path(PATHOGENICITY_CONTROL_SEED_JSON.format(seed=seed))
        if seed_path.exists():
            cached_seed_params = None
            try:
                with open(seed_path) as f:
                    cached_seed_params = json.load(f).get("_params")
            except json.JSONDecodeError:
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
        cells = [
            (fname, pname, split_name, splits)
            for fname in features
            for pname in probes
            for split_name, splits in (("gene", gs), ("family", fs))
        ]

        def _run_cell(fname, pname, split_name, splits, seed=seed):
            res, oof = probes[pname](
                features[fname], y, splits, seed=seed, genes=genes, return_oof=True
            )
            # CI from seed 0's OOF only: each seed reshuffles the CV fold
            # assignment, and the aggregation step below only ever keeps seed
            # 0's CI (matching megascale_stability.py / magnitude_direction.py's
            # R7.5 "seed 0" convention) — computing it for every seed would be
            # pure waste, since seeds 1..n-1's bootstrap runs are discarded.
            if compute_ci and oof is not None and seed == 0:
                clusters = family_or_gene_clusters(
                    oof["genes"], pfam_map, is_family_split=(split_name == "family")
                )
                ci = binary_auroc_cluster_bootstrap_ci(
                    oof, n_resamples=n_boot, seed=seed, clusters=clusters
                )
            else:
                ci = None
            return (fname, pname, split_name, res.get("auroc_mean", float("nan")), ci)

        outcomes = Parallel(n_jobs=n_jobs)(
            delayed(_run_cell)(*c) for c in cells
        )

        # Keep NaN (undefined AUROC: a cell with no valid fold) out of the JSON.
        # json.dump emits the bare token `NaN`, which is invalid JSON; store null
        # so absent data round-trips as None.
        seed_result = {"_params": seed_params}
        for fname, pname, split_name, auroc, ci in outcomes:
            key = f"{fname}_{pname}_{split_name}"
            seed_result[key] = None if np.isnan(auroc) else auroc
            seed_result[f"{key}_ci"] = ci
        atomic_write_json(seed_path, seed_result, indent=2)
        summary = "  ".join(
            f"{k}={v:.3f}" for k, v in seed_result.items()
            if "mlp" in k and not k.endswith("_ci") and v is not None
        )
        print(f"  seed {seed} done -> {seed_path.name}   {summary}")

    # Aggregate per-seed files into the final mean ± std result.
    per_cell = defaultdict(list)
    seed0_ci = {}
    for seed in range(n_seeds):
        seed_path = Path(PATHOGENICITY_CONTROL_SEED_JSON.format(seed=seed))
        try:
            with open(seed_path) as f:
                seed_result = json.load(f)
        except json.JSONDecodeError:
            # Partial write on interrupt. Delete so the next run recomputes this
            # seed rather than aggregating a truncated file.
            print(f"  WARNING: corrupt {seed_path.name} — deleting; re-run to recompute this seed")
            seed_path.unlink()
            raise
        for key, value in seed_result.items():
            if key == "_params":
                continue
            if key.endswith("_ci"):
                if seed == 0:
                    seed0_ci[key[: -len("_ci")]] = value
                continue
            per_cell[key].append(value)

    results = {
        "n_variants": int(len(valid)),
        "n_pathogenic": int(y.sum()),
        "n_benign": int((1 - y).sum()),
        "n_genes": int(len(set(genes))),
        "n_seeds": n_seeds,
        "by_feature": {},
    }
    for fname in features:
        results["by_feature"][fname] = {}
        for pname in probes:
            for split_name in ("gene", "family"):
                key = f"{fname}_{pname}_{split_name}"
                vals = [v for v in per_cell[key] if v is not None]
                mean = float(np.mean(vals)) if vals else None
                std = float(np.std(vals)) if vals else None
                cell = {
                    "auroc_mean": mean,
                    "auroc_std": std,
                    "per_seed": per_cell[key],
                }
                # Cluster-bootstrap CI from seed 0's OOF only (R7.5's "seed 0"
                # convention): each seed reshuffles the CV fold assignment, so a
                # single seed's CI is the coherent unit rather than merging OOF
                # across seeds' differing folds.
                if compute_ci and key in seed0_ci:
                    cell["ci"] = seed0_ci[key]
                results["by_feature"][fname][f"{pname}_{split_name}"] = cell

    atomic_write_json(PATHOGENICITY_CONTROL_JSON, results, indent=2)
    print(f"  Aggregated results written to {PATHOGENICITY_CONTROL_JSON}")
    return results


def _print_headline(results):
    print("\n" + "=" * 60)
    print("HEADLINE — pathogenicity positive control")
    print("=" * 60)

    def cell(feature, key):
        c = results["by_feature"][feature][key]
        return c["auroc_mean"], c["auroc_std"]

    for feature in ("delta_mean", "wt_only"):
        for key in ("logreg_gene", "logreg_family", "mlp_gene", "mlp_family"):
            mean, std = cell(feature, key)
            if mean is None:
                print(f"  {feature:11s} {key:14s} AUROC = undefined (no valid fold)")
            else:
                print(f"  {feature:11s} {key:14s} AUROC = {mean:.3f} ± {std:.3f}")
        print()

    d_mean, _ = cell("delta_mean", "mlp_gene")
    d_fam, _ = cell("delta_mean", "mlp_family")
    if d_mean is None or d_fam is None:
        print("  delta_mean MLP gene/family AUROC is undefined — cannot evaluate the control.")
        return
    print(f"  delta_mean MLP gene→family Δ = {d_mean - d_fam:+.3f}")
    if d_mean >= 0.85:
        print("  ⇒ Pipeline PASSES positive control (delta MLP AUROC ≥ 0.85).")
        print("    The mechanism null is a real absence of signal, not a pipeline failure.")
    elif d_mean >= 0.70:
        print("  ⇒ MODERATE pathogenicity signal (0.70–0.85). Interpretable but weak.")
    else:
        print("  ⇒ Pipeline FAILS positive control (< 0.70). Mechanism null UNINTERPRETABLE.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=ESM2_MODEL_650M, choices=[ESM2_MODEL_650M])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=N_SEEDS, help="number of probe seeds (>=1)")
    parser.add_argument("--n_jobs", type=int, default=-1, help="parallel jobs for probes (-1 = all cores)")
    parser.add_argument("--no_ci", action="store_true", help="skip cluster-bootstrap CIs")
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    args = parser.parse_args()

    if args.seeds < 1:
        parser.error("--seeds must be >= 1")

    variants = load_fetched_variants()
    embed_phase(variants, model=args.model, batch_size=args.batch_size)
    results = probe_phase(
        variants, n_seeds=args.seeds, n_jobs=args.n_jobs,
        compute_ci=not args.no_ci, n_boot=args.n_boot,
    )
    _print_headline(results)


if __name__ == "__main__":
    main()
