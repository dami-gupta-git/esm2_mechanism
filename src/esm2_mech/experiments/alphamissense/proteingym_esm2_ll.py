"""ESM-2 masked-LM delta-LL scoring on ProteinGym human DMS assays, with per-assay Spearman/AUROC."""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

print = functools.partial(print, flush=True)

from esm2_mech.utils.constants import AA_ORDER, MAX_SEQ_LEN
from esm2_mech.utils.io import atomic_write_json
from esm2_mech.utils.paths import (
    PFAM_JSON,
    PROTEINGYM_CACHE_DIR,
    RESULTS_DIR as _RESULTS_DIR,
)
from esm2_mech.utils.sequences import window_sequence

PG_DIR = PROTEINGYM_CACHE_DIR
OUT = _RESULTS_DIR / "proteingym_esm2_ll"

DMS_INDEX = PG_DIR / "DMS_substitutions.csv"
DMS_SUBDIR = PG_DIR / "DMS_ProteinGym_substitutions"
JOBS_CACHE = PG_DIR / "esm2_ll_jobs.json"
JOBS_PARAMS_JSON = PG_DIR / "esm2_ll_jobs.params.json"
SCORE_CACHE = PG_DIR / "esm2_ll_scores.json"
SCORE_PARAMS_JSON = PG_DIR / "esm2_ll_scores.params.json"

# Bump when the parsing or scoring logic changes in a way that makes a cache
# built before the change invalid even though its recorded inputs are unchanged.
JOBS_CACHE_VERSION = 1
SCORE_CACHE_VERSION = 1
AM_CACHE = PG_DIR / "am_scores_proteingym.json"

CHECKPOINT_EVERY = 10  # assays between GPU saves
MIN_VARIANTS = 20  # minimum scored variants to include an assay

# Gate thresholds. Named here so the printed criterion and the comparison cannot
# drift apart, and so it is visible that they are fixed choices rather than
# quantities recomputed from this run.
G1_MIN_MEDIAN_SPEARMAN = 0.40
G2_MAX_FRAC_BELOW_020 = 0.25
G3_MIN_MEDIAN_GAP_OVER_AM = 0.05


def parse_mutant(mut_str: str) -> tuple[str, int, str] | None:
    """Parse 'A673C' into (wt_aa, pos, mut_aa) or None."""
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", mut_str.strip())
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3)


def _load_cache_if_current(cache_path: Path, params_path: Path, params: dict):
    """Return the cached JSON only when its recorded inputs match `params`.

    Both caches were previously keyed on file existence alone, so a changed
    assay list, window length or model silently reused a stale file.
    """
    if not cache_path.exists() or not params_path.exists():
        return None
    try:
        with open(params_path) as handle:
            stored = json.load(handle)
    except json.JSONDecodeError:
        print(f"  {params_path.name} is corrupt — rebuilding {cache_path.name}")
        return None
    if stored != params:
        print(f"  {cache_path.name} was built from different inputs — rebuilding")
        return None
    with open(cache_path) as handle:
        return json.load(handle)


def _save_cache(cache_path: Path, params_path: Path, payload, params: dict) -> None:
    """Write the cache, then the params that produced it."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    atomic_write_json(params_path, params)


def _jobs_cache_params() -> dict:
    """Inputs that determine the parsed job list.

    The DMS index is hashed by content. The per-assay files are recorded by name
    and size, which catches an added, removed or resized assay file but not an
    edit that preserves the byte count — bump JOBS_CACHE_VERSION for that.
    """
    index_digest = hashlib.sha256(DMS_INDEX.read_bytes()).hexdigest()
    assay_files = sorted(
        (path.name, path.stat().st_size)
        for path in DMS_SUBDIR.glob("*.csv")
    )
    return {
        "version": JOBS_CACHE_VERSION,
        "dms_index_sha256": index_digest,
        "assay_files": assay_files,
        "min_variants": MIN_VARIANTS,
    }


def _score_cache_params(jobs: list[dict], model_name: str) -> dict:
    """Every input that determines a delta-LL score.

    The job list is hashed by content, so a changed assay, a changed variant list
    or a reordering all invalidate the cache.
    """
    digest = hashlib.sha256()
    for job in jobs:
        digest.update(job["DMS_id"].encode())
        digest.update(b"\x00")
        for variant in job["variants"]:
            digest.update(variant["mutant"].encode())
            digest.update(b"\x00")
    return {
        "version": SCORE_CACHE_VERSION,
        "jobs_sha256": digest.hexdigest(),
        "model": model_name,
        "max_seq_len": MAX_SEQ_LEN,
    }


def phase1_build_jobs() -> list[dict]:
    """Parse human DMS assays into a cached job list of single-mutant variants."""
    params = _jobs_cache_params()
    jobs = _load_cache_if_current(JOBS_CACHE, JOBS_PARAMS_JSON, params)
    if jobs is not None:
        print(f"Loading cached jobs from {JOBS_CACHE}")
        print(f"  {len(jobs)} assays")
        return jobs

    assays = list(csv.DictReader(open(DMS_INDEX)))
    human = [a for a in assays if a["taxon"] == "Human"]
    print(f"Human assays: {len(human)}")

    jobs = []
    skipped = 0
    for a in human:
        dms_path = DMS_SUBDIR / a["DMS_filename"]
        if not dms_path.exists():
            print(f"  SKIP {a['DMS_id']}: DMS file not found")
            skipped += 1
            continue

        try:
            df = pd.read_csv(dms_path, usecols=["mutant", "DMS_score", "DMS_score_bin"])
        except Exception as e:
            print(f"  SKIP {a['DMS_id']}: {e}")
            skipped += 1
            continue

        df = df[~df["mutant"].astype(str).str.contains(":")].reset_index(drop=True)
        if df.empty:
            skipped += 1
            continue

        wt_seq = a["target_seq"].strip()
        variants = []
        parse_failures = 0
        for _, row in df.iterrows():
            parsed = parse_mutant(str(row["mutant"]))
            if parsed is None:
                parse_failures += 1
                continue
            wt_aa, pos, mut_aa = parsed
            # Validate against target_seq (positions are 1-indexed)
            if pos < 1 or pos > len(wt_seq):
                parse_failures += 1
                continue
            seq_wt_aa = wt_seq[pos - 1].upper()
            if seq_wt_aa != wt_aa:
                # Some assays use an offset that shifts positions; skip mismatches.
                parse_failures += 1
                continue
            variants.append(
                {
                    "mutant": str(row["mutant"]),
                    "pos": pos,
                    "wt_aa": wt_aa,
                    "mut_aa": mut_aa,
                    "DMS_score": float(row["DMS_score"]),
                    "DMS_score_bin": int(row["DMS_score_bin"]),
                }
            )

        if not variants:
            print(
                f"  SKIP {a['DMS_id']}: no parseable variants (parse_failures={parse_failures})"
            )
            skipped += 1
            continue

        jobs.append(
            {
                "DMS_id": a["DMS_id"],
                "UniProt_ID": a["UniProt_ID"],
                "molecule_name": a.get("molecule_name", ""),
                "coarse_selection_type": a["coarse_selection_type"],
                "seq_len": int(a["seq_len"]),
                "wt_seq": wt_seq,
                "variants": variants,
                "parse_failures": parse_failures,
            }
        )
        print(f"  {a['DMS_id']}: {len(variants)} variants, {parse_failures} skipped")

    print(f"\nJobs built: {len(jobs)} assays, {skipped} skipped")
    OUT.mkdir(parents=True, exist_ok=True)
    _save_cache(JOBS_CACHE, JOBS_PARAMS_JSON, jobs, params)
    print(f"Saved → {JOBS_CACHE}")
    return jobs


def phase2_extract_ll(
    jobs: list[dict], batch_size: int = 32
) -> dict[str, dict[str, float]]:
    """Compute masked-LM delta-LL for all variants on GPU; returns {DMS_id: {mutant: dll}}."""
    import torch
    import esm as esm_lib

    from esm2_mech.embeddings.embed_variants import ESM2_MODEL_650M

    params = _score_cache_params(jobs, ESM2_MODEL_650M)
    cached = _load_cache_if_current(SCORE_CACHE, SCORE_PARAMS_JSON, params)
    if cached is not None:
        print(f"Cached LL scores found: {SCORE_CACHE}")
        return cached

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Phase 2 requires a GPU. Run on RunPod.")

    model, alphabet = esm_lib.pretrained.load_model_and_alphabet(ESM2_MODEL_650M)
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    mask_idx = alphabet.mask_idx
    aa_to_idx = {aa: alphabet.get_idx(aa) for aa in AA_ORDER}

    CKPT = PG_DIR / "esm2_ll_ckpt.json"
    done_ids: set[str] = set()
    all_scores: dict[str, dict[str, float]] = {}

    if CKPT.exists():
        with open(CKPT) as f:
            ckpt = json.load(f)
        done_ids = set(ckpt["done"])
        all_scores = ckpt["scores"]
        print(f"Resuming from checkpoint: {len(done_ids)}/{len(jobs)} assays done")

    remaining = [j for j in jobs if j["DMS_id"] not in done_ids]
    print(f"Extracting ΔLL for {len(remaining)} assays on {device}...")

    for assay_num, job in enumerate(remaining):
        dms_id = job["DMS_id"]
        wt_seq = job["wt_seq"]
        variants = job["variants"]

        pos_to_variants: dict[int, list[dict]] = defaultdict(list)
        for v in variants:
            pos_to_variants[v["pos"]].append(v)

        assay_scores: dict[str, float] = {}

        # Build batch: one masked sequence per unique position
        pos_list = sorted(pos_to_variants.keys())
        for batch_start in range(0, len(pos_list), batch_size):
            batch_pos = pos_list[batch_start : batch_start + batch_size]
            batch_data = []
            batch_meta = []  # (pos, wt_aa, windowed_new_pos)

            for pos in batch_pos:
                win_seq, new_pos, _start = window_sequence(wt_seq, pos)
                masked = list(win_seq)
                masked[new_pos - 1] = "<mask>"
                masked_str = "".join(masked)
                batch_data.append((f"{dms_id}_{pos}", masked_str))
                wt_aa = pos_to_variants[pos][0]["wt_aa"]
                batch_meta.append((pos, wt_aa, new_pos))

            _, _, tokens = batch_converter(batch_data)
            tokens = tokens.to(device)

            with torch.inference_mode():
                out = model(tokens)
            logits = out["logits"].cpu().float()  # (B, L+2, vocab)

            for i, (pos, wt_aa, new_pos) in enumerate(batch_meta):
                tok_idx = new_pos  # 1-indexed; with BOS at 0, this is correct
                log_probs = torch.log_softmax(logits[i, tok_idx], dim=-1).numpy()

                ll_wt = (
                    float(log_probs[aa_to_idx[wt_aa]])
                    if wt_aa in aa_to_idx
                    else float("nan")
                )

                for v in pos_to_variants[pos]:
                    mut_aa = v["mut_aa"]
                    if mut_aa in aa_to_idx:
                        ll_mut = float(log_probs[aa_to_idx[mut_aa]])
                        delta_ll = ll_wt - ll_mut
                    else:
                        delta_ll = float("nan")
                    assay_scores[v["mutant"]] = delta_ll

        all_scores[dms_id] = assay_scores
        done_ids.add(dms_id)
        n_scored = sum(1 for v in assay_scores.values() if not np.isnan(v))
        print(
            f"  [{assay_num+1}/{len(remaining)}] {dms_id}: {n_scored}/{len(variants)} variants scored"
        )

        if (assay_num + 1) % CHECKPOINT_EVERY == 0:
            with open(CKPT, "w") as f:
                json.dump({"done": list(done_ids), "scores": all_scores}, f)
            print(f"  Checkpoint saved ({len(done_ids)} assays done)")

    _save_cache(SCORE_CACHE, SCORE_PARAMS_JSON, all_scores, params)
    print(f"Saved ΔLL scores → {SCORE_CACHE}")
    if CKPT.exists():
        CKPT.unlink()

    return all_scores


def phase3_analyse(jobs: list[dict], all_scores: dict[str, dict[str, float]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Load Pfam map for family-split analysis
    pfam_path = PFAM_JSON
    pfam_map: dict[str, str] = {}
    if pfam_path.exists():
        with open(pfam_path) as f:
            pfam_map = json.load(f)

    # Load AM scores for comparison (result_18)
    am_scores: dict[tuple[str, str], float] = {}
    if AM_CACHE.exists():
        with open(AM_CACHE) as f:
            raw = json.load(f)
        for k, v in raw.items():
            parts = k.split("|", 1)
            if len(parts) == 2:
                am_scores[tuple(parts)] = float(v)

    per_assay: dict[str, dict] = {}

    for job in jobs:
        dms_id = job["DMS_id"]
        variants = job["variants"]
        scores = all_scores.get(dms_id, {})

        delta_ll_vals, dms_scores, dms_bins = [], [], []
        for v in variants:
            dll = scores.get(v["mutant"], float("nan"))
            if np.isnan(dll):
                continue
            delta_ll_vals.append(dll)
            dms_scores.append(v["DMS_score"])
            dms_bins.append(v["DMS_score_bin"])

        n = len(delta_ll_vals)
        if n < MIN_VARIANTS:
            per_assay[dms_id] = {
                "skipped": True,
                "reason": "too_few_scored",
                "n_scored": n,
            }
            continue

        delta_ll_arr = np.array(delta_ll_vals)
        dms_score_arr = np.array(dms_scores)
        dms_bin_arr = np.array(dms_bins, dtype=int)

        # Negate Spearman so positive means high ΔLL predicts low fitness.
        rho_raw, pval = spearmanr(dms_score_arr, delta_ll_arr)
        spearman = -float(rho_raw)  # positive = agreement (ΔLL predicts low fitness)

        bin_dmg = (dms_bin_arr == 0).astype(int)
        auroc = None
        if len(np.unique(bin_dmg)) >= 2:
            try:
                auroc = float(roc_auc_score(bin_dmg, delta_ll_arr))
            except Exception:
                pass

        # pfam_map is keyed by gene symbol; try molecule_name then mnemonic prefix.
        uniprot_id = job["UniProt_ID"]
        mol_name = job.get("molecule_name", "")
        mnemonic_prefix = uniprot_id.split("_")[0]
        pfam_family = pfam_map.get(mol_name) or pfam_map.get(mnemonic_prefix)

        per_assay[dms_id] = {
            "skipped": False,
            "UniProt_ID": uniprot_id,
            "coarse_selection_type": job["coarse_selection_type"],
            "seq_len": job["seq_len"],
            "n_variants": n,
            "n_total_variants": len(variants),
            "coverage": float(n / len(variants)),
            "spearman": float(spearman),
            "spearman_pval": float(pval),
            "auroc": auroc,
            "pfam_family": pfam_family,
        }

    ok = [v for v in per_assay.values() if not v.get("skipped")]
    skipped = [v for v in per_assay.values() if v.get("skipped")]
    print(f"\nAssays scored: {len(ok)} / {len(jobs)}")
    if skipped:
        reasons: dict[str, int] = {}
        for v in skipped:
            reasons[v.get("reason", "unknown")] = (
                reasons.get(v.get("reason", "unknown"), 0) + 1
            )
        print(f"Skipped: {reasons}")

    rhos = [v["spearman"] for v in ok]
    aurocs = [v["auroc"] for v in ok if v["auroc"] is not None]
    covs = [v["coverage"] for v in ok]

    def dist(xs: list[float], label: str) -> dict:
        if not xs:
            return {"n": 0}
        arr = np.array(xs)
        d = {
            "n": len(xs),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "q25": float(np.quantile(arr, 0.25)),
            "median": float(np.median(arr)),
            "q75": float(np.quantile(arr, 0.75)),
            "max": float(np.max(arr)),
            "frac_below_0_20": float(np.mean(arr < 0.20)),
            "frac_below_0_40": float(np.mean(arr < 0.40)),
        }
        print(
            f"\n{label}: mean={d['mean']:.3f}±{d['std']:.3f}  "
            f"median={d['median']:.3f}  "
            f"frac<0.20={d['frac_below_0_20']:.2f}  "
            f"frac<0.40={d['frac_below_0_40']:.2f}"
        )
        return d

    print("\n=== ESM-2 ΔLL performance ===")
    spearman_dist = dist(rhos, "Spearman ρ (ESM-2 ΔLL)")
    auroc_dist = dist(aurocs, "AUROC     (ESM-2 ΔLL)")

    sel_types = sorted({v["coarse_selection_type"] for v in ok})
    by_sel: dict[str, dict] = {}
    print("\n=== By selection type ===")
    for sel in sel_types:
        sel_rhos = [v["spearman"] for v in ok if v["coarse_selection_type"] == sel]
        if sel_rhos:
            by_sel[sel] = dist(sel_rhos, f"  {sel} (n={len(sel_rhos)})")

    am_per_assay_path = _RESULTS_DIR / "proteingym_alphamissense" / "per_assay.json"
    am_comparison: dict = {}
    if am_per_assay_path.exists():
        with open(am_per_assay_path) as f:
            am_pa = json.load(f)
        am_rhos = [
            v["spearman_neg"]
            for v in am_pa.values()
            if not v.get("skipped") and v.get("spearman_neg") is not None
        ]
        if am_rhos:
            am_comparison = {
                "n_assays": len(am_rhos),
                "median": float(np.median(am_rhos)),
                "mean": float(np.mean(am_rhos)),
                "std": float(np.std(am_rhos)),
                "frac_below_0_20": float(np.mean(np.array(am_rhos) < 0.20)),
            }
            print(f"\n=== AlphaMissense comparison ===")
            print(
                f"  AM   Spearman median={am_comparison['median']:.3f}  "
                f"frac<0.20={am_comparison['frac_below_0_20']:.2f}  (n={len(am_rhos)})"
            )
            print(
                f"  ESM2 Spearman median={spearman_dist.get('median', float('nan')):.3f}  "
                f"frac<0.20={spearman_dist.get('frac_below_0_20', float('nan')):.2f}  (n={len(rhos)})"
            )

    # A gate with nothing to judge is unavailable, not failed. `dist` returns
    # {"n": 0} when no assay scored, so these keys are absent rather than zero;
    # comparing the resulting NaN would silently record every gate as FAIL.
    esm2_median = spearman_dist.get("median")
    esm2_frac020 = spearman_dist.get("frac_below_0_20")
    am_median = am_comparison.get("median")

    g1 = None if esm2_median is None else esm2_median >= G1_MIN_MEDIAN_SPEARMAN
    g2 = None if esm2_frac020 is None else esm2_frac020 <= G2_MAX_FRAC_BELOW_020
    am_gap = (
        None
        if esm2_median is None or am_median is None
        else float(esm2_median - am_median)
    )
    g3 = None if am_gap is None else am_gap >= G3_MIN_MEDIAN_GAP_OVER_AM

    def _verdict(passed, value):
        if passed is None:
            return "UNAVAILABLE (nothing scored)"
        return f"{value:.3f} → {'PASS ✓' if passed else 'FAIL ✗'}"

    print("\n=== DECISION RULES ===")
    print(
        f"  G1: median Spearman ≥ {G1_MIN_MEDIAN_SPEARMAN} → "
        f"{_verdict(g1, esm2_median)}"
    )
    print(
        f"  G2: frac ρ<0.20 ≤ {G2_MAX_FRAC_BELOW_020} → "
        f"{_verdict(g2, esm2_frac020)}"
    )
    print(
        f"  G3: ESM2 median − AM median ≥ {G3_MIN_MEDIAN_GAP_OVER_AM} → "
        f"{_verdict(g3, am_gap)}"
    )

    ranked = sorted(
        [(k, v) for k, v in per_assay.items() if not v.get("skipped")],
        key=lambda kv: kv[1]["spearman"],
    )
    print("\nWORST 5 (Spearman):")
    for k, v in ranked[:5]:
        print(
            f"  {k:60s}  ρ={v['spearman']:+.3f}  AUROC={v['auroc'] or 'n/a'}  "
            f"type={v['coarse_selection_type']}"
        )
    print("BEST 5 (Spearman):")
    for k, v in ranked[-5:]:
        print(
            f"  {k:60s}  ρ={v['spearman']:+.3f}  AUROC={v['auroc'] or 'n/a'}  "
            f"type={v['coarse_selection_type']}"
        )

    summary = {
        "n_assays_indexed": len(jobs),
        "n_assays_scored": len(ok),
        "n_assays_skipped": len(skipped),
        "spearman": spearman_dist,
        "auroc": auroc_dist,
        "by_selection_type": by_sel,
        "am_comparison": am_comparison,
        "decision_rules": {
            "G1": {
                "criterion": f"median_spearman >= {G1_MIN_MEDIAN_SPEARMAN}",
                "value": esm2_median,
                "passed": g1,
            },
            "G2": {
                "criterion": f"frac_below_0.20 <= {G2_MAX_FRAC_BELOW_020}",
                "value": esm2_frac020,
                "passed": g2,
            },
            "G3": {
                "criterion": (
                    f"esm2_median - am_median >= {G3_MIN_MEDIAN_GAP_OVER_AM}"
                ),
                "value": am_gap,
                "passed": g3,
            },
        },
        "model": "esm2_t33_650M_UR50D",
        "taxon": "Human",
        "min_variants": MIN_VARIANTS,
    }

    with open(OUT / "per_assay.json", "w") as f:
        json.dump(per_assay, f, indent=2, sort_keys=True)
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote → {OUT}/per_assay.json")
    print(f"Wrote → {OUT}/summary.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        default="123",
        help="Phases to run: '1', '2', '3', or '123' (default: all)",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Forward-pass batch size for GPU phase (default: 32)",
    )
    args = ap.parse_args()

    phases = set(args.phase)

    jobs = phase1_build_jobs() if "1" in phases else None

    if "2" in phases or "3" in phases:
        if jobs is None:
            if not JOBS_CACHE.exists():
                print("ERROR: jobs cache not found. Run --phase 1 first.")
                sys.exit(1)
            with open(JOBS_CACHE) as f:
                jobs = json.load(f)

    if "2" in phases:
        print("\n=== Phase 2: GPU ΔLL extraction ===")
        all_scores = phase2_extract_ll(jobs, batch_size=args.batch_size)
    elif "3" in phases:
        if not SCORE_CACHE.exists():
            print("ERROR: LL scores not found. Run --phase 2 first (requires GPU).")
            sys.exit(1)
        with open(SCORE_CACHE) as f:
            all_scores = json.load(f)
        print(f"Loaded ΔLL scores for {len(all_scores)} assays")
    else:
        all_scores = None

    if "3" in phases:
        print("\n=== Phase 3: analysis ===")
        phase3_analyse(jobs, all_scores)


if __name__ == "__main__":
    main()
