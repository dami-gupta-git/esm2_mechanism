"""
ESM-2 log-likelihood on ProteinGym DMS assays (human, all selection types).

For each single-mutant variant:
  ΔLL = log P(wt_aa | context) − log P(mut_aa | context)
scored by ESM-2 650M masked language model.

Per-assay Spearman ρ and AUROC, stratified by selection type and Pfam family.
Compared to AlphaMissense (result_18) to test whether conservation-based scoring
has lower per-assay variance than supervised pathogenicity scoring.

Decision rules (pre-registered):
  G1: median Spearman ≥ 0.40   (replicate ESM-1v published baseline)
  G2: frac assays with ρ < 0.20 ≤ 0.25  (tighter than AM's 0.32)
  G3: ESM-2 LL median Spearman > AM median + 0.05

Phases:
  --phase 1   CPU: parse DMS index, build job list
  --phase 2   GPU: compute ΔLL for all variants (A100, ~1-2 hours)
  --phase 3   CPU: Spearman/AUROC, stratification, decision rules
  (default: all phases)

Usage:
  python3 scripts/proteingym_esm2_ll.py --phase 2   # GPU
  python3 scripts/proteingym_esm2_ll.py --phase 3   # CPU after phase 2
"""

from __future__ import annotations

import argparse
import csv
import functools
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

from esm2_mech.utils.constants import AA_ORDER
from esm2_mech.utils.paths import PROTEINGYM_CACHE_DIR, RESULTS_DIR as _RESULTS_DIR

PG_DIR = PROTEINGYM_CACHE_DIR
OUT = _RESULTS_DIR / "proteingym_esm2_ll"

DMS_INDEX = PG_DIR / "DMS_substitutions.csv"
DMS_SUBDIR = PG_DIR / "DMS_ProteinGym_substitutions"
JOBS_CACHE = PG_DIR / "esm2_ll_jobs.json"
SCORE_CACHE = PG_DIR / "esm2_ll_scores.json"
AM_CACHE = PG_DIR / "am_scores_proteingym.json"

CHECKPOINT_EVERY = 10  # assays between GPU saves
MIN_VARIANTS = 20  # minimum scored variants to include an assay


# ── helpers ───────────────────────────────────────────────────────────────────


def parse_mutant(mut_str: str) -> tuple[str, int, str] | None:
    """Parse 'A673C' → (wt_aa='A', pos=673, mut_aa='C'). Returns None on failure."""
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", mut_str.strip())
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3)


def window_sequence(seq: str, pos_1indexed: int, window: int = 1000) -> tuple[str, int]:
    """
    Return (windowed_seq, new_pos_1indexed).
    Replicates experiment.py window_sequence: centre on pos, clamp to ends.
    If seq fits in window, return unchanged.
    """
    L = len(seq)
    if L <= window:
        return seq, pos_1indexed
    half = window // 2
    start = max(0, pos_1indexed - 1 - half)
    end = start + window
    if end > L:
        end = L
        start = max(0, end - window)
    new_pos = pos_1indexed - start  # still 1-indexed relative to windowed seq
    return seq[start:end], new_pos


# ── Phase 1: build job list ───────────────────────────────────────────────────


def phase1_build_jobs() -> list[dict]:
    """
    For each human assay, parse the DMS CSV and collect every single-mutant variant.
    Returns a job list: one entry per assay with all variants and the WT sequence.
    Caches to JOBS_CACHE.
    """
    if JOBS_CACHE.exists():
        print(f"Loading cached jobs from {JOBS_CACHE}")
        with open(JOBS_CACHE) as f:
            jobs = json.load(f)
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

        # Single mutants only (no ':' in the mutant string)
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
                # Some assays use an offset; skip mismatched residues
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
    with open(JOBS_CACHE, "w") as f:
        json.dump(jobs, f, indent=2)
    print(f"Saved → {JOBS_CACHE}")
    return jobs


# ── Phase 2: GPU ΔLL extraction ────────────────────────────────────────────────


def phase2_extract_ll(
    jobs: list[dict], batch_size: int = 32
) -> dict[str, dict[str, float]]:
    """
    For each assay, for each unique position in the WT sequence:
      - mask that position
      - run one ESM-2 forward pass
      - record log P(wt_aa) and log P(mut_aa) for every variant at that position

    Returns: {DMS_id: {mutant_str: delta_ll}}
    """
    import torch
    import esm as esm_lib

    if SCORE_CACHE.exists():
        print(f"Cached LL scores found: {SCORE_CACHE}")
        with open(SCORE_CACHE) as f:
            return json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Phase 2 requires a GPU. Run on RunPod.")

    from esm2_mech.embeddings.embed_variants import ESM2_MODEL_650M

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

        # Group variants by position — one forward pass per unique position
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
                win_seq, new_pos = window_sequence(wt_seq, pos)
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

    with open(SCORE_CACHE, "w") as f:
        json.dump(all_scores, f)
    print(f"Saved ΔLL scores → {SCORE_CACHE}")
    if CKPT.exists():
        CKPT.unlink()

    return all_scores


# ── Phase 3: analysis ──────────────────────────────────────────────────────────


def phase3_analyse(jobs: list[dict], all_scores: dict[str, dict[str, float]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Load Pfam map for family-split analysis
    pfam_path = DATA / "pfam_families.json"
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

        # Spearman: ΔLL vs DMS_score
        # High ΔLL = model surprised by mutation = low fitness expected
        # So we expect negative raw correlation; report as positive by flipping
        rho_raw, pval = spearmanr(dms_score_arr, delta_ll_arr)
        spearman = -float(rho_raw)  # positive = agreement (ΔLL predicts low fitness)

        # AUROC: damaging (bin==0) as positive class, ΔLL as score
        bin_dmg = (dms_bin_arr == 0).astype(int)
        auroc = None
        if len(np.unique(bin_dmg)) >= 2:
            try:
                auroc = float(roc_auc_score(bin_dmg, delta_ll_arr))
            except Exception:
                pass

        # Pfam family of this protein — pfam_map is keyed by gene symbol.
        # Try molecule_name (often the gene symbol), then the mnemonic prefix
        # (e.g. "ACE2" from "ACE2_HUMAN"), as the UniProt mnemonic never matches.
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

    # ── summary stats ──────────────────────────────────────────────────────────

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

    # ── by selection type ──────────────────────────────────────────────────────
    sel_types = sorted({v["coarse_selection_type"] for v in ok})
    by_sel: dict[str, dict] = {}
    print("\n=== By selection type ===")
    for sel in sel_types:
        sel_rhos = [v["spearman"] for v in ok if v["coarse_selection_type"] == sel]
        if sel_rhos:
            by_sel[sel] = dist(sel_rhos, f"  {sel} (n={len(sel_rhos)})")

    # ── AM comparison ──────────────────────────────────────────────────────────
    # Result_18 summary: AM median Spearman on human ProteinGym = 0.748 AUROC (not spearman directly)
    # We load per_assay from result_18 if available for direct comparison
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

    # ── decision rules ──────────────────────────────────────────────────────────
    esm2_median = spearman_dist.get("median", float("nan"))
    esm2_frac020 = spearman_dist.get("frac_below_0_20", float("nan"))
    am_median = am_comparison.get("median", float("nan"))

    g1 = esm2_median >= 0.40
    g2 = esm2_frac020 <= 0.25
    g3 = (esm2_median - am_median) >= 0.05 if not np.isnan(am_median) else None

    print("\n=== DECISION RULES ===")
    print(
        f"  G1: median Spearman ≥ 0.40 → {esm2_median:.3f} → {'PASS ✓' if g1 else 'FAIL ✗'}"
    )
    print(
        f"  G2: frac ρ<0.20 ≤ 0.25    → {esm2_frac020:.3f} → {'PASS ✓' if g2 else 'FAIL ✗'}"
    )
    if g3 is not None:
        print(
            f"  G3: ESM2 median − AM median ≥ 0.05 → {esm2_median-am_median:.3f} → {'PASS ✓' if g3 else 'FAIL ✗'}"
        )

    # ── worst/best 5 ──────────────────────────────────────────────────────────
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

    # ── write outputs ──────────────────────────────────────────────────────────
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
                "criterion": "median_spearman >= 0.40",
                "value": esm2_median,
                "passed": g1,
            },
            "G2": {
                "criterion": "frac_below_0.20 <= 0.25",
                "value": esm2_frac020,
                "passed": g2,
            },
            "G3": {
                "criterion": "esm2_median - am_median >= 0.05",
                "value": (
                    float(esm2_median - am_median) if not np.isnan(am_median) else None
                ),
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


# ── main ──────────────────────────────────────────────────────────────────────


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
