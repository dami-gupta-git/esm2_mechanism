"""
Result 2 — Gene-split vs family-split baseline comparison.

Tests whether ESM-2 features (WT-only, delta, per-residue, onehot, FoldX, AlphaMissense)
carry mechanism signal beyond protein family identity. Runs each feature under gene-split
and family-split CV across 5 seeds.

Reads from paths.py constants. Writes results to RESULTS_DIR/run1/.
"""

import functools
import json
import os

import numpy as np

from esm2_mech.experiments.mechanism.mechanism_delta_family_split import run as run_family_split
from esm2_mech.experiments.mechanism.mechanism_delta_probe import _load_alphamissense_scores
from esm2_mech.utils.data import load_variants
from esm2_mech.utils.paths import (
    EMB_WT_MEAN,
    EMB_MUT_MEAN,
    EMB_WT_POS,
    EMB_MUT_POS,
    RESULTS_DIR,
    SEQUENCES_JSON,
    VARIANTS_JSON,
)
from esm2_mech.utils.sequences import apply_missense, window_sequence

print = functools.partial(print, flush=True)


def load_data() -> dict:
    """Load and return all shared data needed by the probe experiments."""
    print("\n=== Loading variants ===")
    if not VARIANTS_JSON.exists():
        raise FileNotFoundError(
            f"{VARIANTS_JSON} not found — run fetch_data/fetch_variants.py --step merge first"
        )
    variants = load_variants(VARIANTS_JSON)

    print("\n=== Loading sequences ===")
    if not SEQUENCES_JSON.exists():
        raise FileNotFoundError(
            f"{SEQUENCES_JSON} not found — run fetch_data/fetch_sequences first"
        )
    with open(SEQUENCES_JSON) as f:
        seq_cache = json.load(f)

    # Filter to variants with a valid sequence and a viable WT/mut window.
    # Variants missing a UniProt ID or sequence are skipped with a warning.
    valid_variants = []
    skipped_no_uid = 0
    skipped_no_seq = 0
    for v in variants:
        uid = v.get("uniprot_id")
        if not uid:
            skipped_no_uid += 1
            continue
        if uid not in seq_cache:
            skipped_no_seq += 1
            continue
        # Window the sequence around the mutation site (ESM-2 has a 1022-token limit).
        wt_win, new_pos = window_sequence(seq_cache[uid], v["aa_pos"])
        mut_win = apply_missense(wt_win, new_pos, v["aa_wt"], v["aa_mut"])
        if mut_win is None:
            continue
        valid_variants.append(v)

    if skipped_no_uid:
        print(f"WARNING: {skipped_no_uid} variants skipped — missing UniProt ID")
    if skipped_no_seq:
        print(f"WARNING: {skipped_no_seq} variants skipped — UniProt ID not in sequence cache")
    print(f"Valid variant pairs: {len(valid_variants)}")
    if len(valid_variants) < 50:
        print("WARNING: Very few valid variants. Results may not be reliable.")

    # Four embedding arrays, all aligned by row to valid_variants.
    print("\n=== Loading embeddings ===")
    for path in [EMB_WT_MEAN, EMB_MUT_MEAN, EMB_WT_POS, EMB_MUT_POS]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embedding file missing: {path}")
    emb_wt_mean = np.load(EMB_WT_MEAN)
    emb_mut_mean = np.load(EMB_MUT_MEAN)
    emb_wt_pos = np.load(EMB_WT_POS)
    emb_mut_pos = np.load(EMB_MUT_POS)

    # Build label and metadata arrays aligned to valid_variants.
    labels_3class = np.array([v["label_3class"] for v in valid_variants])
    labels_4class = np.array([v["mechanism"] for v in valid_variants])
    genes_arr = np.array([v["gene"] for v in valid_variants])
    # FoldX ΔΔG: NaN where missing — never impute, restrict subset at probe time.
    foldx_ddg = np.array(
        [v["foldx_ddg"] if v["foldx_ddg"] is not None else np.nan for v in valid_variants]
    )
    aa_wt_list = [v["aa_wt"] for v in valid_variants]
    aa_mut_list = [v["aa_mut"] for v in valid_variants]

    # AlphaMissense: NaN where score unavailable.
    print("\n=== Loading AlphaMissense scores ===")
    alphamissense_scores = _load_alphamissense_scores(valid_variants)

    return {
        "valid_variants": valid_variants,
        "emb_wt_mean": emb_wt_mean,
        "emb_mut_mean": emb_mut_mean,
        "emb_wt_pos": emb_wt_pos,
        "emb_mut_pos": emb_mut_pos,
        "labels_3class": labels_3class,
        "labels_4class": labels_4class,
        "genes_arr": genes_arr,
        "foldx_ddg": foldx_ddg,
        "aa_wt_list": aa_wt_list,
        "aa_mut_list": aa_mut_list,
        "alphamissense_scores": alphamissense_scores,
    }


def main():
    out_dir = str(RESULTS_DIR / "run1")
    os.makedirs(out_dir, exist_ok=True)

    data = load_data()

    print("\n=== Result 2: Gene-split vs family-split baselines ===")
    for seed in [0, 1, 2, 3, 4]:
        print(f"\n--- Seed {seed} ---")
        run_family_split(data=data, out_dir=out_dir, seed=seed)


if __name__ == "__main__":
    main()
