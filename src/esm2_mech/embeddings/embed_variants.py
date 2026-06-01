"""
Extract ESM-2 embeddings for the filtered variant dataset.

Reads data/valid_variants.json (built by fetch_data/build_valid_variants.py) and
data/cache/sequences.json, then runs ESM-2 on each WT/mutant sequence pair.

Pipeline
--------
1. Load pre-filtered variants from data/valid_variants.json.
2. Load sequences from data/cache/sequences.json.
3. For each variant, extract a window of up to 1022 residues centred on the mutation
   site (ESM-2's token limit) and apply the missense substitution.
4. Run WT and mutant sequences through ESM-2 (650M or 3B), interleaved in the same
   batch to halve the number of forward passes.
5. Extract two embedding types per variant from the final transformer layer:
     - Mean-pooled: average over all residue embeddings; captures global protein context.
     - Position-specific: embedding at the exact mutation site; captures local context.

Outputs (under data/embeddings/<model>/):
  embeddings_wt_mean.npy        (N, D)  mean-pooled WT embeddings
  embeddings_mut_mean.npy       (N, D)  mean-pooled mutant embeddings
  embeddings_wt_pos.npy         (N, D)  per-residue WT embedding at variant position
  embeddings_mut_pos.npy        (N, D)  per-residue mutant embedding at variant position
  embedded_variants.json        (N,)    row-aligned subset of variants actually embedded

All four arrays are row-aligned to embedded_variants.json (the row-aligned subset
of the input data/valid_variants.json that was actually embedded).

Checkpointing
-------------
Embeddings are flushed atomically every checkpoint_every variants so a GPU job can
survive an interrupt. On restart, extraction resumes from the last checkpoint or
skips entirely if already complete.

Usage (requires GPU):
    python -m esm2_mech.embeddings.embed_variants \\
        --model esm2_t33_650M_UR50D --batch_size 32
"""

import argparse
import functools
import json
import os
from collections import Counter
import numpy as np

print = functools.partial(print, flush=True)

from esm2_mech.utils.sequences import (
    window_sequence,
    apply_missense,
)
from esm2_mech.utils.embed import get_esm2_embeddings_for_pairs
from esm2_mech.utils.constants import ESM2_MODEL as ESM2_MODEL_650M, ESM2_MODEL_3B
from esm2_mech.utils.paths import VALID_VARIANTS_JSON, SEQUENCES_JSON, DATA_DIR




# ---------------------------------------------------------------------------
# Variant filtering
# ---------------------------------------------------------------------------


def _build_valid_pairs(
    variants: list[dict], seq_cache: dict[str, str]
) -> tuple[list[dict], list[str], list[str], list[int]]:
    """Filter variants to those with a sequence and a valid mutation, build input lists."""
    valid, wt_seqs, mut_seqs, positions = [], [], [], []
    skipped_no_uid = 0

    for v in variants:
        uid = v.get("uniprot_id")
        if not uid:
            skipped_no_uid += 1
            continue
        if uid not in seq_cache:
            continue
        wt_win, new_pos, _ = window_sequence(seq_cache[uid], v["aa_pos"])
        mut_win = apply_missense(wt_win, new_pos, v["aa_wt"], v["aa_mut"])
        if mut_win is None:
            continue
        valid.append(v)
        wt_seqs.append(wt_win)
        mut_seqs.append(mut_win)
        positions.append(new_pos)

    if skipped_no_uid:
        print(f"WARNING: skipped {skipped_no_uid} variants with empty uniprot_id")
    print(f"Valid variant pairs: {len(valid)}")
    if len(valid) < 50:
        print("WARNING: very few valid variants — results may not be reliable")
    return valid, wt_seqs, mut_seqs, positions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=ESM2_MODEL_650M, choices=[ESM2_MODEL_650M, ESM2_MODEL_3B]
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    args = parser.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model: {args.model}")

    if not VALID_VARIANTS_JSON.exists():
        raise FileNotFoundError(
            f"{VALID_VARIANTS_JSON} not found — run fetch_data/build_valid_variants first"
        )
    with open(VALID_VARIANTS_JSON) as f:
        valid_variants = json.load(f)
    print(f"Loaded {len(valid_variants):,} valid variants")

    if not SEQUENCES_JSON.exists():
        raise FileNotFoundError(
            f"{SEQUENCES_JSON} not found — run fetch_data/fetch_sequences first"
        )
    with open(SEQUENCES_JSON) as f:
        seq_cache = json.load(f)

    valid, wt_seqs, mut_seqs, positions = _build_valid_pairs(valid_variants, seq_cache)

    print(f"3-class distribution: {dict(Counter(v['label_3class'] for v in valid))}")

    out_dir = str(DATA_DIR / "embeddings" / args.model)
    os.makedirs(out_dir, exist_ok=True)

    ckpt_wt_mean = os.path.join(out_dir, "embeddings_wt_mean.npy")
    ckpt_mut_mean = os.path.join(out_dir, "embeddings_mut_mean.npy")
    ckpt_wt_pos = os.path.join(out_dir, "embeddings_wt_pos.npy")
    ckpt_mut_pos = os.path.join(out_dir, "embeddings_mut_pos.npy")

    all_ckpts = [ckpt_wt_mean, ckpt_mut_mean, ckpt_wt_pos, ckpt_mut_pos]
    resume_arrays = None
    resume_start = 0
    if all(os.path.exists(p) for p in all_ckpts):
        try:
            row_counts = [np.load(p, mmap_mode="r").shape[0] for p in all_ckpts]
            if len(set(row_counts)) > 1:
                print(f"WARNING: checkpoint row counts inconsistent {row_counts} — re-extracting")
                for ckpt in all_ckpts:
                    if os.path.exists(ckpt):
                        os.remove(ckpt)
            else:
                n_on_disk = row_counts[0]
                if n_on_disk == len(valid):
                    print("Embeddings already complete — nothing to do.")
                    return
                if n_on_disk < len(valid):
                    print(f"Partial checkpoint: {n_on_disk}/{len(valid)} rows — resuming")
                    resume_arrays = (
                        np.load(ckpt_wt_mean),
                        np.load(ckpt_mut_mean),
                        np.load(ckpt_wt_pos),
                        np.load(ckpt_mut_pos),
                    )
                    resume_start = n_on_disk
                else:
                    print(f"WARNING: checkpoint row count mismatch (arrays={n_on_disk}, expected={len(valid)}) — re-extracting")
                    for ckpt in all_ckpts:
                        if os.path.exists(ckpt):
                            os.remove(ckpt)
        except ValueError:
            print("WARNING: corrupt checkpoint — re-extracting")
            for ckpt in all_ckpts:
                if os.path.exists(ckpt):
                    os.remove(ckpt)

    print(f"\nExtracting ESM-2 embeddings ({args.model})...")
    wt_mean, mut_mean, wt_pos, mut_pos = get_esm2_embeddings_for_pairs(
        wt_seqs[resume_start:],
        mut_seqs[resume_start:],
        positions[resume_start:],
        valid_variants=valid,
        out_dir=out_dir,
        model_name=args.model,
        device=device,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        resume_arrays=resume_arrays,
    )

    print(f"\nSaved embeddings: {wt_mean.shape} -> {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
