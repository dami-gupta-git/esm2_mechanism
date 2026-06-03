"""
Extract ESM-2 embeddings for the Tsuboyama stability variants.

Companion to embed_variants.py (the mechanism dataset). Differs because the
Tsuboyama records already carry full WT/mutant domain sequences — there is no
UniProt lookup and no windowing (domains are short, well under ESM-2's 1022-token
limit), so the WT/mut sequence pairs are read straight from the parsed variants.

The heavy lifting (interleaved WT/mut forward passes, atomic checkpointing,
resume) is the shared get_esm2_embeddings_for_pairs helper. That helper writes
fixed filenames into out_dir, so this driver points it at an isolated checkpoint
subdir (MEGASCALE_EMB_CKPT_DIR) to avoid colliding with the mechanism embeddings,
then promotes the four arrays to the MEGASCALE_EMB_* paths the probes read.

Outputs (under data/embeddings/<model>/):
  megascale_wt_mean.npy   megascale_mut_mean.npy
  megascale_wt_pos.npy    megascale_mut_pos.npy
All four are row-aligned to data/megascale_tsuboyama_variants.json.

Usage (requires GPU):
  python -m esm2_mech.embeddings.embed_megascale --model esm2_t33_650M_UR50D --batch_size 32
"""

import argparse
import functools
import os
import shutil

import numpy as np

print = functools.partial(print, flush=True)

from esm2_mech.experiments.stability.tsuboyama_loader import load_tsuboyama_variants
from esm2_mech.utils.embed import get_esm2_embeddings_for_pairs
from esm2_mech.utils.constants import (
    ESM2_MODEL as ESM2_MODEL_650M,
    ESM2_MODEL_3B,
    MAX_SEQ_LEN,
)
from esm2_mech.utils.paths import (
    MEGASCALE_EMB_CKPT_DIR,
    MEGASCALE_EMB_WT_MEAN,
    MEGASCALE_EMB_MUT_MEAN,
    MEGASCALE_EMB_WT_POS,
    MEGASCALE_EMB_MUT_POS,
)

# The shared helper's fixed checkpoint name → the megascale target path it is
# promoted to once extraction is complete.
_CKPT_TO_TARGET = {
    "embeddings_wt_mean.npy": MEGASCALE_EMB_WT_MEAN,
    "embeddings_mut_mean.npy": MEGASCALE_EMB_MUT_MEAN,
    "embeddings_wt_pos.npy": MEGASCALE_EMB_WT_POS,
    "embeddings_mut_pos.npy": MEGASCALE_EMB_MUT_POS,
}


def _build_pairs(variants):
    """WT/mut sequence pairs and 1-indexed positions straight from the records.

    No windowing: a domain longer than ESM-2's limit would be a data error here
    (Tsuboyama domains are short), so we assert rather than silently truncate —
    truncation would move the mutation off var_pos and misalign the pos embedding.
    """
    wt_seqs, mut_seqs, positions = [], [], []
    for variant in variants:
        wt_seq = variant["wt_seq"]
        if len(wt_seq) > MAX_SEQ_LEN:
            raise ValueError(
                f"domain {variant['protein']} length {len(wt_seq)} exceeds ESM-2 "
                f"limit {MAX_SEQ_LEN} — windowing is not implemented for megascale"
            )
        wt_seqs.append(wt_seq)
        mut_seqs.append(variant["mut_seq"])
        positions.append(variant["var_pos"])
    return wt_seqs, mut_seqs, positions


def _promote_checkpoint(ckpt_dir, n_expected):
    """Copy the completed checkpoint arrays to the MEGASCALE_EMB_* target paths."""
    for name, target in _CKPT_TO_TARGET.items():
        src = os.path.join(ckpt_dir, name)
        rows = np.load(src, mmap_mode="r").shape[0]
        if rows != n_expected:
            raise ValueError(
                f"checkpoint {src} has {rows} rows, expected {n_expected} — "
                f"extraction incomplete"
            )
        shutil.copyfile(src, str(target))
    print(f"Promoted 4 arrays ({n_expected} rows) to megascale embedding paths")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=ESM2_MODEL_650M, choices=[ESM2_MODEL_650M, ESM2_MODEL_3B]
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint_every", type=int, default=500)
    args = parser.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model: {args.model}")

    variants = load_tsuboyama_variants()
    wt_seqs, mut_seqs, positions = _build_pairs(variants)
    print(f"Embedding {len(wt_seqs)} WT/mut pairs across "
          f"{len({v['protein'] for v in variants})} domains")

    ckpt_dir = str(MEGASCALE_EMB_CKPT_DIR)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Resume: if a full checkpoint already exists, skip extraction and just promote.
    ckpt_paths = [os.path.join(ckpt_dir, name) for name in _CKPT_TO_TARGET]
    resume_arrays, resume_start = None, 0
    if all(os.path.exists(p) for p in ckpt_paths):
        try:
            row_counts = [np.load(p, mmap_mode="r").shape[0] for p in ckpt_paths]
        except (ValueError, OSError, EOFError):
            # A truncated/partially-written .npy fails to mmap with OSError/EOFError
            # (bad header or data), not only ValueError.
            print("WARNING: corrupt checkpoint — re-extracting")
            row_counts = [-1]
            for p in ckpt_paths:
                if os.path.exists(p):
                    os.remove(p)
        if len(set(row_counts)) == 1 and row_counts[0] == len(wt_seqs):
            print("Embeddings already complete — promoting checkpoint.")
            _promote_checkpoint(ckpt_dir, len(wt_seqs))
            return
        if len(set(row_counts)) == 1 and 0 < row_counts[0] < len(wt_seqs):
            resume_start = row_counts[0]
            print(f"Partial checkpoint: {resume_start}/{len(wt_seqs)} rows — resuming")
            resume_arrays = tuple(np.load(p) for p in ckpt_paths)

    print(f"\nExtracting ESM-2 embeddings ({args.model})...")
    get_esm2_embeddings_for_pairs(
        wt_seqs[resume_start:],
        mut_seqs[resume_start:],
        positions[resume_start:],
        valid_variants=variants,
        out_dir=ckpt_dir,
        model_name=args.model,
        device=device,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        resume_arrays=resume_arrays,
    )

    _promote_checkpoint(ckpt_dir, len(wt_seqs))
    print("Done.")


if __name__ == "__main__":
    main()
