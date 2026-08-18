"""Extract ESM-2 embeddings for Tsuboyama stability variants.

Companion to embed_variants.py (the mechanism dataset). Differs because the
Tsuboyama records already carry full WT/mutant domain sequences — there is no
UniProt lookup and no windowing (domains are short, well under ESM-2's 1022-token
limit), so the WT/mut sequence pairs are read straight from the parsed variants.
"""

import argparse
import functools
import os
import shutil

import numpy as np

print = functools.partial(print, flush=True)

from esm2_mech.experiments.stability.tsuboyama_loader import load_tsuboyama_variants
from esm2_mech.experiments.stability.stability_data import save_fingerprint
from esm2_mech.utils.embed import (
    EMB_ARRAY_NAMES,
    get_esm2_embeddings_for_pairs,
    inspect_four_array_checkpoint,
)
from esm2_mech.utils.constants import (
    ESM2_MODEL as ESM2_MODEL_650M,
    ESM2_MODEL_3B,
    MAX_SEQ_LEN,
)
from esm2_mech.utils.paths import (
    DATA_DIR,
    MEGASCALE_EMB_WT_MEAN,
    MEGASCALE_EMB_MUT_MEAN,
    MEGASCALE_EMB_WT_POS,
    MEGASCALE_EMB_MUT_POS,
)

_CKPT_TO_TARGET_FILENAME = dict(zip(
    EMB_ARRAY_NAMES,
    (
        MEGASCALE_EMB_WT_MEAN.name,
        MEGASCALE_EMB_MUT_MEAN.name,
        MEGASCALE_EMB_WT_POS.name,
        MEGASCALE_EMB_MUT_POS.name,
    ),
))


def _build_pairs(variants):
    """Build WT/mut sequence pairs; raises if any domain exceeds ESM-2's token limit."""
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


def _promote_checkpoint(ckpt_dir, n_expected, target_dir):
    """Promote completed checkpoint arrays to megascale_*.npy under target_dir."""
    for name, target_filename in _CKPT_TO_TARGET_FILENAME.items():
        src = os.path.join(ckpt_dir, name)
        rows = np.load(src, mmap_mode="r").shape[0]
        if rows != n_expected:
            raise ValueError(
                f"checkpoint {src} has {rows} rows, expected {n_expected} — "
                f"extraction incomplete"
            )
        shutil.copyfile(src, os.path.join(target_dir, target_filename))
    print(f"Promoted 4 arrays ({n_expected} rows) to {target_dir}")


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

    target_dir = str(DATA_DIR / "embeddings" / args.model)
    ckpt_dir = os.path.join(target_dir, "megascale_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_paths = [os.path.join(ckpt_dir, name) for name in _CKPT_TO_TARGET_FILENAME]
    resume_arrays, resume_start = None, 0
    status, payload = inspect_four_array_checkpoint(ckpt_paths, len(wt_seqs))
    if status == "complete":
        print("Embeddings already complete — promoting checkpoint.")
        _promote_checkpoint(ckpt_dir, len(wt_seqs), target_dir)
        save_fingerprint(variants)
        return
    if status == "resume":
        resume_start, resume_arrays = payload
        print(f"Partial checkpoint: {resume_start}/{len(wt_seqs)} rows — resuming")

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

    _promote_checkpoint(ckpt_dir, len(wt_seqs), target_dir)
    save_fingerprint(variants)
    print("Done.")


if __name__ == "__main__":
    main()
