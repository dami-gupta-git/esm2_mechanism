"""
Extract ESM-2 embeddings for the pathogenicity positive-control variant set.

Reads {run_dir}/data/clinvar_pathogenicity_variants.json (written by phase 1 of
pathogenicity_fetch.py) and produces mean-pooled WT and mutant embeddings used
by pathogenicity_probes.py.

Inputs:
  {run_dir}/data/clinvar_pathogenicity_variants.json
  data/cache/sequences.json

Outputs (under data/embeddings/<model>/):
  pathogenicity_wt_mean.npy      (N, D)  mean-pooled WT embeddings
  pathogenicity_mut_mean.npy     (N, D)  mean-pooled mutant embeddings
  pathogenicity_meta.json        {valid_indices, n, n_valid, model}

valid_indices are 0-based positions into the input variant list so that
pathogenicity_probes.py can reconstruct the matched variant metadata via
  [variants[i] for i in meta["valid_indices"]]

Checkpointing: if all three output files exist and n_valid matches, the step
is skipped automatically. Re-run the same command to retry after a failure.

Usage (requires GPU, run inside tmux on RunPod):
    python -m esm2_mech.embeddings.embed_pathogenicity \\
        --run_dir run_0 --model esm2_t33_650M_UR50D --batch_size 32
"""

import argparse
import functools
import json
import os

import numpy as np

from esm2_mech.embeddings.embed_variants import get_esm2_embeddings_for_pairs
from esm2_mech.utils.io import atomic_write_json, save_npy
from esm2_mech.utils.paths import EMB_DIR, PATH_EMB_META, PATH_EMB_MUT_MEAN, PATH_EMB_WT_MEAN, SEQUENCES_JSON
from esm2_mech.utils.sequences import window_sequence, apply_missense

print = functools.partial(print, flush=True)

ESM2_MODEL_650M = "esm2_t33_650M_UR50D"
ESM2_MODEL_3B = "esm2_t36_3B_UR50D"


def _build_valid_pairs_indexed(
    variants: list[dict], seq_cache: dict[str, str]
) -> tuple[list[int], list[dict], list[str], list[str], list[int]]:
    """Like embed_variants._build_valid_pairs but also returns the original indices."""
    valid_indices, valid, wt_seqs, mut_seqs, positions = [], [], [], [], []
    skipped_no_uid = 0
    for idx, v in enumerate(variants):
        uid = v.get("uniprot_id")
        if not uid:
            skipped_no_uid += 1
            continue
        if uid not in seq_cache:
            continue
        wt_win, new_pos = window_sequence(seq_cache[uid], v["aa_pos"])
        mut_win = apply_missense(wt_win, new_pos, v["aa_wt"], v["aa_mut"])
        if mut_win is None:
            continue
        valid_indices.append(idx)
        valid.append(v)
        wt_seqs.append(wt_win)
        mut_seqs.append(mut_win)
        positions.append(new_pos)
    if skipped_no_uid:
        print(f"WARNING: skipped {skipped_no_uid} variants with no uniprot_id")
    print(f"Valid variant pairs: {len(valid)}")
    return valid_indices, valid, wt_seqs, mut_seqs, positions


def _already_complete(n_valid: int) -> bool:
    if not all(p.exists() for p in [PATH_EMB_WT_MEAN, PATH_EMB_MUT_MEAN, PATH_EMB_META]):
        return False
    try:
        with open(PATH_EMB_META) as f:
            meta = json.load(f)
    except json.JSONDecodeError:
        print(f"WARNING: corrupt {PATH_EMB_META} — re-extracting")
        PATH_EMB_META.unlink()
        return False
    n_on_disk = np.load(PATH_EMB_WT_MEAN, mmap_mode="r").shape[0]
    return n_on_disk == n_valid == meta.get("n_valid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", default="run_0")
    parser.add_argument("--model", default=ESM2_MODEL_650M, choices=[ESM2_MODEL_650M, ESM2_MODEL_3B])
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    variants_path = os.path.join(args.run_dir, "data", "clinvar_pathogenicity_variants.json")
    if not os.path.exists(variants_path):
        raise FileNotFoundError(
            f"Pathogenicity variants not found: {variants_path}\n"
            "Run phase 1 first:\n"
            "  python -m esm2_mech.experiments.pathogenicity.pathogenicity_fetch"
        )
    with open(variants_path) as f:
        variants = json.load(f)
    print(f"Loaded {len(variants)} pathogenicity variants from {variants_path}")

    if not SEQUENCES_JSON.exists():
        raise FileNotFoundError(
            f"Sequence cache not found: {SEQUENCES_JSON}\n"
            "Run: python -m esm2_mech.fetch_data.fetch_sequences"
        )
    with open(SEQUENCES_JSON) as f:
        seq_cache = json.load(f)

    valid_indices, valid, wt_seqs, mut_seqs, positions = _build_valid_pairs_indexed(variants, seq_cache)

    if _already_complete(len(valid)):
        print("Pathogenicity embeddings already complete — nothing to do.")
        return

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model: {args.model}")

    EMB_DIR.mkdir(parents=True, exist_ok=True)

    wt_mean, mut_mean, _, _ = get_esm2_embeddings_for_pairs(
        wt_seqs,
        mut_seqs,
        positions,
        valid_variants=valid,
        out_dir=None,
        model_name=args.model,
        device=device,
        batch_size=args.batch_size,
    )

    save_npy(str(PATH_EMB_WT_MEAN), wt_mean)
    save_npy(str(PATH_EMB_MUT_MEAN), mut_mean)
    atomic_write_json(
        PATH_EMB_META,
        {
            "valid_indices": valid_indices,
            "n": len(variants),
            "n_valid": len(valid),
            "model": args.model,
        },
    )

    print(f"Saved: {wt_mean.shape} -> {EMB_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
