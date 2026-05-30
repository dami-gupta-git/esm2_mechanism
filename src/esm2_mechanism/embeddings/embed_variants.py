"""
Extract ESM-2 embeddings for the merged variant dataset.

Reads  data/merged_variants.json  (built by fetch_data/fetch_variants.py --step merge)
and    data/cache/sequences.json  (built by fetch_data/fetch_annotations.py --step uniprot).
Fetches UniProt sequences for any IDs not yet in the cache (incremental update).

Outputs (under data/embeddings/<model>/):
  embeddings_wt_mean.npy        (N, D)  mean-pooled WT
  embeddings_mut_mean.npy       (N, D)  mean-pooled mutant
  embeddings_wt_pos.npy         (N, D)  per-residue WT at variant position
  embeddings_mut_pos.npy        (N, D)  per-residue mutant at variant position
  valid_variants.json           filtered variant list aligned with the arrays

Usage (requires GPU):
    python -m esm2_mechanism.embeddings.embed_variants \\
        --data_dir data --model esm2_t33_650M_UR50D --batch_size 32
"""

import argparse
import functools
import json
import os
import time

import numpy as np

print = functools.partial(print, flush=True)

from esm2_mechanism.utils_sequences import fetch_uniprot_sequence, window_sequence, apply_missense

ESM2_MODEL_650M = "esm2_t33_650M_UR50D"
ESM2_MODEL_3B = "esm2_t36_3B_UR50D"


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def get_esm2_embeddings_for_pairs(
    wt_seqs: list[str],
    mut_seqs: list[str],
    aa_positions: list[int],
    model_name: str = ESM2_MODEL_650M,
    device: str = "cuda",
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract ESM-2 embeddings for WT/mutant sequence pairs.

    WT and mutant are interleaved in the same batch to halve forward passes.

    Returns:
        wt_mean:  (N, D) mean-pooled WT embeddings
        mut_mean: (N, D) mean-pooled mutant embeddings
        wt_pos:   (N, D) per-residue WT embedding at variant position
        mut_pos:  (N, D) per-residue mutant embedding at variant position
    """
    import torch
    import esm

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    n_layers = model.num_layers

    wt_mean_list, mut_mean_list = [], []
    wt_pos_list, mut_pos_list = [], []

    total = len(wt_seqs)
    for batch_start in range(0, total, batch_size):
        pairs = list(zip(
            wt_seqs[batch_start:batch_start + batch_size],
            mut_seqs[batch_start:batch_start + batch_size],
            aa_positions[batch_start:batch_start + batch_size],
        ))
        interleaved = []
        for j, (wt, mut, _) in enumerate(pairs):
            interleaved.append((f"wt{j}", wt))
            interleaved.append((f"mut{j}", mut))

        _, _, tokens = batch_converter(interleaved)
        tokens = tokens.to(device, non_blocking=True)
        with torch.inference_mode():
            out = model(tokens, repr_layers=[n_layers])
        reps = out["representations"][n_layers].cpu().float()

        for j, (wt, mut, var_pos) in enumerate(pairs):
            wt_rep  = reps[2 * j]
            mut_rep = reps[2 * j + 1]

            wt_mean_list.append(wt_rep[1:len(wt) + 1].mean(0).numpy())
            mut_mean_list.append(mut_rep[1:len(mut) + 1].mean(0).numpy())

            # var_pos is 1-indexed; token index = var_pos (BOS occupies index 0)
            if 0 < var_pos <= reps.shape[1] - 1:
                wt_pos_list.append(wt_rep[var_pos].numpy())
                mut_pos_list.append(mut_rep[var_pos].numpy())
            else:
                wt_pos_list.append(wt_rep[1:len(wt) + 1].mean(0).numpy())
                mut_pos_list.append(mut_rep[1:len(mut) + 1].mean(0).numpy())

        if (batch_start // batch_size) % 5 == 0:
            print(f"  Embedded {min(batch_start + batch_size, total)}/{total} variant pairs")

    return (
        np.stack(wt_mean_list),
        np.stack(mut_mean_list),
        np.stack(wt_pos_list),
        np.stack(mut_pos_list),
    )


# ---------------------------------------------------------------------------
# Sequence cache (incremental)
# ---------------------------------------------------------------------------

def _load_sequence_cache(cache_path: str) -> dict[str, str]:
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, newline="") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"WARNING: corrupt sequence cache at {cache_path} — starting empty")
        os.remove(cache_path)
        return {}


def _update_sequence_cache(variants: list[dict], cache_path: str) -> dict[str, str]:
    """Fetch any UniProt sequences missing from the cache and write back."""
    seq_cache = _load_sequence_cache(cache_path)
    needed = {v["uniprot_id"] for v in variants if v.get("uniprot_id") and v["uniprot_id"] not in seq_cache}

    if not needed:
        return seq_cache

    print(f"Fetching {len(needed)} missing UniProt sequences...")
    for i, uid in enumerate(sorted(needed)):
        if i % 100 == 0:
            print(f"  {i}/{len(needed)}")
        seq = fetch_uniprot_sequence(uid)
        if seq:
            seq_cache[uid] = seq
        time.sleep(0.3)

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(seq_cache, f)
    print(f"  Sequences cached: {len(seq_cache)}")
    return seq_cache


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
        wt_win, new_pos = window_sequence(seq_cache[uid], v["aa_pos"])
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
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--model", default=ESM2_MODEL_650M,
                        choices=[ESM2_MODEL_650M, ESM2_MODEL_3B])
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model: {args.model}")

    variants_path = os.path.join(args.data_dir, "merged_variants.json")
    if not os.path.exists(variants_path):
        raise FileNotFoundError(
            f"{variants_path} not found — run fetch_data/fetch_variants.py --step merge first"
        )
    with open(variants_path) as f:
        variants = json.load(f)
    print(f"Loaded {len(variants)} merged variants")

    seq_cache_path = os.path.join(args.data_dir, "cache", "sequences.json")
    seq_cache = _update_sequence_cache(variants, seq_cache_path)

    valid, wt_seqs, mut_seqs, positions = _build_valid_pairs(variants, seq_cache)

    from collections import Counter
    print(f"3-class distribution: {dict(Counter(v['label_3class'] for v in valid))}")

    out_dir = os.path.join(args.data_dir, "embeddings", args.model)
    os.makedirs(out_dir, exist_ok=True)

    ckpt_wt_mean  = os.path.join(out_dir, "embeddings_wt_mean.npy")
    ckpt_mut_mean = os.path.join(out_dir, "embeddings_mut_mean.npy")
    ckpt_wt_pos   = os.path.join(out_dir, "embeddings_wt_pos.npy")
    ckpt_mut_pos  = os.path.join(out_dir, "embeddings_mut_pos.npy")
    ckpt_valid    = os.path.join(out_dir, "valid_variants.json")

    all_ckpts = [ckpt_wt_mean, ckpt_mut_mean, ckpt_wt_pos, ckpt_mut_pos, ckpt_valid]
    if all(os.path.exists(p) for p in all_ckpts):
        try:
            with open(ckpt_valid) as f:
                prev = json.load(f)
            if len(prev) == len(valid):
                print("Embeddings already complete — nothing to do.")
                return
        except json.JSONDecodeError:
            print("WARNING: corrupt valid_variants.json checkpoint — re-extracting")
            os.remove(ckpt_valid)

    print(f"\nExtracting ESM-2 embeddings ({args.model})...")
    wt_mean, mut_mean, wt_pos, mut_pos = get_esm2_embeddings_for_pairs(
        wt_seqs, mut_seqs, positions,
        model_name=args.model, device=device, batch_size=args.batch_size,
    )

    # Write arrays before valid_variants.json so its presence signals a complete set
    np.save(ckpt_wt_mean, wt_mean)
    np.save(ckpt_mut_mean, mut_mean)
    np.save(ckpt_wt_pos,  wt_pos)
    np.save(ckpt_mut_pos, mut_pos)
    with open(ckpt_valid, "w") as f:
        json.dump(valid, f)

    print(f"\nSaved embeddings: {wt_mean.shape} -> {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
