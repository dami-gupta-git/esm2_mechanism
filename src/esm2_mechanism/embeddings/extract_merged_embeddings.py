"""
Extract ESM-2 embeddings for the merged Gerasimavicius + G2P/ClinVar dataset.

Reads merged_variants.json (built by build_merged_dataset.py) and sequences.json.
Fetches UniProt sequences for any genes not already in sequences.json cache.
Extracts ESM-2 650M mean-pooled and per-residue embeddings.

Outputs:
  merged_embeddings_wt_mean.npy   (N, 1280)
  merged_embeddings_mut_mean.npy  (N, 1280)
  merged_embeddings_wt_pos.npy    (N, 1280)
  merged_embeddings_mut_pos.npy   (N, 1280)
  merged_valid_variants.json      — filtered variant list aligned with embeddings

Usage (requires GPU):
    python extract_merged_embeddings.py --data_dir run_0/data --batch_size 32
"""

import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="run_0/data")
parser.add_argument("--model", default="esm2_t33_650M_UR50D")
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

from embeddings.esm2_mechanism import get_esm2_embeddings_for_pairs
from utils_sequences import window_sequence, apply_missense, fetch_uniprot_sequence

# Load merged variants
variants_path = os.path.join(args.data_dir, "merged_variants.json")
with open(variants_path) as f:
    variants = json.load(f)
print(f"Loaded {len(variants)} merged variants")

# Load / update sequence cache
seq_cache_path = os.path.join(args.data_dir, "sequences.json")
with open(seq_cache_path) as f:
    seq_cache = json.load(f)

# Fetch sequences for any new UniProt IDs
new_uids = {v["uniprot_id"] for v in variants if v["uniprot_id"] and v["uniprot_id"] not in seq_cache}
if new_uids:
    print(f"Fetching {len(new_uids)} new UniProt sequences...")
    for i, uid in enumerate(sorted(new_uids)):
        if i % 100 == 0:
            print(f"  {i}/{len(new_uids)}")
        seq = fetch_uniprot_sequence(uid)
        if seq:
            seq_cache[uid] = seq
        time.sleep(0.3)
    with open(seq_cache_path, "w") as f:
        json.dump(seq_cache, f)
    print(f"  Sequences now cached: {len(seq_cache)}")

# Build valid variant list and sequences
valid, wt_seqs, mut_seqs, positions = [], [], [], []
skipped_no_uid = 0
for v in variants:
    uid = v["uniprot_id"]
    if not uid:
        skipped_no_uid += 1
        continue
    if uid not in seq_cache:
        continue
    wt_full = seq_cache[uid]
    wt_win, new_pos = window_sequence(wt_full, v["aa_pos"])
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

from collections import Counter
labels = Counter(v["label_3class"] for v in valid)
print(f"3-class distribution: {dict(labels)}")

# Check for partial resume: if all checkpoint files exist and cover all variants, skip extraction
ckpt_valid   = os.path.join(args.data_dir, "merged_valid_variants.json")
ckpt_wt      = os.path.join(args.data_dir, "merged_embeddings_wt_mean.npy")
ckpt_mut     = os.path.join(args.data_dir, "merged_embeddings_mut_mean.npy")
ckpt_wt_pos  = os.path.join(args.data_dir, "merged_embeddings_wt_pos.npy")
ckpt_mut_pos = os.path.join(args.data_dir, "merged_embeddings_mut_pos.npy")
if (os.path.exists(ckpt_valid) and os.path.exists(ckpt_wt) and os.path.exists(ckpt_mut)
        and os.path.exists(ckpt_wt_pos) and os.path.exists(ckpt_mut_pos)):
    prev = json.load(open(ckpt_valid))
    if len(prev) == len(valid):
        print("Embeddings already complete — loading from cache.")
        wt_mean  = np.load(ckpt_wt)
        mut_mean = np.load(ckpt_mut)
        wt_pos   = np.load(ckpt_wt_pos)
        mut_pos  = np.load(ckpt_mut_pos)
        print(f"Loaded embeddings: {wt_mean.shape}")
        print("Done.")
        sys.exit(0)

# Extract embeddings
import torch
import functools
print = functools.partial(print, flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

wt_mean, mut_mean, wt_pos, mut_pos = get_esm2_embeddings_for_pairs(
    wt_seqs, mut_seqs, positions,
    model_name=args.model, device=device, batch_size=args.batch_size
)

# Save: write all npy files first, then valid_variants JSON last so its presence
# signals a fully consistent checkpoint set (crash between npy saves is detectable on resume)
np.save(ckpt_wt,      wt_mean)
np.save(ckpt_mut,     mut_mean)
np.save(ckpt_wt_pos,  wt_pos)
np.save(ckpt_mut_pos, mut_pos)
with open(ckpt_valid, "w") as f:
    json.dump(valid, f)

print(f"\nSaved embeddings: {wt_mean.shape}")
print("Done.")
