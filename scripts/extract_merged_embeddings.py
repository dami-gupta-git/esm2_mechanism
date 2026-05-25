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
    python extract_merged_embeddings.py --data_dir data/raw --emb_dir data/embeddings --batch_size 32
"""

import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data/raw")
parser.add_argument("--emb_dir", default="data/embeddings")
parser.add_argument("--model", default="esm2_t33_650M_UR50D")
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

# Add parent dir to path to import from experiment.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from experiment import (window_sequence, apply_missense,
                            get_esm2_embeddings_for_pairs, fetch_uniprot_sequence)
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from experiment import (window_sequence, apply_missense,
                            get_esm2_embeddings_for_pairs, fetch_uniprot_sequence)

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

# Check for partial resume: if checkpoint exists and covers all variants, skip extraction
os.makedirs(args.emb_dir, exist_ok=True)
ckpt_valid = os.path.join(args.emb_dir, "merged_valid_variants.json")
ckpt_wt = os.path.join(args.emb_dir, "merged_embeddings_wt_mean.npy")
if (os.path.exists(ckpt_wt) and os.path.exists(ckpt_valid)):
    prev = json.load(open(ckpt_valid))
    if len(prev) == len(valid):
        print("Embeddings already complete — loading from cache.")
        wt_mean  = np.load(os.path.join(args.emb_dir, "merged_embeddings_wt_mean.npy"))
        mut_mean = np.load(os.path.join(args.emb_dir, "merged_embeddings_mut_mean.npy"))
        wt_pos   = np.load(os.path.join(args.emb_dir, "merged_embeddings_wt_pos.npy"))
        mut_pos  = np.load(os.path.join(args.emb_dir, "merged_embeddings_mut_pos.npy"))
        print(f"Loaded embeddings: {wt_mean.shape}")
        print("Done.")
        sys.exit(0)

# Extract embeddings
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

wt_mean, mut_mean, wt_pos, mut_pos = get_esm2_embeddings_for_pairs(
    wt_seqs, mut_seqs, positions,
    model_name=args.model, device=device, batch_size=args.batch_size
)

# Save atomically: write valid_variants first so partial runs are detectable
with open(ckpt_valid, "w") as f:
    json.dump(valid, f)
np.save(os.path.join(args.emb_dir, "merged_embeddings_wt_mean.npy"), wt_mean)
np.save(os.path.join(args.emb_dir, "merged_embeddings_mut_mean.npy"), mut_mean)
np.save(os.path.join(args.emb_dir, "merged_embeddings_wt_pos.npy"), wt_pos)
np.save(os.path.join(args.emb_dir, "merged_embeddings_mut_pos.npy"), mut_pos)

print(f"\nSaved embeddings: {wt_mean.shape}")
print("Done.")
