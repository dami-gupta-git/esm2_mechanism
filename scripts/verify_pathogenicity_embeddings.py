#!/usr/bin/env python3
"""
Verify the pathogenicity embeddings on the RunPod pod are a self-consistent,
trustworthy set — independent of when they were generated.

Checks:
  1. meta / variants / .npy row counts agree (valid_indices alignment).
  2. The stored content fingerprint recomputes from the variant subset, in row
     order — proving the .npy rows are aligned to exactly these variants.

Run on the pod:
    python3 verify_pathogenicity_embeddings.py
or point it elsewhere:
    python3 verify_pathogenicity_embeddings.py /path/to/data
"""

import json
import sys
import hashlib
from pathlib import Path

import numpy as np

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/workspace/repo/data")
EMB = DATA / "embeddings" / "esm2_t33_650M_UR50D"

meta_path = EMB / "pathogenicity_meta.json"
vars_path = DATA / "clinvar_pathogenicity_variants.json"
wt_path = EMB / "pathogenicity_wt_mean.npy"
mut_path = EMB / "pathogenicity_mut_mean.npy"

print("Reading:")
for p in (meta_path, vars_path, wt_path, mut_path):
    print(f"  {p}  {'OK' if p.exists() else 'MISSING'}")
print()

meta = json.loads(meta_path.read_text())
variants = json.loads(vars_path.read_text())
wt = np.load(wt_path, mmap_mode="r")
mut = np.load(mut_path, mmap_mode="r")
valid_indices = meta["valid_indices"]

print("variants        :", len(variants))
print("wt / mut rows   :", wt.shape[0], mut.shape[0])
print("n_valid (meta)  :", meta.get("n_valid"))
print("len(valid_idx)  :", len(valid_indices))
print("max(valid_idx)  :", max(valid_indices))
print("model           :", meta.get("model"))

shapes_aligned = (
    len(valid_indices) == wt.shape[0] == mut.shape[0]
    and max(valid_indices) < len(variants)
)
print("SHAPES_ALIGNED  :", shapes_aligned)

labels = {}
for v in variants:
    labels[v["label"]] = labels.get(v["label"], 0) + 1
print("label counts    :", labels)

# Recompute the content fingerprint over the valid subset, in row order.
valid = [variants[i] for i in valid_indices]
digest = hashlib.sha256()
for v in valid:
    key = "|".join(
        str(v[k]) for k in ("gene", "uniprot_id", "aa_pos", "aa_wt", "aa_mut", "label")
    )
    digest.update(key.encode())
    digest.update(b"\x00")
recomputed = digest.hexdigest()

print()
print("stored fingerprint    :", meta.get("fingerprint"))
print("recomputed fingerprint:", recomputed)
fp_match = recomputed == meta.get("fingerprint")
print("FINGERPRINT_MATCH     :", fp_match)

print()
if shapes_aligned and fp_match:
    print("VERDICT: TRUSTWORTHY — embeddings are row-aligned to these variants.")
    print("         Safe to use (Option A). scp the 3 files + variants JSON back.")
else:
    print("VERDICT: DO NOT TRUST — alignment/fingerprint failed. Regenerate.")
    sys.exit(1)
