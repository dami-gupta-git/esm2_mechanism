"""
Extract ESM-2 mean-pooled delta embeddings for the perturbation scan probe list.

Reads data/cache/scan_probes.json (built by perturbation_scan.py --phase 1) and
writes the WT and mutant embeddings to the canonical paths in utils_paths:

  data/embeddings/<model>/scan_wt.npy   (N_probes, D)
  data/embeddings/<model>/scan_mut.npy  (N_probes, D)

Checkpoints are written every --checkpoint_every probes so the run is resumable.

Usage (requires GPU):
    python -m esm2_mech.embeddings.embed_scan --batch_size 128
"""

import functools
import json
import os
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

from esm2_mech.utils.paths import (
    EMB_DIR,
    ESM2_MODEL,
    SCAN_CKPT_MUT,
    SCAN_CKPT_WT,
    SCAN_EMB_MUT,
    SCAN_EMB_WT,
    SCAN_PROBE_CACHE_JSON,
    SEQUENCES_EXTENDED_JSON,
    SEQUENCES_JSON,
)
from esm2_mech.embeddings.embed_variants import get_esm2_embeddings_for_pairs
from esm2_mech.utils.sequences import window_sequence, apply_missense
from esm2_mech.utils.io import save_npy

PROBE_CACHE = SCAN_PROBE_CACHE_JSON
CKPT_IDX = EMB_DIR / "scan_ckpt_idx.txt"
CHECKPOINT_EVERY = 100


def embed_scan(batch_size: int = 128) -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("GPU required for scan embedding extraction. Run on RunPod.")

    if not PROBE_CACHE.exists():
        raise FileNotFoundError(
            f"Probe list not found: {PROBE_CACHE}\n"
            f"Run: python -m esm2_mech.experiments.perturbation.perturbation_scan --phase 1"
        )
    with open(PROBE_CACHE) as f:
        d = json.load(f)
    probes = d["probes"]
    print(f"Loaded {len(probes)} probes from {PROBE_CACHE}")

    if SCAN_EMB_WT.exists() and SCAN_EMB_MUT.exists():
        print(f"Scan embeddings already complete: {SCAN_EMB_WT}")
        return

    # Load sequences
    with open(SEQUENCES_JSON) as f:
        seqs = json.load(f)
    if SEQUENCES_EXTENDED_JSON.exists():
        with open(SEQUENCES_EXTENDED_JSON) as f:
            seqs.update(json.load(f))
    print(f"Sequences loaded: {len(seqs)}")

    # Build the full filtered sequence list first so CKPT_IDX unambiguously
    # counts embedding rows (not raw probes). Probes that fail the WT-match are
    # dropped here; the checkpoint index is an offset into wt_seqs/mut_seqs.
    wt_seqs, mut_seqs, positions = [], [], []
    skipped_mismatch = 0
    for p in probes:
        seq = seqs[p["uniprot_id"]]
        wt_win, new_pos = window_sequence(seq, p["aa_pos"])
        mut_win = apply_missense(wt_win, new_pos, p["aa_wt"], p["aa_mut"])
        if mut_win is None:
            skipped_mismatch += 1
            continue
        wt_seqs.append(wt_win)
        mut_seqs.append(mut_win)
        positions.append(new_pos)
    if skipped_mismatch:
        print(f"WARNING: skipped {skipped_mismatch} probes where WT AA did not match reference sequence")

    n_filtered = len(wt_seqs)
    print(f"Total filtered probes: {n_filtered} ({skipped_mismatch} skipped)")

    # Resume from checkpoint — start_idx is a row count into the filtered lists
    start_idx = 0
    all_wt, all_mut = [], []
    if CKPT_IDX.exists():
        try:
            start_idx = int(CKPT_IDX.read_text().strip())
        except ValueError:
            print("WARNING: corrupt CKPT_IDX — discarding checkpoint and restarting")
            for ckpt in [SCAN_CKPT_WT, SCAN_CKPT_MUT, CKPT_IDX]:
                if ckpt.exists():
                    ckpt.unlink()
            start_idx = 0
        else:
            wt_arr = np.load(SCAN_CKPT_WT)
            mut_arr = np.load(SCAN_CKPT_MUT)
            if wt_arr.shape[0] != start_idx or mut_arr.shape[0] != start_idx:
                print(
                    f"WARNING: checkpoint array row counts ({wt_arr.shape[0]}, {mut_arr.shape[0]}) "
                    f"do not match CKPT_IDX ({start_idx}) — discarding misaligned checkpoint"
                )
                for ckpt in [SCAN_CKPT_WT, SCAN_CKPT_MUT, CKPT_IDX]:
                    if ckpt.exists():
                        ckpt.unlink()
                start_idx = 0
            else:
                all_wt = [wt_arr]
                all_mut = [mut_arr]
                print(f"Resuming from checkpoint: {start_idx}/{n_filtered} filtered rows done")

    wt_seqs = wt_seqs[start_idx:]
    mut_seqs = mut_seqs[start_idx:]
    positions = positions[start_idx:]

    print(f"Extracting {len(wt_seqs)} remaining probe embeddings on {device}...")

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    chunk_size = CHECKPOINT_EVERY * batch_size
    for chunk_start in range(0, len(wt_seqs), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(wt_seqs))
        wt_emb, mut_emb, _, _ = get_esm2_embeddings_for_pairs(
            wt_seqs[chunk_start:chunk_end],
            mut_seqs[chunk_start:chunk_end],
            positions[chunk_start:chunk_end],
            model_name=ESM2_MODEL,
            device=device,
            batch_size=batch_size,
        )
        all_wt.append(wt_emb)
        all_mut.append(mut_emb)
        # n_done counts rows written into the embedding arrays — same coordinate
        # system as CKPT_IDX so resume slicing is consistent.
        n_done = start_idx + chunk_end
        # Checkpoint commit order: write both arrays atomically (save_npy uses
        # .tmp + os.replace internally), then update CKPT_IDX last and
        # atomically. CKPT_IDX is the commit signal: its value is only advanced
        # after both arrays are on disk. On resume, the array row counts are
        # validated against CKPT_IDX — a mismatch means a crash occurred between
        # the two save_npy calls and the checkpoint is discarded.
        save_npy(SCAN_CKPT_WT, np.vstack(all_wt))
        save_npy(SCAN_CKPT_MUT, np.vstack(all_mut))
        ckpt_idx_tmp = str(CKPT_IDX) + ".tmp"
        Path(ckpt_idx_tmp).write_text(str(n_done))
        os.replace(ckpt_idx_tmp, CKPT_IDX)
        try:
            mem_used = torch.cuda.memory_allocated() / 1e9
            mem_res = torch.cuda.memory_reserved() / 1e9
            print(
                f"  Checkpoint: {n_done}/{n_filtered} filtered rows done  "
                f"[GPU mem: {mem_used:.1f}GB alloc / {mem_res:.1f}GB reserved]"
            )
        except Exception:
            print(f"  Checkpoint: {n_done}/{n_filtered} filtered rows done")

    wt_final = np.vstack(all_wt)
    mut_final = np.vstack(all_mut)
    save_npy(SCAN_EMB_WT, wt_final)
    save_npy(SCAN_EMB_MUT, mut_final)
    print(f"Saved {SCAN_EMB_WT}  shape={wt_final.shape}")

    for ckpt in [SCAN_CKPT_WT, SCAN_CKPT_MUT, CKPT_IDX]:
        if ckpt.exists():
            ckpt.unlink()


if __name__ == "__main__":
    import argparse
    import signal
    import traceback

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    def _sig_handler(signum, frame):
        print(f"\n[SIGNAL] Received signal {signum} — exiting")
        sys.exit(1)

    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGHUP, _sig_handler)

    try:
        embed_scan(batch_size=args.batch_size)
    except Exception:
        print("\n[FATAL ERROR]")
        traceback.print_exc()
        sys.exit(1)
