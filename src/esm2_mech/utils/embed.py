"""Shared embedding helpers. Callers pass explicit paths — no filenames are constructed here."""

from __future__ import annotations

import functools
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from esm2_mech.utils.io import save_npy

print = functools.partial(print, flush=True)


def _flush_checkpoint(
    out_dir: str,
    wt_mean_list: list,
    mut_mean_list: list,
    wt_pos_list: list,
    mut_pos_list: list,
    valid_slice: list,
    n_done: int,
) -> None:
    """Atomically write accumulated embeddings and embedded_variants.json to checkpoint files.

    embedded_variants.json is the row-aligned slice of variants whose embeddings
    occupy the .npy arrays (row i of the json == row i of every array). It exists
    purely as a sanity-check / provenance artifact — NO code reads it.

    In practice it is always identical to the input data/valid_variants.json: the
    embed step's _build_valid_pairs re-applies the same three filters (empty
    uniprot_id, uid not in seq_cache, apply_missense → None) that
    fetch_data/build_valid_variants already applied to produce valid_variants.json,
    so the embed step can never drop a row further. It is written anyway so the
    RUNBOOK verification step can confirm row count == .npy array length without
    trusting that invariant. If you ever change the embed-step filters to diverge
    from build_valid_variants, this file becomes the authoritative row index and
    downstream loaders should read it instead of valid_variants.json.
    """
    arrays = {
        "embeddings_wt_mean": wt_mean_list,
        "embeddings_mut_mean": mut_mean_list,
        "embeddings_wt_pos": wt_pos_list,
        "embeddings_mut_pos": mut_pos_list,
    }
    for name, lst in arrays.items():
        save_npy(os.path.join(out_dir, f"{name}.npy"), np.stack(lst))
    valid_path = os.path.join(out_dir, "embedded_variants.json")
    tmp_valid = valid_path + ".tmp"
    with open(tmp_valid, "w") as f:
        json.dump(valid_slice, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_valid, valid_path)
    print(f"  Checkpoint: {n_done} variants flushed to {out_dir}", flush=True)


def load_esm2_model(model_name: str, device: str = "cuda"):
    """Load an ESM-2 model and return (model, alphabet). Callers that run multiple
    embedding passes should call this once and pass the result to
    get_esm2_embeddings_for_pairs to avoid reloading the model on each call."""
    import esm

    print(f"Loading ESM-2 model {model_name}...", flush=True)
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device).eval()
    print(f"Model loaded.", flush=True)
    return model, alphabet


def get_esm2_embeddings_for_pairs(
    wt_seqs: list[str],
    mut_seqs: list[str],
    aa_positions: list[int],
    valid_variants: list[dict] | None = None,
    out_dir: str | None = None,
    model_name: str = "esm2_t33_650M_UR50D",
    device: str = "cuda",
    batch_size: int = 32,
    checkpoint_every: int = 100,
    resume_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    loaded_model=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract ESM-2 embeddings for WT/mutant sequence pairs.

    WT and mutant are interleaved in the same batch to halve forward passes.
    If out_dir is provided, partial results are checkpointed every checkpoint_every variants.
    Pass loaded_model=(model, alphabet) to reuse an already-loaded model across calls.

    Returns:
        wt_mean:  (N, D) mean-pooled WT embeddings
        mut_mean: (N, D) mean-pooled mutant embeddings
        wt_pos:   (N, D) per-residue WT embedding at variant position
        mut_pos:  (N, D) per-residue mutant embedding at variant position
    """
    import torch

    if loaded_model is not None:
        model, alphabet = loaded_model
    else:
        model, alphabet = load_esm2_model(model_name, device)
    batch_converter = alphabet.get_batch_converter()
    n_layers = model.num_layers

    # Seed output lists from checkpoint if resuming, otherwise start empty.
    if resume_arrays is not None:
        wt_mean_r, mut_mean_r, wt_pos_r, mut_pos_r = resume_arrays
        wt_mean_list = [wt_mean_r[i] for i in range(len(wt_mean_r))]
        mut_mean_list = [mut_mean_r[i] for i in range(len(mut_mean_r))]
        wt_pos_list = [wt_pos_r[i] for i in range(len(wt_pos_r))]
        mut_pos_list = [mut_pos_r[i] for i in range(len(mut_pos_r))]
    else:
        wt_mean_list, mut_mean_list = [], []
        wt_pos_list, mut_pos_list = [], []

    total = len(wt_seqs)
    n_done = len(wt_mean_list)
    last_flush = n_done
    for batch_start in range(0, total, batch_size):
        pairs = list(zip(
            wt_seqs[batch_start : batch_start + batch_size],
            mut_seqs[batch_start : batch_start + batch_size],
            aa_positions[batch_start : batch_start + batch_size],
        ))
        # Interleave WT and mutant in the same batch to halve forward passes.
        interleaved = []
        for j, (wt, mut, _) in enumerate(pairs):
            interleaved.append((f"wt{j}", wt))
            interleaved.append((f"mut{j}", mut))

        _, _, tokens = batch_converter(interleaved)
        tokens = tokens.to(device, non_blocking=True)
        with torch.inference_mode():
            out = model(tokens, repr_layers=[n_layers])
        # Keep representations on-device: the mean-pool over residues and the
        # per-position gather are GPU-friendly reductions, and reducing here means
        # we transfer (2*batch, D) instead of the full (2*batch, seqlen, D) tensor.
        reps = out["representations"][n_layers].float()

        # Reduce each item on-device into per-batch stacks, then move once to CPU.
        wt_mean_batch, mut_mean_batch = [], []
        wt_pos_batch, mut_pos_batch = [], []
        for j, (wt, mut, var_pos) in enumerate(pairs):
            wt_rep = reps[2 * j]
            mut_rep = reps[2 * j + 1]

            # Slice off BOS/EOS tokens (token 0 = BOS, tokens 1..len = residues).
            wt_mean_batch.append(wt_rep[1 : len(wt) + 1].mean(0))
            mut_mean_batch.append(mut_rep[1 : len(mut) + 1].mean(0))

            # var_pos is 1-indexed; BOS occupies token 0, so sequence tokens are 1..len(wt)
            if 0 < var_pos <= len(wt):
                wt_pos_batch.append(wt_rep[var_pos])
                mut_pos_batch.append(mut_rep[var_pos])
            else:
                raise ValueError(
                    f"var_pos={var_pos} out of range for sequence length {len(wt)}"
                    f" (batch item {j}) — this should have been caught by _build_valid_pairs"
                )

        # One device→host transfer per batch instead of 4 per variant.
        wt_mean_np = torch.stack(wt_mean_batch).cpu().numpy()
        mut_mean_np = torch.stack(mut_mean_batch).cpu().numpy()
        wt_pos_np = torch.stack(wt_pos_batch).cpu().numpy()
        mut_pos_np = torch.stack(mut_pos_batch).cpu().numpy()
        wt_mean_list.extend(wt_mean_np)
        mut_mean_list.extend(mut_mean_np)
        wt_pos_list.extend(wt_pos_np)
        mut_pos_list.extend(mut_pos_np)

        n_done = len(wt_mean_list)
        if n_done - last_flush >= checkpoint_every:
            if out_dir is not None:
                _flush_checkpoint(
                    out_dir, wt_mean_list, mut_mean_list, wt_pos_list, mut_pos_list,
                    (valid_variants or [])[:n_done], n_done,
                )
            else:
                print(f"  Embedded {n_done}/{total} variant pairs", flush=True)
            last_flush = n_done

    if n_done > last_flush:
        if out_dir is not None:
            _flush_checkpoint(
                out_dir, wt_mean_list, mut_mean_list, wt_pos_list, mut_pos_list,
                (valid_variants or [])[:n_done], n_done,
            )
        else:
            print(f"  Embedded {n_done}/{total} variant pairs", flush=True)

    return (
        np.stack(wt_mean_list),
        np.stack(mut_mean_list),
        np.stack(wt_pos_list),
        np.stack(mut_pos_list),
    )


def unpack_run_data(data: dict) -> dict:
    """Unpack a pre-loaded data dict and compute mean-pooled and per-residue deltas.

    Returns the same keys as the input dict plus:
        deltas_mean : (n, d) float32  — mean-pooled mutant − WT
        deltas_pos  : (n, d) float32  — per-residue mutant − WT
    """
    emb_wt_mean = data["emb_wt_mean"]
    emb_mut_mean = data["emb_mut_mean"]
    emb_wt_pos = data["emb_wt_pos"]
    emb_mut_pos = data["emb_mut_pos"]

    deltas_mean = emb_mut_mean - emb_wt_mean
    deltas_pos = emb_mut_pos - emb_wt_pos
    print(f"Delta embedding shape: {deltas_mean.shape}")

    return {**data, "deltas_mean": deltas_mean, "deltas_pos": deltas_pos}


def load_gene_delta(
    variants_path: Path,
    wt_path: Path,
    mut_path: Path,
) -> dict[str, list[np.ndarray]]:
    """Load mean-pooled embeddings and return a gene → list-of-delta-vectors mapping.

    Keys are upper-cased gene names. Callers average the list to get one vector per gene.
    """
    with open(variants_path) as f:
        variants = json.load(f)
    wt_emb = np.load(wt_path)
    mut_emb = np.load(mut_path)
    delta = mut_emb - wt_emb

    gene_delta: dict[str, list[np.ndarray]] = defaultdict(list)
    for i, v in enumerate(variants):
        gene_delta[v["gene"].upper()].append(delta[i])
    return gene_delta
