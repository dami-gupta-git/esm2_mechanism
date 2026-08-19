"""Shared embedding helpers. Callers pass explicit paths — no filenames are constructed here."""

from __future__ import annotations

import functools
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from esm2_mech.utils.io import save_npy
from esm2_mech.utils.data import validate_embedding_variant_identity

print = functools.partial(print, flush=True)

# The four embedding arrays written by get_esm2_embeddings_for_pairs, in the
# canonical order (wt_mean, mut_mean, wt_pos, mut_pos). Both the checkpoint
# writer (_flush_checkpoint) and the resume readers (inspect_four_array_checkpoint
# and the embed-step drivers) key off this single tuple so the filenames cannot
# drift between writer and reader.
EMB_ARRAY_NAMES = (
    "embeddings_wt_mean.npy",
    "embeddings_mut_mean.npy",
    "embeddings_wt_pos.npy",
    "embeddings_mut_pos.npy",
)


def inspect_four_array_checkpoint(ckpt_paths: list, n_expected: int):
    """Inspect a four-array embedding checkpoint and decide how to proceed.

    ckpt_paths is the list of the four .npy paths (order must match
    EMB_ARRAY_NAMES). Returns one of:

      ("complete", None)                  all four arrays present, equal length,
                                          and == n_expected — nothing to extract.
      ("resume", (start, arrays))         a consistent partial checkpoint of
                                          `start` rows (0 < start < n_expected);
                                          `arrays` is the four loaded np.ndarrays.
      ("reextract", None)                 no checkpoint, or it is corrupt /
                                          inconsistent / longer than expected —
                                          any partial files have been deleted.

    A truncated or partially-written .npy fails to mmap with OSError/EOFError
    (bad header or data), not only ValueError, so all three are caught and
    treated as a corrupt checkpoint to re-extract.
    """
    if not all(os.path.exists(p) for p in ckpt_paths):
        return "reextract", None

    try:
        row_counts = [np.load(p, mmap_mode="r").shape[0] for p in ckpt_paths]
    except (ValueError, OSError, EOFError):
        print("WARNING: corrupt checkpoint — re-extracting", flush=True)
        for p in ckpt_paths:
            if os.path.exists(p):
                os.remove(p)
        return "reextract", None

    if len(set(row_counts)) > 1:
        print(
            f"WARNING: checkpoint row counts inconsistent {row_counts} — re-extracting",
            flush=True,
        )
        for p in ckpt_paths:
            if os.path.exists(p):
                os.remove(p)
        return "reextract", None

    n_on_disk = row_counts[0]
    if n_on_disk == n_expected:
        return "complete", None
    if 0 < n_on_disk < n_expected:
        arrays = tuple(np.load(p) for p in ckpt_paths)
        return "resume", (n_on_disk, arrays)

    # n_on_disk == 0, or n_on_disk > n_expected — either way the checkpoint
    # cannot be trusted; delete and re-extract.
    print(
        f"WARNING: checkpoint row count {n_on_disk} != expected {n_expected} — re-extracting",
        flush=True,
    )
    for p in ckpt_paths:
        if os.path.exists(p):
            os.remove(p)
    return "reextract", None


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
    occupy the .npy arrays (row i of the json == row i of every array). Downstream
    loaders require it and reject arrays whose row identities do not match.

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
    arrays = dict(zip(
        EMB_ARRAY_NAMES,
        (wt_mean_list, mut_mean_list, wt_pos_list, mut_pos_list),
    ))
    for name, lst in arrays.items():
        save_npy(os.path.join(out_dir, name), np.stack(lst))
    valid_path = os.path.join(out_dir, "embedded_variants.json")
    tmp_valid = valid_path + ".tmp"
    with open(tmp_valid, "w") as f:
        json.dump(valid_slice, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_valid, valid_path)
    print(f"  Checkpoint: {n_done} variants flushed to {out_dir}", flush=True)


def inspect_variant_embedding_checkpoint(
    ckpt_paths: list,
    expected_variants: list[dict],
    embedded_variants_path: Path,
    identity_fingerprint=None,
):
    """Inspect an embedding checkpoint and verify its row-identity sidecar.

    A complete or resumable array checkpoint without a matching sidecar cannot be
    associated with the current variants. Such arrays are removed and rebuilt.
    `identity_fingerprint` supports variant schemas other than the default
    gene/UniProt/substitution fields while retaining the same fail-closed behavior.
    """
    status, payload = inspect_four_array_checkpoint(ckpt_paths, len(expected_variants))
    if status == "reextract":
        return status, payload

    n_on_disk = len(expected_variants) if status == "complete" else payload[0]
    try:
        expected_prefix = expected_variants[:n_on_disk]
        if identity_fingerprint is None:
            validate_embedding_variant_identity(
                expected_prefix, embedded_variants_path
            )
        else:
            if not embedded_variants_path.exists():
                raise FileNotFoundError(
                    f"embedding row-identity sidecar missing: {embedded_variants_path}"
                )
            with open(embedded_variants_path) as handle:
                embedded_variants = json.load(handle)
            if identity_fingerprint(expected_prefix) != identity_fingerprint(
                embedded_variants
            ):
                raise ValueError(
                    f"{embedded_variants_path} does not match the current embedding "
                    "inputs and row order"
                )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"WARNING: {exc} — re-extracting embedding checkpoint", flush=True)
        for path in ckpt_paths:
            if os.path.exists(path):
                os.remove(path)
        if embedded_variants_path.exists():
            embedded_variants_path.unlink()
        return "reextract", None
    return status, payload


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


def masked_aa_log_probs(
    model, alphabet, device: str, items: list, aa_order: str, batch_size: int = 32,
) -> dict:
    """Batch masked-token log P(aa) over aa_order at one scored position per item.

    items: list of (key, masked_seq, tok_idx) where masked_seq already has the
    scored residue replaced with '<mask>' and tok_idx is that residue's index into
    the model's token axis (position 0 is BOS, so tok_idx equals the masked
    residue's 1-indexed position in the unmasked sequence).

    Returns {key: np.ndarray[len(aa_order)]} of log-softmax probabilities, one row
    per item, restricted to aa_order.
    """
    import torch

    batch_converter = alphabet.get_batch_converter()
    aa_idx = {aa: alphabet.get_idx(aa) for aa in aa_order}

    out = {}
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        data = [(key, seq) for key, seq, _ in batch]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        with torch.inference_mode():
            logits = model(tokens)["logits"].cpu().float()
        for i, (key, _, tok_idx) in enumerate(batch):
            log_probs = torch.log_softmax(logits[i, tok_idx], dim=-1).numpy()
            out[key] = np.array([log_probs[aa_idx[aa]] for aa in aa_order], dtype=np.float32)
    return out


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
    # The variant list and both embedding arrays must be row-aligned. Check
    # explicitly: a silent mismatch would either drop trailing embedding rows
    # (len(variants) < rows) or raise an opaque IndexError (len(variants) > rows),
    # and a wt/mut shape mismatch would broadcast wrong instead of erroring.
    if not (len(variants) == wt_emb.shape[0] == mut_emb.shape[0]):
        raise ValueError(
            f"row mismatch: {len(variants)} variants vs {wt_emb.shape[0]} wt "
            f"vs {mut_emb.shape[0]} mut embedding rows — not row-aligned."
        )
    delta = mut_emb - wt_emb

    gene_delta: dict[str, list[np.ndarray]] = defaultdict(list)
    for i, v in enumerate(variants):
        gene_delta[v["gene"].upper()].append(delta[i])
    return gene_delta
