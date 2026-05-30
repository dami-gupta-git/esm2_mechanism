"""Shared embedding helpers. Callers pass explicit paths — no filenames are constructed here."""

from __future__ import annotations

import functools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)


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
