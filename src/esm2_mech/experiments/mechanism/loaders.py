"""Data loaders specific to the mechanism experiments (Gerasimavicius + merged datasets)."""

from __future__ import annotations

import functools
import json

import numpy as np

from esm2_mech.utils.paths import (
    VALID_VARIANTS_JSON,
    EMB_WT_MEAN,
    EMB_MUT_MEAN,
    EMB_WT_POS,
    EMB_MUT_POS,
)

print = functools.partial(print, flush=True)


def _label_3class(variant: dict) -> str:
    """Collapse a variant's mechanism to the 3-class GOF/DN/LOF label.

    HI and AR both map to LOF; GOF/DN/LOF pass through. Raises on anything else
    rather than guessing — an unexpected mechanism is a data error, not LOF.
    """
    if "label_3class" in variant:
        return variant["label_3class"]
    mech = variant.get("mechanism")
    if mech in ("HI", "AR", "LOF"):
        return "LOF"
    if mech in ("GOF", "DN"):
        return mech
    raise ValueError(
        f"Variant {variant.get('gene')} pos {variant.get('aa_pos')} has unexpected mechanism {mech!r}"
    )


def load_mechanism_variants(
    pfam_map: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the mechanism-task variants + ESM-2 embeddings, row-aligned.

    Reads VALID_VARIANTS_JSON — the variant list the main embeddings were
    extracted from (same identity and order as the .npy rows) — and labels every
    variant GOF/DN/LOF. This is the full merged set (Gerasimavicius + ClinVar/G2P),
    not Gerasimavicius alone.

    Returns (delta_mean, delta_pos, labels, genes). pfam_map is accepted for
    backward-compatible call sites but is unused (labels come from the variants).
    """
    with open(VALID_VARIANTS_JSON) as f:
        variants = json.load(f)

    labels = np.array([_label_3class(v) for v in variants])
    genes = np.array([v["gene"] for v in variants])

    wt = np.load(EMB_WT_MEAN)
    mut = np.load(EMB_MUT_MEAN)
    delta_mean = mut - wt
    wt_p = np.load(EMB_WT_POS)
    mut_p = np.load(EMB_MUT_POS)
    delta_pos = mut_p - wt_p

    if len(delta_mean) != len(labels):
        raise ValueError(
            f"embedding/variant row mismatch: {len(delta_mean)} embedding rows vs "
            f"{len(labels)} variants — {VALID_VARIANTS_JSON.name} is not row-aligned."
        )
    print(f"  Mechanism set: {len(variants)} variants, {len(set(genes))} genes")
    return delta_mean, delta_pos, labels, genes


def load_merged(pfam_map: dict | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-pooled mechanism variants + embeddings: (delta_mean, labels, genes).

    Thin wrapper over load_mechanism_variants for callers that don't need the
    per-residue (delta_pos) view.
    """
    delta_mean, _delta_pos, labels, genes = load_mechanism_variants(pfam_map)
    return delta_mean, labels, genes
