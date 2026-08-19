"""Data loaders specific to the mechanism experiments (Gerasimavicius + merged datasets)."""

from __future__ import annotations

import functools

import numpy as np

from esm2_mech.utils.io import load_variants_and_delta
from esm2_mech.utils.paths import (
    VALID_VARIANTS_JSON,
    EMB_VALID_VARIANTS_JSON,
    EMB_WT_MEAN,
    EMB_MUT_MEAN,
    EMB_WT_POS,
    EMB_MUT_POS,
)

print = functools.partial(print, flush=True)


def _label_3class(variant: dict) -> str:
    """Collapse to 3-class GOF/DN/LOF. Raises on unexpected mechanism values."""
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
    """Load mechanism variants + ESM-2 embeddings. Returns (delta_mean, delta_pos, labels, genes)."""
    variants, _labels, genes, delta_mean, delta_pos = load_variants_and_delta(
        VALID_VARIANTS_JSON,
        EMB_VALID_VARIANTS_JSON,
        EMB_WT_MEAN,
        EMB_MUT_MEAN,
        EMB_WT_POS,
        EMB_MUT_POS,
        verbose=False,
    )
    labels = np.array([_label_3class(v) for v in variants])
    print(f"  Mechanism set: {len(variants)} variants, {len(set(genes))} genes")
    return delta_mean, delta_pos, labels, genes


def load_merged(
    pfam_map: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-pooled mechanism variants: (delta_mean, labels, genes)."""
    variants, _labels, genes, delta_mean, _delta_pos = load_variants_and_delta(
        VALID_VARIANTS_JSON,
        EMB_VALID_VARIANTS_JSON,
        EMB_WT_MEAN,
        EMB_MUT_MEAN,
        verbose=False,
    )
    labels = np.array([_label_3class(variant) for variant in variants])
    print(f"  Mechanism set: {len(variants)} variants, {len(set(genes))} genes")
    return delta_mean, labels, genes
