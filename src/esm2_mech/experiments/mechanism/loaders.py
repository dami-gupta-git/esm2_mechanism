"""Data loaders specific to the mechanism experiments (Gerasimavicius + merged datasets)."""

from __future__ import annotations

import functools
import json

import numpy as np

from esm2_mech.utils.paths import (
    GERAS_VALID_VARIANTS_JSON,
    VALID_VARIANTS_JSON,
    EMB_WT_MEAN,
    EMB_MUT_MEAN,
    EMB_WT_POS,
    EMB_MUT_POS,
)

print = functools.partial(print, flush=True)


def load_geras(pfam_map: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Gerasimavicius variants + embeddings.

    Returns (delta_mean, delta_pos, labels, genes).
    Raises FileNotFoundError if the geras_valid_variants.json cache is missing.
    """
    if not GERAS_VALID_VARIANTS_JSON.exists():
        raise FileNotFoundError(
            f"{GERAS_VALID_VARIANTS_JSON} not found.\n"
            "This file must list exactly the variants whose embeddings were extracted "
            "(in the same order), so that label and embedding indices align.\n"
            "Generate it by running esm2_mechanism.py and saving the valid_variants "
            "list after apply_missense filtering."
        )
    with open(GERAS_VALID_VARIANTS_JSON) as f:
        variants = json.load(f)

    for v in variants:
        if "label_3class" not in v:
            mech = v.get("mechanism")
            if mech in ("HI", "AR"):
                v["label_3class"] = "LOF"
            elif mech in ("GOF", "DN", "LOF"):
                v["label_3class"] = mech
            else:
                raise ValueError(
                    f"Variant {v.get('gene')} pos {v.get('aa_pos')} has unexpected mechanism {mech!r}"
                )
    labels = np.array([v["label_3class"] for v in variants])
    genes = np.array([v["gene"] for v in variants])

    wt = np.load(EMB_WT_MEAN)
    mut = np.load(EMB_MUT_MEAN)
    delta_mean = mut - wt
    wt_p = np.load(EMB_WT_POS)
    mut_p = np.load(EMB_MUT_POS)
    delta_pos = mut_p - wt_p

    assert len(delta_mean) == len(labels), (
        f"Geras embedding/variant count mismatch: {len(delta_mean)} vs {len(labels)}"
    )
    print(f"  Gerasimavicius: {len(variants)} variants, {len(set(genes))} genes")
    return delta_mean, delta_pos, labels, genes


def load_merged(pfam_map: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load merged (valid_variants.json) variants + embeddings.

    Returns (delta_mean, labels, genes).
    """
    with open(VALID_VARIANTS_JSON) as f:
        variants = json.load(f)
    for v in variants:
        if "label_3class" not in v:
            mech = v.get("mechanism")
            if mech in ("HI", "AR"):
                v["label_3class"] = "LOF"
            elif mech in ("GOF", "DN", "LOF"):
                v["label_3class"] = mech
            else:
                raise ValueError(
                    f"Variant {v.get('gene')} pos {v.get('aa_pos')} has unexpected mechanism {mech!r}"
                )
    labels = np.array([v["label_3class"] for v in variants])
    genes = np.array([v["gene"] for v in variants])

    wt = np.load(EMB_WT_MEAN)
    mut = np.load(EMB_MUT_MEAN)
    delta_mean = mut - wt

    assert len(delta_mean) == len(labels), (
        f"Merged embedding/variant count mismatch: {len(delta_mean)} vs {len(labels)}"
    )
    print(f"  Merged: {len(variants)} variants, {len(set(genes))} genes")
    return delta_mean, labels, genes
