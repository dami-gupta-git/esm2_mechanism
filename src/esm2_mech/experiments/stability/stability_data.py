"""Shared input loading for the Megascale stability probes."""

import functools
import hashlib
import json
from collections import namedtuple

import numpy as np

from esm2_mech.experiments.stability.tsuboyama_loader import load_tsuboyama_variants
from esm2_mech.experiments.stability.build_domain_families import build_family_map
from esm2_mech.utils.constants import ESM2_MODEL, N_FOLDS
from esm2_mech.utils.data import embedding_fingerprint
from esm2_mech.utils.io import atomic_write_json, load_json_or_discard
from esm2_mech.utils.splits import random_split_cv, gene_split_cv, family_split_cv
from esm2_mech.utils.paths import (
    MEGASCALE_EMB_WT_MEAN,
    MEGASCALE_EMB_MUT_MEAN,
    MEGASCALE_EMB_WT_POS,
    MEGASCALE_EMB_MUT_POS,
    MEGASCALE_EMB_FINGERPRINT,
    MEGASCALE_DOMAIN_FAMILIES_JSON,
)

print = functools.partial(print, flush=True)

STABILITY_EMBEDDING_METADATA_VERSION = 2

# delta_pos is None unless include_pos=True (only the linear probe uses the
# per-residue delta); n_families/n_orphans are derived from the family map so the
# callers do not each recompute them.
StabilityInputs = namedtuple(
    "StabilityInputs",
    [
        "variants",
        "proteins",
        "ddg",
        "family_map",
        "delta_mean",
        "delta_pos",
        "n_families",
        "n_orphans",
    ],
)


def stability_splits(seed, n_variants, proteins, family_map, n_folds=N_FOLDS):
    """The three CV schemes for one seed, as a name→splits dict.

    random (in-distribution), domain-holdout (never train+test on the same
    domain), family-holdout (never train+test on related Pfam families). Shared by
    every stability probe so the scheme/arg triple lives in one place.
    """
    return {
        "random": random_split_cv(n_variants, n_folds, seed),
        "domain": gene_split_cv(proteins, n_folds=n_folds, seed=seed),
        "family": family_split_cv(proteins, family_map, n_folds=n_folds, seed=seed),
    }


def _load_family_map(variants):
    """Domain → Pfam family map, from cache if present else built via HMMER.

    Orphan domains (no Pfam hit) are absent from the map and so are excluded from
    the family-split only.
    """
    cached = load_json_or_discard(MEGASCALE_DOMAIN_FAMILIES_JSON)
    if cached is not None:
        return cached
    return build_family_map(variants=variants)


def variant_fingerprint(variants):
    """SHA-256 of ordered (protein, mutation_code) pairs.

    Captures both identity and order of the variant list so that a reordered or
    changed variant cache is detected when compared against the fingerprint
    saved alongside the embedding arrays.
    """
    content = "\n".join(f"{v['protein']}|{v['mutation_code']}" for v in variants)
    return hashlib.sha256(content.encode()).hexdigest()


def embedding_input_fingerprint(variants):
    """SHA-256 of every ordered input that determines a Megascale embedding row."""
    digest = hashlib.sha256()
    required_fields = ("protein", "mutation_code", "wt_seq", "mut_seq", "var_pos")
    for row_index, variant in enumerate(variants):
        missing = [key for key in required_fields if key not in variant]
        if missing:
            raise ValueError(
                f"Megascale variant row {row_index} lacks embedding input fields {missing}"
            )
        for key in required_fields:
            digest.update(str(variant[key]).encode())
            digest.update(b"\x00")
    return digest.hexdigest()


def save_fingerprint(
    variants,
    wt_mean,
    mut_mean,
    wt_pos,
    mut_pos,
    model,
    path=MEGASCALE_EMB_FINGERPRINT,
):
    """Write extraction-time input and array fingerprints for Megascale embeddings."""
    arrays = (wt_mean, mut_mean, wt_pos, mut_pos)
    if any(len(array) != len(variants) for array in arrays):
        raise ValueError(
            "Megascale embedding arrays and variants must have the same row count "
            "before extraction metadata is written"
        )
    if len({array.shape for array in arrays}) != 1:
        raise ValueError(
            f"Megascale embedding array shapes differ: {[array.shape for array in arrays]}"
        )
    atomic_write_json(
        path,
        {
            "metadata_version": STABILITY_EMBEDDING_METADATA_VERSION,
            "sha256": variant_fingerprint(variants),
            "embedding_input_fingerprint": embedding_input_fingerprint(variants),
            "n_variants": len(variants),
            "model": model,
            "mean_embedding_fingerprint": embedding_fingerprint(wt_mean, mut_mean),
            "position_embedding_fingerprint": embedding_fingerprint(wt_pos, mut_pos),
        },
    )


def _check_alignment(embedding, n_variants, path):
    if len(embedding) != n_variants:
        raise ValueError(
            f"embedding/variant row mismatch: {len(embedding)} embedding rows vs "
            f"{n_variants} variants — {path} is not row-aligned."
        )


def _check_fingerprint(variants):
    """Verify extraction metadata matches the current ordered embedding inputs."""
    if not MEGASCALE_EMB_FINGERPRINT.exists():
        raise FileNotFoundError(
            f"no embedding fingerprint file at {MEGASCALE_EMB_FINGERPRINT}; "
            "re-run embed_megascale before using the embedding arrays"
        )
    with open(MEGASCALE_EMB_FINGERPRINT) as fh:
        stored = json.load(fh)
    expected = {
        "metadata_version": STABILITY_EMBEDDING_METADATA_VERSION,
        "sha256": variant_fingerprint(variants),
        "embedding_input_fingerprint": embedding_input_fingerprint(variants),
        "n_variants": len(variants),
        "model": ESM2_MODEL,
    }
    mismatches = {
        key: {"cached": stored.get(key), "current": value}
        for key, value in expected.items()
        if stored.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Megascale embedding metadata does not match the current ordered "
            f"embedding inputs: {mismatches}. Re-run embed_megascale."
        )
    return stored


def _check_embedding_content(metadata, wt_mean, mut_mean, wt_pos=None, mut_pos=None):
    """Require loaded arrays to match their extraction-time content fingerprints."""
    current_mean = embedding_fingerprint(wt_mean, mut_mean)
    if metadata.get("mean_embedding_fingerprint") != current_mean:
        raise ValueError(
            "Megascale mean embedding arrays do not match their extraction-time "
            "content fingerprint; re-run embed_megascale"
        )
    if wt_pos is not None or mut_pos is not None:
        if wt_pos is None or mut_pos is None:
            raise ValueError("both Megascale position embedding arrays are required")
        current_position = embedding_fingerprint(wt_pos, mut_pos)
        if metadata.get("position_embedding_fingerprint") != current_position:
            raise ValueError(
                "Megascale position embedding arrays do not match their extraction-time "
                "content fingerprint; re-run embed_megascale"
            )


def load_stability_inputs(include_pos=False):
    """Load the inputs shared by every stability probe.

    Returns a StabilityInputs namedtuple. delta_pos is None unless include_pos is
    True (the per-residue delta is only used by the linear probe).
    """
    variants = load_tsuboyama_variants()
    proteins = np.array([variant["protein"] for variant in variants])
    ddg = np.array([variant["ddg"] for variant in variants])
    print(
        f"Loaded {len(variants)} Tsuboyama variants across "
        f"{len(set(proteins))} natural domains"
    )

    family_map = _load_family_map(variants)
    n_families = len(set(family_map.values()))
    n_orphans = len(set(proteins)) - len(
        {protein for protein in proteins if protein in family_map}
    )
    print(
        f"Pfam families: {len(set(proteins))} domains → {n_families} families "
        f"({n_orphans} orphans excluded from family-split)"
    )

    embedding_metadata = _check_fingerprint(variants)
    wt_mean = np.load(MEGASCALE_EMB_WT_MEAN)
    mut_mean = np.load(MEGASCALE_EMB_MUT_MEAN)
    _check_alignment(wt_mean, len(variants), MEGASCALE_EMB_WT_MEAN)
    _check_alignment(mut_mean, len(variants), MEGASCALE_EMB_MUT_MEAN)
    if wt_mean.shape != mut_mean.shape:
        raise ValueError(
            f"Megascale WT/mutant mean embedding shapes differ: "
            f"{wt_mean.shape} vs {mut_mean.shape}"
        )
    _check_embedding_content(embedding_metadata, wt_mean, mut_mean)
    delta_mean = mut_mean - wt_mean

    delta_pos = None
    if include_pos:
        wt_pos = np.load(MEGASCALE_EMB_WT_POS)
        mut_pos = np.load(MEGASCALE_EMB_MUT_POS)
        _check_alignment(wt_pos, len(variants), MEGASCALE_EMB_WT_POS)
        _check_alignment(mut_pos, len(variants), MEGASCALE_EMB_MUT_POS)
        if wt_pos.shape != mut_pos.shape:
            raise ValueError(
                f"Megascale WT/mutant position embedding shapes differ: "
                f"{wt_pos.shape} vs {mut_pos.shape}"
            )
        _check_embedding_content(
            embedding_metadata, wt_mean, mut_mean, wt_pos, mut_pos
        )
        delta_pos = mut_pos - wt_pos

    return StabilityInputs(
        variants=variants,
        proteins=proteins,
        ddg=ddg,
        family_map=family_map,
        delta_mean=delta_mean,
        delta_pos=delta_pos,
        n_families=n_families,
        n_orphans=n_orphans,
    )
