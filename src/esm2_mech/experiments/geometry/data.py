"""Validated inputs shared by the pathogenicity geometry analyses."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from esm2_mech.embeddings.embed_variants import ESM2_MODEL_650M
from esm2_mech.utils.data import (
    embedding_fingerprint,
    pfam_fingerprint,
    variants_fingerprint,
)
from esm2_mech.utils.paths import (
    PATH_EMB_META,
    PATH_EMB_MUT_MEAN,
    PATH_EMB_WT_MEAN,
    PATHOGENICITY_CANONICAL_VARIANTS_JSON,
)


@dataclass(frozen=True)
class PathogenicityGeometryInputs:
    variants: list[dict]
    wt: np.ndarray
    mut: np.ndarray
    delta: np.ndarray
    genes: np.ndarray
    labels: np.ndarray
    variant_fingerprint: str
    embedding_fingerprint: str
    model: str


def pathogenicity_label(label: str) -> int:
    """Map the two allowed labels without treating an unknown label as benign."""
    if label == "pathogenic":
        return 1
    if label == "benign":
        return 0
    raise ValueError(
        f"unexpected pathogenicity label {label!r} (expected 'pathogenic' or 'benign')"
    )


def load_pathogenicity_geometry_inputs() -> PathogenicityGeometryInputs:
    """Load row-aligned variants and embeddings, verified by content fingerprint."""
    with open(PATHOGENICITY_CANONICAL_VARIANTS_JSON) as handle:
        variants = json.load(handle)
    with open(PATH_EMB_META) as handle:
        metadata = json.load(handle)

    wt = np.load(PATH_EMB_WT_MEAN)
    mut = np.load(PATH_EMB_MUT_MEAN)
    n_variants = len(variants)
    if not (n_variants == len(wt) == len(mut)):
        raise ValueError(
            f"geometry input row mismatch: {n_variants} canonical variants, "
            f"{len(wt)} WT embedding rows, and {len(mut)} mutant embedding rows"
        )

    current_variant_fingerprint = variants_fingerprint(variants)
    if metadata.get("fingerprint") != current_variant_fingerprint:
        raise ValueError(
            "canonical pathogenicity variants do not match the embedding metadata "
            "fingerprint; rebuild the canonical list or re-embed"
        )
    if metadata.get("n_valid") != n_variants:
        raise ValueError(
            f"embedding metadata records {metadata.get('n_valid')} valid variants, "
            f"but the canonical geometry file contains {n_variants}"
        )
    if metadata.get("model") != ESM2_MODEL_650M:
        raise ValueError(
            f"geometry expects model {ESM2_MODEL_650M!r}, but embedding metadata "
            f"records {metadata.get('model')!r}"
        )

    labels = np.array([pathogenicity_label(variant["label"]) for variant in variants])
    genes = np.array([variant["gene"] for variant in variants])
    return PathogenicityGeometryInputs(
        variants=variants,
        wt=wt,
        mut=mut,
        delta=mut - wt,
        genes=genes,
        labels=labels,
        variant_fingerprint=current_variant_fingerprint,
        embedding_fingerprint=embedding_fingerprint(wt, mut),
        model=metadata["model"],
    )


def pathogenicity_geometry_provenance(
    inputs: PathogenicityGeometryInputs, pfam_map: dict
) -> dict:
    """Fingerprints for every mutable input used by a geometry result."""
    return {
        "variant_fingerprint": inputs.variant_fingerprint,
        "embedding_fingerprint": inputs.embedding_fingerprint,
        "pfam_fingerprint": pfam_fingerprint(pfam_map, inputs.genes.tolist()),
        "model": inputs.model,
        "n_variants": len(inputs.variants),
    }


def mechanism_geometry_provenance(
    delta: np.ndarray, labels: np.ndarray, genes: np.ndarray, pfam_map: dict
) -> dict:
    """Fingerprint the row-aligned mechanism inputs used by geometry probes."""
    if not (len(delta) == len(labels) == len(genes)):
        raise ValueError("mechanism geometry inputs are not row-aligned")
    digest = hashlib.sha256()
    for gene, label in zip(genes, labels):
        digest.update(f"{gene}|{label}".encode())
        digest.update(b"\x00")
    return {
        "row_label_fingerprint": digest.hexdigest(),
        "delta_embedding_fingerprint": embedding_fingerprint(delta),
        "pfam_fingerprint": pfam_fingerprint(pfam_map, genes.tolist()),
        "n_variants": len(labels),
    }
