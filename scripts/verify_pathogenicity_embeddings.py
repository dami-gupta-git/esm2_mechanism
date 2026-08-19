#!/usr/bin/env python3
"""Validate the pathogenicity fetch and embeddings with production checks.

Run from the repository environment on the machine holding the data files:

    python scripts/verify_pathogenicity_embeddings.py
"""

import json

from esm2_mech.experiments.pathogenicity.pathogenicity_control import (
    ESM2_MODEL_650M,
    _derive_expected_selection,
    _validate_embedding_cache,
    load_fetched_variants,
)
from esm2_mech.utils.paths import SEQUENCES_JSON


def main():
    variants, fetch_metadata = load_fetched_variants()
    with open(SEQUENCES_JSON) as handle:
        sequence_cache = json.load(handle)

    expected = _derive_expected_selection(variants, sequence_cache)
    wt_mean, mut_mean, metadata = _validate_embedding_cache(
        expected, fetch_metadata, ESM2_MODEL_650M
    )

    print(f"Fetched variants: {len(variants)}")
    print(f"Expected scored variants: {len(expected.variants)}")
    print(f"Embedding rows: wt={len(wt_mean)} mut={len(mut_mean)}")
    print(f"Variant fingerprint: {expected.fingerprint}")
    print(f"Embedding fingerprint: {metadata['embedding_fingerprint']}")
    print("VERDICT: VALIDATED by the current fetch, selection, and embedding contracts")


if __name__ == "__main__":
    main()
