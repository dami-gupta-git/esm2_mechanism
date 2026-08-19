"""Build the canonical pathogenicity variant list — the row-aligned subset that
matches the pathogenicity embeddings. Pure re-indexing, no GPU.
"""

from __future__ import annotations

import functools
import json

from esm2_mech.experiments.pathogenicity.pathogenicity_control import (
    ESM2_MODEL_650M,
    _derive_expected_selection,
    _validate_embedding_cache,
    load_fetched_variants,
)
from esm2_mech.utils.io import atomic_write_json
from esm2_mech.utils.paths import (
    PATHOGENICITY_CANONICAL_VARIANTS_JSON,
    SEQUENCES_JSON,
)

print = functools.partial(print, flush=True)


def main():
    print("=== Build canonical pathogenicity variant list ===")

    all_variants, fetch_metadata = load_fetched_variants()
    with open(SEQUENCES_JSON) as handle:
        sequence_cache = json.load(handle)
    expected = _derive_expected_selection(all_variants, sequence_cache)
    wt_mean, mut_mean, _ = _validate_embedding_cache(
        expected, fetch_metadata, ESM2_MODEL_650M
    )
    canonical = expected.variants

    print(f"  variants (full)   : {len(all_variants)}")
    print(f"  expected rows     : {len(canonical)}")
    print(f"  embedding rows    : wt={len(wt_mean)} mut={len(mut_mean)}")

    atomic_write_json(PATHOGENICITY_CANONICAL_VARIANTS_JSON, canonical)
    print(
        f"  Wrote {len(canonical)} variants -> {PATHOGENICITY_CANONICAL_VARIANTS_JSON}"
    )


if __name__ == "__main__":
    main()
