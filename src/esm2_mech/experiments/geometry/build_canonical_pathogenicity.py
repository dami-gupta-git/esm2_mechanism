"""Build the canonical pathogenicity variant list — the row-aligned subset that
matches the pathogenicity embeddings. Pure re-indexing, no GPU.
"""

from __future__ import annotations

import functools
import json

import numpy as np

from esm2_mech.utils.data import variants_fingerprint
from esm2_mech.utils.io import atomic_write_json
from esm2_mech.utils.paths import (
    CLINVAR_PATHOGENICITY_VARIANTS_JSON,
    PATH_EMB_META,
    PATH_EMB_MUT_MEAN,
    PATH_EMB_WT_MEAN,
    PATHOGENICITY_CANONICAL_VARIANTS_JSON,
)

print = functools.partial(print, flush=True)


def main():
    print("=== Build canonical pathogenicity variant list ===")

    with open(CLINVAR_PATHOGENICITY_VARIANTS_JSON) as handle:
        all_variants = json.load(handle)
    with open(PATH_EMB_META) as handle:
        meta = json.load(handle)

    valid_indices = meta["valid_indices"]
    wt_rows = np.load(PATH_EMB_WT_MEAN, mmap_mode="r").shape[0]
    mut_rows = np.load(PATH_EMB_MUT_MEAN, mmap_mode="r").shape[0]

    print(f"  variants (full)   : {len(all_variants)}")
    print(f"  valid_indices     : {len(valid_indices)}")
    print(f"  embedding rows    : wt={wt_rows} mut={mut_rows}")

    if not (len(valid_indices) == wt_rows == mut_rows):
        raise ValueError(
            f"valid_indices ({len(valid_indices)}) must equal embedding rows "
            f"(wt={wt_rows}, mut={mut_rows}); meta and .npy disagree."
        )

    # Guard against variant list regeneration since embed time.
    if meta["n"] != len(all_variants):
        raise ValueError(
            f"variant list length {len(all_variants)} != meta n {meta['n']}; "
            f"embeddings were built from a different variant list. "
            f"Re-embed, or restore the variant list the embeddings were built from."
        )
    if max(valid_indices) >= len(all_variants):
        raise ValueError(
            f"valid_indices max {max(valid_indices)} out of range for "
            f"{len(all_variants)} variants."
        )

    canonical = [all_variants[i] for i in valid_indices]

    fingerprint = variants_fingerprint(canonical)
    if fingerprint != meta.get("fingerprint"):
        raise ValueError(
            "Fingerprint of the selected subset does not match the embed-time "
            f"fingerprint ({meta.get('fingerprint')}); the variant list has drifted "
            "from the one the embeddings were built on. Re-embed, or restore the "
            "original variant list."
        )

    atomic_write_json(PATHOGENICITY_CANONICAL_VARIANTS_JSON, canonical)
    print(
        f"  Wrote {len(canonical)} variants -> {PATHOGENICITY_CANONICAL_VARIANTS_JSON}"
    )


if __name__ == "__main__":
    main()
