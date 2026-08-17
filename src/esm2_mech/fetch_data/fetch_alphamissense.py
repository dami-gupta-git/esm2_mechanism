"""Fetch AlphaMissense scores for pathogenicity variants."""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path
from typing import Optional

from esm2_mech.fetch_data.alphamissense_common import (
    AM_URL,
    build_gene_uniprot_map,
    build_lookup,
    download_am,
    stream_am_filter,
)
from esm2_mech.utils.io import atomic_write_json
from esm2_mech.utils.paths import AM_CACHE_FILE, DATA_DIR, VALID_VARIANTS_JSON

print = functools.partial(print, flush=True)

AM_CACHE = AM_CACHE_FILE
AM_OUT = DATA_DIR / "alphamissense_scores_full.json"
AM_MERGED_VALID_VARIANTS = VALID_VARIANTS_JSON
AM_PATHOGENICITY_VARIANTS = DATA_DIR / "pathogenicity_valid_variants.json"


def main(
    am_file: Optional[Path] = None,
    no_download: bool = False,
    out: Optional[Path] = None,
) -> None:
    am_file = am_file or AM_CACHE
    out = out or AM_OUT

    required = [AM_MERGED_VALID_VARIANTS, AM_PATHOGENICITY_VARIANTS]
    missing_inputs = [p for p in required if not p.exists()]
    if missing_inputs:
        raise FileNotFoundError(
            "Required input(s) not found:\n" + "\n".join(f"  {p}" for p in missing_inputs)
        )

    am_file.parent.mkdir(parents=True, exist_ok=True)

    with open(AM_MERGED_VALID_VARIANTS) as f:
        merged_variants = json.load(f)
    g2u = build_gene_uniprot_map(merged_variants)
    print(f"gene -> uniprot: {len(g2u):,} entries")
    with open(AM_PATHOGENICITY_VARIANTS) as f:
        variants = json.load(f)
    print(f"target variants: {len(variants):,}")
    index, skipped_no_uniprot, skipped_key_collision = build_lookup(variants, g2u)
    print(f"variants with UniProt mapping: {len(index):,}")
    if skipped_no_uniprot:
        print(
            f"variants missing UniProt mapping (first 5): {[s['gene'] for s in skipped_no_uniprot[:5]]}"
        )
    if skipped_key_collision:
        print(f"variants dropped due to key collision: {len(skipped_key_collision)}")

    if not am_file.exists():
        if no_download:
            print(
                f"ERROR: --no-download set but {am_file} does not exist",
                file=sys.stderr,
            )
            sys.exit(2)
        download_am(AM_URL, am_file)

    scores = stream_am_filter(am_file, index)
    print(f"matched scores: {len(scores):,} / {len(index):,}")

    atomic_write_json(out, scores, indent=2, sort_keys=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch AlphaMissense scores for target variants."
    )
    parser.add_argument(
        "--am-file",
        type=Path,
        default=None,
        help="Local path for the AM bulk file (default: data/cache/AlphaMissense_aa_substitutions.tsv.gz).",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download; require --am-file to exist.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: data/alphamissense_scores_full.json).",
    )
    args = parser.parse_args()
    main(am_file=args.am_file, no_download=args.no_download, out=args.out)
