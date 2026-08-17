"""Fetch AlphaMissense scores for mechanism variants."""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path
from typing import Optional

from esm2_mech.fetch_data.alphamissense_common import (
    build_gene_uniprot_map,
    build_lookup,
    download_am,
    stream_am_filter,
    AM_URL,
)
from esm2_mech.utils.io import atomic_write_json
from esm2_mech.utils.paths import AM_CACHE_FILE, DATA_DIR, VARIANTS_JSON

print = functools.partial(print, flush=True)

AM_OUT = DATA_DIR / "alphamissense_scores_full.json"


def main(am_file: Optional[Path] = None, no_download: bool = False, out: Optional[Path] = None) -> None:
    am_file = am_file or AM_CACHE_FILE
    out = out or AM_OUT

    if not VARIANTS_JSON.exists():
        raise FileNotFoundError(
            f"Required input not found: {VARIANTS_JSON}\n"
            "Run: python -m esm2_mech.fetch_data.fetch_variants --step merge first"
        )

    am_file.parent.mkdir(parents=True, exist_ok=True)

    with open(VARIANTS_JSON) as f:
        variants = json.load(f)
    print(f"Loaded {len(variants):,} mechanism variants")

    g2u = build_gene_uniprot_map(variants)
    print(f"gene -> uniprot: {len(g2u):,} entries")

    index, skipped_no_uniprot, skipped_key_collision = build_lookup(variants, g2u)
    print(f"variants with UniProt mapping: {len(index):,}")
    if skipped_no_uniprot:
        print(f"  variants missing UniProt mapping (first 5): {[s['gene'] for s in skipped_no_uniprot[:5]]}")

    if not am_file.exists():
        if no_download:
            print(f"ERROR: --no-download set but {am_file} does not exist", file=sys.stderr)
            sys.exit(2)
        download_am(AM_URL, am_file)

    scores = stream_am_filter(am_file, index)
    print(f"matched scores: {len(scores):,} / {len(index):,}")

    atomic_write_json(out, scores)
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--am-file", type=Path, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    main(am_file=args.am_file, no_download=args.no_download, out=args.out)
