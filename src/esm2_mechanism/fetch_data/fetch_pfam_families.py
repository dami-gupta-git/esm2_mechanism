"""
Fetch primary Pfam family for each gene in merged_variants.json via UniProt REST API.

Reads data/merged_variants.json, collects unique (gene, uniprot_id) pairs, and
queries the UniProt API to get the primary Pfam domain annotation for each gene.

Output: data/pfam_families.json  (dict[gene -> pfam_id | null])

This file is required by build_proteome_features.py and several downstream scripts.

Usage:
    python -m esm2_mechanism.fetch_data.fetch_pfam_families
    python -m esm2_mechanism.fetch_data.fetch_pfam_families --from-scratch
"""

from __future__ import annotations

import argparse
import functools
import json
import time
import urllib.request
from typing import Optional

from esm2_mechanism.utils_paths import DATA_DIR

print = functools.partial(print, flush=True)

MERGED_VARIANTS = DATA_DIR / "merged_variants.json"
OUT_PATH = DATA_DIR / "pfam_families.json"

UNIPROT_REST = "https://rest.uniprot.org/uniprotkb"
DELAY = 0.3
RETRIES = 3


def fetch_pfam_for_uniprot(uniprot_id: str) -> Optional[str]:
    """Return the first Pfam cross-reference ID for a UniProt entry, or None."""
    url = f"{UNIPROT_REST}/{uniprot_id}.json"
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode())
            for xref in data.get("uniProtKBCrossReferences", []):
                if xref.get("database") == "Pfam":
                    return xref.get("id")
            return None
        except Exception as e:
            if attempt < RETRIES - 1:
                time.sleep(DELAY * (attempt + 1))
            else:
                print(f"  WARNING: failed to fetch {uniprot_id}: {e}")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Ignore existing output and re-fetch everything",
    )
    args = parser.parse_args()

    if not MERGED_VARIANTS.exists():
        raise FileNotFoundError(f"Required input not found: {MERGED_VARIANTS}")

    with open(MERGED_VARIANTS) as f:
        variants = json.load(f)

    unique_pairs: set[tuple[str, str]] = {
        (v["gene"], v["uniprot_id"])
        for v in variants
        if v.get("gene") and v.get("uniprot_id")
    }
    print(f"Unique (gene, uniprot_id) pairs: {len(unique_pairs)}")

    existing: dict[str, Optional[str]] = {}
    if not args.from_scratch and OUT_PATH.exists():
        with open(OUT_PATH) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing entries from {OUT_PATH.name}")

    to_fetch = sorted(
        (gene, uid) for gene, uid in unique_pairs if gene not in existing
    )
    print(f"Already have: {len(existing)}, need to fetch: {len(to_fetch)}")

    results: dict[str, Optional[str]] = dict(existing)

    for i, (gene, uniprot_id) in enumerate(to_fetch):
        pfam_id = fetch_pfam_for_uniprot(uniprot_id)
        results[gene] = pfam_id
        time.sleep(DELAY)
        if (i + 1) % 100 == 0 or (i + 1) == len(to_fetch):
            print(f"  {i+1}/{len(to_fetch)} fetched")
            OUT_PATH.write_text(json.dumps(results))

    if not to_fetch:
        print("Nothing to fetch.")
        OUT_PATH.write_text(json.dumps(results))

    n_annotated = sum(1 for v in results.values() if v is not None)
    print(f"\nPfam annotations: {n_annotated}/{len(results)} genes assigned")
    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
