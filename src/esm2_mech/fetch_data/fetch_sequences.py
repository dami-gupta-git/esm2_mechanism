"""
Pre-fetch UniProt sequences for all variants in variants.json.

Writes results to data/cache/sequences.json (same path read by embed_variants.py).
Resume-safe: accessions already in the cache, or already known to be absent
(HTTP 404), are skipped. Accessions that 404 are recorded in
data/cache/sequences_not_found.json so they are not re-fetched on every run.
Checkpoints to disk every 50 fetches. Transient failures are logged and left
for the next run to retry — successfully fetched sequences are always written.

Usage:
    python -m esm2_mech.fetch_data.fetch_sequences
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path

from esm2_mech.utils.paths import (
    CACHE_DIR,
    SEQUENCES_JSON,
    SEQUENCES_NOT_FOUND_JSON,
    VARIANTS_JSON,
)
from esm2_mech.utils.io import atomic_write_json, load_json_or_discard
from esm2_mech.fetch_data.uniprot_fetch import fetch_uniprot_sequence, TransientFetchError

print = functools.partial(print, flush=True)


def prefetch_sequences() -> None:
    if not VARIANTS_JSON.exists():
        raise FileNotFoundError(
            f"Required input not found: {VARIANTS_JSON}\n"
            "Run fetch_variants --step merge first."
        )

    with open(VARIANTS_JSON) as f:
        variants = json.load(f)
    print(f"Loaded {len(variants)} variants from {VARIANTS_JSON}")

    all_uids = sorted({v["uniprot_id"] for v in variants if v.get("uniprot_id")})
    print(f"Unique UniProt IDs: {len(all_uids)}")

    seq_cache: dict[str, str] = load_json_or_discard(SEQUENCES_JSON) or {}
    if seq_cache:
        print(f"Loaded {len(seq_cache)} sequences from existing cache")

    cached_not_found = load_json_or_discard(SEQUENCES_NOT_FOUND_JSON)
    not_found: set[str] = set(cached_not_found) if cached_not_found else set()
    if not_found:
        print(f"Loaded {len(not_found)} known-absent (404) accessions")

    needed = [uid for uid in all_uids if uid not in seq_cache and uid not in not_found]
    print(f"Need to fetch: {len(needed)}")

    if not needed:
        print("Cache already complete — nothing to do.")
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    transient_failures: list[str] = []
    for i, uid in enumerate(needed):
        if i % 50 == 0:
            print(f"  [{i}/{len(needed)}] fetching...")
        try:
            seq = fetch_uniprot_sequence(uid)
        except TransientFetchError:
            transient_failures.append(uid)
        else:
            if seq:
                seq_cache[uid] = seq
            else:
                # Definitive 404 (or empty FASTA body): no usable sequence exists
                # at this accession. Record it so the next run does not re-fetch.
                not_found.add(uid)
        if (i + 1) % 50 == 0 or i == len(needed) - 1:
            atomic_write_json(SEQUENCES_JSON, seq_cache)
            atomic_write_json(SEQUENCES_NOT_FOUND_JSON, sorted(not_found))
        time.sleep(0.3)

    if transient_failures:
        print(
            f"WARNING: {len(transient_failures)} transient failures — will retry on next run: "
            f"{transient_failures[:50]}{'...' if len(transient_failures) > 50 else ''}"
        )

    n_found = sum(1 for seq in seq_cache.values() if seq)
    print(
        f"Done. {n_found}/{len(all_uids)} sequences cached to {SEQUENCES_JSON} "
        f"({len(not_found)} accessions absent — recorded in {SEQUENCES_NOT_FOUND_JSON})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    prefetch_sequences()


if __name__ == "__main__":
    main()
