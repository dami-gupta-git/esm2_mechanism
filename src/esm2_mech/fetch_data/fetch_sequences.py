"""
Pre-fetch UniProt sequences for all variants in variants.json.

Writes results to data/cache/sequences.json (same path read by embed_variants.py).
Resume-safe: accessions already in the cache are skipped.
Checkpoints to disk every 50 fetches. Transient failures are logged and left
for the next run to retry — successfully fetched sequences are always written.

Usage:
    python -m esm2_mechanism.fetch_data.prefetch_sequences
    python -m esm2_mechanism.fetch_data.prefetch_sequences --data_dir data
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time
from pathlib import Path

from esm2_mech.utils.paths import DATA_DIR, VARIANTS_JSON
from esm2_mech.fetch_data.sequences import fetch_uniprot_sequence, TransientFetchError

print = functools.partial(print, flush=True)


def prefetch_sequences(data_dir: Path) -> None:
    variants_path = VARIANTS_JSON
    if not variants_path.exists():
        raise FileNotFoundError(
            f"Required input not found: {variants_path}\n"
            "Run fetch_variants --step merge first."
        )

    with open(variants_path) as f:
        variants = json.load(f)
    print(f"Loaded {len(variants)} variants from {variants_path}")

    all_uids = sorted({v["uniprot_id"] for v in variants if v.get("uniprot_id")})
    print(f"Unique UniProt IDs: {len(all_uids)}")

    cache_dir = data_dir / "cache"
    cache_path = cache_dir / "sequences.json"

    seq_cache: dict[str, str] = {}
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                seq_cache = json.load(f)
            print(f"Loaded {len(seq_cache)} sequences from existing cache")
        except json.JSONDecodeError:
            print(f"WARNING: corrupt cache at {cache_path} — starting empty")
            cache_path.unlink()

    needed = [uid for uid in all_uids if uid not in seq_cache]
    print(f"Need to fetch: {len(needed)}")

    if not needed:
        print("Cache already complete — nothing to do.")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".json.tmp")
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
        if (i + 1) % 50 == 0 or i == len(needed) - 1:
            with open(tmp_path, "w") as f:
                json.dump(seq_cache, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, cache_path)
        time.sleep(0.3)

    if transient_failures:
        print(
            f"WARNING: {len(transient_failures)} transient failures — will retry on next run: "
            f"{transient_failures[:50]}{'...' if len(transient_failures) > 50 else ''}"
        )

    n_found = sum(1 for seq in seq_cache.values() if seq)
    print(f"Done. {n_found}/{len(all_uids)} sequences cached to {cache_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DATA_DIR,
        help="Root data directory (default: data/)",
    )
    args = parser.parse_args()
    prefetch_sequences(args.data_dir)


if __name__ == "__main__":
    main()
