"""
Fetch UniProt sequences for all variants in merged_variants.json.

Reads data/merged_variants.json (Gerasimavicius + ClinVar/G2P), collects all
unique uniprot_ids, skips any already in data/sequences.json or
data/cache/uniprot_sequences_extended.json, and fetches the rest in batches
of 100 via the UniProt REST API.

Output: data/cache/uniprot_sequences_extended.json (dict[uniprot_id -> sequence]).
Combined with data/sequences.json this gives full coverage for all variants.

Usage:
    python3 scripts/fetch_data/fetch_uniprot_sequences.py
    python3 scripts/fetch_data/fetch_uniprot_sequences.py --from-scratch   # ignore existing cache
"""

import argparse
import json
import os
import time
import urllib.request
import urllib.parse
import functools
from io import StringIO
from pathlib import Path


from Bio import SeqIO

from esm2_mechanism.utils_paths import DATA_DIR

print = functools.partial(print, flush=True)

DATA = DATA_DIR
CACHE = DATA / "cache"
MERGED_VARIANTS = DATA / "merged_variants.json"
SEQUENCES_JSON = DATA / "sequences.json"
OUT_PATH = CACHE / "uniprot_sequences_extended.json"

BATCH_SIZE = 100
RETRIES = 3
USER_AGENT = "dami-mechanism-prediction/0.1 (academic research)"


def fetch_batch(accs: list[str]) -> str:
    query = " OR ".join(f"accession:{a}" for a in accs)
    params = {"query": query, "format": "fasta", "size": len(accs)}
    url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/plain"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode()


def parse_fasta(text: str) -> dict[str, str]:
    out = {}
    for record in SeqIO.parse(StringIO(text), "fasta"):
        parts = record.id.split("|")
        acc = parts[1].split("-")[0] if len(parts) > 1 else parts[0].split("-")[0]
        out[acc] = str(record.seq)
    return out


def select_extended(
    base: dict[str, str],
    extended_existing: dict[str, str],
    new_results: dict[str, str],
    needed: set[str],
) -> dict[str, str]:
    """Sequences to persist in the extended cache: those required by variants
    (``needed``) that are not already covered by ``base`` (sequences.json).

    Excludes base sequences (avoids duplicating sequences.json) and sequences
    for IDs no variant references (avoids unbounded bloat). New fetches take
    precedence over previously-cached extended entries.
    """
    merged = {**extended_existing, **new_results}
    return {k: v for k, v in merged.items() if k in needed and k not in base}


def atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-scratch", action="store_true",
                        help="Ignore existing cache and re-fetch everything")
    args = parser.parse_args()

    if not MERGED_VARIANTS.exists():
        raise FileNotFoundError(f"Required input not found: {MERGED_VARIANTS}")

    with open(MERGED_VARIANTS) as f:
        variants = json.load(f)

    all_uniprots = {v["uniprot_id"] for v in variants if v.get("uniprot_id")}
    print(f"Unique UniProt IDs in merged_variants.json: {len(all_uniprots)}")

    base: dict[str, str] = {}
    extended_existing: dict[str, str] = {}

    if not args.from_scratch:
        if SEQUENCES_JSON.exists():
            with open(SEQUENCES_JSON) as f:
                base = json.load(f)
            print(f"Loaded {len(base)} sequences from sequences.json")

        if OUT_PATH.exists():
            with open(OUT_PATH) as f:
                extended_existing = json.load(f)
            print(f"Loaded {len(extended_existing)} sequences from uniprot_sequences_extended.json")

    new_results: dict[str, str] = {}
    have_keys = set(base) | set(extended_existing)

    todo = sorted(all_uniprots - have_keys)
    print(f"Already have: {len(all_uniprots & have_keys)}, need to fetch: {len(todo)}")

    CACHE.mkdir(parents=True, exist_ok=True)

    if not todo:
        print("Nothing to fetch.")
        atomic_write_json(OUT_PATH, select_extended(base, extended_existing, new_results, all_uniprots))
        print(f"Wrote: {OUT_PATH}")
        return

    n_batches = (len(todo) + BATCH_SIZE - 1) // BATCH_SIZE
    t0 = time.time()

    for bi in range(n_batches):
        batch = todo[bi * BATCH_SIZE: (bi + 1) * BATCH_SIZE]
        for attempt in range(RETRIES):
            try:
                text = fetch_batch(batch)
                parsed = parse_fasta(text)
                new_results.update(parsed)
                break
            except Exception as e:
                print(f"  batch {bi+1}/{n_batches} attempt {attempt+1} error: {e}")
                time.sleep(2 * (attempt + 1))
        else:
            print(f"  GAVE UP on batch {bi+1}/{n_batches}")

        elapsed = time.time() - t0
        print(f"  batch {bi+1}/{n_batches}: {len(new_results)} fetched ({elapsed:.0f}s)")
        atomic_write_json(OUT_PATH, select_extended(base, extended_existing, new_results, all_uniprots))
        time.sleep(0.5)

    atomic_write_json(OUT_PATH, select_extended(base, extended_existing, new_results, all_uniprots))

    have_all = set(base) | set(extended_existing) | set(new_results)
    found = len(all_uniprots & have_all)
    missing = sorted(all_uniprots - have_all)
    print(f"\nFetched {found}/{len(all_uniprots)} sequences")
    if missing:
        print(f"Missing ({len(missing)}): {missing[:20]}")
    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
