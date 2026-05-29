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
import time
import urllib.request
import urllib.parse
import functools
from io import StringIO


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-scratch", action="store_true",
                        help="Ignore existing cache and re-fetch everything")
    args = parser.parse_args()

    with open(MERGED_VARIANTS) as f:
        variants = json.load(f)

    all_uniprots = {v["uniprot_id"] for v in variants if v.get("uniprot_id")}
    print(f"Unique UniProt IDs in merged_variants.json: {len(all_uniprots)}")

    already_have: dict[str, str] = {}

    if not args.from_scratch:
        if SEQUENCES_JSON.exists():
            with open(SEQUENCES_JSON) as f:
                base = json.load(f)
            already_have.update(base)
            print(f"Loaded {len(base)} sequences from sequences.json")

        if OUT_PATH.exists():
            with open(OUT_PATH) as f:
                extended = json.load(f)
            already_have.update(extended)
            print(f"Loaded {len(extended)} sequences from uniprot_sequences_extended.json")

    results: dict[str, str] = {} if args.from_scratch else {
        k: v for k, v in already_have.items() if k in all_uniprots
    }

    todo = sorted(all_uniprots - set(results))
    print(f"Already have: {len(results)}, need to fetch: {len(todo)}")

    if not todo:
        print("Nothing to fetch.")
        return

    CACHE.mkdir(parents=True, exist_ok=True)
    n_batches = (len(todo) + BATCH_SIZE - 1) // BATCH_SIZE
    t0 = time.time()

    for bi in range(n_batches):
        batch = todo[bi * BATCH_SIZE: (bi + 1) * BATCH_SIZE]
        for attempt in range(RETRIES):
            try:
                text = fetch_batch(batch)
                parsed = parse_fasta(text)
                results.update(parsed)
                break
            except Exception as e:
                print(f"  batch {bi+1}/{n_batches} attempt {attempt+1} error: {e}")
                time.sleep(2 * (attempt + 1))
        else:
            print(f"  GAVE UP on batch {bi+1}/{n_batches}")

        elapsed = time.time() - t0
        print(f"  batch {bi+1}/{n_batches}: {len(results)} fetched ({elapsed:.0f}s)")
        OUT_PATH.write_text(json.dumps(results))
        time.sleep(0.5)

    found = sum(1 for a in all_uniprots if a in results)
    missing = [a for a in all_uniprots if a not in results]
    print(f"\nFetched {found}/{len(all_uniprots)} sequences")
    if missing:
        print(f"Missing ({len(missing)}): {missing[:20]}")
    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
