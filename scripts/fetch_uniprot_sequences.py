"""
Fetch missing UniProt sequences for the merged dataset.

Reads data/cache/missing_uniprots.json (list of UniProt accessions), fetches
each via UniProt REST (in batches of 100, with simple retry), and writes
data/cache/uniprot_sequences_extended.json (dict[uniprot_id -> sequence]).

Combined with the existing data/sequences.json (948 Gerasimavicius entries),
this gives us complete coverage for the 1,983 unique UniProt IDs in
merged_valid_variants.json.

Usage:
    python3 scripts/fetch_uniprot_sequences.py
"""

import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from io import StringIO
import functools
print = functools.partial(print, flush=True)

ROOT = Path(__file__).parent.parent
CACHE = ROOT / "data" / "cache"
MISSING_LIST = CACHE / "missing_uniprots.json"
OUT_PATH = CACHE / "uniprot_sequences_extended.json"

BATCH_SIZE = 100
RETRIES = 3
USER_AGENT = "dami-mechanism-prediction/0.1 (academic research)"


def fetch_batch(accs):
    """Fetch a batch of UniProt entries via REST as FASTA."""
    # UniProt REST: stream FASTA for a set of accessions via "query=accession:X1 OR accession:X2 OR ..."
    # Simpler: use the /uniprotkb/{acc}.fasta endpoint per accession — but that is many requests.
    # Better: use the search endpoint with `accession:X1 OR accession:X2 OR ...` as the query.
    query = " OR ".join(f"accession:{a}" for a in accs)
    params = {
        "query": query,
        "format": "fasta",
        "size": len(accs),
    }
    url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/plain"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode()


def parse_fasta(text):
    """Yield (accession, sequence) tuples from a FASTA stream."""
    out = {}
    acc = None
    buf = []
    for line in text.splitlines():
        if line.startswith(">"):
            if acc and buf:
                out[acc] = "".join(buf)
            # Header format: >sp|P12345|GENE_HUMAN ...  or  >tr|...
            parts = line[1:].split("|")
            acc = parts[1] if len(parts) > 1 else parts[0]
            buf = []
        elif line.strip():
            buf.append(line.strip())
    if acc and buf:
        out[acc] = "".join(buf)
    return out


def main():
    with open(MISSING_LIST) as f:
        accs = json.load(f)
    print(f"Need to fetch {len(accs)} UniProt sequences")

    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            results = json.load(f)
        print(f"Resuming with {len(results)} already fetched")
    else:
        results = {}

    todo = [a for a in accs if a not in results]
    print(f"Remaining: {len(todo)}")

    n_batches = (len(todo) + BATCH_SIZE - 1) // BATCH_SIZE
    t0 = time.time()
    for bi in range(n_batches):
        batch = todo[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]
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
            print(f"  GAVE UP on batch {bi+1}")
        elapsed = time.time() - t0
        print(f"  batch {bi+1}/{n_batches}: {len(results)} fetched ({elapsed:.0f}s)")
        OUT_PATH.write_text(json.dumps(results))
        time.sleep(0.5)  # be polite

    # Coverage report
    found = sum(1 for a in accs if a in results)
    missing = [a for a in accs if a not in results]
    print(f"\nFetched {found}/{len(accs)} sequences")
    if missing:
        print(f"Missing ({len(missing)}): {missing[:20]}")
    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
