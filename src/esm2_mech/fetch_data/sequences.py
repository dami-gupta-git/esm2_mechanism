"""UniProt sequence and Pfam family fetching with resume-safe caching."""

from __future__ import annotations

import functools
import json
import os
import time
import urllib.request

from esm2_mech.utils.constants import UNIPROT_REST
from esm2_mech.utils.io import atomic_write_json
from esm2_mech.utils.paths import PFAM_FILE

print = functools.partial(print, flush=True)


class TransientFetchError(Exception):
    """Raised when all retries fail due to a transient network or server error (not a
    definitive 404). Callers must not cache the result — the next run should retry."""


def fetch_uniprot_sequence(
    uniprot_id: str, retries: int = 3, delay: float = 1.0
) -> str | None:
    """Fetch canonical protein sequence from UniProt.

    Returns the sequence string on success, or None when the server definitively
    reports that the accession does not exist (HTTP 404).

    Raises TransientFetchError when all retries are exhausted due to a transient
    network or server error. Callers must not cache this outcome.
    """
    import urllib.error

    url = f"{UNIPROT_REST}/{uniprot_id}.fasta"
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                fasta = resp.read().decode()
            lines = fasta.strip().split("\n")
            seq = "".join(line for line in lines if not line.startswith(">"))
            return seq.upper() if seq else None
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            last_exc = exc
        except Exception as exc:
            last_exc = exc
        if attempt < retries - 1:
            time.sleep(delay)
    print(
        f"  WARNING: transient fetch failure for {uniprot_id}: {last_exc} — will retry next run"
    )
    raise TransientFetchError(uniprot_id) from last_exc



def fetch_pfam_families(variants: list[dict]) -> dict[str, str | None]:
    """Fetch primary Pfam family for each unique gene via UniProt.

    Returns dict: gene -> pfam_id (or None when the protein has no Pfam annotation).
    Genes that fail with a transient network error are omitted from the returned dict
    and not written to cache, so the next run retries them.
    """
    import urllib.error

    cache_path = PFAM_FILE
    if os.path.exists(cache_path):
        try:
            with open(cache_path, newline="") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"WARNING: corrupt Pfam cache at {cache_path} — re-fetching")
            os.remove(cache_path)

    pfam_map: dict[str, str | None] = {}
    unique_pairs = {
        (v["gene"], v["uniprot_id"]) for v in variants if v.get("uniprot_id")
    }
    print(f"Fetching Pfam families for {len(unique_pairs)} genes...")

    transient_failures = 0
    for gene, uniprot_id in sorted(unique_pairs):
        url = f"{UNIPROT_REST}/{uniprot_id}.json"
        pfam_id = None
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode())
            for xref in data.get("uniProtKBCrossReferences", []):
                if xref.get("database") == "Pfam":
                    pfam_id = xref.get("id")
                    break
            pfam_map[gene] = pfam_id
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                pfam_map[gene] = None
            else:
                print(
                    f"  WARNING: transient HTTP {exc.code} for {uniprot_id} — skipping, will retry next run"
                )
                transient_failures += 1
        except Exception as exc:
            print(
                f"  WARNING: transient fetch failure for {uniprot_id}: {exc} — skipping, will retry next run"
            )
            transient_failures += 1
        time.sleep(0.3)

    if transient_failures:
        print(
            f"  WARNING: {transient_failures} genes had transient failures — skipping cache write, will retry next run"
        )
        n_annotated = sum(1 for v in pfam_map.values() if v is not None)
        print(f"  Pfam annotations so far: {n_annotated}/{len(pfam_map)} genes (partial)")
        return pfam_map

    atomic_write_json(cache_path, pfam_map)
    n_annotated = sum(1 for v in pfam_map.values() if v is not None)
    print(f"  Pfam annotations: {n_annotated}/{len(pfam_map)} genes")
    return pfam_map
