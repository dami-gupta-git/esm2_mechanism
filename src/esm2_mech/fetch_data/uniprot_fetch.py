"""UniProt sequence and Pfam family fetching with resume-safe caching.

Owns the shared urllib fetch helpers used across the fetch_data package:
`TransientFetchError`, `fetch_with_retries`, and `parse_pfam_id`.
"""

from __future__ import annotations

import functools
import json
import time
import urllib.error
import urllib.request

from esm2_mech.utils.constants import UNIPROT_REST
from esm2_mech.utils.io import atomic_write_json, load_json_or_discard
from esm2_mech.utils.paths import PFAM_JSON

print = functools.partial(print, flush=True)


class TransientFetchError(Exception):
    """Raised when all retries fail due to a transient network or server error (not a
    definitive 404). Callers must not cache the result — the next run should retry."""


def fetch_with_retries(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    retries: int = 3,
    delay: float = 1.0,
    label: str | None = None,
) -> str | None:
    """GET `url` with retry + linear backoff, returning the decoded body.

    Returns the response text on success, or None when the server definitively
    reports the resource does not exist (HTTP 404 — a real result, safe to cache).
    Raises TransientFetchError when all retries are exhausted on a transient
    network/server error; callers must not cache that outcome.

    `delay` is scaled by (attempt + 1) so successive waits grow linearly. `label`
    is used only in the warning message (defaults to the URL).
    """
    request = urllib.request.Request(url, headers=headers or {})
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                return resp.read().decode()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            last_exc = exc
        except Exception as exc:
            last_exc = exc
        if attempt < retries - 1:
            time.sleep(delay * (attempt + 1))
    print(
        f"  WARNING: transient fetch failure for {label or url}: {last_exc} — will retry next run"
    )
    raise TransientFetchError(label or url) from last_exc


def parse_pfam_id(entry: dict) -> str | None:
    """Return the first Pfam cross-reference ID in a UniProt JSON entry, or None."""
    for xref in entry.get("uniProtKBCrossReferences", []):
        if xref.get("database") == "Pfam":
            return xref.get("id")
    return None


def fetch_uniprot_sequence(
    uniprot_id: str, retries: int = 3, delay: float = 1.0
) -> str | None:
    """Fetch canonical protein sequence from UniProt.

    Returns the sequence string on success, or None when the server definitively
    reports that the accession does not exist (HTTP 404).

    Raises TransientFetchError when all retries are exhausted due to a transient
    network or server error. Callers must not cache this outcome.
    """
    fasta = fetch_with_retries(
        f"{UNIPROT_REST}/{uniprot_id}.fasta",
        retries=retries,
        delay=delay,
        label=uniprot_id,
    )
    if fasta is None:
        return None
    lines = fasta.strip().split("\n")
    seq = "".join(line for line in lines if not line.startswith(">"))
    return seq.upper() if seq else None



def fetch_pfam_families(variants: list[dict]) -> dict[str, str | None]:
    """Fetch primary Pfam family for each unique gene via UniProt.

    Returns dict: gene -> pfam_id (or None when the protein has no Pfam annotation).
    Genes that fail with a transient network error are omitted from the returned dict
    and not written to cache, so the next run retries them.
    """
    cache_path = PFAM_JSON
    cached = load_json_or_discard(cache_path)
    if cached is not None:
        return cached

    pfam_map: dict[str, str | None] = {}
    unique_pairs = {
        (v["gene"], v["uniprot_id"]) for v in variants if v.get("uniprot_id")
    }
    print(f"Fetching Pfam families for {len(unique_pairs)} genes...")

    transient_failures = 0
    for gene, uniprot_id in sorted(unique_pairs):
        url = f"{UNIPROT_REST}/{uniprot_id}.json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode())
            pfam_map[gene] = parse_pfam_id(data)
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
