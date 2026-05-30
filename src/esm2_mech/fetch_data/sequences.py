"""UniProt sequence and Pfam family fetching with resume-safe caching."""

from __future__ import annotations

import functools
import json
import os
import time
import urllib.request

from esm2_mech.utils.constants import UNIPROT_REST
from esm2_mech.utils.io import atomic_write_json

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


def build_sequence_cache(variants: list[dict], cache_dir: str) -> dict[str, str]:
    """Fetch and cache UniProt sequences for all unique UniProt IDs in variants.

    Checkpoints to disk after every successful fetch so an interrupted run
    resumes rather than re-fetching from scratch. Only IDs not already in the
    cache are fetched.

    Returns dict: uniprot_id -> canonical sequence.
    """
    cache_path = os.path.join(cache_dir, "sequences.json")
    os.makedirs(cache_dir, exist_ok=True)

    sequences: dict[str, str] = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, newline="") as f:
                sequences = json.load(f)
        except json.JSONDecodeError:
            print(f"WARNING: corrupt sequence cache at {cache_path} — re-fetching all")
            os.remove(cache_path)

    unique_uniprots = {v["uniprot_id"] for v in variants if v.get("uniprot_id")}
    needed = sorted(unique_uniprots - sequences.keys())
    if not needed:
        return sequences

    print(f"Fetching {len(needed)} UniProt sequences (resuming from {len(sequences)} cached)...")
    transient_failures = 0
    for idx, uid in enumerate(needed):
        if idx % 50 == 0:
            print(f"  {idx}/{len(needed)}")
        try:
            seq = fetch_uniprot_sequence(uid)
        except TransientFetchError:
            transient_failures += 1
        else:
            if seq:
                sequences[uid] = seq
        if (idx + 1) % 50 == 0 or idx == len(needed) - 1:
            atomic_write_json(cache_path, sequences)
        time.sleep(0.3)

    if transient_failures:
        print(f"  WARNING: {transient_failures} UIDs had transient failures — will retry next run")
    print(f"  Sequences cached: {len(sequences)}/{len(unique_uniprots)}")
    return sequences


def fetch_pfam_families(variants: list[dict], cache_dir: str) -> dict[str, str | None]:
    """Fetch primary Pfam family for each unique gene via UniProt.

    Returns dict: gene -> pfam_id (or None when the protein has no Pfam annotation).
    Genes that fail with a transient network error are omitted from the returned dict
    and not written to cache, so the next run retries them.
    """
    import urllib.error

    cache_path = os.path.join(cache_dir, "pfam_families.json")
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
