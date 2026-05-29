"""
Shared sequence utilities: UniProt fetching, missense application, windowing, caching.

Used by esm2_mechanism.py, esm3_mechanism.py, score_esm1v.py and any other script
that needs to prepare WT/mutant sequence pairs for embedding models.
"""

from __future__ import annotations

import functools
import json
import os
import time
import urllib.request

print = functools.partial(print, flush=True)

UNIPROT_REST = "https://rest.uniprot.org/uniprotkb"
MAX_SEQ_LEN = 1022   # ESM-2 token limit
WINDOW_HALF = 500    # half-window size for sequences > MAX_SEQ_LEN


# ---------------------------------------------------------------------------
# UniProt sequence fetching
# ---------------------------------------------------------------------------

def fetch_uniprot_sequence(uniprot_id: str, retries: int = 3, delay: float = 1.0) -> str | None:
    """Fetch canonical protein sequence from UniProt. Returns None on failure."""
    url = f"{UNIPROT_REST}/{uniprot_id}.fasta"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                fasta = resp.read().decode()
            lines = fasta.strip().split("\n")
            seq = "".join(l for l in lines if not l.startswith(">"))
            return seq.upper() if seq else None
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)
    return None


def build_sequence_cache(variants: list[dict], cache_dir: str) -> dict[str, str]:
    """Fetch and cache UniProt sequences for all unique UniProt IDs in variants.

    Returns dict: uniprot_id -> canonical sequence.
    """
    cache_path = os.path.join(cache_dir, "sequences.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    os.makedirs(cache_dir, exist_ok=True)
    sequences: dict[str, str] = {}
    unique_uniprots = {v["uniprot_id"] for v in variants if v["uniprot_id"]}
    print(f"Fetching sequences for {len(unique_uniprots)} UniProt IDs...")

    for i, uid in enumerate(sorted(unique_uniprots)):
        if i % 50 == 0:
            print(f"  {i}/{len(unique_uniprots)}")
        seq = fetch_uniprot_sequence(uid)
        if seq:
            sequences[uid] = seq
        time.sleep(0.3)

    with open(cache_path, "w") as f:
        json.dump(sequences, f)

    print(f"  Fetched {len(sequences)}/{len(unique_uniprots)} sequences")
    return sequences


# ---------------------------------------------------------------------------
# Pfam family fetching
# ---------------------------------------------------------------------------

def fetch_pfam_families(variants: list[dict], cache_dir: str) -> dict[str, str | None]:
    """Fetch primary Pfam family for each unique gene via UniProt.

    Returns dict: gene -> pfam_id (or None if not found).
    """
    cache_path = os.path.join(cache_dir, "pfam_families.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    pfam_map: dict[str, str | None] = {}
    unique_pairs = {(v["gene"], v["uniprot_id"]) for v in variants if v["uniprot_id"]}
    print(f"Fetching Pfam families for {len(unique_pairs)} genes...")

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
        except Exception:
            pass
        pfam_map[gene] = pfam_id
        time.sleep(0.3)

    with open(cache_path, "w") as f:
        json.dump(pfam_map, f)

    n_annotated = sum(1 for v in pfam_map.values() if v is not None)
    print(f"  Pfam annotations: {n_annotated}/{len(pfam_map)} genes")
    return pfam_map


# ---------------------------------------------------------------------------
# Sequence manipulation
# ---------------------------------------------------------------------------

def apply_missense(sequence: str, aa_pos: int, aa_wt: str, aa_mut: str) -> str | None:
    """Apply a missense mutation (1-indexed aa_pos). Returns None on mismatch or OOB."""
    idx = aa_pos - 1
    if idx < 0 or idx >= len(sequence):
        return None
    if sequence[idx] != aa_wt:
        return None
    seq_list = list(sequence)
    seq_list[idx] = aa_mut
    return "".join(seq_list)


def window_sequence(sequence: str, aa_pos: int,
                    window_half: int = WINDOW_HALF,
                    max_len: int = MAX_SEQ_LEN) -> tuple[str, int]:
    """Extract a window of at most max_len residues centred on aa_pos.

    Returns (windowed_seq, new_aa_pos) where new_aa_pos is 1-indexed in the
    windowed sequence. Sequences already within max_len are returned unchanged.
    """
    if len(sequence) <= max_len:
        return sequence, aa_pos

    idx = aa_pos - 1  # 0-indexed
    start = max(0, idx - window_half)
    end = min(len(sequence), idx + window_half)
    if end - start > max_len:
        half = max_len // 2
        start = max(0, idx - half)
        end = min(len(sequence), start + max_len)

    windowed = sequence[start:end]
    new_pos = idx - start + 1  # back to 1-indexed
    return windowed, new_pos
