"""Shared utilities for AlphaMissense score fetching."""

from __future__ import annotations

import functools
import gzip
import os
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

AM_URL = (
    "https://storage.googleapis.com/dm_alphamissense/"
    "AlphaMissense_aa_substitutions.tsv.gz"
)


def build_gene_uniprot_map(variants: list[dict]) -> dict[str, str]:
    """Build gene -> most-frequent UniProt ID mapping from a variant list."""
    counts: dict[str, dict[str, int]] = {}
    for r in variants:
        g, u = r.get("gene"), r.get("uniprot_id")
        if not g or not u:
            continue
        counts.setdefault(g, {}).setdefault(u, 0)
        counts[g][u] += 1

    g2u: dict[str, str] = {}
    for g, uid_counts in counts.items():
        best = max(uid_counts, key=lambda u: uid_counts[u])
        if len(uid_counts) > 1:
            print(
                f"WARN: gene {g} has multiple UniProt IDs {uid_counts} — using most frequent: {best}",
                file=sys.stderr,
            )
        g2u[g] = best
    return g2u


def build_lookup(
    variants: list[dict], g2u: dict[str, str]
) -> tuple[dict, list, list]:
    """Build (uniprot, protein_variant) -> vkey index for AM streaming.

    Returns (index, skipped_no_uniprot, skipped_key_collision).
    """
    index: dict[tuple[str, str], str] = {}
    skipped_no_uniprot = []
    skipped_key_collision = []
    for v in variants:
        uniprot = g2u.get(v["gene"])
        if not uniprot:
            skipped_no_uniprot.append(v)
            continue
        pv = f"{v['aa_wt']}{v['aa_pos']}{v['aa_mut']}"
        vkey = f"{v['gene']}_{v['aa_pos']}_{v['aa_wt']}_{v['aa_mut']}"
        key = (uniprot, pv)
        if key in index and index[key] != vkey:
            skipped_key_collision.append(v)
            continue
        index[key] = vkey
    if skipped_key_collision:
        print(
            f"  WARNING: {len(skipped_key_collision)} variants dropped due to duplicate "
            f"(uniprot, protein_variant) key. First 5: {[v['gene'] for v in skipped_key_collision[:5]]}"
        )
    return index, skipped_no_uniprot, skipped_key_collision


def download_am(url: str, dest: Path) -> None:
    """Download AlphaMissense bulk file atomically."""
    import urllib.request
    if dest.exists():
        print(f"already exists: {dest} ({dest.stat().st_size:,} bytes)")
        return
    print(f"downloading {url}\n  -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as out:
        total = 0
        chunk = 1 << 20
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            out.write(buf)
            total += len(buf)
            if total % (50 << 20) < chunk:
                print(f"  {total / 1e9:.2f} GB", file=sys.stderr)
    os.replace(tmp, dest)
    print(f"done: {dest.stat().st_size:,} bytes")


def stream_am_filter(am_gz: Path, index: dict[tuple[str, str], str]) -> dict[str, float]:
    """Stream AlphaMissense bulk file and return scores for indexed variants."""
    scores: dict[str, float] = {}
    needed = len(index)
    print(f"streaming {am_gz}, looking for {needed:,} (uniprot, variant) pairs")
    with gzip.open(am_gz, "rt") as f:
        header_skipped = False
        for i, line in enumerate(f):
            if i % 5_000_000 == 0 and i:
                print(f"  read {i:,} rows, matched {len(scores):,}/{needed:,}", file=sys.stderr)
            if not line or line.startswith("#"):
                continue
            if not header_skipped:
                header_skipped = True
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            uniprot, pv, score_s = parts[0], parts[1], parts[2]
            key = (uniprot, pv)
            if key in index:
                try:
                    scores[index[key]] = float(score_s)
                except ValueError:
                    continue
    return scores
