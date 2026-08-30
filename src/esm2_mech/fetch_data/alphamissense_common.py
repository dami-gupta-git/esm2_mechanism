"""Shared helpers for AlphaMissense score fetching."""

from __future__ import annotations

import functools
import gzip
import math
import os
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

AM_URL = (
    "https://storage.googleapis.com/dm_alphamissense/"
    "AlphaMissense_aa_substitutions.tsv.gz"
)


def build_gene_uniprot_map(variants: list[dict]) -> dict[str, str]:
    """Build gene -> most-frequent UniProt ID mapping."""
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


def build_lookup(variants: list[dict], g2u: dict[str, str]) -> dict:
    """Build (uniprot, protein_variant) -> vkey index for AM streaming.

    Every dropped variant is reported here with its total, so each caller does
    not have to remember to report the skip buckets itself.
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
    print(f"variants with UniProt mapping: {len(index):,} / {len(variants):,}")
    if skipped_no_uniprot:
        print(
            f"  {len(skipped_no_uniprot):,} variants dropped with no UniProt mapping. "
            f"First 5 genes: {[v['gene'] for v in skipped_no_uniprot[:5]]}"
        )
    if skipped_key_collision:
        print(
            f"  {len(skipped_key_collision):,} variants dropped on a duplicate "
            f"(uniprot, protein_variant) key. "
            f"First 5 genes: {[v['gene'] for v in skipped_key_collision[:5]]}"
        )
    return index


def download_am(url: str, dest: Path) -> None:
    """Download AlphaMissense bulk file."""
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
    """Stream AM bulk file and return scores for indexed variants."""
    scores: dict[str, float] = {}
    needed = len(index)
    skipped_unparseable = 0
    skipped_non_finite = 0
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
                    score = float(score_s)
                except ValueError:
                    skipped_unparseable += 1
                    continue
                # float() accepts "nan"/"inf" — drop non-finite scores.
                if not math.isfinite(score):
                    skipped_non_finite += 1
                    continue
                scores[index[key]] = score
    if skipped_unparseable or skipped_non_finite:
        print(
            f"  skipped matched rows with bad scores: "
            f"{skipped_unparseable} unparseable, {skipped_non_finite} non-finite"
        )
    return scores
