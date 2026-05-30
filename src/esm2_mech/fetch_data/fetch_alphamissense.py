"""
Fetch AlphaMissense scores for target variants.

Stream the AlphaMissense bulk file and extract scores for variants in
pathogenicity_valid_variants.json.

  Input : data/embeddings/esm2_t33_650M_UR50D/valid_variants.json,
          data/pathogenicity_valid_variants.json,
          data/cache/AlphaMissense_aa_substitutions.tsv.gz  (auto-downloaded)
  Output: data/alphamissense_scores_full.json

Usage:
    python -m esm2_mechanism.fetch_data.fetch_alphamissense
    python -m esm2_mechanism.fetch_data.fetch_alphamissense --no-download --am-file /path/to/file.tsv.gz
    python -m esm2_mechanism.fetch_data.fetch_alphamissense --out /path/to/output.json
"""

from __future__ import annotations

import argparse
import functools
import gzip
import json
import os
import sys
from pathlib import Path
import urllib.request
from typing import Optional

from esm2_mech.utils.paths import DATA_DIR, VALID_VARIANTS_JSON

print = functools.partial(print, flush=True)

AM_CACHE = DATA_DIR / "cache" / "AlphaMissense_aa_substitutions.tsv.gz"
AM_OUT = DATA_DIR / "alphamissense_scores_full.json"
AM_MERGED_VALID_VARIANTS = VALID_VARIANTS_JSON
AM_PATHOGENICITY_VARIANTS = DATA_DIR / "pathogenicity_valid_variants.json"

AM_URL = (
    "https://storage.googleapis.com/dm_alphamissense/"
    "AlphaMissense_aa_substitutions.tsv.gz"
)


def _build_am_gene_uniprot_map() -> dict[str, str]:
    with open(AM_MERGED_VALID_VARIANTS) as f:
        rows = json.load(f)

    counts: dict[str, dict[str, int]] = {}
    for r in rows:
        g, u = r["gene"], r["uniprot_id"]
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


def _build_am_lookup(
    variants: list[dict], g2u: dict[str, str]
) -> tuple[dict, list, list]:
    """
    Returns (index, skipped_no_uniprot, skipped_key_collision).

    skipped_no_uniprot: variants whose gene has no UniProt mapping.
    skipped_key_collision: variants dropped because another gene shares the same
        (uniprot, protein_variant) key — they will receive no AM score.
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
            f"(uniprot, protein_variant) key — these will have no AM score. "
            f"First 5: {[v['gene'] for v in skipped_key_collision[:5]]}"
        )
    return index, skipped_no_uniprot, skipped_key_collision


def _download_am(url: str, dest: Path) -> None:
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


def _stream_am_filter(
    am_gz: Path, index: dict[tuple[str, str], str]
) -> dict[str, float]:
    scores: dict[str, float] = {}
    needed = len(index)
    print(f"streaming {am_gz}, looking for {needed:,} (uniprot, variant) pairs")
    with gzip.open(am_gz, "rt") as f:
        header_skipped = False
        for i, line in enumerate(f):
            if i % 5_000_000 == 0 and i:
                print(
                    f"  read {i:,} rows, matched {len(scores):,}/{needed:,}",
                    file=sys.stderr,
                )
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


def main(
    am_file: Optional[Path] = None,
    no_download: bool = False,
    out: Optional[Path] = None,
) -> None:
    am_file = am_file or AM_CACHE
    out = out or AM_OUT

    required = [AM_MERGED_VALID_VARIANTS, AM_PATHOGENICITY_VARIANTS]
    missing_inputs = [p for p in required if not p.exists()]
    if missing_inputs:
        raise FileNotFoundError(
            "Required input(s) not found:\n" + "\n".join(f"  {p}" for p in missing_inputs)
        )

    am_file.parent.mkdir(parents=True, exist_ok=True)

    g2u = _build_am_gene_uniprot_map()
    print(f"gene -> uniprot: {len(g2u):,} entries")
    with open(AM_PATHOGENICITY_VARIANTS) as f:
        variants = json.load(f)
    print(f"target variants: {len(variants):,}")
    index, skipped_no_uniprot, skipped_key_collision = _build_am_lookup(variants, g2u)
    print(f"variants with UniProt mapping: {len(index):,}")
    if skipped_no_uniprot:
        print(
            f"variants missing UniProt mapping (first 5): {[s['gene'] for s in skipped_no_uniprot[:5]]}"
        )
    if skipped_key_collision:
        print(f"variants dropped due to key collision: {len(skipped_key_collision)}")

    if not am_file.exists():
        if no_download:
            print(
                f"ERROR: --no-download set but {am_file} does not exist",
                file=sys.stderr,
            )
            sys.exit(2)
        _download_am(AM_URL, am_file)

    scores = _stream_am_filter(am_file, index)
    print(f"matched scores: {len(scores):,} / {len(index):,}")

    with open(out, "w") as f:
        json.dump(scores, f, indent=2, sort_keys=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch AlphaMissense scores for target variants."
    )
    parser.add_argument(
        "--am-file",
        type=Path,
        default=None,
        help="Local path for the AM bulk file (default: data/cache/AlphaMissense_aa_substitutions.tsv.gz).",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download; require --am-file to exist.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: data/alphamissense_scores_full.json).",
    )
    args = parser.parse_args()
    main(am_file=args.am_file, no_download=args.no_download, out=args.out)
