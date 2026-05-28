"""
Fetch AlphaMissense scores for the ClinVar variant set used in result_6.

Approach:
1. Build gene -> UniProt mapping from data/merged_valid_variants.json.
2. Stream AlphaMissense_aa_substitutions.tsv.gz (5.3 GB compressed).
3. Filter to (UniProt, protein_variant) pairs matching our 17,236 ClinVar variants.
4. Cache as data/alphamissense_scores_full.json keyed by GENE_POS_WT_MUT.

The bulk file URL is:
    https://storage.googleapis.com/dm_alphamissense/AlphaMissense_aa_substitutions.tsv.gz

Format (tab-separated, header rows start with '#'):
    uniprot_id   protein_variant   am_pathogenicity   am_class
where protein_variant is e.g. "M1A".
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path
from urllib.request import urlopen
import functools
print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

AM_URL = (
    "https://storage.googleapis.com/dm_alphamissense/"
    "AlphaMissense_aa_substitutions.tsv.gz"
)


def build_gene_uniprot_map() -> dict[str, str]:
    """Map gene symbol -> UniProt ID, using merged_valid_variants.json."""
    rows = json.load(open(DATA / "merged_valid_variants.json"))
    g2u: dict[str, str] = {}
    for r in rows:
        g, u = r["gene"], r["uniprot_id"]
        if g in g2u and g2u[g] != u:
            # Inconsistency — pick the first one we saw and warn.
            print(f"WARN: gene {g} has multiple UniProt IDs: {g2u[g]} vs {u}", file=sys.stderr)
            continue
        g2u[g] = u
    return g2u


def load_target_variants() -> list[dict]:
    return json.load(open(DATA / "pathogenicity_valid_variants.json"))


def build_lookup_set(variants: list[dict], g2u: dict[str, str]) -> tuple[dict, list[dict]]:
    """
    Build (uniprot, protein_variant) -> variant_key index.
    protein_variant format matches AM file: e.g. 'R330M' for aa_wt R pos 330 aa_mut M.
    """
    index: dict[tuple[str, str], str] = {}
    skipped = []
    for v in variants:
        uniprot = g2u.get(v["gene"])
        if not uniprot:
            skipped.append(v)
            continue
        pv = f"{v['aa_wt']}{v['aa_pos']}{v['aa_mut']}"
        vkey = f"{v['gene']}_{v['aa_pos']}_{v['aa_wt']}_{v['aa_mut']}"
        index[(uniprot, pv)] = vkey
    return index, skipped


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"already exists: {dest} ({dest.stat().st_size:,} bytes)")
        return dest
    print(f"downloading {url}\n  -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urlopen(url) as r, open(tmp, "wb") as out:
        total = 0
        chunk = 1 << 20  # 1 MB
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            out.write(buf)
            total += len(buf)
            if total % (50 << 20) < chunk:
                print(f"  {total / 1e9:.2f} GB", file=sys.stderr)
    tmp.rename(dest)
    print(f"done: {dest.stat().st_size:,} bytes")
    return dest


def stream_filter(am_gz: Path, index: dict[tuple[str, str], str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    needed = len(index)
    print(f"streaming {am_gz}, looking for {needed:,} (uniprot, variant) pairs")
    with gzip.open(am_gz, "rt") as f:
        for i, line in enumerate(f):
            if i % 5_000_000 == 0 and i:
                print(f"  read {i:,} rows, matched {len(scores):,}/{needed:,}", file=sys.stderr)
            if not line or line.startswith("#"):
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--am-file",
        type=Path,
        default=DATA / "cache" / "AlphaMissense_aa_substitutions.tsv.gz",
        help="local path for the AM bulk file",
    )
    ap.add_argument(
        "--no-download",
        action="store_true",
        help="do not download; require --am-file to exist",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DATA / "alphamissense_scores_full.json",
    )
    args = ap.parse_args()

    args.am_file.parent.mkdir(parents=True, exist_ok=True)

    g2u = build_gene_uniprot_map()
    print(f"gene -> uniprot: {len(g2u):,} entries")
    variants = load_target_variants()
    print(f"target variants: {len(variants):,}")
    index, skipped = build_lookup_set(variants, g2u)
    print(f"variants with UniProt mapping: {len(index):,}")
    if skipped:
        print(f"variants missing UniProt mapping (first 5): "
              f"{[s['gene'] for s in skipped[:5]]}")

    if not args.am_file.exists():
        if args.no_download:
            print(f"ERROR: --no-download set but {args.am_file} does not exist", file=sys.stderr)
            return 2
        download(AM_URL, args.am_file)

    scores = stream_filter(args.am_file, index)
    print(f"matched scores: {len(scores):,} / {len(index):,}")

    with open(args.out, "w") as f:
        json.dump(scores, f, indent=2, sort_keys=True)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
