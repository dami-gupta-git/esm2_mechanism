"""Shared data loaders. Callers pass explicit paths — no filenames are constructed here."""

from __future__ import annotations

import csv
import functools
import hashlib
import json
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)


def variants_fingerprint(variants: list[dict]) -> str:
    """Order-sensitive content hash of a variant list's identities.

    Pins which variants (and in which row order) a downstream artifact was built
    from, so a count-collision or reordering after a seed/cap change is detected
    rather than silently reusing a stale, misaligned artifact. The fields chosen
    uniquely identify a variant and its embedding inputs.
    """
    digest = hashlib.sha256()
    for v in variants:
        key = f"{v['gene']}|{v['uniprot_id']}|{v['aa_pos']}|{v['aa_wt']}|{v['aa_mut']}|{v['label']}"
        digest.update(key.encode())
        digest.update(b"\x00")
    return digest.hexdigest()


def load_variants(path: Path) -> list[dict]:
    """Load and filter the merged variant dataset from path.

    Filters to variants with uniprot_id, aa_wt, aa_mut, and aa_pos > 0.
    """
    with open(path) as f:
        variants = json.load(f)
    variants = [
        v
        for v in variants
        if v.get("uniprot_id")
        and v.get("aa_wt")
        and v.get("aa_mut")
        and v.get("aa_pos", 0) > 0
    ]
    print(f"After filtering: {len(variants)} variants")
    return variants


def build_gene_to_row(gene_list_path: Path) -> dict[str, int]:
    """Return {gene: row_index} for indexing into aligned feature matrices.

    Row order matches the first appearance of each gene in gene_list_path.
    Uses DictReader so column order changes in the TSV do not silently break the index.
    """
    seen: dict[str, int] = {}
    idx = 0
    with open(gene_list_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene = row["gene"].strip()
            if gene and gene not in seen:
                seen[gene] = idx
                idx += 1
    return seen
