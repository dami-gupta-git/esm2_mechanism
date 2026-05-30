"""Shared data loaders. Callers pass explicit paths — no filenames are constructed here."""

from __future__ import annotations

import csv
import functools
import json
from pathlib import Path

print = functools.partial(print, flush=True)


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
