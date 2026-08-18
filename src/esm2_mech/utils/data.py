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


def embedding_fingerprint(*arrays: np.ndarray) -> str:
    """Content hash of one or more embedding arrays.

    Catches cases where embeddings change (e.g. model weights updated)
    while the variant list and model name stay the same.
    """
    digest = hashlib.sha256()
    for arr in arrays:
        digest.update(arr.tobytes())
    return digest.hexdigest()


def pfam_fingerprint(pfam_map: dict, genes: list[str]) -> str:
    """Content hash of the Pfam family assignments for a set of genes.

    Changes to Pfam mappings alter family-split CV folds, invalidating any
    cached probe results that used the old splits.
    """
    digest = hashlib.sha256()
    for gene in sorted(set(genes)):
        family = pfam_map.get(gene, "")
        digest.update(f"{gene}|{family}".encode())
        digest.update(b"\x00")
    return digest.hexdigest()


def load_pfam_map(path: Path) -> dict[str, str | None]:
    """Load the gene -> Pfam family accession map from pfam_families.json.

    A gene with no Pfam hit is present with a null value, not absent — callers
    checking annotation status must test `is not None`, not truthiness or membership.
    """
    with open(path) as f:
        return json.load(f)


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
    duplicates: list[str] = []
    idx = 0
    with open(gene_list_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene = row["gene"].strip()
            if not gene:
                continue
            if gene in seen:
                # Keep-first by design (one row per gene expected). Surface any
                # duplicate so a gene list with unexpected repeats is visible
                # rather than silently collapsing later rows to the first index.
                duplicates.append(gene)
                continue
            seen[gene] = idx
            idx += 1
    if duplicates:
        print(
            f"WARNING: {gene_list_path} has {len(duplicates)} duplicate gene rows; "
            f"kept the first index per gene. Examples: {duplicates[:5]}",
            flush=True,
        )
    return seen


def build_source_mask(valid_variants: list[dict], source: str) -> np.ndarray:
    """Boolean mask (aligned to valid_variants / every row-aligned feature array)
    selecting rows whose `source` field equals `source`.

    Rows with no `source` are excluded and counted rather than silently assigned —
    absent provenance is not a default value.
    """
    flags = []
    n_missing = 0
    for variant in valid_variants:
        variant_source = variant.get("source")
        if variant_source is None:
            n_missing += 1
        flags.append(variant_source == source)
    if n_missing:
        print(f"WARNING: {n_missing} variants have no `source` field — excluded from subset")
    return np.array(flags, dtype=bool)


def observed_rows_mask(
    X: np.ndarray, col_idx: list[int] | None = None, label: str = ""
) -> np.ndarray:
    """Boolean mask selecting rows of `X` with no NaN in the given columns.

    For a probe restricted to a feature subset, only the columns that probe
    actually uses decide whether a row is usable — a NaN elsewhere in the matrix
    is irrelevant. `col_idx` names those columns; None means every column.

    This is the complete-case restriction required for models that cannot
    consume NaN (LogReg, MLP): the caller must subset X, y and genes by this
    mask AND recompute CV splits on the subset, never impute. Imputing a value
    (median or otherwise) computed over the whole dataset both leaks test-fold
    statistics into training and produces cells indistinguishable from real
    measurements. For a multi-feature matrix where complete-case would discard
    a large, non-random share of rows, prefer a NaN-native learner
    (probes.run_histgb_cv) over restricting.
    """
    cols = list(range(X.shape[1])) if col_idx is None else col_idx
    mask = ~np.isnan(X[:, cols]).any(axis=1)
    n_dropped = int((~mask).sum())
    if n_dropped:
        print(
            f"  {label + ': ' if label else ''}complete-case restriction drops "
            f"{n_dropped}/{len(mask)} rows ({100 * n_dropped / len(mask):.1f}%) "
            f"with a NaN in the {len(cols)} column(s) used"
        )
    return mask


def subset_data(data: dict, mask: np.ndarray) -> dict:
    """Return a copy of a row-aligned data dict with every array/list filtered by mask.

    Every value in `data` must be aligned by row to the same N (e.g. the per-variant
    arrays and lists produced by the mechanism experiments' load_data), so a single
    boolean mask applies to all of them: numpy arrays are indexed, python lists are
    comprehended. The expected row count N is taken from `data["valid_variants"]`; a
    value whose length differs raises rather than silently mis-aligning.
    """
    n_full = len(data["valid_variants"])
    subset: dict = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            if value.shape[0] != n_full:
                raise ValueError(
                    f"data['{key}'] has {value.shape[0]} rows, expected {n_full} "
                    f"(not row-aligned to valid_variants — cannot subset by mask)"
                )
            subset[key] = value[mask]
        elif isinstance(value, list):
            if len(value) != n_full:
                raise ValueError(
                    f"data['{key}'] has {len(value)} rows, expected {n_full} "
                    f"(not row-aligned to valid_variants — cannot subset by mask)"
                )
            subset[key] = [item for item, keep in zip(value, mask) if keep]
        else:
            raise TypeError(f"Unexpected non-row-aligned value in data['{key}']: {type(value)}")
    return subset
