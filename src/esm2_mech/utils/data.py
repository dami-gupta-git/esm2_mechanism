"""Shared data loaders. Callers pass explicit paths — no filenames are constructed here."""

from __future__ import annotations

import csv
import functools
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)


def pathogenicity_label(label: str) -> int:
    """Map the two allowed pathogenicity labels to integers."""
    if label == "pathogenic":
        return 1
    if label == "benign":
        return 0
    raise ValueError(
        f"unexpected pathogenicity label {label!r} "
        "(expected 'pathogenic' or 'benign')"
    )


def protein_substitution_key(variant: dict) -> tuple[str, int, str, str]:
    """Protein-level identity used to deduplicate ClinVar records."""
    return (
        variant["gene"],
        int(variant["aa_pos"]),
        variant["aa_wt"],
        variant["aa_mut"],
    )


def validate_balanced_pathogenicity_variants(
    variants: list[dict], *, require_unique_substitutions: bool
) -> dict:
    """Validate labels, protein-substitution uniqueness, and per-gene balance.

    Returns compact realised-design counts for provenance. Any violation raises;
    an invalid scientific input is never coerced into one of the two classes.
    """
    if not variants:
        raise ValueError("pathogenicity variant set is empty")

    per_gene = defaultdict(Counter)
    substitutions: dict[tuple[str, int, str, str], str] = {}
    duplicate_keys = set()
    for variant in variants:
        label = variant["label"]
        pathogenicity_label(label)
        gene = variant["gene"]
        per_gene[gene][label] += 1

        key = protein_substitution_key(variant)
        previous_label = substitutions.get(key)
        if previous_label is None:
            substitutions[key] = label
        else:
            duplicate_keys.add(key)
            if previous_label != label:
                raise ValueError(
                    "protein substitution has conflicting pathogenicity labels: "
                    f"{key!r} is both {previous_label!r} and {label!r}"
                )

    if require_unique_substitutions and duplicate_keys:
        examples = sorted(duplicate_keys)[:5]
        raise ValueError(
            f"pathogenicity set contains {len(duplicate_keys)} repeated protein "
            f"substitution(s); examples: {examples}"
        )

    unbalanced = {
        gene: {
            "pathogenic": counts["pathogenic"],
            "benign": counts["benign"],
        }
        for gene, counts in per_gene.items()
        if counts["pathogenic"] != counts["benign"]
    }
    if unbalanced:
        examples = list(sorted(unbalanced.items()))[:5]
        raise ValueError(
            f"pathogenicity set has {len(unbalanced)} gene(s) with unequal class "
            f"counts; examples: {examples}"
        )

    per_gene_count_distribution = Counter(
        counts["pathogenic"] for counts in per_gene.values()
    )
    label_counts = Counter(variant["label"] for variant in variants)
    return {
        "n_variants": len(variants),
        "n_genes": len(per_gene),
        "n_pathogenic": label_counts["pathogenic"],
        "n_benign": label_counts["benign"],
        "per_gene_class_count_distribution": {
            str(count): n_genes
            for count, n_genes in sorted(per_gene_count_distribution.items())
        },
        "n_duplicate_substitution_keys": len(duplicate_keys),
    }


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


def embedding_variant_fingerprint(variants: list[dict]) -> str:
    """Order-sensitive hash of the fields that identify an embedding row.

    Labels and other annotations are excluded because they do not determine the
    ESM-2 input. The variant identity, substitution, and row order do.
    """
    required_fields = ("gene", "uniprot_id", "aa_pos", "aa_wt", "aa_mut")
    digest = hashlib.sha256()
    for row_index, variant in enumerate(variants):
        missing = [field for field in required_fields if field not in variant]
        if missing:
            raise ValueError(
                f"variant row {row_index} lacks embedding identity fields {missing}"
            )
        key = "|".join(str(variant[field]) for field in required_fields)
        digest.update(key.encode())
        digest.update(b"\x00")
    return digest.hexdigest()


def labeled_variant_fingerprint(
    variants: list[dict], labels: list | np.ndarray
) -> str:
    """Hash embedding-row identities together with their analysis labels."""
    if len(variants) != len(labels):
        raise ValueError(
            f"cannot fingerprint {len(variants)} variants with {len(labels)} labels"
        )
    digest = hashlib.sha256()
    digest.update(embedding_variant_fingerprint(variants).encode())
    digest.update(b"\x00")
    for row_index, label in enumerate(labels):
        if label is None:
            raise ValueError(f"variant row {row_index} has no analysis label")
        digest.update(str(label).encode())
        digest.update(b"\x00")
    return digest.hexdigest()


def validate_embedding_variant_identity(
    variants: list[dict],
    embedded_variants_path: Path,
) -> str:
    """Require the current variants to match the embedding row sidecar exactly."""
    if not embedded_variants_path.exists():
        raise FileNotFoundError(
            f"embedding row-identity sidecar missing: {embedded_variants_path}"
        )
    with open(embedded_variants_path) as handle:
        embedded_variants = json.load(handle)

    current_fingerprint = embedding_variant_fingerprint(variants)
    embedded_fingerprint = embedding_variant_fingerprint(embedded_variants)
    if current_fingerprint != embedded_fingerprint:
        raise ValueError(
            f"{embedded_variants_path} does not match the current variant identities "
            "and row order; the embedding arrays are stale or misaligned"
        )
    return current_fingerprint


def embedding_fingerprint(*arrays: np.ndarray) -> str:
    """Content hash of one or more embedding arrays.

    Catches cases where embeddings change (e.g. model weights updated)
    while the variant list and model name stay the same. Arrays are streamed into
    the digest so fingerprinting the Tsuboyama matrix does not allocate another
    copy of the full embedding array.
    """
    digest = hashlib.sha256()
    for arr in arrays:
        contiguous = np.ascontiguousarray(arr)
        byte_view = memoryview(contiguous).cast("B")
        chunk_size = 8 * 1024 * 1024
        for start in range(0, byte_view.nbytes, chunk_size):
            digest.update(byte_view[start : start + chunk_size])
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


def load_gene_universe(tsv_path: Path) -> tuple[list[str], dict[str, str]]:
    """Return (genes_in_order, {gene: pfam_family}) from gene_universe.tsv."""
    genes: list[str] = []
    families: dict[str, str] = {}
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            g = row["gene"].strip()
            if g and g not in families:
                genes.append(g)
                families[g] = row["pfam_family"].strip()
    print(f"Gene universe: {len(genes)} genes from {tsv_path.name}")
    return genes, families


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
        print(
            f"WARNING: {n_missing} variants have no `source` field — excluded from subset"
        )
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
            raise TypeError(
                f"Unexpected non-row-aligned value in data['{key}']: {type(value)}"
            )
    return subset
