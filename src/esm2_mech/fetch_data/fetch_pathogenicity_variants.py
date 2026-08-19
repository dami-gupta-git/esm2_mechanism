"""
Fetch balanced pathogenic/benign ClinVar variants for the pathogenicity
positive control (Experiment 2).

This is a separate fetch from fetch_variants.py's ClinVar step: that step
keeps pathogenic-only variants to label genes by mechanism (DN/LOF/GOF); this
one keeps pathogenic AND benign variants, capped and balanced per gene, to
train a pathogenic-vs-benign classifier. Network-only (no GPU), so it runs
locally rather than on the pod.

Download ClinVar variant_summary.txt.gz, keep balanced pathogenic/benign
missense variants in the Gerasimavicius gene set (<= max_per_gene_per_class
each), restricted to the GRCh38 assembly (the only place a genome build is
selected — see CLINVAR_ASSEMBLY), attach UniProt IDs.

Caches its variant set. A stale or partial cache requires an explicit ``--force``
refetch so the existing scientific input is never replaced implicitly.

  Input : data/variants.json (target gene set + gene -> UniProt map)
  Output: data/clinvar_pathogenicity_variants.json,
          data/clinvar_pathogenicity_variants.params.json

Usage:
    python -m esm2_mech.fetch_data.fetch_pathogenicity_variants
        --max_per_gene_per_class 20 --fetch_seed 42
"""

from __future__ import annotations

import argparse
import functools
import gzip
import hashlib
import io
import json
import re
import urllib.request
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

from esm2_mech.utils.constants import HTTP_USER_AGENT
from esm2_mech.utils.data import (
    load_variants,
    pathogenicity_label,
    protein_substitution_key,
    validate_balanced_pathogenicity_variants,
    variants_fingerprint,
)
from esm2_mech.utils.io import atomic_write_json
from esm2_mech.utils.paths import (
    CLINVAR_PATHOGENICITY_PARAMS_JSON,
    CLINVAR_PATHOGENICITY_VARIANTS_JSON,
    VARIANTS_JSON,
)

print = functools.partial(print, flush=True)

CLINVAR_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)

# variant_summary.txt.gz lists each variant once per genome assembly (GRCh37 and
# GRCh38). Restrict to GRCh38 — the current standard build — so the same variant
# is not counted twice. This is the only place in the pipeline a genome build is
# selected; all other variant data is handled in protein coordinates (UniProt
# position + amino acid), which are assembly-independent.
CLINVAR_ASSEMBLY = "GRCh38"

AA3 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}
HGVSP_PAT = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})(?=[^a-zA-Z]|$)")

# Bump whenever selection, deduplication, or balancing changes.
_BALANCE_VERSION = 3
_FETCH_METADATA_VERSION = 3
_DUPLICATE_POLICY = "deduplicate_protein_substitution_before_balance"


class StalePathogenicityCacheError(RuntimeError):
    """The cached ClinVar set was not produced by the current selection contract."""


def _deduplicate_protein_substitutions(variants):
    """Merge ClinVar records encoding the same protein substitution.

    Distinct records are retained as ``clinvar_ids`` provenance on the one
    protein-level row. A substitution carrying conflicting labels across
    ClinVar records is dropped entirely rather than having either label
    selected for it.
    """
    unique: dict[tuple[str, int, str, str], dict] = {}
    ordered_keys = []
    duplicate_keys = set()
    conflicting_keys = set()
    n_duplicate_rows_removed = 0
    for variant in variants:
        pathogenicity_label(variant["label"])
        key = protein_substitution_key(variant)
        if key in conflicting_keys:
            continue
        if key not in unique:
            row = dict(variant)
            row["clinvar_ids"] = [variant["clinvar_id"]]
            unique[key] = row
            ordered_keys.append(key)
            continue

        duplicate_keys.add(key)
        n_duplicate_rows_removed += 1
        existing = unique[key]
        if existing["label"] != variant["label"]:
            conflicting_keys.add(key)
            del unique[key]
            continue
        if variant["clinvar_id"] not in existing["clinvar_ids"]:
            existing["clinvar_ids"].append(variant["clinvar_id"])

    deduplicated = []
    for key in ordered_keys:
        if key in conflicting_keys:
            continue
        row = unique[key]
        row["clinvar_ids"] = sorted(row["clinvar_ids"])
        deduplicated.append(row)
    return deduplicated, {
        "duplicate_policy": _DUPLICATE_POLICY,
        "n_duplicate_substitution_keys": len(duplicate_keys),
        "n_duplicate_rows_removed": n_duplicate_rows_removed,
        "n_conflicting_substitutions_dropped": len(conflicting_keys),
    }


def _fetch_clinvar(target_genes, max_per_gene_per_class, seed):
    """Download and filter ClinVar to balanced P/B missense variants."""
    print("  Downloading ClinVar variant_summary.txt.gz (~150 MB compressed) ...")
    req = urllib.request.Request(CLINVAR_URL, headers={"User-Agent": HTTP_USER_AGENT})
    with urllib.request.urlopen(req, timeout=600) as resp:
        raw = resp.read()
        last_modified = resp.headers.get("Last-Modified")

    clinvar_source = {
        "url": CLINVAR_URL,
        "assembly": CLINVAR_ASSEMBLY,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "compressed_sha256": hashlib.sha256(raw).hexdigest(),
        "compressed_bytes": len(raw),
        "last_modified": last_modified,
    }

    print(f"  Downloaded {len(raw) / 1e6:.0f} MB, decompressing ...")
    gz = gzip.GzipFile(fileobj=io.BytesIO(raw))
    text = io.TextIOWrapper(gz, encoding="utf-8", errors="replace")
    header = text.readline().rstrip("\n").split("\t")
    col = {(h[1:] if h.startswith("#") else h): i for i, h in enumerate(header)}
    needed = ["Type", "GeneSymbol", "ClinicalSignificance", "Name", "VariationID", "Assembly"]
    for c in needed:
        if c not in col:
            raise RuntimeError(f"Missing ClinVar column: {c}")
    max_needed_idx = max(col[c] for c in needed)

    target_set = set(target_genes)
    matched = []
    n_seen = 0
    try:
        for line in text:
            n_seen += 1
            parts = line.rstrip("\n").split("\t")
            # Only the read columns need to be present; guard their max index,
            # not the full header width (trailing empty fields may be omitted).
            if len(parts) <= max_needed_idx:
                continue
            if parts[col["Type"]] != "single nucleotide variant":
                continue
            if parts[col["Assembly"]] != CLINVAR_ASSEMBLY:
                continue
            gene = parts[col["GeneSymbol"]].upper()
            if gene not in target_set:
                continue
            sig_low = parts[col["ClinicalSignificance"]].strip().lower()
            if any(s in sig_low for s in ["conflict", "uncertain", "not provided", "other", "no assertion"]):
                continue
            if "pathogenic" in sig_low and "non-pathogenic" not in sig_low:
                label = "pathogenic"
            elif "benign" in sig_low:
                label = "benign"
            else:
                continue
            m = HGVSP_PAT.search(parts[col["Name"]])
            if not m:
                continue
            wt3, pos_s, mut3 = m.groups()
            if wt3 not in AA3 or mut3 not in AA3 or wt3 == mut3:
                continue
            matched.append({
                "gene": gene,
                "aa_pos": int(pos_s),
                "aa_wt": AA3[wt3],
                "aa_mut": AA3[mut3],
                "label": label,
                "clinvar_id": parts[col["VariationID"]],
            })
    except (EOFError, gzip.BadGzipFile, zlib.error) as exc:
        # A truncated download decompresses partway then fails. Do NOT return a
        # partial result that the caller would cache as success.
        raise RuntimeError(
            f"ClinVar download appears truncated after {n_seen} rows ({exc}). "
            "Re-run to retry; nothing was cached."
        ) from exc

    print(f"  Scanned {n_seen} ClinVar rows; matched {len(matched)} records")

    deduplicated, duplicate_accounting = _deduplicate_protein_substitutions(matched)
    print(
        f"  Protein substitutions: {len(deduplicated)} unique; removed "
        f"{duplicate_accounting['n_duplicate_rows_removed']} repeated record(s)"
    )

    by_gene_class = defaultdict(list)
    for variant in deduplicated:
        by_gene_class[(variant["gene"], variant["label"])].append(variant)

    rng = np.random.RandomState(seed)
    capped = {}
    for (gene, label), lst in by_gene_class.items():
        rng.shuffle(lst)
        capped[(gene, label)] = lst[:max_per_gene_per_class]

    genes_with_both = {
        gene for (gene, label) in capped
        if (gene, "pathogenic") in capped and (gene, "benign") in capped
    }
    chosen = []
    for gene in sorted(genes_with_both):
        n = min(len(capped[(gene, "pathogenic")]), len(capped[(gene, "benign")]))
        chosen.extend(capped[(gene, "pathogenic")][:n])
        chosen.extend(capped[(gene, "benign")][:n])

    n_dropped = len(set(g for (g, _) in capped) - genes_with_both)
    if n_dropped:
        print(f"  Dropped {n_dropped} genes with only one class")
    accounting = {
        "n_clinvar_rows_scanned": n_seen,
        "n_matched_records": len(matched),
        "n_unique_substitutions": len(deduplicated),
        **duplicate_accounting,
        "n_after_cap_before_balance": sum(len(rows) for rows in capped.values()),
        "n_genes_with_both_classes": len(genes_with_both),
        "n_single_class_genes_dropped": n_dropped,
        "n_after_balance_before_uniprot": len(chosen),
    }
    return chosen, clinvar_source, accounting


def _attach_uniprot_ids(variants, mechanism_variants):
    gene_to_uid = {
        v["gene"].upper(): v["uniprot_id"]
        for v in mechanism_variants
        if v.get("gene") and v.get("uniprot_id")
    }
    out = []
    for v in variants:
        uid = gene_to_uid.get(v["gene"].upper())
        if uid:
            out.append({**v, "uniprot_id": uid})
    print(f"  {len(out)}/{len(variants)} variants mapped to a UniProt ID")
    return out


def _source_fingerprint(mechanism_variants):
    """Content hash of the target gene set and gene→UniProt map this fetch reads
    from VARIANTS_JSON, so a cache built before that file changed (genes added or
    removed, a UniProt ID corrected) is detected even though the CLI args match.
    """
    pairs = sorted({
        (v["gene"].upper(), v.get("uniprot_id"))
        for v in mechanism_variants if v.get("gene")
    })
    digest = hashlib.sha256()
    for gene, uid in pairs:
        digest.update(f"{gene}|{uid}".encode())
        digest.update(b"\x00")
    return digest.hexdigest()


def _selection_params(mechanism_variants, max_per_gene_per_class, seed):
    return {
        "max_per_gene_per_class": int(max_per_gene_per_class),
        "seed": int(seed),
        "source_fingerprint": _source_fingerprint(mechanism_variants),
        "balance_version": _BALANCE_VERSION,
        "duplicate_policy": _DUPLICATE_POLICY,
    }


def _validate_fetch_metadata(metadata, current_selection):
    if metadata.get("metadata_version") != _FETCH_METADATA_VERSION:
        raise StalePathogenicityCacheError(
            "ClinVar pathogenicity cache metadata_version is missing or stale; "
            "rerun with --force to rebuild both cache files"
        )
    cached_selection = metadata.get("selection")
    if not isinstance(cached_selection, dict):
        raise StalePathogenicityCacheError(
            "ClinVar pathogenicity cache has no selection metadata; rerun with "
            "--force to rebuild both cache files"
        )
    missing = sorted(set(current_selection) - set(cached_selection))
    if missing:
        raise StalePathogenicityCacheError(
            "ClinVar pathogenicity cache is missing selection key(s) "
            f"{missing}; rerun with --force"
        )
    changed = {
        key: {"cached": cached_selection[key], "current": value}
        for key, value in current_selection.items()
        if cached_selection[key] != value
    }
    if changed or set(cached_selection) != set(current_selection):
        raise StalePathogenicityCacheError(
            "ClinVar pathogenicity cache selection differs from the current "
            f"contract: {changed}; rerun with --force"
        )
    for required in ("clinvar_source", "accounting", "variant_fingerprint"):
        if required not in metadata:
            raise StalePathogenicityCacheError(
                f"ClinVar pathogenicity cache metadata is missing {required!r}; "
                "rerun with --force"
            )
    if not isinstance(metadata["clinvar_source"], dict):
        raise StalePathogenicityCacheError(
            "ClinVar pathogenicity source provenance is malformed; rerun with --force"
        )
    if not isinstance(metadata["accounting"], dict):
        raise StalePathogenicityCacheError(
            "ClinVar pathogenicity accounting is malformed; rerun with --force"
        )


def validate_cached_pathogenicity_variants(variants, metadata, current_selection):
    """Validate a fetched variant JSON and its metadata as one artifact."""
    _validate_fetch_metadata(metadata, current_selection)
    actual_fingerprint = variants_fingerprint(variants)
    if metadata["variant_fingerprint"] != actual_fingerprint:
        raise StalePathogenicityCacheError(
            "ClinVar pathogenicity variant JSON does not match the fingerprint in "
            "its metadata; rerun with --force"
        )
    realised = validate_balanced_pathogenicity_variants(
        variants, require_unique_substitutions=True
    )
    recorded = metadata["accounting"].get("realised_design")
    if recorded != realised:
        raise StalePathogenicityCacheError(
            "ClinVar pathogenicity realised-design accounting does not match the "
            "variant JSON; rerun with --force"
        )
    return realised


def load_validated_pathogenicity_cache(max_per_gene_per_class, seed):
    """Load the fetched set only when both files match the caller's contract."""
    if not CLINVAR_PATHOGENICITY_VARIANTS_JSON.exists():
        raise FileNotFoundError(
            f"{CLINVAR_PATHOGENICITY_VARIANTS_JSON} not found; run "
            "`python -m esm2_mech.fetch_data.fetch_pathogenicity_variants --force`"
        )
    if not CLINVAR_PATHOGENICITY_PARAMS_JSON.exists():
        raise FileNotFoundError(
            f"{CLINVAR_PATHOGENICITY_PARAMS_JSON} not found; the variant set has no "
            "verifiable fetch provenance"
        )
    try:
        with open(CLINVAR_PATHOGENICITY_PARAMS_JSON) as handle:
            metadata = json.load(handle)
        with open(CLINVAR_PATHOGENICITY_VARIANTS_JSON) as handle:
            variants = json.load(handle)
    except json.JSONDecodeError as exc:
        raise StalePathogenicityCacheError(
            "ClinVar pathogenicity cache JSON is corrupt; rerun the fetch with --force"
        ) from exc

    mechanism_variants = load_variants(VARIANTS_JSON)
    current_selection = _selection_params(
        mechanism_variants,
        max_per_gene_per_class,
        seed,
    )
    validate_cached_pathogenicity_variants(
        variants, metadata, current_selection
    )
    return variants, metadata


def fetch_phase(max_per_gene_per_class=20, seed=42, force=False):
    """Returns the variant list; caches to CLINVAR_PATHOGENICITY_VARIANTS_JSON."""
    print("=== Fetch ClinVar pathogenic/benign variants ===")

    # Merged mechanism variants (Gerasimavicius + G2P) — defines the target gene set
    # and the gene→UniProt map for the ClinVar pathogenicity fetch. Read before the
    # cache check: the cache key must reflect this file's current content, not just
    # the CLI args, or a stale gene list silently survives a VARIANTS_JSON change.
    mechanism_variants = load_variants(VARIANTS_JSON)
    target_genes = sorted({v["gene"].upper() for v in mechanism_variants if v.get("gene")})
    print(f"  Target gene set: {len(target_genes)} genes")

    selection = _selection_params(mechanism_variants, max_per_gene_per_class, seed)
    cache_files_exist = (
        CLINVAR_PATHOGENICITY_VARIANTS_JSON.exists(),
        CLINVAR_PATHOGENICITY_PARAMS_JSON.exists(),
    )
    if any(cache_files_exist) and not all(cache_files_exist) and not force:
        raise StalePathogenicityCacheError(
            "only one ClinVar pathogenicity cache file exists; rerun with --force "
            "to rebuild the pair"
        )
    if all(cache_files_exist) and not force:
        try:
            with open(CLINVAR_PATHOGENICITY_PARAMS_JSON) as f:
                metadata = json.load(f)
            with open(CLINVAR_PATHOGENICITY_VARIANTS_JSON) as f:
                cached = json.load(f)
        except json.JSONDecodeError as exc:
            raise StalePathogenicityCacheError(
                "ClinVar pathogenicity cache JSON is corrupt; rerun with --force"
            ) from exc
        validate_cached_pathogenicity_variants(cached, metadata, selection)
        print(
            f"  Loaded cached set: {len(cached)} variants "
            f"({Counter(v['label'] for v in cached)})"
        )
        return cached

    variants, clinvar_source, accounting = _fetch_clinvar(
        target_genes, max_per_gene_per_class, seed
    )
    n_before_uniprot = len(variants)
    variants = _attach_uniprot_ids(variants, mechanism_variants)
    realised_design = validate_balanced_pathogenicity_variants(
        variants, require_unique_substitutions=True
    )
    accounting = {
        **accounting,
        "n_unmapped_to_uniprot_removed": n_before_uniprot - len(variants),
        "n_fetched_variants": len(variants),
        "realised_design": realised_design,
    }
    metadata = {
        "metadata_version": _FETCH_METADATA_VERSION,
        "selection": selection,
        "clinvar_source": clinvar_source,
        "accounting": accounting,
        "variant_fingerprint": variants_fingerprint(variants),
    }
    print(
        f"  Final set: {len(variants)} variants ({Counter(v['label'] for v in variants)}), "
        f"{len({v['gene'] for v in variants})} genes"
    )

    CLINVAR_PATHOGENICITY_VARIANTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(CLINVAR_PATHOGENICITY_VARIANTS_JSON, variants)
    atomic_write_json(CLINVAR_PATHOGENICITY_PARAMS_JSON, metadata)
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_per_gene_per_class", type=int, default=20)
    parser.add_argument("--fetch_seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing cache after validating that a full refetch is intended",
    )
    args = parser.parse_args()
    fetch_phase(
        max_per_gene_per_class=args.max_per_gene_per_class,
        seed=args.fetch_seed,
        force=args.force,
    )


if __name__ == "__main__":
    main()
