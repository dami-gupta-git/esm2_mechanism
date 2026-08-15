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

Caches its variant set and re-fetches only if the fetch params change.

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

import numpy as np

from esm2_mech.utils.constants import HTTP_USER_AGENT
from esm2_mech.utils.data import load_variants
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


def _fetch_clinvar(target_genes, max_per_gene_per_class, seed):
    """Download and filter ClinVar to balanced P/B missense variants."""
    print("  Downloading ClinVar variant_summary.txt.gz (~150 MB compressed) ...")
    req = urllib.request.Request(CLINVAR_URL, headers={"User-Agent": HTTP_USER_AGENT})
    with urllib.request.urlopen(req, timeout=600) as resp:
        raw = resp.read()

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
    by_gene_class = defaultdict(list)
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
            by_gene_class[(gene, label)].append({
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

    print(f"  Scanned {n_seen} ClinVar rows; matched {sum(len(v) for v in by_gene_class.values())} variants")

    rng = np.random.RandomState(seed)
    chosen = []
    for lst in by_gene_class.values():
        rng.shuffle(lst)
        chosen.extend(lst[:max_per_gene_per_class])
    return chosen


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
            v["uniprot_id"] = uid
            out.append(v)
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


def fetch_phase(max_per_gene_per_class=20, seed=42):
    """Returns the variant list; caches to CLINVAR_PATHOGENICITY_VARIANTS_JSON."""
    print("=== Fetch ClinVar pathogenic/benign variants ===")

    # Merged mechanism variants (Gerasimavicius + G2P) — defines the target gene set
    # and the gene→UniProt map for the ClinVar pathogenicity fetch. Read before the
    # cache check: the cache key must reflect this file's current content, not just
    # the CLI args, or a stale gene list silently survives a VARIANTS_JSON change.
    mechanism_variants = load_variants(VARIANTS_JSON)
    target_genes = sorted({v["gene"].upper() for v in mechanism_variants if v.get("gene")})
    print(f"  Target gene set: {len(target_genes)} genes")

    params = {
        "max_per_gene_per_class": max_per_gene_per_class,
        "seed": seed,
        "source_fingerprint": _source_fingerprint(mechanism_variants),
    }
    if CLINVAR_PATHOGENICITY_VARIANTS_JSON.exists():
        cached_params = None
        if CLINVAR_PATHOGENICITY_PARAMS_JSON.exists():
            try:
                with open(CLINVAR_PATHOGENICITY_PARAMS_JSON) as f:
                    cached_params = json.load(f)
            except json.JSONDecodeError:
                cached_params = None
        if cached_params != params:
            print(
                f"  Cache was built with {cached_params}, current params {params} — re-fetching"
            )
        else:
            try:
                with open(CLINVAR_PATHOGENICITY_VARIANTS_JSON) as f:
                    cached = json.load(f)
                print(f"  Loaded cached set: {len(cached)} variants ({Counter(v['label'] for v in cached)})")
                return cached
            except json.JSONDecodeError:
                print("  WARNING: corrupt cache — re-fetching")

    variants = _fetch_clinvar(target_genes, max_per_gene_per_class, seed)
    variants = _attach_uniprot_ids(variants, mechanism_variants)
    print(
        f"  Final set: {len(variants)} variants ({Counter(v['label'] for v in variants)}), "
        f"{len({v['gene'] for v in variants})} genes"
    )

    CLINVAR_PATHOGENICITY_VARIANTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(CLINVAR_PATHOGENICITY_VARIANTS_JSON, variants)
    atomic_write_json(CLINVAR_PATHOGENICITY_PARAMS_JSON, params)
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_per_gene_per_class", type=int, default=20)
    parser.add_argument("--fetch_seed", type=int, default=42)
    args = parser.parse_args()
    fetch_phase(max_per_gene_per_class=args.max_per_gene_per_class, seed=args.fetch_seed)


if __name__ == "__main__":
    main()
