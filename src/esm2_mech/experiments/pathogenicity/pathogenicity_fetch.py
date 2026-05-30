"""
Pathogenicity positive control — phase 1: fetch ClinVar variants.

Downloads ClinVar variant_summary.txt.gz, filters to missense variants in the
Gerasimavicius gene set, balances pathogenic vs benign per gene, and attaches
UniProt IDs from the Gerasimavicius variant table.

Input:  {run_dir}/data/variants.json  (Gerasimavicius merged variants)
Output: {run_dir}/data/clinvar_pathogenicity_variants.json

Re-running loads from cache if the output file already exists.

Usage:
    python -m esm2_mech.experiments.pathogenicity.pathogenicity_fetch \\
        --run_dir run_0
"""

import functools
import gzip
import io
import json
import os
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

from esm2_mech.utils.data import load_variants

print = functools.partial(print, flush=True)

CLINVAR_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)

AA3 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}
HGVSP_PAT = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})(?=[^a-zA-Z]|$)")


def fetch_clinvar_variants(target_genes, cache_dir, max_per_gene_per_class=20, seed=42):
    """
    Download ClinVar variant_summary.txt.gz, filter to:
      - Type == "single nucleotide variant"
      - GeneSymbol in target_genes
      - HGVSp parseable as a single missense
      - ClinicalSignificance in {Pathogenic/Likely_pathogenic, Benign/Likely_benign}
      - Per gene per class: cap at max_per_gene_per_class (random subsample)

    Returns list of dicts: {gene, aa_pos, aa_wt, aa_mut, label, clinvar_id}
    (uniprot_id filled later by attach_uniprot_ids)
    """
    cache = os.path.join(cache_dir, "clinvar_pathogenicity_variants.json")
    if os.path.exists(cache):
        try:
            with open(cache) as f:
                data = json.load(f)
            print(f"  Loading cached ClinVar variants from {cache}")
            return data
        except json.JSONDecodeError:
            print(f"  WARNING: corrupt cache {cache} — re-fetching")
            os.remove(cache)

    print("  Downloading ClinVar variant_summary.txt.gz (~150 MB compressed) ...")
    req = urllib.request.Request(CLINVAR_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        raw = resp.read()

    print(f"  Downloaded {len(raw)/1e6:.0f} MB, decompressing ...")
    gz = gzip.GzipFile(fileobj=io.BytesIO(raw))
    text = io.TextIOWrapper(gz, encoding="utf-8", errors="replace")
    header = text.readline().rstrip("\n").split("\t")
    col = {h.lstrip("#"): i for i, h in enumerate(header)}
    needed = ["Type", "GeneSymbol", "ClinicalSignificance", "Name", "VariationID", "Assembly"]
    for c in needed:
        if c not in col:
            raise RuntimeError(f"Missing ClinVar column: {c}")

    target_set = set(target_genes)
    by_gene_class = defaultdict(list)
    n_seen = 0
    for line in text:
        n_seen += 1
        parts = line.rstrip("\n").split("\t")
        if len(parts) < len(header):
            continue
        if parts[col["Type"]] != "single nucleotide variant":
            continue
        if parts[col["Assembly"]] != "GRCh38":
            continue
        gene = parts[col["GeneSymbol"]].upper()
        if gene not in target_set:
            continue
        sig = parts[col["ClinicalSignificance"]].strip()
        sig_low = sig.lower()
        if any(s in sig_low for s in ["conflict", "uncertain", "not provided", "other", "no assertion"]):
            continue
        if "pathogenic" in sig_low and "non-pathogenic" not in sig_low:
            label = "pathogenic"
        elif "benign" in sig_low:
            label = "benign"
        else:
            continue
        name = parts[col["Name"]]
        m = HGVSP_PAT.search(name)
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

    print(f"  Scanned {n_seen} ClinVar rows; matched {sum(len(v) for v in by_gene_class.values())} variants")

    rng = np.random.RandomState(seed)
    chosen = []
    for (gene, label), lst in by_gene_class.items():
        rng.shuffle(lst)
        chosen.extend(lst[:max_per_gene_per_class])

    print(
        f"  After per-gene-per-class cap: {len(chosen)} variants "
        f"({sum(1 for v in chosen if v['label']=='pathogenic')} pathogenic, "
        f"{sum(1 for v in chosen if v['label']=='benign')} benign, "
        f"{len(set(v['gene'] for v in chosen))} genes)"
    )

    tmp = cache + ".tmp"
    with open(tmp, "w") as f:
        json.dump(chosen, f)
    os.replace(tmp, cache)
    return chosen


def attach_uniprot_ids(variants, gerasimavicius_variants):
    gene_to_uid = {}
    for v in gerasimavicius_variants:
        if v.get("gene") and v.get("uniprot_id"):
            gene_to_uid[v["gene"].upper()] = v["uniprot_id"]
    out = []
    for v in variants:
        uid = gene_to_uid.get(v["gene"].upper())
        if uid:
            v["uniprot_id"] = uid
            out.append(v)
    print(f"  {len(out)}/{len(variants)} variants mapped to UniProt IDs present in seq_cache")
    return out


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", default="run_0")
    p.add_argument("--max_per_gene_per_class", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data_dir = os.path.join(args.run_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    gerasimavicius = load_variants(Path(data_dir) / "variants.json")
    target_genes = sorted(set(v["gene"].upper() for v in gerasimavicius if v.get("gene")))
    print(f"Target gene set: {len(target_genes)} genes from Gerasimavicius")

    variants = fetch_clinvar_variants(
        target_genes, data_dir,
        max_per_gene_per_class=args.max_per_gene_per_class,
        seed=args.seed,
    )
    variants = attach_uniprot_ids(variants, gerasimavicius)

    from collections import Counter
    print(
        f"\nFinal ClinVar set: {len(variants)} variants  "
        f"({Counter(v['label'] for v in variants)})  "
        f"{len(set(v['gene'] for v in variants))} genes"
    )
    print(f"Written to {data_dir}/clinvar_pathogenicity_variants.json")


if __name__ == "__main__":
    main()
