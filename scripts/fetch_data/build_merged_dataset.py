"""
Build merged variant dataset: Gerasimavicius + G2P/ClinVar.

Priority: Gerasimavicius for genes present in both sources (has FoldX ddG).
ClinVar variants used only for genes NOT in Gerasimavicius.
Mechanism labels come from merged_gene_list.tsv (G2P-only genes have LOF/GOF/DN).

Output: merged_variants.json — same schema as gerasimavicius_variants.json:
  gene, uniprot_id, aa_pos, aa_wt, aa_mut, mechanism, foldx_ddg, clinvar_id
  + label_3class (GOF/DN/LOF, collapsing HI+AR)
  + source (gerasimavicius / clinvar_g2p)

Usage:
    python scripts/fetch_data/build_merged_dataset.py --data_dir data --out data/merged_variants.json
"""

import argparse
import csv
import json
import os
from collections import Counter
import functools
print = functools.partial(print, flush=True)

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="../data")
parser.add_argument("--out", default="../data/merged_variants.json")
parser.add_argument("--pathogenic_only", action="store_true",
                    help="Restrict ClinVar variants to 'pathogenic' only (excludes likely pathogenic)")
args = parser.parse_args()

# Load Gerasimavicius variants
geras_path = os.path.join(args.data_dir, "gerasimavicius_variants.json")
with open(geras_path) as f:
    geras = json.load(f)
geras_genes = set(v["gene"].upper() for v in geras)
print(f"Gerasimavicius: {len(geras)} variants, {len(geras_genes)} genes")

# Load merged gene list for mechanism labels (G2P source)
gene_list_path = os.path.join(args.data_dir, "merged_gene_list.tsv")
with open(gene_list_path) as f:
    gene_mech_map = {r["gene"].upper(): r["mechanism"]
                     for r in csv.DictReader(f, delimiter="\t")}

# Load ClinVar variants (G2P-only genes)
clinvar_path = os.path.join(args.data_dir, "clinvar_variants.tsv")
with open(clinvar_path) as f:
    clinvar_rows = list(csv.DictReader(f, delimiter="\t"))

# Keep only variants for genes NOT in Gerasimavicius
if args.pathogenic_only:
    clinvar_rows = [r for r in clinvar_rows if r.get("clinsig","").lower() == "pathogenic"]
    print(f"Filtered to pathogenic only: {len(clinvar_rows)} variants")

new_variants = []
skipped_no_mech = 0
for r in clinvar_rows:
    gene = r["gene"].upper()
    if gene in geras_genes:
        continue
    mech = gene_mech_map.get(gene)
    if not mech:
        skipped_no_mech += 1
        continue
    # G2P uses "LOF" where Gerasimavicius distinguishes "HI" and "AR".
    # Map to "HI" so the 3-class collapse (HI+AR→LOF) works correctly.
    # WARNING: all G2P LOF genes become mechanism="HI" in the merged dataset.
    # Do NOT run the 4-class HI/AR secondary probe on merged data — G2P LOF
    # genes will all appear as HI, inflating that class and making HI/AR meaningless.
    # The source field ("clinvar_g2p") identifies these rows.
    if mech == "LOF":
        mech = "HI"
    if mech not in ("GOF", "DN", "HI", "AR"):
        continue
    try:
        aa_pos = int(r["aa_pos"])
    except (ValueError, TypeError):
        continue
    uniprot_id = r.get("uniprot_id", "").strip()
    if not uniprot_id:
        skipped_no_mech += 1  # reuse counter — "no usable row"
        continue
    new_variants.append({
        "gene": gene,
        "uniprot_id": uniprot_id,
        "aa_pos": aa_pos,
        "aa_wt": r["aa_wt"].upper(),
        "aa_mut": r["aa_mut"].upper(),
        "mechanism": mech,
        "foldx_ddg": None,
        "clinvar_id": "",
        "source": "clinvar_g2p",
    })

print(f"ClinVar new-gene variants: {len(new_variants)} "
      f"(skipped {skipped_no_mech} with no mechanism label)")

# Tag Gerasimavicius variants with source
for v in geras:
    v["source"] = "gerasimavicius"

# Merge
merged = geras + new_variants

# Add label_3class
for v in merged:
    v["label_3class"] = "LOF" if v["mechanism"] in ("HI", "AR") else v["mechanism"]

# Stats
mechs = Counter(v["mechanism"] for v in merged)
mechs3 = Counter(v["label_3class"] for v in merged)
genes = len(set(v["gene"] for v in merged))
sources = Counter(v["source"] for v in merged)

print(f"\nMerged dataset: {len(merged)} variants, {genes} genes")
print(f"Mechanism (4-class): {dict(mechs)}")
print(f"Mechanism (3-class): {dict(mechs3)}")
print(f"Sources: {dict(sources)}")

with open(args.out, "w") as f:
    json.dump(merged, f)
print(f"\nWritten to {args.out}")
