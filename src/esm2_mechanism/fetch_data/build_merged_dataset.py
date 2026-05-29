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
    python -m esm2_mechanism.fetch_data.build_merged_dataset
"""

import argparse
import csv
import json
from collections import Counter
import functools

from esm2_mechanism.utils_paths import DATA_DIR

print = functools.partial(print, flush=True)

GERAS_PATH    = DATA_DIR / "gerasimavicius_variants.json"
GENE_LIST_PATH = DATA_DIR / "merged_gene_list.tsv"
CLINVAR_PATH  = DATA_DIR / "clinvar_variants.tsv"
OUT_PATH      = DATA_DIR / "merged_variants.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathogenic_only", action="store_true",
                        help="Restrict ClinVar variants to 'pathogenic' only (excludes likely pathogenic)")
    args = parser.parse_args()

    missing = [p for p in [GERAS_PATH, GENE_LIST_PATH, CLINVAR_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError("Required input(s) not found:\n" + "\n".join(f"  {p}" for p in missing))

    # Load Gerasimavicius variants
    with open(GERAS_PATH) as f:
        geras = json.load(f)
    geras_genes = set(v["gene"].upper() for v in geras)
    print(f"Gerasimavicius: {len(geras)} variants, {len(geras_genes)} genes")

    # Load merged gene list for mechanism labels (G2P source)
    with open(GENE_LIST_PATH) as f:
        gene_mech_map = {r["gene"].upper(): r["mechanism"]
                         for r in csv.DictReader(f, delimiter="\t")}

    # Load ClinVar variants (G2P-only genes)
    with open(CLINVAR_PATH) as f:
        clinvar_rows = list(csv.DictReader(f, delimiter="\t"))

    if args.pathogenic_only:
        clinvar_rows = [r for r in clinvar_rows if r.get("clinsig","").lower() == "pathogenic"]
        print(f"Filtered to pathogenic only: {len(clinvar_rows)} variants")

    new_variants = []
    skipped_no_mech = 0
    skipped_no_uniprot = 0
    # Keep only variants for genes NOT in Gerasimavicius
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
            skipped_no_uniprot += 1
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
          f"(skipped {skipped_no_mech} with no mechanism label, "
          f"{skipped_no_uniprot} with no UniProt ID)")

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
    n_genes = len(set(v["gene"] for v in merged))
    sources = Counter(v["source"] for v in merged)

    print(f"\nMerged dataset: {len(merged)} variants, {n_genes} genes")
    print(f"Mechanism (4-class): {dict(mechs)}")
    print(f"Mechanism (3-class): {dict(mechs3)}")
    print(f"Sources: {dict(sources)}")

    with open(OUT_PATH, "w") as f:
        json.dump(merged, f)
    print(f"\nWritten to {OUT_PATH}")


if __name__ == "__main__":
    main()
