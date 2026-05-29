"""
Parse Gerasimavicius et al. variant table from the local Excel file.

Input : data/downloads/DiseaseMech_Stability_VEPS.xlsx  (sheet: ClinVar_gene_level)
Output: data/gerasimavicius_variants.json

Columns used: Gene, Uniprot_id, Uniprot_variant, Disease_mechanism, raw_FoldX_Monomer
Output schema per variant: gene, uniprot_id, aa_pos, aa_wt, aa_mut, mechanism, foldx_ddg, clinvar_id

Usage:
    python -m esm2_mechanism.fetch_data.fetch_gerasimavicius_variants
"""

import json
import re
import functools
from collections import Counter

import openpyxl

from esm2_mechanism.utils_paths import DATA_DIR

print = functools.partial(print, flush=True)

XLSX_PATH = DATA_DIR / "downloads" / "DiseaseMech_Stability_VEPS.xlsx"
OUT_PATH  = DATA_DIR / "gerasimavicius_variants.json"

_VARIANT_PAT = re.compile(r"^([A-Z])(\d+)([A-Z])$")

_MECH_MAP = {
    "GOF": "GOF", "DN": "DN", "HI": "HI",
    "AR": "AR", "AR, HET": "AR", "AR, HOM": "AR",
}


def parse_variants(xlsx_path) -> list[dict]:
    print(f"Loading {xlsx_path.name}...")
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    try:
        ws = wb["ClinVar_gene_level"]
        rows = list(ws.iter_rows(values_only=True))
    finally:
        wb.close()

    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    col = {h: i for i, h in enumerate(header)}
    print(f"  Sheet columns: {header[:10]}")

    required = {"Gene", "Uniprot_id", "Uniprot_variant", "Disease_mechanism", "Class"}
    missing_cols = required - col.keys()
    if missing_cols:
        raise ValueError(f"Missing expected columns in ClinVar_gene_level sheet: {missing_cols}")

    variants = []
    skipped = 0
    for row in rows[1:]:
        try:
            row_class = str(row[col["Class"]] or "").strip().upper()
            if "CLINVAR" not in row_class:
                continue

            gene        = row[col["Gene"]]
            uniprot     = row[col["Uniprot_id"]]
            variant_str = row[col["Uniprot_variant"]]
            mech_raw    = row[col["Disease_mechanism"]]
            foldx_raw   = row[col["raw_FoldX_Monomer"]] if "raw_FoldX_Monomer" in col else None

            if not all([gene, uniprot, variant_str, mech_raw]):
                skipped += 1
                continue

            mech = _MECH_MAP.get(str(mech_raw).strip().upper())
            if mech is None:
                skipped += 1
                continue

            m = _VARIANT_PAT.match(str(variant_str).strip())
            if not m:
                skipped += 1
                continue
            aa_wt, aa_pos_str, aa_mut = m.groups()

            foldx_ddg = None
            if foldx_raw is not None:
                try:
                    foldx_ddg = float(foldx_raw)
                except (ValueError, TypeError):
                    pass

            variants.append({
                "gene":       str(gene).upper(),
                "uniprot_id": str(uniprot).strip(),
                "aa_pos":     int(aa_pos_str),
                "aa_wt":      aa_wt.upper(),
                "aa_mut":     aa_mut.upper(),
                "mechanism":  mech,
                "foldx_ddg":  foldx_ddg,
                "clinvar_id": "",
            })
        except Exception as exc:
            print(f"WARNING: skipping row due to error: {exc}")
            skipped += 1

    print(f"  Parsed {len(variants)} variants ({skipped} rows skipped)")
    mechs = Counter(v["mechanism"] for v in variants)
    n_genes = len(set(v["gene"] for v in variants))
    print(f"  Genes: {n_genes} | Mechanism counts: {dict(mechs)}")
    return variants


def main():
    if not XLSX_PATH.exists():
        raise FileNotFoundError(f"Required input not found: {XLSX_PATH}")

    variants = parse_variants(XLSX_PATH)

    if not variants:
        raise ValueError("No variants parsed — check the Excel file and sheet name")

    with open(OUT_PATH, "w") as f:
        json.dump(variants, f)
    print(f"\nWritten to {OUT_PATH}")


if __name__ == "__main__":
    main()
