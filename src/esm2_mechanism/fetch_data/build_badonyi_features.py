"""
Build Badonyi 2024 feature columns for the merged gene set.

Source: Badonyi & Marsh 2024 PLOS One (DOI: 10.1371/journal.pone.0307312)
        S3 Table — per-gene SVM probability scores pDN, pGOF, pLOF for 20,365 human proteins.
        Downloaded from OSF: https://osf.io/download/7bftj/

Outputs:
    data/badonyi_features.tsv          — gene x 10 columns (raw + missing + familyresid)
    data/badonyi_features_aligned.npy  — float32 matrix aligned to merged_gene_list.tsv row order
    data/badonyi_feature_columns.json  — column metadata
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import functools

from esm2_mechanism.utils_paths import DATA_DIR

print = functools.partial(print, flush=True)

DATA = DATA_DIR
CACHE = DATA / "cache" / "badonyi"

S3_PATH = CACHE / "table_S3.xlsx"
MERGED_GENE_LIST = DATA / "merged_gene_list.tsv"
PFAM_FAMILIES = DATA / "pfam_families.json"

OUT_TSV = DATA / "badonyi_features.tsv"
OUT_NPY = DATA / "badonyi_features_aligned.npy"
OUT_COLS = DATA / "badonyi_feature_columns.json"


def load_badonyi_predictions():
    print("Loading Badonyi S3 Table...")
    df = pd.read_excel(S3_PATH, sheet_name="table_S3")
    df = df[["gene", "uniprot_id", "pDN", "pGOF", "pLOF"]].copy()
    df = df.rename(columns={"gene": "gene_badonyi"})
    # Some gene symbols may differ — keep both for join diagnostics
    print(f"  Loaded {len(df)} genes from S3 Table")
    return df


def load_merged_genes():
    genes = pd.read_csv(MERGED_GENE_LIST, sep="\t")
    print(f"  Merged gene list: {len(genes)} genes")
    return genes


def load_pfam_families():
    with open(PFAM_FAMILIES) as f:
        pfam = json.load(f)
    # pfam maps gene_symbol -> pfam_family string
    return pfam


def compute_family_residuals(df, pfam, feature_cols):
    """For each continuous feature, subtract the mean of genes in the same Pfam family."""
    df = df.copy()
    family_map = {g: pfam.get(g, None) for g in df["gene"]}
    df["pfam_family"] = df["gene"].map(family_map)

    is_singleton = []
    for _, row in df.iterrows():
        fam = row["pfam_family"]
        if fam is None:
            is_singleton.append(1)
            continue
        n_in_fam = (df["pfam_family"] == fam).sum()
        is_singleton.append(1 if n_in_fam <= 1 else 0)
    df["is_singleton_family_badonyi"] = is_singleton

    for col in feature_cols:
        resid_col = f"{col}_familyresid"
        df[resid_col] = 0.0
        for fam in df["pfam_family"].dropna().unique():
            mask = df["pfam_family"] == fam
            if mask.sum() <= 1:
                continue
            fam_mean = df.loc[mask, col].mean()
            df.loc[mask, resid_col] = df.loc[mask, col] - fam_mean
        # Singleton residuals stay 0 (uninformative)

    return df


def main():
    missing = [p for p in [S3_PATH, MERGED_GENE_LIST, PFAM_FAMILIES] if not p.exists()]
    if missing:
        raise FileNotFoundError("Required input(s) not found:\n" + "\n".join(f"  {p}" for p in missing))

    CACHE.mkdir(parents=True, exist_ok=True)

    bad = load_badonyi_predictions()
    merged = load_merged_genes()
    pfam = load_pfam_families()

    # Join on gene symbol (case-sensitive exact match)
    merged_genes = merged["gene"].tolist()
    bad_lookup = bad.set_index("gene_badonyi")[["pDN", "pGOF", "pLOF"]]

    result = merged[["gene"]].copy()
    result["pDN"] = result["gene"].map(bad_lookup["pDN"])
    result["pGOF"] = result["gene"].map(bad_lookup["pGOF"])
    result["pLOF"] = result["gene"].map(bad_lookup["pLOF"])

    n_covered = result["pDN"].notna().sum()
    print(f"\nCoverage: {n_covered} / {len(result)} merged genes ({100*n_covered/len(result):.1f}%)")
    missing_genes = result[result["pDN"].isna()]["gene"].tolist()
    print(f"  Missing: {len(missing_genes)} genes")
    if missing_genes[:10]:
        print(f"  First 10 missing: {missing_genes[:10]}")

    # Missingness indicators
    feature_cols = ["pDN", "pGOF", "pLOF"]
    for col in feature_cols:
        result[f"{col}_missing"] = result[col].isna().astype(float)

    # Median impute missing values
    for col in feature_cols:
        median_val = result[col].median()
        result[col] = result[col].fillna(median_val)
        print(f"  {col}: median={median_val:.4f}, imputed {result[f'{col}_missing'].sum():.0f} genes")

    # Family-mean-centred residuals
    result = compute_family_residuals(result, pfam, feature_cols)

    # Residual missingness indicators (singleton genes get residual=0, mark them)
    for col in feature_cols:
        resid_col = f"{col}_familyresid"
        result[f"{resid_col}_missing"] = result["is_singleton_family_badonyi"].astype(float)

    # Save TSV
    result.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\nSaved TSV: {OUT_TSV} ({result.shape})")

    # Build aligned numpy matrix (same row order as merged_gene_list.tsv)
    numeric_cols = (
        feature_cols
        + [f"{c}_missing" for c in feature_cols]
        + [f"{c}_familyresid" for c in feature_cols]
        + [f"{c}_familyresid_missing" for c in feature_cols]
        + ["is_singleton_family_badonyi"]
    )
    mat = result[numeric_cols].values.astype(np.float32)
    np.save(OUT_NPY, mat)
    print(f"Saved NPY: {OUT_NPY} shape={mat.shape}")

    # Column metadata
    col_meta = []
    for c in numeric_cols:
        kind = "missing_indicator" if "_missing" in c else ("familyresid" if "_familyresid" in c else "raw")
        col_meta.append({"name": c, "kind": kind, "source": "badonyi_2024_plosone"})
    with open(OUT_COLS, "w") as f:
        json.dump(col_meta, f, indent=2)
    print(f"Saved column metadata: {OUT_COLS}")

    print("\n--- Coverage summary ---")
    print(f"Genes covered (raw):    {n_covered} / {len(result)} ({100*n_covered/len(result):.1f}%)")
    print(f"Genes imputed:          {len(missing_genes)}")
    print(f"Matrix shape:           {mat.shape}")
    print(f"Columns:                {numeric_cols}")


if __name__ == "__main__":
    main()
