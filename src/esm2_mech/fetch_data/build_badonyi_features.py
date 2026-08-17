"""Build Badonyi 2024 feature columns (pDN, pGOF, pLOF) for the merged gene set."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import functools

from esm2_mech.utils.io import save_npy
from esm2_mech.utils.paths import (
    BADONYI_FEATURES_ALIGNED,
    BADONYI_FEATURES_TSV,
    BADONYI_FEATURE_COLUMNS_JSON,
    DATA_DIR,
    GENE_UNIVERSE,
    TABLE_S3_FILE,
)

print = functools.partial(print, flush=True)

DATA = DATA_DIR
S3_PATH = TABLE_S3_FILE

OUT_TSV = BADONYI_FEATURES_TSV
OUT_NPY = BADONYI_FEATURES_ALIGNED
OUT_COLS = BADONYI_FEATURE_COLUMNS_JSON


def load_badonyi_predictions():
    print("Loading Badonyi S3 Table...")
    df = pd.read_excel(S3_PATH, sheet_name="table_S3")
    df = df[["gene", "pDN", "pGOF", "pLOF"]].copy()
    df = df.rename(columns={"gene": "gene_badonyi"})
    print(f"  Loaded {len(df)} genes from S3 Table")
    return df


def load_gene_universe():
    df = pd.read_csv(GENE_UNIVERSE, sep="\t")
    print(f"  Gene universe: {len(df)} genes")
    return df


def compute_family_residuals(df, pfam, feature_cols, observed_mask=None):
    """Subtract the observed-only Pfam family mean from each continuous feature."""
    df = df.copy()
    df["pfam_family"] = df["gene"].map(pfam)

    if observed_mask is None:
        observed_mask = pd.Series(True, index=df.index)

    observed_family_counts = (
        df.loc[observed_mask, "pfam_family"].dropna().value_counts().to_dict()
    )
    df["is_singleton_family_badonyi"] = df["pfam_family"].map(
        lambda f: (
            1
            if (f is None or pd.isna(f) or observed_family_counts.get(f, 0) <= 1)
            else 0
        )
    )

    for col in feature_cols:
        resid_col = f"{col}_familyresid"
        resid_missing_col = f"{resid_col}_missing"
        df[resid_col] = float("nan")
        df[resid_missing_col] = 1

        for fam, fam_df in df.groupby("pfam_family", dropna=True):
            fam_idx = fam_df.index
            observed_in_fam = fam_idx[observed_mask.loc[fam_idx]]
            if len(observed_in_fam) <= 1:
                continue
            fam_mean = df.loc[observed_in_fam, col].mean()
            df.loc[observed_in_fam, resid_col] = df.loc[observed_in_fam, col] - fam_mean
            df.loc[observed_in_fam, resid_missing_col] = 0

    return df


def main():
    missing = [p for p in [S3_PATH, GENE_UNIVERSE] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required input(s) not found:\n" + "\n".join(f"  {p}" for p in missing)
        )

    bad = load_badonyi_predictions()
    merged = load_gene_universe()
    pfam = dict(zip(merged["gene"], merged["pfam_family"]))

    dup_mask = bad["gene_badonyi"].duplicated(keep=False)
    if dup_mask.any():
        dup_genes = sorted(bad.loc[dup_mask, "gene_badonyi"].unique())
        print(
            f"WARNING: {len(dup_genes)} Badonyi gene symbols appear on multiple rows; "
            f"keeping the first per gene. Examples: {dup_genes[:5]}"
        )
        bad = bad.drop_duplicates(subset="gene_badonyi", keep="first")
    bad_lookup = bad.set_index("gene_badonyi")[["pDN", "pGOF", "pLOF"]]

    result = merged[["gene"]].copy()
    result["pDN"] = result["gene"].map(bad_lookup["pDN"])
    result["pGOF"] = result["gene"].map(bad_lookup["pGOF"])
    result["pLOF"] = result["gene"].map(bad_lookup["pLOF"])

    n_covered = result["pDN"].notna().sum()
    print(
        f"\nCoverage: {n_covered} / {len(result)} merged genes ({100*n_covered/len(result):.1f}%)"
    )
    missing_genes = result[result["pDN"].isna()]["gene"].tolist()
    print(f"  Missing: {len(missing_genes)} genes")
    if missing_genes[:10]:
        print(f"  First 10 missing: {missing_genes[:10]}")

    feature_cols = ["pDN", "pGOF", "pLOF"]

    for col in feature_cols:
        if result[col].notna().sum() == 0:
            raise ValueError(
                f"Column '{col}' has no observed values after join — "
                "the gene-symbol join likely failed (column rename drift?). "
                "Refusing to fabricate imputation values."
            )

    for col in feature_cols:
        result[f"{col}_missing"] = result[col].isna().astype(float)

    observed_mask = result["pDN_missing"] == 0

    result = compute_family_residuals(result, pfam, feature_cols, observed_mask)

    for col in feature_cols:
        print(
            f"  {col}: {int(result[f'{col}_missing'].sum())} genes missing "
            f"(left as NaN — no imputation; consumers must restrict to the "
            f"observed subset and recompute CV splits)"
        )

    result.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\nSaved TSV: {OUT_TSV} ({result.shape})")

    numeric_cols = (
        feature_cols
        + [f"{c}_missing" for c in feature_cols]
        + [f"{c}_familyresid" for c in feature_cols]
        + [f"{c}_familyresid_missing" for c in feature_cols]
        + ["is_singleton_family_badonyi"]
    )
    mat = result[numeric_cols].values.astype(np.float32)
    save_npy(OUT_NPY, mat)
    print(f"Saved NPY: {OUT_NPY} shape={mat.shape}")

    col_meta = []
    for c in numeric_cols:
        kind = (
            "missing_indicator"
            if "_missing" in c
            else ("familyresid" if "_familyresid" in c else "raw")
        )
        col_meta.append({"name": c, "kind": kind, "source": "badonyi_2024_plosone"})
    with open(OUT_COLS, "w") as f:
        json.dump(col_meta, f, indent=2)
    print(f"Saved column metadata: {OUT_COLS}")

    print("\n--- Coverage summary ---")
    print(
        f"Genes covered (raw):    {n_covered} / {len(result)} ({100*n_covered/len(result):.1f}%)"
    )
    print(f"Genes missing (NaN):    {len(missing_genes)}")
    print(f"Matrix shape:           {mat.shape}")
    print(f"Columns:                {numeric_cols}")


if __name__ == "__main__":
    main()
