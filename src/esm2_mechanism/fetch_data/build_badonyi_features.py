"""
Build Badonyi 2024 feature columns for the merged gene set.

Source: Badonyi & Marsh 2024 PLOS One (DOI: 10.1371/journal.pone.0307312)
        S3 Table — per-gene SVM probability scores pDN, pGOF, pLOF for 20,365 human proteins.
        Downloaded from OSF: https://osf.io/download/7bftj/

Outputs:
    data/badonyi_features.tsv          — gene x 10 columns (raw + missing + familyresid)
    data/badonyi_features_aligned.npy  — float32 matrix aligned to gene_list.tsv row order
    data/badonyi_feature_columns.json  — column metadata
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import functools

from esm2_mechanism.utils_paths import DATA_DIR, TABLE_S3_FILE

print = functools.partial(print, flush=True)

DATA = DATA_DIR
S3_PATH = TABLE_S3_FILE
GENE_UNIVERSE = DATA / "gene_universe.tsv"

OUT_TSV = DATA / "badonyi_features.tsv"
OUT_NPY = DATA / "badonyi_features_aligned.npy"
OUT_COLS = DATA / "badonyi_feature_columns.json"


def load_badonyi_predictions():
    print("Loading Badonyi S3 Table...")
    df = pd.read_excel(S3_PATH, sheet_name="table_S3")
    df = df[["gene", "pDN", "pGOF", "pLOF"]].copy()
    df = df.rename(columns={"gene": "gene_badonyi"})
    # Some gene symbols may differ — keep both for join diagnostics
    print(f"  Loaded {len(df)} genes from S3 Table")
    return df


def load_gene_universe():
    df = pd.read_csv(GENE_UNIVERSE, sep="\t")
    print(f"  Gene universe: {len(df)} genes")
    return df


def compute_family_residuals(df, pfam, feature_cols, observed_mask=None):
    """For each continuous feature, subtract the mean of observed genes in the same Pfam family.

    observed_mask: boolean Series (same index as df) marking genes with real (non-imputed)
    scores. Family means are computed only over observed genes, so imputed values don't
    contaminate the family mean. If None, all genes are treated as observed.

    Residual assignment rules:
    - All genes in a family with ≥2 observed members: residual = value − observed family mean.
      _familyresid_missing = 0. (Family mean computed from observed-only genes.)
    - Singletons or families with ≤1 observed member: residual = NaN, _familyresid_missing = 1.
    is_singleton_family_badonyi = 1 when the gene has no Pfam entry or its family has ≤1 observed
    member — exactly matching the condition that produces NaN residuals above.
    The raw _missing column (e.g. pDN_missing) already records whether the score was imputed;
    _familyresid_missing records only whether a residual exists, not how the score was obtained.
    """
    df = df.copy()
    df["pfam_family"] = df["gene"].map(pfam)

    if observed_mask is None:
        observed_mask = pd.Series(True, index=df.index)

    # Count observed genes per family — mirrors the residual loop's ≥2 observed condition.
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
            df.loc[fam_idx, resid_col] = df.loc[fam_idx, col] - fam_mean
            df.loc[fam_idx, resid_missing_col] = 0

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

    # Join on gene symbol (case-sensitive exact match)
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

    # Guard: if join matched nothing, all scores are NaN — median would be NaN and
    # fillna would be a no-op, silently writing an all-NaN matrix.
    for col in feature_cols:
        if result[col].notna().sum() == 0:
            raise ValueError(
                f"Column '{col}' has no observed values after join — "
                "the gene-symbol join likely failed (column rename drift?). "
                "Refusing to fabricate imputation values."
            )

    # Missingness indicators — record before imputation
    for col in feature_cols:
        result[f"{col}_missing"] = result[col].isna().astype(float)

    # observed_mask: genes with real scores (used for family mean computation)
    observed_mask = result["pDN_missing"] == 0

    # Family-mean-centred residuals — computed before imputation so family means
    # are never contaminated by imputed values
    result = compute_family_residuals(result, pfam, feature_cols, observed_mask)

    # Median impute raw scores after residuals are computed
    for col in feature_cols:
        median_val = result[col].median()
        result[col] = result[col].fillna(median_val)
        print(
            f"  {col}: median={median_val:.4f}, imputed {result[f'{col}_missing'].sum():.0f} genes"
        )

    # Save TSV
    result.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\nSaved TSV: {OUT_TSV} ({result.shape})")

    # Build aligned numpy matrix (same row order as gene_universe.tsv)
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
    print(f"Genes imputed:          {len(missing_genes)}")
    print(f"Matrix shape:           {mat.shape}")
    print(f"Columns:                {numeric_cols}")


if __name__ == "__main__":
    main()
