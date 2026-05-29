# Run progress

## Preloaded files (data/downloads/)

These files were manually downloaded before the pipeline ran — not produced by any script:

| File | What it is | Source |
|---|---|---|
| `DiseaseMech_Stability_VEPS.xlsx` | Variant-level FoldX ddG + disease mechanism labels | Gerasimavicius et al. 2022, *Nature Communications* 13:3895 — OSF [10.17605/OSF.IO/H62FQ](https://osf.io/rct6d/download) |
| `AllG2P.csv` | Gene-disease-mechanism database; `molecular mechanism` field used for merging | G2P bulk download — gene2phenotype.org |
| `table_S3.xlsx` | Per-gene SVM scores (pDN, pGOF, pLOF) for 20,365 human proteins | Badonyi & Marsh 2024, *PLOS One* — OSF [osf.io/download/7bftj/](https://osf.io/download/7bftj/) |
| `9606-WHOLE_ORGANISM-integrated.txt` | Human proteome abundance (PaxDb v5.0) | PaxDb — pax-db.org (requires account; HTTP 403 without auth) |
| `natcom_gene_list.tsv` | Gene list from the Nature Communications paper | Gerasimavicius et al. 2022 (same paper as xlsx) |
| `megascale/` | ~800k mutations across ~500 small domains with experimental ΔG values | Tsuboyama et al. 2023 — Zenodo [zenodo.org/records/7844779](https://zenodo.org/records/7844779) |

---

## Run 1 — started 2026-05-29

| # | Stage | Command | Inputs | Outputs | Status | Notes |
|---|---|---|---|---|---|---|
| 1 | A1 | `python -m esm2_mechanism.fetch_data.build_merged_gene_list` | `downloads/DiseaseMech_Stability_VEPS.xlsx`, `downloads/AllG2P.csv` | `merged_gene_list.tsv` | ✅ 2026-05-29 | 2376 genes (gerasimavicius=950, g2p=1426); 475 g2p_disagrees; 61 genes excluded for unresolvable conflicting G2P mechanism |
| 2 | A2 | `python -m esm2_mechanism.fetch_data.fetch_clinvar_variants` | `merged_gene_list.tsv` | `clinvar_variants.tsv` | ✅ 2026-05-29 | 44,866 variants across 2376 genes; gene cache reused, only ~254 new genes fetched |
| 3 | A3 | `python -m esm2_mechanism.fetch_data.fetch_gerasimavicius_variants` | `downloads/DiseaseMech_Stability_VEPS.xlsx` | `gerasimavicius_variants.json` | ✅ 2026-05-29 | 10,233 variants, 948 genes; GOF=1983, DN=894, HI=1678, AR=5678 |
| 4 | A4 | `python -m esm2_mechanism.fetch_data.build_merged_dataset --pathogenic_only` | `gerasimavicius_variants.json`, `merged_gene_list.tsv`, `clinvar_variants.tsv` | `merged_variants.json` | ✅ 2026-05-29 | 17,268 variants, 1916 genes; sources: gerasimavicius=10233, clinvar_g2p=7035 |
