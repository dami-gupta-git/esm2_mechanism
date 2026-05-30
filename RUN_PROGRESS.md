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
| 1 | A1 | `python -m esm2_mechanism.fetch_data.build_merged_gene_list` | `downloads/DiseaseMech_Stability_VEPS.xlsx`, `downloads/AllG2P.csv` | `gene_list.tsv` | ✅ 2026-05-29 | 2376 genes (gerasimavicius=950, g2p=1426); 475 g2p_disagrees; 61 genes excluded for unresolvable conflicting G2P mechanism |
| 2 | A2 | `python -m esm2_mechanism.fetch_data.fetch_clinvar_variants` | `gene_list.tsv` | `clinvar_variants.tsv` | ✅ 2026-05-29 | 44,866 variants across 2376 genes; gene cache reused, only ~254 new genes fetched |
| 3 | A3 | `python -m esm2_mechanism.fetch_data.fetch_gerasimavicius_variants` | `downloads/DiseaseMech_Stability_VEPS.xlsx` | `gerasimavicius_variants.json` | ✅ 2026-05-29 | 10,233 variants, 948 genes; GOF=1983, DN=894, HI=1678, AR=5678 |
| 4 | A4 | `python -m esm2_mechanism.fetch_data.fetch_pfam_families` | `variants.json` | `pfam_families.json` | ✅ 2026-05-29 | 1883/1916 genes assigned a Pfam family; 33 unannotated |
| 5 | A5 | `python -m esm2_mechanism.fetch_data.build_merged_dataset --pathogenic_only` | `gerasimavicius_variants.json`, `gene_list.tsv`, `clinvar_variants.tsv` | `variants.json` | ✅ 2026-05-29 | 17,268 variants, 1916 genes; sources: gerasimavicius=10233, clinvar_g2p=7035 |
| 6 | A6 | `python -m esm2_mechanism.fetch_data.fetch_clingen` | — | `downloads/ClinGen_gene_curation_list_GRCh38.tsv` | ✅ 2026-05-29 | 1642 data rows; gene=0, HI=5, TS=13 |
| 7 | A7 | `python -m esm2_mechanism.fetch_data.fetch_uniprot_sequences` | `variants.json` | `cache/uniprot_sequences_extended.json` | ✅ 2026-05-29 | 1914/1914 sequences fetched; ~30s |

---

## Run 2 — started 2026-05-29

| # | Step | Command | Inputs | Outputs | Status | Notes |
|---|---|---|---|---|---|---|
| 1 | 1 | `python -m esm2_mechanism.fetch_data.build_gene_universe --step gene-list` | `downloads/DiseaseMech_Stability_VEPS.xlsx`, `downloads/AllG2P.csv` | `gene_list.tsv` | ✅ 2026-05-29 | 2376 genes (gerasimavicius=950, g2p=1426); 475 g2p_disagrees; 61 genes excluded for unresolvable conflicting G2P mechanism |
| 2 | 2 | `python -m esm2_mechanism.fetch_data.fetch_variants --step gerasimavicius` | `downloads/DiseaseMech_Stability_VEPS.xlsx` | `gerasimavicius_variants.json` | ✅ 2026-05-29 | 10,233 variants, 948 genes; AR=5678, GOF=1983, HI=1678, DN=894 |
| 3 | 3 | `python -m esm2_mechanism.fetch_data.fetch_variants --step clinvar` | `gene_list.tsv` | `clinvar_variants.tsv` | ✅ 2026-05-29 | 47,752 variants across 2376 genes |

---

## Run 3 — started 2026-05-29

Uses `run_fetch_pipeline.py` with step 3 (ClinVar) commented out — already completed in Run 2.

| # | Step | Command | Inputs | Outputs | Status | Notes |
|---|---|---|---|---|---|---|
| 1 | 1 | `run_fetch_pipeline` step 1 | `downloads/DiseaseMech_Stability_VEPS.xlsx`, `downloads/AllG2P.csv` | `gene_list.tsv` | ✅ 2026-05-29 | 2376 genes (gerasimavicius=950, g2p=1426); 475 g2p_disagrees; 61 excluded |
| 2 | 2 | `run_fetch_pipeline` step 2 | `downloads/DiseaseMech_Stability_VEPS.xlsx` | `gerasimavicius_variants.json` | ✅ 2026-05-29 | 10,233 variants, 948 genes; AR=5678, GOF=1983, HI=1678, DN=894 |
| 3 | 3 | *(skipped — reused Run 2 output)* | — | `clinvar_variants.tsv` | ✅ 2026-05-29 | |
| 4 | 4 | `run_fetch_pipeline` step 4 | `gerasimavicius_variants.json`, `gene_list.tsv`, `clinvar_variants.tsv` | `variants.json` | ✅ 2026-05-29 | 17,921 variants, 1941 genes; gerasimavicius=10233, clinvar_g2p=7688 |
| 5 | 5 | `run_fetch_pipeline` step 5 | `variants.json` | `pfam_families.json` | ✅ 2026-05-29 | 1908/1941 genes assigned a Pfam family; 33 unannotated |
| 6 | 6 | `run_fetch_pipeline` step 6 | `gene_list.tsv`, `pfam_families.json` | `gene_universe.tsv` | ✅ 2026-05-29 | 1908 genes retained; 468 dropped (no Pfam annotation) |
| 7 | 7 | `run_fetch_pipeline` step 7 | `variants.json` | `cache/uniprot_sequences_extended.json` | ✅ 2026-05-29 | 1939 unique UniProt IDs |
| 8 | 8 | `run_fetch_pipeline` step 8 | `variants.json`, `gene_list.tsv` | `enzyme_labels.tsv` | ✅ 2026-05-30 | 2376 rows; kinase=131, protease=68, oxidoreductase=123, non-enzyme=1619, missing=436 |
| 9 | 9 | `run_fetch_pipeline` step 9 | `gene_universe.tsv` + manual files | `gene_proteome_features.tsv`, `proteome_features_aligned.npy`, `proteome_feature_columns.json` | ✅ 2026-05-30 | 1908 genes × 35 cols; matrix shape (1908, 33) |
| 10 | 10 | `run_fetch_pipeline` step 10 | `downloads/table_S3.xlsx`, `gene_universe.tsv` | `badonyi_features.tsv`, `badonyi_features_aligned.npy`, `badonyi_feature_columns.json` | ✅ 2026-05-30 | 1900/1908 genes covered (99.6%); 8 missing |
