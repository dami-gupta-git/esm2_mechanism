# Fetch Pipeline Runbook

## Running the pipeline

```
python -m esm2_mechanism.fetch_data.run_pipeline
```

Resumes automatically from the step after the last recorded success (`data/.pipeline_state.json`).
Exits immediately if any step fails; re-run the same command to retry from that step.

| Flag | Effect |
|---|---|
| `--from-step N` | Force start at step N, ignoring saved state |
| `--pathogenic-only` | Step 4: restrict ClinVar variants to pathogenic only |
| `--from-scratch` | Steps 5, 7: ignore existing cache and re-fetch |

---

## Prerequisites — manually placed files

These must be in `data/downloads/` before the pipeline starts. None are fetched automatically.

| File | Source |
|---|---|
| `DiseaseMech_Stability_VEPS.xlsx` | Gerasimavicius et al. 2022 — OSF [10.17605/OSF.IO/H62FQ](https://osf.io/rct6d/download) |
| `AllG2P.csv` | G2P bulk download — gene2phenotype.org |
| `table_S3.xlsx` | Badonyi & Marsh 2024 — OSF [osf.io/download/7bftj/](https://osf.io/download/7bftj/) |
| `9606-WHOLE_ORGANISM-integrated.txt` | PaxDb v5.0 — pax-db.org (requires account) |
| `s_het_estimates.genebayes.tsv` | Zeng et al. 2023 — Zenodo [10.5281/zenodo.7939767](https://doi.org/10.5281/zenodo.7939767) |
| `gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz` | gnomAD release 2.1.1 |

---

## Step reference

| Step | Description | Inputs | Outputs |
|---|---|---|---|
| 1 | Build merged gene list | `downloads/DiseaseMech_Stability_VEPS.xlsx`<br>`downloads/AllG2P.csv` | `merged_gene_list.tsv` |
| 2 | Fetch Gerasimavicius variants | `downloads/DiseaseMech_Stability_VEPS.xlsx` | `gerasimavicius_variants.json` |
| 3 | Fetch ClinVar variants | `merged_gene_list.tsv` | `clinvar_variants.tsv` |
| 4 | Merge variant datasets | `gerasimavicius_variants.json`<br>`merged_gene_list.tsv`<br>`clinvar_variants.tsv` | `merged_variants.json` |
| 5 | Fetch Pfam families | `merged_variants.json` | `pfam_families.json` |
| 6 | Build gene universe | `merged_gene_list.tsv`<br>`pfam_families.json` | `gene_universe.tsv` |
| 7 | Fetch UniProt sequences | `merged_variants.json` | `cache/uniprot_sequences_extended.json` |
| 8 | Fetch enzyme labels | `merged_variants.json`<br>`merged_gene_list.tsv` | `enzyme_labels.tsv` |
| 9 | Build proteome feature matrix | `gene_universe.tsv`<br>`downloads/9606-WHOLE_ORGANISM-integrated.txt`<br>`downloads/s_het_estimates.genebayes.tsv`<br>`downloads/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz` | `gene_proteome_features.tsv`<br>`proteome_features_aligned.npy`<br>`proteome_feature_columns.json` |
| 10 | Build Badonyi feature matrix | `downloads/table_S3.xlsx`<br>`gene_universe.tsv` | `badonyi_features.tsv`<br>`badonyi_features_aligned.npy`<br>`badonyi_feature_columns.json` |
| 11 | Fetch AlphaMissense scores ¹ | `merged_valid_variants.json`<br>`pathogenicity_valid_variants.json` | `alphamissense_scores_full.json` |

All paths are relative to `data/`. Steps 3, 5, 7, 8 are resume-safe (already-fetched entries are skipped).

¹ Step 11 requires files produced by downstream embedding scripts, not the pipeline itself.
Downloads the ~5 GB AlphaMissense bulk file on first run; subsequent runs reuse the cache.

---

## Running steps individually

```
python -m esm2_mechanism.fetch_data.build_gene_universe --step gene-list
python -m esm2_mechanism.fetch_data.fetch_variants --step gerasimavicius
python -m esm2_mechanism.fetch_data.fetch_variants --step clinvar
python -m esm2_mechanism.fetch_data.fetch_variants --step merge [--pathogenic_only]
python -m esm2_mechanism.fetch_data.fetch_annotations --step pfam [--from-scratch]
python -m esm2_mechanism.fetch_data.build_gene_universe --step universe
python -m esm2_mechanism.fetch_data.fetch_annotations --step uniprot [--from-scratch]
python -m esm2_mechanism.fetch_data.fetch_annotations --step enzyme
python -m esm2_mechanism.fetch_data.build_proteome_features [--force-redownload]
python -m esm2_mechanism.fetch_data.build_badonyi_features
python -m esm2_mechanism.fetch_data.fetch_annotations --step alphamissense
```

---

## Re-running from scratch

Delete `data/.pipeline_state.json` and the generated files in `data/`, then run:

```
python -m esm2_mechanism.fetch_data.run_pipeline
```
