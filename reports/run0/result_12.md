# Result 12 — Phase 1+2: Gene-level proteome feature matrix assembled
## Date: May 26, 2026 | No model run | Local CPU | Scripts: build_proteome_features.py

---

## TL;DR

Six public data sources were downloaded, parsed, and integrated into a single gene × feature matrix covering all 2,424 unique genes in the merged dataset. Final output: a **2,424 × 37 float32 matrix** (`data/proteome_features_aligned.npy`) aligned to the gene order in `merged_gene_list.tsv`, plus a human-readable TSV and column metadata JSON. Coverage ranged from 19% (ClinGen HI) to 100% (paralogs); all missing values are handled by binary missingness indicators. Family-mean-centred residuals are pre-computed for all continuous features as a built-in anti-leakage measure. One source (HPA n_tissues) was not recoverable from public endpoints and is excluded; PaxDb was provided as a manual download (98.4% coverage).

---

## What we did

This is the data collection phase (Phases 1 and 2) preceding the modelling gate (Phase 3). No model was trained here.

**Goal:** build a gene-level feature table covering the biological context information hypothesised to carry the DN and haploinsufficiency signal that ESM-2 delta embeddings lack.

**Feature sources attempted:**

| Source | Features | URL / method |
|---|---|---|
| gnomAD v4.1 | pLI, LOEUF, mis_z | Bulk TSV (95 MB), MANE-select transcript preference |
| Ensembl Compara | paralog_count | Per-gene REST API, 10 req/s, resume-safe cache |
| Human Protein Atlas | tissue_specificity_tau | Bulk ZIP (proteinatlas.tsv); text label mapped to [0, 0.8] |
| PaxDb v5.0 | log_abundance_ppm | Manually downloaded (98.4% coverage) |
| BioPlex 3.0 | PPI_degree | 293T 10K network TSV (13 MB) |
| ClinGen dosage | HI_score, TS_score | FTP TSV (0.3 MB); no-header file, positional parsing |

gnomAD and paralogs were already cached from the Stage 0 pilot (result_11).

---

## Coverage

| Feature | Covered | / 2424 | % |
|---|---|---|---|
| pLI | 2257 | 2424 | 93.1% |
| LOEUF | 2257 | 2424 | 93.1% |
| mis_z | 2260 | 2424 | 93.2% |
| paralog_count | 2424 | 2424 | 100.0% |
| tissue_specificity_tau | 2405 | 2424 | 99.2% |
| n_tissues_expressed | 0 | 2424 | 0.0% |
| log_abundance_ppm | 2385 | 2424 | 98.4% |
| PPI_degree | 1819 | 2424 | 75.0% |
| HI_score | 465 | 2424 | 19.2% |
| TS_score | 893 | 2424 | 36.8% |

n_tissues_expressed (0% coverage) is excluded from the final matrix. The final matrix is 2,424 × 37 (8 continuous features × 4 derived columns each + 1 singleton indicator).

---

## Engineering decisions (Phase 2)

### Missing-data policy
For every continuous feature, a paired binary `<feature>_missing` indicator is appended. Missing values in the `.npy` matrix are median-imputed column-wise. Raw NaN values are preserved in the TSV.

### Family-mean-centred residuals
For every continuous feature, a `<feature>_familyresid` column is computed: each gene's value minus the mean of genes in its protein family. Singleton families receive residual=0 and a `is_singleton_family=1` indicator. This is the primary mitigation for family-level leakage: features like PPI_degree and constraint scores are correlated within protein families, and a model trained on absolute values could learn family identity rather than per-gene variation. Family-centred residuals force the model to use within-family deviations — the signal that should survive family-split CV — rather than cross-family absolute differences.

### Final matrix dimensions
- **2,424 genes** × **41 numerical columns**
- Columns: 10 continuous features + 10 missing indicators + 10 family residuals + 10 family-residual missing indicators + 1 singleton indicator
- dtype: float32

---

## Source-specific notes

### gnomAD v4.1
No issues. The pilot's cached 95 MB TSV was reused. Column schema confirmed: gene=col 0, pLI=col 18, LOEUF=col 22, mis_z=col 35, MANE-select=col 4. Selection logic: MANE-select transcript preferred; if absent, transcript with highest expected LoF count.

### Paralogs
All 2,424 genes were in cache from the pilot (1,260 genes) or the extended REST fetch in this phase (1,164 additional genes). Zero new REST calls needed on re-run. 100% coverage.

### Human Protein Atlas
The expected download URLs returned 404. The current bulk download is `proteinatlas.tsv.zip` (6.7 MB compressed). Column `RNA tissue specificity` (text label) was mapped to a continuous score: enriched→0.8, group→0.7, enhanced→0.6, low→0.2, not detected→0.0. 99.2% coverage. The `n_tissues_expressed` feature is not available from the bulk export; treated as fully-missing.

### PaxDb
URL `https://pax-db.org/download/5.0/datasets/9606/9606-WHOLE_ORGANISM-integrated.txt` returned HTTP 403. The site now requires account authentication. Treated as fully-missing in the automated pipeline; file placed manually at `data/9606-WHOLE_ORGANISM-integrated.txt`. 98.4% coverage.

### BioPlex 3.0
Downloaded successfully (293T 10K Dec 2019 TSV, 12.9 MB). The column names are quoted in the file — matched after stripping quotes via `csv.DictReader`. Degree = number of unique interaction partners. 75% coverage.

### ClinGen dosage
Downloaded from FTP (0.3 MB). File has no header row — all lines after comment block (`#`) are data. Text labels mapped to integer scores: "Sufficient evidence for dosage pathogenicity"→3, "Some evidence"→2, "Emerging/little evidence"→1, "No evidence available"→0. Low coverage expected and correct (19% HI, 37% TS) — ClinGen only curates a subset of clinically relevant genes; missingness carries information (uncurated genes effectively have "no strong evidence").

---

## What this does not include

- **Mathieson 2018 protein half-life** — supplementary table from a 2018 *Nat Comms* paper; not included in this automated pull. ~60% coverage expected.
- **PhosphoSitePlus PTM density** — requires registration. Not pursued.
- **PaxDb abundance** — recovered via manual download (98.4% coverage). Automated download blocked.

---

## Outputs

| File | Description |
|---|---|
| `data/gene_proteome_features.tsv` | Human-readable 2,424 × 43 table (gene + pfam_family + 41 numerical cols) |
| `data/proteome_features_aligned.npy` | float32 matrix (2,424, 41), median-imputed, aligned to merged_gene_list.tsv gene order |
| `data/proteome_feature_columns.json` | Column names, metadata, notes on imputation and scaling |
| `data/cache/proteome_features/` | Raw downloaded files (gnomAD reuses pilot cache) |

---

## Next step

Phase 3 modelling (Gate 1 / Stage 1): run V2 (proteome features only, full feature set) under 5-fold family-split CV with 5 seeds. Gate criterion: V2 macro-F1 ≥ 0.35 → proceed to V3 (ESM-2 delta + proteome concatenated). The pilot (result_11) established that the 4-feature subset already reaches 0.417, so the full 41-column matrix is expected to comfortably pass Gate 1 — but the gate is still required to confirm the added features don't hurt.

---

## Plain-English summary

This result is about collecting data, not training a model. Before we can test whether combining protein-level biology with ESM-2 sequence features helps predict disease mechanism, we needed to assemble a table of gene-level properties for all 2,424 genes in our dataset.

Think of it like building a spreadsheet where each row is one gene and each column is a different piece of information about that gene:

- **How tolerant is the human population to mutations in this gene?** (gnomAD constraint — three columns)
- **How many close relatives does it have in the human genome?** (paralog count — one column)
- **How tissue-specific is its expression?** (Human Protein Atlas — one column; a score close to 1 means the gene is only active in a few tissues, 0 means it's active everywhere)
- **How many protein interaction partners does it have?** (BioPlex — one column; genes at the hub of large protein complexes are more likely to cause dominant disease by disrupting the whole complex)
- **Has a clinical genetics database formally classified this gene for dosage sensitivity?** (ClinGen — two columns; rare, but high-value signal for haploinsufficiency)

We tried two additional data sources (protein abundance from PaxDb, tissue count from HPA) but couldn't download them automatically — one is behind a login wall, one changed its format. Both are included in the matrix as blank columns so that the model can still learn from the fact that the value is missing.

For each measured number, we also computed how much that gene differs from the average gene in the same protein family. This "family-centred" version is the scientifically safer input: it forces the model to look at within-family variation rather than exploiting the fact that, say, all kinases have similar constraint scores.

The result is a matrix that will be fed into Phase 3 alongside the ESM-2 sequence embeddings to test whether the combination predicts mechanism better than either alone.
