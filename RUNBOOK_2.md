# Fetch Pipeline Runbook

## Running the fetch pipeline

```
python -m esm2_mechanism.fetch_data.run_fetch_pipeline
```

Resumes automatically from the step after the last recorded success (`data/.pipeline_state.json`).
Exits immediately if any step fails; re-run the same command to retry from that step.

| Flag | Effect |
|---|---|
| `--from-step N` | Force start at step N, ignoring saved state |
| `--pathogenic-only` | Step 4: restrict ClinVar variants to pathogenic only |
| `--from-scratch` | Steps 5, 7: ignore existing cache and re-fetch |

---

### Prerequisites — manually placed files

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

### Step reference

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

### Running steps individually

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

### Re-running from scratch

Delete `data/.pipeline_state.json` and the generated files in `data/`, then run:

```
python -m esm2_mechanism.fetch_data.run_fetch_pipeline
```

---

## Extracting embeddings

Run after the fetch pipeline completes (see above). Requires a GPU and the `esm` package.

```
python -m esm2_mechanism.embeddings.embed_variants \
    --data_dir data \
    --model esm2_t33_650M_UR50D \
    --batch_size 32
```

Reads `data/merged_variants.json` and `data/cache/sequences.json`. Fetches any UniProt sequences not yet in the cache (incremental — safe to re-run). Outputs are written to `data/embeddings/<model>/`:

| File | Description |
|---|---|
| `embeddings_wt_mean.npy` | (N, D) mean-pooled WT embeddings |
| `embeddings_mut_mean.npy` | (N, D) mean-pooled mutant embeddings |
| `embeddings_wt_pos.npy` | (N, D) per-residue WT embedding at variant position |
| `embeddings_mut_pos.npy` | (N, D) per-residue mutant embedding at variant position |
| `valid_variants.json` | Filtered variant list aligned with the arrays (rows in same order) |

If all five output files exist and `valid_variants.json` covers the same number of variants as the current `merged_variants.json` after filtering, the step is skipped automatically.

**Model options**

| `--model` | Parameters | Notes |
|---|---|---|
| `esm2_t33_650M_UR50D` | 650M | Default |
| `esm2_t36_3B_UR50D` | 3B | Requires more GPU memory |

**prerequisite** — `alphamissense_scores_full.json` must be present before running the analysis scripts (not the embedding step itself). Run embedding first, then step 11, then analysis.

---

## Analysis

Run after embeddings and step 11 (AlphaMissense) are complete. No GPU required — all scripts load cached `.npy` files.

The analysis scripts expect `--out_dir` to be the same run directory used for embeddings, with `data/` as a subdirectory. `--data_dir` (where used) should point to `<out_dir>/data`.

### Primary mechanism probe

Fits stability subspace, runs linear probes (gene-split and family-split CV), baselines, negative controls, and probe direction orthogonality analysis.

```
python -m esm2_mechanism.embeddings.esm2_mechanism \
    --out_dir run_0 \
    --model esm2_t33_650M_UR50D \
    --seeds 0 1 2 3 4
```

Reads from `run_0/data/`:

| Input | Source |
|---|---|
| `merged_variants.json` | fetch pipeline step 4 |
| `embeddings/<model>/embeddings_wt_mean.npy` | embed_variants |
| `embeddings/<model>/embeddings_mut_mean.npy` | embed_variants |
| `embeddings/<model>/embeddings_wt_pos.npy` | embed_variants |
| `embeddings/<model>/embeddings_mut_pos.npy` | embed_variants |
| `embeddings/<model>/valid_variants.json` | embed_variants |
| `alphamissense_scores_full.json` | fetch pipeline step 11 |

Writes to `run_0/`:

| Output | Description |
|---|---|
| `final_info_seed<N>.json` | Per-seed headline metrics |
| `detailed_results_seed<N>.json` | Full CV fold results |
| `final_info.json` | Mean ± stderr across seeds |

### Family split baselines

Runs gene-family-split CV for WT-only, mutant-only, delta, one-hot, FoldX and AlphaMissense baselines.

```
python -m esm2_mechanism.mechanism.family_split_baselines \
    --run_dir run_0 \
    --model esm2_t33_650M_UR50D \
    --seed 0
```

### Family clustering diagnostic

Tests whether embeddings cluster by Pfam family (leakage check).

```
python -m esm2_mechanism.mechanism.family_clustering \
    --run_dir run_0 \
    --model esm2_t33_650M_UR50D \
    --out run_0/family_clustering.json
```

### MLP probe

```
python -m esm2_mechanism.mechanism.experiment_mlp \
    --data_dir run_0/data \
    --emb_dir run_0/data/embeddings/esm2_t33_650M_UR50D \
    --out_dir run_0 \
    --model esm2_t33_650M_UR50D \
    --seed 0
```

### Contrastive mechanism

```
python -m esm2_mechanism.mechanism.contrastive_mechanism \
    --data_dir run_0/data \
    --emb_dir run_0/data/embeddings/esm2_t33_650M_UR50D \
    --out_dir run_0
```

---

### Scripts with stale embedding paths (not yet updated)

`multiseed_v1.py` still references the old flat `data/embeddings/embeddings_wt_esm2_t33_650M_UR50D.npy` naming convention rather than the new `data/embeddings/<model>/embeddings_wt_mean.npy` layout. Do not run it until that is fixed.
