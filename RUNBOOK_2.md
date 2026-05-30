# Fetch Pipeline Runbook

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
| `--from-scratch` | Steps 6, 8: ignore existing cache and re-fetch |

---

### Step reference

| Step | Description | Inputs | Outputs |
|---|---|---|---|
| 1 | Build merged gene list | `downloads/DiseaseMech_Stability_VEPS.xlsx`<br>`downloads/AllG2P.csv` | `gene_list.tsv` |
| 2 | Fetch Gerasimavicius variants | `downloads/DiseaseMech_Stability_VEPS.xlsx` | `gerasimavicius_variants.json` |
| 3 | Fetch ClinVar variants | `gene_list.tsv` | `clinvar_variants.tsv` |
| 4 | Merge variant datasets | `gerasimavicius_variants.json`<br>`gene_list.tsv`<br>`clinvar_variants.tsv` | `variants.json` |
| 5 | Fetch UniProt sequences | `variants.json` | `cache/sequences.json` |
| 6 | Fetch Pfam families | `variants.json` | `pfam_families.json` |
| 7 | Build gene universe | `gene_list.tsv`<br>`pfam_families.json` | `gene_universe.tsv` |
| 8 | Fetch UniProt sequences (extended) | `variants.json` | `cache/uniprot_sequences_extended.json` |
| 9 | Fetch enzyme labels | `variants.json`<br>`gene_list.tsv` | `enzyme_labels.tsv` |
| 10 | Build proteome feature matrix | `gene_universe.tsv`<br>`downloads/9606-WHOLE_ORGANISM-integrated.txt`<br>`downloads/s_het_estimates.genebayes.tsv`<br>`downloads/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz` | `gene_proteome_features.tsv`<br>`proteome_features_aligned.npy`<br>`proteome_feature_columns.json` |
| 11 | Build Badonyi feature matrix | `downloads/table_S3.xlsx`<br>`gene_universe.tsv` | `badonyi_features.tsv`<br>`badonyi_features_aligned.npy`<br>`badonyi_feature_columns.json` |

All paths are relative to `data/`. Steps 3, 5, 6, 8, 9 are resume-safe (already-fetched entries are skipped).


---

### Running steps individually

```
python -m esm2_mechanism.fetch_data.build_gene_universe --step gene-list
python -m esm2_mechanism.fetch_data.fetch_variants --step gerasimavicius
python -m esm2_mechanism.fetch_data.fetch_variants --step clinvar
python -m esm2_mechanism.fetch_data.fetch_variants --step merge [--pathogenic_only]
python -m esm2_mechanism.fetch_data.fetch_sequences
python -m esm2_mechanism.fetch_data.fetch_annotations --step pfam [--from-scratch]
python -m esm2_mechanism.fetch_data.build_gene_universe --step universe
python -m esm2_mechanism.fetch_data.fetch_annotations --step uniprot [--from-scratch]
python -m esm2_mechanism.fetch_data.fetch_annotations --step enzyme
python -m esm2_mechanism.fetch_data.build_proteome_features [--force-redownload]
python -m esm2_mechanism.fetch_data.build_badonyi_features
```

---

### Re-running from scratch

Delete `data/.pipeline_state.json` and the generated files in `data/`, then run:

```
python -m esm2_mechanism.fetch_data.run_fetch_pipeline
```

---

## Extracting embeddings

Run the steps below in order. GPU steps require the `esm` package and a CUDA device.

### 1. Mechanism embeddings (GPU)

```
python -m esm2_mechanism.embeddings.embed_variants \
    --data_dir data \
    --model esm2_t33_650M_UR50D \
    --batch_size 32
```

Reads `data/variants.json` and `data/cache/sequences.json`. Outputs are written to `data/embeddings/<model>/`:

| File | Description |
|---|---|
| `embeddings_wt_mean.npy` | (N, D) mean-pooled WT embeddings |
| `embeddings_mut_mean.npy` | (N, D) mean-pooled mutant embeddings |
| `embeddings_wt_pos.npy` | (N, D) per-residue WT embedding at variant position |
| `embeddings_mut_pos.npy` | (N, D) per-residue mutant embedding at variant position |
| `valid_variants.json` | Filtered variant list aligned with the arrays (rows in same order) |

If all five output files exist and row counts match, the step is skipped automatically.

**Model options**

| `--model` | Parameters | Notes |
|---|---|---|
| `esm2_t33_650M_UR50D` | 650M | Default |
| `esm2_t36_3B_UR50D` | 3B | Requires more GPU memory |

---

### 2. Perturbation scan embeddings (phase 1: CPU, GPU, phase 3: CPU)

In-silico perturbation scan: mutates 100 evenly-spaced positions per gene to 3 probe amino acids (Ala, Asp, Trp) and extracts ESM-2 650M delta embeddings (~600k forward passes, ~3h on A100).

```
# Phase 1 — CPU: build probe variant list
python -m esm2_mechanism.perturb.perturbation_scan --run_phase 1

# GPU: extract embeddings (~600k forward passes, ~3h on A100)
python -m esm2_mechanism.embeddings.embed_scan --batch_size 128

# Phase 3 — CPU: compute scan features from cached embeddings
python -m esm2_mechanism.perturb.perturbation_scan --run_phase 3
```

Outputs `data/scan_features.npy` with 5 pre-registered scalar features per gene.

---

### 3. ESM-3 embeddings (optional; phase 1: CPU, phase 2: GPU)

Tests whether ESM-3 structure tokens rescue the mechanism null from ESM-2. Two conditions: sequence-only (`seq`) and sequence + AlphaFold2 structure tokens (`seq_struct`).

```
# Phase 1 — CPU: download AF2 structures
python -m esm2_mechanism.mechanism.esm3_mechanism --phase 1

# Phase 2 — GPU: extract ESM-3 embeddings
python -m esm2_mechanism.mechanism.esm3_mechanism --phase 2
```

---

### 4. Pathogenicity control embeddings (GPU)

Extracts embeddings for ClinVar pathogenic vs benign variants. Produces `pathogenicity_valid_variants.json`, which is required by `fetch_alphamissense` and `score_esm1v`.

```
python -m esm2_mechanism.embeddings.pathogenicity_control --phase 2
```

---

### 5. Fetch AlphaMissense scores (CPU)

Requires `valid_variants.json` (step 1) and `pathogenicity_valid_variants.json` (step 4).

```
python -m esm2_mechanism.fetch_data.fetch_alphamissense
```

Downloads the ~5 GB AlphaMissense bulk file on first run; subsequent runs reuse the cache. Writes `data/alphamissense_scores_full.json`, which is required by the analysis scripts.

---

### 6. ESM-1v scores (GPU)

Requires `pathogenicity_valid_variants.json` (step 4).

```
python -m esm2_mechanism.embeddings.score_esm1v
```

Writes `data/esm1v_scores_full.json`.

---

## Analysis

Run after all embeddings and AlphaMissense scores are complete. No GPU required — all scripts load cached `.npy` files.

The analysis scripts expect `--out_dir` to be the same run directory used for embeddings, with `data/` as a subdirectory. `--data_dir` (where used) should point to `<out_dir>/data`.

### Primary mechanism probe

Fits stability subspace, runs linear probes (gene-split and family-split CV), baselines, negative controls, and probe direction orthogonality analysis.

```
python -m esm2_mechanism.mechanism.esm2_mechanism \
    --out_dir run_0 \
    --model esm2_t33_650M_UR50D \
    --seeds 0 1 2 3 4
```

Reads from `run_0/data/`:

| Input | Source |
|---|---|
| `variants.json` | fetch pipeline step 4 |
| `embeddings/<model>/embeddings_wt_mean.npy` | embed_variants |
| `embeddings/<model>/embeddings_mut_mean.npy` | embed_variants |
| `embeddings/<model>/embeddings_wt_pos.npy` | embed_variants |
| `embeddings/<model>/embeddings_mut_pos.npy` | embed_variants |
| `embeddings/<model>/valid_variants.json` | embed_variants |
| `alphamissense_scores_full.json` | fetch_alphamissense |

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
python -m esm2_mechanism.mechanism.mlp \
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

### Pathogenicity control

Validates the pipeline: runs gene-split and family-split probes for ClinVar pathogenic vs benign. Requires pathogenicity control embeddings from step 4 above.

```
python -m esm2_mechanism.mechanism.pathogenicity_control \
    --run_dir run_0 \
    --model esm2_t33_650M_UR50D
```

### ESM-3 mechanism analysis

Runs the probe comparison for ESM-3 conditions (CPU only). Requires ESM-3 embeddings from step 3 above.

```
python -m esm2_mechanism.mechanism.esm3_mechanism --phase 3
```

### Megascale stability analysis

Positive control for stability encoding (result_21). Runs random-split, protein-holdout, and cluster-split CV on S1724 benchmark. Requires megascale embeddings from `esm2_mechanism.embeddings.embed_variants`. No GPU required.

```
python -m esm2_mechanism.analysis.megascale_stability --run_dir run_0 --model esm2_t33_650M_UR50D
```

### ESM-1v family split analysis

Per-Pfam-family AUROC analysis of ESM-1v ΔLL. Requires `data/esm1v_scores_full.json` from step 6 above.

```
python -m esm2_mechanism.analysis.esm1v_family_split
```

Writes `results/esm1v_family/overall.json`, `per_family.json`, and `summary.json`.

---

### Scripts with stale embedding paths (not yet updated)

`multiseed_v1.py` still references the old flat `data/embeddings/embeddings_wt_esm2_t33_650M_UR50D.npy` naming convention rather than the new `data/embeddings/<model>/embeddings_wt_mean.npy` layout. Do not run it until that is fixed.
