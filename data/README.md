# data/

## Layout

```
data/
  raw/        — input files (variant lists, gene lists, scores, sequences)
  embeddings/ — ESM-2 .npy arrays (gitignored)
```

`raw/` and `embeddings/` are both gitignored. Transfer to/from RunPod with `scp` — see `RUN_EXPERIMENTS.md`.

---

## raw/

| File | Description |
|---|---|
| `DiseaseMech_Stability_VEPS.xlsx` | Gerasimavicius et al. 2022 primary dataset (233MB). Sheet: `ClinVar_gene_level`. |
| `gerasimavicius_variants.json` | Parsed variant list from Gerasimavicius (10,233 variants, 948 genes) |
| `gerasimavicius_gene_list.tsv` | Gene list with mechanism labels and variant counts |
| `AllG2P.csv` | G2P bulk download — `molecular mechanism` field used for merged dataset |
| `merged_gene_list.tsv` | 2,424 genes merging Gerasimavicius + G2P (Gerasimavicius takes priority) |
| `merged_variants.json` | Merged variant list (built by `scripts/build_merged_dataset.py`) |
| `merged_valid_variants.json` | Filtered merged variants aligned with embeddings |
| `natcom_gene_list.tsv` | Nature Communications gene list |
| `clinvar_variants.tsv` | ClinVar pathogenic missense variants for G2P-only genes (fetched by `scripts/fetch_clinvar_variants.py`) |
| `clinvar_pathogenicity_variants.json` | ClinVar pathogenic/benign variants for positive control (result 6) |
| `sequences.json` | UniProt sequence cache keyed by UniProt ID |
| `pfam_families.json` | Pfam family annotations keyed by gene name |
| `alphamissense_scores.json` | AlphaMissense per-variant scores (fetched via MyVariant.info) |
| `pathogenicity_control.json` | Pathogenicity control variant metadata |
| `emb_meta_pathogenicity_esm2_t33_650M_UR50D_n17259.json` | Embedding metadata for pathogenicity set |

---

## embeddings/

All `.npy` files are gitignored. Shape is `(N, 1280)` for ESM-2 650M.

| File | Description | N |
|---|---|---|
| `embeddings_wt_esm2_t33_650M_UR50D.npy` | WT mean-pooled, Gerasimavicius | ~10k |
| `embeddings_mut_esm2_t33_650M_UR50D.npy` | Mutant mean-pooled, Gerasimavicius | ~10k |
| `embeddings_wt_pos_esm2_t33_650M_UR50D.npy` | WT per-residue at variant position, Gerasimavicius | ~10k |
| `embeddings_mut_pos_esm2_t33_650M_UR50D.npy` | Mutant per-residue at variant position, Gerasimavicius | ~10k |
| `emb_wt_mean_pathogenicity_esm2_t33_650M_UR50D_n17259.npy` | WT mean-pooled, ClinVar pathogenicity set | 17,259 |
| `emb_mut_mean_pathogenicity_esm2_t33_650M_UR50D_n17259.npy` | Mutant mean-pooled, ClinVar pathogenicity set | 17,259 |
| `merged_embeddings_wt_mean.npy` | WT mean-pooled, merged dataset | ~19k |
| `merged_embeddings_mut_mean.npy` | Mutant mean-pooled, merged dataset | ~19k |
| `merged_embeddings_wt_pos.npy` | WT per-residue, merged dataset | ~19k |
| `merged_embeddings_mut_pos.npy` | Mutant per-residue, merged dataset | ~19k |
| `merged_valid_variants.json` | Variant metadata aligned with merged embeddings | — |

---

## Transferring to/from RunPod

```bash
# Push raw data to pod
scp -r -i ~/.ssh/id_runpod_2 -P <PORT> \
  data/raw/ root@<POD_IP>:/workspace/esm2_mechanism/data/raw/

# Pull embeddings after extraction
scp -r -i ~/.ssh/id_runpod_2 -P <PORT> \
  root@<POD_IP>:/workspace/esm2_mechanism/data/embeddings/ \
  data/embeddings/
```
