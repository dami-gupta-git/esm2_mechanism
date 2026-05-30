# Run runbook — esm2_mech package

Purpose: run the full pipeline using the refactored `esm2_mech` package after the
2026-05 restructuring. All commands use `python -m esm2_mech.<module>` from the
project root with the package installed (`pip install -e .`). The old
`esm2_mechanism` package is now deleted; all scripts live under `src/esm2_mech/`.

**GPU vs CPU:** only the embedding extraction steps need a GPU (RunPod A100/H100).
Everything else is local CPU. Run GPU steps inside a `tmux` session on RunPod.

---

## Prerequisites — manually placed files

These must be in `data/downloads/` before the pipeline starts.

| File | Source |
|---|---|
| `DiseaseMech_Stability_VEPS.xlsx` | Gerasimavicius et al. 2022 — OSF [10.17605/OSF.IO/H62FQ](https://osf.io/rct6d/download) |
| `AllG2P.csv` | G2P bulk download — gene2phenotype.org |
| `table_S3.xlsx` | Badonyi & Marsh 2024 — OSF [osf.io/download/7bftj/](https://osf.io/download/7bftj/) |
| `9606-WHOLE_ORGANISM-integrated.txt` | PaxDb v5.0 — pax-db.org (requires account) |
| `s_het_estimates.genebayes.tsv` | Zeng et al. 2023 — Zenodo [10.5281/zenodo.7939767](https://doi.org/10.5281/zenodo.7939767) |
| `gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz` | gnomAD release 2.1.1 |

---

## Stage 0 — environment setup

```bash
cd /Users/dgupta/code/portfolio/ESM2/esm2_mechanism
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

---

## Stage 1 — build gene list (CPU)

Shared foundation for all experiments. Run once before any experiment.

**Inputs (manually placed):**
- `data/downloads/DiseaseMech_Stability_VEPS.xlsx`
- `data/downloads/AllG2P.csv`

```bash
python -m esm2_mech.fetch_data.build_gene_list
```

**Output:** `data/gene_list.tsv`

---

## Experiment 1 — ESM-2 delta-embedding mechanism geometry (results 1–2)

### Step 1 — fetch data (CPU)

| Step | Command | Description | Inputs | Outputs |
|---|---|---|---|---|
| 2 | `python -m esm2_mech.fetch_data.fetch_variants --step gerasimavicius` | Fetch Gerasimavicius variants | `downloads/DiseaseMech_Stability_VEPS.xlsx` | `gerasimavicius_variants.json` |
| 3 | `python -m esm2_mech.fetch_data.fetch_variants --step clinvar` | Fetch ClinVar variants | `gene_list.tsv` | `clinvar_variants.tsv` |
| 4 | `python -m esm2_mech.fetch_data.fetch_variants --step merge` | Merge variant datasets | `gerasimavicius_variants.json`, `gene_list.tsv`, `clinvar_variants.tsv` | `variants.json` |
| 5 | `python -m esm2_mech.fetch_data.fetch_sequences` | Fetch UniProt sequences | `variants.json` | `cache/sequences.json` |
| 6 | `python -m esm2_mech.fetch_data.fetch_annotations --step pfam` | Fetch Pfam families | `variants.json` | `pfam_families.json` |

### Step 2 — embed variants (GPU, RunPod)

| Command | Description | Inputs | Outputs |
|---|---|---|---|
| `python -m esm2_mech.embeddings.embed_variants --data_dir data --model esm2_t33_650M_UR50D --batch_size 32` | Extract ESM-2 embeddings | `variants.json`, `cache/sequences.json` | `embeddings_wt_mean.npy`, `embeddings_mut_mean.npy`, `embeddings_wt_pos.npy`, `embeddings_mut_pos.npy`, `valid_variants.json` |

After completion, `scp` the `.npy` files and `valid_variants.json` back to `data/embeddings/esm2_t33_650M_UR50D/` locally.

---

## Verification checklist

- [ ] `data/embeddings/esm2_t33_650M_UR50D/valid_variants.json` row count matches all four `.npy` arrays.
- [ ] `data/pfam_families.json` has entries for ≥ 1,900 genes (< 1,900 suggests a partial fetch).
- [ ] `data/enzyme_labels.tsv` spot-checked against UniProt EC numbers for a handful of kinases and proteases.
- [ ] `data/alphamissense_scores_full.json` non-empty and covers > 90% of `valid_variants.json`.
- [ ] result 6 (pathogenicity AUROC ~0.88) and result 7 (mechanism family-split floor ~0.35–0.39) reproduce their headline numbers — these are the pipeline spine and should not move.
- [ ] result 26: ESM-2 and ESM-3 scored under identical fold rule before reporting M1/M2 gap (see Stage E caveat).
- [ ] `git status` clean; results committed.
