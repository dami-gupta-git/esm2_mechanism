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

## Stage A — fetch data (CPU, network)

Run via the pipeline orchestrator, which resumes from the last completed step
(`data/.pipeline_state.json`). Re-run the same command to retry after a failure.

```bash
python -m esm2_mech.fetch_data.run_fetch_pipeline
```

| Step | Description | Inputs | Outputs |
|---|---|---|---|
| 1 | Build merged gene list | `downloads/DiseaseMech_Stability_VEPS.xlsx`, `downloads/AllG2P.csv` | `gene_list.tsv` |
| 2 | Fetch Gerasimavicius variants | `downloads/DiseaseMech_Stability_VEPS.xlsx` | `gerasimavicius_variants.json` |
| 3 | Fetch ClinVar variants | `gene_list.tsv` | `clinvar_variants.tsv` |
| 4 | Merge variant datasets | `gerasimavicius_variants.json`, `gene_list.tsv`, `clinvar_variants.tsv` | `variants.json` |
| 5 | Fetch UniProt sequences | `variants.json` | `cache/sequences.json` |
| 6 | Fetch Pfam families | `variants.json` | `pfam_families.json` |
| 7 | Build gene universe | `gene_list.tsv`, `pfam_families.json` | `gene_universe.tsv` |
| 8 | Fetch UniProt sequences (extended) | `variants.json` | `cache/uniprot_sequences_extended.json` |
| 9 | Fetch enzyme labels | `variants.json`, `gene_list.tsv` | `enzyme_labels.tsv` |
| 10 | Build proteome feature matrix | `gene_universe.tsv`, `downloads/9606-WHOLE_ORGANISM-integrated.txt`, `downloads/s_het_estimates.genebayes.tsv`, `downloads/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz` | `gene_proteome_features.tsv`, `proteome_features_aligned.npy`, `proteome_feature_columns.json` |
| 11 | Build Badonyi feature matrix | `downloads/table_S3.xlsx`, `gene_universe.tsv` | `badonyi_features.tsv`, `badonyi_features_aligned.npy`, `badonyi_feature_columns.json` |

Steps 3, 5, 6, 8, 9 are resume-safe — already-fetched entries are skipped.

**Running steps individually:**
```bash
python -m esm2_mech.fetch_data.build_gene_list
python -m esm2_mech.fetch_data.fetch_variants --step gerasimavicius
python -m esm2_mech.fetch_data.fetch_variants --step clinvar
python -m esm2_mech.fetch_data.fetch_variants --step merge [--pathogenic_only]
python -m esm2_mech.fetch_data.fetch_sequences
python -m esm2_mech.fetch_data.fetch_annotations --step pfam [--from-scratch]
python -m esm2_mech.fetch_data.build_gene_universe
python -m esm2_mech.fetch_data.fetch_annotations --step uniprot [--from-scratch]
python -m esm2_mech.fetch_data.fetch_annotations --step enzyme
python -m esm2_mech.fetch_data.build_proteome_features
python -m esm2_mech.fetch_data.build_badonyi_features
```

---

## Stage B — embeddings (GPU, RunPod)

Run each step inside a `tmux` session. After completion, `scp` all `.npy` files back
to `data/embeddings/` locally before running Stage C.

### B1 — mechanism embeddings

```bash
python -m esm2_mech.embeddings.embed_variants \
    --data_dir data \
    --model esm2_t33_650M_UR50D \
    --batch_size 32
```

Reads `data/variants.json` and `data/cache/sequences.json`.
Outputs to `data/embeddings/esm2_t33_650M_UR50D/`:

| File | Description |
|---|---|
| `embeddings_wt_mean.npy` | (N, 1280) mean-pooled WT embeddings |
| `embeddings_mut_mean.npy` | (N, 1280) mean-pooled mutant embeddings |
| `embeddings_wt_pos.npy` | (N, 1280) per-residue WT embedding at variant position |
| `embeddings_mut_pos.npy` | (N, 1280) per-residue mutant embedding at variant position |
| `valid_variants.json` | Filtered variant list aligned with the arrays (same row order) |

If all five output files exist and row counts match, the step is skipped automatically.

### B2 — perturbation scan embeddings

```bash
# Phase 1 — CPU: build probe list
python -m esm2_mech.experiments.perturbation.perturbation_scan --run_phase 1

# GPU: extract embeddings (~600k forward passes, ~3h on A100)
python -m esm2_mech.embeddings.embed_scan --batch_size 128

# Phase 3 — CPU: compute scan features from cached embeddings
python -m esm2_mech.experiments.perturbation.perturbation_scan --run_phase 3
```

Outputs `data/scan_features.npy` (1935 genes × 5 features).

### B3 — pathogenicity control

```bash
# Step 1 — CPU: fetch ClinVar variants (cached on re-run)
python -m esm2_mech.experiments.pathogenicity.pathogenicity_fetch --run_dir run_0

# Step 2 — GPU: extract pathogenicity embeddings
python -m esm2_mech.embeddings.embed_pathogenicity \
    --run_dir run_0 --model esm2_t33_650M_UR50D --batch_size 32

# Step 3 — CPU: run probes
python -m esm2_mech.experiments.pathogenicity.pathogenicity_probes --run_dir run_0
```

Produces `data/embeddings/esm2_t33_650M_UR50D/pathogenicity_{wt,mut}_mean.npy` and
`{run_dir}/pathogenicity_control.json` — required by Stage C and `esm1v_family_split`.

### B4 — ESM-3 embeddings (optional)

```bash
# Phase 1 — CPU: download AF2 structures and tokenise
python -m esm2_mech.experiments.esm3.esm3_mechanism --phase 1

# Phase 2 — GPU: extract ESM-3 embeddings
python -m esm2_mech.experiments.esm3.esm3_mechanism --phase 2
```

### B5 — ESM-1v scores

```bash
python -m esm2_mech.embeddings.score_esm1v
```

Requires `data/pathogenicity_valid_variants.json` from B3. Produces
`data/esm1v_scores_full.json`.

---

## Stage C — AlphaMissense scores (CPU, network, large download)

Requires `data/embeddings/esm2_t33_650M_UR50D/valid_variants.json` from B1 and
`data/pathogenicity_valid_variants.json` from B3. Run after both complete.

```bash
python -m esm2_mech.fetch_data.fetch_alphamissense
```

Downloads a ~5 GB AlphaMissense bulk file on first run; subsequent runs reuse the
cache. Produces `data/alphamissense_scores_full.json`.

---

## Stage D — analyses (CPU, local)

All scripts read cached `.npy` files; no GPU required. Run after Stage B `.npy` files
are scp'd back locally. Stage C (`alphamissense_scores_full.json`) must also be
complete before D4 and D6.

### D1 — mechanism core (results 1–10)

```bash
python -m esm2_mech.experiments.mechanism.esm2_mechanism \
    --out_dir run_0 --model esm2_t33_650M_UR50D --seeds 0 1 2 3 4
python -m esm2_mech.experiments.mechanism.family_split_baselines \
    --run_dir run_0 --model esm2_t33_650M_UR50D
python -m esm2_mech.experiments.mechanism.family_clustering \
    --run_dir run_0 --model esm2_t33_650M_UR50D
python -m esm2_mech.experiments.mechanism.mlp \
    --data_dir run_0/data --emb_dir run_0/data/embeddings/esm2_t33_650M_UR50D --out_dir run_0
python -m esm2_mech.experiments.mechanism.mut_only_mlp \
    --data_dir run_0/data --emb_dir run_0/data/embeddings/esm2_t33_650M_UR50D
python -m esm2_mech.experiments.mechanism.contrastive_mechanism \
    --data_dir run_0/data --emb_dir run_0/data/embeddings/esm2_t33_650M_UR50D --out_dir run_0
python -m esm2_mech.experiments.mechanism.clan_holdout
python -m esm2_mech.experiments.mechanism.mmseqs_cluster_holdout
python -m esm2_mech.experiments.mechanism.multiseed_v1
python -m esm2_mech.experiments.mechanism.within_family_mechanism
```

### D2 — pathogenicity control (result 6)

```bash
python -m esm2_mech.experiments.pathogenicity.pathogenicity_probes --run_dir run_0
python -m esm2_mech.experiments.pathogenicity.pathogenicity_5seed
```

### D3 — proteome features + enzyme control (results 11–14, 25)

Requires `data/enzyme_labels.tsv` (Stage A step 9) and
`data/proteome_features_aligned.npy` (Stage A step 10).

```bash
python -m esm2_mech.experiments.proteome_features.proteome_pilot
python -m esm2_mech.experiments.proteome_features.proteome_mechanism
python -m esm2_mech.experiments.proteome_features.per_gene_ablation
python -m esm2_mech.experiments.proteome_features.clinical_utility
python -m esm2_mech.experiments.proteome_features.enzyme_classification
```

### D4 — Badonyi (results 15–16)

Requires `data/badonyi_features_aligned.npy` (Stage A step 11).

```bash
python -m esm2_mech.experiments.badonyi.badonyi_mechanism
python -m esm2_mech.experiments.badonyi.badonyi_holdout_survival
python -m esm2_mech.experiments.badonyi.badonyi_leakage_analysis
```

### D5 — AlphaMissense + ProteinGym (results 17–18, 24)

Requires `data/alphamissense_scores_full.json` (Stage C) and
`data/esm1v_scores_full.json` (Stage B5).

```bash
python -m esm2_mech.experiments.alphamissense.alphamissense_family_split
python -m esm2_mech.experiments.alphamissense.proteingym_alphamissense
python -m esm2_mech.experiments.alphamissense.proteingym_esm2_ll
python -m esm2_mech.experiments.alphamissense.esm1v_family_split
```

### D6 — perturbation + stability (results 19–22)

Requires `data/scan_features.npy` (Stage B2 phase 3).

```bash
python -m esm2_mech.experiments.perturbation.perturbation_pattern
python -m esm2_mech.experiments.perturbation.perturbation_probe
python -m esm2_mech.experiments.perturbation.ll_scan
python -m esm2_mech.experiments.stability.megascale_stability --run_dir run_0 --model esm2_t33_650M_UR50D
python -m esm2_mech.experiments.stability.megascale_mlp
```

### D7 — geometry + transferability synthesis (result 23)

```bash
python -m esm2_mech.experiments.geometry.magnitude_direction
python -m esm2_mech.experiments.geometry.direction_geometry
python -m esm2_mech.experiments.geometry.transfer_contrast
python -m esm2_mech.experiments.geometry.conservation_axis
python -m esm2_mech.experiments.geometry.probe4_axis_identity
```

---

## Stage E — ESM-3 mechanism comparison (result 26)

Requires ESM-3 embeddings from Stage B4.

```bash
python -m esm2_mech.experiments.esm3.esm3_mechanism --phase 3
```

**⚠️ Fold-eligibility caveat (unresolved):** ESM-3 uses `< 3` classes-in-train guard
(via `utils.probes.run_mlp_cv`) but `mlp.py` still uses `< 2`. ESM-2 and ESM-3 are
therefore not scored under identical fold rules. The +0.125 F1 gap (result 26) may be
partly a fold-eligibility artifact. Fix before treating result 26 as final: route
`mlp.py` through `utils.probes.run_mlp_cv` and rerun D1 + E.

---

## Verification checklist

- [ ] `data/embeddings/esm2_t33_650M_UR50D/valid_variants.json` row count matches all four `.npy` arrays.
- [ ] `data/pfam_families.json` has entries for ≥ 1,900 genes (< 1,900 suggests a partial fetch).
- [ ] `data/enzyme_labels.tsv` spot-checked against UniProt EC numbers for a handful of kinases and proteases.
- [ ] `data/alphamissense_scores_full.json` non-empty and covers > 90% of `valid_variants.json`.
- [ ] result 6 (pathogenicity AUROC ~0.88) and result 7 (mechanism family-split floor ~0.35–0.39) reproduce their headline numbers — these are the pipeline spine and should not move.
- [ ] result 26: ESM-2 and ESM-3 scored under identical fold rule before reporting M1/M2 gap (see Stage E caveat).
- [ ] `git status` clean; results committed.
