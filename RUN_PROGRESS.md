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

---

## Run 3 — Embeddings

| # | Stage | Command | Inputs | Outputs | Status | Notes |
|---|---|---|---|---|---|---|
| 1 | embed_variants | `python -m esm2_mechanism.embeddings.embed_variants --data_dir data --model esm2_t33_650M_UR50D --batch_size 32` | `data/variants.json`, `data/cache/sequences.json` | `data/embeddings/esm2_t33_650M_UR50D/embeddings_wt_mean.npy`, `embeddings_mut_mean.npy`, `embeddings_wt_pos.npy`, `embeddings_mut_pos.npy`, `valid_variants.json` | ✅ 2026-05-30 | 17,826 valid variants; shape (17826, 1280); run on RunPod H100 |
| 2 | perturbation_scan phase 1 | `python -m esm2_mechanism.perturb.perturbation_scan --run_phase 1` | `data/cache/sequences.json`, `data/cache/uniprot_sequences_extended.json` | `data/cache/scan_probes.json` | ✅ 2026-05-30 | 553,476 probes for 1,935 genes; run locally (CPU) |
| 3 | embed_scan | `python -m esm2_mechanism.embeddings.embed_scan --batch_size 128` | `data/cache/scan_probes.json`, `data/cache/sequences.json` | `data/embeddings/esm2_t33_650M_UR50D/scan_wt.npy`, `scan_mut.npy` | ✅ 2026-05-30 | 553,476 probes; shape (553476, 1280); run on RunPod H100 |
| 4 | perturbation_scan phase 3 | `python -m esm2_mechanism.perturb.perturbation_scan --run_phase 3` | `data/cache/scan_probes.json`, `scan_wt.npy`, `scan_mut.npy` | `data/scan_features.npy` | ✅ 2026-05-30 | 1935 genes × 5 features; run locally (CPU) |
| 5 | esm3_mechanism phase 1 | `python -m esm2_mechanism.mechanism.esm3_mechanism --phase 1` | `data/gerasimavicius_variants.json`, `data/cache/sequences.json` | `data/cache/esm3_struct_tokens.json` | ❌ cancelled | downloading AF2 structures + tokenising; run locally (CPU) |

---

## Run 4 — started 2026-05-30

Uses `esm2_mech` package (RUNBOOK_3). All commands use `python -m esm2_mech.<module>`.

| # | Step | Command | Inputs | Outputs | Status | Notes |
|---|---|---|---|---|---|---|
| 1 | A1 | `python -m esm2_mech.fetch_data.build_gene_list` | `downloads/DiseaseMech_Stability_VEPS.xlsx`, `downloads/AllG2P.csv` | `gene_list.tsv` | ✅ 2026-05-30 | 2376 genes (gerasimavicius=950, g2p=1426); AR=727, DN=108, GOF=148, HI=82, LOF=1311; 475 g2p_disagrees; 61 excluded (unresolvable conflicting G2P mechanism) |
| 2 | A2 | `python -m esm2_mech.fetch_data.fetch_variants --step gerasimavicius` | `downloads/DiseaseMech_Stability_VEPS.xlsx` | `gerasimavicius_variants.json` | ✅ prior run | 10,233 variants, 948 genes; AR=5678, GOF=1983, HI=1678, DN=894 — file verified on disk |
| 3 | A3 | `python -m esm2_mech.fetch_data.fetch_variants --step clinvar` | `gene_list.tsv` | `clinvar_variants.tsv` | ✅ prior run | 47,752 variants across 2376 genes — file verified on disk |
| 4 | A4 | `python -m esm2_mech.fetch_data.fetch_variants --step merge` | `gerasimavicius_variants.json`, `gene_list.tsv`, `clinvar_variants.tsv` | `variants.json` | ✅ prior run | 17,921 variants, 1941 genes; gerasimavicius=10233, clinvar_g2p=7688 — file verified on disk |
| 5 | A5 | `python -m esm2_mech.fetch_data.fetch_sequences` | `variants.json` | `cache/uniprot_sequences_extended.json` | ✅ prior run | 1939 unique UniProt IDs — file verified on disk |
| 6 | A6 | `python -m esm2_mech.fetch_data.fetch_annotations --step pfam` | `variants.json` | `pfam_families.json` | ✅ prior run | 1908/1941 genes annotated, 33 unannotated — file verified on disk |
| 7 | A7 | `python -m esm2_mech.fetch_data.build_gene_universe` | `gene_list.tsv`, `pfam_families.json` | `gene_universe.tsv` | ✅ prior run | 1908 genes retained — file verified on disk |
| 8 | A8 | `python -m esm2_mech.fetch_data.fetch_annotations --step uniprot` | `variants.json` | `cache/uniprot_sequences_extended.json` | ✅ prior run | same file as step 5; 1939 entries — file verified on disk |
| 9 | A9 | `python -m esm2_mech.fetch_data.fetch_annotations --step enzyme` | `variants.json`, `gene_list.tsv` | `enzyme_labels.tsv` | ✅ prior run | 2376 rows; kinase=131, protease=68, oxidoreductase=123, non-enzyme=1619, missing=436 — file verified on disk |
| 10 | A10 | `python -m esm2_mech.fetch_data.build_proteome_features` | `gene_universe.tsv` + manual downloads | `gene_proteome_features.tsv`, `proteome_features_aligned.npy`, `proteome_feature_columns.json` | ✅ prior run | 1908 genes × 33 features; shape (1908, 33) — file verified on disk |
| 11 | A11 | `python -m esm2_mech.fetch_data.build_badonyi_features` | `downloads/table_S3.xlsx`, `gene_universe.tsv` | `badonyi_features.tsv`, `badonyi_features_aligned.npy`, `badonyi_feature_columns.json` | ✅ prior run | 1908 genes × 13 features; shape (1908, 13); 8 genes missing raw scores (handled via missing indicator columns) — file verified on disk |

---

## Run 5 — started 2026-05-30 (RUNBOOK_4)

| # | RUNBOOK_4 step | Command | Inputs | Outputs | Status | Notes |
|---|---|---|---|---|---|---|
| 1 | Stage 1 | `python -m esm2_mech.fetch_data.build_gene_list` | `downloads/DiseaseMech_Stability_VEPS.xlsx`, `downloads/AllG2P.csv` | `gene_list.tsv` | ✅ 2026-05-30 | 2376 genes (gerasimavicius=950, g2p=1426); AR=727, DN=108, GOF=148, HI=82, LOF=1311; 475 g2p_disagrees; 61 excluded |
| 2 | Exp1 Step 1 | `python -m esm2_mech.fetch_data.fetch_variants --step gerasimavicius` | `downloads/DiseaseMech_Stability_VEPS.xlsx` | `gerasimavicius_variants.json` | ✅ 2026-05-30 | 10,233 variants, 948 genes; AR=5678, GOF=1983, HI=1678, DN=894 |
| 3 | Exp1 Step 1 | `python -m esm2_mech.fetch_data.fetch_variants --step clinvar` | `gene_list.tsv` | `clinvar_variants.tsv` | ✅ 2026-05-30 | 47,752 variants across 2376 genes |
| 4 | Exp1 Step 1 | `python -m esm2_mech.fetch_data.fetch_variants --step merge` | `gerasimavicius_variants.json`, `gene_list.tsv`, `clinvar_variants.tsv` | `variants.json` | ✅ 2026-05-30 | 17,921 variants, 1941 genes; gerasimavicius=10233, clinvar_g2p=7688 |
| 5 | Exp1 Step 1 | `python -m esm2_mech.fetch_data.fetch_sequences` | `variants.json` | `data/cache/sequences.json` | ✅ 2026-05-30 | 1939 unique UniProt IDs |
| 6 | Exp1 Step 1 | `python -m esm2_mech.fetch_data.fetch_annotations --step pfam` | `variants.json` | `data/pfam_families.json` | ✅ 2026-05-30 | 1908/1941 genes annotated, 33 unannotated |
| 7 | Exp1 Step 2 | `python -m esm2_mech.embeddings.embed_variants --model esm2_t33_650M_UR50D --batch_size 32` | `variants.json`, `cache/sequences.json` | `embeddings_wt_mean.npy`, `embeddings_mut_mean.npy`, `embeddings_wt_pos.npy`, `embeddings_mut_pos.npy`, `valid_variants.json` | ✅ 2026-05-30 | 17,826 valid variants; shape (17826, 1280); run on RunPod H100 |
| 8 | Exp1 Step 3 | `python -m esm2_mech.experiments.classify_by_mechanism` | `variants.json`, `cache/sequences.json`, `pfam_families.json`, `embeddings_*.npy`, `alphamissense_scores_full.json` | `results/run1/family_split_baselines_seed{0..4}.json` | ✅ 2026-05-30 | 17,826 variants, 1,935 genes, 1,136 families; 5 seeds; PCA 256 components (98.0% variance); AlphaMissense missing for seed 0; run on RunPod |

---

## Run 6 — started 2026-05-30 (RUNBOOK_4, fresh from scratch)

| # | RUNBOOK_4 step | Command | Inputs | Outputs | Status | Notes |
|---|---|---|---|---|---|---|
| 1 | Stage 0 | `python3 -m venv .venv && source .venv/bin/activate && pip install -e .` | — | `.venv/` | ✅ 2026-05-30 | Environment setup; all deps already satisfied |
| 2 | Stage 1 | `python -m esm2_mech.fetch_data.build_gene_list` | `downloads/DiseaseMech_Stability_VEPS.xlsx`, `downloads/AllG2P.csv` | `gene_list.tsv` | ✅ 2026-05-30 | 2376 genes (gerasimavicius=950, g2p=1426); AR=727, DN=108, GOF=148, HI=82, LOF=1311; 475 g2p_disagrees; 61 excluded |
| 3 | Exp1 Step 1 (step 2) | `python -m esm2_mech.fetch_data.fetch_variants --step gerasimavicius` | `downloads/DiseaseMech_Stability_VEPS.xlsx` | `gerasimavicius_variants.json` | ✅ 2026-05-30 | 10,233 variants, 948 genes; AR=5678, GOF=1983, HI=1678, DN=894 |
| 4 | Exp1 Step 1 (step 3) | `python -m esm2_mech.fetch_data.fetch_variants --step clinvar` | `gene_list.tsv` | `clinvar_variants.tsv` | ✅ prior run | 47,752 variants across 2376 genes — file verified on disk |
| 5 | Exp1 Step 1 (step 4) | `python -m esm2_mech.fetch_data.fetch_variants --step merge --pathogenic_only` | `gerasimavicius_variants.json`, `gene_list.tsv`, `clinvar_variants.tsv` | `variants.json` | ✅ 2026-05-30 | 17,921 variants, 1941 genes; gerasimavicius=10233, clinvar_g2p=7688; --pathogenic_only drops likely pathogenic |
| 6 | Exp1 Step 1 (step 5) | `python -m esm2_mech.fetch_data.fetch_sequences` | `variants.json` | `cache/sequences.json` | ✅ 2026-05-30 | 1939 unique UniProt IDs; cache already complete |
| 7 | Exp1 Step 1 (step 6) | `python -m esm2_mech.fetch_data.fetch_annotations --step pfam` | `variants.json` | `pfam_families.json` | ✅ 2026-05-30 | 1908/1941 genes annotated, 33 unannotated; cache already complete |
| 8 | Exp1 Step 1 (step 7) | `python -m esm2_mech.fetch_data.fetch_alphamissense_mechanism` | `variants.json` | `alphamissense_scores_full.json` | ✅ 2026-05-30 | matched 17,820/17,895 (99.6%); 24 variants dropped (dup uniprot+variant key); streamed 1.1GB AlphaMissense_aa_substitutions.tsv.gz (~215M rows, no cache) |
| 9 | Exp1 Step 1 (step 8) | `python -m esm2_mech.fetch_data.build_valid_variants` | `variants.json`, `cache/sequences.json` | `valid_variants.json` | ✅ 2026-05-30 | 17,826 valid variants; 95 skipped (invalid WT/mut window) |
| 10 | Exp1 Step 2 (GPU) | `python -m esm2_mech.embeddings.embed_variants --model esm2_t33_650M_UR50D --batch_size 32` | `valid_variants.json`, `cache/sequences.json` | `embeddings_wt_mean.npy`, `embeddings_mut_mean.npy`, `embeddings_wt_pos.npy`, `embeddings_mut_pos.npy`, `embedded_variants.json` | ✅ 2026-05-30 | RunPod H100 (31.24.80.34:13300, key id_runpod_2). Cloned esm2_mechanism @ 4210e96 to /workspace/repo. **Verified clean:** EXIT_CODE=0, all four arrays (17826, 1280), embedded_variants.json len=17826, index aligned. Output renamed valid_variants.json→embedded_variants.json on pod; all 5 files scp'd back to local data/embeddings/esm2_t33_650M_UR50D/ |
| 11 | Exp1 Step 3 | `python -m esm2_mech.experiments.mechanism.classify_by_mechanism` | `variants.json`, `cache/sequences.json`, `pfam_families.json`, `embeddings_*.npy`, `alphamissense_scores_full.json` | `results/run6/family_split_baselines_seed{0..4}.json`, `aggregate.json` | ✅ 2026-05-30 | RunPod (tmux `step3`), CLASSIFY_EXIT=0. 17,826 variants, 1,935 genes, 1,136 Pfam families, 5 seeds; PCA 256 comp (98.0% var); AlphaMissense 17,733/17,826. Headline macro-F1 (5-seed mean): wt_only_mean gene-split 0.545 vs family-split 0.442 (Δ +0.10 = homology leakage); delta_mean ~0.288 both splits; alphamissense 0.288; foldx_ddg 0.279 |
| 12 | Exp1 Step 3 | `python -m esm2_mech.experiments.mechanism.mlp --seed {0..4}` | `valid_variants.json`, `pfam_families.json`, `embeddings_*.npy` | `results/run6/nonlinear_results_seed{0..4}.json` | ✅ 2026-05-31 | RunPod, all 5 seeds (seed 0 via tmux `step3`; seeds 1-4 via tmux `mlp4`, ALL_DONE, all SEED_x_EXIT=0). Nonlinear (MLP/GBM/RF/kNN) on delta embeddings. 5-seed mean±std: mlp_delta_mean gene 0.399±0.009 / family 0.380±0.010; kNN 0.408±0.008; GBM 0.309±0.004; RF 0.298±0.004. Stable (std ≤0.013). Results scp'd to local results/run6/ |
| 13 | Exp1 Step 3 | `python -m esm2_mech.experiments.mechanism.family_clustering` | `valid_variants.json`, `pfam_families.json`, `embeddings_wt_mean.npy`, `embeddings_mut_mean.npy` | `results/run6/family_clustering.json` | ✅ 2026-05-31 | Local CPU. WT k=5 family purity 0.254 (~50× chance, z=+249); family-probe 61% (vs 4.4%); within/between 0.514; 83% mechanism-family overlap (multi-gene families). Delta collapses family signal to chance. Report: report_protein_family.md |
| 14 | (baseline) | `python -m esm2_mech.experiments.mechanism.naive_baseline` | `valid_variants.json`, `pfam_families.json` | `results/run6/naive_baseline.json` | ✅ 2026-05-31 | Local CPU. DummyClassifier floor: most_frequent macro-F1 0.288, stratified 0.329, AUROC 0.50 (5-seed). The 0.288 floor matched by delta_mean/onehot/foldx/alphamissense in report_classifier |
| 15 | Exp2 (control) | `python -m esm2_mech.experiments.pathogenicity.pathogenicity_control --model esm2_t33_650M_UR50D --batch_size 32` | `variants.json`, `cache/sequences.json`, `pfam_families.json` (ClinVar bulk auto-downloaded) | `clinvar_pathogenicity_variants.json`, `pathogenicity_{wt,mut}_mean.npy`, `results/run6/pathogenicity_control{,_seed0..4}.json` | ✅ 2026-05-31 | RunPod H200 (31.24.80.57:13143, key id_runpod_2). Consolidated fetch+embed+probe. 38,698 balanced variants (P/B), 37,218 embedded, 1,929 genes, GRCh38. 5-seed (parallel, per-seed files). **PASSES** (delta MLP ≥ 0.85). 5-seed mean±std AUROC: delta_mean MLP gene 0.897±0.001 / family 0.894±0.001 (Δ +0.003 = no leakage); delta_mean logreg 0.862 / 0.859; wt_only MLP 0.616 / 0.605. Dissociation reproduced: ESM-2 delta predicts pathogenicity ~0.90 but mechanism at chance. Result JSONs + embeddings (pathogenicity_{wt,mut}_mean.npy, pathogenicity_meta.json, clinvar_pathogenicity_variants.json) scp'd back to local 2026-05-31 from pod 31.24.80.57:15686; embeddings verified (37,218 rows, valid_indices-aligned, fingerprint b69da387… matches). Report: report_control.md |
| 16 | Exp3 (within-family) | `python -m esm2_mech.experiments.mechanism.mechanism_within_family --seeds 5` | `valid_variants.json`, `pfam_families.json`, `embeddings_wt_mean.npy`, `embeddings_mut_mean.npy` | `results/run6/within_family_mechanism.json` | ✅ 2026-05-31 | Local CPU (no GPU; reads existing run6 embeddings). Within-family gene-split CV per Pfam family (≥6 genes, ≥2 classes); wt_only vs delta × logreg/MLP, 5 seeds; per-family macro-F1 + per-class AUROC (mean±std) and majority baseline. 28 families qualify, 8 unscorable (single-gene minority class). Result: delta sits at the per-family baseline in nearly every family — within-family null matching the cross-family null. The largest balanced family PF00520 (ion channel, 1,044 variants) delta 0.256–0.299 vs base 0.253. Report: report_within_family.md |
| 17 | (diagnostic) | `python -m esm2_mech.experiments.mechanism.leakage_fraction` | `family_split_baselines_seed{0..4}.json`, `naive_baseline.json`, `family_clustering.json` | `results/run6/leakage_fraction.json` | ✅ 2026-05-31 | Local CPU (no model; reads existing result JSONs). Leakage fraction = (gene − family macro-F1)/(gene − chance), chance=0.288 measured. Absolute-embedding features ~40% (wt_only 40.1%, mut_only 40.3%, wt_concat 39.4%, delta_per_residue 38.2%); delta_mean/onehot/foldx/alphamissense undefined (at floor). Computed from seed-averaged F1 (per-seed ratios unstable). Report: report_leakage_fraction.md |
| 18 | Exp4 (ESM-3) | `python -m esm2_mech.experiments.esm3.esm3_mechanism --phase {1,2,3} --dataset merged` | `valid_variants.json`, `cache/sequences.json`, `esm3_struct_tokens.json`, `pfam_families.json` | `data/embeddings/esm3-sm-open-v1/merged/{seq,seq_struct}_mean.npy` (+`_wt`/`_mut`), `results/run6/esm3_mechanism/merged/summary.json` | ✅ 2026-06-01 | RunPod H100 (31.24.80.44, key id_runpod_2; pod restarted mid-run for disk resize, port 15749→11545). ESM-3 esm3-sm-open-v1 (1.4B). 17,826 merged variants, 0 dropped (apply_missense WT-check → row set identical to ESM-2 classifier). AF2 structure on 94.5% (16,852/17,826; 400 coord fallbacks). 5 seeds. ESM-2 floor read at runtime (mlp_delta_mean_family 5-seed mean = 0.380, threshold 0.430). Family-split MLP: seq 0.438±0.009, seq_struct 0.453±0.012. **M1 pass (0.453>0.430), M2 pass (0.438>0.430), M3 fail (0.014<0.030)** → scale suffices, structure adds little. wt/mut arrays preserved for future contrastive/mut-only work. Geras-only run (10,231 variants) superseded + marked contaminated (93 WT-mismatch rows, KNOWN_ISSUES.md). Report: report_esm3_mechanism.md |
| 19 | Exp5 (geometry) prep | `python -m esm2_mech.experiments.geometry.build_canonical_pathogenicity` | `clinvar_pathogenicity_variants.json`, `pathogenicity_{wt,mut}_mean.npy`, `pathogenicity_meta.json` | `data/pathogenicity_valid_variants_canonical.json` | ✅ 2026-06-01 | Re-indexes the embedded subset to a row-aligned canonical variant list; fingerprint-checked against the embeddings (b69da387…). 38,698 variants → 37,218 embedded rows. Built locally + on pod. |
| 20 | Exp5 (geometry) CPU | `python -m esm2_mech.experiments.geometry.run_geometry --seeds 5` | `pathogenicity_valid_variants_canonical.json`, `pathogenicity_{wt,mut}_mean.npy`, `valid_variants.json` + main `embeddings_*.npy`, `pfam_families.json` | `results/run6/magnitude_direction/{probe_results,geometry_results,transfer_contrast,probe4_axis_identity}.json` | ✅ 2026-06-01 | RunPod (31.24.80.42:15583, key id_runpod_2; 192-core). One orchestrator, 4 CPU probes, 5 seeds (joblib over seeds). Pathogenicity family-split: magnitude 0.673, direction 0.901 (MLP), full 0.893 → pathogenicity is directional (gates P1/P2 fail = hypothesis inverted). Direction family-universal (transfer AUROC 0.848). Transfer contrast: path 0.85/0.90 (lin/gbm), mechanism 0.63/0.64. Biochem axis R²=0.07 (not chemistry). Mechanism uses merged set so P3 differs from result_23 (dir F1 0.415 vs floor 0.288). Stability arm skipped (no megascale embeddings). scp'd back. Report: report_geometry.md |
| 21 | Exp5 (geometry) GPU+CPU | `python -m esm2_mech.experiments.geometry.conservation_axis --extract` then (no flag) | `pathogenicity_valid_variants_canonical.json`, `cache/sequences.json`, `pathogenicity_{wt,mut}_mean.npy`, `pfam_families.json` | `data/conservation_pathogenicity.npy` (+meta), `results/run6/magnitude_direction/conservation_axis.json` | ✅ 2026-06-01 | RunPod. Phase 1 (GPU): masked-LM logP_wt/logP_mut/entropy per variant, 37,218/37,218 covered. Phase 2 (CPU, 5 seeds): conservation alone 0.891 ≥ embedding delta 0.859; conservation+delta +0.002. **K1 PASS, K2 FAIL → the pathogenicity axis IS conservation.** Spearman(axis, masked_marginal)=+0.74. scp'd back. Report: report_geometry.md |
