# Scripts

Shared utilities at the top level; all pipeline scripts are organised into packages below.

## Shared utilities

**`utils_probes.py`** — CV split generators (`gene_split_cv`, `family_split_cv`, `family_split_indices`), metric helpers (`compute_metrics`, `aggregate_folds`), and probe runners (`run_logreg_cv`, `run_mlp_cv`, `run_ridge_cv`). Imported by almost every package.

**`utils_sequences.py`** — UniProt sequence fetching, Pfam family fetching, `apply_missense`, and `window_sequence`. Imported by every embedding-extraction script.

**`go_smoke_test.py`** — End-to-end smoke test. Runs a minimal pipeline on synthetic data to verify the environment is configured correctly.

**`launch_scientist.py`** — Orchestration launcher for remote RunPod experiments.

---

## `fetch_data/` — data fetching and build scripts (Stage A + B)

**`fetch_uniprot_sequences.py`** — Fetches canonical protein sequences from UniProt REST API for all UniProt IDs in `merged_variants.json`. Resume-safe. Output: `data/cache/uniprot_sequences_extended.json`.

**`fetch_clinvar_variants.py`** — Fetches ClinVar pathogenic/likely-pathogenic missense variants for the gene list in `merged_gene_list.tsv`. Resume-safe, rate-limited. Output: `data/clinvar_variants.tsv`.

**`fetch_alphamissense.py`** — Downloads the full AlphaMissense substitution score table. Output: `data/cache/AlphaMissense_aa_substitutions.tsv.gz`.

**`fetch_enzyme_labels.py`** — Fetches EC number labels for genes via UniProt. Output: `data/enzyme_labels.tsv`.

**`build_merged_dataset.py`** — Merges Gerasimavicius variants with G2P/ClinVar variants. Gerasimavicius takes priority for genes present in both. Output: `data/merged_variants.json`.

**`build_proteome_features.py`** — Assembles the 41-feature gene-level proteome matrix from gnomAD, Ensembl, HPA, PaxDb, BioPlex, ClinGen. Applies family-mean centring. Output: `data/proteome_features_aligned.npy`.

**`build_badonyi_features.py`** — Builds the Badonyi 2024 feature matrix (pDN, pGOF, pLOF + derived features) aligned to the merged gene list. Output: `data/badonyi_features_aligned.npy`.

---

## `embeddings/` — GPU embedding extraction (Stage C)

**`esm2_mechanism.py`** — Main ESM-2 pipeline. Downloads Gerasimavicius dataset, fetches sequences, extracts ESM-2 650M embeddings, fits stability subspace, runs linear probe with gene-split and family-split CV, baselines, and probe direction orthogonality. Requires GPU. Output: `results/run_0/final_info.json`.

**`extract_merged_embeddings.py`** — Extracts ESM-2 mean-pooled embeddings for the merged variant set. Requires GPU. Output: `data/embeddings/merged_embeddings_wt_mean.npy` etc.

**`pathogenicity_control.py`** — Positive control: predicts ClinVar pathogenic vs benign. Phase 2 requires GPU for embedding extraction; phases 1 and 3 run on CPU. Output: `pathogenicity_control.json`.

**`megascale_stability.py`** — Extracts ESM-2 embeddings for the Megascale S1724 stability benchmark and runs stability regression probes (Ridge, GBM, RF) under protein family-split CV. Requires GPU for extraction. Output: `results/megascale_stability/`.

**`esm3_mechanism.py`** — ESM-3 1.4B mechanism experiment. Phase 1: download AF2 structures (CPU). Phase 2: extract seq-only and seq+struct embeddings (GPU). Phase 3: run probes and evaluate decision rules M1/M2/M3 (CPU). Output: `results/esm3_mechanism/summary.json`.

**`esm1v_family_split.py`** — Runs ESM-1v masked-marginal ΔLL scores through family-split CV as a mechanism probe comparison. CPU (reads cached scores).

**`score_esm1v.py`** — Scores ClinVar pathogenicity variants with ESM-1v masked-marginal ΔLL averaged over two checkpoints. Requires GPU. Output: `data/esm1v_scores_full.json`.

**`perturbation_scan.py`** — In-silico scan: extracts ESM-2 delta embeddings for 100 evenly-spaced probe positions × 3 amino acids per gene. Requires GPU. Output: `data/scan_features.npy`.

---

## `mechanism/` — core mechanism CV (results 1–10)

**`experiment_mlp.py`** — Nonlinear probes on cached delta embeddings. PyTorch MLP (1280→256→64→3) with dropout and class weighting. Also runs GBM, RF (PCA-50), and kNN. CPU-only. Output: `mlp_results_seed{N}.json`.

**`mut_only_mlp.py`** — Mutant-only embedding MLP probe. Tests whether the mutant embedding alone (without the WT reference) carries mechanism signal. CPU-only.

**`family_split_baselines.py`** — Runs all baselines (WT-only, mutant-only, concat, delta, one-hot, FoldX, AlphaMissense) under both gene-split and family-split CV. Reports Δ(gene−family) leakage fraction per feature. Output: `family_split_baselines.json`.

**`family_clustering.py`** — Diagnostic: quantifies ESM-2 family clustering via silhouette, k-NN family purity, within/between cosine ratio, and linear family-probe accuracy. Output: `family_clustering.json`.

**`multiseed_v1.py`** — 5-seed replication of v1 headline numbers: mechanism MLP (Gerasimavicius + merged) and pathogenicity control under gene-split and family-split. CPU-only. Output: `results/v1_multiseed/summary.json`.

**`pathogenicity_5seed.py`** — 5-seed canonical replication of the pathogenicity positive control (n=16,576 variants). CPU-only (reads cached embeddings). Output: `results/pathogenicity_5seed/`.

**`contrastive_mechanism.py`** — Contrastive projection head (TripletMarginLoss) + k-NN probe under family-split CV. V9 contrastive head with family-invariant positive pairs. Output: `results/contrastive_mechanism/`.

**`clan_holdout.py`** — Leave-one-clan-out evaluation. Tests whether mechanism signal generalises to completely unseen Pfam clans. Output: `clan_holdout_results_seed{N}.json`.

**`mmseqs_cluster_holdout.py`** — MMseqs2 sequence-cluster holdout (30% identity). Tests generalisation beyond family-level homology. Output: `results/mmseqs_cluster_holdout/`.

---

## `proteome/` — proteome / gene-level models (results 11–14)

**`proteome_pilot.py`** — Pilot: tests proteome features under family-split CV before the full V1–V4 run.

**`proteome_mechanism.py`** — Full phase 3 modelling. V1 (ESM-2 delta MLP), V2 (proteome LogReg + LGBM + MLP), V3 (concat MLP), V4 (contrastive head + k-NN). 5-fold family-split CV, 5 seeds. Output: `results/proteome_mechanism/`.

**`per_gene_ablation.py`** — Leave-one-gene-out ablation on the proteome feature matrix. Identifies which genes drive V2 performance. Output: `results/per_gene_ablation/`.

**`clinical_utility.py`** — GOF/DN identification within ClinGen HI=3 genes. Evaluates clinical utility of the proteome model for therapy selection. Output: `results/clinical_utility/`.

---

## `badonyi/` — Badonyi + within-family (results 15–16)

**`badonyi_mechanism.py`** — Adds Badonyi 2024 pDN/pGOF/pLOF as a modality. Tests V_bad alone and in combination with ESM-2 delta and proteome features under family-split CV. Output: `results/badonyi_mechanism/`.

**`badonyi_holdout_survival.py`** — Tests whether Badonyi's published predictions survive Pfam family-split and MMseqs2 cluster holdout. Stratifies by Badonyi training-set membership. Output: `results/badonyi_survival/`.

**`badonyi_leakage_analysis.py`** — Quantifies how much of Badonyi's reported performance is attributable to training-set leakage. Output: `results/badonyi_leakage/`.

**`within_family_mechanism.py`** — Within-family mechanism classification using LOGO CV on 24 Pfam families. Tests whether proteome residual features distinguish mechanism within a single family. Output: `results/within_family/`.

---

## `alphamissense/` — AlphaMissense / ProteinGym externals (results 17–18)

**`alphamissense_family_split.py`** — Per-family AUROC stratification of AlphaMissense on ClinVar variants. Computes mean and min per-family AUROC to test family-robustness. Output: `results/alphamissense_family/`.

**`proteingym_alphamissense.py`** — AlphaMissense on ProteinGym DMS assays. Computes per-assay AUROC distribution, identifies assays below threshold. Output: `results/proteingym_alphamissense/`.

**`proteingym_esm2_ll.py`** — ESM-2 masked-marginal log-likelihood on ProteinGym DMS assays. Phases: GPU extraction, CPU probe. Output: `results/proteingym_esm2_ll/`.

---

## `perturb/` — perturbation + stability (results 19–22)

**`perturbation_pattern.py`** — Builds per-gene spatial pattern features from observed variant delta magnitudes and positions. Tests whether perturbation geometry adds mechanism signal beyond mean-pooled delta. Output: `results/perturbation_pattern/`.

**`perturbation_probe.py`** — Probe runs on scan features from `embeddings/perturbation_scan.py`. Tests scan-only, scan+delta, scan+proteome under family-split CV. Evaluates gates G1/G2/G3. Output: `results/perturbation_scan/probe_results.json`.

**`megascale_mlp.py`** — Nonlinear (GBM, RF, MLP) stability probes on Megascale S1724. Reads cached embeddings from `embeddings/megascale_stability.py`. CPU-only. Output: `results/megascale_stability/`.

**`ll_scan.py`** — Log-likelihood scan at the same 100 probe positions as the embedding scan. Phase 2 requires GPU; phase 3 (features + probe) is CPU. Output: `results/ll_scan/`.

---

## `analysis/` — transferability, geometry, and benchmarks (results 23+)

**`magnitude_direction.py`** — Decomposes ESM-2 deltas into magnitude and direction. Runs pathogenicity and mechanism probes on each component. Evaluates pre-registered gates P1–P4. Output: `results/magnitude_direction/`.

**`conservation_axis.py`** — Phase 1 (GPU): extracts masked-LM conservation scores for pathogenicity variants. Phase 2 (CPU): compares conservation axis to result_23 pathogenicity direction under family-split. Output: `results/magnitude_direction/`.

**`direction_geometry.py`** — Rank, cosine similarity, and family-transfer analysis of the pathogenicity direction. Tests whether the direction is universal across families. Output: `results/magnitude_direction/`.

**`transfer_contrast.py`** — Linear vs GBM cross-task transfer: pathogenicity → stability → mechanism. Tests whether a probe trained on one task transfers to another. Output: `results/magnitude_direction/`.

**`probe4_axis_identity.py`** — Tests whether the pathogenicity axis, stability axis, and conservation axis are the same or orthogonal directions in embedding space. Output: `results/magnitude_direction/`.

**`enzyme_classification.py`** — External benchmark: ESM-2 delta embeddings on enzyme mechanism classification (EC number). Tests whether the mechanism null generalises to a different task with ground-truth biochemical labels. Output: `results/enzyme_classification/`.

**`plot.py`** — Matplotlib plots from `final_info.json`: AUROC bar chart, probe direction cosine matrix heatmap, variance-explained bar chart. Run as `python plot.py run_0`.
