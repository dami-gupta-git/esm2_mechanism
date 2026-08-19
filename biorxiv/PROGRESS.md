# run_biorxiv progress

Live status record for `RUNBOOK_biorxiv.md`, which holds the steps only. Each section here mirrors
a section of the runbook, in the same order.

Every analysis script computes the confidence intervals. A confidence interval is a range around a score
saying how much that score would plausibly move if the same analysis were re-run on a different
but similarly-drawn sample of genes, rather than reporting a single number as if it were exact.

## Prerequisites — manually placed files

| # | Item | Outputs | Status | Notes |
|---|---|---|---|---|
| 1 | `DiseaseMech_Stability_VEPS.xlsx` | `data/downloads/DiseaseMech_Stability_VEPS.xlsx` | ⬜ | |
| 2 | `AllG2P.csv` | `data/downloads/AllG2P.csv` | ⬜ | |

## 0. Preconditions

| Step | Item | Status | Notes |
|---|---|---|---|
| 0.0 | Environment setup | ⬜ | |
| 0.1 | Pathogenicity provenance | ⬜ | |
| 0.2 | Stats machinery wired | ⬜ | |
| 0.3 | Methodology rules | ⬜ | |
| 0.4 | Paired cluster bootstrap | ⬜ | |
| 0.5/0.6 | Pre-registered decision rules | ⬜ | |
| 0.7 | Pinned environment | ⬜ | |
| 0.8 | Configuration (`RUN_NAME` flip) | ⬜ | |
| 0.9 | Working tree clean | ⬜ | |
| 0.10 | Megascale embedding provenance | ⬜ | |

## 1. Build gene list

| Step | Command | Outputs | Status |
|---|---|---|---|
| 1.1 | `python -m esm2_mech.fetch_data.build_gene_list` | `data/gene_list.tsv` | ⬜ |

## 2. Fetch variant data

| Step | Command | Outputs | Status |
|---|---|---|---|
| 2.1 | `python -m esm2_mech.fetch_data.fetch_variants --step gerasimavicius` | `gerasimavicius_variants.json` | ⬜ |
| 2.2 | `python -m esm2_mech.fetch_data.fetch_variants --step clinvar` | `clinvar_variants.tsv` | ⬜ |
| 2.3 | `python -m esm2_mech.fetch_data.fetch_variants --step merge --pathogenic_only` | `variants.json` | ⬜ |
| 2.4 | `python -m esm2_mech.fetch_data.fetch_sequences` | `cache/sequences.json` | ⬜ |
| 2.5 | `python -m esm2_mech.fetch_data.fetch_annotations --step pfam` | `pfam_families.json` | ⬜ |
| 2.6 | `python -m esm2_mech.fetch_data.fetch_alphamissense_mechanism` | `alphamissense_scores_full.json` | ⬜ |
| 2.7 | `python -m esm2_mech.fetch_data.build_valid_variants` | `valid_variants.json` | ⬜ |
| 2.8 | `python -m esm2_mech.fetch_data.fetch_pathogenicity_variants --max_per_gene_per_class 20 --fetch_seed 42 --force` | `clinvar_pathogenicity_variants.json`, `clinvar_pathogenicity_variants.params.json` | ⬜ |

## 3. Embed variants

| Step | Command | Status |
|---|---|---|
| 3.1 | `scp valid_variants.json` to pod | ⬜ |
| 3.2 | `scp sequences.json` to pod | ⬜ |
| 3.3 | `python -m esm2_mech.embeddings.embed_variants --model esm2_t33_650M_UR50D` | ⬜ |
| 3.4 | Copy embeddings back to local | ⬜ |

## 4. Experiment: ESM-2 delta-embedding mechanism

| Step | Command | Outputs | Status |
|---|---|---|---|
| 4.1 | `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 5` | `results/run_biorxiv/family_split_baselines_seed{0..4}.json`, `mechanism_oof_cache_seed{0..4}.json` | ⬜ |
| 4.2 | `python -m esm2_mech.experiments.mechanism.mlp --seeds 5` | `results/run_biorxiv/nonlinear_results_seed{0..4}.json` | ⬜ |
| 4.3 | `python -m esm2_mech.experiments.mechanism.family_clustering --seeds 5` | `results/run_biorxiv/family_clustering.json` | ⬜ |
| 4.4 | `python -m esm2_mech.experiments.mechanism.naive_baseline` | `results/run_biorxiv/naive_baseline.json` | ⬜ |
| 4.6 | `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 5 --n_permutations 1000` | `results/run_biorxiv/...` | ⬜ |
| 4.5 | `python -m esm2_mech.experiments.mechanism.leakage_fraction` | `results/run_biorxiv/leakage_fraction.json` | ⬜ |
| 4.7 | `python -m esm2_mech.experiments.mechanism.single_source_mechanism --seeds 5` | `results/run_biorxiv/single_source_gerasimavicius/...` | ⬜ |

## 5. Experiment: Pathogenicity positive control

| Step | Command | Outputs | Status |
|---|---|---|---|
| 5.1 | `scp clinvar_pathogenicity_variants.json` + `.params.json` to pod | | ⬜ |
| 5.2 | `scp pfam_families.json` to pod | | ⬜ |
| 5.3 | `python -m esm2_mech.experiments.pathogenicity.pathogenicity_control --phase embed --model esm2_t33_650M_UR50D --force_embed` | `pathogenicity_{wt,mut}_mean.npy`, `pathogenicity_meta.json` | ⬜ |
| 5.4 | Copy embeddings back to local | | ⬜ |
| 5.5 | `python -m esm2_mech.experiments.pathogenicity.pathogenicity_control --phase probe --seeds 5 --n_jobs <workers> --n_boot 1000` | `results/run_biorxiv/pathogenicity_control_seed{0..4}.json`, `pathogenicity_control.json` | ⬜ |

## 6. Experiment: Geometry of the pathogenicity direction

| Step | Command | Outputs | Status |
|---|---|---|---|
| 6.1 | `python -m esm2_mech.experiments.geometry.build_canonical_pathogenicity` | `data/pathogenicity_valid_variants_canonical.json` | ⬜ |
| 6.2 | `python -m esm2_mech.experiments.geometry.run_geometry --seeds 5 --stability-dataset tsuboyama` | `results/run_biorxiv/magnitude_direction/{probe_results,geometry_results,transfer_contrast,probe4_axis_identity}.json` | ⬜ |
| 6.3 | `scp pathogenicity_valid_variants_canonical.json` to pod | | ⬜ |
| 6.4 | `scp sequences.json` to pod | | ⬜ |
| 6.5 | `python -m esm2_mech.experiments.geometry.conservation_axis --extract` | `data/conservation_pathogenicity.npy`, `data/conservation_pathogenicity_meta.json` | ⬜ |
| 6.6 | Copy conservation outputs back to local | | ⬜ |
| 6.7 | `python -m esm2_mech.experiments.geometry.conservation_axis` | `results/run_biorxiv/magnitude_direction/conservation_axis.json` | ⬜ |

## 7. Experiment: Megascale stability positive control

| Step | Command | Outputs | Status |
|---|---|---|---|
| 7.1 | `python -m esm2_mech.experiments.stability.build_domain_families` | `data/megascale_domain_families.json` | ⬜ |
| 7.2 | `python -m esm2_mech.experiments.stability.megascale_stability --n_jobs 4` | `results/run_biorxiv/megascale_stability/per_protein_spearman.json`, `stability_projection_3c.json`, `summary.json` | ⬜ |
| 7.3 | `python -m esm2_mech.experiments.stability.megascale_mlp` | `results/run_biorxiv/megascale_stability/mlp_summary.json` | ⬜ |
| 7.4 | `python -m esm2_mech.experiments.stability.megascale_mlp --xgboost` | `results/run_biorxiv/megascale_stability/mlp_summary_xgb.json` | ⬜ |
| 7.5 | `python -m esm2_mech.experiments.stability.stability_baselines --n_jobs 4` | `results/run_biorxiv/megascale_stability/baselines.json` | ⬜ |

## 8. Experiment: Enzyme type classification (positive control)

| Step | Command | Outputs | Status |
|---|---|---|---|
| 8.1 | `python -m esm2_mech.experiments.proteome_features.enzyme_classification --seeds 5` | `results/run_biorxiv/enzyme_classification/enzyme_classification_summary.json` | ⬜ |

## Verification checklist

⬜ Not started.
