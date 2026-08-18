# run_biorxiv progress

Live status record for `RUNBOOK_biorxiv.md`, which holds the steps only. Each section here mirrors
a section of the runbook, in the same order.

Every analysis script computes the confidence intervals. A confidence interval is a range around a score
saying how much that score would plausibly move if the same analysis were re-run on a different
but similarly-drawn sample of genes, rather than reporting a single number as if it were exact.

## Prerequisites — manually placed files

These are the source files the pipeline needs but cannot fetch itself; they must be placed in
`data/downloads/` before anything else runs.

| # | Item | Command | Outputs | Status | Notes |
|---|---|---|---|---|---|
| 1 | `DiseaseMech_Stability_VEPS.xlsx` | *(manually placed)* | `data/downloads/DiseaseMech_Stability_VEPS.xlsx` | ✅ reused | From previous run |
| 2 | `AllG2P.csv` | *(manually placed)* | `data/downloads/AllG2P.csv` | ✅ reused | From previous run |

## 0. Preconditions

These are the checks and setup steps that must all pass before the run is allowed to start.

| Step | Item | Command | Status | Notes |
|---|---|---|---|---|
| 0.0 | Environment setup | `cd /Users/dgupta/code/portfolio/ESM2/esm2_mechanism`<br>`python3 -m venv .venv && source .venv/bin/activate`<br>`pip install -e .` | ⬜ | |
| 0.1 | Pathogenicity provenance | *(verification, no command)* | ⬜ | |
| 0.2 | Stats machinery wired | *(verification, no command)* | ⬜ | |
| 0.3 | Methodology rules | *(verification, no command)* | ⬜ | |
| 0.4 | Paired cluster bootstrap | *(verification, no command)* | ⬜ | |
| 0.5/0.6 | Pre-registered decision rules | *(verification, no command)* | ⬜ | |
| 0.7 | Pinned environment | *(verification, no command)* | ⬜ | |
| 0.8 | Configuration | *(config change in `utils/paths.py:11`)* | ⬜ | |
| 0.9 | Working tree clean | `git status` | ⬜ | |

## 1. Build gene list

Builds the list of genes every later experiment uses.

| Step | Command | Inputs | Outputs | Status | Notes |
|---|---|---|---|---|---|
| 1.1 | `python -m esm2_mech.fetch_data.build_gene_list` | `downloads/DiseaseMech_Stability_VEPS.xlsx`, `downloads/AllG2P.csv` | `data/gene_list.tsv` | ✅ reused | From previous run |

## 2. Fetch variant data

Shared foundation for sections 4, 5, 6, and 7.

| Step | Command | Outputs | Status | Notes |
|---|---|---|---|---|
| 2.1 | `python -m esm2_mech.fetch_data.fetch_variants --step gerasimavicius` | `gerasimavicius_variants.json` | ✅ reused | From previous run |
| 2.2 | `python -m esm2_mech.fetch_data.fetch_variants --step clinvar` | `clinvar_variants.tsv` | ✅ reused | From previous run |
| 2.3 | `python -m esm2_mech.fetch_data.fetch_variants --step merge --pathogenic_only` | `variants.json` | ✅ reused | From previous run |
| 2.4 | `python -m esm2_mech.fetch_data.fetch_sequences` | `cache/sequences.json` | ✅ reused | From previous run |
| 2.5 | `python -m esm2_mech.fetch_data.fetch_annotations --step pfam` | `pfam_families.json` | ✅ reused | From previous run |
| 2.6 | `python -m esm2_mech.fetch_data.fetch_alphamissense_mechanism` | `alphamissense_scores_full.json` | ✅ reused | From previous run |
| 2.7 | `python -m esm2_mech.fetch_data.build_valid_variants` | `valid_variants.json` | ✅ reused | From previous run |
| 2.8 | `python -m esm2_mech.fetch_data.fetch_pathogenicity_variants` | `clinvar_pathogenicity_variants.json`, `clinvar_pathogenicity_variants.params.json` | ✅ reused | From previous run |

## 3. Embed variants

Shared foundation for sections 4 and 6.

| Step | Command | Status | Notes |
|---|---|---|---|
| 3.1 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> data/valid_variants.json root@<pod-ip>:/workspace/repo/data/` | ⬜ | |
| 3.2 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> data/cache/sequences.json root@<pod-ip>:/workspace/repo/data/cache/` | ⬜ | |
| 3.3 | `python -m esm2_mech.embeddings.embed_variants --model esm2_t33_650M_UR50D` | ⬜ | |
| 3.4 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> root@<pod-ip>:/workspace/repo/data/embeddings/esm2_t33_650M_UR50D/*.npy root@<pod-ip>:/workspace/repo/data/embeddings/esm2_t33_650M_UR50D/embedded_variants.json data/embeddings/esm2_t33_650M_UR50D/` | ⬜ | |

## 4. Experiment: ESM-2 delta-embedding mechanism

Tests whether ESM-2 embeddings can predict a variant's mechanism (DN/LOF/GOF), and checks that the
model isn't just recognizing which protein family the gene belongs to rather than actually
learning something about the mechanism.

| Step | Command | Outputs | Status |
|---|---|---|---|
| 4.1 | `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 5` | `results/run_biorxiv/family_split_baselines_seed{0..4}.json`, `aggregate.json` | ⬜ |
| 4.2 | `python -m esm2_mech.experiments.mechanism.mlp --seeds 5` | `results/run_biorxiv/nonlinear_results_seed{0..4}.json` | ⬜ |
| 4.3 | `python -m esm2_mech.experiments.mechanism.family_clustering --seeds 5` | `results/run_biorxiv/family_clustering.json` | ⬜ |
| 4.4 | `python -m esm2_mech.experiments.mechanism.naive_baseline` | `results/run_biorxiv/naive_baseline.json` | ⬜ |
| 4.5 | `python -m esm2_mech.experiments.mechanism.leakage_fraction` | `results/run_biorxiv/leakage_fraction.json` | ⬜ |
| 4.6 | `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 1 --n_permutations 1000` | `results/run_biorxiv/...` | ⬜ |
| 4.7 | `python -m esm2_mech.experiments.mechanism.single_source_mechanism --seeds 5` | `results/run_biorxiv/single_source_gerasimavicius/...` | ⬜ |

## 5. Experiment: Pathogenicity positive control

Tests whether the same embeddings can at least tell pathogenic from benign variants, to confirm
they carry usable signal at all.

| Step | Command | Outputs | Status |
|---|---|---|---|
| 5.1 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> data/clinvar_pathogenicity_variants.json root@<pod-ip>:/workspace/repo/data/` | | ⬜ |
| 5.2 | `python -m esm2_mech.experiments.pathogenicity.pathogenicity_control --model esm2_t33_650M_UR50D` | `results/run_biorxiv/pathogenicity_control.json`, per-seed files, embedding arrays | ⬜ |
| 5.3 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> root@<pod-ip>:/workspace/repo/data/embeddings/esm2_t33_650M_UR50D/pathogenicity_*.npy root@<pod-ip>:/workspace/repo/data/embeddings/esm2_t33_650M_UR50D/pathogenicity_meta.json root@<pod-ip>:/workspace/repo/results/run_biorxiv/pathogenicity_control.json data/embeddings/esm2_t33_650M_UR50D/` | | ⬜ |

## 6. Experiment: Geometry of the pathogenicity direction

Asks what the pathogenicity direction in embedding space actually corresponds to.

| Step | Command | Outputs | Status |
|---|---|---|---|
| 6.1 | `python -m esm2_mech.experiments.geometry.build_canonical_pathogenicity` | `data/pathogenicity_valid_variants_canonical.json` | ⬜ |
| 6.2 | `python -m esm2_mech.experiments.geometry.run_geometry --seeds 5` | `results/run_biorxiv/magnitude_direction/{probe_results,geometry_results,transfer_contrast,probe4_axis_identity}.json` | ⬜ |
| 6.3 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> data/pathogenicity_valid_variants_canonical.json root@<pod-ip>:/workspace/repo/data/` | | ⬜ |
| 6.4 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> data/cache/sequences.json root@<pod-ip>:/workspace/repo/data/cache/` | | ⬜ |
| 6.5 | `python -m esm2_mech.experiments.geometry.conservation_axis --extract` | `data/conservation_pathogenicity.npy`, `data/conservation_pathogenicity_meta.json` | ⬜ |
| 6.6 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> root@<pod-ip>:/workspace/repo/data/conservation_pathogenicity.npy root@<pod-ip>:/workspace/repo/data/conservation_pathogenicity_meta.json data/` | | ⬜ |
| 6.7 | `python -m esm2_mech.experiments.geometry.conservation_axis` | `results/run_biorxiv/magnitude_direction/conservation_axis.json` | ⬜ |

## 7. Experiment: Megascale stability positive control

A second positive control, using physical protein-stability measurements instead of clinical
labels, to confirm the embeddings carry signal independent of ClinVar curation.

| Step | Command | Outputs | Status |
|---|---|---|---|
| 7.1 | `python -m esm2_mech.experiments.stability.build_domain_families` | `data/megascale_domain_families.json` | ⬜ |
| 7.2 | `python -m esm2_mech.experiments.stability.megascale_stability` | `results/run_biorxiv/megascale_stability/per_protein_spearman.json`, `h3_stability_projection.json`, `summary.json` | ⬜ |
| 7.3 | `python -m esm2_mech.experiments.stability.megascale_mlp --xgboost` | `results/run_biorxiv/megascale_stability/mlp_summary_xgb.json` | ⬜ |
| 7.4 | `python -m esm2_mech.experiments.stability.stability_baselines` | `results/run_biorxiv/megascale_stability/baselines.json` | ⬜ |

## 8. Experiment: Enzyme type classification (positive control)

A third positive control classifying each gene as kinase, protease, oxidoreductase, or non-enzyme
from its WT mean-pooled ESM-2 embedding.

| Step | Command | Outputs | Status |
|---|---|---|---|
| 8.1 | `python -m esm2_mech.experiments.proteome_features.enzyme_classification --seeds 5` | `results/run_biorxiv/enzyme_classification/enzyme_classification_summary.json` | ⬜ |

## Verification checklist

⬜ Not started.
