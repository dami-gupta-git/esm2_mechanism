# run_biorxiv progress

Live status record for `RUNBOOK_biorxiv.md`, which holds the steps only. Each section here mirrors
a section of the runbook, in the same order, with its own table in the style of `RUN_PROGRESS.md`.  

Every analysis script computes the confidence intervals. A confidence interval is a range around a score
saying how much that score would plausibly move if the same analysis were re-run on a different
but similarly-drawn sample of genes, rather than reporting a single number as if it were exact.

## Prerequisites — manually placed files

These are the source files the pipeline needs but cannot fetch itself; they must be placed in
`data/downloads/` before anything else runs.

| # | Item | Command | Outputs | Status | Notes |
|---|---|---|---|---|---|
| 1 | `DiseaseMech_Stability_VEPS.xlsx` | *(manually placed)* | `data/downloads/DiseaseMech_Stability_VEPS.xlsx` | ✅ 2026-08-14 | |
| 2 | `AllG2P.csv` | *(manually placed)* | `data/downloads/AllG2P.csv` | ✅ 2026-08-14 | |

## 0. Preconditions

These are the checks and setup steps that must all pass before the run is allowed to start.

| Step | Item | Command | Status | Notes |
|---|---|---|---|---|
| 0.0 | Environment setup | `cd /Users/dgupta/code/portfolio/ESM2/esm2_mechanism`<br>`python3 -m venv .venv && source .venv/bin/activate`<br>`pip install -e .` | ✅ 2026-08-14 | |
| 0.1 | Pathogenicity provenance | *(verification, no command)* | ✅ 2026-08-14 | Locked to one canonical variant set; `pathogenicity_control.py` fingerprints it |
| 0.2 | Stats machinery wired | *(verification, no command)* | ✅ 2026-08-14 | Every result-producing script wired to `utils/bootstrap.py`, emits CI keys |
| 0.3 | Methodology rules | *(verification, no command)* | ✅ 2026-08-14 | Rule 3/Rule 4 implemented |
| 0.4 | Paired cluster bootstrap | *(verification, no command)* | ✅ 2026-08-14 | Wired at `conservation_axis.py`, `mechanism_delta_family_split.py` |
| 0.5/0.6 | Pre-registered decision rules | *(verification, no command)* | ✅ 2026-08-14 | CI decision rule and confirmatory/exploratory split recorded |
| 0.7 | Pinned environment | *(verification, no command)* | ✅ 2026-08-14 | `pytest tests/` green |
| 0.8 | Configuration | *(config change in `utils/paths.py:11`)* | ✅ 2026-08-14 | `RUN_NAME` flipped `"run6"` → `"run_biorxiv"` |
| 0.9 | Working tree clean | `git status` | ✅ 2026-08-14 | |

## 1. Build gene list

Builds the list of genes every later experiment uses.

| Step | Command | Inputs | Outputs | Status | Notes |
|---|---|---|---|---|---|
| 1.1 | `python -m esm2_mech.fetch_data.build_gene_list` | `downloads/DiseaseMech_Stability_VEPS.xlsx`, `downloads/AllG2P.csv` | `data/gene_list.tsv` | ✅ 2026-08-14 | |

## 2. Fetch variant data

Shared foundation for sections 4, 5, 6, and 7.

| Step | Command | Outputs | Status | Notes |
|---|---|---|---|---|
| 2.1 | `python -m esm2_mech.fetch_data.fetch_variants --step gerasimavicius` | `gerasimavicius_variants.json` | ✅ 2026-08-11 | |
| 2.2 | `python -m esm2_mech.fetch_data.fetch_variants --step clinvar` | `clinvar_variants.tsv` | ✅ 2026-08-12 | |
| 2.3 | `python -m esm2_mech.fetch_data.fetch_variants --step merge --pathogenic_only` | `variants.json` | ✅ 2026-08-12 | |
| 2.4 | `python -m esm2_mech.fetch_data.fetch_sequences` | `cache/sequences.json` | ✅ 2026-08-12 | |
| 2.5 | `python -m esm2_mech.fetch_data.fetch_annotations --step pfam` | `pfam_families.json` | ✅ 2026-08-12 | |
| 2.6 | `python -m esm2_mech.fetch_data.fetch_alphamissense_mechanism` | `alphamissense_scores_full.json` | ✅ 2026-08-12 | |
| 2.7 | `python -m esm2_mech.fetch_data.build_valid_variants` | `valid_variants.json` | ✅ 2026-08-12 | |
| 2.8 | `python -m esm2_mech.fetch_data.fetch_pathogenicity_variants` | `clinvar_pathogenicity_variants.json`, `clinvar_pathogenicity_variants.params.json` | ✅ 2026-08-14 | Fetches balanced pathogenic/benign variants for section 5, separate from step 2.2's pathogenic-only fetch used for mechanism labels; ran locally (network-only, no GPU needed) |

**Results (2026-08-11/12 fetch):**

| Step | Count |
|---|---|
| Gerasimavicius | 10,233 variants / 948 genes |
| ClinVar | 48,152 rows / 2,115 genes |
| Merged `variants.json` | 17,865 variants, 1,937 genes (gerasimavicius=10,233, clinvar_g2p=7,632) |
| Sequences fetched | 1,935 genes |
| Pfam | 1,913/1,937 genes annotated, 24 unannotated |
| AlphaMissense matched | 17,765 variants |
| `valid_variants.json` | 17,770 rows |

A WT-mismatch check on this fetch flagged 9 genes in the Gerasimavicius set — see
[`FINDINGS.md`](../docs/FINDINGS.md#wt-mismatch-check-flagged-9-genes-in-the-gerasimavicius-set-2026-08-12).

## 3. Embed variants

Shared foundation for sections 4 and 6.

| Step | Command | Status | Notes |
|---|---|---|---|
| 3.1 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> data/valid_variants.json root@<pod-ip>:/workspace/repo/data/` | ✅ 2026-08-14 | |
| 3.2 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> data/cache/sequences.json root@<pod-ip>:/workspace/repo/data/cache/` | ✅ 2026-08-14 | |
| 3.3 | `python -m esm2_mech.embeddings.embed_variants --model esm2_t33_650M_UR50D` | ✅ 2026-08-14 | Ran on pod. Outputs: `embeddings_{wt,mut}_{mean,pos}.npy`, `embedded_variants.json` |
| 3.4 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> root@<pod-ip>:/workspace/repo/data/embeddings/esm2_t33_650M_UR50D/*.npy root@<pod-ip>:/workspace/repo/data/embeddings/esm2_t33_650M_UR50D/embedded_variants.json data/embeddings/esm2_t33_650M_UR50D/` | ✅ 2026-08-14 | All four arrays and `embedded_variants.json` have 17,770 rows, matching `valid_variants.json`; spot-checked rows 0, 100, 5000, 17769 on gene/uniprot_id/position/wt/mut — all match |

## 4. Experiment: ESM-2 delta-embedding mechanism

Tests whether ESM-2 embeddings can predict a variant's mechanism (DN/LOF/GOF), and checks that the
model isn't just recognizing which protein family the gene belongs to rather than actually
learning something about the mechanism. This ran on a rented 32-core computer.

| Step | Command | Outputs | Status |
|---|---|---|---|
| 4.1 | `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 5` | `results/run_biorxiv/family_split_baselines_seed{0..4}.json`, `aggregate.json` | ✅ 2026-08-14 |
| 4.2 | `python -m esm2_mech.experiments.mechanism.mlp --seeds 5` | `results/run_biorxiv/nonlinear_results_seed{0..4}.json` | ✅ 2026-08-14 |
| 4.3 | `python -m esm2_mech.experiments.mechanism.family_clustering --seeds 5` | `results/run_biorxiv/family_clustering.json` | ✅ 2026-08-14 |
| 4.4 | `python -m esm2_mech.experiments.mechanism.naive_baseline` | `results/run_biorxiv/naive_baseline.json` | ✅ 2026-08-14 |
| 4.5 | `python -m esm2_mech.experiments.mechanism.leakage_fraction` | `results/run_biorxiv/leakage_fraction.json` | ✅ 2026-08-14 |
| 4.6 | `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 1 --n_permutations 1000` | `results/run_biorxiv/backup_step2_permutation_seed0.json` (renamed from `family_split_baselines_seed0_step2_permutation.json` — see note 8) | ✅ 2026-08-14 |
| 4.7 | `python -m esm2_mech.experiments.mechanism.single_source_mechanism --seeds 5` | `results/run_biorxiv/single_source_gerasimavicius/...` | ✅ 2026-08-15 |

**Summary:** ESM-2 embeddings do not encode disease mechanism. Full write-up: [`reports/run_biorxiv/report_mechanism.md`](../reports/run_biorxiv/report_mechanism.md).

## 5. Experiment: Pathogenicity positive control

Tests whether the same embeddings can at least tell pathogenic from benign variants, to confirm
they carry usable signal at all. The embedding step needed a GPU, so it ran on a rented H100 pod.

| Step | Command | Outputs | Status |
|---|---|---|---|
| 5.1 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> data/clinvar_pathogenicity_variants.json root@<pod-ip>:/workspace/repo/data/` | | ✅ 2026-08-14 |
| 5.2 | `python -m esm2_mech.experiments.pathogenicity.pathogenicity_control --model esm2_t33_650M_UR50D` | `results/run_biorxiv/pathogenicity_control.json`, `results/run_biorxiv/pathogenicity_control_seed{0..4}.json`, `data/embeddings/esm2_t33_650M_UR50D/pathogenicity_{wt,mut}_mean.npy` | ✅ 2026-08-14 |
| 5.3 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> root@<pod-ip>:/workspace/repo/data/embeddings/esm2_t33_650M_UR50D/pathogenicity_*.npy root@<pod-ip>:/workspace/repo/data/embeddings/esm2_t33_650M_UR50D/pathogenicity_meta.json root@<pod-ip>:/workspace/repo/results/run_biorxiv/pathogenicity_control.json data/embeddings/esm2_t33_650M_UR50D/` | | ✅ 2026-08-14 |

**Summary:** ESM-2 embeddings separate pathogenic from benign variants well above the 0.85 pass bar, confirming the pipeline recovers known signal. Full write-up: [`reports/run_biorxiv/report_pathogenicity_control.md`](../reports/run_biorxiv/report_pathogenicity_control.md).


## 6. Experiment: Geometry of the pathogenicity direction

Asks what the pathogenicity direction in embedding space actually corresponds to.

| Step | Command | Outputs | Status |
|---|---|---|---|
| 6.1 | `python -m esm2_mech.experiments.geometry.build_canonical_pathogenicity` | `data/pathogenicity_valid_variants_canonical.json` | ✅ 2026-08-15 |
| 6.2 | `python -m esm2_mech.experiments.geometry.run_geometry --seeds 5` | `results/run_biorxiv/magnitude_direction/{probe_results,geometry_results,transfer_contrast,probe4_axis_identity}.json` | ✅ 2026-08-15 |
| 6.3 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> data/pathogenicity_valid_variants_canonical.json root@<pod-ip>:/workspace/repo/data/` | | ✅ 2026-08-17 |
| 6.4 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> data/cache/sequences.json root@<pod-ip>:/workspace/repo/data/cache/` | | ✅ 2026-08-17 |
| 6.5 | `python -m esm2_mech.experiments.geometry.conservation_axis --extract` | `data/conservation_pathogenicity.npy`, `data/conservation_pathogenicity_meta.json` | ✅ 2026-08-17 |
| 6.6 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> root@<pod-ip>:/workspace/repo/data/conservation_pathogenicity.npy root@<pod-ip>:/workspace/repo/data/conservation_pathogenicity_meta.json data/` | | ✅ 2026-08-17 |
| 6.7 | `python -m esm2_mech.experiments.geometry.conservation_axis` | `results/run_biorxiv/magnitude_direction/conservation_axis.json` | ✅ 2026-08-17 |

**Notes:**

1. Step 6.1 re-indexes section 5's pathogenicity variant list down to the 37,258 variants that were actually embedded, in the same row order as the embedding arrays, so later steps can read variant details and embeddings together without a separate lookup. Ran locally on CPU (no model, no GPU): 38,797 variants in, 37,258 written out, matching the embedding row count exactly.

## 7. Experiment: Megascale stability positive control

A second positive control, using physical protein-stability measurements instead of clinical
labels, to confirm the embeddings carry signal independent of ClinVar curation.

| Step | Command | Outputs | Status |
|---|---|---|---|
| 7.1 | `python -m esm2_mech.experiments.stability.build_domain_families` | `data/megascale_domain_families.json` | ⬜ |
| 7.2 | `python -m esm2_mech.experiments.stability.megascale_stability` | `results/run_biorxiv/megascale_stability/per_protein_spearman.json`, `h3_stability_projection.json`, `summary.json` | ⬜ |
| 7.3 | `python -m esm2_mech.experiments.stability.megascale_mlp --xgboost` | `results/run_biorxiv/megascale_stability/mlp_summary_xgb.json` | ⬜ |
| 7.4 | `python -m esm2_mech.experiments.stability.stability_baselines` | `results/run_biorxiv/megascale_stability/baselines.json` | ⬜ |

## Verification checklist

Final checks confirming the run's data and statistics are correct before its reports are written.

⬜ Not started.
