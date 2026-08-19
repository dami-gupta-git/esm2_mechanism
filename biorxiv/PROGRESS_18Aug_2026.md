# run_biorxiv progress — re-run of 2026-08-18

Live status record for `RUNBOOK_biorxiv.md`, which holds the steps only. Each section here mirrors
a section of the runbook, in the same order.

This file replaces the previous `PROGRESS.md`. Every step is reset because the shared scoring code
is being changed: ranking metrics were computed on probabilities pooled across independently
fitted folds, which distorts every interval that depends on them. The diagnosis is in
[`exp4_issues.md`](issues/exp4_issues.md) and the work order in [`exp4_fixes.md`](issues/exp4_fixes.md).

Every analysis script computes the confidence intervals. A confidence interval is a range around a score
saying how much that score would plausibly move if the same analysis were re-run on a different
but similarly-drawn sample of genes, rather than reporting a single number as if it were exact.

## Code fix — must complete before any experiment runs

Steps are in [`exp4_fixes.md`](issues/exp4_fixes.md). Sections 4 through 8 all depend on this.

| Step | Item | Status | Notes |
|---|---|---|---|
| F.0 | Rare-class rule decided with project owner | ✅ 2026-08-18 | Strict: every fold must score the class; draws where one cannot are discarded |
| F.1 | Fold index carried through the out-of-fold collector | ✅ 2026-08-18 | Fold argument is mandatory, so nothing can fall back to pooling |
| F.2 | Ranking metrics scored within fold | ⬜ | |
| F.3 | Permutation shuffling constrained within fold | ⬜ | |
| F.4 | Private fold loops folded into the shared helper | ⬜ | Mechanism and stability scripts |
| F.5 | Family probe resamples genes within families, not families | ⬜ | |
| F.6 | Leakage headline and interval computed on the same basis | ⬜ | |
| F.7 | Call sites and tests updated; regression test added | ⬜ | |
| F.8 | Preregistration amendment committed | ⬜ | Four items; must predate any output being inspected |

## Prerequisites — manually placed files

These are the source files the pipeline needs but cannot fetch itself; they must be placed in
`data/downloads/` before anything else runs.

| # | Item | Command | Outputs | Status | Notes |
|---|---|---|---|---|---|
| 1 | `DiseaseMech_Stability_VEPS.xlsx` | *(manually placed)* | `data/downloads/DiseaseMech_Stability_VEPS.xlsx` | ✅ reused | |
| 2 | `AllG2P.csv` | *(manually placed)* | `data/downloads/AllG2P.csv` | ✅ reused | |

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
| 1.1 | `python -m esm2_mech.fetch_data.build_gene_list` | `downloads/DiseaseMech_Stability_VEPS.xlsx`, `downloads/AllG2P.csv` | `data/gene_list.tsv` | ✅ reused | |

## 2. Fetch variant data

Shared foundation for sections 4, 5, 6, and 7.

Reused from the previous run. Nothing in the fetch or merge path is affected by the scoring fix.
One check before relying on them: the path-resolution change of 2026-08-18 altered how the project
root is located, so confirm these files resolve to the same locations they were written to.

| Step | Command | Outputs | Status | Notes |
|---|---|---|---|---|
| 2.1 | `python -m esm2_mech.fetch_data.fetch_variants --step gerasimavicius` | `gerasimavicius_variants.json` | ✅ reused | |
| 2.2 | `python -m esm2_mech.fetch_data.fetch_variants --step clinvar` | `clinvar_variants.tsv` | ✅ reused | |
| 2.3 | `python -m esm2_mech.fetch_data.fetch_variants --step merge --pathogenic_only` | `variants.json` | ✅ reused | |
| 2.4 | `python -m esm2_mech.fetch_data.fetch_sequences` | `cache/sequences.json` | ✅ reused | |
| 2.5 | `python -m esm2_mech.fetch_data.fetch_annotations --step pfam` | `pfam_families.json` | ✅ reused | |
| 2.6 | `python -m esm2_mech.fetch_data.fetch_alphamissense_mechanism` | `alphamissense_scores_full.json` | ✅ reused | |
| 2.7 | `python -m esm2_mech.fetch_data.build_valid_variants` | `valid_variants.json` | ✅ reused | |
| 2.8 | `python -m esm2_mech.fetch_data.fetch_pathogenicity_variants` | `clinvar_pathogenicity_variants.json`, `clinvar_pathogenicity_variants.params.json` | ✅ reused | |

## 3. Embed variants

Shared foundation for sections 4 and 6.

Embeddings are reused. The scoring fix changes how metrics are computed from predictions, not how
the embeddings are produced, so re-extracting them would return identical arrays at significant
GPU cost.

| Step | Command | Status | Notes |
|---|---|---|---|
| 3.1 | `scp valid_variants.json` to pod | ✅ reused | |
| 3.2 | `scp sequences.json` to pod | ✅ reused | |
| 3.3 | `python -m esm2_mech.embeddings.embed_variants --model esm2_t33_650M_UR50D` | ✅ reused | |
| 3.4 | Copy embeddings back to local | ✅ reused | |

## 4. Experiment: ESM-2 delta-embedding mechanism

Tests whether ESM-2 embeddings can predict a variant's mechanism (DN/LOF/GOF), and checks that the
model isn't just recognizing which protein family the gene belongs to rather than actually
learning something about the mechanism.

| Step | Command | Outputs | Status |
|---|---|---|---|
| 4.1 | `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 5` | `results/run_biorxiv/family_split_baselines_seed{0..4}.json`, `aggregate.json` | ✅ 2026-08-18 | Re-ran on pod (GPU box, CPU-bound step) after the bootstrap discard-reason fix. Log: `logs/biorxiv_18Aug_2026/step_4_1.log` |
| 4.2 | `python -m esm2_mech.experiments.mechanism.mlp --seeds 5` | `results/run_biorxiv/nonlinear_results_seed{0..4}.json` | ✅ 2026-08-18 | Ran on pod. Log: `logs/biorxiv_18Aug_2026/step_4_2.log` |
| 4.3 | `python -m esm2_mech.experiments.mechanism.family_clustering --seeds 5` | `results/run_biorxiv/family_clustering.json` | ✅ 2026-08-18 | Ran on pod. Log: `logs/biorxiv_18Aug_2026/step_4_3.log` |
| 4.4 | `python -m esm2_mech.experiments.mechanism.naive_baseline` | `results/run_biorxiv/naive_baseline.json` | ✅ 2026-08-18 | Ran on pod. Log: `logs/biorxiv_18Aug_2026/step_4_4.log` |
| 4.5 | `python -m esm2_mech.experiments.mechanism.leakage_fraction` | `results/run_biorxiv/leakage_fraction.json` | ⚠️ 2026-08-18 | Ran on pod. Log: `logs/biorxiv_18Aug_2026/step_4_5.log`. Leakage-fraction CIs for `wt_only_mean`, `mut_only_mean`, `wt_concat_mut` discard 99.4% of bootstrap resamples (994/1000) — far above the <1% tolerance `exp4_fixes.md` set as a fault threshold. Needs investigation before these CIs are used. |
| 4.6 | `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 5 --n_permutations 1000` | `results/run_biorxiv/...` | ⬜ | Five seeds, not one — the previous run used seed 0 only |
| 4.7 | `python -m esm2_mech.experiments.mechanism.single_source_mechanism --seeds 5` | `results/run_biorxiv/single_source_gerasimavicius/...` | ⬜ | |
| 4.8 | Recompute the 2A threshold from the fixed nonlinear delta score | *(recorded in the preregistration amendment)* | ⬜ | Must precede any adjudication of 2A |

## 5. Experiment: Pathogenicity positive control

Tests whether the same embeddings can at least tell pathogenic from benign variants, to confirm
they carry usable signal at all.

| Step | Command | Outputs | Status |
|---|---|---|---|
| 5.1 | `scp clinvar_pathogenicity_variants.json` to pod | | ✅ 2026-08-18 | Refetched — cached copy predated the per-gene balancing fix (`balance_version=1`); 25,858 variants, 1,837 genes, balanced 12,929/12,929 |
| 5.2 | `scp pfam_families.json` to pod | | ✅ 2026-08-18 | |
| 5.3 | `python -m esm2_mech.experiments.pathogenicity.pathogenicity_control --phase embed --model esm2_t33_650M_UR50D` | `pathogenicity_{wt,mut}_mean.npy`, `pathogenicity_meta.json` | ✅ 2026-08-18 | Re-embedded on pod against the refetched variant set; 24,516 variant pairs |
| 5.4 | Copy embeddings back to local | | ✅ 2026-08-18 | |
| 5.5 | `python -m esm2_mech.experiments.pathogenicity.pathogenicity_control --phase probe` | `results/run_biorxiv/pathogenicity_control.json` | ✅ 2026-08-18 | delta_mean MLP AUROC 0.885 (family-split), 0.885 (gene-split) — passes ≥0.85 gate; wt_only ~0.52 (chance) |

## 6. Experiment: Geometry of the pathogenicity direction

Asks what the pathogenicity direction in embedding space actually corresponds to.

| Step | Command | Outputs | Status |
|---|---|---|---|
| 6.1 | `python -m esm2_mech.experiments.geometry.build_canonical_pathogenicity` | `data/pathogenicity_valid_variants_canonical.json` | ✅ 2026-08-18 | Rebuilt — section 5's refetch/re-embed changed the row set; 24,516 variants, matches embedding row count |
| 6.2 | `python -m esm2_mech.experiments.geometry.run_geometry --seeds 5 --stability-dataset tsuboyama` | `results/run_biorxiv/magnitude_direction/{probe_results,geometry_results,transfer_contrast,probe4_axis_identity}.json` | 🔄 running | On pod, tmux `step6_2`. Log: `logs/biorxiv_18Aug_2026/step_6_2.log` |
| 6.5 | `python -m esm2_mech.experiments.geometry.conservation_axis --extract` | `data/conservation_pathogenicity.npy`, `data/conservation_pathogenicity_meta.json` | 🔄 running | Re-extraction against the new canonical variant list. On pod (GPU), tmux `step6_5`. Log: `logs/biorxiv_18Aug_2026/step_6_5.log` |
| 6.7 | `python -m esm2_mech.experiments.geometry.conservation_axis` | `results/run_biorxiv/magnitude_direction/conservation_axis.json` | ⬜ | Previous output also used obsolete K1/K2/C4 identifiers; confirm the numbering correction is in before running |

## 7. Experiment: Megascale stability positive control

A second positive control, using physical protein-stability measurements instead of clinical
labels, to confirm the embeddings carry signal independent of ClinVar curation.

| Step | Command | Outputs | Status |
|---|---|---|---|
| 7.1 | `python -m esm2_mech.experiments.stability.build_domain_families` | `data/megascale_domain_families.json` | ✅ reused | |
| 7.2 | `python -m esm2_mech.experiments.stability.megascale_stability --n_jobs 4` | `results/run_biorxiv/megascale_stability/per_protein_spearman.json`, `stability_projection_3c.json`, `summary.json` | ⬜ | Previous output also used obsolete H2/H3 identifiers; confirm the numbering correction is in before running |
| 7.3 | `python -m esm2_mech.experiments.stability.megascale_mlp` | `results/run_biorxiv/megascale_stability/mlp_summary.json` | ⬜ | cuML GPU RF |
| 7.4 | `python -m esm2_mech.experiments.stability.megascale_mlp --xgboost` | `results/run_biorxiv/megascale_stability/mlp_summary_xgb.json` | ⬜ | GPU XGBoost |
| 7.5 | `python -m esm2_mech.experiments.stability.stability_baselines --n_jobs 64` | `results/run_biorxiv/megascale_stability/baselines.json` | ⬜ | |

## 8. Experiment: Enzyme type classification (positive control)

A third positive control classifying each gene as kinase, protease, oxidoreductase, or non-enzyme
from its WT mean-pooled ESM-2 embedding.

| Step | Command | Outputs | Status |
|---|---|---|---|
| 8.1 | `python -m esm2_mech.experiments.proteome_features.enzyme_classification --seeds 5` | `results/run_biorxiv/enzyme_classification/enzyme_classification_summary.json` | ⬜ | Previous result predated the commit fixing its decision-rule labels, so it required re-running regardless |

## Reports

Regenerate rather than editing numbers in place.

| Step | Item | Status | Notes |
|---|---|---|---|
| R.1 | Regenerate all `reports/run_biorxiv/` reports from the new results | ⬜ | |
| R.2 | Mark `reports/run_biorxiv/bak/` reports superseded in their own text | ⬜ | Produced by pre-fix code |

## Verification checklist

⬜ Not started.

Acceptance criteria for the code fix are in [`exp4_fixes.md`](issues/exp4_fixes.md). The load-bearing
ones: the delta's per-class family-split AUROCs sit near their per-fold values rather than below
0.5, the permutation null centres near 0.5, and every reported interval brackets the point
estimate it is attached to.
