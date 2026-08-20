# run_biorxiv progress

Live status record for `RUNBOOK_biorxiv.md`, which holds the steps only. Each section here mirrors
a section of the runbook, in the same order.

Every analysis script computes the confidence intervals. A confidence interval is a range around a score
saying how much that score would plausibly move if the same analysis were re-run on a different
but similarly-drawn sample of genes, rather than reporting a single number as if it were exact.

## Prerequisites — manually placed files

| # | Item | Outputs | Status |
|---|---|---|---|
| 1 | `DiseaseMech_Stability_VEPS.xlsx` | `data/downloads/DiseaseMech_Stability_VEPS.xlsx` | ✅ |
| 2 | `AllG2P.csv` | `data/downloads/AllG2P.csv` | ✅ |

## 0. Preconditions

| Step | Item | Status | Notes |
|---|---|---|---|
| 0.0 | Environment setup | ✅ 2026-08-19 | Local and pod environments initialized; versions are recorded in `ENV_SNAPSHOT.md`. |
| 0.1 | Pathogenicity provenance | ✅ 2026-08-19 | Variant and embedding provenance validation passed. |
| 0.2 | Stats machinery wired | ✅ 2026-08-19 | Confidence-interval and paired-gap outputs were verified. |
| 0.3 | Methodology rules | ✅ 2026-08-19 | Resampling and rare-class rules were verified in the result outputs. |
| 0.4 | Paired cluster bootstrap | ✅ 2026-08-19 | Paired bootstrap outputs were verified for claims 2B and 2E. |
| 0.5/0.6 | Pre-registered decision rules | ✅ 2026-08-19 | Decision rules and the confirmatory/exploratory split are recorded in `PREREGISTRATION_run_biorxiv.md`. |
| 0.7 | Pinned environment | ✅ 2026-08-19 | Full test suite: 758 passed, 1 skipped, 1 xfailed; package versions are recorded in `ENV_SNAPSHOT.md`. |
| 0.8 | Configuration (`RUN_NAME` flip) | ✅ 2026-08-19 | `RUN_NAME = "run_biorxiv"`; registered settings were verified. |
| 0.9 | Working tree clean | ✅ 2026-08-19 | Clean at the run branch point; final repository state is tracked in the verification checklist. |
| 0.10 | Megascale embedding provenance | ✅ 2026-08-19 | Four aligned 177,315 × 1,280 arrays and their extraction fingerprint were verified. |

## 1. Build gene list

| Step | Command | Outputs | Status |
|---|---|---|---|
| 1.1 | `python -m esm2_mech.fetch_data.build_gene_list` | `data/gene_list.tsv` | ✅ 2026-08-19 |

## 2. Fetch variant data

| Step | Command | Outputs | Status |
|---|---|---|---|
| 2.1 | `python -m esm2_mech.fetch_data.fetch_variants --step gerasimavicius` | `gerasimavicius_variants.json` | ✅ 2026-08-19 |
| 2.2 | `python -m esm2_mech.fetch_data.fetch_variants --step clinvar` | `clinvar_variants.tsv` | ✅ 2026-08-19 |
| 2.3 | `python -m esm2_mech.fetch_data.fetch_variants --step merge --pathogenic_only` | `variants.json` | ✅ 2026-08-19 |
| 2.4 | `python -m esm2_mech.fetch_data.fetch_sequences` | `cache/sequences.json` | ✅ 2026-08-19 |
| 2.5 | `python -m esm2_mech.fetch_data.fetch_annotations --step pfam` | `pfam_families.json` | ✅ 2026-08-19 |
| 2.6 | `python -m esm2_mech.fetch_data.fetch_alphamissense_mechanism` | `alphamissense_scores_full.json` | ✅ 2026-08-19 |
| 2.7 | `python -m esm2_mech.fetch_data.build_valid_variants` | `valid_variants.json` | ✅ 2026-08-19 |
| 2.8 | `python -m esm2_mech.fetch_data.fetch_pathogenicity_variants --max_per_gene_per_class 20 --fetch_seed 42 --force` | `clinvar_pathogenicity_variants.json`, `clinvar_pathogenicity_variants.params.json` | ✅ 2026-08-19 |
| 2.9 | `python -m esm2_mech.fetch_data.fetch_annotations --step enzyme` | `data/enzyme_labels.tsv` | ✅ 2026-08-19 |

## 3. Embed variants

| Step | Command | Status |
|---|---|---|
| 3.1 | `scp valid_variants.json` to pod | ✅ 2026-08-19 |
| 3.2 | `scp sequences.json` to pod | ✅ 2026-08-19 |
| 3.3 | `python -m esm2_mech.embeddings.embed_variants --model esm2_t33_650M_UR50D` | ✅ 2026-08-19 |
| 3.4 | Copy embeddings back to local | ✅ 2026-08-19 |

## 4. Experiment: ESM-2 delta-embedding mechanism

| Step | Command | Outputs | Status |
|---|---|---|---|
| 4.1 | `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 5` | `results/run_biorxiv/family_split_baselines_seed{0..4}.json`, `mechanism_oof_cache_seed{0..4}.json` | ✅ 2026-08-19 |
| 4.2 | `python -m esm2_mech.experiments.mechanism.mlp --seeds 5` | `results/run_biorxiv/nonlinear_results_seed{0..4}.json` | ✅ 2026-08-19 |
| 4.3 | `python -m esm2_mech.experiments.mechanism.family_clustering --seeds 5` | `results/run_biorxiv/family_clustering.json` | ✅ 2026-08-19 |
| 4.4 | `python -m esm2_mech.experiments.mechanism.naive_baseline` | `results/run_biorxiv/naive_baseline.json` | ✅ 2026-08-19 |
| 4.7 | `python -m esm2_mech.experiments.mechanism.single_source_mechanism --seeds 5` | `results/run_biorxiv/single_source_gerasimavicius/...` | ✅ 2026-08-19 |
| 4.6 | `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 5 --n_permutations 1000` | `results/run_biorxiv/...` | ✅ 2026-08-19 |
| 4.5 | `python -m esm2_mech.experiments.mechanism.leakage_fraction` | `results/run_biorxiv/leakage_fraction.json` | ✅ 2026-08-19 |

## 5. Experiment: Pathogenicity positive control

| Step | Command | Outputs | Status |
|---|---|---|---|
| 5.1 | `scp clinvar_pathogenicity_variants.json` + `.params.json` to pod | | ✅ 2026-08-19 |
| 5.2 | `scp pfam_families.json` to pod | | ✅ 2026-08-19 |
| 5.3 | `python -m esm2_mech.experiments.pathogenicity.pathogenicity_control --phase embed --model esm2_t33_650M_UR50D --force_embed` | `pathogenicity_{wt,mut}_mean.npy`, `pathogenicity_meta.json` | ✅ 2026-08-19 |
| 5.4 | Copy embeddings back to local | | ✅ 2026-08-19 |
| 5.5 | `python -m esm2_mech.experiments.pathogenicity.pathogenicity_control --phase probe --seeds 5 --n_jobs <workers> --n_boot 1000` | `results/run_biorxiv/pathogenicity_control_seed{0..4}.json`, `pathogenicity_control.json` | ✅ 2026-08-19 |

## 6. Experiment: Geometry of the pathogenicity direction

| Step | Command | Outputs | Status |
|---|---|---|---|
| 6.1 | `python -m esm2_mech.experiments.geometry.build_canonical_pathogenicity` | `data/pathogenicity_valid_variants_canonical.json` | ✅ 2026-08-19 |
| 6.2 | `python -m esm2_mech.experiments.geometry.run_geometry --seeds 5 --stability-dataset tsuboyama` | `results/run_biorxiv/magnitude_direction/{probe_results,geometry_results,transfer_contrast,probe4_axis_identity}.json` | ✅ 2026-08-19 |
| 6.3 | `scp pathogenicity_valid_variants_canonical.json` to pod | | ✅ 2026-08-19 |
| 6.4 | `scp sequences.json` to pod | | ✅ 2026-08-19 |
| 6.5 | `python -m esm2_mech.experiments.geometry.conservation_axis --extract` | `data/conservation_pathogenicity.npy`, `data/conservation_pathogenicity_meta.json` | ✅ 2026-08-19 |
| 6.6 | Copy conservation outputs back to local | | ✅ 2026-08-19 |
| 6.7 | `python -m esm2_mech.experiments.geometry.conservation_axis` | `results/run_biorxiv/magnitude_direction/conservation_axis.json` | ✅ 2026-08-19 |

## 7. Experiment: Megascale stability positive control

| Step | Command | Outputs | Status |
|---|---|---|---|
| 7.1 | `python -m esm2_mech.experiments.stability.build_domain_families` | `data/megascale_domain_families.json` | ✅ 2026-08-19 |
| 7.2 | `python -m esm2_mech.experiments.stability.megascale_stability --n_jobs 4` | `results/run_biorxiv/megascale_stability/per_protein_spearman.json`, `stability_projection_3c.json`, `summary.json` | ✅ 2026-08-19 |
| 7.3 | `python -m esm2_mech.experiments.stability.megascale_mlp` | `results/run_biorxiv/megascale_stability/mlp_summary.json` | ✅ 2026-08-19 |
| 7.4 | `python -m esm2_mech.experiments.stability.megascale_mlp --xgboost` | `results/run_biorxiv/megascale_stability/mlp_summary_xgb.json` | ✅ 2026-08-19 |
| 7.5 | `python -m esm2_mech.experiments.stability.stability_baselines --n_jobs 4` | `results/run_biorxiv/megascale_stability/baselines.json` | ✅ 2026-08-19 |

## 8. Experiment: Enzyme type classification (positive control)

| Step | Command | Outputs | Status |
|---|---|---|---|
| 8.1 | `python -m esm2_mech.experiments.proteome_features.enzyme_classification --seeds 5` | `results/run_biorxiv/enzyme_classification/enzyme_classification_summary.json` | ✅ 2026-08-19 |

## Dataset and verification records

### Dataset construction

- Step 2.2 retrieved ClinVar rows for 2,115 of 2,376 genes. The other 261 genes were confirmed to
  have no missense pathogenic or likely pathogenic records; they were not fetch failures.
- Step 2.3 produced 17,865 mechanism variants across 1,937 genes.
- Step 2.4 retained 1,935 sequences, and step 2.5 assigned Pfam families to 1,913 of 1,937 genes.
- Step 2.6 matched AlphaMissense scores for 17,765 of 17,840 variants. Step 2.7 produced 17,770
  schema-validated variants across 1,931 genes.
- Step 2.8 produced 25,740 pathogenicity variants across 1,837 genes, balanced between 12,870
  pathogenic and 12,870 benign variants. IDS F155L had conflicting labels across ClinVar records
  and was excluded during deduplication. The balance and fetch-metadata versions were increased to 3.
- Step 2.9 regenerated the enzyme labels with the required exclusion flag. All 1,935 UniProt entries
  were found, and 2,376 rows were written: 130 kinase, 68 protease, 123 oxidoreductase, 1,135
  non-enzyme, 481 excluded other-enzyme, and 439 missing labels.

### Provenance and quality control

- Steps 3.3–3.4 reused an embedding checkpoint whose fingerprint matched `valid_variants.json`.
  All four embedding arrays and `embedded_variants.json` contain the same 17,770 aligned rows.
- Step 4.7 discarded 2–3% of bootstrap resamples for some intervals because a fold lost a class in
  the small single-source subset. Those intervals require review before citation.
- Step 4.5 suppressed the `delta_per_residue` interval because too few valid bootstrap resamples
  remained.
- Steps 5.3–5.4 embedded 24,384 of 25,740 pathogenicity variants. Variants without a usable sequence
  or position were excluded. Both 24,384 × 1,280 arrays match the current variant fingerprint.
- Steps 6.1 and 6.5 used the same 24,384 aligned pathogenicity variants, and conservation scores were
  obtained for every row.
- Step 7.1 used the existing non-empty domain-family file, as allowed by the runbook. Step 7.2 was
  rerun at clean commit `6937c85`; its result provenance records `commit_dirty: false`, and the files
  were copied locally.
- Step 7.3 completed at clean commit `6937c85`. Its result and execution log were copied locally,
  its input fingerprints match the verified step 7.2 inputs, and its environment is recorded in
  `ENV_SNAPSHOT.md`.
- Step 7.4 was rerun after fingerprint tracking was added. Its input fingerprints match the current
  step 7.2 result, and its environment is recorded in `ENV_SNAPSHOT.md`.
- Step 8.1 was run at clean commit `6937c85`. Its result fingerprints match the current enzyme
  cohort, wildtype embeddings, Pfam assignments, proteome features, and mechanism reference. The
  Section 8 report was regenerated from this result, and claims 2F–2H were verified. Its environment
  is recorded in `ENV_SNAPSHOT.md`.

### Active work and manuscript scope

- All experiment steps are complete. The stability result files and logs are present locally, the
  stability report has been regenerated, and claims 3A through 3D have been verified.
- Step 7.5 is complete at clean commit `6937c85`. Its result was copied locally, and its input
  fingerprints match the current stability inputs. It remains exploratory and is not used in the
  manuscript.

## Verification checklist

### Global checks

- [x] Full test suite. The manuscript-freeze suite passed with 758 tests passed, 1 skipped, and
      1 expected failure.
- [x] Final repository state. The manuscript-freeze release is recorded by tag
      `run_biorxiv-manuscript-freeze-2026-08-19` on `main`.
- [x] Execution environments. Every local and pod environment used by the completed steps is recorded
      in `ENV_SNAPSHOT.md`, including the environment used for step 7.3.
- [x] Result fingerprints. Every completed result used by the manuscript was checked against its
      current mechanism, pathogenicity, conservation, enzyme, or stability inputs. All stored
      fingerprints match, including fingerprints stored under legacy nested provenance fields.
- [x] Run comparison. `scripts/compare_runs.py run6 run_biorxiv` flagged 2,090 movements. Every flag
      is assigned to an explained result group in `DELTA_run6_to_run_biorxiv.md`; none remain
      unexplained.
- [x] Report and figure sources. The five active manuscript reports cite only `run_biorxiv` result
      files, and every linked local source exists. Four unreferenced figures from the earlier cohort,
      including the withdrawn homology-partition panel, were moved to `reports/run_biorxiv/bak/figures/`.
- [x] Cross-report numbers and provenance. Claims 2A-1 through 2H and 3A through 3D match their
      cited result files, and all 63 local links in the quantitative reports, literature audit,
      manuscript files, and verification record resolve. The mechanism, pathogenicity, and geometry
      outputs record commit `b502952` with `commit_dirty: true`; their stored scientific-input
      fingerprints match the final audited inputs. Enzyme and stability outputs record clean commit
      `6937c85`. The confirmatory mechanism score of 0.290, exploratory geometry score of 0.387,
      and shared-family mechanism score of 0.280 remain separately labeled by probe and cohort.

### Per-claim checks

Claim verification has started with the linear mechanism-classification criterion.

| Claim | Status |
|---|---|
| 2A-1 | ✅ 2026-08-20 |
| 2A-2 | ✅ 2026-08-20 |
| 2B | ✅ 2026-08-20 |
| 2C | ✅ 2026-08-20 |
| 2D | ✅ 2026-08-20 |
| 2E | ✅ 2026-08-20 |
| 2F | ✅ 2026-08-20 |
| 2G | ✅ 2026-08-20 |
| 2H | ✅ 2026-08-20 |
| 3A | ✅ 2026-08-20 |
| 3B | ✅ 2026-08-20 |
| 3C | ✅ 2026-08-20 |
| 3D | ✅ 2026-08-20 |

Claim 2A-1 was verified against all five `mechanism_oof_cache_seed{0..4}.json` and
`family_split_baselines_seed{0..4}.json` files. The measured family-split floor is 0.289631, so the
registered threshold is 0.339631. All five `delta_mean` interval upper bounds are below the
threshold, ranging from 0.303863 to 0.305453. The report's seed-0 result of 0.290 [0.276, 0.305],
five-seed count, and affirmed verdict match the source files and the registered rule.

Claim 2A-2 was verified against the permutation outputs in all five
`family_split_baselines_seed{0..4}.json` files. The fixed-prediction family-block test uses macro
one-vs-rest AUROC and 1,000 permutations per seed. Four p-values are below 0.05, so the report's
overturned verdict follows the registered three-of-five rule. No result is resolution-limited, and
the report now gives each seed's count of families without a same-size swap partner.

The mechanism and geometry reports were reconciled for their full-delta logistic-regression values.
The confirmatory mechanism value of 0.290 uses the preregistered 256-component PCA probe without
standardization or class weighting. The exploratory geometry value of 0.387 uses all 1,280
dimensions with per-fold standardization and balanced class weights on the same cohort. Both reports
now identify the probe specification beside the result. This clarification does not change any
preregistered verdict.

Claim 2B was verified against the paired `wt_only_mean` split gaps in all five
`family_split_baselines_seed{0..4}.json` files and their `aggregate.json` summary. Four of five
family-bootstrap intervals exclude zero, so the leakage account is supported and is not overturned
under the registered three-of-five rule. The reported gap range, seed intervals, descriptive leakage
fraction, and Gerasimavicius-only seed-0 sensitivity result match their source files.

Claim 2C was verified against `pathogenicity_control_seed0.json` and the five-seed
`pathogenicity_control.json` summary. The seed-0 family-split `delta_mean` MLP AUROC is 0.887601,
with a 1,072-family bootstrap interval of [0.882320, 0.893363]. The lower bound exceeds the
registered 0.85 threshold, so the affirmed verdict is established. The descriptive five-seed mean
is 0.885004, and the reported deduplication, balancing, and exclusion counts match the source.

Claim 2D was verified against `magnitude_direction/conservation_axis.json`. The seed-0
conservation-only family-split AUROC is 0.887626, with a one-arm 1,072-family bootstrap interval of
[0.880693, 0.894730]. The lower bound exceeds the registered 0.85 threshold, so the affirmed
verdict is established. The descriptive five-seed mean is 0.887541, and all five seed fold means
are retained in the result file.

Claim 2E was verified against `magnitude_direction/conservation_axis.json`. On seed 0,
conservation plus the embedding delta minus conservation alone is -0.004738, with a paired
1,072-family bootstrap interval of [-0.008391, -0.001341] over 24,176 shared variants. The point
estimate does not reach the registered +0.02 improvement, and the interval does not span that
threshold. The report's failed, established verdict therefore follows the registered underpowered
rule.

Claim 2F was verified against `enzyme_classification/enzyme_classification_summary.json`. The
seed-0 family-split logistic-regression macro-F1 is 0.787472, with a one-arm 835-family bootstrap
interval of [0.732364, 0.817868]. The lower bound exceeds the registered 0.70 threshold, so the
affirmed verdict is established. The descriptive five-seed mean is 0.778624.

Claim 2G was verified against `enzyme_classification/enzyme_classification_summary.json`, the
current mechanism `aggregate.json`, and the seed-0 mechanism result and OOF cache. Across 835 shared
Pfam clusters, enzyme macro-F1 is 0.787472 and mechanism macro-F1 is 0.280071. Their paired
difference is +0.507401, with a bootstrap interval of [+0.446608, +0.540746]. The point estimate
exceeds the registered +0.05 minimum and the interval excludes zero, so the affirmed verdict is
established.

Claim 2H was verified against `enzyme_classification/enzyme_classification_summary.json`. On seed
0, MLP minus logistic-regression macro-F1 is -0.074258, with a paired 835-family bootstrap interval
of [-0.118063, -0.042576] over 1,429 shared genes. The absolute point difference exceeds the
registered 0.05 equivalence margin, while the interval crosses the -0.05 boundary. The report's
failed, underpowered verdict therefore follows the registered equivalence rule: equivalence is not
established, although the MLP does not improve performance.

Claim 3A was verified against `megascale_stability/summary.json`. The seed-0 random-split
`delta_mean` ridge Spearman correlation is 0.693111, with a 181-domain bootstrap interval of
[0.674840, 0.709067]. The complete interval exceeds the registered 0.50 threshold, so the affirmed
verdict is established. The descriptive five-seed mean is 0.692980.

Claim 3B was verified against the paired seed-0 comparison in
`megascale_stability/summary.json`. On the cohort shared by the random and family arms, the
random-minus-family Spearman difference is 0.153053, with a 77-family bootstrap interval of
[0.112310, 0.192212]. The originating plan affirms family robustness at a decrease of at most 0.05,
triggers `LEAKY` at a decrease of at least 0.10, and leaves the interval between them not
adjudicated. The complete observed interval exceeds the 0.10 refutation boundary, so the claim
fails and the result is established rather than underpowered. The executed result stores 0.10 as a
single upper-bound gate; the audit clarification in `PREREGISTRATION_run_biorxiv.md`, Part 3,
records why this does not change the verdict. The descriptive five-seed random and family means are
0.692980 and 0.553658.

Claim 3C was verified against `megascale_stability/stability_projection_3c.json`. The five-seed
mechanism macro-F1 changes from 0.394888 before projection to 0.394091 after the fitted stability
direction is removed. The seed-0 projected-minus-baseline difference is -0.000893, with a paired
1,144-family bootstrap interval of [-0.002530, 0.000681]. The complete interval is below the
registered upper limit of +0.01, so the affirmed verdict is established.

Claim 3D was verified against `megascale_stability/summary.json` and
`megascale_stability/per_protein_spearman.json`. Across 181 domains, the per-domain Spearman
standard deviation is 0.159540, with a 181-domain bootstrap interval of [0.131929, 0.183124]. The
complete interval is above the registered maximum of 0.10, so the claim fails and heterogeneous
performance is established. The mean per-domain correlation is 0.636438, the range is 0.021050 to
0.863969, and none of the domain correlations is negative.
