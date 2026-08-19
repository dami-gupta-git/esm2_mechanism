# run_biorxiv progress

Live status record for `RUNBOOK_biorxiv.md`, which holds the steps only. Each section here mirrors
a section of the runbook, in the same order.

Every analysis script computes the confidence intervals. A confidence interval is a range around a score
saying how much that score would plausibly move if the same analysis were re-run on a different
but similarly-drawn sample of genes, rather than reporting a single number as if it were exact.

## Prerequisites — manually placed files

| # | Item | Outputs | Status | Notes |
|---|---|---|---|---|
| 1 | `DiseaseMech_Stability_VEPS.xlsx` | `data/downloads/DiseaseMech_Stability_VEPS.xlsx` | ✅ | Present, unaffected by code changes |
| 2 | `AllG2P.csv` | `data/downloads/AllG2P.csv` | ✅ | Present, unaffected by code changes |

## 0. Preconditions

| Step | Item | Status | Notes |
|---|---|---|---|
| 0.0 | Environment setup | ✅ 2026-08-19 | Local venv + pip install -e . done; pod pulled to `b502952`, matches local HEAD |
| 0.1 | Pathogenicity provenance | ✅ 2026-08-19 | `pathogenicity_control.py` fingerprints variants/embeddings and raises on mismatch (code check) |
| 0.2 | Stats machinery wired | ✅ 2026-08-19 | Verified on pod: `classify_by_mechanism --seeds 1` populates `ci_low`/`ci_high` for both splits and the paired split-gap. Test run only — did not touch local `results/run_biorxiv/`; real 5-seed run still pending |
| 0.3 | Methodology rules | ✅ 2026-08-19 | Covered by 0.2 verification (same wiring implements Rules 3/4) |
| 0.4 | Paired cluster bootstrap | ✅ 2026-08-19 | `mechanism_delta_family_split.py` confirmed via 0.2's `split_gap_paired` CI output; `conservation_axis.py` confirmed by code check (`paired_oof_diff`, claim 2E) |
| 0.5/0.6 | Pre-registered decision rules | ✅ 2026-08-19 | `PREREGISTRATION_run_biorxiv.md` has Parts 1-5 with decision/resampling rules and confirmatory/exploratory split written up |
| 0.7 | Pinned environment | ✅ 2026-08-19 | `pytest tests/` → 756 passed, 1 skipped, 1 xfailed locally. Version snapshot for both machines saved to `ENV_SNAPSHOT.md` |
| 0.8 | Configuration (`RUN_NAME` flip) | ✅ 2026-08-19 | `RUN_NAME = "run_biorxiv"` already set in `utils/paths.py`; 0.2/0.4 gates now pass |
| 0.9 | Working tree clean | ✅ 2026-08-19 | `git status` clean on `main`, HEAD `b502952` |
| 0.10 | Megascale embedding provenance | ✅ 2026-08-19 | Ran on pod GPU; old June checkpoint didn't match new identity sidecar so it fully re-extracted (177,315 rows). Verified locally: 4 arrays 177,315×1,280, fingerprint written |

## 1. Build gene list

| Step | Command | Outputs | Status |
|---|---|---|---|
| 1.1 | `python -m esm2_mech.fetch_data.build_gene_list` | `data/gene_list.tsv` | ✅ 2026-08-19 |

## 2. Fetch variant data

| Step | Command | Outputs | Status |
|---|---|---|---|
| 2.1 | `python -m esm2_mech.fetch_data.fetch_variants --step gerasimavicius` | `gerasimavicius_variants.json` | ✅ 2026-08-19 |
| 2.2 | `python -m esm2_mech.fetch_data.fetch_variants --step clinvar` | `clinvar_variants.tsv` | ✅ 2026-08-19 | 2,115/2,376 genes have ≥1 row; remaining 261 confirmed zero missense P/LP, not fetch failures |
| 2.3 | `python -m esm2_mech.fetch_data.fetch_variants --step merge --pathogenic_only` | `variants.json` | ✅ 2026-08-19 | 17,865 variants, 1,937 genes |
| 2.4 | `python -m esm2_mech.fetch_data.fetch_sequences` | `cache/sequences.json` | ✅ 2026-08-19 | 1,935 sequences, cache already complete |
| 2.5 | `python -m esm2_mech.fetch_data.fetch_annotations --step pfam` | `pfam_families.json` | ✅ 2026-08-19 | 1,913/1,937 genes assigned |
| 2.6 | `python -m esm2_mech.fetch_data.fetch_alphamissense_mechanism` | `alphamissense_scores_full.json` | ✅ 2026-08-19 | matched 17,765/17,840 |
| 2.7 | `python -m esm2_mech.fetch_data.build_valid_variants` | `valid_variants.json` | ✅ 2026-08-19 | 17,770 valid variants, 1,931 genes; verified schema |
| 2.8 | `python -m esm2_mech.fetch_data.fetch_pathogenicity_variants --max_per_gene_per_class 20 --fetch_seed 42 --force` | `clinvar_pathogenicity_variants.json`, `clinvar_pathogenicity_variants.params.json` | ✅ 2026-08-19 | 25,740 variants (12,870 path / 12,870 benign), 1,837 genes. Hit one ClinVar substitution (IDS F155L) with conflicting pathogenic/benign labels across records; fixed `_deduplicate_protein_substitutions` to drop conflicting substitutions instead of aborting (was previously unhandled — no dedup existed before today's `c42a75a`). Bumped `_BALANCE_VERSION`/`_FETCH_METADATA_VERSION` to 3 |

## 3. Embed variants

| Step | Command | Status |
|---|---|---|
| 3.1 | `scp valid_variants.json` to pod | ✅ 2026-08-19 |
| 3.2 | `scp sequences.json` to pod | ✅ 2026-08-19 |
| 3.3 | `python -m esm2_mech.embeddings.embed_variants --model esm2_t33_650M_UR50D` | ✅ 2026-08-19 | Pod's existing checkpoint fingerprint matched current `valid_variants.json` exactly — reused, no re-embedding needed |
| 3.4 | Copy embeddings back to local | ✅ 2026-08-19 | Verified: all 4 arrays + `embedded_variants.json` = 17,770 rows, matches `valid_variants.json` |

## 4. Experiment: ESM-2 delta-embedding mechanism

| Step | Command | Outputs | Status |
|---|---|---|---|
| 4.1 | `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 5` | `results/run_biorxiv/family_split_baselines_seed{0..4}.json`, `mechanism_oof_cache_seed{0..4}.json` | ✅ 2026-08-19 | Ran on pod (208 cores), parallel with 4.2/4.3/4.4/4.7. No errors |
| 4.2 | `python -m esm2_mech.experiments.mechanism.mlp --seeds 5` | `results/run_biorxiv/nonlinear_results_seed{0..4}.json` | ✅ 2026-08-19 | No errors |
| 4.3 | `python -m esm2_mech.experiments.mechanism.family_clustering --seeds 5` | `results/run_biorxiv/family_clustering.json` | ✅ 2026-08-19 | Strong family clustering confirmed (k=5 purity z=+271.6) — homology leakage present in gene-split CV, as expected |
| 4.4 | `python -m esm2_mech.experiments.mechanism.naive_baseline` | `results/run_biorxiv/naive_baseline.json` | ✅ 2026-08-19 | No errors |
| 4.7 | `python -m esm2_mech.experiments.mechanism.single_source_mechanism --seeds 5` | `results/run_biorxiv/single_source_gerasimavicius/...` | ✅ 2026-08-19 | No crash; log shows a few bootstrap resamples (2-3%) discarded on some CIs due to small-subset class loss in a fold — self-reported QC flag, to review before citing those specific intervals |
| 4.6 | `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 5 --n_permutations 1000` | `results/run_biorxiv/...` | ✅ 2026-08-19 | Timed 1/5/50-permutation re-fits first (~190-224s, parallelism absorbs most of the cost on 208 cores); full run ~2h. `delta_mean` 4/5 seeds p<0.05, `wt_only_mean` 5/5 seeds p<0.05 — both clear the ≥3/5 significance bar (claim 2A not refuted) |
| 4.5 | `python -m esm2_mech.experiments.mechanism.leakage_fraction` | `results/run_biorxiv/leakage_fraction.json` | ✅ 2026-08-19 | `wt_only_mean`/`wt_concat_mut`/`mut_only_mean` leakage fraction ~38-39% (CI excludes 0); `delta_per_residue` CI suppressed (QC flag, resample denominator below threshold); rest undefined at floor (gene≈family≈chance) |

## 5. Experiment: Pathogenicity positive control

| Step | Command | Outputs | Status |
|---|---|---|---|
| 5.1 | `scp clinvar_pathogenicity_variants.json` + `.params.json` to pod | | ✅ 2026-08-19 |
| 5.2 | `scp pfam_families.json` to pod | | ✅ 2026-08-19 |
| 5.3 | `python -m esm2_mech.experiments.pathogenicity.pathogenicity_control --phase embed --model esm2_t33_650M_UR50D --force_embed` | `pathogenicity_{wt,mut}_mean.npy`, `pathogenicity_meta.json` | ✅ 2026-08-19 | 24,384/25,740 variant pairs embedded (rest dropped for missing seq/position, as expected). Also synced the uncommitted `fetch_pathogenicity_variants.py` fix to the pod first since `pathogenicity_control.py` imports from it and the cache-version contract changed |
| 5.4 | Copy embeddings back to local | | ✅ 2026-08-19 | Verified: 24,384×1,280 arrays, fingerprint matches fresh fetch |
| 5.5 | `python -m esm2_mech.experiments.pathogenicity.pathogenicity_control --phase probe --seeds 5 --n_jobs <workers> --n_boot 1000` | `results/run_biorxiv/pathogenicity_control_seed{0..4}.json`, `pathogenicity_control.json` | ✅ 2026-08-19 | Ran locally per runbook (avoids burning pod GPU-hours on CPU work). No errors. Claim 2C passes: seed-0 family-split delta_mean MLP AUROC 0.888, CI [0.882, 0.893], excludes 0.85 threshold |

## 6. Experiment: Geometry of the pathogenicity direction

| Step | Command | Outputs | Status |
|---|---|---|---|
| 6.1 | `python -m esm2_mech.experiments.geometry.build_canonical_pathogenicity` | `data/pathogenicity_valid_variants_canonical.json` | ✅ 2026-08-19 | 24,384 variants, matches embedding row count exactly |
| 6.2 | `python -m esm2_mech.experiments.geometry.run_geometry --seeds 5 --stability-dataset tsuboyama` | `results/run_biorxiv/magnitude_direction/{probe_results,geometry_results,transfer_contrast,probe4_axis_identity}.json` | ✅ 2026-08-19 | Ran on pod. No errors. Pathogenicity/stability axes transfer well (AUROC/rho ~0.80-0.89) under group-disjoint CV; mechanism axis weaker (~0.62-0.66), consistent with section 4's null. Biochem-only AUROC 0.703 vs ESM-2 delta 0.838 |
| 6.3 | `scp pathogenicity_valid_variants_canonical.json` to pod | | ✅ 2026-08-19 |
| 6.4 | `scp sequences.json` to pod | | ✅ 2026-08-19 |
| 6.5 | `python -m esm2_mech.experiments.geometry.conservation_axis --extract` | `data/conservation_pathogenicity.npy`, `data/conservation_pathogenicity_meta.json` | ✅ 2026-08-19 | Ran on pod GPU. No errors. 24,384/24,384 variants scored |
| 6.6 | Copy conservation outputs back to local | | ✅ 2026-08-19 |
| 6.7 | `python -m esm2_mech.experiments.geometry.conservation_axis` | `results/run_biorxiv/magnitude_direction/conservation_axis.json` | ✅ 2026-08-19 | Ran on pod. No errors. Claim 2D passes (conservation-alone AUROC 0.888, clears 0.85). Claim 2E fails — delta-over-conservation gap is -0.005 (CI excludes both 0 and the +0.02 threshold): pathogenicity axis is mostly conservation, embedding adds nothing measurable beyond it |

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
