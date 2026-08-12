# run_biorxiv runbook — inferential statistics

Purpose: produce `run_biorxiv` end-to-end so every report carries error bars that account for genes
in the same family not being independent, p-values where a claim needs one, and a real test behind
every "beats" claim — in place of run6's 5-seed fold-jitter error bars.

Supersedes `RUNBOOK_4.md` (run6). The experiments, gates, and hypotheses are unchanged — run_biorxiv
re-scores the same science with correct error bars. Statistical methodology is
[`reports/run6/STATS_PLAN.md`](../reports/run6/STATS_PLAN.md).

This document is both the execution spec and the live status record: each stage and step carries its
own status, so a step and its state cannot disagree. The decision rules, resampling units and
confirmatory/exploratory split are stated once, in
[`PREREGISTRATION_run_biorxiv.md`](PREREGISTRATION_run_biorxiv.md), and referenced from here rather
than restated.

**What changed from run6.** run6's error bars show only how much a number wobbles across five
seeds, which says nothing about whether the result would hold on a different set of genes.
run_biorxiv attaches cluster-bootstrap intervals that account for genes in the same family not being
independent, a permutation p-value where a claim needs one, and a paired test behind every claim
that one thing beats another. Five experiments run. ESM-3 (Experiment 4) and contrastive metric
learning (Experiment 6) are cut: the preprint's argument is the mechanism null, its chance floor,
the homology-leakage account, and the two positive controls, and both cut experiments sit outside
it. Their run6 results stand in the run6 archive. Work that does not gate this run is in
[`FOLLOWUP_biorxiv.md`](FOLLOWUP_biorxiv.md).

Outputs go to `results/run_biorxiv/` and `reports/run_biorxiv/`; `results/run6/` and `reports/run6/`
are preserved untouched as the comparison baseline, which is the point of making this a separate run
rather than an in-place fix.

All commands use `python -m esm2_mech.<module>` from the project root with the package installed
(`pip install -e .`).

**RunPod:** connect with the `id_runpod_2` key (`id_runpod` does NOT work). Run inside `tmux`.

```bash
ssh -i ~/.ssh/id_runpod_2 root@<pod-ip> -p <pod-port>
```

---

## Which embeddings are re-extracted

On 2026-08-11 every ClinVar-derived input was rebuilt from scratch. `data/clinvar_variants.tsv`
and the per-gene cache under `data/cache/clinvar/` were deleted, so `variants.json` and
`valid_variants.json` come from a current ClinVar snapshot rather than May's. The pathogenicity
control set is refetched on the same snapshot: its params sidecar
`clinvar_pathogenicity_variants.params.json` was lost, and rather than reconstruct a provenance
record that could not be fully verified, the set is refetched so its provenance is written by the
run that produced it. The consequence is that no ClinVar-derived array from run6 is reusable, and
each one is re-extracted.

**Re-extract (GPU):**

| Array | Command | Why |
|---|---|---|
| `embeddings_{wt,mut}_{mean,pos}.npy` | `embed_variants` | row-aligned to `valid_variants.json` |
| `pathogenicity_{wt,mut}_mean.npy` | `pathogenicity_control` phase 2 | refetched variant set; the old fingerprint no longer matches |
| `conservation_pathogenicity.npy` | `conservation_axis --extract` | masked-LM scores keyed to the pathogenicity set |

**Reused as-is:**

| Array | Why it is unaffected |
|---|---|
| `megascale_{wt,mut}_{mean,pos}.npy` | Tsuboyama-derived, a physical ΔΔG label with no ClinVar dependency |

The megascale *arrays* are unaffected, but the H3 stability-projection test inside
`megascale_stability` reads `valid_variants.json` and the ESM-2 arrays
(`megascale_stability.py:487`), so H3 must run after the ESM-2 re-extract.

Experiment 7 is therefore the only experiment whose inputs are identical to run6, and the only one
where a run6→run_biorxiv difference is attributable to the new statistics alone. Everywhere else the
comparison carries two changes at once — new statistics and a refreshed variant set — and the delta
note must attribute movement to both.

No copying is needed: embedding paths are keyed by *model*, not by run. `EMB_DIR` is
`data/embeddings/<ESM2_MODEL>/` (`utils/paths.py:68`), which does not contain `RUN_NAME`. The arrays are ~10 GB and
gitignored, so copying per run would cost the space for no provenance gain.

Each run_biorxiv result JSON records the array fingerprint (the existing check used by Experiments 2
and 5), so which arrays a result was scored on shows up in the output rather than only here.

GPU steps:

| Step | Why GPU |
|---|---|
| Experiment 1 step 2 — ESM-2 variant embeddings | forward pass per variant |
| Experiment 2 phase 2 — pathogenicity embeddings | forward pass per variant |
| Experiment 5 step 3 — conservation extract | masked-LM forward pass per variant |
| Experiment 7 step 4 — megascale nonlinear probe | MLP/XGBoost training |
| Permutation tests (Experiment 1) | the probe is refit once per permutation |

---

## Stage 0 — preconditions

run_biorxiv must not start until all of these hold. 0.2–0.4 are the substance of the run, 0.5–0.6
fix how its results may be read, and 0.7–0.9 protect its provenance. All nine passed before
`RUN_NAME` was flipped on 2026-08-11.

### 0.1. Pathogenicity provenance ✅

run6 already consolidated Experiment 2 onto a single canonical variant set and made
`pathogenicity_control.py` fingerprint it and refuse to run against non-matching embeddings
(lines 306, 332, 360), so the two-variant-set ambiguity behind run0's 0.74–0.88 band does not exist
here. What this precondition covered was documentation: `docs/README.md`'s "pending due to
provenance issue" note is corrected, and `result_6.md`'s 0.74–0.88 band is marked superseded so it
cannot be cited by accident. Both are done. The five-seed fingerprint agreement itself can only be
checked once Experiment 2 runs, and is a verification-checklist item below.

### 0.2. Stats machinery wired and verified ✅

Every module on the result path imports `utils/bootstrap.py` and emits CI keys: `naive_baseline`,
`mechanism_delta_family_split`, `mechanism_within_family`, `mechanism/mlp.py`,
`pathogenicity/pathogenicity_control.py`, `geometry/run_geometry.py`,
`stability/megascale_stability.py`, `mechanism/family_clustering.py` (`--seeds`, since run6 was
seed 0 only), and `leakage_fraction.py`. Anything short of that reproduces run6's CI-less result
files and wastes the run.

`leakage_fraction.py`'s ~40% figure is a headline in `INTRO_REPORT.md` and `ESM2_REPORT.md` §4. It
is a derived ratio sharing the gene-split term between numerator and denominator, so the whole
ratio is recomputed once per bootstrap replicate rather than combined from two separate intervals.

`classify_by_mechanism` is the reference implementation: gene/family clusters passed to
`bootstrap_mechanism_metrics`, `--no_ci` / `--n_boot` flags, CI keys in the result JSON.

**Verification gate.** Run each module for one seed and confirm `ci_low`/`ci_high` are populated in
the emitted JSON, not merely that it exited cleanly. Wiring that silently no-ops is the failure this
gate exists to catch, and it has caught three: family-split CIs resampling genes instead of families
across seven call sites; `pathogenicity_control.py` computing CIs for five seeds but keeping only
seed 0's; and `family_clustering.py`'s k-NN-purity and within/between CIs biased by duplicate points
under a with-replacement bootstrap, which is why `cluster_subsample_ci` exists (R7.3 addendum).

### 0.3. Methodology rules the wiring implements ✅

The rules are R7.3 (resampling unit and pairing) and R7.4 (rare-class intervals) of the
pre-registration. They are properties of the emitted numbers, so getting them wrong means
re-running, not just re-reporting. The resampling unit is enforced in code by the shared
`family_or_gene_clusters` helper.

### 0.4. Paired cluster bootstrap ✅

`utils/bootstrap.py` provides `paired_cluster_bootstrap_diff` (same-fold) and
`paired_cluster_bootstrap_diff_cross_partition`, with `paired_oof_diff` wrapping both: it aligns two
arms by `row_ids`, takes the class list as a parameter, and supports macro-F1, binary AUROC and
one-vs-rest AUROC. `adjudicate_diff` and `adjudicate_level` render the R7.1 verdict for a difference
and for a level respectively. The call sites are `conservation_axis.py` and
`mechanism_delta_family_split.py`.

The comparisons below rest on two point estimates with separated error bars, and the thinnest
margins are smaller than a seed of spread. **Two are paired**; the transfer contrast is not, for
the reason given in its row:

| Claim | Margin | Report |
|---|---|---|
| Conservation vs embedding delta (gate K2) | +0.002 | `report_geometry.md` |
| Pathogenicity vs mechanism cross-family transfer | 0.85–0.90 vs 0.62–0.64 | `report_geometry.md` — **not paired**: different datasets, no shared row space |
| Gene-split minus family-split gap | the leakage account (C2) | `report_classifier.md` — cross-partition pairing, resampled over families |

The design these must implement — one shared resample per replicate applied to both arms, the
shared-cluster restriction, family resampling for the split gap with the gene-resampled interval
alongside as a sensitivity check, and the two pairing modes as separate code paths — is R7.3 of the
pre-registration. Both modes are implemented, wired into the modules that need them, and asserted
directly in `tests/utils/test_bootstrap.py` rather than only checked for returning an interval.

### 0.5 / 0.6. Pre-registered decision rules ✅

Both were written into [`PREREGISTRATION_run_biorxiv.md`](PREREGISTRATION_run_biorxiv.md) before the
run, with the run6 point estimates recorded, so the rules cannot be tuned to the run_biorxiv
intervals. 0.5 is the CI decision rule for every confirmatory gate (R7.1); 0.6 is the
confirmatory/exploratory split (R7.2). Read them there.

### 0.7. A pinned environment ✅

`pytest tests/` passes on the commit that produces the run — a green suite is a precondition for
flipping `RUN_NAME`, run locally; there is no CI job. It was green at 622 tests.
`scripts/compare_runs.py` exists and passes its self-diff invariant (run6 against run6 reports zero
movement). Runtime dependencies are pinned via `uv.lock`, and the versions below are what the
numbers were produced under; reports cite this section in Provenance. Trimming unused exploratory
packages is housekeeping, not part of this run.

There are two environments, because the GPU steps do not run locally. Each result file records which
one produced it via the step table below; a report covering both must say so.

**CPU (local — all probe and bootstrap steps):**

```
python           3.13.7   macOS-15.6-arm64
numpy            2.2.5
scipy            1.17.1
scikit-learn     1.8.0
pandas           2.3.3
torch            2.10.0
biopython        1.86
matplotlib       3.10.8
openpyxl         3.1.5
requests         2.32.5
tqdm             4.67.1
joblib           1.5.3
pytest           9.1.1
```

scikit-learn 1.8.0 is safe for this code. `multi_class=` was removed from `LogisticRegression` in
1.8 and `CLAUDE.md` records that as a hazard; no module passes it — the only `multi_class` strings
in the tree are a `multi_class_flag` data column in `fetch_annotations.py`, unrelated to sklearn.
Multinomial is the 1.8 default, which is the behaviour the probes want.

`fair-esm` and `xgboost` are not installed locally. Both are on the result path but only for GPU
steps: `fair-esm` blocks Experiment 5 step 3 (`conservation_axis --extract`) and `xgboost` blocks
Experiment 7 step 4 (`megascale_mlp --xgboost`). Neither blocks the CPU work.

**GPU (RunPod): not yet recorded.** Capture it on the pod before Experiment 5 step 3 and
Experiment 7 step 4 — a report citing a GPU-produced number against the CPU version list would be
wrong. Record the CUDA and driver versions too, since the permutation refits and the megascale probe
are the steps whose cost and numerics depend on them.

```bash
python -c "
import platform, importlib.metadata as md
print('python', platform.python_version(), '|', platform.platform())
for p in ['numpy','scipy','scikit-learn','pandas','torch','fair-esm','xgboost','biopython','joblib']:
    try: print(f'{p}=={md.version(p)}')
    except Exception: print(f'{p}: MISSING')
"
```

### 0.8. Configuration ✅ (flipped 2026-08-11)

- `RUN_NAME = "run6"` → `"run_biorxiv"` in [`utils/paths.py:11`](../src/esm2_mech/utils/paths.py#L11). One
  line; `RESULTS_DIR`, `RUN_REPORTS_DIR`, and `FIGURES_DIR` all derive from it. **Flip this only
  after Stage 0.2 and 0.4 pass their gates** — flipping first means the replay writes CI-less files into
  `results/run_biorxiv/`, and fixing them later either overwrites run_biorxiv provenance or forces a run8.
- `PERMUTATION_FEATURES` stays at `("delta_mean", "wt_only_mean")`. `delta_mean` is C1's
  instrument and `wt_only_mean` is the above-floor comparison; the remaining features are
  exploratory, so each one added costs another 2,000 refits for a p-value no claim reads. The same
  constant gates which features cache OOF for the split gap, so widening it enlarges that cache too.
- `PERMUTATION_N_RESAMPLES` already defaults to 1000 in `constants.py`. The 200 in the run6
  files came from a run-time override; do not repeat it. At 200 the smallest resolvable p-value
  is 1/(200+1) = 0.0099, which is exactly what `wt_only_mean` reported — an unresolved floor,
  not a measurement.

### 0.9. Working tree clean ✅

`git status` must be clean before the run_biorxiv branch point, or run6 and run_biorxiv provenance
become impossible to separate. Commit or set aside everything, including outstanding report edits.
`results/run_biorxiv/homology_partition_panel/panel.json` is deleted first — it is the withdrawn
smoke-scale file (see Task 2b in [`FOLLOWUP_biorxiv.md`](FOLLOWUP_biorxiv.md)), and the results
directory should not start with a retracted result in it.

---

## Stage 1 — build gene list (CPU) ✅ run 2026-08-11, 2,376 genes

Unchanged from run6. Shared foundation; run once.

**Inputs (manually placed in `data/downloads/`):** see the prerequisites table in `RUNBOOK_4.md`
— the file set is unchanged.

```bash
python -m esm2_mech.fetch_data.build_gene_list
```

**Output:** `data/gene_list.tsv`

---

## Experiment 1 — ESM-2 delta-embedding mechanism ⬜

### Step 1 — fetch data (CPU) ✅

Unchanged from run6 in code, but **not skippable in run_biorxiv**: the ClinVar step was re-run from
scratch on 2026-08-11, so every downstream file in this table is rebuilt against a current ClinVar
snapshot. The steps overwrite their outputs unconditionally, so nothing needs deleting first.

Step 3 is the long one — with `data/cache/clinvar/` deleted it fetches all 2,376 genes from NCBI.
Genes whose esearch or esummary call fails are not written and not cached; re-running the same
command retries exactly those.

| Step | Command | Outputs | Status |
|---|---|---|---|
| 2 | `fetch_variants --step gerasimavicius` | `gerasimavicius_variants.json` | ✅ 2026-08-11 |
| 3 | `fetch_variants --step clinvar` | `clinvar_variants.tsv` | ✅ 2026-08-12 |
| 4 | `fetch_variants --step merge --pathogenic_only` | `variants.json` | ✅ 2026-08-12 |
| 5 | `fetch_sequences` | `cache/sequences.json` | ✅ 2026-08-12 |
| 6 | `fetch_annotations --step pfam` | `pfam_families.json` | ✅ 2026-08-12 |
| 7 | `fetch_alphamissense_mechanism` | `alphamissense_scores_full.json` | ✅ 2026-08-12 |
| 8 | `build_valid_variants` | `valid_variants.json` | ✅ 2026-08-12 |

**Results:** gerasimavicius 10,233 variants / 948 genes. ClinVar 48,152 rows / 2,115 genes. Merged
`variants.json`: 17,865 variants, 1,937 genes (gerasimavicius=10,233, clinvar_g2p=7,632). Sequences
fetched for 1,935 genes. Pfam: 1,913/1,937 genes annotated, 24 unannotated. AlphaMissense matched
17,765 variants. `valid_variants.json`: 17,770 rows.

**WT-mismatch check flagged 9 genes in the gerasimavicius set (2026-08-12).** The stored
wild-type residue at the variant position does not match the sequence on file for: MEN1 (38/47
variants mismatched), CYP21A2 (34/37), and to a lesser extent SHANK3, TUFM, TPI1, ARID1B, FDX2,
AGT, TRPC3. MEN1 and CYP21A2 account for most of the mismatched variants. For those two, the
likely cause is that the stored sequence is the wrong transcript isoform — both genes have
multiple annotated isoforms, which is the exact case the WT-check exists to catch. Not yet
root-caused for the other seven genes. `build_valid_variants` drops these variants (WT mismatch
fails `apply_missense`, counted under `skipped_invalid`), so they do not reach the embeddings —
the cost is coverage, not silent contamination: MEN1 and CYP21A2 lose most of their variants and
effectively drop out of any per-gene analysis until the isoform mapping is fixed at the
sequence-fetch step.

### Step 2 — embed variants (GPU) ⬜

**Re-extracted in run_biorxiv** (see *Which embeddings are re-extracted*) — the run6 arrays are aligned
to the pre-refresh variant list and cannot be reused.

```bash
python -m esm2_mech.embeddings.embed_variants --model esm2_t33_650M_UR50D
```

**Output:** `data/embeddings/<ESM2_MODEL>/embeddings_{wt,mut}_{mean,pos}.npy` and
`embedded_variants.json`. Verify before proceeding: all four `.npy` arrays have the same row count,
that count equals the number of rows in `embedded_variants.json`, and the width is 1280.

### Step 3 — run analysis ⬜

| Command | Outputs | Status |
|---|---|---|
| `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 5` | `results/<run>/family_split_baselines_seed{0..4}.json`, `aggregate.json` | ⬜ |
| `python -m esm2_mech.experiments.mechanism.mlp --seeds 5` | `results/<run>/nonlinear_results_seed{0..4}.json` | ⬜ |
| `python -m esm2_mech.experiments.mechanism.family_clustering --seeds 5` | `results/<run>/family_clustering.json` | ⬜ |
| `python -m esm2_mech.experiments.mechanism.naive_baseline` | `results/<run>/naive_baseline.json` — the measured chance floor | ⬜ |
| `python -m esm2_mech.experiments.mechanism.leakage_fraction` | `results/<run>/leakage_fraction.json` | ⬜ |

`family_clustering` gains `--seeds` in run_biorxiv: run6 reported the family-probe accuracy at seed 0
only, and `STATS_PLAN.md` requires a spread to match the other reports.

`leakage_fraction` reads only the result JSONs above (no model inference), so it runs last, and it
needs a CI — run6 reported ~40% as a bare point estimate, quoted as a headline in the intro report.

**Stats flags.** `classify_by_mechanism`, `single_source_mechanism`, and
`mechanism_delta_family_split` accept:

- `--no_ci` — skip cluster-bootstrap CIs (faster; not for the run_biorxiv replay).
- `--n_boot N` — bootstrap resamples, default 1000.
- `--n_permutations N` — label-permutation p-value against chance, default 0 = off.

CIs are on by default and add roughly a minute per seed.

### Step 3b — permutation tests (GPU, seed 0 only) ⬜

Run separately from Step 3: `wt_only_mean` refits the probe once per repeat, so it multiplies
per-seed probe time by N and belongs in its own tmux window.

```bash
python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 1 --n_permutations 1000
```

**The two headline features run different tests** (R7.5, amended 2026-08-12). `wt_only_mean` refits
per permutation and scores macro-F1. `delta_mean` scores macro one-vs-rest AUROC against the cached
out-of-fold predictions and refits nothing, because macro-F1 cannot fire for a probe sitting at the
floor — it predicts the majority class almost everywhere regardless of what the ranking holds, and
run6 measured it *below* its own shuffled-label mean at p = 1.0. Both tests permute at the family
level, swapping whole families' label blocks, because both score a family-split metric and the
permutation unit has to match what the interval clusters on — run6's gene-level shuffle broke the
label structure homologous genes share and built too tight a null. The pre-registration carries the
reasoning and the one thing that must reach the reports: the two p-values come from different nulls,
and each report says which. The emitted result records the statistic, the null type, the permutation
unit, the null's width, and how many families had no same-size partner to swap with.

**Seed 0 only, deliberately.** A permutation test constructs its own null by shuffling labels, so
running it across 5 seeds mostly re-measures the fold jitter run_biorxiv exists to replace. Cuts the
step 5× at no inferential cost.

**Linear probe only, at 1,000 permutations.** The headline claim — `delta_mean` sits at the chance
floor — is a linear-probe claim, so that is the load-bearing test and it runs at full N. The MLP is
not permutation-tested: no claim rests on an MLP permutation p-value and its refits are the
expensive tail. Never report a p-value sitting at its resolution floor of 1/(N+1), the unresolved
`wt_only_mean` = 0.0099 case from run6.

**Before launching, time a single refit on the pod.** The refit cost is now one feature × the
family split × 1,000 permutations = 1,000 refits, and the per-refit cost has never been measured.
It decides whether the step is hours or days, and whether it needs joblib parallelism across the
pod's cores. Earlier planning text quoted 4,000 refits, counting two features across two splits;
only the family split is permuted, so it was 2,000 before this change. This was the run's main
schedule risk and it is now halved.

**This step does NOT cover the split gap.** That is a paired-bootstrap quantity (Stage 0.4).

### Step 4 — single-source robustness check (CPU) ⬜

Unchanged in design; gains CIs. Re-runs the Step 3 probe on the Gerasimavicius-only subset
(10,138 variants, all three classes from one curation pipeline) and recomputes the majority
floor on the subset (0.279, not the merged 0.288).

```bash
python -m esm2_mech.experiments.mechanism.single_source_mechanism --seeds 5
```

**Output:** `results/<run>/single_source_gerasimavicius/{family_split_baselines_seed{0..4}.json,
aggregate.json, naive_baseline.json}`

The run_biorxiv question is narrower than run6's: not whether `delta_mean` sits at the floor, but
whether its **interval still straddles the floor**, and what the CI on `wt_only`'s gene-minus-family
gap (0.612 → 0.445) looks like.

---

## Experiment 2 — pathogenicity positive control ⬜

Tests whether the same delta embeddings that classify mechanism at chance predict ClinVar
pathogenic-vs-benign. Pass criterion: `delta_mean` MLP AUROC ≥ 0.85.

**All three phases run in run_biorxiv.** The variant set is refetched from the current ClinVar
snapshot and the embeddings are re-extracted against it, so phase 2 needs GPU.

Before starting, delete the stale set so the fetch is unambiguous rather than a cache-miss
side effect:

```bash
rm -f data/clinvar_pathogenicity_variants.json
python -m esm2_mech.experiments.pathogenicity.pathogenicity_control --model esm2_t33_650M_UR50D
```

The fetch is driven by `--max_per_gene_per_class` (default 20) and `--fetch_seed` (default 42), and
both are written to `clinvar_pathogenicity_variants.params.json` by the run. Keep the defaults: the
run6 set was capped at 20 per gene per class, which is verifiable in the surviving data (982
gene×class groups sit at exactly 20, none above), so keeping the cap holds the sampling design
fixed while the snapshot changes.

Expect the set to differ from run6's 38,698 variants / 37,218 embedded rows / 1,937 genes. Record
the new counts; a large shift is a real change in ClinVar coverage, not an error, but it must be
stated rather than absorbed.

Requires wiring per Stage 0.2. Classes are balanced here, but the gene-level dependency structure
still applies, so CIs resample whole genes. Add the calibration note: the probes measure
discrimination only and are not risk estimates.

**Why the run6 set was not reused.** Its params sidecar was deleted and is not recoverable. The
cap was verifiable from the data but the fetch seed was not, and the seed determines which variants
were sampled at capped genes. Refetching produces a set whose provenance is written by the run that
made it, rather than one carrying a partly reconstructed record.

---

## Experiment 3 — within-family mechanism (CPU) ⬜

Reads the existing embedding arrays under `data/embeddings/<ESM2_MODEL>/`; no GPU, no RunPod.

```bash
python -m esm2_mech.experiments.mechanism.mechanism_within_family --seeds 5
```

**Output:** `results/<run>/within_family_mechanism.json`

Already imports `utils/bootstrap.py`. Three additions required by `STATS_PLAN.md`, all changing how
the table is *read* rather than what it contains:

1. **The table is labelled an exploratory screen.** The run6 "beats baseline and std < 0.10"
   highlight is uncorrected and is labelled rather than corrected; correcting it would imply it had
   been a confirmatory test.
2. **Minimal-detectable-effect per family**, so the nulls read as underpowered rather than as
   evidence of absence. At 6–33 genes per family the test cannot establish absence.
3. **Cluster-bootstrap CIs over genes within each family**, replacing seed-std.

---

## Experiment 5 — geometry of the pathogenicity direction ⬜

| Step | Command | GPU | Status |
|---|---|---|---|
| 1 build | `geometry.build_canonical_pathogenicity` | no | ⬜ |
| 2 probes | `geometry.run_geometry --seeds 5` | no | ⬜ |
| 3 conservation extract | `geometry.conservation_axis --extract` | **yes** | ⬜ |
| 4 conservation analysis | `geometry.conservation_axis` | no | ⬜ |

Step 3 masks each variant position and reads the masked-LM `logP_wt` / `logP_mut` / entropy. It can
share a pod session with the Step 3b permutation work.

The whole experiment sits downstream of Experiment 2: step 1 reads
`clinvar_pathogenicity_variants.json` and the pathogenicity arrays, so all four steps run only
after Experiment 2's refetch and re-embed. The existing `conservation_pathogenicity.npy` and
`pathogenicity_valid_variants_canonical.json` are keyed to the run6 set and are rebuilt, not
reused.

run_biorxiv additions: gene-cluster CIs on each pathogenicity AUROC (effective N ≈ 1,929 genes, not
37,218 variants), and a **paired cluster bootstrap on the two load-bearing gaps** — conservation
(0.891) versus embedding delta (0.859), and the conservation-plus-delta increment (+0.002, which
is the entire basis for gate K2). Also attach a paired CI to the task transfer contrast
(pathogenicity 0.85–0.90 vs mechanism 0.62–0.64).

---

## Experiment 7 — megascale stability positive control ⬜

A second positive control with a purely physical label (Tsuboyama 2023 ΔΔG), free of clinical
curation.

**Step 2 (embedding) is SKIPPED** — `megascale_{wt,mut}_{mean,pos}.npy` already exist and are
Tsuboyama-derived, so the ClinVar refresh does not touch them. Step 3's H3 stability-projection
test does read `valid_variants.json` and the ESM-2 arrays (`megascale_stability.py:487`), so run
step 3 only after Experiment 1 step 2.

| Step | Command | GPU | Status |
|---|---|---|---|
| 1 families | `stability.build_domain_families` | no | ⬜ |
| 2 embed | *(skipped — arrays reused)* | — | ⏭️ |
| 3 linear probe | `stability.megascale_stability` | no | ⬜ |
| 4 nonlinear probe | `stability.megascale_mlp --xgboost` | **yes** | ⬜ |
| 5 controls | `stability.stability_baselines` | no | ⬜ |

Step 1 needs `hmmscan` and a hmmpress-ed Pfam-A; skip if `megascale_domain_families.json` is
present and non-empty. Step 4 is the third genuine GPU step.

Gates unchanged: H1 random-split ρ ≥ 0.5, H2 LEAKY threshold 0.10 on the random→family drop, H4
per-domain ρ std, H3 mechanism-F1 change ≤ +0.01.

---

## Stage 2 — remaining statistical work

Machinery exists for all of these; none was implemented in run6.

- ✅ **AUPRC with prevalence baseline, and PPV/NPV at class prevalence**, for the rare classes
  (DN ≈ 9%, GOF ≈ 15%). AUROC alone overstates usefulness at those rates. **Implemented
  2026-08-11** in `utils/metrics.py` (`imbalance_metrics`, and the per-class keys emitted by
  `compute_metrics` and `add_flat_class_metrics`), `utils/probes.py` for the binary path, and
  `utils/bootstrap.py`, which now emits an `auprc_<cls>` CI beside every `auroc_<cls>` one. Every
  probe on the result path picks these up without a per-module change.

  How the numbers are defined, since the reports must explain them: AUPRC's no-signal value is the
  class prevalence, not 0.5, so the prevalence is emitted next to it as its baseline — with its own
  interval, and with an interval on the gap between the two computed within each resample, since a
  fixed baseline under a moving AUPRC is the misreading the pair exists to prevent. The gap is the
  number to read for "better than no signal". PPV and NPV
  are read at the prevalence-matched operating point — the top `prevalence × n` scores are called
  positive, so the predicted positive rate equals the observed one. That point needs only the
  ranking, not calibrated probabilities, which is what makes it reportable for an uncalibrated
  probe.

- ⬜ **Calibration note in every probe report** — the probes are uncalibrated and measure
  discrimination only, not risk. State it rather than fix it; the claims are about
  discrimination. In practice each probe report carries one sentence saying a score of 0.8 is a
  rank, not an 80% chance of the class, and the PPV/NPV pair is what answers "if it flags this
  variant, how often is that right".
- ⬜ Exploratory labelling and minimal-detectable-effect for the within-family table — part of
  Experiment 3.
- ⬜ Multi-seed family probe — part of Experiment 1 Step 3.
- ⬜ `python scripts/compare_runs.py run6 run_biorxiv` — both the regression test for the run and
  the delta-note deliverable, generated rather than transcribed.

---

## Stage 3 — regenerate reports ⬜

The run6 reports are rewritten against run_biorxiv result files into `reports/run_biorxiv/`, minus the
experiments cut from this run. The ESM-3 and contrastive reports are not regenerated; they stay in
the run6 archive and are cited from there if referenced at all. `STATS_PLAN.md` moves from a plan to
a record of what was done.

Thirteen documents: eight per-experiment reports — `report_classifier`, `report_control`,
`report_protein_family`, `report_leakage_fraction`, `report_within_family`, `report_geometry`,
`report_stability`, `report_single_source` — plus three cross-cutting ones: `ESM2_REPORT.md` (the
assembled paper), `INTRO_REPORT.md` (the lay summary) and `STATS_PLAN.md`. Both cross-cutting
narrative reports quote numbers from every section, so both must be regenerated.
`INTRO_REPORT.md` stays in `reports/run_biorxiv/` rather than moving to `docs/`.

Two are new for run_biorxiv. The first is a paired-difference summary: the tested differences are
otherwise scattered across reports, and this collects each one with its CI into a single table. The
second is the run6→run_biorxiv delta note — every headline number, its run6 value, its run_biorxiv
value, and whether the CI changes the reading — generated by `scripts/compare_runs.py`, not
transcribed. The writing task there is reviewing its output and explaining each flagged movement.

None of the run6 reports cites a confidence interval, including `report_classifier.md` (the headline)
and `report_leakage_fraction.md` (no interval on the ~40% figure).

Per the project report rules: a result file and its report share the same `RUN_NAME`, and every
number traces to a file under `results/run_biorxiv/` cited in Provenance. Provenance also states which
variant snapshot and which embedding arrays a number was computed on: everything ClinVar-derived
comes from the refreshed 2026-08-11 snapshot and newly extracted arrays, while the megascale
numbers come from the retained run6 arrays.

Because the inputs moved, the run6→run_biorxiv delta note reads differently than planned. Only
Experiment 7 isolates the effect of the new statistics. Every other movement carries a variant-set
change as well, and the note attributes it to both rather than to the statistics alone.

---

## Verification checklist

Data and alignment:

- [ ] `data/embeddings/<ESM2_MODEL>/embedded_variants.json` row count matches all four `.npy`
      arrays and the new `valid_variants.json`. This file is a write-only provenance artifact — no
      code reads it.
- [ ] The refreshed ClinVar fetch covers all 2,376 genes, and any gene skipped by a failed
      esearch/esummary call has been retried to completion.
- [ ] The new variant count is recorded and compared against run6's 17,826; a large shift is
      expected from the ClinVar refresh but must be explained rather than absorbed silently.
- [ ] `data/pfam_families.json` has entries for ≥ 1,900 genes.
- [ ] `data/alphamissense_scores_full.json` covers > 90% of `valid_variants.json`.
- [ ] `clinvar_pathogenicity_variants.params.json` exists next to the refetched variant set and
      records the cap and fetch seed the run actually used. It is the file whose loss forced the
      refetch, so a run that does not leave one behind has repeated the problem.
- [ ] The new pathogenicity counts are recorded against run6's 38,698 variants / 37,218 embedded
      rows / 1,937 genes.
- [ ] Embedding fingerprints recorded in every run_biorxiv result file match the newly extracted
      arrays, except megascale, which matches the retained run6 arrays.
- [ ] **All five pathogenicity seeds share one variant-set fingerprint, and the AUROC spread across
      them is ≤ 0.01.** If any seed disagrees, stop the run — that is a real data defect, and it is
      the failure that produced run0's 0.74–0.88 band, where two different variant sets across seeds
      were read as sampling uncertainty. `pathogenicity_control.py` already refuses to proceed when
      the embeddings do not match the current variant set; this checks the recorded fingerprints
      agree across seeds after the fact as well.

Statistics — the point of this run:

- [ ] Every result file under `results/run_biorxiv/` that reports a macro-F1 or AUROC carries a CI key.
      Zero CI-less headline files.
- [ ] **Family-split CIs resample families, gene-split CIs resample genes.** Effective cluster
      count is emitted alongside every family-split interval. Family-split intervals are wider
      than gene-split ones — if they are not, the unit was applied wrongly.
- [ ] Rare-class (DN, GOF) AUROC intervals are labelled as the least trustworthy in their table.
- [ ] Linear-probe permutation ran at 1,000, and it is the only permutation test in the run — the
      MLP is not permutation-tested. No p-value equal to 1/(N+1) is reported as a measurement —
      that is the resolution floor, not a result.
- [ ] **Both permutation p-values record a family-level permutation unit**, not a gene-level one:
      they score family-split metrics, so the permutation unit must match the interval's clustering
      unit. Each also records its statistic and null type, and the reports say which test produced
      which p-value — `wt_only_mean` refits per permutation on macro-F1, `delta_mean` re-scores
      macro AUROC against the cached out-of-fold predictions.
- [ ] The count of families with no same-size partner is reported beside `delta_mean`'s p-value, so
      how much of the data actually moved under the null is visible rather than assumed.
- [ ] The two paired claims (geometry K2 and the gene-vs-family split gap) report a paired
      cluster-bootstrap CI on the difference, not two separated error bars.
- [ ] The pathogenicity-vs-mechanism transfer contrast reports two independent intervals and says
      so. Its arms are different datasets — ClinVar pathogenicity variants against mechanism
      variants — with no shared row space, so no pairing is defined over them. It is exploratory
      and no confirmatory claim reads it.
- [ ] The gene-vs-family split gap reports a paired-bootstrap CI, **not** a label-permutation
      p-value — the permutation null for that gap is zero by construction.
- [ ] Paired bootstrap resamples the cluster unit once per replicate and applies the same
      resample to both arms; unit tests assert this directly, and cover both the same-fold and
      cross-partition pairing modes.
- [ ] Every report carries the calibration note.
- [ ] Every confirmatory gate reports its verdict under the pre-registered CI rule — affirmed /
      not distinguishable / failed / underpowered — never a bare "pass" beside an interval
      through zero.
- [ ] The five confirmatory claims are reported as the enumerated set, with C1 adjudicated against
      its 0.05 equivalence margin. Exploratory tables are labelled, not corrected.
- [ ] `scripts/compare_runs.py run6 run_biorxiv` run, its output archived as the delta note, and every
      material movement explained.

Science — the spine, which should not move:

- [ ] Pathogenicity `delta_mean` MLP AUROC ≈ 0.89 and passes ≥ 0.85.
- [ ] Mechanism `delta_mean` at the measured floor on both splits; `wt_only` gene-split lift
      collapses under family-split.
- [ ] Stability H1 random-split ρ ≥ 0.5.
- [ ] Any run_biorxiv number that moves materially from run6 is explained, not silently adopted.

- [ ] Environment pinned and recorded in Provenance; `pytest tests/` green on the commit that
      produced the run.
- [ ] `git status` clean; results committed.
