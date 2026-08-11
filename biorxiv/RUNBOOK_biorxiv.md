# Run runbook 5 — run_biorxiv, inferential statistics

Purpose: produce `run_biorxiv` end-to-end with dependency-aware confidence intervals, permutation
p-values, and tested difference claims, so every report carries inferential statistics rather
than 5-seed fold-jitter error bars.

Supersedes `RUNBOOK_4.md` (run6). The experiments, gates, and hypotheses are unchanged — run_biorxiv
re-scores the same science with correct error bars. Statistical methodology is
[`reports/run6/STATS_PLAN.md`](../reports/run6/STATS_PLAN.md); the change list is
[`PLAN_2026-07-20.md`](PLAN_2026-07-20.md).

All commands use `python -m esm2_mech.<module>` from the project root with the package installed
(`pip install -e .`).

**RunPod:** connect with the `id_runpod_2` key (`id_runpod` does NOT work). Run inside `tmux`.

```bash
ssh -i ~/.ssh/id_runpod_2 root@<pod-ip> -p <pod-port>
```

---

## Embeddings are NOT re-extracted

Nothing upstream of the probes changed between run6 and run_biorxiv, so every GPU embedding step is
skipped and the existing arrays are reused as-is. **Do not re-run `embed_variants`,
`embed_megascale`, `embed_scan`, or `esm3_mechanism --phase 2`.**

This is safe and requires no copying: embedding paths are keyed by *model*, not by run.
`EMB_DIR` is `data/embeddings/<ESM2_MODEL>/` and `ESM3_EMB_DIR` is
`data/embeddings/<ESM3_MODEL>/` (`utils/paths.py:68-69`), neither of which contains `RUN_NAME`.
The arrays are ~10 GB and are gitignored; duplicating them per run would cost 10 GB for no
provenance gain, since the model directory already identifies them unambiguously.

Recorded consequence: **run_biorxiv result files are scored on embeddings extracted during run6.**
That is intentional. Each run_biorxiv result JSON records the array fingerprint (the existing
fingerprint check used by Experiments 2 and 5), so the reuse is recorded in the output rather
than only in this runbook.

GPU is still required for three steps, which are computed rather than cached:

| Step | Why GPU |
|---|---|
| Experiment 5 step 3 — conservation extract | masked-LM forward pass per variant |
| Experiment 7 step 4 — megascale nonlinear probe | MLP/XGBoost training |
| Permutation tests (Experiment 1) | the probe is refit once per permutation |

---

## Stage 0 — preconditions

run_biorxiv must not start until all of these hold. 0a/0a-bis/0b are the substance of the run,
0b-bis fixes how its results may be read, and 0c–0e protect its provenance.

### 0a. Stats machinery wired and verified

`utils/bootstrap.py` (`cluster_bootstrap_ci`, `bootstrap_mechanism_metrics`,
`label_permutation_pvalue`) is built, but as of run6 only three modules import it
(`naive_baseline`, `mechanism_delta_family_split`, `mechanism_within_family`). Seven more must be
wired before the replay, or run_biorxiv reproduces run6's CI-less result files and the run is wasted.

| Module | Experiment |
|---|---|
| `mechanism/mlp.py` | 1 (Step 3, nonlinear probes) |
| `mechanism/contrastive_mechanism.py` | 6 |
| `esm3/esm3_mechanism.py` (phase 3) | 4 |
| `pathogenicity/pathogenicity_control.py` | 2 |
| `geometry/run_geometry.py` | 5 |
| `stability/megascale_stability.py` | 7 |
| `mechanism/family_clustering.py` | 1 (needs a new `--seeds` flag; run6 is seed 0 only) |

`leakage_fraction.py` also needs an interval: its ~40% figure is a headline in `INTRO_REPORT.md`
and `ESM2_REPORT.md` §4 and is currently a bare point estimate. It is a derived ratio sharing the
gene-split term between numerator and denominator, so recompute the whole ratio once per
bootstrap replicate rather than combining two separate intervals.

`classify_by_mechanism` is the working reference implementation to copy: pass gene/family
clusters to `bootstrap_mechanism_metrics`, add the `--no_ci` / `--n_boot` flags, emit CI keys
into the result JSON.

**Verification gate:** run one module for one seed locally and confirm a CI key is actually
present in the emitted JSON. Wiring that silently no-ops is the failure this gate exists to
catch.

### 0a-bis. Methodology rules the wiring must implement

Settled in `PLAN_2026-07-20.md` Task 0. These are properties of the emitted numbers, so getting
them wrong means re-running, not just re-reporting.

1. **The resampling unit matches the split.** Gene-split metrics resample genes; family-split
   metrics resample **families**, because under family-split the family is the held-out unit and
   genes within one are not independent draws. Family-split CIs will be visibly wider — 1,134
   families but 833 singletons, so far fewer effective clusters than genes. That is correct, not
   a bug. Emit the effective cluster count next to every family-split interval.
2. **Rare-class AUROC uses BCa, and is flagged regardless.** DN (≈ 9%, ~150–170 genes) and GOF
   (≈ 15%) sit in the regime where percentile bootstrap undercovers for a bounded metric with few
   clusters. Use BCa where the acceleration estimate is computable, keep the existing degenerate-
   fold suppression guard, and label rare-class intervals as the least trustworthy in their table
   either way — with ~150 jackknife clusters, BCa's own correction is noisy.

### 0b. Paired cluster bootstrap implemented

`paired_cluster_bootstrap_diff(...)` in `utils/bootstrap.py`, plus unit tests. This is the only
genuinely new statistical primitive in run_biorxiv. Six claims rest on comparing two point estimates with
separated error bars, and the thinnest margins are smaller than a seed of spread:

| Claim | Margin | Report |
|---|---|---|
| ESM-3 seq vs ESM-2 MLP delta_mean | M2 gate clears its 0.430 threshold by 0.008 | `report_esm3_mechanism.md` |
| Contrastive k-NN vs raw-delta k-NN | +0.041 | `report_contrastive.md` |
| Conservation vs embedding delta (gate K2) | +0.002 | `report_geometry.md` |
| Pathogenicity vs mechanism cross-family transfer | 0.85–0.90 vs 0.62–0.64 | `report_geometry.md` |
| Contrastive per-class DN "unmoved" | a null asserted from a 0.577 → 0.545 point drop | `report_contrastive.md` |
| Gene-split minus family-split gap | see below | `report_classifier.md` |

Required design:

- Resample the cluster unit **once per bootstrap replicate and apply that same resample to both
  arms.** Resampling the arms independently inflates the variance of the difference and is wrong
  for paired data.
- Restrict both arms to the shared cluster subset.
- Fold handling depends on the pairing mode — identical folds in same-fold mode only, **not** as a
  blanket rule. See below.

**The split gap uses the bootstrap, not a permutation test.** Under a shuffled-label null both
gene-split and family-split collapse to the floor, so the null gap is centred near zero by
construction; such a test asks "does leakage exist" — already answered by the ~40% leakage
fraction — and says nothing about the observed gap's sampling variability. Use the paired
bootstrap instead.

**The split-gap CI resamples families, not genes** — its family-split arm's variance is only
correct under family resampling, so a gene-resampled gap understates it. Resample the coarser of
the two units; report the gene-resampled interval alongside as a labelled sensitivity check.

**Two pairing modes, and they are different code paths.** The ESM-3, contrastive, and
conservation comparisons share a fold assignment. The split-gap comparison does not — gene-split
and family-split are different CV partitions by definition — so its pairing is at the **gene
level across two fold assignments**: resample **families** (the coarser unit — see above), then
recompute each arm under its own partition. Written without distinguishing these, the cross-partition case gets silently
implemented as the same-fold path and is wrong.

### 0b-bis. Pre-registered decision rules written into PREREGISTRATION_run_biorxiv.md

Two documents must exist **before** run_biorxiv executes, or the run produces intervals with no stated
reading and any interpretation chosen afterwards is retro-fitted:

1. **CI decision rule for every confirmatory gate.** A gate is affirmed only if its point estimate
   clears the threshold *and* the paired difference 95% CI excludes zero; if the point estimate
   clears but the CI spans zero, the claim is restated as **not distinguishable**. A gate failing
   with a CI that also spans the threshold is reported as **underpowered to detect an effect of
   the pre-registered size**, not as evidence of no effect.
2. **The confirmatory / exploratory split.** Six confirmatory claims (C1–C6 in
   `PLAN_2026-07-20.md` Task 0.2) carry BH-FDR correction across the set, reported raw and
   adjusted. Everything else is labelled exploratory and is not corrected.

Both go into `PREREGISTRATION_run_biorxiv.md` with the run6 point estimates recorded, so the
rules cannot be tuned to the run_biorxiv intervals.

### 0c. Green CI and a pinned environment

- The test suite (38 files) passes in CI. **Green CI is a precondition for flipping `RUN_NAME`.**
- Runtime dependencies trimmed to the result path and pinned; `wandb`, `aider-chat`, `openai`,
  `google-generativeai` and the other exploratory packages removed or moved to an optional extra.
  The pinned set is recorded in the reports' Provenance — it is what the numbers were produced
  under.
- `scripts/compare_runs.py` exists and passes its self-diff invariant (run6 against run6 must
  report zero movement).

### 0d. Configuration

- `RUN_NAME = "run6"` → `"run_biorxiv"` in [`utils/paths.py:11`](../src/esm2_mech/utils/paths.py#L11). One
  line; `RESULTS_DIR`, `RUN_REPORTS_DIR`, and `FIGURES_DIR` all derive from it. **Flip this only
  after 0a and 0b pass their gates** — flipping first means the replay writes CI-less files into
  `results/run_biorxiv/`, and fixing them later either overwrites run_biorxiv provenance or forces a run8.
- Widen `PERMUTATION_FEATURES` at
  [`mechanism_delta_family_split.py:108`](../src/esm2_mech/experiments/mechanism/mechanism_delta_family_split.py#L108)
  from `("delta_mean", "wt_only_mean")` to also include `wt_concat_mut` and `mut_only_mean` —
  all four above-floor features, plus `delta_mean` retained as the negative control. The four
  at-floor features are left out: permuting a feature already at chance adds nothing its CI does
  not show, and costs half the budget.
- `PERMUTATION_N_RESAMPLES` already defaults to 1000 in `constants.py`. The 200 in the run6
  files came from a run-time override; do not repeat it. At 200 the smallest resolvable p-value
  is 1/(200+1) = 0.0099, which is exactly what `wt_only_mean` reported — an unresolved floor,
  not a measurement.

### 0e. Working tree clean

`git status` must be clean before the run_biorxiv branch point, or run6 and run_biorxiv provenance become
impossible to separate. Commit or set aside everything, including the report and plan edits
outstanding at the time of writing.

---

## Stage 1 — build gene list (CPU)

Unchanged from run6. Shared foundation; run once.

**Inputs (manually placed in `data/downloads/`):** see the prerequisites table in `RUNBOOK_4.md`
— the file set is unchanged.

```bash
python -m esm2_mech.fetch_data.build_gene_list
```

**Output:** `data/gene_list.tsv`

---

## Experiment 1 — ESM-2 delta-embedding mechanism

### Step 1 — fetch data (CPU)

Unchanged from run6, and skippable if the outputs are already on disk and verified — these are
`data/`-level files not keyed by run. Re-run only if `data/` was cleared.

| Step | Command | Outputs |
|---|---|---|
| 2 | `fetch_variants --step gerasimavicius` | `gerasimavicius_variants.json` |
| 3 | `fetch_variants --step clinvar` | `clinvar_variants.tsv` |
| 4 | `fetch_variants --step merge --pathogenic_only` | `variants.json` |
| 5 | `fetch_sequences` | `cache/sequences.json` |
| 6 | `fetch_annotations --step pfam` | `pfam_families.json` |
| 7 | `fetch_alphamissense_mechanism` | `alphamissense_scores_full.json` |
| 8 | `build_valid_variants` | `valid_variants.json` |

### Step 2 — embed variants (GPU)

**SKIPPED.** Reuses the existing arrays under `data/embeddings/<ESM2_MODEL>/` (see *Embeddings
are NOT re-extracted*). Verify before proceeding: all four `.npy` arrays are `(17826, 1280)` and
`embedded_variants.json` has 17,826 rows.

### Step 3 — run analysis

| Command | Outputs |
|---|---|
| `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 5` | `results/<run>/family_split_baselines_seed{0..4}.json` |
| `python -m esm2_mech.experiments.mechanism.mlp --seeds 5` | `results/<run>/nonlinear_results_seed{0..4}.json` |
| `python -m esm2_mech.experiments.mechanism.family_clustering --seeds 5` | `results/<run>/family_clustering.json` |
| `python -m esm2_mech.experiments.mechanism.naive_baseline` | `results/<run>/naive_baseline.json` |
| `python -m esm2_mech.experiments.mechanism.leakage_fraction` | `results/<run>/leakage_fraction.json` |

`family_clustering` gains `--seeds` in run_biorxiv: run6 reported the family-probe accuracy at seed 0
only, and `STATS_PLAN.md` requires a spread to match the other reports.

`leakage_fraction` reads only the result JSONs above (no model inference), so it runs last. It
also needs a CI: run6 reported ~40% as a bare point estimate with no interval, and that number
is quoted in the intro report as a headline.

**Stats flags.** `classify_by_mechanism`, `single_source_mechanism`, and
`mechanism_delta_family_split` accept:

- `--no_ci` — skip cluster-bootstrap CIs (faster; not for the run_biorxiv replay).
- `--n_boot N` — bootstrap resamples, default 1000.
- `--n_permutations N` — label-permutation p-value against chance, default 0 = off.

CIs are on by default and add roughly a minute per seed.

### Step 3b — permutation tests (GPU, seed 0 only)

Run separately from Step 3: the permutation test refits the probe once per repeat, so it
multiplies per-seed probe time by N and belongs in its own tmux window.

```bash
python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 1 --n_permutations 1000
```

**Seed 0 only, deliberately.** A permutation test constructs its own null by shuffling labels;
running it across 5 seeds mostly re-measures fold jitter, which is precisely what run_biorxiv exists to
replace. Seed 0 cuts this 5× at no inferential cost.

**Budget split by probe.** The headline claim — `delta_mean` sits at the chance floor — is a
linear-probe claim, so the linear permutation is load-bearing and the MLP permutation is the
expensive tail:

- **Linear probe: 1,000 permutations.** Non-negotiable; this is the tested claim.
- **MLP: whatever N the measured per-refit cost supports, with N stated explicitly** in the
  result file and the report. A smaller, honestly-labelled N beats delaying the run for a round
  number — but never report a p-value sitting at its resolution floor of 1/(N+1), which is
  exactly the unresolved `wt_only_mean` = 0.0099 case from run6.

**Before launching either, time a single refit on the pod.** At 4 features × 2 splits × 1,000
permutations this is 8,000 refits and the per-refit cost has never been measured. It determines
whether this step is hours or days, whether it needs joblib parallelism across the pod's cores,
and what N the MLP tail gets. This is the run's main schedule risk — everything else in run_biorxiv is
cheap.

**This step does NOT cover the split gap.** That is a paired-bootstrap quantity (Stage 0b), not a
permutation quantity.

### Step 4 — single-source robustness check (CPU)

Unchanged in design; gains CIs. Re-runs the Step 3 probe on the Gerasimavicius-only subset
(10,138 variants, all three classes from one curation pipeline) and recomputes the majority
floor on the subset (0.279, not the merged 0.288).

```bash
python -m esm2_mech.experiments.mechanism.single_source_mechanism --seeds 5
```

**Output:** `results/<run>/single_source_gerasimavicius/{family_split_baselines_seed{0..4}.json,
aggregate.json, naive_baseline.json}`

The run_biorxiv question is narrower than run6's: not whether `delta_mean` sits at the floor, but
whether its **interval still straddles the floor**, and what the CI on `wt_only`'s
gene-minus-family gap (0.612 → 0.445) looks like.

---

## Experiment 2 — pathogenicity positive control

Tests whether the same delta embeddings that classify mechanism at chance predict ClinVar
pathogenic-vs-benign. Pass criterion: `delta_mean` MLP AUROC ≥ 0.85.

**Phase 2 (embedding) is SKIPPED** — `pathogenicity_{wt,mut}_mean.npy` already exist and are
fingerprint-verified. Only the probe phase re-runs, on CPU.

```bash
python -m esm2_mech.experiments.pathogenicity.pathogenicity_control --model esm2_t33_650M_UR50D
```

Requires wiring per Stage 0a. Classes are balanced here, but the gene-level dependency structure
still applies, so CIs resample whole genes. Add the calibration note: the probes measure
discrimination only and are not risk estimates.

---

## Experiment 3 — within-family mechanism (CPU)

Reads the existing embedding arrays under `data/embeddings/<ESM2_MODEL>/`; no GPU, no RunPod.

```bash
python -m esm2_mech.experiments.mechanism.mechanism_within_family --seeds 5
```

**Output:** `results/<run>/within_family_mechanism.json`

Already imports `utils/bootstrap.py`. Three additions required by `STATS_PLAN.md`, all of which
change how the table must be *read* rather than what it contains:

1. **Benjamini-Hochberg FDR correction** across the 28-family screen (two views × two probes), or
   an explicit restatement of the table as exploratory. The run6 "beats baseline and std < 0.10"
   highlight is an uncorrected screen.
2. **Minimal-detectable-effect per family**, so the nulls read as underpowered rather than as
   evidence of absence. At 6–33 genes per family the test cannot establish absence.
3. **Cluster-bootstrap CIs over genes within each family**, replacing seed-std.

---

## Experiment 4 — ESM-3: scale and structure

**Phases 1 and 2 are SKIPPED** — structure tokens are cached in
`data/cache/esm3_struct_tokens.json` and embeddings exist under
`data/embeddings/<ESM3_MODEL>/merged/`. Phase 3 only:

```bash
python -m esm2_mech.experiments.esm3.esm3_mechanism --phase 3 --dataset merged --seeds 5
```

**Use `--dataset merged` for the headline.** It matches the Experiment 1 ESM-2 variant set
exactly (same `valid_variants.json`), so it is the only apples-to-apples comparison for a scale
claim. The geras run is not a matched baseline and must not be used for it.

Gates M1/M2/M3 are unchanged. The run_biorxiv addition is the **paired cluster bootstrap** on `seq` −
ESM-2 delta and on `seq_struct` − `seq`, over genes on the shared variant set. M2 clears its
threshold by 0.008, so the current claim is a point-estimate comparison at a margin thinner than
one seed of spread; run_biorxiv either supports it with a tested gap or does not.

---

## Experiment 5 — geometry of the pathogenicity direction

| Step | Command | GPU |
|---|---|---|
| 1 build | `geometry.build_canonical_pathogenicity` | no |
| 2 probes | `geometry.run_geometry --seeds 5` | no |
| 3 conservation extract | `geometry.conservation_axis --extract` | **yes** |
| 4 conservation analysis | `geometry.conservation_axis` | no |

Step 3 is one of the three genuine GPU steps: it masks each variant position and reads the
masked-LM `logP_wt` / `logP_mut` / entropy. It can share a pod session with the Step 3b
permutation work.

run_biorxiv additions: gene-cluster CIs on each pathogenicity AUROC (effective N ≈ 1,929 genes, not
37,218 variants), and a **paired cluster bootstrap on the two load-bearing gaps** — conservation
(0.891) versus embedding delta (0.859), and the conservation-plus-delta increment (+0.002, which
is the entire basis for gate K2). Also attach a paired CI to the task transfer contrast
(pathogenicity 0.85–0.90 vs mechanism 0.62–0.64).

---

## Experiment 6 — contrastive metric learning

```bash
python -m esm2_mech.experiments.mechanism.contrastive_mechanism --seeds 5
```

GPU-resident but small — ~2 minutes for 5 seeds. Reads the MLP floor live from `aggregate.json`;
never hardcode it.

run_biorxiv additions: paired cluster bootstrap over genes on the contrastive-vs-raw-kNN gap (+0.041);
a permutation test for the contrastive macro-F1 against **both** the 0.288 MLP floor and the
raw-kNN baseline; and gene-cluster CIs on the per-class AUROCs, so run6's honest caveat — that
the gain is class balance rather than per-class separability, with DN unmoved — is reported as a
tested null rather than a point drop.

---

## Experiment 7 — megascale stability positive control

A second positive control with a purely physical label (Tsuboyama 2023 ΔΔG), free of clinical
curation.

**Step 2 (embedding) is SKIPPED** — `megascale_{wt,mut}_{mean,pos}.npy` already exist.

| Step | Command | GPU |
|---|---|---|
| 1 families | `stability.build_domain_families` | no |
| 3 linear probe | `stability.megascale_stability` | no |
| 4 nonlinear probe | `stability.megascale_mlp --xgboost` | **yes** |
| 5 controls | `stability.stability_baselines` | no |

Step 1 needs `hmmscan` and a hmmpress-ed Pfam-A; skip if `megascale_domain_families.json` is
present and non-empty. Step 4 is the third genuine GPU step.

Gates unchanged: H1 random-split ρ ≥ 0.5, H2 LEAKY threshold 0.10 on the random→family drop, H4
per-domain ρ std, H3 mechanism-F1 change ≤ +0.01.

---

## Stage 2 — remaining statistical work

Machinery exists for all of these; none was implemented in run6.

- **AUPRC with prevalence baseline, and PPV/NPV at class prevalence**, for the rare classes
  (DN ≈ 9%, GOF ≈ 15%). AUROC alone overstates usefulness at those rates.
- **Calibration note in every probe report** — the probes are uncalibrated and measure
  discrimination only, not risk. State it rather than fix it; the claims are about
  discrimination.
- FDR correction and minimal-detectable-effect for the within-family table (Experiment 3).
- Multi-seed family probe (Experiment 1 Step 3).

---

## Stage 3 — regenerate reports

All 13 reports in `reports/run6/` are rewritten against run_biorxiv result files into
`reports/run_biorxiv/`. None of the run6 reports cites a confidence interval, including
`report_classifier.md` (the headline) and `report_leakage_fraction.md` (no interval on the ~40%
figure).

Per the project report rules: a result file and its report share the same `RUN_NAME`, and every
number traces to a file under `results/run_biorxiv/` cited in Provenance. Where a run_biorxiv report cites a
quantity computed on the reused embeddings, Provenance says so.

---

## Verification checklist

Data and alignment:

- [ ] `data/embeddings/<ESM2_MODEL>/embedded_variants.json` row count matches all four `.npy`
      arrays (17,826). This file is a write-only provenance artifact — no code reads it.
- [ ] `data/pfam_families.json` has entries for ≥ 1,900 genes.
- [ ] `data/alphamissense_scores_full.json` covers > 90% of `valid_variants.json`.
- [ ] Embedding fingerprints recorded in the run_biorxiv result files match the run6 arrays.

Statistics — the point of this run:

- [ ] Every result file under `results/run_biorxiv/` that reports a macro-F1 or AUROC carries a CI key.
      Zero CI-less headline files.
- [ ] **Family-split CIs resample families, gene-split CIs resample genes.** Effective cluster
      count is emitted alongside every family-split interval. Family-split intervals are wider
      than gene-split ones — if they are not, the unit was applied wrongly.
- [ ] Rare-class (DN, GOF) AUROC intervals use BCa where computable, and are labelled as the
      least trustworthy in their table regardless of method.
- [ ] Linear-probe permutation ran at 1,000. MLP permutation N is stated explicitly wherever its
      p-value appears. No p-value equal to 1/(N+1) is reported as a measurement — that is the
      resolution floor, not a result.
- [ ] `PERMUTATION_FEATURES` covers all four above-floor features plus the `delta_mean` control.
- [ ] All six paired claims (ESM-3 M2, contrastive +0.041, geometry K2, transfer contrast,
      contrastive DN null, and the gene-vs-family split gap) report a paired cluster-bootstrap CI
      on the difference, not two separated error bars.
- [ ] The gene-vs-family split gap reports a paired-bootstrap CI, **not** a label-permutation
      p-value — the permutation null for that gap is zero by construction.
- [ ] Paired bootstrap resamples the cluster unit once per replicate and applies the same
      resample to both arms; unit tests assert this directly, and cover both the same-fold and
      cross-partition pairing modes.
- [ ] Every report carries the calibration note.
- [ ] Every confirmatory gate reports its verdict under the pre-registered CI rule — affirmed /
      not distinguishable / failed / underpowered — never a bare "pass" beside an interval
      through zero.
- [ ] BH-FDR applied across the six confirmatory claims, raw and adjusted values both shown.
      Exploratory tables are labelled, not corrected.
- [ ] `scripts/compare_runs.py run6 run_biorxiv` run, its output archived as the delta note, and every
      material movement explained.

Science — the spine, which should not move:

- [ ] Pathogenicity `delta_mean` MLP AUROC ≈ 0.89 and passes ≥ 0.85.
- [ ] Mechanism `delta_mean` at the measured floor on both splits; `wt_only` gene-split lift
      collapses under family-split.
- [ ] Stability H1 random-split ρ ≥ 0.5.
- [ ] ESM-3 compared against ESM-2 only on `--dataset merged`.
- [ ] Any run_biorxiv number that moves materially from run6 is explained, not silently adopted.

- [ ] Environment pinned and recorded in Provenance; CI green on the commit that produced the run.
- [ ] `git status` clean; results committed.
