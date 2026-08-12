# run_biorxiv runbook — inferential statistics

Purpose: produce `run_biorxiv` end-to-end so every report carries error bars that account for genes
in the same family not being independent, p-values where a claim needs one, and a real test behind
every "beats" claim — in place of run6's 5-seed fold-jitter error bars.

Supersedes `RUNBOOK_4.md` (run6). The experiments, gates, and hypotheses are unchanged — run_biorxiv
re-scores the same science with correct error bars. Statistical methodology is
[`reports/run6/STATS_PLAN.md`](../reports/run6/STATS_PLAN.md); the change list is
[`PLAN_biorxiv.md`](PLAN_biorxiv.md).

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

No copying is needed: embedding paths are keyed by *model*, not by run. `EMB_DIR` is
`data/embeddings/<ESM2_MODEL>/` and `ESM3_EMB_DIR` is `data/embeddings/<ESM3_MODEL>/`
(`utils/paths.py:68-69`), neither of which contains `RUN_NAME`. The arrays are ~10 GB and
gitignored, so copying per run would cost the space for no provenance gain.

Recorded consequence: **run_biorxiv result files are scored on embeddings extracted during run6.**
Each run_biorxiv result JSON records the array fingerprint (the existing check used by Experiments 2
and 5), so the reuse shows up in the output rather than only in this runbook.

GPU is still required for three steps, which are computed rather than cached:

| Step | Why GPU |
|---|---|
| Experiment 5 step 3 — conservation extract | masked-LM forward pass per variant |
| Experiment 7 step 4 — megascale nonlinear probe | MLP/XGBoost training |
| Permutation tests (Experiment 1) | the probe is refit once per permutation |

---

## Stage 0 — preconditions

run_biorxiv must not start until all of these hold. 0a/0a-bis/0b are the substance of the run,
0b-bis fixes how its results may be read, and 0c–0e protect its provenance. Live status is in
`RUN_PROGRESS_biorxiv.md`; this document states what must be true, not what has been done.

### 0a. Stats machinery wired and verified

Eleven modules import `utils/bootstrap.py` and emit CI keys: `naive_baseline`,
`mechanism_delta_family_split`, `mechanism_within_family`, `mechanism/mlp.py`,
`mechanism/contrastive_mechanism.py`, `esm3/esm3_mechanism.py` (phase 3),
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

### 0a-bis. Methodology rules the wiring must implement

Settled in `PLAN_biorxiv.md` Task 0. These are properties of the emitted numbers, so getting
them wrong means re-running, not just re-reporting.

1. **The resampling unit matches the split.** Gene-split metrics resample genes; family-split
   metrics resample **families**, because under family-split the family is the held-out unit and
   genes within one are not independent draws. Family-split CIs will be visibly wider — 1,134
   families but 833 singletons, so far fewer effective clusters than genes. That is correct, not
   a bug. Emit the effective cluster count next to every family-split interval.
2. **Rare-class AUROC is flagged, not bias-corrected.** DN (≈ 9%, ~150–170 genes) and GOF
   (≈ 15%) sit in the regime where percentile bootstrap undercovers for a bounded metric with few
   clusters. They use the same percentile cluster bootstrap as everything else, keep the existing
   degenerate-fold suppression guard, and are labelled the least trustworthy intervals in their
   table. No confirmatory claim rests on them — per-class AUROCs are exploratory under R7.2.

### 0b. Paired cluster bootstrap

`utils/bootstrap.py` provides `paired_cluster_bootstrap_diff` (same-fold) and
`paired_cluster_bootstrap_diff_cross_partition`, with `paired_oof_diff` wrapping both: it aligns two
arms by `row_ids`, takes the class list as a parameter, and supports macro-F1, binary AUROC and
one-vs-rest AUROC. `adjudicate_diff` and `adjudicate_level` render the R7.1 verdict for a difference
and for a level respectively. The call sites are `esm3_mechanism.py`,
`contrastive_mechanism.py`, `conservation_axis.py` and `mechanism_delta_family_split.py`.

Six comparisons below rest on two point estimates with separated error bars, and the thinnest
margins are smaller than a seed of spread. **Five are paired**; the transfer contrast is not, for
the reason given in its row:

| Claim | Margin | Report |
|---|---|---|
| ESM-3 seq vs ESM-2 MLP delta_mean | clears `m1_threshold` (the measured floor + 0.05) by 0.008 in run6 | `report_esm3_mechanism.md` |
| Contrastive k-NN vs raw-delta k-NN | +0.041 | `report_contrastive.md` |
| Conservation vs embedding delta (gate K2) | +0.002 | `report_geometry.md` |
| Pathogenicity vs mechanism cross-family transfer | 0.85–0.90 vs 0.62–0.64 | `report_geometry.md` — **not paired**: different datasets, no shared row space |
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
correct under family resampling, so a gene-resampled gap understates it. Report the gene-resampled
interval alongside as a labelled sensitivity check.

**Two pairing modes, and they are different code paths.** The ESM-3, contrastive, and conservation
comparisons share a fold assignment. The split-gap comparison does not — gene-split and family-split
are different CV partitions by definition — so its pairing is across two fold assignments: resample
families, then recompute each arm under its own partition. Written without distinguishing these, the
cross-partition case gets silently implemented as the same-fold path and is wrong.

### 0b-bis. Pre-registered decision rules written into PREREGISTRATION_run_biorxiv.md

Both rules below must be written down **before** run_biorxiv executes, or the run produces intervals
with no stated reading and any interpretation chosen afterwards is retro-fitted:

1. **CI decision rule for every confirmatory gate.** A gate is affirmed only if its point estimate
   clears the threshold *and* the paired difference 95% CI excludes zero; if the point estimate
   clears but the CI spans zero, the claim is restated as **not distinguishable**. A gate failing
   with a CI that also spans the threshold is reported as **underpowered to detect an effect of
   the pre-registered size**, not as evidence of no effect.
2. **The confirmatory / exploratory split.** Five confirmatory claims (C1–C5 in
   `PLAN_biorxiv.md` Task 0.2), enumerated before the run. Everything else is labelled
   exploratory and asserts nothing the paper relies on. No multiplicity correction is applied;
   R7.2 records why none is needed across a set this size. C1 is a null claim and is adjudicated
   against the pre-registered 0.05 equivalence margin, not by an interval overlapping the floor.

Both go into `PREREGISTRATION_run_biorxiv.md` with the run6 point estimates recorded, so the
rules cannot be tuned to the run_biorxiv intervals.

### 0c. A pinned environment

- `pytest tests/` passes on the commit that produces the run. **A green suite is a precondition
  for flipping `RUN_NAME`**, run locally; there is no CI job.
- Runtime dependencies pinned via `uv.lock`, with the pinned set recorded in the reports'
  Provenance — it is what the numbers were produced under, and it prevents the sklearn-version
  hazard documented in `CLAUDE.md` (`multi_class=` removed in ≥ 1.8). Trimming unused exploratory
  packages is housekeeping, not part of this run.
- `scripts/compare_runs.py` exists and passes its self-diff invariant (run6 against run6 must
  report zero movement).

### 0d. Configuration

- `RUN_NAME = "run6"` → `"run_biorxiv"` in [`utils/paths.py:11`](../src/esm2_mech/utils/paths.py#L11). One
  line; `RESULTS_DIR`, `RUN_REPORTS_DIR`, and `FIGURES_DIR` all derive from it. **Flip this only
  after 0a and 0b pass their gates** — flipping first means the replay writes CI-less files into
  `results/run_biorxiv/`, and fixing them later either overwrites run_biorxiv provenance or forces a run8.
- `PERMUTATION_FEATURES` stays at `("delta_mean", "wt_only_mean")`. `delta_mean` is C1's
  instrument and `wt_only_mean` is the above-floor comparison; the remaining features are
  exploratory, so each one added costs another 2,000 refits for a p-value no claim reads. The same
  constant gates which features cache OOF for the split gap, so widening it enlarges that cache too.
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

`leakage_fraction` reads only the result JSONs above (no model inference), so it runs last, and it
needs a CI — run6 reported ~40% as a bare point estimate, quoted as a headline in the intro report.

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

**Seed 0 only, deliberately.** A permutation test constructs its own null by shuffling labels, so
running it across 5 seeds mostly re-measures the fold jitter run_biorxiv exists to replace. Cuts the
step 5× at no inferential cost.

**Linear probe only, at 1,000 permutations.** The headline claim — `delta_mean` sits at the chance
floor — is a linear-probe claim, so that is the load-bearing test and it runs at full N. The MLP is
not permutation-tested: no claim rests on an MLP permutation p-value and its refits are the
expensive tail. Never report a p-value sitting at its resolution floor of 1/(N+1), the unresolved
`wt_only_mean` = 0.0099 case from run6.

**Before launching, time a single refit on the pod.** At 2 features × 2 splits × 1,000 permutations
this is 4,000 refits and the per-refit cost has never been measured. It decides whether the step is
hours or days, and whether it needs joblib parallelism across the pod's cores. This is the run's
main schedule risk — everything else in run_biorxiv is cheap.

**This step does NOT cover the split gap.** That is a paired-bootstrap quantity (Stage 0b).

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
whether its **interval still straddles the floor**, and what the CI on `wt_only`'s gene-minus-family
gap (0.612 → 0.445) looks like.

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

Already imports `utils/bootstrap.py`. Three additions required by `STATS_PLAN.md`, all changing how
the table is *read* rather than what it contains:

1. **The table is labelled an exploratory screen.** The run6 "beats baseline and std < 0.10"
   highlight is uncorrected and is labelled rather than corrected; correcting it would imply it had
   been a confirmatory test.
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
threshold by 0.008 — a margin thinner than one seed of spread — so run_biorxiv either supports it with
a tested gap or does not.

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
a permutation test for the contrastive macro-F1 against **both** the 0.288 MLP floor and the raw-kNN
baseline; and gene-cluster CIs on the per-class AUROCs, so run6's caveat — the gain is class balance
rather than per-class separability, with DN unmoved — is reported as a tested null rather than a
point drop.

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
- Exploratory labelling and minimal-detectable-effect for the within-family table (Experiment 3).
- Multi-seed family probe (Experiment 1 Step 3).

---

## Stage 3 — regenerate reports

The 14 documents in `reports/run6/` — 11 per-experiment reports plus `ESM2_REPORT.md`,
`INTRO_REPORT.md`, and `STATS_PLAN.md` — are rewritten against run_biorxiv result files into
`reports/run_biorxiv/`, with two exceptions and two additions. `report_esm3_mechanism_geras.md` is
dropped and cited from the run6 archive instead (it is already marked superseded, and regenerating a
report that must not be cited invites citing it); `STATS_PLAN.md` moves from a plan to a record of
what was done. The additions are a paired-difference summary table and the run6→run_biorxiv delta note
generated by `compare_runs.py`.

None of the run6 reports cites a confidence interval, including `report_classifier.md` (the headline)
and `report_leakage_fraction.md` (no interval on the ~40% figure).

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
- [ ] The five paired claims (ESM-3 M2, contrastive +0.041, contrastive per-class DN null,
      geometry K2, and the gene-vs-family split gap) report a paired cluster-bootstrap CI on the
      difference, not two separated error bars.
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
- [ ] ESM-3 compared against ESM-2 only on `--dataset merged`.
- [ ] Any run_biorxiv number that moves materially from run6 is explained, not silently adopted.

- [ ] Environment pinned and recorded in Provenance; `pytest tests/` green on the commit that
      produced the run.
- [ ] `git status` clean; results committed.
