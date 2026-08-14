# run_biorxiv plan — re-run the pipeline with inferential statistics

**Drafted:** 2026-07-20
**Revised:** 2026-07-22

**Goal.** Re-run the pipeline as `run_biorxiv` so every reported number carries a real error bar
that accounts for genes in the same family not being independent, a p-value where a claim needs
one, and an actual test wherever the paper says one thing beats another. The run6 error bars only
show seed-to-seed wobble across 5 seeds, which says nothing about whether a result holds on a
different set of genes.

Five experiments run. ESM-3 (scale and structure) and contrastive metric learning are cut from
this run; the preprint's argument is the mechanism null, its chance floor, the homology-leakage
account, and the two positive controls. Run6 results for both cut experiments stand in the run6
archive.

The 2026-07-22 review turned up problems that must be settled before implementation (marked ★
below): pathogenicity provenance, pre-registered pass/fail rules, a named list of the paper's
load-bearing claims, resampling the unit the split uses, rare-class error bars, the permutation
budget, a paired test for the split-gap error bar, and enforced production quality.

Reading Badonyi & Marsh 2025 (`papers/mechanism_2025.pdf`) added a citable objection: they show one
gene often carries two different mechanisms, which makes "the gene labels are noisy" a specific
challenge to claim C1. Three tasks answer it — 2d (re-run C1 on genes with unambiguous labels), 8
(measure how much wrong labels hurt a working probe), and 9 (show other features find mechanism in
the same labels) — all deferred to [`FOLLOWUP_biorxiv.md`](FOLLOWUP_biorxiv.md).

**Dropped from scope.** Task 2b (homology-partition robustness panel) is cut outright. Task 2d
moves to the follow-up doc: a clean-label filter leaves DN at ~50 genes, too few for a
family-resampled CI, and Task 8 answers the same objection without depending on subset size.
Task 4 (runbook) is done; `RUNBOOK_biorxiv.md` supersedes it.

Scope reference: [`reports/run6/STATS_PLAN.md`](../reports/run6/STATS_PLAN.md) is the statistical
methodology; [`RUNBOOK_biorxiv.md`](RUNBOOK_biorxiv.md) is the execution spec (supersedes
`RUNBOOK_4.md`); [`RUN_PROGRESS_biorxiv.md`](RUN_PROGRESS_biorxiv.md) is the live status table.
This plan covers what must change before the runbook is replayed.

**Outputs.** Results go to `results/run_biorxiv/`, reports to `reports/run_biorxiv/`.
`results/run6/` and `reports/run6/` stay untouched as the comparison baseline.

**Every ClinVar-derived input is rebuilt.** The ClinVar fetch was re-run from scratch on
2026-08-11, so `valid_variants.json` comes from a current snapshot, and the pathogenicity control
set is refetched on that snapshot rather than reused, since its params sidecar was lost. Every
array row-aligned to those lists is extracted again: ESM-2 variant embeddings, pathogenicity
embeddings, and conservation scores. Megascale arrays are reused, since they derive from Tsuboyama
data with no ClinVar dependency. Embedding paths are keyed by model, not run
(`utils/paths.py:68-69`), so no copy is needed. Result files record the embedding fingerprint, so
the arrays behind each result are traceable.

Because inputs changed, only Experiment 7 lets a run6→run_biorxiv movement be attributed to the
new statistics alone. Everywhere else two things moved at once, and the delta note must attribute
movement to both rather than reading it as a pure statistical correction.

GPU is needed for: Experiment 1 step 2 (ESM-2 embed), Experiment 2 phase 2 (pathogenicity embed),
Experiment 5 step 3 (conservation extract), Experiment 7 step 4 (megascale MLP), and the
Experiment 1 permutation tests, which refit the probe per repeat.

---

## Current state

`utils/bootstrap.py` is built and complete: `cluster_bootstrap_ci`, `bootstrap_mechanism_metrics`,
`label_permutation_pvalue`. Three experiment modules import it — `naive_baseline.py`,
`mechanism_delta_family_split.py`, `mechanism_within_family.py` — not six, as the original plan
assumed.

What has landed in `results/run6/`:

| | Status |
|---|---|
| Files with cluster-bootstrap CIs | 5 of 27 (`family_split_baselines_seed{0..4}.json` only) |
| Files with permutation p-values | the same 5, and only 2 features within them |
| Reports citing a CI | 0 of 14 |
| Paired difference tests | not implemented |

The two existing p-values (family-split, seed 0, 200 permutations):

- `wt_only_mean` — observed 0.409, null mean 0.325, p = 0.0099
- `delta_mean` — observed 0.289, null mean 0.319, p = 1.0

`wt_only_mean`'s p = 0.0099 is exactly 1/(200+1), the resolution floor of a 200-permutation test;
the true value is unresolved below 0.01. This is the reason to run at the plan's specified 1,000.
`delta_mean` scoring below its own null is consistent with the at-floor result reported elsewhere.

The restriction to two features is `PERMUTATION_FEATURES = ("delta_mean", "wt_only_mean")` at
[`mechanism_delta_family_split.py:108`](../src/esm2_mech/experiments/mechanism/mechanism_delta_family_split.py#L108),
a module-level constant, not a flag. `PERMUTATION_N_RESAMPLES` already defaults to 1000 in
`constants.py`; the 200 in the current files came from a run-time override.

---

## Task 0 — ★ correctness and methodology (settle before any code)

Each item below changes what the machinery must emit, so it is decided before implementation
rather than mid-run.

### Freeze and document the pathogenicity variant set

The paper's headline positive control was recorded on two different variant sets. Seed 0 ran on
RunPod on the full set and scored AUROC 0.878; seeds 1–4 ran locally on a truncated set and scored
0.742 ± 0.006 (`result_6.md`). `docs/README.md` still calls this pending.

The 0.74–0.88 range looks like seed variation but is two experiments stacked together: the
±0.006 shows the probe is stable within either set, so the whole width comes from the sets
differing. Bootstrapping does not fix this, since it measures sensitivity to gene sampling, not to
a change of dataset.

★ **Run6 already fixed this in code (verified 2026-07-22).** Experiment 2 was rebuilt as a single
fetch → embed → probe module over one canonical set of 37,218 balanced ClinVar variants (1,929
genes, GRCh38). `pathogenicity_control.py` fingerprints the variant set (`variants_fingerprint`,
line 306), stores the fingerprint in the embedding metadata (line 332), and refuses to proceed if
the embeddings do not match the current variant set (line 360). All five run6 seeds agree:
`delta_mean` MLP family-split = 0.894, gene-split = 0.897, std ≤ 0.001.

What remains is documentation, not re-derivation:

- Correct `docs/README.md`'s "pending" note to state the canonical set (37,218 variants / 1,929
  genes / GRCh38) and the run6 numbers.
- Mark `result_6.md`'s 0.74–0.88 band as superseded by run6.
- Record the variant-set fingerprint in the run_biorxiv result files and quote it in each report's
  Provenance.
- Verify at run_biorxiv time that all five seeds share one fingerprint and that the spread is
  ≤ 0.01. If any seed disagrees, stop — that is a data defect requiring a freeze-and-rerun.

### ★ Pre-registered CI decision rules for the confirmatory gates

K2 currently passes by a thin margin (0.002). A paired CI puts one error bar on the gap between
two arms rather than a separate error bar on each. That raises a case the current gates do not
define: a gate can pass on its point estimate while its difference CI spans zero.

**Rule, pre-registered before the run:**

> A gate is affirmed only if its point estimate clears the threshold and the paired difference
> 95% CI excludes zero. If the point estimate clears but the CI spans zero, the claim is restated
> as not distinguishable — not a pass, not a refutation. If the point estimate fails, the gate
> fails regardless of the CI.

This applies to the confirmatory gates only.

| Gate | Claim | Margin vs 0.430 |
|---|---|---|
| K1 | conservation alone clears 0.85 | +0.041 |
| K2 | embedding adds over conservation | +0.002 (fails) |
| H2 | stability random→family drop under LEAKY 0.10 | descriptive |

K2 already fails, so a CI spanning zero reinforces that reading.

★ For failing gates, add an underpowered-null distinction. K2 (+0.002 against a 0.02 bar) is a
null from an underpowered comparison. If its CI is wide, the honest reading is that the data
cannot resolve an effect of the pre-registered size, not that no effect exists.

Everything else — per-class AUROCs, the 28-family table, within-family per-family cells,
biochemistry R², the magnitude/direction decomposition, per-feature leakage fractions — is
exploratory and labelled as such in its report.

★ No multiplicity correction is applied; R7.2 of `PREREGISTRATION_run_biorxiv.md` records why.

★ C1 is adjudicated as a null claim. An interval straddling the chance floor is not evidence of
sitting at it, since a wide enough interval straddles anything. C1 is affirmed only if its CI upper
bound falls below the measured floor plus 0.05, and is otherwise recorded as not adjudicated. The
permutation test can refute C1 but not confirm it.

C1 and C3 are the load-bearing pair (the dissociation); C2 is the leakage account. If the
confirmatory set needs trimming, trim from C4, not C1–C3.

★ **The label-heterogeneity threat is named and cited.** Badonyi & Marsh 2025 (bioRxiv
2025.03.13.642984) report that 43% of multi-phenotype dominant genes and 49% of mixed-inheritance
genes carry both LOF and non-LOF mechanisms. Since this project assigns one mechanism label per
gene, some fraction of the 17,826 variants is mislabelled by construction — a citable alternative
explanation for C1: the delta sits at the floor because the labels are noisy, not because the
embedding lacks mechanism signal. No confidence interval addresses this, since the threat is to
what the labels mean, not to sample size. Tasks 2d and 8 in
[`FOLLOWUP_biorxiv.md`](FOLLOWUP_biorxiv.md) answer it. C1's statement in the reports must cite
this paper and point at those results rather than asserting the labels are adequate.

Deliverable: written into [`PREREGISTRATION_run_biorxiv.md`](PREREGISTRATION_run_biorxiv.md)
alongside the CI decision rules, before the run.

### Resample the unit that defines the split

The split is how cross-validation folds are drawn. Under gene-split, all variants of one gene go
into the same fold. Under family-split, all genes in one Pfam family go into the same fold, which
additionally blocks the model from scoring one gene using a close relative it saw in training.

The unit is what the bootstrap draws with replacement when building the error bar: a gene, or a
whole family.

`STATS_PLAN.md` says CIs resample whole genes or whole families depending on the unit, but
currently defaults to gene-resampling for the classifier report on both splits. A bootstrap treats
drawn items as independent, and family-split has already declared that genes in the same family
are not — that is why they share a fold. Resampling genes therefore claims 1,935 independent
items when the evaluation has 1,134 families, 833 of them singletons. Too many assumed-independent
items narrows the interval below what the data supports.

**Rule: the resampling unit matches the split.** Gene-split resamples genes; family-split
resamples families. This applies to every CI on a family-split metric, in every report.

**Expected consequence, stated in advance.** Family-split CIs will be visibly wider than
gene-split ones, since the effective cluster count (1,134 families, 833 singletons) is far below
the gene count. Report the effective cluster count alongside each family-split interval.

**The split-gap comparison resamples families**, the coarser of the two units, since a family
resample induces a valid gene resample but not the reverse. Report the gene-resampled interval
alongside as a sensitivity check, labelled as the narrower and anticonservative of the two.

### Rare-class intervals carry a health warning, not a correction

DN is roughly 9% of variants — 150–170 genes, fewer effective clusters under family-split.
Percentile bootstrap undercovers for a bounded metric (AUROC ∈ [0,1]) near its boundary with few
clusters, which is this regime.

- Rare-class one-vs-rest AUROC uses the same percentile cluster bootstrap as every other interval.
  No bias correction is applied, since over ~150 clusters the correction is itself noisy. No
  confirmatory claim reads these intervals; per-class AUROCs are exploratory.
- Flag every rare-class interval as the least trustworthy in its table.
- Keep the existing suppression guard for degenerate folds.

### Permutation budget — linear probe only

The permutation test refits the probe per repeat; the MLP is the expensive tail.

- **Linear probe: 1,000 permutations.** The headline claim — `delta_mean` sits at the chance
  floor — is a linear-probe claim, so this fully resolves it. Cheap.
- **MLP: not permutation-tested.** ★ Revised 2026-08-11: no claim rests on an MLP permutation
  p-value, and its refits are the run's main schedule risk. Dropped rather than run at reduced N.

No p-value is reported at its resolution floor (see the `wt_only_mean` p = 1/(200+1) case above).
A single timed refit on the pod is still needed before launching the linear permutation.

---

## Task 1 — paired cluster bootstrap (new code) ✅ done 2026-08-10

The one new statistics code in the run: an error bar on the difference between two results rather
than on each separately, as the CI decision rules require. K2 (+0.002) previously showed as two
point estimates with separate error bars at a margin smaller than a seed of spread.

Two versions exist, since the arms are not always scored the same way.
`paired_cluster_bootstrap_diff` handles arms sharing one fold assignment (conservation vs.
embedding). `paired_cluster_bootstrap_diff_cross_partition` handles arms from different fold
layouts — only the gene-split-minus-family-split gap — resampling families and re-scoring each arm
under its own partition. Its optional `sensitivity_clusters` arg returns the gene-resampled version
the resampling-unit rule requires alongside.

Both live in `utils/bootstrap.py` and hand `metric_fn_a`/`metric_fn_b` the identical drawn
row-index array per replicate, so the difference is paired rather than independently resampled.
`tests/utils/test_bootstrap.py` covers both modes, the shared-resample property, planted-difference
and no-difference recovery, undefined-replicate dropping, and `ci_suppressed`.

Wiring into the experiment modules is Task 2.

---

## Task 2 — wire existing machinery into 5 modules

Mechanical: `classify_by_mechanism` is the reference implementation to copy. Each module needs
gene/family clusters passed to `bootstrap_mechanism_metrics`, the `--no_ci` / `--n_boot` flags
added, and CI keys emitted into its result JSON.

| Module | Experiment |
|---|---|
| `mechanism/mlp.py` | 1 (Step 3, nonlinear probes) |
| `pathogenicity/pathogenicity_control.py` | 2 |
| `geometry/run_geometry.py` | 5 |
| `stability/megascale_stability.py` | 7 |
| ★ `mechanism/family_clustering.py` | 1 (Step 3, diagnostic) |

★ `family_clustering.py` was missing from the original plan. `STATS_PLAN.md` requires the
family-probe accuracy to be multi-seed (run6 is seed 0 only) with cluster-bootstrap CIs over
families for both probe accuracy and k-NN purity. That is a code change — the module has no
`--seeds` flag — so it belongs here rather than in Task 5.

★ `leakage_fraction.py` also needs a CI. `report_leakage_fraction.md` quotes ~40% with no
interval, in both `INTRO_REPORT.md` and `ESM2_REPORT.md` §4. It is a derived ratio,
(gene − family) / (gene − chance), of quantities that will each carry CIs. Recompute the whole
ratio per bootstrap replicate rather than combining two separate intervals, since numerator and
denominator share the gene-split term and are strongly dependent.

**Verification gate.** Run one wired module for one seed and confirm a CI key is present in the
emitted JSON. Wiring that silently no-ops is the failure mode this catches.

Without Task 2, replaying the runbook produces run_biorxiv with the same 22 CI-less result files
run6 has today.

---

## Task 2b — homology-partition robustness panel (cut from run_biorxiv)

The mechanism null is measured under the Pfam family partition only. C6 is removed from the
confirmatory set, and the coarser-partition check is named as follow-up work. The paper must not
imply partition-independence.

**The committed panel is defective and is withdrawn, not amended.** Reviving it means rerunning
from scratch, not starting from `results/run_biorxiv/homology_partition_panel/panel.json`. Three
defects produced its apparent finding that the null strengthens under stricter partitions:

- **The clan arm ran on a different dataset, not a stricter split.** Only genes whose Pfam family
  belongs to a clan can be clan-split, dropping roughly two thirds of genes — a smaller,
  differently-composed subset biased toward well-studied superfamilies.
- **Every arm was scored against the same chance floor.** All three rows carry
  `measured_floor: 0.2884`, the family-split floor on the full dataset. Held-out folds under
  different partitions have different class balance, so the floor moves; the clan arm was graded
  on the wrong curve, and the leakage fraction inherited the error through its denominator.
- **The MMseqs2 clustering was never validated.** At 20% identity it produced 1,215 clusters from
  ~1,935 genes, finer than Pfam family's 1,134, when that threshold should merge aggressively.
  Check the coverage threshold (`-c` / `--cov-mode`) first if this arm is revived.

On a matched subset with each arm against its own floor, family and clan scores are close and no
decline is visible. `for_me/homology_partition_findings.md` records the retracted version. No
smoke-scale CI may be quoted: at `n_boot=20` a nominal 95% interval sits at essentially the min and
max of the resample distribution.

**Two bugs found here were real and their fixes stand.** `mmseqs_cluster_holdout.py` fed
int-coded labels into probes that compare against the string `MECHANISM_CLASSES` internally,
silently zeroing every class bucket and crashing on the first balanced fold. `clan_holdout.py`
compared against hardcoded stale reference floors (0.352, 0.387) instead of live measured numbers.

**Related outstanding item, independent of this task.** `delta_mean` family-split macro-F1 is on
record with three different values — 0.380 (`ESM2_REPORT.md` §2 nonlinear table), 0.415 (§7 Table
10), and 0.418 (this panel). run_biorxiv takes the floor from a single source, so the run reports
one number; the run6 discrepancy is noted rather than carried forward.

---

## Task 2c — ★ enforce production quality as a gate, not prose

The `RUNBOOK_biorxiv.md` verification checklist is qualitative and manual, and nothing in the repo
enforces "production quality." Two changes turn that prose into a gate that fails loudly.

### 2c.1 Pin the environment

Pin versions for the result path. `uv.lock` exists; the pinned set must match what produced the
run_biorxiv numbers, and be recorded in each report's Provenance. This prevents the
sklearn-version hazard already documented in `CLAUDE.md` (`multi_class=` removed in ≥ 1.8).

`pytest tests/`, run locally on the commit that produces the run, is the precondition for flipping
`RUN_NAME`. The tests need no GPU, network, or embedding arrays, so a green local run is the whole
gate.

### 2c.2 Automate the run6→run_biorxiv numeric regression

Task 6's run6→run_biorxiv delta note is a script, not prose discipline.

`scripts/compare_runs.py <old_run> <new_run>` reads both runs' result JSONs, diffs every headline
number, and emits an old/new/delta table with a material-movement flag (proposed threshold: any
headline metric moving more than one run6 seed-std, or any gate verdict changing).

- It is the regression test for the run. run_biorxiv changes error bars, not point estimates, so
  any point estimate that moves materially is either a wiring bug or a finding that needs
  explaining.
- Its output is the delta-note deliverable, generated rather than transcribed, so it cannot drift
  from the result files.

Wire it into CI as a smoke test against the committed run6 files: diffing run6 against itself must
produce zero movement, a cheap invariant that catches parser drift.

---

## Task 3 — configuration

- `RUN_NAME = "run6"` → `"run_biorxiv"` in [`utils/paths.py:11`](../src/esm2_mech/utils/paths.py#L11).
  Single line; `RESULTS_DIR`, `RUN_REPORTS_DIR`, and `FIGURES_DIR` all derive from it.
- ★ Flip `RUN_NAME` only after Tasks 1 and 2 pass their verification gates. Replaying against
  unwired modules would land run_biorxiv with the same CI-less files run6 has, and fixing them
  afterward means either overwriting run_biorxiv (destroying provenance) or a run8. Tasks 1 and 2
  are CPU-testable locally against `run6` paths before the flip.
- `PERMUTATION_FEATURES` stays at `("delta_mean", "wt_only_mean")`; widening it to other
  above-floor features was cut on 2026-08-11. `delta_mean` is C1's instrument and `wt_only_mean`
  the above-floor comparison; the rest are exploratory, and each added feature multiplies an
  8,000-refit GPU step. The same constant gates which features cache OOF for the split gap, so
  widening it enlarges that cache too. The per-probe N is set by the permutation budget.
- **Resolve the working tree first.** ★ Largely done as of commit `427c3db` ("cleanup") — the
  modified `constants.py`, `paths.py`, five `family_split_baselines_seed*.json`, and
  `reports/run6/INTRO_REPORT.md`, plus the untracked `docs/` additions and
  `experiments/llm_judge/`, are all committed. Remaining at 2026-08-10 (`git status --short`):
  `RUNBOOK_5.md` deleted (superseded by the `RUNBOOK_biorxiv.md` rename); `RUN_PROGRESS.md` and
  `docs/EXPERIMENT.md` modified (the run7→run_biorxiv rename sweep); `PLAN_biorxiv.md` and
  `RUNBOOK_biorxiv.md` untracked; and three new untracked files under `reports/run6/` —
  `ESM2_REPORT.txt`, `report_control.pdf`, `report_within_family.pdf` (manual exports, confirm
  they belong in history before committing). Commit all of these before the run_biorxiv branch
  point so run6 and run_biorxiv provenance stay separable.
- ★ `SESSION.md` is matched by the user's global gitignore (`~/.gitignore_global:46`), so it is
  local-only and will not travel with the repo. Rename it if the handoff should be in history.

---

## Task 5 — remaining statistical work

Machinery exists for all of these; none is implemented.

- AUPRC with prevalence baseline, and PPV/NPV at class prevalence, for the rare classes
  (DN ≈ 9%, GOF ≈ 15%). AUROC alone overstates usefulness at those rates.
- ★ Multiplicity control is now set by the confirmatory/exploratory split: no correction is
  applied to the five confirmatory claims, and the 28-family within-family table is labelled
  exploratory rather than corrected.
- Minimal-detectable-effect statement per family, so within-family nulls read as underpowered
  rather than as evidence of absence.
- Multi-seed family probe in `report_protein_family.md` (currently seed 0 only) — ★ now a Task 2
  code change, tracked there.
- Calibration note in each probe report: the probes are uncalibrated and measure discrimination
  only, not risk.

---

## Task 6 — regenerate all reports

★ 16 documents: the 14 existing in `reports/run6/` (11 per-experiment plus 3 cross-cutting)
regenerated, plus 2 new for run_biorxiv.

| Group | Files |
|---|---|
| Per-experiment reports (8) | `report_classifier`, `report_control`, `report_protein_family`, `report_leakage_fraction`, `report_within_family`, `report_geometry`, `report_stability`, `report_single_source` |
| ★ Cross-cutting (3) | `ESM2_REPORT.md` (assembled paper), `INTRO_REPORT.md` (lay summary), `STATS_PLAN.md` |
| New for run_biorxiv (2) | a paired-difference summary; a run6→run_biorxiv delta note (generated by `compare_runs.py`, Task 2c.2) |

`ESM2_REPORT.md` and `INTRO_REPORT.md` quote numbers from every section, so both are regenerated.
`STATS_PLAN.md` moves from a plan to a record of what was done. ESM-3 and contrastive reports are
not regenerated, since neither experiment runs; they stay in the run6 archive.

★ **Paired-difference summary (new).** The tested differences are currently scattered across
reports; this collects each one with its CI into one table.

★ **Run6→run_biorxiv delta note (new).** Every headline number, run6 value, run_biorxiv value, and
whether the CI changes the reading, generated by `scripts/compare_runs.py` (Task 2c.2), not
transcribed. The writing task is reviewing its output and explaining each flagged movement.

Per the project report rules: a result file and its report must share the same `RUN_NAME`, and
every number must trace to a file under `results/run_biorxiv/` cited in Provenance. Provenance
names the variant snapshot and embedding arrays behind each number, and states explicitly that the
megascale numbers are the run's only quantities computed on retained run6 arrays.

---

## Sequencing

1. **Task 0** — settle correctness and methodology. The pathogenicity variant set is a docs
   deliverable; the CI decision rules and the confirmatory/exploratory split are written into
   `PREREGISTRATION_run_biorxiv.md` before the run. The resampling unit, rare-class intervals, and
   permutation budget become code in Tasks 1–2.
2. **Task 3 (partial)** — working-tree cleanup only. Cheap; establishes the run_biorxiv branch
   point.
3. ✅ **Task 1** — paired cluster bootstrap, both pairing modes. Unblocks the difference claims.
4. **Task 2** — wire the five modules; verify a CI key appears in real output.
   **Task 2b** — cut from run_biorxiv; deferred to follow-up work.
   **Task 2c** — pin deps, build `compare_runs.py`.
5. **Task 3 (remainder)** — flip `RUN_NAME`. Only after the Task 1 and 2 verification gates pass
   and `pytest tests/` is green (Task 2c.1).
6. **run_biorxiv** — replay `RUNBOOK_biorxiv.md`, skipping every embedding step.
7. **Task 5**, then **Task 6** — remaining statistics, then reports.

Everything after that is in [`FOLLOWUP_biorxiv.md`](FOLLOWUP_biorxiv.md) and gates nothing here.

Tasks 1 and 2 are CPU-testable locally and can proceed while the pod is being provisioned.

---

## Open questions

1. **Which features get permutation p-values?** ★ Resolved 2026-08-11: the two already in
   `PERMUTATION_FEATURES` — `delta_mean` (C1's instrument) and `wt_only_mean` (the above-floor
   comparison). Adding `wt_concat_mut` and `mut_only_mean` was cut; the rest are exploratory.

2. **Is 1,000 permutations firm?** ★ Resolved by the permutation budget. The linear-probe test at
   two features runs 2 × 2 splits × 1,000 = 4,000 refits at seed 0, covering the load-bearing
   claim. The MLP is not permutation-tested. Per-refit cost has not been measured; a single timed
   refit on the pod is needed before scoping this as hours versus days and deciding whether it
   needs joblib parallelism. This is the run's main schedule risk; everything else is cheap.

3. **Do permutations need all 5 seeds?** ★ Resolved: seed 0 only. A permutation test constructs
   its own null by shuffling; running it across 5 seeds would mostly re-measure fold jitter, which
   `STATS_PLAN.md` argues should be replaced. Cuts the step 5× at no inferential cost.

4. **Paired-bootstrap resampling scheme.** ★ Resolved: one shared resample applied to both arms,
   restricted to the shared cluster subset. Folds are identical only in the same-fold mode; the
   cross-partition mode (the split gap) pairs across two different fold assignments. Both modes
   are asserted directly in the Task 1 unit tests.

6. **Which unit for the split-gap CI?** ★ Resolved by the resampling-unit rule: resample families,
   the coarser unit, with the gene-resampled interval reported alongside as a labelled sensitivity
   check.

7. **Does `INTRO_REPORT.md` stay in `reports/run_biorxiv/`?** ★ Decided: yes. The earlier tone
   issue is resolved — hype phrasing that violated the report-style rule in `CLAUDE.md` has been
   neutralised — and the file stays in `reports/run_biorxiv/` rather than moving to `docs/`.

8. **Does stability gate H2 get a paired test?** Recommendation: yes if cheap, otherwise an
   explicit note that the gate is descriptive rather than tested. See Task 1.
