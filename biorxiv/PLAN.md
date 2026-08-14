# run_biorxiv plan — re-run the pipeline with inferential statistics

**Drafted:** 2026-07-20
**Revised:** 2026-07-22

**Goal.** Re-run the whole pipeline as `run_biorxiv` so every number in every report comes with a
real error bar — one that accounts for genes in the same family not being independent — plus a
p-value where a claim needs one, and an actual test whenever the paper says one thing beats
another. The run6 error bars only show how much a number wobbles between the 5 seeds, which says
nothing about whether the result would hold on a different set of genes.

Five experiments. ESM-3 (scale and structure) and contrastive metric learning are cut from this
run: the preprint's argument is the mechanism null, its chance floor, the homology-leakage account,
and the two positive controls. The run6 results for both cut experiments stand in the run6 archive.

**What the 2026-07-22 review added.** The original scope was just "put error bars on the existing
numbers." Reading the run6 reports against `STATS_PLAN.md` turned up problems that needed settling
first (all marked ★):

- **Pathogenicity provenance** — pin down exactly which variant set the headline control ran
  on, and record its fingerprint.
- **Decide the pass/fail rules before the run** — so a gate that clears by 0.008 with an error
  bar through zero has a stated meaning, instead of one picked afterwards.
- **Name the claims the paper rests on** — five of them; everything else is labelled a
  look-around, so nobody can say we tested many things and reported the wins.
- **Resample the same unit the split uses** — family-split numbers resample families, not genes.
- **Rare-class error bars** — DN and GOF have too few genes for the plain method.
- **How many permutations** — and what that costs.
- **Task 1 Fix the split-gap error bar** — the two arms are paired, so the test must be too.
- **Task 2c Production quality** — enforced, not assumed.

**What reading Badonyi & Marsh 2025 added** (`papers/mechanism_2025.pdf`). They measure how often
one gene carries two different mechanisms, which makes "your gene labels are wrong a lot of the
time" a specific, citable objection to C1 rather than a vague caveat. Three new tasks answer it:

The confirmatory / exploratory split records the objection. The three tasks that answer it — 2d (re-run C1 on genes with
unambiguous labels), 8 (measure how much wrong labels hurt a probe that works) and 9 (show other
features find mechanism in the same labels) — are all after the run, in
[`FOLLOWUP_biorxiv.md`](FOLLOWUP_biorxiv.md).

**Dropped from scope.** Task 2b (homology-partition robustness panel), cut outright. Task 2d, moved
to the follow-up doc: a clean-label filter leaves DN near ~50 genes, below what a family-resampled
CI can constrain, and Task 8 answers the same objection without depending on subset size. Task 4
(runbook) is done — `RUNBOOK_biorxiv.md` supersedes it.

Scope reference: [`reports/run6/STATS_PLAN.md`](../reports/run6/STATS_PLAN.md) is the statistical
methodology; [`RUNBOOK_biorxiv.md`](RUNBOOK_biorxiv.md) is the execution spec (supersedes `RUNBOOK_4.md`);
[`RUN_PROGRESS_biorxiv.md`](RUN_PROGRESS_biorxiv.md) is the live status table. This plan covers what must
change before the runbook is replayed.

**Outputs:** results to `results/run_biorxiv/`, reports to `reports/run_biorxiv/`. `results/run6/` and
`reports/run6/` are preserved untouched as the comparison baseline — that is the point of making
run_biorxiv a separate run rather than an in-place fix.

**Every ClinVar-derived input is rebuilt.** The ClinVar fetch was re-run from scratch on
2026-08-11, so `valid_variants.json` comes from a current snapshot, and the pathogenicity control
set is refetched on that same snapshot rather than reused — its params sidecar was lost, and a
refetch gives a set whose provenance is written by the run that produced it instead of one
carrying a partly reconstructed record. Every array row-aligned to those lists is extracted again:
the ESM-2 variant embeddings, the pathogenicity embeddings, and the
conservation scores. The megascale arrays are reused, being Tsuboyama-derived with no ClinVar
dependency. Embedding paths are keyed by *model*, not run (`utils/paths.py:68-69`), so no copy is
needed — and at ~10 GB, copying per run would cost the space for no provenance gain. run_biorxiv
result files record the embedding fingerprint, so which arrays each result was scored on is visible
in the output.

This changes what the run6→run_biorxiv comparison can show. Experiment 7 is the only one whose inputs
are unchanged, so it is the only place a movement is attributable to the new statistics alone.
Everywhere else two things moved at once, and the delta note attributes movement to both rather
than reading it as a statistical correction.

GPU is needed for: Experiment 1 step 2 (ESM-2 embed), Experiment 2 phase 2 (pathogenicity embed),
Experiment 5 step 3 (conservation extract),
Experiment 7 step 4 (megascale MLP), and the Experiment 1 permutation tests, which refit the
probe per repeat.

---

## Current state

The shared machinery in `utils/bootstrap.py` is built and complete: `cluster_bootstrap_ci`,
`bootstrap_mechanism_metrics`, `label_permutation_pvalue`.

★ **Correction to the original plan:** it said six experiment modules import this machinery. Three
do — `naive_baseline.py`, `mechanism_delta_family_split.py`, `mechanism_within_family.py` — so the
wiring gap is larger than assumed.

What has actually landed in `results/run6/`:

| | Status |
|---|---|
| Files with cluster-bootstrap CIs | 5 of 27 (`family_split_baselines_seed{0..4}.json` only) |
| Files with permutation p-values | the same 5, and only 2 features within them |
| Reports citing a CI | 0 of the 14 existing documents |
| Paired difference tests | not implemented |

The two existing p-values (family-split, seed 0, 200 permutations):

- `wt_only_mean` — observed 0.409, null mean 0.325, p = 0.0099
- `delta_mean` — observed 0.289, null mean 0.319, p = 1.0

`wt_only_mean`'s p = 0.0099 is exactly 1/(200+1), the resolution floor of a 200-permutation
test — the true value is unresolved below 0.01. This is the concrete reason to run at the
plan's specified 1,000. `delta_mean` scoring below its own null is consistent with the at-floor
result reported everywhere else.

The restriction to two features is `PERMUTATION_FEATURES = ("delta_mean", "wt_only_mean")` at
[`mechanism_delta_family_split.py:108`](../src/esm2_mech/experiments/mechanism/mechanism_delta_family_split.py#L108)
— a module-level constant, not a flag. `PERMUTATION_N_RESAMPLES` already defaults to 1000 in
`constants.py`; the 200 in the current files came from a run-time override.

---

## Task 0 — ★ correctness and methodology (settle before any code)

Everything below is decided before implementation, because each one changes what the machinery
must emit — discovering them mid-run means re-running, not just re-reporting.

### Freeze and document the pathogenicity variant set

The paper's headline positive control was recorded twice on two different variant sets. Seed 0 ran on
RunPod with the full set and scored AUROC **0.878**; seeds 1–4 ran locally on a truncated version and
scored **0.742 ± 0.006** (`result_6.md`). `docs/README.md` still calls a "clean 5-seed mean on a
consistent variant set" pending.

The resulting 0.74–0.88 range looks like seed-to-seed variation but is two different experiments
stacked together — the `± 0.006` shows the probe is stable within either set, so the whole width
comes from the sets differing. No bootstrap repairs that: it says how much a number would move under
a different sample of genes, not how much it moves under a different dataset.

★ **Verified 2026-07-22: run6 already fixed this in code.** Experiment 2 was rebuilt as a single
consolidated fetch → embed → probe module over one canonical set of 37,218 balanced ClinVar
variants (1,929 genes, GRCh38). `pathogenicity_control.py` fingerprints the variant set
(`variants_fingerprint`, line 306), stores the fingerprint in the embedding metadata (line 332),
and **hard-refuses to proceed** if the embeddings do not match the current variant set
(line 360). All five run6 seeds agree: `delta_mean` MLP family-split = 0.894, gene-split = 0.897,
std ≤ 0.001. The two-variant-set ambiguity does not exist in run6 and is not inherited by
run_biorxiv.

What remains is **documentation, not re-derivation** — the stale text is what a reader meets first:

- Correct `docs/README.md`'s "pending due to provenance issue" note; it now describes a resolved
  problem. State the canonical set (37,218 variants / 1,929 genes / GRCh38) and the run6 numbers.
- Mark `result_6.md`'s 0.74–0.88 band as superseded by run6, so the old band cannot be cited by
  accident.
- Record the variant-set fingerprint in the run_biorxiv result files and quote it in the report's
  Provenance, so the canonical set is identified by hash rather than by description.
- **Verification, not assumption:** confirm at run_biorxiv time that all five seeds share one fingerprint
  and that the spread is ≤ 0.01. If any seed disagrees, stop — that is a real data defect and the
  freeze-and-rerun this task originally called for becomes necessary after all.

### ★ Pre-registered CI decision rules for the confirmatory gates

A load-bearing conclusion rests on a gate that "passes" by a thin margin — K2 by 0.002. A **paired CI** puts one error bar on the gap between the two arms rather than a separate
error bar on each: resample the genes, recompute the difference on that same resample for both arms,
repeat. Attaching one creates a case the current gates do not define: **a gate can pass on its point
estimate while its difference-CI spans zero** — the gap cleared the bar, but the interval includes
zero, so it could equally be nothing. Without a rule fixed in advance,
run_biorxiv leaves "cleared by 0.008" next to an interval through zero and no stated reading, and
any reading chosen afterwards is unfalsifiable.

**Rule, pre-registered before the run:**

> A gate is **affirmed** only if its point estimate clears the threshold *and* the paired
> difference 95% CI excludes zero. If the point estimate clears but the CI spans zero, the claim
> is restated as **not distinguishable** — not as a pass, and not as a refutation. If the point
> estimate fails, the gate fails regardless of the CI.

This applies to the **confirmatory** gates, not only the speculative follow-up tests.

| Gate | Claim (criterion as recorded) | Margin vs 0.430 | Lift vs bare floor 0.380 |
|---|---|---|---|
| K1 | conservation alone clears 0.85 | +0.041 | — |
| K2 | embedding adds over conservation | +0.002 (fails) | — |
| H2 | stability random→family drop under LEAKY 0.10 | descriptive | — |

K2 already **fails**, so a CI spanning zero reinforces that reading.

★ **For the failing gates, add the underpowered-null distinction.** K2 (+0.002 against a 0.02 bar)
is a null from an underpowered comparison. If its CI is wide, the honest reading is that the data
cannot resolve an effect of the pre-registered size, which is not the same as showing there is none.


Everything else — per-class AUROCs, the 28-family table, within-family per-family cells, the
biochemistry R², magnitude/direction decomposition, per-feature leakage fractions — is
**exploratory** and labelled as such in its report.

K2 needed the embedding to add something over conservation, got +0.002, and fails. Under the CI
decision rules' failing-gate clause it is reported as "no effect of the pre-registered size was
detected, and the test is underpowered to rule one out" rather than as a demonstrated absence.

★ **No multiplicity correction is applied.** R7.2 of `PREREGISTRATION_run_biorxiv.md` records why.

★ **C1 is adjudicated as a null claim.** An interval straddling the chance floor is not evidence of
sitting at it — a wide enough interval straddles anything. C1 is affirmed only if its CI upper bound
falls below the measured floor plus 0.05, and is otherwise recorded as *not adjudicated*. The permutation test can refute C1 but
cannot confirm it.

C1 and C3 are the load-bearing pair (the dissociation) and C2 is the leakage account. If the
confirmatory set needs trimming, trim from C4, not C1–C3.

★ **Amendment 2026-07-22 — the label-heterogeneity threat is now named and cited.** Badonyi & Marsh
2025 (`papers/mechanism_2025.pdf`, bioRxiv 2025.03.13.642984) report that **43% of multi-phenotype
dominant genes and 49% of mixed-inheritance genes carry both LOF and non-LOF mechanisms**. Because
this project assigns one mechanism label per gene, some fraction of the 17,826 variants is
mislabelled by construction. That gives a reviewer a citable alternative explanation for C1: the
delta sits at the floor because the labels are noisy, not because the embedding lacks mechanism
signal.

No confidence interval addresses this — the threat is to what the labels mean, not to how many
samples there are. It is answered by Tasks 2d and 8 in
[`FOLLOWUP_biorxiv.md`](FOLLOWUP_biorxiv.md) — does the null survive on cleanly-labelled genes, and
how far does realistic label noise move a working probe. C1's statement in the
reports must cite this paper and point at those two results rather than asserting the labels are
adequate.

Deliverable: written into [`PREREGISTRATION_run_biorxiv.md`](PREREGISTRATION_run_biorxiv.md)
alongside the CI decision rules, before the run.

### Resample the unit that defines the split

Two different things are in play, and the rule is that they must match.

**The split** is how the cross-validation folds are drawn — what is held out together. Under
gene-split, all variants of one gene go into the same fold, so the model never sees gene X in both
training and testing. Under family-split, all genes in one Pfam family go into the same fold, which
is stricter: it blocks the model from scoring gene X by having memorised its close relative Y.

**The unit** is what gets drawn when the error bar is built. The bootstrap re-picks items with
replacement, re-scores, and repeats a thousand times; the spread of those scores is the interval. The
unit is what counts as one item — a gene, or a whole family.

`STATS_PLAN.md` says CIs "resample whole genes (or whole families, where the unit is the family)" but
then defaults to gene-resampling for the classifier report on *both* splits. A bootstrap treats the
items it draws as independent, and family-split has already declared that genes in the same family
are not — that is why they were forced into the same fold. Resampling genes therefore claims 1,935
independent items when the evaluation really has 1,134 families, 833 of them single genes. Too many
assumed-independent items shrinks the spread, so the interval comes out narrower than the data
supports.

**Rule: the resampling unit matches the split.** Gene-split → resample genes. Family-split →
resample families. Applies to every CI on a family-split metric, in every report.

**Expected consequence, stated in advance so it is not mistaken for a bug:** family-split CIs
will be visibly wider than gene-split ones. There are 1,134 families but 833 are singletons, so
the effective cluster count is far below the gene count. That widening is the correct answer, not
something to tune away. Report the effective cluster count alongside each family-split interval.

**The split-gap comparison resamples families.** Its two arms are gene-split and family-split, and
the family-split arm's variance is only correct under family resampling, so a gene-resampled gap
understates it. Resample the **coarser** of the two units: a family resample induces a valid gene
resample, but not the reverse. Report the gene-resampled interval alongside as a sensitivity check,
labelled as the narrower and anticonservative of the two.

### Rare-class intervals carry a health warning, not a correction

DN is ≈ 9% of variants — roughly 150–170 genes, and fewer effective clusters under family-split.
Percentile bootstrap undercovers for a bounded metric (AUROC ∈ [0,1]) near its boundary with few
clusters, which is precisely this regime.

- Rare-class one-vs-rest AUROC uses the same percentile cluster bootstrap as every other interval
  in the project. A bias correction is not applied: over ~150 clusters the correction is itself
  noisy, so it trades one inaccuracy for another, and no confirmatory claim reads these intervals —
  per-class AUROCs are exploratory under the confirmatory / exploratory split.
- **Flag every rare-class interval as the least trustworthy in its table.** DN intervals are
  indicative, not authoritative, and saying so is worth more than a false-precision interval.
- Keep the existing suppression guard for degenerate folds.

### Permutation budget — linear probe only

The permutation test refits the probe per repeat, and the MLP is the expensive tail.

- **Linear probe: 1,000 permutations.** The headline claim — `delta_mean` sits at the chance
  floor — is a linear-probe claim, so the load-bearing test is fully resolved. Cheap.
- **MLP: not permutation-tested.** ★ **Revised 2026-08-11.** No claim rests on an MLP
  permutation p-value, and its refits are the run's main schedule risk. Dropped rather than run
  at a reduced N.

No p-value is reported at its resolution floor (see the `wt_only_mean` p = 1/(200+1) case above).
Still requires timing a single refit on the pod before the linear permutation is launched.

---

## Task 1 — paired cluster bootstrap (new code) ✅ done 2026-08-10

The one piece of new statistics code in the run: an error bar on the **difference** between two
results rather than on each result separately (the paired CI required by the CI decision rules). The
thinnest — K2 at +0.002 — is currently two point estimates with separated error bars at a margin
smaller than a seed of spread.

Two versions, because the arms are not always scored the same way.
`paired_cluster_bootstrap_diff` handles arms sharing one fold assignment: conservation vs
embedding. `paired_cluster_bootstrap_diff_cross_partition` handles arms
from different fold layouts, which is only the gene-split-minus-family-split gap — resample families,
then re-score each arm under its own partition. Its optional `sensitivity_clusters` arg returns the
gene-resampled version that the resampling-unit rule requires alongside.

Both are in `utils/bootstrap.py`, and both hand `metric_fn_a`/`metric_fn_b` the identical drawn
row-index array per replicate, so the difference is paired rather than two independently resampled
arms. Resampling the arms separately would still return a plausible-looking interval, just a wrongly
wide one, which is why the tests assert the shared array directly rather than only checking that an
interval came back. `tests/utils/test_bootstrap.py` covers both modes, the shared-resample property,
planted-difference and no-difference recovery, undefined-replicate dropping, and `ci_suppressed`.

Wiring them into the experiment modules is Task 2.

---

## Task 2 — wire existing machinery into 5 modules

Mechanical: `classify_by_mechanism` is the working reference implementation to copy. Each module
needs gene/family clusters passed to `bootstrap_mechanism_metrics`, the `--no_ci` / `--n_boot`
flags added, and CI keys emitted into its result JSON.

| Module | Experiment |
|---|---|
| `mechanism/mlp.py` | 1 (Step 3, nonlinear probes) |
| `pathogenicity/pathogenicity_control.py` | 2 |
| `geometry/run_geometry.py` | 5 |
| `stability/megascale_stability.py` | 7 |
| ★ `mechanism/family_clustering.py` | 1 (Step 3, diagnostic) |

★ `family_clustering.py` was missing from the original plan's list. `STATS_PLAN.md` requires the
family-probe accuracy to be multi-seed (run6 is seed 0 only) with cluster-bootstrap CIs over
families for both the probe accuracy and the k-NN purity metrics. That is a code change — the
module has no `--seeds` flag — not a reporting change, so it belongs here rather than in Task 5.

★ **Also needs a CI: `leakage_fraction.py`.** The original plan noted (Task 6) that
`report_leakage_fraction.md` has no interval on the ~40% figure, but no task added one. That
number is a headline in both `INTRO_REPORT.md` and `ESM2_REPORT.md` §4. It is a derived ratio,
(gene − family) / (gene − chance), of quantities that will each carry CIs, so it needs an
explicit decision: propagate through the bootstrap (resample genes once, recompute the whole
ratio per replicate — correct, since numerator and denominator share the gene-split term and are
strongly dependent) rather than combining two separate intervals. **Recommendation:** recompute
per replicate.

**Verification gate for this task:** run one wired module for one seed and confirm a CI key is
actually present in the emitted JSON. Wiring that silently no-ops is the failure mode this
catches, and without it Task 2 "completing" is unverified.

Without Task 2, replaying the runbook produces run_biorxiv with the same 22 CI-less result files it has
today.

---

## Task 2b — homology-partition robustness panel (cut from run_biorxiv)

The mechanism null is measured under the Pfam family partition only. C6 is removed from the
confirmatory set and the coarser-partition check is named in the paper as follow-up work, so the
paper claims partition-independence nowhere and must not imply it.

**The committed panel is defective and is withdrawn, not amended.** Anyone reviving it starts from
a rerun, not from `results/run_biorxiv/homology_partition_panel/panel.json`. Three defects produced
its apparent finding that the null strengthens under stricter partitions:

- **The clan arm ran on a different dataset, not a stricter split.** Only genes whose Pfam family
  belongs to a clan can be clan-split, which drops roughly two thirds of the genes — a smaller,
  differently-composed subset biased toward well-studied superfamilies.
- **Every arm was scored against the same chance floor.** All three rows carry
  `measured_floor: 0.2884`, the family-split floor on the full dataset. Held-out folds under
  different partitions have different class balance, so the floor moves; the clan arm was graded on
  the wrong curve, and the leakage fraction inherited the error through its denominator.
- **The MMseqs2 clustering was never validated.** At 20% identity it produced 1,215 clusters from
  ~1,935 genes — finer than Pfam family's 1,134, when that threshold should merge aggressively.
  Check the coverage threshold (`-c` / `--cov-mode`) first if this arm is revived.

On a matched subset with each arm against its own floor, family and clan scores are close and no
decline is visible. `for_me/homology_partition_findings.md` records the retracted version.
No smoke-scale CI may be quoted: at `n_boot=20` a nominal 95% interval sits at essentially the min
and max of the resample distribution.

**Two bugs found here were real and their fixes stand:** `mmseqs_cluster_holdout.py` fed int-coded
labels into probes that compare against the string `MECHANISM_CLASSES` internally, silently zeroing
every class bucket and crashing on the first balanced fold; and `clan_holdout.py` compared against
hardcoded stale reference floors (`0.352`, `0.387`) instead of live measured numbers.

**Related outstanding item, independent of this task.** `delta_mean` family-split macro-F1 is on
record with three different values — 0.380 (`ESM2_REPORT.md` §2 nonlinear table), 0.415 (§7 Table
10), and 0.418 (this panel). run_biorxiv takes the floor from a single source, so the run reports one
number; the discrepancy in the run6 text is noted rather than carried forward.

---

## Task 2c — ★ enforce production quality as a gate, not prose

The verification checklist in `RUNBOOK_biorxiv.md` is qualitative and manual, and "production quality"
appears nowhere as something the repo enforces. Two pieces turn that prose into a gate that fails
loudly.

### 2c.1 Pin the environment

Pin versions for the result path. `uv.lock` exists; the pinned set must be what the run_biorxiv
numbers were produced under, and recorded in the reports' Provenance. This is what prevents the
sklearn-version hazard already documented in `CLAUDE.md` (`multi_class=` removed in ≥ 1.8).

`pytest tests/`, run locally on the commit that produces the run, is the precondition for flipping
`RUN_NAME`. The tests need no GPU, network, or embedding arrays, so a green local run is the whole
gate; there is no CI job.

### 2c.2 Automate the run6→run_biorxiv numeric regression

Task 6 describes a run6→run_biorxiv delta note as a manual writing exercise. It should be a
**script**, not prose discipline:

`scripts/compare_runs.py <old_run> <new_run>` reads both runs' result JSONs, diffs every headline
number, and emits a table of old / new / delta with a **material-movement flag** (proposed
threshold: any headline metric moving more than one run6 seed-std, or any gate verdict changing).

- It is the regression test for the run. run_biorxiv changes error bars, not point estimates, so any
  point estimate that moves materially is either a bug introduced by the wiring or a finding that
  needs explaining. Nothing currently catches the former.
- Its output *is* the delta-note deliverable, generated rather than transcribed, so it cannot drift
  from the result files.

Wire it into CI as a smoke test against the committed run6 files (diffing run6 against itself must
produce zero movement — a cheap invariant that catches parser drift).

---

## Task 3 — configuration

- `RUN_NAME = "run6"` → `"run_biorxiv"` in [`utils/paths.py:11`](../src/esm2_mech/utils/paths.py#L11).
  Single line; `RESULTS_DIR`, `RUN_REPORTS_DIR`, and `FIGURES_DIR` all derive from it.
- ★ **Ordering constraint:** flip `RUN_NAME` only *after* Tasks 1 and 2 pass their verification
  gates. If the replay runs against unwired modules, run_biorxiv lands with the same CI-less files run6
  has, and fixing them afterwards means either overwriting run_biorxiv files (destroying the provenance
  of what was actually run) or a run8. Tasks 1 and 2 are CPU-testable locally against `run6`
  paths before the flip.
- `PERMUTATION_FEATURES` stays at `("delta_mean", "wt_only_mean")`; widening it to the other
  above-floor features was cut on 2026-08-11. `delta_mean` is C1's instrument and `wt_only_mean`
  the above-floor comparison; the rest are exploratory, and each added feature multiplies an
  8,000-refit GPU step. The same constant also gates which features cache OOF for the split gap,
  so widening it enlarges that cache as a side effect. The per-probe N is set by the permutation budget.
- **Resolve the working tree first.** ★ **Largely done as of commit `427c3db` ("cleanup")** — the
  modified `constants.py`, `paths.py`, five `family_split_baselines_seed*.json`, and
  `reports/run6/INTRO_REPORT.md`, plus the untracked `docs/` additions and `experiments/llm_judge/`,
  are all committed. Remaining at 2026-08-10 (`git status --short`): `RUNBOOK_5.md` deleted (superseded
  by the `RUNBOOK_biorxiv.md` rename); `RUN_PROGRESS.md` and `docs/EXPERIMENT.md` modified (the
  run7→run_biorxiv rename sweep); `PLAN_biorxiv.md` and `RUNBOOK_biorxiv.md` untracked; and three
  new untracked files under `reports/run6/` — `ESM2_REPORT.txt`, `report_control.pdf`,
  `report_within_family.pdf` (manual exports, not produced by any script in the pipeline — confirm
  they're wanted in history, not scratch output, before committing). Commit all of these before the
  run_biorxiv branch point so run6 and run_biorxiv provenance stay separable.
- ★ **Note:** `SESSION.md` is matched by the user's global gitignore
  (`~/.gitignore_global:46`), so it is local-only and will not travel with the repo. Rename it if
  the handoff should be in history.

---

## Task 5 — remaining statistical work

Machinery exists for all of these; none is implemented.

- AUPRC with prevalence baseline, and PPV/NPV at class prevalence, for the rare classes
  (DN ≈ 9%, GOF ≈ 15%) — AUROC alone overstates usefulness at those rates.
- ★ Multiplicity control is now set by the confirmatory / exploratory split, which supersedes this bullet's original scope: no
  correction is applied to the five confirmatory claims, and the 28-family within-family table is
  labelled **exploratory** rather than corrected.
- Minimal-detectable-effect statement per family, so the within-family nulls read as
  underpowered rather than as evidence of absence.
- Multi-seed family probe in `report_protein_family.md` (currently seed 0 only) — ★ now a Task 2
  code change, tracked there.
- Calibration note in each probe report: the probes are uncalibrated and measure discrimination
  only, not risk.

---

## Task 6 — regenerate all reports

★ **16 documents** — the 14 that exist in `reports/run6/` (11 per-experiment reports plus three
cross-cutting) regenerated, plus 2 new for run_biorxiv. The original plan said 13 and counted only the
first group:

| Group | Files |
|---|---|
| Per-experiment reports (8) | `report_classifier`, `report_control`, `report_protein_family`, `report_leakage_fraction`, `report_within_family`, `report_geometry`, `report_stability`, `report_single_source` |
| ★ Cross-cutting (3) | `ESM2_REPORT.md` (the assembled paper), `INTRO_REPORT.md` (lay summary), `STATS_PLAN.md` |
| New for run_biorxiv (2) | a paired-difference summary; a run6→run_biorxiv delta note (generated by `compare_runs.py`, Task 2c.2 — reviewed and annotated, not hand-written) |

`ESM2_REPORT.md` and `INTRO_REPORT.md` quote numbers from every section, so both must be
regenerated — neither was in the original Task 6. `STATS_PLAN.md` moves from a plan to a record
of what was done.

The ESM-3 and contrastive reports are not regenerated, since neither experiment runs. They stay in
the run6 archive.

★ **New for run_biorxiv: a paired-difference summary.** The tested differences are currently scattered
across reports; this collects each one with its CI into one table.

★ **New for run_biorxiv: a run6→run_biorxiv delta note.** Every headline number, run6 value, run_biorxiv value, and
whether the CI changes the reading. **Generated by `scripts/compare_runs.py` (Task 2c.2), not
transcribed** — the writing task is reviewing its output and explaining each flagged movement.

Per the project report rules: a result file and its report must share the same `RUN_NAME`, and
every number must trace to a file under `results/run_biorxiv/` cited in Provenance. Provenance names
the variant snapshot and the embedding arrays behind each number, and says so explicitly for the
megascale numbers, which are the run's only quantities computed on retained run6 arrays.

---

## Sequencing

1. **Task 0** — settle correctness and methodology. The pathogenicity variant set is a docs
   deliverable; the CI decision rules and the confirmatory / exploratory split are written into
   `PREREGISTRATION_run_biorxiv.md` before the run; **the resampling unit, rare-class intervals and
   permutation budget become code in Tasks 1–2** (split-unit resampling changes what `bootstrap.py`
   must compute).
2. **Task 3 (partial)** — working-tree cleanup only. Cheap, and establishes the run_biorxiv branch point.
3. ✅ **Task 1** — paired cluster bootstrap, both pairing modes. Unblocks the difference claims.
4. **Task 2** — wire the seven modules; verify a CI key appears in real output.
   **Task 2b** — cut from run_biorxiv; the homology-partition panel is deferred to follow-up work
   (see the Task 2b section).
   **Task 2c** — pin deps, build `compare_runs.py`.
5. **Task 3 (remainder)** — flip `RUN_NAME`. ★ Only now, after the Task 1 and 2 verification gates
   pass **and `pytest tests/` is green** (Task 2c.1).
6. **run_biorxiv** — replay `RUNBOOK_biorxiv.md`, skipping every embedding step.
7. **Task 5**, then **Task 6** — remaining statistics, then reports.

Everything after that is in [`FOLLOWUP_biorxiv.md`](FOLLOWUP_biorxiv.md) and gates nothing here.

Tasks 1 and 2 are CPU-testable locally and can proceed while the pod is being provisioned.

---

## Open questions

1. **Which features get permutation p-values?** ★ **Resolved 2026-08-11: the two already in
   `PERMUTATION_FEATURES`** — `delta_mean` (C1's instrument) and `wt_only_mean` (the above-floor
   comparison). The earlier recommendation to add `wt_concat_mut` and `mut_only_mean` was cut; the
   rest are exploratory. See Task 3.

2. **Is 1,000 permutations firm?** ★ **Resolved by the permutation budget.** The permutation test is linear-probe
   only, at two features (open question 1): 2 features × 2 splits × 1,000 = **4,000 refits at
   seed 0**, which covers the load-bearing claim. The MLP is not permutation-tested.
   **Per-refit cost has still not been measured** — a single timed refit on the pod is needed before
   this is scoped as hours versus days, and before deciding whether it needs joblib parallelism
   across the pod's cores. ★ This is the run's main schedule risk; everything else in run_biorxiv is
   cheap.

3. **Do permutations need all 5 seeds?** ★ **Resolved: seed 0 only.** A permutation test
   constructs its own null by shuffling; running it across 5 seeds mostly re-measures fold
   jitter, which is precisely what `STATS_PLAN.md` argues should be replaced. Cuts the step 5×
   at no inferential cost.

4. **Paired-bootstrap resampling scheme** — ★ **Resolved:** one shared resample applied to both
   arms, restricted to the shared cluster subset. Folds are identical only in the same-fold mode;
   the cross-partition mode (the split gap) pairs across two different fold assignments. Both
   modes asserted directly in the Task 1 unit tests.

6. ★ **Which unit for the split-gap CI?** ★ **Resolved by the resampling-unit rule: resample families**, the coarser
   unit, with the gene-resampled interval reported alongside as a labelled sensitivity check.

7. ★ **Does `INTRO_REPORT.md` stay in `reports/run_biorxiv/`?** ★ **Decided: yes** — the tone issue
   was already resolved (the hype phrasing that previously violated the report-style rule in
   `CLAUDE.md` has been neutralised, so the file now reads consistently with the per-experiment
   reports), and it stays in `reports/run_biorxiv/` rather than moving to `docs/`.

8. ★ **Does stability gate H2 get a paired test?** Recommendation: yes if cheap, otherwise an
   explicit note that the gate is descriptive rather than tested. See Task 1.
