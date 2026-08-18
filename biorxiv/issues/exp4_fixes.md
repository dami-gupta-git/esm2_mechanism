# Experiment 4 — implementation brief

Work order for the defects diagnosed in [`exp4_issues.md`](exp4_issues.md). That document holds
the evidence and reasoning; this one holds the changes. Read issue 1 there before starting, since
every step below depends on understanding why pooling breaks ranking metrics.

Scope is the `run_biorxiv` sequence only. Twelve experiments outside it are affected by the same
defect and are deliberately deferred — see [`TODO.md`](../../TODO.md). Do not fix them here.

**Do not adjust any number by hand.** Every value changes as a consequence of changing how it is
computed, or not at all.

---

## Before writing code

**Step 2 moves the number that the main claim is judged against, so the order is not negotiable.**
The 2A threshold is defined as the nonlinear delta probe's own family-split score. That probe's
score changes when the metrics become fold-aware. So the sequence must be: fix the metrics,
recompute the threshold, commit the amendment recording it, and only then look at whether the
claim passes. Computing the threshold first, or judging the claim against the old one, turns a
preregistered rule into a result chosen after the fact — which is worse for the paper than either
possible verdict. If the threshold moves enough to change the verdict, that is a finding to
report, not a problem to tune away.

The rare-class rule that was open in the first version of this brief has been decided. It is in
the table below and specified in step 2.

---

## Decisions already made

These were settled during review. Implement them; do not re-open them.

| Question | Decision |
|---|---|
| Fold argument on the shared helpers | Required, not optional. An optional argument that falls back to pooling is how this defect survives its own fix. |
| Basis for macro-F1 | Per-fold, averaged across folds — the same basis as the ranking metrics. |
| Reason to change macro-F1 | Consistency, not correctness. Pooled macro-F1 is not corrupted; argmax is per-row. Record this reason accurately in any comment or report text. |
| Rank-transform-then-pool as an alternative | Rejected. The headline tables are already per-fold. |
| Rare-class rule | Strict: every fold must score the class, and any bootstrap draw where one cannot is discarded. Decided 2026-08-18 on the evidence in step 2. |

---

## Step 1 — carry the fold index through the out-of-fold collector

**File:** `src/esm2_mech/utils/probes.py`

`_OofCollector` (around line 90) accumulates four aligned arrays — true labels, probabilities,
gene ids and row ids — and `finalize()` concatenates them. The fold each row came from is never
recorded. That omission is the root cause: once the fold is gone, no consumer can be fold-aware
even if it wants to be, so pooling is the only thing the data structure permits.

Add the fold index as a fifth aligned array. `add()` takes the fold number; `finalize()` returns
it under a `"folds"` key alongside the existing four.

Every producer of an OOF dict must populate it. Within this file that is `_run_multiclass_cv`
(around line 164), which already has `fold_i` in scope.

This step is the enabler. Nothing else in this brief is possible until it lands.

---

## Step 2 — make the ranking metrics fold-aware

**File:** `src/esm2_mech/utils/bootstrap.py`

Four functions rank values and must score within fold, then average across folds:

| Function | Line | Used by |
|---|---|---|
| `bootstrap_mechanism_metrics` | 452 | the multiclass AUROC, AUPRC and lift intervals |
| `binary_auroc_cluster_bootstrap_ci` | 700 | the pathogenicity control and geometry |
| `paired_oof_diff` | 556 | the split gap and the paired comparisons |
| `macro_ovr_auroc` | 810 | the permutation statistic |

Each takes the folds array and computes its metric per fold on the resampled rows, then averages.
The macro-F1 path inside `bootstrap_mechanism_metrics` moves to the same basis, per the decision
above.

### The rare-class rule

Dominant-negative is 9% of the labels, so a fold can in principle contain none of it and the class
cannot be scored there. The current code silently skips a class it cannot score. Under bootstrap
resampling that makes the set of contributing folds vary between draws, so each draw scores a
different statistic — the same failure diagnosed for the family probe in issue 10.

**The rule is strict: every fold must score the class, and any draw where one cannot is
discarded.** Apply it to the ranking metrics, the precision-recall areas and the prevalence
figures alike; the latter two are the more fragile at small positive counts.

**Why strict costs nothing here.** Every family-split fold already contains dominant-negative
variants, on every seed, on both the merged set and the Gerasimavicius-only subset. The thinnest
fold on the merged set carries 9 families and 133 variants of it; the thinnest on the subset
carries 5 families. A fold therefore only loses the class when a draw happens to exclude all of
its dominant-negative families at once, which for 9 families is on the order of one draw in ten
thousand. Across a thousand draws the expected number of discards is a fraction of one.

The alternative — accepting a fold once it holds some minimum count — buys nothing measurable and
puts an arbitrary threshold into the paper that has to be justified to a reviewer.

**Record the discard count** in the output next to the existing `valid_frac`.

**Treat a high discard rate as a fault, not a cost.** The evidence above says it should be far
below 1%. If it comes back materially higher, something else is wrong — most likely the resampling
unit or the fold construction — and it must be investigated rather than absorbed into the result.

### Verify before moving on

The delta's per-class family-split AUROCs should come back near 0.55–0.61, matching the per-fold
values already in the result files, rather than the 0.40–0.48 the pooled path produces.

---

## Step 3 — permute within fold

**File:** `src/esm2_mech/utils/bootstrap.py`

Both permutation paths — `oof_permutation_pvalue` (line 831) and `label_permutation_pvalue`
(line 898) — currently shuffle labels across the whole dataset, in cluster blocks.

A fold-aware statistic scored against a whole-dataset shuffle undoes step 2, because moving labels
between folds changes each fold's class composition. Constrain the shuffle so labels stay within
their own fold while still permuting in cluster blocks inside it.

`label_permutation_pvalue` refits per permutation and receives labels rather than an OOF dict, so
it needs the fold assignment passed in alongside the clusters.

**Verify:** the `delta_mean` null should centre near 0.5. It currently centres on 0.456. Judge
against "near 0.5" rather than exactly 0.5 — block permutation with a 76/15/9 class split moves it
slightly on its own. The current displacement is larger than that cause alone accounts for.

---

## Step 4 — fold the private copies into the shared helper

Two in-scope scripts carry their own fold loop rather than calling `probes.py`, which is how the
two versions diverged in the first place. Replace the loops; do not patch them in place.

**`src/esm2_mech/experiments/mechanism/mechanism_delta_family_split.py`**

`run_probe_on_splits` (around line 40) duplicates the shared body: per-fold PCA, logistic
regression, per-fold metrics, then the same concatenation and pooled macro-F1 at lines 100–112.
Route it through the shared helper. Its three consumers then need the folds passed through — the
two `bootstrap_mechanism_metrics` calls at lines 220 and 241, and the `paired_oof_diff` call at
line 312.

**`src/esm2_mech/experiments/stability/megascale_stability.py`**

The regressor loop around lines 51–105 has the same shape, concatenating predictions and computing
a single pooled Spearman at line 103. Fold-wise rank correlation must be averaged, not pooled.
`megascale_mlp.py` shares the pattern.

`naive_baseline.py` and `family_clustering.py` also have their own loops. Neither computes a
ranking metric over pooled probabilities, so check them rather than assuming a change is needed.

---

## Step 5 — stop resampling the label unit in the family probe

**File:** `src/esm2_mech/experiments/mechanism/family_clustering.py`

`_family_probe_bootstrap_ci` (line 232) resamples `oof["families"]` for both metrics. That probe
predicts the family, so resampling families changes the class set on every draw: each resample
averages macro-F1 over a different and smaller set of classes, which shifts the value
systematically instead of scattering it around the point estimate. The symptom is an interval that
does not contain its own point — 0.4938 against [0.369, 0.473] for the wildtype view, and the same
for the mutant view. The accuracy entry is unaffected because accuracy does not average per class.

Resample genes within families instead, which leaves the class set intact. If that is not
practical, report accuracy alone for this probe and state why in the report.

Do not attempt to repair the macro-F1 computation. It is correct; the resampling scheme is wrong.

---

## Step 6 — compute the leakage headline and its interval identically

**File:** `src/esm2_mech/experiments/mechanism/leakage_fraction.py`

Two mismatches between the headline and its interval:

- The headline averages pooled macro-F1 across five seeds (`_pick_macro_f1`, line 54). The
  interval is computed from the seed 0 out-of-fold cache alone (`leakage_fraction_ci`, line 86).
- The headline takes the chance floor from `naive_baseline.json` and holds it fixed
  (`_measured_chance`, line 47). The interval recomputes the floor on every resample (line 118).

Put both on the same basis, including the treatment of the floor, and report the interval
alongside the headline rather than only in the result file.

Note for whoever writes the report afterwards: whichever basis is chosen, the interval includes
zero, so the merged-dataset leakage fraction is not distinguishable from zero and must be stated
that way.

---

## Step 7 — update call sites and tests in the same change

The project convention is that a shared contract change fixes every caller and test in the same
commit.

In-scope callers needing the folds threaded through: `mechanism/mlp.py` (line 59),
`pathogenicity/pathogenicity_control.py` (line 268), `geometry/magnitude_direction.py`
(lines 197, 280), `geometry/conservation_axis.py` (lines 193, 266),
`proteome_features/enzyme_classification.py` (lines 398, 418).

Out-of-scope callers listed in [`TODO.md`](../../TODO.md) will fail to import or call once the fold
argument becomes required. That is intended — it is what stops them silently producing pooled
numbers. Leave them failing; do not add a default to keep them running.

Tests to update: `tests/utils/test_bootstrap.py`,
`tests/experiments/mechanism/test_mechanism_delta_family_split.py`,
`tests/experiments/mechanism/test_leakage_fraction.py`,
`tests/experiments/stability/test_megascale_stability.py`.

Add a regression test that a fold-aware AUROC on synthetic data with deliberately offset per-fold
probability scales returns the per-fold value and not the pooled one. That is the defect in one
assertion.

---

## Before re-running: preregistration amendment

Dated, committed, and written before any output is inspected. Four items:

1. The two numbers currently both called the chance floor are given separate names. The measured
   majority-class value keeps "chance floor"; the 2A threshold is named for what it is, the score
   set by the strongest mutation-only probe.
2. The 2A threshold is re-derived from the fixed numbers. Step 2 moves the nonlinear delta score,
   and that score *is* the threshold, so the bar must be recomputed before the claim is judged
   against it.
3. The 2B adjudication is pinned on all three open choices — which feature, which quantity, and
   how the five seeds combine. It currently specifies none of them, and each changes the verdict.
4. The rare-class rule from step 2 is recorded as part of the metric definition.

---

## Re-run order

1. Section 4, all steps, five seeds. The permutation step runs on all five seeds, not one.
2. Section 5, then 6, then 7.
3. Section 8. Note its current result predates the commit that fixed its own decision-rule labels,
   so it needs re-running regardless of this work.

Record the commit hash alongside the seed in every result file.

Regenerate the reports rather than editing numbers in them. Mark the reports under
`reports/run_biorxiv/bak/` superseded in their own text.

---

## Acceptance criteria

- The delta's per-class family-split AUROCs sit near their per-fold values, not below 0.5.
- The `delta_mean` permutation null centres near 0.5.
- Every reported interval brackets the point estimate it is attached to, including the family
  probe's macro-F1.
- The headline value and its interval are the same quantity for the leakage fraction.
- The 2A threshold in the report is the recomputed one, and is not called the chance floor.
- A regression test fails if pooled ranking is reintroduced.
- The out-of-scope scripts in `TODO.md` fail loudly rather than silently producing pooled numbers.
