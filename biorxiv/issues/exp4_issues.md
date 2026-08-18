# Experiment 4 — issues found in `results/run_biorxiv`

Review of runbook section 4 (ESM-2 delta-embedding mechanism, steps 4.1–4.7) against the result
files in `results/run_biorxiv/` and the decision rules in `PREREGISTRATION_run_biorxiv.md`.

Dataset as run: 17,770 variants, 1,931 genes, 1,144 Pfam families; LOF 13,556 / GOF 2,668 /
DN 1,546. Measured majority-class macro-F1 is 0.2883 under gene-split and 0.2896 under
family-split.

Issues are ordered by severity. Each states the defect, the evidence, why it matters, and the
root-cause fix.

---

## 1. Ranking metrics are computed on probabilities pooled across independently fitted folds

**Defect.** `run_probe_on_splits` concatenates the out-of-fold probability vectors from all five
folds into a single array, and the downstream AUROC, AUPRC and permutation code rank that
concatenation as one list. Each fold has its own fitted model with its own probability scale, and
under family-split each fold holds a different set of protein families with a different class
composition. Ranking across the concatenation therefore compares scores that were never on a
common scale.

**Evidence.** Comparing the per-fold average AUROC against the pooled-OOF AUROC on seed 0,
family-split:

| Feature | Class | Per-fold mean | Pooled OOF | Difference |
|---|---|---|---|---|
| wt_only_mean | GOF | 0.764 | 0.768 | +0.004 |
| wt_only_mean | DN | 0.760 | 0.771 | +0.012 |
| wt_only_mean | LOF | 0.808 | 0.831 | +0.022 |
| delta_per_residue | GOF | 0.601 | 0.552 | −0.049 |
| delta_per_residue | DN | 0.604 | 0.596 | −0.007 |
| delta_per_residue | LOF | 0.594 | 0.560 | −0.035 |
| delta_mean | GOF | 0.608 | 0.403 | −0.204 |
| delta_mean | DN | 0.546 | 0.475 | −0.071 |
| delta_mean | LOF | 0.581 | 0.424 | −0.157 |

The size of the distortion tracks the weakness of the underlying signal. Where within-fold
discrimination is strong the between-fold scale offsets are swamped and pooling is nearly
harmless; where it is weak the offsets dominate and drive the pooled value below 0.5. The delta —
the feature the experiment exists to characterise — is the worst-affected case.

**Why it matters.** Two published quantities inherit the defect.

The AUROC confidence intervals in `family_split_baselines_seed{0..4}.json` are computed from the
pooled predictions, while the AUROC values in `aggregate.json` and the report tables are per-fold
averages. For `delta_mean` the reported family-split GOF AUROC is 0.584 and the interval attached
to it is centred on 0.403. The interval does not bracket the number it is presented alongside.

The `delta_mean` permutation test scores macro one-vs-rest AUROC on the same pooled predictions.
Its observed statistic is 0.4339, which is exactly the mean of the three pooled per-class values
above, and is below 0.5 despite every per-class per-fold value lying between 0.55 and 0.61.

**The enabler.** The out-of-fold collector in `utils/probes.py` gathers four aligned arrays — true
labels, probabilities, gene ids and row ids — and discards which fold each row came from. That
omission is the root cause: once the fold index is gone, no downstream consumer can be fold-aware
even if it wants to be, so pooling is the only thing the data structure permits. Carrying the fold
index through the collector is the first change, and every other fix in this document depends on
it.

**Fix.** Compute ranking metrics within each fold and combine across folds, with the cluster
bootstrap resampling inside that structure, so that the intervals and the headline values are the
same quantity. Rank-transforming each fold's scores to a common scale before pooling is a valid
alternative but is the less direct of the two, since the headline tables are already per-fold.

The permutation test has to change in the same way and at the same time. A fold-aware statistic
scored against labels that are shuffled across the whole dataset reintroduces exactly the
confound the fix removes, because a shuffle that moves labels between folds changes each fold's
class composition. The shuffling must happen within fold, so that the null holds the fold
structure fixed and varies only the label assignment.

Do not adjust the affected numbers without changing how they are computed.

---

## 2. The pooling is in shared probe code and reaches four other sections

**Defect.** The concatenation described in issue 1 is not local to the mechanism experiment. The
same out-of-fold collection and pooled scoring lives in `utils/probes.py`, which the pathogenicity
control (§5), the geometry analysis (§6), the enzyme classification (§8) and several follow-up
scripts all call. The stability control (§7) carries its own copy of the pattern in
`megascale_stability.py`, where fold-wise rank correlations are pooled into a single Spearman
value the same way.

**Why it matters.** Four preregistered claims rest on intervals produced by this code path. Claim
2C requires the pathogenicity AUROC interval to exclude 0.85, claims 2D and 2E rest on paired
bootstraps over the conservation and delta axes, and claims 2F–2H rest on enzyme family-split
scores. None of those intervals can be relied on until the shared code is fixed, and the pairing
in 2E is the most exposed of them, because a paired difference between two pooled quantities
carries both distortions at once.

The distortion is largest where the underlying signal is weakest, so the strong positive controls
are likely to move least. That is a prediction to check rather than a reason to skip them.

**Fix.** Fix the shared helper once, then re-run every section that calls it. Treat this as the
first task in the sequence: the adjudication of 2A depends on issue 5 below, which depends on the
metric being settled first.

---

## 3. The permutation null for `delta_mean` is not centred near chance

**Defect.** A label-permutation null for an AUROC-type statistic should sit close to 0.5. The
`delta_mean` null centres on 0.456 with the observed value at 0.434. This follows from issue 1:
labels are permuted in cluster blocks, which preserves the family-block composition that the
pooled probability scale is confounded with, so both the observed statistic and the null are
displaced below 0.5 together.

Block permutation with three classes at 76/15/9 percent does not have to land exactly on 0.5 even
when the scoring is sound, because each shuffle draws a different class mix into each block. The
expected departure from that cause alone is small, and the observed displacement is larger than
it, but the recheck should be judged against "near 0.5" rather than against 0.5 exactly.

**Why it matters.** The p-value of 0.639 may survive, since the observed value and the null are
displaced by the same mechanism. But the reported observed statistic cannot be described as an
AUROC in the paper, and no reader can check the test without reproducing the artefact.

**Fix.** Follows from issue 1. Recompute once the statistic is fold-aware and the shuffling is
within fold, then confirm the null centres near 0.5.

---

## 4. The permutation test ran on one seed, and on the seed least favourable to the leakage account

**Defect.** Step 4.6 runs with a single seed. Seed 0 has the highest family-split score of the
five for `wt_only_mean` — 0.5123 against 0.4330, 0.4935, 0.5091 and 0.5012 on pooled macro-F1,
the basis used throughout this document — which makes it the seed with the smallest gene-to-family
gap, at 0.042 against 0.064 to 0.157 elsewhere. Its paired gap interval, [−0.035, 0.106], is the
only one of the five that straddles zero.

**Why it matters.** The leakage evidence rests on the gap between the two splits. Presenting the
single-seed analysis from the seed with the smallest gap understates the effect and invites the
objection that the seed was chosen.

**Fix.** Run the permutation test across all five seeds and report the distribution, or state
explicitly which seed was used and why. Note that the paired gap intervals themselves were
computed on all five seeds and are not affected.

---

## 5. The preregistration uses "chance floor" for two different numbers, and the rule is self-referential

**Defect.** Claim 2A defines the floor as the run's own five-seed average of the nonlinear
`delta_mean` family-split probe. For this run that is **0.3703**. The measured majority-class
score is **0.2883**. Both are called the chance floor in the preregistration text.

**Why it matters.** 0.370 is not a no-information baseline. It is the score achieved by the best
mutation-only probe in the same run, and issue 7 below shows that probe is consistently above the
majority-class floor, so the bar is being set by a measurement that carries signal. Describing it
as chance in the paper is inaccurate, and a reviewer checking the majority-class calculation will
get a different number.

The rule is also self-referential in a way that has practical consequences. The threshold is
derived from the nonlinear delta probe and then used to judge the delta, so if a future run's
nonlinear probe scores higher the bar rises with it and 2A becomes easier to affirm rather than
harder. The fix in issue 1 will itself move the nonlinear probe's score, which means the threshold
is not stable across the very change this document recommends.

**Fix.** Settle the metric first. Once ranking and macro-F1 are both computed fold-aware, fix the
2A threshold to the measured majority-class value, which is a genuine no-information baseline and
does not move when the probes move. Report the nonlinear probe's score beside it as a named
comparator rather than as the threshold. This is a change to the rule, not only to its vocabulary,
and it has to be recorded as a preregistration amendment with its date and reason.

**Adjudication under the rule as written is unaffected.** The rule requires the family-split
interval's upper bound to fall below the threshold plus 0.05, that is below 0.420. The linear
delta's interval upper bound is 0.306, so 2A is affirmed. It is also affirmed under the
majority-class threshold proposed here, since 0.306 falls below 0.338.

---

## 6. The 2A comparison mixes two ways of computing macro-F1

**Defect.** The 2A threshold (0.3703) is a per-fold average taken from the nonlinear probe
results, which do not record a pooled value at all. The interval it is compared against is
computed on pooled out-of-fold predictions (point 0.2883). The two sides of the comparison are
computed differently.

**Why it matters.** It is the same class of problem as issue 1. Here it changes nothing, because
the pooled and per-fold macro-F1 for `delta_mean` agree closely (0.2883 against 0.2903), but the
comparison should not depend on that coincidence.

**Fix.** Make both sides of the rule use the same computation, and record both quantities for the
nonlinear probes so the choice is explicit.

---

## 7. The nonlinear delta clears the chance floor on every seed

**Finding, not a defect.** Under family-split, the MLP and kNN probes on `delta_mean` produce
cluster-bootstrap intervals whose lower bounds sit above the measured majority-class floor of
0.2883 on all five seeds:

| Probe | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---|---|---|---|---|
| MLP | [0.329, 0.386] | [0.342, 0.444] | [0.347, 0.437] | [0.332, 0.408] | [0.350, 0.452] |
| kNN | [0.330, 0.395] | [0.334, 0.395] | [0.327, 0.395] | [0.336, 0.400] | [0.337, 0.401] |

These intervals are themselves computed on pooled predictions, so issue 1 applies to them, but in
the conservative direction: pooling depresses weak scores, so the fold-aware values will be at
least this high.

Per-gene scoring gives essentially the same value as per-variant for the MLP (0.375 against
0.370), so this is not an artefact of high-variant-count genes dominating the metric.

**Why it matters.** The statement that the delta sits at chance is true of the linear probe only.
The nonlinear probes are modestly but consistently above the majority-class floor. This does not
overturn the finding — the values remain far below the absolute-embedding score of roughly 0.45,
and far below any usable level — but the wording in the reports needs to distinguish the linear
result from the nonlinear one.

---

## 8. The per-residue delta reads differently on the two metrics, and the reports state only one

**Defect in the current description.** `delta_per_residue` is described in the run 6 reports as
sitting marginally above the floor, on the strength of macro-F1 alone. On macro-F1 that is not
supportable: its family-split intervals have lower bounds of 0.286–0.287 across all five seeds,
against a floor of 0.2883, so every interval touches or crosses the floor.

On ranking the same feature is above chance. Its family-split LOF AUROC interval excludes 0.5 on
all five seeds, and its GOF interval excludes 0.5 on four of the five:

| Seed | AUROC GOF | AUROC DN | AUROC LOF |
|---|---|---|---|
| 0 | 0.552 [0.513, 0.613] | 0.596 [0.505, 0.711] | 0.560 [0.527, 0.607] |
| 1 | 0.576 [0.531, 0.625] | 0.556 [0.456, 0.674] | 0.573 [0.548, 0.604] |
| 2 | 0.550 [0.499, 0.621] | 0.586 [0.500, 0.694] | 0.551 [0.522, 0.593] |
| 3 | 0.570 [0.528, 0.622] | 0.594 [0.496, 0.720] | 0.571 [0.539, 0.613] |
| 4 | 0.597 [0.563, 0.642] | 0.603 [0.515, 0.718] | 0.589 [0.563, 0.622] |

These are the pooled values, and the per-fold means in issue 1 are higher still (0.59–0.60), so
the conclusion holds on both computations.

**Why it matters.** The preregistration's own stated reason for scoring the delta on AUROC is that
macro-F1 stays pinned near the floor when a probe predicts the majority class everywhere, whatever
ranking signal is present. That reasoning applies to `delta_per_residue` exactly as it applies to
`delta_mean`, and it is the same reasoning issue 7 uses to credit the nonlinear probes. Judging
this feature on macro-F1 alone applies a standard the rest of the document does not.

**Fix.** Describe it as at the floor on macro-F1 and weakly above chance on ranking, and name the
metric in the sentence. Do not describe it as carrying weak signal without saying which metric
that refers to, and do not describe it as indistinguishable from chance, which is false on
ranking.

---

## 9. The leakage fraction and its confidence interval are computed on different bases

**Defect.** `leakage_fraction.json` reports a leakage fraction for `wt_only_mean` of 0.2996,
derived from the five-seed mean pooled macro-F1 of both splits. The confidence interval attached
to the same entry has a point estimate of 0.159, which is the seed 0 value alone
(0.042 / (0.554 − 0.288)). The interval also recomputes the majority-class floor on each
resample, while the headline takes the floor from `naive_baseline.json` and holds it fixed, so the
two differ in the denominator as well as in the seed basis. They are presented as one result.

**Why it matters.** The interval is [−0.181, +0.387] and therefore includes zero. Whichever basis
is chosen, the merged-dataset leakage fraction is not distinguishable from zero, and that must be
stated. Presenting a headline of 30% next to an interval built from a different computation
obscures it.

**Fix.** Compute the headline and the interval the same way, including the treatment of the floor.
Report the interval alongside the headline in the reports rather than only in the result file.

**Context.** This value has fallen across runs: 62.8% in run 0, 40.1% in run 6, 30.0% here. The
single-source replication remains the stronger evidence for leakage — see issue 13.

---

## 10. Two smaller defects in the clustering and baseline results

**A bootstrap that resamples the quantity it is predicting.** In `family_clustering.json`, the
`wt_mean` family-probe macro-F1 has a point estimate of 0.4938 and an interval of [0.369, 0.473],
so the point lies outside the interval. The `mut_mean` view has the same defect, at 0.4914 against
[0.368, 0.471].

The cause is that the bootstrap resamples Pfam families while the family probe's prediction target
is the Pfam family. Every resample drops some families and duplicates others, so each resample
scores macro-F1 over a different and smaller class set than the point estimate does, and averaging
over fewer classes shifts the value systematically rather than scattering it around the point.
The accuracy entry for the same probe is unaffected because accuracy does not average per class.

The fix is to stop resampling the label unit. For the family probe, resample genes within
families, which leaves the class set intact, or report accuracy alone for that probe and say why.
Repairing the macro-F1 computation itself would not help, because the computation is correct and
the resampling scheme is what is wrong.

**Unequal bootstrap coverage across features.** The cluster bootstrap for `foldx_ddg` resamples
666 clusters and `alphamissense` 1,139, against 1,144 for the embedding features. FoldX in
particular covers well under half the family set, so its row is not computed over the same
variants as the others and should not be compared to them without a coverage note.

---

## 11. One result predates its own fix

**Checked and clear for this section.** Step 4's outputs were all produced by one version of the
code. Seven commits landed after the mechanism results were written, but none of them touch the
mechanism path — they change the stability, geometry, pathogenicity, enzyme and path-resolution
code. Sections 5, 6 and 7 are likewise each downstream of the commits that changed them.

**The exception is section 8.** `enzyme_classification_summary.json` was written at 15:23:07 on
2026-08-18. The commit that fixed the enzyme decision-rule labels to match the preregistration
(`f1369ee`) landed at 15:23:19, twelve seconds later. That result therefore predates its own fix
and must be re-run before any enzyme number is cited. Claims 2F–2H rest on it.

**Fix.** Re-run section 8. Record the commit hash alongside the seed in result files so this is
answerable by reading a file rather than by comparing timestamps against a git log.

**On the earlier report.** `reports/run_biorxiv/bak/report_mechanism.md` was produced by pre-fix
code, so its numbers differ from the current ones as expected and are not a competing result.
Mark it superseded in its own text so it is not read as a current analysis.

---

## 12. The adjudication rule for claim 2B does not determine a verdict

**Defect.** Claim 2B is overturned if "the split-gap CI spans zero". The rule does not say which
feature's gap adjudicates it, which seed, or how the five seeds combine. All three choices change
the answer.

| Feature | Seeds whose gap interval excludes zero |
|---|---|
| `wt_only_mean` | 4 of 5 (seed 0 straddles) |
| `mut_only_mean` | 3 of 5 (seeds 0 and 2 straddle) |
| `wt_concat_mut` | 2 of 5 (seeds 0, 1 and 4 straddle) |

The claim points at runbook step 4.5, whose output is the leakage fraction rather than the gap,
and that quantity's interval spans zero comfortably (issue 9). So even the choice of which
artifact adjudicates the claim is open.

**Why it matters.** A preregistered rule exists to fix the verdict before the numbers are seen.
This one currently leaves three degrees of freedom, each of which moves the answer, so whatever
verdict is recorded will have been chosen after the fact. That is the failure mode preregistration
is meant to prevent, and it is more damaging to the paper than either possible verdict.

**Current reading, stated so the choice is visible.** On the feature the reports treat as the
headline, the gap excludes zero on four of five seeds, so leakage is present. It is smaller and
less certain than run 6 reported, and the single-seed view happens to fall on the one seed that
straddles zero (issue 4). The single-source subset excludes zero clearly (issue 13) and is the
cleaner design, since it removes the curation-source confound.

**Fix.** Pre-specify the three missing choices before re-running: the adjudicating feature, the
adjudicating quantity, and the seed-combination rule. The natural choices are `wt_only_mean` as
the headline absolute-embedding feature, the paired split gap rather than the leakage fraction,
and a rule stated over all five seeds rather than seed 0. Record them in the preregistration as an
amendment with its date, not as a silent edit.

---

## 13. Findings that replicate cleanly

Recorded so that the issues above are read in proportion.

The linear delta sits at the measured floor under both splits on all five seeds, with a
family-split interval of [0.268, 0.306] that contains the floor. The wildtype and mutant
embeddings are indistinguishable from each other on every metric, so what clears the floor is
protein identity rather than the mutation. AlphaMissense is at the floor for mechanism on both
splits.

Family clustering is strong and now has intervals: five-nearest-neighbour family purity is 0.2543
against a shuffled null of 0.0052, and a linear probe recovers which of 145 families a gene
belongs to with 60.3% accuracy against a 4.4% majority baseline. The delta reduces that probe to
0.0437, exactly the baseline. Within-family mechanism agreement is 83.2%. The macro-F1 figures for
the family probe are subject to issue 10; the accuracy figures quoted here are not.

The permutation test for `wt_only_mean` returns p = 0.001 against a refit null centred on 0.328,
so its family-split score is above chance by the test the preregistration treats as load-bearing.

The single-source replication holds and is the strongest leakage evidence in the section. On the
Gerasimavicius-only subset (10,138 variants, 942 genes, 666 families, recomputed floor 0.279) the
delta stays at the floor at 0.280 while `wt_only_mean` falls from 0.611 to 0.463, and the paired
gap interval on seed 0 is [0.016, 0.257], excluding zero.

---

## 14. Dataset drift

This run covers 17,770 variants across 1,931 genes and 1,144 families. The run 6 reports describe
17,826 variants across 1,935 genes and 1,134 families. Every carried-over number needs
recomputing rather than copying, and the family count moved in the opposite direction to the
variant count, which is worth confirming is intended.

---

## 15. Remediation plan, in order

The order matters. Each step depends on the one before it, and two of them must happen before any
number is looked at again.

**1. Carry the fold index through the out-of-fold collector.** The enabler described in issue 1.
Nothing else is possible until the fold survives collection.

**2. Make the ranking metrics fold-aware.** AUROC, the precision-recall areas, and the rank
correlations in the stability control should be scored within each fold on the resampled rows and
averaged, rather than ranking one concatenated list.

*Decide the rare-class rule before implementing.* Dominant-negative is 9% of the labels and
family-split folds are uneven, so some folds will lack a class entirely. The current code silently
skips a class when a fold cannot score it. Combined with bootstrap resampling, that makes the set
of contributing folds vary between draws, which is the same "each resample scores a different
statistic" failure diagnosed for the family probe in issue 10. Choose one rule — require every
fold to score the class, or discard the resample — and record how many resamples were discarded.

**3. Put the threshold metric on the same basis.** Macro-F1 is currently pooled in the reports and
per-fold in the tables, and for `wt_only_mean` the two differ by enough to matter (0.512 against
0.488 on seed 0).

State the reason accurately. Pooled macro-F1 is not corrupted by the scale confound — the class is
decided per row by argmax, so there is no cross-fold comparison in it. The reason to change it is
consistency with the ranking metrics and with the 2A threshold, not correctness. Recording the
wrong reason invites the objection that other numbers were changed without cause.

**4. Permute within fold.** Both permutation paths, as set out in issue 1. A fold-aware statistic
scored against a whole-dataset shuffle undoes step 2.

**5. Fold the private copies into the shared helper.** The mechanism experiment and the stability
control each carry their own fold loop, and several follow-up scripts hold partial copies. Fixing
only the shared helper leaves those behind, which is how the two versions diverged in the first
place.

**6. Stop resampling the label unit in the family-recognition probe.** Resample genes within
families, as set out in issue 10.

**7. Compute the leakage headline and its interval identically** — same seeds, same treatment of
the floor, as set out in issue 9.

### Before re-running: one dated preregistration amendment

Written and committed before any output is inspected, covering all four open specification gaps:

- The two numbers currently both called the chance floor are given separate names (issue 5).
- The 2A threshold is re-derived from the fixed numbers. Step 2 will move the nonlinear delta
  score, and that score *is* the threshold, so the bar has to be recomputed before the claim is
  judged against it — not after (issues 5 and 6).
- The 2B adjudication is pinned down on all three open choices: which feature, which quantity, and
  how the five seeds combine (issue 12).
- The rare-class rule from step 2 is recorded as part of the metric definition.

### Out of scope, deferred

Twelve experiments outside the `run_biorxiv` sequence call the same helpers or carry their own
copy of the fold loop. They are not being re-run, so they are not blocking. They are listed with
their remediation conditions in [`TODO.md`](../../TODO.md), including the call-site decision that has
to be made before the shared fix lands.

### Also required, independent of the above

Re-run section 8. Its enzyme result predates the commit that fixed its decision-rule labels, and
claims 2F–2H rest on it (issue 11). Record the commit hash next to the seed in result files so
this class of question is answerable from the file rather than from timestamps.
