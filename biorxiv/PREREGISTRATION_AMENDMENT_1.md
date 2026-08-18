# run_biorxiv pre-registration — amendment 1

**Date: 2026-08-18. Written and committed before any output of the re-run is inspected.**

This amendment governs where it differs from `PREREGISTRATION_run_biorxiv.md`. The original text
is left unedited so that what was specified beforehand stays readable; each rule changed here
carries a pointer back to this file at its place in that document.

The amendment exists because an audit of the run found that the scoring code and the specification
were both underdetermined in ways that let a verdict be chosen after the numbers were seen. The
scoring defects and their fixes are recorded in the implementation history. This document covers
only the specification: four rules that either named the wrong quantity, moved with the result they
were meant to judge, or did not determine a verdict at all.

Nothing here changes a hypothesis, a gate direction, or a threshold that was already unambiguous.

---

## 1. The two quantities that were both called the chance floor

The original text uses "chance floor" for two different numbers. One is the score a no-information
predictor achieves. The other is the score achieved by the run's own strongest mutation-only probe.
Only the first is a floor.

| Name | Definition | Where it comes from |
|---|---|---|
| Measured chance floor | Macro-F1 of a majority-class predictor under the split in question, averaged over the run's five seeds | The run's own naive baseline result file |
| Nonlinear delta reference | Family-split macro-F1 of the run's nonlinear `delta_mean` probe, averaged over five seeds | The run's own nonlinear probe results |

The measured chance floor is what the term means from here on. The nonlinear delta reference keeps
its own name and is never described as chance, in this document, in the reports, or in the paper.
It is a measurement that carries signal, and the audit found it sits consistently above the
measured chance floor on every seed.

The measured chance floor is read live from the run's own result file rather than carried between
runs. It does not move when the probes are refit: a majority-class prediction is decided per row,
so no fold-scale comparison enters it.

## 2. The 2A threshold

**Original rule.** The mechanism delta is affirmed to sit at chance if the family-split interval's
upper bound falls below the chance floor plus 0.05, where the floor is the nonlinear delta probe's
own family-split score.

**Problem.** The threshold was derived from a probe and then used to judge a probe. A run whose
nonlinear probe scored higher would raise its own bar, making the claim easier to affirm rather
than harder. The bar also moves whenever the scoring code changes, which it now has.

**Amended rule.** The 2A threshold is the measured chance floor under the family split, plus 0.05.
Claim 2A is affirmed when the family-split interval's upper bound for the linear `delta_mean` probe
falls below that threshold. The verdict rule is otherwise unchanged: a straddling interval is not
adjudicated rather than confirmed, and the permutation test remains refutation-only.

The nonlinear delta reference is reported beside the threshold as a named comparator, so a reader
can see how far the strongest mutation-only probe sits above the floor. It does not set the bar.

The threshold's value is filled in from the re-run's own naive baseline. It is not carried over
from the pre-fix results, and it is recorded before the claim is judged against it.

## 3. The permutation test: seeds and combination

**Original rule.** The permutation test runs on seed 0.

**Problem.** The audit found seed 0 has the smallest gene-to-family gap of the five and the only
paired gap interval that straddles zero. A single-seed test on the least favourable seed both
understates the effect and invites the objection that the seed was chosen. Neither reading should
be available.

**Amended rule.** The permutation test runs on all five seeds and the full distribution of p-values
is reported. Because the test is refutation-only for 2A, the refutation fires when at least three
of the five seeds return a p-value below 0.05. A minority of significant seeds is reported as a
split result and refutes nothing.

The resolution limit in the original text still applies per seed: a p-value sitting at the
1/(N+1) floor is reported as unresolved rather than as a measurement.

## 4. The 2B adjudication

**Original rule.** Claim 2B is overturned if the split-gap confidence interval spans zero.

**Problem.** The rule does not say which feature's gap adjudicates, which quantity is the gap, or
how the five seeds combine. The audit found each of the three choices changes the answer, which
means any verdict recorded under the rule as written would have been chosen after the fact.

**Amended rule**, pinning all three:

| Open choice | Decision |
|---|---|
| Adjudicating feature | `wt_only_mean`, the headline absolute-embedding feature |
| Adjudicating quantity | The paired gene-split-minus-family-split macro-F1 gap, not the leakage fraction |
| Seed combination | The gap's interval must exclude zero on at least three of the five seeds |

The leakage fraction is a derived ratio whose denominator is itself an estimate, and the audit
found its interval spans zero on the merged dataset under either basis. It is reported as a
descriptive quantity with its interval, and it does not adjudicate 2B.

The single-source subset, where the curation-source confound is absent, is reported alongside as a
named secondary analysis. It is the cleaner design, but it was not pre-specified as the adjudicator
and does not become one here.

## 5. The rare-class rule, as part of the metric definition

Every ranking metric and every macro-F1 in this run is computed inside each cross-validation fold
and averaged across folds. Folds are fitted independently and their probabilities are on their own
scales, so a metric that ranks the concatenation of all folds is not the quantity these rules
intend.

Under bootstrap resampling a fold can lose a rare class entirely. Dominant-negative is roughly nine
percent of the labels, so this is the case that matters.

**Rule.** Every fold must be able to score every class the metric averages over. A resample where
any fold cannot is discarded whole rather than scored over the folds that survive, because a draw
averaging over a different set of folds or classes is estimating a different quantity. The number
of discarded resamples is recorded next to every interval.

The expected discard rate is far below one percent, because every family-split fold already
contains dominant-negative variants on every seed. A materially higher rate is treated as a fault
in the resampling unit or the fold construction and investigated, not absorbed into the result.

## 6. What this amendment does not do

It does not change any hypothesis, any gate direction, or any threshold in claims 2C through 2H.
It does not adjust a reported number. Every value affected by the scoring fixes changes as a
consequence of being recomputed, and the reports are regenerated rather than edited.

If the amended 2A threshold changes the verdict on 2A relative to the rule as originally written,
that is reported as a finding with both verdicts stated, not resolved silently in favour of either.
Under the rule as written and on the pre-fix numbers, 2A was affirmed, and it was also affirmed
under the threshold adopted here; whether that survives the re-run is what the re-run decides.
