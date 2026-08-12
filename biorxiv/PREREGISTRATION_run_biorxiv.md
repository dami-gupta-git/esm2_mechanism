# run_biorxiv pre-registration — inferential statistics (added 2026-07-22)

**Written before run_biorxiv executes.** This document governs run_biorxiv only — the original
(run0-era) pre-registration and its outcomes live in `docs/EXPERIMENT.md`. run_biorxiv re-scores
run6's science with dependency-aware error bars; the experiments, hypotheses, and gates are
unchanged.

These rules exist because run_biorxiv will attach confidence intervals to gates that currently
pass or fail on point estimates alone. Without a reading fixed in advance, an interval that lands
awkwardly invites a framing chosen after the fact. The run6 point estimates are recorded here so
the rules cannot be retro-fitted to the run_biorxiv intervals.

Methodology: `reports/run6/STATS_PLAN.md`. Execution, live status, and what changed since run6:
`RUNBOOK_biorxiv.md`.

## R7.1 — CI decision rule for gate verdicts

Every gate below is evaluated against a paired cluster-bootstrap 95% CI on its difference, not a
point estimate alone.

> **Affirmed** — the point estimate clears the threshold **and** the paired difference 95% CI
> excludes zero.
>
> **Not distinguishable** — the point estimate clears the threshold but the CI spans zero. This is
> reported as neither a pass nor a refutation.
>
> **Failed** — the point estimate does not clear the threshold. The verdict is failure regardless
> of the CI.
>
> **Underpowered** — a failed gate whose difference CI also spans the pre-registered threshold is
> reported as *underpowered to detect an effect of the pre-registered size*, **not** as evidence of
> no effect.

### Gates in scope, with run6 point estimates

**The measured chance floor is a rule, not a number.** It is the ESM-2 family-split floor read
from that run's own MLP `delta_mean` family-split probe, averaged over all five seeds, raising
rather than substituting if any seed is missing. What is pre-registered is the rule and the source,
so the floor moves with what the run measures.

In run6 the floor was 0.380. It is on record elsewhere in the run6 reports as 0.415 and 0.418;
those are report-text drift rather than competing definitions, and the run_biorxiv floor comes from
the single source named above. C1's 0.05 equivalence margin is read against whatever the run
measures. `scripts/compare_runs.py` flags the floor's movement so it is read rather than absorbed.

Margins are stated against the threshold, never against the bare floor. Against the bare floor the
lifts look much larger (seq +0.058, seq_struct +0.072) and the CI question becomes trivial; the two
framings have opposite robustness and must not be interchanged.

| Gate | Criterion as recorded | Run6 value | Margin | Run6 verdict |
|---|---|---|---|---|
| K1 | conservation alone AUROC > 0.85 | 0.891 | +0.041 | pass |
| K2 | conservation + delta improves over conservation by > 0.02 | +0.0023 | −0.018 | fail |
| K2b | conservation + delta improves over delta alone | +0.0345 | — | descriptive (no threshold) |
| H2 | stability random→family rho drop < 0.10 (LEAKY) | — | descriptive | descriptive |

**Exposure is concentrated in K2**, which already fails, so a CI spanning zero *reinforces* that
reading; it is the case where R7.1's underpowered clause applies rather than threatening the
conclusion.

## R7.2 — Confirmatory / exploratory split

The paper runs gate comparisons across seven reports. The protection against selecting a result
after the fact is that the claims the thesis rests on are enumerated **before** the run, and
everything else is labelled exploratory and asserts nothing.

### Confirmatory claims (five)

| # | Claim | Instrument |
|---|---|---|
| C1 | The mechanism delta sits at the measured chance floor under family-split | equivalence margin: CI upper bound below floor + 0.05, with the permutation p as a refutation test |
| C2 | The absolute-embedding gene→family gap is non-zero (homology leakage exists) | paired bootstrap on the split gap |
| C3 | Pathogenicity clears AUROC 0.85 family-split (positive control) | CI excludes 0.85 |
| C4 | Conservation alone matches or beats the embedding delta for pathogenicity | paired bootstrap (K1/K2) |

C1 and C3 are the load-bearing pair (the dissociation) and C2 is the leakage account. Those three
are the paper. C4 is the characterisation payoff; if the confirmatory set must be trimmed, trim
from C4, never C1–C3.

**C1 is a null claim and is adjudicated as one.** A confidence interval that straddles the chance
floor is not evidence that the score sits at the floor — an interval wide enough to straddle
anything straddles the floor too. C1 is affirmed only if the family-split CI's upper bound falls
below the measured floor plus 0.05. If the interval is wider than that, C1 is recorded as **not adjudicated** — underpowered to
separate a real effect from none — never as confirmed. The permutation test runs on C1 but in one
direction only: a significant p refutes C1, and a non-significant p does not confirm it.

**No multiplicity correction is applied.** No verdict in a set this size turns on one: C2 and C3
clear their thresholds by wide margins, and C1 is a null claim, where raising the bar for rejection would make the claim easier to assert rather than harder.
Enumerating the set in advance is the safeguard.

### Exploratory (labelled, not corrected)

Per-class AUROCs, the 28-family within-family table and its per-family cells, the biochemistry R²,
the magnitude/direction decomposition, per-feature leakage fractions, and all descriptive geometry.
These are labelled exploratory in their reports and assert nothing the paper relies on. The
28-family within-family table is included here: it is restated as an exploratory screen rather than
corrected, since correcting a screen implies it was confirmatory.

**Failing gates make no confirmatory claim.** K2 is reported under R7.1's underpowered clause and
is not a confirmatory item, because reporting it as such would imply it was a positive finding
under test.

**Amendment 2026-07-22 — the label-heterogeneity threat is named and cited.** Badonyi & Marsh 2025
(`papers/mechanism_2025.pdf`, bioRxiv 2025.03.13.642984) report that 43% of multi-phenotype
dominant genes and 49% of mixed-inheritance genes carry both LOF and non-LOF mechanisms. This
project assigns one mechanism label per gene, so some fraction of variants is mislabelled by
construction, and a reviewer has a citable alternative explanation for C1: the delta sits at the
floor because the labels are noisy, not because the embedding lacks mechanism signal. No confidence
interval addresses this — the threat is to what the labels mean, not to how many samples there are.
It is answered by Tasks 2d and 8 in [`FOLLOWUP_biorxiv.md`](FOLLOWUP_biorxiv.md): whether the null
survives on cleanly-labelled genes, and how far realistic label noise moves a working probe. C1's
statement in the reports cites this paper and points at those two results rather than asserting the
labels are adequate.

**Homology partitions beyond Pfam family are out of scope.** The mechanism null is measured under
the Pfam family partition only. Whether it also holds under a coarser partition (Pfam clan) or a
sequence-identity clustering is named in the paper as work for a follow-up, not tested here. The
paper therefore makes no claim that the null is independent of the Pfam family definition, and no
claim about how leakage varies with partition strictness.

## R7.3 — Resampling unit

The resampling unit matches the unit the split holds out.

| Metric | Resample |
|---|---|
| Gene-split | genes |
| Family-split | **families** |
| Gene-split minus family-split gap | **families** (the coarser of the two arms) |
| Permutation null for a family-split metric | **families**, swapping whole label blocks (R7.5) |

The split-gap case resamples families because its family-split arm's variance is only correct
under family resampling; a gene-resampled gap understates it. A family resample induces a valid
gene resample, but not the reverse. The gene-resampled interval is reported alongside as a
labelled sensitivity check.

**Expected and pre-registered:** family-split CIs will be visibly wider than gene-split ones.
There are 1,134 families but 833 are singletons, so the effective cluster count is far below the
gene count. That widening is the correct answer, not an artifact to tune away. The effective
cluster count is reported next to every family-split interval.

**Pairing.** A difference between two arms is resampled once per replicate and that same resample
is applied to both arms, restricted to the shared cluster subset. Resampling the arms independently
inflates the variance of the difference and is wrong for paired data. There are two pairing modes
and they are separate rules, not one blanket rule: arms that share a fold assignment (conservation
versus embedding delta) pair within identical folds; arms from different CV partitions — which is
only the gene-split-minus-family-split gap — resample families and then re-score each arm under its
own partition. Written without distinguishing the two, the cross-partition case is silently
implemented as the same-fold path and is wrong.

**The split gap uses the bootstrap, not a permutation test.** Under a shuffled-label null both
gene-split and family-split collapse to the floor, so the null gap is centred near zero by
construction. Such a test asks whether leakage exists — already answered by the leakage fraction —
and says nothing about the observed gap's sampling variability.

**Addendum, added 2026-08-10 — distance/graph statistics use a subsample, not a bootstrap.**
`family_clustering.py`'s k-NN family-purity and within/between pairwise-distance-ratio CIs
(Section 4 of `ESM2_REPORT.md`, exploratory, not in the C1–C4 confirmatory set) are not additive
statistics like F1 or AUROC — they depend on the neighbor graph or pairwise distances between
points. A standard with-replacement cluster bootstrap duplicates a family's rows whenever that
family is drawn more than once, and the duplicate sits at distance exactly 0 from itself, which
inflates same-family neighbor purity and deflates the within-family mean distance (confirmed
empirically before this fix: `ci_low` sometimes exceeded the point estimate). These two CIs use
`cluster_subsample_ci` instead — an m-out-of-n subsample of families **without** replacement
(`subsample_frac=0.632`, matching the expected unique-cluster fraction of a same-size
with-replacement bootstrap), which never duplicates a point. Every other family-split CI in the
project (F1, AUROC, macro_f1, Spearman rho) keeps the ordinary `cluster_bootstrap_ci`, where draw
multiplicity is a correct resampling weight, not an artifact.

## R7.4 — Rare-class intervals

DN (≈ 9%, ~150–170 genes) and GOF (≈ 15%) sit in the regime where percentile bootstrap undercovers
for a bounded metric near its boundary with few clusters.

- One-vs-rest AUROC for the rare classes uses the same percentile cluster bootstrap as every other
  interval in the project. No bias correction is applied: over ~150 clusters a correction is itself
  noisy, so it would trade one inaccuracy for another while implying a precision the panel does not
  have.
- Rare-class intervals are **flagged as the least trustworthy in their table.** DN intervals are
  indicative, not authoritative, and no confirmatory claim rests on them — per-class AUROCs are
  exploratory under R7.2.
- The existing degenerate-fold suppression guard is retained.

## R7.5 — Permutation budget

**Amendment 2026-08-12, before the run — the two headline features get different permutation
tests.** The budget and the seed are unchanged; what changes is the statistic for `delta_mean` and
how its null is built.

`wt_only_mean` is unchanged: refit the probe once per permutation and score macro-F1, at 1,000
permutations. Its p-value is the load-bearing one and run6 left it unresolved at the 1/(N+1) floor.

`delta_mean` moves to macro one-vs-rest AUROC scored against the cached out-of-fold predictions,
which costs no refits. The reason is sensitivity, not cost. Macro-F1 is a weak instrument for C1: a
probe sitting at the chance floor predicts the majority class almost everywhere, so macro-F1 stays
near the floor whether or not the ranking carries a small amount of mechanism signal — in run6
`delta_mean` scored 0.289 against a shuffled-label mean of 0.319, giving p = 1.0. A refutation test
that cannot fire is not a test. AUROC reads the ranking directly and can detect an effect macro-F1
would miss, which is what C1 needs, since a significant p refutes C1 and a non-significant p
confirms nothing.

This is a different null, recorded here rather than discovered later: permuting labels against
fixed predictions conditions on the model that was fit to the real labels and asks whether its
held-out predictions carry label information, where the refit version rebuilds the model under each
shuffle. The predictions are out-of-fold, so the test is not circular, but every report stating this
p-value must say which of the two produced it. The emitted result records the statistic, the null
type, the permutation unit and the width of the null alongside the p-value.

**Amendment 2026-08-12 — the permutation unit matches the clustering unit.** Both tests score a
family-split metric, so both now permute at the family level, the same unit R7.3 already requires
for the interval. Shuffling one label per gene, which is what run6 did, breaks the label structure
that homologous genes share: if related genes tend to carry the same mechanism, a gene-level shuffle
builds a null narrower than the truth and every p-value comes out smaller than it deserves. This
bites the no-refit test hardest, since nothing else in that path absorbs the mismatch.

Families are permuted by swapping whole label blocks between families with the same gene count,
rather than by drawing one label per family. The block swap preserves the observed degree of
within-family label mixing; assigning one label per family would make every family homogeneous,
which is more clustered than the data really is, and would widen the null instead — costing the
test the power it needs, since refuting C1 is the only thing this p-value can do. Blocks are only
exchangeable with blocks of the same length, so a family whose gene count is unique in the data has
no partner and keeps its own labels; that count is reported next to the p-value so it is visible how
much of the data moved. Most families are single genes, so most of the mass permutes freely.

The refit cost falls from 2,000 to 1,000. It was never the 4,000 quoted in the run6-era planning
text: only the family split is permuted, not both splits.

- **Linear probe: 1,000 permutations**, seed 0. The headline claim (`delta_mean` at the chance
  floor) is a linear-probe claim, so the load-bearing test is fully resolved.
- **The MLP is not permutation-tested.** No claim rests on an MLP permutation p-value, and its
  refits are the expensive tail; the linear probe is the only permutation test in run_biorxiv.
- **No p-value is reported at its resolution floor of 1/(N+1)** — that is an unresolved bound, not
  a measurement. The run6 `wt_only_mean` p = 0.0099 at 200 permutations is exactly this case and is
  not carried forward.
- Seed 0 only: a permutation test constructs its own null by shuffling, so running it across five
  seeds mostly re-measures the fold jitter run_biorxiv exists to replace.

## R7.6 — Calibration

The probes are uncalibrated and measure **discrimination only**. Reported scores are not risk
estimates. This is stated in every probe report rather than fixed, because every claim in the paper
is a discrimination claim.

## R7.7 — What would change the conclusions

Recorded in advance so that "the CIs corroborated the point estimates" is a falsifiable statement
rather than an expectation:

- **C1 overturned** if `delta_mean`'s family-split CI excludes the measured floor from above and
  its permutation p is significant. The mechanism null would not survive.
- **C1 recorded as not adjudicated** if its CI upper bound sits above floor + 0.05 without the
  permutation p being significant — the family-split panel would be underpowered to tell a real
  effect of the pre-registered size from none, which is not the same as the null holding.
- **C2 overturned** if the split-gap CI spans zero — the homology-leakage account would be
  unsupported at this sample size.
- **C3 overturned** if the pathogenicity CI includes 0.85; the positive control would no longer
  license the dissociation, and the whole paper weakens.

run_biorxiv changes error bars, not point estimates. Any point estimate that moves materially from run6
is either a bug introduced by the wiring or a finding that needs explaining; `scripts/compare_runs.py`
flags these, and each flagged movement is explained rather than silently adopted.

