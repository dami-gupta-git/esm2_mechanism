# run_biorxiv pre-registration — inferential statistics (added 2026-07-22)

**Written before run_biorxiv executes.** This document governs run_biorxiv only — the original
(run0-era) pre-registration and its outcomes live in `docs/EXPERIMENT.md`. run_biorxiv re-scores
run6's science with dependency-aware error bars; the experiments, hypotheses, and gates are
unchanged.

These rules exist because run_biorxiv will attach confidence intervals to gates that currently
pass or fail on point estimates alone. Without a reading fixed in advance, an interval that lands
awkwardly invites a framing chosen after the fact. The run6 point estimates are recorded here so
the rules cannot be retro-fitted to the run_biorxiv intervals.

Methodology: `reports/run6/STATS_PLAN.md`. Change list: `PLAN_2026-07-20.md`. Execution:
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

**`m1_threshold` is a rule, not a number.** It is the ESM-2 family-split floor **plus a
pre-registered 0.05 effect-size requirement**, and `esm3_mechanism.py` derives it at run time:
`esm2_family_floor()` reads the floor from that run's own `nonlinear_results_seed*.json` — the MLP,
`delta_mean`, family-split probe averaged over all five seeds — and raises rather than substituting
if any seed is missing. What is pre-registered is the rule and the source, so the threshold moves
with the floor the run measures.

In run6 that evaluated to 0.430 (floor 0.380), which is the value the margins below are stated
against. The floor is on record elsewhere in the run6 reports as 0.415 and 0.418; those are
report-text drift rather than competing definitions, and the run_biorxiv floor comes from the single
source named above. **If the measured floor lands near 0.415, the threshold becomes ~0.465 and M2
fails.** That is the pre-registered rule operating, not a moved goalpost, and it is the outcome C5
is most exposed to. `scripts/compare_runs.py` flags the floor's movement so it is read rather than
absorbed.

Margins are stated against the threshold, never against the bare floor. Against the bare floor the
lifts look much larger (seq +0.058, seq_struct +0.072) and the CI question becomes trivial; the two
framings have opposite robustness and must not be interchanged.

| Gate | Criterion as recorded | Run6 value | Margin | Run6 verdict |
|---|---|---|---|---|
| M1 | seq_struct family-split F1 > `m1_threshold` (0.430 in run6) | 0.4528 | +0.023 | pass |
| M2 | seq family-split F1 > `m1_threshold` (0.430 in run6; scale alone) | 0.4384 | +0.008 | pass |
| M3 | seq_struct − seq > 0.030 | +0.0143 | −0.016 | fail |
| K1 | conservation alone AUROC > 0.85 | 0.891 | +0.041 | pass |
| K2 | conservation + delta improves over conservation by > 0.02 | +0.0023 | −0.018 | fail |
| K2b | conservation + delta improves over delta alone | +0.0345 | — | descriptive (no threshold) |
| H2 | stability random→family rho drop < 0.10 (LEAKY) | — | descriptive | descriptive |
| Contrastive | contrastive k-NN > raw-delta k-NN | +0.041 | — | pass |

**Exposure is concentrated in M2 and the contrastive gate** — the only two claiming a pass on a
margin thinner than a seed of spread. M3 and K2 already fail, so a CI spanning zero *reinforces*
those readings; they are the cases where R7.1's underpowered clause applies rather than
threatening the conclusion.

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
| C5 | ESM-3 scale lifts the mechanism floor above the pre-registered gate (M2: seq > `m1_threshold`, the measured floor + 0.05) | paired bootstrap |

C1 and C3 are the load-bearing pair (the dissociation) and C2 is the leakage account. Those three
are the paper. C4 and C5 are the characterisation payoff; if the confirmatory set must be trimmed,
trim from C4/C5, never C1–C3.

**C1 is a null claim and is adjudicated as one.** A confidence interval that straddles the chance
floor is not evidence that the score sits at the floor — an interval wide enough to straddle
anything straddles the floor too. C1 is affirmed only if the family-split CI's upper bound falls
below the measured floor plus 0.05, the same margin the M2 gate uses, so both are read on one
scale. If the interval is wider than that, C1 is recorded as **not adjudicated** — underpowered to
separate a real effect from none — never as confirmed. The permutation test runs on C1 but in one
direction only: a significant p refutes C1, and a non-significant p does not confirm it.

**No multiplicity correction is applied.** No verdict in a set this size turns on one: C2 and C3
clear their thresholds by wide margins, C5 already reads as *not distinguishable*, and C1 is a null
claim, where raising the bar for rejection would make the claim easier to assert rather than harder.
Enumerating the set in advance is the safeguard.

### Exploratory (labelled, not corrected)

Per-class AUROCs, the 28-family within-family table and its per-family cells, the biochemistry R²,
the magnitude/direction decomposition, per-feature leakage fractions, and all descriptive geometry.
These are labelled exploratory in their reports and assert nothing the paper relies on. The
28-family within-family table is included here: it is restated as an exploratory screen rather than
corrected, since correcting a screen implies it was confirmatory.

**Failing gates make no confirmatory claim.** M3 ("structure adds nothing", a stated headline in
`ESM2_REPORT.md` §6) and K2 are reported under R7.1's underpowered clause and are not confirmatory
items, because reporting them as such would imply they were positive findings under test.

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

The split-gap case resamples families because its family-split arm's variance is only correct
under family resampling; a gene-resampled gap understates it. A family resample induces a valid
gene resample, but not the reverse. The gene-resampled interval is reported alongside as a
labelled sensitivity check.

**Expected and pre-registered:** family-split CIs will be visibly wider than gene-split ones.
There are 1,134 families but 833 are singletons, so the effective cluster count is far below the
gene count. That widening is the correct answer, not an artifact to tune away. The effective
cluster count is reported next to every family-split interval.

**Addendum, added 2026-08-10 — distance/graph statistics use a subsample, not a bootstrap.**
`family_clustering.py`'s k-NN family-purity and within/between pairwise-distance-ratio CIs
(Section 4 of `ESM2_REPORT.md`, exploratory, not in the C1–C5 confirmatory set) are not additive
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
- **C5 restated** as *not distinguishable* if M2's paired CI spans zero, which the +0.008 margin
  makes plausible. The scale claim becomes "consistent in direction, not established".
- **C5 fails outright** if the run's measured floor comes back materially above run6's 0.380, since
  the threshold is the floor + 0.05 and ESM-3's seq arm sits at 0.438. A floor near 0.415 puts the
  gate at ~0.465 and M2 fails on its point estimate, which R7.1 reports as failure regardless of the
  CI. The floor is the run's own measurement, so this is the rule operating rather than a threshold
  chosen after the fact.

run_biorxiv changes error bars, not point estimates. Any point estimate that moves materially from run6
is either a bug introduced by the wiring or a finding that needs explaining; `scripts/compare_runs.py`
flags these, and each flagged movement is explained rather than silently adopted.
