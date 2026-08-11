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

`m1_threshold` = 0.430 is the ESM-2 family-split floor (0.380) **plus a pre-registered 0.05
effect-size requirement**. M1 and M2 are recorded in
`results/run6/esm3_mechanism/merged/summary.json` as `family-split F1 > 0.430`, so their margins
below are against 0.430 — not against the bare floor. Stated against the bare floor the lifts are
much larger (seq +0.058, seq_struct +0.072) and the CI question becomes trivial; the two framings
have opposite robustness and must not be interchanged.

| Gate | Criterion as recorded | Run6 value | Margin | Run6 verdict |
|---|---|---|---|---|
| M1 | seq_struct family-split F1 > 0.430 | 0.4528 | +0.023 | pass |
| M2 | seq family-split F1 > 0.430 (scale alone) | 0.4384 | +0.008 | pass |
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

The paper runs gate comparisons across seven reports. Correcting one table while leaving the rest
uncorrected is the weaker half of a defence, so the confirmatory set is enumerated **before** the
run and multiplicity control is applied across it.

### Confirmatory claims (six)

| # | Claim | Instrument |
|---|---|---|
| C1 | The mechanism delta sits at the measured chance floor under family-split | CI straddles the floor + permutation p |
| C2 | The absolute-embedding gene→family gap is non-zero (homology leakage exists) | paired bootstrap on the split gap |
| C3 | Pathogenicity clears AUROC 0.85 family-split (positive control) | CI excludes 0.85 |
| C4 | Conservation alone matches or beats the embedding delta for pathogenicity | paired bootstrap (K1/K2) |
| C5 | ESM-3 scale lifts the mechanism floor by at least the pre-registered 0.05 (M2: seq > 0.430) | paired bootstrap |
| C6 | The mechanism null is stable across homology partitions | family / clan / MMseqs2 robustness panel |

C1 and C3 are the load-bearing pair (the dissociation), C2 is the leakage account, and C6 makes C1
partition-independent. Those four are the paper. C4 and C5 are the characterisation payoff; if the
confirmatory set must be trimmed, trim from C4/C5, never C1–C3.

**Benjamini-Hochberg FDR is applied across these six claims only.** Raw and adjusted values are
both reported so a reader can see the correction rather than only its result.

### Exploratory (labelled, not corrected)

Per-class AUROCs, the 28-family within-family table and its per-family cells, the biochemistry R²,
the magnitude/direction decomposition, per-feature leakage fractions, and all descriptive geometry.
These are labelled exploratory in their reports. They are **not** FDR-corrected — correcting an
exploratory screen implies it was confirmatory.

**Failing gates are excluded from the correction set.** M3 ("structure adds nothing", a stated
headline in `ESM2_REPORT.md` §6) and K2 are reported under R7.1's underpowered clause and take no
part in the BH-FDR set, because correcting them would imply they were positive findings under test.

## R7.3 — Resampling unit

The resampling unit matches the unit the split holds out.

| Metric | Resample |
|---|---|
| Gene-split | genes |
| Family-split | **families** |
| Clan-split | clans |
| MMseqs2 cluster-split | clusters |
| Gene-split minus family-split gap | **families** (the coarser of the two arms) |

The split-gap case resamples families because its family-split arm's variance is only correct
under family resampling; a gene-resampled gap understates it. A family resample induces a valid
gene resample, but not the reverse. The gene-resampled interval is reported alongside as a
labelled sensitivity check.

**Expected and pre-registered:** family-split CIs will be visibly wider than gene-split ones.
There are 1,134 families but 833 are singletons, so the effective cluster count is far below the
gene count. That widening is the correct answer, not an artifact to tune away. The effective
cluster count is reported next to every family-split interval.

## R7.4 — Rare-class intervals

DN (≈ 9%, ~150–170 genes) and GOF (≈ 15%) sit in the regime where percentile bootstrap undercovers
for a bounded metric near its boundary with few clusters.

- One-vs-rest AUROC for the rare classes uses **BCa** wherever the acceleration estimate is
  computable.
- Rare-class intervals are **flagged as the least trustworthy in their table regardless of
  method** — with a jackknife over ~150 clusters, BCa's own correction is noisy. DN intervals are
  indicative, not authoritative.
- The existing degenerate-fold suppression guard is retained; BCa does not replace it.

## R7.5 — Permutation budget

- **Linear probe: 1,000 permutations**, seed 0. The headline claim (`delta_mean` at the chance
  floor) is a linear-probe claim, so the load-bearing test is fully resolved.
- **MLP: N set by the measured per-refit cost, stated explicitly** wherever its p-value appears.
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
- **C2 overturned** if the split-gap CI spans zero — the homology-leakage account would be
  unsupported at this sample size.
- **C3 overturned** if the pathogenicity CI includes 0.85; the positive control would no longer
  license the dissociation, and the whole paper weakens.
- **C5 restated** as *not distinguishable* if M2's paired CI spans zero, which the +0.008 margin
  makes plausible. The scale claim becomes "consistent in direction, not established".
- **C6 overturned** if the mechanism null does not hold under clan or MMseqs2 partitions — the
  result would be an artifact of the Pfam family definition.

run_biorxiv changes error bars, not point estimates. Any point estimate that moves materially from run6
is either a bug introduced by the wiring or a finding that needs explaining; `scripts/compare_runs.py`
flags these, and each flagged movement is explained rather than silently adopted.
