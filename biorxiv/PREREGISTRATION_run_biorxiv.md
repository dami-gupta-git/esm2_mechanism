# run_biorxiv pre-registration — inferential statistics

**Written and committed before run_biorxiv executes**, so that no rule here can be fitted to a
result. Governs run_biorxiv only. Every threshold is either an absolute number fixed here or a
quantity this run measures for itself; no threshold is carried in from another run, and no
threshold is derived from a probe this document then judges.

Methodology: `reports/run6/STATS_PLAN.md`. Step-by-step execution: `RUNBOOK_biorxiv.md` (section
numbers below refer to that document).

---

## Part 1 — Rules that apply to every result

### 1.1 How a gate's verdict is decided

Every difference gate is evaluated against a paired cluster-bootstrap 95% CI. A level gate uses a
cluster-bootstrap interval on that level. Neither type is evaluated from a point estimate alone.

> **Affirmed** — the point estimate clears the threshold and the relevant 95% CI excludes the
> boundary: zero for a difference gate, or the stated threshold for a level gate.
>
> **Not distinguishable** — the point estimate clears the threshold but the CI includes the
> relevant boundary. Neither a pass nor a refutation.
>
> **Failed** — point estimate does not clear the threshold, regardless of the CI.
>
> **Underpowered** — a failed gate whose CI also spans the pre-registered threshold is reported as
> underpowered to detect an effect of that size, not as evidence of no effect.

### 1.2 What gets resampled

The resample unit matches the unit the split holds out — never resample something finer than what
was held out. *(Runbook §0.2–0.3 wires this; §4, §5, §6, §7 all consume it.)*

| Metric | Resample unit |
|---|---|
| Gene-split | genes |
| Family-split | **families** |
| Gene-split minus family-split gap | **families** (the coarser of the two arms) |
| Permutation null for a family-split metric | **families**, whole label blocks swapped (§2.1 below) |

Family-split CIs are expected to be visibly wider than gene-split ones — of 1,134 families, 833
are singletons, so the effective cluster count is far below the gene count. That widening is the
correct answer, not an artifact to tune away, and the effective cluster count is reported next to
every family-split interval.

**Pairing.** A paired difference is resampled once per replicate and that same draw is applied to
both arms, restricted to the shared cluster subset — resampling arms independently inflates the
variance of the difference. Two modes: arms sharing a fold assignment (e.g. conservation vs.
embedding delta) pair within identical folds; arms from different CV partitions (only the
gene-split-minus-family-split gap) resample families once and re-score each arm under its own
partition.

**Exceptions to the standard cluster bootstrap:**
- The gene-split-minus-family-split gap is tested by bootstrap, not permutation — a shuffled-label
  null collapses both splits to the floor by construction, so it can only ask whether leakage
  exists (already answered separately), not how variable the observed gap is.
- The k-NN family-purity and pairwise-distance-ratio diagnostics in `family_clustering.py`
  (exploratory, Runbook §4.3) depend on the neighbor graph itself, not an additive statistic like
  F1 or AUROC. A with-replacement bootstrap duplicates a family's rows, and a duplicate sits at
  distance 0 from itself — this measurably inflates purity and deflates within-family distance
  (`ci_low` was observed exceeding the point estimate before the fix). These two CIs use an
  m-out-of-n subsample without replacement instead (`subsample_frac=0.632`). Every other
  family-split CI in the project keeps the ordinary bootstrap.

### 1.3 Rare classes get a caveat, not a fix

**Fold-aware scoring and the rare-class rule.** Every ranking metric and every macro-F1 in this
run is computed inside each cross-validation fold and averaged across folds. Folds are fitted
independently and their probabilities are on their own scales, so a metric that ranks the
concatenation of all folds is not the quantity these rules intend.

Under bootstrap resampling a fold can lose a rare class entirely, and DN is the case where that
matters. Every fold must be able to score every class the metric averages over; a resample where
any fold cannot is discarded whole rather than scored over the folds that survive, because a draw
averaging over a different set of folds or classes is estimating a different quantity. The number
of discarded resamples is recorded next to every interval.

The expected discard rate is far below one percent, because every family-split fold already
contains DN variants on every seed. A materially higher rate is treated as a fault in the
resampling unit or the fold construction and investigated, not absorbed into the result.

DN (≈9%, ~150–170 genes) and GOF (≈15%) sit in a regime where a percentile bootstrap undercovers
for a bounded metric near its boundary with few clusters. No bias correction is applied — over
~150 clusters a correction is itself noisy, trading one inaccuracy for another while implying
precision the panel doesn't have. Instead, their one-vs-rest AUROC intervals are flagged as the
least trustworthy in their table: indicative, not authoritative, and no confirmatory claim rests
on them. *(Runbook §4.1/§4.2, per-class breakdown.)*

### 1.4 Calibration

All probes are uncalibrated and measure discrimination only; reported scores are not risk
estimates. Stated in every probe report, since every claim in the paper is a discrimination claim.

### 1.5 No multiplicity correction

No verdict in the confirmatory set (Part 2) turns on a borderline p or CI. The stated thresholds and
null-claim rules govern each verdict. Enumerating the set in advance (Part 2) is the safeguard.

---

## Part 2 — The confirmatory claims

Enumerated before the run so a result cannot be selected as load-bearing after the fact. Everything
not listed here is either a stability control (Part 3) or exploratory (Part 4) and asserts nothing
the paper relies on.

| # | Claim | Instrument | Runbook |
|---|---|---|---|
| 2A | The mechanism delta sits at the measured chance floor, family-split | CI upper bound below floor + 0.05; permutation p as a refutation-only test | §4 |
| 2B | The absolute-embedding gene→family gap is non-zero (homology leakage exists) | paired bootstrap on the split gap | §4.5 |
| 2C | Pathogenicity clears AUROC 0.85, family-split (positive control) | CI excludes 0.85 | §5 |
| 2D | Conservation alone clears AUROC 0.85 for pathogenicity | family-cluster bootstrap CI | §6.7 |
| 2E | Adding the embedding delta to conservation improves AUROC by at least 0.02 | paired bootstrap | §6.7 |
| 2F | Enzyme type classification clears family-split LogReg macro-F1 0.70 | cluster-bootstrap CI | §8 |
| 2G | Enzyme family-split F1 substantially exceeds the mechanism family-split floor | paired cluster-bootstrap CI | §8 |
| 2H | The enzyme signal is linearly separable: MLP does not substantially outperform LogReg under family-split | paired cluster-bootstrap CI | §8 |

Claims 2A and 2C form the load-bearing dissociation, and 2B provides the leakage account. Claims
2D–2E address characterisation, and 2F–2H address task specificity. If the set must be trimmed,
retain 2A–2C.

### 2A — mechanism null (Runbook §4, permutation §4.6)

**Two quantities, named separately.** One is the score a no-information predictor achieves. The
other is the score the run's own strongest mutation-only probe achieves. Only the first is a floor.

| Name | Definition | Source |
|---|---|---|
| Measured chance floor | Macro-F1 of a majority-class predictor under the split in question, averaged over the run's five seeds | The run's own naive baseline result file |
| Nonlinear delta reference | Family-split macro-F1 of the run's nonlinear `delta_mean` probe, averaged over five seeds | The run's own nonlinear probe results |

The measured chance floor is what "floor" means throughout this document. It is read live from the
run's own result file rather than carried between runs, and it does not move when the probes are
refit, because a majority-class prediction is decided per row and no fold-scale comparison enters
it. `scripts/compare_runs.py` flags its movement between runs.

The nonlinear delta reference is a measurement that carries signal, and it sits consistently above
the measured chance floor. It is never described as chance, here or in the reports or the paper.

**Verdict.** A CI straddling the floor is not evidence the score sits at the floor — a wide enough
interval straddles anything. 2A is affirmed only if the family-split CI's upper bound for the
linear `delta_mean` probe falls below the measured chance floor under the family split plus 0.05;
otherwise it is **not adjudicated**, never confirmed. The permutation test runs in one direction
only: a significant p refutes 2A, a non-significant p does not confirm it.

The threshold is not derived from a probe. A threshold set by the nonlinear probe's own score would
rise whenever that probe scored higher, making the claim easier to affirm rather than harder, and
would move whenever the scoring code changed. The nonlinear delta reference is reported beside the
threshold as a named comparator so a reader can see how far the strongest mutation-only probe sits
above the floor, and it sets no bar.

The threshold's value is filled in from the re-run's own naive baseline, recorded before the claim
is judged against it, and never carried over from earlier results.

**Permutation test — the two headline features are tested differently:**

| Feature | Statistic | Refits | Permutations | Why this choice |
|---|---|---|---|---|
| `wt_only_mean` | macro-F1 | yes, once per permutation | 1,000, all five seeds | Load-bearing p-value, so it gets the rigorous variant and enough permutations to resolve below the 1/(N+1) floor |
| `delta_mean` | macro one-vs-rest AUROC, cached out-of-fold predictions | none | 1,000, all five seeds | Sensitivity, not cost — macro-F1 stays near the floor even when real ranking signal is present, so a macro-F1 permutation test cannot fire; AUROC reads the ranking directly |

There are two ways to run a shuffle test:

- **Rigorous way** (`wt_only_mean`): scramble the labels, retrain the model from scratch, see how
  well it does — repeated 1,000 times.
- **Cheaper way** (`delta_mean`): scramble the labels but keep the predictions the model already
  made when trained on the real labels, and check whether those predictions still line up with the
  scrambled version. This isn't circular — the predictions being checked are on data the model
  never saw during training — but it is the less rigorous of the two, so any report citing this
  p-value must say which variant produced it.

Both shuffle whole **families**, not individual genes (§1.2):

- Genes in the same family tend to share a mechanism label, so they aren't independent — shuffling
  gene-by-gene breaks that clustering and produces a "random" baseline that's more spread out than
  reality, making a real result look more impressive than it should.
- This matters most for the no-refit test, which has no other safeguard against it.

The family shuffle works by swapping each family's whole label block with another family of the
same gene count, not by assigning one random label per family:

- Assigning one label per family would make every family internally uniform — more clustered than
  the real data — and weaken the test's ability to catch a real effect.
- A family with no size-match keeps its own real labels; how many families this affects is reported
  next to the p-value.
- Most families are a single gene, so almost all of them shuffle freely.

**Seeds and how they combine.** The test runs on all five seeds and the full distribution of
p-values is reported. Because the test is refutation-only for 2A, the refutation fires when at
least three of the five seeds return a p-value below 0.05; a minority of significant seeds is
reported as a split result and refutes nothing. A single-seed test is not used: the seeds differ in
their gene-to-family gap, so one seed can understate the effect and invites the objection that the
seed was chosen.

Scope and cost:

- Only the linear probe gets this test — the headline claim is a linear-probe result, and no
  conclusion depends on the MLP's permutation p-value, so the MLP isn't tested this way.
- Refit cost: 1,000 per seed. Only the family split is permuted, not both splits.

Resolution limit:

- With 1,000 shuffles, the smallest measurable p-value is about 1 in 1,000. A p-value landing
  exactly there means only "nothing more extreme was detectable at this resolution," not a precise
  measurement, and is reported as unresolved rather than as a result.

**Label-heterogeneity threat, named and cited.** Badonyi & Marsh 2025
(`papers/mechanism_2025.pdf`, bioRxiv 2025.03.13.642984) report 43% of multi-phenotype dominant
genes and 49% of mixed-inheritance genes carry both LOF and non-LOF mechanisms. This project
assigns one label per gene, so some variants are mislabelled by construction — a citable
alternative explanation for 2A: the delta sits at the floor because labels are noisy, not because
the embedding lacks signal. No CI addresses this, since the threat is to what the labels mean, not
sample size. Answered by Tasks 2d and 8 in [`FOLLOWUP_biorxiv.md`](FOLLOWUP_biorxiv.md) (does the
null survive on cleanly-labelled genes; how far does realistic label noise move a working probe),
which 2A's statement cites rather than asserting the labels are adequate.

**Out of scope.** The null is measured under the Pfam family partition only. Whether it holds under
a coarser partition (Pfam clan) or a sequence-identity clustering is follow-up work; the paper
claims neither independence from the Pfam family definition nor how leakage varies with partition
strictness.

**Would overturn 2A:** family-split CI excludes the measured floor from above **and** the
permutation p is significant.
**Would leave 2A not adjudicated:** CI upper bound above floor + 0.05 without a significant p —
underpowered to tell a real effect of the pre-registered size from none, which is not the same as
the null holding.

**Checklist:**
- [ ] The measured chance floor is read from this run's own majority-class baseline under the
      family split, not carried over from an earlier run and not derived from any probe.
- [ ] The nonlinear delta reference is reported beside the threshold and is nowhere called a floor
      or called chance.
- [ ] Verdict recorded is "affirmed" or "not adjudicated" only, never "confirmed" outright.
- [ ] "Affirmed" is used only if the linear `delta_mean` family-split CI's upper bound is below the
      measured chance floor + 0.05.
- [ ] `delta_mean`'s reported p-value states which test variant (refit macro-F1 vs. no-refit AUROC)
      produced it.
- [ ] Both `wt_only_mean` and `delta_mean` permutation nulls are built by shuffling at the family
      level, not the gene level.
- [ ] The family block-swap only exchanges blocks of equal gene count; the count of families with
      no size-match (kept their own labels) is reported next to the p-value.
- [ ] No p-value is reported sitting exactly at the 1/(N+1) resolution floor without being flagged
      as unresolved.
- [ ] The MLP has no permutation p-value attached.
- [ ] The permutation test is reported for all five seeds, and a refutation is claimed only when at
      least three of them return p below 0.05.

### 2B — homology leakage (Runbook §4.5)

Paired bootstrap on the gene-split-minus-family-split gap, resampled by family (§1.2).

**What adjudicates.** Three choices decide this verdict and each of them changes the answer, so all
three are fixed here rather than left to whoever reads the results.

| Choice | Decision |
|---|---|
| Adjudicating feature | `wt_only_mean`, the headline absolute-embedding feature |
| Adjudicating quantity | The paired gene-split-minus-family-split macro-F1 gap |
| Seed combination | The gap's interval must exclude zero on at least three of the five seeds |

**Would overturn 2B:** the gap's interval spans zero on three or more of the five seeds — the
leakage account would be unsupported at this sample size.

The leakage fraction does not adjudicate 2B. It is a derived ratio whose denominator is itself an
estimate, and its interval spans zero on the merged dataset. It is reported as a descriptive
quantity with its interval alongside the gap.

The single-source subset, where the curation-source confound is absent, is reported as a named
secondary analysis. It is the cleaner design, but it was not pre-specified as the adjudicator and
does not become one.

**Checklist:**
- [ ] The split-gap CI is resampled by family, using one shared draw applied to both arms (§1.2
      pairing), not two independent bootstraps.
- [ ] The gap is tested with the bootstrap, not a permutation test.
- [ ] The verdict cites `wt_only_mean`, the paired gap, and the count of seeds whose interval
      excludes zero.
- [ ] The leakage fraction is reported with its interval and is not used to adjudicate.

### 2C — pathogenicity positive control (Runbook §5)

**Post-result specification amendment, 2026-08-19.** The initial results were inspected before
these two omitted rules were recorded. The repaired run uses seed 0 as the inferential unit: its
family-split out-of-fold predictions supply both the point estimate and the cluster-bootstrap CI.
The five-seed mean remains descriptive and is not paired with the seed-0 interval. ClinVar records
encoding the same gene, protein position, wildtype residue, and mutant residue are deduplicated
before per-gene balancing. Their ClinVar identifiers are retained as provenance, and a
protein-level substitution carrying conflicting labels is an error. This amendment resolves the
two omissions for the repaired run; it is not prospective preregistration.

CI must exclude 0.85. Resampled by family, not gene — this experiment's classes are balanced by
construction, but genes still cluster into families, so the resampling unit rule (§1.2) still
applies; the calibration caveat (§1.4) still governs how the result is described. Each seed
reshuffles the CV fold assignment, so one seed's predictions form the coherent unit to resample.
**Would overturn 2C:** the CI includes 0.85 — the positive control would no longer license the
dissociation, and the whole paper weakens.

**Checklist:**
- [ ] CI is resampled by family, not gene.
- [ ] CI is computed from seed-0 OOF predictions.
- [ ] Seed-0 point estimate and seed-0 CI are reported together; the five-seed mean is labelled
      descriptive.
- [ ] Repeated protein-level substitutions are deduplicated before balancing, with the removed
      count and retained ClinVar identifiers recorded.
- [ ] Report states the probe measures discrimination only, not a calibrated risk estimate (§1.4).
- [ ] Verdict recorded as overturned only if the CI includes 0.85.

### 2D–2E — conservation vs. embedding delta (Runbook §6.7)

**Post-result specification amendment, 2026-08-19.** The earlier Section 6 result was inspected
before the inferential seed was stated. The repaired run uses seed 0 for inference: the seed-0
held-out-fold AUROC supplies the point estimate and the family-cluster bootstrap interval. The
five-seed mean and each seed's fold mean are saved as descriptive results. Claim 2D uses a
single-arm family-cluster interval. Claim 2E uses a paired family-cluster interval with the same
fold assignment and resample applied to both arms. This amendment records the omitted seed rule; it
does not change either threshold.

| Gate | Criterion |
|---|---|
| 2D | Conservation alone reaches AUROC of at least 0.85 |
| 2E | Conservation plus the embedding delta improves on conservation alone by at least 0.02 |
| Descriptive | Conservation plus the delta compared against the delta alone, reported without a threshold |

Margins are stated against the threshold, never against the bare chance floor. Measured against the
floor the lifts look much larger and the interval question becomes trivial, so the two framings
must not be interchanged. The underpowered clause (§1.1) governs 2E however its interval lands, and
a failed gate makes no positive claim.

**Checklist:**
- [ ] Margins reported are stated against the threshold (e.g. 0.85, 0.02), never against the bare
      chance floor.
- [ ] Seed-0 point estimates and seed-0 intervals are reported together; the five-seed means are
      labelled descriptive and the individual seed fold means are retained in the result file.
- [ ] 2E's CI, however it lands, is reported under the underpowered clause (§1.1) — not relabeled
      as a confirmatory pass.
- [ ] The 2D interval is a one-arm family-cluster bootstrap. The 2E gap interval is a paired
      family-cluster bootstrap with same-fold pairing (§1.2).

### 2F–2H — enzyme type classification (Runbook §8)

A positive control using a wildtype-sequence property: classifying each gene as kinase, protease,
oxidoreductase, or non-enzyme from its WT mean-pooled ESM-2 embedding. Enzyme class is strongly
associated with protein fold, so ESM-2's known Pfam clustering should help — making this a direct
test of whether the mechanism null (2A) is a property of the task, not a failure of the pipeline.
Governed by the same verdict rule (§1.1) and resampling rule (§1.2) as the other experiments.
Decision rules from `docs/plans/plan_enzyme_classification.md`:

| # | Criterion | Interpretation if met |
|---|---|---|
| 2F | Family-split LogReg macro-F1 at or above 0.70 | Enzyme class is strongly encoded in ESM-2 wildtype embeddings |
| 2G | Enzyme family-split F1 substantially exceeds the mechanism family-split floor | The mechanism null is task-specific, not a probe or data failure |
| 2H | MLP does not substantially outperform LogReg under family-split (\|ΔF1\| below 0.05) | A linear readout is sufficient, paralleling pathogenicity |

2G is the central claim: the same pipeline that shows mechanism at floor achieves strong enzyme
classification, so the mechanism null reflects what ESM-2 encodes, not a methodological ceiling.
2F sets an absolute bar. 2H tests whether the signal is linearly separable (as for
pathogenicity) or requires nonlinear probes (as for stability).

The mechanism reference F1 in 2G is read from section 4's aggregate result for this run, never
hardcoded, on the same principle as the measured chance floor in 2A.

CIs are cluster-bootstrap on seed-0 family-split OOF predictions, resampled by family (§1.2). The
rare-class caveat (§1.3) applies to protease, the smallest class here. A proteome-features baseline runs
alongside as a negative control — enzyme class is a structural property, not a population-genetics
one, so proteome features should be near chance.

**Checklist:**
- [ ] Mechanism reference F1 is read from this run's own aggregate, never hardcoded.
- [ ] CIs are resampled by family, not gene.
- [ ] Protease per-class AUROC is flagged under the rare-class caveat (§1.3).

---

## Part 3 — Stability controls (Runbook §7)

A second positive control with its own gates, 3A–3D, governed by the same verdict rule (§1.1) and
resampling rule (§1.2). These gates are not part of the confirmatory set.

---

## Part 4 — Exploratory analyses

Per-class AUROCs, the 28-family within-family table and its per-family cells, the biochemistry R²,
the magnitude/direction decomposition, per-feature leakage fractions, and all descriptive geometry.
Labelled exploratory in their reports and assert nothing the paper relies on. The 28-family table
is restated as an exploratory screen rather than corrected, since correcting a screen implies it
was confirmatory.

---

## Part 5 — What would change the conclusions

Recorded in advance so that "the intervals corroborated the point estimates" is a falsifiable
statement rather than an assumption. Per-claim conditions are under each claim in Part 2.
`scripts/compare_runs.py` flags every point estimate that moves materially between runs, and each
flagged movement is explained rather than silently adopted.
