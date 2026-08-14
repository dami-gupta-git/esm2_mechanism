# Findings — run_biorxiv

Observations and design decisions from run_biorxiv, as distinct from `RUNBOOK_biorxiv.md` (what
was run and its status).

## WT-mismatch check flagged 9 genes in the Gerasimavicius set (2026-08-12)

The stored wild-type residue does not match the sequence on file for MEN1 (38/47 variants) and
CYP21A2 (34/37), which account for most of the mismatches, plus SHANK3, TUFM, TPI1, ARID1B, FDX2,
AGT, and TRPC3 to a lesser extent.

MEN1 and CYP21A2 are confirmed isoform mismatches: the cached sequence is a different UniProt
isoform than the one the variants are numbered against. The other seven are not yet root-caused.

`build_valid_variants` drops these variants (`skipped_invalid`), so the cost is coverage, not
contamination — MEN1 and CYP21A2 drop out of per-gene analysis until the isoform mapping is fixed
at the sequence-fetch step.

## Bootstrap wiring

Every script that produces results computes confidence intervals through the shared bootstrap
code (`utils/bootstrap.py`) and writes those intervals into its output, rather than each script
reimplementing resampling on its own. A script that isn't wired this way produces result files
with point estimates but no uncertainty range — a gap that does not show up as an error, only as
missing keys in the output JSON, which is why a verification gate (checking `ci_low`/`ci_high` are
actually populated, not just that the script exited cleanly) exists downstream of this wiring. That
gate has caught real cases of scripts running cleanly while silently not producing intervals.

The reference implementation is the mechanism classification script
(`classify_by_mechanism`): it passes gene/family cluster assignments into the shared bootstrap
function, exposes flags to turn CIs off or change the number of resamples, and writes the interval
into the result JSON.

## The leakage ratio

The leakage figure is a ratio built from two pieces of data that share a common term (the
gene-split component appears in both the numerator and the denominator). Because of that overlap,
its confidence interval has to come from resampling the whole ratio together on each bootstrap
replicate, not from computing separate intervals for the two pieces and combining them afterward —
combining separately-computed intervals would understate the uncertainty, since it ignores the
correlation induced by the shared term.

## Paired cluster bootstrap

`utils/bootstrap.py` provides two pairing modes for comparing two arms of a result (e.g. one
metric before and after a change, or two splits of the same experiment):

- **Same-fold pairing** (`paired_cluster_bootstrap_diff`) — both arms share the same row space,
  so one resample of the cluster unit is drawn per replicate and applied to both arms at once.
- **Cross-partition pairing** (`paired_cluster_bootstrap_diff_cross_partition`) — used when the
  two arms are resampled over different partitions of the same underlying units (e.g. a gene-split
  metric compared against a family-split metric on the same genes).

`paired_oof_diff` wraps both, aligning the two arms by row id, taking the class list as an
explicit parameter, and supporting macro-F1, binary AUROC, and one-vs-rest AUROC.
`adjudicate_diff` and `adjudicate_level` render the pre-registered CI verdict (affirmed / not
distinguishable / failed / underpowered) for a difference and for a single level respectively.

The design principle both modes implement: one shared resample per replicate applied to both
arms, so the two numbers being compared are always evaluated on the same resampled data rather
than on independently-resampled data that would inflate the apparent spread of the difference.

## Permutation tests

Two different features are tested with two different nulls, because they measure different
things and a single test would misrepresent one of them:

- `wt_only_mean` refits the probe once per permutation and scores macro-F1.
- `delta_mean` re-scores macro one-vs-rest AUROC against cached out-of-fold predictions and
  refits nothing. Macro-F1 cannot register a change for a probe sitting at the chance floor — it
  predicts the majority class almost everywhere regardless of what the underlying ranking holds —
  so a ranking-sensitive metric is used instead for this feature.

Both permute at the family level, not the gene level: both score a family-split metric, and the
permutation unit has to match the unit the confidence interval clusters on, or the null is too
narrow (a gene-level shuffle breaks the label structure that homologous genes share). Every
permutation result records which statistic was used, which null type, which unit was permuted, the
null's width, and how many families had no same-size partner to swap with, so a report reading the
number does not have to assume any of this.

A p-value sitting exactly at the resolution floor (`1/(N+1)` for `N` permutations) is reported as
an unresolved floor, not as a measurement — it means the true p-value could be anywhere below that
floor, not that it equals it.

## AUPRC, PPV, and NPV for rare classes

AUROC alone overstates usefulness for rare classes, because it is insensitive to class imbalance.
For those classes the machinery also emits:

- **AUPRC** alongside its no-signal baseline, which is the class prevalence (not 0.5, as for
  AUROC). The gap between AUPRC and prevalence is computed within each bootstrap resample, rather
  than comparing a moving AUPRC against a fixed baseline — a fixed baseline under a moving AUPRC
  is the misreading this pairing exists to prevent.
- **PPV and NPV at the prevalence-matched operating point** — the top `prevalence × n` scores by
  rank are called positive, so the predicted positive rate equals the observed one. This needs
  only the ranking, not calibrated probabilities, which is what makes it reportable for a probe
  that is not calibrated.

Every probe report states that its scores are a ranking (discrimination), not a calibrated risk
estimate — the PPV/NPV pair is what answers "if it flags this variant, how often is that right."
