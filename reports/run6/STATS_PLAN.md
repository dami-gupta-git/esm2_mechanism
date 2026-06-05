# Statistical analysis plan (pre-preprint)

Tracking document for the inferential statistics to add before this work is submitted as a
preprint. The run6 reports each carry a short "Statistical limitations and planned analyses"
section; this file holds the fuller rationale and the shared methodology so the per-report
sections can stay tight.

The shared resampling machinery is now built (`src/esm2_mech/utils/bootstrap.py`:
`cluster_bootstrap_ci`, `bootstrap_mechanism_metrics`, `label_permutation_pvalue`), and the
mechanism probe and naive-baseline floor emit cluster-bootstrap CIs by default. The result files
under `results/run6/` will carry these CIs after the planned pipeline re-run; the per-report
tables are updated from those files at that point.

The central issue throughout: the reported error bars are 5-seed spreads, and a seed only
reshuffles the cross-validation folds on a fixed dataset. That measures fold-assignment jitter,
not sampling uncertainty, and it understates the true error because every seed reuses all the
data. The plan replaces it with dependency-aware inference.

---

## Shared methodology

**Resampling unit — genes, not variants.** Mechanism labels are gene-level and variants cluster
within genes (and genes within families), so observations are not independent. The effective
sample size is closer to the gene count (≈ 1,935) than the variant count (17,826), and far
smaller for the rare classes (DN ≈ 9%, GOF ≈ 15%). Every confidence interval and significance
test must resample whole genes (or whole families, where the unit is the family) — resampling
variants would be anticonservative by a large factor.

**Confidence intervals.** Replace seed-std error bars with 95% CIs from a cluster bootstrap that
resamples the dependency unit (gene or family) with replacement, recomputing the metric on each
resample (≥ 1,000 resamples).

**Significance against chance.** Attach a p-value via a label-permutation test: shuffle the
mechanism labels within the cross-validation, recompute the metric (≥ 1,000 permutations), and
locate the observed value in the resulting null. The family-clustering report already does this
(shuffled-label nulls and z-scores) — that is the framework the other reports adopt.

**Significance of a difference.** For "feature/model A beats B" claims, use a paired cluster
bootstrap on the shared subset and report a 95% CI on the difference, not separated error bars.

**Metrics for imbalanced classes.** Report AUPRC with its prevalence baseline alongside AUROC,
and PPV/NPV at class prevalence for the rare classes; AUROC alone overstates usefulness at
9–15% prevalence.

**Multiple comparisons.** Where many cells are tested (the within-family table), apply a
Benjamini-Hochberg false-discovery-rate correction, or restate the table explicitly as
exploratory rather than inferential.

**Calibration.** The probes are not calibrated; all reported scores measure discrimination only
and are not risk estimates. State this rather than fixing it — the claims are about
discrimination, not calibration.

---

## Per-report plan

### report_classifier.md (main mechanism result)

- Dependency-aware confidence intervals: 95% CIs from a cluster bootstrap that resamples whole
  genes, on every macro-F1 and AUROC, replacing the seed-std error bars. Note the effective N is
  ≈ 1,935 genes, not 17,826 variants — and smaller still for the rare classes (DN ≈ 9%,
  GOF ≈ 15%).
- Significance against chance: a label-permutation test (mechanism labels shuffled within the
  CV, ≥ 1,000 repeats) to attach a p-value to "macro-F1 above chance" and to the gene-split
  minus family-split gap.
- Metrics for imbalanced classes: AUPRC with its prevalence baseline alongside AUROC, and
  PPV/NPV at class prevalence for the rare classes, since AUROC alone overstates usefulness at
  9–15% prevalence.
- Calibration: the probes are not calibrated; the reported scores measure discrimination only
  and are not risk estimates.

### single-source robustness check (Gerasimavicius-only)

- Same plan as the classifier report, recomputed on the single-source subset (10,138 variants;
  LOF 7,262 / GOF 1,982 / DN 894) so the source/class confound is removed.
- Dependency-aware confidence intervals: 95% CIs from a cluster bootstrap that resamples whole
  genes on the subset, on every macro-F1 and AUROC. The effective N is smaller than the merged
  result and far smaller for the rare classes, so the CIs are expected to be wide — the point is
  whether delta_mean's interval still straddles the floor.
- Significance against chance: the label-permutation test compares against the recomputed
  single-source floor (most-frequent macro-F1 ≈ 0.279), not the merged 0.288 floor.
- Headline to confirm with intervals: delta_mean sits at the floor on both splits, while wt_only
  drops from 0.612 (gene) to 0.445 (family) — a CI on that gene-minus-family gap quantifies the
  cross-family collapse.

### report_control.md (pathogenicity positive control)

- Dependency-aware confidence intervals: a 95% CI from a cluster bootstrap that resamples whole
  genes, on each AUROC, replacing the seed-std error bars. The classes are balanced here, but
  the dependency structure still applies.
- Calibration: as in the classifier report, the probes measure discrimination only; the scores
  are not calibrated risk estimates.

### report_protein_family.md (family clustering, single seed)

- Multi-seed probe metrics: the family-probe accuracy is reported for one seed (seed 0); repeat
  over at least 5 seeds and report a spread, matching the other reports.
- Dependency-aware confidence intervals: 95% CIs from a cluster bootstrap that resamples whole
  families, for both the probe accuracy and the k-nearest-neighbour purity metrics.
- Note: this report already uses shuffled-label nulls and z-scores — the permutation framework
  the mechanism reports will adopt.

### report_within_family.md (28 families, small n)

- Multiple-comparison control: 28 families are tested across two views and two probes; the
  current "beats baseline and std < 0.10" highlight is an uncorrected screen. Apply a
  Benjamini-Hochberg false-discovery-rate correction, or restate the table as exploratory.
- Power: at 6–33 genes per family the test cannot establish absence of signal, only failure to
  detect it. Add a minimal-detectable-effect statement per family so the nulls are read as
  underpowered rather than as evidence of no effect.
- Dependency-aware confidence intervals: 95% CIs from a cluster bootstrap over the genes within
  each family, replacing the seed-std error bars.

### report_esm3_mechanism.md (merged) and report_esm3_mechanism_geras.md (superseded)

- Significance of the scale lift: the ESM-3 seq family-split macro-F1 (0.438) and the matched
  ESM-2 MLP delta_mean baseline (0.380) are compared as point estimates, and the M2 gate clears its
  0.430 threshold by only 0.008 — about one seed of spread. Run a paired cluster bootstrap over
  genes on the shared variant set, reporting a 95% CI on the difference (and on seq_struct − seq),
  so "ESM-3 beats ESM-2" and "structure adds nothing" rest on tested gaps rather than separated
  error bars and a thin threshold margin.
- Significance against chance: a label-permutation test for the family-split scores of both
  conditions, matching the classifier report.
- Effective sample size: labels are gene-level and variants cluster within genes, so all CIs and
  tests resample whole genes rather than variants.
- The merged report (`report_esm3_mechanism.md`) is the live one; its scale-lift CI points at the
  merged shared subset (17,826 variants, 1,935 genes). The geras report is superseded (different
  dataset and a now-fixed data defect) and is not cited.

### report_contrastive.md (cross-family contrastive head)

- Significance of the gain: the contrastive k-NN macro-F1 (0.395) beats the raw-delta k-NN
  baseline (0.354) by +0.041 on the shared family-split subset. Run a paired cluster bootstrap
  over genes on that subset and report a 95% CI on the difference, so the gain rests on a tested
  gap rather than two separated point estimates.
- Significance against chance: a label-permutation test for the contrastive macro-F1, located in
  the gene-shuffled null, and compared against BOTH the 0.288 MLP floor and the raw-kNN baseline.
- Per-class caveat to confirm: the report attributes the gain to class balance rather than
  per-class separability, with DN staying at chance (AUROC 0.577 → 0.545). Attach gene-cluster
  CIs to the per-class AUROCs so the "DN unmoved" claim is read as a tested null, not a point drop.
- This report's own "Statistical limitations" section already names this plan; the shared
  machinery now supports it.

### report_geometry.md (magnitude/direction/conservation)

- Dependency-aware confidence intervals: 95% CIs from a cluster bootstrap that resamples whole
  genes, on each pathogenicity AUROC (effective N ≈ 1,929 genes, not 37,218 variants), replacing
  the seed-std error bars.
- Significance of the conservation result: the decisive claim is that conservation
  (masked-marginal, 0.891) beats the mean-pooled embedding delta (0.859) and that the delta adds
  nothing on top (+0.002, gate K2). Run a paired cluster bootstrap over genes on the shared variant
  set and report a 95% CI on both differences, so K2 rests on a tested gap.
- Significance of the transfer contrast: pathogenicity transfers across families (0.85–0.90) while
  mechanism does not (0.62–0.64); attach a paired cluster-bootstrap CI to that task gap.
- Calibration: the probes are uncalibrated; the scores measure discrimination only.
- Note: the magnitude/direction decomposition rejected its own pre-registered gates (P1/P2 expected
  pathogenicity to be magnitude); the report states this. The cosine null in the geometry table is
  the shuffled-label permutation framework the other reports adopt.

---

## Priority

The two analyses that most change the standing of the work — and that reuse existing data with
no GPU — are the cluster bootstrap over genes (confidence intervals) and the label-permutation
test (p-value against chance). The shared machinery for both is built (`utils/bootstrap.py`) and
wired into the mechanism probe (CIs on by default; permutation opt-in via `--n_permutations`,
slow because it refits per repeat) and the naive-baseline floor. What remains is the planned
pipeline re-run to populate the result files, then the rest (AUPRC/PPV-NPV, FDR, power,
multi-seed, calibration note), which build on the same resampling machinery.
