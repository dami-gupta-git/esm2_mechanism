# Does ESM-2 encode disease mechanism?

**run_biorxiv · 2026-08-14** · ESM-2 `esm2_t33_650M_UR50D` · 17,770 variants · 1,931 genes ·
1,144 protein families · classes LOF 76% / GOF 15% / DN 9%. Confirmatory claims and decision
rules: [`PREREGISTRATION_run_biorxiv.md`](../../../biorxiv/PREREGISTRATION_run_biorxiv.md).

---

## The question

A protein language model (ESM-2) reads a protein's amino-acid sequence and produces a
high-dimensional numerical summary of it, called an embedding. When a disease-causing
mutation changes one amino acid, the embedding changes too. The change between the original
("wildtype") embedding and the mutated embedding is the delta. This experiment asks whether
that delta encodes the *mechanism* by which the mutation causes disease: does it act by
disabling the protein (loss-of-function, LOF), by making it overactive (gain-of-function,
GOF), or by poisoning the normal copies (dominant-negative, DN)?

Because proteins fall into families that share sequence and function, a model might appear
to predict mechanism when it is really just recognizing which family a protein belongs to.
Kinases, for example, are enriched for GOF. To control for this, each feature is tested
under two cross-validation setups: gene-split, where related proteins can leak across
train/test, and family-split, where entire protein families are held out.

run_biorxiv replaces run6's seed-to-seed standard deviations with cluster-bootstrap
confidence intervals (CIs) that account for genes in the same family not being independent,
and adds permutation p-values for the two features under test.

---

## Setup

- **Dataset:** 17,770 pathogenic missense variants across 1,931 genes grouped into 1,144
  Pfam families. Class distribution: LOF 13,556 (76.3%), GOF 2,668 (15.0%), DN 1,546
  (8.7%).
- **Cross-validation:** 5-fold, gene-level (no gene appears in both train and test), run at
  5 random seeds. Family-split holds out entire Pfam families.
- **Metrics:** macro-F1 (equal weight per class) and one-vs-rest AUROC per class.
- **Chance floors:** macro-F1 = 0.288 (majority-class DummyClassifier, gene-split) / 0.290
  (family-split). AUROC = 0.50.
- **Confidence intervals:** 95% cluster bootstrap, 1,000 resamples, from seed 0. Gene-split
  CIs resample genes (1,931 clusters); family-split CIs resample families (1,144 clusters).
  Family-split CIs are wider because 846 of the 1,144 families are single-gene, reducing the
  effective cluster count.
- **Permutation tests:** 1,000 permutations at seed 0, family-level block permutation. Two
  features tested: `wt_only_mean` (refit-per-permutation, macro-F1 statistic) and
  `delta_mean` (out-of-fold predictions, macro one-vs-rest AUROC statistic).
- **Probes:** logistic regression (linear) for the main tables; MLP, kNN, GBM, and random
  forest for the nonlinear panel. All probes are uncalibrated and measure discrimination
  only, not risk.

---

## Glossary

**Features (rows):**

| Name | Dimensionality | Description |
|---|---|---|
| `wt_only_mean` | 1280-d | ESM-2 embedding of the original protein, mean-pooled over residues |
| `mut_only_mean` | 1280-d | ESM-2 embedding of the mutant protein, mean-pooled |
| `wt_concat_mut` | 2560-d | Wildtype and mutant embeddings concatenated |
| `delta_mean` | 1280-d | Mutant minus wildtype embedding (the mutation's effect) |
| `delta_per_residue` | 1280-d | Same delta at the single mutated position instead of mean-pooled |
| `onehot_aa` | 40-d | Amino-acid substitution identity, no sequence context |
| `foldx_ddg` | scalar | Physics-based stability change estimate (kcal/mol) |
| `alphamissense` | scalar | DeepMind pathogenicity score |

**Metrics (columns):**

| Metric | What it measures | Range | Chance value |
|---|---|---|---|
| macro-F1 | Per-class F1 averaged equally across the 3 classes | 0--1 | 0.288 |
| AUROC GOF | Separating gain-of-function from everything else | 0.5--1 | 0.50 |
| AUROC DN | Separating dominant-negative from everything else | 0.5--1 | 0.50 |
| AUROC LOF | Separating loss-of-function from everything else | 0.5--1 | 0.50 |

The macro-F1 chance floor of 0.288 is the score a majority-class classifier achieves (it
always predicts LOF). It is not zero because the imbalanced class distribution gives LOF a
non-zero F1 even without learning. Each AUROC is one-vs-rest (one class against the other
two combined), so its chance value is 0.50 regardless of the number of classes.

---

## Table 1. Gene-split (leakage-prone)

Each row is a different input feature; each column is a different way of scoring how well
that feature predicts mechanism. Gene-split means that no single gene appears in both the
training and test sets, but genes from the same protein family can appear on opposite sides.
This is the standard setup but it lets the model exploit family resemblance as a shortcut,
so scores here may overstate what the feature actually knows about mechanism.

**Note.**   
Chance floors: macro-F1 = 0.288, AUROC = 0.50.
All values and 95% CIs are from seed 0 (cluster bootstrap, 1,000 resamples over 1,931 genes). 5-seed means (in aggregate.json) lie within 0.01–0.02 of these seed-0 values for features above the floor. For floor-level features, per-fold AUROC varies more across seeds (predictions are near-random, so AUROC is unstable), while macro-F1 remains stable.

| Feature | macro-F1 | AUROC GOF | AUROC DN | AUROC LOF |
|---|---:|---:|---:|---:|
| wt_only_mean | 0.566 [0.483, 0.636] | 0.821 [0.725, 0.887] | 0.712 [0.614, 0.799] | 0.820 [0.756, 0.876] |
| mut_only_mean | 0.567 [0.484, 0.639] | 0.821 [0.724, 0.886] | 0.711 [0.614, 0.798] | 0.820 [0.756, 0.876] |
| wt_concat_mut | 0.565 [0.486, 0.632] | 0.820 [0.727, 0.884] | 0.704 [0.605, 0.789] | 0.815 [0.752, 0.872] |
| delta_mean | 0.288 [0.277, 0.299] | 0.438 [0.334, 0.547] | 0.369 [0.302, 0.458] | 0.413 [0.335, 0.486] |
| delta_per_residue | 0.310 [0.293, 0.335] | 0.571 [0.517, 0.620] | 0.561 [0.458, 0.679] | 0.555 [0.516, 0.595] |
| onehot_aa | 0.288 [0.277, 0.299] | 0.489 [0.426, 0.551] | 0.508 [0.424, 0.596] | 0.486 [0.432, 0.534] |
| foldx_ddg | 0.278 [0.257, 0.295] | 0.576 [0.532, 0.627] | 0.493 [0.409, 0.607] | 0.570 [0.524, 0.625] |
| alphamissense | 0.288 [0.275, 0.299] | 0.564 [0.505, 0.632] | 0.545 [0.491, 0.607] | 0.585 [0.547, 0.627] |
| *naive baseline* | *0.288* | *0.500* | *0.500* | *0.500* |

## Table 2. Family-split (homology-controlled)

Same features and metrics as Table 1, but now entire protein families are held out of
training. A model tested here cannot score well by recognizing which family a protein belongs
to, so this is the honest test of whether a feature carries genuine mechanism signal. Any
score that drops from Table 1 to Table 2 was partly relying on family resemblance.

**Note.** Chance floors: macro-F1 = 0.290 (family-split), AUROC = 0.50. All values and 95%
CIs are from seed 0 (cluster bootstrap, 1,000 resamples over 1,144 families). DN intervals
(approximately 150 genes, approximately 90 families) are the least trustworthy in the table.

| Feature | macro-F1 | AUROC GOF | AUROC DN | AUROC LOF |
|---|---:|---:|---:|---:|
| wt_only_mean | 0.502 [0.404, 0.553] | 0.752 [0.631, 0.833] | 0.766 [0.676, 0.820] | 0.817 [0.737, 0.872] |
| mut_only_mean | 0.508 [0.419, 0.559] | 0.754 [0.633, 0.833] | 0.762 [0.672, 0.816] | 0.818 [0.738, 0.873] |
| wt_concat_mut | 0.476 [0.391, 0.526] | 0.736 [0.610, 0.823] | 0.745 [0.656, 0.797] | 0.801 [0.720, 0.859] |
| delta_mean | 0.288 [0.268, 0.306] | 0.400 [0.307, 0.542] | 0.466 [0.385, 0.586] | 0.418 [0.325, 0.546] |
| delta_per_residue | 0.310 [0.277, 0.352] | 0.520 [0.484, 0.576] | 0.575 [0.485, 0.689] | 0.529 [0.499, 0.573] |
| onehot_aa | 0.288 [0.268, 0.306] | 0.461 [0.409, 0.550] | 0.521 [0.456, 0.588] | 0.482 [0.423, 0.560] |
| foldx_ddg | 0.278 [0.247, 0.302] | 0.555 [0.505, 0.630] | 0.483 [0.420, 0.603] | 0.550 [0.486, 0.634] |
| alphamissense | 0.288 [0.266, 0.305] | 0.468 [0.389, 0.578] | 0.552 [0.502, 0.617] | 0.506 [0.437, 0.587] |
| *naive baseline* | *0.290* | *0.500* | *0.500* | *0.500* |

---

## Pre-registered claims tested in this experiment

Before the run, three claims were written down with explicit pass/fail rules (in
[`PREREGISTRATION_run_biorxiv.md`](../../../biorxiv/PREREGISTRATION_run_biorxiv.md)), so the
verdict could not be adjusted after seeing the results. This section applies those rules.

### 2A. The delta embedding does not encode mechanism

The question is whether the delta is genuinely at the chance floor, not just close to it.

The floor (0.375 +/- 0.05) is where the best nonlinear model landed on the delta. So
0.425 is the highest the linear probe's score could plausibly be before the
claim that the delta carries no mechanism signal would need to be reconsidered.

The linear probe scored 0.288, and its CI says the true value could be as high as 0.306 but
no higher. Since 0.306 is below 0.425, even in the worst case the linear probe does not come
close to exceeding the floor by a meaningful amount. Verdict: ✅ **affirmed.**

- **Floor (MLP `delta_mean` family-split, 5-seed mean):** 0.375.
- **Linear `delta_mean` family-split macro-F1:** 0.288 [0.268, 0.306].
- ✅ **CI upper bound (0.306) < floor + 0.05 (0.425): affirmed.**

A permutation test provides a second line of evidence. Labels were shuffled 1,000 times at
the family level (block permutation, seed 0), and the model was re-scored each time to build
a reference distribution of what the score looks like when the labels are meaningless.

- ✅ `delta_mean` macro one-vs-rest AUROC against out-of-fold predictions: observed 0.428, null
  mean 0.454, p = 0.678. 3 families with unique gene counts kept their own labels. The
  real-label score is not distinguishable from shuffled labels. This does not refute 2A.
- ✅ `wt_only_mean` macro-F1 with refit: observed 0.466, null mean 0.326, p < 0.001. The
  protein embedding's above-floor score is not attributable to chance.

### 2B. Homology leakage exists

The question is whether the drop from gene-split to family-split is real, confirming that
part of the gene-split score comes from family resemblance. Tested with a paired cluster
bootstrap on the gap (family-resampled, 1,144 clusters, seed 0).

- **`wt_only_mean` split gap:** 0.064 [0.001, 0.142]. The CI excludes zero: leakage is
  present.

---

## Reading the tables

**1. The linear delta carries no detectable mechanism signal.** `delta_mean` family-split
macro-F1 = 0.288 [0.268, 0.306], indistinguishable from the 0.290 floor. Its CI sits
entirely within the floor's own CI [0.270, 0.304]. The permutation test confirms: p = 0.678.

**1b. The nonlinear delta recovers weak signal, but not mechanism understanding.** The MLP
on `delta_mean` family-split scores 0.375 [0.331, 0.410], above the 0.290 floor but
substantially below the linear absolute-embedding score (0.502). The signal is also reduced
under family holdout (kNN drops from 0.414 to 0.357), consistent with residual family
structure rather than a learned mechanism representation.

**2. The protein embedding predicts mechanism, but through family identity.** `wt_only_mean`
scores 0.566 on gene-split but drops to 0.502 on family-split. The paired gap is 0.064
[0.001, 0.142] (family-resampled), excluding zero. Part of the gene-split score comes from
recognizing which protein family a gene belongs to. Families correlate with mechanism
(kinases are enriched for GOF, structural proteins for DN) because labels are per-gene, so
the model uses family identity as a proxy.

**3. Reference predictors also fail.** AlphaMissense scores macro-F1 = 0.288 on family-split,
at the floor. FoldX scores 0.278, slightly below it. The limitation is not specific to
ESM-2.

**4. LOF is the most separable class.** Under family-split, `wt_only_mean` reaches AUROC
0.817 for LOF versus 0.766 for DN and 0.752 for GOF. LOF is the most directly encoded
property; DN, the rarest class, carries the widest CI.

---

## Nonlinear probes on the delta

A linear probe recovers only linearly separable signal. Four nonlinear models test whether
the delta contains structure the linear probe misses, and whether any such structure
survives when whole families are held out.

**Note.** Chance floor: macro-F1 = 0.288 (gene-split) / 0.290 (family-split). Values are
5-seed means; 95% CIs in brackets are from seed 0.

| Model (on `delta_mean`) | Gene-split macro-F1 | Family-split macro-F1 |
|---|---:|---:|
| MLP | 0.395 [0.368, 0.471] | 0.375 [0.331, 0.410] |
| kNN | 0.414 [0.365, 0.431] | 0.357 [0.330, 0.395] |
| GBM | 0.309 [0.290, 0.332] | 0.297 [0.270, 0.317] |
| Random forest | 0.298 [0.282, 0.316] | 0.290 [0.268, 0.306] |

On gene-split, kNN (0.414) and the MLP (0.395) raise the delta above the 0.288 floor,
indicating weak nonlinear structure not accessible to a linear probe. Under family-split,
every model falls, and kNN drops the most (0.414 to 0.357), consistent with residual family
structure rather than mechanism signal. The MLP (0.375) is the strongest nonlinear delta
probe on the honest split, and its CI [0.331, 0.410] overlaps with the floor's CI. The tree
models collapse back toward the floor (GBM 0.297, RF 0.290).

None of the nonlinear family-split scores reach the linear protein-embedding family-split
score (0.470 five-seed mean, 0.502 at seed 0), so the nonlinear delta does not exceed the
linear absolute embedding.

---

## Leakage fraction

The leakage fraction measures what share of each feature's above-chance gene-split score is
attributable to family recognition: (gene-split F1 minus family-split F1) divided by
(gene-split F1 minus chance). It is defined only for features whose gene-split score is
meaningfully above chance. Values are 5-seed means from `leakage_fraction.json`.

| Feature | Gene-split F1 | Family-split F1 | Drop | Leakage fraction |
|---|---:|---:|---:|---:|
| wt_only_mean | 0.559 | 0.470 | 0.089 | 0.33 |
| mut_only_mean | 0.560 | 0.468 | 0.091 | 0.34 |
| wt_concat_mut | 0.559 | 0.463 | 0.096 | 0.35 |
| delta_per_residue | 0.316 | 0.308 | 0.008 | 0.30 |

`delta_mean`, `onehot_aa`, `foldx_ddg`, and `alphamissense` are at the floor on gene-split,
so the leakage fraction is undefined.

The `wt_only_mean` leakage fraction has a bootstrap CI of 0.230 [0.005, 0.517] (1,144
family clusters). The wide interval reflects the small effective cluster count and the
compounding uncertainty of a ratio whose numerator and denominator are both bootstrapped.

---

## Single-source robustness check

The merged dataset combines variants from two curation sources (Gerasimavicius and ClinVar
via G2P). To confirm the mechanism result is not an artifact of merging differently-curated
datasets, the probe was re-run on the 10,138 Gerasimavicius-only variants (948 genes).

| Feature | Gene-split macro-F1 | Family-split macro-F1 | Chance floor |
|---|---:|---:|---:|
| wt_only_mean | 0.611 | 0.456 | 0.279 |
| delta_mean | 0.279 | 0.280 | 0.280 |

The pattern holds: the protein embedding predicts mechanism above chance, and the delta sits
at the floor. The Gerasimavicius subset has more GOF (19.5%) and DN (8.8%), explaining the
slightly higher `wt_only_mean` score. The delta result is unchanged.

---

## Interpretation

The linear delta is at the chance floor under family-split (macro-F1 0.288, p = 0.678),
showing no detectable mechanism signal. Nonlinear probes recover weak signal (MLP 0.375),
but it is substantially smaller than the absolute-embedding baseline (0.502) and reduced
under family holdout, consistent with residual family structure rather than a learned
mechanism representation. The protein embedding's above-floor score comes from family
identity acting as a proxy for mechanism, not from the mutation itself.

A single missense substitution shifts ESM-2's protein-level embedding only slightly, so
the delta is dominated by per-variant variation rather than a consistent per-mechanism
direction. There is also a granularity mismatch: mechanism labels are per-gene (every
variant in a gene shares one label), whereas the delta is per-variant. A variant-level
feature is poorly matched to a gene-level label, while the gene-level wildtype embedding
is well matched.

---

## What this is and is not

- This experiment shows that ESM-2's delta embedding does not linearly separate disease
  mechanisms, and that nonlinear models recover only weak signal consistent with family
  structure. It does not show that mechanism is unrecoverable from protein language models in
  general; a larger model, a different architecture, or a site-restricted representation
  might perform differently.
- One ESM-2 size (650M parameters) was tested.
- Mechanism labels are per-gene. Badonyi & Marsh 2025 report that 43% of multi-phenotype
  dominant genes carry both LOF and non-LOF mechanisms. Some fraction of the 17,770 variants
  is therefore mislabelled by construction. This gives an alternative explanation for the
  null: the delta may sit at the floor because the labels are noisy, not because the
  embedding lacks mechanism signal. Tasks 2d and 8 in
  [`FOLLOWUP_biorxiv.md`](../../../biorxiv/FOLLOWUP_biorxiv.md) address this: whether the null
  survives on cleanly-labelled genes, and how far realistic label noise moves a working
  probe.
- DN (8.7%, approximately 150 genes) and GOF (15.0%) are rare. Their per-class AUROC
  intervals are the least trustworthy in their tables and no confirmatory claim rests on
  them.
- All probes are uncalibrated; reported scores measure discrimination, not calibrated risk.

---

## Provenance

Embeddings: four arrays `(17770, 1280)`, row-aligned to `valid_variants.json` (17,770
rows). AlphaMissense coverage 17,765 / 17,770 (>99.9%).

| Result | Source files |
|---|---|
| Linear baselines (Tables 1--2) | `results/run_biorxiv/family_split_baselines_seed{0..4}.json`, `aggregate.json` |
| Nonlinear probes | `results/run_biorxiv/nonlinear_results_seed{0..4}.json` |
| Naive baseline | `results/run_biorxiv/naive_baseline.json` |
| Permutation tests | `results/run_biorxiv/backup_step2_permutation_seed0.json` |
| Leakage fraction | `results/run_biorxiv/leakage_fraction.json` |
| Family clustering | `results/run_biorxiv/family_clustering.json` |
| Single-source check | `results/run_biorxiv/single_source_gerasimavicius/aggregate.json` |

Progress log: [`PROGRESS.md`](../../../biorxiv/PROGRESS.md).
