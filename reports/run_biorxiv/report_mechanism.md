# Does ESM-2 encode disease mechanism?

**run_biorxiv · 2026-08-19** · ESM-2 `esm2_t33_650M_UR50D` · 17,770 variants ·
1,931 genes · 1,144 Pfam families · classes LOF 76% / GOF 15% / DN 9%.
Confirmatory rules: [`PREREGISTRATION_run_biorxiv.md`](../../biorxiv/PREREGISTRATION_run_biorxiv.md).

## Summary

Under the preregistered linear PCA probe, the mutation-induced change in the ESM-2 embedding does
not support reliable classification of loss-of-function, gain-of-function, and dominant-negative
mechanisms. The unmutated protein embedding performs better, but part of that result comes from
protein-family identity. A weak mutation-related ranking signal remains, so the finding is not a
complete absence of information.

## The question

ESM-2 converts a protein sequence into a numerical representation called an embedding. A missense
variant changes one amino acid and slightly changes that embedding. The difference between the
mutant and wildtype embeddings is the delta.

This experiment asks whether the delta distinguishes three disease mechanisms: loss-of-function
(LOF), gain-of-function (GOF), and dominant-negative (DN). Because related proteins can share both
sequence and mechanism, the analysis also tests whether a classifier is recognizing protein family
rather than reading the mutation's effect.

## Setup

- Dataset: 17,770 pathogenic missense variants across 1,931 genes and 1,144 Pfam families.
  The class counts are 13,556 LOF, 2,668 GOF, and 1,546 DN.
- Cross-validation: five folds and five random seeds. No gene appears in both training and test
  data.
- Gene split: related genes from the same Pfam family can appear on opposite sides of the split.
- Family split: complete Pfam families are held out together.
- Main probe: logistic regression. Embedding features are reduced to 256 principal components.
- Metrics: macro-F1 for three-class classification and one-vs-rest AUROC for class ranking.
- Measured floors: macro-F1 of 0.288 for the gene split and 0.290 for the family split. AUROC has a
  no-signal value of 0.500.
- Confidence intervals: 95% cluster bootstrap with 1,000 resamples. Seed 0 intervals resample genes
  for the gene split and families for the family split.

All probes measure discrimination, not calibrated clinical risk.

## Glossary

| Feature | Dimensionality | Description |
|---|---:|---|
| `wt_only_mean` | 1,280 | Mean-pooled embedding of the unmutated protein |
| `mut_only_mean` | 1,280 | Mean-pooled embedding of the mutant protein |
| `wt_concat_mut` | 2,560 | Wildtype and mutant embeddings combined |
| `delta_mean` | 1,280 | Mutant minus wildtype embedding, averaged across the protein |
| `delta_per_residue` | 1,280 | The same embedding change at the mutated residue |
| `onehot_aa` | 40 | Identity of the original and substituted amino acids |
| `foldx_ddg` | 1 | Predicted change in protein stability |
| `alphamissense` | 1 | AlphaMissense pathogenicity score |

The measured macro-F1 floor is produced by always predicting LOF, the most common class. AUROC can
detect weak ranking even when the classifier still assigns nearly every variant to LOF.

## Table 1. Gene-split (leakage-prone)

The gene split permits related proteins to appear in both training and test data, so family
resemblance can inflate performance.

| Feature | Macro-F1, five-seed mean | Macro-F1, seed 0 (95% CI) | AUROC GOF | AUROC DN | AUROC LOF |
|---|---:|---:|---:|---:|---:|
| wt_only_mean | 0.552 | 0.534 [0.458, 0.583] | 0.814 | 0.747 | 0.838 |
| mut_only_mean | 0.548 | 0.533 [0.457, 0.581] | 0.816 | 0.745 | 0.837 |
| wt_concat_mut | 0.552 | 0.536 [0.462, 0.583] | 0.810 | 0.736 | 0.833 |
| delta_mean | 0.288 | 0.287 [0.275, 0.298] | 0.629 | 0.553 | 0.607 |
| delta_per_residue | 0.338 | 0.325 [0.306, 0.348] | 0.633 | 0.596 | 0.625 |
| onehot_aa | 0.288 | 0.287 [0.275, 0.298] | 0.554 | 0.556 | 0.555 |
| foldx_ddg | 0.279 | 0.279 [0.260, 0.294] | 0.619 | 0.589 | 0.629 |
| alphamissense | 0.288 | 0.288 [0.276, 0.299] | 0.595 | 0.638 | 0.631 |
| *Measured floor* | *0.288* | *0.287 [0.275, 0.298]* | *0.500* | *0.500* | *0.500* |

AUROC values are five-seed means. `foldx_ddg` is evaluated only where the stability estimate is
present, so its matched macro-F1 floor is 0.279.

## Table 2. Family-split (homology-controlled)

The family split tests transfer after all genes from each Pfam family are held out together.

| Feature | Macro-F1, five-seed mean | Macro-F1, seed 0 (95% CI) | AUROC GOF | AUROC DN | AUROC LOF |
|---|---:|---:|---:|---:|---:|
| wt_only_mean | 0.449 | 0.488 [0.414, 0.528] | 0.759 | 0.729 | 0.799 |
| mut_only_mean | 0.451 | 0.486 [0.413, 0.526] | 0.760 | 0.727 | 0.799 |
| wt_concat_mut | 0.449 | 0.465 [0.400, 0.504] | 0.749 | 0.717 | 0.786 |
| delta_mean | 0.290 | 0.290 [0.276, 0.305] | 0.584 | 0.524 | 0.557 |
| delta_per_residue | 0.318 | 0.320 [0.296, 0.337] | 0.597 | 0.582 | 0.595 |
| onehot_aa | 0.290 | 0.290 [0.276, 0.305] | 0.543 | 0.538 | 0.555 |
| foldx_ddg | 0.280 | 0.279 [0.256, 0.301] | 0.618 | 0.600 | 0.622 |
| alphamissense | 0.289 | 0.290 [0.276, 0.305] | 0.598 | 0.640 | 0.631 |
| *Measured floor* | *0.290* | *0.290 [0.276, 0.305]* | *0.500* | *0.500* | *0.500* |

AUROC values are five-seed means. DN is the smallest class, so its class-specific estimates are
less precise. The matched `foldx_ddg` macro-F1 floor is 0.280.

Table 2 reports the preregistered linear probe: 256 PCA components fitted within each training fold,
without feature standardization or class weighting. The exploratory magnitude-and-direction analysis
in [`report_geometry.md`](report_geometry.md) uses a full-dimensional, fold-standardized,
class-balanced logistic regression and obtains macro-F1 0.387 for the same full delta and mechanism
cohort. That value is a different probe specification and does not replace the 0.290 confirmatory
result or change claim 2A-1.

## Table 3. Nonlinear probes on the delta

Flexible probes recover some information from the mean-pooled delta, but remain below the linear
wildtype embedding.

| Model on `delta_mean` | Gene split, seed 0 (95% CI) | Gene split, five-seed mean | Family split, seed 0 (95% CI) | Family split, five-seed mean |
|---|---:|---:|---:|---:|
| MLP | 0.408 [0.365, 0.434] | 0.395 | 0.380 [0.333, 0.399] | 0.375 |
| k-nearest neighbours | 0.393 [0.359, 0.417] | 0.414 | 0.357 [0.328, 0.369] | 0.357 |
| Gradient-boosted trees | 0.306 [0.290, 0.327] | 0.310 | 0.297 [0.280, 0.313] | 0.297 |
| Random forest | 0.295 [0.281, 0.313] | 0.298 | 0.290 [0.276, 0.306] | 0.290 |

The MLP is the strongest delta model under family holdout at a five-seed mean of 0.375. This is
above the 0.290 measured floor but below the 0.449 linear wildtype result.

## Table 4. Family identity in the embedding

The wildtype and mutant embeddings strongly encode protein family, while subtraction removes most
of that information.

The family probe predicts one of 145 Pfam families for 755 genes. Its majority-class accuracy is
4.37%.

| Feature | Accuracy, seed 0 (95% CI) | Macro-F1, seed 0 |
|---|---:|---:|
| wt_mean | 60.1% [58.7%, 61.7%] | 0.465 |
| mut_mean | 59.7% [58.3%, 61.2%] | 0.463 |
| delta_mean | 4.37% [4.37%, 4.37%] | 0.001 |

In a separate nearest-neighbour check, 25.4% of the wildtype embedding's five nearest neighbours
share its Pfam family, compared with 0.52% after family labels are shuffled. The delta retains a
smaller local family signal: 5.20% purity compared with a 0.52% shuffled reference.

Among genes in multi-gene families, 83.2% match their family's majority mechanism label. The
wildtype leakage fraction is 0.389, with a 95% family-bootstrap interval of 0.241 to 0.542. This
means that about 39% of its performance above the measured floor does not survive family holdout.

## Table 5. Single-source check

Restricting the analysis to Gerasimavicius variants reproduces the main pattern without the G2P
additions.

The subset contains 10,138 variants from 942 genes and 666 families.

| Feature | Gene-split macro-F1, five-seed mean | Family-split macro-F1, five-seed mean | Family split, seed 0 (95% CI) |
|---|---:|---:|---:|
| wt_only_mean | 0.611 | 0.463 | 0.467 [0.387, 0.517] |
| delta_mean | 0.279 | 0.280 | 0.279 [0.256, 0.301] |

The delta remains at the subset's measured floor, while the wildtype embedding remains above it.

## Reading the tables

1. In Table 2, `delta_mean` has a family-split macro-F1 of 0.290, equal to the measured floor. A
   preregistered linear PCA probe cannot reliably assign the three mechanism labels from the
   mutation-induced change.
2. The family-split AUROC for `delta_mean` is highest for GOF at 0.584. This is weak ranking
   information, not useful three-class classification.
3. `wt_only_mean` falls from a five-seed mean of 0.552 in Table 1 to 0.449 in Table 2. Because this
   feature never sees the mutation, its performance reflects the protein and its family.
4. `mut_only_mean` performs almost identically to `wt_only_mean`, and combining the two embeddings
   does not improve the family-split result. The mutation adds no meaningful linear classification
   improvement.
5. In Table 3, nonlinear probes raise the delta above the floor, but the strongest family-split
   result remains below the linear wildtype result.
6. In Table 4, the wildtype embedding predicts Pfam family with 60.1% accuracy against a 4.37%
   baseline. Family identity is directly available to the mechanism classifier.

## Pre-registered claims

The interval and permutation checks distinguish floor-level classification from weak ranking
information.

### Table 6. Statistical checks by seed

The delta ranking test is significant in four seeds, while the wildtype gene-minus-family gap is
supported in four seeds.

| Seed | Delta permutation p-value | Families without a swap partner | Wildtype split gap | Split-gap 95% CI |
|---:|---:|---:|---:|---:|
| 0 | 0.029 | 18 | 0.046 | [-0.028, 0.121] |
| 1 | 0.003 | 14 | 0.140 | [0.043, 0.222] |
| 2 | 0.011 | 26 | 0.122 | [0.044, 0.189] |
| 3 | 0.003 | 16 | 0.116 | [0.035, 0.188] |
| 4 | 0.054 | 17 | 0.088 | [0.031, 0.146] |

Families without a same-size partner in their fold retain their observed labels in every
permutation. Every split-gap interval resamples the 1,144 Pfam families and applies each shared
bootstrap draw to both split arms.

### 2A-1. Preregistered linear PCA classification sits at the measured floor

The interval threshold is the measured family-split floor of 0.290 plus 0.05, giving 0.340. The
upper confidence bound for the preregistered linear PCA `delta_mean` probe is below this threshold
in all five seeds. In seed 0, macro-F1 is 0.290 with a 95% family-bootstrap interval of 0.276 to
0.305.

✅ **Affirmed.** The classification-floor criterion is met in all 5 seeds. Under the preregistered
linear PCA probe, `delta_mean` does not support reliable three-class mechanism classification under
family holdout.

### 2A-2. The preregistered linear mechanism delta has no detectable ranking signal

The family-block permutation sensitivity test uses fixed out-of-fold predictions and macro
one-vs-rest AUROC. Four of five seeds have p-values below 0.05.

❌ **Overturned.** The permutation sensitivity test detects ranking signal in 4 of 5 seeds, which
exceeds the 3-of-5 refutation threshold. The delta retains weak but reproducible ranking
information even though its three-class predictions remain at the measured floor.

### 2B. The wildtype gene-to-family gap is non-zero

The paired family-bootstrap interval excludes zero in four of five seeds.

✅ **Supported, not overturned.** Only 1 of 5 seed intervals spans zero. The gap is positive but
varies across seeds, from 0.046 to 0.140. The Gerasimavicius-only subset, which contains 666
families, gives a gap of 0.147 with a 95% family-bootstrap interval of -0.004 to 0.227. Its direction
is consistent with the merged analysis, but its interval spans zero and is not independently
conclusive. The leakage fraction in Table 4 is descriptive and does not adjudicate this claim.

## Interpretation

Frozen ESM-2 embeddings represent protein identity and family more strongly than mutation
mechanism. Protein families correlate with curated gene-level mechanisms, so an absolute protein
embedding can predict mechanism partly by recognizing the type of protein. Subtracting the
wildtype embedding removes most family information and most classification performance.

The result does not establish that the delta contains no mechanism-related information. AUROC,
nonlinear probes, and the exploratory alternative linear probe in `report_geometry.md` detect weak
signal, particularly for GOF. The confirmatory classification-floor conclusion is specific to the
preregistered linear PCA probe.

## What this is and is not

- This experiment tests frozen embeddings from one ESM-2 model size. It does not establish that
  mechanism is unrecoverable from other models or after fine-tuning.
- Mechanism labels are assigned at the gene level, although variants in one gene can act through
  different mechanisms. This mismatch can weaken a real variant-level signal.
- DN is the smallest class. Its class-specific estimates are less precise.
- The reported scores measure discrimination, not calibrated clinical probabilities.

## Provenance

Embeddings are four arrays with shape `(17770, 1280)`, row-aligned to 17,770 variant records. The
result files record the shared input fingerprints and commit
`b50295205940aca08ce3f733b651db684387e25e` with `commit_dirty: true`. The stored scientific-input
fingerprints match the final audited inputs.

| Result | Source |
|---|---|
| Linear probes, bootstrap intervals, and permutation tests | [`aggregate.json`](../../results/run_biorxiv/aggregate.json) and `family_split_baselines_seed{0..4}.json` |
| Measured floors | [`naive_baseline.json`](../../results/run_biorxiv/naive_baseline.json) |
| Nonlinear probes | `nonlinear_results_seed{0..4}.json` under `results/run_biorxiv/` |
| Family identity and nearest-neighbour measurements | [`family_clustering.json`](../../results/run_biorxiv/family_clustering.json) |
| Leakage fractions | [`leakage_fraction.json`](../../results/run_biorxiv/leakage_fraction.json) |
| Single-source check | [`single_source_gerasimavicius/aggregate.json`](../../results/run_biorxiv/single_source_gerasimavicius/aggregate.json) and `single_source_gerasimavicius/family_split_baselines_seed{0..4}.json` |

Execution status is recorded in [`PROGRESS.md`](../../biorxiv/PROGRESS.md).
