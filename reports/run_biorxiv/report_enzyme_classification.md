# Does ESM-2 encode enzyme type in wildtype embeddings?

**run_biorxiv · 2026-08-19** · ESM-2 `esm2_t33_650M_UR50D` · 1,451 genes ·
4 enzyme classes · 5 seeds. Confirmatory rules:
[`PREREGISTRATION_run_biorxiv.md`](../../biorxiv/PREREGISTRATION_run_biorxiv.md).

## Summary

ESM-2 wildtype embeddings distinguish kinase, protease, oxidoreductase, and non-enzyme genes after
related protein families are held out. The same classifier performs much better on enzyme type
than on disease mechanism, supporting the conclusion that the mechanism result is task-specific
rather than a general pipeline failure. Logistic regression performs better than the nonlinear
probe, although the preregistered equivalence test is underpowered.

## The question

The mechanism experiment found that a linear readout of the mutation delta sits at the measured
classification floor. This positive control asks whether the same embeddings and evaluation
pipeline recover a sequence property that is expected to be associated with protein structure.

Each gene is classified as kinase, protease, oxidoreductase, or non-enzyme from its unmutated,
mean-pooled ESM-2 embedding. A proteome-feature model provides a negative control.

## Setup

- Dataset: 1,451 genes, comprising 130 kinases, 68 proteases, 123 oxidoreductases, and 1,130
  non-enzymes.
- Primary representation: the 1,280-dimensional mean-pooled ESM-2 embedding of the wildtype
  protein.
- Negative control: 37 gene-level proteome features that do not directly represent protein fold.
- Probes: class-balanced logistic regression and a multilayer perceptron (MLP).
- Cross-validation: five folds and five seeds. Gene-split and family-split results are reported.
- Family split: 1,429 genes assigned to 835 Pfam family clusters for the embedding analysis.
- Confidence intervals: seed-0 out-of-fold predictions with 1,000 family-cluster bootstrap
  resamples.
- No-signal reference: majority-class macro-F1 of 0.219.

All probes are uncalibrated and measure discrimination rather than class probabilities.

## Glossary

| Term | Description | No-signal reference |
|---|---|---:|
| Wildtype embedding | Mean-pooled ESM-2 representation of the unmutated protein | Macro-F1 0.219 |
| Proteome features | Gene-level composition and population-genetics features | Macro-F1 0.218 |
| Gene split | No gene appears in both training and test data | Not applicable |
| Family split | No Pfam family appears in both training and test data | Not applicable |
| Macro-F1 | Classification score averaged equally across the four classes | About 0.219 |
| AUROC | One-versus-rest ranking score for an individual class | 0.500 |

## Table 1. Enzyme classification

The wildtype embedding supports enzyme classification after complete protein families are held
out.

| Representation | Split | Probe | Five-seed macro-F1 | Seed-0 macro-F1 (95% CI) |
|---|---|---|---:|---:|
| Wildtype embedding | Gene split | Logistic regression | 0.831 | Not reported |
| Wildtype embedding | Family split | Logistic regression | 0.779 | 0.788 [0.732, 0.817] |
| Wildtype embedding | Family split | MLP | 0.701 | 0.713 |
| Proteome features | Gene split | Logistic regression | 0.310 | Not reported |
| Proteome features | Family split | Logistic regression | 0.291 | 0.292 [0.257, 0.318] |
| Proteome features | Family split | MLP | 0.340 | 0.353 |
| *Majority-class reference* | *Not applicable* | *None* | *0.219* | *0.219* |

The embedding logistic-regression score drops by 0.052 from gene split to family split. The
reported leakage fraction, measured relative to the available score above the floor, is 8.5%.
Proteome features perform substantially below the embedding under family holdout.

## Table 2. Family-split performance by class

The embedding ranks every enzyme class well, including the smaller protease class.

| Class | Genes in full dataset | Embedding AUROC, seed 0 (95% CI) | Proteome-feature AUROC, seed 0 (95% CI) |
|---|---:|---:|---:|
| kinase | 130 | 0.951 [0.922, 0.973] | 0.641 [0.558, 0.707] |
| non-enzyme | 1,130 | 0.940 [0.918, 0.959] | 0.666 [0.620, 0.713] |
| oxidoreductase | 123 | 0.949 [0.920, 0.975] | 0.771 [0.715, 0.816] |
| protease | 68 | 0.900 [0.830, 0.960] | 0.509 [0.401, 0.619] |
| *No-signal reference* | *Not applicable* | *0.500* | *0.500* |

Protease is the smallest class and its interval is the least precise under the preregistered
rare-class caveat. The negative control contains some uneven class-level signal, especially for
oxidoreductases, but does not approach the embedding result.

## Table 3. Pre-registered findings

The absolute enzyme-classification gate and the comparison with mechanism classification are
affirmed; the linear-versus-nonlinear equivalence gate is underpowered.

| Claim | Preregistered criterion | Observed result | Verdict |
|---|---|---:|---|
| 2F | Family-split logistic-regression macro-F1 at least 0.70 | 0.788 [0.732, 0.817] | ✅ Affirmed |
| 2G | Enzyme minus mechanism macro-F1 at least +0.05 | +0.508 [+0.446, +0.539] | ✅ Affirmed |
| 2H | Absolute MLP-minus-logistic difference below 0.05 | -0.074 [-0.119, -0.042] | ⚠️ Failed, underpowered |

### 2F. Enzyme classification clears macro-F1 0.70

✅ **Affirmed.** Seed-0 family-split logistic-regression macro-F1 is 0.788 [0.732, 0.817]. The
complete interval is above the preregistered threshold of 0.70. The five-seed descriptive mean is
0.779.

### 2G. Enzyme classification exceeds mechanism classification

✅ **Affirmed.** On the shared family subset, enzyme macro-F1 is 0.788 and mechanism macro-F1 is
0.280. The difference is +0.508 [+0.446, +0.539], exceeding the preregistered minimum of +0.05
with the complete interval above zero.

### 2H. MLP and logistic regression differ by less than 0.05

⚠️ **Failed, underpowered.** The MLP-minus-logistic difference is -0.074 [-0.119, -0.042]. The MLP
performs worse, not better, but the interval crosses the equivalence boundary at -0.05. The result
does not establish that the two probes are within 0.05 of each other.

## Reading the tables

1. The wildtype embedding reaches family-split macro-F1 0.779 across five seeds. Enzyme type can be
   recovered after related protein families are removed from training.
2. Logistic regression outperforms the MLP by 0.074 on seed 0. A nonlinear readout is not needed
   to obtain the strongest result in this experiment.
3. The proteome-feature negative control reaches family-split macro-F1 0.291, compared with 0.779
   for the embedding. The task is not solved equally well by any gene-level feature set.
4. Enzyme classification exceeds mechanism classification by 0.508 macro-F1 on the shared family
   subset. The difference is much larger than the preregistered minimum.

## Interpretation

The positive control shows that ESM-2 wildtype embeddings contain enzyme-type information that can
be read out across protein families. The same family-held-out pipeline therefore recovers a
structural sequence property when that property is represented in the embedding.

This result supports the task-specific interpretation of the mechanism result. As reported in
[`report_mechanism.md`](report_mechanism.md), the linear mutation delta remains at the measured
mechanism-classification floor. The enzyme comparison does not show that mechanism labels are
error-free or that enzyme and mechanism classification have equal difficulty.

## What this is and is not

- This is a positive control for the embedding and evaluation pipeline, not a general-purpose
  enzyme classifier.
- The four labels are broad functional categories rather than a complete enzyme taxonomy.
- Macro-F1 gives each class equal weight despite the large non-enzyme majority.
- Protease has 68 genes, so its class-specific interval is less precise than the other intervals.
- Claim 2H tests numerical equivalence within 0.05. The MLP does not outperform logistic
  regression, but equivalence is not established under the written rule.

## Provenance

The result file was produced from commit `b50295205940aca08ce3f733b651db684387e25e`.

| Result | Source |
|---|---|
| Dataset, five-seed results, confidence intervals, and claims 2F to 2H | [`enzyme_classification_summary.json`](../../results/run_biorxiv/enzyme_classification/enzyme_classification_summary.json) |
| Mechanism comparison | [`aggregate.json`](../../results/run_biorxiv/aggregate.json) and [`report_mechanism.md`](report_mechanism.md) |
| Decision rules | [`PREREGISTRATION_run_biorxiv.md`](../../biorxiv/PREREGISTRATION_run_biorxiv.md) and [`RUNBOOK_biorxiv.md`](../../biorxiv/RUNBOOK_biorxiv.md) |

Execution status is recorded in [`PROGRESS.md`](../../biorxiv/PROGRESS.md).
