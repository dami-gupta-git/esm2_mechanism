# Does ESM-2 encode enzyme type in wildtype embeddings?

**run_biorxiv · 2026-08-19** · ESM-2 `esm2_t33_650M_UR50D` · 1,451 labeled genes ·
1,429 genes in 835 Pfam clusters for the primary analysis · 4 enzyme classes · 5 seeds.
Confirmatory rules:
[`PREREGISTRATION_run_biorxiv.md`](../../biorxiv/PREREGISTRATION_run_biorxiv.md).

## Summary

ESM-2 wildtype embeddings distinguish kinase, protease, oxidoreductase, and non-enzyme genes when
related protein families are held out. Enzyme type is recovered much more accurately than disease
mechanism with the same evaluation framework. Logistic regression performs better than the MLP,
but the preregistered equivalence claim is not established.

## What was measured and why

Each gene was classified as kinase, protease, oxidoreductase, or non-enzyme from its unmutated,
mean-pooled ESM-2 embedding. The primary evaluation held out complete Pfam families, so a test gene
could not be classified from a close family member in the training set. A 33-feature gene-level
proteome matrix was evaluated as a negative control.

This experiment tests whether the embedding and probe pipeline recover a sequence-associated
property when such information is present. It complements the mechanism experiment reported in
[`report_mechanism.md`](report_mechanism.md).

## Setup

- The dataset contains 1,451 genes: 130 kinases, 68 proteases, 123 oxidoreductases, and 1,130
  non-enzymes.
- The primary representation is the 1,280-dimensional mean-pooled ESM-2 wildtype embedding.
- The probes are class-balanced logistic regression and a multilayer perceptron.
- Five-fold cross-validation was repeated over five seeds under gene and family splits.
- The embedding family-split analysis contains 1,429 Pfam-annotated genes in 835 clusters.
- The proteome-feature analysis contains 1,424 aligned genes; its family split contains 1,422 genes
  in 828 clusters.
- Seed-0 confidence intervals use 1,000 family-cluster bootstrap resamples.
- The majority-class macro-F1 references are 0.219 for the embedding cohort and 0.218 for the
  proteome-feature cohort.

The probes are uncalibrated and measure discrimination rather than class probabilities.

## Glossary

| Term | Description | No-signal reference |
|---|---|---:|
| Wildtype embedding | Mean-pooled ESM-2 representation of the unmutated protein | Macro-F1 0.219 |
| Proteome features | Gene-level constraint, abundance, interaction, and related features | Macro-F1 0.218 |
| Gene split | No gene appears in both training and test data | Not applicable |
| Family split | No Pfam family appears in both training and test data | Not applicable |
| Macro-F1 | Classification score averaged equally across the four classes | About 0.219 |
| AUROC | One-versus-rest ranking score for an individual class | 0.500 |

## Table 1. Enzyme classification

The wildtype embedding supports enzyme classification after complete protein families are held
out.

| Representation | Split | Probe | Five-seed macro-F1 | Seed-0 macro-F1 (95% CI) |
|---|---|---|---:|---:|
| Wildtype embedding | Gene split | Logistic regression | 0.832 | Not reported |
| Wildtype embedding | Family split | Logistic regression | 0.779 | 0.787 [0.732, 0.818] |
| Wildtype embedding | Family split | MLP | 0.701 | 0.713 |
| Proteome features | Gene split | Logistic regression | 0.310 | Not reported |
| Proteome features | Family split | Logistic regression | 0.291 | 0.292 [0.257, 0.318] |
| Proteome features | Family split | MLP | 0.340 | 0.353 |
| *Embedding majority-class reference* | *Not applicable* | *None* | *0.219* | *0.219* |
| *Proteome-feature majority-class reference* | *Not applicable* | *None* | *0.218* | *0.218* |

The embedding logistic-regression score decreases by 0.053 from gene split to family split. The
reported leakage fraction, measured relative to the score above the majority-class reference, is
8.6%. The corresponding proteome-feature leakage fraction is 21.1%. The proteome-feature
family-split score is 0.291, compared with 0.779 for the embedding.

## Table 2. Family-split performance by class

The embedding ranks every class above the AUROC no-signal value of 0.500.

| Class | Genes in full dataset | Embedding AUROC, seed 0 (95% CI) | Proteome-feature AUROC, seed 0 (95% CI) |
|---|---:|---:|---:|
| kinase | 130 | 0.951 [0.923, 0.973] | 0.641 [0.558, 0.707] |
| non-enzyme | 1,130 | 0.940 [0.917, 0.959] | 0.666 [0.620, 0.713] |
| oxidoreductase | 123 | 0.949 [0.920, 0.975] | 0.771 [0.715, 0.816] |
| protease | 68 | 0.899 [0.830, 0.960] | 0.509 [0.401, 0.619] |
| *No-signal reference* | *Not applicable* | *0.500* | *0.500* |

Protease is the smallest class and has the widest interval. The proteome features contain uneven
class-level signal, strongest for oxidoreductases, but do not approach the embedding result.

## Table 3. Preregistered findings

The absolute enzyme-classification gate and the comparison with mechanism classification are
affirmed. The linear-versus-nonlinear equivalence gate is not supported.

| Claim | Preregistered criterion | Observed result | Verdict |
|---|---|---:|---|
| 2F | Family-split logistic-regression macro-F1 at least 0.70 | 0.787 [0.732, 0.818] | ✅ Affirmed |
| 2G | Enzyme minus mechanism macro-F1 at least +0.05 | +0.507 [+0.447, +0.541] | ✅ Affirmed |
| 2H | Absolute MLP-minus-logistic difference below 0.05 | -0.074 [-0.118, -0.043] | ⚠️ Failed, underpowered |

### 2F. Enzyme classification clears macro-F1 0.70

✅ Affirmed. Seed-0 family-split logistic-regression macro-F1 is 0.787 [0.732, 0.818] after
resampling 835 Pfam clusters. The full interval is above the preregistered threshold of 0.70. The
five-seed descriptive mean is 0.779.

### 2G. Enzyme classification exceeds mechanism classification

✅ Affirmed. Across 835 shared Pfam clusters, enzyme macro-F1 is 0.787 and mechanism macro-F1 is
0.280. The difference is +0.507 [+0.447, +0.541], above the preregistered minimum of +0.05 with
the full interval above zero.

### 2H. MLP and logistic regression differ by less than 0.05

⚠️ Failed, underpowered. The MLP-minus-logistic difference is -0.074 [-0.118, -0.043]. The MLP
performs worse, but the interval overlaps the equivalence region at its -0.05 boundary. The result
does not establish that the two probes are within 0.05 of each other. The paired analysis contains
1,429 genes in 835 Pfam clusters; 999 of 1,000 bootstrap resamples were valid.

## Reading the tables

1. The wildtype embedding reaches family-split macro-F1 0.779 across five seeds, compared with a
   majority-class reference of 0.219.
2. Every embedding class has seed-0 AUROC of at least 0.899 under family holdout. The overall score
   is therefore not driven by one class.
3. Logistic regression exceeds the MLP by 0.074 on seed 0. A nonlinear readout does not improve
   this result.
4. The proteome-feature negative control reaches family-split macro-F1 0.291, compared with its
   majority-class reference of 0.218 and with 0.779 for the embedding.
5. Enzyme classification exceeds mechanism classification by 0.507 macro-F1 on the shared family
   subset.

## Interpretation

ESM-2 wildtype embeddings contain enzyme-type information that transfers across protein families.
The family-held-out pipeline therefore recovers a sequence-associated functional property when it
is represented in the embedding.

The comparison with [`report_mechanism.md`](report_mechanism.md) supports a task-specific reading
of the mechanism result. The same evaluation framework separates enzyme classes while the mutation
delta remains at the measured mechanism-classification floor. This comparison does not establish
that mechanism labels are error-free or that the two classification tasks have equal difficulty.

## Limitations

- This is a positive control for the embedding and evaluation pipeline, not a general-purpose
  enzyme classifier.
- The four labels are broad functional categories rather than a complete enzyme taxonomy.
- Macro-F1 gives each class equal weight despite the large non-enzyme majority.
- Protease has 68 genes, so its class-specific interval is less precise than the other intervals.
- Claim 2H tests equivalence within 0.05. The MLP does not outperform logistic regression, but
  equivalence is not established under the written rule.

## Provenance

This report was regenerated from the final result produced at clean commit
`c9945b43dbc279af988ce888febd570fd1e2d5df`. The result records fingerprints for the labeled gene
cohort, wildtype embeddings, Pfam assignments, proteome-feature rows and columns, and the mechanism
reference used for claim 2G.

| Source | File |
|---|---|
| Dataset, five-seed results, confidence intervals, fingerprints, and claims 2F to 2H | [`enzyme_classification_summary.json`](../../results/run_biorxiv/enzyme_classification/enzyme_classification_summary.json) |
| Execution log | [`rerun_clean.log`](../../results/run_biorxiv/enzyme_classification/rerun_clean.log) |
| Execution environment | [`environment_snapshot.txt`](../../results/run_biorxiv/enzyme_classification/environment_snapshot.txt) |
| Mechanism comparison | [`aggregate.json`](../../results/run_biorxiv/aggregate.json), [`family_split_baselines_seed0.json`](../../results/run_biorxiv/family_split_baselines_seed0.json), and [`mechanism_oof_cache_seed0.json`](../../results/run_biorxiv/mechanism_oof_cache_seed0.json) |
| Decision rules | [`PREREGISTRATION_run_biorxiv.md`](../../biorxiv/PREREGISTRATION_run_biorxiv.md) and [`RUNBOOK_biorxiv.md`](../../biorxiv/RUNBOOK_biorxiv.md) |

Execution status is recorded in [`PROGRESS.md`](../../biorxiv/PROGRESS.md).
