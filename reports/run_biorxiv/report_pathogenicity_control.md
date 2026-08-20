# Can the same ESM-2 delta predict pathogenicity?

**run_biorxiv · 2026-08-19** · ESM-2 `esm2_t33_650M_UR50D` · 24,384 variants ·
12,192 pathogenic / 12,192 benign · 1,802 genes · 5 seeds.
Confirmatory rules: [`PREREGISTRATION_run_biorxiv.md`](../../biorxiv/PREREGISTRATION_run_biorxiv.md).

## Summary

The ESM-2 mutation delta separates pathogenic from benign variants under both gene and protein-family
holdout. The signal comes from the mutation-induced change rather than the unmutated protein, and it
changes little when related protein families are held out. This positive control shows that the
embedding and probe pipeline can recover mutation-level information on a task where such signal is
expected.

## The question

The mechanism experiment found that the ESM-2 mutation delta does not support reliable
classification of loss-of-function, gain-of-function, and dominant-negative mechanisms. This
control asks whether the same representation can answer a simpler question: is a missense variant
pathogenic or benign?

Success would rule out a general inability of the embeddings or probes to detect mutation-level
signal. It would not explain why mechanism classification remains at the floor.

## Setup

- Dataset: 24,384 ClinVar missense variants, balanced to 12,192 pathogenic and 12,192 benign
  observations across 1,802 genes.
- Selection: at most 20 variants per gene per class. Only genes containing both classes were kept.
- Duplicate handling: protein-level substitutions were deduplicated before balancing. This removed
  1,219 duplicate rows, and one substitution with conflicting labels was dropped.
- Embedding filter: 1,063 variants could not be applied to the reference sequence. The remaining
  set was rebalanced, removing 293 additional rows and 22 genes that no longer contained both
  classes.
- Cross-validation: five folds and five random seeds. No gene appears in both training and test
  data.
- Family split: 24,176 variants with Pfam annotations were evaluated across 1,072 family clusters;
  208 variants without family annotations were excluded from that split.
- Probes: logistic regression and a multilayer perceptron (MLP).
- Confidence intervals: seed 0, 95% cluster bootstrap with 1,000 resamples. Gene-split intervals
  resample genes and family-split intervals resample Pfam families.

All probes are uncalibrated and measure discrimination rather than clinical risk.

## Glossary

| Term | Description | No-signal reference |
|---|---|---:|
| `delta_mean` | Mean-pooled mutant embedding minus wildtype embedding | AUROC 0.500 |
| `wt_only` | Mean-pooled embedding of the unmutated protein | AUROC 0.500 |
| Logistic regression | Linear classifier | Not applicable |
| MLP | Classifier with a nonlinear hidden layer | Not applicable |
| AUROC | Probability that a randomly selected pathogenic variant is ranked above a benign variant | 0.500 |

## Table 1. Pathogenicity discrimination

The mutation delta predicts pathogenicity and changes little after complete protein families are
held out.

| Feature | Probe | Gene-split AUROC, five-seed mean | Family-split AUROC, five-seed mean | Family split, seed 0 (95% CI) | Split drop |
|---|---|---:|---:|---:|---:|
| delta_mean | MLP | 0.887 | 0.885 | 0.886 [0.880, 0.891] | 0.003 |
| delta_mean | Logistic regression | 0.840 | 0.838 | 0.835 [0.828, 0.842] | 0.002 |
| wt_only | MLP | 0.535 | 0.525 | 0.527 [0.519, 0.536] | 0.010 |
| wt_only | Logistic regression | 0.530 | 0.518 | 0.516 [0.509, 0.523] | 0.012 |
| *No-signal reference* | *None* | *0.500* | *0.500* | *0.500* | *Not applicable* |

The split drop is the gene-split mean minus the family-split mean. A small drop means performance
is not substantially dependent on having related protein families in training.

## Pre-registered claim

The positive-control threshold is evaluated on the seed-0 family-split MLP result.

### 2C. Pathogenicity clears AUROC 0.85 under family holdout

The seed-0 `delta_mean` MLP AUROC is 0.886, with a 95% bootstrap interval from 0.880 to 0.891 after
resampling 1,072 Pfam families. The complete interval is above the preregistered threshold of 0.85.

✅ **Affirmed.** The CI lower bound is 0.880, which exceeds 0.85. The descriptive five-seed
family-split mean is 0.885.

## Reading the table

1. The `delta_mean` MLP reaches a family-split AUROC of 0.885. The mutation-induced embedding change
   distinguishes pathogenic from benign variants after related protein families are held out.
2. The MLP drops by 0.003 from gene split to family split. Logistic regression drops by 0.002.
   Family recognition contributes little to the pathogenicity result.
3. The unmutated protein embedding is near the 0.500 no-signal reference, reaching 0.525 with the
   MLP under family holdout. Protein identity alone does not explain the delta result.
4. The family-split MLP exceeds logistic regression by 0.047. Pathogenicity information is present
   in a linear readout, but the nonlinear probe recovers more of it and is the probe that clears the
   preregistered threshold.

## Table 2. Comparison with mechanism classification

The same delta supports pathogenicity prediction but not reliable three-class mechanism
classification.

| Task | Feature and probe | Family-split result | Reading |
|---|---|---:|---|
| Pathogenicity | delta_mean, MLP | AUROC 0.885 | Pathogenic and benign variants are separable |
| Mechanism | delta_mean, logistic regression | Macro-F1 0.290 | Equal to the measured classification floor |
| Mechanism | delta_mean, MLP | Macro-F1 0.375 | Weak nonlinear signal, below the wildtype result |

AUROC and macro-F1 measure different properties and are not numerically comparable. The comparison
is whether each task produces usable separation under family holdout. Full mechanism results are in
[`report_mechanism.md`](report_mechanism.md).

## Interpretation

The positive control succeeds because ESM-2's mutation delta contains information about whether a
substitution is damaging. That information transfers across protein families and is largely absent
from the wildtype embedding. The result rules out a general failure of the delta extraction,
cross-validation, or probe pipeline.

The control does not restore the stronger mechanism claim. As shown in
[`report_mechanism.md`](report_mechanism.md), the linear delta remains at the three-class
classification floor, although weak mechanism-ranking information is detectable.

## What this is and is not

- This is a balanced ClinVar evaluation. Its class balance does not represent pathogenicity
  prevalence in clinical sequencing.
- AUROC measures ranking, not calibrated probability. The reported outputs should not be read as
  patient-level risks.
- Confidence intervals come from seed 0. Five-seed means describe stability across fold
  assignments but do not share those intervals.
- The family-split result excludes 208 variants without Pfam annotations.
- Success on pathogenicity shows that the pipeline can recover mutation-level signal. It does not
  establish that the mechanism labels are sufficiently precise for variant-level learning.

## Provenance

The final analysis uses result version 4 and input fingerprints recorded in the result files. The
probe was rerun from the fingerprint-verified inputs at clean commit
`c9945b43dbc279af988ce888febd570fd1e2d5df`. The result files record `commit_dirty: false`, and
their stored scientific-input fingerprints match the audited inputs.

| Result | Source |
|---|---|
| Five-seed means, data accounting, and claim 2C | [`pathogenicity_control.json`](../../results/run_biorxiv/pathogenicity_control.json) |
| Seed-0 AUROCs and bootstrap intervals | [`pathogenicity_control_seed0.json`](../../results/run_biorxiv/pathogenicity_control_seed0.json) |
| Remaining seed results | `pathogenicity_control_seed{1..4}.json` under `results/run_biorxiv/` |
| Execution logs | [`step_5_3.log`](../../logs/biorxiv/step_5_3.log) and [`step_5_5.log`](../../logs/biorxiv/step_5_5.log) |

Execution status is recorded in [`PROGRESS.md`](../../biorxiv/PROGRESS.md).
