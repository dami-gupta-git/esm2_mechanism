# What is the shape of ESM-2's pathogenicity signal?

**run_biorxiv · 2026-08-19** · ESM-2 `esm2_t33_650M_UR50D` · 24,384 ClinVar
missense variants · 1,802 genes · 5 seeds. Confirmatory rules:
[`PREREGISTRATION_run_biorxiv.md`](../../biorxiv/PREREGISTRATION_run_biorxiv.md).

## Summary

Pathogenicity is encoded mainly by the direction of the mutation-induced embedding change, not
by how far the embedding moves. This directional signal transfers across protein families and is
only weakly explained by context-free amino-acid properties. ESM-2's conservation score explains
the signal better than the embedding delta, and the delta adds no measurable information beyond
conservation.

## The question

The pathogenicity control showed that the ESM-2 mutation delta separates pathogenic from benign
variants after related protein families are held out. This experiment asks what that signal
represents: distance, direction, substitution chemistry, or conservation.

It also compares how well the full delta transfers across held-out groups for pathogenicity,
mechanism, and protein stability.

## Setup

- Pathogenicity dataset: 24,384 balanced ClinVar missense variants across 1,802 genes. Family-split
  analyses use 24,176 variants assigned to 1,072 Pfam family clusters.
- Mechanism comparison: 17,770 variants assigned to 1,144 Pfam family clusters.
- Delta: the mutant mean-pooled embedding minus the wildtype mean-pooled embedding, with 1,280
  dimensions.
- Magnitude: the length of the delta, represented by one number.
- Direction: the delta divided by its length, retaining which way the embedding changed.
- Cross-validation: five folds and five seeds, with whole protein families held out together.
- Confidence intervals: 95% family-cluster bootstrap with 1,000 resamples where reported.
- Confirmatory inference: seed 0 for claims 2D and 2E. Five-seed results are descriptive.

All probes are uncalibrated and measure discrimination rather than clinical risk.

## Glossary

| Term | Description | No-signal reference |
|---|---|---:|
| Full delta | Complete mutant-minus-wildtype embedding change | AUROC 0.500 |
| Magnitude | How far the embedding moves | AUROC 0.500 |
| Direction | Which way the embedding moves | AUROC 0.500 |
| AUROC | Probability that a positive example is ranked above a negative example | 0.500 |
| Macro-F1 | Average classification score across mechanism classes | 0.290 measured floor |
| Masked-marginal score | How much more ESM-2 expects the wildtype residue than the mutant residue | Not applicable |

## Table 1. Magnitude and direction for pathogenicity

The directional component retains the pathogenicity signal, while magnitude alone is much
weaker.

| Feature | Logistic-regression AUROC | MLP AUROC |
|---|---:|---:|
| Full delta | 0.838 [0.831, 0.845] | 0.885 [0.880, 0.891] |
| Magnitude | 0.610 [0.603, 0.619] | 0.610 [0.603, 0.619] |
| Direction | 0.855 [0.848, 0.862] | 0.892 [0.887, 0.898] |
| *No-signal reference* | *0.500* | *0.500* |

Values are five-seed family-split means with 95% family-bootstrap intervals. Direction slightly
outperforms the full delta for both probes. Magnitude retains some signal but is substantially
weaker.

## Table 2. Magnitude and direction for mechanism classification

The same decomposition does not reveal a strong mechanism signal.

| Feature | Logistic-regression macro-F1 | MLP macro-F1 |
|---|---:|---:|
| Full delta | 0.387 | 0.375 |
| Magnitude | 0.256 | 0.319 |
| Direction | 0.382 | 0.385 |
| *Measured chance floor* | *0.290* | *0.290* |

Values are five-seed family-split means. Direction performs similarly to the full delta, while
magnitude is at or near the measured floor. This analysis does not replace the preregistered
mechanism tests in [`report_mechanism.md`](report_mechanism.md).

## Table 3. Exploratory geometry of the pathogenicity direction

The signal is distributed across the embedding but transfers to previously unseen families.

| Quantity | Result |
|---|---:|
| Full linear AUROC | 0.838 ± 0.007 |
| AUROC after removing 1 direction | 0.841 ± 0.009 |
| AUROC after removing 2 directions | 0.841 ± 0.009 |
| AUROC after removing 3 directions | 0.835 ± 0.009 |
| AUROC after removing 4 directions | 0.833 ± 0.008 |
| AUROC after removing 5 directions | 0.828 ± 0.008 |
| Similarity of directions fit on separate family halves | 0.218 ± 0.026 |
| Shuffled-label similarity reference | 0.002 ± 0.028 |
| Transfer AUROC from family half A to half B | 0.823 ± 0.006 |

Removing five fitted directions reduces AUROC by only 0.010. Directions fitted on separate family
halves have low direct similarity but still predict pathogenicity in the other half. A binary
linear classifier is one-dimensional by construction, so these results do not establish that the
biological signal itself is a single axis.

## Table 4. Exploratory transfer across tasks

Pathogenicity and stability transfer across held-out groups more strongly than mechanism.

| Task | Probe | Group-CV AUROC | Half-group transfer AUROC | Difference |
|---|---|---:|---:|---:|
| Pathogenicity | Linear | 0.837 | 0.823 | 0.015 |
| Pathogenicity | GBM | 0.891 | 0.886 | 0.005 |
| Mechanism, GOF versus rest | Linear | 0.644 | 0.620 | 0.023 |
| Mechanism, GOF versus rest | GBM | 0.655 | 0.636 | 0.020 |
| Stability, ΔΔG above median | Linear | 0.811 | 0.802 | 0.009 |
| Stability, ΔΔG above median | GBM | 0.847 | 0.842 | 0.005 |

Values are five-seed means. Group cross-validation and half-group transfer use different training
set sizes, so the difference is descriptive and is not an isolated estimate of transfer failure.

## Table 5. Exploratory biochemical explanation

Simple amino-acid substitution properties explain only a small part of the pathogenicity axis.

| Feature | Spearman correlation with axis |
|---|---:|
| BLOSUM62 score | -0.275 |
| Absolute hydropathy change | 0.221 |
| Absolute volume change | 0.195 |
| Absolute charge change | 0.103 |
| Delta magnitude | 0.336 |

| Predictor | Family-split AUROC |
|---|---:|
| Context-free biochemical features | 0.703 ± 0.009 |
| ESM-2 delta | 0.838 ± 0.007 |
| ESM-2 delta plus biochemical features | 0.852 ± 0.007 |

The four biochemical features explain 7.2% of held-out axis-score variance. Their AUROC is above
chance but below the ESM-2 delta, indicating that sequence context contributes information beyond
the amino-acid substitution alone.

## Table 6. Conservation and the embedding delta

Conservation alone predicts pathogenicity better than the full embedding delta.

| Feature set | Seed-0 family-split AUROC (95% CI) |
|---|---:|
| Conservation features | 0.888 [0.881, 0.895] |
| Masked-marginal score alone | 0.888 [0.881, 0.895] |
| Embedding delta | 0.835 [0.828, 0.842] |
| Conservation plus embedding delta | 0.883 [0.876, 0.890] |

| Paired comparison | Difference (95% CI) |
|---|---:|
| Conservation minus delta | +0.052 [+0.047, +0.058] |
| Conservation plus delta minus conservation | -0.005 [-0.008, -0.001] |
| Conservation plus delta minus delta | +0.048 [+0.044, +0.051] |

The held-out correlation between the masked-marginal score and the pathogenicity axis is 0.684.
The full conservation feature set and the single masked-marginal score perform almost identically.

## Pre-registered findings

### 2D. Conservation alone clears AUROC 0.85

✅ **Affirmed.** Conservation reaches seed-0 AUROC 0.888 [0.881, 0.895]. The complete interval is
above the preregistered threshold of 0.85. The five-seed descriptive mean is 0.888.

### 2E. Adding the embedding delta improves AUROC by at least 0.02

❌ **Failed, established.** Adding the delta changes AUROC by -0.005 [-0.008, -0.001]. The point
estimate does not reach the preregistered improvement of +0.02, and the complete interval is below
both zero and the threshold. The result is not underpowered for an improvement of that size.

## Reading the tables

1. Direction-only performance matches or exceeds the full delta, while magnitude is much weaker.
   Pathogenicity depends mainly on which way the embedding changes.
2. Removing several fitted directions produces little performance loss. The information is
   distributed across multiple embedding dimensions rather than confined to one fitted vector.
3. Context-free amino-acid properties predict some pathogenicity, but they do not account for most
   of the embedding signal.
4. Conservation alone exceeds the delta by 0.052 AUROC. Adding the delta to conservation reduces
   AUROC by 0.005 rather than improving it.

## Interpretation

The mutation delta predicts pathogenicity because its direction reflects whether a substitution
conflicts with the sequence context learned by ESM-2. The masked-marginal conservation score
captures this information more directly and performs better than the complete mean-pooled delta.
Within this experiment, the delta provides no additional pathogenicity discrimination after
conservation is included.

The result characterizes the pathogenicity signal but does not alter the mechanism finding. The
same direction-based decomposition remains weak for three-class mechanism classification, as
reported in [`report_mechanism.md`](report_mechanism.md). The underlying pathogenicity result is
reported in [`report_pathogenicity_control.md`](report_pathogenicity_control.md).

## What this is and is not

- The magnitude, direction-ablation, transfer, and biochemistry analyses are exploratory. Claims
  2D and 2E are the preregistered tests in this report.
- AUROC measures ranking, not calibrated clinical risk.
- Claims 2D and 2E use seed-0 held-out-fold estimates and family-cluster intervals. Five-seed means
  are descriptive.
- Family-split analyses exclude 208 variants without Pfam annotations.
- The transfer comparison uses different training-set sizes for group CV and half-group transfer.
- Conservation is measured by ESM-2's masked-token probabilities, not by an external evolutionary
  alignment.

## Provenance

The result files were produced from commit `b50295205940aca08ce3f733b651db684387e25e`.

| Result | Source |
|---|---|
| Magnitude and direction | [`probe_results.json`](../../results/run_biorxiv/magnitude_direction/probe_results.json) |
| Direction ablation and cross-family transfer | [`geometry_results.json`](../../results/run_biorxiv/magnitude_direction/geometry_results.json) |
| Transfer comparison across tasks | [`transfer_contrast.json`](../../results/run_biorxiv/magnitude_direction/transfer_contrast.json) |
| Biochemical explanation | [`probe4_axis_identity.json`](../../results/run_biorxiv/magnitude_direction/probe4_axis_identity.json) |
| Conservation and claims 2D/2E | [`conservation_axis.json`](../../results/run_biorxiv/magnitude_direction/conservation_axis.json) |
| Canonical pathogenicity set | [`pathogenicity_valid_variants_canonical.json`](../../data/pathogenicity_valid_variants_canonical.json) |
| Execution logs | [`step_6_2.log`](../../logs/biorxiv/step_6_2.log), [`step_6_5.log`](../../logs/biorxiv/step_6_5.log), and [`step_6_7.log`](../../logs/biorxiv/step_6_7.log) |

Execution status is recorded in [`PROGRESS.md`](../../biorxiv/PROGRESS.md).
