# Does ESM-2 encode protein folding stability?

**run_biorxiv · 2026-08-19** · ESM-2 `esm2_t33_650M_UR50D` · 177,315 single-point
missense variants · 181 natural domains · 77 Pfam families · 5 seeds.
Decision rules: [`PREREGISTRATION_run_biorxiv.md`](../../biorxiv/PREREGISTRATION_run_biorxiv.md)
and [`RUNBOOK_biorxiv.md`](../../biorxiv/RUNBOOK_biorxiv.md), Section 7.

## Summary

Mutation-induced ESM-2 embedding changes predict measured folding stability, including when
complete protein families are held out. The preregistered linear signal loses more performance
under family holdout than the registered tolerance permits, and performance varies substantially
among domains. Removing the fitted stability direction from the mechanism representation does not
improve mechanism classification.

## What was measured and why

The experiment predicts the change in folding stability caused by a point mutation using the
Tsuboyama 2023 mega-scale assay. This provides a mutation-level physical target that is independent
of clinical label curation. It tests whether the frozen embedding and evaluation pipeline recover
mutation-induced information outside the pathogenicity task.

The experiment also tests whether a stability direction masks mechanism information. A stability
direction fitted on the Tsuboyama cohort was removed from the mechanism delta before repeating a
family-split mechanism classifier.

## Setup

- The dataset contains 177,315 variants across 181 natural PDB domains. Fourteen domains without a
  Pfam assignment are excluded only from family-split evaluation.
- `delta_mean` is the protein-wide mean mutant-minus-wildtype embedding change. `delta_pos` is the
  embedding change at the mutated residue.
- The preregistered probe is ridge regression under random, domain, and Pfam-family splits.
- MLP, random-forest, XGBoost, and baseline analyses are exploratory and do not adjudicate claims
  3A through 3D.
- Spearman correlation is the primary metric. Its no-signal reference is 0.000. AUROC after
  splitting the stability target at its median has a no-signal reference of 0.500.
- Seed-0 confidence intervals use 1,000 cluster-bootstrap resamples. Random and domain intervals
  resample 181 domains; family intervals resample 77 Pfam families.

## Glossary

| Term | Description | No-signal reference |
|---|---|---:|
| `delta_mean` | Mean-pooled mutant embedding minus mean-pooled wildtype embedding | Spearman 0.000 |
| `delta_pos` | Mutant-minus-wildtype embedding change at the substituted residue | Spearman 0.000 |
| Random split | Variants are divided without holding out domains or families | Not applicable |
| Domain split | Complete PDB domains are held out | Not applicable |
| Family split | Complete Pfam families are held out | Not applicable |
| Spearman correlation | Agreement between predicted and measured stability rankings | 0.000 |
| AUROC | Ranking after dividing the stability target at its median | 0.500 |

## Table 1. Stability prediction across held-out units

Stability remains predictable under domain and family holdout, but every probe performs best under
the random split.

| Probe and feature | Random Spearman, five-seed mean | Domain Spearman, five-seed mean | Family Spearman, five-seed mean | Family Spearman, seed 0 (95% CI) | Family AUROC, five-seed mean |
|---|---:|---:|---:|---:|---:|
| Ridge, `delta_mean` | 0.693 | 0.601 | 0.554 | 0.544 [0.501, 0.583] | 0.774 |
| Ridge, `delta_pos` | 0.679 | 0.634 | 0.592 | 0.590 [0.559, 0.622] | 0.792 |
| MLP, `delta_mean` | 0.868 | 0.703 | 0.627 | 0.623 [0.590, 0.652] | 0.816 |
| Random forest, `delta_mean` | 0.731 | 0.641 | 0.588 | Not reported | 0.800 |
| XGBoost, `delta_mean` | 0.767 | 0.676 | 0.630 | Not reported | 0.819 |
| *No-signal reference* | *0.000* | *0.000* | *0.000* | *0.000* | *0.500* |

The MLP random, domain, and family seed-0 correlations are 0.868 [0.857, 0.878], 0.697
[0.667, 0.726], and 0.623 [0.590, 0.652], respectively. Its random-to-family decrease is
0.241. The exploratory nonlinear result therefore raises absolute family-held-out performance but
does not remove split sensitivity.

## Table 2. Preregistered findings

The linear probe detects stability, but the registered family-robustness and domain-homogeneity
criteria fail.

| Claim | Preregistered criterion | Observed result | Verdict |
|---|---|---:|---|
| 3A | Random-split Spearman at least 0.50 | 0.693 [0.675, 0.709] | ✅ Affirmed |
| 3B | Family-robust at most 0.05; `LEAKY` at least 0.10 | 0.153 [0.112, 0.192] | ❌ Failed, `LEAKY` |
| 3C | Projected-minus-baseline mechanism macro-F1 at most +0.01 | -0.0009 [-0.0025, +0.0007] | ✅ Affirmed |
| 3D | Per-domain Spearman standard deviation at most 0.10 | 0.160 [0.132, 0.183] | ❌ Failed |

### 3A. Stability is recoverable from the mean delta

✅ Affirmed. Seed-0 random-split Spearman correlation is 0.693 [0.675, 0.709]. The complete
interval is above the registered threshold of 0.50.

### 3B. The linear stability signal is not family-robust under the registered rule

❌ Failed. On the seed-0 cohort shared by the random and family arms, the random-minus-family
correlation difference is 0.153 [0.112, 0.192]. The complete interval is above the 0.10 `LEAKY`
boundary. The descriptive five-seed means are 0.693 under the random split and 0.554 under the
family split.

The originating plan uses two boundaries: a decrease of at most 0.05 supports family robustness,
and a decrease of at least 0.10 triggers `LEAKY`; the interval between them is not adjudicated.
The stored result encodes 0.10 as a single upper-bound gate, but the observed interval exceeds
0.10, so this implementation difference does not change the verdict. The audit trail is recorded
in [`PREREGISTRATION_run_biorxiv.md`](../../biorxiv/PREREGISTRATION_run_biorxiv.md), Part 3.

### 3C. Removing the stability direction does not improve mechanism classification

✅ Affirmed. The five-seed mechanism macro-F1 changes from 0.3949 before projection to 0.3941
after projection. The seed-0 projected-minus-baseline difference is -0.0009 [-0.0025, +0.0007],
which is below the registered upper limit of +0.01.

This control uses its registered full-dimensional, standardized, class-balanced mechanism
classifier. It is distinct from the 256-component preregistered mechanism probe reported in
[`report_mechanism.md`](report_mechanism.md).

### 3D. Stability performance varies among domains

❌ Failed. The per-domain correlation standard deviation is 0.160 [0.132, 0.183], above the
registered maximum of 0.10. The mean per-domain correlation is 0.636. Correlations range from
0.021 to 0.864; none of the 181 domains has a negative correlation, and 79 have a correlation of
at least 0.70.

The combined registered outcome is `LEAKY` because the 3B family-dependence condition fires. Claim
3D independently identifies heterogeneous performance among domains.

## Table 3. Exploratory controls

The controls support the measured stability signal and show that its transferable component is not
captured by delta magnitude alone.

| Check | Random Spearman | Domain Spearman | Family Spearman | Reading |
|---|---:|---:|---:|---|
| Delta magnitude | 0.253 | 0.254 | 0.241 | Magnitude alone retains part of the signal |
| Nested ridge regularization | 0.694 | 0.602 | 0.555 | Tuning the ridge penalty does not remove the split pattern |
| Shuffled labels | 0.000 | -0.002 | -0.002 | Performance returns to the no-signal reference |

In a component sweep, family-split correlation peaks at 0.591 with 10 components and falls to
0.544 with 50 components. Random-split correlation continues rising through 50 components,
reaching 0.693.

## Reading the tables

1. The preregistered mean-delta ridge probe reaches family-split correlation 0.554, compared with
   the 0.000 no-signal reference. Stability information transfers to held-out families.
2. The matched random-to-family difference is 0.153, and its interval remains above the registered
   tolerance of 0.10. The linear result is partly dependent on the held-out unit.
3. The mutation-site delta reaches family-split correlation 0.592, compared with 0.554 for the
   mean-pooled delta.
4. The exploratory MLP reaches family-split correlation 0.627, but it also decreases by 0.241 from
   random to family evaluation. Higher absolute performance does not establish family robustness.
5. Removing the fitted stability direction changes mechanism macro-F1 by less than one thousandth.
   Stability does not account for the mechanism classifier's limited performance under this test.

## Interpretation

Frozen ESM-2 mutation deltas contain information about experimentally measured folding stability.
This information remains detectable across held-out domains and Pfam families, providing a
physical-property positive control alongside the pathogenicity result in
[`report_pathogenicity_control.md`](report_pathogenicity_control.md).

The result is not uniformly family-robust. The preregistered linear random-to-family decrease
exceeds its tolerance, and correlations vary across domains. Exploratory MLP, random-forest, and
XGBoost probes all remain above the no-signal reference under family holdout, showing that the
family-held-out signal is not limited to one probe architecture.

The stability projection does not improve mechanism classification. This supports a task-specific
interpretation of [`report_mechanism.md`](report_mechanism.md): limited mechanism classification is
not explained by a dominant folding-stability direction in the mean-pooled delta.

## Limitations

- The Tsuboyama cohort contains short natural domains and does not represent large or multidomain
  proteins.
- This is a representation probe, not a benchmark of competitive stability predictors.
- Fourteen domains without Pfam annotations are excluded from family-split evaluation.
- Pfam-family holdout reduces direct family overlap but does not remove all evolutionary
  relatedness.
- The MLP, random-forest, XGBoost, and baseline results are exploratory. Claims 3A through 3D are
  adjudicated using the registered linear analysis.

## Provenance

The final result files were produced at clean commit
`6937c85bfb90269ae0451b2fe4684caf5c6a6f0f` and record `commit_dirty: false`.

| Source | File |
|---|---|
| Linear probes, confidence intervals, combined verdict, and claims 3A to 3D | [`summary.json`](../../results/run_biorxiv/megascale_stability/summary.json) |
| Per-domain correlations | [`per_protein_spearman.json`](../../results/run_biorxiv/megascale_stability/per_protein_spearman.json) |
| Stability projection for claim 3C | [`stability_projection_3c.json`](../../results/run_biorxiv/megascale_stability/stability_projection_3c.json) |
| MLP and random-forest probes | [`mlp_summary.json`](../../results/run_biorxiv/megascale_stability/mlp_summary.json) |
| XGBoost probe | [`mlp_summary_xgb.json`](../../results/run_biorxiv/megascale_stability/mlp_summary_xgb.json) |
| Baselines and component sweep | [`baselines.json`](../../results/run_biorxiv/megascale_stability/baselines.json) |
| Domain-to-family mapping | [`megascale_domain_families.json`](../../data/megascale_domain_families.json) |
| Step 7.2 execution log | [`step_7_2.log`](../../logs/biorxiv/step_7_2.log) |
| Step 7.3 execution log | [`megascale_mlp_73.log`](../../logs/biorxiv/megascale_mlp_73.log) |
| Step 7.4 execution log | [`step_7_4.log`](../../logs/biorxiv/step_7_4.log) |
| Step 7.5 execution log | [`step_7_5.log`](../../logs/biorxiv/step_7_5.log) |
| Execution environments | [`ENV_SNAPSHOT.md`](../../biorxiv/ENV_SNAPSHOT.md) |

Execution status is recorded in [`PROGRESS.md`](../../biorxiv/PROGRESS.md).
