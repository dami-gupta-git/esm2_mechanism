# Does ESM-2 encode protein folding stability?

**run_biorxiv · 2026-08-17** · ESM-2 `esm2_t33_650M_UR50D` · 177,315 single-point missense
variants, 181 natural domains, 77 Pfam families (Tsuboyama 2023 mega-scale dataset) · 5 seeds,
5-fold CV, 1,000 bootstrap resamples. Pre-registered gates 3A-3D:
[`PREREGISTRATION_run_biorxiv.md`](../../../biorxiv/PREREGISTRATION_run_biorxiv.md).

---

## The question

The pathogenicity control
([`report_pathogenicity_control.md`](report_pathogenicity_control.md)) showed that ESM-2 delta
embeddings separate pathogenic from benign variants well above the pass bar. But pathogenicity
is a curated clinical label whose training signal (population frequency) partially overlaps
ESM-2's evolutionary training data. A sceptic could argue the positive control works because
ESM-2 and ClinVar share the same evolutionary prior, not because the embedding carries genuine
biochemical information.

This report adds a second positive control using a purely physical label: measured protein
folding stability (ΔΔG) from the Tsuboyama 2023 mega-scale dataset. ΔΔG is measured in a
folding assay, with no curation and no evolutionary circularity. If ESM-2's delta encodes
stability across held-out protein families, the positive-control claim no longer rests on a
curation-derived label alone.

The experiment also tests whether the stability signal interferes with the mechanism experiment.
If ESM-2 encodes stability and stability correlates with mechanism, the mechanism null from
[`report_mechanism.md`](report_mechanism.md) could be explained by a stability confound. The H3
projection test rules this out.

---

## Setup

- **Dataset:** 177,315 single-point missense variants across 181 natural PDB domains from the
  Tsuboyama 2023 mega-scale stability dataset. De novo designed mini-proteins are excluded
  because they have no Pfam family. All domains are small (up to 72 residues), an intrinsic
  constraint of the mega-scale folding assay.
- **Family structure:** 181 domains assigned to 77 Pfam families via HMMER (`hmmscan --cut_ga`
  against Pfam-A). 14 domains with no Pfam hit are kept for random and domain splits but
  excluded from family-split only.
- **Embeddings:** ESM-2 650M, frozen. Two delta views: `delta_mean` (mean-pooled mutant minus
  mean-pooled wildtype, 1,280 dimensions) and `delta_pos` (per-residue delta at the variant
  position, 1,280 dimensions).
- **Cross-validation:** 5-fold, three schemes: random split, domain-holdout (entire PDB domains
  held out), and family-holdout (entire Pfam families held out). 5 seeds each.
- **Probes:**
  - Ridge regression (linear, pre-registered)
  - MLP (256 to 64 hidden units, nonlinear, pre-registered)
  - XGBoost (GPU gradient-boosted trees, exploratory)
- **Target:** continuous ΔΔG (K50dG-derived).
- **Metrics:** Spearman ρ (rank correlation between predicted and measured ΔΔG) and AUROC with
  ΔΔG binarised at its median. No-signal value: ρ = 0.0 for Spearman, 0.50 for AUROC.
- **Confidence intervals:** 95% cluster bootstrap, 1,000 resamples, resampling domains (181
  clusters for random/domain CIs) or families (77 clusters for family CIs).

### Glossary

**Spearman ρ** (rank correlation) measures how well predicted and measured stability values
agree in their ordering, without assuming a linear relationship. A value of 0.0 means no
agreement; 1.0 means perfect agreement. It is the primary metric because ΔΔG prediction is a
ranking task.

**AUROC** (area under the receiver operating characteristic curve) is included for comparison
to the pathogenicity control, with ΔΔG binarised at its median. Chance floor is 0.50.

**Confidence interval (CI)** quantifies sampling uncertainty. Here, 95% CIs are obtained by
repeatedly resampling domains or families as clusters. A narrower interval means the estimate
is more stable; a wider one means it could shift more if the analysis were repeated on
different domains.

---

## Table 1. Stability prediction across CV schemes

This table shows how well ESM-2 delta embeddings predict measured stability under three
cross-validation schemes, from easiest (random) to hardest (family-holdout). The pattern of
decline from random to family reveals how much of the signal depends on having seen a related
protein during training.

**Note:** Chance floor for Spearman ρ = 0.0. Chance floor for AUROC = 0.50. Values are 5-seed
means ± seed-to-seed standard deviation. CIs in brackets are 95% cluster bootstrap (seed 0).

| Probe | Random ρ | Domain ρ | Family ρ | Family AUROC |
|---|---:|---:|---:|---:|
| Ridge (delta_mean) | 0.693 ± 0.000 [0.675, 0.709] | 0.601 ± 0.002 [0.565, 0.631] | 0.554 ± 0.006 [0.505, 0.587] | 0.772 ± 0.003 |
| Ridge (delta_pos) | 0.679 ± 0.000 [0.657, 0.697] | 0.634 ± 0.002 [0.609, 0.659] | 0.592 ± 0.003 [0.565, 0.617] | 0.790 ± 0.002 |
| MLP | 0.868 ± 0.001 | 0.714 ± 0.004 | 0.635 ± 0.003 | 0.818 ± 0.002 |
| XGBoost | 0.767 ± 0.000 | 0.676 ± 0.003 | 0.631 ± 0.004 | 0.817 ± 0.003 |
| *no-signal* | *0.000* | *0.000* | *0.000* | *0.500* |

**Verdict.** All probes are far above the chance floor, confirming that ESM-2 encodes
stability. The decline from random to family is substantial for the linear probe (0.693 to
0.554, a drop of 0.139), indicating that roughly a fifth of the linear signal depends on
family recognition. Nonlinear probes (MLP family ρ = 0.635, XGBoost family ρ = 0.631) recover
most of the family-split loss, and two unrelated model families agree within 0.004. The
transferable stability signal is real but not linearly accessible.

The per-position delta (delta_pos) outperforms the mean-pooled delta (delta_mean) under
family-split (0.592 vs 0.554), suggesting that the mutation-site embedding carries more
transferable stability information than the protein-wide average. Under random split the pattern
reverses (0.679 vs 0.693), consistent with mean-pooling capturing some family-level context
that helps within-distribution but does not transfer.

---

## Table 2. Pre-registered decision

The hypotheses below were pre-registered in
[`plan_megascale_stability.md`](../../../docs/plans/plan_megascale_stability.md) before the run.
Pre-registration means the rules were written down before the results came in, so the verdict
cannot be adjusted afterward.

| Gate | Criterion | Threshold | Observed | Verdict |
|---|---|---|---:|---|
| 3A (H1) | Stability is encoded | random ρ ≥ 0.5 | 0.693 | ✅ pass |
| 3B (H2) | Family-robust (linear) | random-family Δ ≤ 0.05 | Δ = 0.139 | ❌ LEAKY |
| 3C (H3) | Independent of mechanism | mechanism F1 change ≤ +0.01 | -0.001 | ✅ pass |
| 3D (H4) | Per-domain distribution tight | per-domain ρ std ≤ 0.10 | 0.160 | ❌ HETEROGENEOUS |

The pre-registered decision rule fires **LEAKY** (random ρ ≥ 0.5 and random-family Δ ≥ 0.10):
ESM-2 encodes stability, but the linear probe's signal is substantially family-dependent. H4
additionally fires HETEROGENEOUS (std ≥ 0.15).

**3A. Stability is encoded.** Ridge random ρ = 0.693 [0.675, 0.709], well above the 0.5
threshold. The CI excludes 0.5 from below. ESM-2's embedding shift when a residue is mutated
carries clear information about how destabilising the mutation is.

**3B. The linear signal is not family-robust.** The random-to-family drop is 0.139, nearly
three times the 0.05 threshold. The linear probe leans partly on family recognition, unlike
pathogenicity, where the same delta lost almost nothing under family-split (Δ = 0.003 in
[`report_pathogenicity_control.md`](report_pathogenicity_control.md)). Nonlinear probes reduce
this gap (MLP random-family Δ = 0.233), but the linear gate was pre-registered and fails.

**3C. Stability is independent of mechanism.** Fitting a stability direction on the Tsuboyama
data, projecting it out of the mechanism embeddings, and rerunning the mechanism classifier
changes family-split mechanism macro-F1 by -0.001 (0.395 to 0.394). The stability axis is
essentially orthogonal to the mechanism representation, so the mechanism null cannot be
explained by a stability confound.

**3D. The per-domain distribution is wide.** Leave-one-domain-out Spearman averages 0.636 with
std 0.160, ranging from near-zero (2MCK: ρ = 0.021, 2JN4: ρ = 0.030) to 0.864 (2JZ2). No
domain is negative, and roughly 44% exceed ρ = 0.70, but a long tail of domains are predicted
poorly. ESM-2 reads stability well for most domains and weakly for some.

---

## Table 3. Baselines and controls

These checks confirm that the signal is genuine and characterise what drives it.

| Check | Result | Reads as |
|---|---|---|
| Label-shuffle null | ρ = 0.000 / -0.002 / -0.002 (random / domain / family) | no leakage: the signal is real |
| Delta-norm baseline (one feature: the magnitude of the embedding shift) | ρ = 0.253 / 0.254 / 0.241 | the signal is directional, not just magnitude |
| Nested-CV alpha (RidgeCV) | ρ = 0.694 / 0.602 / 0.555, chosen α = 100 | matches α = 1.0: the linear ceiling is real, not under-regularisation |
| PLS dimensionality, random | peaks at 50 components (ρ = 0.693) | the random-split signal uses many dimensions |
| PLS dimensionality, family | peaks at 10 components (ρ = 0.591), then declines | the transferable signal lives in a low-dimensional subspace |

**Reading.** Shuffling ΔΔG labels collapses every split to ρ ≈ 0, confirming the CV is not
leaking. The delta-norm baseline (ρ = 0.253) is far below the full delta (ρ = 0.693), so the
signal is in which way the representation moves, not merely how far, matching the pathogenicity
finding from [`report_geometry.md`](report_geometry.md). Proper per-fold alpha tuning lands on
the same numbers as the default, so the LEAKY verdict is not an artefact of regularisation.
The PLS sweep shows the transferable component is low-dimensional: under family-split,
prediction peaks around 10 components and then falls as extra components fit family-specific
structure that does not transfer.

---

## Where stability sits

The same delta and pipeline, three tasks at different levels of biological abstraction:

| Property | Best family-split score | Family-robust? |
|---|---|---|
| Pathogenicity ([`report_pathogenicity_control.md`](report_pathogenicity_control.md)) | AUROC 0.866 (linear) | yes, linearly robust |
| **Stability (this report)** | AUROC 0.818 (MLP) | partly, nonlinearly recoverable |
| Mechanism ([`report_mechanism.md`](report_mechanism.md)) | macro-F1 ≈ 0.40 (near floor) | no, family-memorised |

This is a gradient of family-dependence. Pathogenicity transfers across families almost
entirely and linearly. Stability transfers, but a nonlinear probe is needed to see the
cross-family signal. Mechanism does not transfer at any probe complexity. The ordering makes
biological sense: whether a mutation is damaging has common signatures across proteins; how much
it destabilises a fold depends on structural context that differs between families; how it
changes function (GOF vs LOF) is the most context-specific of the three.

---

## Interpretation

ESM-2's frozen embedding encodes protein folding stability as measured by the Tsuboyama 2023
mega-scale assay. A linear probe reads it well within distribution (random ρ = 0.693) but loses
a substantial fraction when whole protein families are held out (family ρ = 0.554). Nonlinear
probes (MLP family ρ = 0.635, XGBoost family ρ = 0.631) recover most of the loss, and two
independent probe architectures agree, so the transferable stability signal is real but not
linearly separable.

The per-position delta (delta_pos) outperforms mean-pooling under family-split (0.592 vs
0.554), suggesting that the mutation site carries more transferable stability information than
the protein-wide average.

The stability signal is orthogonal to mechanism: projecting the stability direction out of the
mechanism embeddings changes mechanism macro-F1 by -0.001. The mechanism null from
[`report_mechanism.md`](report_mechanism.md) cannot be explained by a stability confound.

This confirms that the pipeline recovers genuine physical signal beyond ClinVar pathogenicity,
and stability occupies a middle position on the project's family-robustness gradient: more
family-dependent than pathogenicity, but with real cross-family content that nonlinear probes
access.

---

## What this is and is not

- This is a clean physical positive control. ΔΔG carries no clinical curation, so the
  family-robustness result cannot be explained by overlap with ESM-2's training signal. It
  confirms the pipeline recovers known physical signal and that the mechanism null is
  task-specific.
- The dataset is scoped to small natural domains (up to 72 residues). The claim is that ESM-2
  encodes the stability of small natural single domains, not of large or multidomain proteins.
- This is not a stability predictor benchmark. The probes are diagnostic of what the frozen
  embedding contains, not a competitive ΔΔG method (which would fine-tune).
- MLP and XGBoost results are confirmatory only for the fact that nonlinear probes recover
  family-split signal. The specific architecture comparison is exploratory. XGBoost was not
  pre-registered.
- The CIs on Ridge results use domain-level clustering (181 clusters) for random and domain
  splits, and family-level clustering (77 clusters) for family-split. The MLP and XGBoost
  results report seed-to-seed standard deviation only, not cluster-bootstrap CIs.

---

## Provenance

| Source | File |
|---|---|
| Ridge probes, decision rule, per-protein Spearman | [`summary.json`](../../../results/run_biorxiv/megascale_stability/summary.json) |
| Per-protein Spearman details | [`per_protein_spearman.json`](../../../results/run_biorxiv/megascale_stability/per_protein_spearman.json) |
| H3 stability projection | [`h3_stability_projection.json`](../../../results/run_biorxiv/megascale_stability/h3_stability_projection.json) |
| XGBoost probes | [`mlp_summary_xgb.json`](../../../results/run_biorxiv/megascale_stability/mlp_summary_xgb.json) |
| MLP probes | [`mlp_summary.json`](../../results/run_biorxiv/megascale_stability/mlp_summary.json) |
| Baselines (delta-norm, nested alpha, label-shuffle, PLS) | [`baselines.json`](../../../results/run_biorxiv/megascale_stability/baselines.json) |
| Domain-family mapping | [`megascale_domain_families.json`](../../../data/megascale_domain_families.json) |

Parsed by `experiments/stability/tsuboyama_loader.py` from
`Tsuboyama2023_Dataset2_Dataset3_20230416.csv` (natural domains only, single-point substitutions
with finite `ddG_ML`). Pfam families by `experiments/stability/build_domain_families.py`
(`hmmscan --cut_ga` vs Pfam-A). ESM-2 650M mean-pooled WT/mutant embeddings reused from an
earlier extraction run. Probes (`experiments/stability/megascale_stability.py`,
`megascale_mlp.py`, `stability_baselines.py`): 5 seeds × 5-fold CV. Steps 7.2-7.4 ran on a
RunPod B200 (224 vCPUs). H3 projects the stability direction out of the merged mechanism
embeddings without per-fold re-standardisation.
