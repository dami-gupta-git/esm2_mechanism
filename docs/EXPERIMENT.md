# ESM-2 Delta-Embedding Mechanism Geometry

> **Scope note (added 2026-05-28):** This document is the pre-registration for results 1–10 (the frozen ESM-2 characterisation arc). Results 11–23 (gene-level proteome features, Badonyi structural priors, within-family analysis, pathogenicity geometry) were designed post-hoc as follow-up arcs and are not covered here. See `docs/README.md` for the full results index. An **Outcomes** section is appended at the end of this document recording how each pre-registered element resolved.

## Overview

Protein language models like ESM-2 encode evolutionary and biophysical constraints in their sequence representations. When a missense mutation occurs, the difference between the mutant and wildtype ESM-2 embeddings — a delta-embedding — captures how the model's representation of the protein shifts in response to the amino acid change. Prior work has established that pathogenic mutations differ systematically by molecular disease mechanism at the structural level: loss-of-function mutations are destabilizing, dominant-negative mutations cluster at protein interfaces, and gain-of-function mutations occur in disordered regions with mild stability effects (Gerasimavicius et al. 2022). Prior work has identified a dominant stability/constraint axis in ESM-2 delta-embedding space [citation needed]. What no study has asked is whether mechanism-specific geometry survives in ESM-2 delta space after the stability axis is removed — and whether GOF, DN, and LOF occupy geometrically independent directions.

The central question: **do ESM-2 delta-embeddings encode gene-level dominant disease mechanism beyond protein stability, and are the mechanism axes orthogonal or correlated?**

If the probe succeeds, the finding is that protein language models encode not just whether a mutation is damaging, but how it is damaging — and this mechanistic geometry is recoverable from sequence alone, without structural information.

---

## Data

**Gerasimavicius et al. 2022** (Nature Communications 13:3895, OSF: 10.17605/OSF.IO/H62FQ)

**Local copy:** `../data/DiseaseMech_Stability_VEPS.xlsx` (233MB, gitignored). Sheet used: `ClinVar_gene_level`. SCP to new pods rather than re-downloading from OSF.

- ~10,200 pathogenic ClinVar missense variants (GOF: 1,983 / DN: 894 / HI: 1,678 / AR: 5,678)
- ~948 Mendelian disease genes (GOF: 81 / DN: 60 / HI: 82 / AR: 725)
- Sheet used: `ClinVar_gene_level`, `Disease_mechanism` column
- Gene-level mechanism labels: GOF (gain-of-function), DN (dominant-negative), HI (haploinsufficiency), AR (autosomal recessive)

| Label | Full name | Mechanism |
|---|---|---|
| GOF | Gain-of-function | Mutation creates new or enhanced activity — protein does something it shouldn't. Example: KRAS G12D stays constitutively active. |
| DN | Dominant-negative | Mutant protein poisons the wildtype — interferes with oligomers or complexes. One bad copy breaks the whole complex. Example: p53 tetramer with one mutant subunit. |
| HI | Haploinsufficiency | One working copy isn't enough — loss of one allele reduces dosage below threshold. No toxic product, just insufficient protein. |
| AR | Autosomal recessive | Both copies must be lost for disease — one working copy is sufficient. Loss-of-function but only manifests when homozygous. |

In the 3-class analysis, HI and AR are collapsed to **LOF** because both result from reduced/absent protein activity. The key question is whether GOF, DN, and LOF leave distinguishable fingerprints in ESM-2 delta space — structurally they affect different parts of the protein (active sites for GOF, interfaces for DN, anywhere for LOF).
- FoldX ddG provided for each variant
- Labels are gene-level, not variant-level: all variants from a gene carry the gene's dominant mechanism label. This is a known limitation — the experiment tests gene-level dominant mechanism, not per-variant mechanism. The headline claim is scoped accordingly.

**Primary 3-class analysis:** HI and AR are collapsed to LOF (loss-of-function), giving GOF / DN / LOF.

**Secondary analyses:** 4-class GOF/DN/HI/AR; HI vs AR 2-class.

**Per-class sample sizes must be counted before running** (n variants and n genes per class). GOF is expected to be the smallest class by a wide margin. If GOF has fewer than 300 variants or fewer than 50 genes, switch from 5-fold gene-split CV to leave-one-gene-out CV or stratified k-fold with k chosen by the smallest class, and document this before seeing any results.

---

## Experimental Design

### 1. Delta-embedding computation

For each variant, fetch the canonical wildtype protein sequence from UniProt and apply the missense substitution to produce the mutant sequence. Extract ESM-2 650M embeddings for both sequences and subtract: **delta = mutant − wildtype**.

Two representations are computed and treated as co-primary:
- **Mean-pooled delta**: average over all residue positions in the sequence (1280-dim). Captures whole-protein representation shift.
- **Per-residue delta at variant position**: embedding at the mutated residue only (1280-dim). Captures local context shift.

If these two representations disagree on the headline result, mean-pooled is pre-registered as the cited finding.

For proteins longer than 1022 tokens (ESM-2's limit), a window of +/-500 residues centered on the variant position is extracted. The identical window is applied to both wildtype and mutant sequences.

### 2. Stability nuisance subspace removal

ESM-2 delta space encodes a strong stability/conservation axis. To test whether mechanism geometry is independent of stability, this axis is projected out before probing. The projection has two possible paths depending on whether a Megascale-fit stability subspace transfers to Gerasimavicius variants. Path A (the disjoint-dataset path) supports stronger claims about probe-to-stability orthogonality; Path B (the same-dataset fallback) does not — see Section 7 for why this matters.

**Path A (primary):** Fit a multivariate stability subspace on the Megascale dataset (Tsuboyama et al. 2023, ~200k ddG measurements across ~500 protein domains). The subspace is defined by regressing each of the 1280 delta dimensions on ddG, then taking the regression direction plus PCA components of the residuals (10 components total). This subspace is then applied to the Gerasimavicius variants.

*Validation:* Before using the Megascale-fit subspace, validate transfer by computing the Spearman correlation between projections onto the subspace's primary direction and FoldX ddG values on the Gerasimavicius variants. Pre-registered threshold: rho > 0.3. If the transfer passes, Path A proceeds. If it fails, fall back to Path B.

**Path B (fallback):** Fit the stability subspace directly on Gerasimavicius variants using FoldX ddG as the target, with leave-one-gene-out CV on the subspace fit to prevent data leakage into the probe.

Results are reported both with and without stability projection in both paths.

**Pre-registered prediction:** GOF variants will have 30%+ less variance explained by the stability subspace than HI+AR (LOF) variants, consistent with Gerasimavicius's finding that GOF mutations have milder structural effects. Operationalization: compute a bootstrap CI (n=1000, gene-level resampling) on the ratio (GOF variance explained) / (LOF variance explained). The prediction holds if the upper bound of the 95% CI is < 0.70.

**Limitation:** "Projecting out the stability subspace" means projecting out the FoldX- and regression-defined approximation of stability. FoldX has known systematic biases (over-weights packing density, under-weights solvation, fails on flexible regions), and ESM-2 delta conflates stability with evolutionary conservation. Mechanism signal that correlates with the blind spots of these proxies will survive projection; mechanism signal that correlates with what they capture will be attenuated. The claim is "independent of FoldX-defined stability," not "independent of stability."

### 3. Linear probe with gene-split cross-validation

A logistic regression classifier is trained on the stability-projected delta-embeddings to predict 3-class mechanism labels. Cross-validation is performed with **gene-split folds**: no gene appears in both training and test sets. This prevents the classifier from learning gene identity rather than mechanism.

5-fold gene-split CV (subject to the per-class sample size check in the Data section). Bootstrap confidence intervals are computed by resampling test-fold genes (not variants) with replacement, since variants within a gene are not independent.

**Metrics:** AUROC per class vs rest, PR-AUC per class, macro-F1.

**Pre-registered effect size thresholds** for the headline metric (mean-pooled projected 3-class macro-AUROC, averaged across GOF/DN/LOF one-vs-rest AUROCs):
- < 0.60: null result — no meaningful mechanism geometry above chance
- 0.60–0.72: weak signal — present but limited generalization across gene families
- > 0.72: meaningful — mechanism geometry recovers across held-out genes

These thresholds are set assuming 3-class chance = 0.50 and the difficulty of gene-split CV on a gene-level-labeled dataset.

### 4. Gene-family-split cross-validation (tertiary CV)

Gene-split CV prevents memorizing individual genes but does not prevent the probe from learning a gene-family signature that correlates with mechanism class (e.g., kinases are enriched for GOF, structural proteins for DN). To test whether signal generalizes beyond gene family, a tertiary CV scheme holds out entire Pfam families rather than individual genes.

**Implementation:** Fetch the primary Pfam family for each gene from UniProt during sequence retrieval. Group genes by Pfam family. Genes lacking a Pfam annotation are dropped from this analysis (not assigned to singleton groups, since singletons behave like gene-level holdout and inflate family-split AUROC toward gene-split AUROC). If fewer than 10 Pfam families are represented in the dataset, report family-split CV as infeasible.

**Interpretation:** If macro-AUROC collapses under family-split CV (drops below 0.60 or falls > 0.10 below gene-split AUROC), the signal is family-level, not mechanism-level. If family-split AUROC is within 0.05 of gene-split AUROC, mechanism geometry generalizes across families. Report both CVs side by side; the headline result uses gene-split, family-split is a robustness check.

### 5. Baselines

All baselines use the same gene-split CV as the primary probe:
- **WT-only ESM-2 embeddings** (no delta): tests whether mechanism separation is already present in the wildtype representation, independent of the mutation
- **One-hot amino acid identity** (40-dim: 20 for WT residue, 20 for mutant residue): if this baseline matches the delta probe, the signal is from which amino acid substitution occurred, not from the protein context ESM-2 encodes
- **FoldX ddG only** (1-dim): tests whether stability alone separates mechanism classes
- **AlphaMissense score** (1-dim per-variant pathogenicity score, fetched from the AlphaMissense API): tests whether existing pathogenicity predictors already capture mechanism class separation

The delta probe must outperform all four baselines to claim delta-embeddings carry mechanism information beyond substitution identity, stability, and existing variant effect predictors.

### 6. Negative controls

- **Shuffled delta**: randomly reassign delta-embeddings across genes, breaking the variant-protein association. Signal should collapse to chance.
- **Benign stability-matched variants**: For each gene in the Gerasimavicius dataset, collect ClinVar 2-star benign missense variants from the same gene. For each pathogenic variant, find the closest-matched benign variant by FoldX ddG (nearest-neighbor matching). Run the same probe on these benign variants using their gene's mechanism label. If the probe separates mechanism classes on benign variants at > 50% of pathogenic macro-AUROC, this flags label leakage through gene-function-category rather than mutation-specific signal. This control requires FoldX ddG for benign variants; if fewer than 100 matched pairs are available, report this control as infeasible.

### 7. Probe direction orthogonality

Three pairwise logistic regression probes are fit: GOF-vs-DN, GOF-vs-LOF, DN-vs-LOF. Each probe is fit in stability-projected delta space using gene-split CV training data. The cosine similarity between pairs of probe weight vectors gives a 3×3 symmetric matrix of pairwise discriminating axis similarities.

**Why pairwise rather than one-vs-rest:** A one-vs-rest GOF probe discriminates GOF from (DN + LOF combined), so its weight direction mixes "what distinguishes GOF from DN" with "what distinguishes GOF from LOF." Pairwise probes each isolate a single discriminating axis, and cosines among them have a direct geometric interpretation: near-zero cosine between GOF-vs-DN and GOF-vs-LOF means the directions that separate GOF from DN and from LOF are geometrically independent.

**Interpretation baseline:** In 1280-dimensional space, two random unit vectors have expected cosine near zero with standard deviation ~1/sqrt(1280) ≈ 0.028. Near-zero inter-probe cosines are therefore the default, not evidence of orthogonality. The test is whether real pairwise probe cosines are **closer to zero than the shuffled-label null** — i.e., whether mechanism probes are more orthogonal than probes fit on random class assignments.

**Path A interpretation:** The stability subspace is fit on Megascale (a disjoint dataset) and the pairwise probe directions are fit on Gerasimavicius. The two fits are independent. If real inter-probe cosines are distinguishable from the shuffled-label null (|z| > 2), the three pairwise discriminating axes are geometrically independent in ESM-2 delta space after stability is removed. The reported figure is a 4×4 cosine matrix: stability direction + three pairwise probe directions.

**Path B interpretation:** The stability subspace and pairwise probe directions are both fit on Gerasimavicius variants. Probe-to-stability orthogonality is by construction and is not reported as an independent finding. The reported figure is a 3×3 pairwise inter-probe cosine matrix with an explicit caption noting that probe-to-stability orthogonality is mechanical in this path.

In both paths, the shuffled-label null is generated by refitting pairwise probes on 50 permutations of class labels (preserving class proportions) and computing inter-probe cosines. Real inter-probe cosines must show |z| > 2 relative to this null to support an orthogonality claim.

### 8. Secondary analysis: interface and disorder directions

Gerasimavicius et al. showed that DN mutations are enriched at protein interfaces and GOF mutations occur at positions with lower pLDDT (more disordered). If ESM-2 delta-embeddings encode mechanism geometry, they should project onto directions corresponding to these structural properties — recoverable from sequence alone.

**Feasibility pre-check (run before the main experiment):** Count interface variants (Livesey & Marsh 2022, high-confidence structural filter) and disordered-region variants (IUPred2 ∩ pLDDT < 50, ClinVar 2-star pathogenic) in genes not overlapping with Gerasimavicius. If either set has fewer than 500 variants, report that direction as infeasible and omit it from the analysis. Do not run this section if the feasibility check fails — report infeasibility explicitly rather than omitting silently.

**Interface direction:** From interface variants in held-out genes, fit an LDA direction (with shrinkage, scikit-learn LinearDiscriminantAnalysis with solver='eigen', shrinkage='auto') separating interface from non-interface variants in stability-projected delta space.

**Disorder direction:** From disordered-region variants in held-out genes, fit an LDA direction (same shrinkage settings) separating disordered-region from structured-region variants.

**Test:** Project all Gerasimavicius variants onto each external direction and compare scores by mechanism class with one-sided Mann-Whitney U tests. Pre-registered directional predictions:
- DN variants score higher on the interface direction than GOF and LOF variants
- GOF variants score higher on the disorder direction than DN and LOF variants

Both directions must be defined from genes with no overlap with Gerasimavicius to preserve independence.

### 9. ESM-2 3B robustness check

The identical pipeline is run on ESM-2 3B delta-embeddings. Pre-registered scale thresholds versus 650M macro-AUROC:
- Within +/-0.03: scale-invariant (mechanism geometry is robust to model scale)
- More than +0.05 higher: scale-emergent (mechanism geometry strengthens with scale)
- Lower: scale-degraded (finding; larger model may overfit to sequence family or compress mechanism axes)

The inter-probe cosine matrix is also compared between 650M and 3B: do mechanism axes become more orthogonal or more correlated at larger scale?

---

## Multiple testing policy

**Pre-registered headline test (one):** Mean-pooled stability-projected delta, 3-class gene-split CV, macro-AUROC averaged across GOF/DN/LOF one-vs-rest AUROCs. This is the single test against which the effect size thresholds apply.

**Confirmatory tests** (interpreted if headline is positive): per-residue delta AUROC, PR-AUC per class, bootstrap CI on macro-F1, gene-family-split AUROC, pairwise probe direction orthogonality z-scores, variance asymmetry bootstrap CI.

**Exploratory tests** (hypothesis-generating only): 4-class probe, HI vs AR 2-class probe, 3B scale comparison, per-baseline comparison, secondary interface/disorder projections.

No correction is applied within families, but confirmatory and exploratory tests are labeled as such in all tables. Any exploratory finding claimed as a contribution requires replication.

---

## Pre-registered decisions

| Decision | Pre-registered rule |
|---|---|
| Headline feature | Mean-pooled delta; per-residue is co-primary; mean-pooled cited if they disagree |
| Headline test | 3-class gene-split CV, macro-AUROC (mean of GOF/DN/LOF one-vs-rest AUROCs) |
| Effect size thresholds | < 0.60 null; 0.60–0.72 weak; > 0.72 meaningful |
| Stability transfer threshold | Spearman rho > 0.3 on Gerasimavicius FoldX ddG |
| Stability asymmetry prediction | Bootstrap CI on (GOF var. exp.) / (LOF var. exp.); upper CI < 0.70 = prediction holds |
| Bootstrap unit | Genes, not variants |
| CV scheme | 5-fold gene-split; switch to LOGO if GOF < 300 variants or < 50 genes |
| Gene-family-split interpretation | Drop > 0.10 below gene-split AUROC = family-level signal; within 0.05 = mechanism-level |
| Pfam singletons | Dropped from family-split analysis; not assigned to singleton groups |
| Scale thresholds (3B vs 650M) | Within +/-0.03 macro-AUROC = invariant; more than +0.05 = emergent; lower = finding |
| Orthogonality probe type | Pairwise (GOF-vs-DN, GOF-vs-LOF, DN-vs-LOF); 3×3 cosine matrix among probe directions |
| Orthogonality claim (Path A) | 4×4 cosine matrix: pairwise probe directions + stability direction |
| Orthogonality claim (Path B) | 3×3 pairwise inter-probe cosine matrix only; probe-to-stability orthogonality not claimed |
| Orthogonality null | Shuffled-label null (50 permutations); real cosines must show \|z\| > 2 to be reported |
| Orthogonality baseline | Near-zero cosines are expected in 1280-dim space; the test is vs. shuffled-label null |
| Benign control leak flag | Benign macro-AUROC > 50% of pathogenic macro-AUROC flags label leakage |
| Secondary analysis feasibility | < 500 variants in held-out set = infeasible; report explicitly, do not omit |

---

## Key scientific claim

ESM-2 delta-embeddings encode gene-level dominant disease mechanism class (GOF / DN / LOF) in a way that is geometrically distinct from protein stability perturbation. The three mechanism classes occupy different directions in ESM-2 delta space. This is recoverable by a linear classifier from protein sequence alone, without structural information, and is independent of the FoldX-defined stability signal that dominates ESM-2's zero-shot variant effect predictions.

---

## Outcomes (added 2026-05-28)

This section records how each pre-registered element resolved. All references are to `docs/result_*.md` files.

### Headline probe (Section 3)

**Result:** Null. Linear delta_mean macro-F1 = 0.279 under gene-split CV (result_1) — at majority-class chance level. The pre-registered effect-size scale (< 0.60 AUROC = null) was confirmed: even allowing for the different metric (F1 vs AUROC), the finding is unambiguous.

The null result was initially obscured by WT-only achieving F1 = 0.580 under gene-split. Family-split CV (result_2) collapsed WT-only to F1 = 0.389, explaining this as Pfam family memorisation. The actual CV design used in the pre-registration (gene-split) was insufficiently strict; family-split became the de facto standard after results 2–4 established causal leakage.

**MLP probes (exploratory, result_3/5/7):** MLP delta gene-split F1 = 0.415; family-split F1 = 0.299 ± 0.034 (Gerasimavicius, 5-seed). 62.8% of gene-split MLP signal is family-recognition leakage — exact and seed-invariant (result_7; see `result_leakage_fraction.md` for diagnostic formulation).

### Stability nuisance subspace removal (Section 2)

**Path A or B:** Path B was used — Megascale was not run due to infrastructure constraints. Stability subspace fit directly on Gerasimavicius FoldX ΔΔG (result_1).

**Stability transfer validation (Path A threshold rho > 0.3):** Not tested — Path B was taken without attempting Path A.

**Effect of projection:** Negligible. Projecting out the FoldX-fit stability subspace made no difference to probe F1 (result_1, Section on stability). Path B may have removed mechanism-correlated variance along with stability, or the probe had insufficient power to detect the difference. The stability projection was not pursued in subsequent results.

**Pre-registered stability asymmetry prediction:** FALSIFIED. GOF variance explained by the stability subspace = 63%; LOF = 60%. The pre-registered prediction was that (GOF var. exp.) / (LOF var. exp.) bootstrap CI upper bound < 0.70 — the observed ratio of ~1.05 is opposite in sign to the prediction. GOF variants show slightly *more* stability variance, not less. This likely reflects Path B overfitting or a difference between FoldX ΔΔG and actual stability perturbations in this dataset (result_1, stability notes).

### Gene-family-split CV (Section 4)

**Result:** Critical finding, not in the pre-registration. Family-split CV was not pre-registered but was added after result_2 showed the large gene→family drop in WT-only signal. This became the de facto primary holdout for all subsequent work. Results 2–10 all use family-split as the honest evaluation baseline. The 62.8% leakage fraction on Gerasimavicius is the structural diagnostic (result_7 + result_6 Part 2).

**Pfam singletons:** Handled per the pre-registered rule — dropped from family-split folds, not assigned to singleton groups.

### Baselines (Section 5)

**WT-only:** F1 = 0.580 gene-split → 0.389 family-split (result_2). Large gene→family drop confirms family memorisation is the primary signal source in WT embeddings.

**One-hot amino acid identity:** Not run as a standalone baseline. Pre-registered but not executed.

**FoldX ddG only:** Implicitly tested via the stability subspace projection (result_1); stability alone does not separate mechanism classes.

**AlphaMissense:** F1 ~ 0.0 (result_2 family-split baselines). AlphaMissense carries zero mechanism information, consistent with it encoding pathogenicity not mechanism.

### Negative controls (Section 6)

**Shuffled delta:** Not run as a formal control. Implicit through the chance-level linear probe result.

**Benign stability-matched variants:** Not run. Infrastructure not pursued (FoldX on ClinVar benign variants was not computed).

### Probe direction orthogonality (Section 7)

**Not run.** Once the headline probe was null under family-split, fitting pairwise probe directions and computing cosine matrices was not scientifically motivated — if there is no cross-family mechanism signal to separate, the probe directions are fitting family identity rather than mechanism geometry, and their orthogonality would not be interpretable.

### Secondary analysis: interface and disorder directions (Section 8)

**Not run.** The feasibility pre-check (≥500 variants in held-out genes from Livesey & Marsh / IUPred2) was not performed. After the primary probe returned null under family-split, this section was deprioritised.

### ESM-2 3B robustness check (Section 9)

**Not run.** After the frozen-PLM arc established a family-split ceiling of F1 ≈ 0.30–0.39 and found that nonlinear probes (MLP) do not improve family-split F1 (result_7), running a larger model under the same conditions would not change the central finding. The result_21 stability experiment (GBM on ESM-2 delta for ΔΔG) showed that nonlinearity does help stability transfer but not mechanism — confirming the mechanism-specific bottleneck. The pre-registered scale thresholds remain meaningful if future work revisits fine-tuned or task-specific models.

### Per-class sample sizes and CV scheme choice

**Result:** GOF has 1,983 variants and 81 genes — above the pre-registered thresholds (300 variants / 50 genes) for switching to LOGO CV. 5-fold gene-split CV was used as pre-registered.

### Data section actuals

| Pre-registered | Actual |
|---|---|
| ~10,200 pathogenic ClinVar missense variants | 10,233 variants after filtering |
| GOF: 1,983 / DN: 894 / HI: 1,678 / AR: 5,678 | Confirmed |
| ~948 Mendelian disease genes | 948 genes confirmed |

### What the pre-registration did not anticipate

1. **Family-split CV as primary holdout.** Gene-split was pre-registered; family-split was added post-hoc after result_2 revealed that gene-split inflates WT-only signal by leakage. Family-split became the de facto honest evaluation for all 23 results.

2. **The leakage fraction as a structural diagnostic.** The 62.8% leakage fraction on Gerasimavicius is exact, seed-invariant, and computable from dataset structure without training. See `result_leakage_fraction.md`.

3. **Post-hoc arcs (results 11–23).** After establishing the ESM-2 ceiling (results 1–10), the project expanded to gene-level proteome features (results 11–14), Badonyi structural priors (results 15–16), perturbation pattern analysis (results 19–20), stability geometry (result 21), log-likelihood scan (result 22), and pathogenicity/stability/mechanism geometry contrast (result 23). These were not pre-registered and are documented in `docs/README.md`.

4. **Pathogenicity positive control (result 6).** Not in the pre-registration. Added to establish pipeline soundness and the pathogenicity–mechanism dissociation under identical conditions.

### Summary verdict on the pre-registered hypothesis

The central hypothesis — that ESM-2 delta-embeddings encode gene-level dominant disease mechanism beyond protein stability — is **not supported under the project's family-split CV standard.** Family-split macro-F1 = 0.299 ± 0.034 (Gerasimavicius, MLP, 5-seed) — near the majority-class baseline of 0.279. 62.8% of the gene-split signal is family-recognition leakage. Stability projection (Path B) had no effect. The pathogenicity–mechanism dissociation (pathogenicity AUROC 0.74–0.88 vs mechanism F1 0.30–0.39, both family-split-stable) is the central finding of the ESM-2 arc. Mechanism signal that does exist in ESM-2 delta is family-correlated and does not generalise across held-out families.

---

# Run 7 pre-registration — inferential statistics (added 2026-07-22)

**Written before run_biorxiv executes.** Everything above this line is the original (run0-era)
pre-registration and its outcomes; the sections below govern run_biorxiv only. run_biorxiv re-scores run6's
science with dependency-aware error bars — the experiments, hypotheses, and gates are unchanged.

These rules exist because run_biorxiv will attach confidence intervals to gates that currently pass or
fail on point estimates alone. Without a reading fixed in advance, an interval that lands
awkwardly invites a framing chosen after the fact. The run6 point estimates are recorded here so
the rules cannot be retro-fitted to the run_biorxiv intervals.

Methodology: `reports/run6/STATS_PLAN.md`. Change list: `PLAN_2026-07-20.md`. Execution:
`RUNBOOK_5.md`.

## R7.1 — CI decision rule for gate verdicts

Every gate below is evaluated against a paired cluster-bootstrap 95% CI on its difference, not a
point estimate alone.

> **Affirmed** — the point estimate clears the threshold **and** the paired difference 95% CI
> excludes zero.
>
> **Not distinguishable** — the point estimate clears the threshold but the CI spans zero. This is
> reported as neither a pass nor a refutation.
>
> **Failed** — the point estimate does not clear the threshold. The verdict is failure regardless
> of the CI.
>
> **Underpowered** — a failed gate whose difference CI also spans the pre-registered threshold is
> reported as *underpowered to detect an effect of the pre-registered size*, **not** as evidence of
> no effect.

### Gates in scope, with run6 point estimates

`m1_threshold` = 0.430 is the ESM-2 family-split floor (0.380) **plus a pre-registered 0.05
effect-size requirement**. M1 and M2 are recorded in
`results/run6/esm3_mechanism/merged/summary.json` as `family-split F1 > 0.430`, so their margins
below are against 0.430 — not against the bare floor. Stated against the bare floor the lifts are
much larger (seq +0.058, seq_struct +0.072) and the CI question becomes trivial; the two framings
have opposite robustness and must not be interchanged.

| Gate | Criterion as recorded | Run6 value | Margin | Run6 verdict |
|---|---|---|---|---|
| M1 | seq_struct family-split F1 > 0.430 | 0.4528 | +0.023 | pass |
| M2 | seq family-split F1 > 0.430 (scale alone) | 0.4384 | +0.008 | pass |
| M3 | seq_struct − seq > 0.030 | +0.0143 | −0.016 | fail |
| K1 | conservation alone AUROC > 0.85 | 0.891 | +0.041 | pass |
| K2 | conservation + delta improves over conservation by > 0.02 | +0.0023 | −0.018 | fail |
| K2b | conservation + delta improves over delta alone | +0.0345 | — | descriptive (no threshold) |
| H2 | stability random→family rho drop < 0.10 (LEAKY) | — | descriptive | descriptive |
| Contrastive | contrastive k-NN > raw-delta k-NN | +0.041 | — | pass |

**Exposure is concentrated in M2 and the contrastive gate** — the only two claiming a pass on a
margin thinner than a seed of spread. M3 and K2 already fail, so a CI spanning zero *reinforces*
those readings; they are the cases where R7.1's underpowered clause applies rather than
threatening the conclusion.

## R7.2 — Confirmatory / exploratory split

The paper runs gate comparisons across seven reports. Correcting one table while leaving the rest
uncorrected is the weaker half of a defence, so the confirmatory set is enumerated **before** the
run and multiplicity control is applied across it.

### Confirmatory claims (six)

| # | Claim | Instrument |
|---|---|---|
| C1 | The mechanism delta sits at the measured chance floor under family-split | CI straddles the floor + permutation p |
| C2 | The absolute-embedding gene→family gap is non-zero (homology leakage exists) | paired bootstrap on the split gap |
| C3 | Pathogenicity clears AUROC 0.85 family-split (positive control) | CI excludes 0.85 |
| C4 | Conservation alone matches or beats the embedding delta for pathogenicity | paired bootstrap (K1/K2) |
| C5 | ESM-3 scale lifts the mechanism floor by at least the pre-registered 0.05 (M2: seq > 0.430) | paired bootstrap |
| C6 | The mechanism null is stable across homology partitions | family / clan / MMseqs2 robustness panel |

C1 and C3 are the load-bearing pair (the dissociation), C2 is the leakage account, and C6 makes C1
partition-independent. Those four are the paper. C4 and C5 are the characterisation payoff; if the
confirmatory set must be trimmed, trim from C4/C5, never C1–C3.

**Benjamini-Hochberg FDR is applied across these six claims only.** Raw and adjusted values are
both reported so a reader can see the correction rather than only its result.

### Exploratory (labelled, not corrected)

Per-class AUROCs, the 28-family within-family table and its per-family cells, the biochemistry R²,
the magnitude/direction decomposition, per-feature leakage fractions, and all descriptive geometry.
These are labelled exploratory in their reports. They are **not** FDR-corrected — correcting an
exploratory screen implies it was confirmatory.

**Failing gates are excluded from the correction set.** M3 ("structure adds nothing", a stated
headline in `ESM2_REPORT.md` §6) and K2 are reported under R7.1's underpowered clause and take no
part in the BH-FDR set, because correcting them would imply they were positive findings under test.

## R7.3 — Resampling unit

The resampling unit matches the unit the split holds out.

| Metric | Resample |
|---|---|
| Gene-split | genes |
| Family-split | **families** |
| Clan-split | clans |
| MMseqs2 cluster-split | clusters |
| Gene-split minus family-split gap | **families** (the coarser of the two arms) |

The split-gap case resamples families because its family-split arm's variance is only correct
under family resampling; a gene-resampled gap understates it. A family resample induces a valid
gene resample, but not the reverse. The gene-resampled interval is reported alongside as a
labelled sensitivity check.

**Expected and pre-registered:** family-split CIs will be visibly wider than gene-split ones.
There are 1,134 families but 833 are singletons, so the effective cluster count is far below the
gene count. That widening is the correct answer, not an artifact to tune away. The effective
cluster count is reported next to every family-split interval.

## R7.4 — Rare-class intervals

DN (≈ 9%, ~150–170 genes) and GOF (≈ 15%) sit in the regime where percentile bootstrap undercovers
for a bounded metric near its boundary with few clusters.

- One-vs-rest AUROC for the rare classes uses **BCa** wherever the acceleration estimate is
  computable.
- Rare-class intervals are **flagged as the least trustworthy in their table regardless of
  method** — with a jackknife over ~150 clusters, BCa's own correction is noisy. DN intervals are
  indicative, not authoritative.
- The existing degenerate-fold suppression guard is retained; BCa does not replace it.

## R7.5 — Permutation budget

- **Linear probe: 1,000 permutations**, seed 0. The headline claim (`delta_mean` at the chance
  floor) is a linear-probe claim, so the load-bearing test is fully resolved.
- **MLP: N set by the measured per-refit cost, stated explicitly** wherever its p-value appears.
- **No p-value is reported at its resolution floor of 1/(N+1)** — that is an unresolved bound, not
  a measurement. The run6 `wt_only_mean` p = 0.0099 at 200 permutations is exactly this case and is
  not carried forward.
- Seed 0 only: a permutation test constructs its own null by shuffling, so running it across five
  seeds mostly re-measures the fold jitter run_biorxiv exists to replace.

## R7.6 — Calibration

The probes are uncalibrated and measure **discrimination only**. Reported scores are not risk
estimates. This is stated in every probe report rather than fixed, because every claim in the paper
is a discrimination claim.

## R7.7 — What would change the conclusions

Recorded in advance so that "the CIs corroborated the point estimates" is a falsifiable statement
rather than an expectation:

- **C1 overturned** if `delta_mean`'s family-split CI excludes the measured floor from above and
  its permutation p is significant. The mechanism null would not survive.
- **C2 overturned** if the split-gap CI spans zero — the homology-leakage account would be
  unsupported at this sample size.
- **C3 overturned** if the pathogenicity CI includes 0.85; the positive control would no longer
  license the dissociation, and the whole paper weakens.
- **C5 restated** as *not distinguishable* if M2's paired CI spans zero, which the +0.008 margin
  makes plausible. The scale claim becomes "consistent in direction, not established".
- **C6 overturned** if the mechanism null does not hold under clan or MMseqs2 partitions — the
  result would be an artifact of the Pfam family definition.

run_biorxiv changes error bars, not point estimates. Any point estimate that moves materially from run6
is either a bug introduced by the wiring or a finding that needs explaining; `scripts/compare_runs.py`
flags these, and each flagged movement is explained rather than silently adopted.
