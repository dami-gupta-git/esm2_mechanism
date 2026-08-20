# Dissecting protein identity, mutation effects, and disease mechanism in ESM-2 embeddings

## Status

This is the working manuscript skeleton, based on the current mechanism, pathogenicity control,
geometry, enzyme classification, stability, and family-split literature-audit reports under
`reports/run_biorxiv/`. A prior draft plan is kept at `biorxiv/bak/MANUSCRIPT.md` for reference
only.

## Length

The main text should contain 3,000 to 4,000 words, excluding references, figure captions, and
supplementary material.

| Section | Target words |
|---|---:|
| Abstract | 200 |
| Introduction | 400-450 |
| Results | 1,300-1,450 |
| Discussion | 600-700 |
| Methods | 700-900 |
| Total | 3,200-3,900 |

Within Results, the mechanism and family sections (§1-3) should receive roughly half the Results
budget, since they carry the study's primary finding; pathogenicity, enzyme, and stability (§4-6)
are controls and should stay tighter.

| Results subsection | Target words |
|---|---:|
| 1. Mechanism classification at the preregistered floor | 200-250 |
| 2. Weak classification and ranking signals depend on probe choice | 200 |
| 3. Wildtype embeddings encode protein family | 200-250 |
| 4. Pathogenicity is recoverable and largely overlaps with conservation | 200-250 |
| 5. Enzyme type is recoverable from the wildtype embedding | 150-200 |
| 6. Stability is recoverable but not uniformly family-robust | 200-250 |

## Central thesis

Frozen ESM-2 embeddings encode protein identity and family strongly. Under the preregistered
linear probe, the mutation-induced embedding change (the delta) does not reliably separate disease
mechanisms (loss-of-function / gain-of-function / dominant-negative) across held-out protein
families; exploratory probes recover weak above-floor performance, so this is a failure of that
specific classifier, not evidence the delta carries no mechanism information at all. The same delta
representation is predictive in a coarser family-held-out task (pathogenic vs. benign), and the same
architecture succeeds at a different protein-level task (enzyme type). For pathogenicity, however,
the delta is largely redundant with ESM-2's masked-marginal conservation score: conservation
outperforms the delta, and adding the delta does not improve discrimination.

The mutation delta also predicts experimentally measured folding stability across held-out
families, providing a physical-property control. Linear stability performance decreases beyond the
preregistered family-robustness tolerance and varies among domains, so transfer to held-out families
varies across biological targets.

## Abstract (elements, in order)

1. Protein language models are used for variant-effect prediction; whether they capture disease
   mechanism, not just pathogenicity, has not been tested consistently under family-held-out
   evaluation.
2. Frozen ESM-2 wildtype, mutant, and delta representations were evaluated for
   loss-of-function / gain-of-function / dominant-negative classification under gene and
   Pfam-family splits.
3. The preregistered delta probe sits at the classification floor under family holdout; a weak,
   probe-dependent ranking signal survives permutation testing.
4. Wildtype embeddings strongly encode Pfam family, and family membership correlates with the
   mechanism label, explaining part of the gene-to-family performance drop.
5. The same delta representation recovers pathogenicity and folding stability under family
   holdout, although stability performance is family-dependent and heterogeneous. The pathogenicity
   signal is mostly redundant with ESM-2's own conservation score; enzyme type is separately
   recoverable from the wildtype embedding.
6. Conclusion: performance depends on task granularity and evaluation unit, not on whether the
   representation carries any biological information at all.

## Introduction

1. Protein language models for variant effects — what "the embedding contains signal" does and
   does not imply about a specific downstream task.
2. Pathogenicity is not mechanism — a damaging variant can be damaging via different functional
   routes, and mechanism labels are usually curated at the gene level.
3. Evaluation unit matters — variant-random, gene-held-out, and family-held-out splits make
   different generalization claims. Prior mechanism-prediction work applies homology control
   inconsistently: some studies (e.g. LoGoFunc) test homology-disjoint splits, others use random
   variant splits or target within-gene transfer, a different, valid question
   (`report_FAMILY_SPLIT_LITERATURE_AUDIT.md`).
4. Study question — does the frozen ESM-2 mutation delta support family-transferable mechanism
   classification, using the same cohort, features, and splits throughout.

## Results

### 1. Mechanism classification at the preregistered floor
Source: `report_mechanism.md`.
Under family holdout, the preregistered linear probe on the mutation delta scores macro-F1 0.290,
equal to the measured chance floor of 0.290 (seed 0: 0.290, 95% CI 0.276-0.305, claim 2A-1
affirmed). Wildtype-only and mutant-only embeddings score far above the floor but drop when moving
from gene split to family split (wildtype-only: 0.552 to 0.449; mutant-only: 0.548 to 0.451). Because
the wildtype feature contains no mutation information, its performance and gene-to-family drop are
consistent with protein and family recognition contributing to mechanism prediction. Mutant-only
performance is nearly identical, indicating little additional classification benefit from the
mutation under this probe.

### 2. Weak classification and ranking signals depend on probe choice
Source: `report_mechanism.md`.
Permutation testing on the delta's ranking (AUROC) is significant in 4 of 5 seeds
(p = 0.029, 0.003, 0.011, 0.003, 0.054), so the preregistered classification-floor result is not a
claim of zero information (claim 2A-2, overturned as "no signal"). An exploratory full-dimensional,
standardized, class-balanced logistic probe reaches family-split macro-F1 0.387, while the MLP
reaches 0.375; both are above the 0.290 measured floor but below the preregistered linear
wildtype-embedding result of 0.449. These exploratory probes do not replace the preregistered test,
but they show that the recovered mechanism signal depends on probe specification.

### 3. Wildtype embeddings encode protein family, which correlates with the mechanism label
Source: `report_mechanism.md`.
The wildtype embedding predicts one of 145 Pfam families with 60.1% accuracy against a 4.37%
majority-class baseline; the delta drops to 4.37% (no signal). Among genes sharing a family, 83.2%
match their family's majority mechanism label, and the estimated wildtype leakage fraction is 0.389
(95% CI 0.241-0.542) — roughly 39% of the wildtype's above-floor performance does not survive
family holdout. The paired gene-to-family gap for the wildtype embedding is positive in 4 of 5
seeds (0.046 to 0.140; claim 2B supported).

### 4. Pathogenicity is recoverable across families and largely overlaps with conservation
Sources: `report_pathogenicity_control.md`, `report_geometry.md`.
The same delta representation, family-held-out, reaches AUROC 0.885 with an MLP (seed 0: 0.888,
95% CI 0.882-0.893), clearing the preregistered 0.85 threshold (claim 2C affirmed) and dropping
only 0.003 from the gene-split result, showing that the family-held-out design can recover signal for
a matched pathogenicity task. The direction of the delta carries the signal (AUROC 0.855-0.892) more
than its magnitude (AUROC 0.610). ESM-2's own masked-marginal conservation score alone slightly
exceeds the full delta-plus-conservation combination (0.888 vs. 0.883) and beats the delta alone
(0.888 vs. 0.835); adding the delta to conservation does not improve discrimination and in fact
reduces it (-0.005, 95% CI -0.008 to -0.001, an interval that excludes zero; claim 2E failed). The
information available from the delta for pathogenicity is therefore largely
redundant with conservation in this evaluation.

### 5. Enzyme type is recoverable from the wildtype embedding
Source: `report_enzyme_classification.md`.
A second, non-mutation control: the wildtype embedding classifies enzyme type at family-split
macro-F1 0.779 (seed 0: 0.787, 95% CI 0.732-0.818), clearing the preregistered 0.70 threshold
(claim 2F affirmed) against a majority-class floor of 0.219, versus a proteome-feature negative
control at 0.291. This exceeds mechanism classification on the same shared families by +0.507
macro-F1 (95% CI 0.447-0.541; claim 2G affirmed), confirming the representation carries strong
protein-level signal for enzyme type, not only mutation-level signal for pathogenicity. Logistic
regression exceeds the MLP by 0.074 on seed 0; the preregistered linear/nonlinear equivalence claim
(2H) was not established.

### 6. Stability is recoverable but not uniformly family-robust
Source: `report_stability.md`.
The mean-pooled mutation delta predicts experimentally measured folding stability under random,
domain, and Pfam-family splits. The preregistered ridge probe reaches random-split Spearman
correlation 0.693 (95% CI 0.675-0.709; claim 3A affirmed); the five-seed mean falls to a family-split
correlation of 0.554. The seed-0 paired analysis, on the matched cohort, gives a random-to-family
decrease of 0.153 (95% CI 0.112-0.192). The originating
rule supports family robustness at a decrease of at most 0.05, triggers `LEAKY` at a decrease of at
least 0.10, and leaves the interval between them not adjudicated. The observed interval is entirely
above 0.10, so claim 3B fails and the registered combined outcome is `LEAKY`.

Performance also varies among domains: the per-domain correlation standard deviation is 0.160
(95% CI 0.132-0.183), above the registered maximum of 0.10 (claim 3D failed). Removing a fitted
stability direction changes mechanism macro-F1 by -0.0009 (95% CI -0.0025 to +0.0007), affirming
claim 3C. Exploratory nonlinear probes retain family-held-out stability signal, with MLP correlation
0.627 and XGBoost correlation 0.630, but these results do not adjudicate 3A-3D.

## Methods (outline)

1. Mechanism cohort and labels — Gerasimavicius and Gene2Phenotype sources, source-priority rules,
   17,770 missense variants across 1,931 genes and 1,144 Pfam families, LOF / GOF / DN class
   counts.
2. Pathogenicity cohort — balanced ClinVar set, 24,384 variants across 1,802 genes, deduplication
   and per-gene balancing, kept separate from the mechanism cohort.
3. Enzyme cohort — 1,451 labeled genes across four classes, proteome-feature negative control.
4. Stability cohort — Tsuboyama 2023 mega-scale folding assay, 177,315 missense variants across
   181 natural domains and 77 Pfam families, with random, domain, and family splits.
5. ESM-2 representations — `esm2_t33_650M_UR50D`, wildtype/mutant/delta mean-pooled and
   per-residue features, masked-marginal conservation score, FoldX and AlphaMissense features.
6. Evaluation splits — gene split vs. Pfam-family split, definition of each; five-fold
   cross-validation over five seeds throughout, including stability's random, domain, and
   Pfam-family splits. Fourteen of the 181 stability domains lack a Pfam assignment and are
   excluded from stability's family-split evaluation.
7. Probes — preregistered linear PCA probe (256 components, no standardization/class weighting)
   defined separately from exploratory balanced/nonlinear probes; per-task probe specification
   (pathogenicity, enzyme, stability's ridge regression) noted where it differs.
8. Statistics — macro-F1, one-vs-rest AUROC, measured chance floors, cluster-bootstrap intervals
   (gene- or family-resampled), permutation tests, seed-combination rule.
9. Provenance — commit, environment, result-file fingerprints per report.

## Discussion

1. Main finding: the preregistered linear probe does not reliably classify mechanism under family
   holdout; a weak ranking signal remains, so this is not a claim of zero information, only
   insufficient information for that specific classifier.
2. Family recognition contributes to the gene-to-family performance drop: protein family predicts
   the wildtype embedding strongly and correlates with mechanism label.
3. Cross-task controls (pathogenicity, folding stability, enzyme type) show that the evaluation can
   recover signal for other biological tasks. Stability also shows that a physical-property signal
   can remain family-dependent and heterogeneous. These controls do not show that the mechanism
   labels are complete or noise-free.
4. The pathogenicity result overlaps strongly with conservation: the conservation score outperforms
   the delta, and adding the delta does not improve discrimination. RESCVE
   (Zhong and Shen 2022) is independent evidence for the same pathogenicity-versus-mechanism split:
   its untrained ESM masked-marginal score reaches AUROC 0.815-0.922 for pathogenicity but only
   0.541-0.653 for gain-of-function-versus-loss-of-function, close to chance. Its evaluation was not
   family-held-out, so it corroborates the pattern rather than replicating this study's confirmatory
   result.
5. Position the contribution against prior mechanism-prediction literature: some studies (e.g.
   LoGoFunc) test homology-disjoint splits; others use random variant splits or target within-gene
   transfer, a different, valid question. This study combines a paired gene-vs-family comparison,
   direct family-information measurement, and matched positive controls. The targeted literature
   audit does not establish priority for this combination.
6. Limitations: gene-level labels applied to variant-level data, small dominant-negative class,
   residual homology beyond Pfam family, frozen single-model-size embeddings, mean pooling.
7. Implications: report held-out unit and measured no-signal floor as standard practice; mechanism
   prediction likely needs variant-level functional data, not gene-level curated labels alone.

## Main figures (draft mapping)

| Figure | Content |
|---|---:|
| 1 | Cohort construction, delta extraction, gene vs. family splits |
| 2 | Delta classification at floor + ranking signal + probe sensitivity |
| 3 | Family prediction from wildtype embedding + family/mechanism-label association |
| 4 | Pathogenicity under family holdout + conservation alone vs. conservation + delta |
| 5 | Stability across random, domain, and family splits; enzyme classification by representation |

## Open items

- Draft the article text, figures, and supplementary material from the verified reports.
