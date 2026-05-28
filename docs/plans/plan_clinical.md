# Plan — Clinical Utility Experiment

**Date drafted:** 2026-05-26
**Status:** Pre-registration
**Builds on:** results 11–13 (proteome feature matrix, V2-LGBM mechanism classifier)

---

## What this experiment is (plain English)

The modelling work in results 11–13 showed that gene-level proteome features can predict disease mechanism (GOF / DN / LOF) with macro-F1 ~0.48 under family-split cross-validation. The natural question is: is that accuracy good enough to be useful in practice?

The current gold standard for clinical mechanism interpretation is **ClinGen dosage sensitivity scoring**. When a clinician encounters a variant of unknown significance in a dosage-sensitive gene, they look up the ClinGen haploinsufficiency (HI) score. HI=3 means "sufficient evidence that loss of one copy causes disease" — it is the most authoritative public statement that a gene is haploinsufficiency-driven (LOF/HI mechanism). Clinicians use this to decide things like: is this patient likely to benefit from a gene augmentation strategy? Should we avoid drugs that further reduce gene activity?

The problem is that ClinGen HI scores are mechanism-agnostic in one specific way: **they certify that the gene is dosage-sensitive, but they don't tell you whether the dosage-sensitive alleles are loss-of-function or gain-of-function.** A gene can have HI=3 and still be primarily GOF — meaning one overactive copy causes disease, not one missing copy. In those cases, using the HI score to guide treatment would point you in the wrong direction.

In our dataset, there are **17 GOF genes and 19 DN genes that also have ClinGen HI=3**. These are the clinical edge cases — genes where the dosage sensitivity database says "haploinsufficient" but the actual disease mechanism is gain-of-function or dominant-negative. Examples include SCN1A (GOF), SCN2A (GOF), CACNA1A (GOF), GRIN2B (GOF), PTPN11 (GOF) — all channelopathies or signalling genes where the GOF/LOF distinction directly changes prescribing decisions.

**The clinical utility question is:** does our model correctly identify these GOF and DN outliers within the HI=3 set? If it does, it is providing information that ClinGen's HI score cannot — it is distinguishing the direction of effect, not just the dosage sensitivity. That is the specific clinical value proposition.

A secondary question: among genes with **no ClinGen annotation** (the majority of genes — ~80% of our dataset has HI=0 or is unannotated), does the model's mechanism prediction generalise to genes where no clinical database has weighed in? This tests whether the model is learning biology or memorising database annotations.

---

## Why this is a different experiment from results 11–13

Results 11–13 evaluated the model on **variants** under **cross-validation** — the model was trained on some variants and tested on held-out variants from held-out protein families. The goal was to measure generalisation across sequence space.

This experiment evaluates the model on **genes** as a **knowledge discovery tool** — we train on all available data and ask whether the model's gene-level mechanism predictions are concordant with independently established clinical ground truth. There is no train/test split in the traditional sense; the question is whether the model's output is informative for clinical interpretation.

This is a retrospective validation design, common in computational biology: you train a model, then evaluate whether its predictions agree with curated expert knowledge that was not part of the training objective.

---

## Specific hypotheses (pre-registered)

**H1 — Direction accuracy within ClinGen HI=3:**
Among the 372 genes with ClinGen HI=3, the model assigns higher GOF probability to the 17 known GOF genes than to the 333 known LOF/HI genes. Measured by AUROC for GOF vs LOF within the HI=3 set.

**H2 — DN identification within ClinGen HI=3:**
Among the 372 genes with ClinGen HI=3, the model assigns higher DN probability to the 19 known DN genes than to the 333 known LOF/HI genes. Measured by AUROC for DN vs LOF within the HI=3 set.

**H3 — Model adds information beyond pLI alone:**
Within the HI=3 set, the model's GOF/DN predictions are not explainable by pLI or LOEUF alone (the features that define HI). A logistic regression on pLI + LOEUF alone should perform worse than the full model at distinguishing GOF/DN from LOF within the HI=3 subset.

**H4 — Unannotated gene coverage:**
Among genes with HI=0 or no ClinGen annotation (~1,500 genes in the dataset), the model's predicted mechanism distribution is not uniform — it assigns confident GOF/DN predictions to a subset, providing testable hypotheses for genes that clinical databases have not yet evaluated.

---

## What we will measure

**Primary metric:** AUROC for GOF-vs-LOF within ClinGen HI=3 genes (H1). This is the cleanest single number — it asks "does the model rank the 17 GOF outliers above the 333 LOF genes within a set that a database has labelled as uniformly haploinsufficient?"

**Secondary metrics:**
- AUROC for DN-vs-LOF within ClinGen HI=3 (H2)
- Confusion matrix for GOF+DN vs LOF within HI=3: how many of the 36 outliers (17 GOF + 19 DN) does the model correctly flag?
- Feature importance analysis: which proteome features drive the GOF/DN predictions for genes the model correctly identifies?
- Predicted mechanism distribution for unannotated genes (H4): top-20 predicted GOF and DN genes that have HI=0, as testable hypotheses

**Baseline comparison:**
- pLI alone (threshold at 0.9): marks everything as LOF, so sensitivity=0 for GOF/DN by definition
- LOEUF alone: same issue
- Random classifier (AUROC=0.5)

The model doesn't need to beat 0.9 AUROC to be useful here — it just needs to beat the baselines, which by construction cannot distinguish GOF from LOF within a dosage-sensitive gene set.

---

## How we will do it

### Step 1 — Gene-level feature matrix (already done)

`data/proteome_features_aligned.npy` (2424 × 37) is already built and aligned to the gene order in `merged_gene_list.tsv`. No new data collection needed.

### Step 2 — Gene-level labels

Collapse the variant-level labels in `merged_gene_list.tsv` to gene-level. For each gene, take the modal mechanism across all variants (most common class). Genes with conflicting variant labels (e.g. some GOF, some LOF variants) are flagged separately — they are scientifically interesting but should not be used as clean ground truth.

Apply the same 3-class collapse as before: HI → LOF, AR → dropped.

**Output:** `data/gene_labels.tsv` — one row per gene, columns: gene, mechanism_3class, n_variants, n_conflicting, conflict_flag.

### Step 3 — Train full-data V2-LGBM

Train LightGBM on all labeled genes (no held-out fold) using the full proteome feature matrix. This gives gene-level mechanism probability vectors P(GOF), P(DN), P(LOF) for every gene.

This is distinct from the cross-validated V2 — here we deliberately use all data to get the best possible probability estimates for the clinical analysis. The evaluation is against external ground truth (ClinGen), not a held-out test set.

**Script:** `scripts/clinical_utility.py`
**Output:** `data/gene_mechanism_probs.tsv` — gene, P_GOF, P_DN, P_LOF, predicted_class, n_variants, conflict_flag.

### Step 4 — ClinGen HI=3 analysis

Restrict to genes with ClinGen HI=3 (n=372 in the labeled set). Within this subset:

1. Compute AUROC for GOF-vs-rest using P_GOF
2. Compute AUROC for DN-vs-rest using P_DN
3. Build precision-recall curve: at threshold P_GOF > 0.4, how many of the 17 GOF outliers are recovered, and how many LOF genes are false positives?
4. Compare to pLI-only baseline (AUROC = 0.5 by construction within HI=3 — all these genes have high pLI)
5. Named gene report: list the 17 GOF and 19 DN genes, their predicted probabilities, and whether the model correctly ranks them above the LOF majority

### Step 5 — Unannotated gene analysis

For genes with HI=0 or no ClinGen score (n ≈ 1,500 labeled genes), report:
- Top-20 genes by P_GOF with their mechanism label (ground truth check)
- Top-20 genes by P_DN with their mechanism label
- Distribution of predicted mechanisms vs actual mechanisms for this subset
- Flag any genes where the model predicts GOF/DN but the label is LOF — these are either model errors or mislabelled genes worth examining

### Step 6 — Feature importance

Use LightGBM's built-in feature importance (gain-based) to identify which proteome features most strongly differentiate GOF/DN from LOF predictions. Report separately for GOF vs DN, since the driving features may differ (PPI_degree expected for DN, constraint features expected for LOF).

---

## Compute and timeline

| Phase | Duration | Compute |
|---|---|---|
| Step 2 — gene labels | ~30 min | Local CPU |
| Step 3 — full-data LGBM | ~5 min | Local CPU |
| Step 4 — ClinGen analysis | ~30 min | Local CPU |
| Step 5 — unannotated analysis | ~30 min | Local CPU |
| Step 6 — feature importance | ~10 min | Local CPU |
| Writeup | ~1 hour | — |
| **Total** | **~3 hours** | **Local CPU** |

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Model correctly predicts LOF for all HI=3 genes (AUROC ~0.5) | Medium | This is a clean negative result: model adds no information beyond HI status. Document and report honestly |
| GOF/DN outliers are mislabelled in source data | Low-medium | Cross-check named genes (SCN1A, SCN2A, CACNA1A etc.) against OMIM and primary literature before reporting |
| pLI/LOEUF features dominate and the model just recapitulates HI | Medium | Feature importance analysis (Step 6) will show this; H3 directly tests it |
| Conflicting variant labels inflate gene-level uncertainty | Low | Conflict flag in gene_labels.tsv; sensitivity analysis excluding conflict genes |

---

## Artifacts produced

- `scripts/clinical_utility.py` — gene-level label collapsing, full-data LGBM, all analyses
- `data/gene_labels.tsv` — gene-level mechanism labels with conflict flags
- `data/gene_mechanism_probs.tsv` — LGBM probability output for all genes
- `results/clinical_utility/` — AUROC plots, precision-recall curves, feature importance
- `docs/result_14.md` — writeup

---

## Pre-registered decision rules

| Outcome | Threshold | Interpretation |
|---|---|---|
| **Informative** | GOF AUROC within HI=3 ≥ 0.65 | Model adds clinically meaningful directional information beyond HI score |
| **Weakly informative** | GOF AUROC within HI=3 ≥ 0.55 | Some signal but insufficient for clinical use; useful as hypothesis generator |
| **Null** | GOF AUROC within HI=3 < 0.55 | Model cannot distinguish GOF from LOF within dosage-sensitive genes; clinical utility not demonstrated |

Every outcome is interpretable and publishable. The null result directly bounds what proteome features can do for clinical mechanism interpretation.
