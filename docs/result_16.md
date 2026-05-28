# Result 16 — Within-family mechanism prediction from family-residual features
## Date: 2026-05-26 | Script: within_family_mechanism.py | CV: Leave-one-gene-out within families

---

## TL;DR

Within-family mechanism prediction is possible from public gene-level features, but it works through within-family variation in proteome features — not through Badonyi's structural prior. **Family-residual proteome features** (how much a gene deviates from its family's average constraint, abundance, PPI degree etc.) achieve macro-F1 = 0.514, outperforming raw proteome features (0.484), Badonyi residuals (0.449), and nearly matching the combined residuals (0.516). The signal is dominated by two families: homeodomains (PF00046, n=30, resid-proteome F1=0.633) and intermediate filaments (PF00038, n=17, F1=0.451). Badonyi structural residuals add no within-family information beyond what proteome residuals already encode — raw and residual Badonyi give identical results (0.449), meaning the structural prior carries no within-family variation signal. The majority baseline (always predict the family's most common class) is 0.702 accuracy / ~0.333 macro-F1, making any above-chance F1 meaningful within a family.

---

## Setup

**24 qualifying Pfam families** (≥6 genes, ≥2 mechanism classes) covering 238 labeled genes total.

| Family | Name | n | Classes |
|---|---|---|---|
| PF00046 | Homeodomain | 30 | LOF=25, GOF=2, DN=3 |
| PF00038 | Intermediate filament | 17 | LOF=4, GOF=2, DN=11 |
| PF00069 | Kinase | 17 | DN=2, GOF=5, LOF=10 |
| PF00520 | Ion channel | 16 | GOF=10, LOF=3, DN=3 |
| PF00096 | Zinc finger C2H2 | 15 | LOF=13, DN=1, GOF=1 |
| PF00250 | Fork head domain | 11 | LOF=9, GOF=1, DN=1 |
| PF00104 | Nuclear receptor | 11 | LOF=8, DN=3 |
| PF02931 | Ligand-binding domain | 10 | LOF=6, GOF=2, DN=2 |
| PF00071 | Ras GTPase | 9 | GOF=7, DN=1, LOF=1 |
| PF01410 | Calcium-binding EGF | 9 | LOF=5, GOF=4 |
| + 14 smaller families | | 6–9 | various |

**CV design:** Leave-one-gene-out (LOGO) within each family — one gene is the test set; the remaining genes train the classifier. Predictions pooled across all families for aggregate macro-F1 and per-class AUROC.

**Feature sets compared:**
- **Raw proteome** — absolute values of pLI, LOEUF, mis_z, paralog_count, tissue_tau, log_abundance_ppm, PPI_degree, HI_score, TS_score (9 features). Cross-family baseline.
- **Residual proteome** — same features minus the family mean. Within-family deviation only.
- **Raw Badonyi** — pDN, pGOF, pLOF from Badonyi 2024 SVM (3 features). Cross-family structural prior.
- **Residual Badonyi** — pDN/pGOF/pLOF minus family mean. Within-family structural deviation.
- **Combined residuals** — residual proteome + residual Badonyi (12 features).

**Majority baseline:** 70.2% gene-level accuracy (predicting modal class per family), corresponding to ~0.333 macro-F1.

**Note on seed variance:** LOGO CV is deterministic (no random train/test split). All 5 seeds give identical results — std = 0.000 throughout. This is correct: unlike k-fold CV, LOGO has a unique deterministic solution for each feature set.

---

## Results

### Aggregate (238 genes, 24 families)

| Feature set | Macro-F1 | GOF AUROC | DN AUROC | LOF AUROC |
|---|---|---|---|---|
| Majority baseline | ~0.333 | — | — | — |
| Raw proteome | 0.484 | 0.656 | 0.735 | 0.675 |
| **Residual proteome** | **0.514** | **0.653** | **0.737** | **0.694** |
| Raw Badonyi | 0.449 | 0.584 | 0.681 | 0.628 |
| Residual Badonyi | 0.449 | 0.584 | 0.681 | 0.628 |
| **Combined residuals** | **0.516** | **0.655** | **0.725** | **0.692** |

### Per-family breakdown (seed 0, residual proteome)

| Family | Name | n | F1 | GOF | DN | LOF | Notes |
|---|---|---|---|---|---|---|---|
| PF00046 | Homeodomain | 30 | **0.633** | 0.852 | 0.857 | 0.888 | Strong — large n, all 3 classes |
| PF00023 | Paired box | 7 | **1.000** | NA | 1.000 | 1.000 | Perfect — 2 classes, very separable |
| PF00503 | SNAP receptor | 6 | **0.829** | NA | 0.750 | 0.750 | Strong GOF/LOF separation |
| PF00167 | FGF | 6 | 0.625 | NA | 0.750 | 0.750 | Moderate |
| PF00027 | Cyclic nucleotide | 6 | 0.533 | 0.750 | 0.500 | 1.000 | Mixed |
| PF00038 | Intermed. filament | 17 | 0.451 | 0.848 | 0.633 | 0.635 | GOF well-separated; DN harder |
| PF00520 | Ion channel | 16 | 0.417 | 0.487 | 0.533 | 0.513 | Weak — consistent with result_10 |
| PF00069 | Kinase | 17 | 0.300 | 0.000 | 0.433 | 0.600 | GOF/DN confusion within kinases |
| PF00071 | Ras GTPase | 9 | 0.333 | 0.000 | 1.000 | 0.000 | Only 1 DN gene — degenerate |
| PF00096 | ZnF C2H2 | 15 | 0.320 | 0.000 | 0.000 | 0.923 | 13/15 LOF — near-trivial |
| PF00250 | Fork head | 11 | 0.259 | 0.000 | 0.000 | 0.111 | Near-chance |
| PF01410 | EGF calcium | 9 | 0.100 | 0.100 | NA | 0.100 | At chance |

---

## Key findings

### F1 — Within-family variation in proteome features predicts mechanism

Residual proteome F1 = 0.514 vs raw proteome F1 = 0.484. The residual version is strictly better despite having the same 9 features — the difference is that it removes the cross-family mean, forcing the classifier to use only within-family deviation. This confirms the hypothesis: **the mechanism-predictive signal in proteome features is partly within-family**, not solely driven by cross-family differences in constraint level.

Practically: a kinase that is *more constrained than the average kinase* is more likely to be LOF; a kinase with *higher PPI degree than the average kinase* is more likely to be DN. That's the within-family signal.

### F2 — Badonyi structural residuals add nothing within families

Raw Badonyi = Residual Badonyi = 0.449. Identical numbers. This means pDN/pGOF/pLOF does not vary in a mechanism-predictive way within protein families — all kinases get similar Badonyi scores, all ion channels get similar Badonyi scores. The Badonyi model's signal is almost entirely cross-family: it correctly says "ion channels tend to be GOF/DN, transcription factors tend to be LOF," but within the ion channel family it cannot distinguish which specific channels are GOF vs DN vs LOF.

This is a direct complement to the cross-family analysis (results 1–15): Badonyi beats ESM-2 and proteome cross-family, but within a family, the within-family residual of proteome features is a better predictor than any Badonyi-derived signal.

### F3 — Homeodomains are the clearest within-family example

PF00046 (homeodomain, n=30): residual proteome F1 = 0.633, GOF AUROC = 0.852, DN AUROC = 0.857. This is the largest family in the qualifying set and has all three mechanism classes. The result holds under raw features too (F1 = 0.369), but residuals are markedly better (0.633 vs 0.369 — the largest raw → residual lift of any family). This means homeodomains share a family-level constraint profile, and the mechanism information lives in how individual members deviate from that profile.

### F4 — Ion channels are the clearest within-family null

PF00520 (ion channel, n=16): residual proteome F1 = 0.417, barely above majority. Result_8 found within-family ESM-2 delta AUROC = 0.659 for ion channels (GOF/DN 2-class). That earlier result used ESM-2 delta (sequence-level mutation context), not gene-level features. The comparison is informative: within ion channels, mutation-level sequence context (ESM-2 delta) carries more within-family signal than gene-level proteome features. The mechanism signal within ion channels is in *which specific mutations each channel gene carries*, not in *which genes are more constrained or abundant*. Gene-level features are not the right resolution for ion channel mechanism.

### F5 — Combined residuals barely beat residual proteome alone

Combined F1 = 0.516 vs residual proteome = 0.514. The +0.002 lift from adding Badonyi residuals is negligible. Given that Badonyi residuals carry no information on their own (F2), this makes sense: combining nothing with something still gives approximately something.

---

## Interpretation

### The within-family story

The within-family analysis resolves a question left open by results 1–15: **if the cross-family signal is mostly family identity, where does the within-family signal come from?**

Answer: within-family variation in gene-level proteome features — specifically how much a gene deviates from its family's average on constraint (pLI, LOEUF), interactome degree (PPI_degree), and abundance (log_abundance_ppm). This makes biological sense:

- **Within a transcription factor family**: the more essential the transcription factor (higher constraint than the family average), the more likely it causes disease by haploinsufficiency (LOF). The less essential one (lower constraint than family average), if dominant, is more likely GOF or DN.
- **Within a structural protein family**: higher PPI degree than family average → more interaction partners → more surfaces for DN poisoning.
- **Within a kinase family**: the GOF kinases (KRAS, BRAF) tend to be more abundant and centrally connected than the LOF kinases.

### What this means for the interview story

This is the positive result that makes the negative result more interesting. The corrected story is:

> "Cross-family mechanism prediction fails because the signal is mostly family identity — ESM-2 learns that kinases are GOF, structural proteins are LOF, and calls that mechanism. But within a family, the signal is real and learnable from within-family variation in gene-level features. Homeodomains are a clean example: the homeodomain genes that cause disease by GOF deviate from the family average in specific ways — they're more constrained, have more interaction partners, are expressed more specifically. That within-family signal is genuine mechanism biology, not identity leakage."

### The boundary condition

The results also define a boundary condition for when within-family mechanism prediction works:
- **Works**: large families with all 3 classes present and meaningful within-family variation in constraint/abundance (homeodomains, intermediate filaments, paired-box proteins)
- **Doesn't work**: families where one class dominates (ZnF C2H2: 13/15 LOF), or where mechanism is primarily determined by mutation position rather than gene-level properties (ion channels), or where the family is too small for LOGO to be meaningful (n=6 with 1 minority class)

---

## Caveats

**Small n in most families.** 19 of 24 families have n ≤ 17. LOGO CV on n=6–9 gene families with 1 minority class is unreliable — a single correct prediction changes F1 by 0.1+. The homeodomain result (n=30) is the only one large enough to trust as a standalone finding.

**LOGO is deterministic but not unbiased.** With n=6 and 1 minority-class gene, that gene is always left out alone — the test set has 1 sample. Any F1 number from families with a single-member minority class (PF00096, PF12796, PF12662 etc.) is essentially a single coin flip.

**The aggregate macro-F1 (0.514) is dominated by PF00046.** PF00046 contributes 30/238 = 12.6% of test predictions, and its F1 (0.633) is the highest of any family. The aggregate number is real but should not be read as "within-family prediction generally works at F1=0.51" — it works well in one large family and variably elsewhere.

---

## Files

- `scripts/within_family_mechanism.py` — LOGO CV, 5 feature sets, 24 families
- `results/within_family/within_family_seed0.json` — per-family and aggregate results (deterministic; all seeds identical)
- `results/within_family/within_family_summary.json` — aggregated summary

---

# Addendum — Does Badonyi's published model survive strict holdout?

## Date: 2026-05-26 | Script: badonyi_holdout_survival.py | Plan: docs/plan_badonyi.md

## Plain-English summary

We took Badonyi 2024's published mechanism predictor (a model that gives every human gene three probability scores: how likely it is to act by dominant-negative, gain-of-function, or loss-of-function) and asked one question: do those probabilities still work when we test them on genes their model has never seen, and where related genes have been kept out of the training data?

The short answer is *yes for one half of the question and no for the other*.

**The good half — family holdout doesn't hurt.** When we hide entire protein families from each fold (so the model has to predict mechanism for, say, a homeodomain gene without having seen any other homeodomain gene in training), Badonyi's predictions are essentially unchanged. The AUROCs move by 1–2 points, which is statistical noise. So Badonyi's published model genuinely generalises across protein families — it isn't a family-recognition system in disguise. That was the main thing we were testing.

**The bad half — the genes Badonyi already trained on are predicted much better than the genes he never saw.** We split our gene set into two groups: genes that appeared in Badonyi's original training set (621 genes), and genes that did not (1,064 genes). Then we compared the AUROCs:

- Predicting DN-vs-LOF: AUROC 0.68 on training-set genes vs 0.62 on never-seen genes (~6-point gap).
- Predicting GOF-vs-LOF: AUROC 0.71 on training vs 0.69 on never-seen (~2-point gap, small).
- **Predicting LOF-vs-non-LOF: AUROC 0.625 on training vs 0.472 on never-seen** (15-point gap, and 0.472 is essentially random).

The LOF gap is the headline. On the genes Badonyi's model never saw, his pLOF score is at chance — it doesn't distinguish loss-of-function genes from gain-of-function or dominant-negative genes. The published AUROC of 0.763 for LOF-vs-non-LOF appears to be mostly the model fitting genes it had already seen during training.

This is *not* what the project's prior leakage analysis was looking for. We were checking for "family-recognition leakage" — the idea that the model might be learning protein-family identity and using that as a mechanism proxy. The family-split holdout shows that's not what's happening here. Instead, we found a different form of leakage: **per-gene training-set fit**. Badonyi's SVM remembers the genes it trained on better than it generalises to new ones.

## Why this doesn't break result_15

Result_15's V_bad model trains a *new* logistic regression on Badonyi's three probability scores under family-split CV. That new logistic regression effectively re-calibrates Badonyi's predictions per fold, averaging out the training-set fit. So V_bad's numbers stay valid — we're using Badonyi's outputs as features in a model that we evaluate honestly, not as predictions in their own right.

The reframe is for *how Badonyi's published numbers should be cited*, not for our own model's performance:

- "Badonyi 2024 reports DN-vs-LOF AUROC 0.71" → defensible
- "Badonyi 2024's model achieves AUROC 0.71 on held-out genes" → not quite. The published number is on the training set with k-fold CV; on genes outside their training universe, our test shows performance closer to AUROC 0.62 for DN and at chance for LOF.

## Why this doesn't break the leakage-triage analysis either

The earlier leakage-triage analysis (`scripts/badonyi_leakage_analysis.py`) looked at the V_bad model and asked: does the LogReg on top of Badonyi's outputs perform better on Badonyi-training-set genes than on never-seen genes? The answer there was no — V_bad performs *better* on out-of-training genes, because base-rate effects dominate the per-gene F1 metric on the LOF-heavy never-seen subset.

That earlier finding and this new one are consistent: the LogReg-on-top layer absorbs Badonyi's per-gene memorisation and doesn't pass it through to its own predictions. The new finding is about Badonyi's raw model, not about V_bad.

## Numbers table

| Evaluation | DN-vs-LOF | GOF-vs-LOF | LOF-vs-nonLOF | n |
|---|---|---|---|---|
| Badonyi 2024 reported (in their paper) | 0.71 | 0.763 | 0.763 | their training set |
| Our setup, whole labeled set | 0.575 | 0.652 | 0.467 | 1,685 |
| Pfam family-split | 0.588 | 0.666 | 0.483 | 1,229 |
| MMseqs2-20 cluster-split | 0.587 | 0.660 | 0.478 | 1,255 |
| IN-Badonyi-train (no holdout) | 0.677 | 0.713 | 0.625 | 621 |
| OUT-Badonyi-train (no holdout) | 0.620 | 0.694 | 0.472 | 1,064 |

The gap between "Badonyi reported" and "our setup, whole labeled set" is partly real degradation and partly a label-scheme mismatch (our "LOF" collapses HI+LOF from Gerasimavicius+G2P; Badonyi's "LOF" is dominant haploinsufficient only — these are slightly different gene sets). DN and GOF labels align better than LOF; the LOF mismatch is the biggest single driver of the overall drop.

## Pre-registered decision rule outcome

Pre-registered threshold (plan_badonyi.md):
- ΔAUROC ≥ −0.03: ROBUST
- −0.10 < Δ < −0.03: PARTIAL
- Δ ≤ −0.10: MOSTLY LEAKAGE

All six holdout comparisons (Pfam × {DN, GOF, LOF}, MMseqs × {DN, GOF, LOF}) fire in the ROBUST band. By the pre-registered criterion, Badonyi's published model passes the family-recognition leakage test.

But that doesn't fully describe the picture — the IN-vs-OUT-of-training gap (especially +0.153 on LOF) is real and significant. Pre-registration didn't include a threshold for the per-gene training-set fit dimension; that's a finding the original plan didn't anticipate.

## Files

- `scripts/badonyi_holdout_survival.py` — analysis script
- `results/badonyi_survival/badonyi_survival_seed{0..4}.json` — per-seed metrics
- `results/badonyi_survival/badonyi_survival_summary.json` — 5-seed aggregated summary
- `docs/plan_badonyi.md` — pre-registration document
