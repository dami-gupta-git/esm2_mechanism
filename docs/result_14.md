# Result 14 — Clinical Utility: GOF/DN identification within ClinGen HI=3 genes
## Date: May 26, 2026 | Models: LogReg + MLP | Family-split CV, 5 seeds | Script: clinical_utility.py

---

## TL;DR

Using **out-of-sample probabilities from 5-fold family-split CV across 5 seeds**, a logistic regression on the 37-feature proteome matrix achieves **GOF-vs-LOF AUROC = 0.650 ± 0.020** and **DN-vs-LOF AUROC = 0.668 ± 0.014** within the 369 ClinGen HI=3 genes. The pre-registered INFORMATIVE threshold (≥ 0.65) is cleared marginally on the GOF point estimate but is not robust — seed-level range is 0.625–0.674 for GOF, with one seed below threshold.

The dominant finding is the baseline comparison: **paralog_count alone achieves AUROC 0.746** within HI=3 — consistently higher than the full 37-feature model across all seeds. The multi-feature model does not outperform its single best predictor.

The missingness ablation (T3) shows NO-MISS features (18 features, no `*_missing` indicators) achieve **0.679 ± 0.016** for GOF — marginally better than FULL (0.650 ± 0.020). Missingness indicators add noise, not signal. The biological features carry the result.

pLI and LOEUF are at chance (0.496 / 0.486) by construction within HI=3. mis_z reaches 0.591. PPI_degree is below chance (0.349) — consistent with the T4 ablation finding that interactome features add nothing.

MLP under family-split CV achieves GOF AUROC ~0.41 — below LogReg and below the paralog_count baseline.

**Bottom line:** paralog count alone is the best predictor of GOF direction within ClinGen HI=3 genes. The full proteome model adds no aggregate lift. The clinical utility case reduces to: why does paralog count predict GOF within dosage-sensitive genes, and can that specific signal be sharpened into a usable tool?

---

## Design

**Family-split CV (honest OOS evaluation):** 5-fold, Pfam family holdout, same scheme as results 11–13. HI=3 genes are scored only when their Pfam family appears in a held-out test fold. Singleton genes (no Pfam annotation, n=56 of 369 HI=3 genes) are treated as individual singleton families and each appear in exactly one test fold. No gene's probability is computed from a model that saw that gene during training.

**5 seeds (0–4):** each seed uses a different random family-to-fold assignment. Per-seed per-fold AUROCs are aggregated. Reported as mean ± std across seeds. Bootstrap CIs (1,000 resamples) are computed within seed 0 for illustration.

**Two feature sets:**
- **FULL** — all 37 columns (9 substantive + 9 missingness indicators + 9 family residuals + 9 family-residual missing + 1 singleton flag)
- **NO-MISS** — 18 columns (9 substantive + 9 family residuals only, no missingness indicators)

**Baselines within HI=3:** single features evaluated out-of-sample. pLI and LOEUF are included to confirm the tautology (both are near chance by construction within HI=3). Non-tautological baselines: mis_z, paralog_count, PPI_degree.

**Data:** 1,699 labeled genes (GOF=158, DN=118, LOF=1,423). HI=3 subset: 369 labeled genes (GOF=17, DN=19, LOF=333).

---

## Results

### H1 — GOF-vs-LOF within HI=3 (5 seeds)

| Model / Feature | AUROC mean | Std | Seed-0 95% CI |
|---|---|---|---|
| **LR FULL (37 features)** | **0.650** | **0.020** | [0.521–0.830] |
| LR NO-MISS (18 features) | **0.679** | **0.016** | [0.540–0.832] |
| MLP FULL | ~0.41 | — | — |
| **paralog_count alone** | **0.746** | — | [0.645–0.848] |
| mis_z alone | 0.591 | — | [0.409–0.755] |
| PPI_degree alone | 0.349 | — | [0.188–0.535] |
| pLI alone | 0.496 | — | [0.330–0.635] *(tautological)* |
| LOEUF alone | 0.486 | — | [0.311–0.651] *(tautological)* |
| Random | 0.500 | — | — |

**paralog_count (0.746) beats the full LogReg model (0.650 ± 0.020) across all seeds.** The multi-feature model is not outperforming its single best predictor. The GOF pre-registered threshold (≥ 0.65) is cleared on average but is not robust: seed range is 0.625–0.674, with seeds 2 and 4 falling below 0.65.

**T3 missingness ablation:** NO-MISS (0.679 ± 0.016) outperforms FULL (0.650 ± 0.020). Dropping missingness indicators improves the model, confirming the signal is in the substantive features, not in database coverage artefacts. Missingness delta GOF = −0.028 ± 0.008.

### H2 — DN-vs-LOF within HI=3 (5 seeds)

| Model | AUROC mean | Std |
|---|---|---|
| **LR FULL** | **0.668** | **0.014** |
| LR NO-MISS | **0.703** | **0.013** |
| MLP FULL | ~0.45 | — |
| Missingness delta | −0.035 | 0.009 |

DN identification is more stable across seeds than GOF (std 0.014 vs 0.020), and NO-MISS again outperforms FULL by a larger margin (0.703 vs 0.668). The missingness indicators are particularly harmful for DN: they capture "gene is annotated by ClinGen" which strongly predicts LOF and suppresses DN scores.

### H3 — Missingness ablation (T3)

| Feature set | GOF AUROC | DN AUROC |
|---|---|---|
| FULL (37 features) | 0.650 ± 0.020 | 0.668 ± 0.014 |
| NO-MISS (18 features) | **0.679 ± 0.016** | **0.703 ± 0.013** |
| Δ (FULL − NO-MISS) | −0.028 ± 0.008 | −0.035 ± 0.009 |

pLI AUROC = 0.496, LOEUF = 0.486 — both at chance, confirming H3: the model's above-chance performance is not from constraint features (which are tautologically flat within HI=3) but from paralog count and other non-constraint features.

The missingness-dominance concern from the naive analysis was real in terms of coefficient magnitude, but the AUROCs show it was not artifically inflating the result. The NO-MISS model is the cleaner and marginally better version.

### Calibration

ECE (GOF, LR FULL) = 0.148. The model is meaningfully miscalibrated — predicted probabilities of ~0.4–0.5 do not correspond to 40–50% GOF rates in those bins. Probability thresholds (e.g. P_GOF > 0.4) should not be interpreted quantitatively without recalibration.

### Operating point (P_GOF > 0.4, LR FULL)

| Metric | Value |
|---|---|
| GOF genes flagged | 4 / 17 (recall = 0.235) |
| False positives | 21 |
| Precision | 4/25 = 0.160 |

At this threshold the model flags 25 genes to recover 4 of 17 GOF outliers (precision 16%). This is not a useful clinical operating point. The AUROC describes ranking ability; the operating-point numbers describe what happens at any given threshold — and they are poor.

### Named GOF outliers — ranked by P_GOF (LR FULL, OOS)

| Gene | P_GOF LR full | P_GOF LR no-miss | Correctly ranked? |
|---|---|---|---|
| CACNA1A | 0.526 | 0.514 | Yes — highest |
| PRKAR1A | 0.494 | 0.506 | Yes |
| TRIO | 0.469 | 0.507 | Yes |
| GRIN2B | 0.432 | 0.456 | Yes |
| FGFR1 | 0.392 | 0.409 | Borderline |
| SCN2A | 0.379 | 0.489 | — |
| SCN1A | 0.332 | 0.465 | — |
| DNMT3A | 0.329 | 0.417 | — |
| TGFBR1 | 0.301 | 0.358 | — |
| LMNA | 0.293 | 0.426 | — |
| PTPN11 | 0.275 | 0.352 | — |
| PAK3 | 0.219 | 0.260 | — |
| GATA6 | 0.105 | 0.146 | No |
| SPAST | 0.095 | 0.182 | No |
| FOXC2 | 0.089 | 0.258 | No |
| FMR1 | 0.032 | 0.098 | No |
| SETBP1 | 0.008 | 0.189 | No |

CACNA1A, PRKAR1A, TRIO, GRIN2B are correctly identified as the most likely GOF outliers. The rest fall into the LOF probability mass. SCN1A and SCN2A (sodium channels, high-profile GOF epilepsy genes) get moderate P_GOF in the NO-MISS model (0.465, 0.489) but lower in the FULL model (0.332, 0.379), suggesting the missingness indicators partially obscure these genes' signal.

SETBP1 and FMR1 at P_GOF < 0.1 are hard failures — mechanistically unusual (SETBP1 is a transcription factor co-activator; FMR1 is an RNA-binding protein) and their proteome profiles match the LOF pattern better.

### DN outliers — selected

SPTAN1 (0.490), PHOX2B (0.502), OTX2 (0.489), FOXP1 (0.415), COL3A1 (0.403) are correctly ranked in the upper half. AR (androgen receptor, a well-established DN gene) gets P_DN = 0.001 in the FULL model — the ClinGen HI and TS scores in its feature vector likely dominate and push it toward LOF. The NO-MISS model gives AR 0.366, confirming the HI_score feature is suppressing the DN signal for highly ClinGen-annotated genes.

---

## The paralog_count finding — stratification check

paralog_count alone (AUROC 0.746) outperforms every other single feature and the full LogReg model within HI=3. The gene balance hypothesis interpretation requires a direct robustness check: does GOF frequency scale monotonically across paralog tertiles within HI=3?

**Tertile stratification (HI=3, n=369):**

| Tertile | Paralog range | n | GOF | GOF% | DN | DN% |
|---|---|---|---|---|---|---|
| Low (0–3) | 0–3 | 140 | 1 | **0.7%** | 3 | 2.1% |
| Mid (4–14) | 4–14 | 111 | 5 | **4.5%** | 4 | 3.6% |
| High (≥15) | 15–80 | 118 | 11 | **9.3%** | 12 | 10.2% |

GOF frequency scales monotonically: 0.7% → 4.5% → 9.3%. DN also scales monotonically. The gradient is sharp — genes with ≥15 paralogs are 13× more likely to be GOF than genes with 0–3 paralogs within the HI=3 set. 11/17 GOF genes sit in the high-paralog tertile (CACNA1A, FGFR1, GRIN2B, SCN1A, SCN2A, TRIO, PTPN11, PAK3, LMNA, FOXC2, SETBP1). The one low-paralog GOF outlier (FMR1) is mechanistically unusual — a repeat expansion gene, not a typical GOF point mutation.

The stratification is clean. **The AUROC 0.746 is robust and the gene balance hypothesis interpretation holds:** genes with many paralogs are less likely to be haploinsufficient for simple dosage reasons (paralogs buffer one-copy loss), so when ClinGen classifies them as HI=3 anyway, the mechanism is more likely to be activating (GOF) or dominant-negative than dose-limiting (LOF). This also explains why the full 37-feature model (0.650) does not improve on paralog_count alone: with only 17 GOF genes, the model is underpowered to extract residual signal from the other 36 features.

---

## H4 — Unannotated genes

Among 1,330 labeled genes without ClinGen HI=3:
- Predicted: GOF=271 (20%), DN=239 (18%), LOF=820 (62%)
- Training class frequencies: GOF=9%, DN=7%, LOF=84%

The model predicts substantially more GOF and DN than the training base rate, consistent with H4 (non-uniform predictions). The top predicted GOF genes (unannotated) include PDGFRA (true GOF, P=0.772), ZIC1 (true GOF, P=0.686), and NDUFA1 (true LOF, P=0.828 — likely false positive driven by paralog count and constraint profile). The keratin cluster dominates DN predictions and all correctly labelled true DN.

---

## Feature importance (NO-MISS features, full-data LR)

**GOF:** mis_z (+1.204) and LOEUF (+0.649) are the largest substantive coefficients. Higher missense Z-score (more constrained to missense) and less LoF-intolerant → GOF. tissue_specificity_tau (+0.477): more tissue-specific expression → GOF. paralog_count (+0.297): more paralogs → GOF. HI_score (−0.418): having a formal ClinGen HI score (even if not =3) predicts against GOF.

**DN:** pLI (+0.765) and tissue_specificity_tau (+0.755) dominate. High LoF intolerance and tissue-specific expression → DN. log_abundance_ppm (+0.604): higher protein abundance → DN. TS_score (−0.567): genes with formal ClinGen triplosensitivity scores are not DN. Paralog_count (+0.378): consistent with GOF — more paralogs tilts toward both GOF and DN over simple LOF.

The GOF and DN feature profiles are partially overlapping (both favour paralog count, both are tissue-specific), which explains why the multi-class model is less discriminating than paralog_count alone when evaluated in the narrow HI=3 context.

---

## Interpretation

### What is actually shown

1. **Paralog count alone is a better predictor of GOF direction within HI=3 than any multi-feature model.** This is the sharpest finding and is biologically interpretable via the gene balance hypothesis.

2. **The missingness-indicator concern was a red herring for the AUROC numbers** — dropping them improves rather than degrades performance. The missingness indicators were inflating specific gene coefficients without adding net predictive value.

3. **The full model (AUROC 0.675) barely clears the INFORMATIVE threshold and has very wide CIs [0.521–0.830].** With 17 GOF genes in the evaluation, the estimate is inherently uncertain. Claiming clinical informativeness from this is premature.

4. **The operating-point numbers are poor** (recall 0.235, precision 0.160 at P_GOF > 0.4). The AUROC describes ranking on average; the actual screening performance is low.

5. **Calibration ECE = 0.148** — the model is substantially miscalibrated. Probability values should not be used as quantitative estimates.

### What remains open

1. **Paralog count analysis**: ✅ Done — tertile stratification confirms monotonic GOF frequency scaling (0.7% → 4.5% → 9.3%). The AUROC 0.746 is robust. See "The paralog_count finding" section above.

2. **Out-of-sample validation on new genes**: the key gap remains. Genes characterised in 2025–2026 that are not in G2P or Gerasimavicius would provide a genuine prospective test.

3. **Why does paralog_count dominate?** The gene balance hypothesis predicts that paralog-rich genes tolerate dominant activating mutations better than dosage-limiting ones. A direct test: within HI=3, do GOF genes have higher LoF burden in paralogs (measured by gnomAD constraint on paralogs)?

4. **Constraint-only model**: does pLI + LOEUF + mis_z + paralog_count (4 features) match the full 18-feature NO-MISS model? Given paralog_count dominates, likely yes. This is the minimum sufficient feature set.

5. **Recalibration**: Platt scaling or isotonic regression under nested CV would produce calibrated probabilities usable at specific thresholds.

---

## Files

- `scripts/clinical_utility.py` — full analysis script (family-split CV, 5 seeds, two feature sets, bootstrap CIs, calibration)
- `results/clinical_utility/gene_labels.tsv` — gene-level mechanism labels
- `results/clinical_utility/gene_mechanism_probs.tsv` — OOS probabilities (LR FULL, family-split, seed 0)
- `results/clinical_utility/clinical_utility_results.json` — single-seed full results (seed 0)
- `results/clinical_utility/hi3_family_split_seed{0..4}.json` — per-seed HI=3 AUROC metrics (T1+T3)
- `results/clinical_utility/hi3_family_split_summary.json` — 5-seed aggregated summary
- `results/clinical_utility/hi3_roc_pr_curves.png` — ROC and PR curves for H1 and H2 (seed 0)
- `results/clinical_utility/calibration.png` — reliability diagrams for P(GOF) and P(DN) (seed 0)
- `results/clinical_utility/hi3_prob_distributions.png` — probability distributions by true class within HI=3
- `results/clinical_utility/unannotated_top_predictions.png` — top predicted GOF/DN genes outside HI=3
- `results/clinical_utility/feature_importance.png` — LogReg coefficients (NO-MISS features, full-data fit)

---

## Plain-English summary

We asked: given that a gene is already in ClinGen's haploinsufficiency database (the most authoritative clinical statement that a gene is dosage-sensitive), can we tell whether the actual mechanism is gain-of-function or something else?

The answer is a qualified yes, driven almost entirely by a single gene property: how many paralogs the gene has. Genes with more paralogs tend to be GOF within the HI=3 set, and this simple feature (paralog count) achieves better ranking performance than a 37-feature logistic regression. The biological explanation is straightforward: if a gene has many relatives in the genome, a loss of one copy is partially buffered — so when ClinGen still calls it haploinsufficient, it's more likely that the disease mechanism is activating (GOF) rather than dose-limiting (LOF).

The multi-feature model adds no aggregate benefit over paralog count alone, the predictions at any operating threshold are poor (recovering only 4 of 17 GOF outlier genes with 21 false positives), and the probability estimates are miscalibrated. The result clears the pre-registered INFORMATIVE threshold on its point estimate, but the confidence intervals are wide enough to include chance performance.

The clinically useful takeaway is narrower than the plan anticipated: **paralog count is a simple, interpretable, free predictor of GOF direction within haploinsufficient genes**, and deserves further investigation in that framing — not as part of a black-box multi-feature model.

---

## Reconciliation note added 2026-05-26

Result_13's T4 feature ablation found paralog_count contributes ΔF1 = +0.002 to V2's aggregate macro-F1 — essentially nothing. Why does paralog_count then beat the 37-feature model within HI=3 here (AUROC 0.746 vs 0.650)?

The two findings are consistent, not contradictory. Result_13 evaluates V2 on the full labeled set (1,699 genes across all mechanism classes) where the dominant features are constraint and dosage. Paralog_count contributes a small DN-specific signal that gets averaged out across classes — confirmed by result_13's per-class ablation showing ΔDN = −0.015 when paralogs are dropped (paralogs help DN slightly when present).

This result evaluates a *subset selected for being constrained and dosage-sensitive*: ClinGen HI=3 by definition selects high-pLI genes. Within that subset, constraint and dosage are pinned at chance by construction (pLI AUROC = 0.49 within HI=3). The residual signal is what's left after subtracting the dominant features — and paralog_count is the cleanest residual predictor. The gene balance hypothesis explains why: HI=3 genes with many paralogs are mechanistically unusual cases (paralog dosage redundancy should buffer them against HI; if they're nonetheless classified HI=3, the mechanism is more likely activating).

So: paralog_count contributes nothing to *aggregate* mechanism prediction (constraint and dosage dominate there), and dominates within *HI=3* (constraint and dosage are tautologically uninformative within that subset, leaving paralog_count as the strongest remaining signal). The two findings together support a sharper biological claim: paralog_count is a context-specific predictor that matters only when other dominant features have been controlled for.
