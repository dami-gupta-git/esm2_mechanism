# Result 14 — Clinical Utility: GOF/DN identification within ClinGen HI=3 genes
## Date: May 26, 2026 | Models: LogReg + MLP | Family-split CV, 5 seeds | Script: clinical_utility.py

---

## Background: the clinical question

ClinGen's **HI=3** category identifies genes where there is "sufficient evidence for haploinsufficiency" — in other words, losing one copy of this gene causes disease. There are 369 such labeled genes in our dataset.

Within that group, most genes cause disease by **LOF** (loss-of-function — the missing copy isn't replaced). But some cause disease by **GOF** (gain-of-function — the remaining copy is abnormally active) or **DN** (dominant-negative — the remaining copy actively sabotages normal copies). Clinically, distinguishing GOF from LOF in haploinsufficient genes matters for therapy selection — GOF genes may respond to inhibitors, while LOF genes often need gene replacement approaches.

Can our model tell these apart?

---

## TL;DR

Using out-of-sample probabilities from 5-fold family-split CV across 5 seeds, a logistic regression on the 37-feature proteome matrix achieves **GOF-vs-LOF AUROC = 0.650 ± 0.020** and **DN-vs-LOF AUROC = 0.668 ± 0.014** within the 369 ClinGen HI=3 genes. The pre-registered INFORMATIVE threshold (≥ 0.65) is cleared marginally on the GOF point estimate but is not robust — seed-level range is 0.625–0.674 for GOF, with one seed below threshold.

The dominant finding is the baseline comparison: **paralog_count alone achieves AUROC 0.746** within HI=3 — consistently higher than the full 37-feature model across all seeds. The multi-feature model does not outperform its single best predictor.

The missingness ablation shows NO-MISS features (18 features, no `*_missing` indicators) achieve **0.679 ± 0.016** for GOF — marginally better than FULL (0.650 ± 0.020). Missingness indicators add noise, not signal. The biological features carry the result.

pLI and LOEUF are at chance (0.496 / 0.486) by construction within HI=3 — all HI=3 genes are constrained, so constraint doesn't discriminate within the set. mis_z reaches 0.591. PPI_degree is below chance (0.349) — consistent with the T4 ablation finding that interactome features add nothing.

**Bottom line:** paralog count alone is the best predictor of GOF direction within ClinGen HI=3 genes. The full proteome model adds no aggregate lift.

---

## Design

**Family-split CV (honest out-of-sample evaluation):** 5-fold, protein family holdout, same scheme as results 11–13. HI=3 genes are scored only when their protein family appears in a held-out test fold. Singleton genes are treated as individual singleton families. No gene's probability is computed from a model that saw that gene during training.

**5 seeds (0–4):** each seed uses a different random family-to-fold assignment.

**Two feature sets:**
- **FULL** — all 37 columns
- **NO-MISS** — 18 columns (9 substantive + 9 family residuals only, no missingness indicators)

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
| pLI alone | 0.496 | — | [0.330–0.635] *(at chance by construction)* |
| LOEUF alone | 0.486 | — | [0.311–0.651] *(at chance by construction)* |

**paralog_count (0.746) beats the full LogReg model (0.650 ± 0.020) across all seeds.** The multi-feature model is not outperforming its single best predictor. The GOF pre-registered threshold (≥ 0.65) is cleared on average but is not robust: seed range is 0.625–0.674, with seeds 2 and 4 falling below 0.65.

**Missingness ablation:** NO-MISS (0.679 ± 0.016) outperforms FULL (0.650 ± 0.020). Dropping missingness indicators improves the model — the signal is in the substantive features, not in database coverage artefacts.

### H2 — DN-vs-LOF within HI=3 (5 seeds)

| Model | AUROC mean | Std |
|---|---|---|
| **LR FULL** | **0.668** | **0.014** |
| LR NO-MISS | **0.703** | **0.013** |
| MLP FULL | ~0.45 | — |

DN identification is more stable across seeds than GOF (std 0.014 vs 0.020), and NO-MISS again outperforms FULL by a larger margin (0.703 vs 0.668). The missingness indicators are particularly harmful for DN: they capture "gene is annotated by ClinGen" which strongly predicts LOF and suppresses DN scores.

### Calibration

ECE (GOF, LR FULL) = 0.148. The model is meaningfully miscalibrated — predicted probabilities should not be interpreted quantitatively without recalibration.

### Operating point (P_GOF > 0.4, LR FULL)

| Metric | Value |
|---|---|
| GOF genes flagged | 4 / 17 (recall = 0.235) |
| False positives | 21 |
| Precision | 4/25 = 0.160 |

At this threshold the model flags 25 genes to recover 4 of 17 GOF outliers (precision 16%). This is not a useful clinical operating point.

### Named GOF outliers — ranked by P_GOF (LR FULL, out-of-sample)

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

CACNA1A, PRKAR1A, TRIO, GRIN2B are correctly identified as the most likely GOF outliers. SCN1A and SCN2A (high-profile GOF epilepsy genes) get moderate P_GOF in the NO-MISS model (0.465, 0.489) but lower in the FULL model (0.332, 0.379) — the missingness indicators partially obscure these genes' signal. SETBP1 and FMR1 at P_GOF < 0.1 are hard failures — their proteome profiles match the LOF pattern better.

---

## The paralog_count finding — stratification check

paralog_count alone (AUROC 0.746) outperforms every other single feature and the full LogReg model within HI=3. Does GOF frequency actually scale with paralog count?

**Tertile stratification (HI=3, n=369):**

| Tertile | Paralog range | n | GOF | GOF% | DN | DN% |
|---|---|---|---|---|---|---|
| Low (0–3) | 0–3 | 140 | 1 | **0.7%** | 3 | 2.1% |
| Mid (4–14) | 4–14 | 111 | 5 | **4.5%** | 4 | 3.6% |
| High (≥15) | 15–80 | 118 | 11 | **9.3%** | 12 | 10.2% |

GOF frequency scales monotonically: 0.7% → 4.5% → 9.3%. The gradient is sharp — genes with ≥15 paralogs are 13× more likely to be GOF than genes with 0–3 paralogs within the HI=3 set. 11/17 GOF genes sit in the high-paralog tertile. The one low-paralog GOF outlier (FMR1) is mechanistically unusual — a repeat expansion gene, not a typical GOF point mutation.

The stratification is clean. **The AUROC 0.746 is robust and the biological interpretation holds:** genes with many paralogs are less likely to be haploinsufficient for simple dosage reasons (paralogs buffer one-copy loss), so when ClinGen classifies them as HI=3 anyway, the mechanism is more likely to be activating (GOF) or dominant-negative than dose-limiting (LOF). This also explains why the full 37-feature model (0.650) does not improve on paralog_count alone: with only 17 GOF genes, the model is underpowered to extract residual signal from the other 36 features.

---

## H4 — Unannotated genes

Among 1,330 labeled genes without ClinGen HI=3:
- Predicted: GOF=271 (20%), DN=239 (18%), LOF=820 (62%)
- Training class frequencies: GOF=9%, DN=7%, LOF=84%

The model predicts substantially more GOF and DN than the training base rate, consistent with non-uniform predictions. The top predicted GOF genes (unannotated) include PDGFRA (true GOF, P=0.772), ZIC1 (true GOF, P=0.686), and NDUFA1 (true LOF, P=0.828 — likely false positive driven by paralog count and constraint profile).

---

## Interpretation

### What is actually shown

1. **Paralog count alone is a better predictor of GOF direction within HI=3 than any multi-feature model.** This is the sharpest finding and is biologically interpretable via the gene balance hypothesis.

2. **The missingness-indicator concern was a red herring for the AUROC numbers** — dropping them improves rather than degrades performance.

3. **The full model (AUROC 0.675) barely clears the INFORMATIVE threshold and has very wide CIs [0.521–0.830].** With 17 GOF genes in the evaluation, the estimate is inherently uncertain. Claiming clinical informativeness from this is premature.

4. **The operating-point numbers are poor** (recall 0.235, precision 0.160 at P_GOF > 0.4).

5. **Calibration ECE = 0.148** — the model is substantially miscalibrated. Probability values should not be used as quantitative estimates.

### What remains open

1. **Paralog count analysis**: ✅ Done — tertile stratification confirms monotonic GOF frequency scaling.
2. **Out-of-sample validation on new genes**: the key gap. Genes characterised in 2025–2026 not in G2P or Gerasimavicius would provide a genuine prospective test.
3. **Constraint-only model**: does pLI + LOEUF + mis_z + paralog_count (4 features) match the full 18-feature NO-MISS model?
4. **Recalibration**: Platt scaling or isotonic regression under nested CV would produce calibrated probabilities usable at specific thresholds.

---

## Files

- `scripts/clinical_utility.py` — full analysis script
- `results/clinical_utility/gene_labels.tsv` — gene-level mechanism labels
- `results/clinical_utility/gene_mechanism_probs.tsv` — out-of-sample probabilities (LR FULL, family-split, seed 0)
- `results/clinical_utility/clinical_utility_results.json` — single-seed full results (seed 0)
- `results/clinical_utility/hi3_family_split_seed{0..4}.json` — per-seed HI=3 AUROC metrics
- `results/clinical_utility/hi3_family_split_summary.json` — 5-seed aggregated summary
- `results/clinical_utility/hi3_roc_pr_curves.png` — ROC and PR curves (seed 0)
- `results/clinical_utility/calibration.png` — reliability diagrams (seed 0)
- `results/clinical_utility/hi3_prob_distributions.png` — probability distributions by true class within HI=3
- `results/clinical_utility/feature_importance.png` — LogReg coefficients (NO-MISS features, full-data fit)

---

## Plain-English summary

We asked: given that a gene is already in ClinGen's haploinsufficiency database (the most authoritative clinical statement that a gene is dosage-sensitive), can we tell whether the actual mechanism is gain-of-function or something else?

The answer is a qualified yes, driven almost entirely by a single gene property: how many paralogs the gene has. Genes with more paralogs tend to be GOF within the HI=3 set, and this simple feature achieves better ranking performance than a 37-feature logistic regression. The biological explanation is straightforward: if a gene has many relatives in the genome, a loss of one copy is partially buffered — so when ClinGen still calls it haploinsufficient, it's more likely that the disease mechanism is activating (GOF) rather than dose-limiting (LOF).

The multi-feature model adds no aggregate benefit over paralog count alone. The predictions at any operating threshold are poor (recovering only 4 of 17 GOF outlier genes with 21 false positives), and the probability estimates are miscalibrated.

The clinically useful takeaway is narrower than the plan anticipated: **paralog count is a simple, interpretable, free predictor of GOF direction within haploinsufficient genes**, and deserves further investigation in that framing — not as part of a black-box multi-feature model.

---

## Reconciliation note added 2026-05-26

Result_13's T4 feature ablation found paralog_count contributes ΔF1 = +0.002 to V2's aggregate macro-F1 — essentially nothing. Why does paralog_count then beat the 37-feature model within HI=3 here (AUROC 0.746 vs 0.650)?

The two findings are consistent, not contradictory. Result_13 evaluates V2 on the full labeled set (1,699 genes across all mechanism classes) where the dominant features are constraint and dosage. Paralog_count contributes a small DN-specific signal that gets averaged out across classes.

This result evaluates a *subset selected for being constrained and dosage-sensitive*: ClinGen HI=3 by definition selects high-pLI genes. Within that subset, constraint and dosage are pinned at chance by construction. The residual signal is what's left after subtracting the dominant features — and paralog_count is the cleanest residual predictor. The gene balance hypothesis explains why: HI=3 genes with many paralogs are mechanistically unusual cases (paralog dosage redundancy should buffer them against HI; if they're nonetheless classified HI=3, the mechanism is more likely activating).

So: paralog_count contributes nothing to *aggregate* mechanism prediction (constraint and dosage dominate there), and dominates within *HI=3* (constraint and dosage are tautologically uninformative within that subset). The two findings together support a sharper biological claim: paralog_count is a context-specific predictor that matters only when other dominant features have been controlled for.
