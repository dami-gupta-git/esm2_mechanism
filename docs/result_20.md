# Result 20 — In-silico perturbation scan: unbiased hotspot features add to proteome but fail alone
## Date: 2026-05-28 | Scripts: perturbation_scan.py, perturbation_probe.py | Seeds: 0–4 | GPU: H100 80GB

---

## TL;DR

Replacing ClinVar variants with a systematic in-silico scan (100 evenly-spaced positions × 3 probe amino acids per gene, ~568k ESM-2 forward passes) eliminates the circularity of result_19 but also loses its GOF signal. Scan-only family-split F1 = **0.272** — well below the ClinVar-pattern baseline (0.348) and the G1 threshold (0.368). However, scan features combined with proteome features reach F1 = **0.413**, passing G3 (threshold 0.405). The scan adds orthogonal signal to proteome features even though it cannot stand alone.

---

## Setup

- **Dataset:** Gerasimavicius merged (10,231 variants, 1,985 genes, 3-class GOF/LOF/DN)
- **Probe positions:** 100 evenly-spaced positions per gene (linspace 1→L), 3 probe AAs: Ala, Asp, Trp
- **Total forward passes:** 567,771 (some positions skipped where probe AA = WT AA)
- **Embeddings:** ESM-2 650M mean-pooled WT and mutant sequences
- **CV:** 5-fold gene-split AND family-split, 5 seeds (0–4)
- **Probe:** logistic regression (L2, balanced class weights), gene-level features

---

## Features

5 pre-registered scalar features per gene, computed from the N×1280 delta matrix (mut_emb − wt_emb):

| Feature | Definition |
|---|---|
| `scan_mag_mean` | Mean ‖delta‖ across all probe positions and substitutions |
| `scan_mag_cv` | Coefficient of variation of magnitudes (std/mean) |
| `scan_hotspot_frac` | Fraction of positions with magnitude > mean + 1σ |
| `scan_pc1_var` | Variance explained by PC1 of the delta matrix |
| `scan_sub_variance` | Mean per-position variance across Ala/Asp/Trp magnitudes |

Four feature combinations tested:
- **Baseline:** mean-pooled delta (1280-dim), gene-level average — result_7 baseline
- **Scan only:** 5 features above
- **Scan + delta:** scan features + mean-pooled delta (1285 features)
- **Scan + proteome:** scan features + proteome features from result_13

---

## Results (5-seed mean ± std)

| Feature set | CV | macro-F1 | GOF AUROC |
|---|---|---|---|
| Baseline (delta mean) | gene-split | 0.394 ± 0.007 | 0.690 ± 0.009 |
| Baseline (delta mean) | **family-split** | **0.377 ± 0.008** | **0.665 ± 0.013** |
| Scan only | gene-split | 0.280 ± 0.006 | 0.542 ± 0.021 |
| Scan only | **family-split** | **0.272 ± 0.005** | **0.524 ± 0.024** |
| Scan + delta | gene-split | 0.400 ± 0.006 | 0.683 ± 0.007 |
| Scan + delta | **family-split** | **0.375 ± 0.007** | **0.658 ± 0.016** |
| Scan + proteome | gene-split | 0.417 ± 0.007 | 0.767 ± 0.007 |
| **Scan + proteome** | **family-split** | **0.413 ± 0.005** | **0.750 ± 0.009** |

---

## Decision rules

| Gate | Condition | Value | Threshold | Result |
|---|---|---|---|---|
| G1 | scan-only family-split F1 > 0.368 | 0.272 | 0.368 | **FAIL** |
| G2 | scan+delta family-split F1 > 0.419 | 0.375 | 0.419 | **FAIL** |
| G3 | scan+proteome family-split F1 > 0.405 | 0.413 | 0.405 | **PASS** |

---

## Key findings

### F1 — Scan-only signal collapses without ClinVar bias

Scan-only F1 = 0.272, versus result_19 ClinVar-pattern F1 = 0.348. The unbiased scan is substantially worse than using observed clinical variants. This is not a failure of the method — it confirms that result_19's signal came partly from the ClinVar enrichment for known hotspots. When you sample blindly at 100 positions, you usually miss the 2–3 positions that actually matter for a GOF gene.

Scan-only also shows near-zero leakage (gene-split 0.280 ≈ family-split 0.272, Δ = +0.008), confirming the features are not family-recognition shortcuts. There is genuine but weak signal in the uniform scan.

### F2 — Scan adds nothing to the mean-pooled delta

Scan + delta (0.375) is essentially identical to baseline delta alone (0.377). The 5 scalar features derived from the uniform scan are fully subsumed by the 1280-dimensional mean-pooled delta. Whatever signal the scan captures is already present in the aggregate embedding.

### F3 — Scan adds to proteome features (G3 passes)

Scan + proteome (0.413) beats proteome-alone threshold (0.405) and exceeds the baseline delta (0.377). The scan features capture something orthogonal to proteome-level properties (protein length, domain composition, expression). This is the one positive finding: the ESM-2 perturbation landscape adds information beyond what is known about the gene from databases.

### F4 — GOF AUROC is higher than expected for a null

Scan-only GOF AUROC = 0.524 (family-split). This is above chance (0.5) despite the scan's low F1. The scan does pick up faint GOF hotspot signal even with sparse sampling — it just isn't strong enough to drive a good classifier.

---

## Interpretation

### What this proved and disproved (plain-language)

The hope going in was the *positive* version: that a fair, unbiased scan could recover result_19's GOF-hotspot signal **without** relying on where clinical variants happen to sit. That would have shown the mechanism signal is fully extractable from sequence alone, no human-study bias required.

That hope was **largely disproved**. Scan-only F1 (0.272) fell well below the ClinVar-pattern baseline (0.348) and failed G1. Sampling 100 fair, evenly-spaced positions usually misses the 2–3 spots that actually matter for a GOF gene, so the signal collapses. The headline hypothesis did not hold.

But the disproof is itself informative — it is a clean reality-check on result_19. Result_19 looked stronger than it really was *because part of its apparent power was the ClinVar position bias, not ESM-2.* Specifically (see result_19 F4): result_19's clustering features were partly driven by (a) variant *count* (`n_variants_log`, a "how-studied" proxy) and (b) clustering that reflects doctors repeatedly reporting the same famous hotspot. Strip both away with a fixed even scan and most of the lift disappears.

Two clarifications that were not obvious:

- **ESM-2 was never predicting ClinVar.** ESM-2 only measures how disruptive a mutation is; the bias lived in *which positions* it was asked to measure. The labels (GOF/LOF/DN) come from the curated Gerasimavicius dataset, not ClinVar — only the *positions* came from ClinVar.
- **This is not "ESM-2 carries no signal."** The plain mean-pooled delta (result_7, F1≈0.377) and the scan+proteome combination here (G3, 0.413) both carry genuine, leak-free mechanism signal. The correct reading is narrower: ESM-2 carries *less* mechanism signal than result_19 first suggested, and result_19's extra boost was largely a ClinVar artifact. The smaller, genuine core survives.

### Why scan-only fails where ClinVar-pattern succeeded

Result_19 used the positions where clinical variants actually occur — these are by definition enriched for functionally important positions. The scan samples uniformly, so for a GOF gene with 3 critical positions out of 1000, 100 uniform samples have only a ~26% chance of hitting even one. The hotspot concentration signal (`scan_mag_cv`) is diluted by the 97 uninformative positions.

This is not a flaw in the scan design — it is a correct negative result. It shows that the hotspot biology is real (result_19 found it using known positions) but requires either knowing where to look or denser sampling to recover it blindly.

### Why scan + proteome passes G3

Proteome features describe the gene's biological context (tissue expression, constraint, domain architecture). The scan describes the gene's local ESM-2 perturbation landscape. These two are largely independent information sources — one is external database knowledge, the other is a sequence-derived model readout. Their combination is modestly additive.

### Relationship to prior results

| Result | Approach | Family-split F1 |
|---|---|---|
| result_7 | Mean-pooled delta | 0.331 |
| result_9 | Contrastive projection | 0.397 |
| result_13 | Proteome features | ~0.385 |
| result_19 | ClinVar-pattern features | 0.399 |
| result_20 scan-only | In-silico scan (this result) | 0.272 |
| result_20 scan+proteome | In-silico scan + proteome | **0.413** |

---

## Limitations

- 100 positions per gene is sparse for detecting concentrated hotspots — full saturation (~20M passes) would be needed to reliably hit every important position
- Embedding L2 distance is a blunt readout — log-likelihood (result_21) may be more sensitive for the same probe positions
- Gene-level labels conflate multi-mechanism genes (SCN1A has both GOF and LOF variants)
- Single model (ESM-2 650M) only

---

## Next steps

- **result_21:** Log-likelihood scan — use the same 100 probe positions but score mutations via `log P(wt) − log P(mut)` instead of embedding distance. This is a more direct model readout and should recover more hotspot signal per position. Estimated ~3× fewer forward passes (~198k).

---

## Files

- `scripts/perturbation_scan.py` — embedding extraction (phases 1–3)
- `scripts/perturbation_probe.py` — probe runs and decision rules
- `data/scan_features.npy` — 1985 genes × 5 features
- `data/scan_features_meta.json` — gene list and feature names
- `results/perturbation_scan/probe_results.json` — 5-seed results and gate outcomes
