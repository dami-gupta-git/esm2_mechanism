# Result 22 — Log-likelihood scan: sharper readout, same sampling problem
## Date: 2026-05-28 | Script: ll_scan.py | Seeds: 0–4 | GPU: H100 80GB

---

## Background: a more principled score

Result_20 replaced ClinVar variant positions with a systematic in-silico scan (100 evenly-spaced positions per gene, 3 probe amino acids), but used the **embedding L2 distance** as the mutation effect score — how much the ESM-2 representation shifts when you make the substitution.

This experiment uses a more principled score instead: **ESM-2's own log-likelihood**. ESM-2 is a masked language model — it was trained to predict which amino acid belongs at a masked position given the surrounding sequence. For any position, it directly outputs `log P(aa | context)` for all 20 amino acids. The mutation effect score is:

```
ΔLL = log P(wt_aa | context) − log P(mut_aa | context)
```

High ΔLL means the model strongly prefers the wildtype — the position is conserved and the mutation is "surprising." This is the same scoring approach used in ESM-1v and EVE for variant effect prediction, and is more principled than embedding L2 distance.

The key question: does using this more principled readout fix the sampling problem from result_20 (scan-only F1 = 0.272)?

---

## TL;DR

Replacing the embedding L2 distance (result_20) with ESM-2's native log-likelihood score at the same 100 evenly-spaced positions gives no improvement. LL-only family-split F1 = **0.261** — marginally worse than the embedding scan (0.272) and well below all three decision thresholds. All gates fail. The readout is not the bottleneck; the sparse 100-position sampling is.

---

## Setup

- **Dataset:** Gerasimavicius merged (10,231 variants, 1,985 genes, 3-class GOF/LOF/DN)
- **Probe positions:** same 100 evenly-spaced positions per gene as result_20 (reused `scan_probes.json`)
- **Score:** `ΔLL = log P(wt_aa | context) − log P(probe_aa | context)` averaged over Ala, Asp, Trp
- **Total forward passes:** ~198k (1 per position, all 20 AAs scored simultaneously via masking — 3× fewer than result_20)
- **CV:** 5-fold gene-split AND family-split, 5 seeds (0–4)
- **Classifier:** logistic regression (L2, balanced class weights), gene-level features

---

## Features

5 pre-registered scalar features per gene, computed from per-position ΔLL scores:

| Feature | Definition |
|---|---|
| `ll_wt_mean` | Mean log P(wt_aa) across positions — overall conservation |
| `ll_delta_mean` | Mean ΔLL across positions and probe AAs |
| `ll_delta_cv` | CV of ΔLL across positions — hotspot concentration |
| `ll_hotspot_frac` | Fraction of positions with ΔLL > mean + 1σ |
| `ll_top_entropy` | Mean entropy of the 20-AA distribution at top-10 ΔLL positions |

Four feature combinations tested:
- **LL only:** 5 LL features
- **LL + delta:** LL features + mean-pooled embedding delta (1285 features)
- **LL + scan:** LL features + result_20 scan features (10 features)
- **LL + scan + delta:** all three combined

---

## Results (5-seed mean ± std)

| Feature set | CV | macro-F1 | GOF AUROC |
|---|---|---|---|
| LL only | gene-split | 0.263 ± 0.007 | 0.541 ± 0.008 |
| LL only | **family-split** | **0.261 ± 0.005** | **0.527 ± 0.025** |
| LL + delta | gene-split | 0.393 ± 0.007 | 0.691 ± 0.008 |
| LL + delta | **family-split** | **0.380 ± 0.006** | **0.660 ± 0.012** |
| LL + scan | gene-split | 0.272 ± 0.009 | 0.546 ± 0.021 |
| LL + scan | **family-split** | **0.266 ± 0.006** | **0.534 ± 0.023** |
| LL + scan + delta | gene-split | 0.399 ± 0.005 | 0.684 ± 0.007 |
| LL + scan + delta | **family-split** | **0.377 ± 0.008** | **0.654 ± 0.014** |

---

## Decision rules

| Gate | Condition | Value | Threshold | Result |
|---|---|---|---|---|
| G1 | ll-only family-split F1 > 0.282 | 0.261 | 0.282 | **FAIL** |
| G2 | ll+delta family-split F1 > 0.385 | 0.380 | 0.385 | **FAIL** |
| G3 | ll+scan family-split F1 > 0.292 | 0.266 | 0.292 | **FAIL** |

G3 threshold = max(ll-only 0.261, scan-only 0.272) + 0.02 = 0.292.

---

## Key findings

### F1 — LL is no better than embedding distance at sparse sampling

LL-only F1 = 0.261, embedding scan-only F1 = 0.272 (result_20). The log-likelihood — despite being the more principled readout — performs marginally *worse*. The two methods are effectively equivalent at this sampling density. This rules out "wrong readout" as an explanation for result_20's failure.

### F2 — LL and scan features are not complementary

LL + scan (0.266) is no better than either alone (0.261, 0.272). The two readouts are capturing the same sparse, noisy signal from the same 100 positions. There is no complementarity because both methods fail for the same reason: they're measuring mostly uninformative positions.

### F3 — LL + delta almost passes G2 (0.380 vs 0.385)

The combined LL + mean-pooled delta misses G2 by 0.005. This is within noise — the scan + delta combination in result_20 achieved 0.375 with the same data, so LL is delivering approximately the same marginal contribution as embedding distance.

### F4 — Near-zero leakage confirmed

LL-only gene-split (0.263) ≈ family-split (0.261), Δ = +0.002. The LL features are genuinely leak-free. The problem is not leakage; it is signal weakness.

---

## Interpretation

### The readout is not the bottleneck

Results 20 and 22 together deliver a clean verdict: it doesn't matter whether you use embedding L2 distance or log-likelihood to score mutations at the 100 probe positions. Both methods give essentially the same (weak) result. The bottleneck is sampling density, not the scoring function.

For a GOF gene with 2–3 critical hotspot positions in a 1000-AA protein, 100 uniform positions have a ~26% chance of hitting even one hotspot. LL is a sharper instrument, but a sharp instrument measuring the wrong positions is no better than a blunt one.

### What result_19 was really measuring

Results 20 and 22 together confirm the interpretation of result_19: the ClinVar-pattern signal was not primarily about ESM-2's ability to detect hotspots — it was about ClinVar variants already being concentrated at known hotspots. When you remove that positional prior (uniform sampling), the signal drops regardless of how you score each position.

This is a methodologically important negative: result_19's F1 = 0.399 was inflated by ~0.13 F1 points relative to what an unbiased scan recovers (0.261–0.272). The genuine, bias-free ESM-2 mechanism signal sits at F1 ≈ 0.27 for scan-based features.

### What carries genuine signal

The mean-pooled delta (result_7, F1 = 0.377) and the scan + proteome combination (result_20 G3, F1 = 0.413) both survive family-split CV. These are the reliable baselines. The scan features contribute marginally when combined with proteome features but not when standing alone.

### Relationship to prior results

| Result | Approach | Family-split F1 |
|---|---|---|
| result_7 | Mean-pooled delta | 0.331 |
| result_9 | Contrastive projection | 0.397 |
| result_13 | Proteome features | ~0.385 |
| result_19 | ClinVar-pattern features | 0.399 |
| result_20 scan-only | Embedding scan (uniform) | 0.272 |
| result_20 scan+proteome | Embedding scan + proteome | 0.413 |
| result_22 ll-only | LL scan (uniform) | **0.261** |
| result_22 ll+delta | LL scan + mean-pooled delta | **0.380** |

---

## Limitations

- 100 positions per gene is too sparse to reliably hit GOF hotspots
- Single model (ESM-2 650M) — larger models may produce sharper LL scores
- Gene-level labels conflate multi-mechanism genes
- ΔLL averaged over only 3 probe AAs (Ala, Asp, Trp) — averaging over all 19 possible mutations would be more informative but 6× more forward passes

---

## What would actually work

To recover unbiased hotspot signal, options in increasing cost:

1. **Denser sampling** — 500 positions per gene instead of 100 would reliably hit most hotspots (~87% chance for a gene with 3 hotspots in 1000 AA). ~1M forward passes.
2. **Full saturation mutagenesis** — all positions × 19 substitutions. ~20M passes. Definitive but expensive.
3. **Structure-guided sampling** — prioritise positions at known functional sites (active sites, interfaces). Requires structural annotation, reduces to ~20 positions per gene.

---

## Files

- `scripts/ll_scan.py` — LL extraction and probe runs
- `data/ll_scores.json` — raw per-position LL scores for all 1,985 genes (116MB)
- `data/ll_features.npy` — 1985 genes × 5 features
- `data/ll_features_meta.json` — gene list and feature names
- `results/ll_scan/probe_results.json` — 5-seed results and gate outcomes
