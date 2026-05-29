# Result 19 — ClinVar variant pattern features: spatial distribution of perturbations predicts mechanism
## Date: 2026-05-26 | Script: perturbation_pattern.py | Seeds: 0–4 | CPU only

---

## Background: a different way to use ESM-2

All prior results used either the delta embedding directly (mutant minus wildtype, a 1280-dimensional vector) or gene-level biological properties. This experiment tries something in between: instead of using the raw embeddings, we compute **summary statistics** about the spatial pattern of ESM-2 perturbations across all observed variants of a gene.

The idea comes from a biological observation: GOF mutations usually have to hit very specific positions to activate a protein (think KRAS G12, BRAF V600). LOF mutations can break function anywhere. If that's true, the distribution of where pathogenic variants cluster across the sequence should look different for GOF vs LOF genes.

We extract 8 scalar features from the pattern of variant deltas across each gene's sequence — things like how clustered or spread out the magnitudes are, how much the first principal component explains. These are then fed to a logistic regression, not the raw embeddings.

---

## TL;DR

The *spatial pattern* of observed clinical variant deltas across a gene's sequence carries mechanism signal that is almost entirely leak-free. Eight scalar features derived from how ESM-2 perturbations cluster across sequence positions push GOF AUROC from 0.578 to **0.646** and combined family-split F1 from 0.331 to **0.399** — with near-zero leakage (gene-split ≈ family-split for the scalar features). The biological interpretation: GOF mutations must hit specific hotspots to activate a protein; LOF mutations can break it anywhere. This hotspot-vs-spread distinction is readable from ESM-2 perturbation magnitudes across the observed variant positions.

---

## Setup

- **Dataset:** Gerasimavicius merged (10,231 variants, 948 genes, 3-class GOF/DN/LOF)
- **Embeddings:** cached per-residue deltas at variant positions + mean-pooled deltas
- **CV:** 5-fold gene-split AND family-split, 5 seeds (0–4)
- **Classifier:** logistic regression (L2, balanced class weights), gene-level features

---

## Features

For each gene, aggregate all observed variants to build 8 scalar features:

| Feature | Definition |
|---|---|
| `delta_mag_mean` | Mean ‖delta‖ at variant positions |
| `delta_mag_std` | Std of magnitudes |
| `delta_mag_cv` | Coefficient of variation (std/mean) — high = concentrated hotspots |
| `pos_mean_norm` | Mean variant position / max position (N vs C terminal bias) |
| `pos_std_norm` | Std of variant positions / max position (spread) |
| `pc1_var_explained` | Variance explained by PC1 of the delta matrix |
| `pc1_mean_proj` | Projection of mean delta onto PC1 |
| `n_variants_log` | log(1 + n_variants) — study depth proxy |

Three feature sets tested:
- **Baseline:** mean-pooled delta (1280-dim), gene-level average — same as result_7
- **Scalar pattern:** 8 features above only
- **Combined:** scalar pattern + mean-pooled delta (1288 features)

---

## Results (5-seed mean ± std)

| Feature set | CV | macro-F1 | GOF AUROC |
|---|---|---|---|
| Baseline (delta mean) | gene-split | 0.345 ± 0.009 | 0.587 ± 0.008 |
| Baseline (delta mean) | **family-split** | **0.331 ± 0.017** | **0.578 ± 0.021** |
| Scalar pattern | gene-split | 0.352 ± 0.014 | 0.648 ± 0.014 |
| Scalar pattern | **family-split** | **0.348 ± 0.006** | **0.646 ± 0.023** |
| Combined | gene-split | 0.405 ± 0.007 | 0.664 ± 0.004 |
| **Combined** | **family-split** | **0.399 ± 0.014** | **0.672 ± 0.012** |

---

## Key findings

### F1 — Scalar pattern features have near-zero leakage

Gene-split vs family-split for scalar pattern: F1 0.352 → 0.348 (Δ = +0.004), GOF AUROC 0.648 → 0.646 (Δ = +0.002). Essentially zero leakage. The spatial distribution of where variants fall in the sequence is a genuine cross-family signal, not a family-recognition shortcut.

Compare to the mean-pooled delta baseline: Δ = +0.014 F1 (4× larger leakage). The pattern features are cleaner than the embedding itself.

### F2 — GOF AUROC is the primary beneficiary

Scalar pattern features push GOF AUROC from 0.578 to 0.646 under family-split — a +0.068 gain. This is consistent with the biological hypothesis: GOF proteins have mutations concentrated at activating hotspots (KRAS G12, BRAF V600), while LOF variants spread across the sequence. That spatial concentration is the dominant signal in the scalar features.

### F3 — Combination reaches F1 = 0.399

Combined (scalar + mean-pooled delta) reaches family-split F1 = 0.399 ± 0.014 — the highest family-split result for ESM-2-only features in this project, on par with the contrastive projection (result_9, F1=0.397). The combination is additive because the scalar features capture spatial structure that the mean-pooled delta discards.

### F4 — Limitation: ClinVar bias

The pattern features are built from the clinical variants that happen to appear in the dataset. Genes with many variants (SCN1A: 373, KCNQ2: 214) get a meaningful spatial pattern; genes with 2–3 variants get noise. More importantly, ClinVar variants are enriched for known hotspots — the observed clustering is partly circular. This motivates result_20: a systematic in-silico scan replacing clinical variants with 100 evenly-spaced positions × 3 probe amino acids per gene.

#### How the bias actually enters the model

It's worth being explicit about the mechanism, because it's not obvious.

ESM-2 is **not** the source of the bias. ESM-2 is trained on raw natural protein sequences and knows nothing about patients or which mutations have been studied. Here it's only a *measuring tool* — it tells us how disruptive a mutation at a given position is. The bias enters **before** ESM-2, in our choice of *which positions to feed it*. We only looked where ClinVar pointed, and ClinVar points at the famous, well-studied spots.

The bias then rides into the model through the **feature values themselves** — two channels in particular:

1. **`n_variants_log` is literally a "how much was this gene studied" counter.** A famous gene has hundreds of reported variants; an obscure one has three. This feature is essentially a popularity score, not biology.

2. **The clustering/spread features are shaped by sampling, not only biology.** A heavily-studied GOF gene has the *same hotspot mutation reported hundreds of times* → the positions look tightly clustered. A lightly-studied gene has a few scattered reports → the positions look spread out. So "clustered vs spread" partly measures *how the gene was sampled*, not only where mutations actually matter.

This is what result_20's fixed, evenly-spaced scan is designed to break: every gene gets the same number of positions at the same spacing, so neither the variant count nor the clustering can carry study bias.

---

## Interpretation

### Why GOF mutations cluster and LOF mutations don't

GOF mutations require hitting precisely the right structural element to activate a protein. This mechanistic constraint creates spatial clustering in the observed variant distribution. LOF mutations have no such constraint — you can disrupt function at hundreds of positions. DN mutations cluster at interaction interfaces. ESM-2 per-residue deltas capture this through the magnitude concentration (`delta_mag_cv`, `pc1_var_explained`) and positional spread (`pos_std_norm`) features.

### Relationship to prior results

Result_9 (contrastive projection, family-split F1=0.397) and result_19 combined features (F1=0.399) reach similar performance via different routes: contrastive training adjusts the representation; pattern features add a new information source. The similar ceiling suggests that ESM-2's per-residue delta information is being approximately exhausted by these approaches on the current dataset size.

---

## Limitations

- Gene-level labels conflate multi-mechanism genes (e.g. SCN1A has both GOF and LOF variants)
- ClinVar enrichment bias — addressed by the in-silico scan (result_20)
- `n_variants_log` may proxy for "how well-studied is this gene" rather than biology — sensitivity analysis warranted
- Single model (ESM-2 650M) only

---

## Files

- `scripts/perturbation_pattern.py` — analysis script
- `results/perturbation_pattern/results.json` — 5-seed results
