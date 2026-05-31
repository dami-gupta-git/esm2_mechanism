# Result 1: ESM-2 Delta-Embedding Mechanism Geometry
## Run: May 23-24, 2026 | Model: ESM-2 650M | Seed: 0

---

## Background: what this experiment is testing

Each protein variant in the dataset has been labelled with a **disease mechanism**: gain-of-function (GOF — the mutant protein does too much), dominant negative (DN — the mutant protein actively interferes with the normal copy), or loss-of-function (LOF — the mutant protein does too little or nothing).

ESM-2 is a protein language model — it converts a protein sequence into a list of numbers (an "embedding") that encodes something about the protein's structure and biology. For each variant, we computed two embeddings: one for the wildtype (normal) sequence and one for the mutant. The **delta** is just the mutant embedding minus the wildtype embedding — it captures how the mutation shifted the model's representation of the protein.

The question: can we use that shift to predict which mechanism class the variant belongs to?

We test this with a **linear classifier** : a simple model that draws a flat decision boundary in the embedding space. If the three mechanism classes cluster in different regions, the classifier will find it.

**AUROC** (area under the ROC curve) measures how well a classifier ranks one class above the others. 0.5 = random, 1.0 = perfect. The pre-registered threshold for "meaningful" in this study is > 0.72.   
**Macro-F1** is the average F1 score across all three classes — it penalises classifiers that just predict the most common class.

---

## Setup

- **Dataset**: Gerasimavicius et al. 2022, `ClinVar_gene_level` sheet
- **Variants**: 10,231 (GOF: 1,983 / DN: 894 / LOF: 7,354)
- **Genes**: 948
- **CV**: 5-fold gene-split (genes are split across folds, so the model can't memorise a gene it saw in training)
- **Stability path**: B_direct (a physics-based stability score was used to define a "stability direction" in embedding space, then projected out — Megascale data wasn't available so we fit this directly on the dataset's own FoldX ΔΔG values)
- **Bootstrap**: disabled for this run (too slow on 10k×1280 dimensions; will re-enable after confirming signal)

---

## Headline Results

### Primary result: mean-pooled delta, 3-class cross-validation

The delta was averaged across all residue positions in the protein (mean-pooled) and fed into the linear classifier.

| Metric | Value | Pre-registered threshold |
|---|---|---|
| macro-F1 | 0.279 | — |
| AUROC GOF | 0.640 | > 0.72 = meaningful |
| AUROC DN | 0.561 | > 0.72 = meaningful |
| AUROC LOF | 0.628 | > 0.72 = meaningful |
| **Mean macro-AUROC** | **0.610** | **0.60–0.72 = weak** |

**Verdict: weak signal.** Mean macro-AUROC of 0.610 falls in the "weak signal" band. Macro-F1 of 0.279 is effectively at chance — a classifier that just always predicts LOF (the most common class, 72% of variants) would score similarly.

### Per-residue delta (co-primary)

Instead of averaging across the whole protein, we used only the delta at the specific mutated position.

| Metric | Value |
|---|---|
| macro-F1 | 0.373 |
| AUROC GOF | 0.649 |

This does better than the whole-protein average (0.373 vs 0.279 F1). The local context right at the mutation site carries more signal than the average across the whole protein.

---

## Baselines

To put the delta results in context, we also tested simpler features:

| Feature | macro-F1 | Notes |
|---|---|---|
| **WT-only ESM-2** | **0.580** | **Strongest result — beats the delta** |
| One-hot AA identity | 0.280 | Just encoding which amino acid changed — at chance |
| FoldX ΔΔG only | 0.279 | A physics-based stability score — at chance |
| AlphaMissense | 0.279 | A state-of-the-art pathogenicity predictor — at chance |
| Shuffled delta (negative control) | 0.279 | Randomly permuted delta — confirms delta is at chance |

**The most striking finding: using only the wildtype embedding (no mutation information at all) gives macro-F1 = 0.580.** That's much better than the delta. The protein's normal sequence encodes enough information to partially predict its mechanism class — even without knowing what mutation occurred.

---

## WT-Only Follow-up: Family-Split CV

The WT-only result looks impressive, but there's a catch. If related proteins (e.g. all kinases) tend to share the same mechanism label, the classifier might just be recognising protein families rather than learning anything about mechanism.

To test this, we re-ran with **family-split CV**: instead of splitting by gene, we held out entire protein families from the test set. This is a harder test — the classifier can't use any protein from the same family as a hint.

| CV scheme | macro-F1 | macro-AUROC |
|---|---|---|
| Gene-split | 0.580 | ~0.62 (estimated) |
| Family-split | 0.298 | 0.528 ± 0.022 |

**Performance drops sharply.** The WT signal mostly disappears when families are held out — it was largely learning "kinases = GOF, structural proteins = DN" rather than anything more fundamental. A small residual above chance (0.528 > 0.50) survives, but it's weak.

Pfam coverage: 10,200/10,231 variants annotated across 662 families.

---

## Stability Subspace

We tried projecting the delta onto (and away from) a "stability direction" in embedding space — the hypothesis being that stability-related variation might be obscuring mechanism signal.

- Variance of the delta explained by the stability direction: GOF=63%, DN=58%, LOF=60%
- **Pre-registered prediction** (GOF variants should have ≥30% less variance along the stability direction than LOF) **does not hold** — GOF actually has *more* variance explained, not less
- After projecting out the stability direction, macro-F1 is unchanged at 0.279 — removing stability information doesn't help or hurt

---

## Orthogonality

We tried to measure whether the three mechanism classes point in distinct directions in delta space. This analysis mostly failed — numerical issues (likely due to how poorly separated the classes are) produced NaN values across most of the matrix. The one partial result shouldn't be interpreted on its own.

---

## Family-Split CV on Delta

| Metric | Gene-split | Family-split |
|---|---|---|
| macro-F1 | 0.279 | 0.281 |
| AUROC GOF | 0.640 | 0.590 |
| AUROC DN | 0.561 | 0.547 |
| AUROC LOF | 0.628 | 0.572 |

The delta's AUROC drops a little under family-split (mean 0.61 → 0.57), but F1 is essentially unchanged. The signal is too weak to say much either way.

---

## Interpretation

### What worked
- The pipeline ran end-to-end on 10,231 real variants
- WT embeddings carry some mechanism signal (macro-F1 0.58 under gene-split) — ESM-2 encodes enough about a protein's identity to partially predict its mechanism class without even seeing the mutation
- The delta at the specific mutated position (0.373) is more informative than averaging across the whole protein (0.279)

### What didn't work
- **The delta doesn't encode mechanism in a way a linear classifier can find.** Subtracting the wildtype embedding removes the information that actually separates mechanism classes. What's left is dominated by stability noise.
- Projecting out stability made no difference.
- The pre-registered prediction about GOF vs LOF variance did not hold.

### Why might this be?
1. **Gene-level labels are coarse.** Every variant in a gene gets the same mechanism label, regardless of what that specific variant actually does. The delta captures a variant-level perturbation; the label is gene-level. This mismatch may be fundamental.
2. **Class imbalance is severe.** LOF = 72% of variants. The classifier tends to just predict LOF. Rebalancing the classes might reveal more signal.
3. **Averaging over the whole protein loses the signal.** Averaging across 500+ residues dilutes the information at the mutation site. Per-residue delta does better, which fits this story.
4. **Stability projection may be removing the wrong thing.** Fitting the stability direction on the same data might accidentally remove mechanism-correlated variance too.

---

## Next Steps

1. **Rebalance classes** — undersample LOF to ~2× GOF count, re-run
2. **Per-residue delta as primary** — run the full experiment using the per-residue delta as the main feature
3. **Nonlinear classifier** — test whether the mechanism signal is there but just not linearly separable
4. **Expand dataset** — merge with G2P (~158 GOF genes, ~118 DN genes vs current 81/60) for more balanced classes
5. **WT embedding as primary** — reframe the experiment around what the wildtype encodes; use the delta as a contrast

---

## Data Location

- Results: `../results/20260524_baseline_run/run_0/final_info_seed0.json`
- Detailed: `../results/20260524_baseline_run/run_0/detailed_results_seed0.json`
- Embeddings: on RunPod (regenerate in ~5-10 min with optimized code)
- Cached data: `../data/` — sequences.json, pfam_families.json, alphamissense_scores.json, gerasimavicius_variants.json
