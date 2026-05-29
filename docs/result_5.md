# Result 5: Nonlinear Probes on ESM-2 Delta Embeddings
## Run: May 24, 2026 | Model: ESM-2 650M | Seed: 0

---

## Setup

- **Dataset**: Gerasimavicius et al. 2022, `ClinVar_gene_level` sheet
- **Variants**: 10,231 (GOF: 1,983 / DN: 894 / LOF: 7,354)
- **Genes**: 948
- **CV**: 5-fold gene-split (same splits as Result 1)
- **Features**: `delta_mean` (mean-pooled MT−WT), `delta_pos` (per-residue at variant position)
- **Stability projection**: none applied (testing raw delta signal)
- **Motivation**: the linear classifier from Result 1 scored macro-F1 = 0.279, which is essentially chance. This experiment tests whether there's real signal in the delta that a linear classifier simply can't find.

---

## Background: what these classifiers are doing

In Result 1 we used a **linear probe** — roughly, a straight-line decision boundary in embedding space. If the mechanism classes (GOF/DN/LOF) aren't arranged in a way a straight line can separate, that probe will fail even if real signal exists.

This experiment swaps in classifiers that can learn curved or irregular decision boundaries:

- **MLP** (multilayer perceptron): a small neural network — two layers with 256 and 64 neurons — that can learn non-straight decision boundaries.
- **kNN** (k-nearest neighbours): classifies a variant by looking at its 10 most similar variants in embedding space and taking a majority vote.
- **GBM / RF** (gradient boosting and random forest): tree-based methods that split the space into regions. Run on a compressed version of the embeddings (PCA-50: 1280 dimensions reduced to 50) for speed.

All metrics are cross-validated across 5 folds. The ± values are standard deviations across folds.

**AUROC** (area under the ROC curve) measures how well a classifier ranks one class above the others. 0.5 = random, 1.0 = perfect. The pre-registered threshold for "meaningful" signal in this study is 0.72.

---

## Results: delta_mean (whole-protein average shift)

| Classifier | macro-F1 | GOF AUROC | DN AUROC | LOF AUROC |
|---|---|---|---|---|
| Linear (Result 1 baseline) | 0.279 | 0.640 | 0.561 | 0.628 |
| **MLP** (256→64, dropout 0.3) | **0.431 ± 0.020** | **0.744 ± 0.064** | 0.548 ± 0.066 | 0.729 ± 0.053 |
| kNN (k=10, cosine) | 0.410 ± 0.027 | 0.681 ± 0.055 | 0.573 ± 0.041 | 0.664 ± 0.024 |
| GBM (50 trees, PCA-50) | 0.336 ± 0.039 | 0.715 ± 0.077 | 0.539 ± 0.032 | 0.698 ± 0.056 |
| RF (50 trees, PCA-50) | 0.292 ± 0.030 | 0.700 ± 0.082 | 0.537 ± 0.041 | 0.676 ± 0.062 |

## Results: delta_pos (shift at the specific mutated position only)

| Classifier | macro-F1 | GOF AUROC | DN AUROC | LOF AUROC |
|---|---|---|---|---|
| Linear (Result 1 baseline) | 0.373 | 0.649 | — | — |
| **MLP** | **0.351 ± 0.031** | **0.631 ± 0.047** | 0.517 ± 0.064 | 0.643 ± 0.029 |
| kNN | 0.338 ± 0.038 | 0.624 ± 0.043 | 0.525 ± 0.056 | 0.615 ± 0.018 |
| GBM (PCA-50) | 0.297 ± 0.017 | 0.618 ± 0.041 | 0.536 ± 0.037 | 0.608 ± 0.024 |
| RF (PCA-50) | 0.285 ± 0.018 | 0.615 ± 0.042 | 0.527 ± 0.032 | 0.604 ± 0.028 |

---

## Key Findings

### 1. More complex classifiers do better on delta_mean

The MLP reaches macro-F1 0.431 and GOF AUROC 0.744, clearing the pre-registered "meaningful" threshold of 0.72. kNN (0.410) is nearly as strong. GBM (0.336) and RF (0.292) are weaker, likely because they run on PCA-compressed embeddings (50 dimensions instead of 1280), losing some information in the process.

So something in the delta embeddings does allow better-than-chance mechanism prediction — but only if the classifier is expressive enough to find it.

### 2. But there are two very different explanations for why

**Explanation A (good news — mechanism signal):** The delta genuinely contains information about disease mechanism, but that information isn't arranged in a simple linear way. More powerful classifiers can find it; the linear probe couldn't.

**Explanation B (bad news — family leakage):** Result 4 showed that subtracting WT from mutant removes most, but not all, of the protein-family signal. A small residual remains. More powerful classifiers are better at picking up faint signals — so the MLP lift may just be the MLP recovering that leftover family information, not mechanism at all.

**Explanation B is currently more likely.** The key evidence: the linear delta probe scores almost identically whether we hold out genes or hold out entire protein families (0.279 vs 0.281). If there were real mechanism signal, holding out families should hurt more. The fact that it doesn't suggests there's no signal to lose — and that the MLP lift only appears under the gene-split setup, where family leakage is still possible.

### 3. Using only the mutated position doesn't help

For `delta_pos` (the shift at just the mutated residue, rather than averaged across the whole protein), more complex classifiers perform about the same as or worse than the linear probe. Whatever the MLP is picking up in `delta_mean`, it's not something concentrated at the mutation site.

### 4. DN (dominant negative) stays hard for all classifiers (AUROC ~0.52–0.57)

Consistent with Result 4: DN mutations are the least correlated with protein family in this dataset, which means neither the family shortcut nor the residual family signal in the delta helps with DN. DN is also the smallest class (894 variants) and covers very different biological mechanisms under one label.

---

## Interpretation

The MLP does better than the linear probe — that's real. But we don't yet know *why*.

- If it's Explanation A, the delta genuinely encodes mechanism nonlinearly, and this is an interesting positive finding.
- If it's Explanation B, the MLP is just better at detecting a faint protein-family signal that we haven't fully scrubbed out, and the result is an artefact of how we split the data.

**The experiment that resolves this:** run the MLP with family-split cross-validation, where entire protein families are held out from the test set. If the MLP lift collapses back to ~0.28, Explanation B is right. If it survives, Explanation A is right and we have a genuine finding.

This is the single most important experiment to run next.

---

## Next Steps

1. **MLP under family-split CV** — resolves the A vs B question directly
2. **MLP on WT-only under family-split** — paired control; if the WT-only signal also collapses under family-split, it confirms the family-shortcut story is complete
3. **Visualize the delta space** — UMAP plot of delta_mean colored by mechanism class and by protein family; if the clusters the kNN is using align with family rather than mechanism, that's further evidence for Explanation B

---

## Data Location

- Results: `../results/20260524_baseline_run/run_0/mlp_results_seed0.json`
- Embeddings: `../data/embeddings/embeddings_*.npy`
