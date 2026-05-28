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
- **Motivation**: linear probe (Result 1) gave macro-F1 = 0.279 (chance). Testing whether mechanism signal is present but nonlinearly organized.

---

## Results: delta_mean

| Probe | macro-F1 | AUROC GOF | AUROC DN | AUROC LOF |
|---|---|---|---|---|
| Linear (Result 1 baseline) | 0.279 | 0.640 | 0.561 | 0.628 |
| **MLP** (256→64, dropout 0.3) | **0.431 ± 0.020** | **0.744 ± 0.064** | 0.548 ± 0.066 | 0.729 ± 0.053 |
| kNN (k=10, cosine) | 0.410 ± 0.027 | 0.681 ± 0.055 | 0.573 ± 0.041 | 0.664 ± 0.024 |
| GBM (50 trees, PCA-50) | 0.336 ± 0.039 | 0.715 ± 0.077 | 0.539 ± 0.032 | 0.698 ± 0.056 |
| RF (50 trees, PCA-50) | 0.292 ± 0.030 | 0.700 ± 0.082 | 0.537 ± 0.041 | 0.676 ± 0.062 |

## Results: delta_pos (per-residue at variant position)

| Probe | macro-F1 | AUROC GOF | AUROC DN | AUROC LOF |
|---|---|---|---|---|
| Linear (Result 1 baseline) | 0.373 | 0.649 | — | — |
| **MLP** | **0.351 ± 0.031** | **0.631 ± 0.047** | 0.517 ± 0.064 | 0.643 ± 0.029 |
| kNN | 0.338 ± 0.038 | 0.624 ± 0.043 | 0.525 ± 0.056 | 0.615 ± 0.018 |
| GBM (PCA-50) | 0.297 ± 0.017 | 0.618 ± 0.041 | 0.536 ± 0.037 | 0.608 ± 0.024 |
| RF (PCA-50) | 0.285 ± 0.018 | 0.615 ± 0.042 | 0.527 ± 0.032 | 0.604 ± 0.028 |

---

## Key Findings

### 1. All nonlinear probes outperform the linear probe on delta_mean
MLP reaches macro-F1 0.431 and GOF AUROC 0.744 — crossing the pre-registered "meaningful" threshold of 0.72. kNN (0.410) is nearly as strong. GBM (0.336) and RF (0.292) are weaker, likely due to information lost in the PCA-50 reduction applied for computational tractability.

### 2. Two competing explanations for the MLP lift

**Explanation A (mechanism signal):** The delta contains real mechanism information that is nonlinearly organised. The MLP and kNN recover it; the linear probe could not.

**Explanation B (residual family leakage):** Result 4 showed that delta embeddings strip most — but not all — family signal (k=5 family purity z=+18 vs null, still 5× weaker than WT). The MLP is nonlinearly recovering this residual family clustering, not learning mechanism. Under this explanation, the 0.28→0.41 lift is the same family-recognition shortcut as the WT-only baseline, just operating on weaker residual signal.

**Explanation B is currently more parsimonious.** The linear delta probe is flat under both gene-split and family-split CV (0.279 vs 0.281) — no signal to lose, consistent with no genuine mechanism signal. The MLP lift appearing only under gene-split, where family leakage is possible, fits the leakage story. The kNN result (mechanism classes locally clustered in delta space) is also consistent with residual family clustering rather than mechanism clustering per se.

### 3. delta_pos does not benefit from nonlinearity
For `delta_pos`, nonlinear probes are roughly on par with the linear probe (0.285–0.351 vs 0.373). No hidden nonlinear structure at the variant position.

### 4. DN remains weak across all probes (AUROC ~0.52–0.57)
Consistent with Result 4: DN is the class least correlated with Pfam family in this dataset, so both the family shortcut and any residual family signal in the delta help it least.

---

## Interpretation

The MLP lift over the linear probe is real but its cause is unresolved. The result is consistent with either (A) nonlinearly-organised mechanism signal in the delta, or (B) nonlinear recovery of residual family clustering. These require different conclusions about what ESM-2 encodes.

**The resolving experiment is MLP under family-split CV.** Under family-split, family leakage is blocked. If the MLP lift collapses (family-split MLP macro-F1 ≈ 0.28), Explanation B is correct and the current results are an artefact of CV design. If the MLP lift survives (family-split MLP macro-F1 >> 0.28), Explanation A is correct and the delta genuinely encodes mechanism nonlinearly.

This is the single highest-priority next step (also listed as #1 in Result 4's next experiments).

---

## Next Steps

1. **MLP under family-split CV** — resolves the explanation A vs B question; the most important experiment to run next
2. **MLP on WT-only under family-split** — paired control; if WT-only also collapses, confirms the family-shortcut explanation is complete
3. **Visualize delta space** — UMAP of delta_mean colored by mechanism and by Pfam family; if the local clustering seen by kNN aligns with family rather than mechanism, supports Explanation B

---

## Data Location

- Results: `../results/20260524_baseline_run/run_0/mlp_results_seed0.json`
- Embeddings: `../data/embeddings/embeddings_*.npy`
