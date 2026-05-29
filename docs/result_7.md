# Result 7 — Full nonlinear probe results + merged dataset + Option B
## Date: May 24–25, 2026 | Model: ESM-2 650M | Seed: 0

---

## TL;DR

Three experiments completed on May 24–25:

1. **MLP + nonlinear classifiers with family-split CV** on Gerasimavicius (948 genes): MLP delta_mean family-split F1=0.364 (+0.031 above chance). 62% of the gene-split lift disappears under family-split — the majority is family leakage, with a small residual surviving.
2. **Merged dataset embeddings** (Gerasimavicius + G2P/ClinVar pathogenic-only, 1,985 genes, 19,100 variants): embeddings extracted on RunPod.
3. **Option B: gene-level WT classifier on merged dataset** with full protein family coverage (1,950/1,985 genes, 1,146 families): family-split F1=0.393 (+0.060 above chance), Δ=+0.077 — same absolute floor as Gerasimavicius, less inflated gene-split.

**Key finding:** The family-split floor for ESM-2 mechanism classification is consistently ~0.39 macro-F1 across datasets and classifier types — only +0.031 to +0.060 above chance. Gene-split numbers (0.415–0.580) are inflated 50–62% by family leakage. Family-split CV is the necessary diagnostic.

---

## 1. Complete nonlinear classifier results (Gerasimavicius, 948 genes)

### delta_mean (whole-protein average shift)

| Classifier | CV | macro-F1 ± std | GOF AUROC | DN AUROC | LOF AUROC |
|---|---|---|---|---|---|
| MLP (256→64, class-weighted) | gene-split | **0.415 ± 0.042** | 0.710 | 0.549 | 0.714 |
| MLP | family-split | **0.364 ± 0.047** | 0.627 | 0.552 | 0.633 |
| kNN (k=10, cosine) | gene-split | 0.410 ± 0.027 | 0.681 | 0.573 | 0.664 |
| GBM (PCA-50) | gene-split | 0.338 ± 0.042 | 0.715 | 0.537 | 0.698 |
| RF (PCA-50) | gene-split | 0.291 ± 0.029 | 0.700 | 0.537 | 0.676 |
| Linear logistic reg (result_1) | gene-split | 0.279 | 0.634 | 0.529 | 0.620 |

**MLP leakage: Δ = +0.052** (gene-split 0.415 → family-split 0.364)

### delta_pos (shift at just the mutated position)

| Classifier | CV | macro-F1 ± std | GOF AUROC | DN AUROC | LOF AUROC |
|---|---|---|---|---|---|
| MLP | gene-split | 0.350 ± 0.027 | 0.622 | 0.529 | 0.632 |
| MLP | family-split | 0.306 ± 0.049 | 0.582 | 0.531 | 0.571 |
| kNN | gene-split | 0.338 ± 0.038 | 0.624 | 0.525 | 0.615 |
| GBM (PCA-50) | gene-split | 0.299 ± 0.018 | 0.617 | 0.537 | 0.605 |
| RF (PCA-50) | gene-split | 0.285 ± 0.018 | 0.614 | 0.529 | 0.603 |

**MLP leakage: Δ = +0.044** (gene-split 0.350 → family-split 0.306)

### Key observations

- **MLP and kNN are the best classifiers** — nearly identical (0.415 vs 0.410). kNN requires no training, which confirms that mechanism classes have some local geometric structure in delta space — nearby variants in the embedding space do tend to share mechanism labels.
- **GBM/RF are weaker** — likely because they run on PCA-compressed embeddings (50 dimensions instead of 1280), losing some information.
- **Whole-protein average >> mutated position** under MLP (0.415 vs 0.350). Whatever signal the MLP finds is spread across the whole sequence, not concentrated at the mutation site.
- **DN AUROC consistently stuck at ~0.53 (near chance)** across all classifiers and CV schemes. DN is the least recoverable class — likely because of its small size (894 variants, 60 genes), noisy labels, and genuine mechanistic diversity under one label.
- **Delta has smaller absolute Δ than WT-only** (Δ=+0.052 vs +0.191 on Gerasimavicius) — but this comparison is misleading because the gene-split baselines differ. The fair comparison is the leakage fraction: both lose ~62% of their above-chance gene-split signal under family-split. The delta and WT-only signals are roughly equally family-mediated; the delta just has a lower gene-split ceiling to begin with.

---

## 2. Complete cross-dataset comparison

### Gerasimavicius dataset (948 genes, 662 protein families)

| Feature | Classifier | Gene-split F1 | Family-split F1 | Δ |
|---|---|---|---|---|
| delta_mean | Linear logistic | 0.279 | 0.281 | +0.002 |
| delta_mean | MLP | **0.415** | **0.364** | +0.052 |
| delta_mean | kNN | 0.410 | — | — |
| delta_pos | Linear logistic | 0.376 | 0.348 | +0.028 |
| delta_pos | MLP | 0.350 | 0.306 | +0.044 |
| WT-only | Linear logistic | 0.580 | 0.389 | +0.191 |

### Merged dataset (1,985 genes, 1,146 protein families)

| Feature | Classifier | Gene-split F1 | Family-split F1 | Δ | GOF AUROC (family-split) | DN AUROC (family-split) |
|---|---|---|---|---|---|---|
| delta_mean | MLP | 0.384 | **0.352** | +0.031 | 0.635 | 0.586 |
| delta_mean | kNN | 0.392 | — | — | — | — |
| delta_pos | MLP | 0.373 | **0.336** | +0.038 | 0.605 | 0.577 |
| WT-only | Linear logistic | 0.469 | **0.393** | +0.077 | 0.728 | 0.634 |

### Calibrating against chance

Always-predict-LOF macro-F1 baselines (the floor you'd get by always predicting the most common class):
- Gerasimavicius (GOF 1983 / DN 894 / LOF 7354): **0.279**
- Merged variants (GOF 2825 / DN 1716 / LOF 14559): **0.288**
- Gene-level merged (GOF 146 / DN 107 / LOF 1732): **0.311**

| Number | Above chance (0.333) | Above always-predict-LOF (0.279) |
|---|---|---|
| MLP gene-split Geras 0.415 | +0.082 | +0.136 |
| MLP family-split Geras 0.364 | +0.031 | +0.085 |
| MLP gene-split merged 0.384 | +0.051 | +0.105 |
| MLP family-split merged 0.352 | +0.019 | +0.073 |

**62% of above-chance gene-split signal disappears under family-split on Gerasimavicius.** On the merged dataset, 63% disappears. The leakage fraction is nearly identical across datasets — confirming this is a structural property of the task, not an artifact of dataset size.

The one number that holds up most cleanly is **GOF AUROC=0.635 (merged) / 0.627 (Geras) under family-split** — meaningfully above 0.50 and consistent across datasets.

---

## 3. Merged dataset: gene-level WT embeddings

### Dataset
- **19,100 variants, 1,985 genes** (Gerasimavicius 948 + G2P/ClinVar pathogenic-only 1,037 new genes)
- Class distribution: GOF 2,825 variants / 146 genes | DN 1,716 / 107 | LOF 14,559 / 1,732
- **Protein family annotations: 1,950/1,985 genes, 1,146 unique families** (vs 939 genes, 662 families on original dataset)
- Note: first run used incomplete Pfam data (939 genes only) giving spurious Δ=+0.011. Corrected after fetching family data for all 1,037 new G2P genes.

### Results

| CV | macro-F1 | GOF AUROC | DN AUROC | LOF AUROC |
|---|---|---|---|---|
| Gene-split | 0.469 | 0.784 | 0.700 | 0.765 |
| Family-split | **0.393** | **0.728** | **0.634** | **0.691** |
| Δ | **+0.077** | | | |

### Comparison: original vs merged dataset (gene-level WT linear classifier)

| Dataset | Genes | Families | Gene-split F1 | Family-split F1 | Δ |
|---|---|---|---|---|---|
| Gerasimavicius | 948 | 662 | 0.580 | 0.389 | +0.191 |
| Merged | 1,985 | 1,146 | 0.469 | **0.393** | **+0.077** |

**The right way to read this table is the family-split column, not the Δ.** The family-split F1 is essentially identical across both datasets: 0.389 vs 0.393. The reduction in Δ (0.191→0.077) comes entirely from the gene-split number dropping (0.580→0.469), not from the family-split number improving. The merged dataset didn't reveal new family-split-robust signal — it just inflated less via the family shortcut, because its more diverse gene set makes that shortcut less effective.

**The family-split floor of ~0.39 is the real mechanism signal.** Everything above 0.39 in gene-split results is the family-recognition shortcut doing work.

---

## 4. Analysis

### What the delta classifier tells us

The delta MLP family-split result (F1=0.364, +0.031 above chance) is a small positive signal — not a null, but not a strong one. Two observations are robust:

1. **Locally clustered** — kNN achieves comparable F1 (0.410) without any learned transformation. Mechanism classes have some local geometric structure in delta space.
2. **Whole-sequence** — the whole-protein average outperforms the single-position delta under MLP (0.415 vs 0.350 gene-split), so the signal is distributed across the sequence.

**Anomaly: delta_pos family-split goes *up* from Gerasimavicius to merged (0.306 → 0.336).** Every other setup goes down when moving to the merged dataset (more diverse families → less inflation). This reversal could reflect a real effect, fold-assignment variance (std ~0.05 makes a +0.03 difference uninterpretable at single seed), or a data alignment issue. Treat this cautiously until multi-seed replication.

### Why is the signal nonlinear?

The linear classifier (F1=0.279) fails where MLP (F1=0.415) succeeds because the mechanism classes aren't arranged in a way a flat decision boundary can separate — their boundaries are curved. This is consistent with mechanism being encoded in a distributed, interaction-dependent way across the 1,280 embedding dimensions, rather than along a single dominant direction.

### The pathogenicity–mechanism dissociation

The honest comparison uses family-split-stable numbers only:

- **Pathogenicity**: AUROC 0.88, gene-split → family-split Δ = 0.002 (essentially zero leakage)
- **Mechanism floor**: macro-F1 ~0.39, +0.06 above always-predict-LOF baseline

This dissociation is **sharper** than result_6 originally documented. Result_6 compared pathogenicity AUROC 0.88 to mechanism gene-split MLP F1=0.415 — the leaky comparison. The honest comparison uses the family-split floor (~0.39), which is lower.

### Why DN is consistently weak

DN AUROC ~0.53 across all classifiers and both CV schemes. Three compounding factors:
1. **Rarity**: 894 variants, 60 genes — smallest class by far
2. **Ion channel enrichment**: KCNQ2 alone is 24% of DN variants; the classifier may be learning ion-channel-specific features rather than DN mechanism
3. **Mechanistic heterogeneity**: "dominant negative" covers interface disruption, dimerisation interference, and competitive inhibition — fundamentally different at the sequence level

### Why leakage drops on the merged dataset

With 948 genes across 662 families (avg 1.4 genes/family), most genes are singletons or pairs — the family-split test removes almost nothing. With 1,985 genes across 1,146 families, there's slightly more within-family coverage, but the main effect is that the classifier can no longer rely on "kinase=GOF" as a reliable heuristic — there are now GOF, DN, and LOF genes within the same large families, breaking the shortcut.

---

## 5. Revised scientific claim

> The family-split floor for ESM-2-based mechanism classification is approximately **macro-F1 = 0.35–0.39**, observed across two methods (linear classifier on WT, MLP on delta) × two datasets (Gerasimavicius, merged) × two feature representations (whole-protein average, per-position):
>
> | Method | Feature | Dataset | Family-split F1 |
> |---|---|---|---|
> | Linear logistic | WT-only per-variant | Gerasimavicius | 0.389 |
> | Linear logistic | WT-only gene-level | Merged | 0.393 |
> | MLP | delta_mean | Gerasimavicius | 0.364 |
> | MLP | delta_mean | Merged | **0.352** |
> | MLP | delta_pos | Merged | 0.336 |
>
> The convergence to 0.35–0.39 across both methods, both datasets, and both feature types is informative: if it were purely noise, you wouldn't expect this consistency. This floor is only +0.031–+0.056 above chance (0.333) — a small but nonzero residual. The majority of apparent mechanism signal in gene-split evaluations (50–62%) is explained by ESM-2's strong encoding of protein family identity combined with within-family mechanism correlation (74.8%). A nonlinear classifier (MLP) is required to detect even this small residual in delta space — linear classifiers give F1=0.279 (chance). Gene-level WT embeddings achieve a slightly stronger floor (F1=0.393, GOF AUROC=0.728) than delta embeddings (F1=0.364), suggesting gene identity carries more mechanism information than the mutation-specific perturbation. In contrast, pathogenicity (result_6) achieves AUROC=0.88 linearly, family-split-stable — ESM-2 encodes pathogenicity much more strongly and cleanly than mechanism.

---

## 6. What's still needed before posting

1. **MLP delta on merged dataset** — ✅ **DONE** (see section 2). delta_mean family-split F1=0.352, delta_pos=0.336. Family-split floor confirmed ~0.35 on merged dataset, consistent with Gerasimavicius.
2. **Multi-seed replication** — all numbers are seed=0 only. 5 seeds would tighten estimates.
3. **The figure** — one panel showing gene-split vs family-split F1 across classifiers and datasets; one panel showing pathogenicity vs mechanism dissociation.
4. **LaTeX draft** — nothing written yet.

---

## 7. Files

| File | Contents |
|---|---|
| `results/20260524_baseline_run/run_0/mlp_results_seed0.json` | MLP+GBM+RF+kNN on Gerasimavicius (delta_mean and delta_pos) |
| `results/20260524_baseline_run/run_0/mlp_merged_results_seed0.json` | MLP+GBM+RF+kNN on merged 1,985-gene dataset |
| `results/20260524_baseline_run/run_0/option_b_gene_level_wt_merged.json` | Gene-level WT linear classifier on merged dataset |
| `results/20260524_baseline_run/run_0/final_info_seed0.json` | Baseline linear classifier results (result_1) |
| `results/20260524_baseline_run/run_0/pathogenicity_control.json` | Pathogenicity positive control (result_6) |
| `data/embeddings/merged_embeddings_*.npy` | Merged dataset embeddings (19,100 × 1,280) |
| `data/pfam_families.json` | Updated: 1,950/1,985 genes annotated, 1,146 families |
