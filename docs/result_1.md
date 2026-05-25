# Result 1: ESM-2 Delta-Embedding Mechanism Geometry
## Run: May 23-24, 2026 | Model: ESM-2 650M | Seed: 0

---

## Setup

- **Dataset**: Gerasimavicius et al. 2022, `ClinVar_gene_level` sheet
- **Variants**: 10,231 (GOF: 1,983 / DN: 894 / LOF: 7,354)
- **Genes**: 948
- **CV**: 5-fold gene-split
- **Stability path**: B_direct (Megascale not available; subspace fit directly on Gerasimavicius FoldX ΔΔG)
- **Bootstrap**: disabled for this run (too slow on 10k×1280; will re-enable after confirming signal)

---

## Headline Results

### Primary probe: mean-pooled stability-projected delta, 3-class gene-split CV

| Metric | Value | Pre-registered threshold |
|---|---|---|
| macro-F1 | 0.279 | — |
| AUROC GOF | 0.640 | > 0.72 = meaningful |
| AUROC DN | 0.561 | > 0.72 = meaningful |
| AUROC LOF | 0.628 | > 0.72 = meaningful |
| **Mean macro-AUROC** | **0.610** | **0.60–0.72 = weak** |

**Verdict: weak signal.** Mean macro-AUROC of 0.610 falls in the "weak signal" band (0.60–0.72). Macro-F1 of 0.279 is effectively at chance for a class-imbalanced 3-class problem (LOF = 72% of variants).

### Per-residue delta (co-primary)

| Metric | Value |
|---|---|
| macro-F1 | 0.373 |
| AUROC GOF | 0.649 |

Per-residue delta outperforms mean-pooled delta on macro-F1 (0.373 vs 0.279). The local context at the variant position carries more signal than the whole-protein mean shift. Per-residue is the more informative representation for mechanism classification.

---

## Baselines

| Baseline | macro-F1 | Notes |
|---|---|---|
| **WT-only ESM-2** | **0.580** | **Strongest result — beats delta probe** |
| One-hot AA identity | 0.280 | At chance — substitution identity carries no signal |
| FoldX ΔΔG only | 0.279 | At chance — stability alone separates nothing |
| AlphaMissense | 0.279 | At chance — pathogenicity score carries no mechanism signal |
| Shuffled delta (neg ctrl) | 0.279 | Confirms delta is at chance |

**Key finding: WT-only embeddings (macro-F1 = 0.580) substantially outperform the delta probe (0.279).** The wildtype protein representation alone classifies mechanism better than any mutation-specific signal.

---

## WT-Only Follow-up: Family-Split CV

Run manually post-hoc to test whether WT signal is family-level or mechanism-level.

| CV scheme | macro-F1 | macro-AUROC |
|---|---|---|
| Gene-split | 0.580 | ~0.62 (estimated) |
| Family-split | 0.298 | 0.528 ± 0.022 |

**AUROC drops from ~0.62 to 0.528 under family-split CV.** Signal partially collapses — the WT probe is largely learning protein family identity (kinases = GOF, structural proteins = DN) rather than mechanism per se. Some residual signal above chance (0.528 > 0.50) survives family-split, but it is weak.

Pfam coverage: 10,200/10,231 variants annotated across 662 families.

---

## Stability Subspace

- **Path B** (direct fit on Gerasimavicius FoldX ΔΔG) — Megascale not available
- Variance explained by stability subspace: GOF=63%, DN=58%, LOF=60%
- **Variance asymmetry GOF vs LOF: -0.056** — pre-registered prediction (GOF ≥ 30% less variance explained than LOF) **does not hold**
- GOF variants actually have *more* variance explained by the stability subspace than LOF, contradicting Gerasimavicius's finding that GOF mutations have milder structural effects. This may reflect Path B overfitting to the data distribution.

Projected and unprojected delta give identical macro-F1 (0.279) — stability projection makes no difference at this performance level.

---

## Orthogonality

Cosine matrix entries are NaN — the pairwise probes failed to fit (likely due to insufficient class separation or numerical issues). `null_cosine_mean = 0.418` which is unusually high for 1280-dim space (expected ~0), suggesting a numerical issue in the orthogonality analysis.

One partial result: `DN_vs_GOF|GOF_vs_LOF` cosine is distinguishable from null (True), but this is an isolated finding without the full 3×3 matrix and should not be interpreted.

---

## Family-Split CV on Delta (from main run)

| Metric | Gene-split | Family-split |
|---|---|---|
| macro-F1 | 0.279 | 0.281 |
| AUROC GOF | 0.640 | 0.590 |
| AUROC DN | 0.561 | 0.547 |
| AUROC LOF | 0.628 | 0.572 |

Delta probe shows a modest AUROC drop under family-split (0.61 → 0.57 mean). The delta signal is not purely family-level, but it's too weak to interpret meaningfully.

---

## Interpretation

### What worked
- Pipeline ran end-to-end on 10,231 real variants
- WT embeddings carry mechanism signal (macro-F1 0.58 gene-split) — ESM-2 encodes enough about protein identity/family to partially predict mechanism class without seeing any mutation
- Per-residue delta (0.373 macro-F1) is more informative than mean-pooled delta (0.279)

### What didn't work
- **Delta embeddings do not linearly encode mechanism beyond gene identity.** Subtracting the WT representation removes the signal that actually separates mechanism classes. What remains (the mutation-specific perturbation) is dominated by stability/noise and carries weak mechanism signal.
- Stability projection (Path B) made no difference — the subspace is not capturing a meaningful stability axis at this data scale.
- The pre-registered variance asymmetry prediction did not hold.

### Root cause hypotheses
1. **Gene-level labels are too coarse.** All variants from a gene get the same label regardless of which mechanism they individually act through. The delta captures variant-level perturbation; the label is gene-level. This mismatch may be fundamental.
2. **Class imbalance is severe.** LOF = 72% of variants. The probe collapses toward predicting LOF. Rebalancing (e.g. undersample LOF to match GOF+DN) might expose more signal.
3. **Stability projection is too aggressive or misdirected.** Path B fits on the same data, potentially removing mechanism-correlated variance along with stability.
4. **Linear probe is appropriate but the representation is wrong.** Mean-pooled delta averages over 500+ residues; the mechanism signal at the variant position is diluted. Per-residue delta does better, supporting this.

---

## Next Steps

1. **Rebalance classes** — undersample LOF to ~2× GOF count, re-run probe
2. **Per-residue delta as primary** — run full experiment with per-residue as headline feature
3. **MLP probe** — test whether mechanism signal is nonlinearly separable in delta space
4. **Expand dataset** — merge with G2P (~158 GOF genes, ~118 DN genes vs current 81/60) for more balanced classes
5. **WT embedding as primary** — reframe the experiment around what WT encodes; use delta as a contrast

---

## Data Location

- Results: `results/20260524_baseline_run/run_0/final_info_seed0.json`
- Detailed: `results/20260524_baseline_run/run_0/detailed_results_seed0.json`
- Embeddings: on RunPod (regenerate in ~5-10 min with optimized code)
- Cached data: `data/` — sequences.json, pfam_families.json, alphamissense_scores.json, gerasimavicius_variants.json
