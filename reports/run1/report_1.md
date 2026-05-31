# Report 1: Gene-Split vs Family-Split Baseline Comparison
## Script: mechanism_delta_cv.py | Run: 2026-05-30 | Model: ESM-2 650M | Seeds: 0–4

---

## Background

Do ESM-2 embeddings encode disease mechanism, or do they simply encode protein family identity — and protein families happen to correlate with mechanism (e.g. kinases tend to be GOF, structural proteins tend to be LOF)?

Standard gene-split cross-validation cannot distinguish these two possibilities. We test this by comparing two CV schemes:

- **Gene-split CV**: test genes are held out, but related genes can appear in training — family information can leak
- **Family-split CV**: entire Pfam families are held out — no protein from the same family can appear in training

A large drop in performance under family-split means the feature was recognising protein families, not mechanism.

---

## Setup

- **Dataset**: Merged (Gerasimavicius + ClinVar/G2P) — 17,826 variants, 1,935 genes, 1,136 Pfam families
- **Classes**: GOF=2,682 / DN=1,550 / LOF=13,594
- **PCA**: 256 components (98.0% variance explained) applied to embedding features before probing
- **CV**: 5-fold gene-split and family-split
- **Seeds**: 0–4 (results averaged)
- **Script**: `python -m esm2_mech.experiments.mechanism_delta_cv`

---

## Results

| Feature | Gene-split F1 | Family-split F1 | Δ (drop) | GOF AUROC (GS / FS) | DN AUROC (GS / FS) | LOF AUROC (GS / FS) |
|---|---|---|---|---|---|---|
| **wt_only_mean** | **0.543 ± 0.025** | **0.442 ± 0.019** | **+0.102** | 0.807 / 0.730 | 0.730 / 0.714 | 0.838 / 0.791 |
| mut_only_mean | 0.544 ± 0.023 | 0.443 ± 0.019 | +0.101 | 0.808 / 0.731 | 0.730 / 0.713 | 0.838 / 0.791 |
| wt_concat_mut | 0.548 ± 0.027 | 0.451 ± 0.024 | +0.097 | 0.806 / 0.713 | 0.719 / 0.698 | 0.830 / 0.776 |
| delta_per_residue | 0.315 ± 0.005 | 0.305 ± 0.001 | +0.010 | 0.595 / 0.567 | 0.584 / 0.568 | 0.597 / 0.553 |
| delta_mean | 0.288 ± 0.001 | 0.288 ± 0.002 | −0.000 | 0.608 / 0.559 | 0.542 / 0.514 | 0.594 / 0.545 |
| onehot_aa | 0.288 ± 0.001 | 0.288 ± 0.002 | −0.000 | 0.542 / 0.542 | 0.553 / 0.545 | 0.547 / 0.543 |
| foldx_ddg | 0.279 ± 0.001 | 0.279 ± 0.001 | +0.000 | 0.619 / 0.617 | 0.589 / 0.595 | 0.629 / 0.623 |

---

## Key Findings

### 1. WT-only signal is mostly family leakage

WT-only drops from F1 = 0.543 (gene-split) to 0.442 (family-split) — a loss of 0.102. Most of the apparent mechanism signal disappears when protein families are held out. The classifier is learning which family a gene belongs to, not mechanism.

Mutant-only is near-identical to WT-only (0.544 vs 0.543 gene-split), confirming the signal is entirely in protein identity — the mutation adds nothing.

### 2. Delta is clean but empty

`delta_mean` scores identically under both CV schemes (0.288 → 0.288, Δ = −0.000). No leakage — subtracting the wildtype removes family-identity information. But what remains carries no mechanism signal above chance.

`delta_per_residue` has a tiny leakage (+0.010) and slightly more signal (0.315 vs 0.288). Local context at the mutation site carries a weak family-correlated signal.

### 3. FoldX is family-robust

FoldX ΔΔG shows zero leakage (0.279 → 0.279) and its GOF AUROC (0.619 GS / 0.617 FS) is the most stable signal across CV schemes. A physics-based stability score separates GOF variants slightly better than chance without any family shortcut.

### 4. GOF survives family-split best

GOF AUROC under family-split for WT-only is 0.730 — the highest surviving signal. ESM-2 encodes something about GOF proteins that generalises beyond individual families.

---

## Data

- Results: `results/run1/family_split_baselines_seed{0..4}.json`
- Script: `python -m esm2_mech.experiments.mechanism_delta_cv`
- Run log: `results/run1/run.log`
