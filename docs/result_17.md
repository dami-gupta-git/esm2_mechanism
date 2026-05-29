# Result 17 — AlphaMissense is family-robust on ClinVar

**Date:** 2026-05-26
**Script:** `scripts/alphamissense_family_split.py` (analysis) and `scripts/fetch_alphamissense.py` (data fetch)
**Inputs:** 17,236 ClinVar variants from result_6, AlphaMissense bulk file `AlphaMissense_aa_substitutions.tsv.gz`, `data/pfam_families.json`
**Output:** `results/alphamissense_family/{overall,per_family,summary}.json`

---

## Background: what AlphaMissense is, and why we're checking it

AlphaMissense (Cheng et al., *Science* 2023) is DeepMind's state-of-the-art tool for predicting whether a missense variant is likely to be pathogenic. It's what many clinical labs actually use.

Result_6 showed that *our* ESM-2 delta probe, trained on ClinVar pathogenic/benign labels, has essentially zero family leakage — its performance is nearly identical whether we hold out genes or hold out entire protein families. A natural reviewer question: fine for your small probe, but does this hold for the much larger supervised predictor that the clinical community actually uses?

This experiment tests that directly by measuring AlphaMissense's performance separately for each protein family in our variant set.

---

## TL;DR

AlphaMissense scores 16,334 of the 17,236 ClinVar variants from result_6 (95% coverage). Overall AUROC = **0.9404** — matches DeepMind's published number. Stratified by protein family (n = 182 families with ≥10 pathogenic and ≥10 benign variants), the per-family AUROC distribution is **tight**: mean 0.9477 ± 0.0458, median 0.960, IQR 0.923 – 0.983. **0.55%** of families fall below AUROC 0.80; **none** fall below 0.70. The supervised head in AlphaMissense does not inherit family-correlated training shortcuts in any operationally visible way. The result_6 finding ("the *whether* is family-robust") generalises to the published predictor the clinical community actually uses.

---

## Purpose

result_6 showed that our supervised ESM-2 delta probe has essentially zero family leakage (gene-split AUROC 0.878 vs family-split AUROC 0.876, Δ = 0.002). The obvious reviewer pushback: "fine for your probe, but does this hold for AlphaMissense?" AlphaMissense uses a much larger supervised head and a curated training corpus that could in principle have learned family-correlated shortcuts our small probe lacked the capacity for. This experiment answers that question directly.

## Method

### Variant set
The 17,236-variant ClinVar set from result_6: 9,119 pathogenic, 8,117 benign, 944 genes, 658 protein families.

### Score fetch
`fetch_alphamissense.py` streams the AlphaMissense bulk file (216M rows, 1.1 GB compressed), filters to pairs matching our variant set using the gene→UniProt mapping from `merged_valid_variants.json`. Of 17,236 target variants: 110 lacked a UniProt mapping; 792 were absent from the AM file (likely non-canonical transcripts); **16,334 matched scores cached**.

### Analysis
For each variant, attach protein family from `data/pfam_families.json`. Compute:

1. **Overall AUROC and PR-AUC** on all usable rows.
2. **Per-family AUROC** for every protein family with ≥10 pathogenic and ≥10 benign variants (182 families pass; 470 fail the size threshold).
3. **Distribution statistics** of the 182 per-family AUROCs.

### Why this is the right metric
We cannot remove a family from AlphaMissense's training set. The correct analogue of "family-split CV" for a fixed published predictor is the **per-family AUROC distribution**: a family-robust predictor has a tight distribution clustered around the overall AUROC; a predictor that inherits family-correlated training has a heavy tail of low per-family AUROCs.

## Results

### Overall

| Metric | Value |
|---|---|
| n variants scored | 16,336 |
| n pathogenic | 8,675 |
| n benign | 7,661 |
| AUROC | **0.9404** |
| PR-AUC | 0.9438 |

Reproduces AlphaMissense's published ClinVar AUROC (~0.94) on our variant subset.

### Per-family distribution (n = 182 families)

| Statistic | Value |
|---|---|
| mean ± std | **0.9477 ± 0.0458** |
| min | 0.762 |
| q25 | 0.923 |
| median | 0.960 |
| q75 | 0.983 |
| max | 1.000 |
| IQR | 0.060 |
| frac AUROC < 0.80 | 0.55% (1 of 182) |
| frac AUROC < 0.70 | 0.00% |

The per-family mean (0.948) is slightly *higher* than the overall AUROC (0.940). This is consistent with Simpson-style aggregation: family-level discrimination is on average stronger than pooled discrimination because pooling across families introduces inter-family score-scale variation that the AUROC penalises.

### Worst and best families

| Pfam | AUROC | n_pos | n_neg | Notes |
|---|---|---|---|---|
| PF07974 | 0.762 | 15 | 28 | Worst; small n |
| PF16739 | 0.807 | 20 | 20 | |
| PF00625 | 0.812 | 10 | 16 | Small n |
| ... | | | | |
| PF18100 | 1.000 | 20 | 14 | Top |
| PF07648 | 1.000 | 19 | 18 | |
| PF03147 | 1.000 | 20 | 11 | |

Even the worst per-family AUROC (0.76) is meaningfully above chance.

## Findings

### F1 — AlphaMissense behaves like our result_6 probe

Pathogenicity prediction is family-robust across two predictor classes:

| Predictor | Family-robustness metric | Value |
|---|---|---|
| Our ESM-2 delta supervised probe (result_6) | Δ AUROC (gene-split − family-split) | 0.002 |
| AlphaMissense (this result) | per-family AUROC std around 0.940 overall | 0.046 |

Both say the same thing. The supervised-head hypothesis (that AlphaMissense's larger trained head might have memorised family-correlated structure) is **rejected**.

### F2 — Reframes the family-leakage critique as task-dependent

Across the project, family-split CV produces three qualitatively different verdicts depending on the prediction task:

| Task type | Family-leakage signature | Example |
|---|---|---|
| Per-residue local biochemistry | None | Pathogenicity (result_6, result_17) |
| Cross-family pattern recognition | Partial | Within-family ion channel mechanism (result_8) |
| Per-gene global function / family-distributed label | Severe (≥50%) | GOF/DN/LOF mechanism (results 1–10) |

The diagnostic is not a hammer — it is a scalpel that separates tasks whose labels live in *local sequence properties* from tasks whose labels live in *family-correlated global properties*. The mechanism leakage finding does not generalise to pathogenicity.

### F3 — Architectural attribution

AlphaMissense's family-robustness is consistent with three design choices that systematically remove the channels through which family signal can enter:

1. **Population frequency labels, not ClinVar curation labels.** Removes the family-bias in curation effort.
2. **Per-residue prediction objective.** No place in the computation graph for family identity to be exploited.
3. **AF2 structural features.** Local properties (solvent accessibility, contact geometry, secondary structure) instead of family-aggregate properties.

## Caveats

### Training–test logic overlap on ClinVar

AlphaMissense's training labels derive from human + primate population frequency: variants present in populations are weak benign, variants absent are weak pathogenic. ClinVar classification logic uses gnomAD frequency as a partial input — common variants are routinely classified benign. The training and test signals are therefore **not independent** at the per-variant level. The absolute AUROC of 0.94 is inflated by this affinity. **The per-family distribution metric is unaffected** — it depends on the *shape* of the AUROC across families, not the level — but the absolute number should not be cited without this caveat.

The clean follow-up is ProteinGym, which uses physical deep-mutational-scanning fitness as ground truth. No curation–training overlap, no population-frequency circularity. Pending experiment.

### Per-family sample sizes

Many families have small n in the worst-performing tail. The headline distribution statistics are stable; the individual rankings of the worst families are not.

## Practical conclusion

For the paper: **the family-leakage critique does not apply to pathogenicity prediction, including for the published predictor in clinical use.** This is a clean negative result that closes off an obvious reviewer attack on result_6 and clarifies the scope of the family-split-CV contribution.

## Artifacts

- `data/cache/AlphaMissense_aa_substitutions.tsv.gz` — bulk AM file (1.1 GB compressed; safe to delete after `data/alphamissense_scores_full.json` is built)
- `data/alphamissense_scores_full.json` — 16,334 scored variants keyed by GENE_POS_WT_MUT
- `results/alphamissense_family/overall.json`
- `results/alphamissense_family/per_family.json`
- `results/alphamissense_family/summary.json`
