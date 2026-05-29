# Result 11 — Stage 0 pilot: 4 gene-level features predict mechanism under family-split CV
## Date: May 25, 2026 | Models: Logistic regression + tiny MLP | Local CPU | Seeds: 0–4 (5-seed replication)

---

## Background: a different kind of feature

All prior results used ESM-2 embeddings — features derived from the protein sequence. This experiment tries something different: **gene-level biological properties** from public databases. Instead of asking "what does this mutation do to the ESM-2 representation?", we ask "what kind of gene is this, based on population genetics?"

The four features tested:
1. **pLI** — from gnomAD v4.1. Probability that a gene is intolerant to loss-of-function mutations. High pLI means humans tolerate almost no LOF variants in this gene — it's essential.
2. **LOEUF** — from gnomAD v4.1. Upper bound of the observed/expected ratio for LOF variants. Low LOEUF also indicates a constrained, essential gene.
3. **mis_z** — from gnomAD v4.1. Z-score for observed vs expected missense mutations. High mis_z means the gene is constrained even for missense changes.
4. **paralog_count** — number of close relatives this gene has in the human genome (from Ensembl).

The intuition: a gene that humans cannot tolerate losing even one copy of (high pLI) is likely to cause disease by haploinsufficiency (LOF). A gene with many paralogs may be less sensitive to losing one copy, so if it's in a disease dataset, it may more often be there for activating (GOF) or dominant-negative reasons.

This is a **sanity check** — a Stage 0 gate. Its only job is to confirm that this kind of gene-level feature carries mechanism information under our standard family-split evaluation, so that the larger follow-up experiment is worth doing.

---

## TL;DR

A 4-feature logistic regression achieves **macro-F1 = 0.4171 ± 0.0091** under 5-fold family-split CV on 1,234 genes (725 protein families) across 5 seeds, **+0.122 above the majority baseline (0.2954)**. Per-class AUROCs:

- GOF: **0.686 ± 0.011**
- **DN: 0.687 ± 0.009**
- LOF: **0.735 ± 0.001**

All 5/5 seeds returned STRONG_SIGNAL under the pre-registered decision rule (best model macro-F1 ≥ 0.40). Outcome: proceed to Phase 1 full data pull.

---

## Setup

**Features (4):**
1. **pLI** — probability of intolerance to loss-of-function variation (gnomAD v4.1)
2. **LOEUF** — upper bound of observed/expected LoF confidence interval (gnomAD v4.1)
3. **mis_z** — Z-score for observed-vs-expected missense burden (gnomAD v4.1)
4. **paralog_count** — number of human paralogues per Ensembl Compara REST

**Labels:** the merged-dataset gene list (`data/merged_gene_list.tsv`), mechanism column collapsed to 3 classes (GOF → GOF, DN → DN, HI → LOF, LOF → LOF; AR dropped).

**Models:**
- Majority baseline (predicts training-fold majority class)
- Logistic regression (L2-regularised, `class_weight="balanced"`)
- Tiny MLP (16, 8) with early stopping, no class weighting

**Evaluation:** 5-fold family-split CV; 5 seeds (0–4); StandardScaler fit on training fold only.

**Coverage and class distribution:**
- 1,234 genes had both a 3-class label and a protein family assignment.
- gnomAD constraint coverage ~98%.
- 725 unique protein families.
- Class distribution: GOF=145 (11.7%), DN=107 (8.7%), LOF=982 (79.6%) — heavily LOF-skewed. macro-F1 is the right metric; raw accuracy of "always predict LOF" would be ~80% but macro-F1 punishes that.

---

## Results

### 5-seed aggregate

| Model | Macro-F1 (mean ± std) | GOF AUROC | DN AUROC | LOF AUROC |
|---|---|---|---|---|
| Majority baseline | 0.2954 ± 0.0000 | — | — | — |
| **Logistic regression** | **0.4171 ± 0.0091** | **0.686 ± 0.011** | **0.687 ± 0.009** | **0.735 ± 0.001** |
| Tiny MLP (16, 8) | 0.3088 ± 0.0151 | 0.577 ± 0.111 | 0.561 ± 0.091 | 0.593 ± 0.123 |

### Per-seed logistic regression stability

| Seed | Macro-F1 | DN AUROC |
|---|---|---|
| 0 | 0.4275 | 0.6970 |
| 1 | 0.4055 | 0.6779 |
| 2 | 0.4140 | 0.6880 |
| 3 | 0.4254 | 0.6959 |
| 4 | 0.4131 | 0.6777 |
| **Range** | **0.022** | **0.019** |

### Read

- **Logistic regression** is +0.122 above majority on average. All three per-class AUROCs are clearly above chance and roughly balanced. Per-seed range on macro-F1 (0.022) and DN AUROC (0.019) is small compared to the gaps above baseline. Numbers are stable.
- **Tiny MLP** macro-F1 sits close to majority. Mean AUROCs (0.56–0.59) show partial learning, but per-seed std (0.09–0.12) is large. With 4 features, heavy class imbalance (80% LOF), and no class weighting, the MLP often collapses to majority-class prediction. Logistic regression is the right model at this dimensionality.

**Pre-registered decision:** STRONG_SIGNAL (best model ≥ 0.40 on every seed). All 5/5 seeds agree. Proceed to Phase 1.

---

## What it means (for the pilot only)

1. **The pipeline works.** Label join, family-split CV, scaling, evaluation, and the decision rule all execute end-to-end and produce reproducible numbers.
2. **Four public per-gene features carry meaningful mechanism signal** under family-split CV. The signal is robust to seed choice.
3. **Paralog count appears to be the differentiating feature for DN.** A constraint-only run (no paralog data) gave noticeably lower DN AUROC (~0.59 in an earlier run); the fully-populated 4-feature run gives DN AUROC 0.687. This is suggestive but not formally tested — a proper feature ablation belongs in Phase 1 with the full feature set.
4. **MLP regime.** With 4 features, logistic regression dominates. The MLP will need explicit class weighting and either more features or stronger regularisation to be worth using in Phase 1.

---

## Decision and next steps

**Decision:** STRONG_SIGNAL on 5/5 seeds → proceed to Phase 1 of Experiment 11.

**Next steps:**

1. Phase 1 data pull: HPA tissue specificity, PaxDb abundance, BioPlex interactome, Mathieson 2018 half-life, PhosphoSitePlus PTM density, ClinGen dosage sensitivity.
2. Coverage-skew sanity check before running V2: confirm class balance in fully-covered genes is not badly skewed vs the overall labeled set.
3. Phase 2 feature engineering, including the family-mean-centred residuals (mandatory per the updated plan).
4. Stage 1 (V2): all ~30 features, 5 seeds, family-split CV.

---

## Files

- `scripts/proteome_pilot.py`
- `data/cache/proteome_pilot/gnomad_v4.1_constraint.tsv` (cached download)
- `data/cache/proteome_pilot/paralogs/` (per-gene Ensembl REST cache)
- `data/gene_features_pilot.tsv` (per-gene feature table)
- `results/proteome_pilot/pilot_results_seed{0..4}.json` (per-seed metrics)
- `results/proteome_pilot/pilot_results_summary_5seed.json` (aggregated mean ± std)

---

## Plain-English summary

We took 1,234 disease genes with known mechanism labels (does the gene fail by gaining a new function, by being knocked down, or by poisoning its own complex?) and trained a simple logistic regression on just four numbers per gene:

- Three measures of how strongly the human population is selected against losing or mutating the gene (gnomAD constraint).
- One count of how many close relatives the gene has in the human genome (paralogs).

Under cross-validation that holds out entire protein families during training, the model achieved balanced predictions across all three mechanism classes (AUROCs around 0.69–0.74). We repeated the whole experiment with 5 different random seeds; every run agreed within a tiny margin. By the pre-registered decision rule, this is a strong-enough signal to justify the larger experiment.

This was a sanity check, not the main experiment. Its only job was to confirm that gene-level features carry mechanism information under our standard evaluation setup, so that the bigger Phase 1 data pull is worth doing.
