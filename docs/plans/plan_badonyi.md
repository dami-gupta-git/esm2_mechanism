# Plan — Badonyi survives strict holdout?

**Date drafted:** 2026-05-26
**Status:** Pre-registration
**Builds on:** result_15 (Badonyi modality), result_7 (leakage quantification)

---

## Question

Does Badonyi & Marsh 2024's published gene-level mechanism predictor (PLOS One SVM, pDN / pGOF / pLOF) maintain its reported AUROC under the project's strict holdout protocols (Pfam family-split, MMseqs2-20 cluster-split)?

Distinct from result_15's V_bad, which trained a *new* LogReg on top of Badonyi's outputs per fold. Here we use Badonyi's predictions directly with no retraining — the published model is the model on test.

---

## Why

- Badonyi's reported numbers (DN-vs-LOF AUROC 0.71, GOF-vs-LOF 0.763, LOF-vs-non-LOF 0.763) are under 3×10-fold gene-split CV. They do not report family-split numbers.
- Result_7 showed that gene-split inflates above-chance signal by ~62% over family-split on the same disease-gene domain.
- If Badonyi's signal is partly family-recognition leakage, the published numbers will not survive family-split. If it is genuine cross-family mechanism signal, the drop will be small.
- This is the single comparison that calibrates "Badonyi's features beat ESM-2" against "Badonyi's features beat ESM-2 *under matched strict evaluation*."

---

## What we already have

- `data/badonyi_features_aligned.npy` — Badonyi's pDN / pGOF / pLOF for all 2,424 genes
- `data/cache/badonyi/table_S3.xlsx` — including the `train_dn_gof_lof` flag per gene
- `data/merged_gene_list.tsv` — 3-class mechanism labels for 1,699 genes
- `data/pfam_families.json` — Pfam family per gene
- `data/mmseqs_clusters.json` — MMseqs2-20 cluster per gene

No new sequences, no retraining, no GPU. ~1 hour of work.

---

## Procedure

For each holdout protocol H ∈ {none (whole-set baseline), Pfam family-split, MMseqs2-20 cluster-split}:

1. Take labeled genes (n=1,699). Split into 5 folds per H, holding out entire groups (families / clusters).
2. For each held-out gene, take Badonyi's *unmodified* pDN, pGOF, pLOF as the prediction.
3. Aggregate held-out predictions across folds (one prediction per gene, all out-of-fold).
4. Compute three binary AUROCs matching Badonyi's setup:
 - **DN-vs-LOF**: among DN+LOF genes, AUROC(pDN, true=DN)
 - **GOF-vs-LOF**: among GOF+LOF genes, AUROC(pGOF, true=GOF)
 - **LOF-vs-non-LOF**: among all labeled genes, AUROC(pLOF, true=LOF)
5. Repeat for 5 seeds (different fold assignments). Report mean ± std.

Additionally — stratified by Badonyi's training-set membership (from result_15's leakage analysis):

6. Recompute under each H restricted to IN-Badonyi-train and OUT-Badonyi-train subsets.

---

## Outputs

- `scripts/badonyi_holdout_survival.py` — analysis script
- `results/badonyi_survival/badonyi_survival_{none,family,mmseqs}_seed{0..4}.json`
- `results/badonyi_survival/badonyi_survival_summary.json`

---

## Pre-registered outcomes

For each holdout H, compute ΔAUROC = AUROC(H) − AUROC(no holdout). Decision per class:

| Outcome | Threshold | Interpretation |
|---|---|---|
| **Robust** | ΔAUROC ≥ −0.03 | Badonyi's published signal is real cross-family mechanism information. Their feature engineering is solid. |
| **Partial leakage** | −0.10 < ΔAUROC < −0.03 | Some family-recognition contamination in their reported numbers, but most of the signal is genuine. Their numbers should be quoted with this caveat. |
| **Mostly leakage** | ΔAUROC ≤ −0.10 | Badonyi's published numbers are inflated by family recognition. Falsifies the "Badonyi's features beat ESM-2" framing as currently stated. Strengthens this project's methodology contribution materially. |

Decision rule is per-class — DN, GOF, and LOF may behave differently.

---

## Predicted outcomes

Based on result_7's 62% leakage fraction estimate:
- DN-vs-LOF AUROC: 0.71 → ~0.62–0.65 under family-split (partial-leakage band)
- GOF-vs-LOF AUROC: 0.76 → ~0.65–0.70 (partial-leakage band)
- LOF-vs-non-LOF AUROC: 0.76 → ~0.65–0.70 (partial-leakage band)

If predictions land here, the project gets a quotable "Badonyi's reported numbers drop by ~0.08 under proper holdout" sentence. If they survive intact, the comparison to ESM-2 becomes harder for the project.

---

## What this does NOT do

- Does not retrain Badonyi's SVM. We are testing the *published* predictions, not redoing their experiment.
- Does not address whether re-training their SVM under family-split would give different numbers. That requires their full feature set and is out of scope.
- Does not change result_15's V_bad / V2+bad numbers. Those use Badonyi's outputs as features in a re-fit LogReg; this is a different (stricter) evaluation question.
