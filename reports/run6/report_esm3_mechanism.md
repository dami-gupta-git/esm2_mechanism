# Results: ESM-3 Scale and Structure on Mechanism

*Companion to [`report_classifier.md`](report_classifier.md) and
[`report_protein_family.md`](report_protein_family.md). The classifier report found that
ESM-2 delta embeddings classify mechanism (GOF/DN/LOF) near the chance floor under family-split
CV. This report asks whether a larger, structure-aware model closes that gap. It runs ESM-3 on
the same merged variant set as the classifier report, under two conditions — sequence only, and
sequence plus AlphaFold2 structure tokens — to separate the effect of model scale from the
effect of explicit structure.*

**Run 6 · 2026-06-01** · ESM-3 `esm3-sm-open-v1` (1.4B, open weights) · 17,826 variants ·
1,935 genes · 5 seeds · phase 2 on an H100 80GB, phase 3 on CPU. Results in
[`results/run6/esm3_mechanism/merged/summary.json`](../../results/run6/esm3_mechanism/merged/summary.json).

---

## Summary

ESM-3 was run on the same merged variant set the ESM-2 classifier report uses, in two forms:
sequence tokens alone, and sequence tokens together with AlphaFold2 structure tokens. Both
conditions clear the matched ESM-2 floor — family-split macro-F1 of 0.438 for sequence-only and
0.453 for sequence-plus-structure, against an ESM-2 floor of 0.380 — so scaling the sequence
model does lift the mechanism floor. Structure tokens add a little on top: 0.453 versus 0.438 is
a gain of 0.014, real but small, and below the pre-registered 0.030 bar for calling structure a
distinct ingredient. So the picture is scale, not structure: the bulk of the improvement over
ESM-2 comes from the larger sequence model, and explicit AlphaFold2 structure contributes a
marginal amount that does not change the conclusion. Mechanism remains far from solved — 0.45
macro-F1 on three classes is well below what practical prediction would need. Function tokens,
the third ESM-3 modality, are not exposed by the open API and were not tested.

This is the matched comparison. The earlier Gerasimavicius-only run
([`report_esm3_mechanism_geras.md`](report_esm3_mechanism_geras.md)) is superseded: it compared
against a different dataset and contained a data defect; this run fixes both.

---

## What is measured, and why

The classifier report found that ESM-2 delta embeddings carry little mechanism signal once whole
protein families are held out. Two intuitions suggest ESM-3 might do better. It is larger (1.4B
vs 650M), so its sequence representations may capture more of whatever weakly correlates with
mechanism. And it can take explicit structure as input, so if mechanism is legible in the 3D
fold rather than the sequence, structure tokens should help where sequence alone does not.

This report separates those two effects by running two conditions through the same probes and
cross-validation, and judging them against an ESM-2 baseline measured on the same variants.

**Conditions:**

| Condition | What it is |
|---|---|
| `seq` | ESM-3 with sequence tokens only — the scale comparison to ESM-2 650M |
| `seq_struct` | ESM-3 with sequence tokens plus AlphaFold2 structure tokens — tests whether explicit structure adds anything |

For each condition the representation is the same mutant-minus-wildtype shift used throughout
the ESM-2 work: `delta = mean_pool(ESM-3(mut)) − mean_pool(ESM-3(wt))`, dimension 1536.
Structure tokens, where used, come from AlphaFold2 coordinates fetched from the EMBL-EBI API and
encoded by ESM-3's own structure tokenizer; they are applied to both the wildtype and mutant
forward passes so the delta cancels everything except the substitution.

**Splits and probes:**

| Term | Meaning | "No signal" reference |
|---|---|---|
| Gene-split | 5-fold CV holding out whole genes; related genes may sit in train and test | — |
| Family-split | 5-fold CV holding out whole Pfam families; the leakage-free measure | ESM-2 delta floor, 0.380 |
| Macro_f1 | mean per-class F1 over GOF/DN/LOF, so rare classes count equally | three-class chance, well below 0.33 |
| GOF / DN / LOF AUROC | one-vs-rest ranking for each mechanism class against the other two | 0.5 |

The probe is a PyTorch MLP (256→64, dropout 0.3, class-weighted cross-entropy, early stopping)
matching the ESM-2 classifier report exactly, with a balanced logistic regression as a
secondary check. The variant set, the GOF/DN/LOF label collapse (HI and AR folded into LOF), the
WT-reference filter, the fold construction, and the seeds (0–4) all match that report, so the
only changed variable is the embedding model. The merged ESM-3 row set is identical to the
ESM-2 classifier's — all 17,826 variants embedded, none dropped — making this a like-for-like
comparison.

**The ESM-2 floor.** The baseline is read at runtime from the matched ESM-2 probe: the 5-seed
mean of MLP delta_mean family-split macro-F1 from `nonlinear_results_seed*.json`, which is
**0.380**. The pass threshold is that floor plus a 0.05 margin, i.e. **0.430**.

**Decision rules** (pre-registered in `docs/plans/plan_esm3_mechanism.md`):

| Gate | Criterion | Reads as |
|---|---|---|
| M1 | `seq_struct` family-split F1 > 0.430 (floor + 0.05) | does ESM-3-with-structure beat ESM-2? |
| M2 | `seq` family-split F1 > 0.430 | does scale alone beat ESM-2? |
| M3 | `seq_struct` − `seq` > 0.030 | does structure add signal beyond scale? |

---

## Table 1 — Mechanism macro-F1 (MLP, 5-seed mean ± std)

| Condition | Gene-split | Family-split |
|---|---|---|
| ESM-2 650M delta_mean (classifier report) | — | 0.380 |
| ESM-3 seq | 0.445 ± 0.023 | 0.438 ± 0.009 |
| ESM-3 seq_struct | 0.448 ± 0.015 | 0.453 ± 0.012 |

## Table 2 — Per-class AUROC and logistic regression (family-split, 5-seed mean)

| Condition | GOF AUROC | DN AUROC | LOF AUROC | LR macro-F1 |
|---|---|---|---|---|
| ESM-3 seq | 0.689 | 0.647 | 0.693 | 0.429 ± 0.003 |
| ESM-3 seq_struct | 0.699 | 0.628 | 0.705 | 0.439 ± 0.005 |

## Table 3 — Decision rules

| Gate | Criterion | Value | Verdict |
|---|---|---|---|
| M1 | `seq_struct` family-split F1 > 0.430 | 0.453 | pass |
| M2 | `seq` family-split F1 > 0.430 | 0.438 | pass |
| M3 | `seq_struct` − `seq` > 0.030 | 0.014 | fail |

---

## Reading the tables

**1. Scale lifts the family-split floor.**
ESM-3 sequence-only reaches a family-split macro-F1 of 0.438, above the matched ESM-2 delta
floor of 0.380 and clear of the 0.430 threshold. Because the variants, labels, splits, probe,
and seeds are all held fixed, the only thing that changed is the embedding model, so this lift is
attributable to scale. M2 passes: a larger sequence model does carry more of whatever weakly
tracks mechanism. The margin is thin, though: 0.438 clears the 0.430 threshold by 0.008 — about
one seed of spread — and the lift over ESM-2 is a modest 0.058, so the difference is reported here
as consistent in direction but not yet tested for significance. A paired cluster bootstrap over
genes on the shared variant set is the planned test (see statistical limitations).

**2. Structure tokens add a little, but not enough to count.**
Sequence-plus-structure reaches 0.453, edging out sequence-only by 0.014. M1 passes (0.453 is
above 0.430), so ESM-3-with-structure also beats ESM-2 — but M3 fails: the seq_struct − seq gap
of 0.014 is below the 0.030 bar pre-registered for calling structure a distinct ingredient. The
gain is real and consistent (seq_struct is higher than seq on family-split MLP, GOF AUROC, LOF
AUROC, and LR), but small, and most of the score is already there from sequence alone. The script's verdict
is "scale suffices."

**3. The lift is not leakage.**
For sequence-only the gene-split and family-split scores are almost identical (0.445 vs 0.438),
so holding out whole families costs almost nothing. The improvement over ESM-2 holds up on the
leakage-free split rather than evaporating when families are removed, so it reflects
family-transferable signal, not gene-identity leakage.

**4. The number is up, but not useful.**
0.45 macro-F1 on three classes is above the chance floor but well below what mechanism
prediction would need to be relied on. The per-class AUROCs sit close together — LOF near 0.70,
GOF near 0.69, DN near 0.63–0.65 — so no single class is cleanly resolved and the separability is
spread across all three rather than a clean three-way split. Scale moved the floor; it did not
solve the task.

---

## What this is and is not

- **Not a test of function tokens.** ESM-3's third modality is not exposed by the open-weights
  API and was dropped. The conclusion is limited to sequence and sequence-plus-structure.
- **Not a claim that structure is irrelevant to mechanism in general** — only that ESM-3's
  AlphaFold2 structure tokens, added to its sequence tokens, do not add enough to this
  delta-based probe to clear the pre-registered bar. This echoes the family report's finding
  that the family-transferable signal these models carry is conservation-like rather than
  structural, so structure that is itself conservation-correlated supplies little that is new.
- Structure tokens were applied to 94.5% of variants (16,852 of 17,826); the remaining 5.5% fell
  back to sequence-only because no AlphaFold2 entry was available or the predicted structure's
  length did not match the sequence window (400 coord-length fallbacks). This slightly dilutes
  the `seq_struct` condition but cannot reverse the direction of M3.

---

## Statistical limitations and planned analyses (pre-preprint)

The seed-std bars reflect fold reshuffling on a fixed set of genes, not sampling uncertainty, and
understate the true error because every seed reuses all the data. The headline is a 0.058
family-split lift over ESM-2 (0.438 vs 0.380), and M2 clears its 0.430 threshold by only 0.008 —
about one seed of spread. Planned before preprint submission, not yet in the result files:

- **Paired difference test** for the scale lift: a paired cluster bootstrap over genes on the
  shared variant set for the `seq` − ESM-2 delta gap (and `seq_struct` − `seq`), with a 95% CI on
  the difference, so "ESM-3 beats ESM-2" and "structure adds nothing" rest on tested gaps rather
  than separated error bars and a thin threshold margin.
- **Confidence intervals** from a cluster bootstrap over genes (labels are gene-level, so the
  effective N is ≈ 1,935 genes, not 17,826 variants — and far smaller for the rare classes,
  DN ≈ 9% and GOF ≈ 15%), replacing the seed-std bars.
- **Permutation test** against the 0.288 majority-class floor for a p-value on "above chance."
- **Calibration:** the probes are uncalibrated; scores are discrimination only, not risks.

---

## Provenance

Computed by `experiments/esm3/esm3_mechanism.py` (phases 1–3, `--dataset merged`) on all 17,826
merged variants (Gerasimavicius + G2P), across 1,935 genes; none were dropped (the mutant is
built through the shared `apply_missense` helper, so the row set matches the ESM-2 classifier's
exactly). AlphaFold2 structures were fetched from the EMBL-EBI API and encoded with ESM-3's
structure tokenizer. Embeddings are mutant-minus-wildtype mean-pooled deltas, dimension 1536, in
[`data/embeddings/esm3-sm-open-v1/merged/`](../../data/embeddings/esm3-sm-open-v1/merged/)
(`seq_mean.npy`, `seq_struct_mean.npy`, plus raw `_wt`/`_mut` arrays, aligned by `valid_idx.npy`).
Probes: PyTorch MLP (256→64, dropout 0.3, class-weighted CE, early stopping) and balanced
logistic regression (C=0.1), under 5-fold gene-split and family-split CV, seeds 0–4. The ESM-2
floor (0.380) is the 5-seed mean of `mlp_delta_mean_family` read from
`results/run6/nonlinear_results_seed*.json`. Coverage stats in `struct_meta.json`; full results
and decision rules in
[`results/run6/esm3_mechanism/merged/summary.json`](../../results/run6/esm3_mechanism/merged/summary.json).
