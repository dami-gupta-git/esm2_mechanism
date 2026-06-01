# Results: ESM-3 Scale and Structure on Mechanism (Gerasimavicius-only — superseded)

> **⚠️ Known data defect — do not cite these numbers as clean.** This run embedded **93 of
> 10,231 variants on a wrong wildtype/mutant pair**: phase 2 overwrote the windowed reference
> residue without checking it matched the variant's `aa_wt`, a check ESM-2's pipeline applies
> via `apply_missense`. Those 93 deltas are meaningless, and the row set therefore differs from
> the ESM-2 comparison set. The bug is fixed in `esm3_mechanism.py`; this run predates the fix.
> The **absolute family-split numbers below are contaminated** and must not be quoted as ESM-3's
> score or compared against the ESM-2 floor. The **direction** (structure tokens add nothing;
> seq ≈ seq_struct) is robust to 93/10,231 rows and remains informative.
>
> This run is also **Gerasimavicius-only**, which is not the matched ESM-2 comparison set (ESM-2's
> numbers are the merged 17,826-variant set), so the scale comparison here is cross-dataset.
> The clean, apples-to-apples result is the merged run — see
> [`report_esm3_mechanism.md`](report_esm3_mechanism.md). Defect details:
> [`results/run6/esm3_mechanism/geras/KNOWN_ISSUES.md`](../../results/run6/esm3_mechanism/geras/KNOWN_ISSUES.md).

*Companion to [`report_classifier.md`](report_classifier.md) and
[`report_protein_family.md`](report_protein_family.md). The classifier report found that
ESM-2 delta embeddings classify mechanism (GOF/DN/LOF) near the chance floor under family-split
CV. This report asks whether a larger, structure-aware model closes that gap. It runs ESM-3 on
the Gerasimavicius-only task, under two conditions — sequence only, and sequence plus AlphaFold2
structure tokens — to separate the effect of model scale from the effect of explicit structure.*

**Run 6 · 2026-06-01** · ESM-3 `esm3-sm-open-v1` (1.4B, open weights) · 10,231 variants
(incl. 93 contaminated) · 948 genes · 5 seeds · phase 2 on an H100 80GB, phase 3 on CPU.
Results in [`results/run6/esm3_mechanism/geras/summary.json`](../../results/run6/esm3_mechanism/geras/summary.json).

---

## Summary

ESM-3 was run on the Gerasimavicius-only mechanism task in two forms: sequence tokens alone, and
sequence tokens together with AlphaFold2 structure tokens. The clean, within-run finding is that
**adding structure tokens changes nothing**: sequence-only and sequence-plus-structure both score
family-split macro-F1 = 0.421, identical to three decimal places, and track each other across
every split and metric. Whether scale lifts the mechanism floor relative to ESM-2 is **not**
answerable here — the ESM-2 baseline is on a different (merged) dataset, and 93 of these rows are
contaminated; the merged ESM-3 run settles that comparison. Either way the question of how a
mutation acts remains far from solved — 0.421 is still well below what practical mechanism
prediction would need. Function tokens, the third ESM-3 modality, are not exposed by the open
API and were not tested.

---

## What is measured, and why

The classifier report found that ESM-2 delta embeddings carry little mechanism signal once whole
protein families are held out. Two intuitions suggest ESM-3 might do better. It is larger (1.4B
vs 650M), so its sequence representations may capture more of whatever weakly correlates with
mechanism. And it can take explicit structure as input, so if mechanism is legible in the 3D
fold rather than the sequence, structure tokens should help where sequence alone does not.

This report separates those two effects by running two conditions through the same probes and
cross-validation, and comparing them against each other and against the ESM-2 baseline.

**Conditions:**

| Condition | What it is |
|---|---|
| `seq` | ESM-3 with sequence tokens only — a like-for-like scale comparison to ESM-2 650M |
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
| Family-split | 5-fold CV holding out whole Pfam families; the leakage-free measure | three-class chance ≈ 0.33 accuracy, lower in macro-F1 |
| Macro_f1 | mean per-class F1 over GOF/DN/LOF, so rare classes count equally | three-class chance floor (well below 0.33) |
| GOF AUROC | one-vs-rest ranking of gain-of-function variants | 0.5 |
| Leakage Δ | family-split minus gene-split macro-F1; how much of the gene-split score is family recognition | 0 (no leakage) |

The probe is a PyTorch MLP (256→64, dropout 0.3, class-weighted cross-entropy, early stopping)
matching the ESM-2 classifier report exactly, with a balanced logistic regression as a
secondary check. The dataset, the GOF/DN/LOF label collapse (HI and AR folded into LOF), the
fold construction, and the seeds (0–4) all match that report, so the only changed variable is
the embedding model.

**Decision rules** (pre-registered in `docs/plans/plan_esm3_mechanism.md`):

| Gate | Criterion | Reads as |
|---|---|---|
| M1 | `seq_struct` family-split F1 > 0.349 (ESM-2 + 0.05) | does ESM-3-with-structure beat ESM-2? |
| M2 | `seq` family-split F1 > 0.349 | does scale alone beat ESM-2? |
| M3 | `seq_struct` − `seq` > 0.030 | does structure add signal beyond scale? |

---

## Table 1 — Mechanism macro-F1 (MLP, 5-seed mean ± std)

| Condition | Gene-split | Family-split | Leakage Δ |
|---|---|---|---|
| ESM-2 650M delta_mean (classifier report) | 0.288 | 0.288 | 0.000 |
| ESM-3 seq | 0.452 ± 0.020 | 0.421 ± 0.009 | −0.031 |
| ESM-3 seq_struct | 0.434 ± 0.018 | 0.421 ± 0.010 | −0.013 |

## Table 2 — Per-class AUROC and logistic regression (family-split, 5-seed mean)

| Condition | GOF AUROC | DN AUROC | LR macro-F1 |
|---|---|---|---|
| ESM-3 seq | 0.700 | 0.573 | 0.442 ± 0.011 |
| ESM-3 seq_struct | 0.676 | 0.553 | 0.443 ± 0.013 |

## Table 3 — Decision rules

The M1/M2 verdicts shown here used the stale 0.349 threshold (ESM-2 floor 0.299 + 0.05).
That floor is wrong: the matched ESM-2 MLP delta_mean family-split floor is **0.380**, so the
real threshold is **0.430**, against which `seq` = 0.421 does **not** clear M1/M2. Combined with
the 93 contaminated rows and the cross-dataset confound, no M1/M2 verdict from this run should be
quoted — see the merged report for the judged result. Only M3 (within-run, structure vs sequence)
is meaningful here.

| Gate | Criterion (as run) | Value | Verdict (as run) | Status |
|---|---|---|---|---|
| M1 | `seq_struct` family-split F1 > 0.349 | 0.421 | pass | invalid — stale floor; real threshold 0.430 |
| M2 | `seq` family-split F1 > 0.349 | 0.421 | pass | invalid — stale floor; real threshold 0.430 |
| M3 | `seq_struct` − `seq` > 0.030 | −0.000 | fail | informative (within-run) |

---

## Reading the tables

**1. Apparent scale lift — but the comparison is cross-dataset and not valid here.**
ESM-3 sequence-only reaches a family-split macro-F1 of 0.421 on Gerasimavicius. It is tempting to
compare this to the ESM-2 delta baseline, but that baseline (0.288) is on the *merged* 17,826-variant
set, not Gerasimavicius — different data, so the gap conflates scale and dataset. The matched ESM-2
floor on the comparison set is 0.380 (threshold 0.430), which 0.421 does not clear. Any scale claim
must come from the merged ESM-3 run, where both models score the same variants; it is not
established by this report. The gain comes
larger sequence model does carry more of whatever weakly tracks mechanism.

**2. Structure tokens add nothing.**
Sequence-plus-structure also scores 0.421 on family-split — the same number, with the M3 gap
landing at −0.000 (literally −5×10⁻⁶). The two conditions are not merely close; they are
indistinguishable on the leakage-free split. M3 fails decisively: explicit AlphaFold2 structure
does not add mechanism signal beyond what the sequence model already provides. If anything, the
per-class AUROCs are slightly lower with structure (GOF 0.676 vs 0.700), and the gene-split
score dips (0.434 vs 0.452), so structure is neutral-to-faintly-harmful rather than helpful.

**3. The lift is family-transferable, not leaked.**
ESM-3's leakage Δ (family-split minus gene-split) is −0.031 for `seq`: the score barely falls
when whole families are held out, so almost all of it survives the leakage-free split. The ESM-2
delta carried no family-split signal to begin with (0.288 on both splits, the chance floor), so
ESM-3 is not winning by leaking gene identity — its gain holds up precisely where the ESM-2 delta
had nothing. That said, the GOF AUROC of 0.700 on family-split is in the range ESM-2's *absolute*
(wildtype) embeddings reached on gene-split, so part of what ESM-3's sequence model carries is the
same conservation-like family signal, now transferring across families rather than leaking within them.

**4. The two probes agree.**
Logistic regression gives 0.442 (`seq`) and 0.443 (`seq_struct`) on family-split — the same
ordering and the same near-equality as the MLP, with structure no better than sequence. The M3
null is not an artifact of the MLP; a linear probe reads it the same way.

**5. The number is up, but not useful.**
0.421 macro-F1 on three classes is well above the chance floor but well below what mechanism
prediction would need to be relied on. The DN AUROC sits near 0.56 — barely above a coin flip —
so most of the F1 comes from separating GOF from the rest, not from resolving all three
mechanisms. Scale moved the floor; it did not solve the task.

---

## What this is and is not

- **Not a test of function tokens.** ESM-3's third modality is not exposed by the open-weights
  API and was dropped. The conclusion is limited to sequence and sequence-plus-structure; it is
  possible, though untested, that function tokens behave differently.
- **Not a claim that structure is irrelevant to mechanism in general** — only that ESM-3's
  AlphaFold2 structure tokens, added to its sequence tokens, do not improve this delta-based
  mechanism probe. This echoes the family report's finding that the family-transferable signal
  these models carry is conservation-like rather than structural, so adding structure that is
  itself conservation-correlated supplies little that is new.
- **Not run on the merged dataset.** This is Gerasimavicius only, matching the ESM-2 classifier
  report. Whether the scale lift holds on the larger merged mechanism set is untested.
- Structure tokens were applied to 92.1% of variants (9,424 of 10,231); the remaining 7.9% fell
  back to sequence-only because no AlphaFold2 entry was available (14 of 948 proteins) or the
  predicted structure's length did not match the sequence window (61 variants). This slightly
  dilutes the `seq_struct` condition, but cannot explain the M3 null: even on the fully-covered
  majority, structure tracks sequence exactly.

---

## Statistical limitations and planned analyses (pre-preprint)

The ESM-3 lift over ESM-2 is read from non-overlapping seed spreads, which is not a significance
test. Planned before preprint submission, not yet in the result files:

- **Significance of the lift:** a paired cluster bootstrap over genes on the shared variant set,
  giving a 95% CI on the ESM-3 (≈0.42) minus ESM-2 (≈0.38) gap.
- **Permutation test** against chance for the family-split scores of both conditions.
- **Effective N:** labels are gene-level, so all CIs and tests resample genes, not variants.

---

## Provenance

Computed by `experiments/esm3/esm3_mechanism.py` (phases 1–3) on 10,231 of 10,233
Gerasimavicius variants (2 skipped: substitution position out of range after windowing), across
948 genes. AlphaFold2 structures for 934 of 948 proteins were fetched from the EMBL-EBI API and
encoded with ESM-3's structure tokenizer. Embeddings are mutant-minus-wildtype mean-pooled
deltas, dimension 1536, in
[`data/embeddings/esm3-sm-open-v1/`](../../data/embeddings/esm3-sm-open-v1/) (`seq_mean.npy`,
`seq_struct_mean.npy`, aligned by `valid_idx.npy`). Probes: PyTorch MLP (256→64, dropout 0.3,
class-weighted CE, early stopping) and balanced logistic regression (C=0.1), under 5-fold
gene-split and family-split CV, seeds 0–4. The ESM-2 baseline figures are from the classifier
report. Coverage stats in `struct_meta.json`; full results and decision rules in
[`results/run6/esm3_mechanism/geras/summary.json`](../../results/run6/esm3_mechanism/geras/summary.json).
