# Results: Can ESM-2 Distinguish Mechanism *Within* a Protein Family?

*Companion to [`report_classifier.md`](report_classifier.md) and
[`report_protein_family.md`](report_protein_family.md). Those reports showed that ESM-2's
above-chance mechanism score is largely family recognition, and that subtracting the wildtype
(the delta) removes almost all family signal. This report asks the natural follow-up: if family
identity is held constant — so it cannot be the shortcut — is there any mechanism signal left
to find?*

**Run 6 · 2026-05-31** · ESM-2 `esm2_t33_650M_UR50D` · 28 qualifying Pfam families ·
5 seeds × 5-fold within-family gene-split CV. Results in
[`results/run6/within_family_mechanism.json`](../../results/run6/within_family_mechanism.json).

---

## Summary

We looked at the signal within each family separately. For a given protein family, the family-recognition shortcut is removed, so any score above the per-family baseline is a candidate for genuine within-family mechanism signal. On the headline classification metric — macro-F1 — we find almost none. Across the 28 families large enough to test, the delta (mutation-induced) embedding sits at or near the per-family majority baseline in nearly every family. The `wt_only` embedding scores higher in most families, but that is the same family-identity effect operating within a family, not mechanism read from the mutation. No family shows a delta result that is both clearly above baseline and stable across seeds once degenerate cases are set aside.

A per-class ranking metric (one-vs-rest AUROC) qualifies that null slightly. Most per-class AUROCs hover at 0.5 (the no-signal value), and DN and LOF do so almost everywhere. But the delta separates GOF from the rest at a modestly above-chance level in most families that contain GOF genes — including the largest, most balanced family, PF00520, where the delta's GOF AUROC is 0.66 (linear) to 0.73 (MLP) even though its macro-F1 is at baseline. This is a faint, mostly-GOF, mostly-linear ranking signal that does not translate into above-baseline classification. So the within-family picture is a null for *classifying* mechanism from the delta, with a weak partial signal for *ranking* GOF specifically — consistent with the cross-family result rather than overturning it.

---

## What is measured, and why

Each gene carries one gene-level mechanism label (GOF/DN/LOF). Within a single protein family,
all genes share the family, so a classifier cannot win by recognising the family — it must use
something else. We restrict to one family at a time and ask whether mechanism is recoverable
from two embedding views:

| View | What it is |
|---|---|
| `wt_only` | ESM-2 embedding of the wildtype protein, mean-pooled (one vector per variant) |
| `delta` | mutant minus wildtype (the mutation-induced shift) |

For each family we run **within-family gene-split Cross-Validation**: folds hold out whole genes, so no gene
appears in both train and test. Two probes are run on each view — a linear logistic regression
and an MLP — and every number is the mean ± std across **5 seeds** (the fold assignment is
reshuffled per seed). Per-family gene counts are tiny (6–33 genes), so single-seed numbers are
dominated by which gene lands in which fold; the seed spread is the honest error bar.

**Two metrics.** We report two numbers per probe. **Macro-F1** is a classification metric — it
asks whether the probe assigns the right label, and it is sensitive to class imbalance, so its
reference point is the per-family majority baseline below. **Per-class one-vs-rest AUROC** is a
ranking metric — for each class it asks whether the probe scores that class's variants above the
rest, independent of any decision threshold or of the class frequencies. AUROC's no-signal value
is a fixed 0.5 for every class and every family, which makes it a cleaner read of "is there *any*
separation" than macro-F1 in these imbalanced, tiny families. A class is only scored on folds
where it appears in both the held-out set and the training set; a class held by a single gene
(present in at most one fold's test set) cannot be scored and is left blank.

**Baseline (for macro-F1).** For each family we report the macro-F1 of always predicting that
family's most common class (`majority_baseline_f1`). A feature only carries within-family
classification signal if it beats this baseline — not merely if it beats the global 0.29 floor,
because a family skewed toward one class has a higher floor of its own. (AUROC needs no such
per-family baseline; 0.5 is the floor everywhere.)

**Why some cells are blank.** A family is kept if it has ≥6 genes and ≥2 classes, but within-
family CV can still fail to produce a scorable fold — a minority class with a single gene is
either absent from training (the probe can't learn it) or absent from test (it can't be scored),
and with only 6–8 genes a fold can end up single-class. When no fold is scorable across all 5
seeds the cell is reported as blank rather than as a fabricated 0. This happened for 8 of the 28
families; it is a property of the sample sizes, not a result.

---

## Table 1 — Within-family macro-F1, delta vs wt_only (mean ± std over 5 seeds)

Families with at least one scorable probe, ordered by gene count. Each probe column is a macro-F1
(0–1, higher is better) as mean ± std over 5 seeds. The last column, **base (majority F1)**, is
the score from ignoring ESM-2 and always predicting the family's most common mechanism — the bar
every probe must beat to show real within-family signal. A probe only carries mechanism signal if
it sits clearly above the `base` *in its own row*. Bold marks a delta cell that clears its base
and has std < 0.10 (the bar for "stable, real-looking signal").

| Family | n genes | n var | logreg wt | logreg delta | mlp wt | mlp delta | base (majority F1) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PF00046 | 33 | 179 | 0.347 ± 0.068 | 0.333 ± 0.054 | 0.408 ± 0.161 | 0.367 ± 0.064 | 0.316 |
| PF00069 | 23 | 192 | 0.538 ± 0.124 | 0.344 ± 0.024 | 0.505 ± 0.092 | 0.368 ± 0.058 | 0.234 |
| PF00520 | 15 | 1044 | 0.304 ± 0.032 | 0.256 ± 0.030 | 0.304 ± 0.044 | 0.299 ± 0.034 | 0.253 |
| PF00168 | 14 | 67 | 0.857 ± 0.118 | 0.493 ± 0.077 | 0.815 ± 0.094 | 0.575 ± 0.111 | 0.294 |
| PF00071 | 13 | 157 | 1.000 ± 0.000 | 0.755 ± 0.027 | 0.569 ± 0.152 | 0.626 ± 0.034 | 0.317 |
| PF00038 | 13 | 84 | 0.333 ± 0.033 | 0.313 ± 0.027 | 0.445 ± 0.131 | 0.266 ± 0.036 | 0.305 |
| PF00104 | 13 | 93 | 0.261 ± 0.057 | 0.408 ± 0.119 | 0.491 ± 0.266 | 0.443 ± 0.103 | 0.422 |
| PF00023 | 11 | 240 | 0.241 ± 0.082 | 0.451 ± 0.102 | 0.296 ± 0.087 | 0.415 ± 0.108 | 0.429 |
| PF00004 | 11 | 125 | 0.410 ± 0.024 | 0.221 ± 0.039 | 0.243 ± 0.271 | 0.210 ± 0.040 | 0.228 |
| PF01094 | 10 | 212 | 0.295 ± 0.069 | 0.236 ± 0.111 | 0.303 ± 0.052 | 0.152 ± 0.072 | 0.197 |
| PF02931 | 9 | 78 | 0.510 ± 0.171 | 0.236 ± 0.051 | 0.479 ± 0.150 | 0.224 ± 0.027 | 0.233 |
| PF00010 | 8 | 44 | 0.318 ± 0.105 | 0.407 ± 0.060 | 0.246 ± 0.166 | **0.565 ± 0.061** | 0.470 |
| PF00001 | 7 | 52 | 0.271 ± 0.108 | 0.246 ± 0.051 | 0.464 ± 0.279 | 0.383 ± 0.134 | 0.373 |
| PF01410 | 7 | 298 | 0.198 ± 0.066 | 0.393 ± 0.074 | 0.343 ± 0.198 | 0.365 ± 0.058 | 0.447 |
| PF13246 | 7 | 139 | 0.147 ± 0.147 | 0.481 ± 0.044 | 0.173 ± 0.120 | 0.449 ± 0.012 | 0.485 |
| PF00130 | 6 | 99 | 0.740 ± 0.260 | 0.394 ± 0.086 | 0.176 ± 0.104 | 0.394 ± 0.086 | 0.476 |
| PF00167 | 6 | 20 | 0.367 ± 0.033 | 0.367 ± 0.033 | 1.000 ± 0.000 | 0.333 ± 0.000 | 0.429 |
| PF00503 | 6 | 57 | 0.683 ± 0.448 | 0.315 ± 0.107 | 0.424 ± 0.414 | 0.367 ± 0.047 | 0.472 |
| PF00431 | 6 | 24 | 0.889 ± 0.111 | 0.571 ± 0.429 | 0.571 ± 0.429 | 0.325 ± 0.075 | 0.385 |
| PF07679 | 6 | 146 | 0.337 ± 0.132 | 0.365 ± 0.123 | 0.439 ± 0.300 | 0.361 ± 0.120 | 0.477 |

Eight further families (PF00096, PF00250, PF00041, PF00008, PF00106, PF07714, PF00076, PF12662)
produced no scorable fold across all seeds and are omitted — see "Why some cells are blank."

![Per-family within-family delta macro-F1 minus each family's own majority baseline. Most families straddle zero; the few clear positives are small or single-gene-class families.](figures/fig4_within_family.png)

*Per-family delta (MLP) macro-F1 minus that family's own majority baseline, ordered by the gap (5-seed mean ± std). Bars to the right of zero exceed the family's baseline. The families are small (6–33 genes), so per-family scores are dominated by fold assignment; hatched families contain a mechanism class held by a single gene and have a degenerate score.*

---

## Table 2 — Within-family per-class AUROC for the delta (mean ± std over 5 seeds)

This is the **delta** view only (the mutation-induced shift — the view that isolates mechanism
from family identity), reported as one-vs-rest AUROC for each mechanism class. Every cell is on a
common 0–1 scale where **0.50 is no signal**, above 0.50 means the probe ranks that class's
variants above the rest, and below 0.50 means it ranks them below (anti-correlated, expected under
noise on tiny samples). A blank means that class is held by a single gene in that family and cannot
be scored (see "Two metrics"). Families are ordered by gene count, as in Table 1.

| Family | n genes | logreg GOF | logreg DN | logreg LOF | mlp GOF | mlp DN | mlp LOF |
|---|---:|---:|---:|---:|---:|---:|---:|
| PF00046 | 33 | 0.87 ± 0.05 | 0.37 ± 0.03 | 0.59 ± 0.08 | 0.94 ± 0.04 | 0.31 ± 0.04 | 0.57 ± 0.17 |
| PF00069 | 23 | 0.61 ± 0.04 | 0.34 ± 0.04 | 0.57 ± 0.06 | 0.58 ± 0.10 | 0.42 ± 0.06 | 0.61 ± 0.04 |
| PF00520 | 15 | 0.66 ± 0.05 | 0.51 ± 0.03 | 0.48 ± 0.04 | 0.73 ± 0.05 | 0.57 ± 0.05 | 0.38 ± 0.03 |
| PF00168 | 14 | 0.74 ± 0.06 | 0.87 ± 0.06 | 0.64 ± 0.13 | 0.81 ± 0.06 | 0.87 ± 0.11 | 0.77 ± 0.15 |
| PF00071 | 13 | 0.95 ± 0.01 | — | 0.95 ± 0.01 | 0.92 ± 0.03 | — | 0.93 ± 0.04 |
| PF00038 | 13 | — | 0.73 ± 0.22 | 0.35 ± 0.11 | — | 0.82 ± 0.22 | 0.47 ± 0.17 |
| PF00104 | 13 | — | 0.60 ± 0.11 | 0.60 ± 0.11 | — | 0.63 ± 0.19 | 0.63 ± 0.19 |
| PF00023 | 11 | 0.57 ± 0.05 | — | 0.57 ± 0.05 | 0.62 ± 0.12 | — | 0.62 ± 0.12 |
| PF00004 | 11 | — | 0.25 ± 0.04 | 0.36 ± 0.18 | — | 0.30 ± 0.19 | 0.35 ± 0.19 |
| PF01094 | 10 | 0.52 ± 0.20 | — | 0.71 ± 0.09 | 0.42 ± 0.17 | — | 0.62 ± 0.10 |
| PF02931 | 9 | 0.20 ± 0.10 | 0.80 ± 0.06 | 0.45 ± 0.09 | 0.24 ± 0.12 | 0.72 ± 0.15 | 0.30 ± 0.09 |
| PF00010 | 8 | — | 0.74 ± 0.06 | 0.74 ± 0.06 | — | 0.65 ± 0.12 | 0.65 ± 0.12 |
| PF00001 | 7 | 0.53 ± 0.12 | — | 0.53 ± 0.12 | 0.52 ± 0.19 | — | 0.52 ± 0.19 |
| PF01410 | 7 | — | 0.48 ± 0.07 | 0.47 ± 0.07 | — | 0.43 ± 0.08 | 0.43 ± 0.08 |
| PF13246 | 7 | 0.25 ± 0.19 | — | 0.25 ± 0.19 | 0.26 ± 0.08 | — | 0.26 ± 0.08 |
| PF00130 | 6 | 0.64 ± 0.19 | — | 0.64 ± 0.19 | 0.67 ± 0.25 | — | 0.67 ± 0.25 |
| PF00167 | 6 | 0.75 ± 0.25 | — | 0.75 ± 0.25 | 0.42 ± 0.08 | — | 0.42 ± 0.08 |
| PF00503 | 6 | 0.36 ± 0.08 | — | 0.36 ± 0.08 | 0.34 ± 0.16 | — | 0.34 ± 0.16 |
| PF00431 | 6 | 1.00 ± 0.00 | — | 1.00 ± 0.00 | 0.00 ± 0.00 | — | 0.00 ± 0.00 |
| PF07679 | 6 | 0.47 ± 0.32 | — | 0.47 ± 0.32 | 0.43 ± 0.33 | — | 0.43 ± 0.33 |

Across the families where GOF is scorable, the delta's GOF AUROC has a median of 0.61 for the
linear probe (range 0.20–1.00) and 0.52 for the MLP (range 0.00–0.94); DN and LOF medians sit at
roughly 0.5. The high and the zero/one extremes (PF00431, PF00071) are the same small single-gene-
class families that produce the degenerate macro-F1 cells in Table 1 — a six-gene 4-vs-2 split is
an easy or a coin-flip rank, not discrimination. The stable, non-degenerate above-0.5 GOF cells —
PF00046 (0.87/0.94), PF00520 (0.66/0.73), PF00069 (0.61/0.58) — are the substantive ones.

---

## Reading the tables

Each point reads one or two cells and states its interpretation.

**1. In the largest, most balanced family, the delta does not classify mechanism — but it does
weakly rank GOF.** PF00520 (ion channel) has the most data — 1,044 variants, all three classes.
Its delta macro-F1 is 0.256 (linear) and 0.299 (MLP) against a 0.253 baseline, so as a *classifier*
the mutation adds nothing here. But Table 2 shows the delta's GOF AUROC is 0.66 (linear) and 0.73
(MLP) — clearly above the 0.5 no-signal line, while DN (0.51/0.57) and LOF (0.48/0.38) sit at
chance. So the delta does carry a faint, GOF-specific ordering in even this family; it is just too
weak, and too concentrated in one class, to move a thresholded macro-F1 above the majority
baseline. This is the one place the two metrics genuinely disagree, and the AUROC is the more
honest read of "is there any separation."

**2. Where wt_only beats delta, it is family structure, not the mutation.**
In PF00069 (kinase) wt_only reaches 0.538 while delta stays at 0.344, and in PF00168 it is 0.857
vs 0.493. The protein embedding scores higher in most families, but it is reading which gene this
is relative to its family-mates — not what the mutation does. The delta, which isolates the
mutation, does not share the lift.

**3. The high scores are degenerate or unstable.**
PF00071 (Ras GTPase) shows wt_only = 1.000, but the family is almost all one class with a single
odd gene out — an easy split, not real discrimination. The wildly swinging cells (PF00431,
PF00503, PF00167, with std up to 0.45) are coin-flips on a handful of genes.

**4. No family clears the macro-F1 bar.**
Only one delta cell beats its baseline and stays stable across seeds (PF00010), and that family
has just 8 genes — one gene flipping moves it. Every other delta result is at baseline, below it,
or too noisy to call. There is no family where the delta *classifies* mechanism.

**5. The AUROC signal that exists is GOF-specific and linear.**
Reading Table 2 down the columns: DN and LOF AUROCs sit at ~0.5 almost everywhere, but the
delta's GOF AUROC is above 0.5 in the stable families that contain GOF (PF00046 0.87/0.94,
PF00520 0.66/0.73, PF00069 0.61/0.58). The lift is generally as large or larger for the linear
probe than the MLP, so it is not a nonlinear effect. This is a real but narrow finding: the
mutation shift orders GOF variants somewhat above non-GOF within a family, without being strong
enough to classify them. It does not extend to DN or LOF.

---

## Interpretation

Within-family CV is the strongest available test for mechanism-in-the-mutation: it strips out the
family-recognition shortcut that inflates the gene-split scores in
[`report_classifier.md`](report_classifier.md). Under that test the delta does not classify
mechanism — every macro-F1 is at or near its per-family baseline. The one qualification is the
per-class AUROC: the delta ranks GOF modestly above non-GOF in most families that contain GOF
genes (median GOF AUROC ≈0.61 linear), while DN and LOF stay at chance. That GOF-ranking signal is
weak, one-class, and does not survive into thresholded classification, so it tightens the
project's central finding rather than overturning it. Notably the lift is linear, not nonlinear,
so it is *not* the same thing as the cross-family MLP-over-linear gap (MLP ≈0.40 vs linear 0.29 in
[`report_classifier.md`](report_classifier.md)); that nonlinear cross-family lift does not reappear
within families, consistent with it being residual family structure the subtraction did not fully
remove (the faint leftover quantified in [`report_protein_family.md`](report_protein_family.md)).

The granularity mismatch noted in the classifier report applies here too: mechanism labels are
gene-level, so within a family there are only a handful of labelled points (6–33 genes), and a
variant-level delta averaged to a gene is a poor match for a gene-level label measured on so few
genes. Even if a faint within-family signal existed, these sample sizes could not establish it.

---

## What this is and is not

- **Not a claim that mechanism is unlearnable within families in principle** — only that the
  frozen ESM-2 delta does not *classify* it at these sample sizes, and that the wt_only advantage
  is within-family identity rather than mutation signal.
- **Not a claim of zero signal.** The delta ranks GOF modestly above non-GOF within most families
  (one-vs-rest AUROC, Table 2). The honest statement is: a faint, GOF-specific, linear ranking
  signal exists; it does not reach above-baseline classification and does not extend to DN or LOF.
- **Not a verified above-chance result either** — the GOF-AUROC lift is reported descriptively
  with seed spread, not tested against 0.5 with a significance threshold; at 6–33 genes per family
  it is suggestive, not established (see Statistical limitations).
- **Not contradicted by the high wt_only cells** — those are explained by the family-structure
  effect, not by mechanism.
- The blank families are a sample-size artifact (single-gene minority classes), reported as blank
  rather than imputed.

---

## Statistical limitations and planned analyses (pre-preprint)

Per-family sizes are 6–33 genes, so both tables are descriptive, not inferential. Planned before
preprint submission, not yet in the result files:

- **Test the GOF-AUROC lift against 0.5.** The GOF ranking signal (Table 2) is the one positive
  result and is currently reported only as mean ± seed-std. Add a cluster bootstrap over genes per
  family, or a permutation test on the labels, to establish whether GOF AUROC is significantly
  above 0.5 — and pool the GOF-bearing families for a single better-powered estimate.
- **Multiple-comparison control:** 28 families × 3 classes are screened; both the "beats baseline,
  std < 0.10" macro-F1 highlight and the per-class AUROC reads are uncorrected. Add
  Benjamini-Hochberg FDR, or restate as exploratory.
- **Power:** at 6–33 genes the macro-F1 test can only fail to detect signal, not rule it out — add
  a minimal-detectable-effect per family so nulls read as underpowered.
- **Confidence intervals** from a cluster bootstrap over genes within each family.

---

## Provenance

Computed by `experiments/mechanism/mechanism_within_family.py` on the run6 embeddings
(`embeddings_wt_mean.npy`, `embeddings_mut_mean.npy`), `valid_variants.json` (gene-level
`label_3class`), and `pfam_families.json`. Qualifying families: ≥6 genes and ≥2 mechanism
classes; 28 families, 8 with no scorable fold. Within-family gene-split CV (5 folds, relaxed
size guards for small families), logistic regression and MLP probes, 5 seeds; per-family
macro-F1 and per-class one-vs-rest AUROC reported as mean ± std. Per-family majority-class
baseline included. Output:
[`results/run6/within_family_mechanism.json`](../../results/run6/within_family_mechanism.json).
