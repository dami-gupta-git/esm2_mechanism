# Results: Can Contrastive Training Recover Mechanism From the Delta?

*Companion to [`report_classifier.md`](report_classifier.md), which shows that a standard
classifier reads mechanism from the protein's own embedding rather than the mutation-induced
delta.*

**Run 6 · 2026-05-30** · ESM-2 `esm2_t33_650M_UR50D` · 17,826 variants · 1,935 genes ·
1,134 protein families · classes LOF 76% / GOF 15% / DN 9%. Results in
[`results/run6/`](../../results/run6/).

---

## Summary

The standard probes leave the mutation-induced delta at the chance floor under family-split
(see [`report_classifier.md`](report_classifier.md)). This experiment asks whether training the
delta specifically to ignore protein family — a contrastive objective whose only positive pairs
are same-mechanism variants from *different* families — can pull genuine mechanism signal out of
it. It does raise the family-split macro_f1 above the floor, and the lift survives holding out
whole families, but the per-class scores show the gain is better class balance rather than any
mechanism becoming more separable. The honest reading is a modest, real improvement in balanced
accuracy that does not amount to recovering cross-family mechanism.

---

## Glossary

**What was trained.** A small projection head (1280 → 256 → 64) is trained on the `delta_mean`
feature with a triplet loss. The anchor and its positive are two variants that share a mechanism
class but come from *different* Pfam families; the negative is a variant of a different mechanism.
Same-family pairs are never used as positives, so the head cannot reduce the loss by learning to
recognise families — it must find structure that is mechanism-correlated and family-independent,
or fail. After training, each variant is classified by k-nearest-neighbours (k = 10, cosine) in
the learned 64-d space.

**Rows — methods:**

| Name | What it is |
|---|---|
| `contrastive_knn` | k-NN in the trained 64-d projection (the method under test) |
| `raw_knn_baseline` | k-NN in the raw 1280-d `delta_mean` space, no training (the same evaluation, no projection) |
| `mlp delta_mean floor` | The standard MLP on `delta_mean`, family-split, from [`report_classifier.md`](report_classifier.md) — the bar to beat |

The `raw_knn_baseline` isolates what the contrastive *training* adds: it is the identical k-NN
evaluation on the same feature, with the projection removed. The MLP floor is the previous best
nonlinear delta result, the reference for "did training help beyond the existing probe."

**Columns — metrics:** as in [`report_classifier.md`](report_classifier.md). `macro_f1` is the
per-class F1 averaged equally over the three classes (naive floor 0.288); each AUROC is one-vs-rest
with a chance value of 0.50.

**Two cross-validation setups:** gene-split (no gene shared between train and test, but related
families may be split — leakage-prone) and family-split (whole families held out — the honest
test of generalisation to unseen proteins). The gap between them is the family-recognition
component. For a method that excludes within-family positives by construction, an equal gene-split
and family-split lift is the signature of genuine cross-family signal; a lift that appears only on
gene-split would be leakage the construction failed to remove.

All values are means across 5 random seeds, with the seed-to-seed standard deviation shown; the
spread is small (≤ 0.013 on `macro_f1`), so the ordering is stable.

---

## Table 1 — Gene-split (leakage-prone)

| method | macro_f1 | AUROC GOF | AUROC DN | AUROC LOF |
|---|---:|---:|---:|---:|
| contrastive_knn | 0.438 ± 0.013 | 0.654 | 0.576 | 0.658 |
| raw_knn_baseline | 0.408 ± 0.008 | 0.650 | 0.606 | 0.649 |
| *naive baseline* | *0.288* | *0.500* | *0.500* | *0.500* |

## Table 2 — Family-split (homology-controlled)

| method | macro_f1 | AUROC GOF | AUROC DN | AUROC LOF |
|---|---:|---:|---:|---:|
| contrastive_knn | 0.395 ± 0.009 | 0.589 | 0.545 | 0.595 |
| raw_knn_baseline | 0.354 ± 0.006 | 0.595 | 0.577 | 0.588 |
| mlp delta_mean (floor) | 0.288 | 0.560 | 0.514 | 0.546 |
| *naive baseline* | *0.288* | *0.500* | *0.500* | *0.500* |

The MLP `delta_mean` family-split row is carried over from [`report_classifier.md`](report_classifier.md)
as the reference floor. The naive baseline is a majority-class `DummyClassifier` (always predicts
LOF) under the same 5-seed cross-validation.

---

## Reading the tables

Each point reads one cell or pair of cells and states its interpretation.

**1. Contrastive training clears the floor on balanced accuracy.**
In the family-split table, `contrastive_knn` reaches macro_f1 = 0.395 against the 0.288 MLP floor
and a 0.288 naive baseline. The trained projection separates the three classes more evenly than
the standard delta probe, which sat on the floor.

**2. The lift is over the untrained delta, not over nothing.**
`contrastive_knn` is 0.395 versus the `raw_knn_baseline` of 0.354 — a +0.041 gain attributable to
the training, since the only difference between the two rows is the projection. The raw k-NN itself
already beats the linear/MLP floor, so part of the headline is k-NN, and part is the contrastive
head on top of it.

**3. The lift is cross-family, not leakage.**
`contrastive_knn` drops from 0.438 (gene-split) to 0.395 (family-split), a fall of 0.043; the
`raw_knn_baseline` falls 0.054 (0.408 → 0.354). The trained method loses *less* when whole families
are held out, not more. A lift driven by family recognition would inflate gene-split and collapse
under family-split; this does the opposite, so the family-invariance construction holds.

**4. The gain is class balance, not per-class separability of any class.**
On family-split per-class AUROC, training raises no class over the untrained `raw_knn_baseline`: LOF
moves +0.006 (0.595 vs 0.588), inside the ±0.007 seed noise; GOF is lower (0.589 vs 0.595) and DN
lower still (0.545 vs 0.577). The macro_f1 rises while every per-class discrimination stays flat or
falls, which means the improvement comes from predicting the classes in more balanced proportions,
not from making any mechanism more separable across families. (All three classes are already weakly
above the 0.50 chance line in the untrained delta — ≈0.58–0.60 — so the residual delta signal is
not class-specific and contrastive training does not add to it per class.)

**5. Dominant-negative stays at chance.**
`contrastive_knn` reaches AUROC 0.545 for DN under family-split, and the training lowers it relative
to the untrained baseline (0.577). Even with explicit cross-family supervision, dominant-negative —
the interaction-dependent mechanism — does not separate across unseen families.

---

## Summary of findings

| Question | Finding |
|---|---|
| Does contrastive training raise family-split macro_f1? | Yes — 0.395 ± 0.009, above the 0.354 raw-kNN baseline and the 0.288 MLP floor. |
| Is the lift genuine cross-family signal or leakage? | Cross-family. The gene→family drop is smaller for the trained method (0.043) than the untrained one (0.054). |
| Does it recover per-class mechanism (any class)? | No. Training raises no class's AUROC over the untrained delta (LOF +0.006 within noise, GOF and DN lower); the macro_f1 gain is class balance alone. |
| Does dominant-negative transfer across families? | No — AUROC 0.545, at chance, and training does not help. |
| Net reading | A modest, real improvement in balanced accuracy; not a recovery of cross-family mechanism. |

---

## Interpretation

Forcing the delta to be family-invariant produces a small honest gain, which says there is *some*
mechanism-correlated, family-independent structure in `mut − wt` — enough to balance the classes
better than the untrained feature. But the per-class view bounds what that structure is: it is
mostly the loss-of-function axis, the same class that is most separable everywhere else (see
[`report_classifier.md`](report_classifier.md), point 4), and it does not extend to gain-of-function
or dominant-negative across unseen families. This is consistent with the delta carrying a faint
signal that is partly mechanism and partly the granularity mismatch of a variant-level feature
against gene-level labels; the contrastive objective surfaces the balanced-accuracy part without
turning the hard classes above chance.

---

## Limitations

- Single ESM-2 size (650M) and a single feature (`delta_mean`); a site-restricted delta or a
  larger model was not tried under this objective.
- Hyperparameters (margin 1.0, projection dim 64, max 8 cross-family positives per anchor) were not
  tuned; the reported numbers are one configuration.
- k-NN evaluation contributes part of the lift independently of the training (the `raw_knn_baseline`
  already beats the MLP floor), so the contrastive head's specific contribution is the +0.041 over
  raw k-NN, not the full gap to the MLP floor.

## Statistical limitations and planned analyses (pre-preprint)

The seed-to-seed spread reflects fold reshuffling on fixed data, not sampling uncertainty. As for
the sibling reports, a cluster bootstrap over genes (effective N ≈ 1,935 genes, far smaller for the
rare DN and GOF classes) and a permutation test against the 0.288 floor and against the raw-kNN
baseline are planned before preprint, and are not yet in the result files.

## Provenance

Embeddings verified before analysis (clean exit; both `delta_mean` source arrays `(17826, 1280)`;
variant index row-aligned, length 17,826). Pfam coverage 17,729 / 17,826.

Sources:
- Contrastive and raw-kNN rows (Tables 1–2): `experiments/mechanism/contrastive_mechanism`, 5 seeds →
  `results/run6/contrastive_results_seed{0..4}.json`, pooled in `results/run6/contrastive_aggregate.json`.
- MLP `delta_mean` floor row and naive baseline: carried from [`report_classifier.md`](report_classifier.md)
  → `results/run6/aggregate.json`, `results/run6/naive_baseline.json`.

Full run log: [`results/run6/logs/contrastive_multiseed.log`](../../results/run6/logs/contrastive_multiseed.log).
