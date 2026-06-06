# Results: Does the Mechanism Null Survive a Single Curation Source?

*Companion to [`report_classifier.md`](report_classifier.md), which found that ESM-2 delta
embeddings classify mechanism (GOF/DN/LOF) at the chance floor under family-split CV, while the
above-chance gene-split score comes from family recognition in the protein's own embedding. This
report repeats that exact probe on the Gerasimavicius-only subset, removing the curation-source
confound built into the merged dataset.*

**Run 6 · 2026-06-04** · ESM-2 `esm2_t33_650M_UR50D` · 10,138 variants · 942 genes ·
660 Pfam families · classes LOF 7,262 / GOF 1,982 / DN 894. Results in
[`results/run6/single_source_gerasimavicius/`](../../results/run6/single_source_gerasimavicius/).

---

## Summary

The merged dataset draws its mechanism labels from two curation sources, and the classes are split
unevenly between them. So a reviewer could reasonably ask whether the classifier report is really
about biology or just about which database a variant came from. To check, we dropped the second
source and reran the exact same probe on the Gerasimavicius variants alone, which still cover all
three classes. Nothing changed: the delta stays at chance on both splits, and the wildtype
embedding still scores well when related genes are in training and falls back sharply once whole
families are held out. The mechanism null is not an artifact of mixing sources.

---

## What is measured, and why

The merged dataset combines Gerasimavicius (the primary mechanism source) with Gene2Phenotype
(an additional set of genes). Because the four label subtypes are drawn unevenly from the two
sources, any class-level effect on the merged set conflates mechanism with curation: a difference
between, say, the AR and HI loss-of-function subtypes could reflect that they come from different
databases rather than anything about the variants.

This report removes the confound the simplest way — by dropping Gene2Phenotype and keeping only
the Gerasimavicius variants, all curated through one pipeline and still spanning all three
mechanism classes. Everything else is held fixed: the same `load_data`, the same feature set, the
same gene-split and family-split cross-validation, the same probe, and the same five seeds as the
classifier report. Only the row set changes. The majority-class floor is recomputed on the subset,
because the subset's class balance — and therefore the floor — differs from the merged set.

The features, metrics, and cross-validation schemes are exactly those defined in
[`report_classifier.md`](report_classifier.md); they are not repeated here. In brief: **gene-split**
holds out whole genes (related genes may sit in train and test, so family resemblance is available
as a shortcut); **family-split** holds out whole Pfam families (the leakage-free test); `macro_f1`
is the per-class F1 averaged equally over the three classes; each AUROC is one-vs-rest with a
chance value of 0.50.

**The subset floor.** The merged-set floor is 0.288; on the Gerasimavicius-only subset it is
**0.279** (gene-split) and **0.279** (family-split), measured from a majority-class
`DummyClassifier` under the same five-seed CV. The floor moves because the subset is slightly more
imbalanced toward LOF than the merged set. This recomputed value, not the merged 0.288, is the bar
a feature must clear here.

---

## Table 1 — Gene-split (leakage-prone)

| Feature | Macro_f1 | AUROC GOF | AUROC DN | AUROC LOF |
|---|---:|---:|---:|---:|
| wt_only_mean | 0.612 | 0.873 | 0.771 | 0.902 |
| mut_only_mean | 0.612 | 0.874 | 0.770 | 0.902 |
| wt_concat_mut | 0.612 | 0.871 | 0.753 | 0.894 |
| delta_mean | 0.279 | 0.628 | 0.480 | 0.602 |
| delta_per_residue | 0.323 | 0.626 | 0.518 | 0.606 |
| onehot_aa | 0.280 | 0.590 | 0.486 | 0.581 |
| foldx_ddg | 0.279 | 0.619 | 0.589 | 0.629 |
| alphamissense | 0.279 | 0.613 | 0.637 | 0.644 |
| *naive baseline* | *0.279* | *0.500* | *0.500* | *0.500* |

## Table 2 — Family-split (homology-controlled)

| Feature | Macro_f1 | AUROC GOF | AUROC DN | AUROC LOF |
|---|---:|---:|---:|---:|
| wt_only_mean | 0.445 | 0.802 | 0.747 | 0.854 |
| mut_only_mean | 0.445 | 0.802 | 0.745 | 0.854 |
| wt_concat_mut | 0.445 | 0.787 | 0.736 | 0.837 |
| delta_mean | 0.279 | 0.574 | 0.500 | 0.559 |
| delta_per_residue | 0.309 | 0.601 | 0.506 | 0.593 |
| onehot_aa | 0.280 | 0.574 | 0.496 | 0.568 |
| foldx_ddg | 0.279 | 0.617 | 0.595 | 0.623 |
| alphamissense | 0.279 | 0.610 | 0.649 | 0.649 |
| *naive baseline* | *0.279* | *0.500* | *0.500* | *0.500* |

The naive baseline is a majority-class classifier (always predicts LOF) measured under the same
five-seed CV on the subset. Its macro_f1 of 0.279 is the value that `delta_mean`, `onehot_aa`,
`foldx_ddg`, and `alphamissense` match — those features are performing at the majority-class
baseline. All values are five-seed means; the seed-to-seed standard deviation on `macro_f1` is
larger here than on the merged set (up to ≈0.03 for the absolute-embedding features) because the
subset has fewer genes and families, so the ordering is stable but the absolute-embedding numbers
carry more fold-assignment jitter.

---

## Reading the tables

Each point below reads one cell or pair of cells and states its interpretation, mirroring the
classifier report on the merged set.

**1. The delta is still at the floor.**
On family-split, `delta_mean` scores macro_f1 = 0.279 — the recomputed subset floor. Given only
the mutation-induced embedding change, the classifier separates GOF/DN/LOF at chance, exactly as
on the merged set. Removing the second curation source does not surface any delta signal.

**2. The wildtype embedding still collapses under family-split.**
`wt_only_mean` scores 0.612 on gene-split but 0.445 on family-split, a drop of 0.167. The feature
performs well when other variants from the same gene may appear in training and loses about a
quarter of its score on unfamiliar families. This is the same family-recognition signature the
merged set showed (0.545 → 0.442 there); the larger gene-split value here reflects the subset's
own gene structure, but the direction and the collapse are identical.

**3. The reference pathogenicity predictor still fails.**
On both splits, `alphamissense` scores at the subset floor (0.279). As on the merged set,
mechanism is not recoverable from a pathogenicity score — the limitation is not specific to ESM-2
and not specific to the merged dataset.

**4. Loss-of-function is still the most separable class.**
On gene-split, `wt_only_mean` reaches AUROC 0.902 for LOF versus 0.771 for DN. Loss-of-function
discriminates more readily than the interaction-dependent dominant-negative mechanism — the same
ordering as the merged set.

**5. Dominant-negative is at chance for the delta.**
On family-split, `delta_mean` reaches AUROC 0.500 for DN — exactly chance. Even on a single-source
subset, the mutation-induced shift carries nothing about the dominant-negative mechanism across
unseen families, consistent with the contrastive report's finding that DN does not transfer.

**6. Per-residue delta edges above the floor, as before.**
`delta_per_residue` scores 0.323 gene-split and 0.309 family-split, marginally above the 0.279
floor — the same weak local-over-global pattern seen on the merged set (0.315 / 0.305 there),
still near the floor.

---

## The robustness read

The merged-set finding and its single-source replication, side by side:

| Quantity | Merged set ([`report_classifier.md`](report_classifier.md)) | Gerasimavicius-only (this report) |
|---|---|---|
| Subset floor (family-split) | 0.288 | 0.279 |
| delta_mean macro_f1 (family-split) | 0.288 (at floor) | 0.279 (at floor) |
| wt_only_mean macro_f1, gene-split | 0.545 | 0.612 |
| wt_only_mean macro_f1, family-split | 0.442 | 0.445 |
| wt_only_mean gene→family drop | 0.103 | 0.167 |

Both signatures survive the removal of the second curation source: the delta sits at the floor on
both splits, and the absolute-embedding gene-split lift is family recognition that does not survive
holding out whole families. The two findings that carry the classifier report — delta-at-floor and
wt-only-collapse — are properties of the representation and the label granularity, not of the
source mixture.

---

## What this is and is not

- **Not a new analysis** — it is the classifier report's probe rerun on a filtered row set, to
  rule out one specific confound (curation source). The contribution is that the null holds when
  the source mixture is removed.
- **Not a claim that the subset is preferable for the headline.** The merged set is larger and is
  the primary result; this subset is the robustness check, and is noisier because it has fewer
  genes (942 vs 1,935) and families (660 vs 1,134).
- **Does not address the granularity mismatch.** Mechanism labels are still gene-level here, so the
  deeper limitation the classifier report names — a variant-level delta against a gene-level label —
  applies unchanged. Single-source filtering removes the source confound, not the granularity one.

---

## Statistical limitations and planned analyses (pre-preprint)

The seed-to-seed spread reflects fold reshuffling on a fixed subset, not sampling uncertainty, and
the subset's smaller gene and family counts make the absolute-embedding spreads wider than on the
merged set. Planned before preprint submission, not yet in the result files (matching the
classifier report, recomputed on this subset):

- **Confidence intervals** from a cluster bootstrap over genes (the effective N is ≈ 942 genes,
  smaller than the merged set and far smaller for the rare classes, DN ≈ 9% and GOF ≈ 20% here),
  replacing the seed-std bars. The CIs are expected to be wide; the point is whether `delta_mean`'s
  interval still straddles the floor.
- **Permutation test** against the recomputed subset floor (0.279, not the merged 0.288) for a
  p-value on "above chance" and on the gene-split versus family-split gap on `wt_only_mean`.

---

## Provenance

Computed by `experiments/mechanism/single_source_mechanism.py`, which filters the merged variant
set to `source == "gerasimavicius"` and reruns the Section 2 gene-split vs family-split probe
unchanged on the subset (10,138 of 17,826 variants; 942 genes; 660 Pfam families). Features are
ESM-2 650M mean-pooled WT/mutant embeddings, PCA-reduced to 256 components (98.7% variance) for the
embedding features, identical to the classifier report. The subset majority-class floor is
recomputed from a `most_frequent` `DummyClassifier` under the same five-seed gene/family-split CV.
AlphaMissense coverage is carried from the merged extraction (17,733 / 17,826 before filtering).
Probes: 5 seeds, gene-split and family-split CV. Results:
[`aggregate.json`](../../results/run6/single_source_gerasimavicius/aggregate.json),
[`family_split_baselines_seed{0..4}.json`](../../results/run6/single_source_gerasimavicius/),
[`naive_baseline.json`](../../results/run6/single_source_gerasimavicius/naive_baseline.json).
Full run log:
[`single_source_gerasimavicius_run.log`](../../results/run6/single_source_gerasimavicius_run.log).
