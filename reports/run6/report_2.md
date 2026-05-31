# Results: Do ESM-2 Embeddings Cluster by Protein Family?

*Companion to [`report_1.md`](report_1.md). report_1 found that ESM-2 predicts mechanism
above chance, but that the signal is the protein's family identity acting as a proxy rather
than the mutation. This report measures that family clustering directly, turning report_1's
inference into a measured result.*

**Run 6 · 2026-05-31** · ESM-2 `esm2_t33_650M_UR50D` · 17,826 variants · 1,935 genes ·
1,902 with a Pfam annotation · 1,134 families (833 singletons). Results in
[`results/run6/family_clustering.json`](../../results/run6/family_clustering.json).

---

## Summary

ESM-2 embeddings cluster very strongly by protein family. A gene's nearest neighbours in
embedding space share its family roughly 50× more often than chance, and a linear probe
recovers which of 145 families a gene belongs to with 61% accuracy versus a 4.4% baseline.
We also find that, among genes in multi-gene families, a large fraction (83%) carry their
family's majority mechanism label, so a classifier that merely recognises the family and
predicts its usual mechanism would score well — which is what the WT-only baseline in report_1
is largely doing. Subtracting the wildtype (the delta) removes
almost all of this family signal: family prediction from the delta falls to exactly the
chance baseline.

---

## What is measured, and why

The question is whether the WT-only mechanism score in report_1 (≈0.55 gene-split, ≈0.44
family-split) reflects genuine mechanism understanding or just family recognition. The
leakage would work like this: related proteins (e.g. BRCA1 and BRCA2) appear in both train
and test under gene-split; the model learns "proteins that look like BRCA1 are LOF" and
recognises BRCA2 by family resemblance, not by anything about the mutation.

This report measures how strongly the embeddings cluster by family, on three embedding views,
using metrics that do not depend on the mechanism labels at all.

**Embedding views:**

| View | What it is |
|---|---|
| `wt_mean` | ESM-2 embedding of the wildtype protein, mean-pooled, averaged to one vector per gene |
| `mut_mean` | same for the mutant protein |
| `delta_mean` | mutant minus wildtype (the mutation-induced shift) |

**Clustering metrics:**

| Metric | The question it answers | "No clustering" value |
|---|---|---|
| k=5 family purity | Of a gene's 5 nearest neighbours, what fraction share its family? | the shuffled-label null (≈0.005 here) |
| within/between ratio | Are same-family genes closer together than different-family genes? (cosine distance) | 1.0 (same-family no closer than different) |
| family-probe accuracy | Can a linear probe identify which of 145 families a gene is in, from its embedding? | the majority-family baseline (0.044) |
| silhouette | Tightness of family clusters | reported but unreliable here — see note |

Metrics are computed on the 1,069 genes in non-singleton families (a singleton has no
same-family neighbour, so cannot form a cluster). The family probe uses the 757 genes in the
145 families with ≥3 members (so each family can appear in train and test), under stratified
3-fold CV. A z-score is the distance of the observed value from the shuffled-label null, in
null standard deviations; large positive z means the clustering is far beyond what label
shuffling produces.

---

## Table 1 — Family clustering by embedding view

| View | k=5 purity (null, z) | k=10 purity (z) | Within/between ratio (z) | Family-probe acc (baseline) |
|---|---|---|---|---|
| wt_mean | 0.254 (0.005, z=+249) | 0.169 (z=+155) | 0.514 (z=−15.0) | 0.612 (0.044) |
| mut_mean | 0.255 (0.005, z=+244) | 0.168 (z=+156) | 0.513 (z=−15.1) | 0.612 (0.044) |
| delta_mean | 0.051 (0.005, z=+33) | 0.035 (z=+32) | 0.973 (z=−1.2) | 0.044 (0.044) |

The silhouette score is negative for every view (wt −0.161, delta −0.390) despite every
other metric showing strong clustering. This is a known failure of silhouette with many
singleton clusters, uneven cluster sizes, and high-dimensional embeddings — all present here.
It is reported in the JSON for completeness and ignored in the interpretation; the other three
metrics are consistent and unambiguous.

## Table 2 — Mechanism–family overlap

| Quantity | Value |
|---|---|
| Fraction of multi-gene-family genes whose mechanism matches their family's majority (leave-one-out) | 0.833 |
| Correlation between a gene's family-tightness and whether it matches its family majority (WT) | r = +0.001 (p = 0.98) |

This fraction is measured over the 1,069 genes in non-singleton families: for each gene, the
family majority is taken over its family-mates (excluding the gene itself), and the gene
"matches" if its label equals that majority. It depends only on labels and families, not the
embedding, so it is identical across all three views. The value is partly inflated by class
imbalance — LOF is 76% of variants, so families skew LOF and "matching the majority" is easier
than it would be for balanced classes.

---

## Reading the tables

**1. WT embeddings cluster strongly by family.**
For `wt_mean`, a gene's 5 nearest neighbours share its protein family 25.4% of the time
versus 0.5% under label shuffling — about 50× chance, at z = +249 (far beyond any
significance threshold). The within/between distance ratio of 0.514 means same-family genes
sit about half as far apart as different-family genes. This is expected: ESM-2 was trained to
capture evolutionary and functional relationships, and protein families are defined by exactly
those. The point is not that it happens, but what it does to downstream evaluation.

**2. Family is directly readable from the embedding.**
A linear probe identifies which of 145 families a gene belongs to with 61.2% accuracy, versus
a 4.4% majority-family baseline — roughly 14× the baseline. Family identity is not a subtle
property of the embedding; it is one of its dominant axes.

**3. The mutant embedding behaves identically to the wildtype.**
`mut_mean` matches `wt_mean` on every metric (purity 0.255 vs 0.254, probe 0.612 vs 0.612).
A single missense substitution barely moves the protein-level embedding, so the mutant carries
the same family signal as the wildtype and no distinct mutation information at the family level.

**4. The delta removes almost all family signal.**
For `delta_mean`, k=5 purity drops to 0.051 and family-probe accuracy falls to 0.044 — exactly
the majority baseline, i.e. family is no longer recoverable at all. The within/between ratio
rises to 0.973 (z = −1.2, essentially no clustering). Subtracting the wildtype cancels the
"this is a kinase" signal. A small residual remains in the purity metric (0.051 vs 0.005 null,
z = +33); this faint leftover is the most likely source of the small nonlinear delta lift seen
in report_1 (MLP ≈0.40 vs linear 0.29) — the MLP picking up residual family structure rather
than learning mechanism.

**5. Family predicts mechanism for most genes.**
We find that 83.3% of genes in multi-gene families carry their family's majority mechanism
label (leave-one-out). Combined with point 2, this is the leakage channel: family is readable
from the WT embedding at 61%, and family implies mechanism for 83% of genes, so a "recognise
the family, predict its usual mechanism" strategy reaches a high score without using the
mutation. This accounts for report_1's WT-only baseline.

**6. Tighter clustering does not predict which genes match their family.**
The per-gene correlation between family-tightness and whether a gene matches its family's
majority mechanism is r = +0.001 (p = 0.98) for WT — no relationship. The delta view is
nominally significant (r = +0.070, p = 0.023) but explains only ~0.5% of the variance, so it
is scientifically negligible. Either way, the family-recognition effect is population-level
(families share mechanism on average); it does not mean the most tightly clustered genes are
the most label-consistent. Reported for completeness; it is a null sub-result.

---

## The causal chain

Three measured numbers connect embedding clustering to the report_1 baseline:

1. Family is recoverable from the WT ESM-2 embedding at 61% accuracy (vs 4.4% baseline).
2. 83% of genes in multi-gene families carry their family's majority mechanism label.
3. Therefore a family-recognition classifier reaches a high mechanism score without learning
   mechanism — and report_1's WT-only macro_f1 (0.545 gene-split) is a worked example. Its
   drop to 0.442 under family-split is the portion of that score that family recognition
   cannot reach once whole families are held out.

The delta removes the family signal, but, as report_1 shows, has little mechanism signal of
its own.

---

## What this is and is not

- **Not a discovery that ESM-2 encodes family.** That is by design (Rives et al. 2021, Lin et
  al. 2023) and expected. The contribution is quantifying how much it inflates the
  mechanism baseline on this dataset.
- **Not a claim that mechanism is unlearnable from sequence** — only that the WT-only
  representation under gene-split CV measures family recognition, not mechanism.
- The silhouette metric is unreliable here and is excluded from the conclusions.

---

## Provenance

Computed by `experiments/mechanism/family_clustering.py` on the run6 embeddings
(`embeddings_wt_mean.npy`, `embeddings_mut_mean.npy`), `valid_variants.json`, and
`pfam_families.json`. Metrics on 1,069 genes in non-singleton families; family probe on 757
genes in families of ≥3 under stratified 3-fold CV. Single seed (0); shuffled-label nulls use
20 shuffles. Output: [`results/run6/family_clustering.json`](../../results/run6/family_clustering.json).
