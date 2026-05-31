# The leakage fraction: how much of the gene-split score is homology leakage

**Run 6 · 2026-05-31** · ESM-2 `esm2_t33_650M_UR50D` · 17,826 variants · 1,935 genes ·
1,134 Pfam families · 5 seeds. Source:
[`results/run6/leakage_fraction.json`](../../results/run6/leakage_fraction.json), computed by
`experiments/mechanism/leakage_fraction.py` from the gene-split and family-split macro-F1 in
`family_split_baselines_seed{0..4}.json`.

---

## The quantity

Mechanism classifiers are commonly evaluated with gene-split cross-validation: test genes are
held out, but related genes from the same protein family may remain in training. Because
protein families tend to share a mechanism, a model can score on gene-split by recognising the
family rather than the variant. Family-split CV removes that route by holding out whole
families.

The gap between the two scores is exactly the part of the gene-split result that came from family
recognition. We call it the **leakage fraction** (LF): it re-expresses that gap as a share of the
feature's above-chance signal, so the question shifts from "how many points did the feature lose?"
to "what proportion of what it appeared to know was really just the family shortcut?"

> LF = (gene-split macro-F1 − family-split macro-F1) / (gene-split macro-F1 − chance)

The numerator is how much score is lost when families are held out. The denominator is how
much above-chance score there was to lose. So LF is the share of the above-chance gene-split
signal that does not survive a family-aware hold-out. `chance` is the measured majority-class
floor (macro-F1 = 0.288 here), not a nominal 1/3.

LF is only meaningful for a feature that scores above chance on gene-split. For a feature
already at the floor the denominator is ~0 and the ratio is noise, so it is reported as
undefined.

---

## Result

Per feature, 5-seed mean macro-F1 and the leakage fraction:

| Feature | Gene-split | Family-split | Drop | Leakage fraction |
|---|---:|---:|---:|---:|
| wt_only_mean | 0.545 | 0.442 | 0.103 | 40.1% |
| mut_only_mean | 0.547 | 0.443 | 0.104 | 40.3% |
| wt_concat_mut | 0.556 | 0.451 | 0.106 | 39.4% |
| delta_per_residue | 0.316 | 0.305 | 0.010 | 38.2% |
| delta_mean | 0.288 | 0.288 | 0.000 | undefined (at floor) |
| onehot_aa | 0.288 | 0.288 | 0.000 | undefined (at floor) |
| foldx_ddg | 0.279 | 0.279 | 0.000 | undefined (at floor) |
| alphamissense | 0.288 | 0.290 | −0.001 | undefined (at floor) |

The leakage fraction is computed from the across-seed mean macro-F1, not by averaging per-seed
ratios — the per-seed route divides by `(gene_f1 − chance)`, a small noisy quantity, so one hot
seed inflates its ratio out of proportion.

*Chance = 0.288 (measured majority-class floor). The "at floor" rows score here, so their LF denominator is ~0.*

---

## Reading the result

**About 40% of the absolute-embedding gene-split signal is homology leakage.** Every feature
that scores above chance — the wildtype embedding, the mutant embedding, and their
concatenation — has a leakage fraction near 40% (39.4–40.3%). Two-fifths of what gene-split CV
credits these features with is family recognition that disappears once whole families are held
out. For a mechanism-generalisation claim, only the family-split number (~0.44) is honest; the
gene-split number (~0.55) overstates it by this fraction.

**The three absolute-embedding features agree.** `wt_only_mean`, `mut_only_mean`, and
`wt_concat_mut` give 40.1%, 40.3%, and 39.4%. They carry essentially the same information —
which protein the variant sits in — so they leak by the same amount. That the mutant embedding
matches the wildtype confirms the leaking signal is protein identity, not anything about the
mutation.

**The delta and the non-embedding features have no leakage to measure.** `delta_mean`,
`onehot_aa`, `foldx_ddg`, and `alphamissense` sit at the chance floor on both splits, so there
is no above-chance signal for family recognition to inflate. Their leakage fraction is
undefined, which is the correct outcome: a feature with no signal cannot leak. `delta_per_residue`
sits just above the floor — its 38.2% is a 0.010 drop over a 0.028 denominator, a ratio of two
small noisy numbers, so the figure is not robust and its closeness to ~40% is coincidence, not
corroboration.

**Why ~40% is plausible from the data alone.** 83% of genes carry their family's majority
mechanism label (within-family agreement, from the family-clustering analysis), and the class
distribution is heavily skewed (LOF 76%). A classifier that recognises the family and predicts
its usual mechanism therefore captures a large, family-mediated share of the gene-split score —
which is exactly the share that LF measures and that family-split removes.

---

## What this is and is not

- It is a per-dataset, per-feature diagnostic of how much a gene-split score is inflated by
  family recognition, computed from macro-F1 values that already exist — no extra model
  training.
- It is not a statistical test, and not a claim about any external dataset or model; it
  characterises this dataset and these features.
- It does not say the absolute-embedding features are useless — only that ~40% of their
  gene-split score is family-mediated and that the family-split number is the one to quote for
  generalisation.

## Provenance

Computed by `experiments/mechanism/leakage_fraction.py` from
`results/run6/family_split_baselines_seed{0..4}.json` (gene/family macro-F1 per feature, 5
seeds), with the chance floor read from `results/run6/naive_baseline.json` (majority-class
macro-F1 = 0.288) and within-family agreement from `results/run6/family_clustering.json`.
Output: [`results/run6/leakage_fraction.json`](../../results/run6/leakage_fraction.json). Full
run log: [`RUN_PROGRESS.md`](../../RUN_PROGRESS.md), Run 6.
