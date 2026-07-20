# Introduction: Does ESM-2 Encode Disease Mechanism?

---

## What is ESM-2?

ESM-2 is a protein language model trained on hundreds of millions of protein sequences. Like a language model trained on text, it learns to predict missing amino acids from context — and in doing so, it builds up internal representations that capture something about protein biology: evolutionary conservation, structural propensity, functional site identity.

The output of ESM-2 is an **embedding** — a vector of 1,280 numbers for each residue in the protein. These numbers encode the model's representation of that position in context. For a given variant, we compute two embeddings: one for the wildtype (normal) sequence and one for the mutant. The difference — the **delta** — captures how the mutation shifts the model's internal representation of the protein.

---

## The Disease Mechanism Classification Problem

When a missense variant causes disease, it does so through one of three main mechanisms:

- **Gain-of-function (GOF)** — the mutant protein does too much. It may be constitutively active, resistant to normal inhibition, or acquire a new activity. GOF variants often act dominantly: a single mutant copy is enough to cause disease.
- **Dominant negative (DN)** — the mutant protein actively interferes with the normal copy. It may form a dysfunctional complex with the wildtype protein and poison it. DN variants also act dominantly.
- **Loss-of-function (LOF)** — the mutant protein does too little or nothing. It may be unstable, misfolded, or unable to bind its substrate. LOF variants often require both copies to be disrupted (recessive), though haploinsufficiency — where losing one copy is enough — is also common.

Knowing the mechanism matters clinically. A GOF variant might respond to an inhibitor; a LOF variant might not. A DN variant can be dominant even when the gene is not haploinsufficient. Current variant interpretation tools (e.g. AlphaMissense, CADD) predict **pathogenicity** — whether a variant is damaging — but not **mechanism** — how it acts. These are different questions.

---

## The Dataset

We use a merged dataset combining two sources:

**Gerasimavicius et al. 2022** (*Nature Communications*) — a curated set of missense variants across 948 disease genes, each labelled with a gene-level mechanism class (GOF, DN, or one of two LOF subtypes: haploinsufficient HI and autosomal recessive AR). Labels are derived from clinical genetics literature and ClinVar curation. This is the primary source of mechanism labels, contributing 10,138 variants to the filtered working set below.

**Gene2Phenotype (G2P)** — a database of gene-disease associations maintained by clinical genetics groups. We use the `molecular mechanism` field to assign mechanism labels to an additional set of genes not covered by Gerasimavicius.

After merging and filtering to variants with available UniProt sequences, the working dataset contains **17,826 variants across 1,935 genes** spanning **1,134 Pfam protein families**.

The remaining 7,688 variants come from Gene2Phenotype.

Class distribution: LOF = 13,594 (76.3%) / GOF = 2,682 (15.0%) / DN = 1,550 (8.7%). LOF is the dominant class, reflecting the general prevalence of loss-of-function disease genetics. The imbalance matters for interpretation: with LOF at 76%, a classifier that predicts LOF for everything already scores well on accuracy, which is why macro-F1 is used throughout and why the chance floor is measured rather than assumed.

---

## The Core Question

**Does ESM-2 encode disease mechanism?**

Specifically: does the shift in ESM-2's representation of a protein caused by a missense variant (the delta embedding) carry information about whether that variant acts through GOF, DN, or LOF?

This is a non-trivial question. ESM-2 was not trained on mechanism labels — it was trained to predict masked amino acids. Any mechanism signal would have to emerge implicitly from sequence patterns that correlate with mechanism. There are reasons to think this might work (GOF variants may concentrate in activation domains; DN variants may cluster at interfaces), and reasons to think it might not (mechanism labels are gene-level, not variant-level, which is a fundamental mismatch with what ESM-2 encodes — every variant in a gene carries the same label regardless of where it falls).

A further complication is that apparent mechanism signal can be homology leakage rather than mechanism knowledge. Because related genes share both family membership and mechanism, a classifier evaluated with gene-split CV can score by recognising the family of a held-out gene from its relatives in training. Quantifying that share is a result in its own right, reported separately in [`report_leakage_fraction.md`](report_leakage_fraction.md).

We evaluate this using **linear probes** (logistic regression) and **nonlinear probes** (MLP) under two cross-validation schemes:

- **Gene-split CV** — standard holdout: test genes are not seen in training
- **Family-split CV** — strict holdout: entire protein families are withheld, so the classifier cannot use family identity as a proxy for mechanism

The family-split is critical. Protein families tend to share mechanism labels — 83% of genes carry their family's majority label in this dataset — so a classifier could appear to work by recognising families rather than mechanism. Only family-split CV can distinguish these.

**Metrics:** macro-F1 (average F1 across all three classes, penalising classifiers that ignore rare classes) and one-vs-rest AUROC per class.

---

## Baselines

To interpret the delta embedding results, we compare against several baselines:

- **WT-only embedding** — the wildtype protein's ESM-2 embedding, with no mutation information. If this outperforms the delta, the mutation itself is not contributing signal — only the protein's identity matters.
- **Mutant-only embedding** — symmetric to WT-only; tests whether the mutant sequence alone carries mechanism signal.
- **One-hot amino acid identity** — encodes only which amino acid changed (e.g. Ala → Val), ignoring all sequence context. A strong result here would mean mechanism is determined by the biochemical properties of the substitution alone.
- **FoldX ΔΔG** — a physics-based estimate of how much the mutation destabilises the protein. Captures thermodynamic stability effects but nothing about protein function or context.
- **AlphaMissense** — a state-of-the-art deep learning pathogenicity predictor. Predicts whether a variant is harmful, not how it acts. Included to test whether pathogenicity and mechanism are correlated.

These baselines span a range from purely biochemical (one-hot, FoldX) to sequence-context-aware (ESM-2 WT) to learned pathogenicity (AlphaMissense). A delta embedding that outperforms all of them under family-split CV would constitute genuine mechanism signal.

---

## Where the results are

This document sets up the question, the data, and the evaluation; it deliberately reports no
results. The findings are in the sibling reports: [`report_classifier.md`](report_classifier.md)
for the main mechanism result, [`report_leakage_fraction.md`](report_leakage_fraction.md) for
how much of the gene-split score is homology leakage,
[`report_protein_family.md`](report_protein_family.md) for whether the embeddings cluster by
family, and [`report_control.md`](report_control.md) for the pathogenicity positive control.

---

## Provenance

Dataset counts (17,826 variants, 1,935 genes, LOF 13,594 / GOF 2,682 / DN 1,550, and the
10,138 Gerasimavicius / 7,688 Gene2Phenotype split) are from `data/valid_variants.json`. The
1,134 Pfam families and the 83% within-family label agreement are from
[`results/run6/family_clustering.json`](../../results/run6/family_clustering.json). Model:
ESM-2 `esm2_t33_650M_UR50D`. Full run log: [`RUN_PROGRESS.md`](../../RUN_PROGRESS.md), Run 6.

**Statistical limitations.** The evaluation described here uses 5-seed spreads as error bars,
which measure cross-validation fold jitter on a fixed dataset rather than sampling uncertainty.
Dependency-aware confidence intervals and permutation tests are planned; see
[`STATS_PLAN.md`](STATS_PLAN.md).
