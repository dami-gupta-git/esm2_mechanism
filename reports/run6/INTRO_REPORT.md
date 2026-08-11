# Does ESM-2 Encode Disease Mechanism? A Homology-Controlled Probe

**Short version:** A protein language model's mutation embedding predicts *whether* a variant is
damaging at AUROC ~0.90, but classifies *how* it's damaging (gain- vs loss-of-function vs
dominant-negative) at the chance floor once you control for protein-family homology. Most of the
apparent mechanism signal under standard evaluation is the model recognising the protein family,
not reading the mutation. And the part it does predict well — pathogenicity — turns out to be a
worse re-encoding of the model's own conservation estimate.

---

## What is ESM-2, and what is the "delta"?

ESM-2 is a protein language model trained on hundreds of millions of sequences. Like a text
language model, it learns to predict masked amino acids from context, and in doing so builds
internal representations that capture evolutionary conservation, structural propensity, and
functional-site identity. Its output is an **embedding**: 1,280 numbers per residue.

For a given variant I compute two embeddings — one for the wildtype (normal) protein, one for the
mutant — and take the difference. That difference, the **delta**, is meant to capture how the
mutation shifts the model's internal picture of the protein. The whole study is about what that
delta does and does not contain.

---

## The question

When a missense variant causes disease, it usually acts through one of three mechanisms:

- **Gain-of-function (GOF)** — the protein does too much (new or constitutive activity).
- **Dominant-negative (DN)** — the mutant poisons the normal copy, e.g. by forming a broken complex.
- **Loss-of-function (LOF)** — the protein does too little (unstable, misfolded, can't bind).

Mechanism matters clinically: a GOF variant might respond to an inhibitor where a LOF variant
won't. Current tools (AlphaMissense, CADD) predict **pathogenicity** — whether a variant is
damaging — not **mechanism** — how it acts. These are different questions, and the second is the
one I tested: does the ESM-2 delta carry mechanism?

---

## The data

A merged set of **17,826 missense variants across 1,935 genes**, spanning 1,134 Pfam protein
families. Mechanism labels come from Gerasimavicius et al. 2022 (*Nat Commun*; 10,138 variants)
and Gene2Phenotype (7,688 variants). Class distribution: LOF 76.3% / GOF 15.0% / DN 8.7%.

With LOF at 76%, a classifier that predicts LOF for everything already scores well on accuracy —
so I report **macro-F1** (equal weight to each class) throughout, and **measure** the chance floor
(macro-F1 = 0.288) with a majority-class dummy classifier under the same cross-validation rather
than assuming it.

A second, subtler trap: labels are **gene-level**. Every variant in a gene shares one mechanism
label regardless of where it falls. Related genes tend to share both family membership and
mechanism — in this dataset, 83% of genes carry their family's majority label. So a classifier can
look like it understands mechanism when it is really recognising the family. Controlling for that
is the core of the evaluation.

---

## Evaluation

Two cross-validation schemes, and the gap between them is the whole point:

- **Gene-split** — hold out whole genes. Standard practice, but related genes from the same family
  can sit in both train and test, so the model can recognise a held-out gene from its relatives.
- **Family-split** — hold out whole Pfam families. Now family identity is no longer available as a
  shortcut, so only genuine, transferable signal survives.

If a feature scores well on gene-split and collapses on family-split, the difference was family
recognition, not mechanism.

---

## Result 1 — the mutation delta carries no mechanism signal

Under family-split, the delta classifies mechanism at the floor.

| Feature (family-split) | Macro-F1 | Reading |
|---|---:|---|
| delta_mean (the mutation shift) | 0.288 | at the chance floor |
| wt_only (protein embedding, no mutation info) | 0.442 | above floor — but see below |
| AlphaMissense (pathogenicity score) | 0.290 | at the floor |
| *chance floor* | *0.288* | |

The mutation-induced delta separates GOF/DN/LOF at chance. A nonlinear MLP lifts it only slightly
(gene-split 0.399, family-split 0.380), and that lift is residual family structure the subtraction
didn't fully remove, not mechanism. AlphaMissense — a state-of-the-art pathogenicity predictor —
sits at the floor too, confirming that mechanism is a different question from pathogenicity.

The one feature that scores above the floor is the *wildtype protein embedding*, which contains no
mutation information at all. What above-chance mechanism prediction exists comes from recognising
the protein family, not from reading the variant.

---

## Result 2 — ~40% of the protein-embedding signal is homology leakage

The wildtype embedding scores 0.545 on gene-split but 0.442 on family-split. Expressed as a
fraction of its *above-chance* signal:

> leakage fraction = (gene-split − family-split) / (gene-split − chance)
> = (0.545 − 0.442) / (0.545 − 0.288) ≈ **40%**

So about 40% of what the protein embedding appears to "know" about mechanism is family recognition
that evaporates once whole families are held out. The mutation-only features (delta, one-hot,
FoldX, AlphaMissense) sit at the floor on both splits, so they have no signal for family
recognition to inflate — their leakage is undefined, which is the correct result: a feature with
no signal cannot leak.

This is measurable directly from the data: family is linearly recoverable from the embedding at
61% accuracy (vs a 4.4% baseline), and family predicts mechanism for 83% of genes. Recognise the
family, guess its usual mechanism, and you score well without ever using the mutation.

---

## Control — the pipeline works, so the null is real

A null result is only interpretable if the same pipeline can recover signal that is known to
exist. So I ran the identical embeddings, features, probes, and cross-validation on a task with a
known answer: ClinVar pathogenic vs benign.

| Task (delta_mean, MLP) | Gene-split | Family-split | Drop |
|---|---:|---:|---:|
| Pathogenicity (AUROC) | 0.897 | 0.894 | 0.003 |
| Mechanism (macro-F1) | ~0.40 | ~0.38 | (near floor) |

The same delta that sits at the floor for mechanism predicts pathogenicity at **AUROC ~0.90**, and
barely moves under family-split (drop 0.003) — so that signal is genuine per-variant biochemistry,
not homology leakage. The pipeline recovers known signal cleanly. The mechanism null is a real
property of the representation, not a broken setup.

---

## Result 3 — the pathogenicity signal is just conservation

Having established that the delta predicts pathogenicity well, I asked what that signal is. The answer: the model's own conservation estimate.

| Feature (pathogenicity, family-split AUROC) | Value |
|---|---:|
| Conservation alone (the model's masked-marginal likelihood, 1 number) | **0.891** |
| The full 1,280-dimension embedding delta | 0.859 |
| Conservation + the full delta | 0.893 |

A single number the model gives you for free — its own "how surprised am I by this amino acid
here" score — predicts pathogenicity **better** than the entire 1,280-dimension delta (0.891 vs
0.859). Adding all 1,280 embedding dimensions on top of that one number improves it by **+0.002**.
The fitted embedding axis correlates 0.74 with the conservation score.

For this task, the full 1,280-dimension delta is matched by a single number read directly from the model's output.

---

## What this is, and what it isn't

- **Not** a claim that ESM-2 is useless — it predicts pathogenicity at ~0.90, and separately
  encodes protein stability that a linear probe can't fully see.
- **Not** a claim that mechanism is unlearnable from sequence — only that this frozen delta,
  evaluated this way, doesn't reveal it, while it does reveal pathogenicity.
- **The conservation result is task-specific.** It says the *mean-pooled delta* adds nothing over
  the model's likelihood *for pathogenicity*. It is not a universal statement that embeddings are
  always just conservation.
- **Labels are gene-level**, a fundamental mismatch with the variant-level delta. Variant-level
  functional labels would change the analysis qualitatively and are the most informative next step.

---

## Reproducibility

Model: ESM-2 `esm2_t33_650M_UR50D`. All results are 5-seed means under gene-split and family-split
cross-validation; linear (logistic regression) and nonlinear (MLP) probes. Pathogenicity control:
37,218 balanced ClinVar missense variants. The chance floor is measured, not assumed.

Code and per-result files: [repository link]. Mechanism labels: Gerasimavicius et al. 2022 (OSF
10.17605/OSF.IO/H62FQ) and Gene2Phenotype.

*Note on error bars: the reported spreads are 5-seed cross-validation jitter on a fixed dataset,
not sampling uncertainty. Dependency-aware confidence intervals (cluster bootstrap over genes) and
label-permutation tests are the planned next step before any formal write-up.*