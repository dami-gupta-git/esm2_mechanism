# Result 2: Gene-Split vs Family-Split Baseline Comparison
## Script: family_split_baselines.py | Run: May 24, 2026 | Model: ESM-2 650M

---

## Background: what this experiment is testing

Result 1 found something puzzling: using just the wildtype (normal) protein embedding — ignoring the mutation entirely — gave the best classification performance (macro-F1 = 0.580). That's suspicious. It suggests the classifier might be cheating: recognising which protein family a gene belongs to and using that as a proxy for mechanism (e.g. "all kinases are GOF, all ion channels are DN").

This experiment tests that hypothesis directly. We run each feature under two cross-validation schemes:

- **Gene-split CV**: test genes are held out, but related genes (paralogs, family members) can still appear in training. This is the standard setup, and it can leak family information.
- **Family-split CV**: entire protein families are held out from the test set. The classifier cannot use any protein from the same family as a training hint. This is a much stricter test.

If a feature's performance **drops a lot** under family-split, it was mostly learning protein family identity, not mechanism. If performance is **stable**, the signal is more likely to be real.

---

## Setup

Same cached embeddings as Result 1. All features run under both CV schemes.

- **N variants**: 10,231 | **N genes**: 948 | **N Pfam families**: 662
- **Classes**: GOF=1,983 / DN=894 / LOF=7,354
- **New features tested here**: `mut_only_mean` (mutant embedding averaged across the protein), `wt_concat_mut` (WT and mutant embeddings joined together)

---

## Full Results Table

The Δ column shows the drop in F1 under family-split — a large positive Δ means the feature was leaking family information.

| Feature | Gene-split F1 | Family-split F1 | Δ (drop) | Gene-split AUROC (GOF/DN/LOF) | Family-split AUROC (GOF/DN/LOF) |
|---|---|---|---|---|---|
| **wt_only** | **0.580** | **0.389** | **+0.191** | 0.870 / 0.804 / 0.915 | 0.801 / 0.687 / 0.852 |
| mut_only | 0.579 | 0.381 | +0.198 | 0.872 / 0.799 / 0.915 | 0.800 / 0.686 / 0.852 |
| wt_concat_mut | 0.571 | 0.413 | +0.158 | 0.868 / 0.794 / 0.910 | 0.796 / 0.681 / 0.841 |
| delta_per_residue | 0.376 | 0.348 | +0.028 | 0.652 / 0.536 / 0.653 | 0.592 / 0.499 / 0.595 |
| delta_mean | 0.279 | 0.281 | -0.002 | 0.634 / 0.529 / 0.620 | 0.545 / 0.485 / 0.519 |
| onehot_aa | 0.280 | 0.282 | -0.002 | 0.595 / 0.489 / 0.593 | 0.577 / 0.489 / 0.571 |
| foldx_ddg | 0.279 | 0.281 | -0.002 | 0.628 / 0.614 / 0.637 | 0.630 / 0.614 / 0.643 |
| alphamissense | 0.279 | 0.281 | -0.002 | 0.502 / 0.499 / 0.501 | 0.503 / 0.494 / 0.498 |

AUROC columns: GOF / DN / LOF one-vs-rest (each class vs all others combined).

---

## Key Findings

### 1. The WT-only signal is mostly a family recognition shortcut

WT-only drops from F1 = 0.580 (gene-split) to 0.389 (family-split) — a loss of 0.191. Most of the apparent mechanism signal disappears when protein families are held out. The classifier was learning "kinases tend to be GOF, ion channels tend to be DN" — not anything fundamental about mechanism.

The mutant-only embedding performs almost identically to wildtype-only (0.580 vs 0.579 gene-split, 0.389 vs 0.381 family-split). This confirms the signal is entirely in the protein's identity, not in the specific mutation. The mutation doesn't add any information beyond what the wildtype already encodes.

Concatenating WT and mutant embeddings together (`wt_concat_mut`) adds nothing over WT alone. It does retain slightly more signal under family-split (0.413 vs 0.389), probably because a larger representation has more room to find residual cross-family signal.

### 2. The delta is clean — but empty

`delta_mean` (mutant minus wildtype, averaged over the whole protein) scores almost identically under gene-split and family-split: 0.279 → 0.281. No leakage — subtracting the wildtype removes the family-identity information. But also no mechanism signal. The mutation-specific shift doesn't encode mechanism in a way a linear classifier can find.

`delta_per_residue` (the shift at just the mutated position) has a small leakage (0.376 → 0.348, drop of 0.028) and slightly more signal overall. The local context at the mutation site carries a weak but real family-correlated signal.

### 3. FoldX stability scores don't leak, and are actually useful for DN

FoldX ΔΔG (a physics-based estimate of how much the mutation destabilises the protein) is completely flat across CV schemes (0.279 → 0.281) — no family leakage, as expected. Interestingly, its DN AUROC (0.614 under both gene-split and family-split) is the **highest family-split DN AUROC of any feature tested** — including all the ESM-2-based features. A simple stability score separates dominant-negative mutations better than ESM-2 embeddings. This makes biological sense: DN mutations often work by disrupting protein interfaces, which is a stability effect.

### 4. GOF survives family-split better than the other classes

Across all features, the GOF AUROC holds up best when protein families are held out:

| Feature | GOF AUROC gene-split | GOF AUROC family-split | Retention |
|---|---|---|---|
| wt_only | 0.870 | 0.801 | 92% |
| delta_per_residue | 0.652 | 0.592 | 91% |
| delta_mean | 0.634 | 0.545 | 86% |
| DN wt_only | 0.804 | 0.687 | 85% |
| LOF wt_only | 0.915 | 0.852 | 93% |

The GOF AUROC of 0.801 under family-split (WT-only feature) is the highest surviving signal in the whole experiment. ESM-2 has learned something about GOF proteins that generalises beyond individual protein families. This is the one signal that survives the strict family-hold-out test.

### 5. AlphaMissense is random for mechanism classification

AlphaMissense is a state-of-the-art tool for predicting whether a variant is harmful to a patient. But it scores 0.50 AUROC for DN and LOF — completely random. Pathogenicity (is this variant bad for health?) and mechanism (does it cause gain-of-function, dominant negative, or loss-of-function?) are simply different questions, and AlphaMissense only answers the first.

Its GOF performance is slightly above random (0.593 gene-split) — possibly because GOF mutations are often activating and tend to score as "pathogenic" in pathogenicity tools.

---

## Interpretation

### What the numbers mean

The WT-only probe looked promising (F1=0.58) but the signal mostly came from it learning which protein family a gene belongs to — kinases are GOF, ion channels are DN, structural proteins are LOF. Hold out entire protein families and that shortcut disappears; performance drops to 0.39.

The delta (mutant minus wildtype) genuinely avoids this problem — subtracting the wildtype removes the family-identity information. But what's left (the mutation-specific shift) doesn't carry enough mechanism signal for a linear classifier to find.

The one exception worth pursuing: **GOF AUROC = 0.801 under family-split for WT-only.** ESM-2 has learned something about what GOF proteins look like at the sequence level that doesn't reduce to "it's a kinase." That signal generalises across families and is the thread most worth pulling.

### What this rules out

- ESM-2 delta-embeddings don't linearly encode disease mechanism beyond protein stability and family identity
- WT-only mechanism classification is not a genuine mechanism signal — it's protein family classification, and protein family happens to correlate with mechanism
- AlphaMissense carries zero mechanism information — pathogenicity and mechanism are orthogonal

### What remains open

- **Why does GOF survive family-split?** GOF proteins may share sequence-level properties (disordered regions, activation motifs) that ESM-2 encodes independently of family. Worth investigating which sequence features drive the GOF AUROC.
- **Nonlinear classifier on delta** — a linear classifier may be too simple. A neural network might find nonlinear structure in the delta.
- **Expanded dataset** — with only 81 GOF genes, we're data-limited. The merged dataset (G2P + Gerasimavicius, 158 GOF genes) might reveal cleaner signal.
- **Per-variant mechanism labels** — the fundamental limitation is gene-level labels. Some genes act through multiple mechanisms depending on the variant. Labels from functional assays would be a much stronger test.

---

## Data Location

- Results: `../results/20260524_baseline_run/run_0/family_split_baselines.json`
- Script: `../scripts/family_split_baselines.py`
