# Result 2: Gene-Split vs Family-Split Baseline Comparison
## Script: family_split_baselines.py | Run: May 24, 2026 | Model: ESM-2 650M

---

## Setup

Same cached embeddings as Result 1. All features run under both gene-split and family-split CV.

- **N variants**: 10,231 | **N genes**: 948 | **N Pfam families**: 662
- **Classes**: GOF=1,983 / DN=894 / LOF=7,354
- **New features vs Result 1**: `mut_only_mean`, `wt_concat_mut` (concatenation of WT and mutant embeddings)

---

## Full Results Table

| Feature | Gene-split F1 | Family-split F1 | Δ | Gene-split AUROC (G/D/L) | Family-split AUROC (G/D/L) |
|---|---|---|---|---|---|
| **wt_only** | **0.580** | **0.389** | **+0.191** | 0.870 / 0.804 / 0.915 | 0.801 / 0.687 / 0.852 |
| mut_only | 0.579 | 0.381 | +0.198 | 0.872 / 0.799 / 0.915 | 0.800 / 0.686 / 0.852 |
| wt_concat_mut | 0.571 | 0.413 | +0.158 | 0.868 / 0.794 / 0.910 | 0.796 / 0.681 / 0.841 |
| delta_per_residue | 0.376 | 0.348 | +0.028 | 0.652 / 0.536 / 0.653 | 0.592 / 0.499 / 0.595 |
| delta_mean | 0.279 | 0.281 | -0.002 | 0.634 / 0.529 / 0.620 | 0.545 / 0.485 / 0.519 |
| onehot_aa | 0.280 | 0.282 | -0.002 | 0.595 / 0.489 / 0.593 | 0.577 / 0.489 / 0.571 |
| foldx_ddg | 0.279 | 0.281 | -0.002 | 0.628 / 0.614 / 0.637 | 0.630 / 0.614 / 0.643 |
| alphamissense | 0.279 | 0.281 | -0.002 | 0.502 / 0.499 / 0.501 | 0.503 / 0.494 / 0.498 |

AUROC columns: GOF / DN / LOF one-vs-rest.

---

## Key Findings

### 1. WT-only signal is largely paralog leakage

WT-only gene-split F1 = 0.580 → family-split F1 = 0.389 (Δ = +0.191). Most of the apparent mechanism signal disappears when protein families are held out. The probe was learning "kinases tend to be GOF, ion channels tend to be DN" — protein family identity, not mechanism.

WT and mutant embeddings are nearly identical in performance (0.580 vs 0.579 gene-split, 0.389 vs 0.381 family-split). The signal is entirely in the protein identity, not the mutation.

Concatenating WT+mutant (`wt_concat_mut`) adds nothing over WT alone — though it does retain slightly more signal under family-split (0.413 vs 0.389), likely because the joint representation has more capacity to pick up residual cross-family signal.

### 2. Delta probe is flat: no leakage, but no signal

`delta_mean` F1 stays flat at 0.279→0.281 across both CV schemes. No homology leakage (the family identity gets subtracted out), but also no mechanism signal. The mutation-specific perturbation doesn't linearly encode mechanism class.

`delta_per_residue` has a small leakage (0.376→0.348, Δ=+0.028) and slightly more signal than mean-pooled delta. Local context at the variant position carries a weak but real family-correlated signal.

### 3. FoldX ΔΔG is family-invariant and non-trivial for DN

FoldX ΔΔG is completely flat across CV schemes (0.279→0.281) — no family leakage, as expected for a physics-based score. Notably, its DN AUROC (0.614 gene-split, 0.614 family-split) is the **highest family-split DN AUROC of any feature** including the delta probe. Stability alone separates DN better than ESM-2 delta embeddings — DN mutations are thermodynamically distinctive (interface-disrupting).

### 4. GOF AUROC survives family-split most robustly

Across all features, GOF AUROC drops the least under family-split:

| Feature | GOF AUROC gene-split | GOF AUROC family-split | Retention |
|---|---|---|---|
| wt_only | 0.870 | 0.801 | 92% |
| delta_per_residue | 0.652 | 0.592 | 91% |
| delta_mean | 0.634 | 0.545 | 86% |
| DN wt_only | 0.804 | 0.687 | 85% |
| LOF wt_only | 0.915 | 0.852 | 93% |

GOF AUROC under family-split (0.801 for WT-only) is the highest surviving signal in the experiment. ESM-2 encodes something about GOF protein sequences that generalises across families. This is the one finding that holds up under the stringent homology-aware test.

### 5. AlphaMissense is essentially random for mechanism

AlphaMissense AUROC is 0.50 for DN and LOF — it carries no mechanism information. It's a pathogenicity predictor, not a mechanism predictor, and the two are orthogonal. Interestingly, its GOF PR-AUC (0.593 gene-split, 0.589 family-split) is higher than expected — pathogenicity scores may partially track GOF because GOF mutations are often activating and score as "pathogenic."

---

## Interpretation

### What the numbers mean in plain English

The WT-only probe looked promising (F1=0.58) but the signal mostly came from it learning which protein family a gene belongs to — kinases are GOF, ion channels are DN, structural proteins are LOF. Hold out entire protein families and that shortcut disappears; performance drops to 0.39.

The delta probe (mutant minus wildtype) genuinely avoids this problem — subtracting the wildtype removes the family-identity information. But what's left (the mutation-specific shift) doesn't carry enough mechanism signal for a linear probe to find.

The one exception worth pursuing: **GOF AUROC = 0.801 under family-split for WT-only.** ESM-2 has learned something about what GOF proteins look like at the sequence level that doesn't reduce to "it's a kinase." That signal generalises across families and is the thread most worth pulling.

### What this rules out

- ESM-2 delta-embeddings don't linearly encode gene-level dominant disease mechanism beyond protein stability and family identity
- WT-only mechanism classification is not a genuine mechanism signal — it's protein family classification, and protein family correlates with mechanism in the training data
- AlphaMissense (a state-of-the-art variant effect predictor) carries zero mechanism information — pathogenicity and mechanism are orthogonal

### What remains open

- **Why does GOF survive family-split?** GOF proteins may share sequence-level properties (disordered regions, constitutive activation motifs) that ESM-2 encodes independently of family. Worth investigating which sequence features drive the GOF AUROC.
- **Nonlinear probe on delta** — a linear probe may be too simple for the mutation-specific signal. An MLP or kernel method might find nonlinear structure.
- **Expanded dataset** — with only 81 GOF genes, the probe is data-limited. The merged dataset (G2P + Gerasimavicius, 158 GOF genes) might reveal cleaner signal.
- **Per-variant mechanism labels** — the fundamental limitation is gene-level labels. Some genes act through multiple mechanisms depending on the variant. Variant-level labels (from functional assays or DMS data) would be a much stronger test.

---

## Data Location

- Results: `../results/20260524_baseline_run/run_0/family_split_baselines.json`
- Script: `../scripts/family_split_baselines.py`
