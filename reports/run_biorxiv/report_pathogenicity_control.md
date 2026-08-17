# Can the same ESM-2 delta predict pathogenicity?

**run_biorxiv · 2026-08-14** · ESM-2 `esm2_t33_650M_UR50D` · 37,258 variants (18,857
pathogenic / 18,401 benign) · 1,925 genes · 1,141 protein families · 5 seeds · 1,000
bootstrap resamples. Confirmatory claims and decision rules:
[`PREREGISTRATION_run_biorxiv.md`](../../biorxiv/PREREGISTRATION_run_biorxiv.md).

---

## The question

The mechanism report found that ESM-2 delta embeddings classify mechanism (GOF/DN/LOF) at
the chance floor. A null result is only interpretable if the pipeline can recover signal that
is known to exist. This experiment therefore runs the identical embedding extraction,
features, probes, and cross-validation on a task with an established answer: distinguishing
ClinVar pathogenic from benign variants. Published work predicts pathogenicity from ESM-2
at AUROC 0.88–0.94 (e.g. Brandes et al. 2023).

- If pathogenicity prediction succeeds, the pipeline is sound and the mechanism null is a
  real absence of mechanism signal.
- If it also fails, the pipeline is broken and the mechanism null is uninterpretable.

---

## Setup

- **Dataset:** 37,258 balanced ClinVar missense variants (18,857 pathogenic, 18,401 benign)
  across 1,925 genes grouped into 1,141 Pfam families. Variants drawn from the same disease
  genes as the mechanism experiment, balanced at up to 20 per gene per class. Fetched in an
  earlier pipeline step by a separate script from the mechanism experiment's pathogenic-only
  pull ([`report_mechanism.md`](report_mechanism.md)), since this experiment also needed
  benign variants; this experiment's own step only embeds that fetched set and runs the probe.
- **Cross-validation:** 5-fold, gene-level, run at 5 random seeds. Family-split holds out
  entire Pfam families.
- **Metric:** AUROC (probability the probe ranks a random pathogenic variant above a random
  benign one). Chance floor = 0.50 regardless of class balance.
- **Confidence intervals:** 95% cluster bootstrap, 1,000 resamples, from seed 0. Gene-split
  CIs resample genes (1,925 clusters); family-split CIs resample families (1,141 clusters).
- **Probes:** logistic regression (linear) and MLP (nonlinear, 256 hidden units). Both are
  uncalibrated and measure discrimination only, not risk.

---

## Glossary

**Features (rows):**

| Name | Dimensionality | Description |
|---|---|---|
| `delta_mean` | 1280-d | Mutant minus wildtype embedding (the mutation’s effect) |
| `wt_only` | 1280-d | ESM-2 embedding of the original protein, mean-pooled over residues |

**Probes (columns):**

| Probe | What it is |
|---|---|
| logreg | Linear probe (logistic regression) |
| mlp | Nonlinear probe (MLP, 256 hidden units) |

---

## Table 1. Pathogenicity AUROC

Each row is a feature–probe combination; each column is a cross-validation scheme. The
table shows how well each feature predicts whether a variant is pathogenic or benign. The
leakage drop is the difference between gene-split and family-split: a drop near zero means
the score survives without family hints; a large drop means the score was inflated by family
recognition.

**Note.** Chance floor: AUROC = 0.50. All values and 95% CIs are from seed 0 (cluster
bootstrap, 1,000 resamples). Pass criterion for claim 2C: `delta_mean` MLP family-split
AUROC must exceed 0.85, with the CI excluding 0.85.

| Feature | Probe | Gene-split | Family-split | Leakage drop |
|---|---|---:|---:|---:|
| delta_mean | mlp | 0.898 [0.893, 0.902] | 0.894 [0.888, 0.899] | 0.004 |
| delta_mean | logreg | 0.861 [0.855, 0.867] | 0.862 [0.856, 0.868] | −0.001 |
| wt_only | mlp | 0.615 [0.604, 0.625] | 0.597 [0.583, 0.609] | 0.018 |
| wt_only | logreg | 0.581 [0.569, 0.593] | 0.551 [0.539, 0.563] | 0.030 |
| *no-skill baseline* | — | *0.500* | *0.500* | — |

---

## Pre-registered claims tested in this experiment

The rules for deciding these verdicts were written down before the results came in
([`PREREGISTRATION_run_biorxiv.md`](../../biorxiv/PREREGISTRATION_run_biorxiv.md)), so the
verdict could not be adjusted afterward.

### 2C. The delta predicts pathogenicity (positive control)

The question is whether the mutation-induced embedding shift carries enough signal to
separate pathogenic from benign variants, confirming that the pipeline works.

The pass bar is 0.85 AUROC for the `delta_mean` MLP under family-split. The MLP scored
0.894, and even at the low end of its confidence interval (0.888) the score stays well above
0.85.

- **`delta_mean` MLP family-split AUROC:** 0.894 [0.888, 0.899].
- ✅ **CI lower bound (0.888) > 0.85: affirmed.**

---

## Reading the tables

**1. The delta predicts pathogenicity well.**  
`delta_mean` MLP reaches 0.898 on gene-split and 0.894 on family-split. The mutation-induced
embedding shift carries strong, largely linear information about whether a variant is
damaging. The linear probe also clears 0.85 on both splits (0.861 / 0.862), so the signal is
mostly linear.

**2. The signal is family-split-stable.**  
`delta_mean` MLP drops from 0.898 to 0.894 under family-split (gap of 0.004). The logistic
regression actually rises slightly (−0.001). The prediction therefore reads the mutation
itself, not the protein family. This contrasts with the mechanism result, where the wildtype
feature lost ~0.06 macro-F1 under family-split
([`report_mechanism.md`](report_mechanism.md)).

**3. The wildtype embedding cannot predict pathogenicity.**  
`wt_only` reaches only 0.615 (MLP) and 0.581 (logreg) on gene-split, both well below the
delta. The unmutated protein sequence alone does not indicate which specific mutation would
be damaging. Its leakage drop is also larger (0.018–0.030), suggesting that what little
signal it has partly reflects family membership.

**4. Nonlinearity adds a modest, real margin.**  
The MLP exceeds logistic regression by 0.037 on gene-split (0.898 vs 0.861) and 0.032 on
family-split (0.894 vs 0.862). Both gaps exceed the CI width, so the nonlinear component is
real but small.

---

## The dissociation

The same ESM-2 delta embedding and the same pipeline, evaluated on two tasks:

| Task | Feature | Best score | Family-split stable? |
|---|---|---|---|
| Pathogenicity (this report) | delta_mean MLP | AUROC 0.894 | Yes (drop 0.004) |
| Mechanism ([`report_mechanism.md`](report_mechanism.md)) | delta_mean linear | macro-F1 0.288 (at floor) | — |

ESM-2 delta embeddings predict *whether* a mutation is damaging at AUROC ~0.90 but do not
classify *how* it acts above chance. The positive control shows that the mechanism result
cannot be explained by a general failure of the embedding/probe pipeline.

---

## What this is and is not

- **Not a new finding that PLMs predict pathogenicity but not mechanism.** This is stated
  qualitatively in prior work (e.g. AlphaMissense, LoGoFunc, PreMode). The contribution here
  is the controlled side-by-side measurement on one dataset, model, and pipeline.
- **Not a claim that mechanism is unlearnable from sequence.** Only that ESM-2’s delta
  representation, evaluated this way, does not reveal it, while it does reveal pathogenicity.
- **Pass criterion met:** `delta_mean` MLP family-split AUROC 0.894 [0.888, 0.899],
  CI excludes 0.85. ✅

---

## Provenance

| Result | Source file |
|---|---|
| Seed 0 AUROCs and CIs | [`pathogenicity_control_seed0.json`](../../results/run_biorxiv/pathogenicity_control_seed0.json) |
| Seed 1–4 AUROCs | [`pathogenicity_control_seed{1..4}.json`](../../results/run_biorxiv/) |
| Aggregate (5-seed means) | [`pathogenicity_control.json`](../../results/run_biorxiv/pathogenicity_control.json) |

Computed by `experiments/pathogenicity/pathogenicity_control.py`. ClinVar `variant_summary.txt.gz`
filtered to GRCh38 missense, balanced pathogenic/benign (up to 20 per gene per class),
38,797 candidates, 37,258 embedded. ESM-2 650M mean-pooled embeddings on RunPod H100.
Probes: 5 seeds, logreg + MLP × {delta_mean, wt_only} × {gene-split, family-split}.
