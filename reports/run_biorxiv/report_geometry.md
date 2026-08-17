# What is the shape of ESM-2's pathogenicity signal?

**run_biorxiv · 2026-08-17** · ESM-2 `esm2_t33_650M_UR50D` · 37,258 canonical ClinVar
variants (18,857 pathogenic / 18,401 benign) · 1,925 genes · 1,141 protein families · 5
seeds · 1,000 bootstrap resamples. Confirmatory claims and decision rules:
[`PREREGISTRATION_run_biorxiv.md`](../../biorxiv/PREREGISTRATION_run_biorxiv.md).

---

## The question

The pathogenicity control ([`report_pathogenicity_control.md`](report_pathogenicity_control.md))
showed that ESM-2 delta embeddings separate pathogenic from benign variants at AUROC 0.894,
while the mechanism experiment ([`report_mechanism.md`](report_mechanism.md)) found mechanism
classification (GOF/DN/LOF) at the chance floor. This report asks what the pathogenicity
signal actually is: where in the delta it lives, whether it is shared across protein families
or specific to each one, what biological quantity it corresponds to, and whether the same
direction transfers to other tasks.

Four questions, answered by four probes and then a conservation decider:

- **Magnitude vs direction.** Is the signal in how far the embedding moves (magnitude), or
  which way it moves (direction)?
- **Geometry.** Is it one shared direction across protein families, or many family-specific
  ones?
- **Transfer.** Under one protocol, does a direction fit on one half of the data transfer to
  the other, and how does that compare between pathogenicity and mechanism?
- **Biochemistry.** How much of the direction is explained by context-free substitution
  chemistry (BLOSUM62 score, hydropathy, charge, and volume changes)?
- **Conservation.** Is the direction largely explained by conservation, meaning ESM-2's own
  sense of how expected each amino acid is at that position?

---

## Setup

- **Dataset:** 37,258 balanced ClinVar missense variants (18,857 pathogenic, 18,401 benign)
  across 1,925 genes grouped into 1,141 Pfam families. Same canonical set as
  [`report_pathogenicity_control.md`](report_pathogenicity_control.md), re-indexed to match
  the embedding row order.
- **Delta decomposition:** for each variant, the delta `d = mut_emb - wt_emb` (1,280
  dimensions) is split into magnitude (how large: `||d||`, one number) and direction (which
  way: `d/||d||`, 1,280-d unit vector).
- **Cross-validation:** 5-fold, family-split (entire Pfam families held out together). 5 seeds.
- **Probes:** logistic regression (linear) and MLP (256 hidden units, nonlinear). Both are
  uncalibrated and measure discrimination only, not risk.
- **Confidence intervals:** 95% cluster bootstrap, 1,000 resamples, resampling families (1,141
  clusters). Paired differences use a paired cluster bootstrap (one shared draw applied to both
  arms).
- **Mechanism comparison:** uses the merged variant set (1,931 genes, 1,144 families) from
  [`report_mechanism.md`](report_mechanism.md). Chance floor for mechanism macro-F1 = 0.290
  (measured majority-class baseline from
  [`naive_baseline.json`](../../results/run_biorxiv/naive_baseline.json)).

### Glossary

**AUROC** (area under the receiver operating characteristic curve) measures how well a probe
separates two classes. A score of 0.50 means no better than a coin flip; 1.0 means perfect
separation. For pathogenicity (two classes), the chance floor is 0.50.

**Macro-F1** is the average F1 score across classes, giving equal weight to each class
regardless of size. For three-class mechanism (GOF/DN/LOF), the measured chance floor is
0.290.

**Confidence interval (CI)** quantifies sampling uncertainty around an estimated score. Here,
95% CIs are obtained by repeatedly resampling protein families as clusters. A narrower
interval means the estimate is more stable; a wider one means it could shift more if the
analysis were repeated on different genes.

---

## Table 1. Magnitude vs direction: pathogenicity AUROC (family-split)

This table asks where the pathogenicity signal lives. If the signal were about how much the
embedding moves, magnitude would score high and direction would be weak. If it were about
which way the embedding moves, the reverse.

**Note:** Chance floor for AUROC = 0.50. All values and 95% CIs are from seed 0 (cluster
bootstrap, 1,000 resamples, 1,141 family clusters).

| Feature | Logreg AUROC | MLP AUROC |
|---|---:|---:|
| full delta | 0.866 [0.860, 0.872] | 0.910 [0.905, 0.915] |
| magnitude `\|\|d\|\|` | 0.672 [0.660, 0.683] | 0.671 [0.660, 0.683] |
| direction `d/\|\|d\|\|` | 0.873 [0.866, 0.878] | 0.916 [0.911, 0.920] |

**Verdict**   
The signal is overwhelmingly in the direction, not the magnitude. Direction-only
MLP AUROC (0.916) matches the full delta (0.910), while magnitude-only is stuck at 0.671.
The intuitive prior, that a more damaging mutation simply moves the embedding further, does
not hold. Essentially all of ESM-2's pathogenicity signal is in which way the representation
shifts, independent of how large the shift is.

Logreg and MLP agree on the same pattern, with the MLP extracting about 0.04 more from
direction-only, suggesting a modest nonlinear component in the directional signal.

## Table 2. Magnitude vs direction: mechanism macro-F1 (family-split)

The same decomposition applied to the mechanism task, to see whether splitting the delta
surfaces any hidden mechanism signal.

**Note:** Chance floor for macro-F1 = 0.290. All values and 95% CIs are from seed 0 (cluster
bootstrap, 1,000 resamples, 1,144 family clusters).

| Feature | Logreg macro-F1 | MLP macro-F1 |
|---|---:|---:|
| full delta | 0.409 [0.371, 0.439] | 0.429 [0.363, 0.483] |
| magnitude `\|\|d\|\|` | 0.311 [0.254, 0.363] | 0.334 [0.291, 0.367] |
| direction `d/\|\|d\|\|` | 0.406 [0.370, 0.431] | 0.422 [0.373, 0.461] |

**Verdict**    
Decomposing the delta does not surface hidden mechanism signal. Magnitude overlaps the chance floor (0.290). Full delta and direction-only sit only modestly above it, with wide intervals, and neither outperforms the other.  
The decomposition was run to test whether a useful mechanism signal might be masked when magnitude and direction are combined. It is not. No component rises far above chance, so the mechanism null from report_mechanism.md survives.  
The asymmetry with pathogenicity is informative. Direction-only recovers the full
pathogenicity signal, but does not recover mechanism signal. Whatever ESM-2 encodes about pathogenicity in the
direction of its embedding shift, it does not encode mechanism in the same way.

---

## Table 3. Geometry of the pathogenicity direction

This table characterises the pathogenicity direction found in Table 1. The first two rows
test whether the signal is concentrated in one direction or spread across many. The remaining
rows test whether the direction is shared across protein families: directions are fit
independently on disjoint family halves and compared.

| Quantity | Value |
|---|---|
| full linear AUROC (family-split) | 0.867 |
| 1-D projection onto the single fitted direction | 0.867 |
| AUROC after removing 1 / 2 / 3 / 4 / 5 fitted directions | 0.861 / 0.861 / 0.855 / 0.852 / 0.846 |
| cosine of directions fit on disjoint family halves | 0.320 ± 0.021 |
| cosine null (labels shuffled within each half) | -0.001 ± 0.029 |
| transfer AUROC (direction fit on half A, scored on B) | 0.850 ± 0.004 |

**Reading**   
One fitted direction recovers the full linear signal (1-D projection = 0.867). Removing that direction, and even the next four, only slowly lowers performance (down to 0.846). Pathogenicity therefore behaves as a single functional degree of freedom that is redundantly encoded across many correlated dimensions.  
The low cosine similarity (0.320) between directions learned on disjoint family halves is misleading. Because the signal is redundant, different linear combinations can still point at the same predictive subspace. The decisive number is transfer AUROC: a direction fit on half A still scores 0.850 on half B, nearly matching the within-set result (0.867). The direction is shared across families.

---

## Table 4. Cross-family transfer by task

Under one identical protocol, a direction is fit on one half of the protein families and
scored on the other. This is done for pathogenicity and for mechanism, with both a linear
probe and a gradient-boosted model, to see whether the signal transfers across families and
whether nonlinearity helps.

| Task | Probe | Pooled AUROC | Transfer AUROC |
|---|---|---:|---:|
| pathogenicity (path vs benign) | linear | 0.868 | 0.849 |
| pathogenicity | GBM | 0.906 | 0.897 |
| mechanism (GOF vs rest) | linear | 0.802 | 0.620 |
| mechanism | GBM | 0.804 | 0.636 |

**Verdict** Pathogenicity transfers strongly across held-out families (0.849-0.897),
losing only 0.01-0.02 from pooled to transfer. Mechanism transfers poorly (0.620-0.636),
dropping 0.17-0.18 from its pooled score. Nonlinearity (GBM) helps pathogenicity slightly
but does not rescue mechanism. Within one frozen model, pathogenicity is a transferable
direction and mechanism is not.

---

## Table 5. What is the direction? Biochemistry

A Ridge regression of the pathogenicity axis onto four context-free substitution features
(BLOSUM62, hydropathy change, charge change, volume change) asks how much of the direction
can be explained by simple amino acid property differences, without any sequence context.

| Feature | Spearman with axis |
|---|---|
| BLOSUM62 | -0.286 |
| abs(hydropathy change) | 0.228 |
| abs(volume change) | 0.202 |
| abs(charge change) | 0.128 |
| magnitude `\|\|d\|\|` | 0.440 |

| Metric | Value |
|---|---|
| R² (axis from all four biochem features) | 0.074 |
| pathogenicity AUROC, context-free biochem only (family-split) | 0.696 ± 0.007 |
| pathogenicity AUROC, ESM-2 delta (family-split) | 0.861 ± 0.006 |
| pathogenicity AUROC, ESM-2 delta + biochem (family-split) | 0.872 ± 0.005 |

**Verdict**   
Only 7.4% of the pathogenicity axis is explained by context-free substitution
chemistry. The biggest single correlate is BLOSUM62 (Spearman -0.286), reflecting that less
conservative substitutions tend to be more damaging. But the context-free features together
reach only AUROC 0.696, well below the embedding's 0.861. The axis is position-aware: ESM-2
is reading the sequence context at the mutation site, not just looking up what kind of amino
acid swap occurred.

The ± values above are seed-to-seed standard deviations (5 seeds), not bootstrap CIs. This
probe does not have cluster-bootstrap confidence intervals wired.

---

## Table 6. The conservation decider

This is the key table. It asks whether the pathogenicity direction is just conservation, meaning
ESM-2's own sense of how expected each amino acid is at each position.

For each of the 37,258 variants, ESM-2 was run with the wildtype residue masked, and the
model's predicted probability of the wildtype residue, the mutant residue, and the entropy
over all 20 amino acids at that position were recorded. The ESM1v masked-marginal
(log-probability of the wildtype minus log-probability of the mutant) is a single number
summarising how surprised the model is by the substitution.

**Note.** All AUROCs are family-split. CIs are 95% cluster bootstrap over 1,141 families.
Paired differences use a paired cluster bootstrap with the same family draw applied to both
arms.

| Feature set | AUROC |
|---|---:|
| conservation (4 masked-LM features) | 0.891 [0.883, 0.898] |
| masked-marginal alone (1 feature) | 0.891 [0.884, 0.898] |
| embedding delta (1,280-d) | 0.866 [0.860, 0.872] |
| conservation + delta | 0.898 [0.892, 0.905] |

| Paired difference | Value | 95% CI |
|---|---:|---|
| conservation minus delta | +0.025 | [+0.020, +0.029] |
| conservation+delta minus conservation (K2) | +0.008 | [+0.005, +0.011] |
| conservation+delta minus delta | +0.032 | [+0.029, +0.036] |

Spearman correlation between the pathogenicity axis and the masked-marginal score: **+0.740**.

### Pre-registered claims tested in this experiment

Pre-registration means the rules for interpreting these results were written down before the
numbers came in, so the verdict cannot be adjusted afterward. The two gates below come from
[`PREREGISTRATION_run_biorxiv.md`](../../biorxiv/PREREGISTRATION_run_biorxiv.md), claim 2D.

**2D-1A. Conservation alone reaches AUROC 0.85 (the axis is mostly conservation).**

Conservation alone scores 0.891, and its CI [0.883, 0.898] excludes the 0.85 threshold.

✅ **Affirmed.** Conservation alone accounts for the pathogenicity direction.

**2D-1B. Adding the embedding delta on top of conservation improves AUROC by at least 0.02
(the embedding carries pathogenicity signal beyond conservation).**

The improvement is +0.008 [+0.005, +0.011]. The point estimate falls short of 0.02, and the
CI excludes 0.02 from below.

❌ **Failed, established.** The embedding delta adds a statistically detectable but
practically negligible increment (+0.008) over conservation. The CI confirms this is not an
underpowered test: it excludes the 0.02 threshold, so the data are precise enough to say the
effect is smaller than the pre-registered bar, not just that it was missed.

**Reading.** Together, 1A passing and 1B failing say: the pathogenicity direction is largely
explained by conservation. The embedding delta adds a small but statistically detectable
amount of predictive information beyond conservation (+0.008 [+0.005, +0.011]), but this
falls below the pre-registered +0.02 threshold for a meaningful incremental contribution.
ESM-2's mean-pooled embedding is, for pathogenicity, a weaker encoding of largely the same
signal the model's own masked-LM likelihood exposes (0.866 vs 0.891).

---

## Interpretation

Pathogenicity behaves as an **angular** property of ESM-2's perturbation space. What matters
is which way the representation shifts when a residue is mutated, not how far. That signal
contains a shared predictive direction that transfers across protein families, and it is
**largely explained by
conservation**: the model's own masked log-likelihood at the variant position predicts
pathogenicity better than the whole embedding (0.891 vs 0.866). Adding the embedding on top
of conservation improves the score by +0.008, a statistically detectable but practically
small increment that falls below the pre-registered +0.02 bar.

The contrast with mechanism is informative. Under the same protocol, pathogenicity transfers
across held-out families (0.849-0.897 transfer AUROC) while mechanism does not (0.620-0.636).
Within one frozen model, pathogenicity rides on a shared conservation axis that crosses
family boundaries, and mechanism has no comparably transferable direction.

This reframes the pathogenicity result as largely a characterisation of the signal rather than evidence of novel information in the embedding. The delta predicts pathogenicity primarily because it reflects conservation, while carrying only a small additional increment (+0.008) beyond what the conservation features already capture.

---

## What this is and is not

- This is not a claim that ESM-2 cannot represent pathogenicity. It predicts it at AUROC
  0.910 (seed 0). The claim is narrower: the mean-pooled embedding delta adds little predictive information over the model's masked-LM likelihood for this task.
- The mechanism comparison is included only as a contrast. Mechanism numbers are from
  [`report_mechanism.md`](report_mechanism.md); this report does not re-adjudicate the
  mechanism null.
- All probes are uncalibrated and measure discrimination only.
- The biochemistry R² (Table 5) is in-sample and describes the axis; it is not a held-out
  generalisation estimate.
- The CIs on the conservation decider (Table 6) use family-level clustering, which is the
  correct unit since genes within the same family are not independent. The effective cluster
  count (1,141) is reported alongside each interval.

---

## Provenance

| Source | File |
|---|---|
| Magnitude/direction + mechanism comparison | [`probe_results.json`](../../results/run_biorxiv/magnitude_direction/probe_results.json) |
| Geometry (rank + family transfer) | [`geometry_results.json`](../../results/run_biorxiv/magnitude_direction/geometry_results.json) |
| Cross-family transfer by task | [`transfer_contrast.json`](../../results/run_biorxiv/magnitude_direction/transfer_contrast.json) |
| Biochemistry identity of the axis | [`probe4_axis_identity.json`](../../results/run_biorxiv/magnitude_direction/probe4_axis_identity.json) |
| Conservation decider | [`conservation_axis.json`](../../results/run_biorxiv/magnitude_direction/conservation_axis.json) |
| Mechanism chance floor | [`naive_baseline.json`](../../results/run_biorxiv/naive_baseline.json) |
| Conservation features | `data/conservation_pathogenicity.npy` (masked ESM-2 650M, 37,258/37,258 covered) |
| Canonical variant set | `data/pathogenicity_valid_variants_canonical.json` (37,258 variants) |

Computed by `experiments/geometry/run_geometry.py` and
`experiments/geometry/conservation_axis.py` on the run_biorxiv embeddings. 5 seeds,
family-split CV. Conservation extraction ran on a RunPod RTX PRO 4000; all other steps ran
locally on CPU.
