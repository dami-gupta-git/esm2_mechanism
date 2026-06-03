# Results: What Is the Shape of ESM-2's Pathogenicity Signal?

*Companion to [`report_control.md`](report_control.md), which showed the ESM-2 delta
(mutant minus wildtype embedding) predicts ClinVar pathogenic-vs-benign at AUROC ≈ 0.90 while
classifying mechanism (GOF/DN/LOF) at chance. This report asks what that pathogenicity signal
actually is — where in the delta it lives, whether it is one direction or many, whether it
transfers across protein families, and what biological quantity it corresponds to.*

**Run 6 · 2026-06-01** · ESM-2 `esm2_t33_650M_UR50D` · pathogenicity: 37,218 canonical ClinVar
variants (18,815 pathogenic / 18,403 benign), 1,929 genes · mechanism: 17,826 merged variants,
1,935 genes · 5 seeds, family-split CV. Results in
[`results/run6/magnitude_direction/`](../../results/run6/magnitude_direction/).

---

## Summary

The delta `d = mut_emb − wt_emb` has a size (magnitude `‖d‖`) and a heading (direction `d/‖d‖`).
The pathogenicity signal is almost entirely in the heading: direction-only matches the full delta
(family-split AUROC 0.90), magnitude-only is weak (0.67). That direction is a single axis that
transfers across held-out protein families (0.85), and it is not context-free chemistry. What it
*is* is **conservation** — the model's own masked log-likelihood at the variant position predicts
pathogenicity as well as the whole embedding (0.891 vs 0.859), and adding the embedding adds
nothing (+0.002). So ESM-2 encodes pathogenicity as a single, family-universal direction that
carries no information beyond its own likelihood head. Mechanism, by contrast, rides on no
comparably transferable direction.

---

## What was measured, and why

For each variant the delta `d = mut_emb − wt_emb` is split into:

| Feature | What it is |
|---|---|
| `full` | the delta itself (1,280-d) |
| `magnitude` | `‖d‖`, a single scalar — how much the representation was disturbed |
| `direction` | `d/‖d‖`, the unit vector — which way it was disturbed |

Each is run through the same probes (logistic regression and an MLP) under family-split CV — whole
protein families held out, so a probe cannot win by recognising a family it saw in training.
Pathogenicity (pathogenic vs benign) and mechanism (GOF/DN/LOF) are both tested, because the
question is *why* the same delta predicts one and not the other. Four follow-up probes then
characterise the pathogenicity direction: its rank, its cross-family transfer, whether it is
context-free biochemistry, and whether it is conservation. All numbers are 5-seed means ± std.

---

## Table 1 — Magnitude vs direction (family-split)

| Feature | Pathogenicity AUROC (logreg / MLP) | Mechanism macro-F1 (MLP) |
|---|---|---|
| full delta | 0.859 / 0.893 | 0.415 ± 0.004 |
| magnitude `‖d‖` | 0.673 / 0.673 | 0.322 ± 0.011 |
| direction `d/‖d‖` | 0.867 / **0.901** | 0.415 ± 0.006 |
| chance floor | 0.500 | 0.288 ± 0.002 |

Pathogenicity std ≤ 0.003 throughout. The mechanism chance floor is the measured majority-class
macro-F1 from [`results/run6/naive_baseline.json`](../../results/run6/naive_baseline.json) — the
same floor the other run6 reports cite.

## Table 2 — Geometry of the pathogenicity direction

| Quantity | Value |
|---|---|
| full linear AUROC (family-split) | 0.859 ± 0.006 |
| 1-D projection onto the single fitted direction | 0.859 ± 0.006 |
| AUROC after removing 1 / 2 / 5 directions and refitting | 0.859 / 0.858 / 0.845 |
| cosine of directions fit on disjoint family-halves | 0.322 ± 0.021 |
| cosine null (labels shuffled) | −0.006 ± 0.036 |
| transfer AUROC (direction fit on half A, scored on B) | **0.848 ± 0.004** |

## Table 3 — Cross-family transfer, by task and probe

| Task | Probe | Pooled AUROC | Transfer AUROC |
|---|---|---|---|
| pathogenicity (path vs benign) | linear | 0.867 | 0.848 |
| pathogenicity | GBM | 0.905 | **0.896** |
| mechanism (GOF vs rest) | linear | 0.799 | 0.625 |
| mechanism | GBM | 0.802 | 0.640 |

Stability (ΔΔG transfer) was not run — the S1724 megascale embeddings are not present in this run.

## Table 4 — What is the direction?

| Test | Value | Reading |
|---|---|---|
| R²(axis ← context-free biochemistry, Ridge) | 0.074 | axis is not substitution chemistry |
| pathogenicity AUROC, context-free biochem only | 0.694 | well below the delta |
| pathogenicity AUROC, ESM-2 delta only | 0.860 | |
| conservation alone (4 masked-LL features) | **0.891 ± 0.007** | beats the delta |
| masked_marginal alone (1 feature) | 0.891 ± 0.007 | one number suffices |
| embedding delta (1,280-d) | 0.859 ± 0.006 | |
| conservation + delta | 0.893 ± 0.007 | delta adds +0.002 |
| Spearman(axis projection, masked_marginal) | +0.741 | axis ≈ conservation |

Two thresholds summarise this: conservation alone clears 0.85 (0.891), and adding the embedding
to conservation moves the AUROC by less than 0.02 (+0.002). Together they say the axis is
conservation and the embedding adds nothing on top of it.

---

## Reading the tables

**1. Pathogenicity is a heading, not a distance.**
In Table 1, direction-only reaches MLP AUROC 0.901 — equal to the full delta (0.893) — while
magnitude-only is stuck at 0.673. The raw size of the perturbation barely matters; essentially
all the pathogenicity signal is in which way the representation shifts. The natural prior guess —
that a more damaging mutation simply moves the embedding further — does not hold: distance is
weak, heading is everything.

**2. That heading is a single axis.**
In Table 2, one fitted direction recovers the entire linear signal (the 1-D projection equals the
full 0.859). Removing that direction and refitting does not collapse the score — it drifts down
only slightly (0.859 → 0.845 after five removals). So pathogenicity is one functional degree of
freedom, redundantly spread across many correlated coordinates rather than concentrated in any
single one.

**3. The axis is family-universal.**
Directions fit on disjoint family-halves have low raw cosine (0.322) yet transfer almost
perfectly (0.848 vs the within-set 0.859). The low cosine is a red herring: because the signal is
redundantly encoded, each fit picks a different mix of correlated features that point at the same
predictive subspace. Transfer AUROC is the metric that matters, and it says the axis is genuinely
shared across families — which is why pathogenicity barely drops under family-split.

**4. It is not context-free chemistry.**
In Table 4, a regression of the axis on BLOSUM / hydropathy / charge / volume explains only 7% of
it, and those features alone reach just 0.694 AUROC versus the delta's 0.860. The axis is
position-aware, not a lookup on the amino-acid swap.

**5. The axis is conservation — and the embedding adds nothing.**
This is the decider. The model's own masked log-likelihood at the variant position — four numbers,
or even the single ESM1v masked-marginal — reaches 0.891, *above* the 1,280-d embedding delta
(0.859). Adding the embedding to conservation moves the score by +0.002 (K2 fails), and the axis
correlates +0.74 with the masked-marginal. So the embedding direction carries no pathogenicity
information beyond what the model's likelihood head already exposes.

**6. Mechanism does not ride on a transferable direction.**
In Table 3, pathogenicity transfers across families (0.85–0.90) while mechanism transfers far
worse (0.62–0.64), and decomposing the mechanism delta (Table 1) surfaces no hidden signal —
direction and full are identical (0.415). The contrast is the point: within one frozen model,
pathogenicity is a transferable linear axis and mechanism is not.

---

## Interpretation

Pathogenicity behaves as an **angular** property of ESM-2's perturbation space — what kind of
disruption a mutation causes, not how large — and that angle is a single, family-universal
direction. The decisive finding is what the direction turns out to be: **conservation**. ESM-2's
mean-pooled embedding delta is, for pathogenicity, a worse and redundant re-encoding of the
model's own masked-LM likelihood (0.859 vs 0.891). Pooling the embedding loses information the
likelihood head exposes directly. This reframes the pathogenicity result as *characterisation* —
the delta predicts pathogenicity because it partially reflects conservation — rather than a claim
that the representation holds anything novel about damage beyond likelihood.

The honest reading of the task contrast is that transferability is task-dependent within one
frozen model: pathogenicity rides on a shared conservation axis that crosses family boundaries,
whereas mechanism has no comparably transferable direction.

---

## What this is and is not

- **Not a claim that ESM-2 cannot represent pathogenicity** — it predicts it at 0.90. The claim is
  narrower: the *mean-pooled embedding delta* adds nothing over the model's masked-LM likelihood
  for this task.
- **Not a mechanism result.** Mechanism is included only as the contrast: it stays well below
  pathogenicity and does not transfer across families. Its delta decomposition surfaces no hidden
  signal (direction = full = 0.415).
- **Stability not tested** — the S1724 transfer arm needs the megascale embeddings, absent here.
- Sections A/B of the biochemistry probe are in-sample descriptions of the axis; the R² there is
  not a held-out generalisation estimate.

---

## Statistical limitations and planned analyses (pre-preprint)

The seed spreads here are tight (pathogenicity AUROC std ≤ 0.007), but a seed only reshuffles the
CV folds on a fixed set of genes, so it measures fold jitter, not sampling uncertainty, and
understates the true error because every seed reuses all the data. Planned before preprint
submission, not yet in the result files:

- **Confidence intervals** from a cluster bootstrap over genes (pathogenicity labels and
  conservation are gene/position-level, so the effective N is ≈ 1,929 genes, not 37,218 variants),
  on each AUROC.
- **Paired difference test** for the two load-bearing gaps — conservation (0.891) versus the
  embedding delta (0.859), and the conservation-plus-delta increment (+0.002, the basis for K2) —
  via a paired cluster bootstrap over genes, so the conclusion rests on a tested difference rather
  than separated point estimates.
- **Calibration:** the probes are uncalibrated; the scores measure discrimination only, not risks.

This report already uses a shuffled-label null (the cosine null in Table 2) — the permutation
framework the other reports adopt.

---

## Provenance

Computed by `experiments/geometry/run_geometry.py` (orchestrating `magnitude_direction`,
`direction_geometry`, `transfer_contrast`, `probe4_axis_identity`) and
`experiments/geometry/conservation_axis.py`, on the run6 embeddings. Pathogenicity uses the
canonical ClinVar set (`pathogenicity_valid_variants_canonical.json`, 37,218 variants,
row-aligned to `pathogenicity_{wt,mut}_mean.npy` by content fingerprint); mechanism uses the
merged `valid_variants.json` (17,826). Conservation features (`conservation_pathogenicity.npy`)
are masked ESM-2 650M log-likelihoods at each variant position (GPU extraction, 37,218/37,218
covered). Chance floor read from `naive_baseline.json` (majority-class macro-F1 = 0.288).
5 seeds, family-split CV. Outputs:
[`probe_results.json`](../../results/run6/magnitude_direction/probe_results.json),
[`geometry_results.json`](../../results/run6/magnitude_direction/geometry_results.json),
[`transfer_contrast.json`](../../results/run6/magnitude_direction/transfer_contrast.json),
[`probe4_axis_identity.json`](../../results/run6/magnitude_direction/probe4_axis_identity.json),
[`conservation_axis.json`](../../results/run6/magnitude_direction/conservation_axis.json).
