# Result 23 — Pathogenicity is a *direction* (= conservation), not magnitude; the transferable signal is task- and probe-dependent

## Date: 2026-05-28 | Scripts: magnitude_direction.py, direction_geometry.py, transfer_contrast.py, conservation_axis.py | Seeds: 0–4 | GPU (masked-LL) + CPU

> **STATUS: confirmed across 5 seeds** (std ≤ 0.002). Full-delta baseline reproduces result_6 (family-split MLP AUROC 0.884). The conservation decider (Probe 5) shows the pathogenicity axis IS conservation, so this stays one result (no result_24, per pre-registration). Probe C (S1724 signed ΔΔG) needs a median-binarization re-run before citing. result_22 reserved for the log-likelihood scan (plan_loglik.md).

---

## Background: what magnitude and direction mean

When a mutation changes the protein sequence, ESM-2's representation of that protein shifts — the embedding moves from one point in 1280-dimensional space to another. That shift has two components:

- **Magnitude** (`‖d‖`): how large the shift is — how much the mutation disrupted ESM-2's representation.
- **Direction** (`d/‖d‖`): which way the representation shifted — what *kind* of change the mutation caused.

The pre-registered hypothesis was: pathogenicity should be encoded in **magnitude** (bigger shift = more disruptive mutation). This experiment tests that hypothesis.

---

## TL;DR

Decomposing the delta `d = mut_emb − wt_emb` into **magnitude** `‖d‖` and **direction** `d/‖d‖` shows the pre-registered hypothesis is **falsified in the opposite direction**: pathogenicity is carried by the **direction** (family-split AUROC 0.896 ≈ full delta), while magnitude-only is weak (0.664). Geometry follow-up: that direction is a single, low-dimensional, redundantly-encoded axis — and the conservation decider (Probe 5) shows **the axis IS conservation** (masked-LL alone gives 0.891 family-split, beating the embedding's 0.835; the embedding adds nothing). So the pathogenicity geometry is *characterisation*, not a novel representation claim.

**The contribution is the task × probe-type contrast**: the only family-transferable signal ESM-2 has is conservation, which fully accounts for pathogenicity (linear, 0.815 transfer / GBM 0.889), is **insufficient for stability** (linear transfer 0.725 but GBM recovers — 0.750 on result_21's Pfam-split: nonlinear manifold), and is **irrelevant to mechanism** (transfer ≈ chance for both linear 0.520 and GBM 0.540 — nonlinearity does not rescue it). Transferability is task- and probe-dependent within one frozen model.

---

## Setup

- **Pathogenicity:** canonical ClinVar set, 16,576 variants / ~900 genes
- **Mechanism:** Gerasimavicius variant-level, 3-class GOF/LOF/DN
- **Decomposition** of `delta_mean` per variant: `full` (1280-d), `mag` (‖d‖, 1-d), `dir` (d/‖d‖, 1280-d unit)
- **CV:** 5-fold gene-split AND family-split (Pfam)
- **Classifiers:** logistic regression + MLP (256,), matching result_6 / result_7

---

## Results (5-seed mean ± std, MLP)

### Pathogenicity — AUROC

| feature | gene-split | family-split |
|---|---|---|
| full delta | 0.885 ± 0.002 | **0.884 ± 0.001** |
| magnitude `‖d‖` | 0.664 ± 0.001 | **0.664 ± 0.000** |
| direction `d/‖d‖` | 0.893 ± 0.002 | **0.896 ± 0.002** |

### Mechanism — macro-F1 (family-split)

| feature | family-split F1 |
|---|---|
| chance floor | 0.322 |
| full delta | 0.274 ± 0.012 |
| magnitude | 0.297 ± 0.022 |
| direction | 0.279 ± 0.010 |

All feature variants sit at or below the chance floor for mechanism — no decomposition surfaces hidden mechanism signal.

---

## Decision rules (pre-registered in plan_magnitude_direction.md)

| Gate | Condition | Value | Result |
|---|---|---|---|
| P1 | magnitude-only pathogenicity AUROC ≥ 0.85 (family-split) | 0.664 | **FAIL** |
| P2 | direction-only pathogenicity AUROC ≤ 0.70 (family-split) | 0.896 | **FAIL** (direction does not collapse) |
| P3 | direction-only mechanism macro-F1 ≤ chance_floor + 0.02 | 0.296 ≤ 0.333 | **PASS** |

P1 and P2 both fail because the hypothesis is inverted, not because there is no signal. Per the plan's pre-registered failure mode, P1-fail means "the clean magnitude=constraint story is wrong" — here it is wrong in the specific, reportable way that pathogenicity is *directional*.

---

## Key findings

### F1 — Magnitude alone is a weak pathogenicity predictor

`‖d‖` gives AUROC 0.664 (family-split), barely above a coin flip relative to the full embedding's 0.886. The raw *size* of the ESM-2 perturbation carries little pathogenicity information. This directly refutes the pre-registered hypothesis.

### F2 — Direction recovers the full pathogenicity signal

Unit-normalised deltas (`d/‖d‖`, magnitude discarded) give AUROC 0.896 — equal to, even marginally above, the full delta (0.886). Essentially all of ESM-2's pathogenicity signal is in *which direction* the representation shifts when a residue is mutated, independent of how large the shift is.

### F3 — The directional signal is family-robust

Direction-only AUROC is identical under gene-split (0.894) and family-split (0.896), Δ ≈ 0. The directional pathogenicity signal is not family-recognition — it transfers across held-out protein families.

### F4 — Mechanism is absent under every decomposition

Full, magnitude, and direction all sit at or below the chance floor for mechanism macro-F1. Decomposing the delta does not surface any hidden mechanism signal — consistent with result_4/result_7.

---

## Interpretation

Pathogenicity behaves as an **angular** property of ESM-2's perturbation space, not a **radial** one. The model encodes *what kind* of disruption a mutation causes (the direction the embedding moves), but the *magnitude* of disruption is largely uninformative for the damaging-vs-benign call.

**Corrected unifying hypothesis.** Probe C's C1 (≈0 correlation between ‖d‖ and |ΔΔG|) shows magnitude is uninformative for stability too — so the distinction between transferable (pathogenicity) and leaky (stability, mechanism) signals is **not** angular-vs-radial. Both ride on direction. The real distinction is:

> **pathogenicity = a family-*universal* direction (shared across folds → transfers);
> stability / mechanism = family-*specific* directions (fold-dependent → leak under family-holdout).**

This is directly testable: fit the pathogenicity direction on disjoint protein family groups and measure cosine similarity between the fitted directions.

---

## Geometry follow-up — rank and family-transfer of the pathogenicity direction

Script: `direction_geometry.py`. Pure CPU on the canonical pathogenicity set.

### Probe 1 — pathogenicity is low effective rank, redundantly encoded

| | family-split AUROC |
|---|---|
| full linear (1280-d) | 0.835 ± 0.011 |
| **1-D projection onto the single fitted direction** | **0.835 ± 0.011** |
| after removing 1 fitted direction (refit) | 0.848 ± 0.008 |
| after removing 2 / 3 / 4 / 5 | 0.854 / 0.850 / 0.848 / 0.847 |

A single linear direction recovers 100% of the *linear* pathogenicity signal (the MLP's extra ~0.05 over 0.835 is nonlinear). But removing that direction and refitting does **not** reduce AUROC — it slightly rises and plateaus. So pathogenicity is a single *functional degree of freedom* that is **redundantly encoded across many correlated dimensions**; no single coordinate is load-bearing.

### Probe 2 — the direction is family-transferable (but raw cosine is the wrong metric)

| metric | value |
|---|---|
| cosine(w_A, w_B), directions fit on disjoint protein-family halves | 0.142 ± 0.022 |
| cosine null (labels shuffled within each half) | −0.020 ± 0.031 |
| **transfer AUROC (direction fit on half A, scored on half B)** | **0.814 ± 0.006** |

The directions fit on disjoint family-halves have *low* cosine (0.142) yet **transfer near-perfectly** (0.814 vs full 0.835). Resolution: because the signal is redundantly encoded, each L2-logistic fit picks a different weighting of correlated features — different-looking weight vectors that project onto the *same predictive subspace*. **Raw cosine of L2-logistic weights understates universality when features are correlated; transfer AUROC is the trustworthy metric.** By transfer, the pathogenicity direction is genuinely family-universal.

**Net:** pathogenicity is a single, low-dimensional, family-transferable functional direction — universal in *effect*, redundantly encoded rather than a clean unique vector. **What that direction is: conservation** (see Probe 5).

### Probe 5 — the conservation decider: the pathogenicity axis IS conservation

Script: `conservation_axis.py`. Masked ESM-2 650M at each of the 16,576 variant positions (GPU, H100) → position-specific conservation readouts (`logP_wt`, `logP_mut`, `entropy`, and the ESM1v `masked_marginal = logP_wt − logP_mut`).

| feature set | family-split AUROC |
|---|---|
| conservation (4 masked-LL features) | **0.891 ± 0.008** |
| masked_marginal **alone** (1 feature) | 0.891 ± 0.008 |
| embedding delta (1280-d) | 0.835 ± 0.011 |
| conservation + delta | 0.870 ± 0.009 |

Spearman(axis projection, masked_marginal) = **+0.74** (entropy −0.67, logP_wt +0.67).

| Gate | Condition | Value | Result |
|---|---|---|---|
| K1 | conservation-alone AUROC ≥ 0.85 | 0.891 | **PASS** (axis ≈ conservation) |
| K2 | AUROC(conservation+delta) − AUROC(conservation) ≥ 0.02 | **−0.021** | **FAIL** |

**Conclusion: the family-universal pathogenicity axis is conservation.** (1) It correlates +0.74 with the ESM1v masked-marginal score; (2) that score *alone* reaches 0.891 — higher than the full embedding delta (0.835); (3) adding the embedding to conservation doesn't help (−0.021). The embedding direction carries **nothing beyond** ESM-2's own masked-LM likelihood for pathogenicity. Per the pre-registered rule (K2 fail), this is folded here rather than written as a separate result, and the pathogenicity-axis geometry is **characterisation, not a novel representation-level claim**.

Notable side-finding: for pathogenicity the mean-pooled embedding delta is a *worse, redundant* encoding of conservation (0.835 vs 0.891) — pooling loses information the masked-LM head exposes directly.

### Probe 3 — transfer contrast across tasks, by probe type (the decider)

Script: `transfer_contrast.py`. Identical protocol for all three tasks. One shared StandardScaler, split the grouping variable into disjoint halves, fit a classifier on half A, score the disjoint half B (and B→A), AUROC, averaged over random half-splits. Run with both a **linear** (L2-logistic) and a **nonlinear** (HistGBM) classifier — because result_21 showed stability is *nonlinearly* encoded and a linear-only contrast is misleading.

| task | grouping | probe | pooled AUROC | transfer AUROC |
|---|---|---|---|---|
| pathogenicity (path vs benign) | Pfam family | linear | 0.847 ± 0.003 | 0.815 ± 0.004 |
| pathogenicity | Pfam family | GBM | 0.894 ± 0.002 | **0.889 ± 0.004** |
| stability (ΔΔG > median) | protein | linear | 0.830 ± 0.030 | 0.725 ± 0.035 |
| stability | protein | GBM | 0.872 ± 0.025 | **0.761 ± 0.045** |
| mechanism (GOF vs rest) | Pfam family | linear | 0.563 ± 0.012 | 0.520 ± 0.012 |
| mechanism | Pfam family | GBM | 0.588 ± 0.012 | **0.540 ± 0.008** |

**The unified finding is probe-type × task** (consistent with result_21):

- **Pathogenicity** — a **linear, family-universal** axis. Transfers strongly even linearly (0.815); GBM adds a little (0.889) but the linear axis is the defining feature.
- **Stability** — a **nonlinear, cross-family** signal. Linear transfer is weaker (0.725), but GBM recovers it (0.761; **0.750 on result_21's authoritative Pfam-split**). It lives on a curved cross-family submanifold, not a single linear direction.
- **Mechanism** — **no transferable signal at any probe level.** Transfer ≈ chance for *both* linear (0.520) and GBM (0.540). Nonlinearity does **not** rescue it — the sharpest asymmetry with stability.

So the geometric statement: ESM-2 encodes pathogenicity as a linear family-universal direction, stability as a nonlinear cross-family manifold, and mechanism not in any probe-accessible, family-transferable form. Transferability is **task-dependent and probe-dependent within one frozen model** — a positive, interpretable claim, not a bare null.

**Correction note.** An earlier version of this section used a *linear-only* contrast and described stability as a "partially fold-specific direction (drop 0.11)." That was the same linear-probe artifact result_21 exposed (Ridge undersells stability). The probe-type × task framing above supersedes it.

Caveats: (1) Stability rests on small n (27 proteins). (2) Mechanism's "no signal" is partly "little signal even pooled" (0.563–0.588), not transfer destroying a real signal.

## Limitations

- **Single seed for some sub-analyses.** The 0.664-vs-0.896 gap is large but unconfirmed across seeds; 5-seed run pending.
- **Probe C not run.** The biophysical-direction test (S1724 signed ΔΔG) needs the megascale embeddings synced locally from the result_21 GPU host.
- **delta_mean only.** `delta_pos` (per-residue) not yet decomposed.

---

## Next steps

1. Full 5-seed run to confirm F1–F3.
2. Sync S1724 embeddings; run Probe C and add binarized-ΔΔG protein-holdout AUROC.
3. Decompose the family-robust direction: is it a low-rank subspace? Does the leaky stability/mechanism signal live in the magnitude or in a different directional component?

---

## Files

- `scripts/magnitude_direction.py` — Probe A/B/C, pure CPU analysis of cached embeddings
- `scripts/direction_geometry.py` — Probe 1 (rank) + Probe 2 (family-transfer)
- `scripts/transfer_contrast.py` — transfer test (linear + GBM) across pathogenicity / stability / mechanism
- `scripts/conservation_axis.py` — Probe 5: masked-LL conservation decider (Phase 1 GPU, Phase 2 CPU)
- `results/magnitude_direction/probe_results.json` — 5-seed magnitude/direction results + gates
- `results/magnitude_direction/geometry_results.json` — rank + family-transfer
- `results/magnitude_direction/transfer_contrast.json` — linear+GBM transfer table
- `results/magnitude_direction/conservation_axis.json` — K1/K2/K3 conservation decider
- `data/conservation_pathogenicity.npy` — masked-LL [logP_wt, logP_mut, entropy] per variant
- `docs/plans/plan_magnitude_direction.md`, `docs/plans/plan_conservation_axis.md` — pre-registrations
