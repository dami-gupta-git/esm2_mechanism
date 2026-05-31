# Result 24 — ESM-2 ΔLL on ProteinGym: conservation predicts DMS fitness (median ρ=0.50) with fewer tail failures than AlphaMissense, but similar overall variance

## Date: 2026-05-28 | Script: proteingym_esm2_ll.py | GPU (H100) phase 2 + CPU phase 3 | 96 human assays

> **STATUS: G1 ✓ G2 ✓ G3 ✗.** Replicated ESM-1v published baseline. ESM-2 LL has fewer low-ρ outliers than AlphaMissense (8% vs 14% below ρ=0.20) but the median gap (+0.041) falls short of the pre-registered +0.05 threshold. Per the pre-registration, this is still result_24 (G1 pass = qualifies).

---

## TL;DR

ESM-2 zero-shot log-likelihood (ΔLL = log P(wt) − log P(mut), masked-LM) predicts DMS fitness scores across 96 human ProteinGym assays with median Spearman ρ = 0.50 — replicating the published ESM-1v baseline. ESM-2 LL has fewer catastrophic failures than AlphaMissense on the same assays (8% vs 14% of assays below ρ=0.20), but the median advantage is only +0.041, not the pre-registered +0.05. **The main finding is negative:** the per-assay variance of a pure conservation signal is nearly as wide as that of a supervised clinical predictor. The bottleneck on ProteinGym is intrinsic assay heterogeneity, not the type of signal. This completes the transferability picture from result_23: conservation transfers to pathogenicity (AUROC 0.891), partially to stability (AUROC 0.750), not at all to mechanism (chance), and to DMS fitness on average (median ρ=0.50) but with high per-assay variance driven by assay type, not protein family.

---

## Setup

- **Data:** ProteinGym v1.3, 96 human single-substitution DMS assays (all assays with ≥20 scoreable variants; 0 skipped)
- **Model:** ESM-2 650M (`esm2_t33_650M_UR50D`), masked-LM
- **Scoring:** mask the mutated position in the WT sequence, one forward pass per unique position, read log-softmax over vocabulary → ΔLL = logP(wt_aa) − logP(mut_aa). Long sequences (BRCA1, BRCA2, CAR11) windowed to 1000 AA centred on the mutated position.
- **Metric:** per-assay Spearman ρ between ΔLL and DMS_score. AUROC: damaging (DMS_score_bin=0) as positive class.
- **Comparison:** AlphaMissense per-assay Spearman from result_18 (93 overlapping assays).
- **Efficiency:** one forward pass per unique position per assay (~200k forward passes total); variants sharing a position scored in a single batch.

---

## Results

### Overall

| metric | mean ± std | median | min | max |
|---|---|---|---|---|
| Spearman ρ | 0.463 ± 0.163 | **0.500** | −0.120 | 0.726 |
| AUROC | 0.766 ± 0.093 | 0.780 | 0.447 | 0.940 |

Coverage: 96/96 assays scored, all variants scored within each assay (no NaN positions).

### Decision rules

| Gate | Criterion | Value | Verdict |
|---|---|---|---|
| G1 | Median Spearman ≥ 0.40 | **0.500** | PASS ✓ |
| G2 | Frac assays with ρ < 0.20 ≤ 0.25 | **0.083** | PASS ✓ |
| G3 | ESM-2 median − AM median ≥ 0.05 | **+0.041** | FAIL ✗ |

### By selection type

| type | n | mean ± std | median | frac < 0.20 |
|---|---|---|---|---|
| Stability | 23 | 0.523 ± 0.150 | **0.589** | 0.04 |
| Activity | 20 | 0.531 ± 0.089 | **0.528** | 0.00 |
| Expression | 15 | 0.445 ± 0.150 | 0.493 | 0.13 |
| OrganismalFitness | 30 | 0.404 ± 0.171 | 0.429 | 0.13 |
| Binding | 8 | 0.381 ± 0.198 | 0.339 | 0.12 |

Ordering (Stability ≈ Activity > Expression > OrganismalFitness > Binding) matches the prediction from the plan.

### Comparison to AlphaMissense (result_18)

| | ESM-2 ΔLL | AlphaMissense |
|---|---|---|
| n assays | 96 | 93 |
| median Spearman | **0.500** | 0.459 |
| frac < 0.20 | **0.083** | 0.140 |

ESM-2 LL has fewer catastrophic failures and a higher median, but the distributions substantially overlap. The gap is real but smaller than pre-registered.

### Worst and best assays

Worst 5 (Spearman):
- `TADBP_HUMAN_Bolognesi_2019`: ρ = −0.120 (OrganismalFitness) — TDP-43, prion-like domain; fitness driven by aggregation, not conservation
- `CD19_HUMAN_Klesmith_2019_FMC_singles`: ρ = +0.050 (Binding)
- `KCNE1_HUMAN_Muhammad_2023_expression`: ρ = +0.106 (Expression)
- `SCN5A_HUMAN_Glazer_2019`: ρ = +0.122 (OrganismalFitness)
- `SYUA_HUMAN_Newberry_2020`: ρ = +0.137 (OrganismalFitness) — α-synuclein, aggregation-prone

Best 5 (Spearman):
- `GRB2_HUMAN_Faure_2021`: ρ = +0.726 (OrganismalFitness)
- `NKX31_HUMAN_Tsuboyama_2023_2L9R`: ρ = +0.697 (Stability)
- `NPC1_HUMAN_Erwood_2022_HEK293T`: ρ = +0.695 (Activity)
- `CP2C9_HUMAN_Amorosi_2021_activity`: ρ = +0.679 (Binding)
- `P53_HUMAN_Kotler_2018`: ρ = +0.678 (OrganismalFitness)

---

## Interpretation

### G3 failure: intrinsic assay heterogeneity, not predictor type

The pre-registered hypothesis was that a pure conservation signal (ESM-2 LL) would be more uniformly applicable across assays than a supervised pathogenicity predictor (AM), because AM's clinical training might not generalise to physical DMS labels. G3's failure (gap +0.041 not +0.05) suggests the per-assay variance is driven more by **what the assay measures** than by predictor type. Both ESM-2 LL and AM struggle on the same failure modes: prion-like/aggregation proteins (TDP-43, α-synuclein), some OrganismalFitness outliers, and Binding assays where fitness depends on interface geometry rather than core conservation.

### Binding is the weak point — makes biological sense

Binding affinity depends on surface residues that interact with a specific partner. ESM-2's conservation signal reflects evolutionary pressure across the entire protein, which is dominated by fold stability and catalytic residues — not binding interfaces. That Binding has median ρ=0.34 and 62% of assays below ρ=0.40 is expected: conservation ≠ partner-specific binding fitness.

### Completing the transferability gradient from result_23

result_23 established: conservation (masked-LL) transfers to pathogenicity (AUROC 0.891, linear, family-robust), stability (AUROC 0.750, nonlinear, cross-family), and not mechanism (chance). This result adds the fourth point:

| target | transfer | character |
|---|---|---|
| pathogenicity | AUROC 0.891 | linear, family-universal |
| stability | AUROC 0.750 | nonlinear, cross-family |
| DMS fitness | median ρ 0.50 | on-average, high assay variance |
| mechanism | ~chance | no transferable signal |

DMS fitness sits between stability and mechanism in terms of conservation's explanatory power — it works on average but the variance is high and assay-type-dependent.

---

## Limitations

- No per-family stratification (pfam_families.json keys are gene symbols not UniProt IDs; most didn't map). The family-heterogeneity question remains open.
- Single model (ESM-2 650M). Larger ESM-2 variants (3B, 15B) would likely improve absolute performance but probably not the assay-type pattern.
- Binding assays are underrepresented (n=8); the low median could partly reflect small sample.

---

## Files

- `scripts/proteingym_esm2_ll.py` — phases 1–3
- `docs/plans/plan_proteingym_esm2_ll.md` — pre-registration
- `data/cache/proteingym/esm2_ll_jobs.json` — phase 1 output (96 assays, ~200k variants)
- `data/cache/proteingym/esm2_ll_scores.json` — phase 2 output (ΔLL per variant)
- `results/proteingym_esm2_ll/per_assay.json` — per-assay Spearman, AUROC, coverage
- `results/proteingym_esm2_ll/summary.json` — aggregate stats, decision rules, AM comparison
