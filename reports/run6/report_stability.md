# Results: Does ESM-2's Delta Encode Protein Stability?

*Companion to [`report_control.md`](report_control.md), which showed the ESM-2 delta
(mutant minus wildtype embedding) predicts ClinVar pathogenic-vs-benign at AUROC ≈ 0.90 while
classifying mechanism (GOF/DN/LOF) at chance. Pathogenicity is a curated clinical label, so a
sceptic could argue its signal overlaps ESM-2's evolutionary training. This report adds a second
positive control with a purely physical label — measured folding stability (ΔΔG) — that has no
connection to clinical curation, and asks the same family-robustness question.*

**Run 6 · 2026-06-03** · ESM-2 `esm2_t33_650M_UR50D` · 177,315 single-point missense variants,
181 natural domains, 77 Pfam families (Tsuboyama 2023 mega-scale dataset) · 5 seeds, 5-fold CV.
Results in [`results/run6/megascale_stability/`](../../results/run6/megascale_stability/).

---

## Summary

ESM-2's delta embedding predicts how much a mutation destabilises a small protein domain, a
purely physical property with no link to clinical curation. A linear probe reads it well in
distribution but loses a chunk of the signal when whole protein families are held out, so part of
what the linear probe sees is family recognition rather than transferable biochemistry. Nonlinear
probes recover most of that loss — two independent ones, a neural net and gradient-boosted trees,
agree — so the transferable stability signal is real but not linearly accessible. The signal is
uneven across domains, and it is independent of the mechanism axis: removing the stability
direction from the mechanism embeddings does not change mechanism prediction. The pipeline again
recovers a known physical signal, and stability lands between pathogenicity and mechanism on the
project's family-robustness gradient.

---

## What is measured, and why

The control report established that the pipeline recovers ClinVar pathogenicity cleanly. But
pathogenicity is curated partly from population-frequency data, and ESM-2 is trained on
evolutionary sequences, so the two are not fully independent. Stability is the cleaner second
control: ΔΔG is measured in a folding assay, with no curation and no evolutionary circularity. If
ESM-2's delta encodes stability — and especially if it does so across held-out families — the
positive-control claim no longer rests on a curation-derived label.

The variant set is the natural-domain, single-point missense subset of the Tsuboyama 2023
mega-scale stability dataset: 177,315 variants across 181 PDB domains (de novo designed
mini-proteins are excluded, since they have no Pfam family). For each variant, ESM-2 mean-pooled
WT and mutant embeddings give `delta_mean = mut − wt`, and probes regress that onto the measured
ΔΔG. Pfam families are assigned by HMMER (`hmmscan` against Pfam-A); 14 domains with no Pfam hit
are kept for the other splits but excluded from family-split only.

**Probes:**

| Probe | What it is | Pre-registered? |
|---|---|---|
| Ridge | Linear regression on the 1280-d delta | yes |
| MLP | Nonlinear (256→64 hidden), trained on GPU | yes |
| XGBoost | Gradient-boosted trees (GPU) | no — exploratory |

**Cross-validation** uses three schemes, each 5 seeds × 5 folds:

| Scheme | What is held out | Tests |
|---|---|---|
| random | random variants | in-distribution ceiling |
| domain | whole domains | generalisation to unseen domains |
| family | whole Pfam families | generalisation to unseen families (the honest test) |

**Metrics:** Spearman ρ (rank correlation between predicted and measured ΔΔG) and AUROC with
ΔΔG binarised at its median (comparable to the pathogenicity control). No-signal value is ρ = 0
/ AUROC = 0.50.

The hypotheses below were pre-registered in
[`docs/plans/plan_megascale_stability.md`](../../docs/plans/plan_megascale_stability.md) before
the run.

---

## Table 1 — Stability prediction across CV schemes (5-seed mean ± std)

| Probe | Random ρ | Domain ρ | Family ρ | Family AUROC |
|---|---:|---:|---:|---:|
| Ridge (linear) | 0.693 ± 0.000 | 0.601 ± 0.002 | 0.554 ± 0.006 | 0.772 ± 0.003 |
| MLP | 0.868 ± 0.000 | 0.715 ± 0.003 | 0.635 ± 0.004 | 0.818 ± 0.003 |
| XGBoost | 0.767 ± 0.000 | 0.676 ± 0.003 | 0.631 ± 0.005 | 0.817 ± 0.003 |
| *no-signal* | *0.000* | *0.000* | *0.000* | *0.500* |

Seed-to-seed std is ≤ 0.006 throughout, so the values are stable across seeds. The larger source
of uncertainty is per-fold (which family lands in which fold), discussed below.

## Table 2 — Pre-registered decision

| Hypothesis | Threshold | Observed | Verdict |
|---|---|---:|---|
| H1 — stability is encoded | random ρ ≥ 0.5 | 0.693 | ✓ pass |
| H2 — family-robust (linear) | random − family Δ ≤ 0.05 | Δ = 0.139 | ✗ LEAKY |
| H4 — tight per-domain distribution | per-domain ρ std ≤ 0.10 | 0.160 | ✗ HETEROGENEOUS |
| H3 — independent of mechanism | mechanism F1 change ≤ +0.01 | −0.001 | ✓ pass |

The pre-registered rule fires **LEAKY** (random ρ ≥ 0.5, family Δ ≥ 0.10): real signal, but
substantially family-dependent at the linear level. H4 additionally fires HETEROGENEOUS.

## Table 3 — Controls and interpretation (not pre-registered)

| Check | Result | Reads as |
|---|---|---|
| Label-shuffle null | ρ = 0.000 / −0.002 / −0.002 (random / domain / family) | no leakage — results are real |
| Delta-norm baseline (`‖delta‖`, 1 feature) | ρ = 0.253 / 0.254 / 0.241 | the signal is directional, not just magnitude |
| Nested-CV alpha (RidgeCV) | ρ = 0.694 / 0.602 / 0.555, chosen α = 100 | matches α = 1.0 — the linear ceiling is real, not under-regularisation |
| PLS dimensionality (family-split) | ρ peaks at ~10 components (0.591), then declines | the transferable signal lives in a low-dimensional subspace |

---

## Reading the tables

**1. ESM-2 encodes stability (H1).**
A linear Ridge probe on the delta reaches random-split ρ = 0.693 (AUROC 0.843). The
mutation-induced embedding shift carries clear information about how destabilising a mutation is —
a physical property with no clinical-curation or evolutionary circularity. H1 passes.

**2. The linear signal is partly family-dependent (H2, LEAKY).**
Ridge drops from 0.693 (random) to 0.601 (domain) to 0.554 (family). The random→family drop of
0.139 is well over the 0.10 LEAKY threshold: roughly a fifth of the linear signal depends on
having seen the family before. What survives (ρ 0.554, AUROC 0.772) is genuine cross-family
biochemistry, but the linear probe leans partly on family recognition. This is unlike
pathogenicity, where the same delta lost almost nothing under family-split (Δ ≈ 0.003 in
[`report_control.md`](report_control.md)).

**3. Nonlinear probes recover most of the loss, and two of them agree.**
The MLP reaches family ρ = 0.635 (AUROC 0.818) and XGBoost family ρ = 0.631 (AUROC 0.817) —
both clearly above Ridge's 0.554 / 0.772. A neural net and gradient-boosted trees, two unrelated
model families, land within 0.005 of each other on the held-out-family test. The transferable
stability signal is therefore real but not *linearly* accessible: a linear probe understates how
much of stability ESM-2 actually encodes across families.

**4. The signal is uneven across domains (H4, HETEROGENEOUS).**
Leave-one-domain-out Spearman averages 0.636 but with std 0.160, ranging from 0.02 (2MCK) to
0.86 (2JZ2). No domain is negative, and 44% exceed ρ 0.70, but a long tail of domains are
predicted poorly. ESM-2 reads stability well for most domains and weakly for some — the same
per-stratum heterogeneity that AlphaMissense showed on ProteinGym.

**5. Stability is independent of mechanism (H3).**
Fitting the stability direction on this dataset, projecting it out of the merged mechanism
embeddings, and re-running the family-split mechanism classifier changes macro-F1 by −0.001
(0.395 → 0.394). The stability axis is essentially orthogonal to whatever mechanism signal
exists, so mechanism does not fail because stability is drowning it out — they are independent
problems in the representation. (This H3 test is only valid because the projected residuals are
not re-standardised per fold, which would reintroduce the removed direction; see Provenance.)

**6. The controls hold.**
Shuffling the ΔΔG labels collapses every probe to ρ ≈ 0, so the CV is not leaking. A
single-scalar `‖delta‖` baseline reaches only ρ ≈ 0.25, far below the full delta's 0.693, so
ESM-2's signal is in *which way* the representation moved, not merely *how much*. Proper
per-fold alpha tuning lands on the same numbers as the default, so the LEAKY verdict is not an
artefact of regularisation. And the PLS sweep shows the family-transferable component is
low-dimensional — under family-split, prediction peaks around 10 components and then falls as
extra components fit family-specific structure that does not transfer.

---

## Where stability sits

The same delta and pipeline, three physical/clinical/functional tasks:

| Property | Best family-split | Family-robust? |
|---|---|---|
| Pathogenicity ([`report_control.md`](report_control.md)) | AUROC 0.894 (linear) | yes — linearly robust |
| **Stability (this report)** | AUROC 0.818 (nonlinear) | partly — nonlinearly recoverable |
| Mechanism ([`report_classifier.md`](report_classifier.md)) | macro-F1 ≈ 0.40 (near floor) | no — family-memorised |

This is a gradient of family-dependence. Pathogenicity transfers across families almost
entirely and linearly. Stability transfers, but the linear probe undersells it and a nonlinear
probe is needed to see the cross-family signal. Mechanism does not transfer at any probe
complexity. The ordering makes biological sense: whether a mutation is damaging has common
signatures; how much it destabilises a fold depends on structural context; how it changes
function (GOF vs LOF) is the most context-specific of the three.

---

## What this is and is not

- **A clean physical positive control.** ΔΔG carries no clinical curation, so the
  family-robustness result cannot be explained by overlap with ESM-2's training signal. It
  confirms the pipeline recovers known physical signal and that the mechanism null is specific.
- **Scoped to small natural domains.** Every domain is ≤ 72 residues — intrinsic to the
  mega-scale folding assay. The claim is "ESM-2 encodes the stability of small natural single
  domains," not of large or multidomain proteins.
- **Not a stability predictor benchmark.** The probes are diagnostic of what the frozen
  embedding contains, not a competitive ΔΔG method (which would fine-tune, e.g. SPURS / THPLM).
- **Tree and MLP results are exploratory.** Only Ridge and the MLP were pre-registered; XGBoost
  was added afterward as a second nonlinear check and agrees with the MLP. It is not presented as
  a confirmed hypothesis.

---

## Statistical limitations and planned analyses (pre-preprint)

- **Per-fold, not seed, is the real uncertainty.** Seed std is ≤ 0.006, but a few families
  dominate the split — SH3 (PF00018) alone is 28 of 181 domains — so the fold that holds out a
  large family drives the family-split spread more than seed reshuffling does. Per-fold intervals
  (a family-level bootstrap) are planned before preprint.
- **Singleton-heavy family structure.** 55 of 77 families are singletons; the cross-family
  signal rests mainly on the 22 multi-member families (112 domains). For singleton families,
  family-split behaves like domain-split.

---

## Provenance

Parsed by `experiments/stability/tsuboyama_loader.py` from
`Tsuboyama2023_Dataset2_Dataset3_20230416.csv` (Processed_K50_dG_datasets): natural domains only
(real PDB-id `WT_name`, no `_MUT` background suffix), single-point substitutions with a finite
`ddG_ML` → 177,315 variants / 181 domains. Pfam families by
`experiments/stability/build_domain_families.py` (`hmmscan --cut_ga` vs Pfam-A) →
[`data/megascale_domain_families.json`](../../data/megascale_domain_families.json), 167/181 domains
assigned, 77 families. ESM-2 650M mean-pooled WT/mutant embeddings extracted on a RunPod H100
(`embeddings/embed_megascale.py`). Probes (`experiments/stability/megascale_stability.py`,
`megascale_mlp.py`, `stability_baselines.py`): 5 seeds × 5-fold CV. XGBoost was run via the
`--xgboost` option (GPU); the sklearn RF/GBM path is the default. H3 projects the stability
direction out of the merged mechanism embeddings *without* per-fold re-standardisation — a
per-fold scaler would rescale columns and reintroduce variance along the removed direction,
silently nullifying the test. Results:
[`summary.json`](../../results/run6/megascale_stability/summary.json),
[`mlp_summary_xgb.json`](../../results/run6/megascale_stability/mlp_summary_xgb.json),
[`baselines.json`](../../results/run6/megascale_stability/baselines.json),
[`h3_stability_projection.json`](../../results/run6/megascale_stability/h3_stability_projection.json),
[`per_protein_spearman.json`](../../results/run6/megascale_stability/per_protein_spearman.json).
