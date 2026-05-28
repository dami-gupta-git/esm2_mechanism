# The leakage fraction: a structural, seed-invariant CV diagnostic

**Date drafted:** 2026-05-26
**Status:** Working note; candidate for v2/v3 paper section and/or standalone methodology note
**Source results:** result_7 (initial observation), result_6 Part 2 (multi-seed confirmation), result_4 (causal explanation), result_10 (clan-split generalisation)

---

## The finding

For each labeled mechanism dataset, define the **leakage fraction**:

> LF = (gene_split_F1 − family_split_F1) / (gene_split_F1 − chance_F1)

It measures what proportion of the above-chance gene-split signal disappears when you switch to family-disjoint folds. On the Gerasimavicius dataset under the project's MLP delta-embedding setup, LF = **62.8%**, and across 5 random seeds the standard deviation is **0.0%** — the value is exact.

The cleanest reading: the leakage fraction is a property of the dataset's structure, not of any particular model run.

## Why it is exact across seeds

Different seeds shuffle the gene → fold and family → fold mappings, so both gene_split_F1 and family_split_F1 are noisy estimators. Empirically they shift seed-to-seed. But their ratio (after subtracting chance) is invariant.

The reason is that the leakage fraction collapses to a ratio of structural quantities:

1. The within-family mechanism agreement rate (~74.8% on Gerasimavicius from result_4 — most genes share their family's modal mechanism)
2. The class distribution (~84% LOF in the labeled set)
3. The Pfam family partition of the gene universe

None of these depend on seed. The numerator and denominator move in proportion when the fold assignment is reshuffled, so the ratio is fixed by dataset structure.

A formal derivation (TODO) should let you compute the leakage fraction from (within-family agreement rate, class balance, family-size distribution) directly — without running a model.

## Why it matters

1. **Pre-flight diagnostic.** Before training anything, compute LF from labels and family assignments alone. If LF > 50%, your gene-split numbers will be substantially inflated by family recognition; family-split is the honest evaluation. If LF < 10%, the holdout choice barely matters.

2. **Ceiling on recoverable signal.** Under family-split CV on Gerasimavicius, the maximum F1 any model can show is roughly 37.2% of whatever it shows under gene-split. That's an intrinsic ceiling. Anyone reporting a higher family-split number than that ratio on this dataset has either (a) a genuinely cross-family-generalising model — a positive finding — or (b) a bug in their family-split implementation. The project has a precedent for case (b): the Pfam-coverage bug flagged in result_7 inflated apparent family-split lift by ~7× before the fix.

3. **Quantitative leakage critique.** Most leakage critiques in ML are qualitative ("k-fold can leak"). This is quantitative, dataset-specific, computable without training, and seed-invariant. That's the shape of a genuinely useful methods contribution.

4. **Falsifies the field's assumption that gene-split is good enough.** Most mechanism-prediction papers (Badonyi 2024 PLOS One, Zhong et al. PreMode 2025, Oliveira et al. 2025, ClearVariant 2025) use gene-split CV. If their datasets carry similar leakage fractions (likely, given they're drawn from overlapping OMIM/DDG2P/ClinVar gene sets), their reported numbers come from a similar 60%+ inflation. We can't verify that without their code, but it's a falsifiable prediction.

## How the field currently handles family leakage

| Paper / line of work | Holdout strategy | Quantifies leakage? |
|---|---|---|
| Saadat & Fellay 2025 (iScience) | MMseqs2 clusters at 20% identity, 20% coverage | No, but evaluation is leakage-aware by construction |
| Livesey & Marsh 2023 (review/critique) | N/A — qualitative critique | Notes the problem; no quantitative diagnostic |
| Badonyi & Marsh 2024 (PLOS One) | 3×10-fold gene-split | No |
| Badonyi & Marsh 2025 (NatComms) | Variant-level Bayesian; no family blocking | No |
| Zhong et al. PreMode 2025 | Gene-split (per published methods) | No |
| Oliveira et al. 2025 | Gene-split | No |
| ClearVariant 2025 | Gene-split | No |
| AlphaMissense, EVE, ESM-1b, ESM-1v | Gene/random split | No (and less critical for pathogenicity) |
| MissION (ion channels) | Within-family (by construction) | Different problem |

The dominant pattern: paper trains a mechanism predictor, reports k-fold gene-split AUROC, publishes. Almost nobody quantifies what fraction of that AUROC would survive a family-aware hold-out. The two exceptions (S&F's MMseqs2 protocol, Livesey & Marsh's qualitative critique) confirm the issue is recognised but don't provide a usable per-dataset diagnostic.

## What's still open

1. **Analytic derivation.** Show that LF = f(within-family agreement, class balance, family-size distribution) in closed form. ~1 day of work; would convert an empirical observation into a theorem.

2. **Verify seed-invariance on other datasets.** The 0.0% std is verified on Gerasimavicius. Confirm on the merged dataset and DDG2P. If the seed-invariance generalises, the diagnostic becomes a property of any family-clustered labeled dataset.

3. **Compute LF for pathogenicity.** Predict: it should be near 0% on the ClinVar pathogenicity set, because pathogenicity is per-variant and less family-correlated. This is testable directly and would confirm the diagnostic distinguishes leakage-prone tasks from leakage-resistant tasks. Result_6's empirical finding (gene→family Δ ≈ 0 reproducibly) is consistent with this prediction but the leakage-fraction form is not yet computed.

4. **Apply the diagnostic to existing published predictors.** Pick the major mechanism predictors (PreMode, ClearVariant, Oliveira et al., LoGoFunc). Compute LF on each of their reported datasets. Report how many would have their headline numbers drop under family-split. This is a separate ~1-month effort mostly spent chasing down code+data, but it would have significant field impact: it converts the project's negative result about ESM-2 specifically into a quantitative critique of standard practice across the variant-effect prediction field.

5. **Cross-dataset comparison.** Does LF vary smoothly with within-family agreement rate? If yes, you get a rule of thumb: "datasets with within-family agreement > X% will have leakage fraction > Y%, and require family-aware CV." That's the kind of practical guidance the field would actually use.

## Where to put this in the writeup

Three honest options, in order of ambition:

- **v2 paper section.** Add a methodology section to the v2 bioRxiv plan titled "The leakage fraction: a structural, seed-invariant diagnostic." Anchor it in result_7's initial observation and result_6 Part 2's seed-invariance confirmation. Cite Saadat & Fellay 2025 and Livesey & Marsh 2023 as related but qualitatively different.

- **Standalone methods note.** Build out the analytic derivation, verify on 2–3 datasets, and post as a focused short paper. Possibly to *Bioinformatics* (methodology format) or as a section in a *Nat Methods* commentary if framed broadly. Length: ~6 pages.

- **Applied study.** Compute the leakage fraction for all major published mechanism predictors and report which ones survive. This is the biggest-impact version but also the most work. Could be a v3 of the project's bioRxiv release or its own paper.

The v2 section is the cheapest and lands the contribution within the existing publication arc. The standalone methods note has the highest field-impact potential per unit of effort once the analytic derivation is done. The applied study is the most ambitious and requires external code/data access.

## What this is NOT

- Not a novel idea that family-split CV matters (Livesey & Marsh and the broader field have flagged this qualitatively)
- Not a new statistical test (it's a derived quantity from existing F1 values)
- Not a critique of any specific paper (no published paper has been re-run under this diagnostic yet — that's open work)

It is: **a formal, quantitative, seed-invariant, computable-without-training diagnostic that turns "leakage is a problem" into "leakage on this dataset is X%, and here is how much your reported number would shrink under family-split."** That precision is what's missing from existing critiques.
