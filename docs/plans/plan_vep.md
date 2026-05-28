# Plan: family-split leakage analysis for published VEPs

## What this is

A focused follow-up to result_6. Result 6 showed that a *supervised ESM-2 delta probe* trained on ClinVar pathogenic/benign labels has essentially zero family leakage (gene-split AUROC 0.878 vs family-split AUROC 0.876, Δ = 0.002). That answers the question for our own probe. It does **not** answer the question for the published VEPs that the clinical community actually uses.

The plan here is to evaluate the published predictors — AlphaMissense first, then ESM-1v, then the ensemble methods (REVEL, CADD) — on the same variant set, stratified by Pfam family. The headline number is whether each predictor's AUROC drops on family-disjoint test subsets.

## Why this is worth doing despite result_6

1. **Different model class.** Result 6 used frozen ESM-2 + a small supervised head trained by us. AlphaMissense uses AF2 features + a much larger supervised head trained by DeepMind on a curated variant set that we know is family-biased. The supervised head is where family leakage can hide; our probe head was small and might not have had the capacity to memorize families. AlphaMissense's larger head might.

2. **Different label distribution.** AlphaMissense is evaluated on ClinVar variants that are heavily skewed toward a few gene families (kinases, ion channels, BRCA-class). If those families are over-represented in training, the published AUROCs may not transfer to held-out families.

3. **Methodologically clean experiment.** We do not have to retrain anything. We score variants with each predictor and stratify the AUROC computation by Pfam family. Cheap to run, hard to argue with.

## Hypothesis

**H1.** AlphaMissense AUROC drops when stratified by held-out Pfam families relative to the overall AUROC. Magnitude is open — could be small (consistent with result_6) or large (consistent with the trained-head-memorizes-families story).

**H2.** ESM-1v zero-shot scoring shows essentially no drop (consistent with result_6's "ESM-2 encodes per-variant biochemistry" finding).

**H3.** REVEL and CADD, which fold in family-derived conservation features, show larger drops than AlphaMissense or ESM-1v.

If H1 is true and H2/H3 hold up, the contribution is a clean per-predictor decomposition: zero-shot PLM scoring is family-robust, supervised heads inherit family leakage from their training labels in proportion to how much family-correlated information they consume.

## Experimental design

### Data

- **Variants:** the 17,236-variant ClinVar set already assembled for result_6 (`data/pathogenicity_valid_variants.json`). 9,119 pathogenic, 8,117 benign, 944 genes, 658 Pfam families.
- **Family mapping:** `data/pfam_families.json` (1,985 gene → single Pfam ID).
- **Variant key format:** `GENE_POS_WTAA_MUTAA`.

### Predictors

| Predictor | Source | Scope | Status |
|---|---|---|---|
| AlphaMissense | DeepMind, bulk TSV | All human missense | Need to fetch (~5.3 GB compressed) |
| ESM-1v | facebookresearch/esm | Zero-shot scoring | Can compute on existing GPU |
| ESM-2 delta probe | Our pipeline | Trained head | Already in result_6 (baseline) |
| REVEL | rest.ensembl.org or bulk | All human missense | Bulk: ~500 MB |
| CADD | cadd.gs.washington.edu | All human SNVs | Bulk: large, may use REST per-variant |
| PrimateAI-3D | Optional | All human missense | Skip in v1 |

### Splits

For each predictor:

1. **Overall AUROC** — control. Should reproduce published numbers if our variant subset isn't too biased.
2. **Family-stratified AUROC** — 5-fold CV with families assigned to folds. Each fold's test set contains only families absent from that fold's training analog (we are not training, but the analog of "training" here is the assumption that the predictor's developer saw similar families in their data — so the "family-disjoint" subset is the more honest evaluation).
3. **Leakage fraction** — `(overall_AUROC − mean_family_split_AUROC) / (overall_AUROC − 0.5)`. Same metric as result_7.

### Important methodological note

For published predictors, "family-split" does not have the same operational meaning as for a probe we train ourselves. We cannot remove a family from their training set. What we can do is **compute their AUROC on variants from each Pfam family separately**, then ask whether the family-by-family AUROC is consistent with the overall AUROC.

The right framing is therefore: **per-family AUROC distribution.** If a predictor's per-family AUROCs are tightly clustered around the overall AUROC, the predictor is family-robust. If they show a heavy tail or bimodal distribution (some families at AUROC 0.95, others at 0.6), the predictor's headline number is dominated by a few well-represented families and would not transfer to underrepresented ones.

This is a cleaner and more honest analysis than pretending we can "hold out" a family from a model we did not train.

## Implementation phases

### Phase 1 — AlphaMissense (the main attraction, no GPU)

1. Download `AlphaMissense_aa_substitutions.tsv.gz` from `https://storage.googleapis.com/dm_alphamissense/`.
2. Filter to our 17,236 variants by `UniProt_ID + protein_variant`. Requires gene → UniProt mapping (already in our pipeline).
3. Compute:
   - Overall AUROC, PR-AUC.
   - Per-family AUROC (families with ≥10 pathogenic and ≥10 benign).
   - Distribution of per-family AUROCs: mean, std, median, range, IQR.
   - Quartile breakdown: AUROC in top vs bottom quartile of families (by per-family AUROC).
4. Compare to the published AlphaMissense ClinVar AUROC (~0.94).

### Phase 2 — ESM-1v zero-shot (GPU)

1. Run masked-marginal scoring for each variant: log P(mut | context) − log P(wt | context) using ESM-1v.
2. Same analysis as phase 1.
3. Hypothesis: per-family AUROC distribution is tight, consistent with result_6.

### Phase 3 — REVEL and CADD (no GPU)

1. Fetch REVEL bulk TSV (manageable size).
2. Fetch CADD via REST (or bulk if feasible).
3. Same per-family analysis.
4. Hypothesis: wider per-family distribution because conservation features carry family signal.

### Phase 4 — analysis and writeup (CPU)

1. Cross-predictor comparison: side-by-side per-family AUROC distributions.
2. Identify families that are systematically harder for every predictor (probably orphan families with shallow conservation signal).
3. Identify families where predictors disagree (interesting case studies).
4. Writeup as `docs/result_17.md`.

## Deliverables

- `scripts/fetch_alphamissense.py` — downloads + filters bulk AM file to our variant set.
- `scripts/alphamissense_family_split.py` — runs the per-family AUROC analysis.
- `data/alphamissense_scores_full.json` — cached scores for our 17,236 variants.
- `results/alphamissense_family/` — per-family AUROC table, distribution plot, overall metrics.
- `docs/result_17.md` — writeup.

## Open questions before launch

1. **AlphaMissense download.** The bulk aa_substitutions file is ~5.3 GB compressed. Confirm we want to pull it locally rather than use a remote query approach.
2. **Per-family minimum sample size.** Default proposal: ≥10 pathogenic + ≥10 benign for inclusion in per-family AUROC. Drops the long tail of small families but keeps the analysis honest.
3. **Family definition.** Use the same `pfam_families.json` mapping as the rest of the project (one Pfam ID per gene). Genes outside the mapping are dropped — already standard.
4. **What counts as "leakage" here.** Because we are evaluating fixed predictors, the right metric is per-family AUROC distribution, not a holdout drop. Document this clearly so readers don't expect a 0.88 → 0.50 number — that is not the right framing for published predictors.

## What we expect to find — honest prior

result_6's clean negative for our ESM-2 probe is a strong prior that PLM-based predictors will be family-robust. The most likely outcome is:

- AlphaMissense: per-family AUROC tightly distributed around its overall number; a small handful of poorly-served families.
- ESM-1v: similarly tight.
- REVEL/CADD: wider distribution, some families noticeably worse.

If that holds, the contribution is: **the no-leakage finding from result_6 generalizes across major published VEPs, with a small caveat for conservation-ensemble methods.** That is a useful, defensible, narrowly-scoped methodological result — and it neutralizes the obvious "but does this apply to the predictors people actually use?" reviewer pushback against result_6.

If AlphaMissense surprises us with a bimodal per-family distribution, the contribution becomes sharper and the story flips. We will know after phase 1.
