# Result 18 — AlphaMissense on ProteinGym: the family-robustness narrows when training–test overlap is removed

**Date:** 2026-05-26
**Script:** `scripts/proteingym_alphamissense.py`
**Inputs:** ProteinGym v1.3 substitution benchmark (96 human DMS assays); local AlphaMissense bulk file (`data/cache/AlphaMissense_aa_substitutions.tsv.gz`); UniProt mnemonic → accession mapping via UniProt REST.
**Outputs:** `results/proteingym_alphamissense/{per_assay,summary}.json`; cached score table at `data/cache/proteingym/am_scores_proteingym.json`.

---

## TL;DR

The result_17 finding — that AlphaMissense has a tight per-family AUROC distribution on ClinVar (mean 0.948 ± 0.046, no families below 0.70) — does **not** generalise to ProteinGym deep-mutational-scanning labels. On 91 human DMS assays scored against AlphaMissense, the per-assay AUROC distribution is **wide and bimodal**: mean **0.721 ± 0.150**, median 0.748, range 0.170 – 0.957, with **32% of assays below AUROC 0.70 and 14% below 0.60**. The clean ClinVar story was partly underwritten by the training–test logic overlap between AM's population-frequency labels and ClinVar's frequency-derived curation. When the labels come from physical experiments instead of clinical curation, AlphaMissense's apparent uniform competence breaks down — and the failures cluster on out-of-distribution assays (thermal-stability mini-proteins, less-studied proteins) rather than on classic disease genes.

---

## Purpose

result_17 showed AlphaMissense is family-robust on ClinVar. We flagged a caveat in that writeup: AM's training labels (population frequency) and ClinVar's curation logic (gnomAD frequency contributes to pathogenicity classification) are not independent. The headline AUROC of 0.94 is inflated by this affinity. The per-family *distribution* metric — the actual contribution of result_17 — could in principle still be unaffected; or it could also be inflated.

ProteinGym is the right place to test this because:
1. **Physical ground truth.** DMS measures fitness in a lab. No curation, no frequency overlap.
2. **Diverse proteins.** 96 human assays spanning kinases, channels, TF DBDs, mini-protein stability domains, viral receptor interactions, etc. — not the kinase-and-channel-heavy distribution of ClinVar.
3. **Per-assay structure.** Each assay is one protein, so per-assay AUROC distribution directly probes "does this predictor work on this protein," in the same spirit as result_17's per-family analysis.

If AM remains tight on ProteinGym, the family-robustness claim is bulletproof. If it widens, the ClinVar result was partially an artefact of curation–training overlap and the *generalisation* of result_17's finding is the headline that needs revision.

## Method

### Data acquisition

- `DMS_substitutions.csv` (217 assays, 96 human) and `DMS_ProteinGym_substitutions.zip` (per-assay DMS CSV files) downloaded from `https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/`.
- ProteinGym's `zero_shot_substitutions_scores.zip` was downloaded but does **not** contain AlphaMissense (AM is supervised on population frequency, not zero-shot). The zip was deleted; no per-method "supervised scores" zip is hosted.
- AlphaMissense scores were obtained from our local bulk file (cached for result_17). ProteinGym uses UniProt mnemonics (e.g. `A4_HUMAN`); AM uses accessions (e.g. `P05067`). Mapping built by per-mnemonic UniProt REST query (`https://rest.uniprot.org/uniprotkb/<mnemonic>.tsv?fields=accession`), cached at `data/cache/proteingym/mnemonic_to_acc.json` (81/81 mapped successfully; one accession serves multiple ProteinGym DMS entries for the same protein).
- The AM bulk file was streamed once with a 268,018-pair index built from the 96 human DMS files (all single-mutant variants); 223,891 scores matched and cached at `data/cache/proteingym/am_scores_proteingym.json`. Median per-assay coverage of AM scores: **100%** of single-mutant variants.

### Metric conventions

ProteinGym labels: `DMS_score_bin = 1` means *functional*, `DMS_score_bin = 0` means *damaging* (verified on the Seuma A4 and Tsuboyama RBP1 assays). AlphaMissense scores: higher = more pathogenic.

For consistency with result_17 we treat **damaging as the positive class**: AUROC is computed with `y = (DMS_score_bin == 0)` and score = AM pathogenicity. AUROC > 0.5 ⇒ AM correctly ranks damaging variants higher.

Spearman is reported as `-ρ(DMS_score, AM_score)` (the negation, since high DMS score = functional but high AM = damaging). Positive Spearman after the sign flip ⇒ agreement.

### Per-assay analysis

For each human assay with ≥ 20 scored single-mutants:

- Compute AUROC and signed Spearman.
- Record n_variants, n_damaging, n_functional, coverage of AM scores.

Aggregate over assays: mean, median, std, quartiles, fraction below 0.70 and 0.60 AUROC.

### Skipped assays

3 of 96 human assays scored fewer than 20 variants after merging (`too_few`). 91 assays produced an AUROC; 93 produced a Spearman (two had no class variation in the binarised label).

## Results

### Aggregate distribution

| Statistic | AUROC | -Spearman |
|---|---|---|
| n assays | 91 | 93 |
| mean | **0.721** | 0.391 |
| std | 0.150 | 0.241 |
| min | 0.170 | -0.656 |
| q25 | 0.654 | 0.305 |
| median | 0.748 | 0.459 |
| q75 | 0.816 | 0.542 |
| max | 0.957 | 0.735 |
| frac below 0.70 | **31.9%** | n/a |
| frac below 0.60 | **14.3%** | n/a |

The median Spearman (0.46) matches AlphaMissense's published ProteinGym headline within rounding. We are not contradicting their reported number — we are unpacking it by assay.

### Contrast with ClinVar (result_17)

| Metric | ClinVar per-family (result_17) | ProteinGym per-assay (this result) |
|---|---|---|
| Overall mean | 0.940 | 0.721 |
| Per-stratum mean | 0.948 | 0.721 |
| Per-stratum std | 0.046 | **0.150** |
| Per-stratum min | 0.762 | **0.170** |
| Frac below 0.70 | 0.0% | **31.9%** |
| Frac below 0.60 | 0.0% | **14.3%** |

Per-stratum standard deviation **triples** moving from ClinVar to ProteinGym. The tail of the distribution moves from a worst-case AUROC of 0.76 to a worst-case of 0.17.

### Worst and best assays

| Rank | Assay | AUROC | -ρ | n | Notes |
|---|---|---|---|---|---|
| Worst | RBP1_HUMAN_Tsuboyama_2023_2KWH | 0.170 | -0.656 | 56 | Mini-protein thermal stability |
| | OTU7A_HUMAN_Tsuboyama_2023_2L2D | 0.255 | -0.516 | 43 | Mini-protein stability |
| | TNKS2_HUMAN_Tsuboyama_2023_5JRT | 0.265 | -0.145 | 57 | Mini-protein stability |
| | NKX31_HUMAN_Tsuboyama_2023_2L9R | 0.388 | 0.007 | 95 | Mini-protein stability |
| | MET_HUMAN_Estevam_2023 | 0.451 | -0.096 | 244 | Receptor tyrosine kinase function |
| ... | | | | | |
| Best | P53_HUMAN_Giacomelli_2018_WT_Nutlin | 0.910 | 0.512 | 7,448 | TP53 in Nutlin selection |
| | P53_HUMAN_Kotler_2018 | 0.912 | 0.707 | 1,048 | TP53 tumour growth |
| | OBSCN_HUMAN_Tsuboyama_2023_1V1C | 0.941 | 0.362 | 76 | Mini-protein (exception in cluster) |
| | BRCA2_HUMAN_Erwood_2022_HEK293T | 0.951 | 0.433 | 265 | BRCA2 functional |
| | EPHB2_HUMAN_Tsuboyama_2023_1F0M | 0.957 | 0.596 | 37 | Mini-protein (exception) |

The failures concentrate on Tsuboyama 2023 mini-protein stability assays (4 of 5 worst). The successes concentrate on classic disease genes (P53, BRCA2). The Tsuboyama outliers are particularly informative: AUROCs of 0.17–0.39 are **substantially below chance**, meaning AM systematically ranks variants the *wrong way* on those assays.

## Findings

### F1 — The ClinVar family-robustness was partly underwritten by curation–training overlap

The per-family AUROC distribution on ClinVar (std 0.046) was tight in a way that does not generalise to a benchmark with independent ground-truth labels (std 0.150, three times wider). result_17's headline contribution should therefore be stated more narrowly:

> *Within the ClinVar pathogenic/benign labelling distribution, AlphaMissense's per-Pfam-family AUROC is tightly clustered around the overall AUROC. This is consistent with AM's training-distribution and ClinVar's curation-distribution being well-aligned. It does not establish that AM transfers uniformly to held-out functional assays.*

The original "AM is family-robust, full stop" framing is replaced by "AM is family-robust on the curated benchmark it was trained adjacent to; on physically-grounded benchmarks it is heterogeneous."

### F2 — The variance is interpretable, not just noise

The wide per-assay distribution decomposes into recognisable buckets:

| Assay class | n | Mean AUROC | Interpretation |
|---|---|---|---|
| Tsuboyama 2023 mini-protein stability | many | clustered low, several < 0.5 | Out-of-distribution for AM (canonical full-length proteins → mini-protein domains); stability ≠ pathogenicity |
| Classic disease genes (P53, BRCA2, etc.) | ~10 | 0.85 – 0.95 | In-distribution for AM training |
| Misc human protein DMS | majority | 0.65 – 0.85 | Intermediate; consistent with AM's overall Spearman of 0.46 |

The Tsuboyama outliers are not noise. They are out-of-distribution data points exposing a real limitation: AM was trained to predict clinical pathogenicity for canonical full-length proteins, and that signal does not transfer to thermal stability of isolated mini-protein domains. The four worst Tsuboyama assays all sit at AUROC < 0.4, which is well below the level of any per-family AUROC on ClinVar (min 0.76). The OOD failure is real.

### F3 — The leakage story is task-dependent in a deeper way than result_6 / result_17 alone established

Updating the three-bin classification from result_17:

| Task type | Family-split / cross-stratum behaviour | Caveat |
|---|---|---|
| Per-residue local biochemistry, in-distribution | Robust (result_6, result_17) | Holds only within the predictor's training distribution |
| Per-residue local biochemistry, out-of-distribution | **Wide per-assay variance, frequent failures** (result_18) | Tsuboyama mini-protein stability, less-studied proteins |
| Cross-family pattern recognition | Partial (go_smoke) | unchanged |
| Per-gene global function | Severe leakage (mechanism, results 1–10) | unchanged |

The first row of result_17 should not be cited as a general "VEPs are family-robust" claim. It should be cited as "VEPs trained adjacent to a curation distribution remain robust within that distribution."

## Caveats

### Tsuboyama assays are an aggressive OOD test

Mini-protein stability is genuinely a different prediction target from clinical pathogenicity. Some of the very low AUROCs reflect a real semantic mismatch (functional pathogenic mutations may not destabilise an isolated domain) rather than family-style leakage. The headline contrast with result_17 still holds — the *distribution shape* is qualitatively different — but the worst Tsuboyama outliers should not be characterised as "AM is wrong here," they should be characterised as "AM does not predict this task."

### Per-protein, not per-family

ProteinGym has ~1 assay per protein for most entries. The cross-stratum unit here is "assay" ≈ "protein," not "Pfam family." This is a different (and arguably stricter) test than result_17. Combining the two tests with a per-Pfam-family aggregation across ProteinGym proteins is a sensible follow-up but the n per family would be tiny.

### AlphaMissense's published Spearman on ProteinGym (~0.5) matches our 0.46

We are not contradicting AM's reported benchmark performance. We are stratifying it. The headline number is consistent.

### Mapping coverage

81/81 distinct human mnemonics mapped to accessions. 3/96 assays excluded for fewer than 20 scored variants. Median assay coverage 100%. The selection bias from drop-outs is minimal.

## Practical conclusion

For the paper: result_17's family-robustness finding now has the right framing. It is a claim about within-curation-distribution robustness, not a general claim about VEPs. The ProteinGym per-assay analysis is the broader test, and AM fails it — heterogeneously, in interpretable ways.

For the methodology argument: this strengthens, not weakens, the family-split-CV diagnostic case. The diagnostic produces *different verdicts on the same predictor depending on the evaluation distribution*. That is exactly what a good diagnostic should do — surface that the headline performance number is conditional on the benchmark distribution. The right operational guidance is: family-split CV on multiple independent benchmarks (one curation-derived, one physical-experiment-derived) is the honest evaluation protocol.

## Artifacts

- `data/cache/proteingym/DMS_substitutions.csv` (217 assay index)
- `data/cache/proteingym/DMS_ProteinGym_substitutions/` (217 per-assay DMS files)
- `data/cache/proteingym/mnemonic_to_acc.json` (UniProt mnemonic → accession map, cached)
- `data/cache/proteingym/am_scores_proteingym.json` (223,891 AM scores for human ProteinGym variants)
- `results/proteingym_alphamissense/per_assay.json`
- `results/proteingym_alphamissense/summary.json`
