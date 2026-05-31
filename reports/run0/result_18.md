# Result 18 — AlphaMissense on ProteinGym: the family-robustness narrows when training–test overlap is removed

**Date:** 2026-05-26
**Script:** `scripts/proteingym_alphamissense.py`
**Inputs:** ProteinGym v1.3 substitution benchmark (96 human DMS assays); local AlphaMissense bulk file (`data/cache/AlphaMissense_aa_substitutions.tsv.gz`); UniProt mnemonic → accession mapping via UniProt REST.
**Outputs:** `results/proteingym_alphamissense/{per_assay,summary}.json`; cached score table at `data/cache/proteingym/am_scores_proteingym.json`.

---

## Background: why ProteinGym instead of ClinVar

Result_17 showed AlphaMissense is family-robust on ClinVar — its performance is consistent across protein families. But we flagged a caveat: AlphaMissense was trained on population frequency data, and ClinVar classifications also use population frequency. The two signals aren't independent, which inflates the apparent AUROC and could also inflate the apparent family-robustness.

**ProteinGym** (Notin et al.) is a benchmark of deep mutational scanning (DMS) assays — physical laboratory measurements of how much each variant disrupts protein function. The ground truth is measured in a test tube, not derived from population data. There is no overlap with AlphaMissense's training signal. This is the right benchmark to check whether result_17's family-robustness claim holds up when the circularity is removed.

---

## TL;DR

The result_17 finding — that AlphaMissense has a tight per-family AUROC distribution on ClinVar (mean 0.948 ± 0.046, no families below 0.70) — does **not** generalise to ProteinGym deep-mutational-scanning labels. On 91 human DMS assays scored against AlphaMissense, the per-assay AUROC distribution is **wide and bimodal**: mean **0.721 ± 0.150**, median 0.748, range 0.170 – 0.957, with **32% of assays below AUROC 0.70 and 14% below 0.60**. The clean ClinVar story was partly underwritten by the training–test logic overlap. When the labels come from physical experiments instead of clinical curation, AlphaMissense's apparent uniform competence breaks down — and the failures cluster on out-of-distribution assays (thermal-stability mini-proteins, less-studied proteins) rather than on classic disease genes.

---

## Method

### Data acquisition

- `DMS_substitutions.csv` (217 assays, 96 human) and per-assay DMS CSV files downloaded from ProteinGym v1.3.
- AlphaMissense scores obtained from our local bulk file (cached for result_17). ProteinGym uses UniProt mnemonics (e.g. `A4_HUMAN`); AM uses accessions (e.g. `P05067`). Mapping built by per-mnemonic UniProt REST query, cached at `data/cache/proteingym/mnemonic_to_acc.json` (81/81 mapped successfully).
- The AM bulk file was streamed once with a 268,018-pair index built from the 96 human DMS files; 223,891 scores matched and cached. Median per-assay coverage of AM scores: **100%** of single-mutant variants.

### Metric conventions

ProteinGym labels: `DMS_score_bin = 1` means *functional*, `DMS_score_bin = 0` means *damaging*. AlphaMissense scores: higher = more pathogenic.

For consistency with result_17 we treat **damaging as the positive class**: AUROC > 0.5 ⇒ AM correctly ranks damaging variants higher.

Spearman is reported as `-ρ(DMS_score, AM_score)` (the negation, since high DMS score = functional but high AM = damaging). Positive Spearman after the sign flip ⇒ agreement.

### Per-assay analysis

For each human assay with ≥ 20 scored single-mutants:

- Compute AUROC and signed Spearman.
- Record n_variants, n_damaging, n_functional, coverage of AM scores.

3 of 96 human assays scored fewer than 20 variants. 91 assays produced an AUROC; 93 produced a Spearman.

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

Per-stratum standard deviation **triples** moving from ClinVar to ProteinGym. The worst-case AUROC moves from 0.76 to 0.17.

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
| | BRCA2_HUMAN_Erwood_2022_HEK293T | 0.951 | 0.433 | 265 | BRCA2 functional |
| | EPHB2_HUMAN_Tsuboyama_2023_1F0M | 0.957 | 0.596 | 37 | Mini-protein (exception) |

The failures concentrate on Tsuboyama 2023 mini-protein stability assays (4 of 5 worst). The successes concentrate on classic disease genes (P53, BRCA2).

## Findings

### F1 — The ClinVar family-robustness was partly underwritten by curation–training overlap

The per-family AUROC distribution on ClinVar (std 0.046) was tight in a way that doesn't generalise to a benchmark with independent ground-truth labels (std 0.150, three times wider). Result_17's headline contribution should be stated more narrowly:

> *Within the ClinVar pathogenic/benign labelling distribution, AlphaMissense's per-protein-family AUROC is tightly clustered around the overall AUROC. This is consistent with AM's training distribution and ClinVar's curation distribution being well-aligned. It does not establish that AM transfers uniformly to held-out functional assays.*

The original "AM is family-robust, full stop" framing is replaced by "AM is family-robust on the curated benchmark it was trained adjacent to; on physically-grounded benchmarks it is heterogeneous."

### F2 — The variance is interpretable, not just noise

The wide per-assay distribution decomposes into recognisable buckets:

| Assay class | n | Mean AUROC | Interpretation |
|---|---|---|---|
| Tsuboyama 2023 mini-protein stability | many | clustered low, several < 0.5 | Out-of-distribution for AM (canonical full-length proteins → mini-protein domains); stability ≠ pathogenicity |
| Classic disease genes (P53, BRCA2, etc.) | ~10 | 0.85 – 0.95 | In-distribution for AM training |
| Misc human protein DMS | majority | 0.65 – 0.85 | Intermediate; consistent with AM's overall Spearman of 0.46 |

The Tsuboyama outliers are not noise. They are out-of-distribution data points exposing a real limitation: AM was trained to predict clinical pathogenicity for canonical full-length proteins, and that signal doesn't transfer to thermal stability of isolated mini-protein domains. The four worst Tsuboyama assays all sit at AUROC < 0.4 — well below the level of any per-family AUROC on ClinVar (min 0.76).

### F3 — The leakage story is task-dependent in a deeper way than result_6 / result_17 alone established

Updating the three-bin classification from result_17:

| Task type | Family-split / cross-stratum behaviour | Caveat |
|---|---|---|
| Per-residue local biochemistry, in-distribution | Robust (result_6, result_17) | Holds only within the predictor's training distribution |
| Per-residue local biochemistry, out-of-distribution | **Wide per-assay variance, frequent failures** (result_18) | Tsuboyama mini-protein stability, less-studied proteins |
| Cross-family pattern recognition | Partial (result_8) | unchanged |
| Per-gene global function | Severe leakage (mechanism, results 1–10) | unchanged |

The first row of result_17 should not be cited as a general "VEPs are family-robust" claim. It should be cited as "VEPs trained adjacent to a curation distribution remain robust within that distribution."

## Caveats

### Tsuboyama assays are an aggressive OOD test

Mini-protein stability is genuinely a different prediction target from clinical pathogenicity. Some of the very low AUROCs reflect a real semantic mismatch (functional pathogenic mutations may not destabilise an isolated domain) rather than family-style leakage. The headline contrast with result_17 still holds — the *distribution shape* is qualitatively different — but the worst Tsuboyama outliers should not be characterised as "AM is wrong here," they should be characterised as "AM does not predict this task."

### Per-protein, not per-family

ProteinGym has ~1 assay per protein for most entries. The cross-stratum unit here is "assay" ≈ "protein," not "Pfam family." This is a different (and arguably stricter) test than result_17.

### AlphaMissense's published Spearman on ProteinGym (~0.5) matches our 0.46

We are not contradicting AM's reported benchmark performance. We are stratifying it. The headline number is consistent.

## Practical conclusion

For the paper: result_17's family-robustness finding now has the right framing. It is a claim about within-curation-distribution robustness, not a general claim about VEPs.

For the methodology argument: this strengthens the family-split-CV diagnostic case. The diagnostic produces *different verdicts on the same predictor depending on the evaluation distribution* — which is exactly what a good diagnostic should do.

## Artifacts

- `data/cache/proteingym/DMS_substitutions.csv` (217 assay index)
- `data/cache/proteingym/DMS_ProteinGym_substitutions/` (217 per-assay DMS files)
- `data/cache/proteingym/mnemonic_to_acc.json` (UniProt mnemonic → accession map, cached)
- `data/cache/proteingym/am_scores_proteingym.json` (223,891 AM scores for human ProteinGym variants)
- `results/proteingym_alphamissense/per_assay.json`
- `results/proteingym_alphamissense/summary.json`
