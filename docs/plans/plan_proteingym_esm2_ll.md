# Plan: ESM-2 log-likelihood on ProteinGym DMS (human assays)

## Numbering note

result_23 = magnitude/direction/conservation axis. This experiment is **result_24** if it passes decision rules.

---

## The question

ESM-2 zero-shot log-likelihood (`ΔLL = log P(wt) − log P(mut)`) is known to predict DMS fitness scores (published ESM-1v Spearman ~0.44–0.50 on ProteinGym). We have never tested this directly. The project's central methodological question is: **does the signal survive Pfam family holdout?**

result_18 showed AlphaMissense drops from tight ClinVar AUROC to wide bimodal ProteinGym distribution (mean 0.721 ± 0.150). ESM-2 LL should behave differently — it's a direct conservation signal, not a supervised pathogenicity predictor — but we have not checked.

Secondary question: **does performance vary by selection type** (Activity vs Stability vs OrganismalFitness vs Expression vs Binding)?

---

## Data already in hand

- `data/cache/proteingym/DMS_substitutions.csv` — 96 human assays indexed
- `data/cache/proteingym/DMS_ProteinGym_substitutions/` — 217 per-assay DMS CSVs (all downloaded)
- `data/cache/proteingym/mnemonic_to_acc.json` — UniProt mnemonic → accession (81 mapped)
- `data/cache/proteingym/am_scores_proteingym.json` — AM scores cached (for comparison)
- Each DMS CSV includes `mutated_sequence` — the full protein sequence with the mutation applied

The DMS CSVs include the full `mutated_sequence` column, so we do not need to look up WT sequences or apply mutations manually — we read wt/mut sequences directly from the file.

---

## Experimental design

### What we compute

For each single-mutant variant in each human assay:

```
ΔLL = log P(wt_aa | context) − log P(mut_aa | context)
```

Both computed by ESM-2 650M masked language model: mask the mutated position, run one forward pass, read log-softmax over vocabulary.

- WT sequence: from `target_seq` in the DMS index (same for all variants in an assay)
- Mut AA and position: parsed from the `mutant` field (e.g. `A673C` → pos 673, wt=A, mut=C)
- We do NOT use `mutated_sequence` — we mask the position in the WT sequence, same approach as ll_scan.py

### Per-assay metrics

1. **Spearman ρ**: `corr(ΔLL, DMS_score)` — primary. Positive = ESM-2 agrees with fitness (high ΔLL = conservation violated = low fitness).
2. **AUROC**: `roc_auc_score(DMS_score_bin==0, ΔLL)` — damaging as positive, for comparability with result_18.
3. **Coverage**: fraction of variants in the DMS file successfully scored.

### Family-split robustness

The "family-split" question for a published frozen model is different from the probed-head case. We cannot retrain ESM-2. Instead:

**Per-Pfam-family Spearman distribution**: group assays by Pfam family (using `pfam_families.json` where available; fall back to per-protein grouping). Ask whether the per-assay ρ distribution is tight or bimodal.

Additionally: **within-family vs cross-family comparison**. For proteins with a known Pfam family, compute ρ_within (assays from the same Pfam as the query protein) and ρ_cross (assays from different Pfams). If ESM-2 LL is using family-specific evolutionary statistics rather than per-position conservation, ρ_within >> ρ_cross.

### Stratification by selection type

Report mean ± std Spearman separately for:
- OrganismalFitness (n≈32 human)
- Stability (n≈30 human)
- Activity (n≈19 human)
- Expression (n≈9 human)
- Binding (n≈6 human)

Expected ordering based on published literature: Activity ≈ OrganismalFitness > Stability > Binding > Expression.

---

## Decision rules (pre-registered)

| Gate | Criterion | Threshold | Rationale |
|---|---|---|---|
| G1 | Median Spearman ≥ 0.40 | 0.40 | Published ESM-1v baseline on ProteinGym; below this = we failed to replicate |
| G2 | Frac of assays with ρ < 0.20 ≤ 0.25 | 0.25 | Result_18 showed AM fails on 32% of assays; ESM-2 LL should do better |
| G3 | ESM-2 LL median Spearman > AM median Spearman + 0.05 | relative | ESM-2 LL should beat AM on fitness (direct conservation vs supervised pathogenicity) |

G1 is a sanity check — if we can't replicate published ESM-1v numbers, the script has a bug.  
G2 and G3 are the novel contribution: how does the variance / per-assay distribution compare to AM?

If G1 fails: debug before proceeding.  
If G1 passes but G2/G3 fail: still publishable as "ESM-2 LL and AlphaMissense have similar per-assay variance on ProteinGym."  
If all pass: clean positive story — conservation signal is more uniformly distributed across assays than supervised pathogenicity signal.

---

## Implementation

### Script: `scripts/proteingym_esm2_ll.py`

Phases:

**Phase 1 (CPU, trivial):** Parse DMS index, collect all (assay, position, wt_aa, mut_aa) tuples needed. Group by protein so each protein's positions can be batched in one GPU session. Cache the job list to `data/cache/proteingym/esm2_ll_jobs.json`.

**Phase 2 (GPU, A100 ~1–2 hours):**  
For each assay protein:
- Load WT sequence from DMS index `target_seq`
- For each variant: parse position and mut_aa from the `mutant` field
- Batch positions: mask each position once, score all 20 AAs → get ΔLL for every variant at that position in one forward pass
- Cache scores per-assay to `data/cache/proteingym/esm2_ll_scores.json`
- Checkpoint every 10 assays

Key efficiency: positions are shared across variants that mutate the same residue. Group by position first, run one forward pass per unique position, then look up scores for all variants at that position.

**Phase 3 (CPU, ~10 min):**  
- Compute per-assay Spearman, AUROC, coverage
- Stratify by `coarse_selection_type`
- Per-Pfam-family distribution (using `pfam_families.json`)
- Compare to AM results from `data/cache/proteingym/am_scores_proteingym.json`
- Write `results/proteingym_esm2_ll/per_assay.json` and `summary.json`

### Reuse from existing code

- `proteingym_alphamissense.py`: assay loading, mnemonic→accession mapping, Spearman/AUROC computation, per-assay structure. Copy and adapt — replace AM score lookup with ESM-2 forward pass.
- `ll_scan.py`: ESM-2 masked LM forward pass, log-softmax extraction, batching. The position-masking logic is identical; we just point it at ProteinGym sequences instead of gene scan probes.
- `window_sequence()` from `experiment.py`: may or may not be needed — ProteinGym sequences can be long (up to 3418 AA). ESM-2 650M has a 1022-token limit. Need to window long sequences around the mutated position.

### Windowing for long sequences

ESM-2 650M max input: 1022 tokens. For sequences > 1022 AA: use `window_sequence()` from `experiment.py` (already handles this — 1000 AA window centred on the position). For sequences ≤ 1022 AA: use full sequence.

Check: median human assay seq_len = 363; 75th percentile = 540. Only a handful exceed 1022 (BRCA1=1863, BRCA2=3418, CAR11=1154). Windowing handles these.

---

## What we expect to find

**Plausible outcome A (most likely):** ESM-2 LL Spearman median ~0.44–0.50, wide per-assay distribution (std ~0.15), similar shape to AM's ProteinGym result but shifted right. G1 passes, G2 borderline, G3 passes. Story: ESM-2 LL is better than AM on fitness but still heterogeneous across assays. Stability and Activity assays score best; Expression and Binding worst.

**Plausible outcome B:** ESM-2 LL tighter distribution than AM (std < 0.10). G1/G2/G3 all pass. Story: supervised AM inherits label noise / distribution mismatch; pure conservation signal is more uniformly applicable. Stronger result.

**Plausible outcome C:** ESM-2 LL and AM are comparable (G3 fails). Story: the heterogeneity is intrinsic to ProteinGym task diversity, not to the predictor type. Interesting null.

In all cases the family-split analysis adds something: whether failures cluster on specific Pfam families or are protein-specific would distinguish "ESM-2 can't generalise to this fold" from "this DMS assay measures something unusual."

---

## Files

| File | Status |
|---|---|
| `data/cache/proteingym/DMS_substitutions.csv` | ✓ exists |
| `data/cache/proteingym/DMS_ProteinGym_substitutions/` | ✓ exists (217 CSVs) |
| `data/cache/proteingym/mnemonic_to_acc.json` | ✓ exists |
| `data/cache/proteingym/am_scores_proteingym.json` | ✓ exists (for comparison) |
| `scripts/proteingym_esm2_ll.py` | ✗ to write |
| `data/cache/proteingym/esm2_ll_jobs.json` | ✗ Phase 1 output |
| `data/cache/proteingym/esm2_ll_scores.json` | ✗ Phase 2 output |
| `results/proteingym_esm2_ll/per_assay.json` | ✗ Phase 3 output |
| `results/proteingym_esm2_ll/summary.json` | ✗ Phase 3 output |
| `docs/result_24.md` | ✗ written if passes G1 |
