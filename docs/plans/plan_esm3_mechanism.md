# Plan: ESM-3 mechanism family-split

## The question

ESM-2's conservation signal explains pathogenicity (AUROC 0.891, family-robust) but not mechanism (family-split F1 ≈ 0.364, barely above chance floor 0.322). Is that a property of *sequence-only PLMs*, or of *this specific model*?

ESM-3 is multimodal — it jointly encodes sequence, structure, and function tokens. If structure information is what's missing, ESM-3's delta embedding should show a higher family-split F1 than ESM-2's. If it still fails, the null is strengthened: mechanism is not recoverable from any PLM representation, regardless of modality.

This is the single most important follow-up to the ESM-2 arc.

---

## Pre-registered hypothesis

**H1 (structure rescues mechanism):** ESM-3 full (sequence + structure) family-split macro-F1 > ESM-2 family-split macro-F1 + 0.05 (i.e., > 0.414 on Gerasimavicius or > 0.403 on merged).

**H0 (null, consistent with current story):** ESM-3 family-split F1 ≤ ESM-2 + 0.05. The mechanism null is not a property of the model's modality.

**Secondary:** ESM-3 sequence-only (fair scale comparison) vs ESM-3 full — if the gap between them is large, structure is the operative ingredient. If not, model scale is.

---

## Design

### Model

ESM-3 open, 1.4B parameters (`EvolutionaryScale/esm3-sm-open-v1`, HuggingFace). Non-commercial license — fine for this project. Runnable on H100 80GB.

Do NOT use ESM-3 API or larger variants for the first run. If H1 passes, scale up.

### Representation

Replicate the ESM-2 delta exactly:

```
delta = mean_pool(ESM-3(mut_seq)) − mean_pool(ESM-3(wt_seq))
```

Three conditions:
1. **seq-only:** pass sequence tokens only (no structure, no function) — fair comparison to ESM-2 at larger scale
2. **seq+struct:** pass sequence + AlphaFold2 structure tokens — tests whether structure modality adds mechanism signal
3. **full:** sequence + structure + function tokens (if available for these proteins)

### Structure tokens

ESM-3 encodes structure via a VQ-VAE tokeniser (`ESM3StructureTokenizer`) applied to backbone coordinates. Source: AlphaFold2 predictions from the EMBL AF2 database (available for all human proteins). Download the AF2 PDB for each gene, tokenise with ESM3StructureTokenizer.

Fallback if AF2 structures not available for a gene: use seq-only for that gene (log how many fall back).

### Dataset

Same as result_7: Gerasimavicius variant-level (948 genes, 3-class GOF/LOF/DN), and optionally the merged dataset. Start with Gerasimavicius — it's the canonical benchmark for this project.

Same variants, same CV splits (5-fold gene-split + 5-fold family-split, same random seeds as result_7 for comparability).

### Probe

Same MLP as result_7: 2-layer (256→64), class-weighted, 5 seeds. Report macro-F1 and per-class AUROC.

Also run linear logistic regression — if ESM-3 full passes H1 with a linear probe, the signal is a linear family-transferable direction (same structure as the pathogenicity result). If only MLP passes, it's nonlinear (same structure as the stability result).

---

## Decision rules

| Gate | Criterion | Threshold | Rationale |
|---|---|---|---|
| M1 | ESM-3 full family-split macro-F1 > ESM-2 + 0.05 | > 0.414 (Geras) | H1: structure rescues mechanism |
| M2 | ESM-3 seq-only family-split F1 > ESM-2 + 0.05 | > 0.414 | Scale alone rescues mechanism |
| M3 | ESM-3 full − ESM-3 seq-only > 0.03 | relative | Structure tokens add signal beyond scale |

Interpretation matrix:
- M1 pass, M2 fail, M3 pass → **structure is the operative ingredient** (clean positive)
- M1 pass, M2 pass → scale alone suffices; structure may add on top
- M1 fail → null confirmed; mechanism is not recoverable from PLM representations regardless of modality
- M1 pass, M3 fail → ESM-3 is better but structure tokens aren't the reason (scale or training data)

---

## Phases

**Phase 1 (CPU, local):** Download AF2 structures for all Gerasimavicius genes. Tokenise with ESM3StructureTokenizer. Cache structure tokens. Log fallback rate.

**Phase 2 (GPU, H100):** Extract ESM-3 embeddings for all variants under all three conditions (seq-only, seq+struct, full). Cache deltas. ~same compute as result_7's ESM-2 run.

**Phase 3 (CPU, local):** Run MLP + logistic probes, 5-fold gene-split + family-split, 5 seeds. Evaluate decision rules. Write result_26.

---

## What we expect

Most likely outcome: M1 fails. The mechanism null has been robust across every probe type, representation variant, and dataset in the ESM-2 arc (results 1–10, 23). Structure may lift the ceiling slightly but probably not past the threshold — the fundamental issue is that GOF/LOF/DN are defined by cellular phenotype, not by structural disruption type, and no representation encodes that distinction family-transferably.

If M1 passes, it is the most important finding of the project.

---

## Files

| File | Status |
|---|---|
| `scripts/esm3_mechanism.py` | to write |
| `data/cache/af2_structures/` | to download (Phase 1) |
| `data/cache/esm3_deltas_geras.npy` | Phase 2 output |
| `results/esm3_mechanism/summary.json` | Phase 3 output |
| `docs/result_26.md` | written if experiment completes |
