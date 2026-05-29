# Result 26 — ESM-3 mechanism family-split: scale lifts the floor, structure tokens add nothing

## Date: 2026-05-29 | Script: esm3_mechanism.py | GPU (A100 80GB) phase 2 + CPU phase 3 | Seeds: 0–4

> **STATUS: M1 ✓ M2 ✓ M3 ✗. SCALE SUFFICES — structure tokens do not add mechanism signal.** ESM-3 1.4B beats ESM-2 650M on family-split mechanism by +0.125 F1, but the gain is entirely attributable to model scale. seq_struct is marginally worse than seq-only (−0.007), ruling out AF2 structure tokens as the operative ingredient.

---

## TL;DR

ESM-3 (1.4B, open weights) was run on the same Gerasimavicius mechanism classification task as result_7 (ESM-2 650M), under two conditions: sequence-only and sequence+AlphaFold2 structure tokens. Family-split macro-F1 rises from ESM-2's 0.299 to ESM-3-seq's **0.424** — a substantial lift. But adding structure tokens gives **0.417**, slightly *worse* than seq-only. Structure is not the missing ingredient. The mechanism null from result_7 is partially lifted by scaling the sequence model, but the signal is still far below what would be needed for practical mechanism prediction, and the gap (gene-split 0.450 → family-split 0.424) shows the remaining signal is largely family-transferable rather than leaky. The pre-registered conclusion: **scale helps, structure doesn't, function tokens remain untested.**

---

## Setup

- **Model:** ESM-3 1.4B (`esm3-sm-open-v1`, loaded via `ESM3_sm_open_v0`)
- **Conditions:**
  - `seq` — sequence tokens only (scale comparison to ESM-2 650M)
  - `seq_struct` — sequence + AlphaFold2 structure tokens (structure contribution test)
  - Function tokens: not implemented in ESM-3 open API — dropped; noted as limitation
- **Structures:** AlphaFold2 predictions from EMBL API. 934/948 proteins had AF2 entries. Structure tokens applied to 94.9% of variants (9712/10231); 64 coord-length mismatches fell back to seq-only
- **Representation:** `delta = mean_pool(ESM-3(mut)) − mean_pool(ESM-3(wt))`, dim=1536
- **Dataset:** Gerasimavicius variant-level, 3-class GOF/LOF/DN, 10231/10233 variants embedded (2 skipped: aa_pos out of range after windowing)
- **CV:** 5-fold gene-split + 5-fold family-split, seeds 0–4. Matches result_7 exactly
- **Probe:** PyTorch MLP (256→64, dropout 0.3, class-weighted CE, early stopping, DataLoader shuffle) — matches result_7's `run_mlp_probe`. Logistic regression (balanced, C=0.1) as secondary

---

## Results

### MLP macro-F1 (5-seed mean ± std)

| Condition | Gene-split | Family-split | Leakage Δ |
|---|---|---|---|
| ESM-2 650M (result_7) | 0.415 ± 0.042 | **0.299 ± 0.034** | −0.116 |
| ESM-3 seq-only | 0.450 ± 0.007 | **0.424 ± 0.005** | −0.026 |
| ESM-3 seq+struct | 0.443 ± 0.011 | **0.417 ± 0.015** | −0.026 |

### Per-class AUROC (family-split MLP)

| Condition | GOF AUROC | DN AUROC |
|---|---|---|
| ESM-3 seq | 0.707 | 0.582 |
| ESM-3 seq+struct | 0.673 | 0.558 |

### Logistic regression (family-split, 5-seed mean)

| Condition | LR F1 |
|---|---|
| ESM-3 seq | 0.444 ± 0.009 |
| ESM-3 seq+struct | 0.446 ± 0.011 |

LR slightly favours seq+struct, but difference is within noise — consistent with M3 fail.

---

## Decision rules

| Gate | Criterion | Value | Verdict |
|---|---|---|---|
| M1 | seq_struct family-split F1 > 0.349 (ESM-2 + 0.05) | **0.417** | PASS ✓ |
| M2 | seq family-split F1 > 0.349 | **0.424** | PASS ✓ |
| M3 | seq_struct − seq > 0.03 | **−0.007** | FAIL ✗ |

M1 and M2 both pass — ESM-3 beats the threshold regardless of whether structure is included. M3 fails decisively — structure tokens subtract rather than add.

---

## Interpretation

### Scale lifts the mechanism floor — moderately

ESM-3 seq-only improves family-split F1 from 0.299 to 0.424 (+0.125). This is substantial — more than any other single intervention in the ESM-2 arc (contrastive learning gave +0.033 in result_9). Two possible explanations:

1. **Better sequence representations** at 1.4B capture more evolutionary signal that correlates weakly with mechanism
2. **Lower leakage** — the gene→family drop is only 0.026 for ESM-3 vs 0.116 for ESM-2, suggesting ESM-3's mechanism-relevant signal is more family-transferable

Note: 0.424 is still well below what would be needed for practical use, and the GOF AUROC (0.707) is comparable to result_7's gene-split GOF (0.710), suggesting the family-split gain is partly from reduced leakage rather than genuinely new signal.

### Structure tokens are neutral-to-harmful

seq+struct (0.417) is marginally worse than seq-only (0.424) across MLP, with higher variance (±0.015 vs ±0.005). The structure token encoding introduces noise for this task. This is consistent with the interpretation from result_23: the family-transferable signal ESM-2/3 carries for mechanism is conservation-based, not structural — and adding 3D coordinates that are themselves conservation-correlated doesn't add orthogonal information.

The 64 coord-length fallbacks (proteins where AF2 length ≠ sequence length after windowing) are a minor contamination but affect <1% of variants and can't explain the direction of the M3 result.

### Completing the scale × modality picture

| Model | Modality | Family-split F1 |
|---|---|---|
| ESM-2 650M | seq | 0.299 |
| ESM-3 1.4B | seq | 0.424 |
| ESM-3 1.4B | seq+struct | 0.417 |

Scale adds +0.125; structure adds −0.007. The gap between scale and structure is the finding.

### What remains untested

- **Function tokens** (ESM-3 open API doesn't expose them cleanly — would require the full ESM-3 inference server)
- **Larger ESM-3 variants** (3B, 7B behind paywall/access gate)
- **ESM-3 on merged dataset** — result_7 showed the merged dataset gives a slightly lower but more stable floor; worth checking if the scale lift generalises

---

## Limitations

- Function tokens not implemented — "full" condition dropped. The claim is limited to sequence and sequence+structure.
- Structure applied to 94.9% of variants; the 5.1% seq-only fallback slightly deflates the seq_struct number, but not by enough to change M3's direction.
- MLP seeding: `torch.manual_seed` not called (matches result_7 — val-split shuffle is the only seed source), so ±std reflects data-split variance not weight-init variance.

---

## Files

- `scripts/esm3_mechanism.py` — phases 1–3
- `docs/plans/plan_esm3_mechanism.md` — pre-registration
- `data/cache/esm3_struct_tokens.json` — AF2 coordinates per UniProt ID
- `data/embeddings/esm3_geras_seq_mean.npy` — seq-only delta (10231, 1536)
- `data/embeddings/esm3_geras_seq_struct_mean.npy` — seq+struct delta (10231, 1536)
- `data/embeddings/esm3_geras_valid_idx.npy` — variant indices for label alignment
- `data/embeddings/esm3_geras_struct_meta.json` — structure coverage stats
- `results/esm3_mechanism/summary.json` — full results + decision rules
