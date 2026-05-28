# Plan: The conservation decider — is the pathogenicity axis more than ESM-2's own conservation signal?

## Numbering note

result_22 is reserved for the log-likelihood mechanism scan (plan_loglik). result_23 = magnitude/direction + geometry (the pathogenicity axis). This experiment, if it passes its decision rule, is written up as **result_24**. It reuses the masked-LL machinery from `ll_scan.py` (plan_loglik), pointed at the pathogenicity variant positions instead of the mechanism probe positions.

---

## The finding that motivates this

**result_23** established a positive, geometric result: ESM-2 encodes pathogenicity as a **single, low-dimensional, family-transferable linear direction** in perturbation space (`d = mut_emb − wt_emb`). The transfer contrast then showed pathogenicity is *linearly* cross-family, stability is *nonlinearly* cross-family, and mechanism is not transferable at any probe level.

**Probe 4** (context-free biochemistry) ruled out the trivial interpretation: the axis is **not** substitution identity — context-free biochemistry (BLOSUM62, charge/hydropathy/volume changes) explains only R²=0.07 of the axis and reaches only 0.696 AUROC vs ESM-2's 0.835. So the axis is **context-dependent**.

## The open question (the one a reviewer will ask)

A context-dependent, family-transferable pathogenicity signal is, almost by definition, **position-specific conservation/constraint** — and "ESM-2 encodes conservation" is partly established (ESM1v; Marquet/Rost embeddings-predict-conservation). So the decisive question is:

> **Is the pathogenicity axis just ESM-2's own conservation signal, or does the embedding *direction* carry pathogenicity information beyond what the model's likelihood head exposes?**

This is the single experiment standing between "well-organised characterisation of known facts" and "the embedding holds variant information its own output does not."

---

## The experiment

ESM-2 is a masked language model: mask a position and it returns `P(aa | context)` for all 20 amino acids. From that we get *position-specific conservation* readouts directly from the model — the same quantity ESM1v uses for variant effect prediction. We compare these to the result_23 pathogenicity axis on the **same** 16,576 canonical variants, same family-split CV.

### Conservation readouts (per variant)

For each variant, mask the WT position and read the 20-AA log-probability distribution:

| Readout | Definition | Meaning |
|---|---|---|
| `logP_wt` | log P(wt_aa \| context) | how expected the wild-type is (high = conserved) |
| `logP_mut` | log P(mut_aa \| context) | how plausible the mutant is |
| `masked_marginal` | logP_wt − logP_mut | the ESM1v variant-effect score |
| `entropy` | −Σ P(aa) log P(aa) | positional constraint (low = conserved) |

These four are the "conservation" feature set.

### Comparisons (family-split, 5 seeds, the same Pfam split as result_23)

1. **Correlation:** Spearman(pathogenicity-axis projection, each conservation readout). How aligned is the axis with conservation?
2. **Conservation alone:** pathogenicity AUROC from the conservation features only.
3. **The decider — does the embedding add over conservation?** AUROC of `[conservation]` vs `[conservation + ESM-2 delta]` under family-split. Symmetric check: does conservation add over the delta?

---

## Decision rule (pre-registered)

All on family-split AUROC, 5-seed mean. Reference: ESM-2 delta linear ≈ 0.835, MLP ≈ 0.884 (result_6/result_23); context-free biochem 0.696 (Probe 4).

| Gate | Condition | Interpretation |
|---|---|---|
| **K1** | conservation-alone AUROC ≥ 0.85 | the axis is essentially conservation (the model's likelihood head already has it) |
| **K2 (the decider)** | AUROC(conservation + delta) − AUROC(conservation) ≥ **0.02** | the embedding direction carries pathogenicity signal **beyond** conservation — NOVEL representation-level claim |
| **K3** | Spearman(axis, masked_marginal) | descriptive: how much of the axis is the ESM1v score |

Pre-registered outcomes:
- **K1 true, K2 false (delta adds ~0):** the transferable axis *is* conservation. Clean but partly-known — result stays "characterisation," fold into result_23 as a paragraph, do **not** write result_24.
- **K2 true (delta adds ≥ 0.02 over conservation):** the embedding encodes pathogenicity beyond the model's own likelihood output. Genuine novelty — write result_24.
- **K1 false (conservation-alone < 0.80):** conservation is weaker than expected on this set; report and interpret the gap, but K2 remains the headline test.

A +0.005 lift on K2 is noise; the +0.02 threshold is the minimum to claim "beyond conservation" at this sample size (n=16,576, 5-fold family-split, ~658 Pfam families).

---

## Why this matters

If K2 passes, the story becomes: *ESM-2's frozen embedding holds a family-universal pathogenicity direction that exceeds the conservation signal in its own masked-LM output.* That is a statement about the representation carrying more than the trained objective exposes — genuinely novel, not a benchmark re-run. Combined with result_23 (linear axis / nonlinear stability manifold / absent mechanism), the project becomes a positive, mechanistic account of what frozen PLM embeddings encode about variants.

If K2 fails, that is also clean and worth knowing — it bounds the contribution honestly to "characterisation," and we do not oversell.

---

## Implementation plan

### Phase 1 — masked-LL extraction (GPU, ~minutes on A100/H100)
`scripts/conservation_axis.py`, reusing `ll_scan.py` patterns:
- Load the canonical pathogenicity variants (n=16,576) and `sequences.json`.
- For each variant: `window_sequence` around the position, mask it, run ESM-2 650M, read log P over the 20 AAs at the masked token.
- Record `[logP_wt, logP_mut, entropy]` per variant, aligned by index to the cached embeddings. `masked_marginal` derived in Phase 2.
- ~16,576 masked forward passes, batched. Cache to `data/conservation_pathogenicity.npy` (+ meta with coverage). Checkpoint so it is re-runnable.

### Phase 2 — analysis (CPU)
- Recompute the result_23 pathogenicity axis (logistic direction on standardised delta) → projection scalar.
- Build the conservation feature matrix; drop variants with missing sequence/out-of-range position (report coverage).
- Run the three comparisons under family-split CV (5 seeds), same `family_split_cv` as result_23.
- Fire gates K1–K3.

### Phase 3 — write-up
- result_24 if K2 passes; otherwise a paragraph appended to result_23.

---

## Files

| File | Status |
|---|---|
| `scripts/ll_scan.py` | ✓ exists (plan_loglik) — masked-LL pattern reused |
| `data/pathogenicity_valid_variants_canonical.json` | ✓ exists (n=16,576) |
| `data/embeddings/emb_*_path_canonical_n16576.npy` | ✓ exists — for the axis |
| `data/sequences.json` | ✓ exists — WT sequences for masking |
| `scripts/conservation_axis.py` | ✗ to be written (Phase 1+2) |
| `data/conservation_pathogenicity.npy` | ✗ Phase 1 output (GPU) |
| `results/magnitude_direction/conservation_axis.json` | ✗ Phase 2 output |
| `docs/result_24.md` | ✗ written only if K2 passes |
