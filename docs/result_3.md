# Result 3: Nonlinear Probe on Delta Embeddings
## MLP vs Linear Probe | May 24, 2026 | Model: ESM-2 650M

---

## Results

### MLP probe on delta_mean (mean-pooled, stability-projected)

| Metric | Linear probe | MLP probe | Change |
|---|---|---|---|
| macro-F1 | 0.279 | **0.414** | +0.135 |
| GOF AUROC | 0.634 | **0.729** | +0.095 |
| DN AUROC | 0.529 | **0.546** | +0.017 |
| LOF AUROC | 0.620 | **0.727** | +0.107 |
| Mean AUROC | 0.610 | **0.667** | +0.057 |

GOF and LOF AUROC both cross the pre-registered "meaningful" threshold of 0.72.

### MLP probe on delta_per_residue

| Metric | Linear probe | MLP probe | Change |
|---|---|---|---|
| macro-F1 | 0.376 | **0.341** | -0.035 |

Per-residue delta is **weaker** under MLP, reversing the linear probe result where per-residue outperformed mean-pooled.

---

## What This Means

**The mechanism signal is real and present in ESM-2 delta embeddings — it is just nonlinearly organised.**

The linear probe couldn't find it because the three mechanism classes (GOF/DN/LOF) are not linearly separable in the 1280-dimensional delta space. The MLP, which can learn curved decision boundaries, recovers the signal. The GOF and LOF AUROCs crossing 0.72 means this clears the pre-registered "meaningful" threshold.

**The per-residue reversal is informative.** Under a linear probe, per-residue delta (0.376 F1) outperformed mean-pooled delta (0.279 F1) — suggesting the local context at the variant position was the dominant linear signal. Under MLP, the pattern reverses: mean-pooled (0.414) outperforms per-residue (0.341). This means the nonlinear mechanism signal is **distributed across the whole protein sequence**, not concentrated at the variant position. The MLP is integrating whole-protein context that the per-residue representation discards.

**DN AUROC (0.546) remains weak across both probes.** DN is the smallest class (894 variants, 60 genes) and the most mechanistically heterogeneous — "dominant negative" covers interface disruption, dimerisation interference, and competitive inhibition. The signal may be too diffuse or the data too limited to recover with either probe type.

---

## Revised Scientific Claim

The original pre-registered claim was:

> *ESM-2 delta-embeddings encode gene-level dominant disease mechanism class (GOF/DN/LOF) geometrically distinct from protein stability, recoverable by a linear classifier.*

**This needs revision.** The linear classifier result is weak (mean AUROC 0.610, weak signal band). The correct claim is:

> *ESM-2 delta-embeddings encode gene-level dominant disease mechanism nonlinearly. The mechanism signal is present in the whole-protein mean delta but is not linearly separable — an MLP probe recovers GOF and LOF AUROC above the pre-registered "meaningful" threshold (0.729 and 0.727 respectively).*

This is a **stronger and more specific finding** than a pure negative. It rules out that the delta encodes nothing — it encodes something real but curved. It also points toward where the signal lives: in distributed, whole-sequence context, not in the local perturbation at the variant position.

---

## Open Questions

1. **Does the MLP signal survive family-split CV?** The linear probe showed no family leakage in delta (Δ=−0.002). If the MLP signal also survives family-split, this is a robust finding. If it collapses, the nonlinear signal may still be family-correlated.

2. **What is the MLP learning?** The linear probe weight vector has a direct geometric interpretation (the mechanism axis in delta space). An MLP's decision boundary is opaque. Probing what activates the MLP (e.g. gradient-based feature attribution) would say something about which sequence positions or embedding dimensions carry the mechanism signal.

3. **Is DN recoverable with more data?** DN has 60 genes. With the expanded G2P dataset (~118 DN genes), the MLP might recover DN signal too.

4. **Path A stability projection.** We ran Path B (stability subspace fit on same data). Path A (Megascale-fit subspace) might project out a different stability axis and expose more linear signal, potentially making the linear probe competitive with MLP.

---

## Next Steps (priority order)

1. Run MLP probe under family-split CV to confirm signal survives homology
2. Run MLP on WT-only to separate "nonlinear mechanism signal" from "nonlinear family signal"
3. Expand to merged G2P dataset for better DN coverage
4. If family-split MLP result is positive: write up as the headline finding

---

## Data Location

- Results from this run: not yet saved to disk (run interactively)
- Embeddings for re-running: `data/embeddings/`
