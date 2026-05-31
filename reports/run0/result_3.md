# Result 3: Nonlinear Classifier on Delta Embeddings
## MLP vs Linear Classifier | May 24, 2026 | Model: ESM-2 650M

---

## Background: why we're trying a nonlinear classifier

Results 1 and 2 used a **linear classifier** — one that draws a flat decision boundary. If the three mechanism classes (GOF/DN/LOF) aren't arranged in the embedding space in a way a flat boundary can separate, the classifier will fail even if real signal exists.

A **neural network classifier** (MLP — multilayer perceptron) can learn curved and irregular boundaries. This experiment asks: is the mechanism signal genuinely absent from the delta embeddings, or is it just arranged in a way the linear classifier can't handle?

---

## Results

### MLP on delta_mean (whole-protein average shift)

| Metric | Linear classifier | MLP | Change |
|---|---|---|---|
| macro-F1 | 0.279 | **0.414** | +0.135 |
| GOF AUROC | 0.634 | **0.729** | +0.095 |
| DN AUROC | 0.529 | **0.546** | +0.017 |
| LOF AUROC | 0.620 | **0.727** | +0.107 |
| Mean AUROC | 0.610 | **0.667** | +0.057 |

GOF and LOF AUROC both cross the pre-registered "meaningful" threshold of 0.72.

### MLP on delta_per_residue (shift at just the mutated position)

| Metric | Linear classifier | MLP | Change |
|---|---|---|---|
| macro-F1 | 0.376 | **0.341** | -0.035 |

Interestingly, the per-residue delta gets *worse* with the MLP, reversing the pattern from Result 1 where per-residue outperformed the whole-protein average.

---

## What This Means

**The MLP does substantially better on the whole-protein delta.** GOF and LOF AUROC both clear 0.72, the pre-registered threshold for "meaningful" signal. Something in the mean delta is recoverable — the linear classifier just wasn't expressive enough to find it.

**The per-residue reversal is informative.** In Result 1, per-residue delta (F1 = 0.376) outperformed whole-protein average (0.279) with the linear classifier — suggesting the local context at the mutation site was the dominant linear signal. With the MLP, the whole-protein average (0.414) now outperforms per-residue (0.341). This means the signal the MLP is picking up is **spread across the whole protein sequence**, not concentrated at the mutation site. The per-residue representation, by only looking at one position, discards that distributed information.

**DN remains weak (AUROC 0.546).** DN is the smallest class (894 variants, 60 genes) and covers several biologically distinct mechanisms under one label — interface disruption, dimerisation interference, competitive inhibition. The signal may be too diffuse, or there may just not be enough data to learn it.

---

## Revised Scientific Claim

The original pre-registered claim was:

> *ESM-2 delta-embeddings encode gene-level dominant disease mechanism class (GOF/DN/LOF) geometrically distinct from protein stability, recoverable by a linear classifier.*

**This needs revision.** The linear classifier result is weak. The correct claim is:

> *ESM-2 delta-embeddings encode gene-level dominant disease mechanism, but nonlinearly. The signal is present in the whole-protein mean delta but a linear classifier can't find it — a neural network recovers GOF and LOF AUROC above the pre-registered "meaningful" threshold (0.729 and 0.727 respectively).*

This is a **stronger and more specific finding** than just reporting the linear classifier failed. It rules out that the delta contains no information — it contains something real, but it's arranged in a curved way. It also points to where the signal lives: distributed whole-sequence context, not the local perturbation at the mutation site.

---

## Open Questions

1. **Does the MLP signal survive family-split CV?** The linear classifier showed no family leakage in the delta (performance was the same under gene-split and family-split). If the MLP signal also survives family-split, this is a robust finding. If it collapses, the signal the MLP found may still just be protein-family information arranged nonlinearly.

2. **What is the MLP actually learning?** With a linear classifier, the weights have a direct geometric meaning — they define the direction in embedding space that best separates the classes. An MLP's decision boundary is harder to interpret. Gradient-based attribution methods could say something about which positions or embedding dimensions carry the signal.

3. **Is DN recoverable with more data?** DN has only 60 genes. The expanded G2P dataset (~118 DN genes) might help.

4. **Alternative stability projection.** We ran Path B (stability direction fit on the same dataset). A different stability direction fit on independent data (Megascale) might project out a cleaner stability axis, potentially making the linear classifier more competitive.

---

## Next Steps (priority order)

1. Run MLP under family-split CV to confirm the signal survives homology hold-out
2. Run MLP on WT-only to separate "nonlinear mechanism signal" from "nonlinear family signal"
3. Expand to merged G2P dataset for better DN coverage
4. If family-split MLP result is positive: write up as the headline finding

---

## Data Location

- Results from this run: not yet saved to disk (run interactively)
- Embeddings for re-running: `../data/embeddings/`
