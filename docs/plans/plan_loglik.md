# Plan: Log-likelihood scan for mechanism prediction

## Numbering note

This experiment follows result_20 (in-silico embedding scan). result_21 is already taken (Megascale stability positive control). If this passes decision rules it will be written up as **result_22**.

---

## Background: what is log-likelihood in ESM-2?

ESM-2 was trained as a **masked language model**: given a protein sequence with one position hidden, predict the amino acid at that position from the surrounding context. During training it saw hundreds of millions of real sequences, so it learned which amino acids are plausible at each position given the evolutionary and structural context around them.

This gives us a direct probability score for any amino acid at any position:

```
log P(aa | context)
```

where "context" is the rest of the sequence.

### The mutation effect score

For a specific mutation (wildtype `wt_aa` → mutant `mut_aa` at position `pos`):

```
ΔLL = log P(wt_aa | context) - log P(mut_aa | context)
```

- **High ΔLL** → the model strongly prefers the wildtype. The mutation is "surprising" — it breaks something evolution has conserved. Likely deleterious.
- **Low / negative ΔLL** → the mutation is tolerated. The position isn't critical.

This is exactly what ESM-1v (Meier et al. 2021) and EVE (Frazer et al. 2021) use for variant effect prediction, and it's the most principled readout from a masked language model — it's what the model was explicitly trained to output, not an indirect proxy.

### How this differs from the embedding scan (result_20)

| | Embedding scan (result_20) | Log-likelihood scan (result_21) |
|---|---|---|
| **Score** | ‖mut_emb − wt_emb‖ (L2 norm of embedding difference) | log P(wt) − log P(mut) (direct model output) |
| **Forward passes** | 2 per mutation (wt sequence + mut sequence) | 1 per position (mask that position, read all 20 AAs at once) |
| **Speed** | ~568k passes for 100 positions × 3 probes × 1985 genes | ~198k passes for 100 positions × 1985 genes |
| **What it measures** | How much the internal representation shifts | How surprised the model is by the mutation |
| **Principled?** | Indirect proxy | Direct model output |

The log-likelihood approach is ~3× fewer forward passes because masking one position gives scores for all 20 amino acids simultaneously — no need to run separate wt and mut sequences.

---

## The experiment

For each gene, mask each of the 100 evenly-spaced positions in turn and extract the full 20-AA log-probability distribution from ESM-2. From this, compute per-gene scalar features analogous to the embedding scan, but using ΔLL as the position-level score instead of embedding magnitude.

### Pre-registered feature set (5 features)

| Feature | Definition | Biological motivation |
|---|---|---|
| `ll_wt_mean` | Mean log P(wt_aa) across positions | Overall ESM-2 conservation of the gene |
| `ll_delta_mean` | Mean ΔLL = mean(log P(wt) − log P(mut_probe)) across probe AAs | Average cost of perturbing the gene |
| `ll_delta_cv` | Coefficient of variation of ΔLL across positions | High = hotspot concentration (GOF-like); low = uniform (LOF-like) |
| `ll_hotspot_frac` | Fraction of positions with ΔLL > mean + 1σ | Direct hotspot density |
| `ll_top_entropy` | Mean entropy of the top-10 highest-ΔLL positions | How "peaked" are the hotspots — one dominant AA or several tolerated? |

### Probe amino acids

Same 3 as the embedding scan — Ala, Asp, Trp — to keep results comparable. ΔLL is averaged across the 3 probes at each position.

---

## Decision rules (pre-registered)

Set before running GPU extraction. Thresholds relative to result_20 scan-only family-split F1 (to be filled in after result_20 completes).

| Gate | Threshold | Interpretation |
|---|---|---|
| G1: ll-only vs scan-only | ll F1 > scan F1 + 0.01 | LL is a better readout than embedding distance |
| G2: ll + delta vs scan + delta | Combined F1 > scan+delta F1 + 0.01 | LL adds beyond embedding scan when combined with mean-pooled delta |
| G3: ll + scan vs either alone | Combined F1 > max(ll-only, scan-only) + 0.02 | The two readouts are complementary — capture different signal |

G3 is the most interesting gate: if embedding distance and log-likelihood are complementary, it means the model's *internal representation shift* captures something different from its *explicit probability estimate* — a finding worth reporting in its own right.

---

## Why this might differ from the embedding scan

The embedding L2 distance captures how much the full 1280-dimensional internal representation changes. The log-likelihood captures only the model's final output probability. These can diverge:

- A position where ESM-2 is internally uncertain but ultimately assigns similar probabilities to wt and mut → large embedding shift, small ΔLL
- A position that is highly conserved (only one tolerated AA) → small embedding shift if mut is just slightly worse, but large ΔLL

In practice, the two measures correlate but are not identical. GOF hotspots tend to be under strong epistatic constraint — a specific residue is required for a specific interaction. This might show up more clearly in ΔLL (the model is certain about what should be there) than in embedding distance.

---

## Implementation plan

### Phase 1: Script (half day, local)

Write `scripts/ll_scan.py`:
- Reuse probe list from `data/cache/scan_probes.json` (same 100 positions per gene)
- For each gene: mask each position in turn, run ESM-2 forward pass, extract log-softmax over vocabulary at that position
- Record log P(wt_aa) and log P(probe_aa) for Ala, Asp, Trp
- Compute 5 pre-registered features per gene
- Save to `data/ll_features.npy` + `data/ll_features_meta.json`

### Phase 2: Extraction (GPU, ~1 hour on H100)

- ~198k forward passes (1 per position, not per probe substitution)
- Batch by sequence, not by substitution — much simpler than embedding scan
- Checkpoint every 50 genes

### Phase 3: Probe runs (CPU, ~30 min)

- Same logistic regression setup as `perturbation_probe.py`
- Feature sets: ll-only, ll+delta, ll+scan (if result_20 complete), ll+scan+delta
- 5-seed gene-split and family-split CV

### Phase 4: Write-up

- Check G1–G3
- If G3 passes: frame result_21 around the complementarity finding
- Feature importance: which LL features drive the signal?

---

## Files

| File | Status |
|---|---|
| `scripts/perturbation_scan.py` | ✓ exists (result_20) |
| `data/cache/scan_probes.json` | ✓ exists — reuse same probe positions |
| `scripts/ll_scan.py` | ✗ to be written |
| `data/ll_features.npy` | ✗ Phase 2 output |
| `data/ll_features_meta.json` | ✗ Phase 2 output |
| `results/ll_scan/probe_results.json` | ✗ Phase 3 output |
| `docs/result_22.md` | ✗ written only if passes decision rules |
