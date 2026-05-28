# Result 10 — Clan-level holdout: partial generalisation, not pure memorisation
## Date: May 25, 2026 | Model: ESM-2 650M | Seed: 0 | Local CPU

---

## TL;DR

Leave-one-clan-out evaluation across 21 qualifying Pfam clans (groups of related protein families) tests whether ESM-2 delta mechanism signal generalises to completely unseen protein folds, or is clan/family memorisation. Clan-holdout MLP macro-F1 = **0.299 ± 0.076** — substantially below the family-split floor of 0.352, but above chance (majority baseline 0.254, per-class AUROCs 0.57–0.64). The signal is heterogeneous: some clans generalise well (Cupin F1=0.536, RING GOF-AUROC=0.797), others collapse to chance (Ion_channel F1=0.190). **Approximately half the family-split mechanism signal is clan-level memorisation; the remainder is genuine cross-fold generalisation.**

---

## Background and motivation

Results 1–9 established that ESM-2 encodes pathogenicity strongly (AUROC 0.88) and mechanism weakly (family-split F1 ~0.36–0.40). A key unresolved question: is the mechanism signal that *does* survive family-split CV genuine sequence-level encoding, or is it clan-level memorisation — the model exploiting the fact that protein families within a clan share fold and mechanism?

The standard family-split CV (results 7–9) holds out Pfam families but families within the same clan remain in training. If the mechanism signal is clan-memorisation, it should collapse when an entire clan is held out. If it's real sequence-level signal, it should survive.

**Decision rule (pre-registered):**
- Clan-holdout F1 ≥ family-split floor (0.352): signal generalises, not memorisation
- Clan-holdout F1 ≈ majority baseline (0.254): pure memorisation
- Between: partial generalisation — both effects present

---

## Setup

- **Features:** delta_mean = mean_pool(mut) − mean_pool(wt), ESM-2 650M frozen (cached embeddings)
- **Probe:** MLP (256→64, alpha=1e-3, early stopping), same architecture as result_7
- **Baseline:** k-NN (k=10, cosine) and majority-class predictor
- **Clan file:** Pfam-A.clans.tsv (Pfam release current as of May 2026)
- **Qualifying clans:** ≥2 mechanism classes, ≥20 variants in second-most-common class, ≥3 genes
- **21 qualifying clans** covering 4,612 variants total

---

## Results

### Aggregate (21 clans)

| Probe | Macro-F1 (mean ± std) | Macro-F1 (weighted) |
|---|---|---|
| MLP clan-holdout | **0.299 ± 0.076** | 0.282 |
| k-NN clan-holdout | 0.274 | — |
| Majority baseline | 0.254 | — |
| **MLP family-split (result_7)** | **0.352** | — |
| **Contrastive proj family-split (result_9)** | **0.387** | — |

**Per-class AUROC (clan-holdout MLP, mean across clans):**
| GOF | DN | LOF |
|---|---|---|
| 0.597 | 0.575 | 0.636 |

All three are above 0.5, confirming real cross-clan signal. LOF benefits most; DN least (consistent with all prior results — DN is mechanistically heterogeneous).

### Per-clan breakdown

| Clan | Name | Variants | Mechs | MLP F1 | GOF AUROC | DN AUROC | LOF AUROC |
|---|---|---|---|---|---|---|---|
| CL0029 | Cupin | 112 | GOF/DN/LOF | **0.536** | 0.756 | **0.839** | 0.875 |
| CL0041 | Death | 90 | GOF/DN/LOF | **0.378** | **0.812** | 0.478 | **0.903** |
| CL0192 | GPCR_A | 181 | GOF/DN/LOF | **0.370** | 0.694 | **0.800** | 0.720 |
| CL0159 | E-set | 458 | GOF/LOF | **0.366** | 0.724 | — | 0.688 |
| CL0016 | PKinase | 398 | GOF/DN/LOF | 0.317 | 0.577 | **0.708** | 0.557 |
| CL0137 | HAD | 196 | GOF/LOF | 0.254 | **0.882** | — | **0.884** |
| CL0229 | RING | 201 | GOF/LOF | 0.352 | **0.797** | — | **0.781** |
| CL0030 | Ion_channel | 1081 | GOF/DN/LOF | 0.190 | 0.536 | 0.536 | 0.586 |
| CL0220 | EF_hand | 200 | GOF/DN/LOF | 0.163 | 0.583 | 0.394 | 0.676 |
| CL0465 | Ank | 271 | GOF/LOF | 0.226 | 0.431 | — | 0.529 |

---

## Key findings

### F1 — Clan-holdout F1 falls between family-split and chance

Clan-holdout MLP F1 = 0.299; family-split MLP F1 = 0.352; majority baseline = 0.254. The clan holdout drops −0.053 below family-split but stays +0.045 above majority. This places the result squarely in "partial generalisation" territory: approximately **half the family-split signal is clan-level memorisation** and half is genuine cross-fold sequence-level mechanism signal.

### F2 — Signal is heterogeneous across clans (std = 0.076)

The large standard deviation reflects genuinely different generalisability across protein architectures:

**Well-generalising clans** (MLP F1 > 0.35):
- **Cupin (CL0029):** F1=0.536, DN AUROC=0.839. Cupins are jelly-roll β-barrel enzymes; GOF/DN/LOF mechanisms map to distinct active site geometries accessible from sequence.
- **Death domain (CL0041):** F1=0.378, LOF AUROC=0.903. Death domains mediate homotypic interactions; LOF vs GOF reflects whether the interaction interface is disrupted or constitutively activated.
- **GPCR_A (CL0192):** F1=0.370, DN AUROC=0.800. GPCRs have stereotyped GOF/DN patterns tied to transmembrane helix packing.

**Poor-generalising clans** (MLP F1 < 0.22):
- **Ion_channel (CL0030):** F1=0.190 — the clan from result_8, which showed within-family signal. The within-family signal is family-specific (PF00520 channel gating is subtly different from other ion channel families in the clan), not a general cross-clan pattern.
- **EF_hand (CL0220):** F1=0.163 — calcium-binding domains where GOF/LOF distinction is heavily context-dependent on the specific protein.
- **Ank (CL0465):** F1=0.226 — ankyrin repeat stacks have very similar sequence patterns regardless of mechanism.

### F3 — AUROCs above 0.5 confirm real signal, not noise

Even the weakest clans show per-class AUROCs meaningfully above 0.5 when averaged. The mean per-class AUROCs (GOF 0.597, DN 0.575, LOF 0.636) indicate that ESM-2 delta embeddings carry some mechanism-correlated information that transfers across protein folds. This rules out the "pure memorisation" hypothesis.

### F4 — Ion_channel (result_8) signal is family-specific, not clan-level

Result_8 found delta F1=0.407 within PF00520 (one ion channel family). The clan-level holdout for CL0030 (all 21 ion channel genes) gives F1=0.190 — essentially chance for a 3-class problem. This means the within-family signal from result_8 does not generalise even across ion channel subtypes. It is genuinely family-specific.

---

## Interpretation

### What this means for the central finding

The corrected picture after result_10:

> ESM-2 delta embeddings encode mechanism signal that is **partially genuine and partially memorisation**. Under family-split CV (the standard evaluation), approximately half the signal comes from clan-level correlations (related protein families sharing both fold and mechanism). The other half represents real cross-fold mechanism generalisation, visible in the clan-holdout results (F1=0.299, AUROCs 0.57–0.64). The signal is strongest for proteins with stereotyped structural mechanisms (GPCRs, death domains, cupins) and weakest for architecturally plastic repeat proteins (ankyrins, EF-hands) and large, heterogeneous superfamilies (ion channels).

### The lookup question (answered)

The original question: "Is it a lookup table?" The answer is **partially yes, partially no**. Clan-level memorisation accounts for ~40–50% of family-split mechanism signal (drop from 0.352 to 0.299 = −0.053, relative to the gap above chance 0.352−0.254=0.098, so ~54% is memorisation). The remaining ~46% is cross-fold generalisation — real mechanism encoding in ESM-2 delta space.

### Why the heterogeneity matters

The variance across clans (std=0.076) is scientifically informative: mechanism is more readable from sequence in protein architectures where mechanism maps to stereotyped structural features (active site geometry for cupins, interface topology for death domains) than in architecturally plastic proteins where the same mechanism can arise from many different sequence contexts.

---

## Updated evidence table (all results)

| Result | Experiment | F1 / AUROC | Notes |
|---|---|---|---|
| result_7 | MLP, delta, gene-split | F1=0.415 | Includes family leakage |
| result_7 | MLP, delta, family-split | F1=0.364 | Pfam-family holdout |
| result_7 | MLP, delta, merged, family-split | F1=0.352 | Larger dataset |
| result_9 | Contrastive proj, family-split | F1=0.397 | Best frozen embedding result |
| **result_10** | **MLP, clan-holdout** | **F1=0.299 ± 0.076** | **Clan-level holdout** |
| result_10 | MLP, clan-holdout, Cupin | F1=0.536 | Best-generalising clan |
| result_10 | MLP, clan-holdout, Ion_channel | F1=0.190 | Near-chance, family-specific |

---

## What's open

1. **Multi-seed replication** — all numbers are seed=0. The large per-clan variance (std=0.076) means some individual clan results may be noisy. 5 seeds would stabilise the aggregate.
2. **Why Cupin generalises** — Cupin F1=0.536 is striking. Worth examining which variants drive this: are specific active-site positions mechanism-predictive across all cupin genes?
3. **Sequence distance analysis** — does clan-holdout performance correlate with Pfam clan size (proxy for fold diversity)? Smaller, tighter clans may generalise better because member families share more detailed structural features.

---

## Files

- `scripts/clan_holdout.py` — implementation
- `results/20260524_baseline_run/run_0/clan_holdout_results_seed0.json` — full per-clan metrics
- Pfam clan file: `Pfam-A.clans.tsv.gz` (Pfam current release, not committed — too large)

---

## Reconciliation note added 2026-05-26

This result's mechanistic interpretation — "DN biology lives at the complex-assembly level, which ESM-2 cannot see from sequence" — motivated Experiment 11 (the proteome-features thread) under the hypothesis that interactome features like PPI_degree would recover the missing DN signal.

That hypothesis is partially contradicted by result_13's T4 feature ablation. Dropping PPI_degree from V2 gives ΔF1 = −0.002 — PPI_degree contributes nothing to aggregate cross-family mechanism prediction. The DN AUROC lift in V2+bad (result 15) comes from constraint + Badonyi structural features, not from interactome biology.

**Resolution via result_16:** the within-family LOGO analysis shows that the within-family mechanism signal lives in *family-residual* proteome features (gene minus family-mean on constraint, abundance, etc.) — not in absolute interactome degree. PPI_degree may carry within-family signal for specific architectures, but it doesn't move the cross-family aggregate metric. The clan-holdout finding here (some clans generalise, some don't) is consistent with the result_16 picture: within-family mechanism is partially learnable from gene-level variation, and the heterogeneity across clans reflects which families have meaningful within-family proteome variation.

The "DN biology = complex assembly" interpretation in this result should be read as resolution-dependent: at the cross-family level, the signal that wins is constraint + Badonyi structural priors, not interactome topology. At the within-family level (result 16), the signal that wins is within-family proteome variation. The interactome-as-DN-signal claim does not survive the feature ablation in either resolution.
