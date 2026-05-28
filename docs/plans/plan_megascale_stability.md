# Plan — Megascale stability as a second ESM-2 positive control

**Date:** 2026-05-27
**Status:** Pre-registration, not yet run
**Data:** `/Users/dgupta/code/downloads/AIScientist/AI-Scientist/esm2_mechanism/data/megascale/`

---

## Motivation

The project currently has one positive control: pathogenicity prediction (result_6, AUROC 0.886, family-split-stable Δ=0.002). This rules out a broken pipeline as the explanation for the mechanism null. But pathogenicity is a curation-derived clinical label whose training signal (population frequency) and test signal (ClinVar curation) are not fully independent (see result_17 caveat, result_18 ProteinGym widening).

A **second, physical-ground-truth positive control** would:

1. Strengthen the "pipeline is sound" claim with a label that has no curation–training overlap
2. Test whether ESM-2's local-biochemistry competence extends to a physical property, not just clinical labels
3. Settle whether the Path A stability projection (subspace fit on a proper stability dataset) would have worked where Path B (FoldX-fit on Gerasimavicius, result_1) failed
4. Give a clean comparator for the AlphaMissense ProteinGym wide-distribution finding (result_18): is the wide per-assay distribution AM-specific, or does ESM-2 also show it on stability?

Megascale (Tsuboyama 2023 + Cho/Tsuboyama 2026 recalibration) is the right dataset: ~800k mutations across ~500 small domains, physical ΔG ground truth, no curation circularity.

---

## Pre-registered hypotheses

**H1 — Stability is encoded.** ESM-2 delta embeddings predict Megascale ΔG with Spearman ρ ≥ 0.5 under random split.

**H2 — Stability is family-robust.** ρ drops by ≤ 0.05 under family-split / cluster-split CV. (Same diagnostic used everywhere else in the project.)

**H3 — Stability ≠ mechanism dissociation.** Even though ESM-2 encodes stability, this does not rescue the mechanism null — stability projected out of mechanism predictions does not lift family-split mechanism F1.

**H4 — Per-assay distribution is tight.** Per-domain Spearman distribution std ≤ 0.10 (analogous to result_17's per-Pfam-family AUROC tightness on ClinVar pathogenicity). If wide (≥ 0.15, matching result_18 on ProteinGym), the curation-vs-physical-label distinction is the dominant factor, not the family-leakage one.

---

## Decision rule (pre-registered)

| Outcome | Random Spearman | Family-split Δ | Per-domain std | Verdict |
|---|---|---|---|---|
| ROBUST | ≥ 0.5 | ≤ 0.05 | ≤ 0.10 | ESM-2 encodes stability, family-robust |
| WEAK | 0.3–0.5 | any | any | Partial signal; report and stop |
| HETEROGENEOUS | ≥ 0.5 | ≤ 0.05 | ≥ 0.15 | Like result_18 AM/ProteinGym — works on average, fails on some folds |
| LEAKY | ≥ 0.5 | ≥ 0.10 | any | Strong gene-split, family-split collapse — same shortcut as mechanism (unexpected; would be a major finding) |
| NULL | < 0.3 | any | any | ESM-2 does not encode stability — would reshape the project's central claim |

---

## Design

### Data

- **Primary CSV:** `230515_K50dG_dmsv4_dmsv5_dmsv7_concat260429.csv` (~2.2 GB)
  - Confirm columns: WT sequence (per domain), mutation string, ΔG (K50dG-derived)
  - Restrict to single-point missense (drop indels, stops, multi-residue)
- **Splits:** `dmsv4_filtered_train_splits.csv` for reference; build our own family-aware split from scratch
- **Benchmark:** `benchmarks.zip` (S1724, TED) for cross-paper comparison if time permits

### Domain → family mapping

- Run Pfam/HMMER on each unique WT sequence to assign Pfam family
- Where Pfam fails (mini-proteins, designed sequences), fall back to MMseqs2-20 cluster IDs
- Cache mapping to `data/megascale_domain_families.json`

### Features

- ESM-2 650M frozen embeddings (same pipeline as `experiment.py`)
- Two views, matching the project standard:
  - **delta_mean** — mean-pooled (mutant) − mean-pooled (WT)
  - **delta_pos** — per-residue delta at variant position
- Cache to `data/embeddings/megascale_{wt,mut}_*.npy`

### Probes

- **Linear regression** with L2 (Ridge), continuous ΔG target
- **MLP** 1280→256→64→1
- Match result_6/result_7 architectures so numbers are comparable

### CV schemes

Three flavors of 5-fold CV, 5 seeds each:

1. **Random split** — matches Tsuboyama/Cho-reported numbers; sanity check
2. **Domain-split** — hold out whole domains (analogue of gene-split)
3. **Family-split** — hold out whole Pfam families / MMseqs2-20 clusters (the honest test)

### Metrics

- **Spearman ρ** (primary, continuous)
- **Pearson r** (secondary)
- **AUROC** with ΔG binarized at the median, for direct comparison to pathogenicity (0.886) and ProteinGym (0.72)
- **Per-domain Spearman distribution** (analogue of result_17 / result_18 per-stratum analyses)

### Baselines

- **FoldX ΔΔG** — physics baseline; result_1 used this for Path B, now we have ground truth to validate
- **ThermoMPNN** — reported on Megascale; ~0.7 Spearman in their paper
- **Conservation only** — single-position likelihood from ESM-2 zero-shot

---

## Connection to existing results

| Existing result | What stability control adds |
|---|---|
| **result_1 stability subspace (Path B failed)** | If Path A works on Megascale, retroactively explains why Path B failed: FoldX-on-Gerasimavicius was too noisy. If Path A also fails, stability projection is fundamentally a dead end. |
| **result_6 pathogenicity positive control** | Adds a physical-label control; closes the "your one control might be curation-circular" objection |
| **result_17 / 18 AM family-robust vs ProteinGym wide** | Tests whether the wide per-assay variance is AM-specific or a general PLM property on physical labels |
| **result_19 perturbation patterns** | If stability is the signal that survives family-split in deltas, the spatial pattern features may be partly stability-driven |

---

## What would change about the paper

**If ROBUST (expected):** strongest version of the central claim — "ESM-2 encodes local biochemistry (pathogenicity AND stability) at family-robust AUROC ~0.85+; it does not encode mechanism." Two positive controls instead of one. Reviewer-bulletproof.

**If HETEROGENEOUS:** sharper claim — "ESM-2 works on average for physical properties but heterogeneously across folds. The fold-level variance is the same diagnostic as result_18 surfaced for AlphaMissense. Family-split CV plus per-stratum distribution is the right two-axis evaluation."

**If LEAKY (unexpected):** would force a rewrite. Means even physical properties partly depend on family-recognition in ESM-2 — a stronger negative claim about frozen embeddings than we've made.

**If NULL (very unexpected):** undermines the project's framing. Pathogenicity-only competence would suggest ESM-2 learned clinical-curation patterns, not biochemistry. High-value finding either way.

---

## Engineering plan

1. **Phase 1 — data prep (~30 min, local CPU):**
   - Parse concat CSV, restrict to single-point missense, build (wt_seq, mut, dG) table
   - Pfam-assign each unique WT sequence
   - Save aligned (sequence, mutation, dG, pfam, cluster) table

2. **Phase 2 — embeddings (GPU, ~2–4 h on A100):**
   - Run ESM-2 650M on all WT/mutant pairs
   - Cache delta_mean and delta_pos per variant

3. **Phase 3 — probes (CPU, ~30 min):**
   - Ridge + MLP under 3 CV schemes × 5 seeds
   - Per-domain Spearman aggregation
   - Pre-registered decision rule fires

4. **Phase 4 — writeup:**
   - `docs/result_20.md` (or next available)
   - Match the result_6 structure: TL;DR, primary table, per-domain distribution, dissociation comparison, novelty calibration

---

## Open questions before running

- Do we want to test **only** Megascale single-domain stability, or also include destabilising-vs-stabilising classification on the Cho 2026 benchmark splits? (Latter gives cleanest comparison to their predictor; former is sufficient for the positive-control claim.)
- Do we want a within-domain split too? (Hold out positions within each WT — tests whether the model generalises to unseen residues within a known fold, not just across folds.)
- Pfam coverage on mini-proteins is going to be low (~50%?). Fallback to MMseqs2-20 cluster is mandatory; should also report results restricted to Pfam-annotated subset for cleaner comparison to the rest of the project.

---

## Files (to be created)

- `scripts/megascale_stability.py` — full pipeline
- `data/megascale_variants.json` — parsed (seq, mut, dG, pfam, cluster) records
- `data/megascale_domain_families.json` — domain → Pfam/cluster mapping
- `data/embeddings/megascale_{wt,mut}_{mean,pos}_n*.npy` — cached embeddings
- `results/megascale_stability/{random,domain,family}_seed{0..4}.json` — per-CV per-seed metrics
- `results/megascale_stability/summary.json` — 5-seed aggregated
- `docs/result_20.md` — writeup
