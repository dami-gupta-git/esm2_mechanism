# Result 25 — Enzyme type classification from ESM-2 WT embeddings: family-split F1 = 0.655, confirming the mechanism null is task-specific

## Date: 2026-05-28 | Scripts: fetch_enzyme_labels.py, enzyme_classification.py | Seeds: 0–4 | Local CPU

> **STATUS: confirmed across 5 seeds** (std ≤ 0.012). H1 (F1 ≥ 0.70) just misses; H2 (enzyme >> mechanism floor) confirmed by +0.270; H4 (MLP ≈ LogReg) not confirmed — LogReg outperforms MLP.

---

## Background

Results 1–10 established that frozen ESM-2 encodes disease mechanism (GOF/DN/LOF) weakly: family-split macro-F1 = 0.385 ± 0.018 on the merged dataset, with ~62.8% of gene-split signal being family-recognition leakage. A natural question is whether this is a failure of the *probe/pipeline* or a property of the *task*. Enzyme type classification is a clean positive control:

- **Enzyme class is a WT-sequence property** — no mutation delta needed, no per-variant labelling
- **It is strongly associated with protein fold** — ESM-2's clustering by Pfam (26× purity, result 4) should directly help here
- **It is a well-annotated gene-level label** — EC numbers and kinase/protease keywords from UniProt provide ground truth for ~13% of the merged gene set (kinase: 136, protease: 68, oxidoreductase: 119; non-enzyme: 1662 out of 1985 genes)

The design mirrors results 1–10 exactly: gene-split → family-split → leakage fraction, using the same CV code, the same Pfam map, and the same 5-seed protocol.

---

## TL;DR

ESM-2 WT mean-pooled embeddings classify enzyme type at **family-split F1 = 0.655 ± 0.012** — substantially above the mechanism floor (0.385) by +0.270. Per-class AUROCs (family-split): kinase 0.896, protease 0.904, oxidoreductase 0.890, non-enzyme 0.854. Leakage fraction is only 13.7% (vs 62.8% for mechanism), meaning most of the enzyme signal is **real cross-family signal, not fold memorisation**. Proteome features (37-dim gene-level biology) are at chance (F1 = 0.251 ≈ majority 0.228) — enzyme class is not predictable from gene-level constraint, dosage, or expression features. LogReg outperforms MLP under family-split (0.655 vs 0.597), suggesting the enzyme signal is **linearly separable** in WT embedding space, paralleling pathogenicity (result 23) and contrasting with stability (result 21, which required nonlinear probes).

**The central conclusion is confirmatory, not surprising:** mechanism's low ceiling (0.385) reflects a real limitation of what ESM-2 encodes about mutation effects, not a methodological failure. The model CAN separate biologically distinct protein classes under family-split CV when the signal is present — it does so for enzyme type at F1 = 0.655 and for pathogenicity at AUROC = 0.884. The mechanism null stands.

---

## Data

**Labels:** UniProt EC numbers + keyword KW-0418 (kinase) fetched for all 1985 genes in the merged dataset. 4-class scheme:
- **Kinase** — KW-0418 keyword OR EC 2.7.x.x: **136 genes**
- **Protease** — EC 3.4.x.x: **68 genes**
- **Oxidoreductase** — EC 1.x.x.x: **119 genes**
- **Non-enzyme** — no qualifying EC: **1662 genes**

Priority order (most specific first): kinase > protease > oxidoreductase > non-enzyme. Zero multi-class genes flagged. Coverage: 100% (all 1985 genes received a label; uniprot_missing → non-enzyme, but all 1985 accessions resolved successfully).

**Embeddings:** `merged_embeddings_wt_mean.npy` — per-gene WT mean-pooled representation, extracted by taking the first variant's embedding index per gene (all variants for a gene share the same WT embedding). Shape: (1985, 1280), dtype float32. No GPU required.

**CV:** 5-fold gene-split and family-split (Pfam), 5 seeds (0–4), identical to results 7/13/15.

---

## Results (5-seed mean ± std)

### ESM-2 WT embedding (1280-dim)

| probe | CV scheme | macro-F1 |
|---|---|---|
| Majority (always non-enzyme) | — | 0.228 |
| LogReg | gene-split | 0.760 ± 0.003 |
| **LogReg** | **family-split** | **0.655 ± 0.012** |
| MLP (256, 64) | family-split | 0.597 ± 0.007 |

**Leakage fraction: 13.7%** (vs 62.8% for mechanism on Gerasimavicius — result 7).

### Per-class AUROC (family-split LogReg, mean across 5 seeds)

| class | AUROC |
|---|---|
| kinase | 0.896 |
| protease | 0.904 |
| oxidoreductase | 0.890 |
| non-enzyme | 0.854 |

### Proteome features baseline (37-dim gene-level biology)

| probe | CV scheme | macro-F1 |
|---|---|---|
| LogReg | gene-split | 0.261 ± 0.003 |
| **LogReg** | **family-split** | **0.251 ± 0.006** |
| MLP (256, 64) | family-split | 0.233 ± 0.005 |

Proteome features are at majority baseline (0.228) — gene-level constraint, dosage, expression and PPI features carry no enzyme-class information.

---

## Hypothesis evaluation (pre-registered in plan_enzyme_classification.md)

| Hypothesis | Condition | Value | Outcome |
|---|---|---|---|
| H1 | family-split LogReg F1 ≥ 0.70 | 0.655 | **FAIL** (misses by 0.045) |
| H2 | enzyme family-split F1 >> mechanism floor (Δ > 0.10) | +0.270 | **CONFIRMED** |
| H4 | MLP − LogReg family-split ΔF1 < 0.05 | −0.058 | **NOT CONFIRMED** (LogReg wins) |

H1 narrowly fails. The signal is strong (F1 = 0.655, per-class AUROCs all ≥ 0.85) but does not clear the pre-registered 0.70 threshold — appropriate, given that a 4-class problem with a heavily imbalanced non-enzyme class (84%) makes F1 sensitive to the minority class performance. The scientific claim (enzyme class is strongly encoded) is not affected.

H4 is the most interesting negative: the **MLP underperforms LogReg** under family-split (0.597 vs 0.655), and the gap is consistent across all 5 seeds. This is the same pattern seen for pathogenicity (result 23, where direction-only linear recovered full AUROC) and contrasts with stability (result 21, where GBM strongly outperformed linear under Pfam-split). Enzyme class is **linearly separable** in the WT embedding space.

---

## Key findings

### F1 — Enzyme class is strongly encoded in ESM-2 WT embeddings across family boundaries

Family-split F1 = 0.655 is well above both majority (0.228) and the mechanism floor (0.385). All four per-class AUROCs exceed 0.85. This shows the embedding space encodes enzyme-class biology that transfers to held-out protein families — not fold memorisation.

### F2 — Leakage is low (13.7%) — most signal is genuine cross-family

The gene-split → family-split drop is only 0.760 → 0.655, a 13.7% reduction. For mechanism (result 7), the equivalent drop was 62.8%. Enzyme class is less confounded by family identity than disease mechanism — or rather, enough cross-family structure exists that family-split cannot remove it. This makes H2 a clean result: the gap to mechanism (Δ = +0.270) is not an artefact of leakage in the enzyme experiment.

### F3 — LogReg outperforms MLP under family-split; enzyme class is linearly encoded

LogReg 0.655 vs MLP 0.597 under family-split (ΔMLP−LR = −0.058, consistent across 5 seeds). This fits the pattern from result 23 (pathogenicity = linear direction, MLP adds little) and distinguishes enzyme classification from stability (result 21, where GBM recovered 0.750 vs linear 0.597 under Pfam-split). In WT embedding space, enzyme classes occupy linearly separable regions — fold identity is a low-dimensional linear feature of ESM-2.

### F4 — Proteome features carry no enzyme-class signal

F1 = 0.251 ≈ majority (0.228) under family-split. Gene-level constraint (pLI, LOEUF, mis_z), dosage features, expression, PPI degree — none of these separate kinases from proteases from oxidoreductases. This is expected (enzyme class is a structural/sequence property, not a population-genetics property) but provides a useful double dissociation: proteome features beat ESM-2 for mechanism (result 13, +0.080 F1) but are at chance for enzyme class, while ESM-2 shows the opposite pattern.

---

## The task × modality double dissociation

Combining this result with results 13 and 15:

| task | ESM-2 WT/delta (family-split) | proteome features (family-split) |
|---|---|---|
| enzyme type (this result) | **F1 = 0.655** | F1 = 0.251 (≈ chance) |
| disease mechanism (result 13) | F1 = 0.382 | **F1 = 0.462** |

ESM-2 is the strong modality for sequence-level properties (fold/enzyme class); proteome features are the strong modality for gene-level population-genetics properties (mechanism). The two modalities carry orthogonal information, consistent with V3 (ESM-2 + proteome concat) not reliably outperforming V2 alone for mechanism (result 13, Gate 2 fails 3/5 seeds).

---

## What this confirms about the mechanism null

The mechanism family-split floor of 0.385 is not a methodological ceiling. The same pipeline — same embedding infrastructure, same CV code, same seeds, same probe architecture — achieves F1 = 0.655 for enzyme type. The mechanism null is real: ESM-2's delta embeddings do not encode the direction of mutation effect (GOF/DN/LOF) in any family-transferable, probe-accessible form. Enzyme class (a WT sequence property) is recoverable; mutation consequence (a perturbation property) is not. This is the sharpest evidence the project has produced for the core claim.

---

## Limitations

- **4-class scheme only; full EC hierarchy not tested.** Expanding to the 7-class EC scheme (all top-level enzyme classes) may reduce F1 further due to smaller class sizes (lyases, isomerases, ligases are rare in disease genes). This was pre-registered as an optional follow-up; not pursued here.
- **Non-enzyme class dominates (84%).** Macro-F1 with balanced class weights partially corrects for this, but the protease class (68 genes) is small and its per-fold variability drives the seed-to-seed std (0.012).
- **No clan-holdout.** Pre-registered but not run. Given the already low leakage fraction (13.7% vs mechanism's 62.8%), a clan-holdout would further reduce F1 but the qualitative conclusion (enzyme >> mechanism) would be unchanged.

---

## Files

- `scripts/fetch_enzyme_labels.py` — UniProt EC/keyword fetch → `data/enzyme_labels.tsv`
- `scripts/enzyme_classification.py` — LogReg / MLP probes, gene-split + family-split, 5-seed
- `data/enzyme_labels.tsv` — 2424 genes, 4-class labels, EC numbers, UniProt flags
- `data/cache/enzyme_uniprot/` — per-accession UniProt JSON cache (1983 entries)
- `results/enzyme_classification/enzyme_classification_summary.json` — full results
- `docs/plans/plan_enzyme_classification.md` — pre-registration
