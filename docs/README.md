# esm2_mechanism — results index

**This is a standalone research project, not an AI Scientist run.** All experiments were designed and executed manually. The code lives in `esm2_mechanism/scripts/` and was run directly on RunPod (A100 80GB). The project will likely move to its own repository.

Seven `result_*.md` files written across May 23–25, 2026. Read in the order below for the coherent narrative arc. Results 3 and 5 are superseded by result 7.

---

## Reading order

### 1. `result_1.md` — Baseline: linear probe sets up the puzzle
**Script:** `experiment.py` · **Run:** May 23–24, gene-split CV on 10,231 variants (948 genes)
**Headline numbers:** Linear delta_mean macro-F1 = **0.279** (chance). WT-only mysteriously = **0.580**. Per-residue delta (0.373) beats mean-pooled.
**What it concludes:** Linear delta probe finds no mechanism signal; WT-only's 0.58 is the suspicious number that needs explaining.
**Open question after reading:** *Is WT-only's 0.58 a real gene-level mechanism signal, or homology leakage from gene-split CV?*

### 2. `result_2.md` — Gene-split vs family-split baselines
**Script:** `family_split_baselines.py` · Same embeddings as result 1, 8 feature sets × 2 CV schemes
**Headline numbers:** WT-only macro-F1 **collapses** 0.580 → 0.389 under family-split (Δ = +0.191). Delta probe stays flat. GOF AUROC = **0.801** under family-split WT-only.
**What it concludes:** Most of WT-only signal is paralog leakage. AlphaMissense carries zero mechanism information.
**Open question after reading:** *Is there any nonlinear mechanism signal in the delta the linear probe couldn't see?*

### 3. `result_3.md` — MLP probe on delta (FIRST pass) ⚠ SUPERSEDED BY RESULT 7
**Script:** `experiment_mlp.py` · MLP probe (256→64, dropout 0.3) on delta_mean, gene-split only
**Headline numbers:** MLP delta_mean macro-F1 = **0.414**.
**⚠ Why this is superseded:** No family-split CV — the 0.414 is gene-split only and includes family leakage. Result 7 provides the correct family-split number (0.364) and calibration against chance.

### 4. `result_4.md` — Family clustering: the causal explanation
**Script:** `family_clustering.py` · Pfam clustering analysis on WT, mut, delta embeddings
**Headline numbers:** k=5 family purity = **26× chance** (z = +78). 50-way family probe = **27× majority baseline**. **74.8%** of genes share their family's majority mechanism.
**What it concludes:** WT-only signal explained by family recognition × family-mechanism correlation. Includes novelty assessment (2/5 — folk wisdom, but not yet demonstrated as a controlled comparison).

### 5. `result_5.md` — Nonlinear probes (MLP/kNN/GBM/RF) ⚠ PARTIALLY SUPERSEDED BY RESULT 7
**Script:** `experiment_mlp.py` extended · 4 probes on delta_mean and delta_pos, gene-split only
**Headline numbers:** MLP = 0.431, kNN = 0.410, GBM = 0.336, RF = 0.292.
**⚠ Limitation:** Gene-split only. Result 7 provides the family-split calibration showing 62% of the gene-split lift is leakage.

### 6. `result_6.md` — Pathogenicity positive control
**Script:** `pathogenicity_control.py` · 17,236 ClinVar pathogenic/benign variants, 944 genes
**Headline numbers:** Pathogenicity MLP AUROC = **0.878**, family-split Δ = **0.002**.
**What it concludes:** Pipeline is sound. Pathogenicity AUROC 0.88 (family-split-stable) vs mechanism floor ~0.39 (family-split) — **the dissociation is sharper than originally framed** (see result 7 for correction).

### 7. `result_7.md` — Full calibration: all numbers, honest framing
**Scripts:** `experiment_mlp.py` with family-split, `build_merged_dataset.py`, Option B gene-level WT
**Headline numbers:**
- MLP delta gene-split **0.415**, family-split **0.364** (+0.031 above chance; 62% of lift is leakage)
- Gene-level WT merged dataset family-split **0.393**, GOF AUROC **0.728**
- Always-predict-LOF baseline: **0.279** (Gerasimavicius), **0.311** (gene-level merged)
- Family-split floor **~0.39** consistent across 3 different setups (per-variant/gene-level, 2 datasets, linear/MLP)
**What it concludes:** The ~0.39 floor is real but small. The pathogenicity-mechanism dissociation is sharper than result 6 suggested. The GOF AUROC (0.73–0.80 family-split) is the strongest individual signal. See PUBLISH.md for v1 paper plan built around this.

---

## The coherent story across all 7

1. **(1)** Linear probe at chance on delta; WT-only at 0.58 — suspicious.
2. **(2)** Family-split: WT-only collapses (0.58→0.39). Delta stays flat. GOF AUROC 0.80 survives.
3. **(4)** Family clustering explains the collapse: ESM-2 encodes family identity, family correlates with mechanism.
4. **(6)** Positive control: pathogenicity AUROC 0.88, family-split-stable. Pipeline works when signal is there.
5. **(7)** Full calibration: ~0.39 family-split floor across all setups. 62% of gene-split signal is leakage. Dissociation with pathogenicity is **sharper** than originally thought, not narrower.

**Publication plan:** See `PUBLISH.md` — v1 focuses on the GOF AUROC survival under family-split, framed as a frozen-representation interpretability result. Not AI Scientist output; manual research.

---

## Supporting docs

- `EXPERIMENT.md` — Pre-registration document (original hypothesis and thresholds)
- `PUBLISH.md` — Publication plan: v1/v2/v3 versioned bioRxiv strategy
- `explain.txt` — Plain-English explanation of the experiment design
- `progress_notes.md` — Running log of decisions, bugs fixed, observations
- `scripts/README.md` — What each script does and when to use it

---

## Companion data

| Result | JSON file |
|---|---|
| 1 | `results/20260524_baseline_run/run_0/final_info_seed0.json` |
| 2 | `results/20260524_baseline_run/run_0/family_split_baselines.json` |
| 3, 5 | `results/20260524_baseline_run/run_0/mlp_results_seed0.json` |
| 4 | `results/20260524_baseline_run/run_0/family_clustering.json` |
| 6 | `results/20260524_baseline_run/run_0/pathogenicity_control.json` |
| 7 | `results/20260524_baseline_run/run_0/option_b_gene_level_wt_merged.json` + merged dataset MLP (pending) |

Embeddings under `data/embeddings/`:
- `embeddings_{wt,mut}{,_pos}_esm2_t33_650M_UR50D.npy` — Gerasimavicius (results 1–5, 7)
- `merged_embeddings_{wt,mut}_{mean,pos}.npy` — merged 1,985-gene dataset (result 7)
- `emb_{wt,mut}_mean_pathogenicity_*.npy` — ClinVar pathogenicity set (result 6)

---

## Highest-priority next experiments

1. **MLP delta on merged dataset** — **running on RunPod now** (19,100 variants, 1,985 genes, 1,146 families). Result will fill in the merged-dataset family-split floor for delta.
2. **Multi-seed replication** — all numbers seed=0 only. 5 seeds needed before posting.
3. **The figure** — bar chart: per-class AUROC × CV scheme × dataset (see PUBLISH.md).
4. **Within-family analysis** — is mechanism learnable within a single Pfam family? (potential positive flip side for v2)
5. **DDG2P / SaProt replication** — generalisation evidence for v2/v3.
