# Plan — Enzyme Type Classification from ESM-2 Embeddings

**Date drafted:** 2026-05-28
**Status:** Pre-registration
**Builds on:** Results 1–10 (ESM-2 embedding infrastructure, family-split CV framework, `utils_probes.py`)

---

## What this experiment is (plain English)

The core project has established that ESM-2 embeddings encode pathogenicity strongly (AUROC 0.74–0.88) but disease mechanism (GOF/DN/LOF) weakly — and that most apparent mechanism signal under gene-split CV is family-recognition leakage. A natural follow-up is: what *can* ESM-2 classify correctly from sequence alone, when the label is a gene-level biological property?

Enzyme type classification is a clean positive control for this question. Enzyme classes — **protease, kinase, oxidoreductase, transferase, etc.** — are strongly associated with protein fold and active-site geometry. ESM-2 was pre-trained on ~250M sequences spanning all enzyme classes. Unlike disease mechanism (which requires understanding the *direction* of a missense effect), enzyme type is a **WT-sequence-level property**: you only need the wildtype embedding, no mutation delta.

The experiment tests:
1. Whether ESM-2 WT embeddings reliably separate enzyme classes in a linear probe
2. Whether that signal survives family-split CV — i.e., whether the model is recognising fold/sequence identity vs learning something more general
3. How performance degrades from gene-split → family-split → clan-holdout, directly paralleling the mechanism arc (results 4, 7, 10)

This also serves as a calibration for interpreting the mechanism results: if enzyme classification under family-split is near-perfect (as expected), it confirms that the mechanism floor of ~0.38 F1 is a real biological ceiling, not a methodological artefact.

---

## Why this is different from the mechanism experiments

In the mechanism experiments, the probe input is the **delta embedding** (mut − WT) and the label is a **per-variant property** (what this specific mutation does). Family leakage inflates results because WT embeddings cluster by family and 74.8% of genes share their family's modal mechanism.

Here, the probe input is the **WT mean-pooled embedding** and the label is a **gene-level property** (what enzyme class this protein is). There is no delta. The leakage concern is inverted: family-split leakage is *expected* to be large (enzyme class is almost perfectly correlated with fold family), and the interesting question is whether *any* cross-family generalisation exists, and whether a clan-holdout floor is interpretable.

The comparison with mechanism results (result 7) directly tests whether the mechanism null result reflects a property of the task (mechanism is not encoded) vs the input (delta embeddings are noisy). If enzyme classification is strong under family-split and mechanism is not, the contrast is informative.

---

## Data

### Labels — UniProt EC annotations

Fetch enzyme class annotations from UniProt for all genes in the existing `merged_gene_list.tsv` (1,985 genes). Use UniProt REST API (`https://rest.uniprot.org/uniprotkb/search?query=gene:<GENE>&fields=ec`).

EC numbers have 4 levels (e.g. `3.4.21.1` = serine protease). Map to top-level EC class:
- EC 1.x.x.x → Oxidoreductase
- EC 2.x.x.x → Transferase
- EC 3.x.x.x → Hydrolase (includes proteases at 3.4.x.x)
- EC 4.x.x.x → Lyase
- EC 5.x.x.x → Isomerase
- EC 6.x.x.x → Ligase
- EC 7.x.x.x → Translocase
- No EC entry → Non-enzyme

For the initial experiment, use a **4-class** simplified scheme:
- **Kinase** — subset of EC 2.7.x.x (phosphotransferases), identified by UniProt keyword `KW-0418`
- **Protease** — EC 3.4.x.x
- **Oxidoreductase** — EC 1.x.x.x
- **Non-enzyme** — no EC number

This 4-class setup is motivated by the mechanism literature (GOF is enriched in kinases and ion channels; DN in structural complexes) and by class size balance. Expand to full 7-class EC scheme if the 4-class experiment passes pre-registered gates.

**Expected class sizes** (rough, from UniProt coverage of human proteome):
- Kinase: ~500 genes (530 human kinases known)
- Protease: ~150–200 genes
- Oxidoreductase: ~300–400 genes
- Non-enzyme: ~1,000+ genes (majority of proteome)

Class imbalance will require balanced class weights in all classifiers, consistent with existing pipeline.

### Embeddings — WT only

Use existing `merged_embeddings_wt_mean.npy` for genes already in the merged dataset. For any genes in UniProt EC list not already embedded, generate WT embeddings using the existing `get_esm2_embeddings_for_pairs()` infrastructure (pass the same sequence as both WT and mut, extract WT side only).

No GPU required if restricting to the already-embedded gene set (~1,985 genes).

---

## Specific hypotheses (pre-registered)

**H1 — Family-split classification is strong:**
Linear probe (LogReg) on WT mean-pooled embeddings achieves macro-F1 ≥ 0.70 under family-split CV across the 4-class scheme. Motivated by: enzyme class is stereotyped by fold; ESM-2 WT embeddings cluster strongly by family (result 4, 26× purity); enzyme class is near-perfectly correlated with Pfam family.

**H2 — Family-split >> mechanism floor:**
Family-split enzyme classification F1 substantially exceeds the mechanism floor (0.385 ± 0.018 merged, 5-seed). If H1 passes, the contrast directly shows that the mechanism null result is task-specific, not a probe/data failure.

**H3 — Clan-holdout shows partial cross-fold generalisation:**
Leave-one-clan-out evaluation (mirroring result 10) gives enzyme F1 above majority baseline but below family-split. Expected given that some enzyme classes (e.g. oxidoreductases) span multiple structural superfamilies, providing genuine cross-fold signal.

**H4 — Linear probe is sufficient:**
MLP does not substantially outperform LogReg under family-split CV (ΔF1 < 0.05). This would parallel the pathogenicity result (where family-split Δ ≈ 0 for MLP vs LogReg) and contrast with the mechanism case (where MLP gave small lifts that evaporated under family-split).

---

## What we will measure

**Primary metric:** Macro-F1 under family-split CV (5-fold, 5-seed), 4-class (kinase / protease / oxidoreductase / non-enzyme). This is directly comparable to the mechanism F1 reported in results 6–7.

**Secondary metrics:**
- Per-class AUROC (one-vs-rest), consistent with `compute_metrics()` in `utils_probes.py`
- Gene-split vs family-split Δ F1 — quantifies leakage fraction, directly parallel to result 7's 62.8% figure
- Clan-holdout macro-F1 (parallel to result 10)
- MLP vs LogReg comparison under family-split (H4)
- Confusion matrix: which enzyme class pairs are most confused?

**Baselines:**
- Always-predict-majority (non-enzyme): sets the macro-F1 floor
- Random (1/4 = 0.25 per class for balanced evaluation)
- Proteome features alone (V2 LogReg) — does gene-level biology predict enzyme class?

---

## How we will do it

### Step 1 — Fetch EC annotations and build label file

Script: `scripts/fetch_enzyme_labels.py`

For each gene in `merged_gene_list.tsv`, query UniProt REST API for EC number and keyword annotations. Map to 4-class scheme. Handle:
- Genes with multiple EC numbers (e.g. bifunctional enzymes) — assign the EC class with the most specific annotation; flag as multi-class
- Genes with no UniProt match — assign `Non-enzyme` with a `uniprot_missing` flag
- Sequence variant isoforms — use canonical sequence UniProt ID

Output: `data/enzyme_labels.tsv` — columns: gene, uniprot_id, ec_number, ec_top_class, enzyme_4class, multi_class_flag, uniprot_missing_flag

### Step 2 — Align embeddings and labels

Restrict to genes present in both `merged_gene_list.tsv` (has embeddings) and `enzyme_labels.tsv` (has EC annotation). Use existing `merged_embeddings_wt_mean.npy` directly — no new GPU compute needed.

Output: aligned arrays `X_enzyme.npy` (N × 1280) and `y_enzyme.npy` (N,) with integer class labels.

### Step 3 — Linear probe under gene-split and family-split CV

Use `run_logreg_cv()` from `utils_probes.py` directly. Run:
- Gene-split CV (5-fold, 5-seed)
- Family-split CV (5-fold, 5-seed) using existing `family_split_cv()` with Pfam annotations from `data/pfam_families.json`

Compute leakage fraction = (gene-split F1 − family-split F1) / gene-split F1, parallel to result 7.

Script: `scripts/enzyme_classification.py`
Output: `results/enzyme_classification/logreg_summary.json`

### Step 4 — MLP probe under family-split CV

Use `experiment_mlp.py` architecture (1280→256→64→4, dropout 0.3, early stopping). Run family-split CV only (5-seed). Compare to LogReg (H4).

Output: `results/enzyme_classification/mlp_summary.json`

### Step 5 — Clan-holdout

Reuse `clan_holdout.py` pattern: leave-one-Pfam-clan-out evaluation across all clans with ≥ 5 genes and ≥ 2 represented enzyme classes. Report macro-F1 and per-class AUROC distributions across clans.

Output: `results/enzyme_classification/clan_holdout_summary.json`

### Step 6 — Proteome baseline (V2 LogReg)

Run existing `proteome_features_aligned.npy` (37-dim) through `run_logreg_cv()` for enzyme class prediction under family-split. This tests whether the same gene-level features that predict disease mechanism also predict enzyme class — if yes, they may be partially capturing enzyme-class biology rather than mechanism-specific signal.

Output: appended to `results/enzyme_classification/logreg_summary.json`

---

## Compute and timeline

| Phase | Duration | Compute |
|---|---|---|
| Step 1 — EC label fetch | ~1–2 hours | Local CPU (UniProt API) |
| Step 2 — alignment | ~15 min | Local CPU |
| Step 3 — LogReg CV | ~30 min | Local CPU |
| Step 4 — MLP CV | ~1–2 hours | Local CPU (no GPU needed for WT-only) |
| Step 5 — clan holdout | ~30 min | Local CPU |
| Step 6 — proteome baseline | ~15 min | Local CPU |
| Writeup | ~1 hour | — |
| **Total** | **~5–7 hours** | **Local CPU** |

No GPU required if restricting to the already-embedded gene set.

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Class imbalance makes macro-F1 misleading | Medium | Use balanced class weights throughout; report per-class AUROC separately |
| Non-enzyme class dominates and inflates baseline | Medium | Report macro-F1 (equally weights classes) not accuracy; run with non-enzyme excluded as sensitivity check |
| UniProt EC coverage is sparse for disease genes | Low-medium | Flag uniprot_missing genes; report coverage fraction; restrict primary analysis to EC-annotated genes |
| Enzyme class is too easy (ceiling effect) | Medium — expected | This is a positive control; ceiling is the expected result. The value is the contrast with mechanism, not a surprising F1 |
| Multi-class genes (bifunctional enzymes) confuse the classifier | Low | Flag and exclude from primary analysis; run sensitivity check including them |

---

## Pre-registered decision rules

| Outcome | Threshold | Interpretation |
|---|---|---|
| **H1 passes** | Family-split LogReg macro-F1 ≥ 0.70 | Enzyme class is strongly encoded in ESM-2 WT embeddings; confirms family-split CV is a meaningful discriminator |
| **H1 fails** | Family-split macro-F1 < 0.70 | Unexpected; would suggest either label quality issues or that the 4-class scheme spans too many structural superfamilies. Investigate per-class AUROCs and confusion matrix before concluding |
| **H2 confirmed** | Enzyme family-split F1 >> mechanism family-split F1 (0.385) | Mechanism null result is task-specific: ESM-2 WT embeddings encode enzyme class but delta embeddings do not encode mutation mechanism |
| **H4 confirmed** | MLP − LogReg family-split ΔF1 < 0.05 | Linear readout is sufficient; consistent with pathogenicity result and contrast with stability (result 21) |
| **H4 fails** | MLP − LogReg ΔF1 ≥ 0.05 | Nonlinear organisation in WT enzyme embedding space — interesting; would contrast with the pathogenicity result and would mirror the stability result (nonlinear cross-family structure) |

If H1 passes, proceed to extended 7-class EC scheme as an optional follow-up (not pre-registered here).

---

## Artifacts produced

- `scripts/fetch_enzyme_labels.py` — UniProt EC fetch and 4-class label assignment
- `scripts/enzyme_classification.py` — LogReg / MLP / clan-holdout probes, all in one script
- `data/enzyme_labels.tsv` — gene-level EC annotations with flags
- `results/enzyme_classification/` — JSON summaries for all probe runs
- `docs/result_25.md` — writeup
