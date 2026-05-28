# Plan — Experiment 11: Per-variant ESM-2 + gene-level proteome features

**Date drafted:** 2026-05-25
**Status:** Pre-registration
**Builds on:** results 1–10 (especially result_7 cross-family MLP floor, result_9 contrastive lift, result_10 clan-holdout)

**Execution order:** Pilot → Gate 1 (V2) → Gate 2 (V3) → Gate 3 (V4). Each stage gates the next; see "Staged execution" section below.

---

## Motivation

Results 1–10 established that frozen ESM-2 delta embeddings carry a partially-generalising mechanism signal, but DN (dominant negative) remains essentially unrecoverable from sequence alone. The mechanistic interpretation from result_10 is that DN biology lives at the complex-assembly level, which ESM-2 cannot see from sequence.

This experiment tests whether **gene-level proteome features** (protein abundance, turnover, interactome stoichiometry, PTM density, dosage constraint) carry the missing assembly/dosage information that ESM-2 lacks — and whether combining them with ESM-2 delta surfaces real DN signal.

This stays inside the existing project framework (ML, family-split CV, ESM-2 retained) and directly tests a mechanistically motivated hypothesis derived from prior results.

---

## Hypothesis

ESM-2 delta encodes a disruptiveness gradient (separates LOF from non-LOF). Gene-level proteome features encode the orthogonal information needed for DN (complex stoichiometry, oligomerization) and HI (dosage sensitivity). Concatenating the two feature classes should yield additive lift, with the largest gain on DN and HI.

---

## Staged execution

The experiment runs as a series of gated stages. Each stage is cheap to execute and produces a binary decision about whether to proceed. This avoids spending time on full data pulls or expensive model variants until earlier stages confirm signal exists.

### Stage 0 — Pilot (de-risk before any real work)

**Goal:** confirm the pipeline works and that *any* mechanism signal is detectable from a minimal gene-context feature set.

**Features (4–5 only, all public, no registration, single small download):**
1. pLI (gnomAD v4 constraint)
2. LOEUF (gnomAD v4 constraint)
3. mis_z (gnomAD v4 missense Z-score)
4. paralog_count (Ensembl Compara via BioMart, single bulk query)
5. *(optional)* tissue_τ (Human Protein Atlas tissue specificity)

**Genes:** all 1,985 merged-dataset genes (or 948 Gerasimavicius subset for fastest first pass).

**Model:** logistic regression + tiny MLP (16→8→3) on those 4–5 features. Family-split CV, single seed.

**Compute:** ~20 minutes total on a laptop, including data downloads. No GPU.

**Script:** `scripts/proteome_pilot.py` — single self-contained file (data pull + model).

**Decision rules:**
- **V2-pilot family-split macro-F1 ≥ 0.40** → strong signal from constraint alone; proceed to full Phase 1 with confidence
- **V2-pilot ≈ majority baseline (~0.31)** → constraint alone is insufficient; still proceed to Phase 1 because abundance / interactome / PTM are the real targets, but expectations are lower
- **V2-pilot well below majority** → pipeline bug; debug before any further work

Per-class breakdown also collected to check whether HI/LOF separation moves first (expected from dosage features) — sanity check on the biology.

### Stage 1 — Gate 1: V2 (proteome features only, full feature set)

Only run after Stage 0 passes the bug check.

**Inputs:** full ~30-feature matrix from Phase 1+2.
**Gate criterion:** V2 family-split macro-F1 ≥ 0.35 (clearly above majority by ≥0.04).
- **Pass** → proceed to Gate 2 (V3)
- **Fail** → proteome features alone don't carry mechanism; combining with ESM-2 is unlikely to manufacture signal. Stop, document V2 as a standalone negative result, reassess.

V2 is *its own potentially publishable result*: if experimental proteomics features alone predict mechanism well, that competes directly with Badonyi's AF2-derived features.

### Stage 2 — Gate 2: V3 (ESM-2 delta + proteome, concatenated)

Only run after Gate 1 passes.

**Gate criterion:** V3 ≥ max(V1, V2) + 0.02 macro-F1 → genuine additivity beyond either feature class alone.
- **Pass** → features are complementary; proceed to Gate 3 (V4) for contrastive lift
- **Fail** → features are redundant; ESM-2 and proteome encode overlapping information. Document and stop.

### Stage 3 — Gate 3: V4 (contrastive head on concatenated features)

Only run after Gate 2 passes. This is the most expensive variant; gating it behind V3 success avoids wasted compute on uninformative concatenations.

---

## Feature collection (Phase 1)

For each of the 1,985 genes in the merged dataset, collect the following gene-level features. Listed in expected coverage order (high → low):

| # | Source | Feature(s) | Coverage est. | Access |
|---|---|---|---|---|
| 1 | gnomAD v4 constraint | pLI, LOEUF, mis_z | ~98% | Bulk TSV: `https://gnomad.broadinstitute.org/downloads` |
| 2 | Ensembl Compara | paralog count | ~95% | BioMart or REST API |
| 3 | Human Protein Atlas | tissue specificity τ, n_tissues_expressed | ~90% | Bulk TSV: `proteinatlas.org/about/download` |
| 4 | PaxDb (integrated human) | log_abundance_ppm, abundance_rank | ~80% | `pax-db.org`, integrated H. sapiens dataset |
| 5 | BioPlex 3.0 | PPI_degree, n_complexes | ~70% | `bioplex.hms.harvard.edu/downloads` |
| 6 | Mathieson 2018 (Nat Comms) | protein_half_life_hr | ~60% | Supplementary table from paper |
| 7 | PhosphoSitePlus | n_phospho_sites, n_ubiq_sites, n_acetyl_sites | ~50% | Registration required, bulk download |
| 8 | ClinGen dosage sensitivity | HI_score, TS_score | ~30% (high value for HI) | `search.clinicalgenome.org/kb/gene-dosage` |

Expected final feature vector dimensionality: ~30 columns (continuous features + binary missingness indicators).

**Missing-data policy:** for each numerical feature, add a paired binary `<feature>_missing` indicator. Impute missing values with column median. This way the model can learn whether missingness itself carries signal (it often does — uncharacterised genes are systematically different).

**Coverage-skew sanity check (pre-Phase-1 gate, mandatory):**
Before committing to the full feature pull, compute class balance within "fully covered" genes (non-null across all sources) vs the overall labeled set. If the GOF/DN/LOF distribution in the covered subset differs from the overall distribution by more than ~10 percentage points in any class, missingness is confounded with mechanism (e.g. uncharacterised genes are systematically LOF because LOF genes are less studied). In that case:
- Either restrict the experiment to the fully-covered subset and report on that (smaller N, cleaner labels)
- Or use only the higher-coverage feature classes (gnomAD constraint + paralogs + HPA + PaxDb)

Document the coverage table and the chosen scope in result_11.md regardless of outcome.

**Output artifact:** `data/gene_proteome_features.tsv` — one row per gene, ~30 columns.

---

## Feature engineering (Phase 2)

Script: `scripts/build_proteome_features.py`

Steps:
1. Load `data/merged_gene_list.tsv` and `data/merged_valid_variants.json` to get the target gene universe.
2. Read each raw source file, normalise to gene symbol (UniProt → HGNC mapping where needed).
3. Inner-join all sources on gene symbol, applying the missing-data policy above.
4. Z-score continuous features (fit scaler on training fold during modelling — *do not* leak across folds; save unscaled values here).
5. **Add family-mean-centered duplicates of every continuous feature (mandatory).** For each numerical feature `f`, compute `f_familyresid = f - mean(f over all genes in the same Pfam family)`. Save *both* the raw and the family-centered version. This is computed using feature values only (no labels) so it does not leak label information across folds. Rationale: pLI, LOEUF, abundance, paralog_count, PPI degree are all correlated within Pfam families, and under family-split CV the model could shortcut "family-typical profile → mechanism" without ever generalising to new families. Including the family-residual lets the model learn within-family deviations that *can* generalise. The leakage diagnostic (gene-split − family-split delta) becomes a confirmation rather than the only safeguard.
6. Save:
   - `data/gene_proteome_features.tsv` (human-readable, gene × feature, raw + residual)
   - `data/proteome_features_aligned.npy` (numpy matrix aligned to `merged_gene_list.tsv` row order)
   - `data/proteome_feature_columns.json` (column names, raw vs residual tag, missingness indicators)

**Variant-level broadcast** happens at modelling time: for each variant, look up its gene's row and concatenate to the ESM-2 delta embedding.

**Singleton families:** Pfam families with only one labeled gene cannot be family-centered (residual = 0 by construction). Set those residuals to 0 explicitly and add an `is_singleton_family` indicator so the model knows the residual is uninformative for those rows.

---

## Modelling (Phase 3)

Script: `scripts/proteome_mechanism.py`, mirroring the structure of `experiment_mlp.py` and `contrastive_mechanism.py`.

Four model variants, all under identical 5-fold family-split CV with 5 seeds (seeds 0–4):

| Variant | Features | Architecture | Reference baseline |
|---|---|---|---|
| **V1** | ESM-2 delta only (1280) | MLP 1280→256→64→3 | result_7 = 0.352 macro-F1 |
| **V2** | Proteome features only (~30) | MLP 30→64→32→3 | New baseline |
| **V3** | ESM-2 delta + proteome (concat) | MLP 1310→256→64→3 | Main test |
| **V4** | Contrastive head on V3 inputs | 1310→256→64 projection, TripletMarginLoss (same as result_9) + kNN | **result_9 = 0.387 macro-F1 (primary V4 comparator)** |

**Primary V4 comparison:** V4 vs result_9 (0.387), not vs V1. result_9 already demonstrated contrastive lift on ESM-2 alone; V4 only succeeds if it pushes beyond that. A V4 result of, say, 0.40 macro-F1 would be a +0.013 lift from adding proteome features to the contrastive setup — small. To call V4 a success, it needs to clear result_9 + 0.03 = 0.417.

All variants use:
- Family-split CV (Pfam family holdout) — primary metric
- Gene-split CV — secondary, to quantify leakage delta
- 5 seeds → report mean ± std
- Same class-weighting and early stopping as `experiment_mlp.py`

---

## Diagnostics (Phase 4)

The headline number is family-split macro-F1, but the per-class breakdown is where the scientific interpretation lives.

**Per-class AUROC analysis under family-split CV:**

| Question | Metric to inspect |
|---|---|
| Does DN AUROC jump with proteome features? | DN AUROC V3 − DN AUROC V1 |
| Does GOF AUROC improve with PTM density? | GOF AUROC, PTM-feature ablation |
| Does LOF separation hold or improve? | LOF AUROC V3 vs V1 |

**Feature ablation:** drop each proteome feature *class* (constraint, abundance, interactome, PTM, dosage) one at a time from V3; rank by ΔF1 to identify which feature class carries the lift. Report as a table in result_11.md.

**Leakage diagnostic:** compute (gene-split F1 − family-split F1) for V3 and compare to result_7/result_9. If proteome features cause this gap to widen, they are introducing new family-level leakage and need re-evaluation.

---

## Pre-registered decision rules

| Outcome | Threshold | Interpretation | Action |
|---|---|---|---|
| **Strong positive** | V3 family-split F1 ≥ 0.45 AND DN AUROC ≥ 0.65 | Proteome features rescue ESM-2 mechanism prediction; assembly information is the missing modality for DN | Headline finding; restructure paper around this |
| **Moderate positive** | V3 family-split F1 ≥ 0.42 OR DN AUROC ≥ 0.60 | Proteome features add real lift but don't fully solve DN | Add as final result chapter, sharpens story |
| **Modality identification** | Specific feature class shows ≥0.03 lift in ablation | One feature class (e.g. interactome) carries the signal | Biology paragraph in result_11.md |
| **Null** | V3 family-split F1 < 0.40 AND DN AUROC < 0.55 | Cell biology can't compensate for what frozen pLM misses; structural features (Badonyi) may be required | Publish as a clean negative bound: "even proteome priors don't rescue frozen pLM mechanism" |

Every outcome is publishable; the framing changes but the experiment is informative either way.

---

## Compute & timeline

| Phase | Duration | Compute | Cost |
|---|---|---|---|
| 1. Feature collection (scripts + manual registration for PhosphoSitePlus) | ~1 day | Local CPU | $0 |
| 2. Feature engineering | ~½ day | Local CPU | $0 |
| 3. Modelling (4 variants × 5 seeds × 5 folds × 2 CV schemes) | ~½ day | Local CPU sufficient (cached embeddings) | $0–$5 |
| 4. Diagnostics + writeup | ~½ day | Local CPU | $0 |
| **Total** | **~3 days** | | **<$10** |

---

## Artifacts produced

- `scripts/build_proteome_features.py` — feature collection + alignment
- `scripts/proteome_mechanism.py` — 4 model variants under family-split CV
- `data/gene_proteome_features.tsv` — human-readable feature table
- `data/proteome_features_aligned.npy` — aligned numpy matrix
- `data/proteome_feature_columns.json` — column metadata
- `results/.../proteome_mechanism_seed{0..4}.json` — per-seed metrics
- `docs/result_11.md` — writeup mirroring result_9/10 structure

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Low gene coverage after intersecting all sources | Medium | Use missingness indicators rather than dropping genes; report coverage per feature |
| Proteome features just recapitulate pLI/LOEUF | Medium | Ablation table will show this; report pLI/LOEUF-only baseline as a reference |
| Family-split leakage via paralog-shared gene features | High | **Design-level mitigation (mandatory):** family-mean-centered residuals are added alongside raw features in Phase 2, so the model has within-family deviations available. The gene-split vs family-split delta diagnostic becomes a *confirmation* of the design, not the only safeguard. If the gap is still wide after this, do a residuals-only ablation (drop raw features entirely) to see whether any cross-family signal survives. |
| PhosphoSitePlus registration delay | Low | Drop it if blocked; rest of features are public |
| Cross-dataset gene symbol mismatches | Medium | Use HGNC canonical mapping; UniProt as join key where possible |

---

## What this does NOT do

- Does not pursue Badonyi-style structural features (AF2-derived). That is a separate direction.
- Does not fine-tune ESM-2. V4 only trains a small projection head; the base model stays frozen.
- Does not address GOF mechanism heterogeneity beyond the PTM-density feature.
- Does not increase the labeled gene set. Coverage stays at ~1,985 genes from the merged dataset.

---

## Sign-off checklist before running

- [ ] Plan reviewed by dami
- [ ] Decision rules accepted as pre-registered
- [ ] Phase 1 access confirmed (PhosphoSitePlus registration submitted if pursued)
- [ ] Existing scripts (`experiment_mlp.py`, `contrastive_mechanism.py`) compatible with concat input
- [ ] Compute budget approved (<$10)
