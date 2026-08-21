# esm2_mechanism Progress Notes
## Session: May 23-24, 2026

---

## What This Experiment Is

Testing whether ESM-2 650M delta-embeddings (mutant - wildtype) encode gene-level dominant disease mechanism class (GOF / DN / LOF) in a way that is geometrically distinct from protein stability. Linear probe with gene-split CV, stability nuisance subspace removal, probe direction orthogonality analysis.

Full design: `EXPERIMENT.md`

---

## Dataset

### Gerasimavicius et al. 2022 (primary)
- NatComms 13:3895, OSF: 10.17605/OSF.IO/H62FQ
- Local copy: `../data/DiseaseMech_Stability_VEPS.xlsx` (233MB, gitignored)
- **Sheet used: `ClinVar_gene_level`** (not `HGMD_four_class` — that sheet lacks the DN class)
- `Disease_mechanism` column: GOF / DN / HI / AR / AR, HET / AR, HOM
- Final counts after filtering to ClinVar disease variants with valid AA notation:
  - GOF: 1,983 variants, 81 genes
  - DN: 894 variants, 60 genes
  - HI: 1,678 variants, 82 genes
  - AR: 5,678 variants, 725 genes
  - **Total: 10,233 variants, 948 genes**

### Key data observations
- **KCNQ2 dominates DN** (214/894 variants = 24%) — single gene risk
- **SCN1A + SCN2A dominate GOF** (373 + 174 = 547/1983 = 28%) — sodium channel enrichment
- **DN is heavily ion channels**: KCNQ2, KCNQ4, KCNH2, KCNA1, GRIN2A, GABRA1, HCN4
- **GOF enriched for channelopathies and oncogenes**: SCN1A/2A, RYR2, KRAS, BRAF, FGFR1/2
- Family-split CV will be critical to distinguish mechanism signal from family signal

### Gene lists
- `gerasimavicius_gene_list.tsv` — all 948 genes with mechanism and variant counts
- `merged_gene_list.tsv` — 2,424 genes merging Gerasimavicius + G2P (see below)

---

## Dataset Expansion (planned follow-up)

### Gene2Phenotype (G2P)
- Downloaded `ALLG2P.csv` from G2P bulk download
- `molecular mechanism` field: `gain of function` / `dominant negative` / `loss of function`
- Using `definitive` + `strong` confidence only
- G2P-only genes (not in Gerasimavicius): GOF=84, DN=57, LOF=1,335

### Badonyi & Marsh 2025 (NatComms 16, doi:10.1038/s41467-025-63234-3)
- GitHub: `badonyi/mechanism-prediction`
- OSF: 10.17605/OSF.IO/AH2UC
- 139 labeled dominant genes (GOF=49, DN=39, HI=51) + 129 gnomAD controls
- **Complete subset of Gerasimavicius** — 100% agreement, adds no new genes
- Validates Gerasimavicius label quality

### Merged dataset
- `merged_gene_list.tsv`: 2,424 genes (Gerasimavicius priority, G2P fills gaps)
- Merged counts: GOF=158, DN=118, HI=82, LOF=1,341, AR=725
- 39 disagreements between Gerasimavicius and G2P flagged in `g2p_disagrees` column
- **Next step**: fetch ClinVar pathogenic missense variants for G2P-only genes (script pending)

---

## Bugs Fixed

### 1. Wrong OSF URL
- **Error**: `HTTP Error 500` on `https://osf.io/h62fq/download`
- **Fix**: Correct file URL is `https://osf.io/rct6d/download` (direct file, not project page)
- The dataset is an Excel file (`DiseaseMech_Stability_VEPS.xlsx`), not TSV as originally assumed

### 2. Wrong Excel sheet
- **Error**: Used `HGMD_four_class` sheet — has `OTHER LOF` instead of `DN`, no DN class at all
- **Fix**: Switched to `ClinVar_gene_level` sheet which has explicit GOF/DN/HI/AR labels in `Disease_mechanism` column

### 3. Ridge coef_ shape bug
- **Error**: `ValueError: operands could not be broadcast together with shapes (349,1) (1280,1)`
- **Cause**: `Ridge.fit(ddg.reshape(-1,1), deltas)` produces `coef_` of shape `(1, 1280)` not `(1280,)` — needs `.flatten()`
- **Fix**: Added `coefs = np.array(coefs).flatten()` before computing stability direction; also fixed projection to use `[:, None]` instead of `.reshape(-1, 1)` multiplication
- Applied in both `fit_stability_subspace_direct` and `fit_stability_subspace_megascale`

### 4. Deprecated `multi_class` argument
- **Error**: `TypeError: LogisticRegression.__init__() got an unexpected keyword argument 'multi_class'`
- **Cause**: `multi_class="ovr"` removed in newer scikit-learn
- **Fix**: Removed `multi_class="ovr"` from all three `LogisticRegression` calls

### 5. Dead code in family-split CV
- **Error**: `run_linear_probe()` was called before the family-split loop, generating a gene-split result that was immediately overwritten
- **Fix**: Removed the dead `run_linear_probe` call, leaving only the correct family-split loop

### 6. Stale embedding cache
- **Error**: Embeddings cached from synthetic 349-variant run were used for the 618-variant real run, causing shape mismatch in stability subspace fitting
- **Fix**: Deleted `embeddings_*.npy` and `sequences.json` from `../data/` to force re-extraction

### 7. AlphaMissense API broken
- **Error**: `alphamissense.hegelab.org` API returning 0 results
- **Fix**: Switched to MyVariant.info `dbnsfp.alphamissense.score` field via query `{GENE} p.{WT}{POS}{MUT}&fields=dbnsfp.alphamissense&size=1`
- Tested: returns scores correctly (e.g. BRAF V600E = 0.9853)

### 8. `run_baselines` missing arguments
- **Error**: `aa_wt_list`, `aa_mut_list`, `alphamissense_scores` not passed to `run_baselines()`
- **Fix**: Added all three arguments to the call in `run()`

---

## Infrastructure

### SSH key
- Working key: `~/.ssh/id_runpod_2` (registered as `runpod_2` in RunPod account settings)
- RunPod automatically injects this into pod `authorized_keys` on start
- Previous attempts with `id_ed25519` and `id_runpod` failed due to wrong key in account settings

### RunPod API
- Used GraphQL `updateUserSettings(input: { pubKey: "..." })` to update account SSH key
- API key: stored in RUN_AI_SCIENTIST.md (do not commit)

### Current pod
- A100 SXM 80GB RunPod instance
- Repo at `/workspace/dami-AI-Scientist`, branch `esm2-mechanism`
- Baseline running in tmux session `baseline`

---

## Current Run Status (as of ~05:00 May 24)

- Dataset: Gerasimavicius ClinVar_gene_level, 10,233 variants, 948 genes
- Sequences: fetched for 618 valid variant pairs (sequences.json 633K)
- **Embeddings: extracting now** (GPU at 100%, 5GB VRAM)
- Expected completion: ~30-45 min from embedding start
- After completion: stability subspace, probes, baselines, orthogonality, family-split CV

---

## Session 2 findings (May 24, 2026)

### Results summary
- result_1: Linear probe null (macro-F1 0.279). WT-only 0.580 is the suspicious number.
- result_2: Family-split collapses WT-only 0.580→0.389. Delta flat. GOF family-split AUROC 0.801 survives.
- result_3: MLP delta macro-F1 0.431, GOF AUROC 0.744. Superseded by result_5.
- result_4: Family clustering causal explanation. k=5 purity 26× chance, 74.8% within-family mechanism agreement.
- result_5: Nonlinear probes (MLP/kNN/GBM/RF). MLP lift likely residual family signal.
- result_6: **Pathogenicity positive control AUROC 0.88, family-split stable (Δ=0.002).** Pipeline sound. Central finding: ESM-2 encodes pathogenicity not mechanism.

### Central finding
ESM-2 delta embeddings predict ClinVar pathogenic vs benign at AUROC 0.88 but cannot classify GOF/DN/LOF above chance (macro-F1 0.28). The apparent gene-level mechanism signal is family leakage. This is a methodological consolidation contribution (not a discovery) — publishable at Bioinformatics/Genome Biology.

### Running on RunPod
1. `tmux mlp2` — `experiment_mlp.py --family_split` — MLP probe under family-split CV (the single blocking experiment)
2. `tmux merged_emb` — `extract_merged_embeddings.py` — ESM-2 embeddings for merged dataset (19,102 variants, 1,985 genes)

### Merged dataset
- Gerasimavicius (10,233) + G2P/ClinVar pathogenic-only (8,869 new variants from 1,037 new genes)
- GOF: 2,825 / DN: 1,716 / LOF: 14,561
- Files: `esm2_mechanism/data/merged_variants.json`, `esm2_mechanism/data/clinvar_variants.tsv`

---

## Session 3+ findings (May 25–28, 2026)

### result_7 — Full nonlinear probe results + merged dataset (May 24–25)
- MLP delta_mean family-split F1=0.364 on Gerasimavicius; 62% of gene-split signal is family leakage
- Gene-level WT classifier on merged dataset (1,985 genes): family-split F1=0.393 — same floor as Gerasimavicius
- **Family-split floor ~0.39 is the stable real signal; everything above is the family-recognition shortcut**
- DN AUROC stuck at ~0.53 across all classifiers (rarity + mechanistic heterogeneity)
- Leakage fraction (62%) near-identical on both datasets, confirming it is a structural property of the task

### result_leakage_fraction — Leakage fraction as a diagnostic
- LF = (gene_split_F1 − family_split_F1) / (gene_split_F1 − chance_F1) = 62.8% on Gerasimavicius mechanism
- Seed-invariant (std=0.0%) — proposed as a pre-flight diagnostic for published mechanism predictors

### result_8 — Within-family mechanism (May 25)
- PF00520 ion channel: GOF/DN AUROC=0.659 for delta (directional signal)
- Most families at chance due to small within-family gene sets — not publishable at single seed

### result_9 — Contrastive metric learning (May 25)
- Same-mechanism, different-family positive pairs; family-split F1=0.397, MLP floor +0.033
- Gene-split and family-split lift equal (+0.060 / +0.059) — genuine cross-family signal confirmed
- Merged dataset replication: F1=0.387

### result_10 — Clan-level holdout (May 25)
- 21 Pfam clans holdout: F1=0.299 — between majority (0.254) and family-split (0.352)
- ~Half of family-split signal is clan-level memorisation
- Heterogeneous per-clan: Cupin 0.536, Death domain 0.378; Ion_channel 0.190, EF_hand 0.163

### result_11 — Gene-level proteome features, Stage 0 pilot (May 25–26)
- pLI, LOEUF, mis_z, paralog_count on 1,234 genes
- Family-split F1=0.4171±0.0091 (+0.122 above majority); DN AUROC=0.687±0.009 — **STRONG_SIGNAL on 5/5 seeds**

### result_12 — Phase 1+2 proteome feature matrix (May 26)
- 2,424×37 gene-level feature matrix from gnomAD, Ensembl, HPA, PaxDb, BioPlex, ClinGen
- Family-mean-centred residuals computed to mitigate family leakage; coverage 19–100% across features

### result_13 — Phase 3 modelling: proteome vs ESM-2 (May 26)
- V2 (proteome features) family-split F1=0.462 vs V1 (ESM-2 delta) F1=0.382 — **+0.080 lift**
- V3 (ESM-2 + proteome combined) fails Gate 2 (passes 2/5 seeds only)
- DN AUROC improves 0.663→0.740 with proteome; constraint and dosage are load-bearing, PPI degree adds nothing

### result_14 — Clinical utility on ClinGen HI=3 genes (May 26)
- GOF-vs-LOF AUROC=0.650±0.020 on 369 ClinGen gold-standard genes (marginal)
- paralog_count alone AUROC=0.746 — beats the full 37-feature model
- Poor operating point: recall 0.235, precision 0.160 at P_GOF>0.4

### result_15 — Badonyi 2024 structural priors (May 26–27)
- V_bad (3 Badonyi features: pDN, pGOF, pLOF): F1=0.484, outperforms V1 (0.380) and V2 (0.462)
- V2+bad (proteome + Badonyi): F1=0.511±0.021, DN AUROC=0.827±0.015 — **best mechanism predictor so far**
- ESM-2 delta adds nothing to Badonyi (V1+bad=0.441 < V_bad=0.484)

### result_16 — Within-family mechanism from family-residual features (May 27)
- LOGO CV on 24 families (238 genes): residual proteome F1=0.514 vs raw proteome 0.484
- Badonyi residuals add nothing; Homeodomain PF00046: F1=0.633, ion channel PF00520: F1=0.417
- Within-family signal lives in proteome residuals, not structural priors

### result_17 — AlphaMissense family robustness on ClinVar (May 27)
- 16,334 variants (95% coverage), overall AUROC=0.9404
- Per-family AUROC (182 families): mean 0.9477±0.0458, no families below 0.70 — family-robust pathogenicity predictor

### result_18 — AlphaMissense on ProteinGym DMS (May 27)
- 91 human assays: per-assay AUROC mean 0.721±0.150 — much wider variance than ClinVar
- 32% of assays below AUROC 0.70, 14% below 0.60; worst failures on Tsuboyama mini-protein stability (OOD)
- ClinVar robustness was partly underwritten by curation-training overlap

### result_19 — ClinVar variant pattern features (May 27)
- 8 scalar features from spatial distribution of ESM-2 perturbations across observed ClinVar variants
- GOF AUROC 0.578→0.646 under family-split; near-zero leakage (F1 0.352→0.348)
- Combined with delta_mean: F1=0.399; ClinVar enrichment bias flagged as limitation

### result_20 — In-silico perturbation scan (May 27)
- 100 evenly-spaced positions per gene, 3 probe AAs — unbiased, not ClinVar-position-biased
- Scan-only F1=0.272 (family-split), well below ClinVar-pattern (0.348)
- Scan+proteome F1=0.413 (passes G3); confirms result_19's signal relied partly on ClinVar position bias

### result_21 — Stability positive control on S1724 benchmark (May 27–28)
- Linear Ridge loses 0.167 AUROC under protein-holdout (0.764→0.597); GBM recovers it (0.750)
- **Stability is nonlinearly encoded but cross-family transferable** — unlike mechanism which fails at all levels
- Per-protein heterogeneity large: mean ρ=0.248±0.274

### result_22 — Log-likelihood scan vs embedding scan (May 28)
- ΔLL at 100 positions: family-split F1=0.261 — marginally worse than embedding scan (0.272)
- Neither embedding readout is the bottleneck; sparse sampling is
- LL+delta achieves F1=0.380

### result_23 — Magnitude vs direction decomposition (May 28)
- Magnitude-only AUROC=0.664 (weak); direction (d/‖d‖) recovers AUROC=0.896 (family-split)
- **Pathogenicity is a directional (angular) property; conservation (masked-LL) alone: AUROC=0.891**
- Transfer gradient: conservation → pathogenicity (0.891) → stability (0.750) → DMS (ρ 0.50) → mechanism (chance)

### result_24 — ESM-2 ΔLL on ProteinGym (May 28)
- 96 human assays: median Spearman ρ=0.50 — fewer catastrophic failures than AlphaMissense (8% vs 14% below ρ=0.20)
- Median gap (+0.041) misses G3 threshold (+0.05)
- Gates: G1 ✓ G2 ✓ G3 ✗ — completes the transferability gradient

### result_25 — Enzyme type classification positive control (May 28)
- LogReg on WT embeddings: family-split F1=0.655±0.012 (vs mechanism floor 0.385, **Δ=+0.270**)
- Leakage fraction only 13.7% (vs mechanism 62.8%) — most signal is real cross-family signal
- LogReg outperforms MLP (0.655 vs 0.597) — enzyme class is **linearly separable** in WT embedding space
- Proteome features at chance (F1=0.251≈majority 0.228) — enzyme class is not a population-genetics property
- **Double dissociation**: ESM-2 strong for sequence-level properties (enzyme, pathogenicity); proteome features strong for gene-level properties (mechanism)
- **Confirms the mechanism null is task-specific, not a pipeline failure**

---

## Current state (May 28, 2026)

### Core scientific claims — status
1. ESM-2 encodes pathogenicity at AUROC=0.884, linearly, family-split-stable (result_6, result_23) ✓
2. ESM-2 mechanism floor ~0.385 macro-F1 under family-split; 62.8% of gene-split signal is leakage (result_7) ✓
3. Proteome features (V2) beat ESM-2 for mechanism by +0.080 F1 (result_13) ✓
4. Best mechanism predictor: V2+Badonyi F1=0.511±0.021, DN AUROC=0.827±0.015 (result_15) ✓
5. Mechanism null is task-specific — enzyme type (same pipeline, same CV) achieves F1=0.655 (result_25) ✓
6. Conservation → pathogenicity → stability → DMS → mechanism: complete transferability gradient (result_23, result_24) ✓

### Remaining work
- Figure for paper (leakage comparison, task × modality double dissociation, transferability gradient)
- LaTeX draft
