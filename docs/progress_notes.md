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
- A100 SXM 80GB, `154.54.102.57:19561`
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

### Running on RunPod (154.54.102.28:13732)
1. `tmux mlp2` — `experiment_mlp.py --family_split` — MLP probe under family-split CV (the single blocking experiment)
2. `tmux merged_emb` — `extract_merged_embeddings.py` — ESM-2 embeddings for merged dataset (19,102 variants, 1,985 genes)

### Merged dataset
- Gerasimavicius (10,233) + G2P/ClinVar pathogenic-only (8,869 new variants from 1,037 new genes)
- GOF: 2,825 / DN: 1,716 / LOF: 14,561
- Files: `esm2_mechanism/data/merged_variants.json`, `esm2_mechanism/data/clinvar_variants.tsv`

### Next steps
1. Collect MLP family-split results (mlp2)
2. After embeddings: run linear + MLP probes on merged dataset
3. Option B: gene-level WT mean embeddings probe
4. Within-family mechanism analysis (top Pfam families)
5. Write result_7.md
6. Multi-seed replication (5 seeds)
7. Figure for paper
