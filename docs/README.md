# esm2_mechanism — results index

**This is a standalone research project, not an AI Scientist run.** All experiments were designed and executed manually. The code lives in `esm2_mechanism/scripts/` and was run directly on RunPod (A100 80GB) or local CPU. The project will likely move to its own repository.

Twenty-four `result_*.md` files written across May 23–28, 2026, plus `result_leakage_fraction.md` (working note). Read in the order below for the coherent narrative arc. Results 3 and 5 are superseded by result 7.

---

## Current state (as of result 24, 2026-05-28)

The project has four connected arcs:

1. **Results 1–10 — frozen ESM-2 characterisation.** Mechanism floor under family-split CV is **F1 = 0.385 ± 0.018 (merged, 5-seed) / 0.299 ± 0.034 (Gerasimavicius, 5-seed)** — see result_6 Part 2 for the multi-seed correction. Clan-holdout shows ~half the family-split signal is fold memorisation (result 10). Pathogenicity positive control AUROC 0.74–0.88 across replications, family-split-stable (gene→family Δ ≈ 0 reproducibly; result 6) — confirms pipeline soundness and establishes the pathogenicity–mechanism dissociation. 62.8% of gene-split mechanism signal is family-recognition leakage on Gerasimavicius (exact, seed-invariant — structural property of the dataset; result 7 + result 6 Part 2).

2. **Results 11–14 — gene-level proteome features.** A 4-feature pilot (result 11) hits macro-F1 0.417 family-split. The 37-feature matrix (result 12, sources: gnomAD, paralogs, HPA, PaxDb, BioPlex, ClinGen) gives V2 macro-F1 = 0.462 ± 0.025 — outperforming frozen ESM-2 delta (V1, 0.382) by +0.080 (result 13). Per-gene scoring lifts the V2 advantage to +0.101. Combining ESM-2 + proteome (V3) does not reliably improve over V2 alone (Gate 2 fails 3/5 seeds). Feature ablation (T4) shows constraint + dosage are load-bearing; the proteome-biology features (PPI, paralogs, abundance) contribute little to aggregate F1 but matter per-class. Clinical utility (result 14) collapses to a single column: paralog count alone achieves AUROC 0.746 within ClinGen HI=3, beating the full 37-feature model (0.650). Calibration is poor; operating-point performance is weak.

3. **Results 15–16 — Badonyi structural priors + within-family.** Badonyi 2024's SVM probabilities (3 features) beat ESM-2 (+0.104 macro-F1) and proteome (+0.022). V2+bad reaches macro-F1 = 0.511 and DN AUROC = 0.827 — the project's high-water mark (result 15). ESM-2 is the dispensable modality (V1+bad < V_bad). Robustness analyses (result 15 Appendix A: leakage triage; Appendix B: MMseqs2-20 cluster-holdout) confirm the lift is real and survives a stricter sequence-similarity holdout matched to Saadat & Fellay 2025. Within families (result 16), residual proteome features (gene minus family-mean) achieve F1 = 0.514 in LOGO CV across 24 Pfam families. Badonyi residuals add nothing within-family — the structural prior carries only cross-family signal. Homeodomains (n=30, F1=0.633) are the anchor example. The result 16 addendum tests Badonyi's *raw published model* under family-split: it passes the leakage-free criterion but shows a per-gene training-set fit effect (LOF AUROC 0.625 in-training vs 0.472 never-seen) — does not affect V_bad/V2+bad validity, but affects how Badonyi's published numbers should be cited.

4. **Results 17–24 — pathogenicity geometry, perturbation scans, stability, AlphaMissense, ProteinGym ΔLL.** AlphaMissense is family-robust on ClinVar (mean per-family AUROC 0.948 ± 0.046; result 17) but not on ProteinGym DMS labels (mean per-assay AUROC 0.721 ± 0.150, 32% below 0.70; result 18) — the tight ClinVar distribution reflects curation–training overlap, not general family-robustness. ClinVar variant pattern features (spatial hotspot vs spread) give nearly leak-free GOF signal (family-split AUROC 0.646; result 19); the unbiased in-silico scan loses that GOF signal alone (F1 0.272) but adds orthogonal signal to proteome (V2+scan F1 0.413; result 20). Stability in ESM-2 delta is nonlinearly encoded but cross-family transferable (GBM Pfam-split AUROC 0.750 vs linear 0.597; result 21) — the sharpest contrast with mechanism, where nonlinearity does not rescue family-split performance. Log-likelihood scan gives no improvement over the embedding scan (LL-only F1 0.261; result 22), confirming the bottleneck is sampling density not readout. Pathogenicity is carried by delta direction not magnitude (direction AUROC 0.896 vs magnitude 0.664; result 23); that direction IS conservation (masked-LL alone 0.891 family-split); conservation transfers linearly for pathogenicity, nonlinearly for stability, and not at all for mechanism — the transferability is task- and probe-dependent within one frozen model. ESM-2 ΔLL on 96 human ProteinGym assays replicates published ESM-1v baseline (median ρ=0.50; result 24); fewer tail failures than AlphaMissense (8% vs 14% below ρ=0.20) but the median gap is +0.041, short of the pre-registered +0.05 — the per-assay variance is intrinsic to DMS task heterogeneity, not predictor type, completing the transferability gradient.

The experimental work is essentially done. Remaining: writeup consolidation, one master figure, optional rigour (bootstrap CIs, calibration on V2+bad), optional Path B (raw structural features de novo).

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
**What it concludes:** WT-only signal explained by family recognition × family-mechanism correlation. The causal explanation for why frozen ESM-2 looks like a mechanism predictor under gene-split.

### 5. `result_5.md` — Nonlinear probes (MLP/kNN/GBM/RF) ⚠ PARTIALLY SUPERSEDED BY RESULT 7
**Script:** `experiment_mlp.py` extended · 4 probes on delta_mean and delta_pos, gene-split only
**Headline numbers:** MLP = 0.431, kNN = 0.410, GBM = 0.336, RF = 0.292.
**⚠ Limitation:** Gene-split only. Result 7 provides the family-split calibration showing 62% of the gene-split lift is leakage.

### 6. `result_6.md` — Pathogenicity positive control
**Script:** `pathogenicity_control.py` · 17,236 ClinVar pathogenic/benign variants, 944 genes
**Headline numbers:** Pathogenicity MLP AUROC = **0.878 (seed 0, RunPod variant set) / 0.742 ± 0.006 (seeds 1–4, locally-truncated variant set)**. Family-split Δ ≈ 0 reproducibly across all seeds and both variant sets (family-split stability is the robust claim). Clean 5-seed mean on a consistent variant set is pending due to provenance issue.
**What it concludes:** Pipeline is sound. Pathogenicity AUROC in the 0.74–0.88 range across replications (family-split-stable in all) vs mechanism floor 0.30–0.39 (family-split) — the controlled dissociation under identical pipeline holds.

### 7. `result_7.md` — Full calibration: all numbers, honest framing
**Scripts:** `experiment_mlp.py` with family-split, `build_merged_dataset.py`, Option B gene-level WT
**Headline numbers (single-seed in this file; see result_6 Part 2 for multi-seed correction):**
- MLP delta gene-split **0.415**, family-split **0.364** seed 0 → **0.299 ± 0.034 5-seed** (Gerasimavicius)
- Merged dataset family-split **0.352** seed 0 → **0.385 ± 0.018 5-seed**
- Always-predict-LOF baseline: **0.279** (Gerasimavicius), **0.311** (gene-level merged)
- Family-split floor under multi-seed: **F1 = 0.30 (Gerasimavicius) / 0.39 (merged)** — merged is the more reliable headline
**What it concludes:** The floor is real but lower than the single-seed numbers in this file. The pathogenicity-mechanism dissociation holds (pathogenicity 0.74–0.88 vs mechanism 0.30–0.39, both family-split-stable). The GOF AUROC (0.557 ± 0.036 Geras / 0.655 ± 0.014 merged delta MLP) is the strongest mechanism-class signal that survives family-split, distinct from the WT-only GOF AUROC of 0.73–0.80 which captures gene identity rather than mutation effect.

### 8. `result_8.md` — Within-family mechanism (first pass)
**Script:** ad-hoc analysis on cached Gerasimavicius embeddings · Local CPU, seed=42
**Headline numbers:** Within-family gene-split CV on the 5 largest Pfam families. **PF00520 (ion channel) delta F1=0.407, AUROC=0.659 (2-class GOF/DN)** — the most interpretable result. Other families largely at chance due to tiny sample sizes (6–12 genes).
**What it concludes:** Directional signal that mechanism is partially learnable within a homologous subfamily; consistent with MissION-style findings. Not publishable at single seed + small N; result 16 follows up.

### 9. `result_9.md` — Contrastive metric learning recovers cross-family signal
**Script:** `contrastive_mechanism.py` · A100 80GB, seed=0
**Headline numbers:** Supervised contrastive projection head (1280→256→64, TripletMarginLoss, positives = same mechanism / different family) pushes family-split macro-F1 from MLP's 0.364 to **0.397** on Gerasimavicius (+0.033) and to **0.387** on merged (+0.035 above MLP floor). Lift is equal under gene-split (+0.060) and family-split (+0.059) — the critical diagnostic that the recovered signal is *not* leakage. LOF benefits most; **DN stays flat (+0.012 Geras, −0.025 merged)**.
**What it concludes:** Frozen ESM-2 delta does encode small cross-family mechanism signal not accessible to a standard MLP — but only for LOF. DN remains essentially absent.

### 10. `result_10.md` — Clan-level holdout: partial generalisation, not pure memorisation
**Script:** `clan_holdout.py` · Local CPU, seed=0
**Headline numbers:** Leave-one-Pfam-clan-out evaluation across 21 qualifying clans gives MLP macro-F1 = **0.299 ± 0.076** — below family-split floor (0.352) but above majority (0.254). Per-class AUROCs (GOF 0.597 / DN 0.575 / LOF 0.636) confirm real cross-fold signal. **Approximately half the family-split mechanism signal is clan-level memorisation; the remainder is genuine cross-fold generalisation.** Heterogeneous across clans (Cupin F1=0.536; Ion_channel F1=0.190).
**What it concludes:** The ~0.36 family-split floor is roughly half real, half fold-memorisation. Mechanism is more readable from sequence in architectures with stereotyped structural mechanisms (cupins, death domains, GPCRs) and unreadable in plastic repeat proteins (ankyrins, EF-hands).
**Reconciliation note:** Result_10's interpretation ("DN biology in complex-assembly context") is partly contradicted by result_13 T4's feature ablation, which finds PPI_degree contributes nothing to V2 aggregate F1. Result_16 reconciles: PPI signal is within-family, not cross-family.

### 11. `result_11.md` — Stage 0 pilot: 4 gene-level features predict mechanism under family-split CV
**Script:** `proteome_pilot.py` · Local CPU, seeds 0–4 (5-seed replication)
**Headline numbers:** Logistic regression on 4 public gene-level features (pLI, LOEUF, mis_z, paralog_count) under family-split CV on 1,234 genes / 725 families achieves macro-F1 = **0.417 ± 0.009** (+0.122 above majority 0.295). Per-class AUROCs (mean ± std): GOF **0.686 ± 0.011**, **DN 0.687 ± 0.009**, LOF **0.735 ± 0.001** — balanced and tight across seeds.
**What it concludes:** Stage 0 sanity check passes. Public gene-level features carry meaningful mechanism signal under the project's family-split CV, robust to seed choice. 5/5 seeds returned STRONG_SIGNAL by the pre-registered rule. Proceeded to Phase 1.

### 12. `result_12.md` — Proteome feature matrix assembled (data collection only)
**Script:** `build_proteome_features.py` · No model run
**Output:** 2,424 × 37 float32 matrix at `data/proteome_features_aligned.npy`. Sources: gnomAD constraint (93%), Ensembl paralogs (100%), HPA tissue specificity (99%, mapped from categorical), PaxDb abundance (98%, manual download — automated 403'd), BioPlex 3.0 PPI degree (75%), ClinGen HI/TS (19%/37%). HPA n_tissues failed; Mathieson half-life and PhosphoSitePlus PTM not pursued. Family-mean-centred residuals and binary missingness indicators included.
**What it concludes:** Feature collection complete with documented coverage. Proceeded to Phase 3 modelling.

### 13. `result_13.md` — Phase 3 modelling: proteome features outperform ESM-2
**Script:** `proteome_mechanism.py`, `per_gene_ablation.py` · 5 seeds, family-split CV
**Headline numbers:**
- V1 (ESM-2 delta, 1280-dim) macro-F1 = **0.382 ± 0.007**
- V2 (proteome, 37-dim) macro-F1 = **0.462 ± 0.025**, DN AUROC = **0.727 ± 0.017**
- V3 (concat 1317-dim) macro-F1 = **0.447 ± 0.020** — Gate 2 (V3 ≥ max+0.02) fails 3/5 seeds
- V4 (contrastive on V3) macro-F1 = **0.424** — underperforms V3
- **T2 per-gene scoring:** V1 = 0.359, V2 = 0.460, V3 = 0.413. V2 advantage grows to **+0.101 per-gene** vs V1; V3 now actively below V2
- **T4 feature ablation:** dropping constraint costs ΔF1 = +0.040 (the most important class); dropping dosage costs +0.043; dropping PPI_degree gives ΔF1 = −0.002 (contributes nothing). DN AUROC is *hurt* by constraint features (they conflate DN with LOF in the multi-class model)
**What it concludes:** Frozen ESM-2 delta is dominated by 37 gene-level features. Combining doesn't help; ESM-2 is dispensable. The mechanistic interpretation differs from the original hypothesis (PPI/paralogs were expected to drive DN; they don't — constraint and dosage do, but they *hurt* DN specifically).

### 14. `result_14.md` — Clinical utility: paralog count alone beats the multi-feature model
**Script:** `clinical_utility.py` · Family-split CV, 5 seeds, two feature sets
**Headline numbers:** Within ClinGen HI=3 genes (n=369), under family-split CV:
- LR FULL (37 features) GOF-vs-LOF AUROC = **0.650 ± 0.020** (marginally above INFORMATIVE threshold 0.65; 2/5 seeds below)
- LR NO-MISS (18 features, missingness indicators dropped) = **0.679 ± 0.016**
- **paralog_count alone = 0.746** — beats every multi-feature model
- Operating point P_GOF > 0.4: recall 0.235, precision 0.160 — not clinically useful
- ECE = 0.148 — model is miscalibrated
**What it concludes:** The clinical utility case reduces to one signal: paralog count predicts GOF direction within haploinsufficient genes. Biologically interpretable via the gene balance hypothesis (paralog-rich genes survive dominant mutations better; when ClinGen still calls them HI, mechanism is more likely activating). Multi-feature model adds noise rather than signal in this evaluation subset. Honest framing: narrow ranking signal, not a clinical predictor.

### 15. `result_15.md` — Badonyi 2024 priors as a third modality (project high-water mark)
**Script:** `badonyi_mechanism.py` · 5-fold family-split, 5 seeds
**Headline numbers:**
- V_bad (Badonyi pDN/pGOF/pLOF, 3 features, LogReg) macro-F1 = **0.484 ± 0.021** — beats V1 (+0.104) and V2 (+0.022)
- **V2+bad** (proteome + Badonyi, 40 features) = **0.511 ± 0.021**, DN AUROC = **0.827 ± 0.015** — project high-water mark
- V1+bad (ESM-2 + Badonyi, 1283 features) = 0.441 — *underperforms* V_bad alone
- V_all (all three modalities, 1320 features) = 0.481 — underperforms V2+bad
**What it concludes:** Modality ordering is Badonyi structural priors > proteome > ESM-2. ESM-2 is the dispensable modality. Three Badonyi features beat 1280-dim ESM-2 delta. The combination of structural prior + cellular features is genuinely additive (+0.049 F1, +0.100 DN AUROC over V2 alone).

**Appendix A — Leakage triage of V_bad.** Pre-registered concern: V_bad uses Badonyi predictions for genes that were in Badonyi's training set (621/1,699 labeled = 37% overlap, concentrated in minority classes — 73% of GOF and 72% of DN). Stratified evaluation under family-split CV finds V_bad performs *better* on out-of-Badonyi-training genes than in-training ones (DN AUROC 0.814 OUT vs 0.745 IN). No leakage signature. V_bad's headline is real, not artifact.

**Appendix B — MMseqs2-20 cluster-holdout.** Re-evaluated all variants under sequence-similarity clusters at 20% identity / 20% coverage (matching Saadat & Fellay 2025). All numbers within ±0.03 of family-split. V_bad DN AUROC = 0.776; V2+bad = 0.816. Result 15 conclusions hold under a stricter homology block.

### 16. `result_16.md` — Within-family mechanism + Badonyi raw-model holdout
**Script:** `within_family_mechanism.py`, `badonyi_holdout_survival.py` · LOGO CV across 24 families
**Headline numbers (within-family):**
- Residual proteome (family-mean-centred) macro-F1 = **0.514** — beats raw proteome (0.484), Badonyi residuals (0.449), and combined residuals (0.516)
- **Badonyi raw = Badonyi residual = 0.449** — structural prior carries no within-family variation
- PF00046 (homeodomain, n=30): residual proteome F1 = **0.633**, GOF AUROC 0.852, DN AUROC 0.857 — the anchor example
- PF00520 (ion channel, n=16): residual proteome F1 = 0.417 (near majority) — within-family signal lives in ESM-2 mutation context (result 8) for this family, not gene-level features

**Addendum — Badonyi's raw published model under family-split holdout.** Tests whether Badonyi's *published* SVM (not V_bad's re-fit LogReg) survives the project's strict CV. Pre-registered decision rule (ΔAUROC ≥ −0.03 robust, ≤ −0.10 mostly leakage):
- Pfam family-split: all three classifiers in ROBUST band
- MMseqs2-20 cluster-split: all three classifiers in ROBUST band
- **But:** stratified by Badonyi-training-set membership without any holdout shows a per-gene training-set fit effect — DN AUROC 0.677 in-training vs 0.620 never-seen; GOF 0.713 vs 0.694; **LOF 0.625 in-training vs 0.472 never-seen** (15-point gap, never-seen at chance)
**What it concludes:** Badonyi's model is family-recognition-robust (good) but his published LOF numbers reflect per-gene training-set fit (concern). Does not affect V_bad/V2+bad validity — the LogReg-on-top averages out per-gene memorisation per fold. Affects how Badonyi's published numbers should be cited.

---

### 17. `result_17.md` — AlphaMissense is family-robust on ClinVar
**Script:** `alphamissense_family_split.py` · **Run:** May 26, 16,334 ClinVar variants, 182 Pfam families
**Headline numbers:** Overall AUROC = **0.940**. Per-family AUROC mean **0.948 ± 0.046**, median 0.960, IQR 0.923–0.983. **0% of families below AUROC 0.70**.
**What it concludes:** result_6's family-robustness finding (ESM-2 pathogenicity probe Δ ≈ 0 gene→family) generalises to the published clinical predictor. Caveat: ClinVar–AM training-logic overlap may inflate the absolute number; the tight per-family *distribution* is the durable finding.
**Open question after reading:** *Does the tight distribution survive when labels come from physical experiments rather than clinical curation?*

### 18. `result_18.md` — AlphaMissense on ProteinGym: family-robustness narrows without training–test overlap
**Script:** `proteingym_alphamissense.py` · 91 human DMS assays (ProteinGym v1.3)
**Headline numbers:** Per-assay AUROC mean **0.721 ± 0.150**, range 0.170–0.957. **32% of assays below AUROC 0.70; 14% below 0.60**.
**What it concludes:** The tight ClinVar distribution is partly underwritten by curation–training overlap. On physical DMS labels the distribution is wide and bimodal — failures cluster on OOD assays (thermal-stability mini-proteins, less-studied proteins), not classic disease genes. Reframes result_17 as a within-curation-distribution claim.

### 19. `result_19.md` — ClinVar variant pattern features: spatial hotspot distribution predicts mechanism with near-zero leakage
**Script:** `perturbation_pattern.py` · CPU, seeds 0–4, Gerasimavicius merged
**Headline numbers:** 8 scalar features from spatial distribution of ClinVar delta magnitudes. GOF AUROC **0.646** (vs 0.578 baseline). Family-split F1 **0.399**. Gene-split ≈ family-split — near-zero leakage.
**What it concludes:** GOF hotspot biology (mutations must hit specific sites to activate) vs LOF spread (can break anywhere) is readable from observed ClinVar variant positions via ESM-2 perturbation magnitudes. Signal is real but depends on observing ClinVar variants (circularity concern — addressed in result 20).

### 20. `result_20.md` — In-silico perturbation scan: unbiased hotspot features add to proteome but fail alone
**Scripts:** `perturbation_scan.py`, `perturbation_probe.py` · GPU H100, seeds 0–4, ~568k forward passes
**Headline numbers:** Scan-only family-split F1 = **0.272** (below G1 threshold 0.368). V2+scan F1 = **0.413** (passes G3 threshold 0.405).
**What it concludes:** Removing ClinVar circularity via systematic in-silico scan loses the GOF hotspot signal when used alone, but scan features add orthogonal information to proteome features. Not a standalone modality; a useful complement.

### 21. `result_21.md` — Stability is nonlinearly encoded and cross-family transferable; mechanism is not
**Scripts:** `megascale_stability.py`, `megascale_mlp.py` · GPU A100, S1724 benchmark (1,277 variants, 27 proteins)
**Headline numbers:** Linear Pfam-split AUROC = 0.597. GBM Pfam-split AUROC = **0.750** (≈ linear in-distribution 0.764). RF = 0.735.
**What it concludes:** Stability signal in ESM-2 delta lives in a curved cross-family submanifold — nonlinearly organised but transferable. Mechanism MLP lift evaporates under family-split (result 7); stability GBM lift does not. This is the sharpest task-level distinction in the embedding space.

### 22. `result_22.md` — Log-likelihood scan: sharper readout, same sampling problem
**Script:** `ll_scan.py` · GPU H100, seeds 0–4, ~198k forward passes
**Headline numbers:** LL-only family-split F1 = **0.261** (worse than embedding scan 0.272). All gates fail.
**What it concludes:** The readout (L2 distance vs log-likelihood) is not the bottleneck in result_20. Sparse 100-position sampling is. Follow-up would need denser or adaptive sampling.

### 23. `result_23.md` — Pathogenicity is direction (= conservation), not magnitude; transferability is task- and probe-dependent
**Scripts:** `magnitude_direction.py`, `direction_geometry.py`, `transfer_contrast.py`, `conservation_axis.py` · GPU + CPU, 5 seeds
**Headline numbers:** Direction AUROC **0.896** (≈ full delta 0.884); magnitude AUROC **0.664**. Masked-LL alone = **0.891** (beats embedding direction 0.835 — embedding adds nothing). Conservation transfer: pathogenicity linear 0.815 / GBM 0.889; stability GBM 0.750 (nonlinear manifold); mechanism linear 0.520 / GBM 0.540 (chance).
**What it concludes:** The pre-registered hypothesis (pathogenicity = magnitude) is falsified. Pathogenicity is direction, and that direction IS conservation. Conservation transfers linearly for pathogenicity, only nonlinearly for stability, and not at all for mechanism. Transferability is task- and probe-dependent within one frozen model.

### 25. `result_25.md` — Enzyme type classification from ESM-2 WT embeddings: confirms mechanism null is task-specific
**Scripts:** `fetch_enzyme_labels.py`, `enzyme_classification.py` · Local CPU, 5 seeds. UniProt EC labels for 1985 genes (kinase 136, protease 68, oxidoreductase 119, non-enzyme 1662).
**Headline numbers:** LogReg family-split macro-F1 = **0.655 ± 0.012**. Per-class AUROCs: kinase **0.896**, protease **0.904**, oxidoreductase **0.890**, non-enzyme **0.854**. Leakage fraction **13.7%** (vs 62.8% for mechanism). Proteome features family-split F1 = **0.251** (≈ majority 0.228 — gene-biology is at chance for enzyme class). LogReg beats MLP family-split (0.655 vs 0.597 — enzyme class is linearly separable).
**What it concludes:** The mechanism floor (0.385) is not a methodological ceiling. The same pipeline achieves F1 = 0.655 for enzyme type — Δ = +0.270 above mechanism. Enzyme class (a WT fold property) is strongly and linearly encoded; disease mechanism (a mutation-effect property) is not. The task × modality double dissociation is complete: ESM-2 WT embeddings predict enzyme class strongly, proteome features predict disease mechanism better than ESM-2. The mechanism null result is task-specific.

---

## The coherent story across all 25

1. **(1–2)** Linear probes are at chance on delta. WT-only F1=0.58 collapses to 0.39 under family-split — most apparent mechanism signal is family identity.
2. **(4)** ESM-2 strongly clusters by Pfam (26× purity) and 74.8% of genes share their family's modal mechanism — the causal explanation.
3. **(6)** Pathogenicity positive control AUROC 0.74–0.88 across replications, family-split-stable (Δ ≈ 0 reproducibly) — pipeline works; mechanism null is real, not methodological. Multi-seed update in result 6 Part 2.
4. **(7)** Full calibration: mechanism family-split floor F1 = 0.385 ± 0.018 (merged, 5-seed) / 0.299 ± 0.034 (Gerasimavicius, 5-seed). 62.8% of gene-split signal is family-recognition leakage (exact, seed-invariant on Gerasimavicius). Pathogenicity–mechanism dissociation is the central finding of the ESM-2 arc.
5. **(8)** Within ion-channel family, ESM-2 delta has within-family signal (AUROC 0.659 GOF/DN) — directional, small N.
6. **(9)** Contrastive metric learning lifts cross-family floor from 0.364 → 0.397 with equal gene-/family-split deltas (real signal, not leakage). LOF benefits; DN doesn't.
7. **(10)** Clan-holdout: ~half family-split signal is fold memorisation, ~half real. Heterogeneous across protein architectures (cupins generalise; ion channels collapse).
8. **(11)** Stage 0 pilot for proteome thread: 4 gene-level features hit F1=0.417 — STRONG_SIGNAL, proceed to full pull.
9. **(12)** 37-feature proteome matrix assembled. PaxDb and HPA partially limited; family-mean-centred residuals included.
10. **(13)** V2 (proteome) outperforms V1 (ESM-2) by +0.10 per-gene F1. V3 (combination) doesn't help. Constraint + dosage are load-bearing; PPI/paralogs/abundance contribute little to aggregate F1.
11. **(14)** Clinical utility reduces to paralog_count alone (AUROC 0.746 within HI=3) beating the multi-feature model. Calibration poor; operating-point performance weak.
12. **(15)** Badonyi structural prior (3 features) beats ESM-2 (1280 dims) and proteome (37 dims). V2+bad is the high-water mark (F1=0.511, DN AUROC=0.827). Robust under leakage triage and MMseqs2-20 cluster-split.
13. **(16)** Within-family mechanism lives in residual proteome features (F1=0.514 in LOGO across 24 families); Badonyi residuals add nothing within-family. Homeodomains are the cleanest case. Badonyi's raw published model is family-recognition-robust but shows per-gene training-set fit on LOF.
14. **(17–18)** AlphaMissense is family-robust on ClinVar (mean per-family AUROC 0.948) but not on ProteinGym DMS (mean 0.721, 32% below 0.70) — the tight ClinVar distribution reflects curation–training overlap, not general robustness.
15. **(19–20)** ClinVar variant pattern (hotspot vs spread) gives near-leak-free GOF signal (AUROC 0.646); unbiased in-silico scan loses that signal alone but adds orthogonally to proteome.
16. **(21)** Stability in ESM-2 delta is nonlinearly cross-family transferable (GBM 0.750 Pfam-split); mechanism is not — the sharpest task-level distinction in the embedding space.
17. **(22)** LL scan readout doesn't improve on embedding scan; sampling density is the bottleneck.
18. **(23)** Pathogenicity = direction = conservation. Conservation transfers linearly for pathogenicity, nonlinearly for stability, not at all for mechanism. Unifies the cross-result pattern.
19. **(24)** ESM-2 ΔLL on 96 human ProteinGym DMS assays: median ρ=0.50 (replicates ESM-1v), 8% of assays below ρ=0.20 vs AM's 14%. G3 fails (+0.041 vs +0.05 threshold) — per-assay variance is intrinsic to DMS task heterogeneity. Binding (ρ=0.34) is the weak point; Stability (ρ=0.59) and Activity (ρ=0.53) are strongest. Completes the transferability gradient: conservation → pathogenicity (0.891) > stability (0.750) > DMS fitness (0.50, high variance) > mechanism (chance).
20. **(25)** Enzyme type positive control (kinase/protease/oxidoreductase/non-enzyme): WT embedding family-split F1 = 0.655 ± 0.012, leakage 13.7%, all per-class AUROCs ≥ 0.85. Mechanism floor (0.385) is task-specific, not methodological. Double dissociation: ESM-2 WT embeddings predict enzyme class (F1 = 0.655); proteome features do not (F1 = 0.251 ≈ chance). Proteome features predict mechanism better than ESM-2 (result 13). LogReg outperforms MLP for enzyme type — linearly separable, paralleling pathogenicity (result 23) and contrasting with stability (result 21).

**The narrative shape:** frozen-PLM negative result (1–10) → gene-level proteome features beat ESM-2 (11–13) → clinical utility narrows to one column (14) → structural priors beat both (15) → within-family signal lives in within-family proteome variation (16) → pathogenicity geometry and AlphaMissense robustness characterised (17–18) → perturbation scans bound the ClinVar-pattern signal (19–20) → stability is nonlinearly transferable, mechanism is not (21–22) → conservation unifies the transferability gradient (23) → conservation predicts DMS fitness on average but with intrinsic assay-type variance, completing the gradient (24) → enzyme-type positive control confirms the mechanism null is task-specific and closes the pipeline-validity question (25). ESM-2 is dispensable for mechanism throughout. The only family-transferable signal it carries is conservation, which fully explains pathogenicity, partially explains stability and DMS fitness, and leaves mechanism at chance. Enzyme class — a fold property, not a perturbation property — is the clearest positive demonstration of what the embeddings do encode.

---

## Supporting docs

- `EXPERIMENT.md` — Pre-registration document (original ESM-2 hypothesis, results 1–10 scope)
- `plan_experiment.md` — Experiment 11 plan: per-variant ESM-2 + gene-level proteome features (staged execution: pilot → V2 → V3 → V4)
- `plan_esm2_proteome.md` — Detailed Phase 1+2 plan for proteome feature engineering
- `plan_clinical.md` — Clinical utility analysis plan (result 14)
- `plan_badonyi.md` — Pre-registration for Badonyi raw-model holdout (result 16 addendum)
- `progress_notes.md` — Running log of decisions, bugs fixed, observations
- `../scripts/README.md` — What each script does

---

## Companion data

| Result | Primary JSON file |
|---|---|
| 1 | `results/20260524_baseline_run/run_0/final_info_seed0.json` |
| 2 | `results/20260524_baseline_run/run_0/family_split_baselines.json` |
| 3, 5 | `results/20260524_baseline_run/run_0/mlp_results_seed0.json` |
| 4 | `results/20260524_baseline_run/run_0/family_clustering.json` |
| 6 | `results/20260524_baseline_run/run_0/pathogenicity_control.json` |
| 7 | `results/20260524_baseline_run/run_0/option_b_gene_level_wt_merged.json` + merged MLP |
| 8 | `results/20260524_baseline_run/run_0/within_family_analysis.json` |
| 9 | `results/20260524_baseline_run/run_0/contrastive_results_{geras,merged}_seed0.json` |
| 10 | `results/20260524_baseline_run/run_0/clan_holdout_results_seed0.json` |
| 11 | `results/proteome_pilot/pilot_results_summary_5seed.json` |
| 12 | `data/gene_proteome_features.tsv`, `data/proteome_features_aligned.npy`, `data/proteome_feature_columns.json` |
| 13 | `results/proteome_mechanism/proteome_mechanism_summary.json`, `per_gene_summary.json`, `v2_ablation_summary.json` |
| 14 | `results/clinical_utility/hi3_family_split_summary.json` |
| 15 | `results/badonyi_mechanism/badonyi_mechanism_summary.json` |
| 15-AppA | `results/badonyi_leakage/leakage_summary.json` |
| 15-AppB | `results/mmseqs_cluster_holdout/cluster_summary.json` |
| 16 | `results/within_family/within_family_summary.json` |
| 16-addendum | `results/badonyi_survival/badonyi_survival_summary.json` |
| 25 | `results/enzyme_classification/enzyme_classification_summary.json` |

ESM-2 embeddings under `data/embeddings/`:
- `embeddings_{wt,mut}{,_pos}_esm2_t33_650M_UR50D.npy` — Gerasimavicius 10,231 variants (results 1–5, 7)
- `merged_embeddings_{wt,mut}_{mean,pos}.npy` — merged 1,985-gene / 19,100-variant dataset (results 7, 9, 13–15)
- `emb_{wt,mut}_mean_pathogenicity_*.npy` — ClinVar 17,236 pathogenicity variants (result 6)

Feature matrices:
- `data/proteome_features_aligned.npy` (2,424 × 37) — gnomAD + paralogs + HPA + PaxDb + BioPlex + ClinGen (result 12)
- `data/badonyi_features_aligned.npy` (2,424 × 13) — pDN/pGOF/pLOF + residuals + missingness (result 15)
- `data/mmseqs_clusters.json` — gene → MMseqs2-20 cluster_rep mapping (result 15 Appendix B)
- `data/enzyme_labels.tsv` — 2424 genes, 4-class enzyme labels (kinase/protease/oxidoreductase/non-enzyme), EC numbers, UniProt flags (result 25)

---

## Remaining work

The experimental matrix is essentially complete. What's left:

1. **One master figure** — bar chart of macro-F1 / per-class AUROC × modality (V1/V2/V_bad/V2+bad) × holdout (family-split / MMseqs2-20). ~half day.
2. **Bootstrap CIs on headline numbers** — V_bad and V2+bad DN AUROC have seed std but not within-seed CIs. ~2 hours.
3. **Calibration analysis on V2+bad** — reliability diagram and ECE. ~1 hour.
4. **Path B (raw structural features de novo)** — compute FoldX ΔΔG, SASA, SCRIBER, RSA on AF2 structures for the merged gene set; re-evaluate V_struct vs V_bad. Optional rigour upgrade; ~several days including FoldX runtime. Skip unless reviewer specifically pushes back.
5. **Writeup polish** — update `scripts/README.md` to reflect all current scripts; possibly write a short `OVERVIEW.md` for first-time readers.

Scope expansions deliberately not pursued (per the staged plan's pre-registered gates):
- Fine-tuning ESM-2 (LoRA contrastive) — V3 ≯ V2, so the pre-registered "skip" rule fires
- Per-variant mechanism labels — different problem; requires MAVE/DMS data pull
- DDG2P replication — deferred to v3 of the publication plan
- SaProt / ESM-3 — deferred to v3
