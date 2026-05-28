ESM-2 frozen embeddings encode whether a mutation is damaging (delta MLP AUROC 0.74–0.88 across replications, family-split-stable in all; gene→family Δ ≈ 0 reproducibly across seeds) but how it acts only weakly (delta MLP family-split macro-F1 = 0.385 ± 0.018 on merged dataset 5-seed; 0.299 ± 0.034 on Gerasimavicius 5-seed). What looks like mechanism prediction in standard evaluations is mostly the model recognizing protein families; family-split cross-validation and a mut-only ≈ WT-only control are the tests that distinguish real mechanism signal from family-recognition leakage.

# Publication plan — bioRxiv methods note with versioned releases

The plan is to post a short methods note as v1 (priority date + early feedback), then add diagnostic depth in v2 and (if experiments confirm) generalisation in v3. Each version is a self-contained scientific document; later versions strengthen rather than replace earlier ones.

**Honest scope.** The paper is a **methodological consolidation note**, not a discovery. Its contribution is operationalising family-split CV as a quantitative leakage diagnostic with worked examples — not a novel positive finding. Realistic peer-reviewed venue: *Bioinformatics* short methods note.

---

## Complete results inventory (as of 2026-05-26)

Results 1–10 established the ESM-2 sequence-embedding story. Results 11–16 extended into proteome features, structural priors, and within-family analysis. The full arc:

| Result | Key finding | Firm? |
|---|---|---|
| 1–2 | Linear delta probe at chance; WT-only F1=0.58 but collapses under family-split | ✓ |
| 3–5 | MLP lifts delta to F1=0.41 gene-split; partially leakage | ✓ |
| 4 | ESM-2 clusters by Pfam (26× purity); 74.8% within-family mechanism agreement explains WT-only baseline | ✓ |
| 6 | Pathogenicity positive control: same pipeline AUROC = 0.878 (seed 0, RunPod variant set) / 0.742 ± 0.006 (seeds 1–4, locally-truncated variant set). Family-split-stable Δ ≈ 0 reproducibly across all seeds. Clean 5-seed mean is pending due to variant-set provenance issue. Pipeline is sound; mechanism null is real | ✓ 5 seeds (with variant-set caveat) |
| 7 | Family-split floor F1 = 0.299 ± 0.034 (Gerasimavicius, 5-seed) / 0.385 ± 0.018 (merged, 5-seed). Seed 0 alone gave 0.364/0.352 — the multi-seed correction reveals the Gerasimavicius floor is lower than originally reported. Merged is the more reliable headline | ✓ 5 seeds |
| 8 | Within-family ESM-2 delta signal: ion channels (PF00520) AUROC=0.659 (GOF/DN 2-class); doesn't generalise to related families | ✓ (single seed, small n) |
| 9 | Contrastive projection (cross-family positives only) pushes family-split floor to F1=0.397; equal gene/family-split lift confirms real cross-family signal | ✓ |
| 10 | Clan-holdout F1=0.299: ~half of family-split signal is clan memorisation, half is genuine. Cupin generalises (F1=0.536); ion channels collapse (F1=0.190) | ✓ (single seed) |
| 11 | Stage 0 pilot: 4 public gene features (pLI, LOEUF, mis_z, paralog_count) → F1=0.417 family-split, 5/5 seeds | ✓ |
| 12 | 37-feature proteome matrix assembled (gnomAD, paralogs, HPA, PaxDb, BioPlex, ClinGen). 2,424 × 37, family-mean-centred residuals included | ✓ (data collection) |
| 13 | Proteome (V2, F1=0.462) consistently beats ESM-2 delta (V1, F1=0.382) by +0.08; combination not reliably additive (Gate 2: 2/5 seeds). Constraint + dosage are load-bearing features | ✓ 5 seeds |
| 13-T4 | Feature ablation: constraint and dosage are load-bearing (ΔF1 +0.04 each). PPI_degree adds nothing. DN AUROC *hurt* by constraint features (conflation with LOF) | ✓ 5 seeds |
| 14 | Clinical utility: within ClinGen HI=3, paralog_count alone (AUROC=0.746) beats full 37-feature model (0.650). Monotonic GOF scaling 0.7% → 4.5% → 9.3% across paralog tertiles | ✓ 5 seeds |
| 15 | Badonyi 2024 SVM (pDN/pGOF/pLOF, 3 features) beats ESM-2 (1280-dim) by +0.104 and proteome (37-dim) by +0.022. V2+bad: F1=0.511, DN AUROC=0.827 (project high). ESM-2 is the dispensable modality | ✓ 5 seeds |
| 15-AppA | Leakage triage: V_bad performs *better* on OUT-of-Badonyi-training genes than IN. No label leakage — lift is real | ✓ |
| 15-AppB | MMseqs2-20 cluster-split: all result_15 conclusions hold (Δ ≤ 0.03 across all variants). Matches Saadat & Fellay 2025 protocol | ✓ |
| 16 | Within-family LOGO CV: residual proteome F1=0.514 > raw proteome F1=0.484 > Badonyi residuals (=0.449=raw, no within-family variation). Homeodomains (n=30, F1=0.633) are the anchor example | ✓ (deterministic LOGO) |
| 16-addendum | Badonyi's raw published model survives family-split holdout (ROBUST by pre-registered criterion) but shows per-gene training-set fit: LOF AUROC 0.625 (in-training) vs 0.472 (never-seen). Does not affect V_bad/V2+bad validity | ✓ |
| 17 | AlphaMissense on the result_6 ClinVar set (n=16,334): overall AUROC 0.940, per-family AUROC mean 0.948 ± 0.046 across 182 Pfam families, 0% below 0.70. result_6 family-robustness generalises to the published predictor in clinical use. Caveat: ClinVar–AM training-logic overlap inflates absolute number; per-family *distribution* metric unaffected. ProteinGym replication pending | ✓ |
| 18 | AlphaMissense on ProteinGym v1.3 human DMS assays (n=91): per-assay AUROC mean 0.721 ± 0.150, 32% below 0.70, 14% below 0.60. Tight ClinVar distribution does **not** transfer to physical DMS labels; the wider distribution is interpretable (Tsuboyama mini-protein stability assays are OOD; classic disease genes still hit ≥0.90). Reframes result_17 as a within-curation-distribution claim, not a general "VEPs are family-robust" claim | ✓ |

---

## The story in one paragraph

ESM-2 delta embeddings predict mutation pathogenicity (delta MLP AUROC 0.74–0.88 across replications, family-split-stable in all — gene→family Δ ≈ 0 reproducibly) but mechanism weakly (family-split floor F1 = 0.385 ± 0.018 on merged dataset 5-seed; 0.299 ± 0.034 on Gerasimavicius 5-seed). The mechanism null is explained by ESM-2's strong Pfam family clustering (26× purity) and 74.8% within-family mechanism agreement. Simple public gene-level features (proteome, V2) outperform ESM-2 by +0.08 F1; Badonyi's structural SVM (3 features) beats ESM-2 by +0.10; combining proteome + Badonyi achieves the project's best result (F1=0.511, DN AUROC=0.827). ESM-2 is the dispensable modality. Within protein families, the signal comes from within-family variation in gene-level proteome features (residual proteome F1=0.514), not from Badonyi's structural prior (which carries no within-family variation). For ion channels, however, ESM-2 mutation-level context (delta AUROC=0.659 within PF00520) outperforms gene-level features — pointing to a resolution-dependent split: cross-family mechanism is in gene-level biology, within-family mechanism in specific mutations.

---

## v1 — controlled pathogenicity–mechanism dissociation (target: ~1 week)

### Working title
*"Frozen ESM-2 encodes mutation pathogenicity strongly and disease mechanism weakly: a controlled dissociation under family-split cross-validation"*

### Abstract (draft)

Frozen ESM-2 embeddings encode mutation pathogenicity (delta MLP AUROC 0.878 single-seed on 17,236 ClinVar pathogenic-vs-benign variants, 0.742 ± 0.006 under multi-seed replication on a slightly different variant set; gene→family Δ ≈ 0 reproducibly across seeds — family-split-stable in all configurations) but disease mechanism weakly (delta MLP family-split macro-F1 = 0.385 ± 0.018 on merged dataset 5-seed; 0.299 ± 0.034 on Gerasimavicius 5-seed; GOF AUROC 0.655 ± 0.014 / 0.557 ± 0.036 respectively). **The dissociation holds on the same embeddings, the same probe, and the same cross-validation scheme** — ruling out methodology as the explanation. The apparent above-chance gene-split mechanism signal is largely family-recognition leakage on both datasets — a structural property of standard CV designs on family-clustered disease gene sets. The strongest mechanism-class-level signal that survives family-split is GOF on the merged dataset (delta MLP AUROC 0.66; WT-only linear AUROC 0.73–0.80 — though the WT-only number captures gene identity rather than mutation-specific information). DN and LOF do not exceed AUROC 0.55 and 0.69 respectively. Complementing ESMGain (Glaser et al. 2025), which shows fine-tuned ESM-2 captures GOF in DMS data, we show that the cross-family GOF signal exists in frozen embeddings — but is much smaller than gene-split evaluations suggest, and much smaller than what the same model encodes about damage. **Family-split CV is necessary to recover this dissociation; without it, gene-split evaluations inflate mechanism performance materially.** This framework reconciles apparent positive counterexamples: reports of strong PLM-based mechanism prediction within restricted protein families (e.g., MissION on ion channels, AUROC 0.925) are consistent with our null because within a single Pfam family the family-identity signal that our family-split CV removes is precisely the signal those reports exploit. PLM-based mechanism prediction succeeds within any sufficiently homologous gene set and fails when cross-family generalisation is required.

### What's in v1

| Section | Content |
|---|---|
| Methods | ESM-2 650M, mean-pooled per-variant or per-gene embeddings, logistic regression + MLP probes, 5-fold gene-split AND family-split CV |
| Dataset | Gerasimavicius (948 genes) + merged with G2P/ClinVar pathogenic (1,985 genes total) + ClinVar 17,236 pathogenic/benign variants (944 genes) for the pathogenicity task |
| **Co-headline 1 — Pathogenicity** | **Delta MLP AUROC 0.878 (seed 0) / 0.742 ± 0.006 (seeds 1–4, slightly different variant set); family-split-stable in all configurations (gene→family Δ ≈ 0 across all seeds). Clean 5-seed mean on a consistent variant set is pending. Linear probe is sufficient. Confirms ESM-2 deltas carry per-variant damage signal that survives strict holdout regardless of which variant set is used.** |
| **Co-headline 2 — Mechanism** | **Delta MLP family-split macro-F1 = 0.385 ± 0.018 (merged, 5-seed — primary headline) / 0.299 ± 0.034 (Gerasimavicius, 5-seed — replication). GOF AUROC 0.655 ± 0.014 / 0.557 ± 0.036. DN and LOF do not exceed AUROC 0.59 and 0.67. 62.8% leakage fraction exact and seed-invariant on Gerasimavicius (structural property of the dataset).** |
| **The dissociation** | **Same embeddings, same probe family, same CV scheme — pathogenicity AUROC 0.74–0.88 vs mechanism floor F1 ≈ 0.30–0.39 (above the majority baseline by 0.02–0.07 depending on dataset). The dissociation is the central finding and is family-split-stable on both sides.** |
| Supporting methodology | **The leakage diagnostic**: 61–63% of above-chance gene-split mechanism signal is family-recognition leakage on both datasets. |
| **Reconciling MissION** | PLM mechanism prediction succeeds within homologous subfamilies and fails cross-family. The two findings are consistent, not contradictory. Falsifiable rule: *if a PLM mechanism predictor can't demonstrate performance under family-split CV, it has measured family recognition, not mechanism.* |
| Per-class table | GOF / DN / LOF AUROC under gene-split and family-split CV, both datasets. WT-only included as a contrast. |
| Honest scope | "On Gerasimavicius + G2P + ClinVar, ESM-2 650M, frozen mean-pooled, gene-disjoint and family-disjoint CV" |

### What's deliberately NOT in v1

- Family-clustering quantification (k-NN purity, family probe, 74.8% within-family agreement) — saved for v2
- Proteome features, Badonyi comparison, clinical utility — saved for v2 or v3
- Multi-seed replication
- Second model (SaProt / ESM-3)
- Within-family analysis as a result section
- Stability-subspace projection
- Per-residue delta exploration as headline

### Why v1 is safe to post

- The dissociation is controlled by experimental design: same embeddings, same probe family, same CV scheme, two tasks, opposite outcomes.
- Two-dataset replication of the mechanism floor (5-seed mean ± std): Gerasimavicius F1 = 0.299 ± 0.034, merged F1 = 0.385 ± 0.018. Merged is the more stable headline (lower std); Gerasimavicius result tightens around a lower floor than the seed-0 number originally suggested.
- Substantial gene-split → family-split drop on both datasets (the leakage diagnostic is real; the exact percentage shifts under multi-seed but the directional finding holds).
- Convergence across 6 (method × dataset × feature) combinations all in the 0.34–0.39 band.
- MissION reconciliation addresses the strongest apparent counterexample.
- Reproducibility: link to `esm2_mechanism/` scripts and JSON outputs.

### The one figure
**Two-panel figure** showing the dissociation directly:
- **Panel A — Pathogenicity**: AUROC for delta MLP, gene-split vs family-split, with the ~0 drop visible.
- **Panel B — Mechanism**: per-class AUROC (GOF / DN / LOF) under gene-split vs family-split, both datasets.

### Page target
**6–8 pages.** Abstract + intro + methods + one results table + one figure + brief discussion.

### What v1 must NOT claim
- "ESM-2 encodes mechanism" (claim is the bounded dissociation)
- "We provide a novel methodology" (family-split CV is not novel; v1 makes the narrow claim that it changes the numbers materially)
- "Our results contradict X" (we confirm folk wisdom with controls)
- "Family-split CV is the necessary diagnostic for the field" (needs the clustering quantification in v2)
- "Mechanism is unlearnable from PLM representations" (claim what the data supports: under this setup, the floor is ~0.36)

### Prior-work positioning

| Paper | Year | Key features | How it differs |
|---|---|---|---|
| **ESMGain** (Glaser et al.) | 2025 | Fine-tunes ESM-2 (35M); GOF with rBME metric; DMS datasets | Frozen vs fine-tuned; no family-disjoint CV on large human disease gene set; DMS vs ClinVar mechanism labels |
| **Oliveira et al.** | 2025 | ESM-2 + RF/XGBoost/LR for GOF/LOF/Neutral | No family-disjoint CV; lower GOF performance |
| **ClearVariant** | 2025 | Attention model on ESM-2 for GOF/LOF | Deeper architecture; our claim is that signal is in the frozen mean-pool |
| **PreMode** (Zhong et al.) | 2025 | ESM-2 + structure + MSA | Multimodal; our setup is intentionally minimal |
| **ESMRank** | 2026 | ESM-2 + multimodal for pathogenicity | No family-disjoint CV on mechanism |

The specific frozen-mean-pool + linear-probe + family-split setup is novel. v1 is positioned as an **interpretability/representation-analysis result**, not a competitive predictor.

---

## v2 — add diagnostic, proteome + Badonyi modality comparison (target: ~3 weeks after v1)

### What's added in v2

1. **Family-clustering quantification** — k=5 family purity 26× chance, 50-way family probe 27× majority baseline, 74.8% within-family mechanism agreement. Makes "family-split CV is necessary" defensible rather than asserted.

2. **Pfam-coverage methodological note** — the worked example where silent CV failure inflated Δ from +0.011 to +0.077 when annotations were extended. Cautionary tale for anyone applying family-split CV.

3. **Multi-seed replication** — five seeds on all v1 numbers.

4. **Proteome modality comparison** (results 11–13) — gene-level public features (gnomAD constraint, paralogs, HPA, PaxDb, BioPlex, ClinGen) outperform frozen ESM-2 delta by +0.08 F1 under family-split, 5/5 seeds. V2 + V1 combination not reliably additive (Gate 2: 2/5 seeds). Load-bearing features are constraint and dosage; PPI degree adds nothing. Per-gene scoring confirms result is not variant-count-weighted (V2 advantage grows to +0.101 per-gene).

5. **Badonyi structural prior** (result 15) — Badonyi 2024 SVM (3 features: pDN/pGOF/pLOF) beats ESM-2 (+0.104 F1) and beats proteome (+0.022 F1). V2+bad is the best overall combination (F1=0.511, DN AUROC=0.827). ESM-2 is the dispensable modality (V1+bad < V_bad). Both leakage checks pass: no label leakage (AppA); holds under MMseqs2-20 cluster-split (AppB, Δ ≤ 0.03). The modality ordering is now: structural priors > proteome > ESM-2.

6. **Feature ablation** (result 13-T4) — dropping constraint costs ΔF1=0.040; dropping dosage costs 0.043; dropping PPI degree has no effect. DN AUROC is *hurt* by constraint (constraint pushes predictions toward LOF, conflating DN with LOF). Paralog count is DN-specific (ΔDN=−0.015 when dropped).

7. **Clinical utility framing** (result 14) — within ClinGen HI=3 genes, paralog_count alone achieves AUROC=0.746. Monotonic GOF frequency scaling across paralog tertiles (0.7%→4.5%→9.3%) validates the gene balance hypothesis. Full 37-feature model (AUROC=0.650) does not beat the single best predictor. Operating-point numbers are poor (recall=0.235, precision=0.160 at P_GOF>0.4). Clinical utility case reduces to: paralog count as a simple, interpretable, free predictor of GOF direction within dosage-sensitive genes.

8. **Badonyi raw model holdout** (result 16 addendum) — Badonyi's published model passes the family-recognition leakage test (ΔAUROC ≥ −0.03 by pre-registered criterion) but shows per-gene training-set fit for LOF (AUROC 0.625 in-training vs 0.472 never-seen). Affects how Badonyi's published LOF numbers should be cited; does not affect V_bad/V2+bad validity.

### Updated story for v2

> ESM-2 delta embeddings predict mutation pathogenicity (AUROC 0.74–0.88 across replications, family-split-stable in all — gene→family Δ ≈ 0 reproducibly) but not mechanism (family-split floor F1 = 0.385 ± 0.018 merged 5-seed, 0.299 ± 0.034 Gerasimavicius 5-seed). The mechanism null is explained by ESM-2's strong Pfam family clustering (26× purity) and 74.8% within-family mechanism agreement. When the same mechanism prediction task is given to either (a) simple public gene-level features or (b) Badonyi's structural SVM (3 features), both outperform the 1,280-dimensional frozen ESM-2 embeddings — the former by capturing gene-level biology (constraint, paralogs, abundance) and the latter by capturing structural geometry (variant clustering, interface exposure, FoldX ΔΔG). Combining proteome + Badonyi achieves the best results (F1=0.511, DN AUROC=0.827). ESM-2 is redundant once the other modalities are present. This sharpens the central claim: frozen ESM-2 lacks not just mechanism signal in general, but specifically the structural geometric information that best distinguishes mechanism classes.

### What v2's title becomes
*"Frozen ESM-2 encodes pathogenicity but not mechanism: family-split CV as a leakage diagnostic and structural priors as the missing modality"*

### Page target
**12–15 pages.** Adds family-clustering section, modality comparison section, feature ablation, clinical utility paragraph, Badonyi holdout note.

### v2's stronger claims (defensible with added evidence)
- "Family-split CV is necessary; 62% of apparent mechanism signal is family leakage"
- "Public gene-level features outperform frozen ESM-2 for mechanism prediction — consistently, across 5 seeds"
- "Structural priors (Badonyi SVM, 3 features) outperform ESM-2 (1280 dims) — ESM-2 lacks structural geometric information that mechanism prediction requires"
- "Proteome + Badonyi is additive; ESM-2 is the dispensable modality"
- "Paralog count is a simple, free predictor of GOF direction within haploinsufficient genes"

---

## v3 — within-family analysis + generalisation (target: ~2–3 months after v1)

### What's added in v3

1. **Within-family mechanism** (result 16) as a standalone section — family-residual proteome features achieve F1=0.514 in LOGO CV across 24 families. Homeodomains (n=30, F1=0.633) are the anchor. Badonyi structural residuals add nothing within families (raw=residual=0.449) — structural prior is entirely cross-family signal. Ion channels are the null for gene-level features but respond to ESM-2 mutation context (AUROC=0.659 from result 8). This establishes the resolution-dependent picture: cross-family mechanism = gene-level biology; within-family mechanism = specific mutations (for at least some families).

2. **Second mechanism dataset (DDG2P)** — replicate the full modality comparison on EBI G2P's ~2,000-gene set. If the pattern holds, generalisation is established.

3. **SaProt first, ESM-3 second** — structure-aware PLMs as steelmen. SaProt adds structure tokens only; if it recovers DN/LOF where ESM-2 fails, structure is the missing ingredient (consistent with Badonyi result). If SaProt also fails, the negative result is model-general.

4. **Evo2 comparison** — if relevant (tests whether genomic-context features recover signal that per-protein features miss).

5. **Per-class PR-AUC and calibration** — for clinical relevance.

### v3 title depends on outcomes
- SaProt + ESM-3 + DDG2P confirm the pattern: *"Structure-aware pretraining is necessary for cross-family mechanism encoding in protein language models"*
- SaProt recovers mechanism but ESM-2 doesn't: *"Structure-aware protein language models recover disease mechanism that sequence-only models miss"*
- SaProt + ESM-3 also fail: paper ends at v2 with strengthened negative claim

### Page target
**15–20 pages.** Target *Bioinformatics* methodological note. *Nat Methods* / *Nat Commun* not realistic without a strong positive flip side.

---

## Versioning strategy notes

### Why v1 must be narrower than what you eventually want to claim

bioRxiv versions are a public scientific record. v1 scope is a strict subset of v3 scope — each version strengthens rather than replaces the prior.

### What stays the same across all versions

- The GOF AUROC 0.73–0.80 family-split number (WT-only); delta MLP GOF AUROC 0.557 ± 0.036 (Gerasimavicius) / 0.655 ± 0.014 (merged) under multi-seed
- The DN/LOF chance-level family-split numbers from ESM-2 delta
- The pathogenicity gene→family Δ ≈ 0 (family-split stability) reproducibly across all seeds and variant sets
- The pathogenicity AUROC in the 0.74–0.88 range across replications (clean 5-seed mean pending consistent-variant-set replication)
- The 62.8% leakage fraction on Gerasimavicius (exact, seed-invariant — structural property of the dataset)
- The proteome > ESM-2 ordering (V2 F1=0.462 vs V1 F1=0.382, 5 seeds)
- The Badonyi > proteome > ESM-2 ordering (V_bad 0.484 > V2 0.462 > V1 0.380)

### What might change across versions

- Multi-seed replication might shift point estimates ±0.02–0.04
- DDG2P / SaProt could falsify or strengthen the modality ordering
- Within-family result is robust for homeodomains (n=30) but aggregate F1 is dominated by PF00046

### What gets cut entirely if results don't cooperate

- If SaProt + DDG2P don't replicate: paper ends at v2
- If within-family analysis fails on DDG2P: one sentence in discussion only
- If a 2026 paper preempts the frozen-probe GOF finding before v1 posts: v1 becomes "we also observed this" preliminary note

---

## Practical timeline

| Stage | Time | Blocker |
|---|---|---|
| v1 writing | 3–5 days | None — all data in hand |
| v1 figure | 1 day | None |
| v1 polish + post | 1 day | None |
| **v1 live** | **~1 week from start** | |
| v2 — add clustering, proteome, Badonyi writeup | 5–7 days | Multi-seed reruns for v1 numbers (cheap on cached embeddings) |
| v2 figure additions | 1–2 days | None |
| **v2 live** | **~3–4 weeks from start** | |
| v3 — DDG2P embedding extraction | ~3 days GPU | RunPod |
| v3 — SaProt embedding extraction | ~3 days GPU | RunPod |
| v3 — ESM-3 embedding extraction | ~5–7 days GPU | RunPod + EvolutionaryScale access |
| v3 — within-family on DDG2P | 1 day | After embeddings |
| v3 writing & figures | 2–3 weeks | Above experiments must complete |
| **v3 live** | **~2–3 months from start** | |

---

## Files needed before v1 can post

| File | Status |
|---|---|
| Headline numbers JSON (delta MLP, Gerasimavicius + merged) | ✓ `results/20260524_baseline_run/run_0/mlp_results_seed0.json`, `mlp_merged_results_seed0.json` |
| Pathogenicity positive control | ✓ `results/20260524_baseline_run/run_0/pathogenicity_control.json` |
| Per-class AUROCs all conditions | ✓ in JSONs above |
| Multi-seed for v1 numbers | ✗ Need to run 5 seeds — cheap on cached embeddings |
| Figure | ✗ Need to write `plot_publication_v1.py` |
| LaTeX template | ✗ Pick one (`bioRxiv-style.cls` works) |
| ESMGain citation context | ✗ Read ESMGain paper carefully and confirm framing contrast |

## Additional files available for v2 (all in hand)

| File | Status |
|---|---|
| Family clustering metrics | ✓ `results/20260524_baseline_run/run_0/family_clustering.json` |
| Proteome feature matrix | ✓ `data/proteome_features_aligned.npy`, `data/gene_proteome_features.tsv` |
| Proteome modelling results (V1–V4, 5 seeds) | ✓ `results/proteome_mechanism/proteome_mechanism_summary.json` |
| Per-gene scoring + ablation | ✓ `results/proteome_mechanism/per_gene_summary.json`, `v2_ablation_summary.json` |
| Clinical utility results | ✓ `results/clinical_utility/hi3_family_split_summary.json` |
| Badonyi features | ✓ `data/badonyi_features_aligned.npy` |
| Badonyi modelling results (V_bad, V2+bad, 5 seeds) | ✓ `results/badonyi_mechanism/badonyi_mechanism_summary.json` |
| Badonyi leakage triage (AppA) | ✓ `results/badonyi_leakage/` |
| MMseqs2 cluster-holdout (AppB) | ✓ `results/mmseqs_cluster_holdout/cluster_summary.json` |
| Badonyi raw model holdout | ✓ `results/badonyi_survival/badonyi_survival_summary.json` |
| Within-family LOGO CV | ✓ `results/within_family/within_family_summary.json` |
| Clan-holdout results | ✓ `results/20260524_baseline_run/run_0/clan_holdout_results_seed0.json` |
| Contrastive projection results | ✓ `results/20260524_baseline_run/run_0/contrastive_results_geras_seed0.json` |

---

## Single-line summary

**v1 = frozen-probe dissociation (narrow, fast, safe). v2 = + family clustering + proteome + Badonyi modality comparison + clinical utility + holdout robustness. v3 = + within-family as result + DDG2P + SaProt/ESM-3. Each version is a publishable scientific record; later versions strengthen, never replace.**
