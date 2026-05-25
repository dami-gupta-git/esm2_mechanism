# Publication plan — bioRxiv methods note with versioned releases

The plan is to post a short methods note as v1 (priority date + early feedback), then add diagnostic depth in v2 and (if experiments confirm) generalisation in v3. Each version is a self-contained scientific document; later versions strengthen rather than replace earlier ones.

**Honest scope.** The paper is a **methodological consolidation note**, not a discovery. Its contribution is operationalising family-split CV as a quantitative leakage diagnostic with worked examples — not a novel positive finding. The within-family analysis (kinase delta GOF AUROC 0.777, n=24, std 0.13) is too underpowered to stand alone and belongs only as a single directional sentence in the discussion. Realistic peer-reviewed venue: *Bioinformatics* short methods note.

---

## v1 — controlled pathogenicity–mechanism dissociation (target: ~1 week)

### Working title
*"Frozen ESM-2 encodes mutation pathogenicity strongly and disease mechanism weakly: a controlled dissociation under family-split cross-validation"*

### Abstract (draft)

Frozen ESM-2 embeddings encode mutation pathogenicity strongly (delta MLP AUROC 0.88, family-split-stable on 17,236 ClinVar pathogenic-vs-benign variants) but disease mechanism weakly (GOF AUROC 0.627 / 0.635 delta MLP family-split on Gerasimavicius / merged datasets respectively; macro-F1 floor 0.35–0.39 consistent across two datasets and two probe types). **The dissociation holds on the same embeddings, the same probe, and the same cross-validation scheme** — ruling out methodology as the explanation. 61–63% of the apparent above-chance gene-split mechanism signal is family-recognition leakage on both datasets — a structural property of standard CV designs on family-clustered disease gene sets. The strongest mechanism-class-level signal that survives family-split is GOF (delta MLP AUROC 0.63; WT-only linear AUROC 0.73–0.80 — though the WT-only number captures gene identity rather than mutation-specific information). DN and LOF do not exceed AUROC 0.55 and 0.69 respectively. Complementing ESM-Effect/ESMGain (Tang et al. 2025), which shows fine-tuned ESM-2 captures GOF in DMS data, we show that the cross-family GOF signal exists in frozen embeddings — but is much smaller than gene-split evaluations suggest, and much smaller than what the same model encodes about damage. **Family-split CV is necessary to recover this dissociation; without it, gene-split evaluations inflate mechanism performance by ~62%.** This framework reconciles apparent positive counterexamples: reports of strong PLM-based mechanism prediction within restricted protein families (e.g., MissION on ion channels, AUROC 0.925) are consistent with our null because within a single Pfam family the family-identity signal that our family-split CV removes is precisely the signal those reports exploit. PLM-based mechanism prediction succeeds within any sufficiently homologous gene set and fails when cross-family generalisation is required.

### What's in v1

| Section | Content |
|---|---|
| Methods | ESM-2 650M, mean-pooled per-variant or per-gene embeddings, logistic regression + MLP probes, 5-fold gene-split AND family-split CV |
| Dataset | Gerasimavicius (948 genes) + merged with G2P/ClinVar pathogenic (1,985 genes total) + ClinVar 17,236 pathogenic/benign variants (944 genes) for the pathogenicity task |
| **Co-headline 1 — Pathogenicity** | **Delta MLP AUROC 0.88, family-split-stable (gene-split → family-split Δ = 0.002). Linear probe is sufficient. Confirms ESM-2 deltas carry strong per-variant damage signal.** |
| **Co-headline 2 — Mechanism** | **Delta MLP family-split macro-F1 0.36 (Gerasimavicius) / 0.35 (merged). GOF AUROC 0.627 / 0.635. DN and LOF do not exceed AUROC 0.55 and 0.69. The 0.35–0.39 floor replicates across 6 (probe × feature × dataset) combinations.** |
| **The dissociation** | **Same embeddings, same probe family, same CV scheme — pathogenicity AUROC 0.88 vs mechanism above-chance gain of ~0.06 macro-F1. The dissociation is the central finding.** |
| Supporting methodology | **The leakage diagnostic**: 61–63% of above-chance gene-split mechanism signal is family-recognition leakage on both datasets. Explains *why* family-split CV is the right test and why prior gene-split evaluations overstate. |
| **Reconciling apparent counterexamples (the MissION case)** | **Dedicated discussion subsection.** Reports of strong PLM-based mechanism prediction in restricted protein families (MissION on ion channels, AUROC 0.925) are not counterexamples to this work's finding — they are consistent with it. Within a single Pfam family all genes share the family-identity signal that family-split CV is designed to remove; what MissION exploits is precisely the family-recognition signal that disappears under family-split. The two findings combined predict that PLM mechanism prediction will succeed within any sufficiently homologous gene set and fail when cross-family generalisation is required. Falsifiable rule for future work: *if a PLM-based mechanism predictor cannot demonstrate its performance under family-split CV, it has measured family recognition, not mechanism.* |
| Per-class table | GOF / DN / LOF AUROC under gene-split and family-split CV, both datasets. WT-only included as a contrast (higher AUROC 0.73–0.80 for GOF but captures gene identity, not mutation effect). |
| Brief family-split justification | One paragraph explaining why family-disjoint CV matters (proteins cluster by family; family correlates with mechanism). No deep clustering analysis at v1. |
| Contrast with ESMGain | One paragraph: ESMGain fine-tunes; we don't. ESMGain doesn't compare to pathogenicity on the same pipeline; we do. Complementary, not competing. |
| Within-family pilot — discussion only | **One sentence** noting kinase delta GOF AUROC 0.777 (n=24, std 0.13) as directional motivation for future work; explicitly flagged as underpowered. Not a result. |
| Honest scope statement | "On Gerasimavicius + G2P + ClinVar, ESM-2 650M, frozen mean-pooled, gene-disjoint and family-disjoint CV" |

### What's deliberately NOT in v1

- Family-clustering quantification (k-NN purity, family probe, 74.8% within-family agreement) — saved for v2
- Pfam-coverage methodological cautionary tale — saved for v2
- Pathogenicity-mechanism dissociation **promoted to co-headline** — saved for v2 (v1 reports pathogenicity only as a positive control)
- Multi-seed replication
- Second model (SaProt / ESM-3)
- DDG2P replication
- Within-family analysis as a standalone result section (one sentence in discussion only)
- Stability-subspace projection
- Per-residue delta exploration as headline

### Why v1 is safe to post

- **The dissociation IS the central claim**, controlled by experimental design: same embeddings, same probe family, same CV scheme, two tasks, opposite outcomes. Hard to dismiss as methodological artifact.
- ESMGain is cited as the closest prior work; we do not claim to beat it on prediction accuracy and position as a complementary interpretability characterisation, not a competing predictor.
- Numbers reported with std across folds; nothing oversold.
- **Pathogenicity baseline is methodologically necessary**, not bonus context. Without it, the mechanism numbers can't be interpreted (is the pipeline broken or is mechanism really not there?). With it, the answer is unambiguous.
- **Two-dataset replication of the mechanism floor** (Gerasimavicius MLP delta family-split F1 = 0.364; merged = 0.352) — same floor across two independent disease gene sources.
- **Universal 61–63% leakage fraction**: the gene-split → family-split drop accounts for ~62% of the above-chance signal on Gerasimavicius and ~63% on the merged dataset — nearly identical across very different dataset sizes and family-coverage profiles. Structural property of the task, not a dataset artifact.
- **Convergence across methods**: 6 distinct (method × dataset × feature) combinations all yield family-split macro-F1 in the narrow 0.34–0.39 band (see result_7.md §5). Independent estimates of the same underlying ceiling.
- **MissION reconciliation**: explicitly addresses the strongest apparent counterexample in the literature (MissION ion channels, AUROC 0.925) and shows it is consistent with rather than contradictory to the null finding. Resolves an open tension that prior PLM-mechanism papers (PreMode, AlphaMissense, LoGoFunc, Badonyi & Marsh) leave unaddressed.
- Reproducibility: link to `esm2_mechanism/` scripts and JSON outputs.

### Page target
**6–8 pages.** Abstract + intro + methods + one results table for the dissociation (pathogenicity vs mechanism on the same pipeline) + one figure + brief discussion. Slightly longer than originally planned because pathogenicity is now a co-headline rather than a single-paragraph control.

### The one figure
**Two-panel figure** showing the dissociation directly:
- **Panel A — Pathogenicity (positive task)**: AUROC for delta MLP, gene-split vs family-split, with the ~0 drop visible.
- **Panel B — Mechanism**: per-class AUROC (GOF / DN / LOF) under gene-split vs family-split, both datasets. Makes the GOF survival vs DN+LOF collapse visually obvious. The gap between Panel A's AUROC (~0.88) and Panel B's per-class AUROCs (max 0.73) is the dissociation made visible.

### What v1 must NOT claim
- "ESM-2 encodes mechanism" (too broad — claim is the bounded dissociation, not a positive)
- "We provide a novel methodology" (the leakage quantification is shown but the family-clustering analysis backing it is saved for v2; family-split CV itself is not a novel idea)
- "Our results contradict X" (we contradict no one explicitly at v1; we confirm folk wisdom with controls)
- "Family-split CV is the necessary diagnostic for the field" (claim deserves the family-clustering quantification in v2; v1 makes the narrower claim that family-split changes the dissociation numbers materially on this data)
- "Mechanism is unlearnable from PLM representations" (only claim what the data supports: under this setup, on these datasets, the floor is ~0.36)

### Prior-work positioning (related work table for v1 intro)

The 2025–2026 GOF/LOF prediction literature is dense. The contribution sits in a specific gap none of these works occupy: **frozen mean-pooled ESM-2 embeddings + simple linear probe + family-disjoint CV on a large human disease gene set, with no fine-tuning, no extra modalities, and no architectural complexity.**

| Paper | Year | Key features | How it differs from this work |
|---|---|---|---|
| **ESM-Effect / ESMGain** (Glaser et al.) | 2025 | Fine-tunes ESM-2 (35M); emphasizes GOF with rBME metric | Strongly advocates fine-tuning over frozen embeddings; shows static/frozen embeddings underperform on their setup. Focus on DMS datasets, not large-scale human disease genes with family-disjoint CV. **Our finding (frozen embeddings retain GOF signal) is a direct complement: the signal is already there before fine-tuning, at least on disease-level mechanism labels.** |
| **Prediction of GOF/LOF/Neutral in Missense Variants** (Oliveira et al.) | 2025 | ESM-2 + classical ML (RF, XGBoost, LR) for GOF/LOF/Neutral | Uses embeddings but not frozen mean-pooled + LR only; reports lower GOF performance; does not run family-disjoint CV on a large disease gene set. |
| **ClearVariant** | 2025 | Attention-based model on ESM-2 for GOF/LOF | Requires a deeper learned architecture; our claim is that the signal is recoverable by a simple logistic regression on the frozen mean-pool — no architecture engineering required. |
| **PreMode** (Zhong et al.) | 2025 | Multimodal mode-of-action: ESM-2 + structure + MSA | Adds structure and MSA modalities. Our setup is intentionally minimal — frozen ESM-2 alone — which strengthens the interpretive claim about what ESM-2 representations themselves contain. |
| **ESMRank** | 2026 | ESM-2 embeddings + multimodal features for pathogenicity | Looks at GOF genes but does not evaluate variant-level GOF prediction with family-disjoint CV. |

**The specific frozen-mean-pool-plus-linear-probe-with-family-split setup is, to our knowledge, novel.** All four 2025 papers either fine-tune (ESMGain, ClearVariant), use heavier classical ML pipelines (Oliveira), or combine ESM-2 with additional modalities (PreMode, ESMRank). None of them isolate the question *"what mechanism information is already in frozen ESM-2 representations, without fine-tuning or auxiliary features, after blocking the family-recognition shortcut?"*

### What this means for v1's framing

v1 is positioned as **an interpretability/representation-analysis result**, not a competitive predictor. We are not claiming our setup beats ESMGain or ClearVariant on prediction accuracy; we are characterising what ESM-2's pretrained representation already contains. That framing is robust because:

- ESMGain shows fine-tuning helps → consistent with our finding that frozen signal exists but is modest
- Oliveira shows simpler ML pipelines underperform → consistent with the idea that current setups don't fully exploit what's in the embeddings
- PreMode shows adding modalities helps → consistent with frozen-ESM-2-only being a floor, not a ceiling
- Our contribution: characterise that floor precisely, with the leakage diagnostic that none of the above runs

The honest one-line positioning: *"While recent work focuses on building better mechanism predictors (via fine-tuning, additional modalities, or deeper architectures), we ask what mechanism information is already encoded in frozen ESM-2 representations, evaluated with family-disjoint cross-validation."*

---

## v2 — add diagnostic and dissociation framing (target: ~3 weeks after v1)

Pathogenicity is already in v1 as a positive control. v2 promotes it from "control" to "co-headline" by adding the family-clustering diagnostic that makes the dissociation a methodological contribution rather than just two numbers in a table.

v1 already contains the dissociation as the co-headline finding. v2 adds the *causal explanation* (family clustering) and the *methodological cautionary tale* (Pfam-coverage bug) to upgrade the family-split CV claim from "we found it works on our data" to "we know why it works and we provide a diagnostic with worked failure modes."

### What's added in v2

1. **Family-clustering quantification** — k=5 family purity 26× chance, 50-way family probe 27× majority baseline, 74.8% within-family mechanism agreement. Gives the family-split CV justification quantitative teeth and makes the "family-split CV is necessary" claim defensible. Without this, v1's family-split is just a stricter test we ran; with this, v1's family-split is a principled response to a quantified confound.

2. **Pfam-coverage methodological note** — the worked example showing how silent CV failure (Δ=+0.011 inflated to +0.077 when annotations were extended) makes leakage diagnostics non-trivial to apply correctly. Useful cautionary tale that anyone applying family-split CV needs to know about.

3. **Multi-seed replication** — five seeds on all v1 numbers, tighten confidence intervals from std-across-folds to std-across-seeds.

4. **Expanded dissociation discussion** — section comparing this work to PreMode, AlphaMissense, LoGoFunc, Badonyi & Marsh. v1 only cites these briefly; v2 positions the dissociation explicitly as the first controlled side-by-side demonstration of a claim multiple prior papers state qualitatively as motivation.

### What v2's title becomes
*"Why does frozen ESM-2 encode pathogenicity but not mechanism? Family-clustering as the causal explanation and family-split CV as the diagnostic"*

### Page target
**10–14 pages.** Adds one section on family clustering, one on the Pfam-coverage cautionary tale, expanded related-work discussion, and multi-seed numbers throughout.

### v2's stronger claims (now defensible because of the added diagnostic)
- "Family-split CV is necessary to detect family-recognition shortcuts; we provide the quantitative explanation (74.8% within-family mechanism agreement) and worked examples"
- "Silent CV failure from incomplete annotation coverage is a non-trivial methodological pitfall — we show a worked example where leakage looked 7× smaller than it actually was"
- "The pathogenicity-mechanism dissociation observed in v1 has a specific causal mechanism: family identity is strongly encoded but mechanism class is only weakly encoded beyond family"

---

## v3 — generalisation across models and datasets (target: ~2–3 months after v1)

### What's added in v3

1. **Second mechanism dataset (DDG2P)** — replicate the GOF-survives-family-split finding on an independent ~2,000-gene mechanism dataset curated by EBI G2P. If the pattern replicates, generalisation is established.

2. **Second model: SaProt first, ESM-3 second** — structure-aware PLMs as steelmen. Run in this order for principled reasons:
   - **SaProt** adds *structure tokens only* (foldseek 3Di). If SaProt recovers DN/LOF mechanism where ESM-2 fails, structure is the missing ingredient. If SaProt also fails, the negative result becomes much stronger.
   - **ESM-3** adds *structure + function tokens* (GO terms etc.). If ESM-3 recovers mechanism but SaProt didn't, the cause is function-token supervision, not structure. If both fail, the negative claim generalises across pretraining objectives.
   - **Why SaProt before ESM-3:** SaProt is fully open-weight (ESM-3 is partially gated through EvolutionaryScale), cheaper per embedding, and cleanly isolates *structure* as the variable. ESM-3 alone would conflate two pretraining differences (structure + function) and not tell us which mattered.
   - **Why both, not just one:** SaProt + ESM-3 together let us answer "which pretraining ingredient (if any) recovers mechanism information," which is more decisive than either alone.

3. ~~**Within-family mechanism analysis**~~ — **deferred**. Pilot on existing data (PF00069 kinase, n=24, delta GOF AUROC 0.777, F1 std 0.13; PF00071 Ras, n=13, degenerate due to 90% GOF) showed sample sizes are too small for a publishable standalone finding (≤24 genes per family, 5 per test fold). Belongs as one directional sentence in v1's discussion, not as a planned v3 experiment. To revisit only if a labeled cohort with ≥50 genes per family in ≥2 mechanism classes becomes available (e.g., full human kinome with curated mechanism labels — does not currently exist).

4. **Evo2 comparison** — if relevant. Tests whether genomic-context features (paralogs, dosage) recover signal that pure-protein features miss.

5. **Per-class PR-AUC and calibration** — for clinical relevance.

### What v3's title becomes
Depends on outcomes:
- If SaProt + ESM-3 + DDG2P all confirm the GOF-selective pattern: *"Cross-family disease mechanism is selectively encoded for gain-of-function in protein language models: a multi-model, multi-dataset characterisation"*
- If SaProt recovers mechanism but ESM-2 doesn't: *"Structure-aware protein language models recover disease mechanism that sequence-only models miss"*
- If ESM-3 recovers mechanism but SaProt doesn't: *"Function-aware pretraining is necessary for mechanism encoding in protein language models"*
- If SaProt + ESM-3 also fail: paper ends at v2 with a strengthened negative claim (no v3)

### Page target
**12–18 pages.** Methods-note expansion. Target *Bioinformatics* methodological note. *Nat Methods* / *Nat Commun* are **not realistic targets** given the within-family pilot did not produce a publishable positive flip, the GOF finding overlaps materially with ESMGain, and the dissociation finding overlaps with PreMode + AlphaMissense. Set expectations accordingly.

---

## Versioning strategy notes

### Why v1 must be narrower than what you eventually want to claim

bioRxiv versions are a public scientific record. If v1 says X and v3 walks back to Y, readers cite v1's X. Worst case: v1 gets cited as a retraction. Best practice: v1 scope is a strict subset of v3 scope, so each version *strengthens* rather than *replaces* the prior.

### What stays the same across all versions

- The GOF AUROC 0.73–0.80 family-split number
- The DN/LOF chance-level family-split numbers
- The contrast with ESMGain (frozen vs fine-tuned)
- The methods (mean-pooled ESM-2 650M, logreg + MLP, gene-split + family-split CV)

These are the load-bearing claims. They're already supported by the data in `results/20260524_baseline_run/run_0/`.

### What might change across versions

- Multi-seed replication might shift point estimates ±0.02–0.04 — flag this in v1 by reporting fold std
- ~~Re-running MLP delta on merged dataset with corrected Pfam might shift the MLP numbers (currently pending)~~ ✅ **Done**: merged MLP delta_mean family-split F1 = 0.352 (delta_pos = 0.336), confirming the Gerasimavicius floor. v1 reports both datasets.
- DDG2P / SaProt could falsify the GOF claim — v3 will report honestly either way

### What gets cut entirely if results don't cooperate

- If SaProt + DDG2P don't replicate the GOF survival, the paper ends at v2 (with a discussion noting the boundary)
- If within-family analysis is positive but cross-family isn't, v3 reframes around within-family
- If a 2026 paper preempts the frozen-probe GOF finding before v1 posts, the paper ends at v1 as a "we also observed this" preliminary note

---

## Practical timeline

| Stage | Time | Blocker |
|---|---|---|
| v1 writing | 3–5 days | None — all data in hand |
| v1 figure | 1 day | None |
| v1 polish + post | 1 day | None |
| **v1 live** | **~1 week from start** | |
| v2 — pathogenicity & clustering writeup | 5–7 days | Multi-seed reruns (cheap on cached embeddings) |
| v2 figure additions | 1 day | None |
| **v2 live** | **~3 weeks from start** | |
| v3 — DDG2P embedding extraction | ~3 days GPU | RunPod availability |
| v3 — SaProt embedding extraction | ~3 days GPU | RunPod availability (run before ESM-3 — cleaner control) |
| v3 — ESM-3 embedding extraction | ~5–7 days GPU | RunPod availability + EvolutionaryScale access for larger ESM-3 variants |
| v3 — within-family analysis | 1 day | After embeddings |
| v3 writing & figures | 2–3 weeks | Above experiments must complete |
| **v3 live** | **~2–3 months from start** | Tighter timeline if ESM-3 deferred to v4 |

---

## Files needed before v1 can post

| File | Status |
|---|---|
| Headline numbers JSON (Option B WT, merged) | ✓ `results/20260524_baseline_run/run_0/option_b_gene_level_wt_merged.json` |
| MLP probe results (Gerasimavicius) | ✓ `results/20260524_baseline_run/run_0/mlp_results_seed0.json` |
| MLP probe results (merged dataset) | ✓ `results/20260524_baseline_run/run_0/mlp_merged_results_seed0.json` |
| Pathogenicity positive control | ✓ `results/20260524_baseline_run/run_0/pathogenicity_control.json` |
| Per-class AUROCs all conditions | ✓ in JSONs above |
| Multi-seed for v1 numbers | ✗ Need to run 5 seeds — cheap |
| Figure | ✗ Need to write `plot_publication_v1.py` |
| LaTeX template | ✗ Pick one (`bioRxiv-style.cls` works) |
| ESMGain citation context | ✗ Read ESMGain paper carefully and confirm the framing contrast |

---

## Single-line summary

**v1 = frozen-probe GOF cross-family signal (narrow, fast, safe). v2 = + positive control + leakage diagnostic. v3 = + generalisation across models and datasets. Each version is a publishable scientific record; later versions strengthen, never replace.**
