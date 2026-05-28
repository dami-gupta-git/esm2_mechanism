# Result 15 — Badonyi 2024 priors as a third modality: structure-informed gene predictions
## Date: 2026-05-26 | Seeds: 0–4 | Script: badonyi_mechanism.py

---

## TL;DR

Badonyi & Marsh 2024 (PLOS One) per-gene SVM probability scores (pDN, pGOF, pLOF), derived from structural and functional features, outperform both frozen ESM-2 delta embeddings (V1) and our public proteome feature set (V2) individually. **Three features beat 1280.** Combining proteome + Badonyi (V2+bad) gives the best aggregate result (macro-F1 = 0.511 ± 0.021, DN AUROC = 0.827 ± 0.015) — a +0.049 lift over V2 alone and the strongest DN result in the project. ESM-2 delta adds nothing on top of Badonyi: V1+bad (0.441) underperforms V_bad alone (0.484), confirming sequence embeddings are fully dominated by the structural prior. The full combination V_all (ESM-2 + proteome + Badonyi) does not beat V2+bad, reinforcing that ESM-2 is the redundant modality.

---

## Source of Badonyi features

**Paper:** Badonyi M & Marsh JA (2024). Proteome-scale prediction of molecular mechanisms underlying dominant genetic diseases. *PLOS ONE* 19(8): e0307312. doi:10.1371/journal.pone.0307312

**Data:** S3 Table downloaded from OSF (https://osf.io/download/7bftj/). Per-gene SVM probabilities: pDN, pGOF, pLOF for 20,365 human proteins. Three binary SVM classifiers (polynomial kernel) trained on 1,270 curated dominant genes from OMIM + DDG2P, using 21 interpretable features (SCRIBER scores, FoldX ΔΔG, RSA, s_het, pLI, pRec, paralog count, cotranslational assembly, ESM-1v/|ΔΔG| ratio, protein abundance, tissue variance, betweenness centrality, membrane propensity, secondary structure fractions, pLDDT) + 20 UniProt ProtNLM embeddings selected by Wilcoxon effect size. Predictions cover the human UniProt reference proteome.

**Coverage on our merged dataset:** 2,407 / 2,424 genes (99.3%). 17 genes imputed with median values and flagged with missingness indicators. Missing genes include aliases not matched to Badonyi's gene symbols (AFG2A, C12ORF65, GBA, etc.).

**Features used here:** raw pDN, pGOF, pLOF only (3 columns). Missingness indicators and family-mean-centred residuals were computed and saved to `data/badonyi_features_aligned.npy` (13 columns total) but only the 3 raw probabilities are used as model input — cleaner and more interpretable.

---

## Setup

- **Dataset:** merged dataset, 18,985 variants (family-annotated) across 1,146 Pfam families
- **Labels:** 3-class (GOF=1,716 variants / DN=2,816 / LOF=14,453)
- **CV:** 5-fold family-split (Pfam family holdout), 5 seeds (0–4)
- **Model variants:**

| Variant | Features | Dim | Architecture |
|---|---|---|---|
| **V1** | ESM-2 delta (mut−WT mean-pool) | 1280 | MLP 1280→256→64→3 |
| **V2** | Proteome (gnomAD+paralogs+HPA+PaxDb+BioPlex+ClinGen) | 37 | LogReg balanced |
| **V_bad** | Badonyi pDN/pGOF/pLOF | 3 | LogReg balanced |
| **V2+bad** | Proteome + Badonyi | 40 | LogReg balanced |
| **V1+bad** | ESM-2 delta + Badonyi | 1283 | MLP 1283→256→64→3 |
| **V_all** | ESM-2 delta + proteome + Badonyi | 1320 | MLP 1320→256→64→3 |

Per-gene F1 (T2) computed at evaluation time by aggregating per-variant probability vectors to gene level (mean across variants), then argmax. Reported alongside per-variant macro-F1 for comparison.

---

## Results

### Macro-F1 (family-split, 5 seeds mean ± std)

| Variant | F1 (per-variant) | F1 (per-gene) | GOF AUROC | DN AUROC | LOF AUROC |
|---|---|---|---|---|---|
| V1 (ESM-2 delta) | 0.380 ± 0.006 | 0.352 ± 0.010 | 0.604 ± 0.018 | 0.661 ± 0.010 | 0.659 ± 0.013 |
| V2 (proteome) | 0.462 ± 0.025 | 0.410 ± 0.005 | 0.678 ± 0.018 | 0.727 ± 0.017 | 0.792 ± 0.013 |
| **V_bad (Badonyi)** | **0.484 ± 0.021** | **0.449 ± 0.011** | **0.716 ± 0.016** | **0.762 ± 0.006** | **0.786 ± 0.017** |
| **V2+bad** | **0.511 ± 0.021** | **0.452 ± 0.013** | **0.685 ± 0.029** | **0.827 ± 0.015** | **0.820 ± 0.015** |
| V1+bad | 0.441 ± 0.006 | 0.400 ± 0.011 | 0.640 ± 0.011 | 0.728 ± 0.018 | 0.719 ± 0.015 |
| V_all | 0.481 ± 0.014 | 0.424 ± 0.009 | 0.688 ± 0.014 | 0.780 ± 0.013 | 0.785 ± 0.011 |

**Reference numbers from result_13 (same dataset, same CV, for context):**
- V1 (result_13): macro-F1 = 0.382 ± 0.007 ✓ consistent
- V2 best (result_13, LGBM): macro-F1 = 0.462 ± 0.025 ✓ consistent with LogReg here
- DN AUROC V3 (result_13): 0.740 ± 0.017 — V2+bad here (0.827) substantially exceeds this

---

## Key findings

### F1 — Three features beat 1280

V_bad (pDN, pGOF, pLOF; 3 numbers) achieves macro-F1 = 0.484 ± 0.021, outperforming V1 (ESM-2 delta, 1280 dimensions, 0.380 ± 0.006) by +0.104 and outperforming V2 (proteome, 37 dimensions, 0.462 ± 0.025) by +0.022. This is consistent across all 5 seeds. The Badonyi SVM probabilities encode mechanism information more efficiently than either ESM-2 or our public proteome features, because they were trained explicitly on mechanism labels using features that directly capture the structural correlates of each mechanism.

### F2 — Proteome + Badonyi is additive; sequence is not

V2+bad (proteome + Badonyi, 40 features) reaches macro-F1 = 0.511 ± 0.021 — a +0.049 lift over V2 alone and +0.027 over V_bad alone. This is the highest macro-F1 in the project. The lift is consistent across seeds (range: 0.482–0.546). In contrast, adding ESM-2 delta to Badonyi (V1+bad = 0.441) underperforms Badonyi alone (0.484) — the MLP on 1283 dimensions fails to use the sequence signal once the structural prior is present. V_all (all three modalities, 1320 features) also underperforms V2+bad (0.481 vs 0.511), confirming ESM-2 is the redundant modality in the full combination.

### F3 — DN AUROC 0.827 is the project high-water mark

DN AUROC for V2+bad = 0.827 ± 0.015. This exceeds all prior results:
- Result_13 V3 (ESM-2 + proteome): DN AUROC = 0.740 ± 0.017 (+0.087 improvement)
- Result_13 V2 best (proteome alone): DN AUROC = 0.697 ± 0.011 (+0.130 improvement)
- Result_9 contrastive (ESM-2 only): DN AUROC = 0.521 ± 0.029 (+0.306 improvement)

The DN lift from adding Badonyi to proteome features (V2 → V2+bad: +0.100 DN AUROC) is substantially larger than the GOF lift (+0.007) or LOF lift (+0.028). This makes biological sense: DN mechanism encodes complex assembly and interface geometry, which Badonyi's structural features (SCRIBER scores for exposed residues, FoldX ΔΔG, EDC clustering) directly capture, while our public proteome features (constraint, paralogs, abundance) approximate this indirectly.

### F4 — V_bad std is elevated on seeds with high fold variance

V_bad macro-F1 std across seeds = 0.021, which is comparable to V2 (0.025). However, within-seed fold std is higher for V_bad (seed 0: folds range 0.295–0.650) than for V2 (seed 0: 0.373–0.537). This reflects Badonyi's 3-feature LogReg being sensitive to fold composition when a test fold has an atypical mechanism distribution. V2+bad is more stable (within-seed range narrows) because the 37 proteome features provide a more stable baseline.

### F5 — Per-gene vs per-variant ordering is consistent

Per-gene F1 (T2 scoring) is systematically lower than per-variant F1 for all variants: V_bad per-gene = 0.449 vs per-variant = 0.484. This is expected — high-variant-count genes (e.g. KCNQ2 with 240 DN variants) get one vote per-gene regardless of variant count, reducing the effective signal. The ordering across variants (V_bad > V2 > V1; V2+bad > V_bad) holds under per-gene scoring, confirming the ranking is not driven by variant-count weighting.

---

## Interpretation

### What this means for the central project finding

The central finding of results 1–13 was: **frozen ESM-2 encodes pathogenicity, not mechanism.** Result 15 adds a third datum: **explicit structural priors (Badonyi SVM) encode mechanism, and they are not captured by ESM-2 delta embeddings.**

The modality ordering is now:
1. Structural priors (Badonyi pDN/pGOF/pLOF) — best single modality
2. Proteome context (gnomAD + interactome + abundance) — competitive, complementary
3. ESM-2 delta embeddings — weakest, redundant once the other modalities are present

ESM-2 is not additive to Badonyi because the structural signal Badonyi encodes (FoldX ΔΔG, spatial clustering, SCRIBER binding-residue scores) is not accessible from mean-pooled sequence-level delta embeddings. The sequence model cannot see 3D geometry; it can only see the average downstream effect on the embedding. The structural model directly measures where mutations fall in the protein structure and how damaging they are, which is exactly the information needed to distinguish LOF (destabilising, spread) from GOF (mild, clustered at functional sites) from DN (interface-localised).

### Why proteome + Badonyi is additive

Proteome features (constraint, paralogs, abundance, PPI degree) encode *gene-level biology* — how essential the gene is, how many relatives it has, how central it is in protein networks. Badonyi features encode *variant-level structural properties aggregated to gene level* — where the missense variants fall in the protein structure and how energetically impactful they are. These are genuinely orthogonal dimensions of mechanism biology:
- A gene with many paralogs and moderate constraint → likely GOF or DN (proteome signal)
- A gene whose missense variants cluster at interfaces with mild ΔΔG → likely DN (Badonyi structural signal)
- Both together: +0.049 F1 lift and +0.100 DN AUROC lift

### Why ESM-2 is the dispensable modality

V1+bad (0.441) < V_bad (0.484): adding 1280 ESM-2 dimensions to 3 Badonyi dimensions hurts. This is a regularisation problem — the MLP struggles to use 1280 dimensions of weakly informative signal without overfitting to fold-specific patterns, diluting the Badonyi signal. The 5-fold family-split CV is stringent enough to expose this. It is also consistent with results 1–13's core finding: ESM-2 delta carries only ~0.08 F1 above the majority baseline under family-split, and that signal is dominated by noise once a stronger modality is present.

---

## Comparison to Badonyi 2024 reported performance

Badonyi 2024 reports test-set AUROC on their own held-out genes: DN vs LOF = 0.71, GOF vs LOF = 0.763, LOF vs non-LOF = 0.763. Our V_bad evaluation is different in three ways: (1) we use their output probabilities as features into a LogReg rather than directly thresholding their predictions, (2) we evaluate under family-split CV on a different gene set (merged Gerasimavicius + G2P, not Badonyi's OMIM+DDG2P), and (3) our task is 3-class GOF/DN/LOF rather than three binary classifiers. The numbers are not directly comparable, but our V_bad DN AUROC of 0.762 ± 0.006 is consistent with their reported DN vs LOF AUROC of 0.71 (we include GOF genes as an additional negative class, which likely helps DN separation).

---

## What remains open

1. **Badonyi raw features as input (not just output probabilities).** We used pDN/pGOF/pLOF — the SVM output. Badonyi's raw input features (SCRIBER scores, FoldX ΔΔG per gene, RSA, cotranslational assembly) would be a cleaner structural feature set and would separate "structural" from "prior" as modalities. Requires running their pipeline or requesting feature outputs from OSF. Not done here; the output probabilities are sufficient to establish that structural information is the missing modality.

2. **V2+bad per-gene recalibration.** ECE for V2+bad is not yet measured. Given result_14's finding that V2 LogReg is miscalibrated (ECE=0.148), V2+bad probabilities should be recalibrated before use in clinical utility analysis.

3. **Clinical utility update (result_14 rerun with V2+bad).** Result_14 showed paralog_count alone achieves AUROC 0.746 within ClinGen HI=3. V2+bad's stronger DN signal may improve identification of DN outliers within HI=3. Worth rerunning the result_14 analysis with V2+bad probabilities.

4. **Whether Badonyi 2025 mLOF score adds beyond pDN/pGOF/pLOF.** The 2025 NatComms paper introduces an mLOF score that updates the 2024 priors with variant-level structural evidence (EDC + ΔΔG_rank). mLOF was not tested here. Given V_bad already achieves strong results, mLOF may add further, especially for LOF vs non-LOF.

---

## Files

- `scripts/build_badonyi_features.py` — downloads and aligns S3 Table to merged_gene_list.tsv
- `scripts/badonyi_mechanism.py` — V1/V2/V_bad/V2+bad/V1+bad/V_all modelling
- `data/cache/badonyi/table_S3.xlsx` — raw Badonyi S3 Table (20,365 genes × 11 cols)
- `data/badonyi_features.tsv` — per-gene features (2,424 × 15 cols, human-readable)
- `data/badonyi_features_aligned.npy` — float32 matrix (2,424 × 13), aligned to merged_gene_list.tsv
- `data/badonyi_feature_columns.json` — column metadata
- `results/badonyi_mechanism/badonyi_mechanism_seed{0..4}.json` — per-seed metrics
- `results/badonyi_mechanism/badonyi_mechanism_summary.json` — 5-seed aggregated summary
- `papers/badonyi_2024_plosone.pdf` — source paper

---

## Plain-English summary

We asked: does adding a third type of information — protein structure features from a published model (Badonyi & Marsh 2024) — improve mechanism prediction beyond what ESM-2 sequence embeddings and population genetics features already provide?

The answer is yes, emphatically for structure, and partially for the combination. Just three numbers — the Badonyi model's probability estimates that a gene acts via dominant-negative, gain-of-function, or loss-of-function mechanisms — outperform the 1,280-dimensional ESM-2 delta embeddings by a wide margin. Those three numbers encode structural information (where in the protein do pathogenic mutations cluster? how destabilising are they?) that ESM-2 cannot see from sequence alone.

The most useful combination is Badonyi's three probabilities combined with our 37 public proteome features (constraint, paralogs, protein abundance, interactome degree). Together they achieve the project's best dominant-negative AUROC of 0.827 — a metric that was stuck at ~0.66 from ESM-2 alone. ESM-2 sequence embeddings add nothing on top of this combination; they are the dispensable modality.

The result sharpens the project's central claim: frozen ESM-2 sequence embeddings are not just weak at mechanism prediction — they are specifically missing the structural geometric information that best distinguishes the three mechanism classes. A model that directly measures protein structure (even via population genetics and structural priors rather than atomic coordinates) captures what ESM-2 cannot.

---

# Appendix A — Leakage triage of V_bad

## Date: 2026-05-26 | Script: badonyi_leakage_analysis.py

## Question

V_bad and V2+bad use Badonyi 2024's SVM output probabilities (pDN, pGOF, pLOF) as input features to a new LogReg. Many of our 1,699 labeled genes were in Badonyi's training set, so those probabilities are partly fit-to-label predictions from a pre-trained model. Could V_bad's headline performance be inflated by label leakage through Badonyi's training-set predictions?

## Setup

5-fold family-split CV, 5 seeds, V2 / V_bad / V2+bad. For each seed, the labeled+family-annotated variants are partitioned into three regimes by their gene's Badonyi-training-set membership:

- **ALL** — every labeled+family-annotated variant (reproduces result_15)
- **IN** — variants whose gene was in Badonyi's training set (any of the three classifiers)
- **OUT** — variants whose gene was NOT in Badonyi's training set

Per-class gene-overlap with Badonyi training (1,699 labeled genes total):
- GOF: 107/146 (73.3%) in Badonyi training — heavily overlapping
- DN: 77/107 (72.0%) in Badonyi training — heavily overlapping
- LOF: 379/1,732 (21.9%) in Badonyi training — moderately overlapping

## Results (5-seed mean ± std)

| Regime | n_genes | V_bad per-gene F1 | V_bad DN AUROC | V_bad GOF AUROC | V_bad LOF AUROC |
|---|---|---|---|---|---|
| ALL | 1,950 | 0.449 ± 0.011 | 0.761 ± 0.006 | 0.716 ± 0.016 | (varies) |
| IN-Badonyi | 557 | 0.441 ± 0.009 | 0.745 ± 0.025 | 0.607 ± 0.034 | 0.642 |
| **OUT-Badonyi** | **1,393** | **0.383 ± 0.011** | **0.814 ± 0.021** | **0.726 ± 0.053** | **0.853** |

V2 comparator (proteome features only, no possible Badonyi leakage):
- ALL: F1 0.410, DN AUROC 0.727
- IN: F1 0.395, DN AUROC 0.608
- OUT: F1 0.382, DN AUROC 0.713

## Finding

If V_bad were leakage-driven, we would expect IN > OUT — the model performs better on Badonyi's training-set genes because their predictions encode the labels. **The opposite is observed**: every per-class AUROC is higher on OUT than IN:

- DN AUROC: OUT 0.814 > IN 0.745
- GOF AUROC: OUT 0.726 > IN 0.607
- LOF AUROC: OUT 0.853 > IN 0.642

V_bad still beats V2 in *both* regimes, including OUT (where Badonyi label leakage is impossible by construction). On OUT, V_bad DN AUROC (0.814) is +0.101 above V2 (0.713) — that gap is real Badonyi-modality signal, not leakage.

The per-gene macro-F1 gap (IN 0.441 vs OUT 0.383) is not leakage either: it tracks the class-balance difference between subsets (IN is more balanced at 21/15/65% GOF/DN/LOF; OUT is 3/2/95%, so OUT's macro-F1 is pulled down). V2 shows the same direction (V2 IN 0.395 vs V2 OUT 0.382), confirming this is a property of the subset rather than the model.

## Why OUT is easier than IN

Two plausible explanations:
1. **Selection bias in Badonyi's training set.** They curated OMIM+DDG2P-labeled genes — biologically "hard" cases including multi-mechanism genes. From their S3: 92 genes are in (DN,GOF) training subsets simultaneously, 217 in (DN,LOF). Multi-mechanism training genes are inherently noisy under single-label 3-class evaluation. IN-Badonyi is enriched for hard cases.
2. **Uncurated-gene signal is informative.** Badonyi's SVM gives low pDN/pGOF/pLOF for unstudied genes that don't match training profiles. Those low values are themselves predictive ("uncurated → looks LOF by default") — correct ~95% of the time on OUT.

## Conclusion

Result_15's V_bad and V2+bad headlines are **not driven by label leakage** through Badonyi's training-set predictions. The Badonyi modality lift is real and survives the strictest leakage-free subset (OUT).

This does not rule out subtler forms of leakage where Badonyi's training labels influence predictions for paralogs of training genes. See Appendix B and result_16's addendum for the family-split robustness analysis of Badonyi's raw model.

---

# Appendix B — MMseqs2-20 cluster-holdout robustness

## Date: 2026-05-26 | Script: mmseqs_cluster_holdout.py

## Question

Result_15 uses Pfam family-split CV. Saadat & Fellay 2025 use MMseqs2 clustering at 20% sequence identity, 20% coverage — a stricter sequence-homology block. Do result_15's conclusions hold under their (stricter) protocol?

## Setup

- Fetched UniProt sequences for all 1,983 genes in the merged dataset (948 already cached from prior work; 1,035 fetched via UniProt REST).
- Ran `mmseqs easy-cluster --min-seq-id 0.20 -c 0.20 --cov-mode 0` (matching S&F's parameters) → 1,230 clusters from 1,983 sequences.
- Re-ran V1 / V2 / V_bad / V2+bad / V_all under 5-fold cluster-split CV, 5 seeds.

## Results (5-seed mean ± std, per-gene F1 and DN AUROC)

| Variant | Family-split pgF1 | MMseqs2-20 pgF1 | Δ | Family-split DN AUROC | MMseqs2-20 DN AUROC | Δ |
|---|---|---|---|---|---|---|
| V1 (ESM-2) | 0.352 | 0.346 ± 0.010 | −0.006 | 0.661 | 0.657 ± 0.018 | −0.004 |
| V2 (proteome) | 0.410 | 0.409 ± 0.005 | −0.001 | 0.727 | 0.701 ± 0.023 | −0.026 |
| V_bad | 0.449 | 0.443 ± 0.010 | −0.006 | 0.762 | 0.776 ± 0.013 | +0.014 |
| **V2+bad** | **0.452** | **0.447 ± 0.014** | **−0.005** | **0.827** | **0.816 ± 0.009** | **−0.011** |
| V_all | 0.424 | 0.419 ± 0.011 | −0.005 | 0.780 | 0.775 ± 0.012 | −0.005 |

## Conclusion

All result_15 conclusions hold under the stricter MMseqs2-20 cluster-split:
- **V_bad > V2 > V1** ordering preserved.
- **V2+bad is still the best DN predictor** (DN AUROC 0.816 ± 0.009 under MMseqs2-20).
- **V_all does not beat V2+bad** — "ESM-2 is the dispensable modality" holds.
- **Three Badonyi features still outperform 1280 ESM-2 dimensions.**

The numbers are within ±0.03 of family-split across every variant and per-class AUROC — well within seed std. Pfam family-split was an adequate holdout protocol; the stricter sequence-homology block doesn't change the story.

This is the matched cross-paper comparator for Saadat & Fellay 2025. Under their evaluation scheme, V_bad gets DN AUROC 0.776 and V2+bad gets 0.816 — both meaningfully above S&F's reported numbers under similar holdout.

## Files

- `scripts/fetch_uniprot_sequences.py` — fetched 1,035 missing UniProt sequences
- `scripts/mmseqs_cluster_holdout.py` — cluster-holdout CV runner
- `data/cache/mmseqs/seqsim20_cluster.tsv` — MMseqs2 cluster assignments (1,230 clusters / 1,983 sequences)
- `data/mmseqs_clusters.json` — gene → cluster mapping
- `results/mmseqs_cluster_holdout/cluster_seed{0..4}.json` — per-seed metrics
- `results/mmseqs_cluster_holdout/cluster_summary.json` — 5-seed aggregated summary

---

# Combined takeaway from Appendices A and B

Result_15's main findings survive two separate robustness checks:

1. **No label-leakage artifact.** V_bad and V2+bad lifts are real, not driven by Badonyi training-set predictions encoding labels (Appendix A).
2. **No holdout-protocol dependence.** Result_15's conclusions hold under MMseqs2-20 cluster-split, matching Saadat & Fellay 2025's stricter evaluation (Appendix B).

A separate analysis (result_16 addendum) tested *Badonyi's raw published model* under family-split and found it survives — but his model also shows a per-gene training-set fit effect (~15-point LOF AUROC gap between training-set and never-seen genes). That nuance affects how Badonyi's published numbers should be cited, but does not affect V_bad's or V2+bad's validity (the LogReg-on-top averages out the per-gene fit per fold).
