# ESM-2 mechanism project context

This project asks whether frozen ESM-2 protein representations encode the disease mechanism of a pathogenic missense variant beyond information about the protein and its family. The primary analysis compares wildtype and mutant embeddings for variants labelled as gain-of-function, dominant-negative, or loss-of-function. Pathogenicity, protein stability, enzyme type, and embedding geometry provide controls and context for interpreting the mechanism result.

## Research question

The central question is whether the change in an ESM-2 representation caused by a missense variant contains information about how that variant produces disease. The main encoder is the frozen ESM-2 650M model, `esm2_t33_650M_UR50D`; only the probes fitted on its representations are trained.

Each mechanism variant has a gene-level label from curated disease-mechanism sources. The label describes whether disease is associated with gain-of-function, dominant-negative, or loss-of-function activity. The analysis therefore tests recovery of a curated gene-level mechanism from a variant representation; it does not assign an experimentally verified mechanism to each individual variant. The main representation is the delta embedding, defined as the mutant embedding minus the wildtype embedding. Mean-pooled embeddings summarize the full sequence, while position-level embeddings describe the mutated residue in its sequence context.

Mechanism is distinct from pathogenicity. A representation can separate pathogenic from benign variants without separating gain-of-function, dominant-negative, and loss-of-function variants. The project therefore treats pathogenicity as a positive control rather than as evidence for mechanism encoding.

Protein homology is a second distinction. A probe evaluated with a gene split can still learn properties shared by related genes in the same protein family. The family split holds out complete Pfam families and asks whether a result transfers to proteins that are less closely related to those used for training.

## Study design

The project uses a set of controlled comparisons to separate mutation information, protein identity, homology, and task-specific signal.

| Comparison | What it tests |
|---|---|
| Delta embedding versus wildtype embedding | Whether the mutation-induced change carries information beyond the identity of the unmutated protein. |
| Gene split versus family split | Whether performance transfers after related proteins are kept together on one side of the train-test boundary. |
| Linear versus nonlinear probes | Whether the target is available through a linear readout or requires a more flexible decision boundary. |
| Mechanism versus pathogenicity | Whether weak mechanism performance reflects the task rather than a general inability to use the embeddings. |
| Mechanism versus stability and enzyme type | Whether ESM-2 supports other biochemical and protein-level tasks under comparable family-aware evaluation. |
| Magnitude, direction, and conservation analyses | What biological property the pathogenicity-associated change in embedding space corresponds to. |

The planned comparisons, outcome metrics, resampling units, and reporting rules are defined in [`ANALYSIS_PLAN.md`](docs/improve/ANALYSIS_PLAN.md). Comparisons named there are primary; any other comparison is labelled exploratory when reported. Pre-registration has been withdrawn as the governing framework, and results are presented as effect estimates with confidence intervals rather than as pass-or-fail verdicts against pre-set thresholds.

## Pipeline

The ordered pipeline builds shared biological inputs first, extracts embeddings second, and runs the experiments last.

| Runbook section | Role |
|---|---|
| 1. Build gene list | Merge curated gene-mechanism sources into the gene-level gain-of-function, dominant-negative, and loss-of-function labels. |
| 2. Fetch variant data | Fetch and filter missense variants, sequences, Pfam families, AlphaMissense scores, and the separate pathogenic-versus-benign control set. |
| 3. Embed variants | Extract wildtype and mutant ESM-2 representations on a GPU and preserve their row alignment with the variant records. |
| 4. Mechanism experiment | Test linear and nonlinear mechanism prediction, family clustering, measured baselines, homology leakage, and a single-source replication. |
| 5. Pathogenicity control | Test whether the same embedding change separates pathogenic from benign ClinVar variants. |
| 6. Geometry analysis | Decompose the pathogenicity signal into magnitude and direction and compare it with sequence conservation and stability-related directions. |
| 7. Stability control | Predict experimentally measured Tsuboyama protein-stability changes under domain-aware and family-aware splits. |
| 8. Enzyme control | Classify kinase, protease, oxidoreductase, and non-enzyme proteins from wildtype embeddings. |

The experiment sections share data and statistical utilities. Changes to a shared input, embedding set, split contract, or output schema can therefore affect several downstream sections.

## Data

The project combines curated disease labels, clinical variants, protein sequences, family annotations, and experimental measurements.

| Source | Use in the project |
|---|---|
| Gerasimavicius et al. 2022 | Primary curated gene-level mechanism labels and variants. |
| Gene2Phenotype | Additional high-confidence gene-level mechanism labels. |
| ClinVar | Pathogenic missense variants for the mechanism set and a separate pathogenic-versus-benign control set. |
| UniProt | Reference protein sequences and enzyme annotations. |
| Pfam | Protein-family assignments used for family-aware splitting and homology analyses. |
| AlphaMissense | A comparative variant-level pathogenicity feature. |
| Tsuboyama et al. 2023 Megascale data | Experimental protein-stability changes used as a positive control. |

The `data/` directory is not committed. It contains downloaded source files, fetched records, caches, filtered variant lists, embedding arrays, and row-alignment metadata. Variant lists and embedding artifacts are fingerprinted so that analyses can verify that an array was produced from the records currently being scored.

Missing scientific values remain missing. They are represented as null or NaN with explicit missingness information where needed, and analyses restrict themselves to observed features instead of inventing or imputing values.

## Evaluation

The evaluation is designed around the unit that must generalize.

Multiclass mechanism and enzyme analyses use macro-F1 so that each class contributes to the score. Binary pathogenicity analyses use AUROC. Stability analyses use rank correlation between predicted and measured stability changes. Measured baselines are produced by the run and are read from result files rather than copied into analysis code.

Cross-validation keeps all variants from a gene together. Family-split evaluation also keeps genes from the same Pfam family together. Confidence intervals resample the same unit used by the split, and paired comparisons apply the same resample to both arms where they share an evaluation population.

Multi-seed experiments write each seed as it finishes. Result files retain the information needed to identify the input data, split, seed, and code provenance for the measurement.

## Repository map

The package is organized by pipeline stage and scientific experiment.

| Path | Contents |
|---|---|
| `src/esm2_mech/fetch_data/` | Gene-list construction, variant fetching, sequence retrieval, annotations, and feature building. |
| `src/esm2_mech/embeddings/` | ESM-2 embedding extraction for the mechanism, pathogenicity, and stability datasets. |
| `src/esm2_mech/experiments/mechanism/` | Primary mechanism probes, baselines, homology analyses, and replications. |
| `src/esm2_mech/experiments/pathogenicity/` | Pathogenic-versus-benign positive control. |
| `src/esm2_mech/experiments/geometry/` | Magnitude, direction, conservation, and transfer analyses. |
| `src/esm2_mech/experiments/stability/` | Tsuboyama stability probes and controls. |
| `src/esm2_mech/experiments/proteome_features/` | Enzyme classification and gene-level feature analyses. |
| `src/esm2_mech/utils/` | Shared paths, constants, data validation, splits, probes, metrics, and inference. |
| `docs/improve/` | The analysis plan that defines the run, the revision plan, and the code audit. |
| `biorxiv/` | Ordered runbook, progress records, follow-up scope, manuscript, and supplementary material. |
| `results/<run>/` | Machine-readable outputs for one named run. |
| `reports/<run>/` | Standalone reports and figures derived from that run's result files. |
| `tests/` | Unit and regression tests for shared contracts and experiment behavior. |

All project paths come from `src/esm2_mech/utils/paths.py`. A single run name selects the matching result and report directories. Values used by more than one module belong in `src/esm2_mech/utils/constants.py`.

## Read next

The current documents divide the scientific plan, the repairs it depends on, execution, and deferred work.

1. Read [`ANALYSIS_PLAN.md`](docs/improve/ANALYSIS_PLAN.md) for the questions, cohorts, models, outcome metrics, planned comparisons, and reporting rules.
2. Read [`REVISION_PLAN.md`](docs/improve/REVISION_PLAN.md) and [`audit.md`](docs/improve/audit.md) for the code repairs that must land before the run starts.
3. Read [`RUNBOOK_biorxiv.md`](biorxiv/RUNBOOK_biorxiv.md) for the ordered data and experiment pipeline.
4. Read [`PROGRESS.md`](biorxiv/PROGRESS.md) for live execution state.
5. Read [`reports/run_biorxiv/`](reports/run_biorxiv/) for experiment-level methods, results, interpretation, and provenance.
6. Read [`FOLLOWUP_biorxiv.md`](biorxiv/FOLLOWUP_biorxiv.md) for work outside the current run.

The material under `docs/` outside `docs/improve/` describes the earlier exploratory phase and is retained for historical context only. [`PREREGISTRATION_run_biorxiv.md`](biorxiv/to_retire/PREREGISTRATION_run_biorxiv.md) is superseded by the analysis plan and does not govern any part of the present study.


## Detailed results history

> **Historical and stale.** The text below predates the current `run_biorxiv` analysis and is retained only as a record of earlier project work. Do not use its numbers, verdicts, or conclusions in the bioRxiv manuscript. Manuscript evidence must come from the analysis plan, verified files under `results/run_biorxiv/`, and regenerated reports under `reports/run_biorxiv/`.

The following text is reproduced from [`reports/summaries/report_summary.md`](reports/summaries/report_summary.md). It summarizes results from runs 0, 1, and 6.

### Run 0 — Summary of results 1–26

Summary of `reports/run0/result_1.md` … `result_26.md` and `result_leakage_fraction.md`. All numbers below come
from those reports; each report's own Provenance section traces them to files under `results/`. Where a report has
been superseded (result_6 by run6) that is noted inline.

---

#### The central result

Frozen ESM-2 650M delta embeddings (mutant minus wildtype) predict **whether** a mutation is damaging very well,
and **how** it acts (gain-of-function, dominant-negative, loss-of-function) barely at all. The same pipeline,
embeddings, classifiers and cross-validation produce:

| Task | Best family-split metric |
|---|---|
| ClinVar pathogenic vs benign | AUROC 0.886 ± 0.001 |
| Enzyme type from WT embedding | macro-F1 0.655 ± 0.012 |
| Megascale stability (ΔΔG) | AUROC 0.750 (GBM) |
| Disease mechanism (GOF/DN/LOF) | macro-F1 0.385 ± 0.018 |

The mechanism floor is roughly 0.05 above the always-predict-LOF baseline. Because the positive controls succeed
on the same infrastructure, the mechanism result is a property of the task and the representation, not a broken
pipeline.

---

#### Family leakage: the methodological finding

ESM-2 embeddings cluster strongly by protein family — a gene's five nearest neighbours share its Pfam family 21%
of the time against 0.8% by chance, and a linear probe identifies the family (of 50) with 58.7% accuracy against a
2.2% majority baseline. Separately, **74.8% of disease genes carry their protein family's most common mechanism
label**. A classifier that only recognises the family and predicts the family's modal mechanism therefore scores
well without learning anything about mechanism.

This explains the early results. A wildtype-only probe, which never sees the mutation, reached macro-F1 0.580
under gene-split CV and dropped to 0.389 when whole families were held out.

The **leakage fraction** formalises this:

```
LF = (gene_split_F1 − family_split_F1) / (gene_split_F1 − chance_F1)
```

On Gerasimavicius, LF = **62.8% with a standard deviation of exactly 0.0% across five seeds**. It is invariant
because it reduces to structural properties of the dataset — within-family mechanism agreement, class balance, and
the family partition — none of which depend on the fold assignment. It can in principle be computed from labels
alone, before any model is trained. For comparison, the enzyme-classification task has LF = 13.7%, and
pathogenicity has a gene-to-family delta of 0.002.

Nearly all published mechanism predictors (Badonyi & Marsh 2024, PreMode 2025, Oliveira 2025, ClearVariant 2025,
LoGoFunc) use gene-split CV and do not quantify family leakage.

---

#### Mechanism: what was tried and what it bought

Every attempt to raise the ESM-2 family-split mechanism floor:

| Approach | Family-split macro-F1 | Report |
|---|---|---|
| Linear probe on delta_mean | 0.281 | result_1, result_2 |
| MLP on delta_mean | 0.364 (Gerasimavicius) / 0.352 (merged) | result_7 |
| Contrastive projection, cross-family positives only | 0.397 / 0.387 | result_9 |
| ClinVar variant-pattern scalar features + delta | 0.399 | result_19 |
| Uniform in-silico scan, embedding distance | 0.272 | result_20 |
| Uniform in-silico scan, log-likelihood | 0.261 | result_22 |
| Scan features + proteome features | 0.413 | result_20 |
| ESM-3 1.4B, sequence only | 0.424 | result_26 |

Notes on the individual arcs:

**Nonlinearity helps under gene-split, not under family-split.** The MLP lifts delta_mean from 0.279 to 0.415
under gene-split, but most of that lift is the residual family signal the subtraction did not remove.

**Contrastive training is the only ESM-2-side intervention that adds real cross-family signal.** Training a
projection head where positive pairs must share a mechanism and come from *different* families lifts family-split
F1 by +0.033, and the lift is equal under gene-split (+0.060) and family-split (+0.059) — the diagnostic that
separates signal from leakage.

**Clan-level holdout halves the remaining signal.** Holding out whole Pfam clans drops MLP F1 to 0.299 ± 0.076
against a majority baseline of 0.254 — about half of the family-split signal is clan-level memorisation, the rest
is genuine cross-fold generalisation. Per-clan variance is large: cupins reach 0.536, ion channels 0.190.

**The ClinVar-pattern result was partly circular.** Result_19's spatial-pattern features (how clustered variant
deltas are across a sequence) looked leak-free and lifted GOF AUROC from 0.578 to 0.646. But the positions came
from ClinVar, which concentrates on well-studied hotspots. Replacing them with 100 evenly-spaced positions per
gene collapsed the signal to 0.272, and switching the readout from embedding distance to log-likelihood made no
difference (0.261). The bottleneck is sampling density, not the scoring function: for a gene with three critical
positions in 1000 residues, 100 uniform samples hit one about 26% of the time.

**Scale helps; structure tokens do not.** ESM-3 1.4B lifts family-split F1 from 0.299 to 0.424 — the largest
single gain in the project. Adding AlphaFold2 structure tokens gives 0.417, marginally worse. Function tokens were
not testable in the open API.

---

#### Other modalities beat ESM-2 for mechanism

| Feature set | Per-variant family-split F1 | DN AUROC |
|---|---|---|
| V1 — ESM-2 delta (1280-dim) | 0.382 ± 0.007 | 0.663 |
| V2 — proteome features (37-dim) | 0.462 ± 0.025 | 0.727 |
| V_bad — Badonyi pDN/pGOF/pLOF (3-dim) | 0.484 ± 0.021 | 0.762 |
| V2+bad — proteome + Badonyi | **0.511 ± 0.021** | **0.827 ± 0.015** |
| V_all — all three | 0.481 ± 0.014 | 0.780 |

Three structural probability scores from Badonyi & Marsh 2024 outperform 1280 ESM-2 dimensions. Proteome and
Badonyi features are additive; ESM-2 is not — adding it to Badonyi (0.441) is worse than Badonyi alone (0.484).
ESM-2 is the dispensable modality once either other modality is present.

Two robustness checks hold. Restricting to genes outside Badonyi's own training set does not reduce the lift (DN
AUROC is in fact higher on the held-out genes, 0.814 vs 0.745), so the result is not label leakage. Re-running
under MMseqs2 clustering at 20% sequence identity moves every number by less than 0.03.

The proteome model's power comes mostly from constraint (pLI/LOEUF/mis_z) and ClinGen dosage scores; dropping
either costs about 0.04 F1. PPI degree contributes nothing (ΔF1 = −0.002), which contradicts the earlier
hypothesis that dominant-negative biology would show up in interactome topology.

---

#### Within-family prediction

Cross-family and within-family are different problems.

Within 24 protein families (≥6 genes, ≥2 mechanism classes, leave-one-gene-out CV), **family-residual** proteome
features — each gene's deviation from its family mean — reach macro-F1 0.514, better than raw proteome features
(0.484). Badonyi residuals add nothing (raw and residual give identical 0.449), meaning the structural prior
carries no within-family variation at all: it says "ion channels tend to be GOF" but cannot say which channel.

The aggregate is dominated by homeodomains (PF00046, n=30, F1 = 0.633). Ion channels are the clean within-family
null for gene-level features (F1 = 0.417), although an earlier ESM-2 delta probe on a single ion channel family
did reach GOF-vs-DN AUROC 0.659 — within ion channels the signal is in which mutations a gene carries, not in
gene-level properties. Nineteen of the 24 families have n ≤ 17, so most per-family numbers are unreliable.

---

#### What ESM-2 does encode

**Pathogenicity is a direction, not a magnitude, and the direction is conservation.** Decomposing the delta:
magnitude alone gives family-split AUROC 0.664, unit-normalised direction gives 0.896 — matching the full delta.
That direction is a single, low-dimensional, redundantly encoded axis. The decider: ESM-2's own masked
log-likelihood (`log P(wt) − log P(mut)`) alone reaches 0.891, higher than the full embedding delta (0.835), and
adding the embedding to it *reduces* performance by 0.021. The pathogenicity axis is conservation, and mean-pooling
the embedding is a lossy re-encoding of what the masked-LM head exposes directly.

**Transferability is task- and probe-dependent within one frozen model:**

| Task | Character | Family-transfer AUROC |
|---|---|---|
| Pathogenicity | Linear, family-universal | 0.815 linear / 0.889 GBM |
| Stability | Nonlinear, cross-family manifold | 0.725 linear / 0.761 GBM (0.750 on Pfam-split) |
| Mechanism | No transferable signal at any probe level | 0.520 linear / 0.540 GBM |

Nonlinearity rescues stability and does not rescue mechanism. That asymmetry is the sharpest distinction the
project found between the two tasks.

Projecting the stability direction out of the mechanism features changes family-split F1 by +0.0004 — the two
signals are independent, ruling out the hypothesis that stability noise was masking mechanism.

---

#### External predictors

**AlphaMissense on ClinVar (result_17)** is family-robust: overall AUROC 0.9404, per-family AUROC 0.9477 ± 0.0458
across 182 families, none below 0.70. The family-leakage critique does not apply to pathogenicity prediction.

**AlphaMissense on ProteinGym (result_18)** narrows that claim. Against physical deep-mutational-scanning labels,
the per-assay AUROC distribution is wide and bimodal: 0.721 ± 0.150, 32% of assays below 0.70, 14% below 0.60,
worst case 0.170. Per-stratum standard deviation triples moving from ClinVar to ProteinGym. The failures cluster
on out-of-distribution assays (Tsuboyama mini-protein stability) rather than on classic disease genes. Result_17's
finding should be cited as robustness *within the curation distribution AlphaMissense was trained adjacent to*,
not as a general claim.

**ESM-2 log-likelihood on ProteinGym (result_24)** gives median Spearman ρ = 0.50, replicating the published
ESM-1v baseline and beating AlphaMissense's 0.459, with fewer catastrophic failures (8% vs 14% of assays below
ρ = 0.20). The pre-registered +0.05 median gap was not met (+0.041). Per-assay variance is driven by assay type —
stability and activity assays do best, binding worst (median 0.34) — not by protein family.

**Badonyi's published model (result_16 addendum)** survives family holdout, moving 1–2 AUROC points. But it shows
per-gene training-set fit: on genes outside its training set, its pLOF score is at chance (AUROC 0.472 vs 0.625
on training genes). Result_15's use of Badonyi scores as *features* in a re-fitted, honestly evaluated model is
unaffected; the caveat is about how Badonyi's own published numbers should be cited.

---

#### Clinical utility

Within the 369 ClinGen HI=3 genes (sufficient evidence for haploinsufficiency), distinguishing GOF from LOF gives
AUROC 0.650 ± 0.020 for the full 37-feature model. **Paralog count alone reaches 0.746**, beating the full model
on every seed. GOF frequency scales monotonically with paralog count across tertiles: 0.7% → 4.5% → 9.3%. The
interpretation is dosage buffering — genes with many paralogs should tolerate losing one copy, so when ClinGen
calls them haploinsufficient anyway, the mechanism is more likely activating.

The operating point is not clinically usable: at P_GOF > 0.4 the model flags 25 genes to recover 4 of 17 true GOF
genes (recall 0.235, precision 0.160), and calibration error is 0.148, so probabilities should not be read
quantitatively. Missingness indicators hurt rather than help — dropping them improves GOF AUROC to 0.679 and DN to
0.703.

---

#### Caveats that apply throughout

- **Gene-level labels.** Every variant in a gene inherits one mechanism label, but genes such as SCN1A carry both
  GOF and LOF variants. The delta is a variant-level measurement against a gene-level label.
- **Class imbalance.** LOF is 72–84% of the labelled set depending on the dataset. Macro-F1 with balanced class
  weights is the right metric; accuracy is not.
- **DN is the hardest class throughout.** It is the smallest (894 variants, 60 genes on Gerasimavicius), is
  enriched in ion channels (KCNQ2 alone is 24% of DN variants), and covers biologically distinct mechanisms —
  interface disruption, dimerisation interference, competitive inhibition — under one label.
- **Frozen embeddings only.** No fine-tuning was tested. The null concerns what frozen ESM-2 exposes, not what a
  pLM could learn.
- **Single seed for results 1–10.** Multi-seed replication came later and moved some numbers: Gerasimavicius
  family-split mechanism F1 is 0.299 ± 0.034 across five seeds, not seed 0's 0.364.
- **Small n in several sub-analyses.** Stability rests on 27 proteins, within-family analysis on families of 6–17
  genes, the HI=3 clinical analysis on 17 GOF genes.

---

#### Superseded

`result_6.md` carries a superseded banner. Its AUROC band of 0.74–0.88 across seeds reflects two different variant
sets, not sampling uncertainty. Run 6 rebuilt the experiment over one canonical, fingerprinted set of 37,218
balanced ClinVar variants across 1,929 genes; all five seeds agree at delta_mean MLP family-split AUROC 0.894,
gene-split 0.897. Cite `reports/run6/report_control.md` instead for pathogenicity.

---

### Run 1 — Baseline comparison on the merged dataset

Summary of `reports/run1/INTRO_REPORT.md` and `report_1.md`. Run 1 is a re-run of the run 0 baseline comparison
(run 0 result_2) on a rebuilt pipeline: the merged Gerasimavicius + G2P dataset rather than Gerasimavicius alone,
five seeds rather than one, and PCA to 256 components (98.0% of variance) applied to the embeddings before
probing.

#### Dataset

17,826 variants across 1,935 genes spanning 1,136 Pfam families. Class distribution is GOF 2,682, DN 1,550,
LOF 13,594 — LOF is 76% of the set. Mechanism labels are gene-level, from clinical genetics curation in
Gerasimavicius et al. 2022 and the molecular-mechanism field in Gene2Phenotype.

#### Results (5-seed mean ± std, ESM-2 650M)

| Feature | Gene-split F1 | Family-split F1 | Δ | GOF AUROC (gene / family) |
|---|---|---|---|---|
| wt_only_mean | 0.543 ± 0.025 | 0.442 ± 0.019 | +0.102 | 0.807 / 0.730 |
| mut_only_mean | 0.544 ± 0.023 | 0.443 ± 0.019 | +0.101 | 0.808 / 0.731 |
| wt_concat_mut | 0.548 ± 0.027 | 0.451 ± 0.024 | +0.097 | 0.806 / 0.713 |
| delta_per_residue | 0.315 ± 0.005 | 0.305 ± 0.001 | +0.010 | 0.595 / 0.567 |
| delta_mean | 0.288 ± 0.001 | 0.288 ± 0.002 | −0.000 | 0.608 / 0.559 |
| onehot_aa | 0.288 ± 0.001 | 0.288 ± 0.002 | −0.000 | 0.542 / 0.542 |
| foldx_ddg | 0.279 ± 0.001 | 0.279 ± 0.001 | +0.000 | 0.619 / 0.617 |

AlphaMissense is described as a baseline in the introduction but does not appear in the results table.

#### Findings

**The run 0 pattern replicates with five seeds on the larger dataset.** The wildtype-only probe, which never sees
the mutation, drops from 0.543 to 0.442 when whole families are held out. Mutant-only is statistically identical
to wildtype-only (0.544 vs 0.543), so the signal is in protein identity and the mutation contributes nothing.

**The delta is clean but empty.** Mean-pooled delta scores identically under both schemes (0.288, Δ = −0.000):
subtracting the wildtype removes the family-identity information, and what remains does not separate mechanism
classes. Per-residue delta carries a small family-correlated signal (0.315 → 0.305).

**FoldX ΔΔG is the most family-stable feature.** Zero leakage across schemes, and its GOF AUROC (0.617 under
family-split) is essentially unchanged by the holdout.

**GOF is again the class that survives family-split best**, at AUROC 0.730 for the wildtype-only probe.

#### Comparison to run 0

| | Run 0 (result_2) | Run 1 |
|---|---|---|
| Dataset | Gerasimavicius, 10,231 variants | Merged, 17,826 variants |
| Seeds | 1 | 5 |
| Preprocessing | Raw 1280-dim embeddings | PCA-256 |
| wt_only gene-split F1 | 0.580 | 0.543 ± 0.025 |
| wt_only family-split F1 | 0.389 | 0.442 ± 0.019 |
| wt_only Δ | +0.191 | +0.102 |

The family-split floor rises (0.389 → 0.442) and the gene-split number falls (0.580 → 0.543), so the measured
leakage roughly halves. This matches the explanation given in run 0 result_7 for the same comparison: the more
diverse merged gene set weakens the "kinases are GOF" shortcut, because large families now contain genes of all
three mechanism classes. Read the family-split column, not the delta — the merged dataset inflates less rather
than revealing new family-robust signal.

Expressed as a leakage fraction — (gene-split − family-split) / (gene-split − chance), with chance taken as the
0.288 always-predict-LOF floor observed here — run 1's wildtype-only probe sits near 40%, against the 62.8% run 0
reported for the MLP delta probe on Gerasimavicius. The two are not directly comparable: different dataset,
different feature, different probe. Computing the leakage fraction for the merged dataset directly is listed as
open work in `reports/run0/result_leakage_fraction.md`.

---

### Run 6 — The consolidated study

Summary of `reports/run6/`: the synthesis document `ESM2_REPORT.md` and its nine companion
reports (`report_classifier`, `report_control`, `report_protein_family`, `report_within_family`,
`report_contrastive`, `report_single_source`, `report_stability`, `report_geometry`,
`report_esm3_mechanism`, `report_esm3_mechanism_geras`, `report_leakage_fraction`), plus
`INTRO_REPORT.md` and `STATS_PLAN.md`.

Run 6 is the version of the project intended for a preprint. It re-runs the run 0 experiments on a
single canonical dataset with five seeds throughout, a measured rather than assumed chance floor,
pre-registered decision gates, and explicit statistical-limitations sections. Several run 0
conclusions change materially.

#### Dataset and conventions

17,826 merged variants across 1,935 genes and 1,134 Pfam families, of which 1,902 genes carry a
Pfam annotation and 833 families are singletons. Classes are LOF 76%, GOF 15%, DN 9%. Embeddings
are ESM-2 650M mean-pooled, reduced to 256 principal components (98.0% of variance) before
probing.

The chance floor is **0.288 macro-F1**, measured from a majority-class dummy classifier under the
same five-seed cross-validation rather than assumed to be one third. This matters: several run 0
results that read as weak signal sit exactly on this measured floor.

#### Mechanism classification

| Feature | Gene-split macro-F1 | Family-split macro-F1 |
|---|---|---|
| wt_only_mean | 0.545 | 0.442 |
| mut_only_mean | 0.547 | 0.443 |
| wt_concat_mut | 0.555 | 0.451 |
| delta_per_residue | 0.315 | 0.305 |
| delta_mean | 0.288 | 0.288 |
| onehot_aa | 0.288 | 0.288 |
| foldx_ddg | 0.279 | 0.279 |
| alphamissense | 0.288 | 0.290 |
| naive baseline | 0.288 | 0.288 |

The linear delta sits exactly on the floor under both splits. AlphaMissense also sits on the
floor, so the limitation is not specific to ESM-2. The mutant-only embedding is indistinguishable
from wildtype-only, so the above-floor signal is protein identity and the mutation contributes
nothing.

Nonlinear probes on the delta, five seeds, both splits:

| Model | Gene-split | Family-split |
|---|---|---|
| MLP | 0.399 ± 0.009 | 0.380 ± 0.010 |
| kNN | 0.408 ± 0.008 | 0.354 ± 0.006 |
| GBM | 0.309 ± 0.004 | 0.295 ± 0.001 |
| Random forest | 0.298 ± 0.004 | 0.288 ± 0.002 |

Nonlinearity lifts the delta from the floor to about 0.40 under gene-split. The model that gains
most under gene-split (kNN) loses most under family-split, which is the signature of residual
family structure rather than mechanism: a nearest-neighbour classifier depends on having related
proteins in training. No nonlinear delta score reaches the linear protein-embedding score of 0.44.

*Discrepancy to resolve:* the synthesis document states that family-split was computed for the MLP
only and marks kNN, GBM and random forest as not run, but `report_classifier.md` reports
family-split numbers for all four. The companion report appears to be the current one.

#### Family clustering and the leakage account

Measured directly rather than inferred. A gene's five nearest neighbours share its Pfam family
25.4% of the time against a shuffled-label null of 0.5% — about 50× chance at z = +249. A linear
probe identifies which of 145 families a gene belongs to with 61.2% accuracy against a 4.4%
majority baseline. The mutant embedding matches the wildtype on every metric. The delta drops
family-probe accuracy to exactly the 4.4% baseline, with a small residual visible only in the
purity metric — the most likely source of the nonlinear delta lift above.

**83.3% of genes in multi-gene families carry their family's majority mechanism label**
(leave-one-out), up from the 74.8% reported in run 0. The report notes this is partly inflated by
class imbalance, since families skew LOF.

Tighter clustering does not predict which genes match their family's majority label (r = +0.001,
p = 0.98). The family effect is population-level, not per-gene.

##### Leakage fraction — revised downward

| Feature | Gene-split | Family-split | Leakage fraction |
|---|---|---|---|
| wt_only_mean | 0.545 | 0.442 | 40.1% |
| mut_only_mean | 0.547 | 0.443 | 40.3% |
| wt_concat_mut | 0.556 | 0.451 | 39.4% |
| delta_mean | 0.288 | 0.288 | undefined (at floor) |

**This supersedes run 0's headline 62.8%.** Run 0 computed the fraction for an MLP delta probe on
Gerasimavicius; run 6 computes it per feature on the merged set against a measured floor, and the
three absolute-embedding features agree at about 40%. The delta features are at the floor, so
their leakage fraction is undefined — correctly, since a feature with no signal cannot leak. Run
6 also drops run 0's claim that the fraction is exactly seed-invariant, and notes that the ratio
must be computed from across-seed mean F1 rather than by averaging per-seed ratios, because the
denominator is a small noisy quantity.

#### Positive control: pathogenicity

37,218 balanced ClinVar variants (18,815 pathogenic / 18,403 benign) across 1,929 genes, five
seeds. This is the canonical set that supersedes run 0 result_6.

| Feature | Probe | Gene-split AUROC | Family-split AUROC | Drop |
|---|---|---|---|---|
| delta_mean | MLP | 0.897 ± 0.001 | 0.894 ± 0.001 | 0.003 |
| delta_mean | logreg | 0.862 ± 0.000 | 0.859 ± 0.001 | 0.003 |
| wt_only | MLP | 0.616 ± 0.003 | 0.605 ± 0.002 | 0.011 |
| wt_only | logreg | 0.575 ± 0.003 | 0.555 ± 0.003 | 0.020 |

The same delta that sits on the mechanism floor predicts pathogenicity at 0.897 and loses 0.003
when whole families are held out. The wildtype embedding cannot predict pathogenicity — the mirror
image of the mechanism result, and exactly what the granularity argument predicts. The
pre-registered pass threshold of 0.85 is met, so the mechanism null is interpretable.

#### Second positive control: stability

New in run 6, and a much larger dataset than run 0's: 177,315 single-point missense variants
across 181 natural domains and 77 Pfam families from the Tsuboyama 2023 mega-scale assay. ΔΔG is
measured in a folding assay, so unlike ClinVar it has no curation or evolutionary circularity.

| Probe | Random ρ | Domain ρ | Family ρ | Family AUROC |
|---|---|---|---|---|
| Ridge (linear) | 0.693 | 0.601 | 0.554 | 0.772 |
| MLP | 0.868 | 0.715 | 0.635 | 0.818 |
| XGBoost | 0.767 | 0.676 | 0.631 | 0.817 |

The pre-registered rule fires LEAKY for the linear probe (random-to-family drop of 0.139 against a
0.05 threshold) and HETEROGENEOUS for the per-domain spread (std 0.160, range 0.02 to 0.86). But
two unrelated nonlinear models land within 0.005 of each other on the held-out-family test, well
above Ridge, so the transferable stability signal is real and simply not linearly accessible.

Controls: shuffling labels collapses every probe to zero. A single-scalar delta-norm baseline
reaches only ρ ≈ 0.25 against the full delta's 0.693, so the signal is directional rather than a
matter of magnitude. Nested alpha tuning reproduces the same numbers, so the LEAKY verdict is not
a regularisation artefact. A partial-least-squares sweep shows the family-transferable component
peaks at about 10 components, so it is low-dimensional.

Projecting the stability direction out of the mechanism embeddings changes family-split mechanism
macro-F1 by −0.001. Stability and mechanism are independent in the representation; mechanism does
not fail because stability is drowning it out. The report notes this test is only valid because
the projected residuals are not re-standardised per fold, which would silently reintroduce the
removed direction.

Scope: every domain is 72 residues or fewer, so the claim is about small natural single domains.
55 of the 77 families are singletons, so the cross-family result rests mainly on 22 multi-member
families.

#### Within-family test

28 qualifying families, five seeds, within-family gene-split CV, scored against each family's own
majority baseline rather than the global floor. Where no fold is scorable the cell is left blank
rather than filled with a fabricated zero — this happened for 8 of the 28 families.

**On classification the answer is a clean null.** Exactly one delta cell beats its family baseline
and stays stable across seeds (PF00010, 8 genes — one gene flipping moves it). In PF00520 (ion
channel), the largest and most balanced family at 1,044 variants and all three classes, the delta
scores 0.256 to 0.299 against a 0.253 baseline. With the most data and no shortcut available, the
mutation tells the classifier nothing. The high-looking cells elsewhere are degenerate (PF00071 is
almost all one class) or coin-flips with standard deviations up to 0.45.

**On ranking there is a faint qualification.** Most per-class AUROCs sit at 0.5, and DN and LOF do
so almost everywhere, but the delta separates GOF from the rest at a modestly above-chance level
in most families containing GOF genes — including PF00520, where GOF AUROC reaches 0.66 (linear)
to 0.73 (MLP) despite macro-F1 sitting at baseline. The within-family picture is a null for
classifying mechanism with a weak, mostly-GOF, mostly-linear ranking signal.

#### Contrastive training

Repeating the run 0 experiment with five seeds and a raw k-NN control changes the reading.

| Method | Gene-split | Family-split |
|---|---|---|
| Contrastive k-NN | 0.438 ± 0.013 | 0.395 ± 0.009 |
| Raw k-NN baseline | 0.408 ± 0.008 | 0.354 ± 0.006 |
| MLP delta floor | — | 0.288 |

The lift over the untrained k-NN is +0.041 and is genuinely cross-family — the trained method
loses *less* under family-split (0.043) than the untrained one (0.054), the opposite of what
leakage would produce. But the per-class view bounds what it is: training raises no class's AUROC
over the untrained baseline. LOF moves +0.006 (inside seed noise), GOF falls, and DN falls further
(0.545 vs 0.577). The macro-F1 gain is better class balance, not any mechanism becoming more
separable. Part of the headline is the k-NN evaluation itself, since raw k-NN already beats the
MLP floor.

This is a more conservative reading than run 0, which described the same experiment as recovering
cross-family mechanism signal.

#### Single-source robustness check

New in run 6. Because the merged dataset draws labels from two curation databases and the classes
are split unevenly between them, a reviewer could ask whether the result is about biology or about
which database a variant came from. Dropping Gene2Phenotype and re-running on the 10,138
Gerasimavicius-only variants (942 genes, 660 families, recomputed floor 0.279) changes nothing:
the delta stays at the floor on both splits (0.279 family-split, DN AUROC exactly 0.500), and
wildtype-only still collapses from 0.612 to 0.445. The mechanism null is not an artefact of mixing
sources.

#### Scale and structure: ESM-3

ESM-3 1.4B on the same 17,826 merged variants, sequence tokens alone versus sequence plus
AlphaFold2 structure tokens. The matched ESM-2 floor is the MLP delta family-split score of 0.380,
so the pre-registered pass threshold is 0.430.

| Condition | Gene-split | Family-split |
|---|---|---|
| ESM-2 650M delta_mean | — | 0.380 |
| ESM-3 seq | 0.445 ± 0.023 | 0.438 ± 0.009 |
| ESM-3 seq+struct | 0.448 ± 0.015 | 0.453 ± 0.012 |

Both scale gates pass; the structure gate fails at +0.014 against a 0.030 bar. Scale suffices;
structure tokens add a real but small amount that does not clear the pre-registered threshold.
The lift is not leakage — gene-split and family-split are nearly identical. But the margin is
thin: 0.438 clears 0.430 by 0.008, about one seed of spread, and the report explicitly declines to
call it significant without a paired test.

At 0.45 macro-F1 the task is not solved. DN AUROC is 0.63 to 0.65 and GOF 0.69, so most of the
separability is GOF-versus-rest rather than a three-way resolution.

**The Gerasimavicius-only ESM-3 run carries a data defect and is superseded.** 93 of 10,231
variants were embedded on a wrong wildtype/mutant pair, because phase 2 overwrote the windowed
reference residue without checking it matched the variant's recorded wildtype amino acid — a check
the ESM-2 pipeline applies. The bug is fixed in the script; that run predates the fix. Its
absolute numbers must not be quoted, and it also used a stale ESM-2 floor (0.349 rather than
0.430) so its scale verdicts are invalid. Its within-run structure comparison survives: sequence
and sequence-plus-structure both score 0.421, identical to three decimal places.

Function tokens, ESM-3's third modality, were not implemented in either run. The report notes a
caveat for any future test: function annotations describe the whole protein, so they are identical
for wildtype and mutant and would largely cancel in the delta — the same reason structure tokens
added little.

#### The shape of the pathogenicity signal

| Feature | Pathogenicity AUROC (logreg / MLP) | Mechanism macro-F1 (MLP) |
|---|---|---|
| full delta | 0.859 / 0.893 | 0.415 ± 0.004 |
| magnitude | 0.673 / 0.673 | 0.322 ± 0.011 |
| direction | 0.867 / **0.901** | 0.415 ± 0.006 |

Pathogenicity is a heading, not a distance. The natural guess — that a more damaging mutation
moves the embedding further — does not hold.

That heading is a single axis: one fitted direction recovers the entire linear signal, and
removing it and refitting barely moves the score (0.859 after one removal, 0.845 after five), so
it is one functional degree of freedom redundantly spread across correlated coordinates.
Directions fit on disjoint family-halves have low raw cosine (0.322) yet transfer at 0.848 against
a within-set 0.859 — the low cosine is a red herring caused by that redundancy, and transfer is
the metric that matters.

**The axis is conservation.** ESM-2's own masked log-likelihood at the variant position — four
numbers, or even the single ESM1v masked-marginal — reaches 0.891, above the 1,280-dimensional
embedding delta's 0.859. Adding the embedding to conservation gains +0.002. The axis correlates
+0.74 with the masked-marginal. It is not context-free chemistry: a regression on BLOSUM,
hydropathy, charge and volume explains 7% of it, and those features alone reach only 0.694.

So the mean-pooled embedding delta is, for pathogenicity, a worse and redundant re-encoding of the
model's own likelihood head. This reframes the pathogenicity result as characterisation rather
than a claim that the representation holds anything novel.

Cross-family transfer, by task and probe:

| Task | Probe | Pooled AUROC | Transfer AUROC |
|---|---|---|---|
| pathogenicity | linear | 0.867 | 0.848 |
| pathogenicity | GBM | 0.905 | 0.896 |
| mechanism (GOF vs rest) | linear | 0.799 | 0.625 |
| mechanism | GBM | 0.802 | 0.640 |

Stability transfer was not run in run 6 — the megascale embeddings for that specific probe are not
cached in this run, so the three-way transfer gradient from run 0 result_23 is not reproduced here.

#### Where run 6 lands: the family-robustness gradient

| Property | Best family-split | Family-robust? |
|---|---|---|
| Pathogenicity | AUROC 0.894 (linear) | Yes — linearly robust |
| Stability | AUROC 0.818 (nonlinear) | Partly — nonlinearly recoverable |
| Mechanism | macro-F1 ≈ 0.40 (near floor) | No |

The ordering is biologically sensible: whether a mutation is damaging has common signatures, how
much it destabilises a fold depends on structural context, and how it changes function is the most
context-specific of the three.

#### Statistical limitations, stated by the authors

Every section carries its own limitations subsection, and the shared one is that seed-to-seed
spreads reflect fold reshuffling on a fixed set of genes, not sampling uncertainty — every seed
reuses all the data, so the spread understates the true error. Two analyses are named as
priorities before submission, both CPU-only and runnable on existing result files:

1. **Cluster-bootstrap confidence intervals over genes.** Labels are gene-level, so the effective
   N is about 1,935 genes rather than 17,826 variants, and far smaller for the rare classes.
2. **Label-permutation tests** against the 0.288 floor, for the gene-versus-family gap, the paired
   ESM-3 versus ESM-2 lift, and the conservation-plus-delta increment.

Deferred to journal revision: AUPRC and prevalence-conditional PPV/NPV, FDR control for the
28-family within-family screen, minimum-detectable-effect per family, and calibration curves.

#### What run 6 changes relative to run 0

| Claim | Run 0 | Run 6 |
|---|---|---|
| Chance floor | assumed ≈ 0.333, or always-predict-LOF 0.279–0.288 | measured 0.288 from a dummy classifier |
| Leakage fraction | 62.8%, exactly seed-invariant | ~40% for the absolute-embedding features; undefined for the delta |
| Within-family mechanism agreement | 74.8% | 83.3% (leave-one-out, merged set) |
| Linear delta on mechanism | weak signal (0.281) | exactly at the measured floor |
| Contrastive training | recovers cross-family mechanism signal | improves class balance only; no class becomes more separable |
| Stability control | 27 proteins, S1724 | 177,315 variants, 181 domains, Tsuboyama 2023 |
| ESM-3 vs ESM-2 | +0.125 on Gerasimavicius | +0.058 on the matched merged set, margin declared untested |
| Structure tokens | −0.007, neutral-to-harmful | +0.014, real but below the pre-registered bar |
| Pathogenicity | AUROC 0.886 on 16,576 variants | AUROC 0.894 on 37,218 canonical variants |

Run 0's proteome-feature and Badonyi-prior arc (results 11–16) and its external-predictor arc
(AlphaMissense on ClinVar and ProteinGym, results 17–18 and 24) have no run 6 counterpart. Those
remain run 0 results.

#### Named next steps

The authors identify the gene-level label granularity as the strongest single constraint: every
variant in a gene shares one label while the delta is variant-level, and a kinase has both GOF and
LOF missense variants. Curating even a few hundred variant-level labels from functional assays is
named as more informative than another probe. After that: ESM-3's function tokens, end-to-end
fine-tuning with family-split early stopping, and testing whether mechanism is detectable in a
conservation-residualised space once the conservation axis is projected out.
