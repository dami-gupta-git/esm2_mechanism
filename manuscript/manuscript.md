# [Working title — finalize after claim verdicts are known]

Neutral option: Family-aware evaluation of disease-mechanism information in frozen ESM-2 embeddings

Conditional option (if dissociation holds): Frozen ESM-2 embeddings recover pathogenicity and protein properties but not family-transferable disease mechanism

## Abstract

[6 sentences, written last — see biorxiv.md §Abstract for the required content of each]

1.
2.
3.
4.
5.
6.

## 1. Introduction

[Written after final Results scope is known. Paragraph plan from biorxiv.md:]

1. Protein language models as sequence representations for variant-effect prediction; representation quality vs. task evidence.
2. Pathogenicity vs. disease mechanism (gain-of-function / dominant-negative / loss-of-function).
3. The label/evaluation problem — gene-level curated labels, variant-level measurement, shared homology.
4. Why a gene split alone is insufficient for the main claim.
5. Study design summary (wildtype/mutant/delta/baseline features; gene and family splits; family clustering; pathogenicity, stability, enzyme controls; pathogenicity geometry).
6. Preregistered contribution statement, no preview of verdict.

## 2. Results

### 2.1 A family-aware benchmark tests mutation-induced disease-mechanism information
[Claims: framework for 2A/2B. Figure 1.]
Source: report_mechanism.md (data description sections), PREREGISTRATION_run_biorxiv.md

### 2.2 Mutation-induced ESM-2 representations are tested for family-transferable mechanism classification
[Claim 2A. Figure 2.]
Source: report_mechanism.md

### 2.3 Protein-family structure accounts for the absolute-embedding split gap
[Claim 2B. Figure 3.]
Source: report_mechanism.md (family clustering / leakage sections)

### 2.4 The same mutation representations are tested on pathogenicity
[Claim 2C. Figure 4A–B.]
Source: report_pathogenicity_control.md

### 2.5 Geometry and conservation characterize the pathogenicity-associated representation
[Claims 2D/2E confirmatory; rest exploratory. Figure 4C–F.]
Source: report_geometry.md

### 2.6 Stability tests transfer against a physical measurement
[Claims 3A–3D. Figure 5.]
**STATUS: PENDING — experiment 7 (megascale stability) still running; 7.3/7.4 not complete, report_stability.md not yet regenerated. Do not draft numbers here until that lands.**

### 2.7 Enzyme classification tests signal in the wildtype representation
[Claims 2F–2H. Figure 6.]
Source: report_enzyme_classification.md

### 2.8 Family transfer differs across biological tasks
[Synthesis only, no new test. Figure 6 summary panel.]
Depends on 2.2–2.7 all being final, including stability.

## 3. Discussion

### 3.1 Main finding
[State mechanism result with exact representation/labels/split/probe/interval/verdict.]

### 3.2 Homology
[Gene-to-family gap, direct family measurements, leakage; protein identity vs. mutation-induced info.]

### 3.3 Positive controls
[What pathogenicity/stability/enzyme performance establish and do not establish.]

### 3.4 Representation geometry
[Conservation/magnitude/direction/transfer/stability findings, scoped by confirmatory vs. exploratory status.]

### 3.5 Implications
[Why family-aware evaluation, measured baselines, cluster-aware uncertainty matter.]

### 3.6 Limitations
[Gene-level labels on variants, mixed-mechanism genes, class imbalance, rare DN data, incomplete Pfam coverage, frozen embeddings, pooling choices, direct vs. curated labels, stability domain scope.]

### 3.7 Next studies
[Variant-level functional labels; family-aware fine-tuning; representations separating conservation from mechanism. Align with FOLLOWUP_biorxiv.md.]

## 4. Methods

1. Study design, preregistration, confirmatory/exploratory scope
2. Gerasimavicius / Gene2Phenotype label construction
3. ClinVar mechanism-variant fetching and filtering
4. Separate pathogenic-vs-benign ClinVar construction
5. UniProt sequences, Pfam assignment, AlphaMissense, auxiliary features
6. Frozen ESM-2 650M extraction (wildtype/mutant inputs, pooling, delta construction, row-alignment validation)
7. Mechanism probes (preprocessing, linear/nonlinear models, seeds, outputs)
8. Gene/family/domain/random splits and treatment of missing family assignment
9. Metrics (macro-F1, one-vs-rest AUROC, pathogenicity AUROC, Spearman) and their no-signal references
10. Statistical inference (cluster-bootstrap intervals, paired differences, rare-class draw rejection, permutation design, seed-combination, gate adjudication)
11. Family-clustering and leakage diagnostics
12. Pathogenicity geometry and conservation axis fitting
13. Tsuboyama stability analyses (variant parsing, domain/Pfam assignment, regression probes, direction removal) — **pending final experiment 7 numbers**
14. Enzyme classification (labels, inputs, paired comparison with mechanism)
15. Reproducibility (versions, machines, commit, dirty-tree state, seeds, fingerprints, output locations)

## 5. Figures

| Figure | Panels | Status |
|---|---|---|
| 1. Study design | Label sources, variant construction, WT/mutant extraction, delta construction, gene vs. family split, resampling unit | Ready to build once Table 2 is frozen |
| 2. Mechanism classification | Linear features under both splits, nonlinear delta probes, per-class discrimination, permutation/single-source summaries | Data available (report_mechanism.md) |
| 3. Homology account | Pfam NN purity, within/between distance, family-probe performance, mechanism-family agreement, paired split gap, leakage fraction | Data available |
| 4. Pathogenicity and its geometry | Pathogenicity control both splits; magnitude vs. direction; direction removal and family-half transfer; conservation alone and conservation+delta | Data available |
| 5. Stability | Random/domain/family-split Ridge and MLP; per-domain spread; stability-direction removal; selected controls | **Pending experiment 7** |
| 6. Enzyme control and synthesis | Enzyme linear/nonlinear both splits; ESM-2 vs. proteome features; comparison with mechanism; cross-task summary | Data available except synthesis panel (needs stability) |

## 6. Tables

### Table 1 — Datasets and evaluation
[Dataset source, label unit, task, sample units, split unit, metric, resampling unit, no-signal reference]

### Table 2 — Preregistered outcomes
[Claim ID, tested quantity, threshold/null, point estimate, CI or p-value, verdict, result-file provenance]
**Cannot be finalized until stability claims 3A–3D are verified.**

## 7. Supplementary material

[Cohort flow tables, exclusions, fingerprints, class counts, Pfam coverage, per-seed/per-fold metrics, per-class intervals, bootstrap discard counts, permutation distributions, single-source results, full family-clustering diagnostics, all nonlinear/geometry probes, stability tree models and controls, enzyme proteome baselines, run-to-run provenance.]

---

## Build status

| Section | Blocked on |
|---|---|
| Everything except 2.6/2.8, Table 2, Figure 5 | Nothing — can be drafted from current verified reports |
| §2.6 Stability, §2.8 synthesis, Table 2, Figure 5, Methods §13 | Experiment 7 (megascale stability, steps 7.3/7.4) + regenerated report_stability.md |
| Verification checklist (PROGRESS.md) | Not started — required before any number is written into this document |
