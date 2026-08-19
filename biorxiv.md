# BioRxiv article plan

This document maps the planned `run_biorxiv` pipeline to a bioRxiv manuscript. The article should follow the scientific argument rather than the order in which commands are executed. It should not state numerical results or verdicts until the rerun is complete, the outputs are verified, and the run reports have been regenerated.

## Evidence boundary

The manuscript should use the current bioRxiv study as its evidence base.

The governing sources are [`biorxiv/PREREGISTRATION_run_biorxiv.md`](biorxiv/PREREGISTRATION_run_biorxiv.md), [`biorxiv/RUNBOOK_biorxiv.md`](biorxiv/RUNBOOK_biorxiv.md), verified files under `results/run_biorxiv/`, and regenerated reports under `reports/run_biorxiv/`. The preregistration defines the claims and decision rules. The runbook defines the experiments and their dependencies. The result files supply the measurements, and the reports supply the experiment-level interpretation and provenance.

Earlier runs and `reports/summaries/report_summary.md` are historical context only. Their numbers, verdicts, and conclusions should not enter the title, abstract, main text, figures, tables, or supplement unless the corresponding analysis is part of the current run and the current result has been verified.

Confirmatory and exploratory analyses must remain distinguishable in the article. Claims 2A-1, 2A-2, and 2B through 2H are confirmatory. The stability controls 3A through 3D have their own preregistered rules. The remaining geometry, tree-model, family-clustering, and baseline analyses are descriptive or exploratory unless the preregistration says otherwise.

## Working title

The title should be finalized after the claim verdicts are known.

A neutral working title is:

**Family-aware evaluation of disease-mechanism information in frozen ESM-2 embeddings**

If the completed rerun supports the planned dissociation, a more specific title can state that finding while retaining the scope of the experiment:

**Frozen ESM-2 embeddings recover pathogenicity and enzyme type but do not support reliable disease-mechanism classification across protein families**

The title should not say that ESM-2 contains no mechanism information. The study evaluates a frozen ESM-2 650M representation, defined features, specific probes, gene-level labels, and family-aware splits.

## Central argument

The manuscript should answer one question: does the change in a frozen ESM-2 representation caused by a pathogenic missense variant identify whether disease acts through gain-of-function, dominant-negative, or loss-of-function biology after related protein families are separated between training and testing?

The argument has five parts. First, the mechanism task is evaluated against a measured chance floor under gene and family splits. Second, protein-family structure is measured directly so that any gene-to-family performance loss has a biological and statistical account. Third, pathogenicity, stability, and enzyme classification test whether the same representations and evaluation machinery recover other signals. Fourth, the geometry analysis asks what the pathogenicity-associated representation contains. Fifth, the cross-task comparison establishes which conclusions are specific to mechanism and which concern the representation more generally.

The positive controls can show that the embedding and probe pipeline recovers other biological signals. They cannot establish that mechanism information is absent from the underlying biology or from every possible protein language model representation.

## Abstract

The abstract should report the complete argument in six sentences after the rerun is verified.

1. The first sentence should state that protein language models predict variant effects, but their ability to distinguish disease mechanisms across unrelated protein families is unresolved.
2. The second sentence should define the test: frozen ESM-2 wildtype, mutant, and mutant-minus-wildtype representations evaluated for gain-of-function, dominant-negative, and loss-of-function classification under gene and Pfam-family splits.
3. The third sentence should report both mechanism findings: classification at the measured floor and weak family-robust ranking signal detected by permutation.
4. The fourth sentence should report the direct family-structure and leakage result.
5. The fifth sentence should report the pathogenicity, conservation, stability, and enzyme controls using only the measurements needed to establish the cross-task comparison.
6. The final sentence should give the constrained interpretation, including the gene-level label scope.

The abstract should not contain exploratory model sweeps, individual family results, historical comparisons, or unverified point estimates.

## Introduction

The introduction should move from protein language models to the specific evaluation problem.

| Paragraph | Content |
|---|---|
| 1 | Introduce protein language models as sequence representations used for variant-effect prediction. Distinguish representation quality from evidence about a particular biological task. |
| 2 | Distinguish pathogenicity from disease mechanism. A model can identify that a mutation is damaging without identifying whether it produces gain-of-function, dominant-negative, or loss-of-function activity. |
| 3 | Explain the label and evaluation problem. Mechanism labels are curated at gene level, variants are measured individually, and related genes share both sequence features and biological annotations. |
| 4 | Explain why a gene split is insufficient for the main claim. Genes from the same Pfam family can appear on both sides of a gene split, allowing family recognition to raise apparent mechanism performance. |
| 5 | State the study design. Compare wildtype, mutant, delta, and baseline features under gene and family splits; measure family clustering; run pathogenicity, stability, and enzyme controls; and characterize the pathogenicity direction. |
| 6 | State the preregistered contribution without previewing a preferred verdict. The study tests whether mutation-induced ESM-2 features transfer across protein families and separates confirmatory claims from exploratory characterization. |

The introduction should cite the Gerasimavicius and Gene2Phenotype mechanism sources, ClinVar, Pfam, ESM-2, Tsuboyama stability data, and prior protein-language-model variant studies. It should also cite mechanism-prediction work that uses gene-level evaluation where relevant to the homology question.

## Results

The Results section should follow the dependency of the scientific claims, not the numbering of the runbook.

| Order | Proposed heading | Planned evidence | Claims | Main visual |
|---|---|---|---|---|
| 1 | A family-aware benchmark tests mutation-induced disease-mechanism information | Gene and variant construction, class composition, ESM-2 features, gene and family splits, measured majority and stratified baselines, and the resampling units. | Framework for 2A-1, 2A-2, and 2B | Figure 1 |
| 2 | Mutation-induced ESM-2 representations are tested for family-transferable mechanism classification | Linear feature comparison, nonlinear delta probes, per-class results, five-seed descriptive estimates, seed-0 intervals, all-seed permutation tests, and the Gerasimavicius-only replication. | 2A-1 and 2A-2 | Figure 2 |
| 3 | Protein-family structure accounts for the absolute-embedding split gap | Wildtype and mutant family clustering, nearest-neighbour purity, within-to-between distance, family-probe performance, mechanism-family agreement, paired gene-to-family gaps, and leakage fractions. | 2B | Figure 3 |
| 4 | The same mutation representations are tested on pathogenicity | Balanced pathogenic-versus-benign ClinVar set, wildtype and delta probes, gene and family splits, family-cluster intervals, variant accounting, and the preregistered pathogenicity gate. | 2C | Figure 4A and 4B |
| 5 | Geometry and conservation characterize the pathogenicity-associated representation | Delta magnitude versus direction, direction removal, family-half alignment and transfer, substitution chemistry, masked-LM conservation, conservation alone, and conservation plus delta. | 2D and 2E; other probes exploratory | Figure 4C through 4F |
| 6 | Stability tests transfer against a physical measurement | Tsuboyama data, random, domain, and family splits, Ridge and MLP results, per-domain spread, stability-direction removal from mechanism features, and the planned controls. | 3A through 3D | Figure 5 |
| 7 | Enzyme classification tests signal in the wildtype representation | Four-class enzyme labels, ESM-2 and proteome-feature probes, gene and family splits, linear and nonlinear comparison, and the paired comparison with the current run's mechanism score on the shared family subset. | 2F through 2H | Figure 6 |
| 8 | Family transfer differs across biological tasks | A synthesis of the verified mechanism, pathogenicity, stability, and enzyme measurements. This section should introduce no new test. | Synthesis only | Figure 6 summary panel |

### Benchmark and data description

The first Results section should establish what is being predicted and what counts as independent evidence.

Report the source and priority rules for the gene-level labels, the number of variants, genes, and Pfam families, class balance, missing Pfam coverage, sequence-validation exclusions, and embedding coverage. Explain that every variant from one gene shares the curated gene-level mechanism label. Define the wildtype mean, mutant mean, concatenated, mean delta, position-level delta, amino-acid, FoldX, and AlphaMissense features that enter the baseline comparison.

Show gene and family splits schematically. State that gene-split intervals resample genes, family-split intervals resample families, and paired comparisons reuse the same cluster draws. Introduce the measured majority-class and stratified baselines before interpreting any macro-F1 result.

### Mechanism classification

The primary mechanism section should lead with the family-split result and then use the gene split to diagnose family dependence.

The main table should include the linear feature comparison under both splits, with macro-F1, its confidence interval, and per-class AUROC. A second panel should show the nonlinear delta models. The prose should distinguish the linear confirmatory test from exploratory nonlinear probes. The permutation evidence must report all five seeds and apply the preregistered combination rule. The single-source analysis should appear as a compact robustness panel rather than a separate narrative branch.

The mechanism conclusion must apply the two rules separately. Claim 2A-1 evaluates whether three-class classification sits at the measured floor. Claim 2A-2 tests whether any family-robust ranking signal is detectable by permutation. A floor-level classification result does not imply that ranking signal is absent.

### Homology and leakage

The family analysis should explain the mechanism split gap using direct measurements rather than treating the drop as self-explanatory.

Show whether wildtype and mutant embeddings cluster by Pfam family, whether the delta reduces that structure, and how often genes share their family's mechanism label. Report the paired gene-to-family difference for the preregistered feature and the derived leakage fraction for features with above-floor gene-split performance. A leakage fraction should not be reported when its denominator is undefined or when its interval fails the result contract.

Claim 2B is about the paired split gap. Family clustering and mechanism-family agreement explain that gap, but they do not replace its paired confidence interval.

### Pathogenicity control

The pathogenicity section should test the same mutation representation on a distinct task while keeping family structure controlled.

Describe the separate ClinVar pull, equal pathogenic and benign counts per gene, the embedding fingerprint, filtering and balancing accounting, and the family-split evaluation. Report wildtype and delta results under the probe models specified by the run. Apply the 2C rule to the family-split interval, not to the gene-split point estimate.

The interpretation should be limited to discrimination on the constructed set. The output is not a calibrated clinical risk estimate, and the control does not prove an absence of mechanism information.

### Geometry and conservation

The geometry section should explain what supports pathogenicity prediction after the control establishes that the task is learnable.

Separate confirmatory and exploratory evidence. The conservation-alone and conservation-plus-delta comparisons are confirmatory under 2D and 2E. Magnitude versus direction, repeated direction removal, alignment between family halves, cross-half transfer, full-delta transfer, and substitution chemistry are exploratory characterizations.

The main text should report transfer performance alongside raw vector alignment and should not interpret either measurement as evidence that the biological signal is a single axis. Chemistry correlations should be described as associations with a held-out axis, not as causal explanations.

### Stability control

The stability section should establish performance on a direct physical measurement and quantify how it changes with stricter splits.

Report the random, domain, and family-split correlations for the preregistered Ridge and MLP probes. Apply 3A through 3D separately. Include the random-to-family difference, per-domain spread, and the effect of projecting the stability direction out of the mechanism features. Keep random forest, gradient boosting, XGBoost, component sweeps, delta norm, nested regularization, and label shuffling clearly marked as exploratory controls.

The scope should remain small natural domains represented in the Tsuboyama data. The article should state the number of domains and Pfam families contributing to each split and how singleton families affect the effective sample.

### Enzyme control

The enzyme section should test information in the wildtype representation rather than another mutation effect.

Compare ESM-2 wildtype embeddings with the proteome-feature baseline under gene and family splits. Apply the 2F threshold to the family-split linear probe, compare enzyme performance with the current run's mechanism score on the shared family subset for 2G, and compare MLP with logistic regression using the preregistered equivalence band for 2H. The mechanism reference must be read from the same current run.

This control addresses task specificity. It should not be used to claim that enzyme type and mechanism have equivalent label structure or difficulty.

## Discussion

The Discussion should interpret the completed claim set in a fixed order.

| Section | Required content |
|---|---|
| Main finding | State the mechanism result using the exact representation, labels, split, probe, interval, and verdict. Avoid a model-wide claim. |
| Homology | Explain what the gene-to-family difference and direct family measurements show about evaluation leakage. Distinguish protein identity from mutation-induced information. |
| Positive controls | Explain what pathogenicity, stability, and enzyme performance establish about the embedding and pipeline. State what they cannot establish. |
| Representation geometry | Interpret the conservation, magnitude, direction, transfer, and stability findings only to the extent supported by their confirmatory or exploratory status. |
| Implications | Explain why family-aware evaluation, measured baselines, and cluster-aware uncertainty matter for protein-language-model studies. |
| Limitations | Cover gene-level labels applied to variants, genes with mixed mechanisms, class imbalance, rare DN data, incomplete Pfam coverage, frozen embeddings, pooling choices, direct versus curated labels, and the scope of the stability domains. |
| Next studies | Prioritize variant-level functional labels, then family-aware fine-tuning and representations designed to separate conservation from mechanism. Keep deferred work aligned with `FOLLOWUP_biorxiv.md`. |

The Discussion should not present a positive control as proof that the mechanism null is biologically real. It supports the narrower statement that weak mechanism performance is not explained by a generally unusable representation or evaluation pipeline.

## Methods

The Methods section should follow the data and analysis dependency chain.

1. Study design, preregistration, confirmatory claims, and exploratory scope should be defined first.
2. Gerasimavicius and Gene2Phenotype label construction should describe source priority, confidence filtering, disagreements, and the gene-level label unit.
3. ClinVar mechanism-variant fetching and filtering should describe pathogenicity filters, sequence validation, and the final mechanism set.
4. The separate pathogenic-versus-benign ClinVar construction should describe per-gene balance, deduplication policy, fingerprints, and variant accounting.
5. UniProt sequence retrieval, Pfam assignment, AlphaMissense retrieval, and auxiliary features should be described together.
6. Frozen ESM-2 650M extraction should define wildtype and mutant inputs, sequence windows, pooling, position features, delta construction, and row-alignment validation.
7. Mechanism probes should define feature preprocessing, linear models, nonlinear models, seeds, and outputs.
8. Gene, family, domain, and random splits should define the held-out unit and the treatment of records without a family assignment.
9. Metrics should define macro-F1, one-versus-rest AUROC, pathogenicity AUROC, Spearman correlation, and their measured no-signal references.
10. Statistical inference should define cluster-bootstrap intervals, paired differences, rare-class draw rejection, permutation design, seed-combination rules, and gate adjudication.
11. Family-clustering and leakage analyses should define each diagnostic and the conditions under which a leakage fraction is estimable.
12. Pathogenicity geometry and conservation should define axis fitting inside training folds and held-out-family scoring.
13. Tsuboyama stability analyses should define variant parsing, domain and Pfam assignments, regression probes, and stability-direction removal.
14. Enzyme classification should define the four labels, ESM-2 and proteome inputs, and its paired comparison with mechanism.
15. Reproducibility should report software versions, machines, the final clean release commit or tag, seeds, result fingerprints, and output locations.

The main Methods should contain enough detail to reproduce the confirmatory analyses. Long feature lists, complete hyperparameter grids, per-file schemas, and operational RunPod instructions belong in supplementary methods or the repository runbook.

## Figures

The main figures should each answer one part of the argument.

| Figure | Panels |
|---|---|
| 1. Study design | Label sources and variant construction; wildtype and mutant embedding extraction; delta construction; gene versus family split; resampling unit. |
| 2. Mechanism classification | Linear features under both splits; nonlinear delta probes; per-class discrimination; permutation and single-source summaries. |
| 3. Homology account | Pfam nearest-neighbour purity; within-to-between distance; family-probe performance; mechanism-family agreement; paired split gap and leakage fraction. |
| 4. Pathogenicity and its geometry | Pathogenicity control under both splits; magnitude versus direction; direction removal and family-half transfer; conservation alone and conservation plus delta. |
| 5. Stability | Random, domain, and family-split Ridge and MLP results; per-domain spread; stability-direction removal; selected controls. |
| 6. Enzyme control and synthesis | Enzyme linear and nonlinear results under both splits; ESM-2 versus proteome features; comparison with mechanism; summary of transfer across tasks. |

Figure captions should state the evaluation unit, metric, interval type, seed basis, and whether each panel is confirmatory or exploratory. The figure should not require the reader to infer whether an error bar represents seed variation or cluster-bootstrap uncertainty.

## Tables

Two main tables should be sufficient if the figures carry the detailed measurements.

| Table | Content |
|---|---|
| 1. Datasets and evaluation | Dataset source, label unit, task, sample units, split unit, metric, resampling unit, and no-signal reference. |
| 2. Preregistered outcomes | Claim identifier, tested quantity, threshold or null, point estimate, confidence interval or p-value, verdict, and result-file provenance. |

Per-seed values, per-class details, full model sweeps, and input-accounting tables belong in the supplement.

## Supplementary material

The supplement should provide the detail needed to audit the main claims without widening the manuscript's central scope.

Include complete cohort flow tables, exclusions, fingerprints, class counts, Pfam coverage, per-seed and per-fold metrics, per-class intervals, bootstrap discard counts, permutation distributions, single-source results, complete family-clustering diagnostics, all nonlinear probes, all geometry probes, stability tree models and controls, and enzyme proteome baselines.

Deferred experiments from `FOLLOWUP_biorxiv.md` should not be added to the supplement merely because code or historical outputs exist. An analysis belongs only if it uses the current data contracts, current evaluation rules, verified outputs, and a defined role in the manuscript.

## Claims and wording

The article should keep claims at the level measured by the experiment.

| Avoid | Use instead when supported |
|---|---|
| ESM-2 does not encode mechanism. | The tested frozen ESM-2 features did not support reliable family-transferable three-class mechanism classification under the current labels and probes. |
| The positive control proves the mechanism signal is absent. | The positive control shows that the representation and probe recover signal on another task, so weak mechanism performance is not explained by a generally unusable pipeline. |
| The model is at chance. | Name the measured no-signal reference and state the point estimate, interval, and adjudication rule. |
| Family split removes all homology. | Family split holds out the available Pfam assignments and reduces direct family overlap; relatedness can remain above the Pfam-family level. |
| The pathogenicity score is clinically predictive. | The probe discriminates pathogenic from benign variants in the constructed evaluation set and is not a calibrated risk model. |
| A non-significant result shows no effect. | State whether the result is not distinguishable, failed, or underpowered under the preregistered rule. |

Every number in the manuscript should trace to a verified file under `results/run_biorxiv/` and appear in the Provenance section of the corresponding report. Missing values should remain missing and should not be replaced with zero or another numeric placeholder.

## Writing order

The manuscript should be written after the rerun in an order that keeps claims tied to evidence.

1. Complete the verification checklist and freeze the verified result set.
2. Build Table 2 directly from the preregistration and verified result files.
3. Generate the main figures and confirm that every plotted number matches its source file.
4. Write the Results from the figures, claim table, and regenerated run reports.
5. Write the Methods from the runbook and the final code contracts.
6. Write the Discussion after all confirmatory verdicts are fixed.
7. Write the Introduction after the final scope of the Results is known.
8. Write the title and abstract last.

This order prevents historical expectations or isolated point estimates from determining the article's claims.
