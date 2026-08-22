# Dissecting protein identity, mutation effects, and disease mechanism in ESM-2 embeddings

Dami Gupta ([ORCID: 0009-0009-2510-6104](https://orcid.org/0009-0009-2510-6104))

Faculty of Computing & Data Sciences, Boston University, Boston, Massachusetts, USA

Correspondence: Dami Gupta, dami.gupta@gmail.com

## Abstract

Deep learning models of protein sequence, such as ESM-2, promise representations that capture the functional
consequences of sequence variation. We asked whether those representations predict not just whether a variant is
damaging, but how: by removing function, adding function, or interfering with the normal copy.

We took frozen ESM-2 embeddings of the wildtype sequence, the mutant sequence, and the difference between them (the
"delta"), and trained simple probes to sort missense variants into those three mechanisms. We then varied what was
held out at test time. Holding out genes still leaves close relatives of the test gene in training. Holding out whole
Pfam families does not.

The delta did no better than always guessing the most common mechanism. That was our main test, registered before the
run. Weaker signals did survive: the delta ranked variants slightly better than chance, and more flexible probes beat
the guessing baseline. Wildtype and mutant embeddings beat the baseline too, but their scores fell once whole families
were held out. Those same wildtype embeddings predicted which Pfam family a protein belongs to far above the
majority-class baseline, so much of what they carry is family identity.

Control tasks worked. The delta predicted pathogenicity and folding stability under family holdout, although stability
performance varied across families and domains. The wildtype embedding predicted enzyme type. For pathogenicity,
ESM-2's own masked-marginal conservation score already did the job and the delta added nothing.

Scores from gene-held-out tests can therefore overstate how well a model transfers to protein families it has not
seen. Variant-effect studies should state what they held out, and use family- or homology-disjoint evaluation when
claiming generalization beyond the families seen during probe training.

## Introduction

Protein language models such as ESM-2 are trained on large sequence databases and produce embeddings that can be used to predict variant effects. Whether these embeddings predict any particular downstream property, however, remains an empirical question (Lin et al., 2023). Performance on pathogenicity, for example, does not establish that a representation can distinguish disease mechanisms.

Pathogenicity and disease mechanism are distinct properties. A pathogenic variant may abolish protein function (loss-of-function), create or increase activity (gain-of-function), or interfere with the product of the wildtype allele (dominant-negative). Because these mechanisms can have different clinical and therapeutic consequences, identifying a damaging variant is not the same as classifying its mechanism. In this study, mechanism labels are assigned at the gene level from curated literature rather than direct variant-level assays. Each variant therefore inherits its gene's label, and the analysis cannot represent mechanism differences among variants in the same gene.

Variant-level mechanism prediction could help clinical geneticists and functional-genomics researchers prioritize follow-up assays and distinguish variants that may inform different therapeutic strategies. This is particularly relevant when variants in the same gene act through different mechanisms.

The unit held out during evaluation determines the generalization claim. A variant-random split tests interpolation within genes already represented during training. A gene-disjoint split tests transfer to unseen genes, but related members of the same protein family can remain in the training set; performance may therefore partly reflect family-level similarity. A family- or homology-disjoint split tests transfer beyond protein families represented during training.

Recent disease-mechanism predictors handle sequence relatedness differently. LoGoFunc uses a gene-disjoint primary test and a sensitivity analysis that limits training-to-test sequence identity to 40% (Stein et al., 2023). ClearVariant uses variant-level cross-validation without gene or family holdout (Ha et al., 2025), whereas PreMode evaluates variants within each target gene and also tests transfer between genes that share a domain (Zhong et al., 2025). These designs address different generalization questions, so their performance estimates should be interpreted according to the unit held out.

This study tests whether the frozen ESM-2 mutation delta supports disease-mechanism classification across held-out protein families. Before the confirmatory run reported here, we preregistered the confirmatory claims and stability-control criteria. The preregistration also specified the primary probes, resampling procedures, thresholds, and decision rules. Analyses outside those prospective specifications are labeled exploratory; rules added after results were inspected are identified as post-result amendments.  

Mechanism, pathogenicity, enzyme type, and folding stability were analyzed within a shared framework using 
frozen ESM-2 embeddings, held-out protein families, and uncertainty estimates that account for related proteins. 
Together, these analyses provide context for the preregistered mechanism result.

![Figure 1](../reports/run_biorxiv/figures/figure1_study_design.png)

Figure 1. Study design and shared evaluation framework. (A) The mechanism cohort combined curated
gene-level labels with missense variants, and each variant inherited its gene's loss-of-function,
gain-of-function, or dominant-negative label. (B) Wildtype and mutant sequences were embedded
separately with frozen ESM-2. Mean-pooled wildtype, mutant, and mutation-delta representations were
retained. (C) Gene holdout excluded each test gene while allowing related family members in
training; Pfam-family holdout excluded all members of each test family. (D) Mechanism,
pathogenicity, enzyme-type, and folding-stability analyses used task-specific representations and
held-out units within the shared framework.

## Results

### 1. The preregistered delta probe matches the majority-class reference
Under family holdout, the preregistered linear probe on the mean-pooled mutation delta had a
five-seed mean macro-F1 of 0.290, equal to the measured majority-class reference of 0.290. In every
seed, the upper 95% family-bootstrap confidence bound remained below the preregistered threshold of
0.340. In seed 0, macro-F1 was 0.290 (95% CI 0.276-0.305).

Wildtype-only and mutant-only embeddings scored above the reference, but both declined under family
holdout. Wildtype performance fell from 0.552 under gene holdout to 0.450 under family holdout.
Mutant performance fell from 0.549 to 0.451. Mutant-only and wildtype-only performance were nearly
identical. Under this probe, using the mutant rather than the wildtype representation provided
little classification benefit (Figure 2A).

A single-source analysis of 10,138 Gerasimavicius variants reproduced the same pattern without the
G2P additions. With references recomputed for this subset, delta embeddings matched the
majority-class reference at reported precision under gene holdout (0.279 versus 0.279). They also
matched the reference under family holdout (0.280 versus 0.280). Wildtype macro-F1 fell from 0.611
under gene holdout to 0.462 under family holdout. Thus, the main result did not depend on including
the G2P-derived cohort (Supplementary Figure S1).

### 2. Weak classification and ranking signals depend on probe choice
Across seeds, the delta's macro one-versus-rest AUROC ranged from 0.532 to 0.578. Under family
holdout, the five-seed AUROC was 0.584 for gain-of-function, 0.557 for loss-of-function, and 0.524
for dominant-negative variants. Ranking was highest for gain-of-function, whereas dominant-negative
ranking was close to the 0.500 no-signal value.

Family-block permutation tests detected ranking signal in four of five seeds (p = 0.029, 0.003,
0.011, 0.003, and 0.054). Under the preregistered three-of-five decision rule, this overturned the
claim that the delta had no detectable family-robust ranking signal. Thus, although the
preregistered linear probe matched the majority-class reference in macro-F1, it retained weak
ranking information.

Exploratory probes recovered additional classification performance. Full-dimensional,
standardized, class-balanced logistic regression declined from macro-F1 0.415 under gene holdout to
0.387 under family holdout. A separate nonlinear MLP declined from 0.395 to 0.375. Both exploratory
probes exceeded the 0.290 majority-class reference under family holdout, but neither matched the
wildtype-only score of 0.450 under the preregistered probe. These exploratory results do not replace
the preregistered result (Figure 2B,C).

![Figure 2](../reports/run_biorxiv/figures/figure2_mechanism_delta.png)

Figure 2. Disease-mechanism information in the mutation delta. (A) Family-split macro-F1 from the
preregistered within-fold PCA and logistic-regression probe across five seeds. Points show
out-of-fold estimates and bars show 95% family-bootstrap confidence intervals. The dashed line is
the majority-class reference; the preregistered threshold of 0.340 lies beyond the displayed axis,
and every upper confidence bound remained below it. (B) Five-seed mean macro-F1 under gene and
Pfam-family holdout for the preregistered probe and two exploratory probe specifications. (C)
Five-seed family-split one-versus-rest AUROC for gain-of-function, loss-of-function, and
dominant-negative variants. The dashed line marks the 0.500 no-signal value.

### 3. Wildtype embeddings predict Pfam family
The family-classification subset contained 755 genes spanning 145 Pfam families. A linear probe
predicted Pfam family from wildtype embeddings with a five-seed mean accuracy of 60.2%. The
majority-class baseline was 4.37%. The same probe applied to mutation deltas matched the baseline at
4.37%.

We also used a classifier-independent nearest-neighbour analysis. For wildtype embeddings, 25.4%
of the five nearest neighbours shared the same Pfam family, compared with 0.52% after shuffling
family labels. For delta embeddings, the corresponding values were 5.2% and 0.52%, respectively.
Wildtype embeddings therefore showed more local family structure than delta embeddings.

The paired wildtype gene-to-family gap was positive in all five seeds, ranging from 0.045 to 0.139.
The 95% family-bootstrap confidence intervals excluded zero in four of five seeds. Relative to the
wildtype score above the majority-class reference under gene holdout, 38.9% was lost under family
holdout. The 95% CI was 23.9%-54.3%. Figure 3 summarizes the family-classification,
nearest-neighbour, split-gap, and leakage-fraction analyses.

![Figure 3](../reports/run_biorxiv/figures/figure3_family_information.png)

Figure 3. Pfam-family information and the gene-to-family performance gap. (A) Accuracy of a linear
probe trained to classify 145 Pfam families from wildtype, mutant, or delta representations. Bars
show five-seed means with standard deviations; the dashed line is the majority-class reference.
(B) Fraction of each representation's five nearest neighbours that shared its Pfam family. Filled
points show observed values with 95% bootstrap confidence intervals; open points show shuffled
family-label references. (C) Paired wildtype gene-minus-family macro-F1 for each seed, with 95%
family-bootstrap confidence intervals. (D) Fraction of the wildtype representation's performance
above the majority-class reference that was lost under family holdout, with its 95%
family-bootstrap confidence interval.

### 4. Pathogenicity is recoverable across families
Under family holdout, the mean-pooled mutation delta reached a five-seed mean AUROC of 0.885 with an
MLP. The seed-0 estimate was 0.886 (95% CI 0.880-0.891), entirely above the preregistered threshold
of 0.85. The five-seed mean was 0.003 lower than the gene-holdout result.

The directional component performed nearly as well as, or better than, the full delta. It reached
AUROC 0.855 with logistic regression and 0.893 with the MLP. The magnitude component was weaker,
with AUROC 0.610 under both probes (Figure 4A,B).

### 5. Pathogenicity information is largely redundant with conservation
In a matched seed-0 logistic-regression comparison, ESM-2's masked-marginal conservation score
reached AUROC 0.888. This exceeded both the delta alone (0.835) and the delta-plus-conservation model
(0.883). Adding the delta reduced AUROC by 0.005 (95% CI -0.008 to -0.001). Thus, in this evaluation,
the pathogenicity information in the delta was largely redundant with the conservation score
(Figure 4C).

![Figure 4](../reports/run_biorxiv/figures/figure4_pathogenicity_conservation.png)

Figure 4. Pathogenicity transfer, delta geometry, and conservation. (A) Five-seed mean
pathogenicity AUROC under gene and Pfam-family holdout for logistic-regression and MLP probes on the
mean-pooled mutation delta. (B) Five-seed family-split AUROC from the full delta, its directional
component, and its magnitude under exploratory logistic-regression and MLP probes. The dashed lines
in A and B mark the 0.500 no-signal value. (C) Matched seed-0 family-split logistic-regression
comparison of the delta, ESM-2 masked-marginal conservation score, and their combination. Bars show
95% family-bootstrap confidence intervals. The annotation reports the paired AUROC difference
between the combined and conservation-only models.

### 6. Enzyme type is recoverable from the wildtype embedding
Wildtype embeddings were used to classify genes as kinase, protease, oxidoreductase, or non-enzyme.
Under family holdout, the five-seed mean macro-F1 was 0.779. The seed-0 estimate was 0.787 (95% CI
0.732-0.818), entirely above the preregistered threshold of 0.70 and the 0.219 majority-class
reference. A proteome-feature negative control reached macro-F1 0.291 under the same family holdout.

On the shared set of families, enzyme classification exceeded mechanism classification by 0.507
macro-F1 (95% CI 0.447-0.541; Figure 5A,C). Five-seed family-split AUROC was 0.957 for kinase, 0.952
for oxidoreductase, 0.942 for non-enzyme, and 0.905 for protease. Protease was the smallest class and
had the least precise estimate (Figure 5B).

The MLP scored 0.074 macro-F1 lower than logistic regression (95% CI -0.118 to -0.043). Because this
interval extended beyond the preregistered equivalence range of -0.05 to +0.05, we could not
conclude that the two probes performed equivalently (Figure 5C).

![Figure 5](../reports/run_biorxiv/figures/figure5_enzyme_classification.png)

Figure 5. Enzyme-type classification under family holdout. (A) Five-seed mean macro-F1 under gene
and Pfam-family holdout for the ESM-2 wildtype embedding and the proteome-feature negative control.
The dashed line is the majority-class reference. (B) Five-seed family-split one-versus-rest AUROC
from the logistic-regression probe for each enzyme class. The dashed line marks the 0.500 no-signal
value. (C) Paired family-split comparisons. The upper row shows enzyme-minus-mechanism macro-F1 on
the shared family subset, with the comparison minimum gap marked by the dotted line. The lower
row shows MLP-minus-logistic enzyme macro-F1, with the equivalence range shaded. Points show paired
differences and bars show 95% family-bootstrap confidence intervals.

### 7. Stability is recoverable but not uniformly family-robust
The mean-pooled mutation delta predicted experimentally measured folding stability. Under a random
split, the preregistered ridge probe had a seed-0 Spearman correlation of 0.693 (95% CI
0.675-0.709), entirely above the preregistered threshold of 0.50. Under family holdout, the
five-seed mean fell to 0.554.

On the matched seed-0 cohort, performance fell by 0.153 from random to family holdout (95% CI
0.112-0.192). This exceeded the preregistered 0.10 threshold for family dependence. Performance also
varied across domains. The standard deviation of per-domain correlations was 0.160 (95% CI
0.132-0.183). This was above the preregistered maximum of 0.10 (Figure 6A-C).

An exploratory mechanism analysis removed the fitted stability direction. Mechanism macro-F1
changed by -0.0009 (95% CI -0.0025 to +0.0007; Supplementary Figure S2).
Exploratory nonlinear probes retained family-held-out stability signal, with correlations of 0.627
for the MLP and 0.630 for XGBoost. These analyses did not change the preregistered linear-probe
conclusions.

![Figure 6](../reports/run_biorxiv/figures/figure6_folding_stability.png)

Figure 6. Folding-stability transfer across held-out units. (A) Five-seed mean Spearman correlation
under random, PDB-domain, and Pfam-family holdout. Ridge regression was the preregistered probe;
MLP and XGBoost were exploratory. (B) Matched seed-0 random-minus-family difference for the ridge
probe, with its 95% family-bootstrap confidence interval. The dotted line marks the preregistered
0.10 tolerance for family dependence. (C) Distribution of per-domain Spearman correlations. The
vertical line marks the across-domain mean, and the annotation reports the standard deviation and
its 95% bootstrap confidence interval.

## Methods

### 1. Mechanism cohort and labels

Gene-level disease-mechanism labels were assembled from Gerasimavicius et al. (2022) and
Gene2Phenotype (G2P; Thormann et al., 2019). When both sources assigned a label, we used the
Gerasimavicius assignment. We retained G2P records with strong or definitive gene-disease
associations and an unambiguous molecular-mechanism annotation. Genes with unresolved conflicts
among G2P annotations were excluded.

We grouped haploinsufficiency and autosomal-recessive annotations as loss-of-function (LOF). The
remaining classes were gain-of-function (GOF) and dominant-negative (DN). Labels were assigned at
the gene level, so each variant inherited its gene's label. Consequently, the LOF label did not
represent a direct variant-level functional-assay classification.

Missense variants came from the Gerasimavicius supplementary data and from pathogenic ClinVar
records (Landrum et al., 2018) for genes covered only by G2P; likely-pathogenic records were
excluded. Variants required a UniProt identifier (The UniProt Consortium, 2023), an available
reference sequence, and a complete substitution. The recorded wildtype residue also had to match
the reference sequence. The final cohort contained 17,770 variants across 1,931 genes: 13,556 LOF,
2,668 GOF, and 1,546 DN. Pfam annotations covered 1,907 genes in 1,144 families.

### 2. Pathogenicity cohort

The pathogenicity control used a separate ClinVar extraction from the same target-gene universe
(Landrum et al., 2018).
The extraction retained GRCh38 single-nucleotide missense records. Records classified as
conflicting, uncertain, not provided, or other, and records without assertion criteria, were
excluded. Pathogenic and likely-pathogenic records formed the pathogenic class; benign and
likely-benign records formed the benign class.

Protein-level substitutions were deduplicated, and one substitution with conflicting class labels
was excluded. Within each gene, we sampled up to 20 variants from each class using seed 42 and then
balanced the class counts. We removed variants that could not be applied to the UniProt reference
sequence. We then restored per-gene class balance. The final cohort contained 24,384 variants
across 1,802 genes, with 12,192 variants per class. Family splits used 24,176 variants in 1,072 Pfam
clusters; 208 variants without Pfam assignments were excluded only from those splits.

### 3. Enzyme cohort

Enzyme-type labels were derived from UniProt EC-number and keyword annotations (The UniProt
Consortium, 2023). Genes were
classified as kinase, protease, oxidoreductase, or non-enzyme, with priority in that order when
annotations overlapped. Enzymes outside the three named classes and genes without the required
annotation were excluded. The cohort contained 1,451 genes: 130 kinases, 68 proteases, 123
oxidoreductases, and 1,130 non-enzymes. Here, non-enzyme meant no annotated EC activity. The primary
representation was the wildtype embedding; a 33-feature proteome matrix was the prespecified
negative control. The proteome analysis contained 1,424 aligned genes. Family splits included
1,429 genes in 835 Pfam clusters for the embedding and 1,422 genes in 828 clusters for the aligned
proteome features.

### 4. Stability cohort

Folding-stability measurements were obtained from the Tsuboyama et al. (2023) mega-scale assay.
Single substitutions in natural PDB domains with numeric `ddG_ML` values were retained; designed
proteins, mutated backgrounds, insertions, deletions, and multi-mutants were excluded. The target
was the reported folding-stability change, ΔΔG, for 177,315 variants across 181 domains. Evaluation
used random, domain-disjoint, and Pfam-family-disjoint splits. HMMER searches (Eddy, 2011) against
Pfam-A (Mistry et al., 2021) assigned
the best hit passing the curated gathering threshold. Fourteen unassigned domains were excluded
only from the family split, leaving 167 domains in 77 families.

### 5. ESM-2 representations

Wildtype and mutant sequences were embedded separately with frozen
`esm2_t33_650M_UR50D`. We mean-pooled final-layer residue representations into 1,280-dimensional
vectors. We defined the mutation delta as the mutant vector minus the wildtype vector. The delta at
the substituted residue was also retained. Sequences longer than 1,022 residues were windowed
around the variant. The pathogenicity conservation score was `log P(wildtype) - log P(mutant)` at
the masked variant position.

### 6. Evaluation splits

For gene splits, all variants from a gene were assigned to the same fold. For family splits, all
genes with the same assigned Pfam accession were assigned to the same fold. Mechanism,
pathogenicity, and enzyme analyses used the first Pfam cross-reference returned by UniProt. Genes
without a Pfam assignment were excluded only from family-split analyses. Stability used the HMMER
assignments above and also held out complete PDB domains.

Five-fold cross-validation was repeated over five seeds. Metrics were computed within folds and
averaged. Confidence intervals used 1,000 cluster-bootstrap resamples: genes for gene splits, Pfam
families for family splits, and PDB domains for stability random and domain splits.

### 7. Probes

The preregistered mechanism probe used up to 256 principal components followed by logistic
regression, with both steps fitted within each training fold. It did not use feature standardization
or class weighting. Exploratory mechanism probes used full-dimensional, fold-standardized,
class-balanced logistic regression or a two-hidden-layer multilayer perceptron (MLP).

For pathogenicity, we used fold-standardized logistic regression and a one-hidden-layer MLP. For
enzyme classification, we used logistic regression and the two-layer MLP. The preregistered
stability probe was fold-standardized ridge regression with an L2 penalty of 1.0. Other stability
models were exploratory. No classifier estimated calibrated clinical risk. Full hyperparameters
are reported in Supplementary Methods.

### 8. Preregistration and statistics

We wrote and committed the statistical plan before the confirmatory run. The version-controlled
analysis repository retains the plan and its revision history. The plan specified the prospective
claims, stability controls, primary probes, thresholds, resampling procedures, and decision rules.
Analyses outside those specifications were exploratory.

Dated post-result amendments specified protein-level deduplication and seed-0 inference for
pathogenicity, seed-0 inference for the conservation comparison, and the numerical threshold for
the enzyme comparison. These rules were not prospective. We reran the affected analyses under the
documented amended specifications.

Primary metrics were macro-F1 for multiclass classification, one-versus-rest AUROC for class
ranking, binary AUROC for pathogenicity, and Spearman correlation for stability. Always-LOF and
always-non-enzyme classifiers supplied the multiclass references; AUROC and Spearman no-signal
values were 0.500 and 0.000. Mechanism ranking used 1,000 family-block label permutations per seed.
The mechanism classification, ranking, and gene-to-family claims used their prespecified
three-of-five seed rules. Other threshold tests used seed-0 out-of-fold predictions and
cluster-bootstrap intervals for inference; five-seed means were descriptive.

### 9. Use of generative AI

Generative AI tools, including Claude (Anthropic) and Codex (OpenAI), were used to assist with code
development, analysis review, figure preparation, and manuscript drafting and editing. All outputs
were reviewed and verified by the author, who takes full responsibility for the work.

## Discussion

A central finding was that mechanism-classification performance from frozen ESM-2 wildtype and
mutant embeddings dropped when evaluation moved from held-out genes to held-out Pfam families.
Wildtype embeddings also predicted Pfam family well above the majority-class baseline, despite
containing no mutation information. Together, these observations suggested that gene-held-out
performance may partly reflect family-specific associations learned from other members of the same
family, which were absent under family holdout. The wildtype mechanism classifier performed better
under gene holdout in all five seeds, and the family-bootstrap intervals excluded zero in four of
five seeds. However, these analyses did not show that family membership predicted mechanism labels
beyond class imbalance, so they did not establish family recognition as the cause of the gap.

Under family holdout, the preregistered linear probe on the mutation delta matched the
majority-class reference for three-class disease-mechanism classification. This did not establish
that the delta contained no mechanism information. Permutation testing detected ranking signal in
four of five seeds, and exploratory probes recovered modest above-reference classification. These
findings together indicated weak, probe-dependent signal but did not change the preregistered
classification result.

The mutation delta predicted pathogenicity and folding stability, while the wildtype embedding
predicted enzyme type. These controls showed that the shared evaluation framework could recover
biological signals from frozen ESM-2 embeddings under family holdout. On the shared family subset,
enzyme classification exceeded mechanism classification by 0.507 macro-F1, with a paired 95%
bootstrap interval from 0.447 to 0.541. Because the controls used task-specific cohorts,
representations, and probes, they did not establish equal task difficulty or label quality.

The pathogenicity signal from the mutation delta was substantially redundant with ESM-2's
masked-marginal conservation score. Conservation alone outperformed the delta, and adding the delta
slightly reduced discrimination. The delta's direction retained nearly all
pathogenicity performance, whereas its magnitude was much weaker. This comparison did not establish
that conservation is the delta's only content. Zhong and Shen (2022) report a related zero-shot
contrast:
ESM1b reaches AUROC 0.922 for
pathogenicity but 0.541 to 0.653 for GOF-versus-LOF classification across four families. Their
trained RESCVE model improves on the zero-shot score in three of the four tasks, and they argue that
mechanism prediction should be family-specific. Their evaluation therefore tests prediction within
known families rather than transfer to unseen families.

Folding stability remained predictable under family holdout, but both the random-to-family decline
and cross-domain variability exceeded their preregistered limits. Exploratory nonlinear probes
improved absolute performance but did not change these linear-probe conclusions. Mutation-level
information therefore remained usable while transferring unevenly across protein families.

Existing mechanism predictors use different held-out units. Among the studies reviewed, LoGoFunc
is the closest precedent, with a gene-disjoint primary test and a sensitivity analysis limiting
training-to-test sequence identity to 40% (Stein et al., 2023). ClearVariant uses variant-level
cross-validation without gene or family holdout (Ha et al., 2025), whereas PreMode primarily
evaluates variants within each target gene and also tests transfer between genes sharing a domain
(Zhong et al., 2025). These designs address different generalization questions. The present study
directly compares gene and family holdout, measures family information in the representation, and
uses positive controls within the shared evaluation framework. The literature review was targeted
rather than systematic and does not establish priority for this combination.

The main limitation is label granularity. Mechanism labels are curated per gene and applied to
individual variants, so the analysis cannot represent within-gene heterogeneity and the assigned
label may not describe the mechanism of every variant. Badonyi and Marsh (2025) report that 43% of
multi-phenotype dominant genes and 49% of mixed-inheritance genes carry both loss-of-function and
non-loss-of-function mechanisms. The dominant-negative class is also the smallest, reducing the
precision of class-specific estimates.

Four further constraints apply. One Pfam accession per protein does not eliminate remote homology.
The representation analyses use one frozen ESM-2 model size and emphasize mean pooling. The
pathogenicity control shares the mechanism cohort's target-gene universe, making it a matched
comparison rather than an independent replication. Protein-level deduplication and seed-0 inference
for the pathogenicity and conservation analyses were specified after initial results were inspected,
and those analyses were rerun under documented post-result amendments.

In conclusion, this study evaluated which biological properties can be recovered from frozen ESM-2
embeddings when entire Pfam families are held out from probe training. Wildtype and mutant
mechanism-classification performance declined from gene to family holdout. Under the preregistered
linear probe, the mutation delta matched the majority-class reference for three-class disease
mechanism, although permutation tests and exploratory probes identified weak, probe-dependent
signal. Frozen ESM-2 embeddings retained family-held-out signal for pathogenicity, folding
stability, and enzyme type, although stability performance varied across domains. For
pathogenicity, the delta added no discrimination beyond ESM-2's masked-marginal conservation score.

These findings show that the held-out unit changes the interpretation of variant-effect
performance. Gene-disjoint evaluation can retain information shared among related proteins,
whereas family- or homology-disjoint evaluation tests transfer beyond families represented during
probe training. Studies should therefore report the held-out unit, the no-signal reference, and
whether analyses were preregistered or exploratory.

Future work should combine variant-level mechanism labels established by functional assays with
stricter homology partitions. Evaluating additional model sizes, pooling strategies, and learned
representations under those conditions would help determine whether family-held-out mechanism
performance is constrained by label granularity, representation choice, or limited transferable
sequence signal.

## Author contributions

Dami Gupta conceived the study, developed the methodology and software, curated the data,
performed the analyses, prepared the figures, and wrote and revised the manuscript.

## Code availability

Code used for cohort construction, analysis, and figure generation is available at
https://github.com/dami-gupta-git/esm2_mechanism.

## Data availability

The reproducibility package supporting this study is available in Zenodo at
https://doi.org/10.5281/zenodo.22037471. It contains the processed cohorts, statistical outputs,
out-of-fold predictions, reports, figures, preregistration records, environment records, and frozen
source snapshot. Large ESM-2 embedding arrays and upstream download caches are excluded;
row-identity metadata, sequence inputs, model identifiers, and content fingerprints are included.

## Funding

This work received no external funding.

## Competing interests

The author declares no competing interests.

## References

Badonyi M, Marsh JA. Prevalence of loss-of-function, gain-of-function and dominant-negative
mechanisms across genetic disease phenotypes. *Nature Communications*. 2025;16:8392.
https://doi.org/10.1038/s41467-025-63234-3

Eddy SR. Accelerated profile HMM searches. *PLoS Computational Biology*. 2011;7:e1002195.
https://doi.org/10.1371/journal.pcbi.1002195

Gerasimavicius L, Livesey BJ, Marsh JA. Loss-of-function, gain-of-function and dominant-negative
mutations have profoundly different effects on protein structure. *Nature Communications*.
2022;13:3895. https://doi.org/10.1038/s41467-022-31686-6

Ha D, Kim S, Kwon K, Chung W, Han J. Learning sequence to predict gain- or loss-of-function
variants. *Research Square* [preprint]. 2025. https://doi.org/10.21203/rs.3.rs-6705195/v1

Landrum MJ, Lee JM, Benson M, et al. ClinVar: improving access to variant interpretations and
supporting evidence. *Nucleic Acids Research*. 2018;46:D1062-D1067.
https://doi.org/10.1093/nar/gkx1153

Lin Z, Akin H, Rao R, et al. Evolutionary-scale prediction of atomic-level protein structure with a
language model. *Science*. 2023;379:1123-1130. https://doi.org/10.1126/science.ade2574

Mistry J, Chuguransky S, Williams L, et al. Pfam: the protein families database in 2021. *Nucleic
Acids Research*. 2021;49:D412-D419. https://doi.org/10.1093/nar/gkaa913

Stein D, Kars ME, Wu Y, et al. Genome-wide prediction of pathogenic gain- and loss-of-function
variants from ensemble learning of a diverse feature set. *Genome Medicine*. 2023;15:103.
https://doi.org/10.1186/s13073-023-01261-9

Thormann A, Halachev M, McLaren W, et al. Flexible and scalable diagnostic filtering of genomic
variants using G2P with Ensembl VEP. *Nature Communications*. 2019;10:2373.
https://doi.org/10.1038/s41467-019-10016-3

Tsuboyama K, Dauparas J, Chen J, et al. Mega-scale experimental analysis of protein folding
stability in biology and design. *Nature*. 2023;620:434-444.
https://doi.org/10.1038/s41586-023-06328-6

The UniProt Consortium. UniProt: the Universal Protein Knowledgebase in 2023. *Nucleic Acids
Research*. 2023;51:D523-D531. https://doi.org/10.1093/nar/gkac1052

Zhong G, Shen Y. Representation of missense variants for predicting modes of action. *Machine
Learning in Structural Biology workshop, NeurIPS*. 2022.

Zhong G, Zhao Y, Zhuang D, Chung WK, Shen Y. PreMode predicts mode-of-action of missense variants
by deep graph representation learning of protein sequence and structural context. *Nature
Communications*. 2025;16:7189. https://doi.org/10.1038/s41467-025-62318-4
