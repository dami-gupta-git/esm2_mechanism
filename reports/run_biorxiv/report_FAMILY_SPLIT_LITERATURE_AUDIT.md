# Family-aware evaluation in disease-mechanism prediction

## Scope

This targeted literature audit examines whether computational studies that predict gain-of-function
(GOF), loss-of-function (LOF), or dominant-negative (DN) disease mechanisms prevent related proteins
from appearing in both training and testing data. It covers the papers collected under `papers/` and
recent methods identified through searches of the primary literature. It is not a systematic review.

The audited literature does not support the statement that family or homology control has been
universally ignored. Several audited studies address sequence relatedness, but the controls vary
substantially. Among the recent protein-language-model studies reviewed here, several evaluate random
variants within genes or hold out genes while retaining related proteins. Those designs answer narrower
questions than transfer to an unseen protein family.

## Prediction settings

The appropriate split depends on the intended prediction target. These settings should not be treated as
equivalent.

| Prediction target | What is held out | What the result supports |
|---|---|---|
| Another variant in a gene already represented during training | Variants or residue positions | Within-gene interpolation |
| Variants in an unseen gene whose relatives may be represented | Genes | Transfer to a new gene within the observed family structure |
| Variants in an unseen protein family | Families or sequence clusters | Transfer beyond the represented families or homology groups |
| A gene-level disease mechanism for an unseen protein | Genes, preferably grouped by family or sequence cluster | Proteome-level mechanism prediction with controlled relatedness |

A random variant split can be appropriate for a gene-specific predictor. It cannot support a claim of
generalization to unseen genes or families. A gene split prevents exact gene overlap, but it still permits
paralogues and members of the same protein family to occur on both sides of the split.

## Evaluation practices in the audited literature

| Study | Prediction setting | Reported evaluation | Family or homology control | Assessment for this question |
|---|---|---|---|---|
| [VPatho, 2023](https://doi.org/10.1093/bib/bbac535) | Variant-level pathogenicity followed by GOF versus LOF prediction across genes | Variants were divided randomly into 90% training and 10% testing data | No gene- or family-disjoint split was reported | The reported test performance can use gene-level and protein-level information shared across the split. |
| [LoGoFunc, 2023](https://doi.org/10.1186/s13073-023-01261-9) | Variant-level GOF, LOF, and neutral prediction across genes | The primary test was gene-disjoint. An additional test limited training-to-test protein sequence identity to 40% and retained one representative per homology cluster | Explicit homology-disjoint sensitivity analysis | This is the clearest precedent and prevents a claim that homology-aware mechanism evaluation is new by itself. |
| [Badonyi and Marsh, 2024](https://doi.org/10.1371/journal.pone.0307312) | Gene-level DN, GOF, and LOF propensity | Proteins above 50% sequence identity were removed within each outcome before random training and testing | Partial. Homologues assigned to different outcomes were retained, and clusters were not explicitly held intact across the train-test boundary | Sequence redundancy was reduced, but the evaluation does not directly compare gene-disjoint with family-disjoint performance. |
| [Structural interactomics, 2024 preprint](https://doi.org/10.48550/arXiv.2410.17708) | Gene-level DN, GOF, and haploinsufficiency prediction | Proteins were clustered with MMseqs2 at 20% identity and 20% coverage before an 80/10/10 split | The text does not clearly state whether complete clusters were assigned to one split | Relatedness was considered, but the reported methods are insufficient to classify the test as cluster-disjoint without inspecting the split implementation. |
| [ClearVariant, 2025 preprint](https://doi.org/10.21203/rs.3.rs-6705195/v1) | Variant-level GOF versus LOF prediction across genes | Five-fold cross-validation over the clinical variant dataset | No gene- or family-disjoint evaluation was reported. A gene-bias baseline used each gene's training-label distribution | The main result does not establish transfer to unseen genes or families. |
| [PreMode, 2025](https://doi.org/10.1038/s41467-025-62318-4) | Gene-specific GOF versus LOF prediction | Five random training-testing splits within each evaluated gene | The main benchmark intentionally retains the target gene. Additional experiments transfer from other genes sharing the same domain | This study addresses within-gene prediction and same-domain transfer. Its reported performance should not be interpreted as family-independent generalization. |
| [Badonyi and Marsh, 2025](https://doi.org/10.1038/s41467-025-63234-3) | Structural mechanism score combined with gene-level mechanism priors | Selected evaluations excluded training genes and restricted proteins to below 50% pairwise identity | Homology control was used for a defined test subset | This provides another precedent, although family leakage is not the primary methodological question of the study. |
| [MissION, 2026](https://doi.org/10.1038/s10038-026-01484-9) | Variant-level GOF versus LOF prediction in ion channels | The primary analysis used repeated random variant cross-validation. A secondary analysis held out one gene at a time | Gene holdout was tested, but related ion-channel genes remained available during training | The secondary analysis supports cross-gene transfer within ion channels, not transfer to unrelated protein families. |

## What the literature establishes

Family-aware evaluation is present but is not applied consistently among the audited studies. LoGoFunc
includes an explicit sequence-identity-disjoint test, and other gene-level studies apply sequence
filtering or restricted homology tests. These studies show that sequence relatedness is a recognised
evaluation issue.

Several current variant-level methods use evaluation designs that retain gene or family information.
VPatho and ClearVariant use variant-level splits across a multi-gene dataset. PreMode uses random
splits within each target gene because its stated objective is gene-specific prediction. MissION adds a
gene-holdout analysis, but its test genes remain within the ion-channel scope represented during
training.

The resulting performance estimates refer to different tasks. High performance from a within-gene
split is compatible with poor performance on an unseen family. The literature often describes both as
mechanism prediction without consistently separating those generalization targets.

## Relationship to `run_biorxiv`

The current study tests a distinct and clearly defined target: whether frozen ESM-2 features support
GOF, DN, and LOF classification when the test variants belong to Pfam families absent from training.
Every variant inherits a curated gene-level mechanism label, so exact gene holdout is necessary but not
sufficient. Family holdout tests whether performance transfers beyond the family structure associated
with those labels.

The current results contribute more than the use of a family split:

1. Gene and Pfam-family splits are compared on the same dataset, features, probes, and seeds.
2. The paired gene-to-family difference is evaluated with families as the uncertainty unit.
3. Family identity is measured directly in wildtype, mutant, and mutation-delta embeddings.
4. The association between protein family and curated mechanism labels is measured separately from
   classifier performance.
5. Pathogenicity and enzyme classification are evaluated with related split logic, allowing task-specific
   transfer to be distinguished from a generally unusable representation or probe. The corresponding
   stability comparison remains pending verification.

In `run_biorxiv`, the linear mutation delta reaches the measured family-split classification floor,
while weak ranking information remains detectable. Wildtype and mutant embeddings perform better but
drop under family holdout, and the wildtype embedding predicts Pfam family directly. By contrast, the
same ESM-2 representation supports family-transferable pathogenicity and enzyme classification. The
stability comparison will be included only after experiment 7.3 and claims 3A–3D are verified. The
completed comparisons make the family result an evaluation finding rather than a single-model
performance observation.

No audited study was found that combines all of the following for this disease-mechanism task: a paired
gene-versus-family comparison, direct measurement of family information in the representation,
measurement of family-label agreement, family-cluster uncertainty, and matched positive controls across
other biological tasks. This is a targeted-audit finding, not a claim of exhaustive priority.

## Claims supported by this audit

The literature supports the following manuscript statement:

> Homology-aware evaluation has been used in individual disease-mechanism predictors, but it is not
> applied consistently among the studies audited here. Several recent protein-language-model studies in
> this sample report within-gene, variant-random, or gene-holdout performance without testing transfer
> across disjoint protein families.

The current results can support this additional statement after manuscript verification:

> In the tested frozen ESM-2 representation, gene-level evaluation overstates mechanism performance
> because absolute embeddings encode protein-family identity and protein families are associated with
> curated mechanism labels.

The following statements are not supported:

- No previous mechanism-prediction study controls for homology.
- All published mechanism predictors suffer from family leakage.
- A family split removes all evolutionary relatedness.
- Performance under within-gene evaluation is invalid. It measures a different prediction target.
- The tested ESM-2 result establishes that protein language models cannot encode disease mechanism.

## Implications for the article

The manuscript should frame family-aware evaluation as inconsistently applied among the audited studies
rather than absent from the field. LoGoFunc should be cited as the principal precedent. PreMode and
ClearVariant should illustrate recent protein-language-model work that targets within-gene or
variant-level prediction, not serve as examples of incorrect analysis.

A compact literature table should appear in the main text or supplement with the prediction unit,
held-out unit, relatedness control, and supported generalization target for each method. This prevents
performance numbers from different tasks from being compared as if they answered the same question.

### Compact manuscript table

The audited methods use different held-out units and therefore support different generalization claims.

| Study | Prediction target | Held-out unit | Homology control |
|---|---|---|---|
| [VPatho, 2023](https://doi.org/10.1093/bib/bbac535) | GOF versus LOF across genes | Individual variants in a random 90/10 split | None reported |
| [LoGoFunc, 2023](https://doi.org/10.1186/s13073-023-01261-9) | GOF, LOF, or neutral across genes | Genes in the primary test; homology clusters in a sensitivity test | Explicit sensitivity test limiting training-to-test sequence identity to 40% |
| [Badonyi and Marsh, 2024](https://doi.org/10.1371/journal.pone.0307312) | Gene-level DN, GOF, and LOF propensity | Genes after sequence filtering | Partial: proteins above 50% identity were removed within each outcome, but clusters were not held intact across the split |
| [Structural interactomics, 2024 preprint](https://doi.org/10.48550/arXiv.2410.17708) | Gene-level DN, GOF, and haploinsufficiency prediction | Genes after MMseqs2 clustering and an 80/10/10 split | Relatedness was considered, but the methods do not establish that complete clusters were kept within splits |
| [ClearVariant, 2025](https://doi.org/10.21203/rs.3.rs-6705195/v1) | GOF versus LOF across genes | Individual variants in five-fold cross-validation | No gene- or family-disjoint test reported |
| [PreMode, 2025](https://doi.org/10.1038/s41467-025-62318-4) | Gene-specific GOF versus LOF | Variants within the same target gene | The main benchmark is within-gene; additional tests transfer between genes sharing a domain |
| [Badonyi and Marsh, 2025](https://doi.org/10.1038/s41467-025-63234-3) | Structural mechanism score combined with gene-level mechanism priors | Training genes were excluded in selected evaluations | A defined test subset was restricted to proteins below 50% pairwise identity |
| [MissION, 2026](https://doi.org/10.1038/s10038-026-01484-9) | GOF versus LOF in ion channels | Variants in the primary test; genes in a secondary test | Gene holdout was tested, but related ion-channel genes remained in training |

The broadest defensible contribution is:

> Gene holdout does not guarantee family-independent generalization. Claims that a protein
> representation predicts disease mechanism in unseen proteins should therefore report a family- or
> sequence-cluster-disjoint evaluation in addition to gene-level performance.

An external re-evaluation of available LoGoFunc, Badonyi, or ClearVariant scores under the same Pfam
partition would strengthen a field-wide performance claim. It is not required for the current ESM-2
case study, but without it the manuscript should not claim that the measured inflation applies
quantitatively to other predictors.

## Sources reviewed

- Stein et al. [Genome-wide prediction of pathogenic gain- and loss-of-function variants from ensemble
  learning of a diverse feature set](https://doi.org/10.1186/s13073-023-01261-9). Genome Medicine,
  2023.
- Ge et al. [VPatho: a deep learning-based two-stage approach for accurate prediction of
  gain-of-function and loss-of-function variants](https://doi.org/10.1093/bib/bbac535). Briefings in
  Bioinformatics, 2023.
- Badonyi and Marsh. [Proteome-scale prediction of molecular mechanisms underlying dominant genetic
  diseases](https://doi.org/10.1371/journal.pone.0307312). PLOS ONE, 2024.
- Saadat and Fellay. [Proteome-wide prediction of mode of inheritance and molecular mechanism
  underlying genetic diseases using structural interactomics](https://doi.org/10.48550/arXiv.2410.17708).
  Preprint, 2024.
- Zhong et al. [PreMode predicts mode-of-action of missense variants by deep graph representation
  learning of protein sequence and structural context](https://doi.org/10.1038/s41467-025-62318-4).
  Nature Communications, 2025.
- Ha et al. [Learning sequence to predict gain- or loss-of-function variants](https://doi.org/10.21203/rs.3.rs-6705195/v1).
  Preprint, 2025.
- Badonyi and Marsh. [Prevalence of loss-of-function, gain-of-function and dominant-negative mechanisms
  across genetic disease phenotypes](https://doi.org/10.1038/s41467-025-63234-3). Nature
  Communications, 2025.
- Gies et al. [Functional effect predictions for ion channel missense variants using a protein language
  model](https://doi.org/10.1038/s10038-026-01484-9). Journal of Human Genetics, 2026.
