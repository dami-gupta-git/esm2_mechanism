# bioRxiv submission metadata

This sheet contains the metadata for the frozen manuscript. Fields marked "author confirmation
required" must be confirmed by the submitting author in the bioRxiv interface.

## Submission files

| File type | File |
|---|---|
| Main manuscript | `output/pdf/esm2_mechanism_manuscript.pdf` |
| Supplementary material | `output/pdf/esm2_mechanism_supplementary.pdf` |

## Article details

| Field | Entry |
|---|---|
| Title | Dissecting protein identity, mutation effects, and disease mechanism in ESM-2 embeddings |
| bioRxiv article category | New Results |
| Subject area | Bioinformatics |
| License | CC BY 4.0 |
| Language | English |

### Abstract

Protein language models are increasingly used to predict variant effects, including pathogenicity.
Whether frozen representations can distinguish disease mechanisms under family holdout remains
unevenly studied. Here we evaluate frozen ESM-2 wildtype, mutant, and delta (mutant minus wildtype)
representations for classifying missense variants as loss-of-function, gain-of-function, or
dominant-negative, using both gene-held-out and Pfam-family-held-out evaluation. We find that
mechanism classification from the wildtype and mutant embeddings drops from gene holdout to family
holdout. Wildtype embeddings also predict Pfam family well above the majority-class baseline,
consistent with family recognition partly contributing to the gene-to-family drop. Under a
preregistered linear probe, the delta matches the majority-class reference for this classification,
while exploratory nonlinear probes recover weak performance above that reference. The same delta
representation nevertheless predicts pathogenicity and folding stability under family holdout,
though stability performance is family-dependent and uneven across protein domains. The
pathogenicity signal is largely redundant with ESM-2's own conservation score. Wildtype embeddings
also support enzyme-type classification, confirming that the representation carries usable signal
for other biological tasks. These results show that gene-held-out performance can overstate
transfer to unseen protein families. Variant-effect studies should therefore report the held-out
unit and use family- or homology-disjoint evaluation when claiming transfer beyond represented
families.

### Suggested keywords

Protein language models; ESM-2; variant-effect prediction; disease mechanism; protein embeddings;
family-held-out evaluation.

## Author

| Field | Entry |
|---|---|
| Given name | Dami |
| Family name | Gupta |
| ORCID | https://orcid.org/0009-0009-2510-6104 |
| Affiliation | Faculty of Computing & Data Sciences, Boston University |
| Location | Boston, Massachusetts, USA |
| Email | dami.gupta@gmail.com |
| Corresponding author | Yes |

## Declarations

| Field | Entry |
|---|---|
| Competing interests | The author declares no competing interests. |
| Funding | This work received no external funding. |
| Data availability | https://doi.org/10.5281/zenodo.22037471 |
| Code availability | https://github.com/dami-gupta-git/esm2_mechanism |
| Generative AI disclosure | Generative AI tools, including Claude (Anthropic) and Codex (OpenAI), were used to assist with code development, analysis review, figure preparation, and manuscript drafting and editing. All outputs were reviewed and verified by the author, who takes full responsibility for the work. |

## Author confirmations

The submitting author must confirm the following items in the bioRxiv interface:

- The manuscript has not already been accepted for publication.
- Posting the manuscript does not conflict with any journal or institutional policy.
- The author has the right to post all text and figures under the selected license.
- The manuscript and metadata are complete and accurate.
- Any screening questions concerning ethics, dual-use research, or health risks are answered from
  the author's knowledge of the study.

## Basis for the selected fields

bioRxiv currently asks authors to provide the title, abstract, authors, affiliations, subject area,
article category, license, and funding information. The available article categories include New
Results, Confirmatory Results, and Contradictory Results. The available subject areas include
Bioinformatics, Genetics, Genomics, Molecular Biology, and other life-science categories. New
Results and Bioinformatics are the closest match for this manuscript.

Official references:

- https://www.biorxiv.org/about-biorxiv
- https://connect.biorxiv.org/news/2025/09/04/funder_information
