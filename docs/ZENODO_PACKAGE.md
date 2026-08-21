# Assembling the Zenodo reproducibility package

The package deposited on Zenodo is assembled by hand after the runbook's verification checklist
passes and the manuscript is frozen. There is no packaging script. The archive currently on disk is
`artifacts/zenodo/esm2_mechanism_reproducibility_v1.tar.gz`, roughly 21 MB, built from the tag
`run_biorxiv-manuscript-freeze-2026-08-20-v2`.

## Stage the tree

Tag the frozen commit first, then stage a directory named `esm2_mechanism_reproducibility` and copy
into it the five trees below. Stage it outside `results/` and `reports/` so the copy cannot pick
itself up.

| Directory in the package | What to copy |
|---|---|
| `data/` | The processed cohorts and aligned inputs: `valid_variants.json`, `pathogenicity_valid_variants_canonical.json`, `clinvar_pathogenicity_variants.json` and its `.params.json`, `megascale_tsuboyama_variants.json`, `gene_list.tsv`, `gene_universe.tsv`, `pfam_families.json`, `megascale_domain_families.json`, `enzyme_labels.tsv`, `gene_proteome_features.tsv`, `proteome_features_aligned.npy`, `proteome_feature_columns.json`, `conservation_pathogenicity.npy` and its `_meta.json`, and `sequences.json`. Add an `embedding_metadata/` subdirectory holding `embedded_variants.json`, `pathogenicity_meta.json`, and `megascale_fingerprint.json` copied out of `data/embeddings/esm2_t33_650M_UR50D/` |
| `results/run_biorxiv/` | Every result file the manuscript, supplementary material, or a verified report cites, together with the execution logs and the environment snapshots |
| `reports/run_biorxiv/` | The five experiment reports and the literature audit, plus `figures/` in all three formats |
| `study_documents/` | `README.md`, `PREREGISTRATION_run_biorxiv.md`, `RUNBOOK_biorxiv.md`, `PROGRESS.md`, `ENV_SNAPSHOT.md`, `DELTA_run6_to_run_biorxiv.md`, `manuscript.md`, and `supplementary.md`, all from `biorxiv/` |
| `source/` | `git archive` of the freeze tag, written as `esm2_mechanism_run_biorxiv_v2.tar.gz` |

## What is left out

The ESM-2 embedding arrays and the upstream download caches are excluded because they occupy about
24 GB and can be regenerated from the frozen source. The row-identity metadata, sequence inputs,
model identifiers, and content fingerprints that are included let a regenerated array be checked
against the analysis inputs.

Withdrawn and manuscript-external sensitivity analyses are also excluded. For the v1 archive that
meant the Badonyi and proteome mechanism results, the wildtype-identity variants under
`wt_identity_short_proteins/` and `wt_identity_window_average/`, the `step2` permutation backups,
and `report_notes.md`.

## Write the two top-level documents

`README.md` states the creator and ORCID, describes each of the five trees, records the freeze tag
and commit, records the clean analysis commits the result files were produced at, states what is
excluded and why, links the code repository, gives the DOI, and states the licences: CC BY 4.0 for
the study data, results, reports, figures, and documentation, MIT for the source snapshot, and the
original terms for third-party derived files.

`ZENODO_METADATA.md` holds the fields typed into the Zenodo deposit form: resource type, title,
creator, description, keywords, licence, access level, reserved DOI, and the related identifier
linking the GitHub repository as supplemented by the dataset.

## Record digests and archive

    cd <staging>/esm2_mechanism_reproducibility
    find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 shasum -a 256 > MANIFEST.sha256
    cd <staging>
    tar czf <repo>/artifacts/zenodo/esm2_mechanism_reproducibility_v1.tar.gz esm2_mechanism_reproducibility

The manifest records the SHA-256 digest of every archived file except itself. The archive has the
package directory as its single top-level entry.

Version 1 is published at https://doi.org/10.5281/zenodo.22037471. Confirm that the public record
contains the expected archive and that the DOI in the manuscript resolves to that record. If the
package is revised, create a new Zenodo version and repeat the digest and archive checks before
publishing it.
