# Pod file transfers

This document identifies the data files that must be available on a GPU host before the
run_biorxiv pipeline can run there. It also identifies large files that should not be transferred.

Code reaches the pod through `git pull`. Everything under `data/` is gitignored, so a fresh clone
has an empty data directory. Each required input must either be copied to the pod or generated there
by an earlier runbook step.

Sizes below are approximate and are given so a transfer that stalls or truncates is recognisable.

## Set A: GPU embedding steps

The runbook has four GPU embedding or extraction steps: variant embedding (3.3), pathogenicity
embedding (5.3), conservation extraction (6.5), and megascale embedding (0.10). Copy only the
inputs required by the steps that will run on the pod.

| File | Size | Read by |
|---|---|---|
| `data/valid_variants.json` | 3.8 MB | Variant embedding |
| `data/cache/sequences.json` | 1.6 MB | Variant embedding, pathogenicity embedding, and conservation extraction |
| `data/clinvar_pathogenicity_variants.json` | 4.1 MB | Pathogenicity embedding |
| `data/clinvar_pathogenicity_variants.params.json` | 2 KB | Pathogenicity embedding |
| `data/megascale_tsuboyama_variants.json` | 42 MB | Megascale embedding |
| `data/pathogenicity_valid_variants_canonical.json` | 3.9 MB | Conservation extraction |

The last file is produced by runbook step 6.1 from the pathogenicity embeddings. Copy it only if
step 6.1 has already been run locally; otherwise run 6.1 on the pod after the pathogenicity
embeddings exist there.

Megascale embedding reads the wildtype and mutant domain sequences directly from
`megascale_tsuboyama_variants.json`. It does not read `data/cache/sequences.json`.

## Set B: CPU probe steps

These files are needed in addition to the relevant Set A files when the complete analysis sequence
runs on the same pod. The table assumes that the GPU outputs and earlier result files remain on that
pod. If a CPU step runs on a fresh or different host, copy its embedding arrays, sidecars, and any
earlier result files listed as inputs for that step in the runbook.

| File | Size | Read by |
|---|---|---|
| `data/variants.json` | 3.9 MB | Mechanism probe |
| `data/pfam_families.json` | 38 KB | Every family-split step |
| `data/alphamissense_scores_full.json` | 436 KB | Mechanism probe |
| `data/enzyme_labels.tsv` | 324 KB | Enzyme classification |
| `data/gene_universe.tsv` | 65 KB | Row order for the aligned feature matrices |
| `data/proteome_features_aligned.npy` | 252 KB | Enzyme classification |
| `data/proteome_feature_columns.json` | 3 KB | Enzyme classification |
| `data/megascale_domain_families.json` | 4 KB | Stability probes, family split |

The domain-family map is the output of runbook step 7.1, which needs the 1.9 GB Pfam profile
database. Copying the finished map avoids transferring that database to the pod.

## Do not copy when regenerating embeddings

Do not transfer these paths wholesale when the pod will regenerate the required embeddings. For a
CPU-only run on a fresh host, copy the exact embedding arrays and sidecars required by the selected
runbook steps, not the full embedding directory.

| Path | Size | Reason |
|---|---|---|
| `data/embeddings/` | 10 GB | Contains the required arrays together with unrelated large arrays; transfer only the selected arrays and sidecars for a CPU-only run |
| `data/downloads/megascale/` | 12 GB | Raw Tsuboyama tables and the Pfam profile database, already reduced to the two small derived files listed above |
| `data/cache/AlphaMissense_aa_substitutions.tsv.gz` | 1.1 GB | Already reduced to the scores file in Set B |
| `data/embeddings/.../scan_wt.npy`, `scan_mut.npy` | 5.7 GB | Perturbation scan, not part of run_biorxiv |
| `data/downloads/*.xlsx`, `AllG2P.csv` | 245 MB | Read by the fetch steps, which the runbook runs locally |

## Copying back

The embedding steps write into the pod's embedding directory, while conservation extraction writes
under `data/`. Copy these outputs back to the matching local directories when the run finishes:

- The four variant embedding arrays and their `embedded_variants.json` sidecar.
- The two pathogenicity mean-embedding arrays and `pathogenicity_meta.json`.
- The four megascale arrays and `megascale_fingerprint.json`.
- The conservation array and its metadata sidecar, which are written under `data/`, not under the
  embedding directory.

If CPU analyses ran on the pod, also copy `results/run_biorxiv/` and `logs/biorxiv/` back while
preserving their directory structure. Copy `reports/run_biorxiv/` as well if reports or figures were
generated there.

Each embedding set has a sidecar recording the inputs it was computed from. A set copied without
its sidecar cannot be validated against the current variant list and will be treated as a cache
miss on the next run.
