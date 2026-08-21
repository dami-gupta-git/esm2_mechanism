# esm2_mechanism

Does ESM-2 encode disease mechanism?

This project tests whether frozen ESM-2 650M representations predict loss-of-function,
gain-of-function, or dominant-negative disease mechanism under gene and Pfam-family holdout. The
preregistered linear probe on the mutation delta matched the majority-class reference under family
holdout, while ranking tests and exploratory probes retained weak signal. The shared framework also
recovered pathogenicity, folding stability, and enzyme type, showing that the mechanism result was
specific to the target and probe rather than a general failure of the representations.

`docs/README.md` indexes the earlier exploratory phase and is stale; `biorxiv/README.md` governs.

---

## Directory layout

The code is an installed package under `src/esm2_mech/`, and every experiment is run as a module
rather than as a loose script.

```
src/esm2_mech/
  fetch_data/      — build the gene list, fetch variants, sequences, Pfam annotations,
                     AlphaMissense scores, and the proteome and Badonyi feature matrices
  embeddings/      — ESM-2 embedding extraction (GPU); variants, Megascale, scan
  experiments/
    mechanism/     — the mechanism probes: linear baselines, MLP, family clustering,
                     naive floor, leakage fraction, single-source replication
    pathogenicity/ — the pathogenic-vs-benign positive control
    geometry/      — magnitude vs direction, conservation axis, transfer contrast
    stability/     — Megascale (Tsuboyama) stability probes and their baselines
    proteome_features/ — gene-level proteome probes and the enzyme-type control
    badonyi/       — Badonyi 2024 structural-prior probes
    alphamissense/ — AlphaMissense and ESM-1v comparisons, ProteinGym
    perturbation/  — in-silico perturbation and log-likelihood scans
    esm3/          — ESM-3 mechanism comparison
    llm_judge/     — language-model baseline that predicts a variant's mechanism class
  figures/         — the manuscript figure generator
  utils/           — paths, constants, splits, metrics, bootstrap, shared probe code

biorxiv/           — the current run: pre-registration, runbook, progress, manuscript
docs/              — how figures, reports, the Zenodo package, the tests and the ESM-3
                     embeddings are produced, plus the statistics-machinery notes, the
                     RunPod reference, and the stale exploratory-phase index
results/<run>/     — JSON outputs, one directory per run
reports/<run>/     — write-ups and figures for a run
scripts/           — the few CLI tools that are not part of the package
tests/
data/              — all inputs and embeddings (gitignored in full)
```

Pipeline paths are defined in `src/esm2_mech/utils/paths.py`. The run directory is keyed off a
single run-name constant so a result file and its report select the same run.

---

## Setup

Python 3.13 is the pinned version; the package itself requires 3.10 or later.

```bash
uv sync
```

Or with a plain virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Embedding extraction needs a CUDA GPU; every probe and bootstrap step runs on CPU.

`docs/TESTING.md` covers running the test suite. Note that `pip install -e .` does not install
pytest, which is in the `dev` dependency group.

---

## RUN

`biorxiv/RUNBOOK_biorxiv.md` has the full ordered command list for the experiments, including the
GPU embedding steps, and `biorxiv/PROGRESS.md` records what has been executed. For RunPod setup and
remote execution, see `docs/connect_runpod.md`.

What happens after the experiments finish is documented separately:

| Document | Covers |
|---|---|
| `docs/FIGURES.md` | Regenerating the eight manuscript figures, and which result files each one reads |
| `docs/REPORTS.md` | The seven reports in `reports/<run>/`, how they are written, and diffing a run against the run6 baseline first |
| `docs/ZENODO_PACKAGE.md` | Assembling the reproducibility package, what it excludes, and how the archive is built |

The ESM-3 scale-and-structure experiment is a run6-era result outside this pipeline and needs a
different ESM package and a licensed model download; see `docs/ESM3_EMBEDDINGS.md`.

---

## Data

The whole of `data/` is gitignored, so nothing here is shipped with the repository. It holds the
built variant and gene lists, the ESM-2 embedding arrays under `data/embeddings/`, downloaded
third-party files under `data/downloads/` and `data/cache/`, and the Megascale dataset under
`data/downloads/megascale/`. The processed Megascale variant file is stored directly under
`data/`. Embeddings are produced on a GPU machine and copied back; the loaders fingerprint the
variant list and refuse to run against embeddings that do not match it.

Several source files cannot be fetched automatically and must be placed in `data/downloads/` before
anything else runs. `PRELOADED.md` lists all of them with their sources; the primary two are
`DiseaseMech_Stability_VEPS.xlsx` and `AllG2P.csv`.

Primary dataset: Gerasimavicius et al. 2022 (Nature Communications 13:3895), supplied as
`DiseaseMech_Stability_VEPS.xlsx`.

**Badonyi 2024 SVM scores** (`data/downloads/table_S3.xlsx`): per-gene mechanism probability scores for 20,365 human proteins from Badonyi & Marsh 2024 (PLOS One, DOI: 10.1371/journal.pone.0307312). Three binary SVM classifiers (DN vs LOF, GOF vs LOF, LOF vs non-LOF) were trained on 1,270 curated genes with known mechanisms (OMIM + DDG2P), using AlphaFold structural features, FoldX ΔΔG, ESM-1v scores, ProtNLM embeddings, and population genetics constraints (s_het, gnomAD).

| Column | Description |
|---|---|
| `gene` | Gene symbol |
| `uniprot_id` | UniProt accession |
| `train_dn_gof_lof` | Training set membership per classifier (DN\|GOF\|LOF, 1=yes) |
| `rank_max` | Mechanism with the highest percentile rank |
| `verdict` | Final predicted mechanism (`dn`, `gof`, or `lof`) |
| `pDN` | SVM probability of dominant-negative mechanism |
| `DN_pctl` | Proteome-wide percentile rank of pDN |
| `pGOF` | SVM probability of gain-of-function mechanism |
| `GOF_pctl` | Proteome-wide percentile rank of pGOF |
| `pLOF` | SVM probability of loss-of-function mechanism |
| `LOF_pctl` | Proteome-wide percentile rank of pLOF |

The Badonyi feature builder joins on gene symbol and uses `pDN`, `pGOF`, `pLOF` as features (plus missingness indicators and Pfam family-mean-centred residuals).

**GeneBayes s_het** (`data/downloads/` — to be placed manually): posterior estimates of the selection coefficient against heterozygous loss-of-function (s_het) for all human protein-coding genes, from Zeng et al. 2023 (GeneBayes, Pritchard lab). Used as a replacement for ClinGen HI_score, which has 80% missingness. Data: https://doi.org/10.5281/zenodo.7939767. Cited by: Spence, Jeffrey P. et al. "Specificity, length and luck drive gene rankings in association studies." DOI: 10.1038/s41586-025-09703-7 (2025).

---

## Results summary

Current run (`run_biorxiv`), five seeds, cluster-bootstrap confidence intervals.

| Experiment | Metric | Gene split | Family split |
|---|---|---|---|
| Mechanism, linear probe on delta | macro-F1 | 0.288 | 0.290 |
| Mechanism, linear probe on wildtype-only | macro-F1 | 0.552 | 0.450 |
| Mechanism, exploratory MLP on delta | macro-F1 | 0.395 | 0.375 |
| Majority-class reference | macro-F1 | 0.288 | 0.290 |
| Pathogenicity control, MLP on delta | AUROC | 0.887 | 0.885 |
| Enzyme type control, linear on wildtype | macro-F1 | 0.832 | 0.779 |

| Pathogenicity feature, family split | AUROC |
|---|---:|
| Direction only, MLP | 0.893 |
| Magnitude only, MLP | 0.610 |
| Masked-marginal conservation score | 0.888 |

| Stability probe on mean delta | Random split Spearman | Domain split Spearman | Family split Spearman |
|---|---:|---:|---:|
| Ridge | 0.693 | 0.601 | 0.554 |
| MLP | 0.868 | 0.703 | 0.627 |
| XGBoost | 0.767 | 0.676 | 0.630 |

The descriptive leakage fraction for the wildtype-only mechanism probe is 0.389 (95% CI 0.239 to
0.543). The analysis does not establish family recognition as the cause of the gene-to-family
decrease.

The gene-level proteome and Badonyi structural-prior arcs have not been re-measured in the current
run, so no numbers are quoted for them here. The run0-era figures are in `docs/README.md`.

Read `biorxiv/README.md` for the current run and `biorxiv/PROGRESS.md` for what has been executed.
