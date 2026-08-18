# esm2_mechanism_biorxiv

Does ESM-2 encode disease mechanism?

This repository is the current home of the project. It was previously developed at
`dami-gupta-git/esm2_mechanism`, which is no longer updated.

A research project testing whether frozen ESM-2 650M delta-embeddings (mutant − wildtype) encode gene-level dominant disease mechanism class (GOF / DN / LOF) beyond protein stability, evaluated with gene-split and family-split cross-validation.

**Headline finding:** ESM-2 encodes pathogenicity strongly and family-transferably (delta MLP family-split AUROC 0.897) but carries no mechanism signal in the mutant-minus-wildtype delta: a linear probe on the delta sits at the majority-class floor under both gene-split and family-split cross-validation (macro-F1 0.290 vs a 0.288 floor), and a nonlinear MLP reaches only 0.370 family-split. The apparent mechanism signal in wildtype-only embeddings drops from 0.552 gene-split to 0.449 family-split, about 30% of it family-recognition leakage. The null is task-specific rather than a pipeline failure: the same embeddings predict Megascale stability across held-out Pfam families (Spearman 0.554 linear, 0.634 MLP) and classify enzyme type family-split at macro-F1 0.674.

All numbers above are from the current run (`results/run_biorxiv/`), five seeds, with cluster-bootstrap confidence intervals; see `biorxiv/README.md`.

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
  utils/           — paths, constants, splits, metrics, bootstrap, shared probe code

biorxiv/           — the current run: pre-registration, runbook, progress, findings
docs/              — the earlier exploratory phase (stale) plus per-experiment plans
results/<run>/     — JSON outputs, one directory per run
reports/<run>/     — write-ups and figures for a run
scripts/           — the few CLI tools that are not part of the package
tests/
data/              — all inputs and embeddings (gitignored in full)
```

Directory paths are not written inline anywhere; they all come from `utils/paths.py`, and the run
directory is keyed off a single run-name constant so a result file and its report always match.

---

## Setup

Python 3.13.

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

---

## Quickstart

Run from the repo root. Each command writes into the run directory named by `RUN_NAME` in
`utils/paths.py`.

```bash
# Mechanism: linear baselines under gene-split and family-split CV (CPU, 5 seeds)
python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 5

# Mechanism: nonlinear probes on cached embeddings (CPU, 5 seeds)
python -m esm2_mech.experiments.mechanism.mlp --seeds 5

# Pathogenicity positive control (CPU probe phase, 5 seeds)
python -m esm2_mech.experiments.pathogenicity.pathogenicity_control --phase probe

# Megascale stability positive control (CPU, parallelism must be set explicitly)
python -m esm2_mech.experiments.stability.megascale_stability --n_jobs 4

# Enzyme-type positive control (CPU, 5 seeds)
python -m esm2_mech.experiments.proteome_features.enzyme_classification --seeds 5
```

`biorxiv/RUNBOOK_biorxiv.md` has the full ordered command list, including the GPU embedding steps,
and `biorxiv/PROGRESS.md` records what has been executed. For RunPod setup and remote execution, see
`docs/connect_runpod.md`.

---

## Data

The whole of `data/` is gitignored, so nothing here is shipped with the repository. It holds the
built variant and gene lists, the ESM-2 embedding arrays under `data/embeddings/`, downloaded
third-party files under `data/downloads/` and `data/cache/`, and the Megascale dataset under
`data/megascale/`. Embeddings are produced on a GPU machine and copied back; the loaders fingerprint
the variant list and refuse to run against embeddings that do not match it.

Two source files cannot be fetched automatically and must be placed in `data/downloads/` before
anything else runs: `DiseaseMech_Stability_VEPS.xlsx` and `AllG2P.csv`.

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

| Experiment | Metric | Gene-split | Family-split |
|---|---|---|---|
| Mechanism, linear probe on delta | macro-F1 | 0.288 | 0.290 |
| Mechanism, linear probe on wildtype-only | macro-F1 | 0.552 | 0.449 |
| Mechanism, MLP on delta | macro-F1 | 0.395 | 0.370 |
| Majority-class floor | macro-F1 | 0.288 | 0.290 |
| Pathogenicity control, MLP on delta | AUROC | 0.897 | 0.897 |
| Pathogenicity, direction only | AUROC | — | 0.904 |
| Pathogenicity, magnitude only | AUROC | — | 0.672 |
| Conservation axis alone | AUROC | — | 0.891 |
| Megascale stability, Ridge on delta | Spearman | 0.601 (domain) | 0.554 |
| Megascale stability, MLP on delta | Spearman | 0.715 (domain) | 0.634 |
| Enzyme type control, linear on wildtype | macro-F1 | 0.766 | 0.674 |

Family-recognition leakage in the wildtype-only mechanism probe: 30.0%.

The gene-level proteome and Badonyi structural-prior arcs have not been re-measured in the current
run, so no numbers are quoted for them here. The run0-era figures are in `docs/README.md`.

Read `biorxiv/README.md` for the current run and `biorxiv/PROGRESS.md` for what has been executed.
