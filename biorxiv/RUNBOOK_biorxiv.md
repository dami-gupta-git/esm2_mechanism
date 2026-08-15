# run_biorxiv runbook

This pipeline tests whether ESM-2 embeddings encode the *mechanism* of a pathogenic missense
variant — whether it acts as dominant-negative, loss-of-function, or gain-of-function — beyond
what is explained by homology between genes in the same protein family. Alongside the mechanism
test, two positive controls establish that the embeddings carry usable signal at all: a
pathogenicity classifier (ClinVar pathogenic vs. benign) and a physical stability predictor
(Tsuboyama ΔΔG). A geometry analysis asks what the pathogenicity direction in embedding space
actually corresponds to (conservation, largely). run_biorxiv re-scores this whole pipeline with
statistics that account for genes in the same family not being independent — cluster-bootstrap
confidence intervals, permutation p-values, and paired tests behind every "beats baseline" or "A
beats B" claim — replacing run6's 5-seed fold-jitter error bars. The experiments, hypotheses, and
gates are the same as run6's; only the statistics and, where the data changed, the ClinVar
snapshot are new.

Supersedes `RUNBOOK_biorxiv_old.md`, which became inconsistent after a second ClinVar refetch and
is kept for reference only, not as a status source.

Decision rules, resampling units, and the confirmatory/exploratory split: see
[`PREREGISTRATION_run_biorxiv.md`](PREREGISTRATION_run_biorxiv.md). Design of the statistics
machinery and non-obvious findings: see [`docs/FINDINGS.md`](../docs/FINDINGS.md).

Outputs go to `results/run_biorxiv/` and `reports/run_biorxiv/`. `results/run6/` and
`reports/run6/` are preserved untouched as the comparison baseline.

All commands use `python -m esm2_mech.<module>` from the project root with the package installed
(`pip install -e .`). This document holds the steps only — live status is tracked separately in
[`PROGRESS.md`](PROGRESS.md).

**RunPod:** connect with the `id_runpod_2` key (`id_runpod` does NOT work). Run inside `tmux`.

```bash
ssh -i ~/.ssh/id_runpod_2 root@<pod-ip> -p <pod-port>
```

---

## Prerequisites — manually placed files

These must be in `data/downloads/` before the pipeline starts. Scoped to what run_biorxiv's
experiments (1, 2, 3, 5, 7) actually read — the proteome-features and Badonyi downloads in
`RUNBOOK_4.md` are for experiments not part of this run.

| File | Source |
|---|---|
| `DiseaseMech_Stability_VEPS.xlsx` | Gerasimavicius et al. 2022 — OSF [10.17605/OSF.IO/H62FQ](https://osf.io/rct6d/download) |
| `AllG2P.csv` | G2P bulk download — gene2phenotype.org |

---

## Stage 0 — preconditions

Must all hold before `RUN_NAME` is flipped. 0.0 gets the environment running, 0.2–0.4 are the
substance of the run, 0.5–0.6 fix how results may be read, 0.7–0.9 protect provenance.

**0.0 — Environment setup.**

```bash
cd /Users/dgupta/code/portfolio/ESM2/esm2_mechanism
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

**0.1 — Pathogenicity provenance.** Locked to one canonical variant set;
`pathogenicity_control.py` fingerprints it and refuses mismatched embeddings.

**0.2 — Stats machinery wired.** Every result-producing script wired to `utils/bootstrap.py`,
emits CI keys. Verify by running each module for one seed and confirming `ci_low`/`ci_high` are
actually populated, not just that it exited cleanly.

**0.3 — Methodology rules.** R7.3 (resampling unit/pairing) and R7.4 (rare-class intervals) of the
pre-registration, implemented by the 0.2 wiring.

**0.4 — Paired cluster bootstrap.** Implemented in `utils/bootstrap.py`, call sites
`conservation_axis.py` and `mechanism_delta_family_split.py`. Covers three paired claims: the
conservation-vs-embedding-delta gap that gate K2 turns on, the pathogenicity-vs-mechanism
cross-family transfer contrast (not paired — different datasets, no shared row space), and the
gene-split-minus-family-split gap (the leakage account, C2).

**0.5/0.6 — Pre-registered decision rules.** CI decision rule (R7.1) and confirmatory/exploratory
split (R7.2), written into
[`PREREGISTRATION_run_biorxiv.md`](PREREGISTRATION_run_biorxiv.md) before the run.

**0.7 — Pinned environment.** Confirm before the run:

- `pytest tests/` passes locally.
- This run uses two machines: the CPU probe and bootstrap steps run on your local machine, and the
  embedding/extraction steps run on RunPod (a GPU). Each result file records which machine produced
  it.
- On each machine, run the command below and save its output. This records the exact package
  versions the run's numbers were computed under, so a later report can cite them.

```bash
python -c "
import platform, importlib.metadata as md
print('python', platform.python_version(), '|', platform.platform())
for p in ['numpy','scipy','scikit-learn','pandas','torch','fair-esm','xgboost','biopython','joblib']:
    try: print(f'{p}=={md.version(p)}')
    except Exception: print(f'{p}: MISSING')
"
```

**0.8 — Configuration.** `RUN_NAME = "run6"` → `"run_biorxiv"` in `utils/paths.py:11`, only after
0.2 and 0.4 pass their gates. `PERMUTATION_FEATURES` stays `("delta_mean", "wt_only_mean")`;
`PERMUTATION_N_RESAMPLES` stays at the `constants.py` default of 1000, not run6's 200.

**0.9 — Working tree clean.** `git status` clean at the branch point.

---

## Stage 1 — build gene list

This builds the list of genes every later experiment uses. Each gene in the output is labelled with
the disease mechanism its variants are known to cause: dominant-negative (DN), loss-of-function
(LOF), or gain-of-function (GOF). Later experiments test whether ESM-2 can predict this label from
a variant's sequence.

The label comes from merging two published sources of curated gene-disease data:

- **Gerasimavicius et al. 2022** — a spreadsheet of genes, experts assign the mechanism labels. Used as the primary source.
- **G2P (Gene2Phenotype)** — a database with its own mechanism label per gene
  - For genes Gerasimavicius doesn't cover
  - Filtered by G2P's confidence in that label - "definitive" or "strong."

When both sources cover the same gene and disagree on the label, the Gerasimavicius label is kept,
and the disagreement is recorded in the output.

This step reads the two files listed in Prerequisites above.

| Command | Description | Inputs | Outputs |
|---|---|---|---|
| `python -m esm2_mech.fetch_data.build_gene_list` | Build merged gene list | `downloads/DiseaseMech_Stability_VEPS.xlsx`, `downloads/AllG2P.csv` | `data/gene_list.tsv` |

---

## Stage 2 — fetch variant data (CPU)

The merged variant set and its annotations form a shared foundation used by Experiments 1, 2, 3, 5, and 7 (they are not experiment-specific).
The ClinVar fetch is the slowest step because it queries NCBI once per gene. Genes whose esearch/esummary calls fail are neither written nor cached, so re-running the same command automatically retries only those genes.

| Step | Command | Description | Inputs | Outputs |
|---|---|---|---|---|
| 1 | `python -m esm2_mech.fetch_data.fetch_variants --step gerasimavicius` | Fetch Gerasimavicius variants | `downloads/DiseaseMech_Stability_VEPS.xlsx` | `gerasimavicius_variants.json` |
| 2 | `python -m esm2_mech.fetch_data.fetch_variants --step clinvar` | Fetch ClinVar variants | `gene_list.tsv` | `clinvar_variants.tsv` |
| 3 | `python -m esm2_mech.fetch_data.fetch_variants --step merge --pathogenic_only` | Merge variant datasets (pathogenic only, drops likely pathogenic) | `gerasimavicius_variants.json`, `gene_list.tsv`, `clinvar_variants.tsv` | `variants.json` |
| 4 | `python -m esm2_mech.fetch_data.fetch_sequences` | Fetch UniProt sequences | `variants.json` | `cache/sequences.json` |
| 5 | `python -m esm2_mech.fetch_data.fetch_annotations --step pfam` | Fetch Pfam families | `variants.json` | `pfam_families.json` |
| 6 | `python -m esm2_mech.fetch_data.fetch_alphamissense_mechanism` | Fetch AlphaMissense scores | `variants.json` | `alphamissense_scores_full.json` |
| 7 | `python -m esm2_mech.fetch_data.build_valid_variants` | Build filtered variant list | `variants.json`, `cache/sequences.json` | `valid_variants.json` |

---

## Stage 3 — embed variants (GPU)

This turns each variant's wildtype and mutant sequence into ESM-2 embeddings. It is a shared
foundation, not specific to one experiment — Experiment 1 reads it directly, and Experiment 3
(within-family mechanism) reuses the same arrays. It must be re-extracted whenever Stage 2 (fetch
variant data) produces a new `valid_variants.json`; an embedding array built from an older variant
list will not line up with the current one.

This script runs on the pod but reads `data/valid_variants.json` and `data/cache/sequences.json`
from the pod's own filesystem, so copy those two files there first, before launching the script:

```bash
scp -i ~/.ssh/id_runpod_2 -P <pod-port> data/valid_variants.json root@<pod-ip>:/workspace/repo/data/
scp -i ~/.ssh/id_runpod_2 -P <pod-port> data/cache/sequences.json root@<pod-ip>:/workspace/repo/data/cache/
```

| Command | Description | Inputs | Outputs |
|---|---|---|---|
| `python -m esm2_mech.embeddings.embed_variants --model esm2_t33_650M_UR50D` | Extract ESM-2 embeddings | `valid_variants.json`, `cache/sequences.json` | `embeddings_wt_mean.npy`, `embeddings_mut_mean.npy`, `embeddings_wt_pos.npy`, `embeddings_mut_pos.npy`, `embedded_variants.json` |

This runs on the pod, so afterward copy the output files back to your local machine (all
subsequent steps run on CPU and read from the local `data/embeddings/` directory):

```bash
scp -i ~/.ssh/id_runpod_2 -P <pod-port> \
    root@<pod-ip>:/workspace/repo/data/embeddings/esm2_t33_650M_UR50D/*.npy \
    root@<pod-ip>:/workspace/repo/data/embeddings/esm2_t33_650M_UR50D/embedded_variants.json \
    data/embeddings/esm2_t33_650M_UR50D/
```

**Output:** `data/embeddings/<ESM2_MODEL>/embeddings_{wt,mut}_{mean,pos}.npy`,
`embedded_variants.json`. After this runs, confirm all four `.npy` arrays have the same row count
as `embedded_variants.json` and `valid_variants.json`. Spot-check a few rows to confirm the
three files are in the same row order.

---

## Experiment 1 — ESM-2 delta-embedding mechanism

This experiment tests whether ESM-2 embeddings can predict a variant's mechanism (DN/LOF/GOF), and
whether that prediction still holds once genes from the same protein family are kept out of the
opposite train/test split, so the model can't just be recognizing the family.

Three scripts used in this experiment (`classify_by_mechanism`, `single_source_mechanism`,
`mechanism_delta_family_split`) all accept `--no_ci` (skip the confidence-interval computation,
for faster iteration only), `--n_boot N` (number of bootstrap resamples, default 1000), and
`--n_permutations N` (run a permutation test, default 0 = off). None of the commands below pass
`--no_ci`, so confidence intervals are computed by default everywhere in this experiment.

### Step 1 — run analysis (CPU)

This step runs five scripts. The first four each read the embeddings and write their own result
file, and don't depend on each other, so they can run in any order or in parallel. The fifth,
`leakage_fraction`, does not look at the embeddings at all — it just reads the result files the
first four already wrote and combines their numbers. So it has to run last, after the other four
have finished.

Each script's confidence-interval computation already uses all available CPU cores (`n_jobs=-1`
in `utils/bootstrap.py`), so running the four in parallel on one machine means they split those
cores rather than each getting the full machine — a faster or more-cored CPU (local or a RunPod
CPU instance) helps more than trying to parallelize them on a small machine:

| Step | Command | Description | Inputs | Outputs |
|---|---|---|---|---|
| 1 | `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 5` | Gene-split vs family-split baseline comparison | `variants.json`, `cache/sequences.json`, `pfam_families.json`, `embeddings_*.npy`, `alphamissense_scores_full.json` | `results/<run>/family_split_baselines_seed{0..4}.json` |
| 2 | `python -m esm2_mech.experiments.mechanism.mlp --seeds 5` | Nonlinear classifiers (MLP, GBM, RF, kNN) on delta embeddings | `valid_variants.json`, `pfam_families.json`, `embeddings_*.npy` | `results/<run>/nonlinear_results_seed{0..4}.json` |
| 3 | `python -m esm2_mech.experiments.mechanism.family_clustering --seeds 5` | Diagnostic: do ESM-2 embeddings cluster by Pfam family? (kNN purity, within/between distance, family probe, mechanism–family overlap) — explains the homology leakage in the WT-only baseline | `valid_variants.json`, `pfam_families.json`, `embeddings_wt_mean.npy`, `embeddings_mut_mean.npy` | `results/<run>/family_clustering.json` |
| 4 | `python -m esm2_mech.experiments.mechanism.naive_baseline` | Measured majority-class / stratified macro-F1 + AUROC floor (DummyClassifier, 5 seeds, same CV) — the chance reference for the other tables | `valid_variants.json`, `pfam_families.json` | `results/<run>/naive_baseline.json` |
| 5 | `python -m esm2_mech.experiments.mechanism.leakage_fraction` | Derived diagnostic: leakage fraction per feature = (gene − family macro-F1) / (gene − chance), the share of each feature's above-chance gene-split score attributable to family recognition | `family_split_baselines_seed{0..4}.json`, `naive_baseline.json`, `family_clustering.json` | `results/<run>/leakage_fraction.json` |

All outputs write to `results/<run>/`. Each of `classify_by_mechanism`, `single_source_mechanism`,
and `mechanism_delta_family_split` accepts `--no_ci` (skip the confidence-interval computation, for
faster iteration only), `--n_boot N` (number of bootstrap resamples, default 1000), and
`--n_permutations N` (run a permutation test, default 0 = off).

### Step 2 — permutation tests (CPU, seed 0 only)

A permutation test checks whether the family-split score is better than what pure chance would
produce, by repeatedly shuffling the labels and re-scoring. Run this separately from Step 1, in its
own tmux window, because it is far more expensive: the MLP-probe feature `wt_only_mean` re-trains
the probe once per shuffle.

| Command | Description | Outputs |
|---|---|---|
| `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 1 --n_permutations 1000` | Permutation p-value for the family-split score, seed 0 only | `results/<run>/...` |

This only needs to run at seed 0: a permutation test builds its own reference distribution by
shuffling, so running it at every seed would mostly re-measure the same seed-to-seed noise this
whole run already accounts for elsewhere. It also only needs to run on the linear probe: the
headline claim under test is a linear-probe result, so that is the test that has to be well
powered, and no claim in this run depends on an MLP permutation p-value.

Before launching on the pod, time a single re-fit so you know whether the full 1,000-permutation
run will take hours or days.

### Step 3 — single-source robustness check (CPU)

Re-runs the Step 1 probe on the subset of variants that came from a single curation source
(Gerasimavicius), instead of the merged ClinVar + Gerasimavicius set, as a check that the mechanism
result isn't an artifact of merging two differently-curated datasets.

| Command | Description | Outputs |
|---|---|---|
| `python -m esm2_mech.experiments.mechanism.single_source_mechanism --seeds 5` | Re-run the mechanism probe on the Gerasimavicius-only subset | `results/<run>/single_source_gerasimavicius/{family_split_baselines_seed{0..4}.json, aggregate.json, naive_baseline.json}` |

---

## Experiment 2 — pathogenicity positive control

TBD

---

## Experiment 3 — within-family mechanism

TBD

---

## Experiment 5 — geometry of the pathogenicity direction

TBD

---

## Experiment 7 — megascale stability positive control

TBD

---

## Verification checklist

TBD
