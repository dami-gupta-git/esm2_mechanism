# run_biorxiv runbook

This pipeline tests whether ESM-2 embeddings encode the *mechanism* of a pathogenic missense
variant — whether it acts as dominant-negative, loss-of-function, or gain-of-function — beyond
what is explained by homology between genes in the same protein family. Alongside the mechanism
test, three positive controls establish that the embeddings carry usable signal at all: a
pathogenicity classifier (ClinVar pathogenic vs. benign), a physical stability predictor
(Tsuboyama ΔΔG), and an enzyme type classifier (kinase/protease/oxidoreductase/non-enzyme from WT
embeddings). A geometry analysis asks what the pathogenicity direction in embedding space actually
corresponds to (conservation, largely). run_biorxiv re-scores this whole pipeline with statistics
that account for genes in the same family not being independent — cluster-bootstrap confidence
intervals, permutation p-values, and paired tests behind every "beats baseline" or "A beats B"
claim — replacing run6's 5-seed fold-jitter error bars. The experiments, hypotheses, and gates are
the same as run6's; only the statistics and, where the data changed, the ClinVar snapshot are new.

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

### Compute requirements by step

Each step falls into one of three tiers. "More cores help" means the bootstrap uses all available
CPUs, so a 32-core machine finishes roughly 4× faster than an 8-core laptop.

| Tier | Steps | What drives the cost |
|---|---|---|
| **GPU required** | 3.3 (embed variants), 5.2 (pathogenicity embed+probe), 6.5 (conservation extract), 7.3 (MLP+XGBoost) | ESM-2 forward pass or PyTorch/XGBoost training on GPU |
| **CPU-intensive — more cores help** | 4.1, 4.2, 4.3, 4.6, 4.7, 5.2 probe phase, 6.2, 6.7, 7.2, 8.1 | 1,000-resample cluster bootstrap and/or multi-seed sklearn CV, all with `n_jobs=-1` |
| **Light — runs anywhere** | 4.4, 4.5, 6.1, 7.1, 7.4 | Trivial models, file reads, or simple ratio computations |

---

## Prerequisites — manually placed files

These must be in `data/downloads/` before the pipeline starts. Scoped to what run_biorxiv's
experiments (1, 2, 5, 7) actually read — the proteome-features and Badonyi downloads in
`RUNBOOK_4.md` are for experiments not part of this run.

| File | Source |
|---|---|
| `DiseaseMech_Stability_VEPS.xlsx` | Gerasimavicius et al. 2022 — OSF [10.17605/OSF.IO/H62FQ](https://osf.io/rct6d/download) |
| `AllG2P.csv` | G2P bulk download — gene2phenotype.org |

---

## 0. Preconditions

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

**0.3 — Methodology rules.** Rule 3 (resampling unit/pairing) and Rule 4 (rare-class intervals) of the
pre-registration, implemented by the 0.2 wiring.

**0.4 — Paired cluster bootstrap.** Implemented in `utils/bootstrap.py`, call sites
`conservation_axis.py` and `mechanism_delta_family_split.py`. Covers two paired claims — the
conservation-vs-embedding-delta gap that gate 1B turns on and the
gene-split-minus-family-split gap (the leakage account, 2B) — plus the pathogenicity-vs-mechanism
cross-family transfer contrast (not paired — different datasets, no shared row space).

**0.5/0.6 — Pre-registered decision rules.** CI decision rule (Rule 1) and confirmatory/exploratory
split (Rule 2), written into
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

## 1. Build gene list

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

| Step | Command | Description | Inputs | Outputs |
|---|---|---|---|---|
| 1.1 | `python -m esm2_mech.fetch_data.build_gene_list` | Build merged gene list | `downloads/DiseaseMech_Stability_VEPS.xlsx`, `downloads/AllG2P.csv` | `data/gene_list.tsv` |

---

## 2. Fetch variant data (CPU)

The merged variant set and its annotations form a shared foundation used by sections 4, 5, 6, and 7 (they are not experiment-specific).
The ClinVar fetch is the slowest step because it queries NCBI once per gene. Genes whose esearch/esummary calls fail are neither written nor cached, so re-running the same command automatically retries only those genes.

| Step | Command | Description | Inputs | Outputs |
|---|---|---|---|---|
| 2.1 | `python -m esm2_mech.fetch_data.fetch_variants --step gerasimavicius` | Fetch Gerasimavicius variants | `downloads/DiseaseMech_Stability_VEPS.xlsx` | `gerasimavicius_variants.json` |
| 2.2 | `python -m esm2_mech.fetch_data.fetch_variants --step clinvar` | Fetch ClinVar variants | `gene_list.tsv` | `clinvar_variants.tsv` |
| 2.3 | `python -m esm2_mech.fetch_data.fetch_variants --step merge --pathogenic_only` | Merge variant datasets (pathogenic only, drops likely pathogenic) | `gerasimavicius_variants.json`, `gene_list.tsv`, `clinvar_variants.tsv` | `variants.json` |
| 2.4 | `python -m esm2_mech.fetch_data.fetch_sequences` | Fetch UniProt sequences | `variants.json` | `cache/sequences.json` |
| 2.5 | `python -m esm2_mech.fetch_data.fetch_annotations --step pfam` | Fetch Pfam families | `variants.json` | `pfam_families.json` |
| 2.6 | `python -m esm2_mech.fetch_data.fetch_alphamissense_mechanism` | Fetch AlphaMissense scores | `variants.json` | `alphamissense_scores_full.json` |
| 2.7 | `python -m esm2_mech.fetch_data.build_valid_variants` | Build filtered variant list | `variants.json`, `cache/sequences.json` | `valid_variants.json` |
| 2.8 | `python -m esm2_mech.fetch_data.fetch_pathogenicity_variants` | Fetch balanced pathogenic/benign ClinVar variants for section 5 (separate from step 2.2's pathogenic-only fetch) | `variants.json` | `clinvar_pathogenicity_variants.json`, `clinvar_pathogenicity_variants.params.json` |

Step 2.8 is network-only, so it runs locally rather than on the pod, unlike section 5's embedding step below. It caches its output and only re-fetches if `--max_per_gene_per_class` (default 20) or `--fetch_seed` (default 42) change from what produced the cached file.

---

## 3. Embed variants (GPU)

This turns each variant's wildtype and mutant sequence into ESM-2 embeddings. It is a shared
foundation, not specific to one experiment — section 4 reads it directly. It must be
re-extracted whenever section 2 (fetch variant data) produces a new `valid_variants.json`; an
embedding array built from an older variant list will not line up with the current one.

This script runs on the pod but reads `data/valid_variants.json` and `data/cache/sequences.json`
from the pod's own filesystem, so copy those two files there first, before launching the script.

| Step | Command | Description |
|---|---|---|
| 3.1 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> data/valid_variants.json root@<pod-ip>:/workspace/repo/data/` | Copy valid_variants.json to pod |
| 3.2 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> data/cache/sequences.json root@<pod-ip>:/workspace/repo/data/cache/` | Copy sequences.json to pod |
| 3.3 | `python -m esm2_mech.embeddings.embed_variants --model esm2_t33_650M_UR50D` | Extract ESM-2 embeddings. Inputs: `valid_variants.json`, `cache/sequences.json`. Outputs: `embeddings_wt_mean.npy`, `embeddings_mut_mean.npy`, `embeddings_wt_pos.npy`, `embeddings_mut_pos.npy`, `embedded_variants.json` |
| 3.4 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> root@<pod-ip>:/workspace/repo/data/embeddings/esm2_t33_650M_UR50D/*.npy root@<pod-ip>:/workspace/repo/data/embeddings/esm2_t33_650M_UR50D/embedded_variants.json data/embeddings/esm2_t33_650M_UR50D/` | Copy embeddings and metadata back to local machine |

After step 3.4, confirm all four `.npy` arrays have the same row count as `embedded_variants.json`
and `valid_variants.json`. Spot-check a few rows to confirm the three files are in the same row
order.

---

## 4. Experiment: ESM-2 delta-embedding mechanism

This experiment tests whether ESM-2 embeddings can predict a variant's mechanism (DN/LOF/GOF), and
whether that prediction still holds once genes from the same protein family are kept out of the
opposite train/test split, so the model can't just be recognizing the family.

Several scripts used in this experiment (`classify_by_mechanism`, `single_source_mechanism`,
`mechanism_delta_family_split`) all accept `--no_ci` (skip the confidence-interval computation,
for faster iteration only), `--n_boot N` (number of bootstrap resamples, default 1000), and
`--n_permutations N` (run a permutation test, default 0 = off). None of the commands below pass
`--no_ci`, so confidence intervals are computed by default everywhere in this experiment.

### Run analysis (CPU)

Steps 4.1–4.4 each read the embeddings and write their own result file, and don't depend on each
other, so they can run in any order or in parallel. Step 4.5 (`leakage_fraction`) does not look at
the embeddings at all — it just reads the result files the earlier steps already wrote and combines
their numbers. So it has to run last, after the others have finished.

Each script's confidence-interval computation already uses all available CPU cores (`n_jobs=-1`
in `utils/bootstrap.py`), so running them in parallel on one machine means they split those
cores rather than each getting the full machine — a faster or more-cored CPU (local or a RunPod
CPU instance) helps more than trying to parallelize them on a small machine:

| Step | Command | Description | Inputs | Outputs |
|---|---|---|---|---|
| 4.1 | `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 5` | Gene-split vs family-split baseline comparison | `variants.json`, `cache/sequences.json`, `pfam_families.json`, `embeddings_*.npy`, `alphamissense_scores_full.json` | `results/<run>/family_split_baselines_seed{0..4}.json` |
| 4.2 | `python -m esm2_mech.experiments.mechanism.mlp --seeds 5` | Nonlinear classifiers (MLP, GBM, RF, kNN) on delta embeddings | `valid_variants.json`, `pfam_families.json`, `embeddings_*.npy` | `results/<run>/nonlinear_results_seed{0..4}.json` |
| 4.3 | `python -m esm2_mech.experiments.mechanism.family_clustering --seeds 5` | Diagnostic: do ESM-2 embeddings cluster by Pfam family? (kNN purity, within/between distance, family probe, mechanism–family overlap) — explains the homology leakage in the WT-only baseline | `valid_variants.json`, `pfam_families.json`, `embeddings_wt_mean.npy`, `embeddings_mut_mean.npy` | `results/<run>/family_clustering.json` |
| 4.4 | `python -m esm2_mech.experiments.mechanism.naive_baseline` | Measured majority-class / stratified macro-F1 + AUROC floor (DummyClassifier, 5 seeds, same CV) — the chance reference for the other tables | `valid_variants.json`, `pfam_families.json` | `results/<run>/naive_baseline.json` |
| 4.5 | `python -m esm2_mech.experiments.mechanism.leakage_fraction` | Derived diagnostic: leakage fraction per feature = (gene − family macro-F1) / (gene − chance), the share of each feature's above-chance gene-split score attributable to family recognition | `family_split_baselines_seed{0..4}.json`, `naive_baseline.json`, `family_clustering.json` | `results/<run>/leakage_fraction.json` |

All outputs write to `results/<run>/`. Each of `classify_by_mechanism`, `single_source_mechanism`,
and `mechanism_delta_family_split` accepts `--no_ci` (skip the confidence-interval computation, for
faster iteration only), `--n_boot N` (number of bootstrap resamples, default 1000), and
`--n_permutations N` (run a permutation test, default 0 = off).

### Permutation tests (CPU, seed 0 only)

A permutation test checks whether the family-split score is better than what pure chance would
produce, by repeatedly shuffling the labels and re-scoring. Run this separately from steps 4.1–4.5, in its
own tmux window, because it is far more expensive: the MLP-probe feature `wt_only_mean` re-trains
the probe once per shuffle.

| Step | Command | Description | Outputs |
|---|---|---|---|
| 4.6 | `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 1 --n_permutations 1000` | Permutation p-value for the family-split score, seed 0 only | `results/<run>/...` |

This only needs to run at seed 0: a permutation test builds its own reference distribution by
shuffling, so running it at every seed would mostly re-measure the same seed-to-seed noise this
whole run already accounts for elsewhere. It also only needs to run on the linear probe: the
headline claim under test is a linear-probe result, so that is the test that has to be well
powered, and no claim in this run depends on an MLP permutation p-value.

Before launching on the pod, time a single re-fit so you know whether the full 1,000-permutation
run will take hours or days.

### Single-source robustness check (CPU)

Re-runs the step 4.1 probe on the subset of variants that came from a single curation source
(Gerasimavicius), instead of the merged ClinVar + Gerasimavicius set, as a check that the mechanism
result isn't an artifact of merging two differently-curated datasets.

| Step | Command | Description | Outputs |
|---|---|---|---|
| 4.7 | `python -m esm2_mech.experiments.mechanism.single_source_mechanism --seeds 5` | Re-run the mechanism probe on the Gerasimavicius-only subset | `results/<run>/single_source_gerasimavicius/{family_split_baselines_seed{0..4}.json, aggregate.json, naive_baseline.json}` |

---

## 5. Experiment: Pathogenicity positive control

Tests whether the same delta embeddings that show no mechanism signal in section 4 can still
tell pathogenic from benign ClinVar variants, confirming they carry usable signal at all. Pass
criterion: `delta_mean` MLP AUROC ≥ 0.85.

A small neural network (the MLP) is trained to look at a variant's embedding and guess whether it
is disease-causing or harmless. AUROC is a score from 0.5 to 1 measuring how well it separates the
two: 0.5 means no better than a coin flip, 1.0 means it always gets it right. The 0.85 threshold is
set in advance — scoring at least that well is treated as proof the embeddings carry real
biological signal, since section 4 found mechanism prediction near chance.

This experiment uses its own ClinVar pull, separate from section 2's. Step 2.2 fetched
pathogenic variants only, to label genes by mechanism; step 2.8
(`fetch_pathogenicity_variants`) fetches benign variants too, in equal numbers to pathogenic ones
per gene, to train a pathogenic-vs-benign classifier. Run step 2.8 before this experiment, if
not already done — its output is this experiment's input.

One script runs the remaining phases in sequence: extracting ESM-2 embeddings for the fetched
variants (GPU), then running the pathogenic-vs-benign probe (CPU). Because the first phase needs a
GPU, this runs on the pod.

| Step | Command | Description |
|---|---|---|
| 5.1 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> data/clinvar_pathogenicity_variants.json root@<pod-ip>:/workspace/repo/data/` | Copy pathogenicity variants to pod |
| 5.2 | `python -m esm2_mech.experiments.pathogenicity.pathogenicity_control --model esm2_t33_650M_UR50D` | Embed the fetched pathogenicity variants, run the pathogenic-vs-benign probe. Inputs: `clinvar_pathogenicity_variants.json`, `cache/sequences.json`, `pfam_families.json`. Outputs: `data/embeddings/esm2_t33_650M_UR50D/pathogenicity_{wt,mut}_mean.npy`, `data/embeddings/esm2_t33_650M_UR50D/pathogenicity_meta.json`, `results/run_biorxiv/pathogenicity_control.json` |
| 5.3 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> root@<pod-ip>:/workspace/repo/data/embeddings/esm2_t33_650M_UR50D/pathogenicity_*.npy root@<pod-ip>:/workspace/repo/data/embeddings/esm2_t33_650M_UR50D/pathogenicity_meta.json root@<pod-ip>:/workspace/repo/results/run_biorxiv/pathogenicity_control.json data/embeddings/esm2_t33_650M_UR50D/` | Copy embeddings and results back to local machine |

Step 5.2 errors immediately if `clinvar_pathogenicity_variants.json` is missing.

Classes are balanced by construction (equal numbers of pathogenic and benign variants per gene).
However, genes still cluster into protein families, so confidence intervals continue to resample
whole genes rather than individual variants (as in section 4). The report should also note that the
probe measures discrimination between pathogenic and benign variants, not a calibrated risk
estimate for any single variant.

---

## 6. Experiment: Geometry of the pathogenicity direction

Section 5 shows the delta embeddings separate pathogenic from benign variants. This experiment
asks what that pathogenicity direction actually is: whether it is one shared direction across
protein families or many family-specific ones, whether it is more about how far a variant moves
the embedding or which way, whether it is explained by simple substitution chemistry (e.g. amino
acid size or charge change) or by ESM-2's own sense of how conserved a position is, and whether the
same direction transfers to the stability and mechanism tasks.


### Build canonical variant list (CPU)

Section 5's embedding step drops any variant it cannot embed (missing sequence, position out of
range, wildtype-residue mismatch), so the `.npy` embedding rows are a subset of the fetched variant
list, not a 1:1 match. This step is a re-indexing step: it takes the pathogenicity variants and embeddings 
Section 5 already produced and re-materializes them in a row-aligned file for the geometry scripts to read directly.

| Step | Command | Description | Inputs | Outputs |
|---|---|---|---|---|
| 6.1 | `python -m esm2_mech.experiments.geometry.build_canonical_pathogenicity` | Re-index the pathogenicity variant set to match the embedding row order | `clinvar_pathogenicity_variants.json`, `data/embeddings/esm2_t33_650M_UR50D/pathogenicity_{wt,mut}_mean.npy`, `pathogenicity_meta.json` | `pathogenicity_valid_variants_canonical.json` |

### Geometry probes (CPU)

One script runs the probes in sequence, each writing its own result file. `--probe` can restrict
this to a subset (e.g. `--probe magnitude geometry`); the default is all four.

| Probe | What it asks |
|---|---|
| magnitude | Does the pathogenicity signal come from how far the delta moves the embedding (magnitude), or which direction it moves in? |
| geometry | Is pathogenicity carried by a single direction (rank-1) or a higher-dimensional subspace, and does a direction fit on one set of protein families transfer to a disjoint set? |
| transfer | Under one identical protocol, does a direction fit on one half of the data transfer to the other half, compared for the pathogenicity, stability, and mechanism tasks? |
| biochem | How much of the direction is explained by context-free substitution chemistry (BLOSUM62 score, and changes in hydropathy, charge, and volume) rather than sequence context? |

| Step | Command | Description | Inputs | Outputs |
|---|---|---|---|---|
| 6.2 | `python -m esm2_mech.experiments.geometry.run_geometry --seeds 5` | Run all geometry probes | `pathogenicity_valid_variants_canonical.json`, `data/embeddings/esm2_t33_650M_UR50D/pathogenicity_{wt,mut}_mean.npy` | `results/<run>/magnitude_direction/probe_results.json`, `geometry_results.json`, `transfer_contrast.json`, `probe4_axis_identity.json` |

Only the magnitude probe has cluster-bootstrap confidence intervals wired (`--no_ci` / `--n_boot`
apply to it only); the others are rank and correlation probes with no CI attached. `--seeds`
applies to all probes.

### Conservation extract (GPU)

For each canonical pathogenicity variant, mask its wildtype position and read ESM-2's own predicted
probability of the wildtype residue, the mutant residue, and the entropy over all 20 amino acids at
that position — i.e., how confidently the model expects that position to be conserved. This is the
one GPU step in this experiment; it can share a pod session with any other GPU work already running.

| Step | Command | Description |
|---|---|---|
| 6.3 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> data/pathogenicity_valid_variants_canonical.json root@<pod-ip>:/workspace/repo/data/` | Copy canonical variants to pod |
| 6.4 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> data/cache/sequences.json root@<pod-ip>:/workspace/repo/data/cache/` | Copy sequences to pod |
| 6.5 | `python -m esm2_mech.experiments.geometry.conservation_axis --extract` | Masked-LM forward pass per variant to score how conserved its position is. Inputs: `pathogenicity_valid_variants_canonical.json`, `cache/sequences.json`. Outputs: `data/conservation_pathogenicity.npy`, `data/conservation_pathogenicity_meta.json` |
| 6.6 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> root@<pod-ip>:/workspace/repo/data/conservation_pathogenicity.npy root@<pod-ip>:/workspace/repo/data/conservation_pathogenicity_meta.json data/` | Copy conservation outputs back to local machine |

### Conservation analysis (CPU)

Compares the conservation features from step 6.5 to the pathogenicity direction found in step 6.2, on
the same family-split protocol. Pre-registered gates: 1A conservation alone reaches AUROC ≥ 0.85
(the axis is mostly conservation); 1B adding the embedding delta on top of conservation improves
AUROC by ≥ 0.02 (the embedding carries pathogenicity signal beyond conservation).

| Step | Command | Description |
|---|---|---|
| 6.7 | `python -m esm2_mech.experiments.geometry.conservation_axis` | Compare conservation features to the embedding-derived pathogenicity direction. Inputs: `conservation_pathogenicity.npy`, `pathogenicity_valid_variants_canonical.json`, `data/embeddings/esm2_t33_650M_UR50D/pathogenicity_{wt,mut}_mean.npy`. Output: `results/<run>/magnitude_direction/conservation_axis.json` |

run_biorxiv adds gene-cluster confidence intervals to each pathogenicity AUROC in this experiment
(resampled over genes, not individual variants, the same as sections 4 and 5), and a paired
cluster-bootstrap confidence interval on the 1B gap (conservation-alone AUROC vs. conservation-plus-
embedding-delta AUROC) and on the pathogenicity-vs-mechanism transfer contrast.

---

## 7. Experiment: Megascale stability positive control

A second positive control, alongside section 5, with a purely physical label instead of a
clinically curated one: Tsuboyama et al. 2023's measured folding stability change (ΔΔG) for about
177,000 single point mutations across about 181 natural protein domains. Because this label comes
from a direct physical measurement rather than expert curation, it rules out the concern that
section 5's pathogenicity signal is really the embeddings picking up on curation patterns rather
than biology.

The embedding step is skipped: `megascale_{wt,mut}_{mean,pos}.npy` already exist locally, extracted
in an earlier run, and are unaffected by this run's ClinVar refresh since this experiment has no
ClinVar dependency. Step 7.2's H3 test does read `valid_variants.json` and the section 4
embeddings, so run step 7.2 after step 4.1 has produced a current `valid_variants.json`.

### Assign Pfam families (CPU)

Assigns each Tsuboyama domain to a Pfam family via HMMER, so later steps can hold out whole families
rather than whole domains when testing whether the stability signal generalizes. Needs `hmmscan` on
the system path and a hmmpress-ed Pfam-A database; skip this step if `megascale_domain_families.json`
is already present and non-empty (it is, as of this writing).

| Step | Command | Description | Inputs | Outputs |
|---|---|---|---|---|
| 7.1 | `python -m esm2_mech.experiments.stability.build_domain_families` | Assign each Tsuboyama domain to a Pfam family | `megascale_tsuboyama_variants.json`, `downloads/megascale/Pfam-A.hmm` | `data/megascale_domain_families.json` |

### Embed variants (GPU) — skipped

Not run in run_biorxiv. `megascale_{wt,mut}_{mean,pos}.npy` under
`data/embeddings/esm2_t33_650M_UR50D/` are reused from an earlier run.

### Linear probe (CPU)

Fits a Ridge regression from the embeddings to ΔΔG under three cross-validation schemes — random
split, holding out whole domains, and holding out whole Pfam families — and tests four pre-registered
hypotheses: H1, the random-split correlation (Spearman ρ) reaches at least 0.5; H2, that correlation
drops by no more than 0.10 when switching to a family-split (a big drop would mean the model is
recognizing domains rather than learning a general stability signal); H3, projecting the fitted
stability direction out of section 4's mechanism-classification features does not raise the
family-split mechanism score by more than 0.01 (stability and mechanism should be separable); H4, the
per-domain spread in correlation stays tight (standard deviation ≤ 0.10).

| Step | Command | Description | Inputs | Outputs |
|---|---|---|---|---|
| 7.2 | `python -m esm2_mech.experiments.stability.megascale_stability` | Ridge probe from embeddings to ΔΔG under random/domain/family splits | `megascale_tsuboyama_variants.json`, `megascale_domain_families.json`, `data/embeddings/esm2_t33_650M_UR50D/megascale_{wt,mut}_{mean,pos}.npy`, `valid_variants.json` | `results/<run>/megascale_stability/per_protein_spearman.json`, `h3_stability_projection.json`, `summary.json` |

### Nonlinear probe (GPU)

Repeats step 7.2's three-way split comparison with a small neural network (MLP) alongside Ridge, plus
two exploratory tree-based models (random forest and gradient-boosted trees), to check whether a
nonlinear model finds more signal than the linear probe, and whether any such gain survives the
family-split. Only the Ridge and MLP results are pre-registered; the random forest and
gradient-boosted-tree numbers are exploratory. `--xgboost` adds the gradient-boosted-tree model,
which needs a GPU; without it, this step is CPU-only.

| Step | Command | Description | Inputs | Outputs |
|---|---|---|---|---|
| 7.3 | `python -m esm2_mech.experiments.stability.megascale_mlp --xgboost` | MLP/random-forest/gradient-boosted-tree probes on the same three splits | `megascale_tsuboyama_variants.json`, `megascale_domain_families.json`, `data/embeddings/esm2_t33_650M_UR50D/megascale_{wt,mut}_{mean,pos}.npy` | `results/<run>/megascale_stability/mlp_summary_xgb.json` |

### Controls (CPU)

Exploratory checks on the step 7.2 linear signal, not part of the pre-registered H1–H4 verdict:
whether a single feature (the size of the embedding shift, ignoring its direction) recovers most of
the full signal; the regularization strength chosen by nested cross-validation, so the main probe's
result isn't an artifact of one fixed setting; a label-shuffle null, where the ΔΔG values are
randomly permuted and the correlation should collapse to near zero as a leakage check; and how the
correlation changes as more embedding components are kept, to characterize how many dimensions the
stability signal actually occupies.

| Step | Command | Description | Inputs | Outputs |
|---|---|---|---|---|
| 7.4 | `python -m esm2_mech.experiments.stability.stability_baselines` | Delta-norm baseline, nested-CV alpha, label-shuffle null, and component sweep | `megascale_tsuboyama_variants.json`, `megascale_domain_families.json`, `data/embeddings/esm2_t33_650M_UR50D/megascale_{wt,mut}_{mean,pos}.npy` | `results/<run>/megascale_stability/baselines.json` |

Gates are unchanged from run6: H1 random-split ρ ≥ 0.5, H2 the random-to-family-split drop stays
below 0.10, H3 the mechanism-F1 change from projecting out stability stays ≤ +0.01, H4 per-domain ρ
standard deviation stays tight. run_biorxiv's only addition here is confidence intervals on these
figures; since this experiment has no ClinVar dependency, it is the one part of this run that
isolates the effect of the new statistics from any effect of the refreshed ClinVar snapshot.

---

## 8. Experiment: Enzyme type classification (positive control)

A third positive control that uses a wildtype-sequence property rather than a mutation property:
classifying each gene as kinase, protease, oxidoreductase, or non-enzyme from its WT mean-pooled
ESM-2 embedding. Enzyme class is strongly associated with protein fold, so ESM-2's known Pfam
clustering should help here — making it a direct test of whether the mechanism null (section 4) is a
property of the task, not a failure of the pipeline or the embeddings.

The experiment mirrors section 4's structure: gene-split, family-split, and the gap between them
(leakage fraction), run across 5 seeds with cluster-bootstrap CIs on the seed-0 family-split OOF
predictions. A proteome-features baseline (37 gene-level biology features) runs alongside as a
negative control — enzyme class is a structural property, not a population-genetics one, so
proteome features should be near chance.

### Prerequisites

No new data to fetch or embed. This experiment uses:

- `data/enzyme_labels.tsv` — already produced by step 2's annotation fetch (`fetch_annotations --step enzyme`)
- `data/embeddings/esm2_t33_650M_UR50D/embeddings_wt_mean.npy` — the WT embeddings from section 3
- `data/proteome_features_aligned.npy` — the gene-level proteome feature matrix

All three must exist before running. If `enzyme_labels.tsv` is missing, run:

```bash
python -m esm2_mech.fetch_data.fetch_annotations --step enzyme
```

### Run analysis (CPU)

| Step | Command | Description | Inputs | Outputs |
|---|---|---|---|---|
| 8.1 | `python -m esm2_mech.experiments.proteome_features.enzyme_classification --seeds 5` | Enzyme type classification: LogReg + MLP on WT embeddings and proteome features, gene-split and family-split, with cluster-bootstrap CIs | `valid_variants.json`, `embeddings_wt_mean.npy`, `enzyme_labels.tsv`, `pfam_families.json`, `proteome_features_aligned.npy`, `results/<run>/aggregate.json` (mechanism reference) | `results/<run>/enzyme_classification/enzyme_classification_summary.json` |

The script accepts `--no_ci` (skip cluster-bootstrap CIs), `--n_boot N` (default 1000), and
`--n_permutations N` (OOF permutation test, default 0 = skip), matching sections 4–7. The mechanism
reference F1 is read from section 4's aggregate result, not hardcoded — run section 4 first.

Decision rules (pre-registration §2E):

- **2E.1** — family-split LogReg macro-F1 ≥ 0.70. Enzyme class is strongly encoded in ESM-2 WT
  embeddings and family-split CV is a meaningful discriminator.
- **2E.2** — enzyme family-split F1 substantially exceeds the mechanism family-split floor (read
  from section 4's aggregate, not hardcoded). The mechanism null is task-specific, not a probe or
  data failure.
- **2E.3** — MLP does not substantially outperform LogReg under family-split (|delta F1| < 0.05).
  Linear readout is sufficient, paralleling pathogenicity and contrasting with stability.

---

## Verification checklist

TBD
