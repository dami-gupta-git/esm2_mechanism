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
[`PREREGISTRATION_run_biorxiv.md`](PREREGISTRATION_run_biorxiv.md), whose revision history records
the rules revised on 2026-08-18 before the re-run. Design of the statistics
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

Compute tags appear next to each step: **🟢 light** (runs anywhere), **🟡 CPU — more cores help**
(bootstrap uses all cores; 128-core pod finishes ~16× faster than an 8-core laptop),
**🔴 GPU** (ESM-2 forward pass or PyTorch/XGBoost training).

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
conservation-vs-embedding-delta gap that claim 2E turns on and the
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

**0.10 — Megascale embedding provenance.** Run
`python -m esm2_mech.embeddings.embed_megascale --model esm2_t33_650M_UR50D` on the GPU host before
sections 6 and 7. A complete checkpoint is reused only when its `embedded_variants.json` sidecar
matches the current ordered Tsuboyama inputs. Otherwise the extraction resumes or restarts. Copy
the four `megascale_{wt,mut}_{mean,pos}.npy` arrays and `megascale_fingerprint.json` back to the
local embedding directory. The fingerprint records the exact sequence inputs, model, and array
content.

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
| 2.8 | `python -m esm2_mech.fetch_data.fetch_pathogenicity_variants --max_per_gene_per_class 20 --fetch_seed 42 --force` | Fetch balanced pathogenic/benign ClinVar variants for section 5 (separate from step 2.2's pathogenic-only fetch) | `variants.json` | `clinvar_pathogenicity_variants.json`, `clinvar_pathogenicity_variants.params.json` |

Step 2.8 is network-only, so it runs locally rather than on the pod. The command uses `--force`
because this repaired run must replace the pre-fix cache. Without `--force`, a current matching
cache is reused and a stale, partial, or corrupt cache raises instead of being replaced silently.

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
| 3.3 | `python -m esm2_mech.embeddings.embed_variants --model esm2_t33_650M_UR50D` | 🔴 GPU. Extract ESM-2 embeddings. Inputs: `valid_variants.json`, `cache/sequences.json`. Outputs: `embeddings_wt_mean.npy`, `embeddings_mut_mean.npy`, `embeddings_wt_pos.npy`, `embeddings_mut_pos.npy`, `embedded_variants.json` |
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

Steps 4.1–4.4 each read the embeddings and write their own result file, and do not depend on each
other, so they can run in any order or in parallel. Step 4.5 (`leakage_fraction`) does not look at
the embeddings at all — it just reads the result files the earlier steps already wrote and combines
their numbers. Run step 4.6 before step 4.5 because step 4.6 regenerates the seed results and their
bound OOF caches. Step 4.5 is the final Section 4 aggregation.

Each script's confidence-interval computation already uses all available CPU cores (`n_jobs=-1`
in `utils/bootstrap.py`), so running them in parallel on one machine means they split those
cores rather than each getting the full machine — a faster or more-cored CPU (local or a RunPod
CPU instance) helps more than trying to parallelize them on a small machine:

| Step | Command | Description | Inputs | Outputs |
|---|---|---|---|---|
| 4.1 | `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 5` | 🟡 CPU — more cores help. Gene-split vs family-split baseline comparison | `variants.json`, `cache/sequences.json`, `pfam_families.json`, `embeddings_*.npy`, `alphamissense_scores_full.json` | `results/<run>/family_split_baselines_seed{0..4}.json`, `results/<run>/mechanism_oof_cache_seed{0..4}.json` |
| 4.2 | `python -m esm2_mech.experiments.mechanism.mlp --seeds 5` | 🟡 CPU — more cores help. Nonlinear classifiers (MLP, GBM, RF, kNN) on delta embeddings | `valid_variants.json`, `pfam_families.json`, `embeddings_*.npy` | `results/<run>/nonlinear_results_seed{0..4}.json` |
| 4.3 | `python -m esm2_mech.experiments.mechanism.family_clustering --seeds 5` | 🟡 CPU — more cores help. Diagnostic: do ESM-2 embeddings cluster by Pfam family? (kNN purity, within/between distance, family probe, mechanism–family overlap) — explains the homology leakage in the WT-only baseline | `valid_variants.json`, `pfam_families.json`, `embeddings_wt_mean.npy`, `embeddings_mut_mean.npy` | `results/<run>/family_clustering.json` |
| 4.4 | `python -m esm2_mech.experiments.mechanism.naive_baseline` | 🟢 Light. Measured majority-class / stratified macro-F1 + AUROC floor (DummyClassifier, 5 seeds, same CV) — the chance reference for the other tables | `valid_variants.json`, `pfam_families.json` | `results/<run>/naive_baseline.json` |
| 4.5 | `python -m esm2_mech.experiments.mechanism.leakage_fraction` | 🟢 Light. Derived diagnostic: leakage fraction per feature = (gene − family macro-F1) / (gene − chance), the share of each feature's above-chance gene-split score attributable to family recognition | `family_split_baselines_seed{0..4}.json`, `mechanism_oof_cache_seed{0..4}.json`, `naive_baseline.json`, `family_clustering.json` | `results/<run>/leakage_fraction.json` |

All outputs write to `results/<run>/`. Each of `classify_by_mechanism`, `single_source_mechanism`,
and `mechanism_delta_family_split` accepts `--no_ci` (skip the confidence-interval computation, for
faster iteration only), `--n_boot N` (number of bootstrap resamples, default 1000), and
`--n_permutations N` (run a permutation test, default 0 = off).

### Permutation tests (CPU, all five seeds)

A permutation test checks whether the family-split score is better than what pure chance would
produce, by repeatedly shuffling the labels and re-scoring. Run this separately from steps 4.1–4.5, in its
own tmux window, because it is far more expensive: the MLP-probe feature `wt_only_mean` re-trains
the probe once per shuffle.

| Step | Command | Description | Outputs |
|---|---|---|---|
| 4.6 | `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 5 --n_permutations 1000` | 🟡 CPU — more cores help. Permutation p-value for the family-split score, all five seeds | `results/<run>/...` |

This runs on all five seeds: the seeds differ in their gene-to-family gap, so a single seed can
understate the effect and invites the objection that the seed was chosen. Per the pre-registration,
the refutation for claim 2A fires only when at least three of the five seeds return a p-value below
0.05; a minority of significant seeds is a split result and refutes nothing. It only needs to run on
the linear probe: the headline claim under test is a linear-probe result, so that is the test that
has to be well powered, and no claim in this run depends on an MLP permutation p-value.

Before launching on the pod, time a single re-fit so you know whether the full 1,000-permutation
run will take hours or days.

### Single-source robustness check (CPU)

Re-runs the step 4.1 probe on the subset of variants that came from a single curation source
(Gerasimavicius), instead of the merged ClinVar + Gerasimavicius set, as a check that the mechanism
result isn't an artifact of merging two differently-curated datasets.

| Step | Command | Description | Outputs |
|---|---|---|---|
| 4.7 | `python -m esm2_mech.experiments.mechanism.single_source_mechanism --seeds 5` | 🟡 CPU — more cores help. Re-run the mechanism probe on the Gerasimavicius-only subset | `results/<run>/single_source_gerasimavicius/{family_split_baselines_seed{0..4}.json, aggregate.json, naive_baseline.json}` |

---

## 5. Experiment: Pathogenicity positive control

Tests whether the same delta embeddings used in section 4 can distinguish pathogenic from benign
ClinVar variants on a different task. A passing control shows that the embeddings and probe
pipeline recover strong pathogenicity discrimination; it does not establish that mechanism
information is absent. Pass criterion: the seed-0 family-split `delta_mean` MLP AUROC CI excludes
0.85.

A small neural network (the MLP) is trained to look at a variant's embedding and guess whether it
is disease-causing or harmless. AUROC is a score from 0.5 to 1 measuring how well it separates the
two: 0.5 means no better than a coin flip, 1.0 means it always gets it right. The 0.85 threshold is
set in advance. The seed-0 family-split out-of-fold predictions supply the adjudicating point
estimate and family-bootstrap interval; the five-seed mean is descriptive.

This experiment uses its own ClinVar pull, separate from section 2's. Step 2.2 fetched
pathogenic variants only, to label genes by mechanism; step 2.8
(`fetch_pathogenicity_variants`) fetches benign variants too. It deduplicates identical
protein-level substitutions before selecting equal numbers of pathogenic and benign variants per
gene. Run step 2.8 before this experiment; both output files are required inputs.

The embedding step (GPU) runs on the pod; the probe step (CPU) runs locally after copying the
embeddings back. The `--phase` flag separates them so the bootstrap gets all your local cores
instead of burning GPU-hours on the pod.

| Step | Command | Description |
|---|---|---|
| 5.1 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> data/clinvar_pathogenicity_variants.json data/clinvar_pathogenicity_variants.params.json root@<pod-ip>:/workspace/repo/data/` | Copy pathogenicity variants and fetch metadata to the pod |
| 5.2 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> data/pfam_families.json root@<pod-ip>:/workspace/repo/data/` | Copy the current Pfam mapping to the pod |
| 5.3 | `python -m esm2_mech.experiments.pathogenicity.pathogenicity_control --phase embed --model esm2_t33_650M_UR50D --force_embed` | 🔴 GPU. Replace the pathogenicity embedding cache from the repaired fetched set. Inputs: `clinvar_pathogenicity_variants.json`, `clinvar_pathogenicity_variants.params.json`, `cache/sequences.json`. Outputs: `data/embeddings/esm2_t33_650M_UR50D/pathogenicity_{wt,mut}_mean.npy`, `data/embeddings/esm2_t33_650M_UR50D/pathogenicity_meta.json` |
| 5.4 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> root@<pod-ip>:/workspace/repo/data/embeddings/esm2_t33_650M_UR50D/pathogenicity_*.npy root@<pod-ip>:/workspace/repo/data/embeddings/esm2_t33_650M_UR50D/pathogenicity_meta.json data/embeddings/esm2_t33_650M_UR50D/` | Copy embeddings and metadata back to the local machine |
| 5.5 | `python -m esm2_mech.experiments.pathogenicity.pathogenicity_control --phase probe --seeds 5 --n_jobs <workers> --n_boot 1000` | 🟡 CPU — more cores help. Run the pathogenic-vs-benign probe on the validated embeddings. Inputs: `pathogenicity_{wt,mut}_mean.npy`, `pathogenicity_meta.json`, `clinvar_pathogenicity_variants.json`, `clinvar_pathogenicity_variants.params.json`, `pfam_families.json`. Outputs: `results/run_biorxiv/pathogenicity_control_seed{0..4}.json`, `results/run_biorxiv/pathogenicity_control.json` |

Step 5.3 uses `--force_embed` because this repaired run must replace the old embedding cache. The
embed and probe phases both validate the fetched-set metadata, the exact selected rows, the exact
sequence windows supplied to ESM-2, and the embedding-array fingerprint. A mismatch raises. The
`--phase` flag accepts `embed`, `probe`, or `both`.

Classes are balanced by construction (equal numbers of pathogenic and benign variants per gene).
However, genes still cluster into protein families, so family-split confidence intervals resample
whole Pfam families rather than genes or individual variants. The report should also note that the
probe measures discrimination between pathogenic and benign variants, not a calibrated risk
estimate for any single variant.

---

## 6. Experiment: Geometry of the pathogenicity direction

Section 5 shows the delta embeddings separate pathogenic from benign variants. This experiment
asks what that pathogenicity direction actually is: whether it is one shared direction across
protein families or many family-specific ones, whether it is more about how far a variant moves
the embedding or which way, and whether it is explained by simple substitution chemistry (e.g.
amino acid size or charge change) or by ESM-2's own sense of how conserved a position is. Separate
exploratory probes measure cross-family direction alignment and full-delta generalisation across
pathogenicity, stability, and mechanism tasks.


### Build canonical variant list (CPU)

Section 5's embedding step drops any variant it cannot embed (missing sequence, position out of
range, wildtype-residue mismatch), so the `.npy` embedding rows are a subset of the fetched variant
list, not a 1:1 match. This step is a re-indexing step: it takes the pathogenicity variants and embeddings 
Section 5 already produced and re-materializes them in a row-aligned file for the geometry scripts to read directly.

| Step | Command | Description | Inputs | Outputs |
|---|---|---|---|---|
| 6.1 | `python -m esm2_mech.experiments.geometry.build_canonical_pathogenicity` | 🟢 Light. Re-index the pathogenicity variant set to match the embedding row order | `clinvar_pathogenicity_variants.json`, `data/embeddings/esm2_t33_650M_UR50D/pathogenicity_{wt,mut}_mean.npy`, `pathogenicity_meta.json` | `pathogenicity_valid_variants_canonical.json` |

### Geometry probes (CPU)

One script runs the probes in sequence, each writing its own result file. `--probe` can restrict
this to a subset (e.g. `--probe magnitude geometry`); the default is all four.

| Probe | What it asks |
|---|---|
| magnitude | Does the pathogenicity signal come from how far the delta moves the embedding (magnitude), or which direction it moves in? |
| geometry | How much linear pathogenicity signal remains after fitted directions are removed in sequence, and how well do directions fitted on disjoint family halves align and transfer? |
| transfer | How do full-delta linear and gradient-boosted probes perform under group-disjoint cross-validation and when trained on one group half and scored on the other? The two scores are descriptive because their training-set sizes differ. |
| biochem | How strongly does a family-held-out pathogenicity axis associate with context-free substitution chemistry (BLOSUM62 score, and changes in hydropathy, charge, and volume), and how well does that chemistry predict the held-out axis score? |

| Step | Command | Description | Inputs | Outputs |
|---|---|---|---|---|
| 6.2 | `python -m esm2_mech.experiments.geometry.run_geometry --seeds 5 --stability-dataset tsuboyama` | 🟡 CPU — more cores help. Run all geometry probes, including the Tsuboyama stability arms | Canonical pathogenicity variants, `pathogenicity_meta.json`, pathogenicity WT/mutant mean embeddings, `valid_variants.json`, mechanism WT/mutant mean embeddings, `pfam_families.json`, `naive_baseline.json`, Tsuboyama variants, domain-family map, embedding fingerprint, and WT/mutant mean embeddings | `results/<run>/magnitude_direction/probe_results.json`, `geometry_results.json`, `transfer_contrast.json`, `probe4_axis_identity.json` |

Only the magnitude probe has cluster-bootstrap confidence intervals wired (`--no_ci` / `--n_boot`
apply to it only). The direction ablation, cross-family transfer, full-delta transfer, and
biochemistry analyses are exploratory summaries without bootstrap intervals. `--seeds` applies to
all probes.

### Conservation extract (GPU)

For each canonical pathogenicity variant, mask its wildtype position and read ESM-2's own predicted
probability of the wildtype residue, the mutant residue, and the entropy over all 20 amino acids at
that position — i.e., how confidently the model expects that position to be conserved. This is the
one GPU step in this experiment; it can share a pod session with any other GPU work already running.

| Step | Command | Description |
|---|---|---|
| 6.3 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> data/pathogenicity_valid_variants_canonical.json root@<pod-ip>:/workspace/repo/data/` | Copy canonical variants to pod |
| 6.4 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> data/cache/sequences.json root@<pod-ip>:/workspace/repo/data/cache/` | Copy sequences to pod |
| 6.5 | `python -m esm2_mech.experiments.geometry.conservation_axis --extract` | 🔴 GPU. Masked-LM forward pass per variant to score how conserved its position is. Inputs: `pathogenicity_valid_variants_canonical.json`, `cache/sequences.json`. Outputs: `data/conservation_pathogenicity.npy`, `data/conservation_pathogenicity_meta.json` |
| 6.6 | `scp -i ~/.ssh/id_runpod_2 -P <pod-port> root@<pod-ip>:/workspace/repo/data/conservation_pathogenicity.npy root@<pod-ip>:/workspace/repo/data/conservation_pathogenicity_meta.json data/` | Copy conservation outputs back to local machine |

### Conservation analysis (CPU)

Compares the conservation features from step 6.5 to the pathogenicity direction found in step 6.2, on
the same family-split protocol. Pre-registered claims: 2D conservation alone reaches AUROC ≥ 0.85
(the axis is mostly conservation); 2E adding the embedding delta on top of conservation improves
AUROC by ≥ 0.02 (the embedding carries pathogenicity signal beyond conservation).

| Step | Command | Description |
|---|---|---|
| 6.7 | `python -m esm2_mech.experiments.geometry.conservation_axis` | 🟡 CPU — more cores help. Compare conservation features to the embedding-derived pathogenicity direction. Inputs: `conservation_pathogenicity.npy`, `conservation_pathogenicity_meta.json`, `cache/sequences.json`, `pathogenicity_valid_variants_canonical.json`, `pathogenicity_meta.json`, `data/embeddings/esm2_t33_650M_UR50D/pathogenicity_{wt,mut}_mean.npy`. Output: `results/<run>/magnitude_direction/conservation_axis.json` |

run_biorxiv adds family-cluster confidence intervals to each pathogenicity AUROC in this experiment
(resampled over Pfam families, not genes or individual variants), and a paired cluster-bootstrap confidence
interval on the 2E gap (conservation-alone AUROC vs. conservation-plus-
embedding-delta AUROC). The descriptive correlations with the pathogenicity axis fit that axis
within each training-family fold and score the association only in the held-out families. Claims
2D and 2E use seed-0 held-out-fold point estimates and seed-0 family-bootstrap intervals. The five
seed fold means are saved and reported separately as descriptive results.

---

## 7. Experiment: Megascale stability positive control

A second positive control, alongside section 5, with a purely physical label instead of a
clinically curated one: Tsuboyama et al. 2023's measured folding stability change (ΔΔG) for about
177,000 single point mutations across about 181 natural protein domains. Because this label comes
from a direct physical measurement rather than expert curation, it rules out the concern that
section 5's pathogenicity signal is really the embeddings picking up on curation patterns rather
than biology.

The Megascale arrays are unchanged by the ClinVar refresh, but they are used only after precondition
0.10 has verified their extraction-time row identity and content. Step 7.2's 3C test also reads
`valid_variants.json` and the section 4 embeddings, so run step 7.2 after step 4.1 has produced a
current `valid_variants.json`.

### Assign Pfam families (CPU)

Assigns each Tsuboyama domain to a Pfam family via HMMER, so later steps can hold out whole families
rather than whole domains when testing whether the stability signal generalizes. Needs `hmmscan` on
the system path and a hmmpress-ed Pfam-A database; skip this step if `megascale_domain_families.json`
is already present and non-empty (it is, as of this writing).

| Step | Command | Description | Inputs | Outputs |
|---|---|---|---|---|
| 7.1 | `python -m esm2_mech.experiments.stability.build_domain_families` | 🟢 Light. Assign each Tsuboyama domain to a Pfam family | `megascale_tsuboyama_variants.json`, `downloads/megascale/Pfam-A.hmm` | `data/megascale_domain_families.json` |

### Validate or rebuild embeddings (GPU)

Precondition 0.10 writes extraction metadata for a checkpoint whose row-identity sidecar matches,
or rebuilds the arrays when that identity cannot be established.

### Linear probe (CPU)

Fits a Ridge regression from the embeddings to ΔΔG under three cross-validation schemes — random
split, holding out whole domains, and holding out whole Pfam families — and tests four pre-registered
hypotheses: 3A, the random-split correlation (Spearman ρ) reaches at least 0.5; 3B, that correlation
drops by no more than 0.10 when switching to a family-split (a big drop would mean the model is
recognizing domains rather than learning a general stability signal); 3C, projecting the fitted
stability direction out of section 4's mechanism-classification features does not raise the
family-split mechanism score by more than 0.01 (stability and mechanism should be separable); 3D, the
per-domain spread in correlation stays tight (standard deviation ≤ 0.10).

| Step | Command | Description | Inputs | Outputs |
|---|---|---|---|---|
| 7.2 | `python -m esm2_mech.experiments.stability.megascale_stability --n_jobs 4` | 🟡 CPU — more cores help. Ridge probe from embeddings to ΔΔG under random/domain/family splits | `megascale_tsuboyama_variants.json`, `megascale_domain_families.json`, `data/embeddings/esm2_t33_650M_UR50D/megascale_{wt,mut}_{mean,pos}.npy`, `valid_variants.json` | `results/<run>/megascale_stability/per_protein_spearman.json`, `stability_projection_3c.json`, `summary.json` |

`--n_jobs` is required, not optional: the per-seed, per-protein, and 3C loops each fork a worker
that standardizes and fits against most of the 177k×1280 embedding matrix, so an unbounded worker
count (`-1`) can exhaust RAM. Start at `--n_jobs 4`, watch peak RAM, and raise only if it fits.

### Nonlinear probe (GPU)

Repeats step 7.2's three-way split comparison with a small neural network (MLP), plus
an exploratory random forest, to check whether a
nonlinear model finds more signal than the linear probe, and whether any such gain survives the
family-split. Only the Ridge and MLP results are pre-registered; the random forest and XGBoost
numbers are exploratory. The default command runs the MLP and random forest. The random forest uses
cuML on a GPU when available and otherwise uses scikit-learn on the CPU. The separate `--xgboost`
command runs only XGBoost and needs a GPU.

| Step | Command | Description | Inputs | Outputs |
|---|---|---|---|---|
| 7.3 | `python -m esm2_mech.experiments.stability.megascale_mlp` | 🔴 GPU when available. Run the MLP and exploratory random forest on the three splits; the random forest falls back to scikit-learn on the CPU when cuML is unavailable | `megascale_tsuboyama_variants.json`, `megascale_domain_families.json`, `data/embeddings/esm2_t33_650M_UR50D/megascale_{wt,mut}_mean.npy` | `results/<run>/megascale_stability/mlp_summary.json` |
| 7.4 | `python -m esm2_mech.experiments.stability.megascale_mlp --xgboost` | 🔴 GPU. Run the exploratory XGBoost probe on the three splits | Same as step 7.3 | `results/<run>/megascale_stability/mlp_summary_xgb.json` |

### Controls (CPU)

Exploratory checks on the step 7.2 linear signal, not part of the pre-registered 3A–3D verdict:
whether a single feature (the size of the embedding shift, ignoring its direction) recovers most of
the full signal; the regularization strength chosen by nested cross-validation, so the main probe's
result isn't an artifact of one fixed setting; a label-shuffle null, where the ΔΔG values are
randomly permuted and the correlation should collapse to near zero as a leakage check; and how the
correlation changes as more embedding components are kept, to characterize how many dimensions the
stability signal actually occupies.

| Step | Command | Description | Inputs | Outputs |
|---|---|---|---|---|
| 7.5 | `python -m esm2_mech.experiments.stability.stability_baselines --n_jobs 4` | 🟢 Light. Delta-norm baseline, nested-CV alpha, label-shuffle null, and component sweep | `megascale_tsuboyama_variants.json`, `megascale_domain_families.json`, `data/embeddings/esm2_t33_650M_UR50D/megascale_{wt,mut}_{mean,pos}.npy` | `results/<run>/megascale_stability/baselines.json` |

`--n_jobs` is required here too, for the same reason as step 7.2 — its per-seed loops fork workers
against the full embedding matrix. Start at `--n_jobs 4`.

Gates are unchanged from run6: 3A random-split ρ ≥ 0.5, 3B the random-to-family-split drop stays
below 0.10, 3C the mechanism-F1 change from projecting out stability stays ≤ +0.01, 3D per-domain ρ
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
| 8.1 | `python -m esm2_mech.experiments.proteome_features.enzyme_classification --seeds 5` | 🟡 CPU — more cores help. Enzyme type classification: LogReg + MLP on WT embeddings and proteome features, gene-split and family-split, with cluster-bootstrap CIs | `valid_variants.json`, `embeddings_wt_mean.npy`, `enzyme_labels.tsv`, `pfam_families.json`, `proteome_features_aligned.npy`, `results/<run>/aggregate.json`, `results/<run>/mechanism_oof_cache_seed0.json` | `results/<run>/enzyme_classification/enzyme_classification_summary.json` |

The script accepts `--no_ci` (skip cluster-bootstrap CIs), `--n_boot N` (default 1000), and
`--n_permutations N` (OOF permutation test, default 0 = skip), matching sections 4–7. The mechanism
reference F1 and its family-split OOF predictions are read from section 4's outputs, not hardcoded.
Run section 4 first.

Decision rules (pre-registration §2F–2H):

- **2F** — family-split LogReg macro-F1 ≥ 0.70. Enzyme class is strongly encoded in ESM-2 WT
  embeddings and family-split CV is a meaningful discriminator.
- **2G** — enzyme family-split F1 substantially exceeds the mechanism family-split floor (read
  from section 4's aggregate, not hardcoded). The mechanism null is task-specific, not a probe or
  data failure.
- **2H** — MLP does not substantially outperform LogReg under family-split (|delta F1| < 0.05).
  Linear readout is sufficient, paralleling pathogenicity and contrasting with stability.

---

## Verification checklist

TBD
