# Run runbook — esm2_mech package

Purpose: run the full pipeline using the refactored `esm2_mech` package after the
2026-05 restructuring. All commands use `python -m esm2_mech.<module>` from the
project root with the package installed (`pip install -e .`). The old
`esm2_mechanism` package is now deleted; all scripts live under `src/esm2_mech/`.

**RunPod:** embedding extraction and analysis steps both run on RunPod (A100/H100). Run inside a `tmux` session. Fetch/data steps run locally on CPU.

**RunPod SSH:** connect with the `id_runpod_2` key:
```bash
ssh -i ~/.ssh/id_runpod_2 root@<pod-ip> -p <pod-port>
```
(`id_runpod` does NOT work — use `id_runpod_2`.)

---

## Prerequisites — manually placed files

These must be in `data/downloads/` before the pipeline starts.

| File | Source |
|---|---|
| `DiseaseMech_Stability_VEPS.xlsx` | Gerasimavicius et al. 2022 — OSF [10.17605/OSF.IO/H62FQ](https://osf.io/rct6d/download) |
| `AllG2P.csv` | G2P bulk download — gene2phenotype.org |
| `table_S3.xlsx` | Badonyi & Marsh 2024 — OSF [osf.io/download/7bftj/](https://osf.io/download/7bftj/) |
| `9606-WHOLE_ORGANISM-integrated.txt` | PaxDb v5.0 — pax-db.org (requires account) |
| `s_het_estimates.genebayes.tsv` | Zeng et al. 2023 — Zenodo [10.5281/zenodo.7939767](https://doi.org/10.5281/zenodo.7939767) |
| `gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz` | gnomAD release 2.1.1 |

---

## Stage 0 — environment setup

```bash
cd /Users/dgupta/code/portfolio/ESM2/esm2_mechanism
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

---

## Stage 1 — build gene list (CPU)

Shared foundation for all experiments. Run once before any experiment.

**Inputs (manually placed):**
- `data/downloads/DiseaseMech_Stability_VEPS.xlsx`
- `data/downloads/AllG2P.csv`

```bash
python -m esm2_mech.fetch_data.build_gene_list
```

**Output:** `data/gene_list.tsv`

---

## Experiment 1 — ESM-2 delta-embedding mechanism geometry (results 1–2)

### Step 1 — fetch data (CPU)

| Step | Command | Description | Inputs | Outputs |
|---|---|---|---|---|
| 2 | `python -m esm2_mech.fetch_data.fetch_variants --step gerasimavicius` | Fetch Gerasimavicius variants | `downloads/DiseaseMech_Stability_VEPS.xlsx` | `gerasimavicius_variants.json` |
| 3 | `python -m esm2_mech.fetch_data.fetch_variants --step clinvar` | Fetch ClinVar variants | `gene_list.tsv` | `clinvar_variants.tsv` |
| 4 | `python -m esm2_mech.fetch_data.fetch_variants --step merge --pathogenic_only` | Merge variant datasets (pathogenic only, drops likely pathogenic) | `gerasimavicius_variants.json`, `gene_list.tsv`, `clinvar_variants.tsv` | `variants.json` |
| 5 | `python -m esm2_mech.fetch_data.fetch_sequences` | Fetch UniProt sequences | `variants.json` | `cache/sequences.json` |
| 6 | `python -m esm2_mech.fetch_data.fetch_annotations --step pfam` | Fetch Pfam families | `variants.json` | `pfam_families.json` |
| 7 | `python -m esm2_mech.fetch_data.fetch_alphamissense_mechanism` | Fetch AlphaMissense scores | `variants.json` | `alphamissense_scores_full.json` |
| 8 | `python -m esm2_mech.fetch_data.build_valid_variants` | Build filtered variant list | `variants.json`, `cache/sequences.json` | `valid_variants.json` |

### Step 2 — embed variants (GPU, RunPod)

| Command | Description | Inputs | Outputs |
|---|---|---|---|
| `python -m esm2_mech.embeddings.embed_variants --model esm2_t33_650M_UR50D --batch_size 32` | Extract ESM-2 embeddings | `valid_variants.json`, `cache/sequences.json` | `embeddings_wt_mean.npy`, `embeddings_mut_mean.npy`, `embeddings_wt_pos.npy`, `embeddings_mut_pos.npy`, `embedded_variants.json` |

After completion, `scp` the `.npy` files and `embedded_variants.json` back to `data/embeddings/esm2_t33_650M_UR50D/` locally.

### Step 3 — run analysis (RunPod)

| Command | Description | Inputs | Outputs                                                 |
|---|---|---|---------------------------------------------------------|
| `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 5` | Gene-split vs family-split baseline comparison | `variants.json`, `cache/sequences.json`, `pfam_families.json`, `embeddings_*.npy`, `alphamissense_scores_full.json` | `results/<run_name>/family_split_baselines_seed{0..4}.json` |
| `python -m esm2_mech.experiments.mechanism.mlp --seeds 5` | Nonlinear classifiers (MLP, GBM, RF, kNN) on delta embeddings | `valid_variants.json`, `pfam_families.json`, `embeddings_*.npy` | `results/<run_name>/nonlinear_results_seed{0..4}.json`                 |
| `python -m esm2_mech.experiments.mechanism.family_clustering` | Diagnostic: do ESM-2 embeddings cluster by Pfam family? (kNN purity, within/between distance, family probe, mechanism–family overlap) — explains the homology leakage in the WT-only baseline | `valid_variants.json`, `pfam_families.json`, `embeddings_wt_mean.npy`, `embeddings_mut_mean.npy` | `results/<run_name>/family_clustering.json` |
| `python -m esm2_mech.experiments.mechanism.naive_baseline` | Measured majority-class / stratified macro-F1 + AUROC floor (DummyClassifier, 5 seeds, same CV) — the chance reference for the other tables | `valid_variants.json`, `pfam_families.json` | `results/<run_name>/naive_baseline.json` |
| `python -m esm2_mech.experiments.mechanism.leakage_fraction` | Derived diagnostic: leakage fraction per feature = (gene − family macro-F1) / (gene − chance), the share of each feature's above-chance gene-split score that is homology leakage | `family_split_baselines_seed{0..4}.json`, `naive_baseline.json`, `family_clustering.json` | `results/<run_name>/leakage_fraction.json` |

The first two run on RunPod inside a `tmux` session; `scp` results back to `results/<run_name>/` locally. `family_clustering`, `naive_baseline`, and `leakage_fraction` are CPU-only and run locally. `leakage_fraction` reads only the result JSONs above (no model inference), so run it after `classify_by_mechanism`, `naive_baseline`, and `family_clustering`.

The mechanism probe (`classify_by_mechanism` here, and the Step 4 `single_source_mechanism`) and `naive_baseline` now emit gene/family cluster-bootstrap 95% CIs by default — dependency-aware intervals that resample whole genes (or families), the label unit, rather than 5-seed fold jitter (see `reports/run6/STATS_PLAN.md`; machinery in `utils/bootstrap.py`). CIs add roughly a minute per seed and need no flag.

**`--seeds`:** every multi-seed command takes `--seeds N`, an integer **count** that runs seeds `0..N-1` and defaults to 5 (the single `N_SEEDS` constant in `utils/constants.py`). It is uniform across the runbook — `classify_by_mechanism`, `mlp`, `single_source_mechanism`, `mechanism_within_family`, `run_geometry`, `contrastive_mechanism`, and `esm3_mechanism --phase 3` all use the same flag with the same meaning. To run a subset, lower the count (`--seeds 1` runs seed 0 only).

`classify_by_mechanism`, `single_source_mechanism`, and the standalone `mechanism_delta_family_split` each accept three stats flags:

- `--no_ci` — skip the cluster-bootstrap CIs (faster).
- `--n_boot N` — bootstrap resamples (default 1000).
- `--n_permutations N` — label-permutation p-value against chance for the headline features (default 0 = off). Slow: it refits the probe once per permutation, so a full run multiplies the per-seed probe time by N — a candidate for joblib parallelism on a many-core pod. Off by default; turn it on deliberately when you want the p-value (e.g. `--n_permutations 1000`).

Example with the permutation test on: `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --n_permutations 1000`.

### Step 4 — single-source robustness check (CPU)

The merged dataset confounds mechanism class with curation source: the AR loss-of-function
subtype is entirely Gerasimavicius, HI is mostly Gene2Phenotype, while GOF and DN are
predominantly Gerasimavicius. A reviewer could argue any class-level difference (or its absence)
reflects source/curation rather than biology. This step removes that confound by re-running the
exact Step 3 gene-split vs family-split probe on the Gerasimavicius-only subset, which contains
all three classes from a single curation pipeline. It reuses `load_data()` and `run_family_split()`
unchanged — only the row set is filtered — and recomputes the majority-class floor on the subset
(the floor shifts because the subset's class balance differs from the merged set).

CPU-only — it reuses the existing Step 1/Step 2 inputs (no new fetch or embedding) and runs
locally; no GPU or RunPod needed.

| Command | Description | Inputs | Outputs |
|---|---|---|---|
| `python -m esm2_mech.experiments.mechanism.single_source_mechanism --seeds 5` | Re-run the Step 3 mechanism probe on the Gerasimavicius-only subset; recompute the subset majority-class floor; aggregate across 5 seeds and print the robustness read against merged Step 3 | `valid_variants.json`, `cache/sequences.json`, `pfam_families.json`, `embeddings_*.npy`, `alphamissense_scores_full.json` | `results/<run_name>/single_source_gerasimavicius/{family_split_baselines_seed{0..4}.json, aggregate.json, naive_baseline.json}` |

Args: `--n_folds` (default 5), `--seeds` (count, default 5 — see the `--seeds` note under Step 3), plus the same `--no_ci` / `--n_boot` / `--n_permutations` stats flags documented under Step 3. The null holds on the subset if
`delta_mean` sits at the subset floor on both splits and `wt_only`'s gene-split lift collapses
under family-split — confirming the mechanism null is not a source artifact.

---

## Experiment 2 — pathogenicity positive control (result_control)

Tests whether the same ESM-2 delta embeddings that classify mechanism at chance can predict
ClinVar pathogenic-vs-benign (published ESM-2 work: AUROC 0.88–0.94). If yes, the mechanism
null is a real absence of signal, not a broken pipeline; the gene-split / family-split gap
being ~0 shows the signal is per-variant biochemistry, not homology leakage.

One consolidated module runs all three phases (fetch → embed → 5-seed probe) in sequence.
Run on RunPod (Phase 2 needs GPU; the H200/A100 also has CPU for phases 1 and 3). Each phase
skips itself when its output is already present and matches by content.

| Command | Description | Inputs | Outputs |
|---|---|---|---|
| `python -m esm2_mech.experiments.pathogenicity.pathogenicity_control --model esm2_t33_650M_UR50D --batch_size 32` | Fetch balanced ClinVar P/B variants (GRCh38), extract ESM-2 WT+mut embeddings, run 5-seed logreg+MLP probes on delta_mean and wt_only × gene/family-split | `variants.json`, `cache/sequences.json`, `pfam_families.json` (ClinVar bulk file auto-downloaded) | `data/clinvar_pathogenicity_variants.json`, `pathogenicity_{wt,mut}_mean.npy`, `pathogenicity_meta.json`, `results/<run_name>/pathogenicity_control.json` |

Run inside a `tmux` session on RunPod. `scp` `pathogenicity_control.json` back to
`results/<run_name>/` locally. Headline: `delta_mean` MLP AUROC (pass threshold ≥ 0.85),
and the gene→family Δ (~0 expected). Report written as `reports/<run_name>/report_control.md`.

---

## Experiment 3 — within-family mechanism

Holds protein-family identity constant (so it cannot act as a leakage shortcut)
and asks whether ESM-2 embeddings can distinguish mechanism (GOF/DN/LOF) *within*
a single family. Per-family gene counts are tiny (6–16 genes), so results are
reported as mean ± std per family — `wt_only` vs `delta`, logreg and MLP, with a
per-family always-most-common-class baseline.

CPU-only — it reads the existing run6 `.npy` embeddings (no model inference) and
fits small per-family probes. Runs locally; no GPU or RunPod needed.

| Command | Description | Inputs | Outputs |
|---|---|---|---|
| `python -m esm2_mech.experiments.mechanism.mechanism_within_family --seeds 5` | Within-family gene-split CV per qualifying Pfam family (≥6 genes, ≥2 classes); wt_only vs delta × logreg/MLP; per-family macro-F1 + per-class AUROC (mean ± std) and majority-baseline F1 | `valid_variants.json`, `pfam_families.json`, `embeddings_wt_mean.npy`, `embeddings_mut_mean.npy` | `results/<run_name>/within_family_mechanism.json` |

---

## Experiment 4 — ESM-3 mechanism: scale and structure (report_esm3_mechanism)

Runs ESM-3 (`esm3-sm-open-v1`, 1.4B) on the same GOF/DN/LOF task in two conditions —
sequence-only (`seq`) and sequence + AlphaFold2 structure tokens (`seq_struct`) — to separate
the effect of model scale from explicit structure (the M3 test). Three phases, each takes
`--dataset {geras,merged}`; outputs go to per-dataset subdirectories so the two runs never
collide.

- **`geras`** — Gerasimavicius only (948 genes). The within-run M3 (seq vs seq_struct) result.
- **`merged`** — Gerasimavicius + G2P (1,935 genes). Matches the Experiment 1 ESM-2 classifier
  set exactly (same `valid_variants.json`, identical `label_3class`), so it is the
  apples-to-apples comparison for any scale claim against ESM-2. **Use merged for the headline.**

Phase 1 (CPU, network) and 3 (CPU) run locally or on the pod; phase 2 (GPU) runs on RunPod in a
`tmux` session. The phase-1 structure cache (`data/cache/esm3_struct_tokens.json`) is keyed by
UniProt ID and shared across datasets, so the merged run only fetches the proteins geras did not
already cover. Requires the ESM-3 SDK (`pip install esm==3.2.1.post1` — distinct from `fair-esm`)
and a HuggingFace token with the `esm3-sm-open-v1` licence accepted (`export HF_TOKEN=...`).

| Command | Description | Inputs | Outputs |
|---|---|---|---|
| `python -m esm2_mech.experiments.esm3.esm3_mechanism --phase 1 --dataset merged` | CPU: download AF2 structures from EBI, cache per-residue coordinates | `valid_variants.json` (merged) or `gerasimavicius_variants.json` (geras) | `data/cache/esm3_struct_tokens.json`, `data/cache/af2_structures/*.pdb` |
| `python -m esm2_mech.experiments.esm3.esm3_mechanism --phase 2 --dataset merged` | GPU: extract ESM-3 wt+mut mean-pooled embeddings for both conditions, save deltas + raw wt/mut arrays | `esm3_struct_tokens.json`, `cache/sequences.json`, variants | `data/embeddings/esm3-sm-open-v1/<dataset>/{seq,seq_struct}_mean.npy` (+ `_wt`/`_mut`), `valid_idx.npy`, `struct_meta.json` |
| `python -m esm2_mech.experiments.esm3.esm3_mechanism --phase 3 --dataset merged --seeds 5` | CPU: MLP + logistic probes, gene/family-split, 5 seeds; evaluate M1/M2/M3 | `{seq,seq_struct}_mean.npy`, `valid_idx.npy`, `pfam_families.json`, `results/<run_name>/nonlinear_results_seed{0..4}.json` (ESM-2 floor) | `results/<run_name>/esm3_mechanism/<dataset>/summary.json` |

Run phase 2 inside a `tmux` session on RunPod; `scp` the `<dataset>/` embedding and result
subdirectories back locally. Report written as `reports/<run_name>/report_esm3_mechanism.md`.

> **Comparison caveat:** the ESM-2 numbers in Experiment 1 are the merged set (17,826 variants).
> Compare ESM-3 against ESM-2 only on `--dataset merged`; the geras run is not a matched baseline
> for the ESM-2 classifier and must not be used for the scale claim.

---

## Experiment 5 — geometry of the pathogenicity direction

Decomposes the ESM-2 delta into magnitude (‖d‖) and direction (d/‖d‖) and asks where the
pathogenicity / mechanism signal lives. Run the steps in order: build the canonical variant list,
run the four CPU probes via the orchestrator, then the conservation decider (GPU extract → CPU
analysis). Run on RunPod (more cores; each seed dispatched in parallel), then `scp` the result
JSONs back to `results/<run_name>/magnitude_direction/`.

| Step | Command | Description | Inputs | Outputs |
|---|---|---|---|---|
| 1 build | `python -m esm2_mech.experiments.geometry.build_canonical_pathogenicity` | Materialise the row-aligned canonical pathogenicity variant set (fingerprint-checked against the embeddings); no GPU | `clinvar_pathogenicity_variants.json`, `pathogenicity_{wt,mut}_mean.npy`, `pathogenicity_meta.json` | `data/pathogenicity_valid_variants_canonical.json` |
| 2 probes (CPU) | `python -m esm2_mech.experiments.geometry.run_geometry --seeds 5` | Four CPU probes from one orchestrator (single `--seeds`): magnitude-vs-direction, rank/family-transfer geometry, transfer contrast (path/stability/mechanism), biochemical axis identity. `--probe …` runs a subset; stability rows skip if megascale S1724 embeddings absent | `pathogenicity_valid_variants_canonical.json`, `pathogenicity_{wt,mut}_mean.npy`, `valid_variants.json` + main `embeddings_*.npy` (mechanism), `pfam_families.json` | `results/<run_name>/magnitude_direction/{probe_results,geometry_results,transfer_contrast,probe4_axis_identity}.json` |
| 3 conservation extract (GPU) | `python -m esm2_mech.experiments.geometry.conservation_axis --extract` | Phase 1: mask each variant position, read masked-LM logP_wt/logP_mut/entropy | `pathogenicity_valid_variants_canonical.json`, `cache/sequences.json` | `data/conservation_pathogenicity.npy`, `..._meta.json` |
| 4 conservation analysis (CPU) | `python -m esm2_mech.experiments.geometry.conservation_axis` | Phase 2: is the pathogenicity axis just conservation? | `conservation_pathogenicity.npy`, `pathogenicity_{wt,mut}_mean.npy`, `pathogenicity_valid_variants_canonical.json`, `pfam_families.json` | `results/<run_name>/magnitude_direction/conservation_axis.json` |

---

## Experiment 6 — contrastive metric learning on the delta (report_contrastive)

Asks whether training the `delta_mean` feature to be family-invariant — a supervised
contrastive (triplet) objective whose only positive pairs are same-mechanism variants from
*different* Pfam families, with within-family pairs excluded — can surface cross-family
mechanism signal that the standard probes (Experiment 1) leave at the floor. A small projection
head (1280 → 256 → 64, TripletMarginLoss) is trained per fold, then variants are classified by
k-NN (k=10, cosine) in the learned space, against a raw-kNN baseline on the untrained delta.
Both gene-split and family-split CV, 5 seeds; the verdict reads the MLP `delta_mean` family
floor live from `aggregate.json` (never hardcoded).

CPU-light but GPU-resident: the feature matrix and triplet indices stay on the device, so a full
5-seed run is ~2 minutes on a recent GPU. Run on RunPod in a `tmux` session; `scp` the result
JSONs back to `results/<run_name>/`.

| Command | Description | Inputs | Outputs |
|---|---|---|---|
| `python -m esm2_mech.experiments.mechanism.contrastive_mechanism` | Train the cross-family contrastive head, evaluate k-NN vs raw-kNN baseline under gene/family-split, 5 seeds, then pool across seeds | `valid_variants.json`, `pfam_families.json`, `embeddings_wt_mean.npy`, `embeddings_mut_mean.npy`, `aggregate.json` (MLP floor) | `results/<run_name>/contrastive_results_seed{0..4}.json`, `contrastive_aggregate.json` |

Runs all `N_SEEDS` seeds (default 5) and writes the across-seed pool by default; `--seed N` runs a
single seed without aggregation. Headline: family-split `contrastive_knn` macro_f1 vs the raw-kNN baseline
and the MLP floor, and the gene→family drop (a smaller drop than the baseline's is the signature
of genuine cross-family signal rather than leakage). Report written as
`reports/<run_name>/report_contrastive.md`.

---

## Verification checklist

- [ ] `data/embeddings/esm2_t33_650M_UR50D/embedded_variants.json` row count matches all four `.npy` arrays. (This file is a write-only provenance artifact — no code reads it; it is the row-aligned variant index for the `.npy` arrays and should equal `data/valid_variants.json`. See `utils/embed.py` `_flush_checkpoint`.)
- [ ] `data/pfam_families.json` has entries for ≥ 1,900 genes (< 1,900 suggests a partial fetch).
- [ ] `data/enzyme_labels.tsv` spot-checked against UniProt EC numbers for a handful of kinases and proteases.
- [ ] `data/alphamissense_scores_full.json` non-empty and covers > 90% of `valid_variants.json`.
- [ ] result 6 (pathogenicity AUROC ~0.88) and result 7 (mechanism family-split floor ~0.35–0.39) reproduce their headline numbers — these are the pipeline spine and should not move.
- [ ] report_esm3_mechanism: ESM-3 compared against ESM-2 only on `--dataset merged` (matched 17,826-variant set), not geras (see Experiment 4 comparison caveat). Both scored under the identical fold rule.
- [ ] report_contrastive: family-split `contrastive_knn` macro_f1 clears the MLP `delta_mean` floor (read from `aggregate.json`), and its gene→family drop is no larger than the raw-kNN baseline's (else the lift is leakage, not cross-family signal).
- [ ] `git status` clean; results committed.
