# esm2_mechanism

Does ESM-2 encode disease mechanism?

A research project testing whether frozen ESM-2 650M delta-embeddings (mutant − wildtype) encode gene-level dominant disease mechanism class (GOF / DN / LOF) beyond protein stability, evaluated with gene-split and family-split cross-validation.

**Headline finding:** ESM-2 encodes pathogenicity strongly (delta MLP AUROC 0.88, family-split-stable) but mechanism weakly (family-split macro-F1 ~0.36–0.39). 62% of apparent gene-split mechanism signal is family-recognition leakage. GOF AUROC 0.63–0.80 is the strongest surviving signal under family-split.

See `docs/PUBLISH.md` for the full publication plan.

---

## Directory layout

```
data/
  raw/           — variant lists, gene lists, AlphaMissense scores, sequences (.json, .tsv, .csv, .xlsx)
  embeddings/    — ESM-2 .npy embedding arrays (gitignored — SCP from RunPod)

docs/
  README.md      — results index: reading order and headline numbers for all 7 results
  EXPERIMENT.md  — pre-registration: hypotheses, CV design, effect size thresholds
  PUBLISH.md     — versioned bioRxiv publication plan (v1/v2/v3)
  result_*.md    — per-experiment write-ups
  progress_notes.md — running log of decisions and bugs fixed

results/
  YYYYMMDD_<name>/run_N/  — JSON outputs from each experiment run

scripts/
  experiment.py          — main baseline pipeline (GPU required)
  experiment_mlp.py      — nonlinear probes on cached embeddings (CPU)
  family_clustering.py   — Pfam clustering diagnostic
  family_split_baselines.py — all baselines under gene-split and family-split CV
  pathogenicity_control.py  — positive control: ClinVar pathogenic vs benign
  build_merged_dataset.py   — merge Gerasimavicius + G2P/ClinVar variant sets
  extract_merged_embeddings.py — ESM-2 embeddings for merged dataset (GPU)
  mut_only_mlp.py        — MLP on raw WT/mut embeddings (not delta)
  fetch_clinvar_variants.py  — fetch ClinVar variants for G2P genes
  plot.py                — publication figures from result JSONs
  launch_scientist.py    — AI Scientist orchestrator (separate workflow)
```

---

## Quickstart

All scripts are run from the repo root. GPU required only for embedding extraction.

```bash
# Full baseline run (~2–2.5 hours on A100)
python scripts/experiment.py --out_dir results/run_0

# Nonlinear probes on cached embeddings (CPU)
python scripts/experiment_mlp.py \
  --emb_dir data/embeddings \
  --data_dir results/run_0/data \
  --out_dir results/run_0 \
  --family_split

# Plots
python scripts/plot.py results/run_0
```

For RunPod setup and remote execution, see `RUN_EXPERIMENTS.md`.

---

## Data

Raw data files are in `data/raw/` (committed where small enough) and embedding `.npy` arrays are in `data/embeddings/` (gitignored — too large for GitHub). See `data/README.md` for the full file inventory and how to transfer embeddings to/from RunPod.

Primary dataset: Gerasimavicius et al. 2022 (Nature Communications 13:3895). Local copy: `data/raw/DiseaseMech_Stability_VEPS.xlsx` (233MB, gitignored).

---

## Results

Read `docs/README.md` for the narrative arc across all 7 results. Result JSONs are in `results/20260524_baseline_run/run_0/`.
