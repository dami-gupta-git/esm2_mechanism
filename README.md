# esm2_mechanism

Does ESM-2 encode disease mechanism?

A research project testing whether frozen ESM-2 650M delta-embeddings (mutant − wildtype) encode gene-level dominant disease mechanism class (GOF / DN / LOF) beyond protein stability, evaluated with gene-split and family-split cross-validation.

**Headline finding:** ESM-2 encodes pathogenicity strongly (delta MLP AUROC 0.74–0.88, family-split-stable) but mechanism weakly (family-split macro-F1 ~0.30–0.39). 62.8% of apparent gene-split mechanism signal is family-recognition leakage. Gene-level proteome features (37-dim) outperform ESM-2 delta (1280-dim) by +0.10 F1. Badonyi 2024 structural priors (3 features) beat both. Project high-water mark: V2+bad macro-F1 = 0.511, DN AUROC = 0.827.

See `docs/README.md` for the full results narrative (23 results).

---

## Directory layout

```
data/
  *.json / *.tsv / *.csv    — variant lists, gene lists, scores, sequences
  *.npy                     — computed feature matrices (gitignored)
  embeddings/               — ESM-2 .npy arrays (gitignored — SCP from RunPod)
  cache/                    — downloaded third-party files (gitignored)
  megascale/                — Megascale stability dataset (gitignored)

docs/
  README.md          — results index: all 23 results with headline numbers and reading order
  EXPERIMENT.md      — pre-registration: hypotheses, CV design, effect size thresholds
  result_*.md        — per-experiment write-ups
  progress_notes.md  — running log of decisions and bugs fixed

results/
  <name>/            — JSON outputs from each experiment run

scripts/
  experiment.py                  — main baseline pipeline (GPU required)
  experiment_mlp.py              — nonlinear probes on cached embeddings (CPU)
  family_split_baselines.py      — all baselines under gene-split and family-split CV
  family_clustering.py           — Pfam clustering diagnostic
  pathogenicity_control.py       — positive control: ClinVar pathogenic vs benign
  pathogenicity_5seed.py         — 5-seed replication of pathogenicity control
  multiseed_v1.py                — 5-seed replication of V1 (ESM-2) under family-split
  build_merged_dataset.py        — merge Gerasimavicius + G2P/ClinVar variant sets
  extract_merged_embeddings.py   — ESM-2 embeddings for merged dataset (GPU required)
  build_proteome_features.py     — assemble 37-feature gene-level proteome matrix
  proteome_pilot.py              — 4-feature pilot (result 11)
  proteome_mechanism.py          — V1/V2/V3 modality comparison (result 13)
  per_gene_ablation.py           — per-gene scoring and feature ablation (result 13)
  clinical_utility.py            — clinical utility within ClinGen HI=3 (result 14)
  build_badonyi_features.py      — assemble Badonyi 2024 SVM features
  badonyi_mechanism.py           — V_bad/V2+bad modality comparison (result 15)
  badonyi_leakage_analysis.py    — leakage triage of V_bad (result 15 Appendix A)
  mmseqs_cluster_holdout.py      — MMseqs2-20 cluster-split holdout (result 15 Appendix B)
  within_family_mechanism.py     — within-family LOGO CV (result 16)
  badonyi_holdout_survival.py    — Badonyi raw-model holdout (result 16 addendum)
  contrastive_mechanism.py       — supervised contrastive projection head (result 9)
  clan_holdout.py                — leave-one-clan-out holdout (result 10)
  mut_only_mlp.py                — MLP on raw WT/mut embeddings (not delta)
  fetch_clinvar_variants.py      — fetch ClinVar variants for G2P genes
  fetch_alphamissense.py         — fetch AlphaMissense per-variant scores
  fetch_uniprot_sequences.py     — fetch UniProt sequences
  plot.py                        — publication figures from result JSONs
  utils_probes.py                — shared probe utilities
```

---

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

> `evo2` requires CUDA. Install `flash-attn` separately after the above:
> ```bash
> pip install flash-attn --no-build-isolation
> ```

---

## Quickstart

All scripts are run from the repo root. GPU required only for embedding extraction.

```bash
# Full baseline run (~2–2.5 hours on A100)
python scripts/experiment.py --out_dir results/run_0

# Nonlinear probes on cached embeddings (CPU)
python scripts/experiment_mlp.py \
  --emb_dir data/embeddings \
  --out_dir results/run_0 \
  --family_split

# Proteome modality comparison (CPU, 5-seed)
python scripts/proteome_mechanism.py --out_dir results/proteome_mechanism

# Badonyi modality comparison (CPU, 5-seed)
python scripts/badonyi_mechanism.py --out_dir results/badonyi_mechanism
```

For RunPod setup and remote execution, see `docs/connnect_runpod.md`.

---

## Data

Raw data files are in `data/` (committed where small enough). Embedding `.npy` arrays are in `data/embeddings/` and large third-party files are in `data/cache/` and `data/megascale/` — all gitignored (too large for GitHub, SCP from RunPod). See `data/README.md` for the full file inventory and transfer instructions.

Primary dataset: Gerasimavicius et al. 2022 (Nature Communications 13:3895). `data/DiseaseMech_Stability_VEPS.xlsx` (233 MB, gitignored).

---

## Results summary

| Arc | Results | Key finding |
|---|---|---|
| Frozen ESM-2 characterisation | 1–10 | Mechanism floor F1 = 0.39 (merged, 5-seed). 62.8% of gene-split signal is leakage. Pathogenicity AUROC 0.74–0.88 family-split-stable. |
| Gene-level proteome features | 11–14 | 37 features beat ESM-2 by +0.10 F1. Clinical utility reduces to paralog count alone (AUROC 0.746 within HI=3). |
| Badonyi structural priors | 15–16 | 3 Badonyi features beat ESM-2 and proteome. V2+bad = project high-water mark (F1 = 0.511, DN AUROC = 0.827). Within-family signal lives in residual proteome, not ESM-2. |

Read `docs/README.md` for the full narrative arc across all 23 results.
