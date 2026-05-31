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
| `python -m esm2_mech.experiments.mechanism.classify_by_mechanism` | Gene-split vs family-split baseline comparison | `variants.json`, `cache/sequences.json`, `pfam_families.json`, `embeddings_*.npy`, `alphamissense_scores_full.json` | `results/<run_name>/family_split_baselines_seed{0..4}.json` |
| `python -m esm2_mech.experiments.mechanism.mlp --seed 0` | Nonlinear classifiers (MLP, GBM, RF, kNN) on delta embeddings | `valid_variants.json`, `pfam_families.json`, `embeddings_*.npy` | `results/<run_name>/nonlinear_results_seed0.json`                 |
| `python -m esm2_mech.experiments.mechanism.family_clustering` | Diagnostic: do ESM-2 embeddings cluster by Pfam family? (kNN purity, within/between distance, family probe, mechanism–family overlap) — explains the homology leakage in the WT-only baseline | `valid_variants.json`, `pfam_families.json`, `embeddings_wt_mean.npy`, `embeddings_mut_mean.npy` | `results/<run_name>/family_clustering.json` |
| `python -m esm2_mech.experiments.mechanism.naive_baseline` | Measured majority-class / stratified macro-F1 + AUROC floor (DummyClassifier, 5 seeds, same CV) — the chance reference for the other tables | `valid_variants.json`, `pfam_families.json` | `results/<run_name>/naive_baseline.json` |
| `python -m esm2_mech.experiments.mechanism.leakage_fraction` | Derived diagnostic: leakage fraction per feature = (gene − family macro-F1) / (gene − chance), the share of each feature's above-chance gene-split score that is homology leakage | `family_split_baselines_seed{0..4}.json`, `naive_baseline.json`, `family_clustering.json` | `results/<run_name>/leakage_fraction.json` |

The first two run on RunPod inside a `tmux` session; `scp` results back to `results/<run_name>/` locally. `family_clustering`, `naive_baseline`, and `leakage_fraction` are CPU-only and run locally. `leakage_fraction` reads only the result JSONs above (no model inference), so run it after `classify_by_mechanism`, `naive_baseline`, and `family_clustering`.

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

## Verification checklist

- [ ] `data/embeddings/esm2_t33_650M_UR50D/embedded_variants.json` row count matches all four `.npy` arrays. (This file is a write-only provenance artifact — no code reads it; it is the row-aligned variant index for the `.npy` arrays and should equal `data/valid_variants.json`. See `utils/embed.py` `_flush_checkpoint`.)
- [ ] `data/pfam_families.json` has entries for ≥ 1,900 genes (< 1,900 suggests a partial fetch).
- [ ] `data/enzyme_labels.tsv` spot-checked against UniProt EC numbers for a handful of kinases and proteases.
- [ ] `data/alphamissense_scores_full.json` non-empty and covers > 90% of `valid_variants.json`.
- [ ] result 6 (pathogenicity AUROC ~0.88) and result 7 (mechanism family-split floor ~0.35–0.39) reproduce their headline numbers — these are the pipeline spine and should not move.
- [ ] result 26: ESM-2 and ESM-3 scored under identical fold rule before reporting M1/M2 gap (see Stage E caveat).
- [ ] `git status` clean; results committed.
