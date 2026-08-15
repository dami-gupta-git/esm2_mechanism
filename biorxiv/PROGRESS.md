# run_biorxiv progress

Live status record for `RUNBOOK_biorxiv.md`, which holds the steps only. Each section here mirrors
a section of the runbook, in the same order, with its own table in the style of `RUN_PROGRESS.md`.

## Prerequisites — manually placed files

These are the source files the pipeline needs but cannot fetch itself; they must be placed in
`data/downloads/` before anything else runs.

| # | Item | Command | Outputs | Status | Notes |
|---|---|---|---|---|---|
| 1 | `DiseaseMech_Stability_VEPS.xlsx` | *(manually placed)* | `data/downloads/DiseaseMech_Stability_VEPS.xlsx` | ✅ 2026-08-14 | |
| 2 | `AllG2P.csv` | *(manually placed)* | `data/downloads/AllG2P.csv` | ✅ 2026-08-14 | |

## Stage 0 — preconditions

These are the checks and setup steps that must all pass before the run is allowed to start.

| # | Item | Command | Status | Notes |
|---|---|---|---|---|
| 1 | 0.0 Environment setup | `cd /Users/dgupta/code/portfolio/ESM2/esm2_mechanism`<br>`python3 -m venv .venv && source .venv/bin/activate`<br>`pip install -e .` | ✅ 2026-08-14 | |
| 2 | 0.1 Pathogenicity provenance | *(verification, no command)* | ✅ 2026-08-14 | Locked to one canonical variant set; `pathogenicity_control.py` fingerprints it |
| 3 | 0.2 Stats machinery wired | *(verification, no command)* | ✅ 2026-08-14 | Every result-producing script wired to `utils/bootstrap.py`, emits CI keys |
| 4 | 0.3 Methodology rules | *(verification, no command)* | ✅ 2026-08-14 | R7.3/R7.4 implemented |
| 5 | 0.4 Paired cluster bootstrap | *(verification, no command)* | ✅ 2026-08-14 | Wired at `conservation_axis.py`, `mechanism_delta_family_split.py` |
| 6 | 0.5/0.6 Pre-registered decision rules | *(verification, no command)* | ✅ 2026-08-14 | CI decision rule and confirmatory/exploratory split recorded |
| 7 | 0.7 Pinned environment | *(verification, no command)* | ✅ 2026-08-14 | `pytest tests/` green |
| 8 | 0.8 Configuration | *(config change in `utils/paths.py:11`)* | ✅ 2026-08-14 | `RUN_NAME` flipped `"run6"` → `"run_biorxiv"` |
| 9 | 0.9 Working tree clean | `git status` | ✅ 2026-08-14 | |

## Stage 1 — build gene list

Builds the list of genes every later experiment uses.

| #              | Command | Inputs | Outputs | Status | Notes |
|----------------|---|---|---|---|---|
| 1| `python -m esm2_mech.fetch_data.build_gene_list` | `downloads/DiseaseMech_Stability_VEPS.xlsx`, `downloads/AllG2P.csv` | `data/gene_list.tsv` | ✅ 2026-08-14 | |

## Stage 2 — fetch variant data

Shared foundation for Experiments 1, 2, 3, 5, and 7.

| # | Command | Outputs | Status | Notes |
|---|---|---|---|---|
| 1 | `python -m esm2_mech.fetch_data.fetch_variants --step gerasimavicius` | `gerasimavicius_variants.json` | ✅ 2026-08-11 | |
| 2 | `python -m esm2_mech.fetch_data.fetch_variants --step clinvar` | `clinvar_variants.tsv` | ✅ 2026-08-12 | |
| 3 | `python -m esm2_mech.fetch_data.fetch_variants --step merge --pathogenic_only` | `variants.json` | ✅ 2026-08-12 | |
| 4 | `python -m esm2_mech.fetch_data.fetch_sequences` | `cache/sequences.json` | ✅ 2026-08-12 | |
| 5 | `python -m esm2_mech.fetch_data.fetch_annotations --step pfam` | `pfam_families.json` | ✅ 2026-08-12 | |
| 6 | `python -m esm2_mech.fetch_data.fetch_alphamissense_mechanism` | `alphamissense_scores_full.json` | ✅ 2026-08-12 | |
| 7 | `python -m esm2_mech.fetch_data.build_valid_variants` | `valid_variants.json` | ✅ 2026-08-12 | |

**Results (2026-08-11/12 fetch):**

| Stage | Count |
|---|---|
| Gerasimavicius | 10,233 variants / 948 genes |
| ClinVar | 48,152 rows / 2,115 genes |
| Merged `variants.json` | 17,865 variants, 1,937 genes (gerasimavicius=10,233, clinvar_g2p=7,632) |
| Sequences fetched | 1,935 genes |
| Pfam | 1,913/1,937 genes annotated, 24 unannotated |
| AlphaMissense matched | 17,765 variants |
| `valid_variants.json` | 17,770 rows |

A WT-mismatch check on this fetch flagged 9 genes in the Gerasimavicius set — see
[`FINDINGS.md`](../docs/FINDINGS.md#wt-mismatch-check-flagged-9-genes-in-the-gerasimavicius-set-2026-08-12).

## Stage 3 — embed variants

Shared foundation for Experiment 1 and Experiment 3.

| # | Command | Outputs | Status | Notes |
|---|---|---|---|---|
| 1 | `python -m esm2_mech.embeddings.embed_variants --model esm2_t33_650M_UR50D` | `embeddings_{wt,mut}_{mean,pos}.npy`, `embedded_variants.json` | ✅ 2026-08-14 | Ran on pod, copied back locally. All four arrays and `embedded_variants.json` have 17,770 rows, matching `valid_variants.json`; spot-checked rows 0, 100, 5000, 17769 on gene/uniprot_id/position/wt/mut — all match |

## Experiment 1 — ESM-2 delta-embedding mechanism

Tests whether ESM-2 embeddings can predict a variant's mechanism (DN/LOF/GOF), and checks that the
model isn't just recognizing which protein family the gene belongs to rather than actually
learning something about the mechanism. This ran on a rented 32-core computer.

| # | Command | Outputs | Status |
|---|---|---|---|
| 1 | `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 5` | `results/<run>/family_split_baselines_seed{0..4}.json`, `aggregate.json` | ✅ 2026-08-14 |
| 2 | `python -m esm2_mech.experiments.mechanism.mlp --seeds 5` | `results/<run>/nonlinear_results_seed{0..4}.json` | ✅ 2026-08-14 |
| 3 | `python -m esm2_mech.experiments.mechanism.family_clustering --seeds 5` | `results/<run>/family_clustering.json` | ✅ 2026-08-14 |
| 4 | `python -m esm2_mech.experiments.mechanism.naive_baseline` | `results/<run>/naive_baseline.json` | ✅ 2026-08-14 |
| 5 | `python -m esm2_mech.experiments.mechanism.leakage_fraction` | `results/<run>/leakage_fraction.json` | ✅ 2026-08-14 |
| 6 | `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 1 --n_permutations 1000` | `results/<run>/...` | ⬜ |
| 7 | `python -m esm2_mech.experiments.mechanism.single_source_mechanism --seeds 5` | `results/<run>/single_source_gerasimavicius/...` | ⬜ |

**Notes:**

1. This ran on a rented 32-core computer, with settings added so the statistics step wouldn't overload the machine. The prediction score is macro-F1, a number from 0 to 1 where 1 is a perfect prediction and roughly 0.29 is what you'd get by always guessing the single most common mechanism (measured directly in note 4 below). The embeddings-only feature (`delta_mean`) scored 0.288 whether or not genes from the same protein family were kept separate between training and testing — indistinguishable from guessing, no signal. A different feature based on the unmutated protein sequence alone (`wt_only_mean`) scored 0.560 when family members could appear on both sides of the split, but dropped to 0.470 once they were kept separate. That 0.090-point drop shows most of the 0.560 score was the model recognizing which family a gene belongs to, not learning anything about the mechanism itself.
2. This tried several more flexible models (a small neural network, gradient boosting, random forest, nearest-neighbor) on the same embeddings, in case a simple model was missing a pattern a more complex one could find. The best of them (nearest-neighbor) scored 0.423 macro-F1 — better than the 0.288 floor, but still well below the 0.560 that the family-recognition-driven feature reached, and this is before checking whether it too collapses under family separation.
3. This checked directly whether the embeddings group genes by protein family. They do, strongly: grouping genes into 5 clusters by the unmutated-sequence embedding recovers each gene's true family 25.4% of the time, against roughly 0.5% expected if there were no such grouping at all — about 50 times higher than chance. This is the reason the 0.560 score in note 1 looked good before family separation was enforced, and why it dropped once that separation was applied.
4. This measured the score you would get by always guessing the single most common mechanism, with no model at all: 0.288 macro-F1 whether or not genes were split by family. That number is the reference line ("chance floor") every other score in this table is being compared against.
5. This calculated, for each feature, what fraction of its score before family separation was actually due to family recognition rather than real signal about the mechanism. For the features that looked promising at first (the unmutated-sequence-based ones), about a third of their score — 33% to 36% — was family recognition. For the embeddings-only feature (`delta_mean`), this couldn't even be calculated, because it never scored better than the 0.288 guessing floor to begin with, so there was no above-chance score to explain.
6. This will check whether the family-separated score is meaningfully different from what pure chance would produce, by repeatedly shuffling the mechanism labels and comparing 1,000 shuffled results against the real one.
7. This will re-run the main test on data from a single source (the Gerasimavicius dataset only), instead of the combined dataset used above, to check that the result isn't caused by combining two differently-collected datasets.

## Experiment 2 — pathogenicity positive control

Tests whether the same embeddings can at least tell pathogenic from benign variants, to confirm
they carry usable signal at all.

⬜ Not started.

## Experiment 3 — within-family mechanism

Tests mechanism prediction again, but restricted to genes inside one protein family, to see if any
signal survives once family membership can't be used as a shortcut.

⬜ Not started.

## Experiment 5 — geometry of the pathogenicity direction

Asks what the pathogenicity direction in embedding space actually corresponds to.

⬜ Not started.

## Experiment 7 — megascale stability positive control

A second positive control, using physical protein-stability measurements instead of clinical
labels, to confirm the embeddings carry signal independent of ClinVar curation.

⬜ Not started.

## Verification checklist

Final checks confirming the run's data and statistics are correct before its reports are written.

⬜ Not started.
