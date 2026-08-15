# run_biorxiv progress

Live status record for `RUNBOOK_biorxiv.md`, which holds the steps only. Each section here mirrors
a section of the runbook, in the same order, with its own table in the style of `RUN_PROGRESS.md`.  

Every analysis script computes the confidence intervals. A confidence interval is a range around a score
saying how much that score would plausibly move if the same analysis were re-run on a different
but similarly-drawn sample of genes, rather than reporting a single number as if it were exact.

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
| 1 | `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 5` | `results/run_biorxiv/family_split_baselines_seed{0..4}.json`, `aggregate.json` | ✅ 2026-08-14 |
| 2 | `python -m esm2_mech.experiments.mechanism.mlp --seeds 5` | `results/run_biorxiv/nonlinear_results_seed{0..4}.json` | ✅ 2026-08-14 |
| 3 | `python -m esm2_mech.experiments.mechanism.family_clustering --seeds 5` | `results/run_biorxiv/family_clustering.json` | ✅ 2026-08-14 |
| 4 | `python -m esm2_mech.experiments.mechanism.naive_baseline` | `results/run_biorxiv/naive_baseline.json` | ✅ 2026-08-14 |
| 5 | `python -m esm2_mech.experiments.mechanism.leakage_fraction` | `results/run_biorxiv/leakage_fraction.json` | ✅ 2026-08-14 |
| 6 | `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 1 --n_permutations 1000` | `results/run_biorxiv/family_split_baselines_seed0_step2_permutation.json` (backed up under this name before Step 1's rerun overwrote the plain `seed0` file) | ✅ 2026-08-14 |
| 7 | `python -m esm2_mech.experiments.mechanism.single_source_mechanism --seeds 5` | `results/run_biorxiv/single_source_gerasimavicius/...` | ⬜ |




**Notes:**
Ran on a 32-core machine with settings to keep the statistics step from overloading it. 
1. We measure how good the macro-F1 score is. As a reference, if you ignore all the data and just always say “LOF” (the most common answer), you get a score of about 0.29. Anything close to that number has learned nothing.
   - `delta_mean` (mutation embedding only): 0.288 under both gene-split and family-split → pure chance, no signal.  
   - `wt_only_mean` (unmutated protein only): 0.560 under gene-split, drops to 0.470 under family-split. The 0.090 drop shows most of the original score came from recognising protein families, not from learning mechanism.

2. We also tested more flexible models (small neural net, gradient boosting, random forest, nearest-neighbour) on the same embeddings. The best of them, nearest-neighbour, scored 0.423. That is better than the 0.288 “always say LOF” baseline, but still far below the 0.560 that came from simply recognising protein families. We have not yet checked whether this 0.423 score also drops under family-split.  

3. We checked whether the embeddings group genes by protein family. They do. When we cluster genes into 5 groups using only the unmutated-protein embedding, the clusters match the true protein families 25.4% of the time. By pure chance this would happen only about 0.5% of the time — roughly 50 times higher than expected. This is why the wild-type-only feature looked strong under gene-split and why its score fell once whole families were held out.

4. The naive experiment. As a reference, if you ignore all the data and just always say “LOF” (the most common answer), you get a score of 0.288. This is true under both gene-split and family-split. Every other result is compared against this number.

5. For each feature we estimated how much of its gene-split score came from recognising protein families rather than from real mechanism signal. For the wild-type-based features that initially looked promising, about one-third of the score (33–36%) was family recognition. For the mutation-only feature (`delta_mean`) the calculation cannot be done — it never scored higher than the 0.288 “always say LOF” baseline, so there was no extra signal to explain.

6. We tested whether the family-split score is meaningfully better than chance by randomly shuffling the mechanism labels 1,000 times and seeing how often a shuffled result looks as good as the real one. For `wt_only_mean` (the feature that refits per shuffle), the real result beat all but one of the 1,000 shuffles — about as extreme a result as this test can report, strong evidence the family-split score is not a fluke. For `delta_mean` (the feature scored from cached predictions, not refit per shuffle), the real result looked unremarkable next to the shuffled ones, about as ordinary as a random shuffle — consistent with `delta_mean` carrying no real signal to begin with. We only ran this once, not once per seed, because a permutation test builds its own comparison baseline by shuffling, so repeating it across 5 seeds would mostly just re-measure the same seed-to-seed wobble the rest of this run already accounts for elsewhere. The timing test beforehand showed the full 1,000-shuffle run takes under 2.5 minutes on the 32-core machine, not the hours or days originally worried about. This run also had confidence intervals switched on (not skipped with `--no_ci`), so alongside the shuffle-test result, `wt_only_mean`'s family-split score also has a confidence interval recorded: macro-F1 0.502, with the true value likely somewhere between 0.404 and 0.554.

7. *(Pending)* We will re-run the main analysis using only the Gerasimavicius dataset (instead of the combined dataset) to check that the result is not caused by mixing two differently collected sources.

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
