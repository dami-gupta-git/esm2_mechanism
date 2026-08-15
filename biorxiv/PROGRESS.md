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
| 8 | `python -m esm2_mech.fetch_data.fetch_pathogenicity_variants` | `clinvar_pathogenicity_variants.json`, `clinvar_pathogenicity_variants.params.json` | ✅ 2026-08-14 | Fetches balanced pathogenic/benign variants for Experiment 2, separate from step 2's pathogenic-only fetch used for mechanism labels; ran locally (network-only, no GPU needed) |

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
| 6 | `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 1 --n_permutations 1000` | `results/run_biorxiv/backup_step2_permutation_seed0.json` (renamed from `family_split_baselines_seed0_step2_permutation.json` — see note 8) | ✅ 2026-08-14 |
| 7 | `python -m esm2_mech.experiments.mechanism.single_source_mechanism --seeds 5` | `results/run_biorxiv/single_source_gerasimavicius/...` | ✅ 2026-08-15 |




**Notes:**
Ran on a 32-core machine with settings to keep the statistics step from overloading it. 
1. We measure how good the macro-F1 score is. As a reference, if you ignore all the data and just always say “LOF” (the most common answer), you get a score of about 0.29. Anything close to that number has learned nothing. These are 5-seed averages from the full rerun (re-aggregated after fixing a bug described below):
   - `delta_mean` (mutation embedding only): 0.288 under gene-split, 0.290 under family-split → pure chance, no signal.
   - `wt_only_mean` (unmutated protein only): 0.559 under gene-split, drops to 0.470 under family-split. The 0.089 drop shows most of the original score came from recognising protein families, not from learning mechanism.

2. We also tested more flexible models (small neural net, gradient boosting, random forest, nearest-neighbour) on the same embeddings, averaged across all 5 seeds. The best of them, nearest-neighbour, scored 0.414 under gene-split — better than the 0.288 “always say LOF” baseline, but still below the 0.559 that came from simply recognising protein families. Under family-split it drops to 0.357, the same pattern as `wt_only_mean`: part of its gene-split score was also family recognition, not real mechanism signal.

3. We checked whether the embeddings group genes by protein family. They do. When we cluster genes into 5 groups using only the unmutated-protein embedding, the clusters match the true protein families 25.4% of the time. By pure chance this would happen only about 0.5% of the time — roughly 50 times higher than expected. This is why the wild-type-only feature looked strong under gene-split and why its score fell once whole families were held out.

4. The naive experiment. As a reference, if you ignore all the data and just always say “LOF” (the most common answer), you get a score of 0.288. This is true under both gene-split and family-split. Every other result is compared against this number.

5. For each feature we estimated how much of its gene-split score came from recognising protein families rather than from real mechanism signal. For the wild-type-based features that initially looked promising, about one-third of the score was family recognition: 32.9% for `wt_only_mean`, 33.6% for `mut_only_mean`, 35.4% for `wt_concat_mut`. For the mutation-only feature (`delta_mean`) the calculation cannot be done — it never scored higher than the 0.288 “always say LOF” baseline, so there was no extra signal to explain.

6. We tested whether the family-split score is meaningfully better than chance by randomly shuffling the mechanism labels 1,000 times and seeing how often a shuffled result looks as good as the real one. For `wt_only_mean` (the feature that refits per shuffle), the real result beat all but one of the 1,000 shuffles — about as extreme a result as this test can report, strong evidence the family-split score is not a fluke. For `delta_mean` (the feature scored from cached predictions, not refit per shuffle), the real result looked unremarkable next to the shuffled ones, about as ordinary as a random shuffle — consistent with `delta_mean` carrying no real signal to begin with. We only ran this once, not once per seed, because a permutation test builds its own comparison baseline by shuffling, so repeating it across 5 seeds would mostly just re-measure the same seed-to-seed wobble the rest of this run already accounts for elsewhere. The timing test beforehand showed the full 1,000-shuffle run takes under 2.5 minutes on the 32-core machine, not the hours or days originally worried about. This run also had confidence intervals switched on (not skipped with `--no_ci`), so alongside the shuffle-test result, `wt_only_mean`'s family-split score also has a confidence interval recorded: macro-F1 0.502, with the true value likely somewhere between 0.404 and 0.554.

7. We re-ran the main analysis using only the Gerasimavicius dataset (instead of the combined ClinVar + Gerasimavicius dataset), to check that the results above are not caused by mixing two differently collected sources. On this subset the always-guess-LOF baseline is 0.279 (gene-split) / 0.280 (family-split) — close to, but not identical to, the 0.288 baseline from the combined dataset, because the mix of DN/GOF/LOF cases differs slightly in this smaller subset. The same two patterns from the combined dataset show up again here: `delta_mean` stays at that baseline on both splits (0.279 / 0.280 — still no signal), and `wt_only_mean` drops sharply between splits (0.611 gene-split → 0.456 family-split, a 0.155-point drop, with a confidence interval of roughly 0.40 to 0.58 on the family-split score). This confirms the mechanism result is not an artifact of combining two differently-collected datasets — it holds up on Gerasimavicius data alone.

## Experiment 2 — pathogenicity positive control

Tests whether the same embeddings can at least tell pathogenic from benign variants, to confirm
they carry usable signal at all. The embedding step needed a GPU, so it ran on a rented H100 pod.

| # | Command | Outputs | Status |
|---|---|---|---|
| 1 | `python -m esm2_mech.experiments.pathogenicity.pathogenicity_control --model esm2_t33_650M_UR50D` | `results/run_biorxiv/pathogenicity_control.json`, `results/run_biorxiv/pathogenicity_control_seed{0..4}.json`, `data/embeddings/esm2_t33_650M_UR50D/pathogenicity_{wt,mut}_mean.npy` | ✅ 2026-08-14 |

**Results:**

| Feature | Model | Split | AUROC (mean of 5 seeds) | 95% CI (from seed 0) |
|---|---|---|---|---|
| delta_mean | logistic regression | gene | 0.862 | [0.855, 0.867] |
| delta_mean | logistic regression | family | 0.861 | [0.856, 0.868] |
| delta_mean | small neural net | gene | 0.897 | [0.893, 0.902] |
| delta_mean | small neural net | family | 0.897 | [0.888, 0.899] |
| wt_only | logistic regression | gene | 0.572 | [0.569, 0.593] |
| wt_only | logistic regression | family | 0.555 | [0.539, 0.563] |
| wt_only | small neural net | gene | 0.615 | [0.604, 0.625] |
| wt_only | small neural net | family | 0.601 | [0.583, 0.609] |

**Notes:**

1. 37,258 embeddable variants (18,857 pathogenic, 18,401 benign, 1,925 genes), from a fetch separate from Stage 2's mechanism-labeling fetch since this one needed benign variants too.

2. Pass criterion: the mutation-embedding feature (`delta_mean`), scored with a small neural net, must reach 0.85 AUROC (0.5 = coin flip, 1.0 = perfect). It scored 0.897 on both gene-split and family-split — passes, with almost no drop between splits. That's the opposite of Experiment 1's mechanism result, where the wild-type feature's score fell sharply under family-split; here the signal holds up because it comes from the variant itself, not from recognizing the gene's protein family.

3. The wild-type-only feature scored far lower (0.615/0.601), as expected — the unmutated sequence alone says little about whether one specific mutation in it is pathogenic.

4. Since this control passes, Experiment 1's near-chance mechanism result reads as a genuine absence of signal for that task, not a broken pipeline.


## Experiment 5 — geometry of the pathogenicity direction

Asks what the pathogenicity direction in embedding space actually corresponds to.

| # | Command | Outputs | Status |
|---|---|---|---|
| 1 | `python -m esm2_mech.experiments.geometry.build_canonical_pathogenicity` | `data/pathogenicity_valid_variants_canonical.json` | ✅ 2026-08-15 |
| 2 | `python -m esm2_mech.experiments.geometry.run_geometry --seeds 5` | | ⬜ |
| 3 | `python -m esm2_mech.experiments.geometry.conservation_axis --extract` | | ⬜ |
| 4 | `python -m esm2_mech.experiments.geometry.conservation_axis` | | ⬜ |

**Notes:**

1. Step 1 re-indexes Experiment 2's pathogenicity variant list down to the 37,258 variants that were actually embedded, in the same row order as the embedding arrays, so later steps can read variant details and embeddings together without a separate lookup. Ran locally on CPU (no model, no GPU): 38,797 variants in, 37,258 written out, matching the embedding row count exactly.

## Experiment 7 — megascale stability positive control

A second positive control, using physical protein-stability measurements instead of clinical
labels, to confirm the embeddings carry signal independent of ClinVar curation.

⬜ Not started.

## Verification checklist

Final checks confirming the run's data and statistics are correct before its reports are written.

⬜ Not started.
