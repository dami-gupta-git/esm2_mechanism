# Producing the manuscript figures

Eight figures support the manuscript: six numbered main figures and two supplementary figures. A
single script produces all of them, writing PNG at 300 dpi, SVG, and PDF for each figure into
`reports/<run>/figures/`. That directory is keyed off `RUN_NAME` in `utils/paths.py`, the same
constant that selects the results directory the figures read from, so figures and results always
come from one run. There is no command-line override for the output location.

The script reads finished result files and recomputes nothing, so every experiment in the runbook
must have completed first. A missing input raises `Required figure input is missing` and aborts
rather than dropping a panel.

## Commands

    python -m esm2_mech.figures.manuscript_figures --figure all

`--figure` also accepts a single figure: `1` through `6` for the main figures and `S1` or `S2` for
the supplementary ones. The values are case-sensitive. Rerunning one figure overwrites only that
figure's three files.

## Inputs

Each figure draws on these results, all relative to `results/<run>/` unless stated otherwise.

| Figure | File stem | Result files read |
|---|---|---|
| 1 | `figure1_study_design` | `data/valid_variants.json` (cohort composition, not a result file) |
| 2 | `figure2_mechanism_delta` | `aggregate.json`, `magnitude_direction/probe_results.json` |
| 3 | `figure3_family_information` | `family_clustering.json`, `aggregate.json`, `leakage_fraction.json` |
| 4 | `figure4_pathogenicity_conservation` | `pathogenicity_control.json`, `magnitude_direction/probe_results.json`, `magnitude_direction/conservation_axis.json` |
| 5 | `figure5_enzyme_classification` | `enzyme_classification/enzyme_classification_summary.json` |
| 6 | `figure6_folding_stability` | `megascale_stability/summary.json`, `megascale_stability/mlp_summary.json`, `megascale_stability/mlp_summary_xgb.json`, `megascale_stability/per_protein_spearman.json` |
| S1 | `figureS1_single_source` | `aggregate.json`, `single_source_gerasimavicius/aggregate.json` |
| S2 | `figureS2_stability_direction_ablation` | `megascale_stability/stability_projection_3c.json` |

Figure 6 needs both stability probe summaries, so the stability MLP step must have been run twice,
once with `--xgboost` and once without.

## Dependencies

Figure rendering needs only `matplotlib` and `numpy` from the pinned environment; the heavier model
and probe dependencies are not loaded. The script forces the non-interactive `Agg` backend, embeds
TrueType fonts in the PDFs, and leaves SVG text as text rather than outlines, so an SVG viewer must
have DejaVu Sans available to render it as intended.

## A script not to run

`esm2_mech.experiments.mechanism.make_figures` writes a different and now unused set of portfolio
figures into the same directory, under different filenames, and reads at least one result file that
no longer exists. It is not part of this pipeline.
