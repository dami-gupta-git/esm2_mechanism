# Producing the experiment reports

Seven documents live in `reports/<run>/`. Five report the experiments, one holds plain-language
working notes, and one is a literature audit that supports how family-split results are framed.

| Report | Covers | Claims |
|---|---|---|
| `report_mechanism.md` | The delta-embedding mechanism experiment, including the permutation tests and the single-source check | 2A-1, 2A-2, 2B |
| `report_pathogenicity_control.md` | The pathogenicity positive control | 2C |
| `report_geometry.md` | The geometry of the pathogenicity direction | 2D, 2E |
| `report_stability.md` | The megascale stability positive control | 3A to 3D |
| `report_enzyme_classification.md` | The enzyme type classification positive control | 2F, 2G, 2H |
| `report_notes.md` | Plain-language working notes across all experiments | None |
| `report_FAMILY_SPLIT_LITERATURE_AUDIT.md` | Published family-split and homology-holdout results used for comparison | None |

## How they are written

The reports are written by hand from the result files. No script generates them. Each one follows
the fixed order used by the run6 reports: a summary, what was measured and why, glossary tables
explaining every row and column against a baseline or no-signal reference, the results tables, a
"Reading the tables" section that interprets specific cells in plain language, an interpretation
section stating what the result is and is not, and Provenance.

Every number in a report must come from a file under `results/<run>/` and must be cited in that
report's Provenance table, which also records the commit the result was produced at and the
execution logs under `logs/biorxiv/`. Write a report only after its experiment has finished; a
number quoted from an earlier run is the failure the runbook's verification checklist exists to
catch.

## Checking against the baseline run

Before writing, diff every number against the run6 baseline so that anything which moved is either
explained or investigated:

    python scripts/compare_runs.py run6 run_biorxiv --out biorxiv/DELTA_run6_to_run_biorxiv.md

The script compares every numeric and string leaf in every result JSON by its dotted path, judging
movement against the old run's own seed spread where the file carries one and falling back to an
absolute threshold otherwise. Keys present in only one run are reported separately from movement,
so the confidence-interval fields that run_biorxiv adds throughout do not bury the signal. Its
output is the source of `biorxiv/DELTA_run6_to_run_biorxiv.md`, which is generated from the run
rather than transcribed by hand.
