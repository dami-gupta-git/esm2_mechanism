# run_biorxiv — the current run

A fresh run of the study. The data is reacquired, the embeddings are regenerated, and every reported
number comes from this run's own result files. Nothing from run0, run1 or run6 is carried forward.

`docs/improve/ANALYSIS_PLAN.md` defines what the run measures: the questions, cohorts, models,
outcome metrics, planned comparisons, resampling rules, and reporting rules. It is the document that
governs the science. `docs/improve/REVISION_PLAN.md` and `docs/improve/audit.md` define the code
repairs that must land before the run starts.

Pre-registration has been withdrawn. `PREREGISTRATION_run_biorxiv.md` remains in this directory as a
record of the superseded submission and governs nothing. Its claim numbering (2A–2H, 3A–3D) and its
pass-or-fail gates do not apply to the fresh run, which reports effect estimates with confidence
intervals instead.

`docs/README.md` and `docs/EXPERIMENT.md` index the run0-era exploratory phase; assume both are
stale. `docs/FINDINGS.md` is current and describes the statistics machinery this run uses, and the
runbook cites it as authoritative.

Each fact lives in exactly one of the files below. Where one needs something another states, it
references it by name rather than restating it, so a change lands in a single file.

| File | What it is |
|---|---|
| `RUNBOOK_biorxiv.md` | Preconditions and the commands to run, in order, with the verification checklist. Steps only, with no status markers, dates, or result numbers. |
| `PROGRESS.md` | The live status record, with one table per runbook section in the same order and numbering. |
| `ENV_SNAPSHOT.md` | The package versions each cited result was computed under, per machine. Required by the runbook's verification checklist. |
| `DELTA_run6_to_run_biorxiv.md` | What moved between run6 and this run, and the cohort changes behind it. |
| `FOLLOWUP_biorxiv.md` | Deferred and withdrawn work. Gates nothing here. |
| `manuscript.md` | The paper. |
| `supplementary.md` | Supplementary methods and figures. |

`findings.md` was retired on 2026-08-20 and archived as `bak/findings_biorxiv_19Aug_2026.md`. Its
rules moved to the files that own them: pod operating rules to `docs/connect_runpod.md`,
result-file naming to `for_me/BUG_PATTERNS.md`, runbook and progress conventions to `CLAUDE.md`,
and report conventions to the project report skill.


## Canonical facts

**Every dataset is acquired as a new snapshot for this run**, with raw inputs, retrieval dates,
source identifiers, checksums, and exclusion counts retained. The pathogenicity variant set is
refetched on a current ClinVar snapshot; `pathogenicity_control.py` fingerprints the set and refuses
to run against non-matching embeddings. Embedding paths are keyed by model, not by run.

Counts and scores from earlier runs are not baselines for this one. No earlier number may be quoted
in a report or in the manuscript, and the run is not evaluated by how far it moved from run6.

**Outputs:** `results/run_biorxiv/` and `reports/run_biorxiv/`, both keyed off `RUN_NAME` in
`utils/paths.py`.

Metrics are computed inside each held-out fold and then aggregated. Point estimates use every
requested seed with equal weight; a seed-0 interval is not attached to or compared with a multi-seed
point estimate. The full aggregation and uncertainty rules are in
`docs/improve/ANALYSIS_PLAN.md`.
