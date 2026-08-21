# run_biorxiv — the current run

Re-scores run6's experiments with confidence intervals that account for genes in the same family
not being independent, permutation p-values, and real tests behind every "A beats B" claim. The
experiments, hypotheses and gates are unchanged.

`docs/README.md` and `docs/EXPERIMENT.md` index the run0-era exploratory phase; assume both are
stale. `docs/FINDINGS.md` is current and describes the statistics machinery this run uses, and the
runbook cites it as authoritative.

Each fact lives in exactly one of the files below. Where one needs something another states, it
references it by name rather than restating it, so a change lands in a single file.

| File | What it is |
|---|---|
| `PREREGISTRATION_run_biorxiv.md` | The claims under test (2A–2H), the decision rules, the resampling and pairing rules, what counts as passing, and what would overturn each claim. Its dated amendments identify rules recorded after earlier results were inspected. |
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

**Pathogenicity variant set:** run6 used 38,698 fetched / 37,218 embedded balanced ClinVar
variants over 1,937 genes, GRCh38. run_biorxiv refetches the set on a current snapshot, so these
counts are the baseline to compare against, not the run's own numbers. `pathogenicity_control.py`
fingerprints the set and refuses to run against non-matching embeddings. Run6: `delta_mean` MLP
family-split AUROC **0.894**, gene-split 0.897, std ≤ 0.001 across five seeds. This supersedes the
run0-era 0.74–0.88 band, whose width was two different variant sets across seeds rather than
sampling uncertainty — that band must not be cited.

**Every ClinVar-derived input is rebuilt from scratch (2026-08-11)** — the mechanism set, the
pathogenicity set, and the four arrays aligned to them. Only the megascale arrays are reused, which
makes Experiment 7 the one place a run6→run_biorxiv movement is attributable to the new statistics
alone. Paths are keyed by model, not by run.

**Outputs:** `results/run_biorxiv/` and `reports/run_biorxiv/`, both keyed off `RUN_NAME` in
`utils/paths.py`. run6 is preserved untouched as the comparison baseline.

`scripts/compare_runs.py run6 run_biorxiv` diffs every number and flags material movement. The
repaired result path computes ranking metrics inside each held-out fold. Confirmatory intervals use
seed-0 out-of-fold predictions, while five-seed means are retained as descriptive summaries.
