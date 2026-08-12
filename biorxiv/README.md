# run_biorxiv — the current run

Re-scores run6's experiments with confidence intervals that account for genes in the same family
not being independent, permutation p-values, and real tests behind every "A beats B" claim. The
experiments, hypotheses and gates are unchanged.

Anything under `docs/` indexing the numbered `result_*.md` files is the run0-era exploratory phase.
Assume it is stale.

Three documents, and each fact lives in exactly one of them. Where one needs something another
states, it references it by name rather than restating it, so a change lands in a single file.

| File | What it is |
|---|---|
| `PREREGISTRATION_run_biorxiv.md` | The claims under test (C1–C4), the decision rules, the resampling and pairing rules, what counts as passing, and what would overturn each claim. Frozen before the run. |
| `RUNBOOK_biorxiv.md` | What changed since run6, preconditions, the pinned environment, commands in order with live status, and the verification checklist. |
| `FOLLOWUP_biorxiv.md` | Deferred and withdrawn work. Gates nothing here. |

The run_biorxiv plan, the separate progress table, and the separate environment file are retired:
their live content is in the runbook and their rules are in the pre-registration.

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

`scripts/compare_runs.py run6 run_biorxiv` diffs every number and flags material movement. Expect
K1 and K2 to move: conservation now reports one pooled AUROC over seed-averaged out-of-fold
predictions instead of a mean of per-fold AUROCs, since a paired difference needs both arms scored
on the same per-variant predictions.
