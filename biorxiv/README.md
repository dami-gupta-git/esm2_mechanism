# run_biorxiv — the current run

Re-scores run6's experiments with confidence intervals that account for genes in the same family
not being independent, permutation p-values, and real tests behind every "A beats B" claim. The
experiments, hypotheses and gates are unchanged.

Anything under `docs/` indexing the numbered `result_*.md` files is the run0-era exploratory phase.
Assume it is stale.

| File | What it is |
|---|---|
| `PREREGISTRATION_run_biorxiv.md` | The claims under test (C1–C5), what counts as passing, what would overturn each. **Governs** — where another document disagrees, this one is right. |
| `RUNBOOK_biorxiv.md` | Preconditions, commands in order, verification checklist. |
| `RUN_PROGRESS_biorxiv.md` | Live status. |
| `PLAN_biorxiv.md` | What changed since run6 and why. |
| `FOLLOWUP_biorxiv.md` | Deferred work. Gates nothing here. |

## Canonical facts

**Pathogenicity variant set:** 37,218 balanced ClinVar variants, 1,929 genes, GRCh38.
`pathogenicity_control.py` fingerprints it and refuses to run against non-matching embeddings.
Run6: `delta_mean` MLP family-split AUROC **0.894**, gene-split 0.897, std ≤ 0.001 across five
seeds. This supersedes the run0-era 0.74–0.88 band, whose width was two different variant sets
across seeds rather than sampling uncertainty — that band must not be cited.

**Embeddings are reused from run6**, not re-extracted; paths are keyed by model, not by run.

**Outputs:** `results/run_biorxiv/` and `reports/run_biorxiv/`, both keyed off `RUN_NAME` in
`utils/paths.py`. run6 is preserved untouched as the comparison baseline.

`scripts/compare_runs.py run6 run_biorxiv` diffs every number and flags material movement. Expect
K1 and K2 to move: conservation now reports one pooled AUROC over seed-averaged out-of-fold
predictions instead of a mean of per-fold AUROCs, since a paired difference needs both arms scored
on the same per-variant predictions.
