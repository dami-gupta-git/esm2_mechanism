# Findings

Written 2026-08-19. Nothing is currently running in the background.

## Current state

Executing `RUNBOOK_biorxiv.md` for `run_biorxiv`. Sections 4 and 5 are done. Section 6 has step
6.1 done and step 6.2 results on disk but not yet recorded in `PROGRESS.md`. Sections 6.3-6.7
and all of section 7 are pending. Live status is tracked in `PROGRESS.md`, not in the runbook.

**Within-family mechanism (formerly Experiment 3) is dropped from `run_biorxiv` scope.** It was never
scoped beyond a placeholder, so it has been removed from `RUNBOOK_biorxiv.md` and moved to
`FOLLOWUP_biorxiv.md`. Do not resurrect it as a next action.

## Next action

Remove em-dashes from `reports/run_biorxiv/report_mechanism.md` (rewrite sentences to flow
naturally, not a straight find-and-replace). Then mark step 6.2 as done in `PROGRESS.md` and
continue with steps 6.3-6.7 (GPU pod needed for 6.3-6.6). Write reports for sections 5, 6, and 7
as each completes.

## Operational notes for future pod work

- Pods are ephemeral and have died mid-session before, losing uncopied results. Copy result files
  back to the local machine (`results/run_biorxiv/`) as soon as each script finishes, not at the
  end of a batch.
- Before running anything that computes bootstrap CIs or permutation tests on a pod, export
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` — otherwise joblib's per-worker BLAS
  calls oversubscribe the cores.
- Never name a backup/copy of a result file starting with the exact prefix of a real result file
  (e.g. `family_split_baselines_seed*`) — aggregation steps glob that prefix and will silently
  double-count it. Prefix backups with `backup_...` instead.
- Any cache-freshness check must compare everything that determines the cached content, not just
  the CLI args passed to that run — see `BUG_PATTERNS.md`, which already records this pattern.

## Section 4 WT windowing checks

Two internal sensitivity analyses found that variant-centered WT windows do not account for the
Section 4 gene-to-family gap. The short-protein subset retained 12,499 variants and supported a
positive gap in four of five seeds; on the full dataset, replacing variant-centered WT embeddings
with one observed-window average per UniProt protein left the Claim 2B decision unchanged at four
of five seeds, and every paired interval for the gene score, family score, and gap spanned zero.
These checks are retained under `results/run_biorxiv/wt_identity_short_proteins/` and
`results/run_biorxiv/wt_identity_window_average/` for reviewer questions and are not needed in the
main paper or as changes to the canonical Section 4 results.

## PROGRESS.md / RUNBOOK_biorxiv.md conventions (corrected on this session, don't relearn them)

- The two files are separate: the runbook holds steps only, no status markers, dates, or result
  numbers. `PROGRESS.md` is the live status record — one table per runbook section, in the same
  order and numbering, with a numbered "Notes:" list below each table (not a Notes column in the
  table itself — that was tried and rejected as too cramped).
- Every `<run>` placeholder in `PROGRESS.md` must say `run_biorxiv` explicitly, not `<run>`.
- Command cells in both files show the full runnable command, e.g.
  `python -m esm2_mech.experiments.mechanism.classify_by_mechanism --seeds 5`, never an abbreviated
  form.
- `PROGRESS.md` notes must be understandable to a non-statistician but still include the actual
  numbers (macro-F1, AUROC, p-values, CI bounds), explaining what each number means in plain words
  rather than omitting it. Example: "The score was 0.560, well above the 0.288 you'd get by always
  guessing the most common label" — not just "the score was good" and not just "0.560" with no
  context.
- Don't guess or fabricate numbers. If a result file's actual values haven't been read yet, say so
  plainly rather than writing a plausible-sounding placeholder.
- Write full sentences, not clipped dash-joined fragments, in both files.

## Report conventions

- Reports go in `reports/run_biorxiv/`, one per experiment section (e.g. `report_mechanism.md`).
  A project-level skill at `.claude/skills/report.md` codifies the structure and style.
- Detailed results and interpretation live in the report file. `PROGRESS.md` keeps only the step
  tables, status, and a one-line summary linking to the report.
- Always state the chance floor for a metric before interpreting any number against it. Do not
  write "near chance" or "at chance" without saying what chance is (0.288 for macro-F1, 0.5 for
  AUROC).
- Avoid em-dashes. Rewrite sentences to use commas, periods, or parentheses instead.
- Per-class breakdowns (e.g. per-class AUROC) should be included when the result files contain
  them, to show whether the overall result is driven by one class or consistent across all.
