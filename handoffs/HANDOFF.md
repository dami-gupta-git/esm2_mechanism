---
name: handoff-biorxiv-reports
description: "Session handoff — wrote section 4 mechanism report, created report-writing skill, trimmed PROGRESS.md"
metadata:
  type: project
---

**Date**: 2026-08-17
**Working directory**: /Users/dgupta/code/portfolio/ESM2/esm2_mechanism

## Done
- Wrote `reports/run_biorxiv/report_mechanism.md` covering all of section 4 (mechanism experiment): linear probes, per-class AUROC, nonlinear probes, family clustering, leakage fractions, permutation tests, single-source robustness check.
- Replaced the lengthy notes in `biorxiv/PROGRESS.md` section 4 with a one-line summary linking to the report.
- Created a project-level report-writing skill at `.claude/skills/report.md` encoding the structure, style, and lessons learned (always state chance floors, per-class breakdowns, no em-dashes, plain-language metric definitions).
- User added "avoid em-dashes" to the skill file after reviewing it.

## Open threads
- The report still has em-dashes throughout; user asked for a rewrite to remove them but it was interrupted before starting.
- Section 5 (pathogenicity) results are done but have no dedicated report yet, only notes in PROGRESS.md.
- Section 6 (geometry) step 6.2 results exist on disk (`magnitude_direction/` subfolder) but PROGRESS.md still shows 6.2 as unchecked; steps 6.3-6.7 are pending.
- Section 7 (megascale stability) has not started (7.1-7.4 all pending).
- No reports written yet for sections 5, 6, or 7.

## Context a new session needs
- The user wants reports in the style of `reports/run0/result_1.md`: plain-language question, setup, headline results with verdicts, per-class breakdowns, interpretation, provenance.
- Every number must be verified against the result JSON files, not pulled from conversation context.
- User preference: always state the chance floor explicitly before interpreting a number against it; do not say "near chance" without defining what chance is for that metric.
