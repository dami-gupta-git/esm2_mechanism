---
name: handoff-run-biorxiv-runbook
description: "Session handoff — filled in Experiment 5 and 7 in RUNBOOK_biorxiv.md, ran Experiment 5 Step 2, cleaned up PROGRESS.md and report_pathogenicity_control.md"
metadata:
  type: project
---

**Date**: 2026-08-17
**Working directory**: /Users/dgupta/code/portfolio/ESM2/esm2_mechanism

## Done
- Filled in `biorxiv/RUNBOOK_biorxiv.md`'s Experiment 5 (geometry of the pathogenicity direction) and Experiment 7 (megascale stability positive control) sections, which were previously "TBD" — written from the current source files in `src/esm2_mech/experiments/geometry/` and `.../stability/`, not from the older draft runbooks (`RUNBOOK_biorxiv_old.md`, `_original.md`), since those are out of date (e.g. `run_geometry.py` now bundles four probe scripts the old drafts didn't list).
- Ran Experiment 5 Step 2 (`python -m esm2_mech.experiments.geometry.run_geometry --seeds 5`) locally — completed successfully, wrote `results/run_biorxiv/magnitude_direction/{probe_results,geometry_results,transfer_contrast,probe4_axis_identity}.json`. Headline: pathogenicity direction is largely family-universal (cosine ~0.32 vs ~0 null, transfer AUROC 0.850); pathogenicity transfers across held-out families much better than mechanism (0.849–0.897 vs 0.620–0.636 AUROC); only ~7% of the axis is explained by context-free substitution biochemistry (R²=0.074).
- Reworded a confusing sentence in `RUNBOOK_biorxiv.md` about Experiment 2's class balancing and CI resampling unit, and added a missing detail to `reports/run_biorxiv/report_pathogenicity_control.md`'s Setup section noting the pathogenicity variant fetch is a separate earlier pipeline step (Stage 2 step 8), not part of this experiment's own step.
- Trimmed `biorxiv/PROGRESS.md` section 5 (pathogenicity positive control) down to match section 4's convention: once a report exists, a completed section keeps its step table but drops the full results table and numbered notes in favor of one summary sentence + a link to the report.
- Moved handoff files to live in the project repo itself (this file's location) instead of `~/.claude/projects/.../memory/` — see the updated `~/.claude/skills/handoff/SKILL.md` and `~/.claude/CLAUDE.md`.

## Open threads
- Experiment 5 Step 3 (conservation extract, GPU) and Step 4 (conservation analysis, CPU) not yet run — Step 3 needs a GPU pod.
- Experiment 1 Step 3 (`single_source_mechanism --seeds 5`) was completed since the last handoff (see `biorxiv/PROGRESS.md` row 4.7, ✅ 2026-08-15) — noting this in case [[handoff-biorxiv-reports]] still lists it as pending.
- Experiment 7 (megascale stability) is written into the runbook but not yet run.

## Context a new session needs
- `biorxiv/session.md` and `biorxiv/PROGRESS.md` are the live status sources; `RUNBOOK_biorxiv.md` holds steps only, no status markers — this convention was reinforced again this session.
- See [[handoff-biorxiv-reports]] for earlier context on this same run_biorxiv effort (also at repo root, `HANDOFF.md`).
