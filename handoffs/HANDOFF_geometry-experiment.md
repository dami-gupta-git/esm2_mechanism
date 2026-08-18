---
name: handoff-geometry-experiment
description: "Session handoff — ran Experiment 6 steps 6.3-6.7 and wrote report_geometry.md"
metadata:
  type: project
---

**Date**: 2026-08-17
**Working directory**: /Users/dgupta/code/portfolio/ESM2/esm2_mechanism

## Done
- Ran Experiment 6 steps 6.3-6.7 (conservation extract on RunPod RTX PRO 4000, conservation analysis locally), completing the geometry experiment.
- Updated PROGRESS.md to mark steps 6.2-6.7 as complete (6.2 was done on Aug 15 but not marked).
- Wrote `reports/run_biorxiv/report_geometry.md` following the report skill and using `reports/run6/report_geometry.md` and `reports/run0/result_23.md` as references.
- Report reviewed and refined: softened "axis is conservation" to "largely explained by conservation" (the +0.008 delta increment is statistically detectable but below the pre-registered +0.02 bar); changed "single axis" to "shared predictive direction" (ablation shows redundant encoding); fixed CI definition to avoid incorrect frequentist interpretation; fixed markdown pipe escaping in `||d||` table cells; corrected seed mismatch (seed 0 point estimates with seed 0 CIs throughout).
- Updated `.claude/skills/report.md` to point to `report_geometry.md` as the primary reference for future reports.
- RunPod connection: `ssh -i ~/.ssh/id_runpod_2 -p 30130 root@38.80.152.148` (the `id_ed25519` key does NOT work for this pod).

## Open threads
- Experiment 7 (megascale stability) is next in the runbook — all four steps (7.1-7.4) are not yet run; step 7.3 needs a GPU.
- The RunPod pod at 38.80.152.148:30130 may still be running and billable.
- Report may benefit from one more pass on editorial tone — user flagged "is the point" as editorializing; "only modestly" was acceptable.

## Context a new session needs
- See [[handoff-run-biorxiv-runbook]] for earlier context on the run_biorxiv pipeline.
- The conservation extract took about 2 hours on an RTX PRO 4000 for 37K variants — budget accordingly for similar masked-LM forward passes.
- When writing reports: use seed 0 point estimates with seed 0 CIs (never mix 5-seed means with seed-0 CIs); biochem probes only have seed std, not bootstrap CIs.
