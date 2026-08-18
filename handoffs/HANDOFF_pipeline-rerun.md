---
name: handoff-pipeline-rerun
description: "Session handoff — fresh pipeline rerun on RunPod, section 4 in progress"
metadata:
  type: project
---

**Date**: 2026-08-17
**Working directory**: /Users/dgupta/code/portfolio/ESM2/esm2_mechanism

## Done
- Reset `biorxiv/PROGRESS.md` for a fresh rerun; old progress backed up as `biorxiv/PROGRESS_bak.md`.
- Verified all data files from sections 1-2 are present locally and marked as reused.
- Added inline compute tags (GPU / CPU-intensive / light) to every step in the runbook.
- Set up a 128-core H100 RunPod pod at `ssh -i ~/.ssh/id_runpod_2 root@38.80.152.148 -p 30320`: synced code, installed deps, copied all data and embedding files.
- Ran step 4.1 (`classify_by_mechanism --seeds 5`) on the pod; results and log copied back to local (`logs/step_4_1.log`), PROGRESS.md updated.

## Open threads
- Steps 4.2–4.7 still need to run (4.5 depends on 4.1+4.3+4.4; 4.6 is the expensive permutation test).
- Sections 3, 5, 6, 7, 8 are all still pending — section 3 embeddings may be reusable from the pod (already present there).
- The pod is live and ready; all data/embeddings are already on it.

## Context a new session needs
- Pod connection: `ssh -i ~/.ssh/id_runpod_2 root@38.80.152.148 -p 30320`, repo at `/workspace/repo`.
- The user wants stdout captured and saved to `logs/` for each step, and result files copied back to local immediately after each step finishes.
- Step 4.5 (leakage_fraction) must run after 4.1, 4.3, and 4.4 are all done.
