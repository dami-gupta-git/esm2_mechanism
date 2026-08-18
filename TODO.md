# TODO — deferred fixes

Work that is understood and specified but deliberately not being done now. Each item says what is
wrong, what depends on it, and what has to happen before the affected code is trusted again.

---

## Fold-aware scoring in experiments outside the current runbook

**Status:** deferred. These experiments are not part of the `run_biorxiv` sequence and are not
being re-run, so their results are not blocking. Fix when one of them is next needed.

**The defect.** Ranking metrics are computed on probabilities concatenated across independently
fitted cross-validation folds. Each fold's model has its own probability scale, so ranking the
concatenation compares scores that were never comparable. The distortion grows as the underlying
signal weakens. Full diagnosis, evidence and the remediation order are in
[`biorxiv/exp4_issues.md`](biorxiv/issues/exp4_issues.md), issues 1 and 2.

**Why these files are affected.** They call the shared bootstrap and probe helpers, or carry their
own copy of the fold loop. The fix to the shared helpers changes their behaviour; the ones with
private copies will not pick the fix up at all.

| Script | How it is affected |
|---|---|
| `mechanism/clan_holdout.py` | calls the shared multiclass bootstrap |
| `mechanism/mmseqs_cluster_holdout.py` | calls the shared multiclass bootstrap |
| `mechanism/homology_partition_panel.py` | calls the shared multiclass bootstrap |
| `mechanism/mechanism_within_family.py` | shared multiclass bootstrap, plus the refit permutation path |
| `mechanism/contrastive_mechanism.py` | own fold loop, shared bootstrap, paired difference |
| `esm3/esm3_mechanism.py` | own fold loop, shared bootstrap, paired difference |
| `badonyi/badonyi_mechanism.py` | calls the shared multiclass bootstrap |
| `badonyi/badonyi_leakage_analysis.py` | calls the shared multiclass bootstrap |
| `badonyi/badonyi_holdout_survival.py` | calls the shared binary AUROC interval |
| `proteome_features/proteome_mechanism.py` | own fold loop, shared bootstrap |
| `proteome_features/per_gene_ablation.py` | own fold loop |
| `proteome_features/proteome_pilot.py` | own fold loop |

**What has to happen when one of these is next run.**

1. Confirm the shared helpers have been fixed first. If they have not, the result inherits the
   original defect.
2. If the script has its own fold loop, replace it with the shared helper rather than patching the
   loop in place. The private copies are why the two versions diverged.
3. Re-run rather than reusing any cached result under `results/`, and record the commit hash
   alongside the seed.
4. Any report quoting the old numbers needs regenerating, not editing.
5. Remove the strict expected-failure marks on the tests that cover the script. Two files
   carry them today, `tests/experiments/mechanism/test_homology_partition_panel.py` and
   `tests/experiments/mechanism/test_within_family_pooled.py`, on the tests that reach a
   helper without a fold index. The marks are strict, so a fixed script makes them pass
   unexpectedly and the suite fails until they are removed. That failure is the reminder;
   removing the mark is what restores the coverage.

**Decision to make before the shared fix lands.** The main multiclass bootstrap has roughly
fifteen call sites and the paired-difference and binary-AUROC helpers about ten more, most of them
in this list. Either update every call site in the same change, which is what the project
convention requires, or make the fold argument optional and have the helper refuse to compute
ranking metrics without it. The second is safer, because an optional argument that silently
defaults to the old behaviour is how this defect would survive its own fix.

---

## Reports carrying superseded numbers

**Status:** deferred alongside the above.

Reports under `reports/run_biorxiv/bak/` were produced by pre-fix code. They are not competing
results and should not be cited. Mark each superseded in its own text, the way
`reports/run6/report_esm3_mechanism_geras.md` does, rather than leaving them reading as finished
analyses.

Run 0 and run 6 reports quote the same pooled ranking metrics and are affected by the same defect.
They are historical records of what was believed at the time, so they should be annotated rather
than recomputed.
