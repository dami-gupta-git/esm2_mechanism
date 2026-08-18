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

## `homology_partition_panel.py` import broken

**Status:** deferred alongside the fold-aware scoring fix above — this script is already out of
scope until that lands, so this is a second, independent reason it will not run as-is.

**The defect.** It imports `load_pfam` from `mechanism/clan_holdout.py`, which no longer exists
there — `clan_holdout.py` was switched onto the shared `utils/data.load_pfam_map(PFAM_JSON)`
during the `load_pfam()` duplication cleanup, and the old wrapper was removed rather than kept as
a shim. `tests/experiments/mechanism/test_homology_partition_panel.py` currently
`pytest.importorskip`s the whole module for this reason.

**What has to happen when this script is next run.** Switch its import to
`utils/data.load_pfam_map(PFAM_JSON)` (the same fix already applied to `clan_holdout.py`), then
remove the `importorskip` guard from the test file — at that point the two pre-existing strict
`xfail` marks described above are what should still be gating the suite, not an import error.

---

## `load_pfam()` duplication outside the current runbook

**Status:** deferred. `duplicates.md` found the gene-to-Pfam-family JSON re-read (copy-pasted or
near-identical) in about 20 files. The ones under `experiments/mechanism/`, `experiments/
proteome_features/enzyme_classification.py` and `proteome_mechanism.py`, `experiments/geometry/`,
and `experiments/stability/megascale_stability.py` — the scripts the `run_biorxiv` runbook (section
4, 6, 7) actually runs — were switched onto the shared `utils/data.load_pfam_map(path)`. The rest
were left alone because they sit outside that runbook and touching them is unrelated churn until
they are next run.

**Files still reading the Pfam JSON directly**, by area:

| File | Note |
|---|---|
| `mechanism/mut_only_mlp.py` | not called from the runbook |
| `mechanism/mechanism_delta_probe.py` | not called from the runbook |
| `mechanism/multiseed_v1.py` | not called from the runbook |
| `mechanism/homology_partition_panel.py` | already broken (see the import-error item above); fix both together |
| `proteome_features/proteome_pilot.py` | already deferred above, own fold loop |
| `proteome_features/per_gene_ablation.py` | already deferred above, own fold loop |
| `esm3/esm3_mechanism.py` | separate ESM-3 pipeline, not `run_biorxiv` |
| `badonyi/badonyi_holdout_survival.py` | Badonyi pipeline, not `run_biorxiv` |
| `alphamissense/esm1v_family_split.py`, `alphamissense_family_split.py`, `proteingym_esm2_ll.py` | AlphaMissense/ESM1v comparison pipeline, not `run_biorxiv` |
| `perturbation/ll_scan.py`, `perturbation_probe.py`, `perturbation_pattern.py` | perturbation-scan pipeline, not `run_biorxiv` |

**What has to happen when one of these is next run.** Replace its direct `open(...); json.load(...)`
of the Pfam families file with `esm2_mech.utils.data.load_pfam_map(PFAM_JSON)` (or the file's local
`PFAM_FAMILIES` alias for that same path). Mechanical, one call site or a couple per file — same
change already made across the `run_biorxiv` scripts.

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
