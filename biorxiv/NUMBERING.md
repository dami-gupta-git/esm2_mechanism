# Numbering key

Old identifiers used in earlier drafts, mapped to the current numbering. This file exists so a
reference to an old ID can still be resolved; new writing should use the current ID only.

## Rules

| Old | Current | Title |
|---|---|---|
| R7.1 | Rule 1 | CI decision rule for gate verdicts |
| R7.2 | Rule 2 | Confirmatory / exploratory split |
| R7.3 | Rule 3 | Resampling unit |
| R7.4 | Rule 4 | Rare-class intervals |
| R7.5 | Rule 5 | Permutation budget |
| R7.6 | Rule 6 | Calibration |
| R7.7 | Rule 7 | What would change the conclusions |

## Gates (Rule 1's table)

| Old | Current | Criterion |
|---|---|---|
| K1 | 1A | conservation alone AUROC > 0.85 |
| K2 | 1B | conservation + delta improves over conservation by > 0.02 |
| K2b | 1C | conservation + delta improves over delta alone |
| H2 | 1D | stability random→family rho drop < 0.10 (LEAKY) |

## Confirmatory claims (Rule 2's table)

| Old | Current | Claim |
|---|---|---|
| C1 | 2A | mechanism delta sits at the measured chance floor under family-split |
| C2 | 2B | absolute-embedding gene→family gap is non-zero (homology leakage exists) |
| C3 | 2C | pathogenicity clears AUROC 0.85 family-split (positive control) |
| C4 | 2D | conservation alone matches or beats the embedding delta for pathogenicity |

## Scope

This mapping applies to the live documents only: `PREREGISTRATION_run_biorxiv.md`,
`RUNBOOK_biorxiv.md`, `PROGRESS.md`, `FOLLOWUP_biorxiv.md`, `README.md`, `new_ideas.md`.

Retired documents (`PLAN.md`, `PLAN_short.md`, `RUNBOOK_biorxiv_old.md`,
`RUNBOOK_biorxiv_original.md`) and every reference outside `biorxiv/` — including run0/run6 reports
and the source code that implements these gates — keep the old identifiers. They are frozen
records of what was written and run at the time; renumbering them would misrepresent that record.

Not covered by this mapping: `H1`, `H3`, `H4` in `RUNBOOK_biorxiv.md` section 7 are a separate,
experiment-local hypothesis set for the megascale stability probe. They were never part of the
pre-registration's Rule 1 gate table (only `H2`/`1D` was), so they are left as-is rather than
folded into this scheme.
