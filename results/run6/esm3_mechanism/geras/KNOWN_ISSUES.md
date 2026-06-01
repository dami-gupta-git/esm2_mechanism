# KNOWN ISSUES — geras ESM-3 run (do not cite as clean)

This `geras/` ESM-3 mechanism run contains **93 contaminated variants** and should not
be reported without this caveat. The matched, clean run is `../merged/`.

## The defect

Phase 2 built the mutant sequence by **blind overwrite** of the windowed wildtype
residue, without checking that the windowed reference residue actually matched the
variant's `aa_wt`. ESM-2's pipeline drops such variants via `apply_missense` (returns
`None` on a WT-reference mismatch); this ESM-3 run did not. As a result, **93 variants**
whose windowed reference residue did not equal `aa_wt` (wrong isoform / off-by-one) were
embedded on a **wrong wt/mut pair**, so their delta is meaningless. ESM-2 had already
dropped these, so this run's row set also diverges from the ESM-2 comparison set.

- Variants embedded here: 10,231 (includes the 93 bad rows)
- WT-mismatch rows that should have been dropped: 93
- OOB skips (correct): 2

The fix (use the shared `apply_missense` helper) is in `esm3_mechanism.py` as of the
commit that added this file; **re-running phase 2 + 3 with `--dataset geras` regenerates
these outputs cleanly.** This run predates the fix.

## What the numbers here are, and are not

- Headline as computed: seq family-split F1 = 0.421, seq_struct = 0.421 (M3 gap ≈ 0).
- The **direction** (structure tokens add nothing; seq ≈ seq_struct) is robust to 93
  rows out of 10,231 and is still informative.
- The **absolute values** are contaminated and must not be quoted as ESM-3's geras score,
  nor compared against the ESM-2 floor.

## Use the merged run instead

`results/run6/esm3_mechanism/merged/` is built from `valid_variants.json`, which has
already passed the same WT-reference filter (0 mismatches), so its row set is identical
to the ESM-2 classifier's. That is the apples-to-apples comparison and the basis for
`reports/run6/report_esm3_mechanism.md`.
