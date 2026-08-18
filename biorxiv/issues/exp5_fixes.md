# Experiment 5 — implementation brief

Work order for the defects diagnosed in [`exp5_issues.md`](exp5_issues.md). That document holds the
evidence and reasoning; this one holds the changes. Read issue 1 there before starting, since the
sequencing below is driven by the fact that the whole input chain has to be rebuilt.

This brief depends on [`exp4_fixes.md`](exp4_fixes.md). The confidence intervals in this section
come from the shared bootstrap helper being repaired there, so section 5 cannot be re-run until
that work has landed. Nothing here duplicates it.

**Do not adjust any number by hand.** Every value changes as a consequence of changing how it is
computed, or not at all.

---

## Before writing code

**Every code and specification change lands before anything is re-run.** The verdict logic, the
output metrics and the seed metadata all change what the probe writes, so re-running before they
are fixed means running the probes twice. The re-run is the last stage, not an interleaved one.

**Two specification decisions come first**, because both change what gets computed and both have to
be recorded as dated post-result specification amendments before repaired outputs are generated:

- The 2C seed-combination rule (step 1).
- Whether repeated protein substitutions are kept or deduplicated (step 2).

**The stale input chain starts at the fetch, not at the embeddings.** The fetched ClinVar file
predates per-gene balancing. Re-extracting embeddings from it would carry the unbalanced set
forward, so runbook step 2.8 has to run first.

**Rebuilding the inputs invalidates section 6.** Six geometry scripts and one mechanism follow-up
read the pathogenicity variant file or its embeddings, and claims 2D and 2E rest on two of them.
Section 6 must be re-run in full afterwards. Plan the pod session so both sections' GPU work
happens in one visit.

---

## Decisions already made

These were settled during review. Implement them; do not re-open them.

| Question | Decision |
|---|---|
| Repair the existing embedding cache by dropping rows | Rejected. The cache and the selection code must be provably the same generation; patching rows preserves exactly the ambiguity the staleness check exists to remove. |
| Re-extract from the existing fetched file | Rejected. That file predates balancing. Re-run the fetch first. |
| Staleness check severity | Hard error, not a warning, at both the fetch and the embedding stage. A stale artifact that prints a warning and continues is how this defect reached a result file. |
| Claim 2C adjudication | Family-split interval excluding 0.85, per the preregistration. The gene-split score remains a reported number and stops deciding the claim. |
| Order of work | All code and specification changes first; re-run last. |

Two questions are **open** and are the subject of steps 1 and 2. Because the initial results have
already been inspected, each decision must be recorded as a dated post-result specification
amendment before the repaired outputs are generated.

---

## Step 1 — decide and record the 2C seed-combination rule

**File:** `biorxiv/PREREGISTRATION_run_biorxiv.md`

The aggregate currently presents a five-seed average AUROC beside an interval computed from seed 0
alone, and the code comment attributes that to a preregistered convention. There is no such
convention for 2C. §1.2 specifies resampling units and pairing and is silent on seeds. Seed 0 is
specified explicitly for the mechanism permutation tests and for the enzyme claims 2F–2H, and
neither extends to 2C.

Decide the rule and write it into the 2C section as a dated post-result specification amendment
with its reason. The first results have already been inspected, so this does not restore
prospective preregistration; it resolves the omission transparently for the repaired run. Seed 0
alone is defensible and matches the neighbouring sections: each seed reshuffles the fold
assignment, so one seed's out-of-fold predictions are a coherent resampling unit. The choice has
to be recorded rather than inherited from a comment.

Whatever is chosen, the result file records the seed basis of the headline and of the interval
separately, so a reader who notices the two differ can see why.

Do this before writing code. It is the same class of gap as the open 2B adjudication in
`exp4_issues.md`, and it has to close the same way.

---

## Step 2 — decide and record the duplicate-substitution rule

**File:** `biorxiv/PREREGISTRATION_run_biorxiv.md`, then
`src/esm2_mech/fetch_data/fetch_pathogenicity_variants.py`

The scored set contains 219 distinct protein substitutions appearing more than once, 229 excess
rows in total, with no cross-label conflicts. ESM-2 returns an identical embedding for each
repeat, so those variants are weighted more than once in fitting and scoring.

Decide whether to keep or deduplicate, and record it under the same post-result amendment status as
step 1. Both are defensible: deduplication treats the protein substitution as the unit of
observation, retention keeps ClinVar's record structure.

If deduplication is chosen, it must run **before** balancing. Removing rows unevenly across classes
afterwards would break the balance the balancing step has just established.

Report the count in the outputs either way.

---

## Step 3 — make the fetch validate its cached parameters

**File:** `src/esm2_mech/fetch_data/fetch_pathogenicity_variants.py`

The fetch compares its cached parameter set to the current one and re-fetches on inequality. The
cached file on disk records only the per-gene cap, the seed and a source fingerprint; the current
code also writes a `balance_version`. The comparison therefore does detect this particular stale
file, but only as an incidental inequality — a missing key and a changed value are indistinguishable
to it, so the failure mode it reports is uninformative.

Running step 2.8 is the root correction: the existing inequality detects the missing
`balance_version` and triggers a re-fetch. Harden the diagnostic separately. A key the current code
writes that is absent from the cache raises a named staleness error identifying the key and the
artifact. A key present with a different value produces a separate, equally explicit diagnostic.
This hardening changes how stale inputs are identified, not the selection algorithm's correctness.

---

## Step 4 — make the embedding staleness check derive the expected variant set

**File:** `src/esm2_mech/experiments/pathogenicity/pathogenicity_control.py`

The probe phase rebuilds the variant list from the indices stored in the embedding metadata,
fingerprints that reconstruction, and compares it to the fingerprint stored beside them. That does
catch a variant file whose contents at those indices have changed. What it cannot catch is a change
to the selection algorithm while the underlying file stays put — it never asks what the current code
would select. That is precisely the change that landed on 17 August, and it is why a three-day-old
cache passed.

Derive the expected set: run the current filter and balance code over the current variant file,
fingerprint the result, and require the cache's stored fingerprint to equal it. The same derivation
already exists in the embedding phase, so factor it into one function both phases call rather than
writing it twice.

On mismatch, raise. The message must name which side is stale — a different count, a different
variant set at the same count, or a set the current code would no longer produce — and say to delete
the embedding files and re-extract. Do not fall back to the cache and do not offer to continue.

The existing row-count and row-alignment checks stay. They catch a different failure and neither
subsumes this one.

---

## Step 5 — assert per-gene balance in the probe phase

**File:** `src/esm2_mech/experiments/pathogenicity/pathogenicity_control.py`

Assert that every gene's pathogenic and benign counts are equal, and fail if any gene violates it.
Per-gene balance is the property that makes gene identity uninformative about the label, and it is
what the runbook, the preregistration and the report all assert. It should be enforced by the code
rather than asserted by the prose.

Record the realised balance in the result file: the gene count, the per-gene class count, and the
number of genes dropped for holding only one class.

**On the second rebalancing pass.** `_rebalance_after_filter` keeps the first *n* entries of the
larger class after the embedding filter has dropped variants unevenly. That is not an
order-dependent selection bias: the fetch already shuffles each gene-and-class list under the fetch
seed before writing it, so a prefix of that list is a random subset. Adding an explicit second draw
would improve encapsulation, by making the script's correctness independent of an upstream
guarantee, but it is optional and it corrects nothing.

### Verify before moving on

After balancing, scoring each variant by the pathogenic fraction of its own gene, using no
embedding, must give AUROC exactly 0.5 because every gene's fraction is 0.5 and the score is
constant. On the current data the in-sample oracle value is 0.752; a leave-one-out version that
removes each variant's label from its own group fraction is 0.682.

The same check by Pfam family must also give 0.5. A family is a collection of genes, and a
collection of exactly balanced genes is itself exactly balanced. On the current data the in-sample
oracle value is 0.710 and the leave-one-out value is 0.670.

Both are checks that the confound is gone. Neither is a statement about AUROC's chance level, which
is 0.5 regardless of class or group imbalance.

---

## Step 6 — adjudicate claim 2C on the preregistered rule and record the verdict

**File:** `src/esm2_mech/experiments/pathogenicity/pathogenicity_control.py`

The script decides pass or fail on the gene-split point estimate with no interval involved, and
prints the outcome without writing it anywhere. The preregistration requires the family-split
interval to exclude 0.85.

Adjudicate on the family-split interval using the shared adjudication helpers the other claim gates
already use, and write into the result file the verdict, the quantity it was computed from, the
split, the threshold and the seed basis from step 1. The three-way printed summary can stay as a log
convenience provided it reports the same rule.

Gene-split remains a reported number.

---

## Step 7 — correct the overstated claim at its source

**File:** `src/esm2_mech/experiments/pathogenicity/pathogenicity_control.py`

The script prints that a passing control shows the mechanism null is a real absence of signal rather
than a pipeline failure. A positive control rules out the pipeline explanation only. It cannot
establish that the delta embeddings hold no mechanism information, and section 4's own nonlinear
probes sit above the majority-class floor.

State what the control establishes: the embeddings and the probe pipeline recover strong signal on a
different task, so the weak mechanism result is not explained by a broken pipeline. The report
carries the same sentence and must be corrected in the same change, since regenerating the report
does not touch text originating in the source.

---

## Step 8 — carry the discarded detail and the accounting into the outputs

**Files:** `src/esm2_mech/utils/probes.py` (read only — the values already exist),
`src/esm2_mech/experiments/pathogenicity/pathogenicity_control.py`

The shared binary CV helper already computes, per fold, the area under the precision-recall curve,
the positive-class prevalence, and the positive and negative predictive values at the
prevalence-matched operating point. The script keeps only the mean AUROC. Carry the rest into the
per-seed files and aggregate them the same way, labelled as descriptive discrimination detail.

**Do not present them as calibration evidence.** They measure discrimination, not calibration, so
they do not support the preregistration's 2C statement that the probe is not a calibrated risk
estimate — that would need a calibration analysis of its own. The predictive values are further
limited on an artificially balanced case-control sample, where they reflect the sampling design
rather than any population. Keep the caveat in the report as a statement about what the probe is.

Separately, write the variant accounting into the embedding metadata and forward it into the result
file: the fetched count, the counts skipped by each filter reason, the counts removed by balancing
at each stage, the duplicate count and how it was handled, the number of single-class genes dropped,
and the final scored count. Record the ClinVar retrieval date, source URL and source-file checksum
with the fetched-set parameters. After the re-run the drop is larger than it is now, and a reader
comparing the fetched total to the scored total needs the breakdown to see the loss is intended.

---

## Step 9 — update call sites and tests in the same change

Project convention is that a shared contract change fixes every caller and test in the same commit.

The factored-out variant derivation from step 4 is called by both phases of this script. The metrics
and accounting added in step 8 change the shape of the per-seed result files, so anything reading
them must be updated with them.

Tests to update: `tests/experiments/pathogenicity/test_pathogenicity_control.py`,
`tests/fetch_data/test_fetch_pathogenicity_variants.py`.

Add three regression tests:

- A cached fetch parameter set missing a key the current code writes raises a staleness error naming
  the key. A count-only or equality-only check would not distinguish this from an ordinary change.
- A stale embedding cache is rejected. Build metadata whose stored variant set differs from what the
  current selection code produces *at the same row count and over an unchanged variant file*, and
  assert the probe phase raises. That is the defect in one assertion, and the existing checks do not
  catch it.
- The per-gene balance assertion fires on synthetic input with one unbalanced gene, and passes on
  balanced input.

---

## Re-run order

Only after every step above has landed.

1. Sections 1 through 4 as set out in `exp4_fixes.md`, since the shared helper must be repaired
   first.
2. Re-run the ClinVar fetch, runbook step 2.8.
3. Re-extract the pathogenicity embeddings on the pod.
4. Re-run section 5, five seeds.
5. Re-run section 6 in full. It reads the same embeddings and claims 2D and 2E rest on it.
6. Section 7, then section 8.

Record the commit hash alongside the seed in every result file.

Regenerate the reports rather than editing numbers in them. Mark
`reports/run_biorxiv/bak/report_pathogenicity_control.md` superseded in its own text.

---

## Acceptance criteria

- A fetch parameter set missing a key the current code writes raises a named staleness error.
- A stale embedding cache raises an error naming which side changed, including when the variant file
  itself is unchanged.
- Every gene in the scored set has equal pathogenic and benign counts, asserted by the code.
- Scoring by a variant's own gene's pathogenic fraction gives AUROC 0.5, and the same check by Pfam
  family also gives 0.5.
- Claim 2C's verdict appears in the result file, computed from the family-split interval against
  0.85, with its seed basis recorded.
- The seed-combination rule for 2C and the duplicate-substitution rule are both recorded as dated
  post-result specification amendments before repaired outputs are generated.
- The variant accounting in the result file reconciles the fetched count to the scored count.
- Neither the script nor the report claims the control establishes an absence of mechanism signal,
  and neither presents the precision-recall or predictive-value figures as calibration evidence.
