# Experiment 5 — issues found in `results/run_biorxiv`

Review of runbook section 5 (pathogenicity positive control, steps 5.1–5.5) against
`pathogenicity_control.json` and its five per-seed files, the decision rule for claim 2C in
`PREREGISTRATION_run_biorxiv.md`, and the superseded report under
`reports/run_biorxiv/bak/report_pathogenicity_control.md`.

The headline result is strong. The delta-embedding MLP scores 0.897 under family-split against a
pass bar of 0.85, with an interval lower bound of 0.888. Nothing below suggests the control fails.
What the issues below establish is that the number cannot be published as it stands, because the
data it was computed on is not the data the experiment specifies, the interval attached to it was
computed by the method already condemned in [`exp4_issues.md`](exp4_issues.md), and the rule the
code used to adjudicate the claim is not the rule that was preregistered.

Dataset as run: 37,258 variants, 18,857 pathogenic, 18,401 benign, across 1,925 genes and 1,141
Pfam clusters.

Issues are ordered by severity. Each states the defect, the evidence, why it matters, and the
root-cause fix.

---

## 1. The probe ran on unbalanced data, because the whole input chain predates the balancing fix

**Defect.** This experiment is specified to draw equal numbers of pathogenic and benign variants
per gene. Balancing is done at the fetch step, in `fetch_pathogenicity_variants.py`, with a second
rebalancing pass in `pathogenicity_control.py` that repairs the imbalance introduced when the
embedding filter drops variants unevenly. Neither has ever run on the data that produced this
result.

**Evidence.** The stale chain starts at the fetched variant file, not at the embeddings. That file
is dated 14 August. Two commits changed the fetch code afterwards, `d1e0f2e` and `4fe87fe`, both on
17 August, and between them they added the per-gene balancing to the fetch and a `balance_version`
marker whose stated purpose is to invalidate caches built before it. The parameter file beside the
fetched variants, `clinvar_pathogenicity_variants.params.json`, records only the per-gene cap, the
seed and a source fingerprint. It has no `balance_version` key, which confirms it was written by
the pre-balancing code. The embeddings were built from that file on 14 August, and the probe ran on
18 August.

Running the current balance check against the variant list stored in the embedding metadata:

| Quantity | Value |
|---|---|
| Genes in the scored set | 1,925 |
| Genes with equal pathogenic and benign counts | 265 |
| Genes with unequal counts | 1,660 |
| Genes carrying only one class | 119 |

The class totals in the result file, 18,857 against 18,401, are themselves the signature. Per-gene
balancing forces the two totals to be identical, so any difference between them means the step did
not run.

**Why it matters.** Per-gene balance is what makes this a clean test of the mutation signal. When
every gene contributes equally to both classes, knowing which gene a variant sits in tells you
nothing about its label, so no part of a probe's score can come from recognising disease genes.
That property does not hold in the data as run. An oracle score built from each gene's pathogenic
fraction over the full dataset reaches AUROC 0.752, and the corresponding Pfam-family score reaches
0.710. Each variant contributes its own label to those fractions, so these are in-sample upper
bounds rather than estimates of held-out performance. Leaving each variant's label out of its own
group fraction lowers the values to 0.682 for genes and 0.670 for families.

None of these figures is a correction to subtract from 0.897. They show that gene and family
prevalence are associated with the labels in the data as run. The balanced design removes that
association by construction. Its contribution to the reported probe score is unknown and cannot
be determined without re-running.

Claim 2C is the positive control that licenses the central dissociation in the paper. It should
rest on the design that removes the confound rather than on an argument about how much of the
confound survived.

**The enabler.** The embedding staleness check does catch some changes and cannot catch this one.
It rebuilds the variant list from the indices stored in the cache's own metadata and fingerprints
that reconstruction against the fingerprint stored beside them, which detects a variant file whose
contents at those indices have changed. What it cannot detect is a change to the selection
algorithm while the underlying variant file stays put, because it never derives what the current
code would select. That is exactly the change that landed on 17 August, which is why a three-day-old
cache built by superseded selection logic passed silently.

**Fix.** Three parts, in this order.

Re-run the ClinVar fetch — runbook step 2.8. The file on disk predates balancing, so re-extracting
embeddings from it would carry the same unbalanced set forward. The fetch's own cache check
compares a stored parameter set that now includes the balance marker, and the stored set lacks the
key, so it will detect the stale file and re-download. It was simply never run again after the fix
landed.

Running step 2.8 is the root correction. The current parameter comparison already detects the
missing `balance_version` and triggers a re-fetch. Separately, harden its provenance reporting: a
cached parameter set missing a key the current code writes should raise a named staleness error
rather than being reported as an undifferentiated parameter inequality. This changes how the stale
artifact is diagnosed, not whether the current selection logic is correct.

Make the embedding staleness check derive the expected variant set: run the current filter and
balance code over the current variant file, fingerprint that result, and require the cache's stored
fingerprint to equal it. A mismatch must be an error naming which side changed, not a warning. Then
delete the pathogenicity embeddings, re-extract, and re-run the probe. Do not rebalance after the
fact by dropping rows from the existing embedding matrices — the cache and the selection code must
be provably the same generation.

---

## 2. The confidence intervals use the pooled-ranking method condemned for section 4

**Defect.** The intervals attached to every AUROC in this section are produced by
`binary_auroc_cluster_bootstrap_ci`, which is one of the four functions identified in
[`exp4_issues.md`](exp4_issues.md) issue 2 as ranking a concatenation of out-of-fold probabilities
from five independently fitted models. Each fold's model has its own probability scale, so ranking
across the concatenation compares scores that were never on a common scale.

**Why it matters.** The headline AUROC in this section is a per-fold average and the interval
beside it is computed on the pooled concatenation, so the two are different quantities. That is the
same mismatch documented for the mechanism experiment, where the reported delta AUROC and its
interval did not bracket each other.

The adjudicating quantity for 2C is not borderline. Claim 2C is judged on the MLP under
family-split, whose interval lower bound is 0.888 against a bar of 0.85. The distortion is expected
to be small here in any case, because pooling hurts most when within-fold discrimination is weak
and this probe discriminates strongly. The fix is mandatory because the reported quantity and its
interval must be the same thing, not because the verdict is at risk.

**Fix.** No separate work. The shared helper is already being repaired for section 4. Re-run this
section against the fixed helper.

---

## 3. The script's automatic verdict uses the wrong split and is never recorded

**Defect.** The preregistration adjudicates claim 2C on the family-split confidence interval
excluding 0.85. The script decides pass or fail on the gene-split point estimate, with no interval
involved, and prints the outcome to the log without writing it to any result file.

**Why it matters.** Two separate problems.

The rule the code applies is not the preregistered rule. It reaches the same conclusion in this
run, because both splits clear the bar comfortably, but a run where the two splits diverge would
produce a verdict that contradicts the preregistration while looking authoritative in the log.
Gene-split is the more permissive of the two by design, so the code's rule is systematically easier
to pass than the one that was registered.

Because the verdict exists only as printed text, no result file records how 2C was adjudicated.
Anyone reconstructing the paper's claims from the outputs has to re-derive it, and the record of
which rule was applied is lost as soon as the log scrolls.

**Fix.** Adjudicate on the family-split interval, using the shared adjudication helpers that the
other claim gates already use, and write the verdict, the quantity it was computed from, the split,
the threshold and the seed basis into the result file. The gene-split score stays in the output as
a reported number; it stops deciding the claim.

---

## 4. The seed basis for the 2C interval is undeclared, and the preregistration does not supply one

**Defect.** The aggregate presents a five-seed average AUROC next to an interval computed from
seed 0 alone. For the delta MLP under family-split the average is 0.8967 and the interval's own
point estimate is 0.8939. Nothing in the result file marks them as different bases.

The code comment justifying the choice cites a preregistered seed-0 convention. There is no such
convention for this claim. §1.2 of the preregistration specifies resampling units and pairing and
says nothing about seeds. Seed 0 is specified explicitly in two other places — the permutation
tests in the mechanism section, and the enzyme claims 2F–2H — and in neither case does the text
extend to 2C. The 2C section itself is silent on how the five seeds combine.

**Why it matters.** This is the same failure diagnosed for claim 2B in issue 12 of
[`exp4_issues.md`](exp4_issues.md): a preregistered rule that leaves a degree of freedom open, so
whatever basis is used will have been chosen after the numbers were seen. Here the open choice is
the seed-combination rule, and the code has silently resolved it while attributing the resolution
to a document that does not contain it.

The practical consequence in this run is small, since the two numbers differ in the third decimal
and the verdict is the same either way. The specification problem is not small, because the same
gap will be there on the next run.

**Fix.** Decide the seed-combination rule for 2C and record it as a dated post-result specification
amendment before re-running. The first results have already been inspected, so the amendment does
not restore prospective preregistration; it resolves an omission transparently for the repaired
run. Seed 0 alone is a defensible choice and matches what the neighbouring sections do: each seed
reshuffles the fold assignment, so a single seed's out-of-fold predictions are a coherent
resampling unit. The choice has to be written down rather than inherited from a comment. Whatever
is chosen, the result file must record the seed basis of the headline and of the interval
separately.

---

## 5. The runbook, the preregistration and the report state a property the data does not have

**Defect.** The runbook states that classes are balanced by construction, with equal numbers of
pathogenic and benign variants per gene. The preregistration repeats it in the 2C section. The
superseded report describes the dataset as 37,258 balanced variants.

**Why it matters.** All three describe the design rather than the run, and issue 1 shows the run
did not match the design.

The report's accompanying statement that the chance floor is 0.50 regardless of class balance is
correct and should not be changed. AUROC's no-information value is 0.5 whatever the class ratio and
whatever the group structure. What per-gene imbalance does is create a gene-prevalence predictor
that scores above chance; it does not move chance itself. The problem with the run is that some
unknown share of the reported score is attributable to that predictor rather than to the mutation,
which is a question about what the number means, not about what the floor is.

**Fix.** The balance statements become true once issue 1 is fixed, so they need the run to match
them rather than rewording. Add a balance assertion to the probe phase that fails if any gene's two
class counts differ, so the property is enforced by the code rather than asserted by the prose, and
record the realised per-gene balance in the result file.

---

## 6. The report claims more than a positive control can establish

**Defect.** The script prints, and the superseded report repeats, that a passing control shows the
mechanism null is a real absence of signal rather than a pipeline failure.

**Why it matters.** A positive control rules out one explanation for a null result: that the
pipeline is broken end to end. It cannot establish that the delta embeddings contain no mechanism
information, because the mechanism probe could be failing for reasons that leave pathogenicity
discrimination intact — a different probe class, a different label structure, a signal present but
not linearly accessible. Section 4 itself supports the narrower reading, since its nonlinear probes
sit consistently above the majority-class floor.

The wording is in the source, not only in the report, so regenerating the report does not remove
it.

**Fix.** State what the control establishes: that the embeddings and the probe pipeline carry and
recover strong signal on a different task, so the weak mechanism result is not explained by a
broken pipeline. Change the printed line and the report text in the same edit.

---

## 7. Per-fold discrimination detail is computed and discarded

**Defect.** The shared binary CV helper computes, for every fold, the area under the
precision-recall curve, the positive-class prevalence, and the positive and negative predictive
values at the prevalence-matched operating point. The pathogenicity script keeps only the mean
AUROC and drops the rest, so none of them reach the result file.

**Why it matters.** They are useful descriptive detail about a probe whose only reported number is
a single ranking summary, and they are already being computed at no extra cost.

**What they do not do.** They do not measure calibration. The preregistration's 2C checklist
requires the report to state that the probe measures discrimination only and is not a calibrated
risk estimate, and none of these quantities supports or tests that statement — establishing it
would take a calibration analysis of its own, such as a reliability curve or a calibration slope.
The predictive values are further limited here: on an artificially balanced case-control sample
they reflect the sampling design rather than any population, so they cannot be read as the
probability that a variant with a given score is pathogenic.

**Fix.** Carry them into the per-seed result files alongside the AUROC and aggregate them the same
way, labelled as descriptive discrimination detail. Keep the calibration caveat in the report as a
statement about what the probe is, and do not present these numbers as evidence for it. Whether to
add a real calibration analysis is a separate decision, out of scope here.

---

## 8. Dropped variants and ClinVar provenance are not accounted for in the output

**Defect.** The fetched variant file holds 38,797 variants; 37,258 were scored. The intermediate
counts are printed during the embedding phase and are not written anywhere. The result file records
only the final total, so the difference cannot be reconstructed from the outputs.

**Why it matters.** After issue 1 is fixed the drop will be larger, since balancing removes rows by
design at two stages. A reader comparing the fetched count to the scored count needs the breakdown
to see that the loss is intended filtering rather than an error.

**Fix.** Write the skip counts by reason, and the counts removed by balancing at each stage, into
the embedding metadata and forward them into the result file. Record the ClinVar retrieval date,
source URL and source-file checksum with the fetched-set parameters so the selected variants can be
tied to the source snapshot that produced them.

---

## 9. Repeated protein-level substitutions are weighted more than once

**Defect.** The scored set contains 219 distinct protein substitutions — the same gene, position,
wildtype residue and mutant residue — that appear more than once, accounting for 229 excess rows.
They arise because distinct ClinVar records can encode the same protein-level change, for instance
through different nucleotide changes at the same codon or separate submissions.

There are no cross-label conflicts: every repeated substitution carries the same label on all of
its rows.

**Why it matters.** ESM-2 sees an identical input for each of these rows and returns an identical
embedding, so the affected variants contribute more than once to fitting and to scoring. The effect
is small at 229 rows out of 37,258 and it cannot create a label conflict, but it is an
undocumented weighting choice sitting inside the experiment rather than a decision anyone made.

**Fix.** Decide explicitly whether to keep or deduplicate them, and record the decision. Both are
defensible — deduplication treats the protein substitution as the unit of observation, retention
keeps ClinVar's own record structure — but the choice must be stated rather than inherited. If they
are deduplicated, it has to happen before balancing, since removing rows unevenly across classes
would otherwise break the balance the balancing step just established. Report the count either way.

---

## 10. The same embeddings feed the geometry section

**Not a defect in this section.** Six scripts outside the pathogenicity control read the
pathogenicity variant file or its cached embeddings: the canonical pathogenicity axis build, the
conservation axis analysis, the direction geometry analysis, the magnitude-versus-direction
analysis, the axis identity probe, and the transfer contrast. One mechanism follow-up reads them as
well.

Claims 2D and 2E rest on the conservation axis and the magnitude-direction analyses, so they
inherit the unbalanced data from issue 1 in addition to the shared interval defect already recorded
against them in `exp4_issues.md`.

**Consequence for sequencing.** Re-fetching and re-extracting invalidates every one of those
results. Section 6 cannot be re-run until section 5's inputs are rebuilt, which makes issue 1 a
blocker for two sections rather than one.

---

## 11. Findings that are likely to survive

Provisional until the balanced re-run. The contribution of the imbalance in issue 1 is unknown, so
none of the following is settled, but each is large enough that the imbalance is unlikely to
account for it.

The delta embedding separates pathogenic from benign variants far above the pass bar on every seed,
under both splits, and with both a linear and a nonlinear probe. Seed-to-seed variation is
negligible. The gene- and family-split point estimates are nearly identical on every seed,
providing little descriptive evidence of family dependence. No paired interval on that gap is
available.

The wildtype embedding alone scores 0.60 and the delta scores 0.90, which is the reverse of the
mechanism experiment, where the wildtype carries the signal and the delta does not. That contrast
is the substance of the positive control. It is also the finding most exposed to issue 1, since the
wildtype arm is the one that reads gene identity directly, so its value in particular should be
expected to move.

Bootstrap coverage is complete: every resample was valid, and the family-split analysis covers all
but 376 of the scored variants, the ones whose genes have no Pfam annotation.

---

## 12. Remediation plan, in order

Every code and specification change lands before anything is re-run. Re-running first and fixing
the verdict, the output metrics and the seed metadata afterwards would mean running the probes
twice.

**Specification, before any code:**

1. Decide the 2C seed-combination rule and record it as a dated post-result specification amendment
   that explicitly resolves an omission exposed by the first run (issue 4).
2. Decide whether repeated protein substitutions are kept or deduplicated, and record that choice
   under the same amendment status (issue 9).

**Code, all of it before any re-run:**

3. Harden the fetch's cache diagnostics so a missing parameter key raises a named staleness error;
   the existing inequality already triggers the required re-fetch when step 2.8 runs (issue 1).
4. Make the embedding staleness check derive the expected variant set rather than reconstructing it
   from the cache (issue 1).
5. Add the per-gene balance assertion to the probe phase (issue 5).
6. Land the fold-aware bootstrap helper from `exp4_fixes.md` (issue 2).
7. Adjudicate 2C on the family-split interval and write the verdict into the result file (issue 3).
8. Carry the per-fold discrimination detail, variant accounting, ClinVar source provenance and
   seed basis into the outputs (issues 4, 7, 8).
9. Correct the overstated positive-control claim in the source (issue 6).

**Re-run, only after all of the above:**

10. Re-run the ClinVar fetch, runbook step 2.8.
11. Re-extract the pathogenicity embeddings on the pod.
12. Re-run section 5, five seeds.
13. Re-run section 6 in full, since it reads the same embeddings.

**Then:**

14. Regenerate the report. It must not be edited in place — every number in it changes as a
    consequence of steps 10 through 13.
