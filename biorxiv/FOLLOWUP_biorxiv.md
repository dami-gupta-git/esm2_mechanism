# Follow-up work — after run_biorxiv

Work that does not gate run_biorxiv. Each item is self-contained; none is a precondition for the
paper, and nothing in [`RUNBOOK_biorxiv.md`](RUNBOOK_biorxiv.md) depends on any of them landing.

Task numbering is kept from the retired run_biorxiv plan so existing cross-references still resolve.

The clean-label arm (Task 2d) was in run_biorxiv scope and is now here. Its expected outcome was an
underpowered sensitivity check — a clean-label filter leaves DN near ~50 genes across perhaps ~150
families, below what a family-resampled CI can constrain — and Task 8 answers the same
label-heterogeneity objection without depending on subset size.

---

## Experiment 3 — within-family mechanism (cut from run_biorxiv scope)

TBD. Was listed as an in-scope experiment in `RUNBOOK_biorxiv.md` alongside Experiments 1, 2, 5,
and 7, but never scoped beyond the placeholder. Moved here so the runbook only lists experiments
that are actually specified.

---

## Task 2b — homology-partition robustness panel (cut from run_biorxiv, committed panel withdrawn)

The mechanism null is measured under the Pfam family partition only. C6 was removed from the
confirmatory set and the coarser-partition check is named in the paper as follow-up work, so the
paper claims partition-independence nowhere and must not imply it (R7.2).

**The committed panel is defective and is withdrawn, not amended.** Anyone reviving this starts
from a rerun, not from `results/run_biorxiv/homology_partition_panel/panel.json`, which is deleted.
Three defects produced its apparent finding that the null strengthens under stricter partitions:

- **The clan arm ran on a different dataset, not a stricter split.** Only genes whose Pfam family
  belongs to a clan can be clan-split, which drops roughly two thirds of the genes — a smaller,
  differently-composed subset biased toward well-studied superfamilies.
- **Every arm was scored against the same chance floor.** All three rows carry
  `measured_floor: 0.2884`, the family-split floor on the full dataset. Held-out folds under
  different partitions have different class balance, so the floor moves; the clan arm was graded on
  the wrong curve, and the leakage fraction inherited the error through its denominator.
- **The MMseqs2 clustering was never validated.** At 20% identity it produced 1,215 clusters from
  ~1,935 genes — finer than Pfam family's 1,134, when that threshold should merge aggressively.
  Check the coverage threshold (`-c` / `--cov-mode`) first if this arm is revived.

On a matched subset with each arm against its own floor, family and clan scores are close and no
decline is visible. `for_me/homology_partition_findings.md` records the retracted version. No
smoke-scale CI may be quoted: at `n_boot=20` a nominal 95% interval sits at essentially the min and
max of the resample distribution.

Two bugs found here were real and their fixes stand: `mmseqs_cluster_holdout.py` fed int-coded
labels into probes that compare against the string `MECHANISM_CLASSES` internally, silently zeroing
every class bucket and crashing on the first balanced fold; and `clan_holdout.py` compared against
hardcoded stale reference floors (`0.352`, `0.387`) instead of live measured numbers.

---

## Open — does stability gate H2 get a paired test?

H2 (the random→family Spearman drop against the LEAKY threshold of 0.10) is currently descriptive.
The paired cluster bootstrap exists and could be applied to it. Recommendation: do it if cheap,
otherwise state explicitly in `report_stability.md` that the gate is descriptive rather than tested.
This is the only question left open when the run_biorxiv plan was retired; everything else in it was
either resolved or became code.

---

## Task 2d — ★ clean-label robustness arm (Badonyi & Marsh 2025)

Task 2b asks whether the mechanism null survives a **coarser partition**. This asks whether it
survives **cleaner labels** — same shape, same machinery, same reports section. The objection it
answers is stated in R7.2 of `PREREGISTRATION_run_biorxiv.md`: one label per gene means some fraction of variants is
mislabelled by construction, which caps achievable macro-F1, and a reviewer can claim the floor
result is that cap rather than an absent signal.

**The test.** Re-run the Experiment 1 family-split mechanism probe on the subset of genes whose
labels are unambiguous — dominant genes with a **single** disease phenotype, where the gene-level
mechanism annotation is not averaging over two different mechanisms. If `delta_mean` is still at
the (recomputed) floor on that subset, label noise is excluded empirically rather than argued away.

**Shape: a row filter, not new science.** Structurally identical to
`experiments/mechanism/single_source_mechanism.py`, which already re-runs the same probe on the
Gerasimavicius-only subset to remove the curation-source confound. Reuse `load_data()` and
`run_family_split()` unchanged, filter the row set, and **recompute the majority-class floor on the
subset** — the class balance shifts, so the merged-set floor of 0.288 does not carry over. That
recomputation is the part most likely to be got wrong by copying the merged number across.

**Inputs.** Badonyi & Marsh's Table S1 (N = 2,837 OMIM phenotypes with MIM identifiers, EDC and
ΔΔG_rank values, mLOF scores, mechanism posteriors) at `https://osf.io/29pxr`. Note it is
**phenotype-level, not gene-level** — the clean subset is derived by selecting genes with exactly
one qualifying phenotype, not by reading a "clean" column. Their gene set derives from
Gerasimavicius et al., which is also this project's source, so overlap should be substantial.

★ **Go/no-go check before this is scoped as a robustness row.** Perform the join and count the
intersection against the 1,935 genes in `valid_variants.json` **first**. Under family-split the
effective unit is the family, and the merged set already has 833 singleton families out of 1,134 —
a clean subset of a few hundred genes may leave too few non-singleton families to support an
interval worth reporting.

★ **The binding constraint is the rare classes, not the gene total — measured 2026-07-22.** The
merged set's *gene-level* label distribution is far more skewed than its variant-level one:

| | Genes | Families |
|---|---|---|
| Total | 1,935 (33 with no Pfam) | 1,134 — **833 singleton**, 301 non-singleton |
| In non-singleton families | 1,069 | 301 |
| **LOF** | 1,701 | |
| **GOF** | **136** | |
| **DN** | **98** | |

DN is 98 genes and GOF 136, spread across 301 non-singleton families. Any clean-label filter cuts
directly into those. If the subset retains even half, DN lands near ~50 genes across perhaps ~150
families — below what a family-resampled CI can usefully constrain, and squarely in the rare-class
regime where a rare-class interval is indicative rather than authoritative. **The thin branch is
therefore the expected outcome**, not a first-class row.

★ **Report per-class counts, not just the gene total.** A clean subset of 600 genes containing 30
DN genes is not a usable three-class arm however healthy the total looks. The go/no-go decision is
made on the *minimum* per-class gene count and its family spread, not on the subset size.

- **Adequate overlap** → a first-class robustness row, with a CI resampling families on the subset
  and the subset floor stated next to it.
- **Thin overlap (expected)** → report as an explicitly underpowered sensitivity check under Task
  0.1's underpowered-null language, and do **not** let it gate the run.

Either way the count itself is reported, so the reader can see what the subset could and could not
resolve. This is exploratory (it is not in the C1–C4 confirmatory set) but it is the direct answer
to a confirmatory claim's main threat, so it belongs in the same report as C1.

★ **Consequence for sequencing: Task 8 likely carries this argument, not Task 2d** — it has no
subset-size dependency. Task 2d remains worth running for the count it reports, even in the thin
branch.

**Reviewer-facing note:** Badonyi & Marsh apply a sequence-identity
control (<50% pairwise identity) only to their LoGoFunc comparison test set, not to their headline
mLOF AUROCs (Methods, p.20). Their own discussion (p.13) flags supervised mechanism predictors as
suffering "inflated performance estimates due to circularity issues." The leakage fraction measured
here is a quantified instance of exactly that concern, raised independently by the same lab — cite
it in `report_leakage_fraction.md` and in `ESM2_REPORT.md` §4.

---

---

## Task 7 — conservation-residualised mechanism test (post-run_biorxiv, speculative)

The fourth "what I would do next" item in `ESM2_REPORT.md`: if pathogenicity ≈ conservation in
ESM-2's delta, then mechanism differences might live in a conservation-residualised space —
variants equally likely to be deleterious, differing in *how*. Project the delta onto the
conservation axis and ask whether the residual carries mechanism. Untested.

**Difficulty: low, roughly a day of work.** It is a near-exact clone of an experiment already
built and debugged:
[`megascale_stability.py:171`](../src/esm2_mech/experiments/stability/megascale_stability.py#L171)
`run_h3_stability_projection` has the identical shape — fit a direction, project it out of
`delta_mean`, re-run the family-split mechanism probe on the residual, compare against the
unprojected baseline. Swap the stability axis for the conservation axis. The subtle parts are
already solved there and should be copied, not re-derived:

- Standardize once up front, project last, and never re-standardize after projecting (the
  residuals are rank-deficient along the removed direction; per-column rescaling reintroduces
  variance along it and silently undoes the projection). See the comment at L220.
- Verify the removal with `var_after ≈ 0` along the direction (L234).
- Both arms get identical preprocessing so the projection is the only difference.

`project_out_subspace` at
[`mechanism_delta_probe.py:234`](../src/esm2_mech/experiments/mechanism/mechanism_delta_probe.py#L234)
is a second reference implementation.

**What is genuinely new:**

1. **Conservation must be extracted for the mechanism variant set.** `conservation_axis.py:74`
   currently extracts masked logP_wt / logP_mut / entropy for the *pathogenicity* set only. The
   mechanism set (`valid_variants.json`, 17,826 variants) needs its own masked-LM pass — a GPU
   step, and the only meaningful compute cost in this task.
2. **Deriving the axis.** Conservation is a scalar per variant, not a vector, so "the
   conservation axis" in embedding space has to be constructed — regress `delta_mean` on the
   conservation scalar and take the fitted direction, exactly as H3 does with predicted
   stability. This is the one part worth designing rather than copying.

**Two preconditions, both cheap, both determining whether the result is interpretable at all.**

*(a) Confirm the strip worked, at two levels.* The existing `var_after ≈ 0` assertion
(`megascale_stability.py:229-240`) is a *geometric* check — it proves the direction was removed
from the matrix, and it must be carried over verbatim. But it does not prove conservation was
removed as a *predictive* signal, because a scalar regressed onto 1280 dimensions leaves
conservation-correlated variance in the orthogonal complement. So add a second, functional
check: re-run the pathogenicity probe on `d_resid` and confirm it now scores at chance (≈ 0.50
AUROC). If the residual still predicts pathogenicity above chance, conservation was not fully
removed and a mechanism null is uninterpretable — indistinguishable from the projection having done
nothing. This is a few lines reusing `auroc_family_split` (`conservation_axis.py:199`). Report the
residual pathogenicity AUROC alongside the mechanism number whatever the outcome.

*(b) Pre-register the read before seeing the number.* Both outcomes are publishable, which is
exactly when a framing chosen afterwards is indefensible. Write the prediction into an
`EXPERIMENT.md`-style pre-registration before the run, with the gate stated numerically:

> - Residual family-split macro-F1 > floor + 0.05 → mechanism lives in a subspace orthogonal to
>   conservation, and the run6 null was a conservation-swamping artifact.
> - Residual at floor (and residual pathogenicity AUROC ≈ 0.50) → the mechanism null survives
>   conservation-residualisation.
>
> Floor is the measured majority-class macro-F1 from `naive_baseline.json`, not a nominal 1/3.

**Precedent — the structurally identical test already returned ~zero.** H3 in
`megascale_stability.py` performed the same operation with the *stability* axis and moved
mechanism family-split macro-F1 by **−0.00053** (baseline 0.3947 → projected 0.3942,
`results/run6/megascale_stability/h3_stability_projection.json`). It replicates: the same test
in `results_0` gave +0.00045. Conservation is a different and more dominant axis, so this is
worth running — but a near-zero result is the more likely outcome, and the plan should expect it.

**What a null buys.** If the residual stays at floor, framings A and B collapse into one claim: the
ESM-2 delta encodes a single conservation axis, and neither stability nor mechanism survives its
removal. That closes the standing "your probe was just swamped by conservation" objection, which is
currently unaddressed.

**Expected outcome — likely null.** `delta_mean` sits at the chance floor on both splits in run6, so
projecting a direction out of it removes one direction from a feature with no mechanism signal to
begin with. The more interesting variant tests the residual of `wt_only_mean` or `wt_concat_mut` —
the features that actually score — but those carry protein identity, so a null there is confounded
by the ~40% homology leakage rather than informative about mechanism. A stated negative is
publishable, but this must not displace the run_biorxiv statistics work.

---

---

## Task 8 — ★ label-noise tolerance simulation (post-run_biorxiv, CPU)

Task 2d asks whether the null survives clean labels. This asks the complementary question:
**how far would label noise of the measured magnitude actually move a probe that works?** It turns
"noise alone would not drive a result to exactly the floor" from an assertion into a measurement.

★ **Priority raised above Task 2d (2026-07-22).** The measured gene-level rare-class counts
(DN = 98 genes, GOF = 136) mean Task 2d's clean subset will very likely be too thin to constrain a
family-resampled interval. This task runs on the full pathogenicity set and has no such dependency,
so it should be run first of the two.

**Method.** Take the pathogenicity task — the same embeddings, the same pipeline, and a
demonstrated AUROC of 0.894 family-split — and corrupt its labels at increasing rates spanning the
heterogeneity Badonyi & Marsh report (roughly 20–50%). Plot performance against noise rate under
the same family-split CV. The output is a degradation curve on *these* embeddings, not a textbook
argument about label noise in general.

**The read.** If a probe at 45% label noise still scores well above chance, then noise at the rate
observed in mechanism annotations does not explain a result sitting at the floor, and the
label-noise objection is answered quantitatively. If instead the curve collapses to chance at
realistic noise rates, that is a genuine limitation and must be reported as one — it would mean the
mechanism null cannot be distinguished from a labelling artifact by this dataset, which is a
material caveat on C1 rather than a footnote.

**Design notes.**

- Corrupt labels at the **gene** level, not per-variant. The real noise mechanism is a gene carrying
  two mechanisms and receiving one label; flipping individual variants models a different and
  easier problem.
- Sweep multiple rates with several draws per rate, and report a CI per rate — a single draw at a
  single rate is not a curve.
- The three-class mechanism task and the binary pathogenicity task have different chance floors, so
  the comparison is between each probe and **its own** floor, expressed as fraction of headroom
  retained, never as raw metric values side by side.
- Pre-register the read before running it (same reasoning as Task 7b): both outcomes are
  informative, which is exactly when a post-hoc framing is indefensible.

CPU-only, reuses existing embeddings, no new data.

---

---

## Task 9 — ★ mLOF / ΔΔG_rank comparison (post-run_biorxiv)

The positive control for the labels themselves. Task 2d and Task 8 argue the labels are good enough;
this shows directly that they carry learnable signal, by demonstrating that *other* features recover
mechanism where the ESM-2 delta does not. If a structural score achieves above-chance mechanism
separation on the **same labels** under the **same family-split**, label noise is excluded as the
explanation for the null and the finding becomes a statement about representations rather than about
the dataset. It is new science and must not displace the run_biorxiv statistics work.

**Two arms, and they are not the same difficulty:**

1. **ΔΔG_rank (cheap, directly comparable).** Precomputed proteome-wide values at
   `https://osf.io/g98as`. Variant-level, so it drops straight into the existing probe as a feature
   arm alongside the current FoldX baseline. Subject to the project rule against imputing missing
   scalar features: restrict to the observed subset and recompute CV splits on that subset — never
   fill with 0.0.
2. **mLOF (harder, not a drop-in baseline).** ★ **mLOF scores a *group* of variants, not a single
   variant** — it requires ≥3 distinct missense positions with pLDDT > 70 and produces a
   phenotype/gene-level likelihood. This project's probe is per-variant. A head-to-head therefore
   needs predictions aggregated to gene level, which **changes the task** and its chance floor. Run
   it as an explicitly gene-level comparison with its own floor, or not at all; presenting it beside
   per-variant macro-F1 would be a category error.

Method and notebook: `https://github.com/badonyi/mechanism-prediction`. Their scope limits are worth
stating when citing them — missense only, pLDDT > 70 regions only, assumes the input variants are
causal, and monomeric AlphaFold models (their Discussion, p.14).

**Expected outcome.** Their reported gene-level AUROCs are 0.622–0.714 across the binary mechanism
pairs — real signal, but modest, and measured without a homology-partitioned split. Under this
project's family-split the comparable numbers may well be lower. A modest-but-above-chance
structural result against an at-floor embedding result is the outcome that most sharpens the
paper's claim.

**Third arm — DN high-water-mark replication.** A separate, earlier finding (proteome features
combined with Badonyi structural priors) was previously this project's best DN result. Re-run that
combination under the current labels and the current family-split statistics (paired cluster
bootstrap, recomputed floor) to check whether the lift survives the stricter methodology now in
place. This is not a comparison against the ESM-2 delta like the other two arms — it is a check of
whether a result obtained under looser statistics still holds. Report it alongside the ΔΔG_rank and
mLOF arms since all three use the same proteome/Badonyi inputs, but keep the read separate: the
ΔΔG_rank and mLOF arms answer whether the *labels* carry signal, this arm answers whether a specific
*prior positive result* replicates.

---
