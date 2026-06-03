# bioRxiv outline — characterisation-led framing

**Drafted:** 2026-06-03
**Sourcing rule:** reference only the clean **run6** reports (`reports/run6/`) and run6 results.
Do **not** cite the exploratory run0 `result_N` reports — they are exploratory and contain known
small bugs. Where a beat needs an experiment that does not yet have a run6 report, it is marked
*(needs run6 report)*.

## One-line claim

ESM-2's variant delta encodes pathogenicity — which is just conservation — not disease mechanism;
the apparent mechanism signal under standard CV is protein-family recognition.

## Main text (6 beats)

1. **Naive result** — mechanism looks predictable under gene-split CV.
2. **It's family recognition** — embeddings cluster by Pfam (`report_protein_family.md`) + signal
   collapses under family-split (`report_classifier.md`) + the per-feature leakage diagnostic
   (`report_leakage_fraction.md`). *(core finding)*
3. **The pipeline is sound** — pathogenicity passes family-split (AUROC 0.89, linear, ~0 drop;
   `report_control.md`). Clean, one-glance positive control. Stability gets a single sentence here
   as the non-circular, physically-measured confirmation (full numbers in supplement).
4. **What the delta actually is — conservation** — direction-not-magnitude + masked-LL matches the
   full embedding (`report_geometry.md`), then validated externally on ProteinGym (ESM-2 ΔLL vs DMS
   fitness; *needs run6 report*): conservation transfers to pathogenicity / partially to stability /
   to DMS fitness on average / not to mechanism. *(positive payoff, externally grounded — the heart)*
5. **Not for lack of trying** — a contrastive metric-learning probe, built specifically to surface
   family-invariant mechanism signal, lifts the floor only +0.03 (*needs run6 report*). Frame as the
   *limit case* ("only adversarial extraction surfaces a sliver, an order below the controls"),
   not a counterexample. *(pre-empts the "you didn't try hard enough" objection)*
6. **Scale doesn't fix it** — ESM-3 lifts the floor modestly, structure tokens add nothing
   (`report_esm3_mechanism.md`).

## Control structure (decided)

- Main control: **pathogenicity** (`report_control.md`) — legible, linear, family-robust.
- Stability (`results/run6/megascale_stability/`): one-line mention in beat 3 + full report in
  supplement *(needs run6 report)*. Its value is the circularity rebuttal (test-tube ΔΔG,
  independent of ESM-2's evolutionary training); two independent nonlinear probes (MLP, XGBoost)
  agree at family ρ ≈ 0.63 / AUROC ≈ 0.82.

## Supplement

Within-family (`report_within_family.md`), full stability, enzyme-type control, contrastive
details, AlphaMissense in/out-of-distribution. *(items without a run6 report still need one)*

## Parked for paper 2 ("what *does* predict mechanism")

Gene-level proteome features, structural priors, clinical utility, the modality comparison framing.
These answer a different, positive question and would dilute the clean negative result here.

## Figures still to make

- ProteinGym ΔLL transferability — new panel.
- Contrastive floor lift — new panel.
- Beats 2–4 are largely covered by existing run6 figures (family clustering, family-split bars,
  geometry).

## What this paper must NOT claim

- "ESM-2 encodes mechanism" (claim the bounded null + dissociation).
- "Mechanism is unlearnable from PLMs" (claim the measured floor under this setup).
- Contrastive +0.03 as a positive result (it's the limit case reinforcing the null).

---

## PS — leakage fraction on the bigger dataset (2026-06-03)

The clean single "62.8% structural-invariant leakage fraction" headline (older framing) did **not**
survive the larger merged dataset. On run6 (17,826 variants / 1,935 genes), the per-feature picture
(`report_leakage_fraction.md`) is:

- The **delta is at the chance floor on both gene-split and family-split** (0.288 / 0.288) — its
  leakage fraction is undefined because there was no above-chance gene-split score to leak. This is
  a *cleaner* null than "looks good then collapses."
- The ~40% leakage lives in the **absolute embeddings** (`wt_only`, `mut_only`, `wt_concat_mut`,
  39–40%). Wildtype and mutant leak identically → the leaking signal is **protein identity, not the
  mutation**.

Implication for beat 2 (not yet folded into the beat above): re-anchor the family-recognition
argument on the **wt-vs-delta contrast** rather than a single leakage number —

> The mutation delta is at chance under both splits. The apparent mechanism signal comes from
> absolute embeddings, which encode which protein the variant sits in; ~40% of that gene-split
> score is family recognition that disappears under family-disjoint folds (wt and mut leak
> identically — it's identity, not the mutation). With 83% within-family mechanism agreement and
> the family clustering, standard evaluations measure family recognition, not mechanism.

The leakage fraction stays as a *supporting per-feature diagnostic* (`report_leakage_fraction.md` is
honest about it being per-dataset, not a theorem) — just not the headline.
