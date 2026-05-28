# Plan: Magnitude-vs-direction decomposition — *why* ESM-2 encodes pathogenicity but not mechanism

## Numbering note

result_21 = Megascale/S1724 stability positive control (running). plan_loglik is conditionally result_22. If this passes its decision rules it will be written up as **result_23**. If it fails, it becomes a paragraph in result_6's discussion ("the dissociation is not a magnitude artifact"), not a standalone result.

---

## The finding that motivates this

**result_6** established the project's spine: the *same* ESM-2 delta embeddings (mutant − WT) predict ClinVar pathogenic-vs-benign at MLP AUROC **0.886** (family-split-stable, Δ=0.002) while classifying GOF/DN/LOF at chance (result_4 macro-F1 0.28). Conclusion: **ESM-2 encodes *whether* a mutation matters, not *how*.**

result_6 demonstrates *that* the dissociation exists. It does **not** explain *what kind* of signal pathogenicity is, because it uses the full 1280-dim delta — it only shows the signal is *somewhere* in the embedding. This plan asks the next question: **is pathogenicity carried by the *magnitude* of the perturbation, and is mechanism absent because it would require a *direction* ESM-2 does not encode?**

If yes, the paper upgrades from "ESM-2 doesn't do mechanism" (arguably folk knowledge) to a mechanistic account: *ESM-2 deltas encode the **size** of a perturbation (constraint), which is what pathogenicity needs; GOF-vs-LOF is a **functional-consequence direction** that is a systems-level property absent from single-sequence representations.*

---

## The decomposition

Every variant gives a delta vector `d = mut_emb − wt_emb` (both the `delta_mean` and `delta_pos` views from result_6/result_7). Split it into:

- **magnitude** `m = ‖d‖` — one scalar: "how much does this mutation disturb the representation"
- **direction** `u = d / ‖d‖` — unit vector: "*which way* it disturbs it"

Pre-registered claim: **pathogenicity ≈ magnitude; mechanism would need direction; ESM-2's direction carries biophysical but not functional information.**

---

## Probes (pre-registered)

### Probe A — single-scalar magnitude test

Feature = `m = ‖d‖` only (one scalar per variant), for both `delta_mean` and `delta_pos`.

- **A1 (pathogenicity):** AUROC of the scalar `m` (and 1-feature logistic regression), family-split, 5 seeds. Compare to result_6 full-embedding 0.886.
- **A2 (mechanism):** 3-class macro-F1 from `m` alone, family-split, 5 seeds. Compare to result_4 floor.

### Probe B — magnitude-removed (direction-only) test

L2-normalise every delta to unit length (`u`), discarding `m`. Re-run the result_6/result_7 probes (MLP + logistic) unchanged.

- **B1 (pathogenicity on direction):** does AUROC collapse toward chance once magnitude is removed?
- **B2 (mechanism on direction):** does family-split macro-F1 stay at the floor? (If direction held mechanism signal, this is where it would appear.)

### Probe C — biophysical direction via *signed* ΔΔG (S1724)

Uses result_21's S1724 data (1,277 single-point missense, signed ThermoMutDB ΔΔG). Tests whether ESM-2 deltas encode *any* directional information when the direction is **biophysical** rather than functional.

- **C1 (magnitude↔magnitude):** Spearman(`m`, `|ΔΔG|`), protein-holdout.
- **C2 (direction↔sign):** AUROC of the delta (full + direction-only) for `sign(ΔΔG)` (stabilising vs destabilising), protein-holdout.

The contrast is the payload: if ESM-2 recovers biophysical sign (C2 passes) but never recovers functional direction (B2 at floor), the missing axis is specifically *functional consequence*, not "direction" in general.

---

## Decision rules (pre-registered)

Thresholds set before running. result_6 reference = 0.886; mechanism floor reference = result_4 variant-level macro-F1 (0.28) / result_7 gene-level family-split floor (~0.39 — use the matching granularity for each probe and state it).

| Gate | Condition | Interpretation |
|---|---|---|
| **P1** | A1 magnitude-only pathogenicity AUROC ≥ **0.85** (within 0.04 of full 0.886) | Pathogenicity is magnitude-dominated — a constraint/importance readout |
| **P2** | B1 direction-only pathogenicity AUROC drops to ≤ **0.70** | Removing magnitude destroys most pathogenicity signal — confirms P1 from the other side |
| **P3** | B2 direction-only mechanism family-split macro-F1 ≤ floor + **0.02** | Direction carries no functional-mechanism signal — mechanism is genuinely absent, not hidden in an unexamined axis |
| **P4** | C2 signed-ΔΔG AUROC ≥ **0.65** AND C1 Spearman ≥ **0.30** | ESM-2 *does* encode biophysical direction — so the mechanism gap is specific to *functional* direction |

**Pass = the mechanistic story holds:** P1 ∧ P3 are the load-bearing gates (magnitude carries pathogenicity; direction does not carry mechanism). P2 is confirmatory. P4 sharpens the claim from "magnitude only" to "magnitude + biophysical sign, but not functional direction."

**Failure modes are still informative (pre-registered, not post-hoc):**
- **P1 fails** (magnitude-only AUROC well below 0.85): pathogenicity needs direction too — the clean "magnitude = constraint" story is wrong. Report honestly; the result_6 dissociation stands but its *explanation* is not the magnitude decomposition. Demote to a discussion paragraph.
- **P3 fails** (direction-only lifts mechanism above floor): there *is* recoverable mechanism signal in the embedding direction that every prior probe missed — this would be a **positive** mechanism result and would reopen the prediction track. High-value either way.
- **P4 fails** (no signed-ΔΔG recovery): ESM-2 has no directional sense at all, biophysical or functional. The story simplifies to "magnitude only," which is weaker but still coherent.

---

## Why this is not just restating result_6

| | result_6 | This plan |
|---|---|---|
| Claim | The dissociation *exists* | *What kind* of signal each side is |
| Feature | Full 1280-dim delta | Magnitude scalar vs unit direction |
| Question | Is mechanism signal present? (no) | *Why* is it absent and pathogenicity present? |
| Output | "encodes whether, not how" | "encodes perturbation **size** (+ biophysical sign), not **functional direction**" |

The single-scalar result (A1) is the striking figure: if *one number* nearly matches a 1280-dim MLP for pathogenicity but is at chance for mechanism, the dissociation is visibly a magnitude phenomenon.

---

## Connection to existing results

| Existing result | What this adds |
|---|---|
| **result_4 / 7** mechanism null | Explains the null mechanistically rather than just reporting it |
| **result_6** pathogenicity control | Decomposes *what* the 0.886 signal is (magnitude) instead of leaving it as a black-box embedding |
| **result_19** ClinVar-pattern features | `delta_mag_cv`/`delta_mag_std` were aggregated magnitude features; this isolates magnitude at the variant level and tests it directly |
| **result_21** stability control | Reuses S1724 signed ΔΔG for the biophysical-direction test (Probe C) — depends on result_21 reading out non-LEAKY |

**Dependency:** Probe C's interpretation hinges on result_21 not coming back LEAKY (stability ≠ fold-memorisation). If result_21 is LEAKY, run A/B only and drop the biophysical-direction claim.

---

## What would change about the paper

- **Pass (P1 ∧ P3, ideally + P4):** the discussion gains a mechanistic account — title-worthy framing along the lines of *"Protein language models encode the magnitude of a perturbation, not its functional direction."* This is the line that separates the paper from folk knowledge.
- **P3 fails:** reopens mechanism prediction — pivot back, this becomes the lead positive result.
- **P1 fails:** the magnitude story is dropped; result_6 stands unchanged; this plan yields one honest negative paragraph.

---

## Implementation plan

### Phase 1 — script (half day, local CPU; no GPU needed)
Write `scripts/magnitude_direction.py`:
- Load cached deltas from result_6/result_7 (`delta_mean`, `delta_pos`) — **no re-extraction**, this is pure analysis of existing embeddings.
- Compute `m = ‖d‖` and `u = d/‖d‖` per variant for both views.
- Pathogenicity labels: reuse result_6 ClinVar pathogenic/benign table. Mechanism labels: merged GOF/LOF/DN.
- Implement Probe A (scalar), Probe B (unit-normalised), under the result_6/result_7 probe + family-split CV harness, 5 seeds.

### Phase 2 — Probe C (half day, local CPU)
- Load S1724 deltas from result_21 cache (or extract if result_21 hasn't cached them yet — ~2.5k forward passes, trivial).
- C1 Spearman(`m`, `|ΔΔG|`); C2 AUROC for `sign(ΔΔG)` from full delta and from `u`. Protein-holdout, 5 seeds.

### Phase 3 — decision + write-up (half day)
- Fire gates P1–P4.
- Headline figure: single-scalar magnitude → pathogenicity vs mechanism bar pair.
- Write `docs/result_23.md` if P1 ∧ P3 pass; otherwise fold into result_6 discussion.

---

## Files

| File | Status |
|---|---|
| `scripts/pathogenicity_control.py` | ✓ exists (result_6) — reuse labels + probe harness |
| `scripts/megascale_stability.py` | ✓ exists (result_21) — source of S1724 deltas/ΔΔG |
| cached `delta_mean` / `delta_pos` embeddings | ✓ exist — reused, no GPU |
| `scripts/magnitude_direction.py` | ✓ written (Phase 1) |
| `results/magnitude_direction/probe_results.json` | ✓ single-seed output written |
| `docs/result_23.md` | ✓ written as PRELIMINARY (hypothesis falsified — inverse finding, see below) |

---

## Progress log (2026-05-28)

**Script written and Probe A/B run for a single seed (seed 0).** Pure CPU analysis of cached embeddings, as planned.

**Harness bug found and fixed before trusting any number.** The first run reused `multiseed_v1.load_pathogenicity`, which points at an *older, superseded* pathogenicity extraction (`emb_*_pathogenicity_*_n17259.npy`). That under-reports the full-delta baseline (family-split MLP AUROC 0.746 vs result_6's 0.884). Fixed by loading the **canonical** set (`pathogenicity_valid_variants_canonical.json` + `emb_*_path_canonical_n16576.npy`, n=16,576). After the fix the full-delta baseline reproduces result_6 exactly (family-split MLP **0.886**), so the decomposition numbers are trustworthy.

**The pre-registered hypothesis (P1: magnitude carries pathogenicity) is FALSIFIED — in the opposite direction.** Single-seed family-split pathogenicity AUROC:

| feature | logreg | MLP |
|---|---|---|
| full delta | 0.834 | 0.886 |
| magnitude only `‖d‖` | 0.664 | 0.664 |
| direction only `d/‖d‖` | 0.849 | **0.896** |

Pathogenicity is carried by the **direction** of the embedding shift, not its magnitude. Direction-only ≈ full delta (even marginally higher); magnitude-only is weak. This is the **P1-fails failure mode** pre-registered above — but instead of a null demotion, it is a positive *inverse* finding: pathogenicity is a directional property of ESM-2 perturbation space. Gates: P1 FAIL (0.664 < 0.85), P2 FAIL (0.896 > 0.70 — direction does NOT collapse), P3 PASS (direction-only mechanism F1 0.296 ≤ floor 0.313 + 0.02). Mechanism stays at chance under every decomposition.

**Written up as `docs/result_23.md` (PRELIMINARY).** Marked single-seed; the inverse finding replaces the original framing.

### What remains

1. **Full 5-seed run** of Probe A/B to confirm the 0.664-vs-0.896 gap is not a one-seed fluke (gap is large, so expected to hold). `python3 scripts/magnitude_direction.py` (default seeds 0–4).
2. **Probe C is blocked on data, not code.** result_21 (megascale/S1724) ran on a GPU host; the embeddings (`data/embeddings/megascale_{wt,mut}_mean.npy`) and `data/megascale_variants.json` are **not local**. Sync them down, then Probe C runs automatically. With the LEAKY verdict, Probe C is reframed: not "does ESM-2 have biophysical direction" but "does signed-ΔΔG direction survive protein-holdout" (expected: largely no).
3. **Add binarized-ΔΔG protein-holdout AUROC to Probe C** (median split) — the metric-matched number that hardens the result_21 LEAKY dissociation (already confirmed offline: 0.764→0.642, Δ=0.122).
4. **Reconcile with the family-robust story:** if pathogenicity = direction and that direction is family-robust (0.896 family-split), test whether what *leaks* for stability/mechanism is the direction or the magnitude. Natural follow-up, not yet scoped.
