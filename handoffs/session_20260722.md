# Session handoff — 2026-07-22

Context dump for picking this up cold. Covers the project, what was decided this session, what was
verified against the code (versus taken from docs), and what to do next.

Nothing was implemented this session — it was all planning and verification. The deliverables are
three updated documents: [`PLAN_biorxiv.md`](PLAN_biorxiv.md),
[`RUNBOOK_5.md`](RUNBOOK_5.md) (new), and the Run 7 section of
[`RUN_PROGRESS.md`](RUN_PROGRESS.md).

---

## 1. The project in one paragraph

Does frozen ESM-2's variant delta embedding (`mut − wt`) encode **disease mechanism**
(GOF/DN/LOF)? Answer: no — it encodes **pathogenicity**, and even that turns out to be
conservation. Run 6 is the canonical run: 17,826 missense variants, 1,935 genes, 1,134 Pfam
families; classes LOF 76% / GOF 15% / DN 9%. Under family-split CV the delta classifies mechanism
at the measured chance floor (macro-F1 0.288), while the same delta predicts ClinVar pathogenicity
at AUROC 0.894 with essentially no family-split drop. The apparent mechanism signal under standard
gene-split CV is protein-family recognition.

The current branch is `biorxiv`. The paper is a **negative result with strong controls**, outlined
in [`docs/OUTLINE_biorxiv.md`](docs/OUTLINE_biorxiv.md).

### Headline run6 numbers (memorise these; they anchor everything)

| Quantity | Value |
|---|---|
| Measured chance floor (majority-class, macro-F1) | **0.288** |
| Mechanism `delta_mean`, gene-split / family-split | 0.288 / 0.288 (at floor both) |
| Mechanism `wt_only_mean`, gene-split / family-split | 0.545 / 0.442 |
| Mechanism MLP on delta, gene / family | 0.399 / 0.380 |
| Leakage fraction (absolute embeddings) | ~40% |
| Pathogenicity `delta_mean` MLP, gene / family | 0.897 / 0.894 (Δ 0.003) |
| Conservation alone (masked-marginal) for pathogenicity | **0.891** — beats the 1280-d delta (0.859) |
| Conservation + delta | 0.893 (delta adds +0.002 → gate K2 fails) |
| ESM-3 seq / seq_struct, family-split | 0.438 / 0.453 (M1,M2 pass; M3 fails at +0.014 < 0.030) |
| Contrastive k-NN vs raw-kNN, family-split | 0.395 vs 0.354 |
| Family probe accuracy vs baseline | 61% vs 4.4% |
| Genes carrying their family's majority mechanism | 83% |

### Repo conventions that matter (from `CLAUDE.md`)

- **All paths** come from `utils/paths.py`. Never construct a directory inline.
- **All run outputs** key off the single `RUN_NAME` constant in `utils/paths.py`
  (`results/<RUN_NAME>/`, `reports/<RUN_NAME>/`). A result file and its report must share the same
  `RUN_NAME`.
- **All repeated values** live in `constants.py`.
- **No fallback/default/imputed values** for scientific data. Absent data is `None`/`NaN`, never a
  sentinel like `0.0`.
- **Do not run tests** unless explicitly asked.
- Reports follow a fixed style: neutral tone, no hype words, sentence-case table headers,
  Summary → what was measured → glossary → results → "Reading the tables" → interpretation →
  Provenance.

### User working preferences (learned this session)

- Short replies. Lead with the answer. No option surveys, no recaps, no narrated reasoning.
- **Never** use the `AskUserQuestion` popup — ask in plain text, one question at a time.
- Ask permission before implementing; default to discussion and design.
- Surface disagreement rather than quietly doing something different.
- Dislikes `wandb` (explicitly asked for its removal).
- Sends corrections and additions **mid-turn** — expect new requirements to arrive while you are
  still working, and fold them in rather than finishing the old plan first.

---

## 2. What run 7 is

**Run 7 adds inferential statistics to run 6's science.** The experiments, gates, and hypotheses
do not change. The problem it fixes: every error bar in run6 is a 5-seed spread, and a seed only
reshuffles CV folds on a fixed dataset — that measures fold jitter, not sampling uncertainty, and
understates the true error because every seed reuses all the data.

Confirmed scope decisions:

- **Full run**, all seven experiments (not just the ones with thin-margin claims).
- Results to `results/run_biorxiv/`, reports to `reports/run_biorxiv/`.
- `results/run6/` and `reports/run6/` preserved untouched as the comparison baseline.
- **Embeddings are NOT re-extracted** and NOT copied.

### Why embeddings are reused (verified, not assumed)

`EMB_DIR = DATA_DIR / "embeddings" / ESM2_MODEL` and `ESM3_EMB_DIR = ... / ESM3_MODEL`
(`utils/paths.py:68-69`). These are keyed by **model**, not by `RUN_NAME`, so they are already
run-independent — no code change and no copy needed. The arrays total **10 GB**
(9.4 GB ESM-2 + 747 MB ESM-3) and are gitignored, so duplicating per run would cost 10 GB for zero
provenance gain. run_biorxiv result files record the embedding fingerprint so the reuse is recorded in the
output rather than only in the runbook.

GPU is needed for exactly three computed steps: Exp 5 step 3 (conservation extract), Exp 7 step 4
(megascale MLP), Exp 1 step 3b (permutation refits).

---

## 3. Things verified against the code this session

These corrected the plan's own assumptions. **Do not trust the older prose over these.**

| Claim in docs | Reality |
|---|---|
| "`utils/bootstrap.py` is imported by six experiment modules" | **Three**: `naive_baseline.py`, `mechanism_delta_family_split.py`, `mechanism_within_family.py`. The wiring gap is bigger than the plan assumed. |
| `docs/README.md`: pathogenicity "clean 5-seed mean pending due to provenance issue" (0.878 vs 0.742±0.006) | **Already fixed in run6.** Exp 2 was rebuilt as a consolidated fetch→embed→probe over one canonical 37,218-variant set. `pathogenicity_control.py` fingerprints the variant set (line 306), stores it in embedding metadata (line 332), and **hard-refuses** on mismatch (line 360). All 5 run6 seeds: 0.894 family-split, std ≤ 0.001. The 0.74–0.88 band is from run0-era `result_6.md`. **The stale sentence in `docs/README.md` is still there and must be fixed** — it is what a reviewer reads first. |
| Plan says "13 reports" | **14 `.md` files** in `reports/run6/`: 11 per-experiment reports + `ESM2_REPORT.md` (assembled paper) + `INTRO_REPORT.md` (lay summary) + `STATS_PLAN.md`. The two cross-cutting reports quote numbers from every section and were missing from the plan's Task 6. |
| Outline beat 4 (ProteinGym) and the enzyme control | Modules **are** ported (`experiments/alphamissense/proteingym_{esm2_ll,alphamissense}.py`, `experiments/proteome_features/enzyme_classification.py`) but their only results are in `results/results_0/` — the exploratory run the outline **forbids citing**. Not yet in run_biorxiv scope; see open questions. |
| "38 test files" | Confirmed, and **no `.github/workflows/`** — nothing enforces they pass. |
| `pyproject.toml` runtime deps | Still pull `aider-chat`, `openai`, `anthropic`, `google-generativeai`, `wandb`, `tiktoken`, `datasets`, `pypdf`, `pymupdf4llm` — none on the result path. |
| `clan_holdout.py` / `mmseqs_cluster_holdout.py` | Both exist under `experiments/mechanism/`. Ready to promote into a robustness panel. |
| Existing permutation p-values | `wt_only_mean` p = 0.0099 = exactly 1/(200+1) — the **resolution floor** of a 200-permutation test, not a measurement. `PERMUTATION_N_RESAMPLES` already defaults to 1000; the 200 came from a runtime override. |

---

## 4. Decisions made this session

### Ordering (the most important one)

**Wire the stats machinery FIRST, verify a CI key actually appears in emitted JSON, and only then
flip `RUN_NAME` to `run_biorxiv`.** If the replay runs against unwired modules, run_biorxiv lands with the same
CI-less result files run6 has, and fixing it afterwards means either overwriting run_biorxiv files
(destroying provenance of what was actually run) or a run8. Tasks 1/2 are CPU-testable locally
against run6 paths before the flip.

### Statistical methodology corrections (Task 0 in the plan)

1. **0.0 Pathogenicity provenance** — docs fix + a verification step (confirm all 5 seeds share one
   fingerprint; if any disagrees, stop and do a real freeze-and-rerun).
2. **0.1 Pre-registered CI decision rules** — a gate is **affirmed** only if the point estimate
   clears the threshold *and* the paired difference 95% CI excludes zero. Point estimate clears but
   CI spans zero → **"not distinguishable"**, not a pass. A gate that *fails* with a CI spanning the
   threshold → **"underpowered to detect an effect of the pre-registered size"**, not "no effect".
   Must be written into `docs/EXPERIMENT.md` **before** the run, or it is retro-fitted.
3. **0.2 Whole-paper confirmatory/exploratory split** — six confirmatory claims (C1–C6), BH-FDR
   across that set only, raw and adjusted both reported; everything else labelled exploratory and
   *not* corrected (correcting an exploratory screen implies it was confirmatory). C1–C3 are
   load-bearing; trim from C4/C5 if needed.
4. **0.3 Resampling unit matches the split** — gene-split → resample genes; family-split →
   **resample families**. Family-split CIs will be visibly wider (1,134 families but 833
   singletons); that is correct, not a bug. Emit the effective cluster count next to each interval.
   **The split-gap CI resamples families** (the coarser unit) — a gene-resampled gap understates
   the family-split arm's variance, which is the very anticonservatism this rule removes.
5. **0.4 BCa for rare classes** — DN (~9%, ~150–170 genes) and GOF (~15%) sit where percentile
   bootstrap undercovers for a bounded metric with few clusters. Use BCa where computable, and flag
   rare-class intervals as least trustworthy **regardless** (with ~150 jackknife clusters BCa's own
   correction is noisy).
6. **0.5 Permutation budget split by probe** — linear probe at 1,000 (the headline "delta_mean at
   floor" claim is a linear claim); MLP at whatever the measured per-refit cost supports, with N
   stated explicitly. Never report a p-value at its 1/(N+1) resolution floor.

### The split-gap correction (important, and a real error caught in review)

The original plan proposed a **label-permutation test** for the gene-split-minus-family-split gap.
**That is the wrong instrument**: under a shuffled-label null both scores collapse to the floor, so
the null gap is centred near zero *by construction*. It tests "does leakage exist" — already
answered by the ~40% leakage fraction — and says nothing about the observed gap's sampling
variability. **Use the paired bootstrap instead.**

Consequence: `paired_cluster_bootstrap_diff` needs **two pairing modes**, and they are different
code paths.

- **Same-fold** (ESM-3 vs ESM-2, contrastive vs raw-kNN, conservation vs delta): both arms share a
  fold assignment. Shared cluster subset, identical folds, one resample applied to both arms.
  "Identical folds" holds in this mode only — it is not a blanket rule.
- **Cross-partition** (the split gap): gene-split and family-split are *different CV partitions by
  definition*, so "identical folds" cannot hold. Pairing is across **two fold assignments** —
  resample **families** (the coarser unit; a family resample induces a valid gene resample, not the
  reverse), then recompute each arm under its own partition.

**This is the single most likely thing to be silently implemented wrong.** Written without noticing
the distinction, the cross-partition case gets implemented as the same-fold path and is incorrect.

### Permutation seeds

**Seed 0 only.** A permutation test constructs its own null by shuffling; running across 5 seeds
mostly re-measures fold jitter — precisely what run_biorxiv exists to replace. Cuts the step 5× at no
inferential cost.

---

## 5. Claims requiring a paired difference test

Six, up from the plan's original three:

| Claim | Margin | Report |
|---|---|---|
| ESM-3 seq vs ESM-2 MLP delta_mean (M2) | clears 0.430 threshold by **0.008** | `report_esm3_mechanism.md` |
| Contrastive k-NN vs raw-delta k-NN | +0.041 | `report_contrastive.md` |
| Conservation vs embedding delta (K2) | **+0.002** | `report_geometry.md` |
| Pathogenicity vs mechanism cross-family transfer | 0.85–0.90 vs 0.62–0.64 | `report_geometry.md` |
| Contrastive per-class DN "unmoved" | a **null asserted from a point drop** 0.577 → 0.545 | `report_contrastive.md` |
| Gene-split minus family-split gap | cross-partition mode | `report_classifier.md` |

The 0.008 and 0.002 margins are smaller than a seed of spread — the weakest load-bearing numbers in
the work. The DN claim is subtler: a null asserted from an unmeasured difference, weaker than a thin
positive margin.

A seventh candidate: stability gate **H2** (LEAKY threshold 0.10 on the random→family ρ drop) is
also a gate verdict resting on an untested difference. Lower priority.

---

## 6. Task list (see `PLAN_biorxiv.md` for full detail)

| Task | What | Notes |
|---|---|---|
| **0** | Correctness + methodology decisions (0.0–0.5 above) | 0.0 is a real deliverable; 0.1/0.2 must be written into `docs/EXPERIMENT.md` *before* the run |
| **1** | `paired_cluster_bootstrap_diff` + unit tests | Only new statistical primitive. Tests must assert the shared-resample property directly and cover **both** pairing modes |
| **2** | Wire bootstrap into **7** modules | `mechanism/mlp.py`, `mechanism/contrastive_mechanism.py`, `esm3/esm3_mechanism.py` (phase 3), `pathogenicity/pathogenicity_control.py`, `geometry/run_geometry.py`, `stability/megascale_stability.py`, `mechanism/family_clustering.py` (needs a new `--seeds` flag). Plus a CI on `leakage_fraction`'s ratio. Reference impl: `classify_by_mechanism` |
| **2b** | Homology-partition robustness panel | Promote `clan_holdout.py` + `mmseqs_cluster_holdout.py`. Report the mechanism null and leakage fraction under Pfam family / clan / MMseqs2 (20–30%). Each row's CI resamples that row's own held-out unit |
| **2c** | Production quality as an enforced gate | Trim+pin deps (**remove `wandb`**); add CI running the 38-file suite; build `scripts/compare_runs.py` |
| **3** | Config: `RUN_NAME`→run_biorxiv, widen `PERMUTATION_FEATURES`, clean working tree | **Only after 1, 2 gates pass and CI is green** |
| **4** | Runbook | Done — `RUNBOOK_5.md` written fresh |
| **5** | Remaining stats: AUPRC/PPV/NPV, BH-FDR, minimal-detectable-effect, calibration notes | |
| **6** | Regenerate **16** documents into `reports/run_biorxiv/` | 11 reports + `ESM2_REPORT` + `INTRO_REPORT` + `STATS_PLAN` + 2 new (paired-difference summary, run6→run_biorxiv delta note) |
| **7** | Conservation-residualised mechanism test | Post-run_biorxiv, speculative, pre-mortemed as a likely null. Do **not** let it displace the stats work |

### Notes on specific tasks

**`leakage_fraction` CI** — it is a derived ratio, (gene − family) / (gene − chance), whose
numerator and denominator share the gene-split term and are strongly dependent. **Recompute the
whole ratio once per bootstrap replicate**; do not combine two separate intervals. That ~40% figure
is a headline in both `INTRO_REPORT.md` and `ESM2_REPORT.md` §4 and currently has no interval at
all.

**`compare_runs.py`** — reads both runs' result JSONs, diffs every headline number, flags material
movement (proposed: any headline metric moving more than one run6 seed-std, or any gate verdict
changing). Two reasons it is a script and not prose: (a) run_biorxiv changes error bars, not point
estimates, so any point estimate that moves materially is either a bug introduced by the wiring or a
finding needing explanation — nothing currently catches the former; (b) its output *is* the
delta-note deliverable, generated rather than transcribed, so it cannot drift. Wire a run6-vs-run6
zero-movement invariant into CI.

**Task 2b expected outcome** — run0-era clan-holdout landed *below* the family-split floor
(0.299 ± 0.076 vs 0.352), reading as ~half the family-split signal being clan-level memorisation.
That is more interesting than uniformity and should be reported as such, not smoothed over. MMseqs2
numbers were within ±0.03 of family-split. Both are exploratory numbers on a smaller dataset and
cannot be cited as-is.

---

## 7. Publication readiness (analysis delivered this session)

run_biorxiv is necessary but **not sufficient**. Gaps beyond the statistics:

1. **Beat 4 of the bioRxiv outline has no citable experiment.** ProteinGym ΔLL — described in the
   outline as "the heart," the external validation of the conservation finding — exists only in
   `results/results_0/`. Without it the transferability gradient (conservation → pathogenicity 0.89
   > stability 0.75 > DMS fitness ρ≈0.50 > mechanism ≈ chance) is asserted from two points instead
   of four. **Largest gap; GPU work; not currently in run_biorxiv scope.**
2. **The enzyme-type control is also stranded in `results_0`.** This matters more than its
   supplement placement suggests — see point 4.
3. **Figures.** Seven exist, all supporting beats 2–3, all with seed-std error bars, so **all need
   regenerating against run_biorxiv regardless**. Two new panels named in the outline (ProteinGym
   transferability, contrastive floor lift) do not exist. Suggest a third: a forest plot of the six
   tested differences with CIs — the most reviewer-legible artifact run_biorxiv produces.
4. **Framing risk in the central claim.** The dissociation compares a **3-class macro-F1 at 0.288**
   against a **binary AUROC at 0.89** — different metrics, different tasks, different chance floors.
   "Predicts X but not Y" invites "mechanism is simply the harder task." The strongest existing
   rebuttal is the enzyme control (4-class, family-split F1 0.655, same pipeline) — **which is one
   of the results stranded in `results_0`.** Promoting it would close the strongest structural
   objection to the thesis.
5. **Two structural nulls are underpowered, not negative.** M3 ("structure adds nothing", +0.014 vs a
   0.030 bar) and the 28-family within-family table (6–33 genes each). Task 0.1's underpowered-null
   language and Task 5's minimal-detectable-effect statement are what make these defensible.

---

## 8. Open questions for the user

1. **Do ProteinGym ΔLL and enzyme classification join run_biorxiv?** Both modules are already ported, so it
   is execution not new code; enzyme is CPU-only, ProteinGym needs GPU. Otherwise outline beat 4
   must be rewritten around what exists. **Recommend: yes, include both.**
2. **`INTRO_REPORT.md` location.** Tone is **resolved** — the hype phrasing that violated the
   report-style rule in `CLAUDE.md` has been neutralised, so it now reads consistently with the
   per-experiment reports. Only location remains open: a general-audience explainer in
   `reports/<run>/` alongside the per-experiment record, or in `docs/` beside `linkedin_post.md`?
   It is run-specific, which argues for keeping it. **Recommend: keep in `reports/run_biorxiv/`.**
3. **Does `report_esm3_mechanism_geras.md` carry into run_biorxiv?** Already marked superseded (different
   dataset, a now-fixed data defect in `results/run6/esm3_mechanism/geras/KNOWN_ISSUES.md`).
   **Recommend: drop from run_biorxiv, cite the run6 archive** — regenerating a report that must not be
   cited invites it being cited.
4. **Split-gap CI unit** — gene-resampled headline with a family-resampled sensitivity check, since
   the gap spans two partitions and neither unit is uniquely correct. Confirm.
5. **Stability gate H2** — paired test, or an explicit note that the gate is descriptive?

---

## 9. Known risks

- **Permutation cost is unmeasured.** 4 features × 2 splits × 1,000 = 8,000 refits at seed 0, and
  per-refit cost has never been timed. **Time a single refit on the pod before committing.** This is
  the run's main schedule risk — everything else in run_biorxiv is cheap.
- **Working tree is dirty.** Modified `constants.py`, `paths.py`, all five
  `family_split_baselines_seed*.json`, `reports/run6/INTRO_REPORT.md`; untracked
  `PLAN_biorxiv.md`, `docs/linkedin_post.md`, `docs/connect_runpod.md`, `docs/natera.md`,
  `docs/plan_lora.md`, `experiments/llm_judge/`. Must be committed or set aside **before** the run_biorxiv
  branch point or run6/run_biorxiv provenance is inseparable.
- **Silent no-op wiring.** Adding bootstrap calls that never fire is the failure mode Task 2's
  verification gate exists to catch. Confirm a CI key in real emitted JSON, not just that the code
  runs.
- **`docs/README.md` stale provenance sentence** — see §3. Small fix, high reviewer impact.

---

## 10. Immediate next steps

1. Write Task 0.1 (CI decision rules) and 0.2 (confirmatory/exploratory split) into
   `docs/EXPERIMENT.md`. **These must predate the run** or they are retro-fitted to the intervals.
   `docs/EXPERIMENT.md` already has a "Pre-registered decisions" section at line 166.
2. Fix the stale provenance sentence in `docs/README.md`; mark `result_6.md`'s 0.74–0.88 band
   superseded.
3. Clean the working tree.
4. Implement `paired_cluster_bootstrap_diff` with both pairing modes + unit tests (Task 1).
5. Wire the 7 modules; verify a CI key appears in real output (Task 2).
6. Only then flip `RUN_NAME` and replay `RUNBOOK_5.md`.

**The plan is done. Further planning has negative value — the next action is code.**
