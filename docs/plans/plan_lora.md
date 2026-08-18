# Plan: LoRA fine-tuning of ESM-2 650M for mechanism classification

## One-line summary

We LoRA fine-tuned the top layers of ESM-2 650M to classify GOF/DN/LOF under the
same family-split CV as the frozen probes, testing whether mechanism signal
exists in ESM-2 but is simply unreadable by frozen probes. We ran the identical
recipe on the ClinVar pathogenicity task as a positive control, confirming any
mechanism null reflects the representation rather than a training-setup failure.

## Why

The whole project rests on a frozen-probe null (mechanism family-split macro-F1
≈ floor; pathogenicity AUROC ≈ 0.90). A frozen probe cannot distinguish "the
signal is not in ESM-2" from "the signal is in ESM-2 but not linearly/MLP
readable from the mean-pooled delta." LoRA fine-tuning is the minimal experiment
that separates those two. The report already names this as the natural next step
("What I would do next" → end-to-end fine-tuning).

Pre-registered framing (decide before looking at results):
- The mechanism null **strengthens** if LoRA stays near the frozen family-split
  floor (0.38–0.45) — the signal genuinely is not recoverable.
- The null is **qualified** if LoRA clears the floor by a pre-registered margin
  on family-split — frozen probing was the bottleneck.
- Either outcome is publishable; we commit to reporting both honestly.

## Key design decision: HuggingFace, not fair-esm

The existing extraction path uses **fair-esm** (`utils/embed.py`), which returns
a bare PyTorch module with no clean PEFT integration. For LoRA we use
HuggingFace `transformers` (`facebook/esm2_t33_650M_UR50D`, identical weights to
`esm2_t33_650M_UR50D`) where `peft` attaches via `get_peft_model`. This is
isolated to the new training module; nothing in the frozen-probe pipeline
changes. The frozen results (`results/run6/`) remain the comparison baseline.

We must verify weight-identity once: confirm the HF model's mean-pooled WT
embedding matches the cached fair-esm `embeddings_wt_mean.npy` on a sample of
variants (cosine ≈ 1.0) before trusting any HF-derived number. If they diverge,
stop and reconcile (tokenizer/pooling difference) before training.

## Pre-registered gates (set before running)

Matched baselines (family-split, 5-seed, from `results/run6/`):
- Mechanism frozen floor: MLP delta_mean family-split macro-F1 = **0.380**.
- ESM-3 1.4B reference: family-split macro-F1 = **0.453** (Section 6).
- Pathogenicity frozen: delta_mean MLP family-split AUROC = **0.894** (Section 3).

| Gate | Task | Criterion | Reads as |
|---|---|---|---|
| L1 | mechanism | family-split macro-F1 > 0.430 (floor + 0.05) | LoRA beats frozen ESM-2 |
| L2 | mechanism | gene-split − family-split gap reported, not just peak | quantifies leakage amplification |
| L3 | pathogenicity (control) | family-split AUROC ≥ 0.85 | training pipeline recovers known signal |
| L4 | mechanism | family-split macro-F1 > 0.453 | LoRA beats ESM-3 scale |

L3 is the load-bearing control: if it fails, the mechanism null is uninterpretable
(training setup is broken), so L3 is checked first.

## Data and splits (all reused, unchanged)

- Mechanism variants + labels: `loaders.load_mechanism_variants()` →
  GOF/DN/LOF via `MECHANISM_CLASSES` (`constants.py`). 17,826 variants.
- Pathogenicity variants: `data/clinvar_pathogenicity_variants.json` (cached,
  37,218 variants) — reuse `pathogenicity_control._fetch_clinvar` if cache missing.
- Sequences: `SEQUENCES_JSON` (WT and mutant constructed as in current extraction).
- CV folds: `splits.family_split_cv(genes, pfam_map)` and `gene_split_cv(genes)`,
  `pfam_map` from `data/pfam_families.json`. **Same fold function the frozen
  probes use** — this is what makes the comparison matched.
- Seeds: `N_SEEDS = 5` (`constants.py`), seeds 0..4.

Class imbalance (LOF 76%) is handled with class-weighted cross-entropy
(weights = inverse class frequency computed **per train fold**, never global).

## Model and training

New module: `src/esm2_mech/experiments/lora/lora_finetune.py`.

- Base: HF `facebook/esm2_t33_650M_UR50D`, `EsmForSequenceClassification` head
  (3-class for mechanism, 2-class for pathogenicity).
- LoRA via `peft`: target the query/key/value/output projections of the **top 4
  transformer layers only** (layers 29–32 of 33); freeze everything else +
  embeddings. Rank r=8, alpha=16, dropout=0.1 (pre-registered; not tuned on the
  test split). Trainable-parameter count logged.
- Input: per-variant we feed the **mutant sequence** (the head reads the full
  perturbed protein). Rationale: the frozen null is specifically about the
  mean-pooled *delta*; LoRA on the mutant sequence is the more powerful test —
  if even end-to-end mutant-sequence fine-tuning can't beat the floor, the null
  is strong. (A delta-style twin-tower variant is out of scope for v1; noted in
  "what this is not.")
- Pooling: mean over residue representations → classification head (matches the
  frozen `*_mean` feature so the comparison is apples-to-apples on pooling).
- Optimizer: AdamW, lr 1e-4 on LoRA params, weight decay 0.01, linear warmup
  10%, max ~10 epochs with **early stopping on a family-disjoint validation
  split carved from train** (never the test fold) — mirrors the MLP early-stop
  discipline and the project rule against tuning on test.
- Batch: dynamic by token length; gradient accumulation to a fixed effective
  batch. Mixed precision (bf16) on A100.
- Determinism: seed set per fold; trainable-param count and final epoch logged.

### Leakage discipline (project-critical)

- Standardize/normalize nothing across the train/test boundary.
- Class weights, early-stop validation split, and LoRA all fit **inside** each
  train fold. The test fold is touched only at scoring.
- No imputation/fallback values anywhere (CLAUDE.md data-integrity rules).
- A NaN per-fold metric (e.g. AUROC on a fold missing a class) is filtered with
  `utils.metrics.mean_std_n`, never `np.mean`.

## Metrics and outputs

- Per fold: `metrics.compute_metrics(y_true, y_pred, y_proba)` → macro-F1 +
  per-class one-vs-rest AUROC. Same util the frozen probes use.
- Aggregate per seed → across seeds with `seed_aggregation.aggregate_across_seeds`
  and `metrics.mean_std_n`.
- Both gene-split and family-split run for every seed (L2 needs the gap).

### Paths (add to `utils/paths.py` FIRST, per CLAUDE.md)

- `LORA_RESULTS_DIR = RESULTS_DIR / "lora"` (i.e. `results/run6/lora/`).
- Per-seed files: `lora_mechanism_seed{0..4}.json`,
  `lora_pathogenicity_seed{0..4}.json`.
- Aggregates: `lora_mechanism_aggregate.json`, `lora_pathogenicity_aggregate.json`.
- Naming helpers added next to the existing `seed_result_filename` in
  `constants.py` rather than hardcoded inline.

## Execution (RunPod A100, per CLAUDE.md tmux rule)

1. Add `peft` to `pyproject.toml`; `pip install peft transformers accelerate`.
2. On RunPod, inside a tmux session.
3. Weight-identity check (HF vs cached fair-esm embeddings) — gate before training.
4. Pathogenicity control first (L3). If it fails, stop and debug.
5. Mechanism run, 5 seeds, gene- + family-split.
6. After **each seed**, write that seed's result file (CLAUDE.md multi-seed rule).
7. Aggregate; write the two `*_aggregate.json`.

## Reporting

- New report section drafted to match existing run-report style
  (`reports/run6/`): Summary → what was measured → gates table → results table →
  "Reading the table" → what this is and is not → Provenance.
- Headline numbers must trace to `results/run6/lora/*.json`.
- "What this is not": single LoRA recipe (r=8, top-4 layers), not a
  hyperparameter sweep; mutant-sequence head, not a delta twin-tower; gene-level
  labels (the standing granularity caveat) still apply.

## Steps

1. `utils/paths.py`: add `LORA_RESULTS_DIR`; `constants.py`: add LoRA result-file
   name helpers + LoRA hyperparameter constants (r, alpha, dropout, layers, lr).
2. `pyproject.toml`: add `peft`, `accelerate`.
3. `experiments/lora/hf_model.py`: HF model + LoRA wrapper + tokenizer; weight-
   identity check util vs cached fair-esm embeddings.
4. `experiments/lora/lora_finetune.py`: per-fold train/eval loop, class weights,
   early stop on family-disjoint val, returns per-fold metrics. CLI with
   `--task {mechanism,pathogenicity}`, `--split {gene,family}`, `--seeds`.
5. `experiments/lora/aggregate.py` (or reuse `seed_aggregation`): write aggregates.
6. Run on RunPod (control → mechanism), write per-seed files.
7. Draft `reports/run6/report_lora.md`; update `docs/README.md` result index.

## Out of scope (v1)

- Hyperparameter sweep over rank/layers/lr.
- Delta twin-tower (WT and mutant through shared LoRA, subtract) — possible v2.
- ESM-3 LoRA.
- Conservation-residualised LoRA (the Section 7 follow-up hypothesis) — possible v2.
