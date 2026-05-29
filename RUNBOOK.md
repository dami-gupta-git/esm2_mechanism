# Full-pipeline rerun runbook

Purpose: regenerate every result from scratch after the 2026-05 bug-fix sweep
(see "Why we are rerunning" at the bottom). Execute stages top-to-bottom.
Within a stage, scripts are order-independent unless noted.

**GPU vs CPU:** only Stage C needs a GPU (RunPod A100/H100). Everything else is
local CPU. ESM-2 embeddings are deterministic — once Stage C is cached you never
need the GPU again unless variant lists change.

All commands use `python -m esm2_mechanism.<module>` from the project root with the
package installed (`pip install -e .`). The old `scripts/` directory no longer exists.

---

## What to delete vs keep before rerunning

Blanket-deleting caches buys almost no correctness — the bugs were in analysis/probe
code, not in the cached artifacts. Deterministic caches rebuild to byte-identical
files, so deleting them just costs API + GPU time. Only delete what a bug actually
touched, plus the outputs you're regenerating anyway.

**Delete / regenerate (genuinely suspect):**
```bash
rm data/enzyme_labels.tsv   # only DATA file a bug corrupted (EC-string parsing) -> Stage A5
rm -rf results/             # outputs you regenerate anyway; scripts overwrite. embeddings are NOT here
```

**Keep (deterministic, untouched by any bug):**
- Raw fetches — `data/sequences.json`, `data/pfam_families.json`,
  `data/alphamissense_scores.json`, ClinVar, the Gerasimavicius `.xlsx`. Slow,
  rate-limited; re-fetching returns identical data.
- ESM-2 embeddings — `data/embeddings/*.npy`. Extraction code was clean and the
  merged cache is verified intact (19,100 × 1280 × 4 arrays). Re-embedding = GPU
  hours for identical bytes.
- Feature tables — `proteome_features_aligned.npy`, `badonyi_features_aligned.npy`,
  `mmseqs_clusters.json`, `megascale_protein_clusters.json`. Their build scripts
  were not in the bug sweep; the bugs were in how downstream scripts *consumed* them.

**The one judgment call:** if "sleep easy" means byte-level proof the embeddings
themselves are clean (not just row-count-correct), wipe `data/embeddings/` and rerun
Stage C on GPU. That is the most expensive step in the pipeline for a near-zero
correctness gain — the Stage-C row-count check is the practical guarantee.

---

## Stage 0 — baseline hygiene

```bash
cd /Users/dgupta/code/portfolio/ESM2/esm2_mechanism

# 1. Commit the bug fixes so the rerun has a clean, reproducible baseline.
git add src/ docs/ requirements.txt
git commit -m "Fix class-index/alignment and CV bugs across probe + fetch scripts"

# 2. Pin the environment.
python3 -m venv .venv && source .venv/bin/activate
pip3 install -e .
```

---

## Stage A — raw data fetch (CPU, network)

These hit external APIs (UniProt, NCBI, MyVariant, OSF). Resume-safe; rate-limited.

**Dependency order matters within Stage A — run in the numbered sequence.**

| Order | Command | Reads | Produces | Notes |
|---|---|---|---|---|
| A1 | `python -m esm2_mechanism.fetch_data.build_merged_gene_list` | `data/downloads/DiseaseMech_Stability_VEPS.xlsx`, `data/downloads/AllG2P.csv` | `data/merged_gene_list.tsv` | Must run first; all other Stage A steps depend on this |
| A2 | `python -m esm2_mechanism.fetch_data.fetch_clinvar_variants` | `data/merged_gene_list.tsv` | `data/clinvar_variants.tsv` | Hits NCBI + UniProt; resume-safe via cache |
| A3 | `python -m esm2_mechanism.fetch_data.build_merged_dataset --data_dir data --out data/merged_variants.json` | `data/gerasimavicius_variants.json`, `data/merged_gene_list.tsv`, `data/clinvar_variants.tsv` | `data/merged_variants.json` | A2 must complete first |
| A4 | `python -m esm2_mechanism.fetch_data.fetch_uniprot_sequences` | `data/merged_variants.json` | `data/cache/uniprot_sequences_extended.json` | A3 must complete first; supplements `data/sequences.json` |
| A5 | `python -m esm2_mechanism.fetch_data.build_proteome_features` | `data/merged_gene_list.tsv`, `data/pfam_families.json` | `data/gene_proteome_features.tsv`, `data/proteome_features_aligned.npy` | Independent of A2–A4; hits gnomAD/Ensembl/HPA/PaxDb |
| A6 | `python -m esm2_mechanism.fetch_data.build_badonyi_features` | `data/cache/badonyi/table_S3.xlsx` (pre-downloaded), `data/merged_gene_list.tsv`, `data/pfam_families.json` | `data/badonyi_features.tsv`, `data/badonyi_features_aligned.npy` | Independent of A2–A4; requires manual download of Badonyi S3 table |

**A6 note:** `data/cache/badonyi/table_S3.xlsx` must be downloaded manually from the Badonyi 2024 paper before running.

---

## Stage B — enzyme labels (CPU, network) — **BUG-FIXED, must rerun**

Depends on `data/merged_valid_variants.json`, which is produced by Stage C2.
Run this after Stage C2 completes and the `.npy` files are scp'd back locally.

```bash
python -m esm2_mechanism.fetch_data.fetch_enzyme_labels
```

Reads: `data/merged_valid_variants.json`
Produces: `data/enzyme_labels.tsv`

**Verify:** spot-check a few rows of the new `enzyme_labels.tsv` against UniProt
EC numbers before trusting result_25. This is the one stage where the *data*, not
just the metric, may have been wrong.

---

## Stage C — embeddings (GPU, RunPod)

See `RUN_EXPERIMENTS.md` for pod setup / scp. Run each in a `tmux` session.
After Stage C, `scp` all `.npy` files back to `data/embeddings/` locally.

`data/sequences.json` must exist (from the Gerasimavicius fetch or existing cache)
before C1/C2. `data/merged_variants.json` (from A2) must exist before C2.

| Order | Command | Reads | Produces | Covers |
|---|---|---|---|---|
| C1 | `python -m esm2_mechanism.embeddings.esm2_mechanism --out_dir results/run_0` | OSF dataset (fetched automatically) | `results/run_0/*.npy`, `results/run_0/final_info.json` | results 1, 2, 7 |
| C2 | `python -m esm2_mechanism.embeddings.extract_merged_embeddings --data_dir data --batch_size 32` | `data/merged_variants.json`, `data/sequences.json` | `data/merged_embeddings_{wt,mut}_{mean,pos}.npy`, `data/merged_valid_variants.json` | merged-dataset results |
| C3 | `python -m esm2_mechanism.embeddings.pathogenicity_control --run_dir .` | ClinVar (fetched automatically), `data/pfam_families.json` | `data/embeddings/emb_*_pathogenicity_*.npy`, `results/pathogenicity_control/summary.json` | results 4, 6 |
| C4 | `python -m esm2_mechanism.embeddings.megascale_stability` | `data/megascale/benchmarks.zip`, `data/pfam_families.json` | `data/embeddings/megascale_{wt,mut}_{mean,pos}.npy`, `results/megascale_stability/*.json` | result_21 |
| C5 | `python -m esm2_mechanism.embeddings.perturbation_scan` | `data/sequences.json`, `data/merged_valid_variants.json` | `data/embeddings/scan_ckpt_*.npy` | result_20 scan |

**C2 note:** pass `--data_dir data` explicitly — the default (`run_0/data`) is wrong
for the current layout. The script writes all 4 `.npy` files before the
`merged_valid_variants.json` sentinel and checks all 5 on resume, so a
crashed/resumed run cannot leave a silently-partial cache.

**C4 note:** this one script both extracts the megascale embeddings (on first run;
resume-skips if the 4 `.npy` exist) **and** runs the stability analysis in the same
invocation — so it produces result_21 directly here, including the bug-fixed H3
projection. It reads `data/megascale_variants.json` and the merged embeddings from
C2, so run it after C2. No separate Stage-D rerun needed for `megascale_stability`.

**C5 note:** `--run_phase 2` for GPU-only embedding extraction; `--run_phase 3` to
rerun just the CPU analysis against cached embeddings.

After Stage C2, run Stage B (fetch_alphamissense + fetch_enzyme_labels) which depend
on `data/merged_valid_variants.json`.

---

## Stage B2 — AlphaMissense scores (CPU, network, large download)

Depends on `data/merged_valid_variants.json` (from C2) and
`data/pathogenicity_valid_variants.json` (from C3).

```bash
python -m esm2_mechanism.fetch_data.fetch_alphamissense
```

Reads: `data/merged_valid_variants.json`, `data/pathogenicity_valid_variants.json`
Produces: `data/alphamissense_scores_full.json`
Note: downloads a 5.3 GB file from Google Cloud Storage on first run.

---

## Stage D — analyses (CPU, local)

Each reads cached embeddings/features. Grouped by independence; run in any order
within a group. **Bold = bug-fixed this sweep, number will change.**

### D1 — mechanism core (results 1–10)
```bash
python -m esm2_mechanism.mechanism.experiment_mlp --family_split        # results 3, 5
python -m esm2_mechanism.mechanism.mut_only_mlp --data_dir data --emb_dir data/embeddings   # result_7 ** (was SyntaxError)
python -m esm2_mechanism.mechanism.family_split_baselines                # result_2
python -m esm2_mechanism.mechanism.family_clustering                     # result_4 ** (was SyntaxError)
python -m esm2_mechanism.mechanism.multiseed_v1                          # result_6 mechanism floor
python -m esm2_mechanism.embeddings.pathogenicity_control                # result_6 control (phase 3)
python -m esm2_mechanism.mechanism.contrastive_mechanism                 # result_9 ** (val-split sizing)
python -m esm2_mechanism.mechanism.clan_holdout                          # result_10
python -m esm2_mechanism.mechanism.mmseqs_cluster_holdout                # result_15/10 ** (per-class AUROC label bug)
```

### D2 — proteome / gene-level (results 11–14)
```bash
python -m esm2_mechanism.proteome.proteome_pilot                         # result_11 ** (predict_proba column align)
python -m esm2_mechanism.proteome.per_gene_ablation                      # result_13 ** (fold-append guard)
python -m esm2_mechanism.proteome.proteome_mechanism                     # result_13
python -m esm2_mechanism.proteome.clinical_utility                       # result_14 (verify; likely no-op)
```

### D3 — within-family + Badonyi (results 15–16)
```bash
python -m esm2_mechanism.badonyi.badonyi_mechanism                       # result_15
python -m esm2_mechanism.badonyi.badonyi_holdout_survival                # result_16
python -m esm2_mechanism.badonyi.within_family_mechanism                 # result_16 ** (concat column offset)
```

### D4 — AlphaMissense / ProteinGym externals (results 17–18, 24)
```bash
python -m esm2_mechanism.alphamissense.alphamissense_family_split        # result_17
python -m esm2_mechanism.alphamissense.proteingym_alphamissense          # result_18
python -m esm2_mechanism.alphamissense.proteingym_esm2_ll                # result_24
```

### D5 — perturbation + stability (results 19–22)
```bash
python -m esm2_mechanism.perturb.perturbation_pattern                    # result_19 (verify; minor tweak)
python -m esm2_mechanism.perturb.perturbation_probe                      # result_20 ** (gene_mask misalignment)
# megascale_stability.py already ran in Stage C4 (embed + analysis + fixed H3) — do NOT rerun here
python -m esm2_mechanism.perturb.megascale_mlp                           # result_21 MLP companion (reads C4 caches, CPU)
python -m esm2_mechanism.perturb.ll_scan                                 # result_22
```

### D6 — transferability synthesis (result_23) + enzyme control (result_25)
```bash
python -m esm2_mechanism.analysis.magnitude_direction                    # result_23
python -m esm2_mechanism.analysis.conservation_axis                      # result_23
python -m esm2_mechanism.analysis.direction_geometry                     # result_23 (verify; single-seed no-op)
python -m esm2_mechanism.analysis.transfer_contrast                      # result_23
python -m esm2_mechanism.analysis.probe4_axis_identity                   # result_23
python -m esm2_mechanism.analysis.enzyme_classification                  # result_25 ** (depends on fixed Stage-B labels)
```

### D7 — esm2_mechanism.py derived metrics
`esm2_mechanism.py` (Stage C1) also emits orthogonality + variance-explained metrics
that were **bug-fixed** — the cosine keys were `GOF_vs_DN|...` but the real dict keys
are `DN_vs_GOF|...`, so these were silently **NaN** before. These live in
`results/run_0/final_info.json` from C1 — no separate rerun, but re-read those fields
when updating result_1/result_23 geometry numbers.

---

## Stage E — ESM-3 (result_26) — ⚠️ RE-VERIFY BEFORE TRUSTING

```bash
python -m esm2_mechanism.embeddings.esm3_mechanism
```

Independent of A–D. Run per `docs/plans/plan_esm3_mechanism.md`.

**result_26 is the project's only positive result (M1✓ M2✓ — "scale lifts
mechanism", ESM-3 0.424 vs ESM-2 0.299). It rests on a fairness assumption that a
just-fixed bug may break:**

- `esm3_mechanism.py`'s fold guard was fixed from `len(set(y_tr)) < 2` to `< 3`
  (skip a fold if any of the 3 classes is missing from train). This matches the
  shared `utils_probes.run_mlp_cv` (`< n_classes`).
- BUT the ESM-2 baseline (0.299) comes from `esm2_mechanism.mechanism.experiment_mlp`,
  which **still uses `< 2`** (lines 140/269/327) — it includes degenerate folds that
  drag F1 down.
- So ESM-2 is scored with degenerate folds *in*, ESM-3 with them *out*. **Part of
  the +0.125 gap may be a fold-eligibility artifact, not scale.**

**Required before result_26 can stand or enter PUBLISH.md:** re-score ESM-2 and
ESM-3 under the *identical* fold rule. Cleanest fix — make `experiment_mlp.py` use
the shared `utils_probes.run_mlp_cv` (or bump its guards to `< n_classes`), re-run
the ESM-2 family-split baseline, then re-run `esm3_mechanism.py` phase 3 and
recompute the M1/M2 gap against the new baseline. Note `esm3_mechanism.py` uses its
own `_run_mlp` rather than the shared util — unifying them removes this whole class
of mismatch.

---

## Stage F — regenerate figures / docs

```bash
python -m esm2_mechanism.analysis.plot results/run_0      # AUROC bars, cosine heatmap, variance
```
Then update each `docs/result_*.md` whose number moved (the `**` rows above) and
re-derive the leakage fraction in `result_leakage_fraction.md` from the fresh
gene-split/family-split F1s.

---

## Verification checklist (the "sleep easy" part)

- [ ] Stage C caches: every `.npy` row-count == `len(valid_variants)` (merged = 19,100 × 1280).
- [ ] Stage-B enzyme labels spot-checked against UniProt.
- [ ] `mmseqs_cluster_holdout` + `proteome_pilot` per-class AUROCs now use canonical GOF/DN/LOF order.
- [ ] result_6 (pathogenicity AUROC) and result_7 (mechanism floor) reproduce their **unchanged** headline numbers — these are the v1 spine and should NOT move. If they do, something in the rerun diverged.
- [ ] result_24 (ProteinGym ESM-2 LL) family-split actually groups proteins (pfam lookup was degenerate — verify non-empty family assignments).
- [ ] result_26 (ESM-3): ESM-2 and ESM-3 scored under identical fold-eligibility rule before reporting the M1/M2 gap (see Stage E).
- [ ] `git status` clean; results committed.

---

## Why we are rerunning (bug provenance)

Three sweeps found ~25 script fixes. Recurring classes:

1. **Class-index / column misalignment** — `LabelEncoder` (alphabetical) vs canonical
   `CLASSES` order; `predict_proba().argmax()` assuming column alignment; cosine-key
   name mismatch (`GOF_vs_DN` vs `DN_vs_GOF`). Affected: `mmseqs_cluster_holdout`,
   `proteome_pilot`, `proteome_mechanism`, `badonyi_mechanism`,
   `within_family_mechanism`, `esm2_mechanism` (cosine keys → were NaN). Scrambled
   per-class AUROCs / orthogonality; macro-F1 mostly survived.
2. **Broken / crashed / never-executed scripts** — `import` inside a
   `from ... import (` block = SyntaxError (`family_clustering`, `mut_only_mlp`;
   results 4, 7); `None`-AUROC format crash (`badonyi_holdout_survival`; result_16).
3. **Row-alignment / index bugs** — fold-append before guard (`per_gene_ablation`),
   gene_mask re-index + Badonyi `[:len]` truncation (`perturbation_probe`, two bugs),
   scale-space projection (`megascale_stability` H3), within-family majority baseline
   over wrong family set (`within_family_mechanism`, second bug),
   merged-checkpoint resume (`extract_merged_embeddings`, `score_esm1v`).
4. **Cross-key lookup failures** — ProteinGym Pfam lookup keyed on UniProt mnemonic
   vs gene symbol → degenerate family-split (`proteingym_esm2_ll`; result_24);
   clan aggregation over unfiltered lists vs `weights` (`clan_holdout`; result_10).
5. **Corrupted labels** — EC-string iteration + multi-class keyword fallback
   (`fetch_enzyme_labels`, two fixes; result_25).
6. **Fold-eligibility mismatch** — ESM-3 `< 2` → `< 3` classes-in-train, now
   matching `utils_probes`, but ESM-2's `experiment_mlp.py` still `< 2`. **Threatens
   the result_26 M1/M2 positive claim** (see Stage E).

The refactor ("refactoring duplicate code" commits) consolidated CV/metric helpers
into `utils_probes.py` (`gene_split_cv`, `family_split_cv`, `compute_metrics`,
`align_proba`, `run_mlp_cv`, `run_logreg_cv`) — exactly the dedup that prevents the
class-order bug class. Remaining gap: `esm3_mechanism.py` (`_run_mlp`) and
`experiment_mlp.py` still carry their own probe loops with the old `< 2` guard;
routing them through `utils_probes.run_mlp_cv` would close the fold-rule mismatch.
