# Full-pipeline rerun runbook

Purpose: regenerate every result from scratch after the 2026-05 bug-fix sweep
(see "Why we are rerunning" at the bottom). Execute stages top-to-bottom.
Within a stage, scripts are order-independent unless noted.

**GPU vs CPU:** only Stage C needs a GPU (RunPod A100/H100). Everything else is
local CPU. ESM-2 embeddings are deterministic — once Stage C is cached you never
need the GPU again unless variant lists change.

Legend: `[confirm]` = script inferred (early result docs predate the inline-script
convention); verify the entry point before trusting the number.

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
git add scripts/ docs/ requirements.txt
git commit -m "Fix class-index/alignment and CV bugs across probe + fetch scripts"

# 2. Pin the environment.
python3 -m venv .venv && source .venv/bin/activate   # or your existing env
pip3 install -r requirements.txt
```

---

## Stage A — raw data fetch (CPU, network)

These hit external APIs (UniProt, NCBI, MyVariant, OSF). Resume-safe; rate-limited.

| Order | Command | Produces | Notes |
|---|---|---|---|
| A1 | `python3 scripts/fetch_uniprot_sequences.py` | `data/sequences.json`, `data/pfam_families.json` | foundation for everything |
| A2 | `python3 scripts/build_merged_dataset.py` | `data/merged_variants.json`, `data/merged_gene_list.tsv` | reads `AllG2P.csv` + Gerasimavicius |
| A3 | `python3 scripts/fetch_clinvar_variants.py` | `data/clinvar_variants.tsv` | pathogenicity control variants |
| A4 | `python3 scripts/fetch_alphamissense.py` | `data/alphamissense_scores.json` | needs merged/pathogenicity variant lists first |
| A5 | `python3 scripts/fetch_enzyme_labels.py` | `data/enzyme_labels.tsv` | **BUG-FIXED** (EC-string parsing) — must rerun |

**Verify A5:** spot-check a few rows of the new `enzyme_labels.tsv` against UniProt
EC numbers before trusting result_25. This is the one stage where the *data*, not
just the metric, may have been wrong.

---

## Stage B — derived feature tables (CPU)

| Order | Command | Produces | Result |
|---|---|---|---|
| B1 | `python3 scripts/build_proteome_features.py` | `data/gene_proteome_features.tsv`, `data/proteome_features_aligned.npy` | result_12 |
| B2 | `python3 scripts/build_badonyi_features.py` | `data/badonyi_features.tsv`, `data/badonyi_features_aligned.npy` | result_15 |

---

## Stage C — embeddings (GPU, RunPod)

See `RUN_EXPERIMENTS.md` for pod setup / scp. Run each in a `tmux` session.

| Order | Command | Produces | Covers |
|---|---|---|---|
| C1 | `python3 scripts/experiment.py --out_dir results/run_0` | Gerasimavicius WT/mut/pos `.npy` + `run_0/final_info.json` | results 1, 2, 7 |
| C2 | `python3 scripts/extract_merged_embeddings.py` | `data/embeddings/merged_embeddings_*.npy` | merged-dataset results | 
| C3 | `python3 scripts/pathogenicity_control.py` (phase 2) | `emb_*_pathogenicity_*.npy` | results 4, 6 |
| C4 | `python3 scripts/megascale_stability.py` | `data/embeddings/megascale_{wt,mut}_{mean,pos}.npy` **+** `results/megascale_stability/{summary,per_protein_spearman,h3_stability_projection}.json` | result_21 |
| C5 | `python3 scripts/perturbation_scan.py` | `data/embeddings/scan_ckpt_*.npy` | result_20 scan |

**C4 note:** this one script both extracts the megascale embeddings (on first run;
resume-skips if the 4 `.npy` exist) **and** runs the stability analysis in the same
invocation — so it produces result_21 directly here, including the bug-fixed H3
projection. It reads `data/megascale_variants.json` and the merged embeddings from
C2, so run it after C2. No separate Stage-D rerun needed for `megascale_stability`.

**C2 note:** the `extract_merged_embeddings.py` fix now writes all 4 `.npy` files
*before* the `valid_variants.json` sentinel and checks all 5 on resume — so a
crashed/resumed run can no longer leave a silently-partial cache. Pull all `.npy`
back to `data/embeddings/` after the run.

After Stage C, `scp` the `.npy` files back locally; the rest is CPU.

---

## Stage D — analyses (CPU, local)

Each reads cached embeddings/features. Grouped by independence; run in any order
within a group. **Bold = bug-fixed this sweep, number will change.**

### D1 — mechanism core (results 1–10)
```bash
python3 scripts/experiment_mlp.py --family_split        # results 3 [confirm], 5
python3 scripts/mut_only_mlp.py --data_dir data --emb_dir data/embeddings   # result_7  ** (was SyntaxError)
python3 scripts/family_split_baselines.py               # result_2
python3 scripts/family_clustering.py                    # result_4  ** (was SyntaxError)
python3 scripts/multiseed_v1.py                         # result_6 mechanism floor
python3 scripts/pathogenicity_control.py                # result_6 control (phase 3)
python3 scripts/contrastive_mechanism.py                # result_9  ** (val-split sizing)
python3 scripts/clan_holdout.py                         # result_10
python3 scripts/mmseqs_cluster_holdout.py               # result_15/10  ** (per-class AUROC label bug)
```

### D2 — proteome / gene-level (results 11–14)
```bash
python3 scripts/proteome_pilot.py                       # result_11 ** (predict_proba column align)
python3 scripts/per_gene_ablation.py                    # result_13 ** (fold-append guard)
python3 scripts/proteome_mechanism.py                   # result_13
python3 scripts/clinical_utility.py                     # result_14 (verify; likely no-op)
```

### D3 — within-family + Badonyi (results 15–16)
```bash
python3 scripts/badonyi_mechanism.py                    # result_15
python3 scripts/badonyi_holdout_survival.py             # result_16
python3 scripts/within_family_mechanism.py              # result_16 ** (concat column offset)
```

### D4 — AlphaMissense / ProteinGym externals (results 17–18, 24)
```bash
python3 scripts/alphamissense_family_split.py           # result_17
python3 scripts/proteingym_alphamissense.py             # result_18
python3 scripts/proteingym_esm2_ll.py                   # result_24
```

### D5 — perturbation + stability (results 19–22)
```bash
python3 scripts/perturbation_pattern.py                 # result_19 (verify; minor tweak)
python3 scripts/perturbation_probe.py                   # result_20 ** (gene_mask misalignment)
# megascale_stability.py already ran in Stage C4 (embed + analysis + fixed H3) — do NOT rerun here
python3 scripts/megascale_mlp.py                        # result_21 MLP companion (reads C4 caches, CPU)
python3 scripts/ll_scan.py                              # result_22
```

### D6 — transferability synthesis (result_23) + enzyme control (result_25)
```bash
python3 scripts/magnitude_direction.py                  # result_23
python3 scripts/conservation_axis.py                    # result_23
python3 scripts/direction_geometry.py                   # result_23 (verify; single-seed no-op)
python3 scripts/transfer_contrast.py                    # result_23
python3 scripts/probe4_axis_identity.py                 # result_23
python3 scripts/enzyme_classification.py                # result_25 ** (depends on fixed A5 labels)
```

### D7 — experiment.py derived metrics
`experiment.py` (Stage C1) also emits orthogonality + variance-explained metrics
that were **bug-fixed** (cosine-key mismatch, `np.var` axis). These are in
`run_0/final_info.json` from C1 — no separate rerun, but re-read those fields when
updating result_1/result_23 geometry numbers.

---

## Stage E — ESM-3 (in flight, separate)

`scripts/esm3_mechanism.py` is its own 3-phase pipeline (AF2 download → GPU embed →
probes) writing result_26. Independent of A–D. Run per
`docs/plans/plan_esm3_mechanism.md`.

---

## Stage F — regenerate figures / docs

```bash
python3 scripts/plot.py results/run_0      # AUROC bars, cosine heatmap, variance
```
Then update each `docs/result_*.md` whose number moved (the `**` rows above) and
re-derive the leakage fraction in `result_leakage_fraction.md` from the fresh
gene-split/family-split F1s.

---

## Verification checklist (the "sleep easy" part)

- [ ] Stage C caches: every `.npy` row-count == `len(valid_variants)` (merged = 19,100 × 1280).
- [ ] A5 enzyme labels spot-checked against UniProt.
- [ ] `mmseqs_cluster_holdout` + `proteome_pilot` per-class AUROCs now use canonical GOF/DN/LOF order.
- [ ] result_6 (pathogenicity AUROC) and result_7 (mechanism floor) reproduce their **unchanged** headline numbers — these are the v1 spine and should NOT move. If they do, something in the rerun diverged.
- [ ] `git status` clean; results committed.

---

## Why we are rerunning (bug provenance)

Two sweeps found ~18 uncommitted script fixes. Three recurring classes:

1. **Class-index / column misalignment** — `LabelEncoder` (alphabetical) vs canonical
   `CLASSES` order, and `predict_proba().argmax()` assuming column alignment.
   Affected: `mmseqs_cluster_holdout`, `proteome_pilot`, `within_family_mechanism`,
   `experiment` (cosine keys). Scrambled per-class AUROCs; macro-F1 mostly survived.
2. **Broken/never-executed scripts** — `import` inside a `from ... import (` block =
   SyntaxError in `family_clustering` and `mut_only_mlp` (results 4, 7).
3. **CV/index bugs** — fold-append before guard (`per_gene_ablation`), gene_mask
   re-index (`perturbation_probe`), scale-space projection (`megascale_stability` H3),
   truncation/label-alignment, and corrupted EC labels (`fetch_enzyme_labels`).

Recommended follow-up (not blocking): extract a single audited per-class metric
helper into `utils_probes.py` so class-order bugs can't recur (CLAUDE.md
"check for duplication" rule).
