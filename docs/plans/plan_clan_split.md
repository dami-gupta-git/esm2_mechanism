# Plan: Clan-disjoint CV — does the mechanism signal survive a coarser homology partition?

## The finding that motivates this

The family-split CV (result_1 / run_biorxiv) established that a linear probe on ESM-2 delta embeddings does not classify disease mechanism (GOF / DN / LOF) above chance once Pfam families are held out. The leakage fraction shows that most of the gene-split signal is family recognition, not transferable mechanism signal.

That test partitions on Pfam families. Families within the same **Pfam clan** share a common evolutionary origin — distant homology that a masked language model trained on sequence could still pick up. If two families in the same clan land in different folds, the model might generalise across them not because it learned mechanism but because it learned a shared fold or remote-homology signature.

## The open question

> **Does the family-split result change when the partition unit is coarsened from family to clan?**

If the family-split macro-F1 is at chance and the clan-split macro-F1 is also at chance, the conclusion is the same but the evidence is stronger: the signal does not survive even when clans are held out. If the clan-split score rises above the family-split score, the family split was too aggressive — it was withholding information the model can legitimately generalise from, and the finding needs qualification.

---

## Background: Pfam clans

A Pfam clan groups families whose profile HMMs detect significant similarity (HHsearch p < 1e-3). Clans are the coarsest grouping in Pfam. The mapping is distributed as `Pfam-A.clans.tsv.gz` from the EBI FTP site (columns: pfam_acc, clan_id, clan_name, pfam_name, pfam_description). Not every family belongs to a clan; clan-less families are singletons in this hierarchy.

## The subset problem

Only about a third of Pfam families in our dataset map to a clan. Restricting to clan-annotated genes produces a smaller, biased subset skewed toward well-studied superfamilies. This means:

- The chance floor (majority-class macro-F1) must be recomputed on the clan subset.
- Coverage (how many variants / genes / families survive the restriction) must be reported.
- Any comparison between family-split and clan-split must either (a) run both on the same clan-restricted subset, or (b) note the subset difference and interpret accordingly. Option (a) is cleaner.

---

## The experiment

### Clan-disjoint k-fold CV

Analogous to `family_split_cv`: shuffle the unique clans, split them into k folds, assign all families (and therefore all genes and variants) within each clan to the same fold. Genes whose family has no clan are excluded from both train and test (same convention as unannotated genes in `family_split_cv`).

### Comparisons

All on the clan-annotated subset, 5 seeds, 5 folds.

1. **Clan-split macro-F1 and per-class AUROC** for each feature (delta_mean, wt_only_mean, etc.), with clan-resampled bootstrap CIs.
2. **Family-split on the same subset** — rerun family-split CV restricted to the clan-annotated genes, so the comparison is apples-to-apples.
3. **Clan-split leakage fraction:** `LF_clan = (gene_F1 − clan_F1) / (gene_F1 − chance_subset)`, with clan-resampled CI.
4. **Paired difference:** family-split vs clan-split on the shared OOF predictions, with a paired bootstrap CI.

---

## Decision rules (pre-registered)

All on clan-subset macro-F1, 5-seed mean, delta_mean feature. `chance_subset` is the majority-class macro-F1 on the clan-annotated subset.

| Gate | Condition | Interpretation |
|---|---|---|
| **D1** | clan-split macro-F1 CI includes `chance_subset` | mechanism signal does not survive clan partition — consistent with the family-split null, but stronger |
| **D2** | clan-split macro-F1 CI is entirely above `chance_subset` | some transferable signal survives even at clan level — the family-split null may be too conservative |
| **D3** | clan-split − family-split difference CI includes 0 | no detectable difference between partitions — the family-level result already captures the homology effect |

The headline outcome:
- **D1 true:** the mechanism null holds at clan level. Strengthens the existing claim. Report as a robustness paragraph in the mechanism write-up.
- **D2 true, D3 true:** clan-split detects signal but not significantly more than family-split. The family partition was already sufficient. Report both numbers.
- **D2 true, D3 false (clan > family):** this would mean the family split was destroying legitimate cross-family signal. Unlikely given the family-split result is already near chance, but if it happens, it changes the interpretation and requires a dedicated section.

---

## Implementation plan

### Phase 0 — data: build gene-to-clan map

Parse `Pfam-A.clans.tsv.gz` into a `{gene: clan_id}` lookup by chaining through the existing `pfam_families.json` (gene → pfam_acc) and the clan TSV (pfam_acc → clan_id). Record:
- How many of the dataset's genes have a clan assignment.
- How many unique clans are represented.
- Class distribution on the clan-annotated subset.

Output: an in-memory map, not a cached file (the mapping is deterministic from two existing files).

### Phase 1 — splits: add `clan_split_cv` to `splits.py`

A new function following the same signature pattern as `family_split_cv`:

```
clan_split_cv(genes, pfam_map, clan_map, n_folds=5, seed=42)
```

where `clan_map` is `{pfam_acc: clan_id}`. Internally: gene → pfam_acc (via pfam_map) → clan_id (via clan_map). Shuffle unique clans, split into k folds, assign all genes in each clan to the same fold. Genes with no clan excluded from both sides. Minimum train/test sizes same as `family_split_cv`.

### Phase 2 — probe: add clan-split arm to `mechanism_delta_family_split.py`

For each feature, run `run_probe_on_splits` a third time with clan splits. Compute clan-resampled bootstrap CIs (resample clans, not families or genes). Store results under a `"clan_split"` key alongside `"gene_split"` and `"family_split"`.

Also rerun the family-split arm on the clan-restricted subset so the comparison is controlled. Store under `"family_split_clan_subset"`.

### Phase 3 — leakage: extend `leakage_fraction.py`

Add `leakage_fraction_per_feature` for the clan split, using the clan-subset chance floor. Add `leakage_fraction_ci` with clan-unit resampling.

### Phase 4 — paired difference

Compute paired OOF diff (clan-split minus family-split on the clan subset) using the existing `paired_oof_diff` machinery, resampling at the clan level.

### Phase 5 — report

A section in the mechanism report (or a standalone companion) showing:
- Subset coverage (variants, genes, families, clans).
- Clan-subset chance floor.
- Table: gene-split / family-split / clan-split macro-F1 with CIs, all on the clan subset.
- Leakage fractions at family and clan level.
- Paired difference CI.
- Interpretation keyed to gates D1–D3.

---

## Files

| File | Status |
|---|---|
| `data/downloads/Pfam-A.clans.tsv.gz` | exists (553 KB) |
| `data/pfam_families.json` | exists |
| `src/esm2_mech/utils/splits.py` | modify: add `clan_split_cv` |
| `src/esm2_mech/utils/paths.py` | already has `PFAM_CLANS_TSV_GZ` |
| `src/esm2_mech/experiments/mechanism/mechanism_delta_family_split.py` | modify: add clan-split arm |
| `src/esm2_mech/experiments/mechanism/leakage_fraction.py` | modify: add clan-level LF |
| `results/run_biorxiv/family_split_baselines_seed{N}.json` | modify: add `clan_split` key |
| `results/run_biorxiv/leakage_fraction.json` | modify: add clan-level entries |
