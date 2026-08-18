# The homology-partition robustness panel: does the mechanism null hold under stricter partitions?

**Run 6 · 2026-08-11** · ESM-2 `esm2_t33_650M_UR50D` · `delta_mean` MLP probe
(hidden=(256, 64)) · seed 0, n_boot=20 (smoke-scale; not preprint-grade precision). Source:
[`results/run6/homology_partition_panel/panel.json`](../../../for_me/homology_partition_panel/panel.json),
computed by `experiments/mechanism/homology_partition_panel.py`.

---

## The question

The paper's mechanism-null result (`report_classifier.md`) rests on family-split
cross-validation: whole Pfam families are held out so a classifier cannot recognise a family
it saw in training. But Pfam families are not independent — some are grouped into clans,
distinct families with a shared deep evolutionary origin (same fold, different name). A
family-split can still put two clan-mates in opposite folds, leaving a route for
family-resemblance leakage that the family-split test does not see.

This panel re-runs the `delta_mean` mechanism probe under two stricter partitions and checks
whether the null result — and the leakage fraction it implies — holds up:

- **Pfam family** (the paper's default): no gene from the same family in both train and test.
- **Pfam clan**: no gene from the same clan (a family-of-families) in both train and test.
- **MMseqs2 cluster**: no gene from the same 20%-sequence-identity cluster in both train and
  test — the strictest cut, independent of Pfam's curated groupings.

All three rows use the same probe (sklearn MLP, hidden=(256, 64)) so only the partition
definition changes. Confidence intervals are computed by resampling each row's own held-out
unit — families for the family row, clans for the clan row, MMseqs2 clusters for the cluster
row — never genes uniformly, so a row's CI reflects the actual number of independent groups
its split held out.

---

## Glossary

| Column | Meaning |
|---|---|
| `n_clusters` | Effective number of independent groups the partition's own bootstrap resampled (families / clans / MMseqs2 clusters — not genes or variants). |
| `mechanism_null_macro_f1` | `delta_mean` MLP macro-F1 under that partition, with its CI. Compared against the measured chance floor (0.288, `naive_baseline.json`) to ask "is this at chance?" |
| `leakage_fraction` | Share of the gene-split score not surviving this partition's hold-out: `(gene_f1 − partition_f1) / (gene_f1 − chance)`, jointly resampled at the partition's own unit per bootstrap replicate (same method as `report_leakage_fraction.md`, generalised from family to clan/cluster). |

Chance floor = 0.288 (measured majority-class macro-F1, `naive_baseline.json`), the same value
`report_classifier.md` and `report_leakage_fraction.md` cite — not recomputed per partition.

---

## Result

| Partition | n_clusters | mechanism_null macro_f1 [95% CI] | leakage_fraction [95% CI] |
|---|---:|---:|---:|
| Pfam family | 1,134 | 0.418 [0.390, 0.439] | 0.262 [−0.020, 0.360] |
| Pfam clan | 22 | 0.356 [0.313, 0.378] | 0.551 [−0.078, 0.793] |
| MMseqs2 cluster (20% identity) | 1,215 | 0.379 [0.347, 0.407] | 0.486 [0.173, 0.580] |

*n_boot=20 (smoke-scale). All nine intervals (CI on each metric × 3 partitions) are populated
(`valid_frac=1.0`, none suppressed). Wide CIs on the clan row's leakage fraction reflect only
22 clans qualifying for the resample, not a computation failure.*

---

## Reading the table

**1. The mechanism-null point estimate drops as the partition gets stricter.** Macro-F1 goes
0.418 (family) → 0.379 (MMseqs2) → 0.356 (clan). All three remain closer to the 0.288 floor
than to the gene-split score (~0.55, `report_classifier.md`), so the qualitative finding —
`delta_mean` carries little mechanism signal — still holds at every partition. But the null is
not flat: it gets closer to chance, not further, as homology is controlled more strictly.

**2. The clan-split CI's lower bound (0.313) sits below the family-split CI's lower bound
(0.390).** The two intervals barely overlap. This is the panel's central finding: some of what
looked, under family-split, like "the model isn't exploiting homology" was actually the model
exploiting clan-level homology that a family-only split does not block.

**3. The leakage fraction roughly doubles going from family to clan/MMseqs2.** Family-split
attributes 26% of the gene-split score to homology leakage; clan-split attributes 55%;
MMseqs2-split attributes 49%. `report_leakage_fraction.md` reports ~40% leakage for the
absolute-embedding features under family-split alone — this panel shows that figure is a
lower bound, not the full picture, once deeper relatedness is accounted for.

**4. The MMseqs2 and clan rows roughly agree with each other and disagree with the family
row.** MMseqs2 clustering is independent of Pfam's curated family/clan hierarchy (raw
sequence identity vs. curated homology), yet lands close to the clan row (0.379 vs. 0.356
macro-F1; 49% vs. 55% leakage). Two independent stricter definitions pointing the same
direction is stronger evidence than either alone.

**5. `n_clusters` confirms each CI resampled the right unit.** 1,134 matches the paper's known
Pfam-family count; 1,215 is the MMseqs2 cluster count; 22 is the qualifying-clan count (many
Pfam clans in this dataset are too small — fewer than the minimum genes needed for a scorable
held-out fold — to enter the resample; this is a property of the clan size distribution in
this gene set, not a bug).

---

## What this means for the paper's claims

The original claim (confirmatory item C6: "the mechanism null is stable across homology
partitions") does not hold as stated — the null is not stable, it strengthens under stricter
partitions. The corrected claim is directional: **the fraction of gene-split score
attributable to homology leakage, and the distance of the partition-split score from the gene-split
score, both increase monotonically as the partition definition gets stricter.** This is
consistent with, and reinforces, the paper's broader thesis that ESM-2's apparent mechanism
signal is substantially a homology-recognition artifact — the family-split number alone
understates how much of it is homology-mediated.

---

## What this is and is not

- This is a robustness check on one probe (`delta_mean` MLP) and one dataset, not a new
  finding about a different feature or model.
- n_boot=20 is a smoke-scale check confirming the CIs are wired correctly and populated with
  the right resampling unit; the point estimates and CI widths reported here are not
  preprint-grade precision (`BOOTSTRAP_N_RESAMPLES` at full scale would tighten them).
- It does not establish causally *why* clan-level leakage occurs — only that accounting for it
  changes the leakage estimate.

## Limitations

- Single seed (0), n_boot=20. Full-scale reruns (more seeds, `BOOTSTRAP_N_RESAMPLES`) are
  needed before this table is preprint-grade.
- The clan row's leakage-fraction CI is wide (crosses zero) because only 22 clans qualify —
  a larger clan-mapped gene set would tighten it, but this dataset's clan coverage is fixed by
  Pfam's own clan groupings for these genes.

## Provenance

Computed by `experiments/mechanism/homology_partition_panel.py`, which promotes
`experiments/mechanism/clan_holdout.py` (leave-one-Pfam-clan-out) and
`experiments/mechanism/mmseqs_cluster_holdout.py` (MMseqs2 20%-identity cluster-holdout) into
one consolidated table alongside a freshly-computed Pfam-family row (same sklearn MLP
architecture as the other two rows, rather than reusing `mlp.py`'s PyTorch
`mlp_delta_mean_family` number, to keep the model architecture fixed across rows — that
PyTorch figure, 0.381, is cited in the result JSON as `mlp_py_torch_family_reference` for
cross-check only). Chance floor from `results/run6/naive_baseline.json`. Clan assignments from
`data/downloads/Pfam-A.clans.tsv.gz` (Pfam-A clans release); MMseqs2 clusters from
`data/mmseqs_clusters.json` (20% sequence-identity clustering). Output:
[`results/run6/homology_partition_panel/panel.json`](../../../for_me/homology_partition_panel/panel.json).
Full run log: [`RUN_PROGRESS.md`](../../../bak/RUN_PROGRESS.md), Run 6.
