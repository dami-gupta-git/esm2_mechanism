# Delta note: run6 to run_biorxiv

`scripts/compare_runs.py run6 run_biorxiv` was run on 2026-08-20 with its default absolute
threshold of 0.005. It compared 5,647 shared scalar leaves and flagged 2,090 movements. The new run
also contains 13,501,880 added leaves, chiefly out-of-fold predictions and new confidence-interval,
permutation, provenance, and paired-difference fields; 10,559 run6-only leaves were removed.

The generated movement flags divide into the groups below. Each flag belongs to exactly one group.

| Result group | Flags | Explanation |
|---|---:|---|
| Main mechanism experiment | 1,767 | The mechanism cohort changed from 17,826 variants across 1,935 genes and 1,134 Pfam families to 17,770 variants across 1,931 genes and 1,144 families after the ClinVar refresh and annotation rebuild. The run also replaced run6's fold-jitter error bars with fold-aware point estimates, family-cluster bootstrap intervals, paired differences, and the registered feature-specific permutation tests. The permutation count increased from 200 to 1,000. These changes account for the flagged counts, scores, uncertainty values, cluster counts, and permutation values in `aggregate.json`, `family_split_baselines_seed{0..4}.json`, `nonlinear_results_seed{0..4}.json`, `family_clustering.json`, `leakage_fraction.json`, and `naive_baseline.json`. |
| Gerasimavicius-only mechanism subset | 230 | The subset retained the same 10,138 variants and 942 genes, while updated Pfam annotations changed the family count from 660 to 666. The same fold-aware, family-bootstrap, paired-difference, and permutation changes applied. This accounts for every movement under `single_source_gerasimavicius/`. |
| Pathogenicity control | 52 | The current ClinVar snapshot and corrected selection contract replaced the run6 cohort of 37,218 variants across 1,929 genes with 24,384 variants across 1,802 genes. Run6 contained 18,815 pathogenic and 18,403 benign variants; run_biorxiv contains 12,192 of each. Protein substitutions are deduplicated before balancing, conflicting labels are excluded, each gene is capped and balanced, and the surviving variants are rebalanced after sequence and position filtering. All 52 flags are cohort counts or AUROCs computed on these different cohorts. |
| Geometry experiment | 41 | Three flags record the pathogenicity cohort change from 37,218 to 24,384 variants. The other 38 are mechanism or pathogenicity probe values that inherit the refreshed mechanism cohort, updated Pfam assignments, and corrected pathogenicity cohort. They occur in `magnitude_direction/probe_results.json`, `geometry_results.json`, `probe4_axis_identity.json`, and `conservation_axis.json`. |

The shared Megascale stability outputs had no material point-estimate movement. Enzyme classification
has no run6 counterpart, so its outputs appear only as added keys. No flagged movement remains
unexplained.

The full raw expansion is not retained because the added out-of-fold arrays produce a 1.3 GB Markdown
file. It can be regenerated with the command above; this note records all movement groups and their
counts.
