# Plan: In-silico perturbation scan for mechanism prediction

## Numbering note

The ClinVar-pattern analysis (perturbation_pattern.py) is written up as `docs/result_19.md` (result_17 = AlphaMissense ClinVar, result_18 = AlphaMissense ProteinGym). This plan references that work as **result_19**. The in-silico scan result will be **result_20** if it passes the decision rules.

---

## The finding that motivates this

**result_19** (perturbation_pattern.py) showed that the *spatial pattern* of observed clinical variant deltas across a gene's sequence carries mechanism signal that survives family-split CV. Eight scalar features derived from how variants cluster in ESM-2 embedding space push GOF AUROC from 0.578 to 0.646 and combined F1 from 0.331 to 0.399 under family-split — with almost no leakage (gene-split ≈ family-split for the scalar features).

The biological reason: GOF mutations have to hit specific hotspots (KRAS G12, BRAF V600) to flip a protein from off to on. LOF mutations can break the protein anywhere. DN mutations cluster at interfaces. This hotspot-vs-spread distinction shows up in ESM-2 perturbation space.

## The problem with result_19

The pattern features are built from clinical variants in ClinVar/Gerasimavicius. This has two problems:

1. **Coverage bias:** a gene with 50+ variants gets a meaningful pattern; a gene with 3 gets noise.
2. **Circularity:** ClinVar variants are not random — they're enriched for the known hotspots. We see clustering partly because clinicians already found the hotspots.

The fix: replace clinical variants with a systematic in-silico scan — same positions for every gene, unbiased by what's been reported.

---

## The experiment

Mutate a uniform sample of positions in each gene to 3 probe amino acids, extract ESM-2 deltas, and build pattern features from the scan. Every gene gets the same coverage regardless of how many clinical variants it has.

### Sampling strategy

Full saturation mutagenesis is infeasible (~20M forward passes). Practical approach:

- **100 evenly-spaced positions** per gene (all positions if length < 200)
- **3 probe amino acids:** Ala (structural tolerance), Asp (charge sensitivity), Trp (steric tolerance)
- 1,985 genes × 100 positions × 3 substitutions ≈ 600k forward passes
- ~2–3 hours on A100 80GB at batch_size=128

### Pre-registered feature set (5 features — primary analysis)

These 5 features are pre-registered before running Phase 2. They represent the minimum necessary set that is well-motivated and not overfit:

| Feature | Definition | Biological motivation |
|---|---|---|
| `scan_mag_mean` | Mean \|\|delta\|\| across all positions and substitutions | Overall ESM-2 perturbability of the gene |
| `scan_mag_cv` | Std / mean of magnitudes | High = concentrated hotspots (GOF-like); low = uniform (LOF-like) |
| `scan_hotspot_fraction` | Fraction of positions with magnitude > mean + 1σ | Direct hotspot density measure |
| `scan_pc1_var` | Variance explained by PC1 of the N×1280 delta matrix | How much one direction dominates the perturbation space |
| `scan_sub_variance` | Mean across positions of variance(Ala_mag, Asp_mag, Trp_mag) | Position-level substitution sensitivity — is sensitivity structural or residue-specific? |

**Note on PC features:** only the scalar variance fraction is used, not the PC direction vectors. PC directions on 100×1280 with n=100 are unstable. Restrict to scalars from the PCA spectrum.

### Ablation feature set (additional 4, evaluated separately after primary)

Only run if primary set passes the decision rule. Treated as exploratory, not pre-registered:

- `scan_mag_skew` — skewness of magnitude distribution (long tail = few dominant hotspots)
- `scan_hotspot_spacing_cv` — how evenly spaced are the hotspots (clustered together vs distributed)
- `scan_top5_range` — positional range of top-5 hotspots / protein length
- `scan_pc1_pc2_ratio` — PC1/PC2 eigenvalue ratio (how dominant is the top direction)

---

## Decision rule (pre-registered)

All thresholds set before Phase 2 runs.

| Gate | Threshold | Interpretation |
|---|---|---|
| G1: scan-only vs result_19 ClinVar-pattern baseline | Scan F1 > 0.348 + **0.02** = **0.368** | Unbiased scan adds meaningful signal over clinical variant pattern |
| G2: scan + delta vs result_19 combined baseline | Combined F1 > 0.399 + **0.02** = **0.419** | Scan adds beyond mean-pooled delta when combined |
| G3: scan + proteome vs proteome alone | Combined F1 > 0.385 + **0.02** = **0.405** | Scan adds beyond proteome features |

A +0.005 lift technically passes but should be treated as noise. The +0.02 threshold is the minimum to be scientifically interesting at this sample size (n~1,985 genes, 5-fold family-split).

If G1 fails: scan features don't improve on the ClinVar-pattern baseline — stop, do not proceed to G2/G3.
If G1 passes but G3 fails: scan adds to the delta baseline but not beyond proteome — report as a methodological curiosity, not a practical advance.
If G3 passes: scan features are a useful addition to the best current model — write up as result_20.

---

## Why this matters

The current best predictor (Badonyi + proteome, result_15, F1=0.511) uses pre-computed structural features from external tools. The perturbation scan is pure-sequence — no structure, no external tools. If it adds signal beyond proteome features from sequence alone, it suggests ESM-2 implicitly encodes structural sensitivity information that can be extracted by probing the representation rather than by explicit structure prediction.

It also generalises to the dark proteome — genes with no known structure, no AlphaFold model, no clinical variants. The scan runs on any sequence.

---

## Implementation plan

### Phase 1: Script (1 day, local)
Write `scripts/perturbation_scan.py`:
- Load merged gene list; fetch sequences for G2P-only genes not in sequences.json (check coverage first)
- For each gene: sample 100 positions evenly, apply Ala/Asp/Trp substitutions via apply_missense
- Extract ESM-2 650M mean-pooled WT/mutant embeddings for each probe
- Compute the 5 pre-registered features + 4 ablation features per gene
- Cache scan features to `data/scan_features_1985genes.npy` and per-gene delta tensors optionally

### Phase 2: Embedding extraction (GPU, ~3 hours, A100 80GB)
- ~600k forward passes, batch_size=128
- Checkpoint every 200 genes — re-runnable without restarting from scratch

### Phase 3: Probe runs (CPU, ~1 hour)
- Logistic regression, family-split CV, 5 seeds
- Feature sets: scan-only, scan+delta, scan+proteome, scan+Badonyi
- Report per-seed results and 5-seed mean ± std

### Phase 4: Analysis and write-up (half day)
- Check decision rules G1–G3
- If G1 passes: per-class breakdown (which mechanism benefits most?)
- Feature importance from logistic regression coefficients
- Write result_20.md if G3 passes; add a note to result_19.md otherwise

---

## Sequence coverage check (do before Phase 1)

sequences.json has 948 Gerasimavicius genes. The merged dataset has 1,985 genes; the 1,037 G2P-only genes need sequences. fetch_uniprot_sequences.py exists — run it on the G2P-only gene list and check how many UniProt sequences are available. If coverage < 90%, restrict the scan to the covered subset and note it.

---

## Files

| File | Status |
|---|---|
| `scripts/perturbation_pattern.py` | ✓ exists — ClinVar-pattern analysis (result_19) |
| `results/perturbation_pattern/results.json` | ✓ exists |
| `docs/result_19.md` | ✓ written |
| `scripts/perturbation_scan.py` | ✗ to be written (Phase 1) |
| `data/scan_features_1985genes.npy` | ✗ Phase 2 output |
| `results/perturbation_scan/` | ✗ Phase 3 output |
| `docs/result_20.md` | ✗ written only if G3 passes |
