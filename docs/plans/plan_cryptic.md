# Plan: family-split evaluation of PLM-based cryptic pocket prediction

## The pitch in one sentence

ESM-2 3B per-residue embeddings + a shallow probe match or beat PocketMiner on cryptic-pocket prediction (Bloore et al. AI4D3 2023; CryptoBench Bioinformatics 2025) — but nobody has tested whether the signal survives Pfam-family holdout. We do, and we expect it to be a mosaic: families with conserved cryptic-pocket architectures (kinases' DFG-out, GPCR allosteric sites) survive, novel folds collapse.

## Why this fits

1. The probing methodology (frozen PLM + linear/shallow head + family-split CV) is exactly what we already do for mechanism, GO, and pathogenicity.
2. Cryptic pocket prediction is currently hot. The two best PLM-based papers (ESP workshop, CryptoBench journal) use frozen embeddings — same setup as our pipeline.
3. The honest open question — "does this signal generalise across Pfam families or has the model memorised which families have well-studied cryptic pockets" — has not been asked. Reviewers will ask it.

## What is known

- **PocketMiner (Meller et al., Nat Commun 2023).** GNN + short MD. The structural-method baseline.
- **ESP (Bloore et al., NeurIPS AI4D3 workshop 2023).** Frozen ESM-2 3B and 15B + simple heads. AUROC 0.93 vs PocketMiner 0.87 on PM's test set. Workshop paper, not main-conference, but a real result. (https://ensemtx.com/wp-content/uploads/2023/12/Protein_language_models_AI4D3-final.pdf)
- **CryptoBench (Polák et al., Bioinformatics 2025).** New benchmark + per-residue evaluation. Frozen ESM-2 3B + 3-layer MLP. AUROC 0.88 vs PocketMiner 0.76 on CB-PM. Also beats P2Rank. (https://academic.oup.com/bioinformatics/article/41/1/btae745/7927823)
- **PickPocket (OpenReview).** ESM-2 + GearNet structural encoder. Doesn't isolate the embedding contribution; not directly relevant to our framing.

Open: no published evaluation uses Pfam-family-disjoint or clan-disjoint CV. CryptoBench uses sequence-identity clustering only.

## Hypothesis

**H1 (primary).** Per-residue cryptic-pocket AUROC drops materially under Pfam-family-split CV on CryptoBench, by enough to put the PLM-probe approach back into competitive (rather than dominant) range vs PocketMiner.

**H2 (per-family heterogeneity, the interesting part).** The drop is non-uniform.
- *Survives:* protein families with well-characterised cryptic-pocket architecture and many training examples (kinases' DFG-out, GPCR helix-VIII allosteric sites, β-lactamases, possibly Hsp90).
- *Collapses:* orphan folds and families with few cryptic-pocket examples.

This is the same per-target decomposition pattern we found in result_18 for AlphaMissense on ProteinGym (Tsuboyama mini-proteins collapse, classic disease genes survive).

**H3 (size matters).** The 650M ESM-2 we already have cached is likely insufficient. ESP and CryptoBench both used 3B. 650M may collapse to near-baseline. We test both to characterise the size effect.

## Experimental design

### Data

- **CryptoBench dataset** from the Bioinformatics 2025 paper. Train + test splits, per-residue cryptic-pocket labels. Available via the paper's supplementary or their GitHub.
- **CryptoSite** (Cimermancic et al., 2016) as a held-out evaluation set, exactly as ESP and PocketMiner use it.
- **Pfam-family annotations** for every protein in both sets. Use InterProScan (already known in our pipeline) or pre-computed Pfam → UniProt mapping.

### Embeddings

- **Primary:** ESM-2 3B (`esm2_t36_3B_UR50D`) per-residue embeddings. Mean-pooled is not enough for residue-level prediction — we need per-position vectors.
- **Secondary (ablation):** ESM-2 650M (already cached for some proteins). Establishes the size dependence.
- **Skip:** 15B, on the bet that 3B is the inflection point and 15B adds compute without changing the family-split conclusion.

GPU cost estimate: a few thousand proteins, average ~400 residues, ESM-2 3B. On an A100 80GB this is 2–6 hours depending on batching. Embeddings cache to disk; subsequent probe runs are CPU.

### Probe

Match the published setups so the comparison is honest:
- **Linear probe** on per-residue embeddings — the strictest "what does the representation encode" test.
- **3-layer MLP** (CryptoBench's setup) for the headline number.
- Class imbalance handling via `class_weight='balanced'`.

### Splits

1. **Sequence-identity holdout** (CryptoBench's default; e.g. MMseqs2 50%): reproduces published numbers.
2. **Pfam-family-disjoint holdout** (new): groups proteins by Pfam family, holds entire families out per fold.
3. **Pfam-clan-disjoint holdout** (stricter): groups by clan. Some Pfam families share a clan with others in the training set even if held out by family.

The three-rung ladder mirrors our existing methodology and answers "is the PLM-probe advantage over PocketMiner real, or does it reflect a benchmark-distribution affinity that family-split removes."

### Metrics

- **Per-residue AUROC and AUPRC** — primary, matches CryptoBench.
- **Per-protein top-k pocket residue recovery** — secondary, more practically useful.
- **Per-family AUROC distribution** — the result-17/result-18-style stratification. The headline contribution.
- **Comparison to PocketMiner** at each split. PM is run pre-trained, same as ESP did, with the caveat documented honestly.

## Implementation phases

### Phase 1 — data + embeddings (~1 week)
1. Fetch CryptoBench dataset and CryptoSite. Parse residue-level labels. (~1 day, CPU.)
2. Pfam-annotate all proteins via InterProScan or precomputed mapping. (~1 day.)
3. Extract ESM-2 3B per-residue embeddings. (~1 day on A100, GPU.)
4. Cache to `data/cache/cryptic_embeddings/`.

### Phase 2 — probes + splits (~3 days, CPU)
1. Train linear and 3-layer MLP probes under sequence-identity holdout. Reproduce CryptoBench's headline.
2. Re-run under Pfam-family-disjoint CV. Compute per-family AUROC distribution.
3. Re-run under Pfam-clan-disjoint CV.
4. Run pre-trained PocketMiner on the same evaluation subsets for comparison.

### Phase 3 — 650M ablation (~1 day)
1. Same pipeline at ESM-2 650M. Establishes the size effect.

### Phase 4 — writeup (~1 week)
1. Per-family AUROC distribution scatter plots, sequence-identity vs family-split vs clan-split.
2. Interpretation of survivor / collapse families.
3. Comparison to PM at each split.
4. `docs/result_19.md`.

### Total timeline: ~3 weeks.

## What we expect to find — honest prior

Pattern from result_17 (ClinVar tight) → result_18 (ProteinGym wide):

| Split | Expected per-protein AUROC distribution |
|---|---|
| Sequence-identity holdout (CryptoBench default) | Mean 0.85–0.90, tight (matches paper) |
| Pfam-family-disjoint | Mean 0.75–0.82, wider, heavy tail on novel folds |
| Pfam-clan-disjoint | Mean 0.70–0.78, wider still |

If this holds, the headline is:

> *PLM-probe cryptic-pocket prediction looks dominant on sequence-identity-clustered evaluations because the training distribution has many well-studied cryptic-pocket families. Stratified by Pfam family, the advantage over PocketMiner narrows substantially or disappears for families outside the well-characterised distribution. Like AlphaMissense on ProteinGym, the headline performance number is conditional on the benchmark distribution.*

That's a meaningful contribution: it does not invalidate ESP / CryptoBench, but it pins down what their numbers mean operationally.

## What could go wrong

1. **3B embeddings might be too expensive at scale.** Mitigation: start with the CryptoBench test set only (~1,100 structures), defer training-set embedding extraction if not needed for the probe under our holdouts.
2. **Family-split could be trivially destructive.** Sample size per Pfam family is small in the CryptoSite-style benchmarks; family-disjoint folds may have very few held-out examples. Mitigation: report per-family AUROC only for families with ≥5 positive residues; analyse the rest as aggregated tail.
3. **PocketMiner re-run requirements.** Their codebase needs MD trajectories. Run only on the held-out subsets, not the full benchmark. Mitigation: use their reported numbers where re-running is impractical.
4. **The "cryptic-ness" label is itself ambiguous.** Different benchmarks define cryptic differently (apo→holo conformational change ΔRMSD threshold, hands-on annotation, etc.). Mitigation: report all results on CryptoBench's definition primarily; CryptoSite as secondary.

## Open questions before launch

1. **GPU access.** ESM-2 3B per-residue embedding extraction needs a real GPU (A100 or H100). Confirm availability before committing.
2. **PocketMiner re-run scope.** Do we re-run PM on held-out subsets, or rely on their reported numbers? Re-running is the cleanest comparison but adds ~1 week of setup.
3. **Should we include ESM-2 15B?** It is the largest model and was tested in ESP. 15B is ~22 GB of model weights; embedding extraction is ~3× slower than 3B. Decision: skip in v1, add as an ablation if 3B vs 650M shows a meaningful gap.
4. **Publication venue.** Bioinformatics methods note matches the existing project framing. NeurIPS / ICLR workshop is feasible if reframed as a probing-methodology paper. Decision deferrable.
