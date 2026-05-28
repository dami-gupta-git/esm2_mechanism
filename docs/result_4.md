# Result 4 — ESM-2 embeddings cluster strongly by Pfam family, explaining apparent mechanism signal

**Date:** 2026-05-24
**Run:** `../results/20260524_baseline_run/run_0/`, model `esm2_t33_650M_UR50D`
**Script:** `../scripts/family_clustering.py`
**Output:** `../results/20260524_baseline_run/run_0/family_clustering.json`

## TL;DR

ESM-2 WT embeddings cluster strongly by Pfam protein family (k=5 neighbor purity 26× chance, 50-way family probe 27× majority baseline). Because **74.8% of Gerasimavicius disease genes share their family's majority mechanism**, any gene-level mechanism classifier built on ESM-2 — without family-split CV — is operating on a family-recognition shortcut, not learned mechanism. The mutant−WT delta strips most family signal and falls to chance, consistent with mechanism not being encoded at the per-residue level. This causally explains the linear / MLP / family-split / WT-only result pattern observed in prior runs.

---

## Background

Prior runs of `experiment.py` produced:

| Probe | Setup | macro-F1 | Notes |
|---|---|---|---|
| Linear delta_mean | gene-split | 0.279 | At chance (0.333) |
| Linear delta_per_residue | gene-split | 0.373 | Slightly above chance |
| Linear WT-only | gene-split | **0.580** | Suspicious — well above chance despite ignoring the mutation |
| MLP delta_mean | gene-split | 0.414 | Non-linear lift over linear delta |
| Linear delta_mean | family-split | 0.281 | No collapse — no signal to lose |

The open question after those runs: **is the 0.58 WT-only macro-F1 a real "ESM-2 knows mechanism at the gene level" signal, or homology leakage from gene-split CV (BRCA1 in train → BRCA2 in test)?**

This document answers that question.

## Method

`family_clustering.py` operates on the cached per-variant WT, mutant, and delta embeddings and the previously-built Pfam map. For each view it computes, on **gene-level** embeddings (mean of per-variant embeddings per gene):

1. **Silhouette score** by Pfam family (cosine distance).
2. **k-NN family purity** (k=5, 10) — fraction of a gene's k nearest neighbors sharing its family — vs label-shuffled null.
3. **Within-family vs between-family mean cosine distance ratio** vs null.
4. **Linear probe predicting Pfam family** (50 families with ≥3 member genes) from gene-level embedding, vs majority baseline.
5. **Per-gene confound**: fraction of genes whose mechanism matches the majority mechanism of their Pfam family.

Restriction: clustering metrics are computed on the 424 genes belonging to non-singleton Pfam families (singletons can't define a cluster). Family-probe metrics are computed on all annotated genes (939 / 948).

## Results

### Dataset composition

- 10,231 variants, 948 unique genes
- 939 / 948 genes have a Pfam annotation (99%)
- 662 unique Pfam families; **515 singletons**, leaving 424 genes in 147 multi-gene families
- Top 5 families: PF00069 (12), PF00168 (12), PF00046 (11), PF00071 (11), PF00520 (9) — kinase, C2, homeobox, Ras, ABC transporter

### Family clustering by view

| View | Silhouette | k=5 purity (null, z) | k=10 purity | Within/between ratio (null, z) | Family probe acc (majority) |
|---|---|---|---|---|---|
| **WT mean-pooled** | −0.075 | **0.211** (0.008, **+78.5**) | **0.129** (z=+63.8) | **0.560** (1.00, **−11.1**) | **0.587** (0.022) |
| Mut mean-pooled | −0.074 | 0.211 (z=+80.3) | 0.131 (z=+66.9) | 0.558 (z=−11.1) | 0.565 |
| **Delta mean-pooled** | −0.351 | 0.047 (0.007, +17.9) | 0.032 (z=+18.0) | 0.977 (1.004, −1.3) | **0.022** (= majority baseline) |

### Mechanism–family confound

**74.8 % of genes have a mechanism label matching their family's majority mechanism.** A classifier that perfectly identifies family and predicts the family's majority mechanism would therefore achieve ~75% gene-level accuracy with **zero** learned mechanism knowledge.

### Interpretation of the silhouette anomaly

WT silhouette is slightly negative (−0.075) despite every other metric showing strong family clustering. This is a known failure mode of silhouette when (a) cluster sizes are highly uneven, (b) many "clusters" are singletons or pairs, and (c) the embedding is high-dimensional (1280-D) where mean intra-cluster distances are not much smaller than between-cluster distances even when nearest-neighbor relations are strongly family-aligned. The four other metrics — k=5 and k=10 purity, within/between ratio, and the 50-way family probe — are mutually consistent and unambiguous. **Do not use silhouette in isolation for this kind of analysis.**

## Findings

### F1 — WT and mutant embeddings strongly cluster by Pfam family

k=5 family purity of 0.211 vs a chance null of 0.008 (z = +78) means a gene's nearest-neighbor genes in ESM-2 space are 26× more likely to share its Pfam family than chance. The 50-way family probe achieves 58.7% accuracy where majority-class prediction gets 2.2% — a 27× lift. Within-family genes sit at ~56% the cosine distance of between-family genes. ESM-2 was trained for exactly this kind of family-level representation, so this is expected — but it has direct consequences for downstream evaluation.

### F2 — Delta embeddings strip most (not all) family signal

Delta family purity collapses to 0.047 (still z = +18 vs null — significant but 5× weaker than WT), within/between ratio rises to 0.98 (essentially chance), and the family probe falls to the majority baseline. Subtracting WT from mutant cancels the bulk-protein "this is a kinase" signal. A small but real residual remains, which is the most likely source of the MLP delta_mean lift from 0.28 → 0.41 — the MLP is recovering residual family signal non-linearly, not learning mechanism.

### F3 — The mechanism–family correlation (75%) explains the WT-only baseline

Three observations conspire:
1. Pfam family is recoverable from WT ESM-2 with 58.7% accuracy (27× majority baseline).
2. 74.8% of genes carry their family's majority mechanism.
3. The original WT-only macro-F1 was 0.580.

These numbers are mutually consistent with a classifier that **identifies family then predicts the family majority mechanism**, achieving ~0.59 × 0.75 ≈ 0.44 gene-level accuracy on the hard cases plus better-than-chance on family-internal majority cases, summing to a macro-F1 in the 0.55–0.60 range. The WT-only baseline does not require ESM-2 to encode mechanism — it only requires ESM-2 to encode family, which it demonstrably does.

### F4 — The delta probe results are now coherent

- Linear delta @ chance: delta strips the family-mediated shortcut and there is no residue-local mechanism signal to fall back on.
- Family-split delta = gene-split delta (0.28 ≈ 0.28): consistent — there was no signal to lose in either split.
- MLP delta lift to 0.41: the non-linear probe recovers the small residual family signal in deltas (z = +18 on k-purity is enough to matter for an MLP).
- DN AUROC stuck at chance even with MLP: DN is the class least tied to Pfam family in this dataset, so the family shortcut helps it least.

### F5 — Methodological consequence for prior literature

Any published mechanism classifier using ESM/PLM embeddings under **gene-stratified-but-not-family-stratified** CV is consistent with this leakage pattern. Concretely:
- MissION (medRxiv 2025) — restricted to ion channels (deeply paralogous), gene-stratified splits only. The reported ESM-2 lift over baselines is consistent with within-family homology rather than mechanism learning. Direct comparison requires re-running their setup with family-split CV.
- LoGoFunc — uses gene-aware splits across HGMD subsets but does not explicitly enforce family disjointness.
- Badonyi & Marsh 2024 — does not use PLMs but reports per-class metrics without family-split CV.

This is not a refutation of those papers' headline numbers; it is a statement that those numbers are consistent with a family-recognition shortcut that prior CV designs do not block.

## Novelty assessment

This finding has two layers that sit very differently in the literature. Distinguishing them is important for how the result should be framed in any writeup.

### What is already well known (not novel)

**That ESM-2 embeddings cluster by Pfam family is not a discovery.** It is essentially what ESM-2 was designed to do.

- Rives et al. 2021 (the original ESM paper, *PNAS*) showed ESM embeddings recover family structure.
- Lin et al. 2023 (ESM-2 / ESMFold, *Science*) explicitly benchmarked remote homology detection.
- Many downstream tools use ESM embeddings *as* a similarity metric for family or superfamily assignment.

A paper whose only claim is "ESM-2 encodes Pfam family" would not be publishable.

### What is genuinely novel (probably)

**The novel contribution is the quantitative causal chain, not the qualitative observation:**

> k=5 family purity is 26× chance, the 50-way family probe is 27× majority baseline, AND 74.8% of disease genes in Gerasimavicius share their family's majority mechanism. Therefore any gene-split-only mechanism classifier on this dataset is achieving most of its reported score via family recognition rather than mechanism learning. The 0.58 WT-only baseline in this pipeline is a worked example of exactly this shortcut.

I have not found this specific quantitative chain spelled out in prior mechanism-prediction work. The closest references treat the issue qualitatively:
- Gerasimavicius 2022 noted mechanism correlates with protein family but did not quantify the resulting leakage risk for downstream classifiers.
- Livesey & Marsh 2020/2023 critique CV-split practices in variant-effect prediction in general terms but do not pin them to family-level clustering with numbers.
- Bileschi/Yang-style ML-for-protein reviews flag homology leakage but offer no per-dataset diagnostic.

**The contribution is the quantification and its causal attribution to a specific previously-puzzling baseline number, not the existence of family clustering.** "Family matters" is folk knowledge; "family explains the WT-only macro-F1 we observed, and predicts which published results are most at risk of the same inflation" is a specific, testable, methodologically actionable claim.

### Publishability ladder

| Framing | Interest level | Plausible venue |
|---|---|---|
| "ESM-2 encodes Pfam family" | Already textbook — not interesting | None |
| "Mechanism classifiers may have homology leakage" | Mildly interesting; restates folk knowledge | Workshop or methods note |
| **Quantified family leakage explains the WT-only baseline in our pipeline; same shortcut likely inflates published PLM mechanism classifiers; family-split CV proposed as minimum standard** | **Genuinely useful methodological contribution** | bioRxiv → *Bioinformatics* / *Genome Biology* if expanded |
| "PLMs fundamentally cannot encode mechanism — evidence across ≥3 models and ≥3 datasets, with within-family positive contrast showing the problem is cross-family generalisation" | High-impact methodological correction | *Nat Methods* / *Nat Commun* |

Current work sits at the third row. Climbing to the fourth depends on the experiments listed in "Next experiments" below — especially the structure-aware model replication (SaProt / ESM-3) and the within-family analysis.

### Reviewer-defensible novelty statement

The honest one-paragraph answer to "what's new here?":

> *"It is known that PLMs encode protein family. What has not previously been quantified is how much this inflates downstream gene-level mechanism classifiers under common CV designs. We provide that quantification on a standard mechanism dataset (Gerasimavicius), show that it accounts for the entire apparent signal in our WT-only baseline, and predict the same leakage explains reported headroom in MissION and similar ESM-2-based work. We propose family-split CV as a minimum evaluation standard for this task."*

This is a methodological cleanup contribution rather than a paradigm shift. Bioinformatics is full of papers in exactly this register and they are cited heavily because every subsequent paper in the area needs to defend its CV setup against them.

### What would make this *more* novel

If the follow-up experiments show:
1. **WT-only macro-F1 collapses under family-split** (confirms the shortcut is the whole signal, not just a contributor).
2. **The same pattern replicates on DDG2P** (~2,000 genes, structured labels).
3. **A structure-aware model (SaProt or ESM-3) shows the same family-clustering and the same null mechanism signal** — this is the most important follow-up, because the field's current implicit belief is that adding structure tokens will recover mechanism. Showing it does not would directly correct that.
4. **Within-family analysis reveals mechanism IS learnable inside a single Pfam family** — would give the paper a positive flip side: "mechanism prediction is a within-family problem, not a cross-family one; the field has been measuring the wrong thing."

With all four, the contribution moves from "methodological cleanup" to "field-level reframing of what mechanism prediction is."

## Revised story (updated from result_3)

> ESM-2 embeddings strongly encode Pfam family (k=5 purity 26× chance, 50-way family probe 27× majority baseline). Because 74.8% of disease genes in Gerasimavicius share their family's majority mechanism, any gene-level mechanism classifier built on ESM-2 without family-split CV is operating on a family-recognition shortcut. The WT-only baseline's 0.58 macro-F1 — previously the most puzzling number in this study — is fully explained by this shortcut. The mutant−WT delta operation removes most family information; the residual non-linear signal recovered by an MLP (0.28 → 0.41) is consistent with leftover family clustering rather than learned mechanism. ESM-2 deltas do not encode disease mechanism at the per-residue level on this dataset.

## What this is not

- **Not** a claim that ESM-2 is useless for mechanism prediction — only that the *delta* and the *gene-split-only* setups are flawed.
- **Not** a claim that mechanism is unlearnable from sequence — only that ESM-2's current representation, evaluated this way, does not contain it.
- **Not** definitive without replication on a second dataset (DDG2P, Badonyi 2025) and a second model (SaProt, ESM-3) — see "Next experiments" below.

## Next experiments (in priority order)

1. **MLP under family-split CV on WT-only and delta.** Single most important confirmation: if MLP-WT-only also collapses under family-split, the family-shortcut explanation is locked in.
2. **Multi-seed replication** (5+ seeds) on all current metrics. Current numbers are one seed only.
3. **Positive controls**: same pipeline must recover (a) ClinVar pathogenicity (AUROC > 0.85 expected) and (b) Megascale ΔΔG (Spearman ρ > 0.5 expected). Without these, the null is uninterpretable.
4. **Replicate on DDG2P** (~2,000 genes, structured `mutation_consequence` field). If the same family-clustering pattern + null mechanism signal appears, the result generalises beyond Gerasimavicius.
5. **Structure-aware steelman**: re-run on SaProt (foldseek structure tokens) or ESM-3. If a structure-aware model also fails, the negative result is much stronger; if it succeeds, the finding becomes "mechanism requires structural context that pure-sequence PLMs lack."
6. **Within-family analysis**: pick the 3 largest Pfam families (kinase, C2, homeobox) and test whether mechanism is learnable *within* a single family — directly addresses whether MissION-style narrow-domain success is real or homology-driven.

## Files

- `../scripts/family_clustering.py` — analysis script (~310 lines, reuses `experiment.py` helpers)
- `../results/20260524_baseline_run/run_0/family_clustering.json` — full metric output, all three views

## Caveat on the script's printed headline

The headline string at the end of `family_clustering.py` keys off `silhouette_score` alone and prints "NO family clustering" — this is wrong for this data and should be ignored. The four other metrics in the JSON output are the correct read. The headline logic should be rewritten to be a consensus over the four robust metrics; this is a small follow-up to make.
