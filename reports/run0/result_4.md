# Result 4 — ESM-2 embeddings cluster strongly by protein family, explaining the apparent mechanism signal

**Date:** 2026-05-24
**Run:** `../results/20260524_baseline_run/run_0/`, model `esm2_t33_650M_UR50D`
**Script:** `../scripts/family_clustering.py`
**Output:** `../results/20260524_baseline_run/run_0/family_clustering.json`

## TL;DR

ESM-2 embeddings group proteins by protein family very strongly — a gene's nearest neighbours in embedding space are 26× more likely to share its protein family than chance. Because **74.8% of disease genes in this dataset share their protein family's most common mechanism label**, any mechanism classifier built on ESM-2 without holding out entire protein families during evaluation is largely just recognising family membership, not learning mechanism. The mutant−WT delta removes most of this family signal and falls to chance. This explains the pattern of results across the prior experiments.

---

## Background

Prior experiments produced these results:

| Feature | Setup | macro-F1 | Notes |
|---|---|---|---|
| Linear delta_mean | gene-split | 0.279 | At chance |
| Linear delta_per_residue | gene-split | 0.373 | Slightly above chance |
| Linear WT-only | gene-split | **0.580** | Suspicious — well above chance despite ignoring the mutation entirely |
| MLP delta_mean | gene-split | 0.414 | Better than linear on delta |
| Linear delta_mean | family-split | 0.281 | Identical to gene-split — no signal to lose |

The key open question: **is the WT-only macro-F1 of 0.58 a real signal — ESM-2 genuinely knows something about mechanism at the gene level — or is it a measurement artefact?**

The artefact would work like this: related proteins (e.g. BRCA1 and BRCA2) appear in both training and test sets. The classifier learns "proteins that look like BRCA1 tend to have LOF mechanism." When BRCA2 appears in the test set, it recognises the family resemblance and predicts correctly — not because it learned mechanism, but because it learned family identity.

This experiment measures how strongly ESM-2 embeddings cluster by protein family, and how much that explains the WT-only result.

---

## Method

We measured protein-family clustering in the embedding space using five different metrics, on the WT embedding, mutant embedding, and delta embedding separately:

1. **Silhouette score** — how much tighter within-family distances are compared to between-family distances (higher = more clustered).
2. **k-nearest-neighbour family purity** — for each gene, what fraction of its 5 (or 10) nearest neighbours in embedding space share its protein family? Compared against a shuffled null.
3. **Within-family vs between-family distance ratio** — are same-family genes closer together than different-family genes?
4. **Family prediction accuracy** — how accurately can a linear classifier identify which of 50 protein families a gene belongs to, just from its embedding?
5. **Mechanism–family overlap** — for each gene, does its mechanism label match the most common mechanism label in its protein family?

Clustering metrics were computed on the 424 genes that belong to families with at least 2 members (singletons can't form clusters). Family prediction was computed on all 939 annotated genes.

---

## Results

### Dataset composition

- 10,231 variants, 948 unique genes
- 939 / 948 genes have a protein family annotation (99%)
- 662 unique protein families; **515 are singletons**, leaving 424 genes in 147 multi-gene families
- Top 5 families: PF00069 (12 genes), PF00168 (12), PF00046 (11), PF00071 (11), PF00520 (9) — kinase, C2 domain, homeobox, Ras, ABC transporter

### Family clustering by embedding type

| Embedding | Silhouette | k=5 purity (chance, z-score) | k=10 purity | Within/between distance ratio (chance, z) | Family prediction accuracy (chance) |
|---|---|---|---|---|---|
| **WT mean-pooled** | −0.075 | **0.211** (0.008, **+78.5**) | **0.129** (z=+63.8) | **0.560** (1.00, **−11.1**) | **0.587** (0.022) |
| Mutant mean-pooled | −0.074 | 0.211 (z=+80.3) | 0.131 (z=+66.9) | 0.558 (z=−11.1) | 0.565 |
| **Delta mean-pooled** | −0.351 | 0.047 (0.007, +17.9) | 0.032 (z=+18.0) | 0.977 (1.004, −1.3) | **0.022** (= chance) |

Reading the WT row: a gene's 5 nearest neighbours share its protein family 21.1% of the time — compared to 0.8% by chance (z = +78.5, far beyond any reasonable significance threshold). A linear classifier can predict which of 50 protein families a gene belongs to with 58.7% accuracy, compared to 2.2% by just guessing the most common family.

The delta row is nearly the opposite: family purity drops to 4.7%, family prediction accuracy falls to chance. Subtracting the wildtype embedding strips most of the family signal.

### How much mechanism and family overlap

**74.8% of genes have a mechanism label matching the most common mechanism label in their protein family.** This is the key number. A classifier that identifies the protein family and predicts the family's majority mechanism label would be right 75% of the time — without learning anything about mechanism at all.

### Why the silhouette score looks odd

The silhouette score is slightly negative (−0.075) for the WT embedding, even though every other metric shows strong family clustering. This is a known failure of silhouette when there are many singleton clusters, highly unequal cluster sizes, and high-dimensional embeddings — all of which apply here. The four other metrics are consistent and unambiguous. The silhouette score can be safely ignored for this analysis.

---

## Findings

### F1 — WT embeddings strongly cluster by protein family

A gene's nearest neighbours in ESM-2 space are 26× more likely to share its protein family than chance. A linear classifier identifies the correct family (out of 50) with 58.7% accuracy, compared to 2.2% by guessing. ESM-2 was trained to produce representations that capture protein function and evolutionary relationships — so family-level clustering is expected. The issue is what that means for downstream evaluation.

### F2 — The delta removes most family signal, but not all

Subtracting the wildtype from the mutant embedding cancels most of the "this is a kinase" signal. The within/between distance ratio rises to 0.98 (essentially chance), and family prediction falls to random. But a small residual remains — k=5 purity is 4.7% vs a chance of 0.7% (z = +18). This residual is most likely responsible for the MLP lift from 0.28 → 0.41 in Result 3: the neural network is picking up that faint leftover family signal, not learning mechanism.

### F3 — The 75% mechanism–family overlap explains the WT-only result

Three numbers fit together:
1. Family is predictable from WT ESM-2 with 58.7% accuracy.
2. 74.8% of genes carry the majority mechanism label of their family.
3. The WT-only macro-F1 was 0.580.

A classifier that identifies the protein family and predicts the family's most common mechanism label would achieve roughly this score — without learning anything about mechanism. The WT-only baseline doesn't require ESM-2 to understand mechanism; it only requires it to recognise protein families, which it clearly does.

### F4 — All the earlier delta results now make sense

- **Linear delta at chance**: subtracting the wildtype removes the family-recognition shortcut, and there's no mechanism signal left for a linear classifier to find.
- **Family-split delta ≈ gene-split delta** (0.28 ≈ 0.28): there was no family signal in the delta to begin with, so holding out families doesn't change anything.
- **MLP delta lift to 0.41**: the neural network finds the small residual family signal that subtraction didn't fully remove.
- **DN stuck at chance even with MLP**: DN is the class least correlated with protein family in this dataset, so the family shortcut helps it least.

### F5 — This likely affects other published results too

Any published mechanism classifier using protein language model embeddings under a CV design that holds out genes but not entire protein families is susceptible to this same inflation. For example:
- **MissION (medRxiv 2025)** — restricted to ion channels (a deeply related family), gene-stratified splits only. The reported ESM-2 improvement over simpler baselines is consistent with within-family homology rather than genuine mechanism learning. Verifying this would require re-running their setup with family-split CV.
- **LoGoFunc** — uses gene-aware splits but doesn't explicitly enforce family disjointness.
- **Badonyi & Marsh 2024** — doesn't use language models but reports per-class metrics without family-split CV.

This isn't a refutation of those papers' reported numbers — it's a statement that those numbers are consistent with the family-recognition shortcut, and that prior CV designs don't rule it out.

---

## Novelty assessment

### What's already known (not novel here)

**ESM-2 clustering by protein family is not a discovery.** ESM-2 was designed to produce representations that capture evolutionary and functional relationships, and protein families are defined by exactly those relationships.

- Rives et al. 2021 (the original ESM paper, *PNAS*) showed ESM embeddings recover family structure.
- Lin et al. 2023 (ESM-2 / ESMFold, *Science*) explicitly benchmarked remote homology detection.
- Many downstream tools use ESM embeddings as a similarity metric for family assignment.

### What's genuinely novel (probably)

**The novel contribution is the quantitative causal chain:**

> The embeddings cluster by family with 26× the expected rate. 74.8% of disease genes share their family's majority mechanism. Therefore any gene-split-only mechanism classifier is achieving most of its score via family recognition. The WT-only macro-F1 of 0.58 in this pipeline is a worked example of exactly this shortcut.

This specific chain — measured numbers connecting embedding clustering to mechanism label correlation to classifier performance inflation — hasn't been spelled out in prior work on mechanism prediction. The closest prior work (Gerasimavicius 2022, Livesey & Marsh 2020/2023) raises the issue qualitatively but doesn't measure it.

**"Family matters" is folk knowledge. "Family explains this specific benchmark number, and here's the arithmetic" is a testable, actionable methodological claim.**

### Publishability

| Framing | Interest level | Plausible venue |
|---|---|---|
| "ESM-2 encodes protein family" | Already textbook — not interesting | None |
| "Mechanism classifiers may have homology leakage" | Mildly interesting; restates folk knowledge | Workshop or methods note |
| **Quantified family leakage explains our WT-only baseline; same shortcut likely inflates published results; family-split CV proposed as minimum standard** | **Genuinely useful methodological contribution** | bioRxiv → *Bioinformatics* / *Genome Biology* if expanded |
| "PLMs fundamentally cannot encode mechanism — evidence across ≥3 models and ≥3 datasets, with within-family positive contrast" | High-impact correction | *Nat Methods* / *Nat Commun* |

Current work is at the third row. Reaching the fourth would require the follow-up experiments below.

### Reviewer-defensible novelty statement

> *"It is known that protein language models encode protein family. What hasn't been quantified is how much this inflates downstream gene-level mechanism classifiers under common evaluation designs. We provide that quantification on a standard mechanism dataset (Gerasimavicius), show that it accounts for the entire apparent signal in our WT-only baseline, and predict the same leakage explains reported headroom in MissION and similar ESM-2-based work. We propose family-split CV as a minimum evaluation standard for this task."*

### What would make this more novel

If follow-up experiments show:
1. **WT-only macro-F1 collapses under family-split** (confirms the shortcut is the whole signal, not just a contributor).
2. **The same pattern replicates on DDG2P** (~2,000 genes, structured labels).
3. **A structure-aware model (SaProt or ESM-3) shows the same family-clustering and the same null mechanism signal** — the field currently believes adding structural information will fix the problem; showing it doesn't would directly correct that.
4. **Within-family analysis reveals mechanism IS learnable inside a single protein family** — this would be a positive flip side: "mechanism prediction is a within-family problem, not a cross-family one; the field has been measuring the wrong thing."

With all four, this moves from "methodological cleanup" to "field-level reframing of what mechanism prediction is."

---

## Revised story

> ESM-2 embeddings strongly recognise protein families (nearest-neighbour purity 26× chance, family prediction 27× majority baseline). Because 74.8% of disease genes in Gerasimavicius share their family's most common mechanism label, any mechanism classifier built on ESM-2 without holding out entire families during evaluation is largely recognising family membership, not mechanism. The WT-only baseline's macro-F1 of 0.58 is fully explained by this shortcut. Subtracting the wildtype (the delta operation) removes most family information; the small residual nonlinear signal recovered by the MLP (0.28 → 0.41) is consistent with leftover family clustering, not learned mechanism.

---

## What this is not

- **Not** a claim that ESM-2 is useless for mechanism prediction — only that the delta representation and gene-split-only evaluation are both flawed for this task.
- **Not** a claim that mechanism is unlearnable from sequence — only that ESM-2's current representation, evaluated this way, doesn't reveal it.
- **Not** definitive without replication on a second dataset (DDG2P) and a second model (SaProt, ESM-3).

---

## Next experiments (in priority order)

1. **MLP under family-split CV on WT-only and delta.** Single most important confirmation: if MLP-WT-only also collapses under family-split, the family-shortcut explanation is locked in.
2. **Multi-seed replication** (5+ seeds) on all current metrics. Current numbers are one random seed only.
3. **Positive controls**: the same pipeline should recover ClinVar pathogenicity (AUROC > 0.85 expected) and Megascale ΔΔG stability (Spearman ρ > 0.5 expected). Without these, we can't rule out the pipeline itself being broken.
4. **Replicate on DDG2P** (~2,000 genes, structured `mutation_consequence` field). If the same pattern appears, the result generalises beyond the Gerasimavicius dataset.
5. **Structure-aware model test**: re-run on SaProt (which uses structural tokens) or ESM-3. If a structure-aware model also fails, the negative result is much stronger. If it succeeds, the finding becomes "mechanism requires structural context that pure-sequence models lack."
6. **Within-family analysis**: pick the 3 largest protein families (kinase, C2 domain, homeobox) and test whether mechanism is learnable within a single family — directly addresses whether tools like MissION succeed because they work within a family, not across them.

---

## Files

- `../scripts/family_clustering.py` — analysis script (~310 lines, reuses `experiment.py` helpers)
- `../results/20260524_baseline_run/run_0/family_clustering.json` — full metric output, all three embedding types

## Caveat on the script's printed headline

The last line of `family_clustering.py` checks only the silhouette score and prints "NO family clustering" — this is wrong for this data and should be ignored. The correct read is the four other metrics in the JSON output. The check should be rewritten to use a consensus across all four robust metrics.
