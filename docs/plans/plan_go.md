# Plan: ML/AI framing of the family-split GO experiment

## The pitch in one sentence

Probing benchmarks for frozen foundation-model representations are systematically inflated when probe labels correlate with the latent categorical structure the pretraining process induces; we introduce a holdout-strictness ladder that decomposes probe accuracy into genuine feature learning vs. latent-cluster recognition, and instantiate it on protein language models where the latent hierarchy (Pfam family → clan → CATH superfamily → fold) is explicit and stratifiable.

---

## The ML problem (not biology)

Linear probes on frozen foundation-model representations are the dominant interpretability tool across NLP, vision, audio, and code. Probe accuracy is taken as evidence that the model "knows" some concept.

This is wrong when the probe label correlates with latent clusters the pretraining distribution induces in the representation. A random train/test split then measures cluster-recognition, not concept-learning. The probe is real; the interpretation is not.

Examples of this failure mode across domains:
- Vision models cluster by ImageNet class — probing for "object property" labels that correlate with class is contaminated.
- Language models cluster by domain and register — probing for "semantic property" labels that correlate with corpus topic is contaminated.
- Code models cluster by repository — probing for "code style" labels that correlate with project is contaminated.

The general principle: any probe target whose label distribution is non-uniform across the latent clusters yields probe accuracy that is a mixture of genuine feature learning and cluster-recognition. Random splits cannot separate the two.

---

## Why proteins are the right testbed

Proteins are not the subject of the paper; they are the substrate that makes the methodology measurable.

What makes them useful:

1. **Labeled latent hierarchy.** Pfam family, Pfam clan, CATH superfamily, CATH topology — a sequence of progressively stricter notions of "how related are these inputs." Each rung is an explicit equivalence relation we can hold out.
2. **Stratifiable holdouts.** We can split train/test at any rung and re-run the probe. NLP has no analogous public hierarchy of "domain similarity at increasing strictness."
3. **Quantifiable label-cluster correlation.** For any probe target (GO term, mechanism class, etc.), we can directly measure within-family label agreement before training. This makes the contamination predictable, not just diagnosable.
4. **Frozen 650M-parameter foundation model with a well-developed probing literature.** ESM-2 is a natural target with directly comparable published numbers.

---

## The contribution: holdout-strictness ladder

A general probing-evaluation protocol:

| Rung | Holdout granularity | Probe accuracy reflects |
|---|---|---|
| 0 | Random train/test | Concept-learning + cluster-recognition (entangled) |
| 1 | Sequence/instance-identity holdout (e.g. UniRef50) | Removes near-duplicates; still leaks cluster structure |
| 2 | Latent-cluster holdout, level 1 (Pfam family) | Survivors = cross-family learning |
| 3 | Latent-cluster holdout, level 2 (Pfam clan / CATH superfamily) | Survivors = cross-superfamily / fold-level learning |
| 4 | Latent-cluster holdout, level 3 (CATH topology) | Survivors = architecture-independent feature learning |

Run a probe at every rung. Plot probe accuracy vs. rung for each probe target. The shape of the curve diagnoses the target:

- **Flat curve** = genuine feature learning (target is encoded as a transferable feature, independent of latent cluster).
- **Step down at rung 2** = family memorization (target is a family-level property dressed as a feature).
- **Gradual decline** = layered signal (partly motif-level, partly family-localized).

Aggregate across probe targets: gives a per-target classification of what the foundation model has actually internalized.

---

## Theoretical decomposition

Decompose observed probe accuracy `A` on a random split into:

```
A = α · A_feature + β · A_cluster + ε
```

where:
- `A_feature` is the accuracy attainable if the probe label were independent of latent clusters.
- `A_cluster` is the accuracy attainable from cluster-recognition alone given the label-cluster correlation structure.
- `α + β = 1` in the simplest decomposition; the cluster-label mutual information determines `β`.

A random split estimates `A`. Holdout at rung k estimates the residual `A_feature` after removing cluster correlations up to that level. The ladder traces the decomposition empirically.

This formal framing connects the work to:
- **Probing-as-information-theory** (Pimentel et al., 2020; Voita & Titov, 2020) — but those analyses assume i.i.d. splits and so collapse `A_feature` and `A_cluster`.
- **Control tasks** (Hewitt & Liang, 2019) — control tasks measure probe expressivity, not representation-vs-label entanglement; orthogonal contribution.
- **Spurious correlation / shortcut learning** (Geirhos et al., 2020) — same phenomenon, applied to representation probes rather than supervised classifiers.

---

## Empirical case study (the protein instantiation)

### Part A: Negative — task where probe accuracy is mostly cluster-recognition

Mechanism prediction (GOF / DN / LOF) on disease genes with frozen ESM-2 650M embeddings + linear probe.
- Random/gene-split CV: F1 = 0.58.
- Pfam-family-split CV: F1 = 0.39 (= family-majority baseline).
- 62% of the above-chance signal is cluster-recognition.

This is the cautionary case: standard probing methodology would have reported a real "ESM-2 encodes mechanism" finding. The ladder reveals it as family memorization.

### Part B: Mixed — task where the per-target decomposition is heterogeneous

GO term prediction with the same probe pipeline, across hundreds of testable terms.
- Mean drop random → Pfam-family-split is small (~3 AUROC points after filtering near-chance terms).
- Per-term heterogeneity is large.
- Three regimes observed in smoke (n=50 terms):
  - **Flat across rungs:** mitochondrial localization, extracellular matrix, DNA-binding TF activity (≤2 pt drop).
  - **Step-down at family rung:** ATP binding (~9 pt drop), calcium ion binding (~12 pt drop), visual perception.
  - **Already near chance at random split:** pathway-level BP terms — noise-dominated, excluded from headline.

### Part C: Strictness ladder on survivors

For each term that survives Pfam-family-split, re-probe at Pfam-clan and CATH-superfamily levels. Hypothesis: targeting peptides and pan-fold-recognition concepts survive all the way down; fold-localized motifs collapse between family and superfamily.

This is the rung that converts "PLM works below 30% identity" (DeepGO-SE's claim) into a quantitative per-target map of *why*.

### Part D: Non-protein companion experiment

To argue the methodology generalizes, run one matched experiment in a different modality:
- BERT or RoBERTa frozen embeddings.
- A probing task with a known latent cluster structure in the pretraining corpus (e.g. topic-correlated semantic property classification on a domain-labeled corpus).
- Hold out by topic/domain at increasing strictness.
- Show the same ladder shape emerges.

This is what turns the work from "protein paper with ML packaging" into "ML methodology paper with protein case study." Without it, the venue argument fails.

---

## What the paper claims

1. Random-split probe accuracy systematically overstates concept-learning in frozen foundation-model representations whenever probe labels correlate with the model's latent cluster structure.
2. The inflation is heterogeneous across probe targets — some are genuine features, some are cluster-recognition. Random splits cannot distinguish them.
3. A holdout-strictness ladder produces a per-target decomposition that distinguishes the two.
4. Empirical demonstration: one task (mechanism) where ladder reveals 62% of probe accuracy is latent-cluster recognition; one task (GO terms) where the ladder reveals an interpretable per-target map of genuine vs. cluster-mediated learning; one non-protein companion showing the methodology transfers.

---

## What this is NOT

- Not a debunking of PLM function-prediction benchmarks. The GO data shows most published numbers are roughly right.
- Not a biology paper with ML vocabulary. The biology is a testbed; the contribution is methodological.
- Not a claim that all probing literature is wrong. The claim is narrower: probing literature *that has not stratified by latent-cluster structure* is uninterpretable as concept-learning evidence.

---

## Venues

- **Primary target:** NeurIPS / ICLR / ICML main track (methodology / evaluation).
- **Secondary:** Foundation-model-evaluation workshops at the same venues.
- **Not:** bioRxiv as primary. A short companion bio note pointing to the methodology paper is fine.

---

## Relationship to existing literature

| Work | Limitation our methodology addresses |
|---|---|
| Hewitt & Liang 2019 (control tasks) | Measures probe expressivity, not representation-vs-label entanglement |
| Pimentel et al. 2020 (info-theoretic probing) | Assumes i.i.d. split; collapses feature and cluster contributions |
| Voita & Titov 2020 (MDL probing) | Same i.i.d. assumption |
| Geirhos et al. 2020 (shortcut learning) | Same phenomenon, applied to supervised classifiers not probes |
| ProteInfer (Sanderson 2023) | Closest holdout-stricter probe in proteins; identity-clustered not family-disjoint, CNN not PLM |
| DeepGO-SE 2024 | Argues PLMs work below 30% identity; we provide the per-target decomposition explaining when and why |
| Hermann & Fiedler MLCB 2024 | Pretraining-data leakage for thermostability; single task, no ladder |

---

## Implementation plan

### Phase 1 — methodology + protein ladder (4–6 weeks)
1. Formalize the decomposition `A = α·A_feature + β·A_cluster` (2–3 days).
2. Pfam-family-split is already implemented for mechanism + GO smoke.
3. Add Pfam-clan holdout (clan → family mapping from Pfam database) — 1 week.
4. Add CATH-H-superfamily holdout (map UniProt → CATH via Gene3D) — 1 week.
5. Re-run probe pipeline for mechanism (3 classes) and GO (top ~500 terms after frequency + signal filtering) at all four rungs — compute on existing cached embeddings, CPU-bound for probes.
6. Per-target ladder curves, namespace breakdown, theoretical-decomposition fit. Result writeup.

### Phase 2 — non-protein companion experiment (2–3 weeks)
1. Pick: BERT-base or RoBERTa-base, frozen.
2. Probe target with known topic correlation in a domain-labeled corpus (candidates: 20 Newsgroups subtopics with overlapping vocabulary; AG News with author/source labels; PubMed-RCT with section-vs-topic confound).
3. Build a 3-rung holdout ladder analogous to identity → cluster level 1 → cluster level 2.
4. Show the same curve shape emerges; ideally find one probe target that flattens (genuine learning) and one that collapses (topic memorization).

### Phase 3 — writeup (2 weeks)
1. Methods section: decomposition + ladder protocol.
2. Results: protein ladder + non-protein companion.
3. Related-work positioning against probing / shortcut / pretraining-leakage literature.
4. Target a 9-page conference submission.

### Total timeline
~10 weeks to submission-ready.

---

## Open questions before committing

1. **Companion experiment choice.** The non-protein experiment is load-bearing. We need a public dataset where the latent cluster structure is labeled and at least two rungs are constructible. Candidate shortlist needs to be drawn up.
2. **Theoretical depth.** Is the decomposition `A = α·A_feature + β·A_cluster` enough, or do we need a more rigorous information-theoretic statement (e.g. a CMI-based bound)? Latter is more publishable; former is faster.
3. **Mechanism work positioning.** The existing mechanism preprint (v1) is currently framed as a biology contribution. For this paper, mechanism becomes Part A of the empirical case study. Decision: does mechanism stay as a separate bio preprint and get re-used here, or does this paper subsume it?
4. **Probe choice.** Linear probes are standard but criticized as low-capacity. Should we add a small MLP probe for robustness? Doubles compute; doubles writeup complexity.
