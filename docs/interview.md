# Interview pitch — ESM-2 mechanism project

Rehearsal doc for talking through the project on a call. Two scripted parts (pitch + deep dive), then likely follow-up questions and how to answer them.

---

## 30-second pitch (memorize this)

> *"I tested whether a major protein language model — ESM-2 — encodes disease mutation mechanism, not just whether mutations are damaging. On the same pipeline, ESM-2 predicts pathogenicity at AUROC 0.88 but mechanism barely above the trivial baseline. The dissociation holds across two datasets, and a direct mut-only versus WT-only control shows the mutation itself contributes essentially zero signal beyond gene identity. I posted a preprint on bioRxiv consolidating this with two prior papers whose own ablation data — when actually read — supports the same conclusion. The methodological contribution is a controlled side-by-side dissociation with a leakage diagnostic that prior work missed."*

It's honest, specific, and shows methodological maturity.

---

## 5-minute deep dive structure

If they ask "tell me more":

### 1. The question (~30s)
Can frozen protein language models (PLMs) predict the *mechanism* of disease mutations — gain-of-function vs dominant-negative vs loss-of-function — not just whether the mutation is damaging? This matters clinically because the three call for different treatments.

### 2. The setup (~60s)
- **Model**: ESM-2 650M, frozen (no fine-tuning)
- **Features**: delta embedding = mutant sequence embedding − wildtype sequence embedding, mean-pooled
- **Probes**: logistic regression + MLP
- **CV**: gene-split AND family-split (Pfam-disjoint) 5-fold cross-validation
- **Datasets**: Gerasimavicius 2022 (948 disease genes) + merged with G2P/ClinVar (1,985 genes total)
- **Positive control**: same pipeline on ClinVar pathogenic vs benign (17,236 variants)

### 3. The key result (~60s)
- **Pathogenicity**: AUROC 0.88, family-split-stable (Δ = 0.002 from gene-split to family-split)
- **Mechanism**: family-split macro-F1 floor 0.36 across two datasets, six (probe × feature) combinations
- **Trivial baseline** (always predict majority class LOF): 0.31
- **The dissociation**: same pipeline, same embeddings, same CV — pathogenicity strong (AUROC 0.88), mechanism barely above baseline (F1 0.36 vs 0.31)
- The model knows whether a mutation is damaging but not how it acts

### 4. The control that locks it in (~60s)
- An MLP on raw **mutant** embeddings (not delta) reaches family-split macro-F1 0.49
- An MLP on raw **wildtype** embeddings (no mutation at all) reaches F1 0.49 — statistically indistinguishable
- If the mutation contributed mechanism-specific signal, mut-only would have to beat WT-only. It doesn't.
- The 0.49 ceiling is gene-identity signal that survives family-split CV at our dataset's family density (1.7 genes/family, 55% singleton families)
- The delta operation (mut − WT) is the methodological control that strips this residual leakage, giving the honest 0.36 floor for mutation-specific mechanism signal

### 5. The literature reconciliation (~60s)
Two commonly cited prior works support the same conclusion in their own ablation data, which is rarely cited:
- **MissION (2025)** reports headline AUROC 0.925 on ion channels. Their own ablation: removing ESM-2 pLM features → mean Δ = −1.02, **p = 0.61**. The headline is driven by GO terms and clinical phenotype annotations (HPO), not by ESM-2 representation content.
- **LoGoFunc (2023)** reports headline GOF AP 0.524 under gene-disjoint CV. Their own homology-disjoint test (≤40% sequence identity between train/test proteins) drops GOF AP to 0.37 — **a 29% drop**. Same direction as our family-split finding.
- Both papers report these stricter numbers honestly in their own text. The field cites the headline. We consolidate.

---

## Follow-up Q&A (likely questions)

### "Why does this matter?"
Mechanism-aware variant interpretation could change how clinicians prioritise variants of uncertain significance and which therapies are tried first (e.g., gain-of-function vs loss-of-function suggest opposite drug strategies). If PLMs encoded mechanism, this would scale to thousands of genes cheaply. The result says current PLMs don't, and identifies a specific reason (family-recognition shortcuts in evaluation) why prior work appears to show otherwise.

### "What's novel here? Aren't there already papers on this?"
Three things are new:
1. **First controlled same-pipeline pathogenicity-vs-mechanism dissociation** — prior work tests these separately
2. **The mut-only ≈ WT-only direct control** — nobody has run this specific check; it's dispositive evidence that the mutation contributes nothing to the higher numbers
3. **The consolidation of MissION and LoGoFunc** — reading their own ablation results together reveals the consistent pattern; no prior paper does this

The qualitative claim "PLMs don't predict mechanism" has been suggested before (PreMode, AlphaMissense paper). The contribution is operationalising the test and making the dissociation visible.

### "What did you actually do, technically?"
- Built embedding extraction pipeline (ESM-2 650M on UniProt sequences, with windowing for proteins >1022 residues)
- Implemented gene-split AND Pfam-family-disjoint 5-fold CV
- Probes: logistic regression, MLP (PyTorch, class-weighted, early stopping), kNN, gradient-boosted trees, random forest, contrastive metric learning with cross-family triplet loss
- Pathogenicity positive control: pulled 17k ClinVar variants, ran same pipeline
- Family-clustering diagnostics: k-NN purity, family probe accuracy, within-family mechanism agreement
- All on RunPod A100 80GB, scripts and JSON outputs in the public GitHub repo

### "What didn't work?"
- **Contrastive metric learning** (designed specifically to surface cross-family signal beyond family identity): lifted macro-F1 modestly (+0.03) but did not improve per-class AUROCs over the standard MLP — and on merged dataset, contrastive GOF/DN actually underperformed raw k-NN. This is strong evidence the ~0.36 floor is a property of the ESM-2 representation, not of probe choice.
- **Within-family pilot** (testing if mechanism is learnable inside a single Pfam family): samples are too small (≤24 genes/family) for a publishable result. Belongs as discussion-only future work.

### "How does this compare to MissION's 0.925?"
MissION reports AUROC 0.925 — but their own ablation shows removing ESM-2 from the model has no significant effect (p = 0.61). The 0.925 is driven by GO terms and HPO clinical phenotype annotations. Their leave-one-gene-out test (out-of-gene generalisation) drops to 70.8% accuracy, not significantly better than the prior SCION baseline (p = 0.17). MissION is not a counterexample; it's a confirmation hidden in the ablation table.

### "What would you do next?"
- **DDG2P replication** — confirm the pattern on a third independent mechanism-labelled disease gene set
- **SaProt or ESM-3** — test whether structure-aware PLMs recover the signal that sequence-only ESM-2 misses; would tell us *what* pretraining ingredient matters for mechanism encoding
- **Within-family at scale** — wait for a kinome-scale labelled dataset (~50+ kinases with curated mechanism) to test whether mechanism is learnable inside a single family

### "Is this a positive or negative result?"
Negative on the mechanism question, but with a methodological contribution (the controlled dissociation + the mut-only ≈ WT-only control) that has positive value for the field. Negative results with strong controls are more valuable than weak positive results.

### "How did you handle the prior-work overlap?"
Read MissION and LoGoFunc carefully, found that their own reported stricter-evaluation numbers (MissION's pLM ablation p=0.61; LoGoFunc's homology-disjoint GOF AP drop of 29%) confirm rather than contradict our finding. The contribution narrows from "discovery" to "consolidation with a cleaner controlled setup" — and the framing is honest about that.

### "Why frozen embeddings, not fine-tuning?"
Two reasons: (1) fine-tuning conflates "what the representation contains" with "what fine-tuning can teach the model" — frozen probing isolates the former, which is the actual question; (2) ESMGain (Tang et al. 2025) already shows fine-tuning helps for GOF — we complement that by characterising what's recoverable without it.

---

## How to handle the obvious skeptic question

> *"This sounds like a negative result. Why is it interesting?"*

Honest answer in one sentence: **"Negative results with proper controls and a leakage diagnostic are more valuable to the field than positive results without them, because they save other researchers months of building on a flawed foundation — and the controls themselves (the mut-only ≈ WT-only check, family-split CV applied to frozen PLMs specifically) are reusable methodological tools."**

Don't apologise for it being a negative result. Negative results with strong controls are good science. Frame the contribution as methodological: a diagnostic that anyone evaluating a PLM mechanism predictor should now run.

---

## What to put on your CV

```
Gupta D. Frozen ESM-2 embeddings encode mutation pathogenicity strongly
and disease mechanism weakly: a controlled dissociation under family-split
cross-validation. bioRxiv 2026. doi:10.1101/[fill in once posted]
```

Code: https://github.com/[your-username]/esm2_mechanism

---

## Quick-reference numbers (memorize)

| Metric | Value |
|---|---|
| Pathogenicity AUROC (delta MLP, family-split) | **0.88** |
| Mechanism macro-F1 floor (delta MLP, family-split) | **0.36** |
| Always-predict-LOF baseline | 0.31 |
| Gerasimavicius dataset | 948 genes, 10,231 variants |
| Merged dataset | 1,985 genes, 19,100 variants |
| ClinVar pathogenicity dataset | 944 genes, 17,236 variants |
| MissION pLM ablation p-value | **0.61** |
| LoGoFunc GOF AP drop under homology-disjoint CV | **29%** (0.524 → 0.37) |
| mut-only family-split F1 (merged) | 0.492 |
| wt-only family-split F1 (merged) | 0.494 |
| Family-split macro-F1 floor (across 6 setups) | 0.34–0.39 |
| Above-chance signal lost under family-split | 61–63% |
