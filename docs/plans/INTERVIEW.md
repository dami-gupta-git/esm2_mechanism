# Interview narrative — big-picture story

A 3–5 minute version. Conversational, not a paper.

---

## Quick blurb (grab-and-go)

**30-second version:**
"I looked at whether protein language models — ESM-2 — encode the *mechanism* of a disease mutation: not just whether it's damaging, but *how* — does it break the protein, overactivate it, or jam it. The headline is a clean negative: that information isn't in the embeddings. The same model predicts *whether* a variant is pathogenic at AUROC ~0.88, family-robust — but mechanism sits at chance, and stays there no matter what probe I use. I traced *why*: the transferable signal ESM-2 carries is essentially conservation, which tells you a position matters but not which direction it fails. Mechanism needs gene-level functional and structural context, not sequence likelihood."

**Why it's a finding, not a dead end:**
"People are increasingly throwing PLM embeddings at clinical variant interpretation. My result says: for the *direction* of effect, that's the wrong tool — and I show what does work instead. A proteome/structure feature model recovers mechanism at F1 ~0.5. So it's not 'nothing predicts mechanism,' it's 'mechanism lives in functional context, not in the language model.'"

**Three pocket points:**
1. **Rigor is the story.** Naive gene-split made ESM-2 look like a mechanism predictor (F1 ~0.41); under family-split most of that was homology leakage (F1 ~0.30–0.38). Pre-registered gates + positive controls make the negative trustworthy, not underpowered.
2. **The dissociation.** Same model, same data, same splits: pathogenicity transfers, stability transfers nonlinearly, mechanism doesn't transfer at all. Transferability is task-dependent — a statement about *what the representation encodes*.
3. **The conservation decider.** I tested whether the embedding knows more than the model's own likelihood output. It doesn't — the pathogenicity axis is just conservation (masked-LL alone, AUROC 0.891, beats the embedding's 0.835). That's how I know the mechanism negative is real signal-absence, not failed extraction.

**Pushback — "isn't a negative just a failed project?"**
"A negative is only weak if it's underpowered or sloppy. Mine is neither — controlled splits, positive controls, multiple probe types, ~16,000 variants. Knowing a widely-assumed approach doesn't work, and exactly why, saves the field from chasing it."

**Pushback — "isn't 'PLMs don't do mechanism' obvious?"**
"The prevailing direction assumes PLM embeddings are richly functional — people predict structure, binding, function from them. Mechanism *is* function, so the null cuts against the optimistic prior. And I localized it: it's specifically the *functional-direction* axis that's missing, while conservation is fully there."

---

## The setup

I wanted to know if protein language models actually understand disease mechanism, or if they just look like they do because of how people evaluate them.

ESM-2 is a 650M-parameter transformer trained on a large protein sequence database. It clearly encodes a lot of biology — for example, it predicts whether a missense variant is pathogenic at AUROC in the 0.74–0.88 range (varies by replication; reproducibly family-split-stable). The natural follow-up question: does it also predict *how* a variant causes disease — through loss of function, gain of function, or dominant-negative effects? These three mechanisms call for completely different therapeutic strategies, so it's a clinically important distinction.

Several recent papers said yes. I wasn't convinced.

## The first finding

Under the standard evaluation people were using — train on some genes, test on others — ESM-2 looked like a decent mechanism predictor. Macro-F1 around 0.41, well above chance.

But that evaluation has a problem. Proteins come in families: hemoglobin α and β are similar, kinases are similar to each other, ion channels are similar to each other. So when you hold out one gene, related genes can still be in the training set. The model can recognise "this looks like a kinase" and use that as a shortcut, because in the training data kinases are mostly gain-of-function.

I re-ran the evaluation with stricter splits — hold out entire protein families, then go further and hold out clusters of distantly related proteins. The apparent mechanism signal lost most of its above-chance margin; under family-split with 5 random seeds, mechanism prediction lands at macro-F1 ~0.30–0.38. Meanwhile, the pathogenicity prediction was completely unaffected by the stricter holdout (Δ ≈ 0). Same model, same architecture, same cross-validation procedure — one task survived the strict test, the other didn't.

That's the central scientific finding: a controlled dissociation. ESM-2 encodes whether a mutation is damaging (AUROC 0.74–0.88, family-split-stable), but not how (mechanism F1 ~0.30–0.39 family-split). What looked like mechanism prediction was mostly the model recognising protein families.

## The hypothesis that didn't pan out

I thought I knew why ESM-2 was failing. Dominant-negative variants disrupt protein complexes — that biology lives at the cellular level, not at the sequence level. So I bet that adding protein interactome features, paralog counts, abundance measurements would recover the signal.

I built a 37-feature matrix from public sources, ran the same evaluation, and got a result I hadn't predicted. The feature-engineered model beat ESM-2 by +0.10 macro-F1 — a substantial win. But when I did the feature ablation, the win wasn't from the biology features I bet on. Interactome degree contributes essentially nothing. The win was driven by population constraint scores and clinical dosage-sensitivity ratings — features that have nothing to do with my complex-assembly hypothesis.

Then I tried adding three published probability scores from a competing predictor on top of those features. Three numbers improved the model by another +5 points on macro-F1 and +10 on the hardest class. Three features beat 1,280 dimensions of foundation-model embeddings.

Combining ESM-2 with anything else doesn't help. It's dispensable.

## What this means

The methodological contribution: family-split cross-validation operationalised as a quantitative leakage diagnostic with worked examples, plus multi-seed replication that showed even our single-seed numbers had been slightly optimistic. Most of the apparent above-chance mechanism signal under loose CV turns out to be family recognition wearing mechanism's clothes. Without this kind of evaluation, the field would keep publishing inflated numbers.

The scientific contribution: a controlled demonstration that frozen pre-trained protein language models don't encode mechanism the way they encode pathogenicity. The information that distinguishes mechanism classes lives in feature-engineered gene-level and structural signals — not in self-supervised sequence representations.

The ML lesson is broader: foundation models are extraordinarily powerful, but they aren't automatically the right tool for every downstream task. Sometimes three carefully chosen features encode information that a billion parameters cannot. And the evaluation protocol matters more than the model choice for distinguishing real signal from convenient memorisation.

## Why I picked this problem

I'm interested in the gap between what large models *look like* they're doing and what they're *actually* doing — particularly in domains where the consequences of believing the model are non-trivial. Disease mechanism prediction is one of those domains. The framework I built here — pre-registered gates, stratified holdout, controlled dissociation — generalises to any setting where you want to know whether a model has learned the thing you care about or a shortcut to a related thing.

---

## Optional follow-up depth (if asked)

- **What was your hypothesis going in?** That proteome context (PPI, paralogs, abundance) would carry the dominant-negative signal ESM-2 lacks. I was wrong about the features — constraint and dosage did the work — but right about the modality direction.
- **What didn't work?** Combining ESM-2 with anything. Across multiple architectures and holdout schemes, ESM-2 is consistently the dispensable input.
- **What's the most surprising number?** Three features beat 1,280. A published per-gene predictor's three output probabilities reach dominant-negative AUROC 0.82 under family-split — higher than any ESM-2-based model I tested.
- **What would you do differently?** Pre-register the feature ablation up front. The "interactome carries DN" hypothesis was disconfirmed by the ablation, and I should have built that ablation into the original experimental design rather than as a follow-up.
- **What's the biggest limitation?** Gene-level labels. Many disease genes have variant-dependent mechanism (SCN1A has both activating and inactivating alleles). The whole project predicts modal mechanism per gene, not per variant. Per-variant labels from functional assays would test whether a sharper-resolution version of the same question gives different answers.
- **Did you look at *what* the pathogenicity signal is geometrically?** Yes. It's a single, low-dimensional, family-transferable direction in embedding space — and it's carried by the *direction* of the perturbation, not its magnitude (magnitude-only AUROC 0.664 vs direction-only 0.896). I then ran a conservation decider: masked-LM log-likelihood at the variant position (the ESM1v score) alone reaches 0.891, beats the embedding (0.835), and the embedding adds nothing on top. So the "pathogenicity axis" is just conservation — which is honest to say, and it's why I don't overclaim it as novel.
- **Stability — wasn't that also a negative?** It's the nuanced one. A linear probe makes stability look family-dependent, but a nonlinear probe (GBM) recovers it cross-family (AUROC ~0.75 on family-split). So stability lives on a *curved* cross-family manifold. Mechanism, by contrast, fails at every probe level — nonlinearity doesn't rescue it. That probe-type × task asymmetry is the sharpest single contrast in the project.
- **So what's the honest one-line status?** A rigorous characterization: ESM-2's only family-transferable signal is conservation, which suffices for pathogenicity, needs nonlinearity for stability, and is absent for mechanism — and mechanism is instead recoverable from gene-level functional/structural features (F1 ~0.5). It's a controlled negative plus a working positive, not a foundation-model success story.
