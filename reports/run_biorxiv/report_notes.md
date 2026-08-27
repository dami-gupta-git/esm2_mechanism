# Mechanism results — notes  . Personal notes only.

- Delta embedding (the mutation signal) sits at the chance floor (0.290) whether or not gene families are held out — the confirmed mechanism null.
- Wildtype and mutant embeddings score well but drop sharply once families are held out (~0.55 → ~0.45–0.49) — that's homology leakage, about 38–39% of the score.
- Mutant-only ≈ wildtype-only, so the mutation itself contributes nothing; all signal is protein identity.
- Nonlinear models lift the delta a bit under loose splits, but most of that gain vanishes under family holdout.
- Permutation test: delta has a faint but real ranking signal (4/5 seeds significant), too weak to move classification off the floor.
- Single-source replication (one curation database only) reproduces the same pattern — not a merging artifact.
- Gain-of-function is easiest to detect, dominant-negative hardest, consistent with prior results.

## Pathogenicity control

- Passes clean: `delta_mean` MLP hits AUROC 0.888 family-split, CI [0.882, 0.893] — clears the 0.85 pre-registered bar with room to spare, and barely drops from the gene-split version (0.888 → 0.885), so almost no homology leakage here.
- Wildtype-only embedding is near coin-flip (AUROC ~0.52–0.53) on this task — it can't tell pathogenic from benign at all. Mirror image of the mechanism result: there, wildtype carried the signal and the mutation added nothing; here, only the mutation (the delta) carries signal.
- This is the clean positive control the mechanism null leans on — same pipeline, same embeddings, strong family-robust signal on a different task, so the mechanism floor isn't a broken-pipeline artifact.

## Geometry of the pathogenicity direction

- Pathogenicity is a direction, not a distance: magnitude alone barely beats chance (AUROC ~0.61 family-split), while direction alone matches or beats the full delta (0.855 logreg / 0.892 MLP vs 0.838/0.885 for the raw delta).
- That direction is one dominant, redundant axis — stripping out up to 5 fitted directions barely dents AUROC (0.838 → 0.828). Directions fit on separate halves of the data look different by raw similarity but still transfer well to each other, consistent with a redundantly-encoded single axis rather than many independent ones.
- The axis turns out to be conservation: ESM-2's own confidence about what amino acid "should" be at that position, on its own, clears AUROC 0.888 — a hair above the full embedding delta (0.835). Adding the embedding on top of conservation makes things very slightly worse, not better.
- Net read: the pathogenicity signal in the embedding delta is mostly a re-encoding of the model's own residue-conservation confidence, not new information. Passes the pre-registered "conservation clears 0.85" claim (2D); fails "embedding adds ≥0.02 beyond conservation" (2E).

## Enzyme type classification (positive control)

ESM-2's wildtype embedding classifies enzyme type (kinase/protease/oxidoreductase/non-enzyme) well, and it beats mechanism classification by a wide margin on the same pipeline — so the mechanism null isn't a broken pipeline, it's specific to that task. One sub-check (whether linear and nonlinear probes perform about the same) came out ambiguous.

- ESM-2's wildtype embedding classifies kinase/protease/oxidoreductase/non-enzyme well: macro-F1 0.788 family-split (CI [0.732, 0.817]), clearing the 0.70 pre-registered bar (claim 2F passes). Chance floor here is 0.219.
- It drops from gene-split (0.831) to family-split (0.788), about 8.5% leakage — much smaller than the mechanism task's ~38–39%, so this signal isn't just family recognition.
- Enzyme classification beats mechanism classification by 0.508 macro-F1 on the shared subset (0.788 vs 0.280), comfortably clearing the pre-registered 0.05 minimum with the CI excluding zero (claim 2G passes). This is the central result.
- Claim 2H (linear should be about as good as nonlinear) fails/underpowered: the MLP actually does noticeably worse than logistic regression here (0.713 vs 0.788), the opposite direction from "nonlinear beats linear," and the CI doesn't cleanly resolve it either way.
- Proteome features (the population-genetics negative control) sit near chance (macro-F1 ~0.29–0.34) as expected — confirms the WT-embedding result isn't a trivial artifact of any feature set working.
- Per-class: kinase and oxidoreductase are strongest (AUROC ~0.95), protease weaker but still solid (~0.90) despite being the smallest class.
