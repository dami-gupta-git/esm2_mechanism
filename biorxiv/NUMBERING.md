# Numbering key

`PREREGISTRATION_run_biorxiv.md` is the authority for current identifiers. This file maps identifiers
from earlier drafts to that schema. New writing and generated outputs use only the current identifiers.

## General rules

| Old | Current | Subject |
|---|---|---|
| R7.1 | 1.1 | Gate verdicts |
| R7.3 | 1.2 | Resampling units and pairing |
| R7.4 | 1.3 | Rare-class intervals |
| R7.6 | 1.4 | Calibration |
| R7.2 | Part 2 and Part 4 | Confirmatory and exploratory analyses |
| R7.5 | 2A | Permutation-test design and budget |
| R7.7 | Part 5 | Conditions that would change the conclusions |

## Confirmatory claims

| Old | Current | Claim |
|---|---|---|
| C1 | 2A | Mechanism delta sits at the measured chance floor under family-split evaluation |
| C2 | 2B | The absolute-embedding gene-to-family gap is non-zero |
| C3 | 2C | Pathogenicity clears family-split AUROC 0.85 |
| K1, C4 | 2D | Conservation alone clears family-split AUROC 0.85 |
| K2 | 2E | Adding the embedding delta to conservation improves AUROC by at least 0.02 |
| 2E.1 | 2F | Enzyme classification clears family-split LogReg macro-F1 0.70 |
| 2E.2 | 2G | Enzyme family-split F1 substantially exceeds the mechanism floor |
| 2E.3 | 2H | MLP does not substantially outperform LogReg for enzyme classification |

`K2b`, conservation plus delta versus delta alone, is descriptive and has no confirmatory claim
identifier in the current preregistration.

## Stability controls

| Old | Current | Control |
|---|---|---|
| H1 | 3A | Random-split Spearman correlation reaches 0.5 |
| H2 | 3B | Random-to-family Spearman drop stays below 0.10 |
| H3 | 3C | Removing the stability direction does not improve mechanism F1 by more than 0.01 |
| H4 | 3D | Per-domain Spearman spread remains within its threshold |

## Scope

The current identifiers apply to live bioRxiv documents, source code that emits bioRxiv results,
tests of those contracts, and newly generated reports. Retired plans, prior-run reports, and saved
logs remain historical records and are not rewritten.
