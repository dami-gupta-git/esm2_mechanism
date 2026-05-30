"""
Per-Pfam-family AUROC analysis of AlphaMissense on ClinVar pathogenic/benign.

Inputs:
- data/pathogenicity_valid_variants.json   (17,236 variants, label: pathogenic|benign)
- data/alphamissense_scores_full.json      (variant_key -> AM score; produced by fetch_alphamissense.py)
- data/pfam_families.json                  (gene -> Pfam ID)

Outputs:
- results/alphamissense_family/overall.json       (overall AUROC / PR-AUC / N)
- results/alphamissense_family/per_family.json    (family -> {n_pos, n_neg, auroc, pr_auc})
- results/alphamissense_family/summary.json       (mean / median / quartiles of per-family AUROCs)

The right framing for a fixed published predictor:
not a "holdout drop" but the *distribution of per-family AUROCs*.
A family-robust predictor has a tight distribution around the overall AUROC.
A predictor that inherits family-correlated training has a heavy tail.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import functools

print = functools.partial(print, flush=True)

from esm2_mech.utils.paths import DATA_DIR as DATA, RESULTS_DIR as _RESULTS_DIR
from esm2_mech.utils.eval import vkey, run_family_split_eval

RESULTS = _RESULTS_DIR / "alphamissense_family"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--min-pos", type=int, default=10, help="min pathogenic variants per family"
    )
    ap.add_argument(
        "--min-neg", type=int, default=10, help="min benign variants per family"
    )
    ap.add_argument(
        "--scores", type=Path, default=DATA / "alphamissense_scores_full.json"
    )
    args = ap.parse_args()

    with open(DATA / "pathogenicity_valid_variants.json") as _f:
        variants = json.load(_f)
    with open(args.scores) as _f:
        scores = json.load(_f)
    with open(DATA / "pfam_families.json") as _f:
        pfam = json.load(_f)

    # Assemble (label, score, family) rows. AlphaMissense score is already
    # oriented higher = more pathogenic.
    rows = []
    miss_score = miss_pfam = 0
    for v in variants:
        k = vkey(v)
        s = scores.get(k)
        if s is None:
            miss_score += 1
            continue
        fam = pfam.get(v["gene"])
        if not fam:
            miss_pfam += 1
            continue
        label = 1 if v["label"] == "pathogenic" else 0
        rows.append((label, float(s), fam))

    print(f"variants: {len(variants):,}")
    print(f"  missing AM score:    {miss_score:,}")
    print(f"  missing Pfam family: {miss_pfam:,}")
    print(f"  usable rows:         {len(rows):,}")

    return run_family_split_eval(
        rows,
        overall_path=RESULTS / "overall.json",
        per_family_path=RESULTS / "per_family.json",
        summary_path=RESULTS / "summary.json",
        min_pos=args.min_pos,
        min_neg=args.min_neg,
    )


if __name__ == "__main__":
    raise SystemExit(main())
