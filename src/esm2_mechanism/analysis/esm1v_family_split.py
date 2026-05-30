"""
Per-Pfam-family AUROC analysis of ESM-1v ΔLL on ClinVar pathogenic/benign.

Mirrors alphamissense_family_split.py — same framing, same metrics, same
output structure — so the two results are directly comparable.

Inputs:
  data/pathogenicity_valid_variants.json  (17,236 variants)
  data/esm1v_scores_full.json             (variant_key -> ΔLL; from score_esm1v.py)
  data/pfam_families.json                 (gene -> Pfam ID)

Outputs:
  results/esm1v_family/overall.json
  results/esm1v_family/per_family.json
  results/esm1v_family/summary.json
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

from esm2_mechanism.utils_paths import DATA_DIR as DATA, RESULTS_DIR as _RESULTS_DIR
from esm2_mechanism.utils_eval import vkey, run_family_split_eval

RESULTS = _RESULTS_DIR / "esm1v_family"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--min-pos", type=int, default=10, help="min pathogenic variants per family"
    )
    ap.add_argument(
        "--min-neg", type=int, default=10, help="min benign variants per family"
    )
    ap.add_argument("--scores", type=Path, default=DATA / "esm1v_scores_full.json")
    args = ap.parse_args()

    if not args.scores.exists():
        print(
            f"ERROR: scores file not found: {args.scores}\n"
            f"Run score_esm1v.py first.",
            file=sys.stderr,
        )
        return 1

    with open(DATA / "pathogenicity_valid_variants.json") as _f:
        variants = json.load(_f)
    with open(args.scores) as _f:
        scores = json.load(_f)
    with open(DATA / "pfam_families.json") as _f:
        pfam = json.load(_f)

    # Assemble (label, score, family) rows. ESM-1v ΔLL is lower-is-pathogenic,
    # so negate it to orient higher = more pathogenic for the shared evaluator.
    rows = []
    miss_score = miss_pfam = 0
    for v in variants:
        k = vkey(v)
        s = scores.get(k)
        if s is None or (isinstance(s, float) and np.isnan(s)):
            miss_score += 1
            continue
        fam = pfam.get(v["gene"])
        if not fam:
            miss_pfam += 1
            continue
        label = 1 if v["label"] == "pathogenic" else 0
        rows.append((label, -float(s), fam))

    print(f"variants: {len(variants):,}")
    print(f"  missing ΔLL score:   {miss_score:,}")
    print(f"  missing Pfam family: {miss_pfam:,}")
    print(f"  usable rows:         {len(rows):,}")

    return run_family_split_eval(
        rows, RESULTS, min_pos=args.min_pos, min_neg=args.min_neg
    )


if __name__ == "__main__":
    sys.exit(main())
