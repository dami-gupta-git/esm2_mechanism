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
import statistics
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import functools

print = functools.partial(print, flush=True)

from esm2_mechanism.utils_paths import DATA_DIR as DATA, RESULTS_DIR as _RESULTS_DIR

RESULTS = _RESULTS_DIR / "alphamissense_family"


def vkey(v: dict) -> str:
    return f"{v['gene']}_{v['aa_pos']}_{v['aa_wt']}_{v['aa_mut']}"


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

    RESULTS.mkdir(parents=True, exist_ok=True)

    with open(DATA / "pathogenicity_valid_variants.json") as _f:
        variants = json.load(_f)
    with open(args.scores) as _f:
        scores = json.load(_f)
    with open(DATA / "pfam_families.json") as _f:
        pfam = json.load(_f)

    # Assemble (label, score, family) rows.
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
        rows.append((label, float(s), fam, v["gene"]))

    print(f"variants: {len(variants):,}")
    print(f"  missing AM score:    {miss_score:,}")
    print(f"  missing Pfam family: {miss_pfam:,}")
    print(f"  usable rows:         {len(rows):,}")
    if not rows:
        print("no usable rows; aborting")
        return 1

    y = np.array([r[0] for r in rows])
    s = np.array([r[1] for r in rows])

    overall = {
        "n": int(y.size),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "auroc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
    }
    print(
        f"\nOVERALL  n={overall['n']:,}  "
        f"pos={overall['n_pos']:,}  neg={overall['n_neg']:,}  "
        f"AUROC={overall['auroc']:.4f}  PR-AUC={overall['pr_auc']:.4f}"
    )
    with open(RESULTS / "overall.json", "w") as f:
        json.dump(overall, f, indent=2)

    # Per family.
    by_fam: dict[str, list[tuple[int, float]]] = {}
    for label, score, fam, _g in rows:
        by_fam.setdefault(fam, []).append((label, score))

    per_family = {}
    skipped = 0
    for fam, items in by_fam.items():
        ys = np.array([it[0] for it in items])
        ss = np.array([it[1] for it in items])
        n_pos, n_neg = int(ys.sum()), int((1 - ys).sum())
        if n_pos < args.min_pos or n_neg < args.min_neg:
            skipped += 1
            continue
        per_family[fam] = {
            "n_pos": n_pos,
            "n_neg": n_neg,
            "auroc": float(roc_auc_score(ys, ss)),
            "pr_auc": float(average_precision_score(ys, ss)),
        }

    print(f"\nfamilies total:        {len(by_fam):,}")
    print(
        f"families passing min:  {len(per_family):,} "
        f"(min_pos={args.min_pos}, min_neg={args.min_neg})"
    )
    print(f"families skipped:      {skipped:,}")

    with open(RESULTS / "per_family.json", "w") as f:
        json.dump(per_family, f, indent=2, sort_keys=True)

    aurocs = sorted(d["auroc"] for d in per_family.values())
    if aurocs:
        q = np.quantile(aurocs, [0.0, 0.25, 0.5, 0.75, 1.0])
        summary = {
            "n_families": len(aurocs),
            "overall_auroc": overall["auroc"],
            "per_family_mean": float(np.mean(aurocs)),
            "per_family_std": float(np.std(aurocs)),
            "per_family_min": float(q[0]),
            "per_family_q25": float(q[1]),
            "per_family_median": float(q[2]),
            "per_family_q75": float(q[3]),
            "per_family_max": float(q[4]),
            "per_family_iqr": float(q[3] - q[1]),
            "frac_below_0_8": float(sum(a < 0.8 for a in aurocs) / len(aurocs)),
            "frac_below_0_7": float(sum(a < 0.7 for a in aurocs) / len(aurocs)),
        }
        print("\nPER-FAMILY AUROC DISTRIBUTION")
        print(f"  n_families = {summary['n_families']}")
        print(f"  overall    = {summary['overall_auroc']:.4f}")
        print(
            f"  mean ± std = {summary['per_family_mean']:.4f} ± "
            f"{summary['per_family_std']:.4f}"
        )
        print(
            f"  min / q25 / med / q75 / max = "
            f"{summary['per_family_min']:.3f} / "
            f"{summary['per_family_q25']:.3f} / "
            f"{summary['per_family_median']:.3f} / "
            f"{summary['per_family_q75']:.3f} / "
            f"{summary['per_family_max']:.3f}"
        )
        print(f"  frac AUROC < 0.8 = {summary['frac_below_0_8']:.2%}")
        print(f"  frac AUROC < 0.7 = {summary['frac_below_0_7']:.2%}")

        # Print bottom 5 / top 5 families by AUROC.
        ranked = sorted(per_family.items(), key=lambda kv: kv[1]["auroc"])
        print("\nWORST 5 families:")
        for fam, d in ranked[:5]:
            print(
                f"  {fam}  AUROC={d['auroc']:.3f}  n_pos={d['n_pos']} n_neg={d['n_neg']}"
            )
        print("\nBEST 5 families:")
        for fam, d in ranked[-5:]:
            print(
                f"  {fam}  AUROC={d['auroc']:.3f}  n_pos={d['n_pos']} n_neg={d['n_neg']}"
            )

        with open(RESULTS / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
