"""
Shared per-Pfam-family evaluation for fixed per-variant predictors.

Callers supply rows of (label, score, family) where the score is oriented so
that HIGHER means MORE pathogenic. (ESM-1v ΔLL is lower-is-pathogenic, so the
caller passes -ΔLL.)
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

print = functools.partial(print, flush=True)


def vkey(variant: dict) -> str:
    """Canonical variant key: gene_pos_wt_mut."""
    return f"{variant['gene']}_{variant['aa_pos']}_{variant['aa_wt']}_{variant['aa_mut']}"


def _per_family_summary(per_family: dict[str, dict], overall_auroc: float) -> dict:
    # Distribution shape matters: a tight distribution around overall_auroc means
    # the predictor is family-robust; a heavy low tail means it fails on specific families.
    aurocs = sorted(d["auroc"] for d in per_family.values())
    quartiles = np.quantile(aurocs, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "n_families": len(aurocs),
        "overall_auroc": overall_auroc,
        "per_family_mean": float(np.mean(aurocs)),
        "per_family_std": float(np.std(aurocs)),
        "per_family_min": float(quartiles[0]),
        "per_family_q25": float(quartiles[1]),
        "per_family_median": float(quartiles[2]),
        "per_family_q75": float(quartiles[3]),
        "per_family_max": float(quartiles[4]),
        "per_family_iqr": float(quartiles[3] - quartiles[1]),
        "frac_below_0_8": float(sum(a < 0.8 for a in aurocs) / len(aurocs)),
        "frac_below_0_7": float(sum(a < 0.7 for a in aurocs) / len(aurocs)),
    }


def run_family_split_eval(
    rows: Sequence[tuple],
    overall_path: Path,
    per_family_path: Path,
    summary_path: Path,
    min_pos: int = 10,
    min_neg: int = 10,
) -> int:
    """Compute overall + per-family AUROC/PR-AUC, print, and write JSON outputs.

    rows: sequence of (label, score, family, ...) tuples. Only the first three
          elements are used. label is 1 (pathogenic) / 0 (benign); score is
          oriented so higher = more pathogenic.
    overall_path / per_family_path / summary_path: output file paths.
    min_pos / min_neg: minimum pathogenic / benign variants for a family to be scored.

    Returns a process exit code (0 on success, 1 if no usable rows).
    """
    if not rows:
        print("no usable rows; aborting")
        return 1

    for path in (overall_path, per_family_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    labels = np.array([row[0] for row in rows])
    scores = np.array([row[1] for row in rows])

    # NOTE: labels are assumed to be exactly {0, 1}. n_neg = (1 - labels).sum()
    # and n_pos = labels.sum() are only correct under that encoding — a {-1, 1} or
    # multi-class encoding would silently miscount. Callers must pass 0/1 labels.
    overall = {
        "n": int(labels.size),
        "n_pos": int(labels.sum()),
        "n_neg": int((1 - labels).sum()),
        "auroc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
    }
    print(
        f"\nOVERALL  n={overall['n']:,}  "
        f"pos={overall['n_pos']:,}  neg={overall['n_neg']:,}  "
        f"AUROC={overall['auroc']:.4f}  PR-AUC={overall['pr_auc']:.4f}"
    )
    with open(overall_path, "w") as out_file:
        json.dump(overall, out_file, indent=2)

    by_fam: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        label, score, family = row[0], row[1], row[2]
        by_fam.setdefault(family, []).append((label, score))

    per_family: dict[str, dict] = {}
    skipped = 0
    for family, items in by_fam.items():
        family_labels = np.array([item[0] for item in items])
        family_scores = np.array([item[1] for item in items])
        n_pos, n_neg = int(family_labels.sum()), int((1 - family_labels).sum())
        # Skip families too small to produce a meaningful AUROC — roc_auc_score
        # requires at least one positive and one negative example.
        if n_pos < min_pos or n_neg < min_neg:
            skipped += 1
            continue
        per_family[family] = {
            "n_pos": n_pos,
            "n_neg": n_neg,
            "auroc": float(roc_auc_score(family_labels, family_scores)),
            "pr_auc": float(average_precision_score(family_labels, family_scores)),
        }

    print(f"\nfamilies total:        {len(by_fam):,}")
    print(
        f"families passing min:  {len(per_family):,} "
        f"(min_pos={min_pos}, min_neg={min_neg})"
    )
    print(f"families skipped:      {skipped:,}")

    with open(per_family_path, "w") as out_file:
        json.dump(per_family, out_file, indent=2, sort_keys=True)

    if per_family:
        summary = _per_family_summary(per_family, overall["auroc"])
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

        ranked = sorted(per_family.items(), key=lambda kv: kv[1]["auroc"])
        print("\nWORST 5 families:")
        for family, stats in ranked[:5]:
            print(
                f"  {family}  AUROC={stats['auroc']:.3f}  "
                f"n_pos={stats['n_pos']} n_neg={stats['n_neg']}"
            )
        print("\nBEST 5 families:")
        for family, stats in ranked[-5:]:
            print(
                f"  {family}  AUROC={stats['auroc']:.3f}  "
                f"n_pos={stats['n_pos']} n_neg={stats['n_neg']}"
            )

        with open(summary_path, "w") as out_file:
            json.dump(summary, out_file, indent=2)

    return 0
