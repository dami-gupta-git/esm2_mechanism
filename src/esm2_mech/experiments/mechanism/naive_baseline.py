"""
Measure the naive-classifier baseline for 3-class mechanism prediction.

Computes the macro-F1 and per-class AUROC that a classifier with no learned
signal achieves on the GOF/DN/LOF task, so the floor can be reported alongside
the real feature scores in reports/run6/report_classifier.md (the `naive baseline` row).

To stay consistent with the experiment, this reuses the project's own
cross-validation split functions (gene_split_cv / family_split_cv), the same
3-class labels (label_3class), and the same metrics (macro-F1 via f1_score,
per-class one-vs-rest AUROC via roc_auc_score), averaged over the same 5 seeds.
Only the estimator differs: each rule is fitted from the corresponding training fold.

Three dummy strategies are reported:
  - most_frequent : always predict the majority class (LOF). This is the
                    stricter reference and the value reported in the tables.
  - prior         : same predictions as most_frequent for a single-label argmax.
  - stratified    : predict randomly in proportion to class frequencies.

Reported values are mean ± std across the 5 seeds (per-seed value = mean over
that seed's folds), matching how the experiment aggregates seeds.

  Input : data/valid_variants.json   (label_3class, gene per variant)
          data/pfam_families.json    (gene -> Pfam family, for family-split)
  Output: results/<run>/naive_baseline.json (path from paths.NAIVE_BASELINE_JSON)
          plus a summary table to stdout

Usage:
    python -m esm2_mech.experiments.mechanism.naive_baseline
"""

from __future__ import annotations

import functools
import json
from collections import Counter

import numpy as np

from esm2_mech.utils.constants import (
    BOOTSTRAP_N_RESAMPLES,
    MECHANISM_CLASSES,
    N_FOLDS,
    N_SEEDS,
)
from esm2_mech.utils.data import labeled_variant_fingerprint, load_pfam_map, pfam_fingerprint
from esm2_mech.utils.io import write_result_json
from esm2_mech.utils.paths import NAIVE_BASELINE_JSON, PFAM_JSON, VALID_VARIANTS_JSON
from esm2_mech.utils.splits import family_split_cv, gene_split_cv
from esm2_mech.utils.classification import validate_complete_classification_splits
from esm2_mech.utils.metrics import (
    aggregate_folds,
    compute_metrics,
    empty_aggregate_metrics,
    family_frequency_reference,
    featureless_reference,
)

print = functools.partial(print, flush=True)

STRATEGIES = ["most_frequent", "prior", "stratified", "family_frequency"]


def _unscorable_reference(contract, reason):
    result = empty_aggregate_metrics(
        MECHANISM_CLASSES, contract["requested_folds"], reason
    )
    result.update(
        {
            "status": "unscorable",
            "classes": list(MECHANISM_CLASSES),
            "eligible_rows": contract["eligible_rows"],
            "out_of_fold_rows": 0,
            "held_out_unit": contract.get("held_out_unit"),
            "group_count": contract.get("group_count"),
            "unscorable_reason": reason,
            "split_validation": contract,
        }
    )
    return result


def _eval_reference_one_seed(strategy, labels, families, splits, contract, seed):
    """Score one training-fold reference through the shared metric contract."""
    if contract["status"] != "valid":
        return _unscorable_reference(contract, "split_validation_failed")
    fold_predictions = []
    try:
        for fold_index, (train_idx, test_idx) in enumerate(splits):
            if strategy == "family_frequency":
                predictions, probabilities = family_frequency_reference(
                    labels[train_idx],
                    families[train_idx],
                    families[test_idx],
                    MECHANISM_CLASSES,
                )
            else:
                predictions, probabilities = featureless_reference(
                    labels[train_idx],
                    len(test_idx),
                    MECHANISM_CLASSES,
                    strategy,
                    seed + fold_index,
                )
            fold_predictions.append((predictions, probabilities))
    except ValueError as error:
        return _unscorable_reference(contract, str(error))

    fold_results = []
    for (_train_idx, test_idx), (predictions, probabilities) in zip(
        splits, fold_predictions
    ):
        fold_results.append(
            compute_metrics(
                labels[test_idx], predictions, probabilities, MECHANISM_CLASSES
            )
        )
    aggregate = aggregate_folds(fold_results, MECHANISM_CLASSES, contract["requested_folds"])
    aggregate.update(
        {
            "status": "success",
            "classes": list(MECHANISM_CLASSES),
            "eligible_rows": contract["eligible_rows"],
            "out_of_fold_rows": contract["eligible_rows"],
            "held_out_unit": contract.get("held_out_unit"),
            "group_count": contract.get("group_count"),
            "split_validation": contract,
        }
    )
    return aggregate


def evaluate(strategy, split_name, labels, genes, pfam_map, n_seeds=N_SEEDS, n_folds=N_FOLDS):
    """Mean ± std across n_seeds for one (strategy, split) cell."""
    per_seed = []
    for seed in range(n_seeds):
        if split_name == "gene":
            splits = gene_split_cv(genes, n_folds=n_folds, seed=seed)
        else:
            splits = family_split_cv(genes, pfam_map, n_folds=n_folds, seed=seed)
        groups = (
            genes
            if split_name == "gene"
            else np.array([pfam_map.get(gene) for gene in genes], dtype=object)
        )
        families = np.array([pfam_map.get(gene) for gene in genes], dtype=object)
        contract = validate_complete_classification_splits(
            splits,
            requested_folds=n_folds,
            eligible_rows=np.concatenate([test for _train, test in splits]),
            labels=labels,
            classes=MECHANISM_CLASSES,
            groups=groups,
            held_out_unit=split_name,
        )
        per_seed.append(
            _eval_reference_one_seed(
                strategy, labels, families, splits, contract, seed
            )
        )

    unavailable_seeds = [
        seed for seed, value in enumerate(per_seed) if value["status"] != "success"
    ]
    result = {
        "status": "unavailable" if unavailable_seeds else "success",
        "classes": list(MECHANISM_CLASSES),
        "n_seeds": n_seeds,
        "per_seed": per_seed,
        "unavailable_seeds": unavailable_seeds,
    }
    metric_names = [
        "macro_f1",
        "balanced_accuracy",
        "macro_auroc",
        *[f"f1_{class_name}" for class_name in MECHANISM_CLASSES],
        *[f"auroc_{class_name}" for class_name in MECHANISM_CLASSES],
        *[f"auprc_{class_name}" for class_name in MECHANISM_CLASSES],
    ]
    for metric_name in metric_names:
        values = [value.get(f"{metric_name}_mean") for value in per_seed]
        if unavailable_seeds or any(value is None for value in values):
            result[f"{metric_name}_mean"] = None
            result[f"{metric_name}_std"] = None
        else:
            result[f"{metric_name}_mean"] = float(np.mean(values))
            result[f"{metric_name}_std"] = float(np.std(values))
    return result


def floor_macro_f1_ci(labels, genes, pfam_map, seed=0, n_boot=BOOTSTRAP_N_RESAMPLES):
    """Cluster-bootstrap CIs for the most_frequent floor macro-F1.

    The majority class is refitted inside each fold and bootstrap draw. The gene CI
    resamples whole genes; the family CI resamples whole Pfam families (unannotated
    genes excluded, matching family-split CV). Both metrics are scored within fold
    and discard a draw when a fold loses a mechanism class.
    """
    output = {}
    for split_name in ("gene", "family"):
        point = evaluate(
            "most_frequent",
            split_name,
            labels,
            genes,
            pfam_map,
            n_seeds=1,
            n_folds=N_FOLDS,
        )["macro_f1_mean"]
        output[split_name] = {
            "point": point,
            "ci_low": None,
            "ci_high": None,
            "ci_suppressed": True,
            "missing": True,
            "reason": "blocked_by_audit_1_4",
            "n_resamples": 0,
            "n_resamples_total": 0,
            "n_clusters": (
                len(set(genes.tolist()))
                if split_name == "gene"
                else len({pfam_map.get(gene) for gene in genes} - {None})
            ),
        }
    return output


def main() -> None:
    with open(VALID_VARIANTS_JSON) as fh:
        variants = json.load(fh)
    pfam_map = load_pfam_map(PFAM_JSON)

    labels = np.array([v["label_3class"] for v in variants])
    genes = np.array([v["gene"] for v in variants])

    class_distribution = dict(Counter(labels))
    print(f"n = {len(labels)}  class distribution = {class_distribution}")
    print(f"Averaging over {N_SEEDS} seeds, {N_FOLDS}-fold CV\n")

    results = {
        "input_fingerprints": {
            "labeled_variants": labeled_variant_fingerprint(variants, labels),
            "pfam_assignments": pfam_fingerprint(pfam_map, genes.tolist()),
        },
        "analysis_parameters": {
            "n_seeds": N_SEEDS,
            "n_folds": N_FOLDS,
            "n_bootstrap_resamples": BOOTSTRAP_N_RESAMPLES,
        },
        "n_variants": int(len(labels)),
        "class_distribution": {k: int(v) for k, v in class_distribution.items()},
        "n_seeds": N_SEEDS,
        "n_folds": N_FOLDS,
        "by_strategy": {},
    }

    header = f"{'strategy':14} {'split':7} {'macro_f1':>15}  " + "  ".join(
        f"{c:>5}" for c in MECHANISM_CLASSES
    )
    print(header)

    for strategy in STRATEGIES:
        results["by_strategy"][strategy] = {}
        for split_name in ("gene", "family"):
            cell = evaluate(strategy, split_name, labels, genes, pfam_map)
            results["by_strategy"][strategy][split_name] = cell
            if cell["status"] != "success":
                print(f"{strategy:14} {split_name:7} {'Unscorable':>15}")
                continue
            macro = f"{cell['macro_f1_mean']:.3f} ± {cell['macro_f1_std']:.3f}"
            auroc_str = "  ".join(
                (
                    "NA"
                    if cell[f"auroc_{class_name}_mean"] is None
                    else f"{cell[f'auroc_{class_name}_mean']:.3f}"
                )
                for class_name in MECHANISM_CLASSES
            )
            print(f"{strategy:14} {split_name:7} {macro:>15}  {auroc_str}")

    print("\nMost_frequent floor cluster-bootstrap CIs (macro-F1)...")
    floor_ci = floor_macro_f1_ci(labels, genes, pfam_map)
    results["most_frequent_floor_ci"] = floor_ci
    for split_name in ("gene", "family"):
        cell = floor_ci[split_name]
        point_str = f"{cell['point']:.3f}" if cell["point"] is not None else "n/a"
        if cell["ci_low"] is not None and cell["ci_high"] is not None:
            ci_str = f"95% CI [{cell['ci_low']:.3f}, {cell['ci_high']:.3f}]"
        else:
            ci_str = "95% CI suppressed (too few valid resamples)"
        print(
            f"  {split_name:7} point {point_str}  {ci_str}  "
            f"({cell['reason']})"
        )

    NAIVE_BASELINE_JSON.parent.mkdir(parents=True, exist_ok=True)
    write_result_json(NAIVE_BASELINE_JSON, results, seeds=list(range(N_SEEDS)), indent=2)
    print(f"\nResults written to {NAIVE_BASELINE_JSON}")


if __name__ == "__main__":
    main()
