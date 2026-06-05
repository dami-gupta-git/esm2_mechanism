"""Dependency-aware inference: cluster bootstrap and label-permutation tests.

Mechanism labels are gene-level and variants cluster within genes (genes within
families), so observations are not independent and the effective sample size is the
cluster count, not the variant count. Seed-to-seed spread only reshuffles CV folds on
fixed data and understates the true uncertainty. These helpers replace it with:

  - cluster_bootstrap_ci: a confidence interval that resamples whole clusters
    (genes, or families) with replacement and recomputes the metric on each resample.
  - label_permutation_pvalue: a p-value against chance from shuffling the labels and
    recomputing the metric; the shuffle is cluster-aware (one label per cluster) so it
    respects the gene-level label structure.

See reports/run6/STATS_PLAN.md for the full rationale.
"""

from __future__ import annotations

import functools
from typing import Callable

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import f1_score, roc_auc_score

from esm2_mech.utils.constants import (
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_N_RESAMPLES,
    MECHANISM_CLASSES,
    PERMUTATION_N_RESAMPLES,
)

print = functools.partial(print, flush=True)


def average_oof_over_seeds(oof_list: list[dict | None]) -> dict | None:
    """Collapse per-seed out-of-fold predictions to one proba-per-variant.

    Each entry is a probe's OOF dict {"y_true", "proba", "genes", "row_ids"} from one
    seed, where row_ids index a fixed per-row array (e.g. a family's variant rows). A
    variant appears once per seed's CV; averaging its proba across seeds gives a single
    de-duplicated prediction per variant, so a downstream gene-cluster bootstrap counts
    each variant once instead of n_seeds times (which would falsely narrow the CI).

    None entries (seeds with no scorable fold) are skipped. Returns a single OOF dict
    keyed back to unique row_ids (sorted), or None if no entry had data.
    """
    valid = [oof for oof in oof_list if oof is not None and len(oof["row_ids"])]
    if not valid:
        return None

    proba_sum: dict = {}
    proba_count: dict = {}
    y_by_row: dict = {}
    gene_by_row: dict = {}
    for oof in valid:
        for pos, row in enumerate(oof["row_ids"]):
            row = int(row)
            vec = np.asarray(oof["proba"][pos], dtype=float)
            if row in proba_sum:
                proba_sum[row] = proba_sum[row] + vec
                proba_count[row] += 1
            else:
                proba_sum[row] = vec.copy()
                proba_count[row] = 1
            # y_true and gene are constant per row across seeds; record once.
            y_by_row.setdefault(row, oof["y_true"][pos])
            gene_by_row.setdefault(row, oof["genes"][pos])

    rows_sorted = sorted(proba_sum.keys())
    proba = np.array([proba_sum[row] / proba_count[row] for row in rows_sorted])
    return {
        "y_true": np.array([y_by_row[row] for row in rows_sorted]),
        "proba": proba,
        "genes": np.array([gene_by_row[row] for row in rows_sorted], dtype=object),
        "row_ids": np.array(rows_sorted, dtype=int),
    }


def _cluster_to_rows(clusters: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Group row indices by cluster id. Returns (unique_clusters, row_arrays) aligned."""
    order: dict = {}
    for row, cluster in enumerate(clusters):
        order.setdefault(cluster, []).append(row)
    unique = np.array(list(order.keys()), dtype=object)
    rows = [np.array(order[c], dtype=int) for c in unique]
    return unique, rows


def cluster_bootstrap_ci(
    clusters: np.ndarray,
    metric_fn: Callable[[np.ndarray], float | None],
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    seed: int = 0,
) -> dict:
    """Percentile CI for a metric via a cluster bootstrap.

    `clusters` is a per-row array of cluster ids (e.g. gene or Pfam family). Each
    resample draws len(unique clusters) clusters with replacement, gathers all rows
    of the drawn clusters (a cluster drawn k times contributes its rows k times), and
    calls `metric_fn(row_indices)` — a closure over the held data that returns the
    metric on those rows, or None/NaN when undefined on a resample (e.g. a class
    absent). Undefined resamples are dropped from the percentile, not imputed.

    The point estimate is metric_fn over all rows. Returns point, ci_low, ci_high,
    n_resamples (the count that contributed), n_clusters.
    """
    unique, cluster_rows = _cluster_to_rows(np.asarray(clusters))
    n_clusters = len(unique)
    all_rows = np.arange(len(clusters))
    point = metric_fn(all_rows)

    rng = np.random.RandomState(seed)
    stats: list[float] = []
    for _ in range(n_resamples):
        drawn = rng.randint(0, n_clusters, size=n_clusters)
        rows = np.concatenate([cluster_rows[i] for i in drawn])
        value = metric_fn(rows)
        if value is not None and np.isfinite(value):
            stats.append(float(value))

    if not stats:
        return {
            "point": float(point) if point is not None and np.isfinite(point) else None,
            "ci_low": None,
            "ci_high": None,
            "n_resamples": 0,
            "n_clusters": int(n_clusters),
        }
    lo_pct = (1.0 - ci_level) / 2.0 * 100.0
    hi_pct = (1.0 + ci_level) / 2.0 * 100.0
    return {
        "point": float(point) if point is not None and np.isfinite(point) else None,
        "ci_low": float(np.percentile(stats, lo_pct)),
        "ci_high": float(np.percentile(stats, hi_pct)),
        "n_resamples": len(stats),
        "n_clusters": int(n_clusters),
    }


def bootstrap_mechanism_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    clusters: np.ndarray,
    classes: list[str] = MECHANISM_CLASSES,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    seed: int = 0,
) -> dict:
    """Cluster-bootstrap CIs for macro-F1 and per-class one-vs-rest AUROC.

    `proba` columns must be aligned to `classes` (use utils.metrics.align_proba). Each
    metric reuses the same resampled clusters via cluster_bootstrap_ci. Returns
    {"macro_f1": {...}, "auroc_GOF": {...}, ...} where each value is the CI dict.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = np.array([classes[col] for col in proba.argmax(axis=1)])

    def _macro_f1(rows: np.ndarray) -> float:
        return float(f1_score(y_true[rows], pred[rows], average="macro", zero_division=0))

    out: dict = {
        "macro_f1": cluster_bootstrap_ci(
            clusters, _macro_f1, n_resamples=n_resamples, ci_level=ci_level, seed=seed
        )
    }

    for col_idx, cls in enumerate(classes):
        def _auroc(rows: np.ndarray, _col=col_idx, _cls=cls) -> float | None:
            y_bin = (y_true[rows] == _cls).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                return None
            return float(roc_auc_score(y_bin, proba[rows, _col]))

        out[f"auroc_{cls}"] = cluster_bootstrap_ci(
            clusters, _auroc, n_resamples=n_resamples, ci_level=ci_level, seed=seed
        )
    return out


def _permute_labels(
    labels: np.ndarray, groups: np.ndarray | None, rng: np.random.RandomState
) -> np.ndarray:
    """Permute labels. With `groups`, shuffle one label per group then broadcast back.

    Mechanism labels are constant within a gene, so a variant-level shuffle would break
    that structure and build an unrealistically easy null. Passing groups=genes shuffles
    at the gene level: each gene keeps a single (permuted) label across its variants.
    """
    if groups is None:
        return rng.permutation(labels)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    group_label = {g: labels[groups == g][0] for g in unique}
    permuted = rng.permutation([group_label[g] for g in unique])
    mapping = dict(zip(unique, permuted))
    return np.array([mapping[g] for g in groups])


def _permutation_null_value(
    run_metric_fn: Callable[[np.ndarray], float | None],
    labels: np.ndarray,
    groups: np.ndarray | None,
    child_seed: int,
) -> float | None:
    """Shuffle labels with an independent seeded RNG, then recompute the metric.

    Each call gets its own `RandomState(child_seed)` so the permutations are
    reproducible and order-independent — a requirement for running them in parallel,
    where the sequential RNG state of a single shared generator cannot be relied on.
    """
    rng = np.random.RandomState(child_seed)
    permuted = _permute_labels(np.asarray(labels), groups, rng)
    value = run_metric_fn(permuted)
    if value is not None and np.isfinite(value):
        return float(value)
    return None


def label_permutation_pvalue(
    run_metric_fn: Callable[[np.ndarray], float | None],
    labels: np.ndarray,
    groups: np.ndarray | None = None,
    n_permutations: int = PERMUTATION_N_RESAMPLES,
    seed: int = 0,
    alternative: str = "greater",
    n_jobs: int = -1,
) -> dict:
    """One-sided permutation p-value for a metric against the label-shuffled null.

    `run_metric_fn(labels)` must recompute the FULL cross-validated metric for a label
    vector — i.e. it refits the probe. The labels are permuted before each refit, so
    this cannot be computed from fixed out-of-fold predictions. Pass groups=genes for a
    gene-level shuffle (the correct null when labels are gene-level). The p-value is
    (1 + #{null >= observed}) / (1 + n) for alternative="greater".

    The n_permutations refits are independent and run across cores via joblib
    (n_jobs=-1 = all cores). Each permutation draws from its own RNG seeded by a
    SeedSequence spawned from `seed`, so the null distribution is identical to a
    serial run regardless of how the work is scheduled.
    """
    observed = run_metric_fn(labels)

    # Spawn one independent child seed per permutation up front. SeedSequence.spawn
    # guarantees statistically independent, reproducible streams without relying on
    # a single generator's sequential state (which parallel execution would break).
    child_seqs = np.random.SeedSequence(seed).spawn(n_permutations)
    child_seeds = [int(s.generate_state(1)[0]) for s in child_seqs]

    null_values = Parallel(n_jobs=n_jobs)(
        delayed(_permutation_null_value)(run_metric_fn, labels, groups, child_seed)
        for child_seed in child_seeds
    )
    null = [value for value in null_values if value is not None]

    null_arr = np.array(null)
    if observed is None or not np.isfinite(observed) or len(null_arr) == 0:
        return {
            "observed": float(observed) if observed is not None and np.isfinite(observed) else None,
            "p_value": None,
            "null_mean": float(np.mean(null_arr)) if len(null_arr) else None,
            "n_permutations": int(len(null_arr)),
        }
    if alternative == "greater":
        extreme = int(np.sum(null_arr >= observed))
    elif alternative == "less":
        extreme = int(np.sum(null_arr <= observed))
    else:
        raise ValueError(f"alternative must be 'greater' or 'less', got {alternative!r}")
    p_value = (1 + extreme) / (1 + len(null_arr))
    return {
        "observed": float(observed),
        "p_value": float(p_value),
        "null_mean": float(np.mean(null_arr)),
        "n_permutations": int(len(null_arr)),
    }
