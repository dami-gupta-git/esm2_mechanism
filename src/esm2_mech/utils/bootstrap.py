"""Cluster bootstrap CIs and label-permutation tests for gene/family-clustered data."""

from __future__ import annotations

import functools
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score

from esm2_mech.utils.metrics import binary_class_target, fold_macro_f1
from esm2_mech.utils.constants import (
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_MAX_DISCARD_FRAC,
    BOOTSTRAP_MIN_VALID_FRAC,
    BOOTSTRAP_N_RESAMPLES,
    MECHANISM_CLASSES,
    PERMUTATION_N_RESAMPLES,
)

print = functools.partial(print, flush=True)

# Cannot collide with a real Pfam accession ("PF" + digits).
UNANNOTATED_CLUSTER_PREFIX = "__no_pfam__:"


@dataclass(frozen=True)
class BootstrapMetricResult:
    """One bootstrap metric value plus the reason an undefined draw was rejected.

    Plain floats and ``None`` remain valid metric-function returns. Use this result
    when a caller has more than one possible failure mode and the parent process
    needs the per-draw reason. The object is returned from each joblib worker, so
    reason accounting does not depend on mutating worker-local closure state.
    """

    value: float | None
    discard_reason: str | None = None


def average_oof_over_seeds(oof_list: list[dict | None]) -> dict | None:
    """Average per-seed OOF probas to one prediction per variant.

    The result deliberately carries no fold assignment: each seed reshuffles the
    folds, so an averaged row has no single fold and its metrics could only be scored
    on the pooled concatenation. Use stack_oof_over_seeds for anything that computes
    a ranking metric; a fold-aware consumer handed this dict raises rather than
    silently pooling.
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


def oof_score_arms(oof: dict, label: str) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Normalize an OOF dict to the (proba, folds, fold_ids) blocks a metric scores in.

    Every fold is fitted independently, so its probabilities carry their own scale
    and their own intercept. Ranking one concatenation of all folds compares scores
    that were never on a common scale, which leaves a strong signal roughly intact
    but drives a weak one below 0.5. Metrics therefore score inside a fold and
    average across folds, which is only possible if the fold survived collection.

    A single-seed OOF gives one block set. A multi-seed OOF from
    stack_oof_over_seeds gives one block set per seed, because each seed reshuffles
    the fold assignment and a seed's probabilities are not comparable with another
    seed's either.
    """
    if "proba_by_seed" in oof:
        return [
            (np.asarray(proba), np.asarray(folds), np.unique(np.asarray(folds)))
            for proba, folds in zip(oof["proba_by_seed"], oof["folds_by_seed"])
        ]
    if "folds" not in oof:
        raise KeyError(
            f"{label}: this OOF has no 'folds' array, so its metrics can only be "
            "scored on the pooled concatenation, which is the defect this code "
            "exists to prevent. Produce it with a probe that records the fold, or "
            "combine seeds with stack_oof_over_seeds rather than "
            "average_oof_over_seeds."
        )
    folds = np.asarray(oof["folds"])
    return [(np.asarray(oof["proba"]), folds, np.unique(folds))]


def folds_to_arms(
    proba: np.ndarray, folds: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """One block set from a bare (proba, folds) pair, for callers holding arrays."""
    folds = np.asarray(folds)
    return [(np.asarray(proba), folds, np.unique(folds))]


def score_within_folds(
    rows: np.ndarray,
    arms: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    score_fn: Callable[[np.ndarray, np.ndarray], float | None],
    weights: np.ndarray | None = None,
) -> float | None:
    """Mean of score_fn over every (arm, fold) block of `rows`, or None if any fails.

    score_fn(block_rows, proba) returns the metric for one fold, or None when it is
    undefined there — typically because the resample dropped every variant of a rare
    class from that fold. The whole resample is then discarded rather than scored
    over the folds that survive: a draw that averages over a different set of folds,
    or a different set of classes, is estimating a different quantity, and mixing
    those into one percentile interval is what pushes an interval off its own point
    estimate.
    """
    values = []
    for proba, folds, fold_ids in arms:
        row_folds = folds[rows]
        for fold in fold_ids:
            block = rows[row_folds == fold]
            if len(block) == 0:
                return None
            value = score_fn(block, proba)
            if value is None or not np.isfinite(value):
                return None
            values.append(float(value))
    if not values:
        return None
    return float(np.mean(values))


def stack_oof_over_seeds(oof_list: list[dict | None]) -> dict | None:
    """Align per-seed OOF dicts on the variants every seed scored, keeping folds.

    The seed-averaging alternative collapses the per-seed probabilities into one
    vector and has no fold to report, so its metrics can only be pooled. This keeps
    each seed's probabilities and fold assignment separate; a metric then scores
    inside each seed's folds and averages over folds and seeds.
    """
    valid = [oof for oof in oof_list if oof is not None and len(oof["row_ids"])]
    if not valid:
        return None
    for oof in valid:
        if "folds" not in oof:
            raise KeyError(
                "stack_oof_over_seeds: a per-seed OOF has no 'folds' array"
            )

    pos_by_row = [
        {int(row): pos for pos, row in enumerate(oof["row_ids"])} for oof in valid
    ]
    shared = sorted(set.intersection(*[set(m) for m in pos_by_row]))
    if not shared:
        return None
    dropped = sum(len(m) - len(shared) for m in pos_by_row)
    if dropped:
        print(
            f"  [stack_oof] {len(shared)} variants scored by all {len(valid)} seeds, "
            f"{dropped} seed-specific rows dropped"
        )

    first = valid[0]
    first_idx = np.array([pos_by_row[0][row] for row in shared], dtype=int)
    y_true = np.asarray(first["y_true"])[first_idx]
    for oof, pos_map in zip(valid[1:], pos_by_row[1:]):
        idx = np.array([pos_map[row] for row in shared], dtype=int)
        if not np.array_equal(np.asarray(oof["y_true"])[idx], y_true):
            raise RuntimeError(
                "stack_oof_over_seeds: two seeds disagree on a variant's label — "
                "the OOF row spaces are not the same variants"
            )

    return {
        "y_true": y_true,
        "genes": np.asarray(first["genes"], dtype=object)[first_idx],
        "row_ids": np.array(shared, dtype=int),
        "proba_by_seed": [
            np.asarray(oof["proba"])[
                np.array([pos_map[row] for row in shared], dtype=int)
            ]
            for oof, pos_map in zip(valid, pos_by_row)
        ],
        "folds_by_seed": [
            np.asarray(oof["folds"])[
                np.array([pos_map[row] for row in shared], dtype=int)
            ]
            for oof, pos_map in zip(valid, pos_by_row)
        ],
    }


def _cluster_to_rows(clusters: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Group row indices by cluster id."""
    order: dict = {}
    for row, cluster in enumerate(clusters):
        order.setdefault(cluster, []).append(row)
    unique = np.array(list(order.keys()), dtype=object)
    rows = [np.array(order[c], dtype=int) for c in unique]
    return unique, rows


def _clean_scalar(value) -> float | None:
    """Return float or None for non-finite values."""
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _clean_metric_result(
    result: float | None | BootstrapMetricResult,
) -> tuple[float | None, str | None]:
    """Normalize one metric result without losing its worker-side failure reason."""
    if isinstance(result, BootstrapMetricResult):
        value = _clean_scalar(result.value)
        if value is not None and result.discard_reason is not None:
            raise ValueError("a finite bootstrap metric cannot carry a discard reason")
        return value, result.discard_reason if value is None else None
    return _clean_scalar(result), None


def _evaluate_metric_fns(
    metric_fns: list[Callable[[np.ndarray], dict]], rows: np.ndarray
) -> tuple[dict, dict]:
    """Return flattened metric values and per-metric rejection reasons."""
    values: dict = {}
    reasons: dict = {}
    for metric_fn in metric_fns:
        for name, result in metric_fn(rows).items():
            value, reason = _clean_metric_result(result)
            values[name] = value
            if reason is not None:
                reasons[name] = reason
    return values, reasons


def _multi_bootstrap_resample_values(
    metric_fns: list[Callable[[np.ndarray], dict]],
    cluster_rows: list[np.ndarray],
    n_clusters: int,
    child_seed: int,
) -> tuple[dict, dict]:
    """Draw once and return metric values plus worker-side rejection reasons."""
    rng = np.random.RandomState(child_seed)
    drawn = rng.randint(0, n_clusters, size=n_clusters)
    rows = np.concatenate([cluster_rows[i] for i in drawn])
    return _evaluate_metric_fns(metric_fns, rows)


def _summarize_bootstrap(
    point: float | None,
    stats: list[float],
    n_resamples: int,
    n_clusters: int,
    ci_level: float,
    min_valid_frac: float,
) -> dict:
    """Percentile CI from surviving replicates of one metric."""
    valid_frac = len(stats) / n_resamples if n_resamples else 0.0
    base = {
        "point": point,
        "n_resamples": len(stats),
        "n_resamples_total": int(n_resamples),
        "n_discarded": int(n_resamples) - len(stats),
        "discard_frac": float(1.0 - valid_frac),
        "valid_frac": float(valid_frac),
        "n_clusters": int(n_clusters),
    }
    if not stats or valid_frac < min_valid_frac:
        return {**base, "ci_low": None, "ci_high": None, "ci_suppressed": True}
    lo_pct = (1.0 - ci_level) / 2.0 * 100.0
    hi_pct = (1.0 + ci_level) / 2.0 * 100.0
    return {
        **base,
        "ci_low": float(np.percentile(stats, lo_pct)),
        "ci_high": float(np.percentile(stats, hi_pct)),
        "ci_suppressed": False,
    }


_DEFAULT_DISCARD_REASON = (
    "the metric could not be scored on that many draws (cause not identified by "
    "the caller)"
)


def _warn_on_discards(
    summaries: dict, discard_reasons: dict[str, str | Callable[[], str]] | None = None
) -> None:
    """Print a fault warning for any metric that lost more than the tolerated share.

    With every fold carrying every class several families over in the real splits,
    a fold losing a class entirely needs an improbable draw and should stay far
    below the tolerance — but that is only one of several reasons a metric_fn can
    return None (a ratio's denominator collapsing, or too few surviving rows, are
    others). This function only sees discard counts, not causes, so it never
    asserts one: callers that know their own failure mode pass it via
    `discard_reasons`; callers that do not get an honest placeholder rather than
    a guessed explanation. Per-draw reasons returned by workers take precedence
    over a caller's static description. A rate above tolerance points at the
    resampling unit or the fold/metric construction rather than at sampling noise,
    and is not something to absorb into a wider interval either way.
    """
    discard_reasons = discard_reasons or {}
    for name, summary in summaries.items():
        discard_frac = summary.get("discard_frac")
        if discard_frac is None or discard_frac <= BOOTSTRAP_MAX_DISCARD_FRAC:
            continue
        reason_counts = summary.get("discard_reason_counts") or {}
        if reason_counts:
            parts = [
                f"{count} draw(s): {reason.replace('_', ' ')}"
                for reason, count in sorted(reason_counts.items())
            ]
            unidentified = summary["n_discarded"] - sum(reason_counts.values())
            if unidentified:
                parts.append(f"{unidentified} draw(s): cause not identified")
            reason = "; ".join(parts)
        else:
            reason = discard_reasons.get(name)
            if callable(reason):
                reason = reason()
            reason = reason or _DEFAULT_DISCARD_REASON
        print(
            f"  [bootstrap] {name}: {summary['n_discarded']}/"
            f"{summary['n_resamples_total']} resamples discarded "
            f"({discard_frac:.1%}, tolerance {BOOTSTRAP_MAX_DISCARD_FRAC:.0%}) — "
            f"{reason}. Investigate the resampling unit and the metric construction "
            "before using this interval."
        )


def cluster_bootstrap_ci_multi(
    clusters: np.ndarray,
    metric_fns: list[Callable[[np.ndarray], dict]],
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    min_valid_frac: float = BOOTSTRAP_MIN_VALID_FRAC,
    seed: int = 0,
    n_jobs: int = -1,
    discard_reasons: dict[str, str | Callable[[], str]] | None = None,
) -> dict:
    """Cluster-bootstrap CIs for several metrics over one shared set of resamples.

    `discard_reasons` lets a caller name, per metric, why its metric_fn returns
    None on a resample. Different metric_fns fail for different reasons (a fold
    losing a class is only one of them — a ratio's denominator collapsing, or too
    few rows surviving, are others) and the discard warning must not assert a
    cause it cannot verify.
    """
    unique, cluster_rows = _cluster_to_rows(np.asarray(clusters))
    n_clusters = len(unique)
    all_rows = np.arange(len(clusters))
    points, point_reasons = _evaluate_metric_fns(metric_fns, all_rows)

    child_seqs = np.random.SeedSequence(seed).spawn(n_resamples)
    child_seeds = [int(s.generate_state(1)[0]) for s in child_seqs]

    replicates = Parallel(n_jobs=n_jobs)(
        delayed(_multi_bootstrap_resample_values)(
            metric_fns, cluster_rows, n_clusters, child_seed
        )
        for child_seed in child_seeds
    )

    out = {}
    for name in points:
        stats = [
            values[name]
            for values, _reasons in replicates
            if values.get(name) is not None
        ]
        summary = _summarize_bootstrap(
            points[name], stats, n_resamples, n_clusters, ci_level, min_valid_frac
        )
        reason_counts = Counter(
            reasons[name]
            for values, reasons in replicates
            if values.get(name) is None and reasons.get(name) is not None
        )
        if reason_counts:
            summary["discard_reason_counts"] = dict(reason_counts)
        if point_reasons.get(name) is not None:
            summary["point_invalid_reason"] = point_reasons[name]
        out[name] = summary
    _warn_on_discards(out, discard_reasons)
    return out


def within_stratum_bootstrap_ci(
    strata: np.ndarray,
    metric_fn: Callable[[np.ndarray], float | None],
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    min_valid_frac: float = BOOTSTRAP_MIN_VALID_FRAC,
    seed: int = 0,
    n_jobs: int = -1,
    discard_reason: str | Callable[[], str] | None = None,
) -> dict:
    """Bootstrap that resamples rows with replacement inside each stratum.

    For a probe whose prediction target is the stratum itself. Resampling whole
    strata, as the cluster bootstrap does, drops some of them from every draw, so
    each draw averages macro-F1 over a different and smaller set of classes. That
    shifts the value in one direction instead of scattering it, which is how a point
    estimate ends up outside its own interval. Keeping every stratum and resampling
    the rows inside it leaves the class set fixed across draws.
    """
    unique, stratum_rows = _cluster_to_rows(np.asarray(strata))
    all_rows = np.arange(len(strata))
    point = _clean_scalar(metric_fn(all_rows))

    child_seqs = np.random.SeedSequence(seed).spawn(n_resamples)
    child_seeds = [int(seq.generate_state(1)[0]) for seq in child_seqs]

    values = Parallel(n_jobs=n_jobs)(
        delayed(_within_stratum_resample_value)(metric_fn, stratum_rows, child_seed)
        for child_seed in child_seeds
    )
    stats = [value for value in values if value is not None]
    summary = _summarize_bootstrap(
        point, stats, n_resamples, len(unique), ci_level, min_valid_frac
    )
    _warn_on_discards({"within_stratum": summary}, {"within_stratum": discard_reason})
    return summary


def _within_stratum_resample_value(
    metric_fn: Callable[[np.ndarray], float | None],
    stratum_rows: list[np.ndarray],
    child_seed: int,
) -> float | None:
    """Draw one within-stratum resample and score it."""
    rng = np.random.RandomState(child_seed)
    rows = np.concatenate([
        rows_in[rng.randint(0, len(rows_in), size=len(rows_in))]
        for rows_in in stratum_rows
    ])
    return _clean_scalar(metric_fn(rows))


def cluster_bootstrap_ci(
    clusters: np.ndarray,
    metric_fn: Callable[
        [np.ndarray], float | None | BootstrapMetricResult
    ],
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    min_valid_frac: float = BOOTSTRAP_MIN_VALID_FRAC,
    seed: int = 0,
    n_jobs: int = -1,
    discard_reason: str | Callable[[], str] | None = None,
    metric_name: str = "metric",
) -> dict:
    """Single-metric front end to cluster_bootstrap_ci_multi.

    `discard_reason` names why an untagged `metric_fn` return is None. A
    `BootstrapMetricResult` supplies the reason for its own draw instead.
    `metric_name` identifies the result in warnings. See cluster_bootstrap_ci_multi.
    """
    key = metric_name
    out = cluster_bootstrap_ci_multi(
        clusters,
        [lambda rows: {key: metric_fn(rows)}],
        n_resamples=n_resamples,
        ci_level=ci_level,
        min_valid_frac=min_valid_frac,
        seed=seed,
        n_jobs=n_jobs,
        discard_reasons={key: discard_reason} if discard_reason else None,
    )
    return out[key]


def _subsample_resample_value(
    metric_fn: Callable[[np.ndarray], float | None],
    cluster_rows: list[np.ndarray],
    n_clusters: int,
    subsample_size: int,
    child_seed: int,
) -> float | None:
    """Draw one cluster subsample (without replacement) and return the metric."""
    rng = np.random.RandomState(child_seed)
    drawn = rng.choice(n_clusters, size=subsample_size, replace=False)
    rows = np.concatenate([cluster_rows[i] for i in drawn])
    value = metric_fn(rows)
    if value is not None and np.isfinite(value):
        return float(value)
    return None


def cluster_subsample_ci(
    clusters: np.ndarray,
    metric_fn: Callable[[np.ndarray], float | None],
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    subsample_frac: float = 0.632,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    min_valid_frac: float = BOOTSTRAP_MIN_VALID_FRAC,
    seed: int = 0,
    n_jobs: int = -1,
) -> dict:
    """CI via without-replacement cluster subsample, for distance/graph metrics that break under duplicate points."""
    unique, cluster_rows = _cluster_to_rows(np.asarray(clusters))
    n_clusters = len(unique)
    subsample_size = max(1, round(subsample_frac * n_clusters))
    all_rows = np.arange(len(clusters))
    point = metric_fn(all_rows)

    child_seqs = np.random.SeedSequence(seed).spawn(n_resamples)
    child_seeds = [int(s.generate_state(1)[0]) for s in child_seqs]

    values = Parallel(n_jobs=n_jobs)(
        delayed(_subsample_resample_value)(
            metric_fn, cluster_rows, n_clusters, subsample_size, child_seed
        )
        for child_seed in child_seeds
    )
    stats = [value for value in values if value is not None]

    valid_frac = len(stats) / n_resamples if n_resamples else 0.0
    base = {
        "point": float(point) if point is not None and np.isfinite(point) else None,
        "n_resamples": len(stats),
        "n_resamples_total": int(n_resamples),
        "valid_frac": float(valid_frac),
        "n_clusters": int(n_clusters),
        "subsample_size": int(subsample_size),
    }
    if not stats or valid_frac < min_valid_frac:
        return {**base, "ci_low": None, "ci_high": None, "ci_suppressed": True}
    lo_pct = (1.0 - ci_level) / 2.0 * 100.0
    hi_pct = (1.0 + ci_level) / 2.0 * 100.0
    return {
        **base,
        "ci_low": float(np.percentile(stats, lo_pct)),
        "ci_high": float(np.percentile(stats, hi_pct)),
        "ci_suppressed": False,
    }


def _paired_bootstrap_resample_values(
    metric_fn_a: Callable[[np.ndarray], float | None],
    metric_fn_b: Callable[[np.ndarray], float | None],
    cluster_rows: list[np.ndarray],
    n_clusters: int,
    child_seed: int,
) -> tuple[float | None, float | None]:
    """Draw one cluster resample and score both arms on the identical drawn rows."""
    rng = np.random.RandomState(child_seed)
    drawn = rng.randint(0, n_clusters, size=n_clusters)
    rows = np.concatenate([cluster_rows[i] for i in drawn])
    value_a = metric_fn_a(rows)
    value_b = metric_fn_b(rows)
    value_a = float(value_a) if value_a is not None and np.isfinite(value_a) else None
    value_b = float(value_b) if value_b is not None and np.isfinite(value_b) else None
    return value_a, value_b


def _paired_cluster_bootstrap_diff_ci(
    resample_clusters: np.ndarray,
    metric_fn_a: Callable[[np.ndarray], float | None],
    metric_fn_b: Callable[[np.ndarray], float | None],
    n_resamples: int,
    ci_level: float,
    min_valid_frac: float,
    seed: int,
    n_jobs: int,
    discard_reason: str | Callable[[], str] | None,
) -> dict:
    """Shared resampling machinery for both pairing modes."""
    unique, cluster_rows = _cluster_to_rows(np.asarray(resample_clusters))
    n_clusters = len(unique)
    all_rows = np.arange(len(resample_clusters))
    point_a = metric_fn_a(all_rows)
    point_b = metric_fn_b(all_rows)
    point_a = float(point_a) if point_a is not None and np.isfinite(point_a) else None
    point_b = float(point_b) if point_b is not None and np.isfinite(point_b) else None
    point_diff = point_a - point_b if point_a is not None and point_b is not None else None

    child_seqs = np.random.SeedSequence(seed).spawn(n_resamples)
    child_seeds = [int(seq.generate_state(1)[0]) for seq in child_seqs]

    paired_values = Parallel(n_jobs=n_jobs)(
        delayed(_paired_bootstrap_resample_values)(
            metric_fn_a, metric_fn_b, cluster_rows, n_clusters, child_seed
        )
        for child_seed in child_seeds
    )
    diffs = [
        value_a - value_b
        for value_a, value_b in paired_values
        if value_a is not None and value_b is not None
    ]

    valid_frac = len(diffs) / n_resamples if n_resamples else 0.0
    base = {
        "point_a": point_a,
        "point_b": point_b,
        "point_diff": point_diff,
        "n_resamples": len(diffs),
        "n_resamples_total": int(n_resamples),
        "n_discarded": int(n_resamples) - len(diffs),
        "discard_frac": float(1.0 - valid_frac),
        "valid_frac": float(valid_frac),
        "n_clusters": int(n_clusters),
    }
    _warn_on_discards(
        {"paired_diff": base}, {"paired_diff": discard_reason}
    )
    if not diffs or valid_frac < min_valid_frac:
        return {**base, "ci_low": None, "ci_high": None, "ci_suppressed": True}
    lo_pct = (1.0 - ci_level) / 2.0 * 100.0
    hi_pct = (1.0 + ci_level) / 2.0 * 100.0
    return {
        **base,
        "ci_low": float(np.percentile(diffs, lo_pct)),
        "ci_high": float(np.percentile(diffs, hi_pct)),
        "ci_suppressed": False,
    }


def paired_cluster_bootstrap_diff(
    clusters: np.ndarray,
    metric_fn_a: Callable[[np.ndarray], float | None],
    metric_fn_b: Callable[[np.ndarray], float | None],
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    min_valid_frac: float = BOOTSTRAP_MIN_VALID_FRAC,
    seed: int = 0,
    n_jobs: int = -1,
    discard_reason: str | Callable[[], str] | None = None,
) -> dict:
    """Paired CI on metric_a minus metric_b under one shared fold assignment."""
    return _paired_cluster_bootstrap_diff_ci(
        clusters,
        metric_fn_a,
        metric_fn_b,
        n_resamples=n_resamples,
        ci_level=ci_level,
        min_valid_frac=min_valid_frac,
        seed=seed,
        n_jobs=n_jobs,
        discard_reason=discard_reason,
    )


def independent_cluster_bootstrap_diff(
    clusters_a: np.ndarray,
    clusters_b: np.ndarray,
    metric_fn_a: Callable[[np.ndarray], float | None],
    metric_fn_b: Callable[[np.ndarray], float | None],
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    min_valid_frac: float = BOOTSTRAP_MIN_VALID_FRAC,
    seed: int = 0,
) -> dict:
    """CI on A minus B when the arms come from independent datasets."""
    _, cluster_rows_a = _cluster_to_rows(np.asarray(clusters_a))
    _, cluster_rows_b = _cluster_to_rows(np.asarray(clusters_b))
    n_clusters_a = len(cluster_rows_a)
    n_clusters_b = len(cluster_rows_b)

    point_a = _clean_scalar(metric_fn_a(np.arange(len(clusters_a))))
    point_b = _clean_scalar(metric_fn_b(np.arange(len(clusters_b))))
    point_diff = point_a - point_b if point_a is not None and point_b is not None else None

    seed_sequences = np.random.SeedSequence(seed).spawn(n_resamples)
    differences = []
    for seed_sequence in seed_sequences:
        seed_a, seed_b = seed_sequence.spawn(2)
        rng_a = np.random.RandomState(int(seed_a.generate_state(1)[0]))
        rng_b = np.random.RandomState(int(seed_b.generate_state(1)[0]))
        drawn_a = rng_a.randint(0, n_clusters_a, size=n_clusters_a)
        drawn_b = rng_b.randint(0, n_clusters_b, size=n_clusters_b)
        rows_a = np.concatenate([cluster_rows_a[index] for index in drawn_a])
        rows_b = np.concatenate([cluster_rows_b[index] for index in drawn_b])
        value_a = _clean_scalar(metric_fn_a(rows_a))
        value_b = _clean_scalar(metric_fn_b(rows_b))
        if value_a is not None and value_b is not None:
            differences.append(value_a - value_b)

    valid_frac = len(differences) / n_resamples if n_resamples else 0.0
    result = {
        "point_a": point_a,
        "point_b": point_b,
        "point_diff": point_diff,
        "n_resamples": len(differences),
        "n_resamples_total": int(n_resamples),
        "valid_frac": float(valid_frac),
        "n_clusters_a": n_clusters_a,
        "n_clusters_b": n_clusters_b,
    }
    if not differences or valid_frac < min_valid_frac:
        return {**result, "ci_low": None, "ci_high": None, "ci_suppressed": True}
    tail = (1.0 - ci_level) / 2.0 * 100.0
    return {
        **result,
        "ci_low": float(np.percentile(differences, tail)),
        "ci_high": float(np.percentile(differences, 100.0 - tail)),
        "ci_suppressed": False,
    }


def paired_cluster_bootstrap_diff_cross_partition(
    resample_clusters: np.ndarray,
    metric_fn_a: Callable[[np.ndarray], float | None],
    metric_fn_b: Callable[[np.ndarray], float | None],
    sensitivity_clusters: np.ndarray | None = None,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    min_valid_frac: float = BOOTSTRAP_MIN_VALID_FRAC,
    seed: int = 0,
    n_jobs: int = -1,
    discard_reason: str | Callable[[], str] | None = None,
) -> dict:
    """Paired CI on a difference between arms scored under different CV partitions.

    resample_clusters must be the COARSER unit (family, not gene) so the
    family-split arm's variance is not understated.
    """
    primary = _paired_cluster_bootstrap_diff_ci(
        resample_clusters,
        metric_fn_a,
        metric_fn_b,
        n_resamples=n_resamples,
        ci_level=ci_level,
        min_valid_frac=min_valid_frac,
        seed=seed,
        n_jobs=n_jobs,
        discard_reason=discard_reason,
    )
    if sensitivity_clusters is not None:
        primary["gene_resampled_sensitivity"] = _paired_cluster_bootstrap_diff_ci(
            sensitivity_clusters,
            metric_fn_a,
            metric_fn_b,
            n_resamples=n_resamples,
            ci_level=ci_level,
            min_valid_frac=min_valid_frac,
            seed=seed,
            n_jobs=n_jobs,
            discard_reason=discard_reason,
        )
    return primary


def bootstrap_mechanism_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    clusters: np.ndarray,
    folds: np.ndarray,
    classes: list[str] = MECHANISM_CLASSES,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    seed: int = 0,
) -> dict:
    """Cluster-bootstrap CIs for macro-F1, per-class AUROC, AUPRC, prevalence and lift.

    Every metric is computed inside each fold and averaged over folds. `folds` is
    required: an optional fold argument that falls back to pooling is how the pooled
    ranking defect survived its own fix. Macro-F1 is on the same per-fold basis for
    consistency with the ranking metrics and with the threshold it is compared
    against; the pooled macro-F1 was not itself distorted, because a class is decided
    per row by argmax and no cross-fold comparison enters it.
    """
    return bootstrap_mechanism_metrics_from_oof(
        {"y_true": y_true, "proba": proba, "folds": folds},
        clusters,
        classes=classes,
        n_resamples=n_resamples,
        ci_level=ci_level,
        seed=seed,
    )


def bootstrap_mechanism_metrics_from_oof(
    oof: dict,
    clusters: np.ndarray,
    classes: list[str] = MECHANISM_CLASSES,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    seed: int = 0,
) -> dict:
    """As bootstrap_mechanism_metrics, taking the OOF dict rather than loose arrays.

    Accepts a multi-seed OOF from stack_oof_over_seeds, where the average runs over
    every seed's folds.
    """
    y_true = np.asarray(oof["y_true"])
    arms = oof_score_arms(oof, "bootstrap_mechanism_metrics")
    preds = [
        np.array([classes[col] for col in proba.argmax(axis=1)]) for proba, _, _ in arms
    ]
    pred_arms = [
        (pred, folds, fold_ids) for pred, (_, folds, fold_ids) in zip(preds, arms)
    ]

    def _macro_f1(rows: np.ndarray) -> dict:
        def _fold_f1(block: np.ndarray, arm_pred: np.ndarray) -> float | None:
            return fold_macro_f1(y_true, block, arm_pred, classes)
        return {"macro_f1": score_within_folds(rows, pred_arms, _fold_f1)}

    def _class_metrics(rows: np.ndarray, *, _col: int, _cls: str) -> dict:
        names = (f"auroc_{_cls}", f"auprc_{_cls}", f"prevalence_{_cls}", f"auprc_lift_{_cls}")

        def _fold_metric(block: np.ndarray, arm_proba: np.ndarray, which: str) -> float | None:
            y_bin = binary_class_target(y_true[block], _cls)
            if y_bin is None:
                return None
            scores = arm_proba[block, _col]
            if which == "auroc":
                return float(roc_auc_score(y_bin, scores))
            prevalence = float(y_bin.mean())
            if which == "prevalence":
                return prevalence
            auprc = float(average_precision_score(y_bin, scores))
            return auprc if which == "auprc" else auprc - prevalence

        return {
            name: score_within_folds(
                rows, arms, functools.partial(_fold_metric, which=which)
            )
            for name, which in zip(names, ("auroc", "auprc", "prevalence", "auprc_lift"))
        }

    metric_fns = [_macro_f1] + [
        functools.partial(_class_metrics, _col=col_idx, _cls=cls)
        for col_idx, cls in enumerate(classes)
    ]
    class_metric_reason = (
        "a fold's resampled rows lost the one-vs-rest positive or negative class"
    )
    discard_reasons = {"macro_f1": "a fold's resampled rows lost a mechanism class"}
    for cls in classes:
        discard_reasons.update({
            f"auroc_{cls}": class_metric_reason,
            f"auprc_{cls}": class_metric_reason,
            f"prevalence_{cls}": class_metric_reason,
            f"auprc_lift_{cls}": class_metric_reason,
        })
    out = cluster_bootstrap_ci_multi(
        clusters,
        metric_fns,
        n_resamples=n_resamples,
        ci_level=ci_level,
        seed=seed,
        discard_reasons=discard_reasons,
    )

    for metric_name, ci in out.items():
        if ci.get("ci_suppressed"):
            print(
                f"  [bootstrap] {metric_name}: CI suppressed — only "
                f"{ci['n_resamples']}/{ci['n_resamples_total']} resamples valid "
                f"({ci['valid_frac']:.0%}); the metric was undefined on the rest "
                f"(a fold lost the class on that resample). No CI reported."
            )
    return out


def attach_mechanism_ci(
    result: dict,
    oof: dict | None,
    clusters: np.ndarray | None,
    *,
    compute_ci: bool,
    classes: list[str] = MECHANISM_CLASSES,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    seed: int = 0,
) -> dict:
    """Attach fold-aware mechanism bootstrap intervals to an experiment result.

    `clusters` must be aligned to the rows in `oof`. Callers select the resampling
    unit, such as gene, Pfam family, clan, or sequence cluster. The OOF dict remains
    intact so the bootstrap scorer always receives its fold assignments.
    """
    if not compute_ci or oof is None:
        return result
    if clusters is None:
        raise ValueError("attach_mechanism_ci: clusters are required when CI is enabled")
    if len(clusters) != len(oof["y_true"]):
        raise ValueError(
            "attach_mechanism_ci: clusters and OOF rows are not aligned: "
            f"{len(clusters)} clusters for {len(oof['y_true'])} rows"
        )
    result["ci"] = bootstrap_mechanism_metrics_from_oof(
        oof,
        np.asarray(clusters),
        classes=classes,
        n_resamples=n_resamples,
        ci_level=ci_level,
        seed=seed,
    )
    return result


def family_or_gene_clusters(
    genes: np.ndarray, pfam_map: dict, is_family_split: bool
) -> np.ndarray:
    """Map genes to resampling clusters: families for family-split, genes for gene-split.

    Unannotated genes get singleton clusters (not dropped or shared) to preserve
    row alignment and avoid asserting unrelated genes are non-independent.
    """
    if not is_family_split:
        return genes
    return np.array([
        pfam_map[gene] if pfam_map.get(gene) else f"{UNANNOTATED_CLUSTER_PREFIX}{gene}"
        for gene in genes
    ])


def _align_oof_pair(oof_a: dict, oof_b: dict, label: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Index positions into each arm for variants both arms scored."""
    for name, oof in (("a", oof_a), ("b", oof_b)):
        if "row_ids" not in oof:
            raise KeyError(
                f"{label}: arm {name}'s OOF has no 'row_ids'; paired differences "
                "cannot be aligned positionally"
            )
    rows_a = {int(row): pos for pos, row in enumerate(oof_a["row_ids"])}
    rows_b = {int(row): pos for pos, row in enumerate(oof_b["row_ids"])}
    shared = sorted(set(rows_a) & set(rows_b))
    if not shared:
        print(f"  [paired] {label}: skipped — the arms share no scored variants")
        return None
    dropped = (len(rows_a) - len(shared)) + (len(rows_b) - len(shared))
    if dropped:
        print(
            f"  [paired] {label}: {len(shared)} shared variants, "
            f"{dropped} arm-specific rows dropped"
        )
    idx_a = np.array([rows_a[row] for row in shared], dtype=int)
    idx_b = np.array([rows_b[row] for row in shared], dtype=int)

    y_a = np.asarray(oof_a["y_true"])[idx_a]
    y_b = np.asarray(oof_b["y_true"])[idx_b]
    if not np.array_equal(y_a, y_b):
        raise RuntimeError(
            f"{label}: the two arms disagree on the label of a shared variant — "
            "the OOF row spaces are not the same variants"
        )
    return idx_a, idx_b


def paired_oof_diff(
    oof_a: dict | None,
    oof_b: dict | None,
    pfam_map: dict,
    label: str,
    classes: list[str] | None = None,
    metric: str = "macro_f1",
    pos_class: str | None = None,
    is_family_split: bool = True,
    cross_partition: bool = False,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    seed: int = 0,
) -> dict | None:
    """Paired cluster-bootstrap CI on metric(A) minus metric(B) for two OOF arms."""
    if oof_a is None or oof_b is None:
        print(f"  [paired] {label}: skipped — an arm has no OOF")
        return None
    if metric not in ("macro_f1", "auroc_binary", "auroc_one_vs_rest"):
        raise ValueError(f"{label}: unknown metric {metric!r}")
    if metric == "macro_f1" and not classes:
        raise ValueError(f"{label}: metric 'macro_f1' requires the `classes` list")
    if metric == "auroc_one_vs_rest":
        if not classes:
            raise ValueError(
                f"{label}: metric 'auroc_one_vs_rest' requires the `classes` list"
            )
        if pos_class not in classes:
            raise ValueError(
                f"{label}: pos_class {pos_class!r} is not in classes {classes} — the "
                "proba column would be selected by the wrong index"
            )

    aligned = _align_oof_pair(oof_a, oof_b, label)
    if aligned is None:
        return None
    idx_a, idx_b = aligned
    y_true = np.asarray(oof_a["y_true"])[idx_a]

    def _metric_fn(oof: dict, idx: np.ndarray) -> Callable[[np.ndarray], float | None]:
        # Slice each arm's blocks to the shared rows first, so both arms are scored
        # on the same variants while each keeps its own fold assignment — the two
        # arms are different CV partitions and their folds do not correspond.
        arms = [
            (np.asarray(proba)[idx], np.asarray(folds)[idx], np.unique(np.asarray(folds)[idx]))
            for proba, folds, _ in oof_score_arms(oof, label)
        ]

        if metric == "macro_f1":
            preds = [
                np.array([classes[col] for col in proba.argmax(axis=1)])
                for proba, _, _ in arms
            ]
            f1_arms = [
                (pred, folds, fold_ids)
                for pred, (_, folds, fold_ids) in zip(preds, arms)
            ]

            def _fold_f1(block: np.ndarray, arm_pred: np.ndarray) -> float | None:
                return fold_macro_f1(y_true, block, arm_pred, classes)

            def _macro_f1(rows: np.ndarray) -> float | None:
                return score_within_folds(rows, f1_arms, _fold_f1)
            return _macro_f1

        if metric == "auroc_one_vs_rest":
            column_arms = [
                (proba[:, classes.index(pos_class)], folds, fold_ids)
                for proba, folds, fold_ids in arms
            ]
            y_bin_all = (y_true == pos_class).astype(int)
        else:
            column_arms = arms
            y_bin_all = y_true

        def _fold_auroc(block: np.ndarray, column: np.ndarray) -> float | None:
            y_bin = y_bin_all[block]
            if len(np.unique(y_bin)) < 2:
                return None
            return float(roc_auc_score(y_bin, column[block]))

        def _auroc(rows: np.ndarray) -> float | None:
            return score_within_folds(rows, column_arms, _fold_auroc)
        return _auroc

    shared_genes = np.asarray(oof_a["genes"], dtype=object)[idx_a]
    fn_a, fn_b = _metric_fn(oof_a, idx_a), _metric_fn(oof_b, idx_b)
    if metric == "macro_f1":
        discard_reason = (
            "at least one arm's fold lost a mechanism class on the shared resample"
        )
    else:
        discard_reason = (
            "at least one arm's fold lost the one-vs-rest positive or negative "
            "class on the shared resample"
        )
    if cross_partition:
        out = paired_cluster_bootstrap_diff_cross_partition(
            family_or_gene_clusters(shared_genes, pfam_map, is_family_split=True),
            fn_a,
            fn_b,
            sensitivity_clusters=shared_genes,
            n_resamples=n_resamples,
            seed=seed,
            discard_reason=discard_reason,
        )
    else:
        out = paired_cluster_bootstrap_diff(
            family_or_gene_clusters(shared_genes, pfam_map, is_family_split),
            fn_a,
            fn_b,
            n_resamples=n_resamples,
            seed=seed,
            discard_reason=discard_reason,
        )
    out["n_shared"] = len(idx_a)
    return out


def adjudicate_diff(passed: bool | None, diff_ci: dict | None, threshold: float) -> str:
    """Pre-registration §1.1 verdict for a difference."""
    if passed is None:
        return "not adjudicated (no point estimate)"
    if diff_ci is None or diff_ci.get("ci_suppressed") or diff_ci.get("ci_low") is None:
        return f"{'pass' if passed else 'fail'}, no CI"
    ci_low, ci_high = diff_ci["ci_low"], diff_ci["ci_high"]
    if passed:
        if ci_low > 0:
            return "pass, established (CI excludes zero)"
        return "pass on point estimate, not distinguishable (CI spans zero)"
    if ci_high > threshold:
        return "fail, underpowered (CI spans the pre-registered threshold)"
    return "fail, established (CI excludes the pre-registered threshold)"


def adjudicate_equivalence(
    passed: bool | None, diff_ci: dict | None, margin: float
) -> str:
    """Verdict for an equivalence claim: is |difference| < margin?

    'passed' is True when the point estimate is within [-margin, +margin].
    The CI establishes equivalence only if the entire interval falls within
    that band; it refutes equivalence only if the entire interval falls
    outside it.
    """
    if passed is None:
        return "not adjudicated (no point estimate)"
    if diff_ci is None or diff_ci.get("ci_suppressed") or diff_ci.get("ci_low") is None:
        return f"{'pass' if passed else 'fail'}, no CI"
    ci_low, ci_high = diff_ci["ci_low"], diff_ci["ci_high"]
    ci_within = ci_low > -margin and ci_high < margin
    ci_outside = ci_low > margin or ci_high < -margin
    if passed:
        if ci_within:
            return "pass, established (CI within equivalence band)"
        return "pass on point estimate, not established (CI exceeds equivalence band)"
    if ci_outside:
        return "fail, established (CI outside equivalence band)"
    return "fail, underpowered (CI overlaps equivalence band)"


def adjudicate_level(value: float | None, ci: dict | None, threshold: float) -> str:
    """Pre-registration §1.1 verdict for a level claim."""
    if value is None or not np.isfinite(value):
        return "not adjudicated (no point estimate)"
    passed = value >= threshold
    if ci is None or ci.get("ci_suppressed") or ci.get("ci_low") is None:
        return f"{'pass' if passed else 'fail'}, no CI"
    if passed:
        if ci["ci_low"] > threshold:
            return "pass, established (CI excludes the threshold)"
        return "pass on point estimate, not distinguishable (CI covers the threshold)"
    if ci["ci_high"] >= threshold:
        return "fail, underpowered (CI covers the threshold)"
    return "fail, established (CI excludes the threshold)"


def binary_auroc_cluster_bootstrap_ci(
    oof: dict,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    seed: int = 0,
    clusters: np.ndarray | None = None,
) -> dict:
    """Cluster-bootstrap CI on a binary AUROC, scored within fold and averaged.

    The OOF must carry its fold assignment; see oof_score_arms for why ranking the
    pooled concatenation is not an option.
    """
    y_true = np.asarray(oof["y_true"])
    arms = oof_score_arms(oof, "binary_auroc_cluster_bootstrap_ci")

    def _fold_auroc(block: np.ndarray, arm_proba: np.ndarray) -> float | None:
        y_bin = y_true[block]
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            return None
        return float(roc_auc_score(y_bin, arm_proba[block]))

    def _auroc(rows: np.ndarray) -> float | None:
        return score_within_folds(rows, arms, _fold_auroc)

    resample_unit = oof["genes"] if clusters is None else clusters
    return cluster_bootstrap_ci(
        resample_unit,
        _auroc,
        n_resamples=n_resamples,
        ci_level=ci_level,
        seed=seed,
        discard_reason="a fold's resampled rows lost the positive or the negative class",
        metric_name="binary_auroc",
    )


def _fold_of_each_unit(units: np.ndarray, folds: np.ndarray, what: str) -> dict:
    """Map each permutation unit to its fold, raising if a unit straddles two folds.

    Under both split schemes a permutation unit sits entirely inside one test fold —
    a family-split fold holds whole families, a gene-split fold holds whole genes. If
    that stops being true the within-fold shuffle is not well defined, and silently
    picking one of the folds would put the confound back.
    """
    fold_of: dict = {}
    for unit in np.unique(units):
        unit_folds = np.unique(folds[units == unit])
        if len(unit_folds) != 1:
            raise ValueError(
                f"permutation unit {what} {unit!r} spans {len(unit_folds)} folds; a "
                "within-fold shuffle needs each unit inside a single fold"
            )
        fold_of[unit] = unit_folds[0]
    return fold_of


def _permute_labels(
    labels: np.ndarray,
    groups: np.ndarray | None,
    folds: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Permute labels within fold; with groups, shuffle whole groups.

    The shuffle is confined to a fold because the statistic is now scored within
    fold. Moving a label across folds changes each fold's class composition, which is
    exactly the confound the fold-aware scoring removes, so a whole-dataset shuffle
    would put it back into the null while the observed value no longer carries it.
    """
    labels = np.asarray(labels)
    folds = np.asarray(folds)
    out = np.empty_like(labels)
    if groups is None:
        for fold in np.unique(folds):
            mask = folds == fold
            out[mask] = rng.permutation(labels[mask])
        return out

    groups = np.asarray(groups)
    fold_of_group = _fold_of_each_unit(groups, folds, "group")
    group_label = {g: labels[groups == g][0] for g in fold_of_group}
    mapping: dict = {}
    by_fold: dict = {}
    for group, fold in fold_of_group.items():
        by_fold.setdefault(fold, []).append(group)
    for fold in sorted(by_fold, key=str):
        members = sorted(by_fold[fold], key=str)
        permuted = rng.permutation([group_label[g] for g in members])
        mapping.update(dict(zip(members, permuted)))
    return np.array([mapping[g] for g in groups])


def _cluster_partition(
    groups: np.ndarray,
    clusters: np.ndarray,
    folds: np.ndarray,
) -> tuple[dict, dict]:
    """Group genes by cluster and clusters by (fold, cluster size).

    Purely structural: depends only on which gene belongs to which cluster and fold,
    never on labels or an rng draw. A cluster's swap eligibility (whether it has a
    same-size partner in its own fold) is therefore fixed before any permutation is
    drawn, so it can be counted once instead of re-derived from a single draw.
    """
    groups = np.asarray(groups)
    clusters = np.asarray(clusters)
    folds = np.asarray(folds)

    gene_cluster = {}
    for gene in np.unique(groups):
        mask = groups == gene
        gene_cluster[gene] = clusters[mask][0]

    cluster_genes: dict = {}
    for gene, cluster in gene_cluster.items():
        cluster_genes.setdefault(cluster, []).append(gene)
    for cluster in cluster_genes:
        cluster_genes[cluster] = sorted(cluster_genes[cluster])

    fold_of_cluster = _fold_of_each_unit(clusters, folds, "cluster")

    by_fold_size: dict = {}
    for cluster, genes_in in cluster_genes.items():
        by_fold_size.setdefault((fold_of_cluster[cluster], len(genes_in)), []).append(cluster)

    return cluster_genes, by_fold_size


def count_immovable_clusters(
    groups: np.ndarray,
    clusters: np.ndarray,
    folds: np.ndarray,
) -> int:
    """Count clusters with no same-size partner in their own fold.

    These clusters keep their real labels in every draw of `_permute_labels_by_cluster`
    (there is nothing same-size to swap with), so this count is identical for every
    permutation and must be reported alongside the p-value per the preregistration.
    """
    _, by_fold_size = _cluster_partition(groups, clusters, folds)
    return sum(1 for cluster_ids in by_fold_size.values() if len(cluster_ids) == 1)


def _permute_labels_by_cluster(
    labels: np.ndarray,
    groups: np.ndarray,
    clusters: np.ndarray,
    folds: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Swap whole clusters' label blocks between same-size clusters in the same fold.

    Preserves within-cluster label mixing (unlike one-label-per-cluster
    broadcast, which would widen the null). Swaps are confined to a fold for the
    reason given in _permute_labels. Clusters with no same-size partner inside their
    own fold keep their labels; use `count_immovable_clusters` for that count.
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups)

    gene_label = {}
    for gene in np.unique(groups):
        gene_label[gene] = labels[groups == gene][0]

    cluster_genes, by_fold_size = _cluster_partition(groups, clusters, folds)

    new_gene_label = {}
    for key in sorted(by_fold_size, key=str):
        cluster_ids = sorted(by_fold_size[key], key=str)
        if len(cluster_ids) == 1:
            only = cluster_ids[0]
            for gene in cluster_genes[only]:
                new_gene_label[gene] = gene_label[gene]
            continue
        donors = [cluster_ids[i] for i in rng.permutation(len(cluster_ids))]
        for target, donor in zip(cluster_ids, donors):
            donor_labels = [gene_label[g] for g in cluster_genes[donor]]
            for gene, label in zip(cluster_genes[target], donor_labels):
                new_gene_label[gene] = label

    return np.array([new_gene_label[g] for g in groups])


def _permutation_null_value(
    run_metric_fn: Callable[[np.ndarray], float | None],
    labels: np.ndarray,
    groups: np.ndarray | None,
    folds: np.ndarray,
    child_seed: int,
    clusters: np.ndarray | None = None,
) -> float | None:
    """Shuffle labels within fold and recompute the metric for one permutation draw."""
    rng = np.random.RandomState(child_seed)
    if clusters is not None:
        permuted = _permute_labels_by_cluster(
            np.asarray(labels), np.asarray(groups), clusters, folds, rng
        )
    else:
        permuted = _permute_labels(np.asarray(labels), groups, folds, rng)
    value = run_metric_fn(permuted)
    if value is not None and np.isfinite(value):
        return float(value)
    return None


def macro_ovr_auroc(
    y_true: np.ndarray,
    proba: np.ndarray,
    folds: np.ndarray,
    classes: list[str] = MECHANISM_CLASSES,
) -> tuple[float | None, tuple[str, ...]]:
    """Macro one-vs-rest AUROC scored within fold, and the classes it averaged over.

    A class counts as scored only when every fold can score it, matching the rule the
    bootstrap uses. Scored classes come back alongside the value because a 3-class and
    a 2-class average are different statistics and must not be mixed in one null.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    arms = folds_to_arms(proba, folds)
    all_rows = np.arange(len(y_true))
    values, scored = [], []
    for col_idx, cls in enumerate(classes):
        def _fold_auroc(block: np.ndarray, arm_proba: np.ndarray, _cls=cls, _col=col_idx):
            y_bin = binary_class_target(y_true[block], _cls)
            if y_bin is None:
                return None
            return float(roc_auc_score(y_bin, arm_proba[block, _col]))

        value = score_within_folds(all_rows, arms, _fold_auroc)
        if value is None:
            continue
        values.append(value)
        scored.append(cls)
    if not values:
        return None, ()
    return float(np.mean(values)), tuple(scored)


def oof_permutation_pvalue(
    y_true: np.ndarray,
    proba: np.ndarray,
    folds: np.ndarray,
    groups: np.ndarray | None = None,
    clusters: np.ndarray | None = None,
    classes: list[str] = MECHANISM_CLASSES,
    n_permutations: int = PERMUTATION_N_RESAMPLES,
    seed: int = 0,
) -> dict:
    """Permutation p-value for macro OVR AUROC against fixed OOF predictions (no refit).

    Uses AUROC rather than macro-F1 because a chance-floor probe predicts
    majority class everywhere, pinning F1 near the floor regardless of ranking
    signal.

    Both the observed statistic and the null are scored within fold, and the shuffle
    is confined to a fold. Scoring within fold against a whole-dataset shuffle would
    hold the fold structure fixed on one side of the comparison only.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    folds = np.asarray(folds)
    observed, observed_classes = macro_ovr_auroc(y_true, proba, folds, classes)

    if clusters is not None and groups is None:
        raise ValueError("clusters requires groups (the gene-level label unit)")
    immovable = (
        count_immovable_clusters(groups, clusters, folds) if clusters is not None else None
    )

    rng = np.random.RandomState(seed)
    null = []
    dropped_class_mismatch = 0
    for _ in range(n_permutations):
        if clusters is not None:
            permuted = _permute_labels_by_cluster(
                y_true, groups, clusters, folds, rng
            )
        else:
            permuted = _permute_labels(y_true, groups, folds, rng)
        value, scored = macro_ovr_auroc(permuted, proba, folds, classes)
        if value is None or not np.isfinite(value):
            continue
        # A draw scoring a different class set is a different statistic.
        if scored != observed_classes:
            dropped_class_mismatch += 1
            continue
        null.append(value)

    null_arr = np.array(null)
    common = {
        "statistic": "macro_ovr_auroc",
        "null_type": "oof_fixed_predictions",
        "permutation_unit": "cluster_block" if clusters is not None else "gene",
        "shuffle_scope": "within_fold",
        "classes_scored": list(observed_classes),
        "n_permutations": int(len(null_arr)),
        "n_dropped_class_mismatch": dropped_class_mismatch,
        "n_clusters_immovable": immovable,
    }
    if observed is None or not np.isfinite(observed) or len(null_arr) == 0:
        return {
            **common,
            "observed": float(observed) if observed is not None and np.isfinite(observed) else None,
            "p_value": None,
            "p_value_resolution": (
                float(1 / (1 + len(null_arr))) if len(null_arr) else None
            ),
            "resolution_limited": None,
            "null_mean": float(np.mean(null_arr)) if len(null_arr) else None,
            "null_std": float(np.std(null_arr)) if len(null_arr) else None,
        }
    extreme = int(np.sum(null_arr >= observed))
    return {
        **common,
        "observed": float(observed),
        "p_value": float((1 + extreme) / (1 + len(null_arr))),
        "p_value_resolution": float(1 / (1 + len(null_arr))),
        "resolution_limited": extreme == 0,
        "null_mean": float(np.mean(null_arr)),
        "null_std": float(np.std(null_arr)),
    }


def label_permutation_pvalue(
    run_metric_fn: Callable[[np.ndarray], float | None],
    labels: np.ndarray,
    statistic: str,
    folds: np.ndarray,
    groups: np.ndarray | None = None,
    clusters: np.ndarray | None = None,
    n_permutations: int = PERMUTATION_N_RESAMPLES,
    seed: int = 0,
    alternative: str = "greater",
    n_jobs: int = -1,
) -> dict:
    """One-sided permutation p-value with full refit per permutation.

    `folds` is the fold each row was scored in. The shuffle stays inside a fold, so
    the null holds the fold structure fixed and varies only the label assignment.
    """
    if clusters is not None and groups is None:
        raise ValueError("clusters requires groups (the gene-level label unit)")
    immovable = (
        count_immovable_clusters(groups, clusters, folds) if clusters is not None else None
    )

    observed = run_metric_fn(labels)

    child_seqs = np.random.SeedSequence(seed).spawn(n_permutations)
    child_seeds = [int(s.generate_state(1)[0]) for s in child_seqs]

    null_values = Parallel(n_jobs=n_jobs)(
        delayed(_permutation_null_value)(
            run_metric_fn, labels, groups, folds, child_seed, clusters
        )
        for child_seed in child_seeds
    )
    null = [value for value in null_values if value is not None]

    null_arr = np.array(null)
    common = {
        "statistic": statistic,
        "null_type": "refit_per_permutation",
        "permutation_unit": "cluster_block" if clusters is not None else "gene",
        "shuffle_scope": "within_fold",
        "n_clusters_immovable": immovable,
    }
    if observed is None or not np.isfinite(observed) or len(null_arr) == 0:
        return {
            **common,
            "observed": float(observed) if observed is not None and np.isfinite(observed) else None,
            "p_value": None,
            "p_value_resolution": (
                float(1 / (1 + len(null_arr))) if len(null_arr) else None
            ),
            "resolution_limited": None,
            "null_mean": float(np.mean(null_arr)) if len(null_arr) else None,
            "null_std": float(np.std(null_arr)) if len(null_arr) else None,
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
        **common,
        "observed": float(observed),
        "p_value": float(p_value),
        "p_value_resolution": float(1 / (1 + len(null_arr))),
        "resolution_limited": extreme == 0,
        "null_mean": float(np.mean(null_arr)),
        "null_std": float(np.std(null_arr)),
        "n_permutations": int(len(null_arr)),
    }
