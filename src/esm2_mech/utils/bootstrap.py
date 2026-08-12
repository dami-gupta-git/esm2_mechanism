"""Dependency-aware inference: cluster bootstrap and label-permutation tests.

Mechanism labels are gene-level and variants cluster within genes (genes within
families), so the effective sample size is the cluster count, not the variant count.

  - cluster_bootstrap_ci: CI that resamples whole clusters with replacement.
  - paired_cluster_bootstrap_diff / _cross_partition: CI on the DIFFERENCE between
    two metrics on a shared resample.
  - label_permutation_pvalue: p-value from cluster-aware label shuffling.

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
    BOOTSTRAP_MIN_VALID_FRAC,
    BOOTSTRAP_N_RESAMPLES,
    MECHANISM_CLASSES,
    PERMUTATION_N_RESAMPLES,
)

print = functools.partial(print, flush=True)


def average_oof_over_seeds(oof_list: list[dict | None]) -> dict | None:
    """Collapse per-seed out-of-fold predictions to one proba-per-variant.

    Each entry is a probe's OOF dict {"y_true", "proba", "genes", "row_ids"} from one
    seed. Averaging proba across seeds gives one de-duplicated prediction per variant,
    so a downstream gene-cluster bootstrap counts each variant once, not n_seeds times.

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


def _bootstrap_resample_value(
    metric_fn: Callable[[np.ndarray], float | None],
    cluster_rows: list[np.ndarray],
    n_clusters: int,
    child_seed: int,
) -> float | None:
    """Draw one cluster resample with an independent seeded RNG, return the metric.

    Each call gets its own `RandomState(child_seed)` for reproducible,
    order-independent resamples under parallel execution.
    """
    rng = np.random.RandomState(child_seed)
    drawn = rng.randint(0, n_clusters, size=n_clusters)
    rows = np.concatenate([cluster_rows[i] for i in drawn])
    value = metric_fn(rows)
    if value is not None and np.isfinite(value):
        return float(value)
    return None


def cluster_bootstrap_ci(
    clusters: np.ndarray,
    metric_fn: Callable[[np.ndarray], float | None],
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    min_valid_frac: float = BOOTSTRAP_MIN_VALID_FRAC,
    seed: int = 0,
    n_jobs: int = -1,
) -> dict:
    """Percentile CI for a metric via a cluster bootstrap.

    `clusters` is a per-row array of cluster ids (e.g. gene or Pfam family). Each
    resample draws len(unique clusters) clusters with replacement, gathers all their
    rows, and calls `metric_fn(row_indices)`, which returns the metric on those rows
    or None/NaN when undefined (e.g. a class absent). Undefined resamples are dropped,
    not imputed.

    When the surviving fraction falls below `min_valid_frac`, no CI is returned
    (ci_low/ci_high = None) and `ci_suppressed` is set, so the dropout is visible
    rather than silently narrowing the interval.

    The point estimate is metric_fn over all rows. Returns point, ci_low, ci_high,
    n_resamples (the count that contributed), n_resamples_total, valid_frac,
    ci_suppressed, n_clusters.

    Resamples run in parallel across cores via joblib (n_jobs=-1 = all cores); each
    draws from its own SeedSequence-spawned RNG, so the CI matches a serial run.
    """
    unique, cluster_rows = _cluster_to_rows(np.asarray(clusters))
    n_clusters = len(unique)
    all_rows = np.arange(len(clusters))
    point = metric_fn(all_rows)

    child_seqs = np.random.SeedSequence(seed).spawn(n_resamples)
    child_seeds = [int(s.generate_state(1)[0]) for s in child_seqs]

    values = Parallel(n_jobs=n_jobs)(
        delayed(_bootstrap_resample_value)(
            metric_fn, cluster_rows, n_clusters, child_seed
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


def _subsample_resample_value(
    metric_fn: Callable[[np.ndarray], float | None],
    cluster_rows: list[np.ndarray],
    n_clusters: int,
    subsample_size: int,
    child_seed: int,
) -> float | None:
    """Draw one cluster SUBSAMPLE (without replacement) with an independent
    seeded RNG, return the metric. Mirrors `_bootstrap_resample_value` but
    draws `subsample_size` distinct clusters instead of `n_clusters` with
    replacement, so no cluster's rows are ever duplicated within a replicate.
    """
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
    """Percentile CI via an m-out-of-n cluster SUBSAMPLE (without replacement).

    Use this instead of `cluster_bootstrap_ci` for statistics that break under
    literal duplicate points — nearest-neighbor purity, pairwise-distance
    ratios, or any other distance/graph statistic where "this observation
    counts twice" has no coherent meaning. A standard with-replacement cluster
    bootstrap puts the SAME rows into a replicate more than once whenever a
    cluster is drawn more than once; for a k-NN or pairwise-distance metric
    those duplicate points sit at distance exactly 0 from each other, which
    inflates same-cluster neighbor purity and deflates within-cluster mean
    distance. Subsampling `subsample_frac` of clusters WITHOUT replacement
    never duplicates a point, so that artifact cannot occur. (Metrics that are
    additive over rows — F1, AUROC, macro_f1, Spearman rho — do not have this
    problem and should keep using `cluster_bootstrap_ci`, where draw
    multiplicity is a meaningful, correct resampling weight.)

    `subsample_frac` defaults to 0.632 (~ 1 - 1/e), the expected fraction of
    distinct clusters included in a same-size with-replacement bootstrap —
    the standard choice for an m-out-of-n subsample meant to approximate a
    bootstrap's resampling variability without its duplication.

    Still resamples whole clusters, never splits one, same as
    `cluster_bootstrap_ci`. Returns the same keys (point, ci_low, ci_high,
    n_resamples, n_resamples_total, valid_frac, ci_suppressed, n_clusters)
    plus `subsample_size` (the number of clusters actually drawn per
    replicate, since it differs from `n_clusters`).
    """
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
    """Draw one cluster resample and score BOTH arms on the identical drawn rows.

    The single `rng.randint` draw is the pairing: both metric_fn_a and metric_fn_b
    see the same resampled row-index array, so the difference is a paired statistic,
    not two independently-resampled ones.
    """
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
) -> dict:
    """Shared resampling machinery for both pairing modes.

    `resample_clusters` is the resampling unit (row-aligned cluster ids); the two
    public functions differ only in what that unit means (a shared fold assignment
    for same-fold pairing, or the coarser of two CV partitions for cross-partition
    pairing) and what `metric_fn_a`/`metric_fn_b` do with the row-index array each
    replicate.
    """
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
    # A replicate contributes only if BOTH arms were defined on it.
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
        "valid_frac": float(valid_frac),
        "n_clusters": int(n_clusters),
    }
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
) -> dict:
    """Percentile CI on the difference between two metrics under one shared fold.

    Same-fold pairing mode: both arms are already scored under one fold assignment
    (e.g. ESM-3 seq-track vs ESM-2 delta_mean on the same family-split folds).
    `clusters` is that fold assignment's cluster id per row; `metric_fn_a` and
    `metric_fn_b` are closures over each arm's fixed predictions, called with the
    SAME resampled row-index array on every replicate, making the difference a
    paired statistic. `clusters` must already be restricted to rows present in both
    arms — this function does not intersect the arms itself.

    For the gene-split-minus-family-split gap (two different CV partitions), use
    `paired_cluster_bootstrap_diff_cross_partition` instead.

    Returns point_a, point_b, point_diff (= point_a - point_b, over all rows),
    ci_low, ci_high (percentile CI on the diff), n_resamples (contributing
    replicates), n_resamples_total, valid_frac, ci_suppressed, n_clusters. A
    replicate is dropped when either arm is undefined on it; below `min_valid_frac`
    surviving, no CI is returned and `ci_suppressed` is set.
    """
    return _paired_cluster_bootstrap_diff_ci(
        clusters,
        metric_fn_a,
        metric_fn_b,
        n_resamples=n_resamples,
        ci_level=ci_level,
        min_valid_frac=min_valid_frac,
        seed=seed,
        n_jobs=n_jobs,
    )


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
) -> dict:
    """Percentile CI on a difference between two arms scored under DIFFERENT partitions.

    Cross-partition pairing mode: for the gene-split-minus-family-split gap, arm A
    and arm B are not evaluated under a shared fold assignment — gene-split and
    family-split are different CV partitions of the same rows. `metric_fn_a` and
    `metric_fn_b` must each be self-contained closures that recompute their OWN
    arm's metric under their OWN partition given a row-index array.

    The pairing is still preserved: `resample_clusters` is resampled ONCE per
    replicate and the identical drawn row-index array is handed to both metric fns.

    `resample_clusters` must be the COARSER of the two arms' units — for the split
    gap this is the family, never the gene: the family-split arm's variance is only
    correct under family resampling, and a family resample induces a valid gene
    resample but not the reverse. Passing gene clusters here silently understates
    the family-split arm's true variance.

    `resample_clusters` (and `sensitivity_clusters`, if given) must already be
    restricted to rows present in both arms — this function does not intersect the
    arms itself.

    If `sensitivity_clusters` (the finer unit, e.g. gene) is given, a second CI is
    computed at that finer granularity and returned under `gene_resampled_sensitivity`
    — a labelled sensitivity check, never the primary result (R7.3).

    Return shape matches `paired_cluster_bootstrap_diff`, plus
    `gene_resampled_sensitivity` when requested.
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
        )
    return primary


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

    for metric_name, ci in out.items():
        if ci.get("ci_suppressed"):
            print(
                f"  [bootstrap] {metric_name}: CI suppressed — only "
                f"{ci['n_resamples']}/{ci['n_resamples_total']} resamples valid "
                f"({ci['valid_frac']:.0%}); the metric was undefined on the rest "
                f"(rare class absent on a resample). No CI reported for this metric."
            )
    return out


def family_or_gene_clusters(
    genes: np.ndarray, pfam_map: dict, is_family_split: bool
) -> np.ndarray:
    """The R7.3 resampling-unit choice: family-split CIs resample families, not
    genes — the family is the unit family-split CV actually holds out, so genes
    within one are not independent draws. Gene-split CIs resample genes
    unchanged.

    `genes` is a row-aligned gene-id array (typically an OOF dict's "genes"
    entry). Only annotated genes ever reach a family-split fold
    (`family_split_cv` excludes unannotated genes), so `pfam_map[g]` cannot
    miss when `is_family_split` is True.
    """
    if not is_family_split:
        return genes
    return np.array([pfam_map[g] for g in genes])


def _align_oof_pair(oof_a: dict, oof_b: dict, label: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Positions into each arm for the variants both arms scored, in one order.

    Pairing is defined only on rows both arms scored: a fold skipped in one arm
    (missing class) but not the other leaves rows with no counterpart. Both OOF
    dicts must carry "row_ids" — positional alignment is not assumed, because two
    arms can drop different folds and still produce equal-length arrays.
    """
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
    """Paired cluster-bootstrap CI on metric(A) − metric(B) for two OOF arms.

    Same-fold pairing (the default): both arms were scored over the same variants
    under the same split, so one cluster resample per replicate is handed to both
    arms and the difference is paired rather than a comparison of two independent
    intervals. This is the instrument behind the C4/C5 and contrastive claims.

    `cross_partition=True` is the C2 case — the gene-split-minus-family-split gap,
    where the arms were scored under two DIFFERENT CV partitions of the same rows.
    The resampling unit is then the family (R7.3: the coarser of the two arms; a
    family resample induces a valid gene resample, not the reverse), and the
    gene-resampled interval is returned alongside under
    `gene_resampled_sensitivity` as a labelled sensitivity check, never the primary
    result. `is_family_split` is ignored in this mode.

    `metric` selects what the arms are compared on:

      - "macro_f1"    — `proba` is an (n, n_classes) matrix whose columns are
                        aligned to `classes` (required); predictions are its argmax.
      - "auroc_binary" — `proba` is the 1-D positive-class probability and `y_true`
                        is 0/1, the shape `binary_auroc_cluster_bootstrap_ci` takes.
      - "auroc_one_vs_rest" — one class's column out of the (n, n_classes) matrix
                        against all others; `classes` and `pos_class` are both
                        required. This is how a per-class null ("DN is unmoved") is
                        tested as a difference rather than asserted from two point
                        estimates.

    `classes` is a parameter rather than a closed-over constant because callers
    pass 2-, 3- and 4-class label lists; a hardcoded list would index the wrong
    label or raise. Returns None when either arm is missing or the arms share no
    rows; otherwise the `paired_cluster_bootstrap_diff` shape plus n_shared.
    """
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
        proba = np.asarray(oof["proba"])[idx]
        if metric == "macro_f1":
            pred = np.array([classes[col] for col in proba.argmax(axis=1)])

            def _macro_f1(rows: np.ndarray) -> float:
                return float(
                    f1_score(y_true[rows], pred[rows], average="macro", zero_division=0)
                )
            return _macro_f1

        if metric == "auroc_one_vs_rest":
            column = proba[:, classes.index(pos_class)]
            y_bin_all = (y_true == pos_class).astype(int)
        else:
            column = proba
            y_bin_all = y_true

        def _auroc(rows: np.ndarray) -> float | None:
            y_bin = y_bin_all[rows]
            # A resample with no positives (or no negatives) leaves AUROC undefined;
            # the replicate is dropped, never scored as 0.5.
            if len(np.unique(y_bin)) < 2:
                return None
            return float(roc_auc_score(y_bin, column[rows]))
        return _auroc

    shared_genes = np.asarray(oof_a["genes"], dtype=object)[idx_a]
    fn_a, fn_b = _metric_fn(oof_a, idx_a), _metric_fn(oof_b, idx_b)
    if cross_partition:
        out = paired_cluster_bootstrap_diff_cross_partition(
            family_or_gene_clusters(shared_genes, pfam_map, is_family_split=True),
            fn_a,
            fn_b,
            sensitivity_clusters=shared_genes,
            n_resamples=n_resamples,
            seed=seed,
        )
    else:
        out = paired_cluster_bootstrap_diff(
            family_or_gene_clusters(shared_genes, pfam_map, is_family_split),
            fn_a,
            fn_b,
            n_resamples=n_resamples,
            seed=seed,
        )
    out["n_shared"] = len(idx_a)
    return out


def adjudicate_diff(passed: bool | None, diff_ci: dict | None, threshold: float) -> str:
    """R7.1 verdict for a gate, reading its point estimate against its paired CI.

    The point estimate decides pass/fail; the CI decides whether that reading is
    established or underpowered. A pass whose CI spans zero is reported as consistent
    in direction but not distinguishable, never as an established effect.
    """
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


def adjudicate_level(value: float | None, ci: dict | None, threshold: float) -> str:
    """R7.1 verdict for a claim about a LEVEL rather than a difference.

    K1 ("conservation alone clears 0.85") and C3 ("pathogenicity clears 0.85") are
    claims that one score sits above a threshold, so the interval is read against
    that threshold rather than against zero. Clearing on the point estimate while
    the interval still covers the threshold is not an established result.
    """
    if value is None or not np.isfinite(value):
        return "not adjudicated (no point estimate)"
    passed = value >= threshold
    if ci is None or ci.get("ci_suppressed") or ci.get("ci_low") is None:
        return f"{'pass' if passed else 'fail'}, no CI"
    if passed:
        if ci["ci_low"] > threshold:
            return "pass, established (CI excludes the threshold)"
        return "pass on point estimate, not distinguishable (CI covers the threshold)"
    if ci["ci_high"] > threshold:
        return "fail, underpowered (CI covers the threshold)"
    return "fail, established (CI excludes the threshold)"


def binary_auroc_cluster_bootstrap_ci(
    oof: dict,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    seed: int = 0,
    clusters: np.ndarray | None = None,
) -> dict:
    """Cluster-bootstrap CI on a binary AUROC from an OOF dict.

    `oof` is {"y_true" (0/1), "proba" (positive-class probability, 1-D), "genes"}
    — the shape returned by run_logreg_binary_cv/run_mlp_binary_cv/
    _run_sklearn_probe_impl with return_oof=True on a binary target (e.g. the
    pathogenicity control's pathogenic-vs-benign probe). Distinct from
    bootstrap_mechanism_metrics, which is specific to the 3-class GOF/DN/LOF
    mechanism labels.

    Resamples `oof["genes"]` by default. Pass `clusters` explicitly to resample
    a different unit instead (R7.3: a family-split arm must resample families,
    not genes — the family is the unit family-split CV actually holds out).
    """
    y_true = oof["y_true"]
    proba = oof["proba"]

    def _auroc(rows: np.ndarray) -> float | None:
        y_bin = y_true[rows]
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            return None
        return float(roc_auc_score(y_bin, proba[rows]))

    resample_unit = oof["genes"] if clusters is None else clusters
    return cluster_bootstrap_ci(
        resample_unit, _auroc, n_resamples=n_resamples, ci_level=ci_level, seed=seed
    )


def _permute_labels(
    labels: np.ndarray, groups: np.ndarray | None, rng: np.random.RandomState
) -> np.ndarray:
    """Permute labels. With `groups`, shuffle one label per group then broadcast back.

    Mechanism labels are constant within a gene, so a variant-level shuffle would break
    that structure and build an unrealistically easy null; groups=genes shuffles at the
    gene level instead, so each gene keeps a single (permuted) label across its variants.
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
    """Shuffle labels with an independent seeded RNG, then recompute the metric."""
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
    vector — i.e. it refits the probe — since this can't be computed from fixed
    out-of-fold predictions. Pass groups=genes for a gene-level shuffle (the correct
    null when labels are gene-level). The p-value is (1 + #{null >= observed}) / (1 + n)
    for alternative="greater".

    Permutations run in parallel across cores via joblib (n_jobs=-1 = all cores); each
    draws from its own SeedSequence-spawned RNG, so the null matches a serial run.
    """
    observed = run_metric_fn(labels)

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
