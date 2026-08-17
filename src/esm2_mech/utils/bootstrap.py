"""Cluster bootstrap CIs and label-permutation tests for gene/family-clustered data."""

from __future__ import annotations

import functools
from typing import Callable

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from esm2_mech.utils.metrics import binary_class_target
from esm2_mech.utils.constants import (
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_MIN_VALID_FRAC,
    BOOTSTRAP_N_RESAMPLES,
    MECHANISM_CLASSES,
    PERMUTATION_N_RESAMPLES,
)

print = functools.partial(print, flush=True)

# Cannot collide with a real Pfam accession ("PF" + digits).
UNANNOTATED_CLUSTER_PREFIX = "__no_pfam__:"


def average_oof_over_seeds(oof_list: list[dict | None]) -> dict | None:
    """Average per-seed OOF probas to one prediction per variant."""
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


def _evaluate_metric_fns(
    metric_fns: list[Callable[[np.ndarray], dict]], rows: np.ndarray
) -> dict:
    """Run every metric fn on one row-index array and flatten to {name: value}."""
    out: dict = {}
    for metric_fn in metric_fns:
        for name, value in metric_fn(rows).items():
            out[name] = _clean_scalar(value)
    return out


def _multi_bootstrap_resample_values(
    metric_fns: list[Callable[[np.ndarray], dict]],
    cluster_rows: list[np.ndarray],
    n_clusters: int,
    child_seed: int,
) -> dict:
    """Draw one cluster resample and score every metric on the same drawn rows."""
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


def cluster_bootstrap_ci_multi(
    clusters: np.ndarray,
    metric_fns: list[Callable[[np.ndarray], dict]],
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    min_valid_frac: float = BOOTSTRAP_MIN_VALID_FRAC,
    seed: int = 0,
    n_jobs: int = -1,
) -> dict:
    """Cluster-bootstrap CIs for several metrics over one shared set of resamples."""
    unique, cluster_rows = _cluster_to_rows(np.asarray(clusters))
    n_clusters = len(unique)
    all_rows = np.arange(len(clusters))
    points = _evaluate_metric_fns(metric_fns, all_rows)

    child_seqs = np.random.SeedSequence(seed).spawn(n_resamples)
    child_seeds = [int(s.generate_state(1)[0]) for s in child_seqs]

    replicates = Parallel(n_jobs=n_jobs)(
        delayed(_multi_bootstrap_resample_values)(
            metric_fns, cluster_rows, n_clusters, child_seed
        )
        for child_seed in child_seeds
    )

    return {
        name: _summarize_bootstrap(
            points[name],
            [rep[name] for rep in replicates if rep.get(name) is not None],
            n_resamples,
            n_clusters,
            ci_level,
            min_valid_frac,
        )
        for name in points
    }


def cluster_bootstrap_ci(
    clusters: np.ndarray,
    metric_fn: Callable[[np.ndarray], float | None],
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    min_valid_frac: float = BOOTSTRAP_MIN_VALID_FRAC,
    seed: int = 0,
    n_jobs: int = -1,
) -> dict:
    """Single-metric front end to cluster_bootstrap_ci_multi."""
    key = "metric"
    out = cluster_bootstrap_ci_multi(
        clusters,
        [lambda rows: {key: metric_fn(rows)}],
        n_resamples=n_resamples,
        ci_level=ci_level,
        min_valid_frac=min_valid_frac,
        seed=seed,
        n_jobs=n_jobs,
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
    """Cluster-bootstrap CIs for macro-F1, per-class AUROC, AUPRC, prevalence and lift."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = np.array([classes[col] for col in proba.argmax(axis=1)])

    def _macro_f1(rows: np.ndarray) -> dict:
        return {
            "macro_f1": float(
                f1_score(y_true[rows], pred[rows], average="macro", zero_division=0)
            )
        }

    def _class_metrics(rows: np.ndarray, *, _col: int, _cls: str) -> dict:
        names = (f"auroc_{_cls}", f"auprc_{_cls}", f"prevalence_{_cls}", f"auprc_lift_{_cls}")
        y_bin = binary_class_target(y_true[rows], _cls)
        if y_bin is None:
            return dict.fromkeys(names)
        scores = proba[rows, _col]
        auprc = float(average_precision_score(y_bin, scores))
        prevalence = float(y_bin.mean())
        return {
            names[0]: float(roc_auc_score(y_bin, scores)),
            names[1]: auprc,
            names[2]: prevalence,
            names[3]: auprc - prevalence,
        }

    metric_fns = [_macro_f1] + [
        functools.partial(_class_metrics, _col=col_idx, _cls=cls)
        for col_idx, cls in enumerate(classes)
    ]
    out = cluster_bootstrap_ci_multi(
        clusters, metric_fns, n_resamples=n_resamples, ci_level=ci_level, seed=seed
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
    """R7.1 verdict: point estimate decides pass/fail, CI decides established vs underpowered."""
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
    """R7.1 verdict for a level claim: is the value above the threshold."""
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
    """Cluster-bootstrap CI on a binary AUROC from an OOF dict."""
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
    """Permute labels; with groups, shuffle at the group level to preserve within-group structure."""
    if groups is None:
        return rng.permutation(labels)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    group_label = {g: labels[groups == g][0] for g in unique}
    permuted = rng.permutation([group_label[g] for g in unique])
    mapping = dict(zip(unique, permuted))
    return np.array([mapping[g] for g in groups])


def _permute_labels_by_cluster(
    labels: np.ndarray,
    groups: np.ndarray,
    clusters: np.ndarray,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, int]:
    """Swap whole clusters' label blocks between same-size clusters.

    Preserves within-cluster label mixing (unlike one-label-per-cluster
    broadcast, which would widen the null). Clusters with unique gene counts
    have no swap partner and keep their labels; that count is returned.
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    clusters = np.asarray(clusters)

    gene_label = {}
    gene_cluster = {}
    for gene in np.unique(groups):
        mask = groups == gene
        gene_label[gene] = labels[mask][0]
        gene_cluster[gene] = clusters[mask][0]

    cluster_genes: dict = {}
    for gene, cluster in gene_cluster.items():
        cluster_genes.setdefault(cluster, []).append(gene)
    for cluster in cluster_genes:
        cluster_genes[cluster] = sorted(cluster_genes[cluster])

    by_size: dict = {}
    for cluster, genes_in in cluster_genes.items():
        by_size.setdefault(len(genes_in), []).append(cluster)

    new_gene_label = {}
    immovable = 0
    for size, cluster_ids in by_size.items():
        cluster_ids = sorted(cluster_ids)
        if len(cluster_ids) == 1:
            immovable += 1
            only = cluster_ids[0]
            for gene in cluster_genes[only]:
                new_gene_label[gene] = gene_label[gene]
            continue
        donors = [cluster_ids[i] for i in rng.permutation(len(cluster_ids))]
        for target, donor in zip(cluster_ids, donors):
            donor_labels = [gene_label[g] for g in cluster_genes[donor]]
            for gene, label in zip(cluster_genes[target], donor_labels):
                new_gene_label[gene] = label

    return np.array([new_gene_label[g] for g in groups]), immovable


def _permutation_null_value(
    run_metric_fn: Callable[[np.ndarray], float | None],
    labels: np.ndarray,
    groups: np.ndarray | None,
    child_seed: int,
    clusters: np.ndarray | None = None,
) -> float | None:
    """Shuffle labels and recompute the metric for one permutation draw."""
    rng = np.random.RandomState(child_seed)
    if clusters is not None:
        permuted, _ = _permute_labels_by_cluster(
            np.asarray(labels), np.asarray(groups), clusters, rng
        )
    else:
        permuted = _permute_labels(np.asarray(labels), groups, rng)
    value = run_metric_fn(permuted)
    if value is not None and np.isfinite(value):
        return float(value)
    return None


def macro_ovr_auroc(
    y_true: np.ndarray, proba: np.ndarray, classes: list[str] = MECHANISM_CLASSES
) -> tuple[float | None, tuple[str, ...]]:
    """Macro one-vs-rest AUROC and the classes it averaged over.

    Returns scored classes alongside the value because a 3-class and 2-class
    average are different statistics and must not be mixed in a null.
    """
    y_true = np.asarray(y_true)
    values, scored = [], []
    for col_idx, cls in enumerate(classes):
        y_bin = binary_class_target(y_true, cls)
        if y_bin is None:
            continue
        values.append(float(roc_auc_score(y_bin, proba[:, col_idx])))
        scored.append(cls)
    if not values:
        return None, ()
    return float(np.mean(values)), tuple(scored)


def oof_permutation_pvalue(
    y_true: np.ndarray,
    proba: np.ndarray,
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
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    observed, observed_classes = macro_ovr_auroc(y_true, proba, classes)

    rng = np.random.RandomState(seed)
    null = []
    dropped_class_mismatch = 0
    immovable = None
    for _ in range(n_permutations):
        if clusters is not None:
            if groups is None:
                raise ValueError("clusters requires groups (the gene-level label unit)")
            permuted, immovable = _permute_labels_by_cluster(y_true, groups, clusters, rng)
        else:
            permuted = _permute_labels(y_true, groups, rng)
        value, scored = macro_ovr_auroc(permuted, proba, classes)
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
            "null_mean": float(np.mean(null_arr)) if len(null_arr) else None,
            "null_std": float(np.std(null_arr)) if len(null_arr) else None,
        }
    extreme = int(np.sum(null_arr >= observed))
    return {
        **common,
        "observed": float(observed),
        "p_value": float((1 + extreme) / (1 + len(null_arr))),
        "null_mean": float(np.mean(null_arr)),
        "null_std": float(np.std(null_arr)),
    }


def label_permutation_pvalue(
    run_metric_fn: Callable[[np.ndarray], float | None],
    labels: np.ndarray,
    statistic: str,
    groups: np.ndarray | None = None,
    clusters: np.ndarray | None = None,
    n_permutations: int = PERMUTATION_N_RESAMPLES,
    seed: int = 0,
    alternative: str = "greater",
    n_jobs: int = -1,
) -> dict:
    """One-sided permutation p-value with full refit per permutation."""
    if clusters is not None and groups is None:
        raise ValueError("clusters requires groups (the gene-level label unit)")

    observed = run_metric_fn(labels)

    child_seqs = np.random.SeedSequence(seed).spawn(n_permutations)
    child_seeds = [int(s.generate_state(1)[0]) for s in child_seqs]

    null_values = Parallel(n_jobs=n_jobs)(
        delayed(_permutation_null_value)(run_metric_fn, labels, groups, child_seed, clusters)
        for child_seed in child_seeds
    )
    null = [value for value in null_values if value is not None]

    null_arr = np.array(null)
    if observed is None or not np.isfinite(observed) or len(null_arr) == 0:
        return {
            "statistic": statistic,
            "null_type": "refit_per_permutation",
            "permutation_unit": "cluster_block" if clusters is not None else "gene",
            "observed": float(observed) if observed is not None and np.isfinite(observed) else None,
            "p_value": None,
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
        "statistic": statistic,
        "null_type": "refit_per_permutation",
        "permutation_unit": "cluster_block" if clusters is not None else "gene",
        "observed": float(observed),
        "p_value": float(p_value),
        "null_mean": float(np.mean(null_arr)),
        "null_std": float(np.std(null_arr)),
        "n_permutations": int(len(null_arr)),
    }
