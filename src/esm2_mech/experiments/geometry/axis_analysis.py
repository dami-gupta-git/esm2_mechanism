"""Family-held-out descriptive analyses of a supervised pathogenicity axis."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from esm2_mech.utils.constants import N_FOLDS, N_SEEDS
from esm2_mech.utils.seed_aggregation import (
    aggregate_seed_values,
    make_seed_record,
    read_seed_point_estimate,
)
from esm2_mech.utils.splits import family_split_cv


def _seed_summary(seeds, values_by_seed: dict[int, list[float]]) -> dict:
    records = []
    for seed in seeds:
        fold_values = values_by_seed[seed]
        within_seed_mean = (
            float(np.mean(fold_values))
            if len(fold_values) == N_FOLDS and np.isfinite(fold_values).all()
            else None
        )
        records.append(make_seed_record(seed, within_seed_mean))
    return aggregate_seed_values(seeds, records).to_dict()


def format_axis_summary(summary: dict) -> str:
    """Format a summary without turning a missing estimate into a number."""
    metric = read_seed_point_estimate(summary)
    if not metric.available:
        return f"unavailable ({metric.message})"
    spread = metric.spread
    spread_text = "spread unavailable" if spread is None else f"seed SD {spread:.3f}"
    return f"{metric.value:+.3f} ({spread_text})"


def family_held_out_axis_analysis(
    delta: np.ndarray,
    labels: np.ndarray,
    genes: np.ndarray,
    pfam_map: dict,
    association_features: dict[str, np.ndarray],
    regression_features: np.ndarray | None = None,
    seeds=range(N_SEEDS),
) -> dict:
    """Estimate axis associations without using a held-out family's labels.

    Each fold learns the pathogenicity axis and every scaler on its training
    families. Correlations and optional Ridge R² are scored inside the held-out
    fold, then averaged over folds and seeds. Scores from independently fitted
    folds are never pooled onto one shared scale.
    """
    delta = np.asarray(delta)
    labels = np.asarray(labels)
    genes = np.asarray(genes)
    feature_arrays = {
        name: np.asarray(values) for name, values in association_features.items()
    }
    n_rows = len(labels)
    if not (len(delta) == len(genes) == n_rows):
        raise ValueError("axis-analysis delta, labels, and genes are not row-aligned")
    for name, values in feature_arrays.items():
        if len(values) != n_rows:
            raise ValueError(
                f"axis association feature {name!r} has {len(values)} rows; "
                f"expected {n_rows}"
            )
    if regression_features is not None and len(regression_features) != n_rows:
        raise ValueError(
            f"axis regression features have {len(regression_features)} rows; "
            f"expected {n_rows}"
        )

    requested_seeds = tuple(seeds)
    correlations = {
        name: {seed: [] for seed in requested_seeds} for name in feature_arrays
    }
    r2_values = {seed: [] for seed in requested_seeds}
    for seed in requested_seeds:
        for train_rows, test_rows in family_split_cv(genes, pfam_map, seed=seed):
            if (
                len(np.unique(labels[train_rows])) < 2
                or len(np.unique(labels[test_rows])) < 2
            ):
                continue

            delta_scaler = StandardScaler().fit(delta[train_rows])
            train_delta = delta_scaler.transform(delta[train_rows])
            test_delta = delta_scaler.transform(delta[test_rows])
            axis_model = LogisticRegression(
                max_iter=2000, C=1.0, random_state=seed
            ).fit(train_delta, labels[train_rows])
            train_scores = axis_model.decision_function(train_delta)
            test_scores = axis_model.decision_function(test_delta)

            for name, values in feature_arrays.items():
                correlation = spearmanr(test_scores, values[test_rows]).correlation
                if np.isfinite(correlation):
                    correlations[name][seed].append(float(correlation))

            if regression_features is not None and len(test_rows) >= 2:
                feature_scaler = StandardScaler().fit(regression_features[train_rows])
                train_features = feature_scaler.transform(
                    regression_features[train_rows]
                )
                test_features = feature_scaler.transform(regression_features[test_rows])
                ridge = Ridge(alpha=1.0).fit(train_features, train_scores)
                value = r2_score(test_scores, ridge.predict(test_features))
                if np.isfinite(value):
                    r2_values[seed].append(float(value))

    result = {
        "correlations": {
            name: _seed_summary(requested_seeds, values_by_seed)
            for name, values_by_seed in correlations.items()
        }
    }
    if regression_features is not None:
        result["regression_r2"] = _seed_summary(requested_seeds, r2_values)
    return result
