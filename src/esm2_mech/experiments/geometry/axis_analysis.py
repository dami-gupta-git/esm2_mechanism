"""Family-held-out descriptive analyses of a supervised pathogenicity axis."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from esm2_mech.utils.metrics import mean_std_n
from esm2_mech.utils.splits import family_split_cv


def _summary(values: list[float]) -> dict:
    mean, std, count = mean_std_n(values)
    if count == 0:
        return {"mean": None, "std": None, "n": 0, "missing": True}
    return {"mean": mean, "std": std, "n": count, "missing": False}


def format_axis_summary(summary: dict) -> str:
    """Format a summary without turning a missing estimate into a number."""
    if summary["missing"]:
        return "unavailable (n=0)"
    return f"{summary['mean']:+.3f} ± {summary['std']:.3f} (n={summary['n']})"


def family_held_out_axis_analysis(
    delta: np.ndarray,
    labels: np.ndarray,
    genes: np.ndarray,
    pfam_map: dict,
    association_features: dict[str, np.ndarray],
    regression_features: np.ndarray | None = None,
    seeds=range(5),
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

    correlations = {name: [] for name in feature_arrays}
    r2_values = []
    for seed in seeds:
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
                    correlations[name].append(float(correlation))

            if regression_features is not None and len(test_rows) >= 2:
                feature_scaler = StandardScaler().fit(regression_features[train_rows])
                train_features = feature_scaler.transform(
                    regression_features[train_rows]
                )
                test_features = feature_scaler.transform(regression_features[test_rows])
                ridge = Ridge(alpha=1.0).fit(train_features, train_scores)
                value = r2_score(test_scores, ridge.predict(test_features))
                if np.isfinite(value):
                    r2_values.append(float(value))

    result = {
        "correlations": {
            name: _summary(values) for name, values in correlations.items()
        }
    }
    if regression_features is not None:
        result["regression_r2"] = _summary(r2_values)
    return result
