"""Regression tests for self-contained gene-level mechanism comparisons."""

import pytest

from esm2_mech.experiments.badonyi.badonyi_mechanism import (
    aggregate_seeds,
    build_seed_comparisons,
)


def _seed(v1: float, v2: float, v_bad: float, v2_bad: float, v1_bad: float, v_all: float):
    return {
        "V1": {"macro_f1_mean": v1},
        "V2": {"macro_f1_mean": v2},
        "V_bad": {"macro_f1_mean": v_bad},
        "V2_bad": {"macro_f1_mean": v2_bad},
        "V1_bad": {"macro_f1_mean": v1_bad},
        "V_all": {"macro_f1_mean": v_all},
    }


def test_gene_feature_contrast_uses_esm2_arm_from_same_result():
    result = aggregate_seeds(
        [
            _seed(0.40, 0.45, 0.41, 0.47, 0.42, 0.48),
            _seed(0.42, 0.46, 0.40, 0.46, 0.43, 0.47),
        ]
    )

    comparison = result["comparisons"]["gene_features_minus_esm2"]
    assert comparison["left_arm"] == "V2"
    assert comparison["right_arm"] == "V1"
    assert comparison["left_mean"] == pytest.approx(0.455)
    assert comparison["right_mean"] == pytest.approx(0.41)
    assert comparison["difference_mean"] == pytest.approx(0.045)
    assert comparison["same_classifier"] is False
    assert comparison["left_classifier"] == "hist_gradient_boosting"
    assert comparison["right_classifier"] == "mlp"


def test_esm2_ablation_records_matched_classifier():
    comparisons = build_seed_comparisons(
        _seed(0.40, 0.45, 0.41, 0.47, 0.42, 0.48)
    )

    comparison = comparisons["esm2_added_to_gene_features"]
    assert comparison["left_arm"] == "V_all"
    assert comparison["right_arm"] == "V2_bad"
    assert comparison["difference"] == pytest.approx(0.01)
    assert comparison["same_classifier"] is True
    assert comparison["left_classifier"] == "hist_gradient_boosting"
    assert comparison["right_classifier"] == "hist_gradient_boosting"


def test_missing_comparison_arm_raises_instead_of_using_another_probe():
    seed_result = _seed(0.40, 0.45, 0.41, 0.47, 0.42, 0.48)
    del seed_result["V1"]

    with pytest.raises(KeyError, match="missing arms.*V1"):
        build_seed_comparisons(seed_result)


def test_nonfinite_comparison_score_raises():
    seed_result = _seed(float("nan"), 0.45, 0.41, 0.47, 0.42, 0.48)

    with pytest.raises(ValueError, match="macro-F1 must be finite"):
        build_seed_comparisons(seed_result)
