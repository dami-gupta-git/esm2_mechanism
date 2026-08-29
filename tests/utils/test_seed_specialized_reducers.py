"""Contract tests for non-scalar aggregation across model seeds."""

import numpy as np
import pytest

from esm2_mech.utils.seed_aggregation import (
    SeedUnavailableReason,
    aggregate_oof_dicts,
    aggregate_paired_seed_difference,
    aggregate_seed_confusion_matrices,
    aggregate_seed_vote,
    make_seed_payload_record,
    make_seed_record,
)

REQUESTED_SEEDS = (11, 23, 47)
ROW_IDS = np.array([101, 205, 309])
LABELS = np.array(["A", "B", "A"])
CLUSTERS = np.array(["G1", "G2", "G3"], dtype=object)
CLASSES = ["A", "B"]


def _oof(seed, order=None):
    if order is None:
        order = np.arange(len(ROW_IDS))
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    folds = np.array([0, 1, 2])
    return {
        "row_ids": ROW_IDS[order],
        "y_true": LABELS[order],
        "genes": CLUSTERS[order],
        "proba": proba[order],
        "folds": folds[order],
        "classes": CLASSES,
        "seed": seed,
    }


def _complete_oof_by_seed():
    return {
        11: _oof(11),
        23: _oof(23, np.array([2, 0, 1])),
        47: _oof(47, np.array([1, 2, 0])),
    }


class TestAggregateSeedOof:
    def test_preserves_actual_seed_identity_and_declared_row_order(self):
        result = aggregate_oof_dicts(
            REQUESTED_SEEDS,
            _complete_oof_by_seed(),
            declared_row_ids=ROW_IDS,
            declared_labels=LABELS,
            declared_clusters=CLUSTERS,
            class_order=CLASSES,
            declared_fold_ids=range(3),
        )
        assert result.available
        assert list(result.payload["oof_by_seed"]) == list(REQUESTED_SEEDS)
        assert np.array_equal(result.payload["row_ids"], ROW_IDS)
        for seed in REQUESTED_SEEDS:
            assert result.payload["oof_by_seed"][seed]["seed"] == seed
            assert np.array_equal(
                result.payload["oof_by_seed"][seed]["folds"], [0, 1, 2]
            )

    def test_missing_seed_is_unavailable_not_dropped(self):
        oof_by_seed = _complete_oof_by_seed()
        del oof_by_seed[23]
        result = aggregate_oof_dicts(
            REQUESTED_SEEDS,
            oof_by_seed,
            declared_row_ids=ROW_IDS,
            declared_labels=LABELS,
            declared_clusters=CLUSTERS,
            class_order=CLASSES,
            declared_fold_ids=range(3),
        )
        assert not result.available
        assert result.reason is SeedUnavailableReason.MISSING_SEED
        assert result.affected_seeds == (23,)

    def test_unexpected_shifted_seed_is_unavailable(self):
        oof_by_seed = _complete_oof_by_seed()
        oof_by_seed[24] = oof_by_seed.pop(23)
        result = aggregate_oof_dicts(
            REQUESTED_SEEDS,
            oof_by_seed,
            declared_row_ids=ROW_IDS,
            declared_labels=LABELS,
            declared_clusters=CLUSTERS,
            class_order=CLASSES,
            declared_fold_ids=range(3),
        )
        assert not result.available
        assert result.reason is SeedUnavailableReason.UNEXPECTED_SEED
        assert result.affected_seeds == (23, 24)

    def test_incomplete_row_set_is_unavailable_not_intersected(self):
        oof_by_seed = _complete_oof_by_seed()
        oof_by_seed[23] = {
            key: value[:-1] if isinstance(value, np.ndarray) else value
            for key, value in oof_by_seed[23].items()
        }
        result = aggregate_oof_dicts(
            REQUESTED_SEEDS,
            oof_by_seed,
            declared_row_ids=ROW_IDS,
            declared_labels=LABELS,
            declared_clusters=CLUSTERS,
            class_order=CLASSES,
            declared_fold_ids=range(3),
        )
        assert not result.available
        assert result.reason is SeedUnavailableReason.INVALID_ROW_SET
        assert result.affected_seeds == (23,)

    @pytest.mark.parametrize("field", ["y_true", "genes"])
    def test_metadata_mismatch_is_unavailable(self, field):
        oof_by_seed = _complete_oof_by_seed()
        corrupted = np.array(oof_by_seed[23][field], copy=True)
        corrupted[0] = "wrong"
        oof_by_seed[23][field] = corrupted
        result = aggregate_oof_dicts(
            REQUESTED_SEEDS,
            oof_by_seed,
            declared_row_ids=ROW_IDS,
            declared_labels=LABELS,
            declared_clusters=CLUSTERS,
            class_order=CLASSES,
            declared_fold_ids=range(3),
        )
        assert not result.available
        assert result.reason is SeedUnavailableReason.METADATA_MISMATCH

    def test_class_order_mismatch_is_unavailable(self):
        oof_by_seed = _complete_oof_by_seed()
        oof_by_seed[23]["classes"] = ["B", "A"]
        result = aggregate_oof_dicts(
            REQUESTED_SEEDS,
            oof_by_seed,
            declared_row_ids=ROW_IDS,
            declared_labels=LABELS,
            declared_clusters=CLUSTERS,
            class_order=CLASSES,
            declared_fold_ids=range(3),
        )
        assert not result.available
        assert result.reason is SeedUnavailableReason.CLASS_ORDER_MISMATCH

    def test_invalid_fold_shape_is_unavailable(self):
        oof_by_seed = _complete_oof_by_seed()
        oof_by_seed[23]["folds"] = np.array([0, 1])
        result = aggregate_oof_dicts(
            REQUESTED_SEEDS,
            oof_by_seed,
            declared_row_ids=ROW_IDS,
            declared_labels=LABELS,
            declared_clusters=CLUSTERS,
            class_order=CLASSES,
            declared_fold_ids=range(3),
        )
        assert not result.available
        assert result.reason is SeedUnavailableReason.INVALID_SHAPE

    def test_wrong_fold_set_is_unavailable(self):
        oof_by_seed = _complete_oof_by_seed()
        oof_by_seed[23]["folds"] = np.array([0, 0, 1])
        result = aggregate_oof_dicts(
            REQUESTED_SEEDS,
            oof_by_seed,
            declared_row_ids=ROW_IDS,
            declared_labels=LABELS,
            declared_clusters=CLUSTERS,
            class_order=CLASSES,
            declared_fold_ids=range(3),
        )
        assert not result.available
        assert result.reason is SeedUnavailableReason.INVALID_SHAPE
        assert result.affected_seeds == (23,)

    def test_all_bad_seeds_are_reported(self):
        oof_by_seed = _complete_oof_by_seed()
        oof_by_seed[23]["folds"] = np.array([0, 0, 1])
        oof_by_seed[47]["proba"][0, 0] = np.nan
        result = aggregate_oof_dicts(
            REQUESTED_SEEDS,
            oof_by_seed,
            declared_row_ids=ROW_IDS,
            declared_labels=LABELS,
            declared_clusters=CLUSTERS,
            class_order=CLASSES,
            declared_fold_ids=range(3),
        )
        assert not result.available
        assert result.affected_seeds == (23, 47)
        assert result.contributing_seeds == (11,)


class TestAggregateSeedVote:
    def test_true_and_false_are_available_decisions(self):
        true_vote = aggregate_seed_vote(
            REQUESTED_SEEDS,
            [
                make_seed_record(11, 0.01),
                make_seed_record(23, 0.02),
                make_seed_record(47, 0.20),
            ],
            threshold=0.05,
            minimum_supporting_seeds=2,
        )
        false_vote = aggregate_seed_vote(
            REQUESTED_SEEDS,
            [
                make_seed_record(11, 0.01),
                make_seed_record(23, 0.20),
                make_seed_record(47, 0.30),
            ],
            threshold=0.05,
            minimum_supporting_seeds=2,
        )
        assert true_vote.available and true_vote.payload["decision"] is True
        assert false_vote.available and false_vote.payload["decision"] is False

    def test_missing_value_is_unavailable_not_false(self):
        result = aggregate_seed_vote(
            REQUESTED_SEEDS,
            [
                make_seed_record(11, 0.01),
                make_seed_record(23, None),
                make_seed_record(47, 0.01),
            ],
            threshold=0.05,
            minimum_supporting_seeds=2,
        )
        assert not result.available
        assert result.payload is None
        assert result.reason is SeedUnavailableReason.INVALID_VALUE


def test_paired_difference_is_formed_within_seed_before_reduction():
    result = aggregate_paired_seed_difference(
        REQUESTED_SEEDS,
        [
            make_seed_record(11, 10.0),
            make_seed_record(23, 1.0),
            make_seed_record(47, 4.0),
        ],
        [
            make_seed_record(11, 9.0),
            make_seed_record(23, 3.0),
            make_seed_record(47, 1.0),
        ],
    )
    expected_differences = np.array([1.0, -2.0, 3.0])
    assert result.mean == pytest.approx(float(expected_differences.mean()))
    assert result.spread == pytest.approx(float(expected_differences.std(ddof=1)))


def test_paired_difference_is_unavailable_when_either_arm_is_incomplete():
    result = aggregate_paired_seed_difference(
        REQUESTED_SEEDS,
        [make_seed_record(11, 1.0), make_seed_record(23, 2.0)],
        [
            make_seed_record(11, 1.0),
            make_seed_record(23, 2.0),
            make_seed_record(47, 3.0),
        ],
    )
    assert not result.available
    assert result.reason is SeedUnavailableReason.MISSING_SEED
    assert result.affected_seeds == (47,)


class TestAggregateSeedConfusionMatrices:
    def _records(self):
        matrices = {
            11: [[8, 2], [1, 9]],
            23: [[1, 9], [4, 6]],
            47: [[5, 5], [2, 8]],
        }
        return [
            make_seed_payload_record(seed, {"matrix": matrix, "class_order": CLASSES})
            for seed, matrix in matrices.items()
        ]

    def test_keeps_pooled_counts_separate_from_equal_seed_normalized_mean(self):
        result = aggregate_seed_confusion_matrices(
            REQUESTED_SEEDS, CLASSES, self._records()
        )
        assert result.available
        assert np.array_equal(result.payload["pooled_raw"], [[14, 16], [7, 23]])
        assert np.allclose(
            result.payload["normalized_seed_mean"],
            [[0.4666666667, 0.5333333333], [0.2333333333, 0.7666666667]],
        )

    def test_zero_support_row_is_unavailable(self):
        records = self._records()
        records[1] = make_seed_payload_record(
            23, {"matrix": [[0, 0], [4, 6]], "class_order": CLASSES}
        )
        result = aggregate_seed_confusion_matrices(REQUESTED_SEEDS, CLASSES, records)
        assert not result.available
        assert result.reason is SeedUnavailableReason.ZERO_SUPPORT
        assert result.affected_seeds == (23,)

    def test_class_order_mismatch_is_unavailable(self):
        records = self._records()
        records[1] = make_seed_payload_record(
            23, {"matrix": [[1, 9], [4, 6]], "class_order": ["B", "A"]}
        )
        result = aggregate_seed_confusion_matrices(REQUESTED_SEEDS, CLASSES, records)
        assert not result.available
        assert result.reason is SeedUnavailableReason.CLASS_ORDER_MISMATCH
