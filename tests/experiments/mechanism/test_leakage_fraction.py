"""Tests for fold-aware, seed-complete leakage point estimates."""

import numpy as np
import pytest

from esm2_mech.experiments.mechanism.leakage_fraction import (
    leakage_fraction_per_feature,
)
from esm2_mech.utils.constants import LOF, MECHANISM_CLASSES


def _oof_cache_entry(n=60, seed=42, n_folds=3):
    rng = np.random.RandomState(seed)
    if n % n_folds != 0:
        raise ValueError("test fixture requires equal fold sizes")
    rows_per_fold = n // n_folds
    fold_labels = np.resize(MECHANISM_CLASSES, rows_per_fold)
    y_true = np.tile(fold_labels, n_folds)
    row_ids = list(range(n))
    genes = [f"gene_{index % 6}" for index in range(n)]
    folds = np.repeat(np.arange(n_folds), rows_per_fold).tolist()
    gene_pred = y_true.copy()
    gene_pred[n // 2 :] = rng.choice(MECHANISM_CLASSES, size=n - n // 2)
    family_pred = rng.choice(MECHANISM_CLASSES, size=n)

    def arm(predictions):
        return {
            "row_ids": row_ids,
            "y_true": y_true.tolist(),
            "pred": predictions.tolist(),
            "genes": genes,
            "folds": folds,
        }

    return {"gene_split": arm(gene_pred), "family_split": arm(family_pred)}


def test_missing_cache_is_an_error():
    with pytest.raises(ValueError, match="OOF caches are required"):
        leakage_fraction_per_feature("delta_mean", chance=0.30, requested_seeds=[0])


def test_headline_is_rescored_from_aligned_oof_rows():
    result = leakage_fraction_per_feature(
        "delta_mean",
        chance=0.20,
        requested_seeds=[4],
        oof_cache_entries={4: _oof_cache_entry(seed=4)},
    )
    assert result["gene_macro_f1_seed_aggregate"]["mean"] != pytest.approx(0.99)
    assert result["family_macro_f1_seed_aggregate"]["mean"] != pytest.approx(0.01)
    assert result["chance_macro_f1"] == pytest.approx(0.20)


def test_at_floor_leakage_fraction_is_unavailable():
    n_folds = 3
    rows_per_fold = 6
    n_rows = n_folds * rows_per_fold
    y_true = np.tile(MECHANISM_CLASSES, n_rows // len(MECHANISM_CLASSES))
    predictions = np.full(n_rows, LOF, dtype=object)
    arm = {
        "row_ids": list(range(n_rows)),
        "y_true": y_true.tolist(),
        "pred": predictions.tolist(),
        "genes": [f"gene_{row}" for row in range(n_rows)],
        "folds": np.repeat(np.arange(n_folds), rows_per_fold).tolist(),
    }
    result = leakage_fraction_per_feature(
        "delta_mean",
        chance=1.0 / 6.0,
        requested_seeds=[0],
        oof_cache_entries={0: {"gene_split": arm, "family_split": arm}},
    )
    assert result["leakage_fraction"] is None


def test_different_seed_row_sets_are_rejected():
    entries = {seed: _oof_cache_entry(seed=seed) for seed in range(3)}
    for arm in ("gene_split", "family_split"):
        for key in ("row_ids", "y_true", "pred", "genes", "folds"):
            entries[1][arm][key] = entries[1][arm][key][:40]
    with pytest.raises(ValueError, match="different family-eligible row set"):
        leakage_fraction_per_feature(
            "delta_mean",
            chance=0.28,
            requested_seeds=range(3),
            oof_cache_entries=entries,
        )


def test_missing_requested_seed_makes_summary_unavailable():
    result = leakage_fraction_per_feature(
        "delta_mean",
        chance=0.20,
        requested_seeds=[0, 1],
        oof_cache_entries={0: _oof_cache_entry(seed=0)},
    )
    assert result["status"] == "unscorable"
    assert result["gene_macro_f1_seed_aggregate"]["reason"] == "missing_seed"
