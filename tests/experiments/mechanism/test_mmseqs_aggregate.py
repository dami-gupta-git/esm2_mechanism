"""
Tests for esm2_mech.experiments.mechanism.mmseqs_cluster_holdout.aggregate_seeds.

Invariant:
- a per-seed metric that is NaN (a fold scored but the metric was NaN) is
  filtered out of the across-seed mean/std, matching the None-filter — a single
  NaN must not poison the aggregate.
"""

import numpy as np
import pytest

from esm2_mech.experiments.mechanism.mmseqs_cluster_holdout import aggregate_seeds


def _seed(macro_f1):
    return {"V1": {"macro_f1_mean": macro_f1, "per_gene_f1_mean": macro_f1}}


def test_nan_seed_excluded_from_mean():
    all_res = [_seed(0.40), _seed(float("nan")), _seed(0.50)]
    out = aggregate_seeds(all_res)
    # mean of the two finite seeds = 0.45, not NaN.
    assert out["V1_macro_f1_mean"] == pytest.approx(0.45)
    assert not np.isnan(out["V1_macro_f1_mean"])


def test_none_seed_excluded_from_mean():
    all_res = [_seed(0.40), _seed(None), _seed(0.50)]
    out = aggregate_seeds(all_res)
    assert out["V1_macro_f1_mean"] == pytest.approx(0.45)


def test_all_nan_omits_key():
    all_res = [_seed(float("nan")), _seed(None)]
    out = aggregate_seeds(all_res)
    # No scorable seed → the metric key is omitted entirely (n == 0).
    assert "V1_macro_f1_mean" not in out
