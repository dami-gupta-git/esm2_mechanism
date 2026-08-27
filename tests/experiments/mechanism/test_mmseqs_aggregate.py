"""
Tests for esm2_mech.experiments.mechanism.mmseqs_cluster_holdout.aggregate_seeds.

Invariant:
- a required seed with a missing metric makes the complete across-seed metric
  unavailable; surviving seeds are not averaged into a reduced-seed result.
"""

import numpy as np
import pytest

from esm2_mech.experiments.mechanism.mmseqs_cluster_holdout import aggregate_seeds


def _seed(macro_f1):
    return {"V1": {"macro_f1_mean": macro_f1, "per_gene_f1_mean": macro_f1}}


def test_nan_seed_makes_aggregate_unavailable():
    all_res = [_seed(0.40), _seed(float("nan")), _seed(0.50)]
    out = aggregate_seeds(all_res)
    assert out["V1_macro_f1_mean"] is None


def test_none_seed_makes_aggregate_unavailable():
    all_res = [_seed(0.40), _seed(None), _seed(0.50)]
    out = aggregate_seeds(all_res)
    assert out["V1_macro_f1_mean"] is None


def test_all_nan_retains_explicit_null_key():
    all_res = [_seed(float("nan")), _seed(None)]
    out = aggregate_seeds(all_res)
    assert out["V1_macro_f1_mean"] is None
