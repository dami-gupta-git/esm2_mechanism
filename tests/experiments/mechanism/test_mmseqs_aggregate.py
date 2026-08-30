"""
Tests for esm2_mech.experiments.mechanism.mmseqs_cluster_holdout.aggregate_seeds.

Invariant:
- a required seed with a missing metric makes the complete across-seed metric
  unavailable; surviving seeds are not averaged into a reduced-seed result.
"""

from esm2_mech.experiments.mechanism.mmseqs_cluster_holdout import aggregate_seeds
from esm2_mech.utils.seed_aggregation import seed_result_contract


def _seed(seed, macro_f1):
    return {
        **seed_result_contract(seed),
        "V1": {
            "status": "success",
            "macro_f1_mean": macro_f1,
            "per_gene_f1_mean": macro_f1,
        },
    }


def test_nan_seed_makes_aggregate_unavailable():
    all_res = [_seed(0, 0.40), _seed(1, float("nan")), _seed(2, 0.50)]
    out = aggregate_seeds(all_res, range(3))
    assert out["V1_macro_f1_seed_aggregate"]["state"] == "unavailable"


def test_none_seed_makes_aggregate_unavailable():
    all_res = [_seed(0, 0.40), _seed(1, None), _seed(2, 0.50)]
    out = aggregate_seeds(all_res, range(3))
    assert out["V1_macro_f1_seed_aggregate"]["state"] == "unavailable"


def test_all_nan_retains_explicit_null_key():
    all_res = [_seed(0, float("nan")), _seed(1, None)]
    out = aggregate_seeds(all_res, range(2))
    assert out["V1_macro_f1_seed_aggregate"]["mean"] is None
