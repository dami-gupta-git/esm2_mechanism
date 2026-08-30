from esm2_mech.experiments.proteome_features.proteome_mechanism import aggregate_seeds
from esm2_mech.utils.seed_aggregation import seed_result_contract


def _seed_result(seed, *, v4_status="success"):
    success = {"status": "success", "macro_f1_mean": 0.4}
    result = {
        **seed_result_contract(seed),
        "V1_family_split": dict(success),
        "V2_lgbm_family_split": dict(success),
        "V2_best_macro_f1_mean": 0.4,
        "V2_logreg_observed_subset": {**success, "frac_observed": 0.5},
        "V2_histgb_observed_subset": dict(success),
        "V3_family_split": dict(success),
        "V3_gene_split": dict(success),
        "V3_leakage_delta": 0.0,
    }
    if v4_status == "success":
        result["V4_family_split"] = {
            **success,
            "frac_observed": 0.5,
        }
        result["V3_family_split_matched_to_V4"] = dict(success)
    else:
        skipped = {"status": "skipped", "skipped": True, "reason": "gate_failed"}
        result["V4_family_split"] = skipped
        result["V3_family_split_matched_to_V4"] = dict(skipped)
    return result


def test_variants_only_declares_v4_once_without_seed_aggregate():
    summary = aggregate_seeds([_seed_result(0)], [0], include_v4=False)
    assert summary["arm_exclusions"]["V4"]["reason"] == "excluded_by_variants_only_option"
    assert "V4_family_split_macro_f1_seed_aggregate" not in summary


def test_seed_specific_v4_skip_makes_v4_aggregate_unavailable():
    summary = aggregate_seeds(
        [_seed_result(0), _seed_result(1, v4_status="skipped")],
        [0, 1],
    )
    aggregate = summary["V4_family_split_macro_f1_seed_aggregate"]
    assert aggregate["state"] == "unavailable"
    assert aggregate["reason"] == "skipped_seed"
    assert aggregate["affected_seeds"] == [1]
