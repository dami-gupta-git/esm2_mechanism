import numpy as np

from esm2_mech.experiments.stability.megascale_stability import (
    apply_decision_rule,
    run_h3_stability_projection,
)


def test_h2_leaky_verdict_requires_ci_to_clear_threshold():
    assert apply_decision_rule(
        0.69, 0.55, 0.16, {"ci_low": 0.11, "ci_high": 0.17}
    ) == "LEAKY"
    assert apply_decision_rule(
        0.69, 0.55, 0.16, {"ci_low": 0.08, "ci_high": 0.17}
    ).startswith("UNDERPOWERED")
    assert apply_decision_rule(0.69, 0.55, 0.16).startswith("NOT ADJUDICATED")


def test_h3_returns_paired_family_bootstrap_ci():
    rng = np.random.RandomState(3)
    n_features = 6
    stability_delta = rng.normal(size=(60, n_features))
    stability_ddg = stability_delta[:, 0] + 0.1 * rng.normal(size=60)

    families = ["F1", "F2", "F3", "F4", "F5"]
    genes = np.array([f"G{i}" for i in range(15) for _ in range(3)])
    pfam_map = {f"G{i}": families[i % len(families)] for i in range(15)}
    labels = np.tile(np.array(["DN", "GOF", "LOF"]), 15)
    mechanism_delta = rng.normal(size=(len(labels), n_features))

    result = run_h3_stability_projection(
        mechanism_delta,
        labels,
        genes,
        pfam_map,
        [],
        stability_delta,
        stability_ddg,
        n_folds=5,
        n_seeds=1,
        n_boot=40,
    )

    ci = result["difference_ci"]
    assert ci is not None
    assert ci["n_clusters"] == 5
    assert ci["point_diff"] is not None
    assert result["h3_verdict"] in {
        "pass — established",
        "fail — established",
        "underpowered — CI overlaps +0.01 threshold",
    }
