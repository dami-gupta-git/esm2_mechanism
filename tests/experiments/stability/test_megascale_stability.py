import numpy as np

from esm2_mech.experiments.stability.megascale_stability import (
    apply_decision_rule,
    run_stability_projection_3c,
)
from esm2_mech.utils.constants import BOOTSTRAP_MAX_DISCARD_FRAC

N_FOLDS = 5
N_FAMILIES = 50
VARIANTS_PER_GENE = 3


def test_3b_leaky_verdict_requires_ci_to_clear_threshold():
    assert apply_decision_rule(
        0.69, 0.55, 0.16, {"ci_low": 0.11, "ci_high": 0.17}
    ) == "LEAKY"
    assert apply_decision_rule(
        0.69, 0.55, 0.16, {"ci_low": 0.08, "ci_high": 0.17}
    ).startswith("UNDERPOWERED")
    assert apply_decision_rule(0.69, 0.55, 0.16).startswith("NOT ADJUDICATED")


def _merged_set(rng, n_features):
    """One gene per family, several variants per gene, labels cycling over classes.

    The family count has to exceed the fold count by enough that a bootstrap draw
    still puts at least one family in every fold. The paired difference is scored
    within each fold, so a draw that empties a fold is discarded, and with one
    family per fold almost every draw would be.
    """
    genes = np.array(
        [f"G{i}" for i in range(N_FAMILIES) for _ in range(VARIANTS_PER_GENE)]
    )
    pfam_map = {f"G{i}": f"F{i}" for i in range(N_FAMILIES)}
    labels = np.tile(np.array(["DN", "GOF", "LOF"]), N_FAMILIES)
    delta = rng.normal(size=(len(labels), n_features))
    return delta, labels, genes, pfam_map


def test_3c_returns_paired_family_bootstrap_ci():
    rng = np.random.RandomState(3)
    n_features = 6
    stability_delta = rng.normal(size=(60, n_features))
    stability_ddg = stability_delta[:, 0] + 0.1 * rng.normal(size=60)
    mechanism_delta, labels, genes, pfam_map = _merged_set(rng, n_features)

    result = run_stability_projection_3c(
        mechanism_delta,
        labels,
        genes,
        pfam_map,
        [],
        stability_delta,
        stability_ddg,
        n_folds=N_FOLDS,
        n_seeds=1,
        n_boot=100,
    )

    ci = result["difference_ci"]
    assert ci is not None
    assert ci["n_clusters"] == N_FAMILIES
    assert ci["point_diff"] is not None
    assert result["3C_verdict"] in {
        "pass — established",
        "fail — established",
        "underpowered — CI overlaps +0.01 threshold",
    }


def test_3c_discards_few_resamples_when_every_fold_holds_many_families():
    # A high discard rate means folds are losing whole classes, which makes the
    # surviving draws a different statistic from the point estimate. With ten
    # families per fold it should be near zero, so a regression that empties or
    # unbalances folds shows up here rather than in the interval.
    rng = np.random.RandomState(3)
    n_features = 6
    stability_delta = rng.normal(size=(60, n_features))
    stability_ddg = stability_delta[:, 0] + 0.1 * rng.normal(size=60)
    mechanism_delta, labels, genes, pfam_map = _merged_set(rng, n_features)

    result = run_stability_projection_3c(
        mechanism_delta,
        labels,
        genes,
        pfam_map,
        [],
        stability_delta,
        stability_ddg,
        n_folds=N_FOLDS,
        n_seeds=1,
        n_boot=200,
    )

    assert result["difference_ci"]["discard_frac"] <= BOOTSTRAP_MAX_DISCARD_FRAC
