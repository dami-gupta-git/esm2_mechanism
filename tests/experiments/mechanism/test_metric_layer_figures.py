"""Figure readers preserve explicit unavailable metric values."""

import matplotlib.axes
import pytest

from esm2_mech.experiments.mechanism import make_figures
from esm2_mech.experiments.mechanism.seed_results import read_feature_metric
from esm2_mech.figures import manuscript_figures
from esm2_mech.utils.constants import SEED_AGGREGATION_SCHEMA_VERSION
from esm2_mech.utils.seed_aggregation import (
    aggregate_paired_seed_difference,
    aggregate_seed_values,
    make_seed_record,
)


def _unavailable_family_clustering_views():
    purity = aggregate_seed_values(
        range(3), [make_seed_record(seed, 0.2) for seed in range(3)]
    ).to_dict()
    purity_null = aggregate_seed_values(
        range(3), [make_seed_record(seed, 0.1) for seed in range(3)]
    ).to_dict()
    unavailable = aggregate_seed_values(
        range(3), [make_seed_record(seed, None) for seed in range(3)]
    ).to_dict()
    return {
        view: {
            "knn5_purity_seed_aggregate": purity,
            "knn5_purity_null_seed_aggregate": purity_null,
            "family_probe": {
                "status": "unavailable",
                "accuracy_seed_aggregate": unavailable,
                "majority_baseline_accuracy_seed_aggregate": unavailable,
            },
        }
        for view in ("wt_mean", "mut_mean", "delta_mean")
    }


def test_portfolio_family_figure_accepts_unavailable_probe(monkeypatch):
    monkeypatch.setattr(
        make_figures,
        "_load_json",
        lambda _path: {"by_view": _unavailable_family_clustering_views()},
    )
    saved = []
    monkeypatch.setattr(make_figures, "_save", lambda figure, name: saved.append(name))

    make_figures.fig_family_clustering()

    assert saved == ["fig5_family_clustering.png"]


def test_portfolio_within_family_plots_the_within_seed_paired_lift(monkeypatch):
    # Both arms move together, so every within-seed difference is the same and the
    # paired spread is zero — an error bar taken from one arm alone would not be.
    probe_values = (0.35, 0.40, 0.45)
    reference_values = (0.30, 0.35, 0.40)
    score = aggregate_seed_values(
        range(3), [make_seed_record(seed, value) for seed, value in enumerate(probe_values)]
    ).to_dict()
    baseline = aggregate_seed_values(
        range(3),
        [make_seed_record(seed, value) for seed, value in enumerate(reference_values)],
    ).to_dict()
    lift = aggregate_paired_seed_difference(
        range(3),
        [make_seed_record(seed, value) for seed, value in enumerate(probe_values)],
        [make_seed_record(seed, value) for seed, value in enumerate(reference_values)],
    )
    assert lift.spread == pytest.approx(0.0)
    result = {
        "by_family": {
            "PF1": {
                "n_genes": 8,
                "gene_class_counts": {"GOF": 4, "LOF": 4},
                "majority_reference": {"macro_f1_seed_aggregate": baseline},
                "delta_mlp_minus_majority_seed_aggregate": lift.to_dict(),
                "delta": {"mlp": {"macro_f1": score}},
            }
        }
    }
    monkeypatch.setattr(make_figures, "_load_json", lambda _path: result)
    saved = []
    monkeypatch.setattr(make_figures, "_save", lambda figure, name: saved.append(name))

    make_figures.fig_within_family()

    assert saved == ["fig4_within_family.png"]


def test_manuscript_family_figure_accepts_unavailable_probe(monkeypatch):
    clustering = {"by_view": _unavailable_family_clustering_views()}
    aggregate = {
        "gene_minus_family_split_gap": {
            "gene_minus_family_seed_aggregate": aggregate_seed_values(
                range(3), [make_seed_record(seed, 0.1) for seed in range(3)]
            ).to_dict()
        }
    }
    leakage = {
        "by_feature": {
            "wt_only_mean": {
                "leakage_fraction": 0.5,
                "ci": {"ci_low": 0.4, "ci_high": 0.6},
            }
        }
    }

    def fake_load(path):
        if path == manuscript_figures.FAMILY_CLUSTERING_JSON:
            return clustering
        if path == manuscript_figures.MECHANISM_AGGREGATE_JSON:
            return aggregate
        if path == manuscript_figures.LEAKAGE_FRACTION_JSON:
            return leakage
        raise AssertionError(path)

    monkeypatch.setattr(manuscript_figures, "_load_json", fake_load)
    saved = []
    monkeypatch.setattr(
        manuscript_figures,
        "_save_figure",
        lambda figure, stem: saved.append(stem),
    )

    manuscript_figures.figure3_family_information()

    assert saved == ["figure3_family_information"]


def test_manuscript_enzyme_figure_uses_family_split_reference(monkeypatch):
    def aggregate(value):
        return aggregate_seed_values(
            range(3), [make_seed_record(seed, value) for seed in range(3)]
        ).to_dict()

    enzyme = {
        "esm2_wt_embedding": {
            "majority_reference": {
                "gene_split": aggregate(0.21),
                "family_split": aggregate(0.34),
            },
            "logreg_gene_split": {"macro_f1_seed_aggregate": aggregate(0.70)},
            "logreg_family_split": {
                "macro_f1_seed_aggregate": aggregate(0.65),
                "per_class_auroc_seed_aggregate": {
                    class_name: aggregate(0.8)
                    for class_name in manuscript_figures.ENZYME_CLASS_ORDER
                },
            },
            "paired_logreg_minus_mechanism_seed_aggregate": aggregate(0.20),
            "paired_mlp_minus_logreg_seed_aggregate": aggregate(0.01),
        },
        "proteome_features": {
            "logreg_gene_split": {"macro_f1_seed_aggregate": aggregate(0.60)},
            "logreg_family_split": {"macro_f1_seed_aggregate": aggregate(0.50)},
        },
        "gate_evaluation": {"enzyme_minus_mechanism_minimum_f1_gap": 0.10},
    }
    monkeypatch.setattr(manuscript_figures, "_load_json", lambda _path: enzyme)
    monkeypatch.setattr(manuscript_figures, "_save_figure", lambda _figure, _stem: None)

    horizontal_lines = []
    original_axhline = matplotlib.axes.Axes.axhline

    def record_axhline(axis, y=0, *args, **kwargs):
        horizontal_lines.append(y)
        return original_axhline(axis, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "axhline", record_axhline)

    manuscript_figures.figure5_enzyme_classification()

    assert 0.34 in horizontal_lines
    assert 0.21 not in horizontal_lines


# ---------------------------------------------------------------------------
# across-seed mechanism readers
# ---------------------------------------------------------------------------


def _seed_aggregate(mean, spread, seeds=(0, 1, 2, 3, 4)):
    return {
        "schema_version": SEED_AGGREGATION_SCHEMA_VERSION,
        "state": "available",
        "reason": None,
        "requested_seeds": list(seeds),
        "contributing_seeds": list(seeds),
        "affected_seeds": [],
        "mean": mean,
        "seed_std": spread,
        "sampling_unit": "model_seed",
        "message": None,
    }


def _unavailable_seed_aggregate(seeds=(0, 1, 2, 3, 4), affected=(4,)):
    return {
        "schema_version": SEED_AGGREGATION_SCHEMA_VERSION,
        "state": "unavailable",
        "reason": "failed_seed",
        "requested_seeds": list(seeds),
        "contributing_seeds": [seed for seed in seeds if seed not in affected],
        "affected_seeds": list(affected),
        "mean": None,
        "seed_std": None,
        "sampling_unit": "model_seed",
        "message": "a requested seed failed",
    }


def _mechanism_feature(mean, spread, *, available=True):
    aggregate = (
        _seed_aggregate(mean, spread) if available else _unavailable_seed_aggregate()
    )
    block = {"macro_f1_seed_aggregate": aggregate}
    for class_name in make_figures.MECHANISM_CLASSES:
        block[f"auroc_{class_name}_seed_aggregate"] = aggregate
    return block


def _mechanism_aggregate(available_features, unavailable_features=()):
    across_seed = {"gene_split": {}, "family_split": {}}
    for split in across_seed:
        for feature in available_features:
            across_seed[split][feature] = _mechanism_feature(0.4, 0.02)
        for feature in unavailable_features:
            across_seed[split][feature] = _mechanism_feature(
                None, None, available=False
            )
    return {"across_seed": across_seed}


def _patch_mechanism_figure_inputs(monkeypatch, aggregate):
    def fake_load(path):
        if path == make_figures.MECHANISM_AGGREGATE_JSON:
            return aggregate
        if path == make_figures.NAIVE_BASELINE_JSON:
            return {
                "by_strategy": {
                    "most_frequent": {
                        "gene": {
                            "macro_f1_seed_aggregate": _seed_aggregate(0.29, 0.01)
                        }
                    }
                }
            }
        raise AssertionError(path)

    monkeypatch.setattr(make_figures, "_load_json", fake_load)
    saved = []
    monkeypatch.setattr(make_figures, "_save", lambda figure, name: saved.append(name))
    return saved


def test_probe_ranking_omits_a_feature_without_a_seed_mean(monkeypatch, capsys):
    features = [key for key, _label in make_figures.MECH_FEATURES]
    aggregate = _mechanism_aggregate(features[:-1], unavailable_features=features[-1:])
    saved = _patch_mechanism_figure_inputs(monkeypatch, aggregate)

    make_figures.fig_probe_ranking()

    assert saved == ["fig3_probe_ranking.png"]
    assert f"[omitted] gene_split {features[-1]}" in capsys.readouterr().out


def test_family_split_figure_is_skipped_when_nothing_is_readable(monkeypatch, capsys):
    features = [key for key, _label in make_figures.MECH_FEATURES]
    aggregate = _mechanism_aggregate([], unavailable_features=features)
    saved = _patch_mechanism_figure_inputs(monkeypatch, aggregate)

    make_figures.fig_family_split()

    assert saved == []
    assert "no feature has a readable seed mean" in capsys.readouterr().out


def test_auroc_split_bars_drops_only_the_unavailable_class(monkeypatch):
    aggregate = _mechanism_aggregate(["wt_only_mean", "delta_mean"])
    dropped = make_figures.MECHANISM_CLASSES[0]
    for split in aggregate["across_seed"].values():
        split["delta_mean"][
            f"auroc_{dropped}_seed_aggregate"
        ] = _unavailable_seed_aggregate()
    saved = _patch_mechanism_figure_inputs(monkeypatch, aggregate)

    make_figures.fig_auroc_split_bars()

    assert saved == ["fig6_auroc_split_bars.png"]


def test_a_stale_schema_version_is_not_read_as_a_value(monkeypatch, capsys):
    aggregate = _mechanism_aggregate(["delta_mean", "wt_only_mean"])
    aggregate["across_seed"]["gene_split"]["delta_mean"]["macro_f1_seed_aggregate"][
        "schema_version"
    ] = SEED_AGGREGATION_SCHEMA_VERSION + 1
    _patch_mechanism_figure_inputs(monkeypatch, aggregate)

    read = read_feature_metric(
        aggregate["across_seed"], "gene_split", "delta_mean", "macro_f1"
    )

    assert read.available is False
    assert read.value is None


def test_auroc_panels_are_skipped_when_no_class_is_readable(monkeypatch, capsys):
    aggregate = _mechanism_aggregate(
        [], unavailable_features=["wt_only_mean", "delta_mean"]
    )
    saved = _patch_mechanism_figure_inputs(monkeypatch, aggregate)

    make_figures.fig_auroc_split_bars()
    make_figures.fig_auroc_split_slope()

    assert saved == []
    output = capsys.readouterr().out
    assert "[skipped] fig6_auroc_split_bars" in output
    assert "[skipped] fig7_auroc_split_slope" in output
