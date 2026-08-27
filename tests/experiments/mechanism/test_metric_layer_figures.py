"""Figure readers preserve explicit unavailable metric values."""

import matplotlib.axes

from esm2_mech.experiments.mechanism import make_figures
from esm2_mech.figures import manuscript_figures


def _unavailable_family_clustering_views():
    return {
        view: {
            "knn5_purity": 0.2,
            "knn5_purity_null": 0.1,
            "knn5_purity_ci": {"ci_low": 0.15, "ci_high": 0.25},
            "family_probe": {
                "status": "unavailable",
                "accuracy": None,
                "macro_f1": None,
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


def test_portfolio_within_family_uses_fold_aware_reference(monkeypatch):
    result = {
        "by_family": {
            "PF1": {
                "n_genes": 8,
                "gene_class_counts": {"GOF": 4, "LOF": 4},
                "majority_reference": {"macro_f1_mean": 0.3},
                "delta": {"mlp": {"macro_f1": {"mean": 0.4, "std": 0.05}}},
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
        "claim_2b_split_gap_summary": {
            "per_seed": [
                {"seed": 0, "point_diff": 0.1, "ci_low": 0.05, "ci_high": 0.15}
            ]
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
    enzyme = {
        "esm2_wt_embedding": {
            "majority_reference": {
                "gene_split": {"macro_f1_mean": 0.21},
                "family_split": {"macro_f1_mean": 0.34},
            },
            "logreg_gene_split": {"macro_f1_mean": 0.70},
            "logreg_family_split": {
                "macro_f1_mean": 0.65,
                "per_class_auroc_mean": {
                    class_name: 0.8
                    for class_name in manuscript_figures.ENZYME_CLASS_ORDER
                },
            },
            "paired_ci_logreg_minus_mechanism": {
                "point_diff": 0.20,
                "ci_low": 0.10,
                "ci_high": 0.30,
            },
            "paired_ci_mlp_minus_logreg": {
                "point_diff": 0.01,
                "ci_low": -0.02,
                "ci_high": 0.04,
            },
        },
        "proteome_features": {
            "logreg_gene_split": {"macro_f1_mean": 0.60},
            "logreg_family_split": {"macro_f1_mean": 0.50},
        },
        "gate_evaluation": {"2G_minimum_f1_gap": 0.10},
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
