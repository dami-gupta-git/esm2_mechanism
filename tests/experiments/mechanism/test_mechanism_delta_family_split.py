"""
Tests for the mechanism experiment's probe.

Invariants:
- run_logreg_pca_cv fits PCA inside each fold on training rows only, never on the
  full dataset, and skips PCA when the feature is already narrower than n_pca.
- The out-of-fold predictions carry the fold each row was scored in, so the metrics
  can be computed within fold instead of over the pooled concatenation.
- The reported macro-F1 is the mean over folds, and no pooled macro-F1 is emitted.
"""

import json
from unittest.mock import patch

import numpy as np
import pytest

from esm2_mech.experiments.mechanism import mechanism_delta_family_split as family_probe
from esm2_mech.experiments.mechanism.mechanism_delta_family_split import run
from esm2_mech.utils.probes import run_logreg_pca_cv
from esm2_mech.utils.constants import MECHANISM_CLASSES
from esm2_mech.utils.classification import validate_complete_classification_splits
from esm2_mech.utils.constants import SEED_AGGREGATION_SCHEMA_VERSION
from esm2_mech.utils.metrics import FOLD_SAMPLING_UNIT
from esm2_mech.utils.seed_aggregation import (
    SEED_SCHEMA_KEY,
    SEED_STATUS_KEY,
    SEED_STATUS_SUCCESS,
)


def _contract(labels, genes, splits):
    return validate_complete_classification_splits(
        splits,
        requested_folds=len(splits),
        eligible_rows=np.concatenate([test for _train, test in splits]),
        labels=labels,
        classes=MECHANISM_CLASSES,
        groups=np.arange(len(labels)),
        held_out_unit="row",
    )


def _make_data(n=120, dim=50, n_classes=3, n_genes=6, seed=42):
    """Synthetic data with enough structure for a 3-class probe."""
    rng = np.random.RandomState(seed)
    genes = np.array([f"gene_{i % n_genes}" for i in range(n)])
    classes = ["GOF", "DN", "LOF"]
    labels = np.array([classes[i % n_classes] for i in range(n)])
    X = rng.randn(n, dim)
    for i, cls in enumerate(classes):
        X[labels == cls, :3] += rng.randn(3) * 2
    return X, labels, genes


def _mechanism_data(labels, genes, embedding):
    """The row-aligned bundle `run` expects, built from one set of labels/genes.

    The four embedding views are offset copies of one matrix so a probe fitted on
    the wrong view produces different numbers. FoldX and AlphaMissense are all-NaN
    because these tests exercise the embedding features only.
    """
    n_rows = len(labels)
    return {
        "valid_variants": [
            {
                "gene": genes[row],
                "uniprot_id": f"P{row}",
                "aa_pos": 1,
                "aa_wt": "A",
                "aa_mut": "V",
            }
            for row in range(n_rows)
        ],
        "emb_wt_mean": embedding,
        "emb_mut_mean": embedding + 1,
        "emb_wt_pos": embedding + 2,
        "emb_mut_pos": embedding + 3,
        "labels_3class": labels,
        "genes_arr": genes,
        "foldx_ddg": np.full(n_rows, np.nan),
        "aa_wt_list": ["A"] * n_rows,
        "aa_mut_list": ["V"] * n_rows,
        "alphamissense_scores": np.full(n_rows, np.nan),
    }


def _count_pca_fits(call):
    """Run `call` and return the training-row counts PCA was fitted on."""
    from sklearn.decomposition import PCA as RealPCA

    fit_call_counts = []
    original_fit_transform = RealPCA.fit_transform

    def tracking_fit_transform(self, X_in, *args, **kwargs):
        fit_call_counts.append(X_in.shape[0])
        return original_fit_transform(self, X_in, *args, **kwargs)

    with patch.object(RealPCA, "fit_transform", tracking_fit_transform):
        call()
    return fit_call_counts


class TestPcaPerFold:

    def test_pca_fitted_inside_folds_not_globally(self):
        X, labels, genes = _make_data(n=100, dim=40)
        splits = [(np.arange(0, 80), np.arange(80, 100))]
        counts = _count_pca_fits(
            lambda: run_logreg_pca_cv(
                X, labels, splits, MECHANISM_CLASSES, _contract(labels, genes, splits),
                genes=genes, n_pca=10
            )
        )
        assert counts == [80]

    def test_pca_not_applied_when_dim_below_threshold(self):
        X, labels, genes = _make_data(n=60, dim=5)
        splits = [(np.arange(0, 40), np.arange(40, 60))]
        counts = _count_pca_fits(
            lambda: run_logreg_pca_cv(
                X, labels, splits, MECHANISM_CLASSES, _contract(labels, genes, splits),
                genes=genes, n_pca=10
            )
        )
        assert counts == []

    def test_no_pca_when_n_pca_is_none(self):
        X, labels, genes = _make_data(n=60, dim=40)
        splits = [(np.arange(0, 40), np.arange(40, 60))]
        counts = _count_pca_fits(
            lambda: run_logreg_pca_cv(
                X, labels, splits, MECHANISM_CLASSES, _contract(labels, genes, splits),
                genes=genes, n_pca=None
            )
        )
        assert counts == []


class TestOutOfFoldCarriesItsFold:

    def _two_fold_run(self):
        X, labels, genes = _make_data(n=90, dim=10)
        splits = [
            (np.arange(0, 60), np.arange(60, 90)),
            (np.arange(30, 90), np.arange(0, 30)),
        ]
        return run_logreg_pca_cv(
            X, labels, splits, MECHANISM_CLASSES, _contract(labels, genes, splits),
            genes=genes, return_oof=True
        )

    def test_every_oof_row_records_its_fold(self):
        _agg, oof = self._two_fold_run()
        assert len(oof["folds"]) == len(oof["row_ids"])
        assert sorted(set(oof["folds"].tolist())) == [0, 1]

    def test_each_row_appears_once_under_one_fold(self):
        _, oof = self._two_fold_run()
        assert len(set(oof["row_ids"].tolist())) == len(oof["row_ids"])
        for fold in (0, 1):
            rows = oof["row_ids"][oof["folds"] == fold]
            assert len(rows) == 30

    def test_macro_f1_is_the_fold_mean_and_nothing_is_pooled(self):
        agg, _ = self._two_fold_run()
        assert "macro_f1_mean" in agg
        assert "macro_f1_pooled" not in agg

    def test_oof_requires_genes(self):
        X, labels, _ = _make_data(n=60, dim=5)
        splits = [(np.arange(0, 40), np.arange(40, 60))]
        with pytest.raises(ValueError, match="genes are required"):
            run_logreg_pca_cv(
                X, labels, splits, MECHANISM_CLASSES,
                _contract(labels, np.arange(len(labels)), splits),
                genes=None, return_oof=True
            )


def test_run_rejects_unknown_feature_before_fitting():
    data = _mechanism_data(np.array(["GOF"]), np.array(["G1"]), np.ones((1, 2)))

    with pytest.raises(ValueError, match="unknown mechanism feature"):
        run(data, out_dir="unused", feature_names=("not_a_feature",))


def test_run_fits_only_requested_feature(tmp_path, monkeypatch):
    n_rows = 6
    labels = np.array(["GOF", "DN", "LOF", "GOF", "DN", "LOF"])
    genes = np.array([f"G{row}" for row in range(n_rows)])
    embedding = np.arange(n_rows * 2, dtype=float).reshape(n_rows, 2)
    data = _mechanism_data(labels, genes, embedding)
    splits = [(np.array([0, 1, 2]), np.array([3, 4, 5]))]
    calls = []
    oof = {
        "row_ids": np.array([3, 4, 5]),
        "y_true": labels[[3, 4, 5]],
        "proba": np.eye(3),
        "genes": genes[[3, 4, 5]],
        "folds": np.zeros(3, dtype=int),
    }

    def fake_probe(X, y, cv_splits, *args, **kwargs):
        calls.append(kwargs["label"])
        return {"status": "success", "macro_f1_mean": 0.4, "macro_f1_std": 0.0,
                "auroc_GOF_mean": 0.5, "auroc_DN_mean": 0.5, "auroc_LOF_mean": 0.5}, oof

    pfam_path = tmp_path / "pfam.json"
    pfam_path.write_text("{}")
    monkeypatch.setattr(family_probe, "PFAM_JSON", pfam_path)
    monkeypatch.setattr(
        family_probe, "load_pfam_map", lambda _path: {gene: f"F{index}" for index, gene in enumerate(genes)}
    )
    monkeypatch.setattr(family_probe, "gene_split_cv", lambda *args, **kwargs: splits)
    monkeypatch.setattr(family_probe, "family_split_cv", lambda *args, **kwargs: splits)
    monkeypatch.setattr(family_probe, "run_logreg_pca_cv", fake_probe)
    monkeypatch.setattr(family_probe, "attach_mechanism_ci", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        family_probe,
        "paired_oof_diff",
        lambda *args, **kwargs: {
            "point_diff": 0.125,
            "ci_low": None,
            "n_clusters": 3,
        },
    )

    result = run(
        data,
        out_dir=str(tmp_path),
        compute_ci=False,
        n_folds=1,
        feature_names=("wt_only_mean",),
    )

    assert calls == ["wt_only_mean gene", "wt_only_mean family"]
    assert set(result["gene_split"]) == {"wt_only_mean"}
    assert set(result["family_split"]) == {"wt_only_mean"}
    assert (
        result["family_split"]["wt_only_mean"]["split_gap_paired"]["point_diff"]
        == pytest.approx(0.125)
    )


def test_no_ci_still_writes_oof_cache_with_exact_execution_binding(
    tmp_path, monkeypatch
):
    n_rows = 6
    labels = np.array(["GOF", "DN", "LOF", "GOF", "DN", "LOF"])
    genes = np.array([f"G{row}" for row in range(n_rows)])
    embedding = np.arange(n_rows * 2, dtype=float).reshape(n_rows, 2)
    data = _mechanism_data(labels, genes, embedding)
    splits = [(np.array([0, 1, 2]), np.array([3, 4, 5]))]
    oof = {
        "row_ids": np.array([3, 4, 5]),
        "y_true": labels[[3, 4, 5]],
        "proba": np.eye(3),
        "genes": genes[[3, 4, 5]],
        "folds": np.zeros(3, dtype=int),
    }

    monkeypatch.setattr(family_probe, "PFAM_JSON", tmp_path / "pfam.json")
    (tmp_path / "pfam.json").write_text("{}")
    monkeypatch.setattr(
        family_probe, "load_pfam_map", lambda _path: {gene: f"F{index}" for index, gene in enumerate(genes)}
    )
    monkeypatch.setattr(family_probe, "gene_split_cv", lambda *args, **kwargs: splits)
    monkeypatch.setattr(family_probe, "family_split_cv", lambda *args, **kwargs: splits)
    monkeypatch.setattr(
        family_probe,
        "run_logreg_pca_cv",
        lambda *args, **kwargs: ({
            "status": "success",
            "macro_f1_mean": 1.0,
            "macro_f1_std": 0.0,
            "auroc_GOF_mean": 1.0,
            "auroc_DN_mean": 1.0,
            "auroc_LOF_mean": 1.0,
        }, oof),
    )
    monkeypatch.setattr(family_probe, "attach_mechanism_ci", lambda *args, **kwargs: None)
    monkeypatch.setattr(family_probe, "paired_oof_diff", lambda *args, **kwargs: None)

    run(
        data,
        out_dir=str(tmp_path),
        compute_ci=False,
        n_folds=1,
        n_boot=10,
        feature_names=("wt_only_mean",),
    )

    result = json.loads((tmp_path / "family_split_baselines_seed0.json").read_text())
    cache = json.loads((tmp_path / "mechanism_oof_cache_seed0.json").read_text())
    assert cache["analysis_run_id"] == result["analysis_run_id"]
    assert cache["input_fingerprints"] == result["input_fingerprints"]
    assert cache["analysis_parameters"] == result["analysis_parameters"]
    assert set(cache["features"]) == {"wt_only_mean"}


def test_per_seed_file_declares_the_shared_seed_contract(tmp_path, monkeypatch):
    """The written per-seed file states its schema, seed and status, and each
    metric block states that its spread is over folds."""
    n_rows = 60
    labels = np.array(["GOF", "DN", "LOF"] * (n_rows // 3))
    genes = np.array([f"G{row % 6}" for row in range(n_rows)])
    embedding = np.random.RandomState(0).randn(n_rows, 8)
    data = _mechanism_data(labels, genes, embedding)
    (tmp_path / "pfam.json").write_text("{}")
    monkeypatch.setattr(family_probe, "PFAM_JSON", tmp_path / "pfam.json")
    monkeypatch.setattr(
        family_probe,
        "load_pfam_map",
        lambda _path: {gene: f"F{index % 3}" for index, gene in enumerate(set(genes))},
    )
    monkeypatch.setattr(family_probe, "attach_mechanism_ci", lambda *args, **kwargs: None)

    run(
        data,
        out_dir=str(tmp_path),
        seed=3,
        compute_ci=False,
        n_folds=2,
        feature_names=("wt_only_mean",),
    )

    written = json.loads((tmp_path / "family_split_baselines_seed3.json").read_text())
    assert written[SEED_SCHEMA_KEY] == SEED_AGGREGATION_SCHEMA_VERSION
    assert written["seed"] == 3
    assert written[SEED_STATUS_KEY] == SEED_STATUS_SUCCESS
    block = written["gene_split"]["wt_only_mean"]
    assert block["spread_sampling_unit"] == FOLD_SAMPLING_UNIT
    assert "macro_f1_std" in block
