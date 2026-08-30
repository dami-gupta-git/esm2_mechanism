"""
Tests for enzyme gate evaluation in enzyme_classification.run_multiseed.

Invariants:
- LogReg and MLP delegate to the shared probe runners.
- Every requested seed contributes through the shared seed contract.
- MLP-minus-LogReg differences are paired by seed before aggregation.
- An inferential interval targets the complete across-seed estimate.
- Enzyme-versus-mechanism inference is paired over their shared Pfam families.
"""

import json

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

from esm2_mech.experiments.proteome_features import enzyme_classification
from esm2_mech.experiments.proteome_features.enzyme_classification import (
    ENZYME_CLASSES,
    enzyme_input_fingerprints,
    run_multiseed,
)
from esm2_mech.utils.seed_aggregation import read_seed_point_estimate, seed_result_contract


def _synthetic_data(n_genes=200, dim=32, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_genes, dim).astype(np.float32)

    labels = []
    per_class = n_genes // len(ENZYME_CLASSES)
    for cls in ENZYME_CLASSES:
        labels.extend([cls] * per_class)
    labels.extend([ENZYME_CLASSES[0]] * (n_genes - len(labels)))
    rng.shuffle(labels)

    le = LabelEncoder()
    le.fit(ENZYME_CLASSES)
    y = np.asarray(labels)

    genes = [f"GENE{i}" for i in range(n_genes)]
    pfam_map = {f"GENE{i}": f"PF{i % 40:04d}" for i in range(n_genes)}

    return X, y, genes, pfam_map, le


def test_enzyme_cv_uses_shared_probe_runners():
    assert not hasattr(enzyme_classification, "_run_cv")
    assert hasattr(enzyme_classification, "run_logreg_cv")
    assert hasattr(enzyme_classification, "run_mlp_cv")


def _write_mechanism_seed(tmp_path, recorded_seed=0):
    result = {
        **seed_result_contract(recorded_seed),
        "analysis_run_id": "run-1",
        "input_fingerprints": {"labeled_variants": "variants-1"},
        "analysis_parameters": {"n_folds": 5},
        "family_split": {
            "delta_mean": {"status": "success", "macro_f1_mean": 0.4}
        },
    }
    (tmp_path / "family_split_baselines_seed0.json").write_text(json.dumps(result))


def test_enzyme_reader_accepts_current_seed_result(tmp_path, monkeypatch):
    _write_mechanism_seed(tmp_path)
    monkeypatch.setattr(enzyme_classification, "RESULTS_DIR", tmp_path)

    records = enzyme_classification._load_mechanism_seed_records([0])

    assert records[0]["seed"] == 0
    assert records[0]["mechanism"]["macro_f1_mean"] == pytest.approx(0.4)


def test_enzyme_reader_reports_missing_seed_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(enzyme_classification, "RESULTS_DIR", tmp_path)

    records = enzyme_classification._load_mechanism_seed_records([3])

    assert records == []
    output = capsys.readouterr().out
    assert "mechanism seed 3 result file is missing" in output
    assert str(tmp_path / "family_split_baselines_seed3.json") in output


def test_enzyme_reader_rejects_wrong_seed_identity(tmp_path, monkeypatch):
    _write_mechanism_seed(tmp_path, recorded_seed=1)
    monkeypatch.setattr(enzyme_classification, "RESULTS_DIR", tmp_path)

    with pytest.raises(ValueError, match="declares seed 1"):
        enzyme_classification._load_mechanism_seed_records([0])


def test_mechanism_oof_reader_aligns_by_seed_and_row(monkeypatch):
    row_ids = np.arange(10)
    labels = np.resize(np.array(["LOF", "GOF", "DN"], dtype=object), 10)
    genes = np.array([f"G{row}" for row in row_ids], dtype=object)
    predictions = np.roll(labels, 1)
    folds = np.tile(np.arange(5), 2)

    def load_seed(seed):
        order = row_ids if seed == 0 else row_ids[::-1]
        return {
            "row_ids": row_ids[order],
            "y_true": labels[order],
            "pred": predictions[order],
            "genes": genes[order],
            "folds": folds[order],
        }

    monkeypatch.setattr(
        enzyme_classification,
        "_load_mechanism_family_oof_for_seed",
        load_seed,
    )

    observed, observed_genes, arms = (
        enzyme_classification.load_mechanism_family_oof_arms([0, 1])
    )

    assert np.array_equal(observed, labels)
    assert np.array_equal(observed_genes, genes)
    assert len(arms) == 2
    assert all(
        np.array_equal(predicted, predictions)
        for predicted, _folds, _ids in arms
    )


def _fingerprint_inputs():
    return {
        "X_emb": np.arange(12, dtype=np.float32).reshape(3, 4),
        "genes": ["G1", "G2", "G3"],
        "uniprot_ids": ["P1", "P2", "P3"],
        "labels": np.array(["kinase", "protease", "non-enzyme"]),
        "pfam_map": {"G1": "PF1", "G2": "PF2", "G3": None},
        "X_proteome": np.arange(6, dtype=np.float32).reshape(3, 2),
        "proteome_genes": ["G1", "G2", "G3"],
        "proteome_labels": np.array(["kinase", "protease", "non-enzyme"]),
        "proteome_columns": ["feature_a", "feature_b"],
        "mechanism_reference": {
            "content": "mechanism-content-1",
            "analysis_run_id": "run-1",
            "input_fingerprints": {"labeled_variants": "variants-1"},
            "analysis_parameters": {"n_folds": 5},
        },
    }


def test_enzyme_input_fingerprints_cover_every_scientific_input():
    inputs = _fingerprint_inputs()
    baseline = enzyme_input_fingerprints(**inputs)

    changed_labels = _fingerprint_inputs()
    changed_labels["labels"][0] = "oxidoreductase"
    assert enzyme_input_fingerprints(**changed_labels)["enzyme_labeled_genes"] != baseline[
        "enzyme_labeled_genes"
    ]

    changed_embedding = _fingerprint_inputs()
    changed_embedding["X_emb"][0, 0] += 1
    assert enzyme_input_fingerprints(**changed_embedding)["wt_embedding_content"] != baseline[
        "wt_embedding_content"
    ]

    changed_pfam = _fingerprint_inputs()
    changed_pfam["pfam_map"]["G1"] = "PF9"
    assert enzyme_input_fingerprints(**changed_pfam)["pfam_assignments"] != baseline[
        "pfam_assignments"
    ]

    changed_proteome = _fingerprint_inputs()
    changed_proteome["X_proteome"][0, 0] += 1
    assert enzyme_input_fingerprints(**changed_proteome)[
        "proteome_feature_content"
    ] != baseline["proteome_feature_content"]

    changed_columns = _fingerprint_inputs()
    changed_columns["proteome_columns"][0] = "different_feature"
    assert enzyme_input_fingerprints(**changed_columns)[
        "proteome_feature_columns"
    ] != baseline["proteome_feature_columns"]

    changed_mechanism = _fingerprint_inputs()
    changed_mechanism["mechanism_reference"]["content"] = "mechanism-content-2"
    assert enzyme_input_fingerprints(**changed_mechanism)[
        "mechanism_reference"
    ] != baseline["mechanism_reference"]


def test_enzyme_input_fingerprints_reject_misaligned_rows():
    inputs = _fingerprint_inputs()
    inputs["uniprot_ids"] = ["P1", "P2"]
    with pytest.raises(ValueError, match="embedding inputs are misaligned"):
        enzyme_input_fingerprints(**inputs)


class TestSeedAggregation:

    def test_family_scores_use_shared_seed_aggregates(self):
        X, y, genes, pfam_map, le = _synthetic_data()
        result = run_multiseed(
            X, y, genes, pfam_map, le, seeds=[0], n_folds=3,
        )
        for arm in ("logreg_family_split", "mlp_family_split"):
            metric = read_seed_point_estimate(
                result[arm]["macro_f1_seed_aggregate"]
            )
            assert metric.available
            assert metric.spread is None
        paired = read_seed_point_estimate(
            result["paired_mlp_minus_logreg_seed_aggregate"]
        )
        assert paired.available
        missing_reference = result["paired_logreg_minus_mechanism_seed_aggregate"]
        assert missing_reference["state"] == "unavailable"
        assert missing_reference["reason"] == "missing_seed"


class TestIntervals:

    def test_single_seed_interval_is_computed_and_contains_its_own_point(self):
        X, y, genes, pfam_map, le = _synthetic_data()
        result = run_multiseed(
            X, y, genes, pfam_map, le, seeds=[0], n_folds=3, n_boot=40,
        )
        interval = result["bootstrap_ci"]["macro_f1"]
        assert interval["point"] is not None
        if not interval["ci_suppressed"]:
            assert interval["ci_low"] <= interval["point"] <= interval["ci_high"]

    def test_multiseed_interval_targets_the_across_seed_point(self):
        X, y, genes, pfam_map, le = _synthetic_data(n_genes=120, dim=8)
        requested_seeds = [0, 1, 2]
        result = run_multiseed(
            X,
            y,
            genes,
            pfam_map,
            le,
            seeds=requested_seeds,
            n_folds=3,
            n_boot=20,
        )
        aggregate = read_seed_point_estimate(
            result["logreg_family_split"]["macro_f1_seed_aggregate"]
        )
        interval = result["bootstrap_ci"]["macro_f1"]

        assert aggregate.available
        assert interval["point"] == pytest.approx(aggregate.value)

    def test_enzyme_mechanism_interval_uses_shared_families(self):
        X, y, genes, pfam_map, le = _synthetic_data(n_genes=120, dim=8)
        # The mechanism cohort deliberately half-overlaps the enzyme cohort: 20
        # genes whose families the enzyme side also has, plus 20 genes carrying
        # families it does not. The shared count must therefore be smaller than
        # either cohort's own family count, which it would not be if the code
        # took one side's families wholesale.
        overlapping_genes = genes[:20]
        disjoint_genes = [f"XGENE{i}" for i in range(20)]
        pfam_map.update({gene: f"PFX{i:04d}" for i, gene in enumerate(disjoint_genes)})
        mechanism_genes = np.array(overlapping_genes + disjoint_genes, dtype=object)
        mechanism_labels = np.resize(
            np.array(["LOF", "GOF", "DN"], dtype=object),
            len(mechanism_genes),
        )
        mechanism_predictions = mechanism_labels.copy()
        mechanism_oof = {
            "y_true": mechanism_labels,
            "pred": mechanism_predictions,
            "genes": mechanism_genes,
            "folds": np.arange(len(mechanism_genes)) % 3,
        }
        mechanism_arms = (
            mechanism_labels,
            mechanism_genes,
            [
                (
                    mechanism_predictions,
                    mechanism_oof["folds"],
                    np.unique(mechanism_oof["folds"]),
                )
            ],
        )

        result = run_multiseed(
            X,
            y,
            genes,
            pfam_map,
            le,
            seeds=[0],
            n_folds=3,
            n_boot=20,
            mechanism_family_arms=mechanism_arms,
        )

        interval = result["paired_ci_logreg_minus_mechanism"]
        enzyme_families = {pfam_map[gene] for gene in genes}
        mechanism_families = {pfam_map[gene] for gene in mechanism_genes}
        shared_families = enzyme_families & mechanism_families
        # Guard the guard: if either side were a subset of the other this
        # assertion would pass for a wholesale implementation too.
        assert 0 < len(shared_families) < len(enzyme_families)
        assert len(shared_families) < len(mechanism_families)
        assert interval["n_clusters_shared"] == len(shared_families)

    def test_no_interval_is_produced_when_intervals_are_switched_off(self):
        X, y, genes, pfam_map, le = _synthetic_data()
        result = run_multiseed(
            X, y, genes, pfam_map, le, seeds=[0], n_folds=3, compute_ci=False,
        )
        assert "bootstrap_ci" not in result
