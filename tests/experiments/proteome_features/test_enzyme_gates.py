"""
Tests for enzyme gate evaluation in enzyme_classification.run_multiseed.

Invariants:
- The enzyme module has no private CV loop; LogReg and MLP delegate to the shared
  probe runners.
- The gate point estimates (fs_f1, mlp_f1) come from the seed-0 out-of-fold
  predictions, scored within each fold and averaged — the same basis the
  bootstrap CI uses, so the point estimate matches what the CI is attached to.
- When CIs are computed, a paired CI on MLP minus LogReg is present
  (paired_ci_mlp_minus_logreg key). Without it the preregistered decision
  rule cannot be applied.
- The seed-0 OOF macro-F1 is stored in both logreg_family_split and
  mlp_family_split result dicts, under oof_macro_f1.
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
from esm2_mech.utils.constants import MECHANISM_OOF_CACHE_SCHEMA_VERSION


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


def _write_mechanism_seed_and_cache(tmp_path, cache_run_id="run-1"):
    result = {
        "analysis_run_id": "run-1",
        "input_fingerprints": {"labeled_variants": "variants-1"},
        "analysis_parameters": {"n_folds": 5},
    }
    cache = {
        "cache_schema_version": MECHANISM_OOF_CACHE_SCHEMA_VERSION,
        "seed": 0,
        "analysis_run_id": cache_run_id,
        "input_fingerprints": result["input_fingerprints"],
        "analysis_parameters": result["analysis_parameters"],
        "features": {
            "delta_mean": {
                "family_split": {
                    "row_ids": [0, 1, 2],
                    "y_true": ["GOF", "DN", "LOF"],
                    "pred": ["GOF", "DN", "LOF"],
                    "genes": ["G1", "G2", "G3"],
                    "folds": [0, 1, 2],
                }
            }
        },
    }
    (tmp_path / "family_split_baselines_seed0.json").write_text(json.dumps(result))
    (tmp_path / "mechanism_oof_cache_seed0.json").write_text(json.dumps(cache))


def test_enzyme_reader_accepts_cache_bound_to_seed_result(tmp_path, monkeypatch):
    _write_mechanism_seed_and_cache(tmp_path)
    monkeypatch.setattr(enzyme_classification, "RESULTS_DIR", tmp_path)

    oof = enzyme_classification._load_mechanism_family_oof()

    assert oof["row_ids"] == [0, 1, 2]


def test_enzyme_reader_rejects_cache_from_another_execution(tmp_path, monkeypatch):
    _write_mechanism_seed_and_cache(tmp_path, cache_run_id="run-2")
    monkeypatch.setattr(enzyme_classification, "RESULTS_DIR", tmp_path)

    with pytest.raises(ValueError, match="analysis_run_id"):
        enzyme_classification._load_mechanism_family_oof()


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


class TestOofMacroF1Consistency:

    def test_oof_macro_f1_present(self):
        """run_multiseed should include oof_macro_f1 in both logreg and mlp results."""
        X, y, genes, pfam_map, le = _synthetic_data()
        result = run_multiseed(
            X, y, genes, pfam_map, le, seeds=[0], n_folds=3,
            compute_ci=False, n_boot=50,
        )
        assert "oof_macro_f1" in result["logreg_family_split"]
        assert "oof_macro_f1" in result["mlp_family_split"]

    def test_oof_macro_f1_equals_the_fold_mean_on_a_single_seed(self):
        """With one seed, the OOF score and the reported fold mean are the same
        average over the same folds, so they must agree to floating point.

        This is what distinguishes the two computations. Scoring the concatenated
        out-of-fold predictions as one block gives a different number on imbalanced
        data, so a return to pooling breaks this equality.
        """
        X, y, genes, pfam_map, le = _synthetic_data()
        result = run_multiseed(
            X, y, genes, pfam_map, le, seeds=[0], n_folds=3,
            compute_ci=False, n_boot=50,
        )
        for arm in ("logreg_family_split", "mlp_family_split"):
            oof_f1 = result[arm]["oof_macro_f1"]
            fold_mean = result[arm]["macro_f1_mean"]
            assert oof_f1 is not None
            assert fold_mean is not None
            assert oof_f1 == pytest.approx(fold_mean)


class TestPairedCiPresence:

    def test_paired_ci_computed_when_ci_enabled(self):
        """When compute_ci=True, paired_ci_mlp_minus_logreg must be present."""
        X, y, genes, pfam_map, le = _synthetic_data()
        result = run_multiseed(
            X, y, genes, pfam_map, le, seeds=[0], n_folds=3,
            compute_ci=True, n_boot=50,
        )
        assert "paired_ci_mlp_minus_logreg" in result, (
            "Gate 2H requires a paired CI on MLP-LogReg difference"
        )
        ci = result["paired_ci_mlp_minus_logreg"]
        assert ci is not None
        assert "ci_low" in ci or "ci_suppressed" in ci

    def test_no_paired_ci_when_ci_disabled(self):
        """When compute_ci=False, no CI keys should be present."""
        X, y, genes, pfam_map, le = _synthetic_data()
        result = run_multiseed(
            X, y, genes, pfam_map, le, seeds=[0], n_folds=3,
            compute_ci=False, n_boot=50,
        )
        assert "paired_ci_mlp_minus_logreg" not in result
        assert "bootstrap_ci" not in result
