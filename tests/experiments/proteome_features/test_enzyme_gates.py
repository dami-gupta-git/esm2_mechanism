"""
Tests for enzyme gate evaluation in enzyme_classification.run_multiseed.

Invariants:
- The gate point estimates (fs_f1, mlp_f1) come from the seed-0 out-of-fold
  predictions, scored within each fold and averaged — the same basis the
  bootstrap CI uses, so the point estimate matches what the CI is attached to.
- When CIs are computed, a paired CI on MLP minus LogReg is present
  (paired_ci_mlp_minus_logreg key). Without it the preregistered decision
  rule cannot be applied.
- The seed-0 OOF macro-F1 is stored in both logreg_family_split and
  mlp_family_split result dicts, under oof_macro_f1.
"""

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

from esm2_mech.experiments.proteome_features.enzyme_classification import (
    ENZYME_CLASSES,
    run_multiseed,
)


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
    y = le.transform(labels)

    genes = [f"GENE{i}" for i in range(n_genes)]
    pfam_map = {f"GENE{i}": f"PF{i % 40:04d}" for i in range(n_genes)}

    return X, y, genes, pfam_map, le


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
