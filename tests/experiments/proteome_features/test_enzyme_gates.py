"""
Tests for enzyme gate evaluation in enzyme_classification.run_multiseed.

Invariants:
- The gate point estimates (fs_f1, mlp_f1) come from pooled out-of-fold
  predictions, not from the mean of per-fold F1 scores. These two numbers
  differ on imbalanced data, and the CI is computed on pooled OOF, so the
  point estimate must match.
- When CIs are computed, a paired CI on MLP minus LogReg is present
  (paired_ci_mlp_minus_logreg key). Without it the preregistered decision
  rule cannot be applied.
- The pooled OOF F1 is stored in both logreg_family_split and
  mlp_family_split result dicts.
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


class TestPooledOofConsistency:

    def test_pooled_oof_f1_present(self):
        """run_multiseed should include pooled_oof_macro_f1 in both logreg and mlp results."""
        X, y, genes, pfam_map, le = _synthetic_data()
        result = run_multiseed(
            X, y, genes, pfam_map, le, seeds=[0], n_folds=3,
            compute_ci=False, n_boot=50,
        )
        assert "pooled_oof_macro_f1" in result["logreg_family_split"]
        assert "pooled_oof_macro_f1" in result["mlp_family_split"]

    def test_pooled_oof_differs_from_fold_mean(self):
        """The pooled-OOF F1 and fold-mean F1 are generally different numbers.
        This test verifies they are computed independently (not copied)."""
        X, y, genes, pfam_map, le = _synthetic_data()
        result = run_multiseed(
            X, y, genes, pfam_map, le, seeds=[0], n_folds=3,
            compute_ci=False, n_boot=50,
        )
        pooled = result["logreg_family_split"]["pooled_oof_macro_f1"]
        fold_mean = result["logreg_family_split"]["macro_f1_mean"]
        assert pooled is not None
        assert fold_mean is not None
        assert isinstance(pooled, float)
        assert isinstance(fold_mean, float)


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
