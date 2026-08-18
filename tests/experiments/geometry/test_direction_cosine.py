"""
Tests for direction_geometry.probe2_universal — the cross-family direction cosine.

Invariants:
- The cosine is computed in the original (unscaled) embedding space by
  back-projecting each half's scaled coefficients through its own scaler.
  This avoids both the separate-coordinate-system problem (comparing
  coefficients from different scalers) and the test-set leakage problem
  (fitting a scaler on both halves before training on one).
- When both halves see the same linear signal, the cosine is near 1.
- When labels are shuffled, the cosine is near 0.
- The transfer AUROC uses train-only scaling (A's scaler applied to B),
  never a scaler fitted on both halves.
"""

import numpy as np

from esm2_mech.experiments.geometry.direction_geometry import (
    fit_direction,
    original_space_direction,
    probe2_universal,
)


class TestDirectionCosine:

    def test_identical_signal_gives_high_cosine(self):
        rng = np.random.RandomState(42)
        n = 400
        d = 20
        w_true = rng.randn(d)
        w_true /= np.linalg.norm(w_true)
        X = rng.randn(n, d)
        y = (X @ w_true > 0).astype(int)
        genes = np.array([f"G{i}" for i in range(n)])
        fam = np.array([f"F{i % 50}" for i in range(n)])

        result = probe2_universal(X, y, genes, fam, n_partitions=5, seeds=(0,))

        cos_mean = result["cosine_observed"][0]
        assert cos_mean > 0.7, (
            f"With a shared linear signal, cosine should be high; got {cos_mean:.3f}"
        )

    def test_shuffled_labels_give_low_cosine(self):
        rng = np.random.RandomState(42)
        n = 400
        d = 20
        X = rng.randn(n, d)
        y = rng.randint(0, 2, size=n)
        genes = np.array([f"G{i}" for i in range(n)])
        fam = np.array([f"F{i % 50}" for i in range(n)])

        result = probe2_universal(X, y, genes, fam, n_partitions=5, seeds=(0,))

        cos_null_mean = result["cosine_null_shuffled"][0]
        assert abs(cos_null_mean) < 0.5, (
            f"With random labels, null cosine should be near zero; got {cos_null_mean:.3f}"
        )

    def test_back_projection_recovers_original_space_direction(self):
        """Back-projecting scaled coefficients through the scaler should recover
        the direction in the original embedding space, not a scaler-dependent one."""
        from sklearn.preprocessing import StandardScaler

        rng = np.random.RandomState(42)
        n = 200
        d = 10
        w_true = rng.randn(d)
        w_true /= np.linalg.norm(w_true)
        X = rng.randn(n, d)
        y = (X @ w_true > 0).astype(int)

        sc = StandardScaler().fit(X)
        X_scaled = sc.transform(X)
        w_scaled, _ = fit_direction(X_scaled, y, seed=0)

        w_back = original_space_direction(w_scaled, sc)

        cos_with_truth = float(np.dot(w_back, w_true))
        assert abs(cos_with_truth) > 0.8, (
            f"Back-projected direction should align with ground truth; "
            f"cosine={cos_with_truth:.3f}"
        )

    def test_shift_invariance_via_back_projection(self):
        """When one half's features are shifted, back-projecting through each
        half's own scaler should still produce a valid cosine, because the
        scaling difference is absorbed by the back-projection."""
        from sklearn.preprocessing import StandardScaler

        rng = np.random.RandomState(42)
        n = 400
        d = 20
        w_true = rng.randn(d)
        w_true /= np.linalg.norm(w_true)
        X = rng.randn(n, d)
        y = (X @ w_true > 0).astype(int)

        X_shifted = X.copy()
        X_shifted[n // 2:] += 100.0

        sc_a = StandardScaler().fit(X_shifted[:n // 2])
        sc_b = StandardScaler().fit(X_shifted[n // 2:])
        Xa = sc_a.transform(X_shifted[:n // 2])
        Xb = sc_b.transform(X_shifted[n // 2:])
        wA, _ = fit_direction(Xa, y[:n // 2], seed=0)
        wB, _ = fit_direction(Xb, y[n // 2:], seed=0)

        wA_orig = original_space_direction(wA, sc_a)
        wB_orig = original_space_direction(wB, sc_b)
        cos_back = float(np.dot(wA_orig, wB_orig))

        cos_naive = float(np.dot(wA, wB))

        assert abs(cos_back) > abs(cos_naive) + 0.05 or abs(cos_back) > 0.7, (
            f"Back-projected cosine should be more meaningful than naive cosine "
            f"when halves have different distributions; "
            f"back-projected={cos_back:.3f}, naive={cos_naive:.3f}"
        )

    def test_back_projection_preserves_linear_scores_up_to_intercept(self):
        from sklearn.preprocessing import StandardScaler

        X = np.array([[0.0, 10.0], [1.0, 30.0], [3.0, 70.0]])
        scaler = StandardScaler().fit(X)
        scaled_direction = np.array([0.4, -0.9])
        original_direction = original_space_direction(scaled_direction, scaler)

        scaled_scores = scaler.transform(X) @ scaled_direction
        original_scores = X @ original_direction
        assert np.allclose(
            np.diff(scaled_scores),
            np.diff(original_scores) * np.linalg.norm(scaled_direction / scaler.scale_),
        )
