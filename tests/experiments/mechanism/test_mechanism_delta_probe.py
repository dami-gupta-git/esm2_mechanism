"""
Tests for the pure stability-subspace math in
esm2_mech.experiments.mechanism.mechanism_delta_probe.

These two functions carry the highest silent-corruption risk in the module:
- project_out_subspace removes a nuisance (stability) subspace from the delta
  embeddings; a bug leaves the removed signal in, invalidating the "projected" arm.
- variance_explained_per_class feeds a pre-registered GOF/LOF asymmetry prediction.
"""

import numpy as np
import pytest

from esm2_mech.experiments.mechanism.mechanism_delta_probe import (
    VARIANCE_ASYMMETRY_THRESHOLD,
    assert_subspace_removed,
    project_out_subspace,
    standardize_once,
    subspace_in_standardized_coords,
    variance_explained_per_class,
)
from esm2_mech.utils.probes import run_logreg_cv


def test_project_out_subspace_removes_the_subspace():
    rng = np.random.RandomState(0)
    deltas = rng.randn(50, 8)
    # A 2-dimensional subspace (rows are basis vectors, as the caller passes them).
    subspace = rng.randn(2, 8)

    projected = project_out_subspace(deltas, subspace)

    # After projection the variance along the subspace must be ~0 — this is the
    # exact invariant CLAUDE.md requires when projecting a direction out.
    Q, _ = np.linalg.qr(subspace.T, mode="reduced")
    residual_along_subspace = projected.dot(Q)
    assert np.var(residual_along_subspace) < 1e-12


def test_project_out_subspace_leaves_orthogonal_part_unchanged():
    # Build deltas that live entirely in a direction orthogonal to the subspace.
    subspace = np.array([[1.0, 0.0, 0.0]])
    orthogonal_dir = np.array([0.0, 0.0, 1.0])
    deltas = np.outer(np.linspace(-1, 1, 10), orthogonal_dir)

    projected = project_out_subspace(deltas, subspace)

    # Nothing to remove: a vector orthogonal to the subspace passes through.
    assert np.allclose(projected, deltas, atol=1e-12)


def test_project_out_subspace_none_is_identity():
    deltas = np.arange(12, dtype=float).reshape(4, 3)
    assert np.array_equal(project_out_subspace(deltas, None), deltas)


def test_variance_explained_asymmetry_sign_and_flag():
    # LOF deltas lie inside the subspace (high variance explained);
    # GOF deltas lie orthogonal to it (near-zero variance explained).
    subspace = np.array([[1.0, 0.0, 0.0]])
    rng = np.random.RandomState(1)

    lof = np.zeros((20, 3))
    lof[:, 0] = rng.randn(20)          # variance along the subspace
    gof = np.zeros((20, 3))
    gof[:, 2] = rng.randn(20)          # variance orthogonal to the subspace

    deltas = np.vstack([gof, lof])
    labels = np.array(["GOF"] * 20 + ["LOF"] * 20)

    result = variance_explained_per_class(deltas, labels, subspace)

    assert result["LOF"] > 0.9   # nearly all LOF variance captured
    assert result["GOF"] < 0.1   # almost none of GOF variance captured
    # asymmetry = (LOF - GOF) / LOF, so LOF >> GOF gives a large positive value.
    assert result["gof_lof_asymmetry"] > VARIANCE_ASYMMETRY_THRESHOLD
    assert result["asymmetry_prediction_holds"] is True


def test_variance_explained_no_asymmetry_when_classes_symmetric():
    # Both classes share the same variance structure -> asymmetry ~ 0, flag False.
    subspace = np.array([[1.0, 0.0, 0.0]])
    rng = np.random.RandomState(2)
    block = np.zeros((20, 3))
    block[:, 0] = rng.randn(20)

    deltas = np.vstack([block, block])
    labels = np.array(["GOF"] * 20 + ["LOF"] * 20)

    result = variance_explained_per_class(deltas, labels, subspace)

    assert abs(result["gof_lof_asymmetry"]) < VARIANCE_ASYMMETRY_THRESHOLD
    assert result["asymmetry_prediction_holds"] is False


def test_variance_explained_none_subspace_returns_empty():
    deltas = np.zeros((10, 3))
    labels = np.array(["GOF"] * 5 + ["LOF"] * 5)
    assert variance_explained_per_class(deltas, labels, None) == {}


# ---------------------------------------------------------------------------
# Standardize-once + project ordering
# ---------------------------------------------------------------------------


def test_standardize_once_leaves_constant_columns_at_zero():
    deltas = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    standardized, scale = standardize_once(deltas)

    # Column 1 is constant: centering sends it to exactly zero and scale stays 1,
    # so no value is invented for it.
    assert np.allclose(standardized[:, 1], 0.0)
    assert scale[1] == 1.0
    assert np.isclose(standardized[:, 0].std(), 1.0)


def test_raw_subspace_is_not_removed_by_projecting_it_in_standardized_space():
    # The coordinate change matters: projecting the untransformed raw direction out
    # of standardized data leaves the raw stability coordinate varying.
    rng = np.random.RandomState(0)
    deltas = rng.randn(60, 5) * np.array([10.0, 1.0, 0.5, 3.0, 7.0])
    subspace = rng.randn(1, 5)
    subspace /= np.linalg.norm(subspace)

    standardized, scale = standardize_once(deltas)
    wrong = project_out_subspace(standardized, subspace)
    right = project_out_subspace(standardized, subspace_in_standardized_coords(subspace, scale))

    # Raw stability coordinate of each row, recovered from standardized coords.
    raw_coord_wrong = (wrong * scale[None, :]).dot(subspace[0])
    raw_coord_right = (right * scale[None, :]).dot(subspace[0])

    assert np.var(raw_coord_wrong) > 1e-3
    assert np.var(raw_coord_right) < 1e-20


def test_assert_subspace_removed_raises_when_variance_survives():
    rng = np.random.RandomState(1)
    deltas = rng.randn(40, 6)
    subspace = rng.randn(2, 6)

    assert_subspace_removed(project_out_subspace(deltas, subspace), subspace, "ok")

    with pytest.raises(ValueError, match="projection failed"):
        assert_subspace_removed(deltas, subspace, "unprojected")


def test_prescaled_probe_preserves_projection_and_default_scaler_destroys_it():
    # The bug this guards: run_logreg_cv's per-fold StandardScaler rescales each
    # column independently, putting variance back along a removed direction.
    rng = np.random.RandomState(2)
    deltas = rng.randn(80, 6) * np.array([8.0, 1.0, 0.3, 4.0, 2.0, 6.0])
    subspace = rng.randn(1, 6)

    standardized, scale = standardize_once(deltas)
    subspace_std = subspace_in_standardized_coords(subspace, scale)
    projected = project_out_subspace(standardized, subspace_std)

    from sklearn.preprocessing import StandardScaler

    direction = subspace_std[0] / np.linalg.norm(subspace_std[0])
    assert np.var(projected.dot(direction)) < 1e-20
    rescaled = StandardScaler().fit_transform(projected)
    assert np.var(rescaled.dot(direction)) > 1e-3

    # prescaled=True runs the classifier on exactly the matrix handed in.
    seen = {}
    labels = np.array(["GOF", "LOF"] * 40)
    splits = [(np.arange(0, 60), np.arange(60, 80))]

    import esm2_mech.utils.probes as probes_mod

    original_fit = probes_mod.LogisticRegression.fit

    def recording_fit(self, X, y, **kwargs):
        seen["X"] = np.asarray(X).copy()
        return original_fit(self, X, y, **kwargs)

    probes_mod.LogisticRegression.fit = recording_fit
    try:
        run_logreg_cv(
            projected, labels, splits, classes=["GOF", "LOF"], seed=0,
            label="prescaled", prescaled=True,
        )
        assert np.allclose(seen["X"], projected[splits[0][0]])
        assert np.var(seen["X"].dot(direction)) < 1e-20

        run_logreg_cv(
            projected, labels, splits, classes=["GOF", "LOF"], seed=0,
            label="scaled",
        )
        assert np.var(seen["X"].dot(direction)) > 1e-3
    finally:
        probes_mod.LogisticRegression.fit = original_fit
