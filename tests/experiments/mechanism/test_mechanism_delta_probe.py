"""
Tests for the pure stability-subspace math in
esm2_mech.experiments.mechanism.mechanism_delta_probe.

These two functions carry the highest silent-corruption risk in the module:
- project_out_subspace removes a nuisance (stability) subspace from the delta
  embeddings; a bug leaves the removed signal in, invalidating the "projected" arm.
- variance_explained_per_class feeds a pre-registered GOF/LOF asymmetry prediction.
"""

import numpy as np

from esm2_mech.experiments.mechanism.mechanism_delta_probe import (
    VARIANCE_ASYMMETRY_THRESHOLD,
    project_out_subspace,
    variance_explained_per_class,
)


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
