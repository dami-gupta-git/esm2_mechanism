"""Regression tests for the mechanism nonlinear-probe orchestration.

The family-split MLP must use Pfam family IDs, rather than gene IDs, as its
early-stopping validation groups.
"""

from types import SimpleNamespace

import numpy as np

from esm2_mech.experiments.mechanism import mlp
from esm2_mech.utils.constants import DN, GOF, LOF


def test_run_seed_passes_family_groups_to_family_split_mlp(tmp_path, monkeypatch):
    genes = np.array(["G1", "G2", "G3", "G4", "G5", "G6"])
    labels = np.array([GOF, DN, LOF, GOF, DN, LOF])
    delta_mean = np.zeros((len(genes), 3), dtype=np.float32)
    delta_pos = np.ones((len(genes), 3), dtype=np.float32)
    family_groups = np.array(["PF1", "PF1", "PF2", "PF2", "PF3", "PF3"])
    split = [(np.array([0, 1, 2, 3]), np.array([4, 5]))]
    observed_validation_groups = []

    monkeypatch.setattr(mlp, "gene_split_cv", lambda *args, **kwargs: split)
    monkeypatch.setattr(mlp, "family_split_cv", lambda *args, **kwargs: split)
    monkeypatch.setattr(
        mlp,
        "family_or_gene_clusters",
        lambda *args, **kwargs: family_groups,
    )

    def fake_mlp_probe(*args, **kwargs):
        observed_validation_groups.append(kwargs["validation_groups"])
        return {}, None

    def fake_sklearn_probe(*args, **kwargs):
        return {}, None

    monkeypatch.setattr(mlp, "run_mlp_probe_cv", fake_mlp_probe)
    monkeypatch.setattr(mlp, "run_sklearn_probe_pca", fake_sklearn_probe)
    monkeypatch.setattr(mlp, "run_sklearn_probe", fake_sklearn_probe)
    monkeypatch.setattr(mlp, "write_result_json", lambda *args, **kwargs: None)

    args = SimpleNamespace(
        out_dir=str(tmp_path),
        no_ci=True,
        n_boot=10,
        only_new_family_arms=False,
        max_epochs=1,
        patience=1,
    )
    mlp.run_seed(
        0,
        args,
        labels,
        genes,
        delta_mean,
        delta_pos,
        {"G1": "PF1"},
        {"test": "fingerprint"},
    )

    assert len(observed_validation_groups) == 4
    np.testing.assert_array_equal(observed_validation_groups[0], genes)
    np.testing.assert_array_equal(observed_validation_groups[1], family_groups)
    np.testing.assert_array_equal(observed_validation_groups[2], genes)
    np.testing.assert_array_equal(observed_validation_groups[3], family_groups)
