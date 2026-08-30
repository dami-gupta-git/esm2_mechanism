import numpy as np
import pandas as pd

from esm2_mech.experiments.badonyi import badonyi_holdout_survival as survival


def test_fold_summary_requires_every_requested_fold(monkeypatch):
    frame = pd.DataFrame({"family": ["A", "A", "B", "B"]})
    monkeypatch.setattr(
        survival,
        "assign_folds",
        lambda *_args, **_kwargs: np.array([0, 0, 1, 1]),
    )
    responses = iter(
        (
            {"DN_vs_LOF": 0.8, "GOF_vs_LOF": 0.7, "LOF_vs_nonLOF": 0.6},
            {"DN_vs_LOF": None, "GOF_vs_LOF": 0.9, "LOF_vs_nonLOF": 0.8},
            {"DN_vs_LOF": 0.7, "GOF_vs_LOF": 0.8, "LOF_vs_nonLOF": 0.7},
        )
    )
    monkeypatch.setattr(
        survival, "compute_aurocs", lambda *_args, **_kwargs: next(responses)
    )

    result = survival.run_holdout(
        frame, "family", n_folds=2, seed=0, compute_ci=False
    )["fold_mean_aurocs"]

    assert result["DN_vs_LOF_fold_mean"] is None
    assert result["DN_vs_LOF_n_folds_valid"] == 1
    assert result["GOF_vs_LOF_fold_mean"] == 0.8
    assert result["GOF_vs_LOF_n_folds_valid"] == 2
