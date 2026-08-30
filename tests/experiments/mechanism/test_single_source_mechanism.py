"""
Tests for the majority-class floor recomputed on the single-source subset.

The floor must be recomputed on the subset because its class balance, and so
the score a most-frequent guesser achieves, differs from the merged set. It
must also use the same cross-validation setup as the probe it is compared
against, or the comparison is between two different experiments.

Covers:
- compute_subset_floor: reports the subset's own class distribution
- compute_subset_floor: scores both the gene-split and family-split floors
- compute_subset_floor: the requested fold count reaches the underlying evaluator
- compute_subset_floor: the requested seed count reaches the underlying evaluator
- compute_subset_floor: a floor that cannot be scored raises, returning no value
- compute_subset_floor: the printed floors are the values it returns
- the module's own scoring path executes end to end on a small subset
"""

import numpy as np
import pytest

from esm2_mech.experiments.mechanism import single_source_mechanism
from esm2_mech.experiments.mechanism.single_source_mechanism import compute_subset_floor
from esm2_mech.utils.constants import DN, GOF, LOF
from tests.helpers import available_seed_aggregate, unavailable_seed_aggregate

# A perfectly balanced set leaves the training-fold majority class tied, which the
# evaluator refuses to score; this cycle gives GOF a clear majority.
N_GENES = 120
N_FAMILIES = 15
CLASS_CYCLE = (GOF, GOF, DN, LOF)


@pytest.fixture
def subset():
    """Genes spread over Pfam families with a deliberately uneven class balance."""
    genes = np.array([f"G{index:03d}" for index in range(N_GENES)])
    labels = np.array(
        [CLASS_CYCLE[index % len(CLASS_CYCLE)] for index in range(N_GENES)]
    )
    pfam_map = {gene: f"PF{index % N_FAMILIES}" for index, gene in enumerate(genes)}
    return labels, genes, pfam_map


@pytest.fixture
def patched_pfam(monkeypatch, subset):
    """compute_subset_floor reads the Pfam map from disk; serve the synthetic one."""
    _labels, _genes, pfam_map = subset
    monkeypatch.setattr(
        single_source_mechanism, "load_pfam_map", lambda _path: pfam_map
    )
    return pfam_map


def test_reports_the_subset_class_distribution(subset, patched_pfam):
    labels, genes, _ = subset
    floor = compute_subset_floor(labels, genes, n_seeds=1, n_folds=5)
    assert floor["class_distribution"] == {GOF: 60, DN: 30, LOF: 30}


def test_scores_both_split_schemes(subset, patched_pfam):
    labels, genes, _ = subset
    floor = compute_subset_floor(labels, genes, n_seeds=1, n_folds=5)
    for split in ("gene_split", "family_split"):
        aggregate = floor[split]["macro_f1_seed_aggregate"]
        assert aggregate["state"] == "available"
        assert 0.0 <= aggregate["mean"] <= 1.0


def test_requested_folds_reach_the_evaluator(subset, patched_pfam, monkeypatch):
    """A floor computed under a different fold count is not comparable to the probe."""
    seen = []
    monkeypatch.setattr(
        single_source_mechanism,
        "eval_naive",
        lambda *args, **kwargs: seen.append(kwargs) or _stub_floor(),
    )
    labels, genes, _ = subset
    compute_subset_floor(labels, genes, n_seeds=3, n_folds=4)
    assert [call["n_folds"] for call in seen] == [4, 4]


def test_requested_seeds_reach_the_evaluator(subset, patched_pfam, monkeypatch):
    seen = []
    monkeypatch.setattr(
        single_source_mechanism,
        "eval_naive",
        lambda *args, **kwargs: seen.append(kwargs) or _stub_floor(),
    )
    labels, genes, _ = subset
    compute_subset_floor(labels, genes, n_seeds=3, n_folds=4)
    assert [call["n_seeds"] for call in seen] == [3, 3]


def _stub_floor(mean=0.29):
    return {"macro_f1_seed_aggregate": available_seed_aggregate(mean, seeds=(0, 1, 2))}


def _unscorable_floor():
    return {"macro_f1_seed_aggregate": unavailable_seed_aggregate(seeds=(0, 1, 2))}


def test_an_unscorable_floor_raises(subset, patched_pfam, monkeypatch):
    """No floor means no comparison; returning a plausible number would read as one."""
    monkeypatch.setattr(
        single_source_mechanism,
        "eval_naive",
        lambda *args, **kwargs: _unscorable_floor(),
    )
    labels, genes, _ = subset
    with pytest.raises(ValueError, match="floor is unavailable"):
        compute_subset_floor(labels, genes, n_seeds=3, n_folds=5)


def test_printed_floors_are_the_returned_values(
    subset, patched_pfam, monkeypatch, capsys
):
    monkeypatch.setattr(
        single_source_mechanism,
        "eval_naive",
        lambda *args, **kwargs: _stub_floor(0.375),
    )
    labels, genes, _ = subset
    floor = compute_subset_floor(labels, genes, n_seeds=3, n_folds=5)
    assert "gene-split 0.375" in capsys.readouterr().out
    assert floor["gene_split"]["macro_f1_seed_aggregate"]["mean"] == 0.375


def test_scoring_path_runs_end_to_end(subset, patched_pfam):
    """Every experiment module needs one test that executes its scoring path."""
    labels, genes, _ = subset
    floor = compute_subset_floor(labels, genes, n_seeds=2, n_folds=5)
    assert set(floor) == {"class_distribution", "gene_split", "family_split"}
