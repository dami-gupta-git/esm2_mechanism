"""
Tests for the log-likelihood scan's pure logic.

A scanned position whose wild-type residue is outside the standard alphabet has
no log-likelihood, and is recorded as NaN. Two reductions previously ignored
that: a comparison against NaN is False, so unscoreable positions were counted
as non-hotspots in the denominator, and numpy sorts NaN last, so those same
positions were selected as the "most perturbed" ones whose entropies became a
feature. Both are pinned below.

Covers:
- load_probe_list: one entry per (gene, position), the first occurrence winning
- load_probe_list: a position repeated across probes does not multiply
- compute_ll_features: one row per gene, row-aligned to the gene list
- compute_ll_features: a gene with too few scoreable positions is dropped
- compute_ll_features: a dropped gene is reported, not silently omitted
- compute_ll_features: unscoreable positions are excluded from the hotspot fraction
- compute_ll_features: unscoreable positions are never selected as most-perturbed
- compute_ll_features: features are finite when at least one position is scoreable
- compute_ll_features: a gene absent from the scores is dropped
"""

import json

import numpy as np
import pytest

from esm2_mech.experiments.perturbation.ll_scan import (
    MIN_POSITIONS,
    compute_ll_features,
    load_probe_list,
)

N_AA = 20


def _score(aa_pos, ll_wt, probe_ll=0.0, entropy_marker=1.0):
    """One scanned position. `entropy_marker` scales the residue distribution so
    a test can tell which positions the entropy feature was taken from."""
    probs = np.full(N_AA, entropy_marker, dtype=float)
    return {
        "aa_pos": aa_pos,
        "wt_aa": "A",
        "ll_wt": ll_wt,
        "ll_ala": probe_ll,
        "ll_asp": probe_ll,
        "ll_trp": probe_ll,
        "full_probs": probs.tolist(),
    }


def _feature(gene_list, X, gene, name, feature_names):
    return float(X[list(gene_list).index(gene), feature_names.index(name)])


# ---------------------------------------------------------------------------
# load_probe_list
# ---------------------------------------------------------------------------


def _write_probe_cache(tmp_path, monkeypatch, probes, covered_genes):
    from esm2_mech.experiments.perturbation import ll_scan

    path = tmp_path / "scan_probes.json"
    path.write_text(json.dumps({"probes": probes, "covered_genes": covered_genes}))
    monkeypatch.setattr(ll_scan, "SCAN_PROBE_CACHE_JSON", path)


def _probe(gene="G1", aa_pos=1, aa_wt="A", uniprot_id="P1", seq_len=10):
    return {
        "gene": gene,
        "aa_pos": aa_pos,
        "aa_wt": aa_wt,
        "uniprot_id": uniprot_id,
        "seq_len": seq_len,
    }


def test_probe_list_has_one_entry_per_gene_position(tmp_path, monkeypatch):
    _write_probe_cache(
        tmp_path, monkeypatch, [_probe(aa_pos=1), _probe(aa_pos=2)], ["G1"]
    )
    covered, positions = load_probe_list()
    assert covered == ["G1"]
    assert sorted(positions["G1"]) == [1, 2]


def test_a_repeated_position_keeps_the_first_probe(tmp_path, monkeypatch):
    """The same position appears once per probe residue; it must not multiply."""
    _write_probe_cache(
        tmp_path,
        monkeypatch,
        [_probe(aa_pos=1, aa_wt="A"), _probe(aa_pos=1, aa_wt="W")],
        ["G1"],
    )
    _covered, positions = load_probe_list()
    assert list(positions["G1"]) == [1]
    assert positions["G1"][1]["wt_aa"] == "A"


# ---------------------------------------------------------------------------
# compute_ll_features
# ---------------------------------------------------------------------------


def test_one_row_per_gene_aligned_to_the_gene_list():
    scores = {
        "G1": [_score(i, ll_wt=1.0) for i in range(MIN_POSITIONS)],
        "G2": [_score(i, ll_wt=2.0) for i in range(MIN_POSITIONS)],
    }
    gene_list, X, feature_names = compute_ll_features(["G1", "G2"], scores)
    assert len(gene_list) == X.shape[0] == 2
    assert X.shape[1] == len(feature_names)


def test_a_gene_with_too_few_scoreable_positions_is_dropped():
    scores = {"G1": [_score(i, ll_wt=1.0) for i in range(MIN_POSITIONS - 1)]}
    gene_list, X, _ = compute_ll_features(["G1"], scores)
    assert len(gene_list) == 0
    assert X.shape[0] == 0


def test_a_gene_whose_positions_are_all_unscoreable_is_dropped(capsys):
    """Enough positions, but none with a log-likelihood, is still not enough."""
    scores = {"G1": [_score(i, ll_wt=float("nan")) for i in range(MIN_POSITIONS + 2)]}
    gene_list, _X, _ = compute_ll_features(["G1"], scores)
    assert len(gene_list) == 0
    assert "dropped" in capsys.readouterr().out


def test_unscoreable_positions_are_excluded_from_the_hotspot_fraction():
    """A NaN compares False against the threshold, so counting it in the
    denominator would dilute the fraction toward zero."""
    observed = [_score(0, ll_wt=10.0), _score(1, ll_wt=1.0), _score(2, ll_wt=1.0)]
    unscoreable = [_score(i, ll_wt=float("nan")) for i in range(3, 20)]

    gene_list, X, names = compute_ll_features(["G1"], {"G1": observed})
    with_nan_list, with_nan, _ = compute_ll_features(
        ["G1"], {"G1": observed + unscoreable}
    )

    assert _feature(gene_list, X, "G1", "ll_hotspot_frac", names) == pytest.approx(
        _feature(with_nan_list, with_nan, "G1", "ll_hotspot_frac", names)
    )


def test_unscoreable_positions_are_never_the_most_perturbed():
    """numpy sorts NaN last, so the top-10 selection would otherwise prefer
    exactly the positions that could not be scored."""
    observed = [
        _score(i, ll_wt=float(i), entropy_marker=1.0) for i in range(MIN_POSITIONS)
    ]
    # A distinct distribution marks the unscoreable positions; if any were
    # selected, the entropy feature would move.
    unscoreable = [
        _score(i, ll_wt=float("nan"), entropy_marker=7.0) for i in range(10, 30)
    ]

    gene_list, X, names = compute_ll_features(["G1"], {"G1": observed})
    with_nan_list, with_nan, _ = compute_ll_features(
        ["G1"], {"G1": observed + unscoreable}
    )

    assert _feature(gene_list, X, "G1", "ll_top_entropy", names) == pytest.approx(
        _feature(with_nan_list, with_nan, "G1", "ll_top_entropy", names)
    )


def test_features_are_finite_when_any_position_is_scoreable():
    scores = {
        "G1": [_score(i, ll_wt=float(i)) for i in range(MIN_POSITIONS)]
        + [_score(9, ll_wt=float("nan"))]
    }
    _gene_list, X, _ = compute_ll_features(["G1"], scores)
    assert np.isfinite(X).all()


def test_a_gene_absent_from_the_scores_is_dropped():
    gene_list, _X, _ = compute_ll_features(["G_MISSING"], {})
    assert len(gene_list) == 0
