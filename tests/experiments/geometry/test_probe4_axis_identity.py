"""
Tests for the context-free biochemistry features used by the pathogenicity-axis probe.

The amino-acid property tables are hand-entered and paired positionally with the
AA string, so a transposed entry would shift every feature without any error.
These tests pin each table against published reference values.

Covers:
- AA: the alphabet is the 20 standard residues in BLOSUM62 order
- BLOSUM: the matrix is symmetric
- BLOSUM: every ordered residue pair has a score
- BLOSUM: known diagonal and off-diagonal entries match BLOSUM62
- HYDRO: every residue matches the published Kyte-Doolittle scale
- CHARGE: only the ionisable residues are non-zero, with the right signs
- VOLUME: every residue matches the published residue volumes
- VOLUME: the smallest and largest residues are glycine and tryptophan
- biochem_features: returns one value per declared feature name
- biochem_features: feature values follow the declared FEAT_NAMES order
- biochem_features: an unknown wild-type residue returns None
- biochem_features: an unknown mutant residue returns None
- biochem_features: a synonymous substitution gives zero differences
- biochem_features: the absolute-difference features are sign-independent
"""

import numpy as np
import pytest

from esm2_mech.experiments.geometry.probe4_axis_identity import (
    AA,
    BLOSUM,
    CHARGE,
    FEAT_NAMES,
    HYDRO,
    VOLUME,
    biochem_features,
)


# Kyte-Doolittle hydropathy (J Mol Biol 1982;157:105-32).
REFERENCE_HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Residue volumes in cubic angstroms (Zamyatnin, Annu Rev Biophys Bioeng 1984).
REFERENCE_VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}


# ---------------------------------------------------------------------------
# alphabet and property tables
# ---------------------------------------------------------------------------


def test_alphabet_is_the_twenty_standard_residues_in_blosum_order():
    assert AA == "ARNDCQEGHILKMFPSTWYV"
    assert len(set(AA)) == 20


def test_blosum_is_symmetric():
    for wt in AA:
        for mut in AA:
            assert BLOSUM[(wt, mut)] == BLOSUM[(mut, wt)]


def test_blosum_covers_every_residue_pair():
    assert len(BLOSUM) == len(AA) ** 2


@pytest.mark.parametrize(
    "pair,expected",
    [
        (("W", "W"), 11),   # highest diagonal score
        (("C", "C"), 9),
        (("A", "A"), 4),
        (("W", "C"), -2),
        (("K", "R"), 2),    # conservative, both positively charged
        (("D", "E"), 2),    # conservative, both negatively charged
        (("G", "W"), -2),
        (("I", "V"), 3),
    ],
)
def test_blosum_entries_match_published_matrix(pair, expected):
    assert BLOSUM[pair] == expected


def test_hydropathy_matches_kyte_doolittle():
    assert HYDRO == pytest.approx(REFERENCE_HYDROPATHY)


def test_charge_is_non_zero_only_for_ionisable_residues():
    expected = {"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.5}
    assert {a: c for a, c in CHARGE.items() if c != 0} == expected
    assert set(CHARGE) == set(AA)


def test_volume_matches_published_residue_volumes():
    assert VOLUME == pytest.approx(REFERENCE_VOLUME)


def test_volume_extremes_are_glycine_and_tryptophan():
    assert min(VOLUME, key=VOLUME.get) == "G"
    assert max(VOLUME, key=VOLUME.get) == "W"


# ---------------------------------------------------------------------------
# biochem_features
# ---------------------------------------------------------------------------


def test_biochem_features_returns_one_value_per_declared_name():
    features = biochem_features("A", "V")
    assert len(features) == len(FEAT_NAMES)
    assert all(np.isfinite(value) for value in features)


def test_biochem_features_follow_the_declared_order():
    wt, mut = "D", "K"
    features = dict(zip(FEAT_NAMES, biochem_features(wt, mut)))
    assert features["blosum62"] == BLOSUM[(wt, mut)]
    assert features["d_hydro"] == pytest.approx(HYDRO[mut] - HYDRO[wt])
    assert features["abs_d_hydro"] == pytest.approx(abs(HYDRO[mut] - HYDRO[wt]))
    assert features["d_charge"] == pytest.approx(CHARGE[mut] - CHARGE[wt])
    assert features["abs_d_charge"] == pytest.approx(abs(CHARGE[mut] - CHARGE[wt]))
    assert features["abs_d_volume"] == pytest.approx(abs(VOLUME[mut] - VOLUME[wt]))


def test_biochem_features_unknown_wild_type_returns_none():
    """A non-standard residue must be reported as absent, not scored with a default."""
    assert biochem_features("X", "V") is None


def test_biochem_features_unknown_mutant_returns_none():
    assert biochem_features("A", "*") is None


def test_biochem_features_synonymous_substitution_has_zero_differences():
    features = dict(zip(FEAT_NAMES, biochem_features("A", "A")))
    assert features["d_hydro"] == 0
    assert features["d_charge"] == 0
    assert features["abs_d_volume"] == 0
    assert features["blosum62"] == BLOSUM[("A", "A")]


def test_biochem_features_absolute_differences_are_sign_independent():
    forward = dict(zip(FEAT_NAMES, biochem_features("D", "K")))
    reverse = dict(zip(FEAT_NAMES, biochem_features("K", "D")))
    for name in ("abs_d_hydro", "abs_d_charge", "abs_d_volume"):
        assert forward[name] == pytest.approx(reverse[name])
    assert forward["d_charge"] == pytest.approx(-reverse["d_charge"])
