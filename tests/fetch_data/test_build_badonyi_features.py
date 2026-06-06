"""
Tests for esm2_mech.fetch_data.build_badonyi_features.compute_family_residuals.

This is the data-integrity-critical step: family-mean-centred residuals with a
missingness flag. Per the project rules, a flag and the computed value it
describes must be derived from the same condition — here:

  - residual = value - observed_family_mean, and _familyresid_missing = 0,
    ONLY for genes in a family with >= 2 OBSERVED members.
  - otherwise residual = NaN and _familyresid_missing = 1.
  - is_singleton_family_badonyi must match the exact condition that produces
    a NaN residual (no Pfam entry, or <= 1 observed member in the family).

These tests pin those equivalences so the flag can never silently diverge from
the residual it is supposed to describe.
"""

import numpy as np
import pandas as pd
import pytest

from esm2_mech.fetch_data.build_badonyi_features import compute_family_residuals

FEATURE_COLS = ["pDN"]


def _df(rows):
    """rows: list of (gene, pDN). Returns a DataFrame with a 'gene' column."""
    return pd.DataFrame(rows, columns=["gene", "pDN"])


def test_residual_and_flag_agree_everywhere():
    # Any row with a NaN residual must have _familyresid_missing == 1, and vice versa.
    df = _df([("A", 0.2), ("B", 0.4), ("C", 0.9), ("LONE", 0.5)])
    pfam = {"A": "F1", "B": "F1", "C": "F1", "LONE": "F2"}
    out = compute_family_residuals(df, pfam, FEATURE_COLS)

    resid = out["pDN_familyresid"]
    flag = out["pDN_familyresid_missing"]
    nan_rows = resid.isna()
    assert ((flag == 1) == nan_rows).all()
    assert ((flag == 0) == ~nan_rows).all()


def test_family_with_two_plus_members_gets_centred_residual():
    df = _df([("A", 0.2), ("B", 0.4)])
    pfam = {"A": "F1", "B": "F1"}
    out = compute_family_residuals(df, pfam, FEATURE_COLS).set_index("gene")
    mean = (0.2 + 0.4) / 2
    assert out.loc["A", "pDN_familyresid"] == pytest.approx(0.2 - mean)
    assert out.loc["B", "pDN_familyresid"] == pytest.approx(0.4 - mean)
    assert (out["pDN_familyresid_missing"] == 0).all()
    # residuals within a family sum to zero
    assert out["pDN_familyresid"].sum() == pytest.approx(0.0)


def test_singleton_family_gets_nan_residual():
    df = _df([("A", 0.2), ("B", 0.4), ("LONE", 0.9)])
    pfam = {"A": "F1", "B": "F1", "LONE": "F2"}
    out = compute_family_residuals(df, pfam, FEATURE_COLS).set_index("gene")
    assert np.isnan(out.loc["LONE", "pDN_familyresid"])
    assert out.loc["LONE", "pDN_familyresid_missing"] == 1
    assert out.loc["LONE", "is_singleton_family_badonyi"] == 1


def test_gene_with_no_pfam_entry_is_singleton():
    df = _df([("A", 0.2), ("B", 0.4), ("NOFAM", 0.5)])
    pfam = {"A": "F1", "B": "F1"}  # NOFAM absent → maps to NaN family
    out = compute_family_residuals(df, pfam, FEATURE_COLS).set_index("gene")
    assert out.loc["NOFAM", "is_singleton_family_badonyi"] == 1
    assert np.isnan(out.loc["NOFAM", "pDN_familyresid"])
    assert out.loc["NOFAM", "pDN_familyresid_missing"] == 1


def test_singleton_flag_matches_nan_residual_condition():
    # The singleton flag must mark exactly the rows whose residual is NaN.
    df = _df([("A", 0.2), ("B", 0.4), ("C", 0.6), ("LONE", 0.9), ("NOFAM", 0.1)])
    pfam = {"A": "F1", "B": "F1", "C": "F1", "LONE": "F2"}  # NOFAM has no entry
    out = compute_family_residuals(df, pfam, FEATURE_COLS)
    nan_rows = out["pDN_familyresid"].isna()
    singleton = out["is_singleton_family_badonyi"] == 1
    assert (nan_rows == singleton).all()


def test_observed_mask_excludes_imputed_from_family_mean():
    # An imputed gene (observed_mask False) must not contaminate the family mean,
    # and a family with only one OBSERVED member must yield NaN residuals.
    df = _df([("A", 0.2), ("B_imputed", 99.0)])
    pfam = {"A": "F1", "B_imputed": "F1"}
    observed = pd.Series([True, False], index=df.index)
    out = compute_family_residuals(df, pfam, FEATURE_COLS, observed_mask=observed)
    # Only one observed member (A) → family has <= 1 observed → all NaN residuals.
    assert out["pDN_familyresid"].isna().all()
    assert (out["pDN_familyresid_missing"] == 1).all()
    assert (out["is_singleton_family_badonyi"] == 1).all()


def test_observed_mask_two_observed_uses_observed_mean_only():
    # Two observed members + one imputed: residuals exist and the mean is over
    # the two observed values only (the imputed 99.0 is excluded).
    df = _df([("A", 0.2), ("B", 0.4), ("C_imputed", 99.0)])
    pfam = {"A": "F1", "B": "F1", "C_imputed": "F1"}
    observed = pd.Series([True, True, False], index=df.index)
    out = compute_family_residuals(df, pfam, FEATURE_COLS, observed_mask=observed).set_index("gene")
    observed_mean = (0.2 + 0.4) / 2
    assert out.loc["A", "pDN_familyresid"] == pytest.approx(0.2 - observed_mean)
    assert out.loc["B", "pDN_familyresid"] == pytest.approx(0.4 - observed_mean)
    # The imputed gene gets a residual relative to the observed mean, but it is
    # NOT itself observed; the function assigns residuals to the whole family.
    assert out.loc["C_imputed", "pDN_familyresid"] == pytest.approx(99.0 - observed_mean)
    # All three are in a family with >= 2 observed members → not singletons.
    assert (out["is_singleton_family_badonyi"] == 0).all()
