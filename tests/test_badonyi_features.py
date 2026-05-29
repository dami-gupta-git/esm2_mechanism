"""
Tests for compute_family_residuals in build_badonyi_features.py.

The function subtracts per-family means from each continuous feature column,
leaving singletons (genes whose family has only 1 member in the dataframe) at 0.
"""

import numpy as np
import pandas as pd
import pytest
from build_badonyi_features import compute_family_residuals


def make_df(genes, pDN, pGOF, pLOF):
    return pd.DataFrame({"gene": genes, "pDN": pDN, "pGOF": pGOF, "pLOF": pLOF})


FEATURE_COLS = ["pDN", "pGOF", "pLOF"]


class TestFamilyResiduals:

    def test_within_family_residuals_sum_to_zero(self):
        """Residuals within a family must sum to ~0 (they are mean-centred)."""
        genes = ["A", "B", "C", "D"]
        pfam = {"A": "FAM1", "B": "FAM1", "C": "FAM2", "D": "FAM2"}
        df = make_df(genes, pDN=[0.1, 0.3, 0.6, 0.8], pGOF=[0.2, 0.4, 0.5, 0.7], pLOF=[0.9, 0.1, 0.3, 0.7])
        out = compute_family_residuals(df, pfam, FEATURE_COLS)
        for fam in ["FAM1", "FAM2"]:
            mask = out["pfam_family"] == fam
            for col in FEATURE_COLS:
                resid_sum = out.loc[mask, f"{col}_familyresid"].sum()
                assert abs(resid_sum) < 1e-10, f"{col} residuals in {fam} don't sum to 0: {resid_sum}"

    def test_residual_values_correct(self):
        genes = ["A", "B"]
        pfam = {"A": "FAM1", "B": "FAM1"}
        df = make_df(genes, pDN=[0.2, 0.6], pGOF=[0.1, 0.9], pLOF=[0.4, 0.4])
        out = compute_family_residuals(df, pfam, FEATURE_COLS)
        # Family mean of pDN = 0.4; residuals should be -0.2 and +0.2
        np.testing.assert_allclose(out["pDN_familyresid"].values, [-0.2, 0.2], atol=1e-10)
        # pLOF mean = 0.4; both residuals = 0
        np.testing.assert_allclose(out["pLOF_familyresid"].values, [0.0, 0.0], atol=1e-10)

    def test_singleton_residual_is_zero(self):
        """Genes that are alone in their family get residual=0."""
        genes = ["A", "B", "C"]
        pfam = {"A": "FAM1", "B": "FAM1", "C": "FAM_SOLO"}
        df = make_df(genes, pDN=[0.1, 0.9, 0.5], pGOF=[0.2, 0.8, 0.5], pLOF=[0.3, 0.7, 0.5])
        out = compute_family_residuals(df, pfam, FEATURE_COLS)
        solo_row = out[out["gene"] == "C"]
        for col in FEATURE_COLS:
            assert solo_row[f"{col}_familyresid"].values[0] == 0.0

    def test_singleton_flag_set_correctly(self):
        genes = ["A", "B", "C"]
        pfam = {"A": "FAM1", "B": "FAM1", "C": "FAM_SOLO"}
        df = make_df(genes, pDN=[0.1, 0.9, 0.5], pGOF=[0.2, 0.8, 0.5], pLOF=[0.3, 0.7, 0.5])
        out = compute_family_residuals(df, pfam, FEATURE_COLS)
        flags = dict(zip(out["gene"], out["is_singleton_family_badonyi"]))
        assert flags["A"] == 0
        assert flags["B"] == 0
        assert flags["C"] == 1

    def test_gene_without_pfam_entry_is_singleton(self):
        """Genes missing from pfam are treated as singletons (residual=0, flag=1)."""
        genes = ["A", "B", "UNKNOWN"]
        pfam = {"A": "FAM1", "B": "FAM1"}  # UNKNOWN not in pfam
        df = make_df(genes, pDN=[0.1, 0.9, 0.5], pGOF=[0.2, 0.8, 0.5], pLOF=[0.3, 0.7, 0.5])
        out = compute_family_residuals(df, pfam, FEATURE_COLS)
        unk_row = out[out["gene"] == "UNKNOWN"]
        assert unk_row["is_singleton_family_badonyi"].values[0] == 1
        for col in FEATURE_COLS:
            assert unk_row[f"{col}_familyresid"].values[0] == 0.0

    def test_original_values_unchanged(self):
        """The function must not modify the input feature columns in-place."""
        genes = ["A", "B"]
        pfam = {"A": "FAM1", "B": "FAM1"}
        original_pDN = [0.2, 0.6]
        df = make_df(genes, pDN=original_pDN, pGOF=[0.1, 0.9], pLOF=[0.4, 0.4])
        out = compute_family_residuals(df, pfam, FEATURE_COLS)
        np.testing.assert_array_equal(out["pDN"].values, original_pDN)

    def test_input_dataframe_not_mutated(self):
        """The original df passed in should not be modified (function works on a copy)."""
        genes = ["A", "B"]
        pfam = {"A": "FAM1", "B": "FAM1"}
        df = make_df(genes, pDN=[0.2, 0.6], pGOF=[0.1, 0.9], pLOF=[0.4, 0.4])
        original_cols = set(df.columns)
        compute_family_residuals(df, pfam, FEATURE_COLS)
        assert set(df.columns) == original_cols, "Input df was modified"

    def test_multiple_families(self):
        """Three families; each should be independently mean-centred."""
        genes = ["A", "B", "C", "D", "E", "F"]
        pfam = {"A": "F1", "B": "F1", "C": "F2", "D": "F2", "E": "F3", "F": "F3"}
        pDN = [0.1, 0.3, 0.5, 0.7, 0.2, 0.8]
        df = make_df(genes, pDN=pDN, pGOF=[0.5] * 6, pLOF=[0.5] * 6)
        out = compute_family_residuals(df, pfam, FEATURE_COLS)
        for fam in ["F1", "F2", "F3"]:
            mask = out["pfam_family"] == fam
            resid_sum = out.loc[mask, "pDN_familyresid"].sum()
            assert abs(resid_sum) < 1e-10

    def test_all_singletons_returns_zero_residuals(self):
        genes = ["A", "B", "C"]
        pfam = {"A": "F1", "B": "F2", "C": "F3"}
        df = make_df(genes, pDN=[0.1, 0.5, 0.9], pGOF=[0.2, 0.5, 0.8], pLOF=[0.3, 0.5, 0.7])
        out = compute_family_residuals(df, pfam, FEATURE_COLS)
        for col in FEATURE_COLS:
            np.testing.assert_array_equal(out[f"{col}_familyresid"].values, [0.0, 0.0, 0.0])
        assert list(out["is_singleton_family_badonyi"]) == [1, 1, 1]

    def test_residual_columns_added(self):
        genes = ["A", "B"]
        pfam = {"A": "FAM1", "B": "FAM1"}
        df = make_df(genes, pDN=[0.2, 0.6], pGOF=[0.1, 0.9], pLOF=[0.4, 0.4])
        out = compute_family_residuals(df, pfam, FEATURE_COLS)
        for col in FEATURE_COLS:
            assert f"{col}_familyresid" in out.columns
        assert "is_singleton_family_badonyi" in out.columns
        assert "pfam_family" in out.columns
