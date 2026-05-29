"""
Unit tests for pure-logic functions in the badonyi source modules.

Covers:
- badonyi_leakage_analysis.broadcast: correct shape, known-row values, missing gene → zeros
- badonyi_holdout_survival.assign_folds: no None-group in numbered folds, deterministic,
  different seeds differ, groups exclusive across folds
- badonyi_holdout_survival.compute_aurocs: correct binary AUROCs, missing preds → None,
  single-class → None
- within_family_mechanism.collapse_3class: all mappings, unknown returns None
- within_family_mechanism.get_qualifying_families: min_genes filter, min_classes filter,
  unknown pfam genes excluded
"""

import numpy as np
import pandas as pd
import pytest
from collections import Counter


# ---------------------------------------------------------------------------
# badonyi_leakage_analysis.broadcast
# ---------------------------------------------------------------------------

class TestBroadcast:

    def setup_method(self):
        from esm2_mechanism.badonyi.badonyi_leakage_analysis import broadcast
        self.broadcast = broadcast

    def test_output_shape(self):
        genes = np.array(["A", "B", "C"])
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        gene_to_row = {"A": 0, "B": 1}
        out = self.broadcast(genes, matrix, gene_to_row)
        assert out.shape == (3, 2)

    def test_known_genes_get_correct_rows(self):
        genes = np.array(["A", "B"])
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        gene_to_row = {"A": 0, "B": 1}
        out = self.broadcast(genes, matrix, gene_to_row)
        np.testing.assert_array_almost_equal(out[0], [1.0, 2.0])
        np.testing.assert_array_almost_equal(out[1], [3.0, 4.0])

    def test_missing_gene_gets_zeros(self):
        genes = np.array(["A", "UNKNOWN"])
        matrix = np.array([[1.0, 2.0]], dtype=np.float32)
        gene_to_row = {"A": 0}
        out = self.broadcast(genes, matrix, gene_to_row)
        np.testing.assert_array_equal(out[1], [0.0, 0.0])

    def test_returns_float32(self):
        genes = np.array(["A"])
        matrix = np.array([[1.0, 2.0]], dtype=np.float32)
        gene_to_row = {"A": 0}
        out = self.broadcast(genes, matrix, gene_to_row)
        assert out.dtype == np.float32

    def test_all_missing_returns_zeros(self):
        genes = np.array(["X", "Y"])
        matrix = np.array([[1.0, 2.0]], dtype=np.float32)
        gene_to_row = {}
        out = self.broadcast(genes, matrix, gene_to_row)
        np.testing.assert_array_equal(out, np.zeros((2, 2), dtype=np.float32))


# ---------------------------------------------------------------------------
# badonyi_holdout_survival.assign_folds
# ---------------------------------------------------------------------------

class TestAssignFolds:

    def setup_method(self):
        from esm2_mechanism.badonyi.badonyi_holdout_survival import assign_folds
        self.assign_folds = assign_folds

    def _make_df(self, n=30, n_groups=5):
        groups = [f"FAM{i % n_groups}" for i in range(n)]
        return pd.DataFrame({"group": groups})

    def test_none_groups_get_minus_one(self):
        df = pd.DataFrame({"group": ["A", None, "B", None]})
        folds = self.assign_folds(df, "group", n_folds=3, seed=0)
        none_positions = [i for i, v in enumerate(df["group"]) if v is None]
        for pos in none_positions:
            assert folds[pos] == -1

    def test_non_none_groups_get_valid_fold(self):
        df = self._make_df()
        folds = self.assign_folds(df, "group", n_folds=3, seed=0)
        for i, g in enumerate(df["group"]):
            if g is not None:
                assert 0 <= folds[i] < 3

    def test_deterministic(self):
        df = self._make_df()
        f1 = self.assign_folds(df, "group", n_folds=3, seed=7)
        f2 = self.assign_folds(df, "group", n_folds=3, seed=7)
        np.testing.assert_array_equal(f1, f2)

    def test_different_seeds_differ(self):
        df = self._make_df(n=60, n_groups=10)
        f1 = self.assign_folds(df, "group", n_folds=5, seed=0)
        f2 = self.assign_folds(df, "group", n_folds=5, seed=99)
        assert not np.array_equal(f1, f2)

    def test_all_rows_in_same_group_get_same_fold(self):
        df = pd.DataFrame({"group": ["FAM1"] * 10})
        folds = self.assign_folds(df, "group", n_folds=3, seed=0)
        non_none = folds[folds != -1]
        assert len(set(non_none.tolist())) == 1


# ---------------------------------------------------------------------------
# badonyi_holdout_survival.compute_aurocs
# ---------------------------------------------------------------------------

class TestComputeAurocs:

    def setup_method(self):
        from esm2_mechanism.badonyi.badonyi_holdout_survival import compute_aurocs
        self.compute_aurocs = compute_aurocs

    def _make_df(self, labels, pDN, pGOF, pLOF):
        return pd.DataFrame({
            "label3": labels,
            "pDN": pDN,
            "pGOF": pGOF,
            "pLOF": pLOF,
        })

    def test_perfect_dn_vs_lof(self):
        n = 20
        labels = ["DN"] * 10 + ["LOF"] * 10
        pDN  = [1.0] * 10 + [0.0] * 10
        df = self._make_df(labels, pDN=pDN, pGOF=[0.5]*n, pLOF=[0.5]*n)
        result = self.compute_aurocs(df)
        assert result["DN_vs_LOF"] == pytest.approx(1.0)

    def test_chance_dn_vs_lof(self):
        n = 20
        labels = ["DN"] * 10 + ["LOF"] * 10
        pDN = [0.5] * n
        df = self._make_df(labels, pDN=pDN, pGOF=[0.5]*n, pLOF=[0.5]*n)
        result = self.compute_aurocs(df)
        assert result["DN_vs_LOF"] == pytest.approx(0.5)

    def test_single_class_dn_vs_lof_returns_none(self):
        n = 10
        df = self._make_df(["DN"]*n, pDN=[0.9]*n, pGOF=[0.5]*n, pLOF=[0.1]*n)
        result = self.compute_aurocs(df)
        assert result["DN_vs_LOF"] is None

    def test_missing_preds_excluded(self):
        df = pd.DataFrame({
            "label3": ["DN", "LOF", "DN", "LOF"],
            "pDN":  [None, None, 0.9, 0.1],
            "pGOF": [None, None, 0.1, 0.9],
            "pLOF": [None, None, 0.1, 0.9],
        })
        result = self.compute_aurocs(df)
        # Only 2 rows have predictions — still computable if both classes present
        assert result["n_total"] == 2

    def test_n_total_correct(self):
        n = 15
        labels = ["DN"] * 5 + ["LOF"] * 5 + ["GOF"] * 5
        df = self._make_df(labels, pDN=[0.5]*n, pGOF=[0.5]*n, pLOF=[0.5]*n)
        result = self.compute_aurocs(df)
        assert result["n_total"] == n

    def test_keys_present(self):
        n = 20
        labels = ["DN"] * 7 + ["LOF"] * 7 + ["GOF"] * 6
        df = self._make_df(labels, pDN=[0.5]*n, pGOF=[0.5]*n, pLOF=[0.5]*n)
        result = self.compute_aurocs(df)
        assert "DN_vs_LOF" in result
        assert "GOF_vs_LOF" in result
        assert "LOF_vs_nonLOF" in result


# ---------------------------------------------------------------------------
# within_family_mechanism.collapse_3class
# ---------------------------------------------------------------------------

class TestCollapse3class:

    def setup_method(self):
        from esm2_mechanism.badonyi.within_family_mechanism import collapse_3class
        self.collapse_3class = collapse_3class

    def test_gof(self):
        assert self.collapse_3class("GOF") == "GOF"

    def test_dn(self):
        assert self.collapse_3class("DN") == "DN"

    def test_hi_maps_to_lof(self):
        assert self.collapse_3class("HI") == "LOF"

    def test_lof_maps_to_lof(self):
        assert self.collapse_3class("LOF") == "LOF"

    def test_unknown_returns_none(self):
        assert self.collapse_3class("AR") is None
        assert self.collapse_3class("GAIN") is None
        assert self.collapse_3class("") is None


# ---------------------------------------------------------------------------
# within_family_mechanism.get_qualifying_families
# ---------------------------------------------------------------------------

class TestGetQualifyingFamilies:

    def setup_method(self):
        from esm2_mechanism.badonyi.within_family_mechanism import get_qualifying_families
        self.get_qualifying_families = get_qualifying_families

    def test_family_with_enough_genes_and_classes_included(self):
        genes = ["G1", "G2", "G3", "G4"]
        mechs = ["GOF", "LOF", "GOF", "LOF"]
        pfam  = {"G1": "FAM1", "G2": "FAM1", "G3": "FAM1", "G4": "FAM1"}
        result = self.get_qualifying_families(genes, mechs, pfam, min_genes=2, min_classes=2)
        assert "FAM1" in result

    def test_family_with_too_few_genes_excluded(self):
        genes = ["G1"]
        mechs = ["GOF"]
        pfam  = {"G1": "FAM1"}
        result = self.get_qualifying_families(genes, mechs, pfam, min_genes=2, min_classes=1)
        assert "FAM1" not in result

    def test_family_with_single_class_excluded_when_min_classes_2(self):
        genes = ["G1", "G2", "G3"]
        mechs = ["GOF", "GOF", "GOF"]
        pfam  = {"G1": "FAM1", "G2": "FAM1", "G3": "FAM1"}
        result = self.get_qualifying_families(genes, mechs, pfam, min_genes=2, min_classes=2)
        assert "FAM1" not in result

    def test_genes_without_pfam_excluded(self):
        genes = ["G1", "G2", "G3"]
        mechs = ["GOF", "LOF", "GOF"]
        pfam  = {"G1": "FAM1", "G2": "FAM1"}  # G3 not in pfam
        result = self.get_qualifying_families(genes, mechs, pfam, min_genes=2, min_classes=2)
        if "FAM1" in result:
            assert "G3" not in result["FAM1"]

    def test_unknown_mechanism_excluded(self):
        genes = ["G1", "G2", "G3"]
        mechs = ["GOF", "AR", "LOF"]   # AR → None via collapse_3class
        pfam  = {"G1": "FAM1", "G2": "FAM1", "G3": "FAM1"}
        result = self.get_qualifying_families(genes, mechs, pfam, min_genes=2, min_classes=2)
        if "FAM1" in result:
            assert "G2" not in result["FAM1"]

    def test_multiple_families_independently_filtered(self):
        genes = ["G1", "G2", "G3", "G4", "G5"]
        mechs = ["GOF", "LOF", "GOF", "GOF", "GOF"]
        pfam  = {"G1": "FAM1", "G2": "FAM1", "G3": "FAM2", "G4": "FAM2", "G5": "FAM2"}
        result = self.get_qualifying_families(genes, mechs, pfam, min_genes=2, min_classes=2)
        assert "FAM1" in result   # 2 genes, 2 classes
        assert "FAM2" not in result  # 3 genes, 1 class
