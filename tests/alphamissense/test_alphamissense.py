"""
Unit tests for pure-logic functions in the alphamissense modules.

Covers:
- alphamissense_family_split.vkey: correct key format
- proteingym_alphamissense._dist: empty input, single value, multi-value stats,
  frac_below thresholds
"""

import pytest
import numpy as np


class TestVkey:

    def setup_method(self):
        from esm2_mechanism.alphamissense.alphamissense_family_split import vkey
        self.vkey = vkey

    def test_basic_format(self):
        v = {"gene": "BRCA1", "aa_pos": 100, "aa_wt": "A", "aa_mut": "V"}
        assert self.vkey(v) == "BRCA1_100_A_V"

    def test_different_genes_differ(self):
        v1 = {"gene": "BRCA1", "aa_pos": 100, "aa_wt": "A", "aa_mut": "V"}
        v2 = {"gene": "BRCA2", "aa_pos": 100, "aa_wt": "A", "aa_mut": "V"}
        assert self.vkey(v1) != self.vkey(v2)

    def test_different_positions_differ(self):
        v1 = {"gene": "G", "aa_pos": 1, "aa_wt": "A", "aa_mut": "V"}
        v2 = {"gene": "G", "aa_pos": 2, "aa_wt": "A", "aa_mut": "V"}
        assert self.vkey(v1) != self.vkey(v2)

    def test_different_aa_mut_differ(self):
        v1 = {"gene": "G", "aa_pos": 1, "aa_wt": "A", "aa_mut": "V"}
        v2 = {"gene": "G", "aa_pos": 1, "aa_wt": "A", "aa_mut": "E"}
        assert self.vkey(v1) != self.vkey(v2)

    def test_same_variant_same_key(self):
        v = {"gene": "TP53", "aa_pos": 248, "aa_wt": "R", "aa_mut": "W"}
        assert self.vkey(v) == self.vkey(v)


class TestDist:

    def setup_method(self):
        from esm2_mechanism.alphamissense.proteingym_alphamissense import _dist
        self._dist = _dist

    def test_empty_returns_n_zero(self):
        result = self._dist([])
        assert result == {"n": 0}

    def test_single_value_stats(self):
        result = self._dist([0.8])
        assert result["n"] == 1
        assert result["mean"] == pytest.approx(0.8)
        assert result["min"] == pytest.approx(0.8)
        assert result["max"] == pytest.approx(0.8)

    def test_mean_correct(self):
        result = self._dist([0.4, 0.6, 0.8])
        assert result["mean"] == pytest.approx(np.mean([0.4, 0.6, 0.8]))

    def test_n_correct(self):
        result = self._dist([0.1, 0.2, 0.3, 0.4])
        assert result["n"] == 4

    def test_frac_below_0_7(self):
        # 3 of 4 values below 0.7
        result = self._dist([0.5, 0.6, 0.65, 0.9])
        assert result["frac_below_0_7"] == pytest.approx(0.75)

    def test_frac_below_0_6(self):
        # 2 of 4 values below 0.6
        result = self._dist([0.5, 0.55, 0.65, 0.9])
        assert result["frac_below_0_6"] == pytest.approx(0.5)

    def test_all_above_thresholds(self):
        result = self._dist([0.8, 0.9, 0.95])
        assert result["frac_below_0_7"] == pytest.approx(0.0)
        assert result["frac_below_0_6"] == pytest.approx(0.0)

    def test_all_below_thresholds(self):
        result = self._dist([0.1, 0.2, 0.3])
        assert result["frac_below_0_7"] == pytest.approx(1.0)
        assert result["frac_below_0_6"] == pytest.approx(1.0)

    def test_median_and_quartiles(self):
        result = self._dist([1.0, 2.0, 3.0, 4.0])
        assert result["median"] == pytest.approx(np.median([1.0, 2.0, 3.0, 4.0]))
        assert result["q25"] == pytest.approx(np.quantile([1.0, 2.0, 3.0, 4.0], 0.25))
        assert result["q75"] == pytest.approx(np.quantile([1.0, 2.0, 3.0, 4.0], 0.75))
