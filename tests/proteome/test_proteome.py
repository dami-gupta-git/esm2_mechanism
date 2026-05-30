"""
Unit tests for pure-logic functions in the proteome and mechanism modules.

Covers:
- proteome_mechanism.gene_split_indices: no family leakage (with pfam_map),
  deterministic, min-size respected, no-pfam_map falls back to gene-split
- proteome_mechanism.aggregate_fold_results: empty → error, mean/std correct, n_folds
- proteome_mechanism.build_triplets_v4: positives cross-family, negatives different
  mechanism, no positives when single family
- proteome_mechanism.run_knn_v4: returns compute_metrics keys
- clinical_utility.get_feature_sets: FULL includes all, NO_MISS excludes _missing
- clinical_utility.compute_ece: zero for perfect calibration, in [0, 1]
- clinical_utility.bootstrap_auroc: CI contains point estimate, width > 0
- clinical_utility.apply_decision_rule: three verdict strings
- per_gene_ablation.get_drop_indices: drops base and derived columns, leaves others
- mechanism.family_split_baselines.build_onehot: shape, known positions set correctly
- mechanism.clan_holdout.aggregate: correct means, handles empty
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# proteome_mechanism.gene_split_indices
# ---------------------------------------------------------------------------


class TestGeneSplitIndices:

    def setup_method(self):
        from esm2_mechanism.proteome.proteome_mechanism import gene_split_indices

        self.gene_split_indices = gene_split_indices

    def _genes(self, n=120, n_genes=12):
        return np.array([f"G{i % n_genes}" for i in range(n)])

    def _pfam(self, n_genes=12, n_fams=4):
        return {f"G{i}": f"FAM{i % n_fams}" for i in range(n_genes)}

    def test_no_family_leakage_with_pfam(self):
        genes = self._genes()
        pfam = self._pfam()
        for tr, te in self.gene_split_indices(genes, n_folds=4, seed=0, pfam_map=pfam):
            tr_fams = {pfam.get(g) for g in genes[tr] if pfam.get(g)}
            te_fams = {pfam.get(g) for g in genes[te] if pfam.get(g)}
            assert not (tr_fams & te_fams)

    def test_train_test_disjoint(self):
        genes = self._genes()
        for tr, te in self.gene_split_indices(genes, n_folds=4, seed=0):
            assert len(set(tr.tolist()) & set(te.tolist())) == 0

    def test_deterministic(self):
        genes = self._genes()
        pfam = self._pfam()
        s1 = list(self.gene_split_indices(genes, n_folds=4, seed=7, pfam_map=pfam))
        s2 = list(self.gene_split_indices(genes, n_folds=4, seed=7, pfam_map=pfam))
        assert len(s1) == len(s2)
        for (tr1, te1), (tr2, te2) in zip(s1, s2):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(te1, te2)

    def test_min_size_respected(self):
        genes = self._genes()
        pfam = self._pfam()
        for tr, te in self.gene_split_indices(genes, n_folds=4, seed=0, pfam_map=pfam):
            assert len(tr) >= 10
            assert len(te) >= 5

    def test_no_pfam_fallback_still_splits(self):
        genes = self._genes()
        splits = list(self.gene_split_indices(genes, n_folds=4, seed=0, pfam_map=None))
        assert len(splits) > 0


# ---------------------------------------------------------------------------
# proteome_mechanism.aggregate_fold_results
# ---------------------------------------------------------------------------


class TestAggregateFoldResults:

    def setup_method(self):
        from esm2_mechanism.proteome.proteome_mechanism import (
            aggregate_fold_results,
            CLASSES,
        )

        self.aggregate_fold_results = aggregate_fold_results
        self.CLASSES = CLASSES

    def _fold(self, f1, auroc_val):
        return {
            "macro_f1": f1,
            "per_class_auroc": {c: auroc_val for c in self.CLASSES},
        }

    def test_empty_returns_error(self):
        result = self.aggregate_fold_results([])
        assert "error" in result

    def test_mean_std_correct(self):
        folds = [self._fold(0.4, 0.7), self._fold(0.6, 0.9)]
        result = self.aggregate_fold_results(folds)
        assert result["macro_f1_mean"] == pytest.approx(0.5)
        assert result["macro_f1_std"] == pytest.approx(0.1)

    def test_n_folds_correct(self):
        folds = [self._fold(0.5, 0.8)] * 3
        result = self.aggregate_fold_results(folds)
        assert result["n_folds"] == 3

    def test_auroc_keys_present(self):
        folds = [self._fold(0.5, 0.8)]
        result = self.aggregate_fold_results(folds)
        for cls in self.CLASSES:
            assert f"auroc_{cls}_mean" in result

    def test_none_auroc_handled(self):
        folds = [
            {"macro_f1": 0.5, "per_class_auroc": {"GOF": None, "DN": 0.8, "LOF": 0.7}},
            {"macro_f1": 0.6, "per_class_auroc": {"GOF": 0.9, "DN": 0.85, "LOF": 0.75}},
        ]
        result = self.aggregate_fold_results(folds)
        assert result["auroc_GOF_mean"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# proteome_mechanism.build_triplets_v4
# ---------------------------------------------------------------------------


class TestBuildTripletsV4:

    def setup_method(self):
        from esm2_mechanism.proteome.proteome_mechanism import (
            build_triplets_v4,
            CLASSES,
        )

        self.build_triplets_v4 = build_triplets_v4
        self.n_classes = len(CLASSES)

    def _data(self, n=120, n_fams=4, seed=0):
        rng = np.random.RandomState(seed)
        y_int = np.array([i % self.n_classes for i in range(n)], dtype=np.int32)
        gene_pfam = np.array([f"FAM{i % n_fams}" for i in range(n)], dtype=object)
        return y_int, gene_pfam

    def test_positives_are_same_mechanism(self):
        y_int, gene_pfam = self._data()
        anchors, positives, _ = self.build_triplets_v4(
            y_int, gene_pfam, max_triplets=200, seed=0
        )
        for a, p in zip(anchors, positives):
            assert y_int[a] == y_int[p]

    def test_negatives_are_different_mechanism(self):
        y_int, gene_pfam = self._data()
        anchors, _, negatives = self.build_triplets_v4(
            y_int, gene_pfam, max_triplets=200, seed=0
        )
        for a, n in zip(anchors, negatives):
            assert y_int[a] != y_int[n]

    def test_positives_are_cross_family(self):
        y_int, gene_pfam = self._data()
        anchors, positives, _ = self.build_triplets_v4(
            y_int, gene_pfam, max_triplets=200, seed=0
        )
        for a, p in zip(anchors, positives):
            assert gene_pfam[a] != gene_pfam[p]

    def test_single_family_produces_no_triplets(self):
        y_int = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
        gene_pfam = np.array(["FAM1"] * 6, dtype=object)
        anchors, positives, negatives = self.build_triplets_v4(
            y_int, gene_pfam, max_triplets=100, seed=0
        )
        assert len(anchors) == 0

    def test_max_triplets_cap(self):
        y_int, gene_pfam = self._data(n=300, n_fams=6)
        anchors, _, _ = self.build_triplets_v4(
            y_int, gene_pfam, max_triplets=50, seed=0
        )
        assert len(anchors) <= 50

    def test_output_lengths_consistent(self):
        y_int, gene_pfam = self._data()
        anchors, positives, negatives = self.build_triplets_v4(
            y_int, gene_pfam, max_triplets=200, seed=0
        )
        assert len(anchors) == len(positives) == len(negatives)


# ---------------------------------------------------------------------------
# clinical_utility.get_feature_sets
# ---------------------------------------------------------------------------


class TestGetFeatureSets:

    def setup_method(self):
        from esm2_mechanism.proteome.clinical_utility import get_feature_sets

        self.get_feature_sets = get_feature_sets

    def test_full_includes_all(self):
        names = ["pLI", "pLI_missing", "pLI_familyresid", "oe_lof"]
        result = self.get_feature_sets(names)
        assert result["FULL"] == list(range(len(names)))

    def test_no_miss_excludes_missing_columns(self):
        names = ["pLI", "pLI_missing", "pLI_familyresid", "oe_lof", "oe_lof_missing"]
        result = self.get_feature_sets(names)
        no_miss = result["NO_MISS"]
        missing_cols = [i for i, n in enumerate(names) if n.endswith("_missing")]
        for idx in missing_cols:
            assert idx not in no_miss

    def test_no_miss_includes_non_missing(self):
        names = ["pLI", "pLI_missing", "pLI_familyresid", "oe_lof"]
        result = self.get_feature_sets(names)
        # pLI and pLI_familyresid and oe_lof should be in NO_MISS
        assert 0 in result["NO_MISS"]  # pLI
        assert 3 in result["NO_MISS"]  # oe_lof

    def test_is_singleton_excluded_from_no_miss(self):
        names = ["pLI", "is_singleton_family", "pLI_familyresid"]
        result = self.get_feature_sets(names)
        assert 1 not in result["NO_MISS"]


# ---------------------------------------------------------------------------
# clinical_utility.compute_ece
# ---------------------------------------------------------------------------


class TestComputeEce:

    def setup_method(self):
        from esm2_mechanism.proteome.clinical_utility import compute_ece

        self.compute_ece = compute_ece

    def test_perfect_calibration_near_zero(self):
        # y_prob == y_true (binary) → perfectly calibrated
        y_true = np.array([0.0, 1.0, 0.0, 1.0] * 10)
        y_prob = np.array([0.0, 1.0, 0.0, 1.0] * 10)
        ece = self.compute_ece(y_true, y_prob)
        assert ece == pytest.approx(0.0, abs=1e-6)

    def test_ece_in_range(self):
        rng = np.random.RandomState(0)
        y_true = rng.randint(0, 2, 100).astype(float)
        y_prob = rng.rand(100)
        ece = self.compute_ece(y_true, y_prob)
        assert 0.0 <= ece <= 1.0

    def test_worst_case_high_ece(self):
        # Confident wrong predictions: high score but all labels are 0
        # Use 0.95 so samples land in the [0.9, 1.0) bin
        y_true = np.zeros(100)
        y_prob = np.full(100, 0.95)
        ece = self.compute_ece(y_true, y_prob)
        assert ece > 0.5


# ---------------------------------------------------------------------------
# clinical_utility.bootstrap_auroc
# ---------------------------------------------------------------------------


class TestBootstrapAuroc:

    def setup_method(self):
        from esm2_mechanism.proteome.clinical_utility import bootstrap_auroc

        self.bootstrap_auroc = bootstrap_auroc

    def test_ci_contains_point_estimate(self):
        rng = np.random.RandomState(0)
        y_true = np.array([0, 1] * 50)
        y_score = rng.rand(100)
        point, lo, hi = self.bootstrap_auroc(y_true, y_score, n_boot=200, seed=0)
        assert lo <= point <= hi

    def test_ci_width_positive(self):
        rng = np.random.RandomState(0)
        y_true = np.array([0, 1] * 50)
        y_score = rng.rand(100)
        _, lo, hi = self.bootstrap_auroc(y_true, y_score, n_boot=200, seed=0)
        assert hi > lo

    def test_point_estimate_in_range(self):
        y_true = np.array([0, 1] * 20)
        y_score = np.linspace(0, 1, 40)
        point, _, _ = self.bootstrap_auroc(y_true, y_score, n_boot=100, seed=0)
        assert 0.0 <= point <= 1.0


# ---------------------------------------------------------------------------
# clinical_utility.apply_decision_rule
# ---------------------------------------------------------------------------


class TestApplyDecisionRule:

    def setup_method(self):
        from esm2_mechanism.proteome.clinical_utility import apply_decision_rule

        self.apply_decision_rule = apply_decision_rule

    def test_informative(self):
        result = self.apply_decision_rule(
            {
                "H1_GOF_vs_LOF_AUROC_LR_FULL": 0.75,
                "H1_GOF_vs_LOF_CI95_LR_FULL": [0.65, 0.85],
            }
        )
        assert "INFORMATIVE" in result

    def test_weakly_informative(self):
        result = self.apply_decision_rule(
            {
                "H1_GOF_vs_LOF_AUROC_LR_FULL": 0.60,
                "H1_GOF_vs_LOF_CI95_LR_FULL": [0.50, 0.70],
            }
        )
        assert "WEAKLY" in result

    def test_null(self):
        result = self.apply_decision_rule(
            {
                "H1_GOF_vs_LOF_AUROC_LR_FULL": 0.50,
                "H1_GOF_vs_LOF_CI95_LR_FULL": [0.40, 0.60],
            }
        )
        assert "NULL" in result

    def test_missing_ci_no_crash(self):
        result = self.apply_decision_rule({"H1_GOF_vs_LOF_AUROC_LR_FULL": 0.70})
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# per_gene_ablation.get_drop_indices
# ---------------------------------------------------------------------------


class TestGetDropIndices:

    def setup_method(self):
        from esm2_mechanism.proteome.per_gene_ablation import get_drop_indices

        self.get_drop_indices = get_drop_indices

    def test_base_column_dropped(self):
        names = ["pLI", "oe_lof", "gnomad_z"]
        drop = self.get_drop_indices(names, base_features=["pLI"])
        assert 0 in drop  # pLI
        assert 1 not in drop  # oe_lof kept

    def test_derived_columns_dropped(self):
        names = ["pLI", "pLI_missing", "pLI_familyresid", "oe_lof"]
        drop = self.get_drop_indices(names, base_features=["pLI"])
        assert 0 in drop  # pLI
        assert 1 in drop  # pLI_missing
        assert 2 in drop  # pLI_familyresid
        assert 3 not in drop

    def test_empty_base_features_drops_nothing(self):
        names = ["pLI", "oe_lof"]
        drop = self.get_drop_indices(names, base_features=[])
        assert drop == []

    def test_multiple_base_features(self):
        names = ["pLI", "pLI_missing", "oe_lof", "oe_lof_missing", "gnomad_z"]
        drop = self.get_drop_indices(names, base_features=["pLI", "oe_lof"])
        assert 0 in drop  # pLI
        assert 1 in drop  # pLI_missing
        assert 2 in drop  # oe_lof
        assert 3 in drop  # oe_lof_missing
        assert 4 not in drop  # gnomad_z kept


# ---------------------------------------------------------------------------
# mechanism.family_split_baselines.build_onehot
# ---------------------------------------------------------------------------


class TestBuildOnehot:

    def setup_method(self):
        from esm2_mech.experiments.mechanism.mechanism_delta_family_split import build_onehot

        self.build_onehot = build_onehot

    def test_shape(self):
        onehot = self.build_onehot(["A", "V", "G"], ["V", "E", "A"])
        assert onehot.shape == (3, 40)

    def test_wt_position_set(self):
        from esm2_mech.experiments.mechanism.mechanism_delta_family_split import AA_INDEX

        onehot = self.build_onehot(["A"], ["V"])
        assert onehot[0, AA_INDEX["A"]] == 1.0

    def test_mut_position_set(self):
        from esm2_mech.experiments.mechanism.mechanism_delta_family_split import AA_INDEX

        onehot = self.build_onehot(["A"], ["V"])
        assert onehot[0, 20 + AA_INDEX["V"]] == 1.0

    def test_unknown_aa_leaves_zeros(self):
        onehot = self.build_onehot(["X"], ["Z"])
        assert onehot[0].sum() == pytest.approx(0.0)

    def test_dtype_float32(self):
        onehot = self.build_onehot(["A"], ["V"])
        assert onehot.dtype == np.float32


# ---------------------------------------------------------------------------
# mechanism.clan_holdout.aggregate
# ---------------------------------------------------------------------------


class TestClanAggregate:

    def setup_method(self):
        from esm2_mechanism.mechanism.clan_holdout import aggregate

        self.aggregate = aggregate

    def _result(self, f1, gof=None, dn=None, lof=None, n_test=10):
        mlp = {"macro_f1": f1}
        if gof is not None:
            mlp["auroc_GOF"] = gof
        if dn is not None:
            mlp["auroc_DN"] = dn
        if lof is not None:
            mlp["auroc_LOF"] = lof
        return {
            "clan": "CL0001",
            "clan_name": "test",
            "n_train": 100,
            "n_test": n_test,
            "n_genes_test": 3,
            "test_mechs": {"GOF": 5, "LOF": 5},
            "mlp": mlp,
            "knn_macro_f1": f1 - 0.05,
            "majority_macro_f1": 0.33,
        }

    def test_mean_f1_correct(self):
        results = [self._result(0.4), self._result(0.6)]
        agg = self.aggregate(results)
        assert agg["mlp_macro_f1_mean"] == pytest.approx(0.5)

    def test_n_clans_correct(self):
        results = [self._result(0.5)] * 3
        agg = self.aggregate(results)
        assert agg["n_clans"] == 3

    def test_empty_returns_nan(self):
        agg = self.aggregate([])
        assert np.isnan(agg["mlp_macro_f1_mean"])

    def test_error_mlp_excluded(self):
        r1 = self._result(0.6)
        r2 = {
            "clan": "CL0002",
            "clan_name": "bad",
            "n_train": 50,
            "n_test": 10,
            "n_genes_test": 2,
            "test_mechs": {},
            "mlp": {"error": "failed"},
            "knn_macro_f1": float("nan"),
            "majority_macro_f1": 0.33,
        }
        agg = self.aggregate([r1, r2])
        assert agg["mlp_macro_f1_mean"] == pytest.approx(0.6)

    def test_per_class_auroc_present(self):
        results = [self._result(0.5, gof=0.7, dn=0.6, lof=0.8)]
        agg = self.aggregate(results)
        assert "GOF" in agg["per_class_auroc_mean"]
        assert "DN" in agg["per_class_auroc_mean"]
        assert "LOF" in agg["per_class_auroc_mean"]
