"""
Unit tests for pure-logic functions in the perturb modules.

Covers:
- ll_scan.compute_ll_features: correct feature shapes, MIN_POSITIONS filter,
  delta_cv formula, hotspot_frac threshold, entropy computation, NaN handling
- megascale_mlp.pfam_split_cv: no family leakage, min-size respected, deterministic
- megascale_mlp.run_sklearn_probe: returns expected keys, handles empty splits,
  spearman on correlated data is high
- perturbation_pattern.build_gene_features: correct scalar feature count, output
  shapes, singleton and multi-variant genes, mag_cv formula
- perturbation_pattern.run_probe / perturbation_probe.run_probe: both return
  expected keys; empty splits returns empty dict
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# ll_scan — compute_ll_features
# ---------------------------------------------------------------------------


class TestComputeLlFeatures:

    def setup_method(self):
        from esm2_mechanism.perturb.ll_scan import compute_ll_features, MIN_POSITIONS

        self.compute_ll_features = compute_ll_features
        self.MIN_POSITIONS = MIN_POSITIONS

    def _make_scores(self, n_positions=10, seed=0):
        rng = np.random.RandomState(seed)
        aa_order = "ACDEFGHIKLMNPQRSTVWY"
        scores = []
        for i in range(n_positions):
            probs = rng.dirichlet(np.ones(20))
            scores.append(
                {
                    "aa_pos": i + 1,
                    "wt_aa": "A",
                    "ll_wt": float(np.log(probs[0] + 1e-12)),
                    "ll_ala": float(np.log(probs[0] + 1e-12)),
                    "ll_asp": float(np.log(probs[1] + 1e-12)),
                    "ll_trp": float(np.log(probs[2] + 1e-12)),
                    "full_probs": probs.tolist(),
                }
            )
        return scores

    def test_output_shapes(self):
        covered = ["GENE1", "GENE2"]
        all_scores = {g: self._make_scores(10, seed=i) for i, g in enumerate(covered)}
        gene_list, X, feature_names = self.compute_ll_features(covered, all_scores)
        assert X.shape == (2, 5)
        assert len(gene_list) == 2
        assert len(feature_names) == 5

    def test_min_positions_filter(self):
        covered = ["GENE_OK", "GENE_SHORT"]
        all_scores = {
            "GENE_OK": self._make_scores(10),
            "GENE_SHORT": self._make_scores(self.MIN_POSITIONS - 1),
        }
        gene_list, X, _ = self.compute_ll_features(covered, all_scores)
        assert "GENE_SHORT" not in gene_list
        assert "GENE_OK" in gene_list

    def test_missing_gene_excluded(self):
        covered = ["GENE_PRESENT", "GENE_ABSENT"]
        all_scores = {"GENE_PRESENT": self._make_scores(10)}
        gene_list, X, _ = self.compute_ll_features(covered, all_scores)
        assert "GENE_ABSENT" not in gene_list

    def test_feature_names_correct(self):
        covered = ["G"]
        all_scores = {"G": self._make_scores(10)}
        _, _, feature_names = self.compute_ll_features(covered, all_scores)
        assert feature_names == [
            "ll_wt_mean",
            "ll_delta_mean",
            "ll_delta_cv",
            "ll_hotspot_frac",
            "ll_top_entropy",
        ]

    def test_hotspot_frac_in_range(self):
        covered = ["G"]
        all_scores = {"G": self._make_scores(20)}
        _, X, _ = self.compute_ll_features(covered, all_scores)
        hotspot_frac = X[0, 3]
        assert 0.0 <= hotspot_frac <= 1.0

    def test_entropy_is_positive(self):
        covered = ["G"]
        all_scores = {"G": self._make_scores(20)}
        _, X, _ = self.compute_ll_features(covered, all_scores)
        ll_top_entropy = X[0, 4]
        assert ll_top_entropy > 0.0

    def test_uniform_probs_give_high_entropy(self):
        # Uniform distribution over 20 AAs → max entropy ≈ ln(20) ≈ 2.996
        scores = []
        for i in range(15):
            scores.append(
                {
                    "aa_pos": i + 1,
                    "wt_aa": "A",
                    "ll_wt": -3.0,
                    "ll_ala": -3.0,
                    "ll_asp": -3.0,
                    "ll_trp": -3.0,
                    "full_probs": [1.0 / 20] * 20,
                }
            )
        _, X, _ = self.compute_ll_features(["G"], {"G": scores})
        ll_top_entropy = float(X[0, 4])
        assert ll_top_entropy > 2.5

    def test_delta_cv_nonnegative(self):
        covered = ["G"]
        all_scores = {"G": self._make_scores(15)}
        _, X, _ = self.compute_ll_features(covered, all_scores)
        delta_cv = X[0, 2]
        assert delta_cv >= 0.0

    def test_returns_float32(self):
        covered = ["G"]
        all_scores = {"G": self._make_scores(10)}
        _, X, _ = self.compute_ll_features(covered, all_scores)
        assert X.dtype == np.float32


# ---------------------------------------------------------------------------
# megascale_mlp — pfam_split_cv
# ---------------------------------------------------------------------------


class TestPfamSplitCv:

    def setup_method(self):
        from esm2_mechanism.perturb.megascale_mlp import pfam_split_cv, PFAM

        self.pfam_split_cv = pfam_split_cv
        self.PFAM = PFAM

    def _proteins(self):
        return np.array(list(self.PFAM.keys()))

    def test_no_family_leakage(self):
        from esm2_mechanism.perturb.megascale_mlp import PFAM

        proteins = self._proteins()
        for tr, te in self.pfam_split_cv(proteins, n_folds=5, seed=0):
            tr_fams = {PFAM.get(p, p) for p in proteins[tr]}
            te_fams = {PFAM.get(p, p) for p in proteins[te]}
            assert not (tr_fams & te_fams), f"Family leakage: {tr_fams & te_fams}"

    def test_min_size_respected(self):
        proteins = self._proteins()
        for tr, te in self.pfam_split_cv(proteins, n_folds=5, seed=0):
            assert len(tr) >= 10
            assert len(te) >= 5

    def test_deterministic(self):
        proteins = self._proteins()
        s1 = self.pfam_split_cv(proteins, n_folds=5, seed=42)
        s2 = self.pfam_split_cv(proteins, n_folds=5, seed=42)
        assert len(s1) == len(s2)
        for (tr1, te1), (tr2, te2) in zip(s1, s2):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(te1, te2)

    def test_different_seeds_differ(self):
        proteins = self._proteins()
        s1 = self.pfam_split_cv(proteins, n_folds=5, seed=0)
        s2 = self.pfam_split_cv(proteins, n_folds=5, seed=99)
        if len(s1) > 0 and len(s2) > 0:
            any_diff = any(
                not (np.array_equal(tr1, tr2) and np.array_equal(te1, te2))
                for (tr1, te1), (tr2, te2) in zip(s1, s2)
            )
            assert any_diff

    def test_indices_in_bounds(self):
        proteins = self._proteins()
        n = len(proteins)
        for tr, te in self.pfam_split_cv(proteins, n_folds=5, seed=0):
            assert tr.min() >= 0 and tr.max() < n
            assert te.min() >= 0 and te.max() < n


# ---------------------------------------------------------------------------
# megascale_mlp — run_sklearn_probe
# ---------------------------------------------------------------------------


class TestRunSklearnProbe:

    def setup_method(self):
        from esm2_mechanism.perturb.megascale_mlp import run_sklearn_probe

        self.run_sklearn_probe = run_sklearn_probe

    def _correlated_data(self, n=100, dim=10, seed=0):
        rng = np.random.RandomState(seed)
        X = rng.randn(n, dim).astype(np.float32)
        y = X[:, 0] * 3.0 + rng.randn(n) * 0.3
        splits = [(np.arange(n // 2), np.arange(n // 2, n))]
        return X, y, splits

    def _ridge(self):
        from sklearn.linear_model import Ridge

        return Ridge(alpha=1.0)

    def test_returns_spearman_keys(self):
        X, y, splits = self._correlated_data()
        result = self.run_sklearn_probe(X, y, splits, clf_fn=self._ridge)
        assert "spearman_mean" in result
        assert "spearman_std" in result

    def test_returns_auroc_keys(self):
        X, y, splits = self._correlated_data()
        result = self.run_sklearn_probe(X, y, splits, clf_fn=self._ridge)
        assert "auroc_mean" in result
        assert "auroc_std" in result

    def test_returns_n_folds(self):
        X, y, splits = self._correlated_data()
        result = self.run_sklearn_probe(X, y, splits, clf_fn=self._ridge)
        assert result["n_folds"] == len(splits)

    def test_high_spearman_on_correlated_data(self):
        X, y, splits = self._correlated_data()
        result = self.run_sklearn_probe(X, y, splits, clf_fn=self._ridge)
        assert result["spearman_mean"] > 0.8

    def test_empty_splits_returns_empty(self):
        X, y, _ = self._correlated_data()
        result = self.run_sklearn_probe(X, y, [], clf_fn=self._ridge)
        assert result == {}


# ---------------------------------------------------------------------------
# perturbation_pattern — build_gene_features
# ---------------------------------------------------------------------------


class TestBuildGeneFeatures:

    def setup_method(self):
        from esm2_mechanism.perturb.perturbation_pattern import build_gene_features

        self.build_gene_features = build_gene_features

    def _make_variants(self, n_genes=4, variants_per_gene=5, dim=8, seed=0):
        rng = np.random.RandomState(seed)
        classes = ["GOF", "LOF", "DN"]
        variants = []
        for i in range(n_genes):
            label = classes[i % 3]
            for j in range(variants_per_gene):
                variants.append(
                    {
                        "gene": f"GENE{i:02d}",
                        "aa_pos": j + 1,
                        "label_3class": label,
                    }
                )
        n = len(variants)
        delta_pos = rng.randn(n, dim).astype(np.float32)
        delta_mean = rng.randn(n, dim).astype(np.float32)
        return variants, delta_pos, delta_mean

    def test_output_count_matches_genes(self):
        variants, delta_pos, delta_mean = self._make_variants(n_genes=4)
        gene_list, X, labels, n_scalar = self.build_gene_features(
            variants, delta_pos, delta_mean
        )
        assert len(gene_list) == 4
        assert len(labels) == 4
        assert X.shape[0] == 4

    def test_scalar_feature_count_is_8(self):
        variants, delta_pos, delta_mean = self._make_variants()
        _, X, _, n_scalar = self.build_gene_features(variants, delta_pos, delta_mean)
        assert n_scalar == 8

    def test_total_feature_dim_is_scalar_plus_delta(self):
        dim = 16
        variants, delta_pos, delta_mean = self._make_variants(dim=dim)
        _, X, _, n_scalar = self.build_gene_features(variants, delta_pos, delta_mean)
        assert X.shape[1] == n_scalar + dim

    def test_mag_cv_is_nonnegative(self):
        variants, delta_pos, delta_mean = self._make_variants()
        _, X, _, _ = self.build_gene_features(variants, delta_pos, delta_mean)
        mag_cv_col = 2
        assert (X[:, mag_cv_col] >= 0).all()

    def test_singleton_gene_pc1_zero(self):
        # Gene with only 1 variant → PCA not run → pc1_var=0, pc1_proj=0
        variants = [{"gene": "SOLO", "aa_pos": 5, "label_3class": "LOF"}]
        delta_pos = np.random.randn(1, 8).astype(np.float32)
        delta_mean = np.random.randn(1, 8).astype(np.float32)
        _, X, _, _ = self.build_gene_features(variants, delta_pos, delta_mean)
        pc1_var_col = 5
        pc1_proj_col = 6
        assert X[0, pc1_var_col] == pytest.approx(0.0)
        assert X[0, pc1_proj_col] == pytest.approx(0.0)

    def test_pos_mean_norm_in_range(self):
        variants, delta_pos, delta_mean = self._make_variants()
        _, X, _, _ = self.build_gene_features(variants, delta_pos, delta_mean)
        pos_mean_norm_col = 3
        assert (X[:, pos_mean_norm_col] >= 0).all()
        assert (X[:, pos_mean_norm_col] <= 1.0 + 1e-6).all()

    def test_n_variants_log_positive(self):
        variants, delta_pos, delta_mean = self._make_variants()
        _, X, _, _ = self.build_gene_features(variants, delta_pos, delta_mean)
        n_var_col = 7
        assert (X[:, n_var_col] > 0).all()

    def test_returns_float32(self):
        variants, delta_pos, delta_mean = self._make_variants()
        _, X, _, _ = self.build_gene_features(variants, delta_pos, delta_mean)
        assert X.dtype == np.float32


# ---------------------------------------------------------------------------
# perturbation_pattern.run_probe and perturbation_probe.run_probe
# (Both have the same signature — test both implementations)
# ---------------------------------------------------------------------------


def _make_probe_inputs(n=90, dim=8, seed=0):
    rng = np.random.RandomState(seed)
    classes = ["GOF", "LOF", "DN"]
    labels = np.array(classes * (n // 3))
    X = rng.randn(n, dim).astype(np.float32)
    fold = n // 3
    splits = [
        (
            np.concatenate([np.arange(fold), np.arange(2 * fold, n)]),
            np.arange(fold, 2 * fold),
        ),
    ]
    return X, labels, splits


@pytest.mark.parametrize(
    "module_path",
    [
        "esm2_mechanism.perturb.perturbation_pattern",
        "esm2_mechanism.perturb.perturbation_probe",
    ],
)
class TestRunProbe:

    def _get_run_probe(self, module_path):
        import importlib

        mod = importlib.import_module(module_path)
        return mod.run_probe

    def test_returns_macro_f1_mean(self, module_path):
        run_probe = self._get_run_probe(module_path)
        X, labels, splits = _make_probe_inputs()
        result = run_probe(X, labels, splits)
        assert "macro_f1_mean" in result

    def test_returns_macro_f1_std(self, module_path):
        run_probe = self._get_run_probe(module_path)
        X, labels, splits = _make_probe_inputs()
        result = run_probe(X, labels, splits)
        assert "macro_f1_std" in result

    def test_f1_in_valid_range(self, module_path):
        run_probe = self._get_run_probe(module_path)
        X, labels, splits = _make_probe_inputs()
        result = run_probe(X, labels, splits)
        assert 0.0 <= result["macro_f1_mean"] <= 1.0

    def test_empty_splits_returns_empty(self, module_path):
        run_probe = self._get_run_probe(module_path)
        X, labels, _ = _make_probe_inputs()
        result = run_probe(X, labels, [])
        assert result == {}

    def test_auroc_keys_present(self, module_path):
        run_probe = self._get_run_probe(module_path)
        X, labels, splits = _make_probe_inputs()
        result = run_probe(X, labels, splits)
        for cls in ["GOF", "DN", "LOF"]:
            assert f"auroc_{cls}_mean" in result

    def test_single_class_fold_skipped(self, module_path):
        """A fold where train has only one class is silently skipped."""
        run_probe = self._get_run_probe(module_path)
        rng = np.random.RandomState(0)
        n = 60
        X = rng.randn(n, 8).astype(np.float32)
        labels = np.array(["GOF"] * n)  # single class → all folds skipped
        fold = n // 2
        splits = [(np.arange(fold), np.arange(fold, n))]
        result = run_probe(X, labels, splits)
        assert result == {}
