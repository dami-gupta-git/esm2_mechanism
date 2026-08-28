"""
Tests for T2 per-gene scoring and the T4 V2 feature-class ablation
(per_gene_ablation.py).

Invariants:
- Every name the module uses at runtime is imported. The NaN-native tree
  classifier is used in all three arms; without its import both T2 and T4 fail
  with a NameError on the first fold.
- Ablation drops a feature and its derived columns, and nothing that merely
  contains the feature name as a substring.
- Label encoding follows the declared class order, not sklearn's alphabetical
  order, and decoding is its exact inverse.
- T2 scores one row per test gene, so a gene with many variants does not
  outvote a gene with one.
- The proteome block keeps its missing cells: nothing is imputed before the
  split, and rows are not dropped for being incomplete.
- A fold set that cannot supply enough variant rows reports unscorable with
  None metrics rather than a number computed from too little data.
"""

import numpy as np
import pandas as pd
import pytest

from esm2_mech.experiments.proteome_features import per_gene_ablation
from esm2_mech.experiments.proteome_features.per_gene_ablation import (
    CLASSES,
    FEATURE_CLASSES,
    _decode,
    _encode,
    get_drop_indices,
    run_per_gene_cv,
    run_v2_ablation,
)
from esm2_mech.utils.constants import N_FOLDS

PROTEOME_COLUMNS = [
    "pLI",
    "LOEUF",
    "mis_z",
    "pLI_zscore",
    "paralog_count",
    "tissue_specificity_tau",
    "log_abundance_ppm",
    "PPI_degree",
    "HI_score",
    "TS_score",
    "unrelated_feature",
]


def test_module_imports_the_classifier_it_uses():
    """All three arms construct this estimator; a missing import is a NameError
    on the first fold, after the data load has already run."""
    assert hasattr(per_gene_ablation, "HistGradientBoostingClassifier")


class TestGetDropIndices:

    def test_drops_the_named_feature_and_its_derived_columns(self):
        drop = get_drop_indices(PROTEOME_COLUMNS, FEATURE_CLASSES["constraint"])

        dropped_names = {PROTEOME_COLUMNS[index] for index in drop}
        assert dropped_names == {"pLI", "LOEUF", "mis_z", "pLI_zscore"}

    def test_does_not_drop_a_column_that_only_contains_the_name(self):
        columns = ["pLI", "log_pLI", "pLI_rank", "otherpLI"]

        drop = get_drop_indices(columns, ["pLI"])

        assert {columns[index] for index in drop} == {"pLI", "pLI_rank"}

    def test_returns_no_indices_when_the_feature_is_absent(self):
        assert get_drop_indices(["a", "b"], ["pLI"]) == []


class TestLabelEncoding:

    def test_encoding_follows_declared_class_order(self):
        encoded = _encode(np.array(CLASSES))

        assert list(encoded) == list(range(len(CLASSES)))

    def test_decode_inverts_encode(self):
        labels = np.array([CLASSES[2], CLASSES[0], CLASSES[1], CLASSES[2]])

        assert list(_decode(_encode(labels))) == list(labels)

    def test_encoding_is_not_alphabetical(self):
        """LabelEncoder would sort these; the declared order must win, or the
        probability columns line up with the wrong classes."""
        assert list(CLASSES) != sorted(CLASSES)
        assert _decode(np.array([0]))[0] == CLASSES[0]


def _gene_level_cohort(n_families=15, genes_per_family=4, seed=0, with_nan=True):
    """Gene-level proteome matrix with real missing cells, one family per group."""
    rng = np.random.RandomState(seed)
    genes, families, labels = [], [], []
    for family_index in range(n_families):
        for gene_index in range(genes_per_family):
            genes.append(f"GENE_{family_index}_{gene_index}")
            families.append(f"PF{family_index:04d}")
            # LOF in the majority so the class distribution is not tied.
            labels.append(CLASSES[gene_index] if gene_index < len(CLASSES) else "LOF")
    X = rng.randn(len(genes), len(PROTEOME_COLUMNS)).astype(np.float32)
    if with_nan:
        X[::5, 0] = np.nan
        X[::7, 4] = np.nan
    return (
        X,
        np.array(labels, dtype=object),
        np.array(families),
        np.array(genes),
    )


class TestRunV2Ablation:

    @pytest.mark.slow
    def test_reports_full_and_every_feature_class(self):
        X, y, groups, _genes = _gene_level_cohort()

        result = run_v2_ablation(X, y, groups, PROTEOME_COLUMNS, seed=0)

        assert set(result) == {"FULL", *FEATURE_CLASSES}
        assert result["FULL"]["status"] == "success"

    @pytest.mark.slow
    def test_delta_f1_is_full_minus_ablated(self):
        X, y, groups, _genes = _gene_level_cohort()

        result = run_v2_ablation(X, y, groups, PROTEOME_COLUMNS, seed=0)

        full_f1 = result["FULL"]["macro_f1_mean"]
        for class_name in FEATURE_CLASSES:
            arm = result[class_name]
            assert arm["delta_f1"] == pytest.approx(full_f1 - arm["macro_f1_mean"])

    @pytest.mark.slow
    def test_feature_counts_account_for_every_column(self):
        X, y, groups, _genes = _gene_level_cohort()

        result = run_v2_ablation(X, y, groups, PROTEOME_COLUMNS, seed=0)

        for class_name in FEATURE_CLASSES:
            arm = result[class_name]
            assert (
                arm["n_features_dropped"] + arm["n_features_kept"]
                == len(PROTEOME_COLUMNS)
            )
        assert result["constraint"]["n_features_dropped"] == 4

    @pytest.mark.slow
    def test_missing_cells_are_kept_rather_than_imputed_or_dropped(self):
        """The tree model consumes NaN directly. Imputing a median over the whole
        matrix would leak test-fold statistics into training."""
        X, y, groups, _genes = _gene_level_cohort(with_nan=True)
        assert np.isnan(X).any()

        result = run_v2_ablation(X, y, groups, PROTEOME_COLUMNS, seed=0)

        assert result["FULL"]["status"] == "success"
        assert result["FULL"]["completed_folds"] == N_FOLDS

    def test_unscorable_splits_report_none_not_zero(self):
        """An undefined delta must be None. Zero would read as 'this feature
        class does not matter', and NaN would poison the across-seed mean."""
        X, y, groups, _genes = _gene_level_cohort(n_families=2)

        result = run_v2_ablation(X, y, groups, PROTEOME_COLUMNS, seed=0)

        assert result["FULL"]["status"] == "unscorable"
        assert result["FULL"]["macro_f1_mean"] is None
        for class_name in FEATURE_CLASSES:
            assert result[class_name]["delta_f1"] is None
            assert result[class_name]["delta_auroc_DN"] is None


def _variant_cohort(genes, labels_by_gene, variant_counts, seed=0, dim=8):
    """Expand genes into variant rows, giving some genes many more than others."""
    rng = np.random.RandomState(seed)
    variant_genes, variant_labels = [], []
    for gene in genes:
        for _ in range(variant_counts[gene]):
            variant_genes.append(gene)
            variant_labels.append(labels_by_gene[gene])
    delta = rng.randn(len(variant_genes), dim).astype(np.float32)
    return np.array(variant_genes), np.array(variant_labels, dtype=object), delta


def _per_gene_inputs(seed=0, lopsided=True):
    X_gene, y_gene, families, genes = _gene_level_cohort(seed=seed)
    pfam_map = dict(zip(genes, families))
    labels_by_gene = dict(zip(genes, y_gene))
    if lopsided:
        counts = {
            gene: (30 if index % 4 == 0 else 1) for index, gene in enumerate(genes)
        }
    else:
        counts = {gene: 3 for gene in genes}
    variant_genes, variant_labels, delta = _variant_cohort(
        list(genes), labels_by_gene, counts, seed=seed
    )
    X_prot_var = np.tile(
        X_gene[[list(genes).index(gene) for gene in variant_genes]], (1, 1)
    )
    gene_level_df = pd.DataFrame({"gene": genes, "mech3": y_gene})
    return {
        "delta": delta,
        "X_prot_var": X_prot_var,
        "var_labels": variant_labels,
        "var_genes": variant_genes,
        "gene_level_df": gene_level_df,
        "X_prot_gene": X_gene,
        "pfam_map": pfam_map,
        "seed": 0,
    }


class TestRunPerGeneCv:

    @pytest.mark.slow
    def test_scores_one_row_per_test_gene_regardless_of_variant_count(
        self, monkeypatch
    ):
        """A gene contributing thirty variants must weigh the same as a gene
        contributing one, which is the point of re-scoring per gene."""
        inputs = _per_gene_inputs(lopsided=True)
        scored_row_counts = []
        real_compute_metrics = per_gene_ablation.compute_metrics

        def _recording_compute_metrics(y_true, y_pred, y_proba, classes):
            scored_row_counts.append(len(y_true))
            return real_compute_metrics(y_true, y_pred, y_proba, classes)

        monkeypatch.setattr(
            per_gene_ablation, "compute_metrics", _recording_compute_metrics
        )

        result = run_per_gene_cv(**inputs)

        assert result["V1_per_gene"]["completed_folds"] == N_FOLDS
        n_test_genes_per_fold = len(inputs["gene_level_df"]) / N_FOLDS
        assert max(scored_row_counts) <= len(inputs["gene_level_df"])
        assert max(scored_row_counts) == pytest.approx(n_test_genes_per_fold, abs=4)
        # The variant rows far outnumber the genes; no arm may score them directly.
        assert len(inputs["var_genes"]) > 2 * len(inputs["gene_level_df"])

    @pytest.mark.slow
    def test_reports_all_three_arms(self):
        result = run_per_gene_cv(**_per_gene_inputs(lopsided=False))

        assert set(result) == {"V1_per_gene", "V2_per_gene", "V3_per_gene"}
        for arm in result.values():
            assert arm["macro_f1_mean"] is not None

    def test_too_few_variant_rows_is_unscorable_in_every_arm(self):
        inputs = _per_gene_inputs(lopsided=False)
        # One variant row per gene leaves folds below the preflight minimum.
        keep = np.zeros(len(inputs["var_genes"]), dtype=bool)
        keep[:6] = True
        inputs["var_genes"] = inputs["var_genes"][keep]
        inputs["var_labels"] = inputs["var_labels"][keep]
        inputs["delta"] = inputs["delta"][keep]
        inputs["X_prot_var"] = inputs["X_prot_var"][keep]

        result = run_per_gene_cv(**inputs)

        for arm in result.values():
            assert arm["status"] == "unscorable"
            assert arm["macro_f1_mean"] is None
        assert result["V1_per_gene"]["preflight_failures"]
