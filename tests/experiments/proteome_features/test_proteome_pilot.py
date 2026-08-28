"""
Tests for the Stage 0 proteome pilot (proteome_pilot.py).

Invariants:
- AR and unlabeled genes are dropped; HI collapses to LOF. Nothing else is remapped.
- The gnomAD parser keeps one row per gene, preferring the MANE transcript and
  then the larger lof.exp. A missing lof.exp never wins a tie-break, because
  coercing it to 0.0 would let an incomplete row displace a complete one.
- Unparseable or NA cells become None, never 0.0.
- A gnomAD file missing a required column exits rather than proceeding with a
  partial feature set.
- The feature matrix carries a missingness indicator per feature column, so an
  imputed cell stays distinguishable from a measured one.
- Genes without a Pfam family are dropped before the family split, and the
  feature matrix, labels, and groups stay row-aligned to the surviving rows.
- The scaler is fitted once per fold on training rows only.
- An arm whose splits fail validation reports None metrics with an unscorable
  status, never a number.
"""

import csv
import json

import numpy as np
import pytest

from esm2_mech.experiments.proteome_features import proteome_pilot
from esm2_mech.experiments.proteome_features.proteome_pilot import (
    CLASSES,
    FEATURE_COLS,
    build_feature_table,
    evaluate_model,
    fetch_paralog_count,
    load_gene_labels,
    majority_baseline,
    parse_gnomad_constraint,
    save_feature_table,
)

GNOMAD_HEADER = [
    "gene",
    "transcript",
    "mane_select",
    "lof.exp",
    "lof.pLI",
    "lof.oe_ci.upper",
    "mis.z_score",
]


def _write_tsv(path, header, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _write_gnomad(tmp_path, rows, header=GNOMAD_HEADER):
    return _write_tsv(tmp_path / "constraint.tsv", header, rows)


class TestLoadGeneLabels:

    def test_collapses_hi_and_drops_ar_and_blank(self, tmp_path):
        path = _write_tsv(
            tmp_path / "genes.tsv",
            ["gene", "mechanism"],
            [
                ["GENE_GOF", "GOF"],
                ["GENE_DN", "DN"],
                ["GENE_LOF", "LOF"],
                ["GENE_HI", "HI"],
                ["GENE_AR", "AR"],
                ["GENE_BLANK", ""],
            ],
        )

        labels = load_gene_labels(path)

        assert labels == {
            "GENE_GOF": "GOF",
            "GENE_DN": "DN",
            "GENE_LOF": "LOF",
            "GENE_HI": "LOF",
        }

    def test_unknown_mechanism_is_not_silently_kept(self, tmp_path):
        path = _write_tsv(
            tmp_path / "genes.tsv",
            ["gene", "mechanism"],
            [["GENE_A", "GOF"], ["GENE_WEIRD", "SOMETHING_ELSE"]],
        )

        labels = load_gene_labels(path)

        assert "GENE_WEIRD" not in labels


class TestParseGnomadConstraint:

    def test_na_cells_become_none_not_zero(self, tmp_path):
        path = _write_gnomad(
            tmp_path, [["GENE_A", "T1", "TRUE", "100", "NA", "", "nan"]]
        )

        parsed = parse_gnomad_constraint(path)

        assert parsed["GENE_A"]["pLI"] is None
        assert parsed["GENE_A"]["LOEUF"] is None
        assert parsed["GENE_A"]["mis_z"] is None

    def test_mane_row_wins_over_non_mane_in_either_file_order(self, tmp_path):
        mane_last = _write_gnomad(
            tmp_path,
            [
                ["GENE_A", "T1", "FALSE", "500", "0.10", "1.0", "1.0"],
                ["GENE_A", "T2", "TRUE", "5", "0.90", "0.2", "3.0"],
            ],
        )
        assert parse_gnomad_constraint(mane_last)["GENE_A"]["pLI"] == 0.90

        mane_first = _write_tsv(
            tmp_path / "constraint_reordered.tsv",
            GNOMAD_HEADER,
            [
                ["GENE_A", "T2", "TRUE", "5", "0.90", "0.2", "3.0"],
                ["GENE_A", "T1", "FALSE", "500", "0.10", "1.0", "1.0"],
            ],
        )
        assert parse_gnomad_constraint(mane_first)["GENE_A"]["pLI"] == 0.90

    def test_larger_lof_exp_wins_when_mane_status_ties(self, tmp_path):
        path = _write_gnomad(
            tmp_path,
            [
                ["GENE_A", "T1", "FALSE", "10", "0.10", "1.0", "1.0"],
                ["GENE_A", "T2", "FALSE", "50", "0.90", "0.2", "3.0"],
            ],
        )

        assert parse_gnomad_constraint(path)["GENE_A"]["pLI"] == 0.90

    def test_missing_lof_exp_never_displaces_a_row_that_has_one(self, tmp_path):
        """A missing lof.exp read as 0.0 would still lose here, but read as a
        real value it could win any tie-break against a genuinely smaller
        expectation. The incomplete row must never replace the complete one."""
        path = _write_gnomad(
            tmp_path,
            [
                ["GENE_A", "T1", "FALSE", "0.5", "0.10", "1.0", "1.0"],
                ["GENE_A", "T2", "FALSE", "NA", "0.90", "0.2", "3.0"],
            ],
        )

        assert parse_gnomad_constraint(path)["GENE_A"]["pLI"] == 0.10

    def test_row_with_lof_exp_replaces_a_row_without_one(self, tmp_path):
        path = _write_gnomad(
            tmp_path,
            [
                ["GENE_A", "T1", "FALSE", "NA", "0.10", "1.0", "1.0"],
                ["GENE_A", "T2", "FALSE", "0.5", "0.90", "0.2", "3.0"],
            ],
        )

        assert parse_gnomad_constraint(path)["GENE_A"]["pLI"] == 0.90

    def test_short_rows_are_skipped_not_index_errors(self, tmp_path):
        path = _write_gnomad(
            tmp_path,
            [
                ["GENE_A", "T1", "TRUE", "10", "0.5", "0.4", "2.0"],
                ["GENE_TRUNCATED", "T1"],
            ],
        )

        parsed = parse_gnomad_constraint(path)

        assert "GENE_A" in parsed
        assert "GENE_TRUNCATED" not in parsed

    def test_missing_required_column_exits(self, tmp_path):
        header = [column for column in GNOMAD_HEADER if column != "mis.z_score"]
        path = _write_tsv(
            tmp_path / "constraint.tsv",
            header,
            [["GENE_A", "T1", "TRUE", "10", "0.5", "0.4"]],
        )

        with pytest.raises(SystemExit) as excinfo:
            parse_gnomad_constraint(path)

        assert excinfo.value.code == 2


class TestFetchParalogCount:

    def test_cached_count_is_returned_without_a_network_call(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(proteome_pilot, "PARALOG_CACHE", tmp_path)
        (tmp_path / "GENE_A.json").write_text(json.dumps({"paralog_count": 7}))

        def _forbidden(*args, **kwargs):
            raise AssertionError("cached gene must not trigger a request")

        monkeypatch.setattr(proteome_pilot.urllib.request, "urlopen", _forbidden)

        assert fetch_paralog_count("GENE_A") == 7

    def test_failed_fetch_records_the_error_alongside_the_null(
        self, tmp_path, monkeypatch
    ):
        """A None in the cache must stay distinguishable from a real zero-paralog
        answer, so the failure reason is written next to it."""
        monkeypatch.setattr(proteome_pilot, "PARALOG_CACHE", tmp_path)

        def _fail(*args, **kwargs):
            raise TimeoutError("connection timed out")

        monkeypatch.setattr(proteome_pilot.urllib.request, "urlopen", _fail)

        assert fetch_paralog_count("GENE_B") is None
        cached = json.loads((tmp_path / "GENE_B.json").read_text())
        assert cached["paralog_count"] is None
        assert "connection timed out" in cached["error"]


def _feature_inputs():
    labels = {
        "GENE_A": "GOF",
        "GENE_B": "DN",
        "GENE_C": "LOF",
        "GENE_NOFAM": "GOF",
    }
    families = {
        "GENE_A": "PF0001",
        "GENE_B": "PF0002",
        "GENE_C": "PF0001",
        "GENE_NOFAM": None,
    }
    gnomad = {
        "GENE_A": {"pLI": 0.9, "LOEUF": 0.2, "mis_z": 3.0},
        "GENE_B": {"pLI": 0.1, "LOEUF": 1.4, "mis_z": 0.5},
        "GENE_C": {"pLI": None, "LOEUF": 0.8, "mis_z": 1.5},
    }
    paralogs = {"GENE_A": 3, "GENE_B": 0, "GENE_C": 11}
    return labels, families, gnomad, paralogs


class TestBuildFeatureTable:

    def test_drops_genes_without_a_family_and_stays_row_aligned(self):
        rows, X, y, groups = build_feature_table(*_feature_inputs())

        assert [row["gene"] for row in rows] == ["GENE_A", "GENE_B", "GENE_C"]
        assert X.shape[0] == len(rows) == len(y) == len(groups)
        assert list(y) == ["GOF", "DN", "LOF"]
        assert list(groups) == ["PF0001", "PF0002", "PF0001"]

    def test_missingness_indicators_are_appended_for_every_feature(self):
        _rows, X, _y, _groups = build_feature_table(*_feature_inputs())

        assert X.shape[1] == 2 * len(FEATURE_COLS)
        pli_column = FEATURE_COLS.index("pLI")
        indicator = X[:, len(FEATURE_COLS) + pli_column]
        # GENE_C has no pLI; the other two do.
        assert list(indicator) == [0.0, 0.0, 1.0]

    def test_missing_cell_is_filled_with_the_observed_median_not_zero(self):
        _rows, X, _y, _groups = build_feature_table(*_feature_inputs())

        pli_column = FEATURE_COLS.index("pLI")
        filled = X[2, pli_column]
        assert filled == pytest.approx(np.median([0.9, 0.1]))
        assert filled != 0.0

    def test_entirely_missing_column_is_flagged_on_every_row(self):
        labels, families, gnomad, _paralogs = _feature_inputs()
        paralogs = {"GENE_A": None, "GENE_B": None, "GENE_C": None}

        _rows, X, _y, _groups = build_feature_table(
            labels, families, gnomad, paralogs
        )

        paralog_column = FEATURE_COLS.index("paralog_count")
        indicator = X[:, len(FEATURE_COLS) + paralog_column]
        assert list(indicator) == [1.0, 1.0, 1.0]


def test_save_feature_table_writes_blanks_for_missing_values(tmp_path):
    rows, _X, _y, _groups = build_feature_table(*_feature_inputs())
    path = tmp_path / "features.tsv"

    save_feature_table(rows, path)

    with open(path, newline="") as handle:
        written = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["gene"] for row in written] == ["GENE_A", "GENE_B", "GENE_C"]
    assert written[2]["pLI"] == ""
    assert "None" not in written[2].values()


def _cv_cohort(n_families=15, seed=0):
    """Every family carries all three classes, with LOF in the majority so the
    training-frequency reference has an unambiguous most-common label."""
    rng = np.random.RandomState(seed)
    labels, groups = [], []
    for family_index in range(n_families):
        for class_name in list(CLASSES) + ["LOF"]:
            labels.append(class_name)
            groups.append(f"PF{family_index:04d}")
    y = np.array(labels, dtype=object)
    group_array = np.array(groups)
    X = rng.randn(len(y), 2 * len(FEATURE_COLS))
    return X, y, group_array


class TestEvaluateModel:

    @pytest.mark.slow
    def test_scorable_cohort_reports_success_and_every_declared_class(self):
        from sklearn.linear_model import LogisticRegression

        X, y, groups = _cv_cohort()

        result = evaluate_model(
            LogisticRegression(max_iter=500), X, y, groups, n_folds=5, seed=0
        )

        assert result["status"] == "success"
        assert result["n_test"] == len(y)
        assert set(result["per_class_auroc"]) == set(CLASSES)
        assert result["macro_f1"] is not None

    @pytest.mark.slow
    def test_scaler_is_fitted_once_per_fold_on_training_rows_only(self, monkeypatch):
        """Fitting a single scaler before splitting would leak test-fold means and
        variances into training."""
        import sklearn.preprocessing
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler as RealScaler

        X, y, groups = _cv_cohort()
        fitted_row_counts = []

        class CountingScaler(RealScaler):
            def fit(self, X_fit, y_fit=None, **kwargs):
                fitted_row_counts.append(len(X_fit))
                return super().fit(X_fit, y_fit, **kwargs)

        monkeypatch.setattr(sklearn.preprocessing, "StandardScaler", CountingScaler)

        evaluate_model(
            LogisticRegression(max_iter=500), X, y, groups, n_folds=5, seed=0
        )

        assert len(fitted_row_counts) == 5
        assert all(count < len(y) for count in fitted_row_counts)

    def test_unscorable_splits_report_none_metrics(self):
        from sklearn.linear_model import LogisticRegression

        # Two families cannot fill five family-disjoint folds.
        X, y, groups = _cv_cohort(n_families=2)

        result = evaluate_model(
            LogisticRegression(max_iter=500), X, y, groups, n_folds=5, seed=0
        )

        assert result["status"] == "unscorable"
        assert result["macro_f1_mean"] is None
        assert result["split_validation"]["status"] != "valid"


class TestMajorityBaseline:

    def test_reports_a_reference_score_on_a_scorable_cohort(self):
        _X, y, groups = _cv_cohort()

        result = majority_baseline(y, groups, n_folds=5, seed=0)

        assert result["status"] == "success"
        assert result["macro_f1"] is not None

    def test_unscorable_splits_report_none_metrics(self):
        _X, y, groups = _cv_cohort(n_families=2)

        result = majority_baseline(y, groups, n_folds=5, seed=0)

        assert result["status"] == "unscorable"
        assert result["macro_f1_mean"] is None
