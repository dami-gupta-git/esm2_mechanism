"""Probe runners: logistic regression and MLP CV over pre-computed splits."""

from __future__ import annotations

import functools

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from esm2_mech.utils.classification import validate_classes, validate_observed_labels
from esm2_mech.utils.metrics import (
    aggregate_folds,
    align_proba,
    compute_metrics,
    empty_aggregate_metrics,
    mean_std_n,
    standardize,
)

print = functools.partial(print, flush=True)


def _per_gene_f1(
    y_true: np.ndarray,
    proba: np.ndarray,
    genes: np.ndarray,
    classes: list[str],
) -> float:
    """Aggregate per-variant probabilities to per-gene predictions and compute macro-F1.

    `classes` must be the same label list `proba`'s columns are aligned to — the
    caller's, not MECHANISM_CLASSES, since the multiclass runners accept an
    arbitrary class list (e.g. the 4-class GOF/DN/HI/AR probe).
    """
    declared = validate_classes(classes)
    validate_observed_labels(y_true, declared, "gene-level y_true")
    unique = sorted(set(genes.tolist()), key=str)
    y_g, p_g = [], []
    for gene in unique:
        mask = genes == gene
        gene_labels = y_true[mask]
        observed = np.unique(gene_labels)
        if len(observed) != 1:
            raise ValueError(
                f"gene {gene!r} has inconsistent mechanism labels: {observed.tolist()!r}"
            )
        true_label = observed[0]
        pred_label = declared[int(proba[mask].mean(0).argmax())]
        y_g.append(true_label)
        p_g.append(pred_label)
    identity = np.eye(len(declared), dtype=float)
    predicted_proba = identity[[declared.index(value) for value in p_g]]
    return compute_metrics(
        np.asarray(y_g), np.asarray(p_g), predicted_proba, declared
    )["macro_f1"]


def _record_per_gene_f1(
    pg_f1s: list,
    y_true: np.ndarray,
    proba: np.ndarray,
    genes_sub: np.ndarray,
    classes: list[str],
) -> str:
    """Compute the fold's per-gene macro-F1, append it to pg_f1s, return the log suffix.

    `genes_sub` is the gene-id array for the test rows (already sliced). Shared by the
    three multiclass runners, which each computed and accumulated this identically.
    """
    pg = _per_gene_f1(y_true, proba, genes_sub, classes)
    pg_f1s.append(pg)
    return f"  per_gene_f1={pg:.3f}"


def _add_per_gene_f1(agg: dict, pg_f1s: list) -> None:
    """Add per_gene_f1 mean/std to `agg` in place when any per-gene F1 was recorded."""
    if pg_f1s:
        agg["per_gene_f1_mean"] = float(np.mean(pg_f1s))
        agg["per_gene_f1_std"] = float(np.std(pg_f1s))


def _log_fold(label: str, fold_i: int, fm: dict, classes: list[str], pg_str: str) -> None:
    """Print one per-fold line: macro-F1, optional per-gene F1, and per-class AUROC.

    `fm` is a compute_metrics dict (has macro_f1 and per_class_auroc). Shared by
    run_logreg_cv and run_mlp_cv, which logged this identically.
    """
    auroc_str = "  ".join(
        (
            f"{cls}={fm['per_class_auroc'].get(cls, float('nan')):.3f}"
            if fm["per_class_auroc"].get(cls) is not None
            else f"{cls}=NA"
        )
        for cls in classes
    )
    print(f"    [{label}] Fold {fold_i+1}: macro_f1={fm['macro_f1']:.3f}{pg_str}  " + auroc_str)


class _OofCollector:
    """Accumulate per-fold out-of-fold test predictions for dependency-aware inference.

    Shared by run_logreg_cv and run_mlp_cv, which both gather the same five aligned
    arrays per fold (y_true, proba aligned to `classes`, gene ids, original row ids,
    fold index) and concatenate them into one OOF dict, or None if no fold contributed.

    The fold index is mandatory. Each fold is fitted independently, so its
    probabilities are on their own scale; a consumer that ranks the concatenation as
    one list compares scores that were never comparable, which pushes weak-signal
    AUROCs below 0.5. Downstream ranking metrics score within fold and average, and
    they can only do that if the fold survives collection.
    """

    def __init__(
        self,
        classes: list[object],
        eligible_rows: np.ndarray,
        requested_folds: int,
    ) -> None:
        self._classes = validate_classes(classes)
        self._eligible_rows = np.asarray(eligible_rows, dtype=int)
        self._requested_folds = int(requested_folds)
        self._y: list = []
        self._proba: list = []
        self._genes: list = []
        self._rows: list = []
        self._folds: list = []

    def add(
        self,
        y_te: np.ndarray,
        proba: np.ndarray,
        genes_te: np.ndarray | None,
        te: np.ndarray,
        fold_i: int,
    ) -> None:
        te = np.asarray(te)
        if proba.shape != (len(te), len(self._classes)):
            raise ValueError(
                f"OOF probability shape {proba.shape} does not match "
                f"({len(te)}, {len(self._classes)})"
            )
        if not np.isfinite(proba).all():
            raise ValueError(f"OOF probabilities contain non-finite values in fold {fold_i}")
        validate_observed_labels(y_te, self._classes, "OOF y_true")
        self._y.append(y_te)
        self._proba.append(proba)
        if genes_te is not None:
            self._genes.append(genes_te)
        self._rows.append(te)
        self._folds.append(np.full(len(te), int(fold_i), dtype=int))

    def finalize(self) -> dict | None:
        if not self._y:
            raise ValueError("cannot finalize an empty OOF record")
        row_ids = np.concatenate(self._rows)
        if len(np.unique(row_ids)) != len(row_ids):
            unique_rows, row_counts = np.unique(row_ids, return_counts=True)
            duplicates = unique_rows[row_counts > 1]
            raise ValueError(f"OOF record contains duplicate rows: {duplicates[:20].tolist()}")
        missing = sorted(set(self._eligible_rows.tolist()) - set(row_ids.tolist()))
        unexpected = sorted(set(row_ids.tolist()) - set(self._eligible_rows.tolist()))
        if missing or unexpected:
            raise ValueError(
                "OOF row coverage mismatch: "
                f"missing={missing[:20]!r}, unexpected={unexpected[:20]!r}"
            )
        folds = np.concatenate(self._folds)
        observed_folds = sorted(np.unique(folds).tolist())
        expected_folds = list(range(self._requested_folds))
        if observed_folds != expected_folds:
            raise ValueError(
                f"OOF fold labels {observed_folds!r} do not match {expected_folds!r}"
            )
        order = np.argsort(row_ids, kind="stable")
        return {
            "y_true": np.concatenate(self._y)[order],
            "proba": np.concatenate(self._proba)[order],
            "genes": np.concatenate(self._genes)[order] if self._genes else None,
            "row_ids": row_ids[order],
            "folds": folds[order],
            "classes": list(self._classes),
        }


def require_no_nan(X: np.ndarray, caller: str) -> None:
    """Raise if X contains NaN, naming the two sanctioned ways to handle it.

    sklearn's own "Input contains NaN" is raised deep inside a fit call and says
    nothing about which arm fed it or what to do. This fails at the probe
    boundary instead, because the wrong fix (imputing to make the error go away)
    is the bug this guard exists to prevent: a value filled in before the CV
    split leaks test-fold statistics into training and is indistinguishable from
    a real measurement afterwards.

    The two correct responses, per feature block:
      - Sparse block (proteome / Badonyi features, or anything concatenated with
        one): use run_histgb_cv, which consumes NaN natively — no row is dropped.
      - A single scalar feature: restrict to the observed subset with
        data.observed_rows_mask AND recompute the CV splits on that subset.
    """
    n_nan = int(np.isnan(X).sum())
    if not n_nan:
        return
    n_rows = int(np.isnan(X).any(axis=1).sum())
    raise ValueError(
        f"{caller} received {n_nan} NaN cells across {n_rows}/{len(X)} rows. "
        f"{caller} standardizes and fits a model that cannot consume missing "
        f"values. Do NOT impute to silence this. Either use run_histgb_cv "
        f"(NaN-native, keeps every row — correct for the proteome/Badonyi "
        f"blocks and anything concatenated with them), or restrict to the "
        f"observed subset with data.observed_rows_mask and recompute the CV "
        f"splits on that subset."
    )


def _unscorable_result(split_contract: dict, classes: list[object]) -> dict:
    result = empty_aggregate_metrics(
        classes,
        split_contract["requested_folds"],
        "split_validation_failed",
    )
    result.update(
        {
            "status": "unscorable",
            "classes": list(classes),
            "eligible_rows": split_contract["eligible_rows"],
            "out_of_fold_rows": 0,
            "held_out_unit": split_contract.get("held_out_unit"),
            "group_count": split_contract.get("group_count"),
            "split_validation": split_contract,
        }
    )
    return result


def _failed_result(
    split_contract: dict,
    classes: list[object],
    completed_folds: int,
    failed_fold: int,
    error: Exception,
) -> dict:
    result = empty_aggregate_metrics(
        classes,
        split_contract["requested_folds"],
        "runtime_failure",
    )
    result.update(
        {
            "status": "failed",
            "classes": list(classes),
            "eligible_rows": split_contract["eligible_rows"],
            "out_of_fold_rows": 0,
            "held_out_unit": split_contract.get("held_out_unit"),
            "group_count": split_contract.get("group_count"),
            "completed_folds": int(completed_folds),
            "failed_fold": int(failed_fold),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "split_validation": split_contract,
        }
    )
    return result


def _run_multiclass_cv(
    fit_proba_fn,
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple],
    classes: list[str],
    split_contract: dict,
    seed: int,
    genes: np.ndarray | None,
    label: str,
    return_oof: bool,
    compute_per_gene: bool = True,
):
    """Shared multiclass CV body: guard folds → fit → align proba → aggregate.

    fit_proba_fn(X_tr, y_tr, X_te, seed) -> (raw_proba, clf_classes), where
    clf_classes are the string labels of raw_proba's columns. Shared by
    run_logreg_cv (per-fold StandardScaler + LogReg) and run_histgb_cv
    (NaN-native gradient boosting, unscaled), which differ only in that step.
    """
    declared = validate_classes(classes)
    if return_oof and genes is None:
        raise ValueError("genes are required when return_oof=True")
    if split_contract.get("classes") != declared:
        raise ValueError("runner classes do not match the split contract")
    if split_contract.get("status") != "valid":
        result = _unscorable_result(split_contract, declared)
        return (result, None) if return_oof else result
    fold_results, pg_f1s = [], []
    oof = _OofCollector(
        declared,
        np.asarray(split_contract["eligible_row_ids"], dtype=int),
        split_contract["requested_folds"],
    )
    for fold_i, (tr, te) in enumerate(splits):
        try:
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y[tr], y[te]
            raw_proba, clf_classes = fit_proba_fn(X_tr, y_tr, X_te, seed)
            proba = align_proba(
                raw_proba,
                clf_classes,
                declared,
                allow_missing_classes=split_contract["allow_missing_classifier_classes"],
            )
            pred = np.array([declared[idx] for idx in proba.argmax(axis=1)])
            fm = compute_metrics(y_te, pred, proba, declared)
        except Exception as error:
            result = _failed_result(
                split_contract, declared, len(fold_results), fold_i, error
            )
            return (result, None) if return_oof else result
        fold_results.append(fm)
        oof.add(y_te, proba, None if genes is None else genes[te], te, fold_i)

        pg_str = ""
        if genes is not None and compute_per_gene:
            # `classes` (not the MECHANISM_CLASSES default) — proba's columns are
            # aligned to the caller's class list, which may not be the 3-class one.
            pg_str = _record_per_gene_f1(pg_f1s, y_te, proba, genes[te], declared)

        _log_fold(label, fold_i, fm, declared, pg_str)

    agg = aggregate_folds(
        fold_results, declared, split_contract["requested_folds"]
    )
    agg.update(
        {
            "status": "success",
            "classes": declared,
            "eligible_rows": split_contract["eligible_rows"],
            "out_of_fold_rows": split_contract["eligible_rows"],
            "held_out_unit": split_contract.get("held_out_unit"),
            "group_count": split_contract.get("group_count"),
            "split_validation": split_contract,
        }
    )
    _add_per_gene_f1(agg, pg_f1s)
    if return_oof:
        return agg, oof.finalize()
    return agg


def run_logreg_cv(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple],
    classes: list[str],
    split_contract: dict,
    seed: int = 42,
    genes: np.ndarray | None = None,
    label: str = "",
    return_oof: bool = False,
    prescaled: bool = False,
    compute_per_gene: bool = True,
):
    """Run LogReg + StandardScaler over pre-computed splits, return aggregated metrics.

    X must not contain NaN — LogReg cannot fit on missing values. Restrict to the
    observed subset with data.observed_rows_mask and recompute `splits` on that
    subset, or use run_histgb_cv when complete-case would drop too many rows.

    genes : if provided, also computes per_gene_f1 per fold
    label : prefix for per-fold log lines
    prescaled : set True when X has already been standardized by the caller AND a
        direction has been projected out of it. The per-fold StandardScaler rescales
        each column independently, which reintroduces variance along a removed
        direction and silently undoes the projection; with prescaled=True the
        scaler is skipped so the projection survives to the classifier. Callers
        must standardize once up front and make the projection the last transform.
    min_train_classes : minimum distinct classes a fold's train split must have to
        be kept. Defaults to n_classes (the historical behaviour — every class must
        be present). Pass 2 to keep folds where a rare class falls entirely in test
        (a classifier only needs two classes to fit); this also lets a caller make
        the probe's fold set match a separately-computed chance floor that skips on
        the same condition.
    return_oof : if True, return (agg, oof) where oof collects the out-of-fold test
        predictions for dependency-aware inference (cluster bootstrap / permutation):
        {"y_true", "proba" (aligned to `classes`), "genes", "row_ids", "folds"}, or None if no fold was
        scorable. `genes` must be provided for oof to carry gene ids. Default False
        keeps the bare-`agg` return for existing callers.
    """
    require_no_nan(X, "run_logreg_cv")

    def _fit(X_tr, y_tr, X_te, fold_seed):
        if prescaled:
            X_tr_s, X_te_s = X_tr, X_te
        else:
            sc = StandardScaler().fit(X_tr)
            X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)
        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=fold_seed
        )
        clf.fit(X_tr_s, y_tr)
        return clf.predict_proba(X_te_s), clf.classes_

    return _run_multiclass_cv(
        _fit,
        X,
        y,
        splits,
        classes,
        split_contract,
        seed,
        genes,
        label,
        return_oof,
        compute_per_gene,
    )


def run_logreg_pca_cv(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple],
    classes: list[str],
    split_contract: dict,
    seed: int = 42,
    genes: np.ndarray | None = None,
    label: str = "",
    n_pca: int | None = None,
    return_oof: bool = False,
):
    """Unscaled per-fold PCA then plain multinomial LogReg, over the shared CV body.

    The mechanism experiment's probe. It is not run_logreg_cv: that one standardizes
    per fold and weights the classes, and swapping the model would change every
    headline number for a reason unrelated to the metric it is being fixed for. What
    it does share is the fold loop and the out-of-fold collector, so the fold index
    reaches the metrics here the same way it does everywhere else.

    n_pca : components fitted on the training rows only, and only when the feature is
        wider than that. A one-column feature such as FoldX passes through untouched.
    """
    from sklearn.decomposition import PCA

    require_no_nan(X, "run_logreg_pca_cv")

    def _fit(X_tr, y_tr, X_te, fold_seed):
        if n_pca is not None and X_tr.shape[1] > n_pca:
            pca = PCA(n_components=min(n_pca, X_tr.shape[0] - 1), random_state=fold_seed)
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=fold_seed)
        clf.fit(X_tr, y_tr)
        return clf.predict_proba(X_te), clf.classes_

    return _run_multiclass_cv(
        _fit, X, y, splits, classes, split_contract, seed, genes, label, return_oof,
    )


def run_histgb_cv(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple],
    classes: list[str],
    split_contract: dict,
    seed: int = 42,
    genes: np.ndarray | None = None,
    label: str = "",
    return_oof: bool = False,
    max_iter: int = 200,
    compute_per_gene: bool = True,
):
    """NaN-native multiclass CV (HistGradientBoosting), same return shape as run_logreg_cv.

    Use this instead of run_logreg_cv/run_mlp_cv when the feature matrix has
    missing cells and complete-case restriction would discard a large or
    non-random share of rows. HistGradientBoostingClassifier consumes NaN
    directly — at each split it learns which side missing values go to — so no
    value is ever fabricated and no row is dropped. Imputing instead (a median
    over the whole dataset) would both leak test-fold statistics into training
    and make the filled cells indistinguishable from real measurements.

    Not scaled: gradient-boosted trees are invariant to monotone feature
    rescaling, and StandardScaler would need its own missing-value handling.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    def _fit(X_tr, y_tr, X_te, fold_seed):
        clf = HistGradientBoostingClassifier(
            max_iter=max_iter, class_weight="balanced", random_state=fold_seed
        )
        clf.fit(X_tr, y_tr)
        return clf.predict_proba(X_te), clf.classes_

    return _run_multiclass_cv(
        _fit, X, y, splits, classes, split_contract, seed, genes, label, return_oof,
        compute_per_gene,
    )


def _pos_class_col(clf_classes: np.ndarray, pos_label) -> int:
    """Return the column index for pos_label in clf.classes_, raising clearly if absent."""
    cols = np.where(clf_classes == pos_label)[0]
    if len(cols) == 0:
        raise ValueError(
            f"pos_label {pos_label!r} not found in classifier classes {clf_classes.tolist()}. "
            "Ensure the fold contains both classes before fitting."
        )
    return int(cols[0])


def auroc_for_clf(clf, X: np.ndarray, y: np.ndarray, pos_label=1) -> float:
    """AUROC of a fitted binary classifier scored on (X, y).

    Returns NaN if y has fewer than two classes (AUROC undefined). Uses
    _pos_class_col so a pos_label absent from clf.classes_ raises clearly rather
    than failing with a bare list.index ValueError.
    """
    if len(set(np.asarray(y).tolist())) < 2:
        return float("nan")
    proba = clf.predict_proba(X)[:, _pos_class_col(clf.classes_, pos_label)]
    return float(roc_auc_score(y, proba))


def _run_binary_cv(
    fold_predict_fn,
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple],
    classes: list[object],
    split_contract: dict,
    seed: int,
    pos_label,
    genes: np.ndarray | None = None, return_oof: bool = False,
):
    """Run binary folds through the shared scorer and complete-fold aggregator."""
    require_no_nan(X, "binary probe CV")
    declared = validate_classes(classes)
    if return_oof and genes is None:
        raise ValueError("genes are required when return_oof=True")
    if len(declared) != 2:
        raise ValueError(f"binary runner requires two declared classes, got {declared!r}")
    if pos_label not in declared:
        raise ValueError(f"positive label {pos_label!r} is not in {declared!r}")
    if split_contract.get("classes") != declared:
        raise ValueError("runner classes do not match the split contract")
    if split_contract.get("status") != "valid":
        result = _unscorable_result(split_contract, declared)
        return (result, None) if return_oof else result
    fold_results = []
    oof = _OofCollector(
        declared,
        np.asarray(split_contract["eligible_row_ids"], dtype=int),
        split_contract["requested_folds"],
    )
    positive_column = declared.index(pos_label)
    negative_column = 1 - positive_column
    for fold_i, (tr, te) in enumerate(splits):
        try:
            positive_probability = fold_predict_fn(
                X, y, tr, te, seed, fold_i, pos_label
            )
            if positive_probability is None:
                raise RuntimeError("binary fold predictor returned no probabilities")
            probabilities = np.empty((len(te), 2), dtype=float)
            probabilities[:, positive_column] = positive_probability
            probabilities[:, negative_column] = 1.0 - positive_probability
            predictions = np.array(
                [declared[index] for index in probabilities.argmax(axis=1)]
            )
            fold_metrics = compute_metrics(y[te], predictions, probabilities, declared)
        except Exception as error:
            result = _failed_result(
                split_contract, declared, len(fold_results), fold_i, error
            )
            return (result, None) if return_oof else result
        fold_results.append(fold_metrics)
        oof.add(y[te], probabilities, None if genes is None else genes[te], te, fold_i)

    aggregate = aggregate_folds(
        fold_results, declared, split_contract["requested_folds"]
    )
    aggregate.update(
        {
            "status": "success",
            "classes": declared,
            "eligible_rows": split_contract["eligible_rows"],
            "out_of_fold_rows": split_contract["eligible_rows"],
            "held_out_unit": split_contract.get("held_out_unit"),
            "group_count": split_contract.get("group_count"),
            "split_validation": split_contract,
            "auroc_mean": aggregate[f"auroc_{pos_label}_mean"],
            "auroc_std": aggregate[f"auroc_{pos_label}_std"],
            "auprc_mean": aggregate[f"auprc_{pos_label}_mean"],
            "auprc_std": aggregate[f"auprc_{pos_label}_std"],
            "prevalence_mean": aggregate[f"prevalence_{pos_label}_mean"],
            "prevalence_std": aggregate[f"prevalence_{pos_label}_std"],
            "ppv_mean": aggregate[f"ppv_{pos_label}_mean"],
            "ppv_std": aggregate[f"ppv_{pos_label}_std"],
            "npv_mean": aggregate[f"npv_{pos_label}_mean"],
            "npv_std": aggregate[f"npv_{pos_label}_std"],
        }
    )
    if return_oof:
        return aggregate, oof.finalize()
    return aggregate


def run_logreg_binary_cv(
    X: np.ndarray, y: np.ndarray, splits: list[tuple], classes: list[object],
    split_contract: dict, seed: int = 42, pos_label=1,
    genes: np.ndarray | None = None, return_oof: bool = False, max_iter: int = 1000,
):
    """Binary LogReg CV returning AUROC mean ± std.

    `max_iter` is exposed because callers fitting high-dimensional features (the
    1280-d embedding delta) need more iterations to converge than the default;
    raising it here rather than for every caller keeps other probes unchanged.
    """
    def _predict_fold(
        X_all, y_all, train_idx, test_idx, fold_seed, _fold_i, positive_label
    ):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_all[train_idx])
        X_test = scaler.transform(X_all[test_idx])
        classifier = LogisticRegression(
            max_iter=max_iter, C=1.0, random_state=fold_seed
        )
        classifier.fit(X_train, y_all[train_idx])
        positive_col = _pos_class_col(classifier.classes_, positive_label)
        return classifier.predict_proba(X_test)[:, positive_col]

    return _run_binary_cv(
        _predict_fold,
        X, y, splits, classes, split_contract, seed, pos_label,
        genes=genes, return_oof=return_oof,
    )


def run_mlp_cv(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple],
    classes: list[str],
    split_contract: dict,
    hidden: tuple = (256, 64),
    seed: int = 42,
    genes: np.ndarray | None = None,
    label: str = "",
    return_oof: bool = False,
    max_iter: int = 500,
    activation: str = "relu",
    alpha: float = 0.0001,
    validation_fraction: float = 0.15,
    n_iter_no_change: int = 10,
    oversample: bool = True,
    balanced_sample_weight: bool = False,
    compute_per_gene: bool = True,
):
    """Sklearn MLP CV: scale → optional class balancing → fit → metrics.

    splits : pre-computed list of (train_idx, test_idx)
    genes  : if provided, also computes per_gene_f1 per fold
    compute_per_gene : set False when genes are supplied only for OOF dependency
        tracking and the cohort does not define one unique label per gene.
    label  : prefix for per-fold log lines
    return_oof : if True, return (agg, oof) with out-of-fold test predictions
        {"y_true", "proba" (aligned to `classes`), "genes", "row_ids", "folds"} for dependency-aware
        inference, or None if no fold was scorable. `genes` must be provided.
        Default False keeps the bare-`agg` return for existing callers.
    oversample : balance the training fold by duplicating minority-class rows.
        Set False for experiments whose registered estimator was fit on the
        original class distribution.
    balanced_sample_weight : give each class equal total training weight without
        duplicating rows. Requires scikit-learn 1.7 or newer and cannot be used
        together with oversample.
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.utils.class_weight import compute_sample_weight

    require_no_nan(X, "run_mlp_cv")
    if oversample and balanced_sample_weight:
        raise ValueError("run_mlp_cv cannot combine oversampling and balanced sample weights")
    declared = validate_classes(classes)
    if return_oof and genes is None:
        raise ValueError("genes are required when return_oof=True")
    if split_contract.get("classes") != declared:
        raise ValueError("runner classes do not match the split contract")
    if split_contract.get("status") != "valid":
        result = _unscorable_result(split_contract, declared)
        return (result, None) if return_oof else result
    n_classes = len(declared)
    cls_to_idx = {cls: idx for idx, cls in enumerate(declared)}
    fold_results, pg_f1s = [], []
    oof = _OofCollector(
        declared,
        np.asarray(split_contract["eligible_row_ids"], dtype=int),
        split_contract["requested_folds"],
    )

    for fold_i, (tr, te) in enumerate(splits):
        try:
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y[tr], y[te]
            sc = StandardScaler().fit(X_tr)
            X_tr_s = sc.transform(X_tr)
            X_te_s = sc.transform(X_te)

            fit_idx = np.arange(len(y_tr))
            if oversample:
                counts = {cls: int((y_tr == cls).sum()) for cls in declared}
                max_c = max(counts.values())
                rng = np.random.RandomState(seed)
                sampled_idx = []
                for cls in declared:
                    class_idx = np.where(y_tr == cls)[0]
                    if len(class_idx) == 0:
                        continue
                    sampled_idx.append(np.tile(class_idx, max_c // len(class_idx)))
                    remainder = max_c % len(class_idx)
                    if remainder:
                        sampled_idx.append(rng.choice(class_idx, remainder, replace=False))
                fit_idx = np.concatenate(sampled_idx)
                rng.shuffle(fit_idx)

            clf = MLPClassifier(
                hidden_layer_sizes=hidden,
                activation=activation,
                alpha=alpha,
                max_iter=max_iter,
                early_stopping=True,
                validation_fraction=validation_fraction,
                n_iter_no_change=n_iter_no_change,
                random_state=seed,
            )
        # Fit on integer-encoded labels: sklearn's early_stopping scores an
        # internal validation split with np.isnan(y_pred), which raises on string
        # arrays. clf.classes_ are then mapped back to strings for align_proba so
        # column ordering stays explicit (never positionally assumed).
            y_tr_enc = np.array([cls_to_idx[lab] for lab in y_tr])
            if balanced_sample_weight:
                sample_weight = compute_sample_weight(
                    class_weight="balanced", y=y_tr_enc[fit_idx]
                )
                clf.fit(
                    X_tr_s[fit_idx],
                    y_tr_enc[fit_idx],
                    sample_weight=sample_weight,
                )
            else:
                clf.fit(X_tr_s[fit_idx], y_tr_enc[fit_idx])

            clf_str_classes = np.array([declared[idx] for idx in clf.classes_])
            proba = align_proba(
                clf.predict_proba(X_te_s),
                clf_str_classes,
                declared,
                allow_missing_classes=split_contract["allow_missing_classifier_classes"],
            )
            pred = np.array([declared[idx] for idx in proba.argmax(axis=1)])
            fm = compute_metrics(y_te, pred, proba, declared)
        except Exception as error:
            result = _failed_result(
                split_contract, declared, len(fold_results), fold_i, error
            )
            return (result, None) if return_oof else result
        fold_results.append(fm)
        oof.add(y_te, proba, None if genes is None else genes[te], te, fold_i)

        pg_str = ""
        if genes is not None and compute_per_gene:
            pg_str = _record_per_gene_f1(pg_f1s, y_te, proba, genes[te], declared)

        _log_fold(label, fold_i, fm, declared, pg_str)

    agg = aggregate_folds(
        fold_results, declared, split_contract["requested_folds"]
    )
    agg.update(
        {
            "status": "success",
            "classes": declared,
            "eligible_rows": split_contract["eligible_rows"],
            "out_of_fold_rows": split_contract["eligible_rows"],
            "held_out_unit": split_contract.get("held_out_unit"),
            "group_count": split_contract.get("group_count"),
            "split_validation": split_contract,
        }
    )
    if compute_per_gene:
        _add_per_gene_f1(agg, pg_f1s)
    if return_oof:
        return agg, oof.finalize()
    return agg


def run_mlp_binary_cv(
    X: np.ndarray, y: np.ndarray, splits: list[tuple], classes: list[object],
    split_contract: dict, validation_groups: np.ndarray,
    seed: int = 42, pos_label=1,
    genes: np.ndarray | None = None, return_oof: bool = False,
):
    """Binary sklearn MLP CV with group-disjoint early stopping.

    `validation_groups` is the dependency unit for the internal 10% validation
    holdout. Pass genes for gene-split CV and Pfam-family IDs for family-split
    CV. A group that crosses an outer train/test boundary is rejected.
    """
    from sklearn.neural_network import MLPClassifier

    validation_groups = np.asarray(validation_groups)
    if len(validation_groups) != len(X):
        raise ValueError(
            f"validation_groups has {len(validation_groups)} rows for {len(X)} samples"
        )

    validation_masks = []
    internal_failures = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        training_groups = validation_groups[train_idx]
        overlap = sorted(
            set(training_groups.tolist())
            & set(validation_groups[test_idx].tolist()),
            key=str,
        )
        validation_mask = _validation_group_mask(
            training_groups, seed + fold_i, validation_fraction=0.1
        )
        if overlap:
            internal_failures.append(
                {
                    "scope": "fold",
                    "fold": fold_i,
                    "reason": "validation_group_crosses_outer_boundary",
                    "groups": overlap[:20],
                }
            )
        if validation_mask is None:
            internal_failures.append(
                {
                    "scope": "fold",
                    "fold": fold_i,
                    "reason": "insufficient_validation_groups",
                }
            )
            validation_masks.append(None)
            continue
        fit_mask = ~validation_mask
        fit_rows = int(fit_mask.sum())
        validation_rows = int(validation_mask.sum())
        if fit_rows < 10 or validation_rows < 5:
            internal_failures.append(
                {
                    "scope": "fold",
                    "fold": fold_i,
                    "reason": "insufficient_early_stopping_rows",
                    "fit_rows": fit_rows,
                    "validation_rows": validation_rows,
                }
            )
        fit_classes = set(np.asarray(y)[train_idx][fit_mask].tolist())
        if len(fit_classes) < 2:
            internal_failures.append(
                {
                    "scope": "fold",
                    "fold": fold_i,
                    "reason": "insufficient_early_stopping_fit_classes",
                    "observed_classes": sorted(fit_classes, key=repr),
                }
            )
        validation_masks.append(validation_mask)
    if internal_failures:
        internal_contract = dict(split_contract)
        internal_contract["status"] = "unscorable"
        internal_contract["failures"] = [
            *split_contract.get("failures", []),
            *internal_failures,
        ]
        result = _unscorable_result(internal_contract, validate_classes(classes))
        return (result, None) if return_oof else result

    def _predict_fold(
        X_all, y_all, train_idx, test_idx, fold_seed, fold_i, positive_label
    ):
        training_groups = validation_groups[train_idx]
        test_groups = set(validation_groups[test_idx].tolist())
        outer_overlap = set(training_groups.tolist()) & test_groups
        if outer_overlap:
            examples = sorted(outer_overlap, key=str)[:5]
            raise ValueError(
                "validation group spans the outer CV train/test boundary; "
                f"examples: {examples}"
            )

        validation_mask = validation_masks[fold_i]
        fit_mask = ~validation_mask
        X_fit = X_all[train_idx][fit_mask]
        X_validation = X_all[train_idx][validation_mask]
        y_fit = y_all[train_idx][fit_mask]
        y_validation = y_all[train_idx][validation_mask]
        X_fit, X_validation, X_test = standardize(
            X_fit, X_validation, X_all[test_idx]
        )
        classifier = MLPClassifier(
            hidden_layer_sizes=(256,),
            max_iter=300,
            random_state=fold_seed,
            early_stopping=False,
        )
        training_classes = np.unique(y_all[train_idx])
        best_validation_score = float("-inf")
        best_parameters = None
        no_improvement_count = 0
        for epoch in range(classifier.max_iter):
            if epoch == 0:
                classifier.partial_fit(X_fit, y_fit, classes=training_classes)
            else:
                classifier.partial_fit(X_fit, y_fit)
            validation_score = float(classifier.score(X_validation, y_validation))
            # Patience is measured against the best score from an earlier epoch.
            # Updating the best score first makes every epoch fall short of
            # best + tol, including a genuinely improving epoch.
            if validation_score < best_validation_score + classifier.tol:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            if validation_score > best_validation_score:
                best_validation_score = validation_score
                best_parameters = (
                    [weights.copy() for weights in classifier.coefs_],
                    [bias.copy() for bias in classifier.intercepts_],
                )
            if no_improvement_count > classifier.n_iter_no_change:
                break

        if best_parameters is None:
            raise RuntimeError("MLP early stopping produced no fitted checkpoint")
        classifier.coefs_, classifier.intercepts_ = best_parameters
        positive_col = _pos_class_col(classifier.classes_, positive_label)
        return classifier.predict_proba(X_test)[:, positive_col]

    return _run_binary_cv(
        _predict_fold,
        X, y, splits, classes, split_contract, seed, pos_label,
        genes=genes, return_oof=return_oof,
    )


def pca_reduce(X_tr: np.ndarray, X_te: np.ndarray, n_components: int = 50):
    """Fit PCA on training data and transform both train and test."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=0)
    return pca.fit_transform(X_tr), pca.transform(X_te)


def _run_sklearn_probe_impl(
    clf_fn, X: np.ndarray, labels: np.ndarray, genes: np.ndarray,
    classes: list[str], split_contract: dict,
    n_folds: int, seed: int, splits: list,
    normalize: bool, n_pca: int | None, return_oof: bool,
):
    """Shared generic sklearn probe using the classification contracts."""
    require_no_nan(X, "run_sklearn_probe")
    declared = validate_classes(classes)
    if return_oof and genes is None:
        raise ValueError("genes are required when return_oof=True")
    class_to_index = {class_name: index for index, class_name in enumerate(declared)}

    def _fit(X_tr, y_tr, X_te, fold_seed):
        X_tr = X_tr.astype(np.float32)
        X_te = X_te.astype(np.float32)
        if normalize or n_pca is not None:
            X_tr, X_te = standardize(X_tr, X_te)
        if n_pca is not None:
            from sklearn.decomposition import PCA
            pca = PCA(
                n_components=min(n_pca, X_tr.shape[1], X_tr.shape[0] - 1),
                random_state=fold_seed,
            )
            X_tr, X_te = pca.fit_transform(X_tr), pca.transform(X_te)
        classifier = clf_fn(fold_seed)
        encoded = np.array([class_to_index[value] for value in y_tr])
        classifier.fit(X_tr, encoded)
        if not hasattr(classifier, "predict_proba"):
            raise ValueError("generic classification probes require predict_proba")
        fitted_classes = np.array([declared[index] for index in classifier.classes_])
        return classifier.predict_proba(X_te), fitted_classes

    return _run_multiclass_cv(
        _fit,
        X,
        labels,
        splits,
        declared,
        split_contract,
        seed,
        genes,
        "sklearn",
        return_oof,
    )


def run_sklearn_probe(
    clf_fn, X: np.ndarray, labels: np.ndarray, genes: np.ndarray,
    classes: list[str], split_contract: dict, splits: list,
    n_folds: int = 5, seed: int = 42, normalize: bool = False,
    return_oof: bool = False,
):
    """Generic gene-split CV runner for any sklearn classifier."""
    return _run_sklearn_probe_impl(
        clf_fn, X, labels, genes, classes, split_contract, n_folds, seed, splits,
        normalize=normalize, n_pca=None, return_oof=return_oof,
    )


def run_sklearn_probe_pca(
    clf_fn, X: np.ndarray, labels: np.ndarray, genes: np.ndarray,
    classes: list[str], split_contract: dict, splits: list,
    n_folds: int = 5, seed: int = 42, n_pca: int = 50,
    return_oof: bool = False,
):
    """Gene-split CV with per-fold PCA reduction and normalization."""
    return _run_sklearn_probe_impl(
        clf_fn, X, labels, genes, classes, split_contract, n_folds, seed, splits,
        normalize=True, n_pca=n_pca, return_oof=return_oof,
    )


def _validation_group_mask(
    training_groups: np.ndarray,
    seed: int,
    validation_fraction: float = 0.15,
) -> np.ndarray | None:
    """Select whole groups for early stopping; return None if fewer than two exist."""
    training_groups = np.asarray(training_groups)
    unique_groups = np.array(sorted(set(training_groups.tolist()), key=str), dtype=object)
    if len(unique_groups) < 2:
        return None
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_groups)
    n_validation_groups = max(1, int(validation_fraction * len(unique_groups)))
    n_validation_groups = min(n_validation_groups, len(unique_groups) - 1)
    validation_group_set = set(unique_groups[:n_validation_groups])
    return np.array([group in validation_group_set for group in training_groups])


def run_mlp_probe_cv(
    X: np.ndarray,
    labels: np.ndarray,
    splits: list[tuple],
    classes: list[str],
    split_contract: dict,
    validation_groups: np.ndarray,
    seed: int = 42,
    hidden: tuple = (256, 64),
    dropout: float = 0.3,
    lr: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 10,
    batch_size: int = 256,
    genes: np.ndarray | None = None,
    label: str = "",
    return_oof: bool = False,
):
    """Run the PyTorch multiclass probe under a validated complete split set."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    require_no_nan(X, "run_mlp_probe_cv")
    declared = validate_classes(classes)
    if return_oof and genes is None:
        raise ValueError("genes are required when return_oof=True")
    if split_contract.get("classes") != declared:
        raise ValueError("runner classes do not match the split contract")
    if split_contract.get("status") != "valid":
        result = _unscorable_result(split_contract, declared)
        return (result, None) if return_oof else result
    n_classes = len(declared)
    cls_to_idx = {cls: idx for idx, cls in enumerate(declared)}
    y = np.array([cls_to_idx[lab] for lab in labels])
    validation_groups = np.asarray(validation_groups)
    if len(validation_groups) != len(X):
        raise ValueError(
            f"validation_groups has {len(validation_groups)} rows for {len(X)} samples"
        )
    validation_masks = []
    internal_failures = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        training_groups = validation_groups[train_idx]
        overlap = sorted(
            set(training_groups.tolist())
            & set(validation_groups[test_idx].tolist()),
            key=str,
        )
        validation_mask = _validation_group_mask(training_groups, seed + fold_i)
        if overlap:
            internal_failures.append(
                {
                    "scope": "fold",
                    "fold": fold_i,
                    "reason": "validation_group_crosses_outer_boundary",
                    "groups": overlap[:20],
                }
            )
        if validation_mask is None:
            internal_failures.append(
                {
                    "scope": "fold",
                    "fold": fold_i,
                    "reason": "insufficient_validation_groups",
                }
            )
            validation_masks.append(None)
            continue
        fit_mask = ~validation_mask
        if int(fit_mask.sum()) < 10 or int(validation_mask.sum()) < 5:
            internal_failures.append(
                {
                    "scope": "fold",
                    "fold": fold_i,
                    "reason": "insufficient_early_stopping_rows",
                    "fit_rows": int(fit_mask.sum()),
                    "validation_rows": int(validation_mask.sum()),
                }
            )
        fit_labels = np.asarray(labels)[train_idx][fit_mask]
        fit_present = set(fit_labels.tolist())
        required_fit = split_contract.get("required_train_classes") or []
        missing_fit = [
            class_name for class_name in required_fit if class_name not in fit_present
        ]
        minimum_fit = split_contract.get("minimum_train_classes")
        if missing_fit:
            internal_failures.append(
                {
                    "scope": "fold",
                    "fold": fold_i,
                    "reason": "early_stopping_fit_missing_required_classes",
                    "classes": missing_fit,
                }
            )
        elif minimum_fit is not None and len(fit_present) < minimum_fit:
            internal_failures.append(
                {
                    "scope": "fold",
                    "fold": fold_i,
                    "reason": "early_stopping_fit_insufficient_classes",
                    "required": int(minimum_fit),
                    "observed": int(len(fit_present)),
                }
            )
        validation_masks.append(validation_mask)
    if internal_failures:
        internal_contract = dict(split_contract)
        internal_contract["status"] = "unscorable"
        internal_contract["failures"] = [
            *split_contract.get("failures", []),
            *internal_failures,
        ]
        result = _unscorable_result(internal_contract, declared)
        return (result, None) if return_oof else result

    fold_results, pg_f1s = [], []
    oof = _OofCollector(
        declared,
        np.asarray(split_contract["eligible_row_ids"], dtype=int),
        split_contract["requested_folds"],
    )

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        try:
            X_tr = X[train_idx].astype(np.float32)
            X_te = X[test_idx].astype(np.float32)
            y_tr = y[train_idx]
            labels_te = labels[test_idx]
            validation_mask = validation_masks[fold_i]
            fit_mask = ~validation_mask
            X_fit, y_fit = X_tr[fit_mask], y_tr[fit_mask]
            X_val, y_val = X_tr[validation_mask], y_tr[validation_mask]
            X_fit, X_val, X_te_normalized = standardize(X_fit, X_val, X_te)

            class_counts = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
            class_weights = np.zeros(n_classes, dtype=np.float32)
            present_indices = np.where(class_counts > 0)[0]
            class_weights[present_indices] = 1.0 / class_counts[present_indices]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            weights_tensor = torch.tensor(class_weights).to(device)
            torch.manual_seed(seed + fold_i)

            layers: list = []
            previous_width = X_fit.shape[1]
            for hidden_width in hidden:
                layers += [
                    nn.Linear(previous_width, hidden_width),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
                previous_width = hidden_width
            layers.append(nn.Linear(previous_width, n_classes))
            model = nn.Sequential(*layers).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
            criterion = nn.CrossEntropyLoss(weight=weights_tensor)
            dataset = TensorDataset(
                torch.tensor(X_fit), torch.tensor(y_fit, dtype=torch.long)
            )
            shuffle_generator = torch.Generator().manual_seed(seed + fold_i)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                generator=shuffle_generator,
            )
            best_validation_loss = float("inf")
            patience_count = 0
            best_state = None
            for _epoch in range(max_epochs):
                model.train()
                for features_batch, labels_batch in loader:
                    optimizer.zero_grad()
                    criterion(
                        model(features_batch.to(device)), labels_batch.to(device)
                    ).backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    validation_loss = criterion(
                        model(torch.tensor(X_val).to(device)),
                        torch.tensor(y_val, dtype=torch.long).to(device),
                    ).item()
                if validation_loss < best_validation_loss - 1e-4:
                    best_validation_loss = validation_loss
                    patience_count = 0
                    best_state = {
                        key: value.clone() for key, value in model.state_dict().items()
                    }
                else:
                    patience_count += 1
                    if patience_count >= patience:
                        break
            if best_state is None:
                raise RuntimeError("MLP early stopping produced no fitted checkpoint")
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                proba = torch.softmax(
                    model(torch.tensor(X_te_normalized).to(device)), 1
                ).cpu().numpy()
            missing_indices = np.where(class_counts == 0)[0]
            if len(missing_indices):
                if not split_contract["allow_missing_classifier_classes"]:
                    missing_classes = [declared[index] for index in missing_indices]
                    raise ValueError(
                        f"training fold is missing required classes: {missing_classes!r}"
                    )
                proba[:, missing_indices] = 0.0
                row_sums = proba.sum(axis=1, keepdims=True)
                if np.any(row_sums == 0):
                    raise RuntimeError("MLP produced no probability mass on fitted classes")
                proba = proba / row_sums
            pred_labels = np.array([declared[index] for index in proba.argmax(1)])
            fold_metrics = compute_metrics(labels_te, pred_labels, proba, declared)
        except Exception as error:
            result = _failed_result(
                split_contract, declared, len(fold_results), fold_i, error
            )
            return (result, None) if return_oof else result
        fold_results.append(fold_metrics)
        oof.add(
            labels_te,
            proba,
            None if genes is None else genes[test_idx],
            test_idx,
            fold_i,
        )

        pg_str = ""
        if genes is not None:
            pg_str = _record_per_gene_f1(
                pg_f1s, labels_te, proba, genes[test_idx], declared
            )
        _log_fold(label, fold_i, fold_metrics, declared, pg_str)

    agg = aggregate_folds(
        fold_results, declared, split_contract["requested_folds"]
    )
    agg.update(
        {
            "status": "success",
            "classes": declared,
            "eligible_rows": split_contract["eligible_rows"],
            "out_of_fold_rows": split_contract["eligible_rows"],
            "held_out_unit": split_contract.get("held_out_unit"),
            "group_count": split_contract.get("group_count"),
            "split_validation": split_contract,
        }
    )
    _add_per_gene_f1(agg, pg_f1s)
    if return_oof:
        return agg, oof.finalize()
    return agg
