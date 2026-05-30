"""
Shared probe utilities: CV splits, metrics, and regression helpers.

Used by badonyi_mechanism, per_gene_ablation, within_family_mechanism,
multiseed_v1, pathogenicity_control, contrastive_mechanism, megascale_stability,
badonyi_leakage_analysis, mmseqs_cluster_holdout, proteome_mechanism,
mlp, enzyme_classification, go_smoke_test.
"""

from __future__ import annotations

import functools
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

print = functools.partial(print, flush=True)

MECHANISM_CLASSES = ["GOF", "DN", "LOF"]


# ---------------------------------------------------------------------------
# CV split generators
# ---------------------------------------------------------------------------


def gene_split_cv(genes: np.ndarray, n_folds: int = 5, seed: int = 42) -> list[tuple]:
    """Gene-disjoint CV: each fold holds out a disjoint set of genes."""
    u = np.array(sorted(set(genes)))
    np.random.RandomState(seed).shuffle(u)
    splits = []
    for fold in np.array_split(u, n_folds):
        tr = np.where(~np.isin(genes, fold))[0]
        te = np.where(np.isin(genes, fold))[0]
        if len(tr) >= 10 and len(te) >= 5:
            splits.append((tr, te))
    return splits


def family_split_cv(
    genes: np.ndarray, pfam_map: dict, n_folds: int = 5, seed: int = 42
) -> list[tuple]:
    """Family-disjoint CV using a {gene: pfam_family} map.

    Only annotated genes (present in pfam_map) are included in train/test;
    unannotated genes are excluded from both sides.
    """
    g2p = {g: pfam_map[g] for g in np.unique(genes) if pfam_map.get(g)}
    fams = np.array(sorted(set(g2p.values())))
    np.random.RandomState(seed).shuffle(fams)
    n = len(genes)
    splits = []
    for fold_fams in np.array_split(fams, n_folds):
        fs = set(fold_fams)
        te = np.array([genes[i] in g2p and g2p[genes[i]] in fs for i in range(n)])
        tr = np.array([genes[i] in g2p and g2p[genes[i]] not in fs for i in range(n)])
        if tr.sum() >= 10 and te.sum() >= 5:
            splits.append((np.where(tr)[0], np.where(te)[0]))
    return splits


def family_split_indices(groups: np.ndarray, n_folds: int, seed: int):
    """Yield (train_idx, test_idx) for family-disjoint CV.

    groups is a 1-D array of Pfam family strings (or None for unannotated).
    None-group samples are distributed evenly across folds rather than excluded.
    """
    rng = np.random.RandomState(seed)
    unique_fams = np.array(sorted(f for f in set(groups) if f is not None))
    rng.shuffle(unique_fams)
    fam_fold = {f: i % n_folds for i, f in enumerate(unique_fams)}

    none_positions = [i for i, g in enumerate(groups) if g is None]
    rng.shuffle(none_positions)
    none_fold = {pos: i % n_folds for i, pos in enumerate(none_positions)}

    fold_of = np.array(
        [
            fam_fold[g] if g is not None else none_fold.get(i, 0)
            for i, g in enumerate(groups)
        ]
    )
    for k in range(n_folds):
        test = np.where(fold_of == k)[0]
        train = np.where(fold_of != k)[0]
        yield train, test


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: list[str] = MECHANISM_CLASSES,
) -> dict:
    """Compute macro-F1 and per-class AUROC.

    Returns {"macro_f1": float, "per_class_auroc": {cls: float|None}}.
    """
    if len(y_true) == 0:
        return {"macro_f1": None, "per_class_auroc": {c: None for c in classes}, "n": 0}
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    auroc: dict[str, float | None] = {}
    for i, cls in enumerate(classes):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            auroc[cls] = None
        else:
            auroc[cls] = float(roc_auc_score(y_bin, y_proba[:, i]))
    out: dict = {"macro_f1": macro_f1, "per_class_auroc": auroc}
    if len(y_true) > 0:
        out["n"] = int(len(y_true))
    return out


def per_gene_f1(y_true: np.ndarray, proba: np.ndarray, genes: np.ndarray) -> float:
    """Aggregate per-variant probabilities to per-gene predictions and compute macro-F1."""
    unique = list(set(genes.tolist()))
    y_g, p_g = [], []
    for g in unique:
        m = genes == g
        true_label = int(np.bincount(y_true[m]).argmax())
        mean_proba = proba[m].mean(0)
        pred_label = int(mean_proba.argmax())
        y_g.append(true_label)
        p_g.append(pred_label)
    return float(f1_score(y_g, p_g, average="macro", zero_division=0))


def aggregate_folds(
    fold_list: list[dict], classes: list[str] = MECHANISM_CLASSES
) -> dict:
    """Aggregate per-fold dicts from compute_metrics into mean ± std."""
    if not fold_list:
        return {"error": "no folds"}
    out: dict = {}
    f1_vals = [f["macro_f1"] for f in fold_list if f.get("macro_f1") is not None]
    out["macro_f1_mean"] = float(np.mean(f1_vals)) if f1_vals else None
    out["macro_f1_std"] = float(np.std(f1_vals)) if f1_vals else None
    for cls in classes:
        vals = [
            f["per_class_auroc"][cls]
            for f in fold_list
            if f.get("per_class_auroc", {}).get(cls) is not None
        ]
        out[f"auroc_{cls}_mean"] = float(np.mean(vals)) if vals else None
        out[f"auroc_{cls}_std"] = float(np.std(vals)) if vals else None
    out["n_folds"] = len(fold_list)
    return out


# ---------------------------------------------------------------------------
# Probability alignment
# ---------------------------------------------------------------------------


def align_proba(
    proba: np.ndarray, clf_classes: np.ndarray, n_classes: int
) -> np.ndarray:
    """Reorder classifier probability output to match canonical class indices."""
    aligned = np.zeros((len(proba), n_classes), dtype=np.float32)
    for ci, c in enumerate(clf_classes):
        if c < n_classes:
            aligned[:, c] = proba[:, ci]
    return aligned


# ---------------------------------------------------------------------------
# Logistic regression probe
# ---------------------------------------------------------------------------


def run_logreg_cv(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple],
    classes: list[str] = MECHANISM_CLASSES,
    seed: int = 42,
    genes: np.ndarray | None = None,
    label: str = "",
) -> dict:
    """Run LogReg + StandardScaler over pre-computed splits, return aggregated metrics.

    genes : if provided, also computes per_gene_f1 per fold
    label : prefix for per-fold log lines
    """
    n_classes = len(classes)
    fold_results, pg_f1s = [], []
    for fold_i, (tr, te) in enumerate(splits):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        if len(set(y_tr.tolist())) < n_classes:
            print(f"    [{label}] Fold {fold_i+1}: skipped (missing class in train)")
            continue
        if len(set(y_te.tolist())) < 2:
            print(f"    [{label}] Fold {fold_i+1}: skipped (< 2 classes in test)")
            continue
        sc = StandardScaler().fit(X_tr)
        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed
        )
        clf.fit(sc.transform(X_tr), y_tr)
        proba = align_proba(
            clf.predict_proba(sc.transform(X_te)), clf.classes_, n_classes
        )
        pred = proba.argmax(axis=1)
        fm = compute_metrics(y_te, pred, proba, classes)
        fold_results.append(fm)

        pg_str = ""
        if genes is not None:
            pg = per_gene_f1(y_te, proba, genes[te])
            pg_f1s.append(pg)
            pg_str = f"  per_gene_f1={pg:.3f}"

        print(
            f"    [{label}] Fold {fold_i+1}: macro_f1={fm['macro_f1']:.3f}{pg_str}  "
            + "  ".join(
                (
                    f"{cls}={fm['per_class_auroc'].get(cls, float('nan')):.3f}"
                    if fm["per_class_auroc"].get(cls) is not None
                    else f"{cls}=NA"
                )
                for cls in classes
            )
        )

    agg = aggregate_folds(fold_results, classes)
    if pg_f1s:
        agg["per_gene_f1_mean"] = float(np.mean(pg_f1s))
        agg["per_gene_f1_std"] = float(np.std(pg_f1s))
    return agg


def run_mlp_cv(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple],
    hidden: tuple = (256, 64),
    seed: int = 42,
    classes: list[str] = MECHANISM_CLASSES,
    genes: np.ndarray | None = None,
    label: str = "",
) -> dict:
    """Sklearn MLP CV: scale → oversample → fit → aggregate metrics.

    splits : pre-computed list of (train_idx, test_idx)
    genes  : if provided, also computes per_gene_f1 per fold
    label  : prefix for per-fold log lines
    """
    from sklearn.neural_network import MLPClassifier

    n_classes = len(classes)
    fold_results, pg_f1s = [], []

    for fold_i, (tr, te) in enumerate(splits):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        if len(set(y_tr.tolist())) < n_classes:
            print(f"    [{label}] Fold {fold_i+1}: skipped (missing class in train)")
            continue
        if len(set(y_te.tolist())) < 2:
            print(f"    [{label}] Fold {fold_i+1}: skipped (< 2 classes in test)")
            continue

        sc = StandardScaler().fit(X_tr)
        X_tr_s = sc.transform(X_tr)
        X_te_s = sc.transform(X_te)

        counts = np.bincount(y_tr, minlength=n_classes)
        max_c = counts.max()
        rng = np.random.RandomState(seed)
        os_idx = []
        for c in range(n_classes):
            ci = np.where(y_tr == c)[0]
            if len(ci) == 0:
                continue
            os_idx.append(np.tile(ci, max_c // len(ci)))
            rem = max_c % len(ci)
            if rem:
                os_idx.append(rng.choice(ci, rem, replace=False))
        os_idx = np.concatenate(os_idx)
        rng.shuffle(os_idx)

        clf = MLPClassifier(
            hidden_layer_sizes=hidden,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed,
        )
        clf.fit(X_tr_s[os_idx], y_tr[os_idx])

        proba = align_proba(clf.predict_proba(X_te_s), clf.classes_, n_classes)
        pred = proba.argmax(axis=1)
        fm = compute_metrics(y_te, pred, proba, classes)
        fold_results.append(fm)

        pg_str = ""
        if genes is not None:
            pg = per_gene_f1(y_te, proba, genes[te])
            pg_f1s.append(pg)
            pg_str = f"  per_gene_f1={pg:.3f}"

        print(
            f"    [{label}] Fold {fold_i+1}: macro_f1={fm['macro_f1']:.3f}{pg_str}  "
            + "  ".join(
                (
                    f"{cls}={fm['per_class_auroc'].get(cls, float('nan')):.3f}"
                    if fm["per_class_auroc"].get(cls) is not None
                    else f"{cls}=NA"
                )
                for cls in classes
            )
        )

    agg = aggregate_folds(fold_results, classes)
    if pg_f1s:
        agg["per_gene_f1_mean"] = float(np.mean(pg_f1s))
        agg["per_gene_f1_std"] = float(np.std(pg_f1s))
    return agg


def run_logreg_binary_cv(
    X: np.ndarray, y: np.ndarray, splits: list[tuple], seed: int = 42
) -> dict:
    """Binary LogReg CV returning AUROC mean ± std."""
    aurocs = []
    for tr, te in splits:
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        clf.fit(X_tr, y[tr])
        proba = clf.predict_proba(X_te)[:, list(clf.classes_).index(1)]
        aurocs.append(float(roc_auc_score(y[te], proba)))
    if not aurocs:
        return {}
    return {
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "n_folds": len(aurocs),
    }


# ---------------------------------------------------------------------------
# Ridge regression probe (for continuous targets)
# ---------------------------------------------------------------------------


def run_ridge_cv(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple],
    alpha: float = 1.0,
    include_pearson: bool = True,
) -> dict:
    """Ridge regression CV returning Spearman (and optionally Pearson) r."""
    from scipy.stats import spearmanr, pearsonr

    rhos, rs = [], []
    for tr, te in splits:
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        clf = Ridge(alpha=alpha)
        clf.fit(X_tr, y[tr])
        pred = clf.predict(X_te)
        rho, _ = spearmanr(y[te], pred)
        rhos.append(float(rho))
        if include_pearson:
            r, _ = pearsonr(y[te], pred)
            rs.append(float(r))
    if not rhos:
        return {}
    out = {
        "spearman_mean": float(np.mean(rhos)),
        "spearman_std": float(np.std(rhos)),
        "n_folds": len(rhos),
    }
    if include_pearson and rs:
        out["pearson_mean"] = float(np.mean(rs))
        out["pearson_std"] = float(np.std(rs))
    return out
