"""Probe runners: logistic regression and MLP CV over pre-computed splits."""

from __future__ import annotations

import functools

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler

from esm2_mech.utils.constants import MECHANISM_CLASSES
from esm2_mech.utils.metrics import aggregate_folds, align_proba, compute_metrics

print = functools.partial(print, flush=True)


def _per_gene_f1(y_true: np.ndarray, proba: np.ndarray, genes: np.ndarray) -> float:
    """Aggregate per-variant probabilities to per-gene predictions and compute macro-F1."""
    unique = list(set(genes.tolist()))
    y_g, p_g = [], []
    for g in unique:
        mask = genes == g
        gene_labels = y_true[mask]
        counts = {cls: int((gene_labels == cls).sum()) for cls in MECHANISM_CLASSES}
        true_label = max(counts, key=counts.__getitem__)
        pred_label = MECHANISM_CLASSES[int(proba[mask].mean(0).argmax())]
        y_g.append(true_label)
        p_g.append(pred_label)
    return float(f1_score(y_g, p_g, average="macro", zero_division=0))


def run_logreg_cv(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple],
    classes: list[str] = MECHANISM_CLASSES,
    seed: int = 42,
    genes: np.ndarray | None = None,
    label: str = "",
    min_train_classes: int | None = None,
) -> dict:
    """Run LogReg + StandardScaler over pre-computed splits, return aggregated metrics.

    genes : if provided, also computes per_gene_f1 per fold
    label : prefix for per-fold log lines
    min_train_classes : minimum distinct classes a fold's train split must have to
        be kept. Defaults to n_classes (the historical behaviour — every class must
        be present). Pass 2 to keep folds where a rare class falls entirely in test
        (a classifier only needs two classes to fit); this also lets a caller make
        the probe's fold set match a separately-computed chance floor that skips on
        the same condition.
    """
    n_classes = len(classes)
    min_train_classes = n_classes if min_train_classes is None else min_train_classes
    fold_results, pg_f1s = [], []
    for fold_i, (tr, te) in enumerate(splits):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        if len(set(y_tr.tolist())) < min_train_classes:
            print(f"    [{label}] Fold {fold_i+1}: skipped (< {min_train_classes} classes in train)")
            continue
        if len(set(y_te.tolist())) < 2:
            print(f"    [{label}] Fold {fold_i+1}: skipped (< 2 classes in test)")
            continue
        sc = StandardScaler().fit(X_tr)
        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed
        )
        clf.fit(sc.transform(X_tr), y_tr)
        proba = align_proba(clf.predict_proba(sc.transform(X_te)), clf.classes_, classes)
        pred = np.array([classes[idx] for idx in proba.argmax(axis=1)])
        fm = compute_metrics(y_te, pred, proba, classes)
        fold_results.append(fm)

        pg_str = ""
        if genes is not None:
            pg = _per_gene_f1(y_te, proba, genes[te])
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


def run_logreg_binary_cv(
    X: np.ndarray, y: np.ndarray, splits: list[tuple], seed: int = 42, pos_label=1
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
        proba = clf.predict_proba(X_te)[:, _pos_class_col(clf.classes_, pos_label)]
        aurocs.append(float(roc_auc_score(y[te], proba)))
    if not aurocs:
        return {}
    return {
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "n_folds": len(aurocs),
    }


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
    cls_to_idx = {cls: idx for idx, cls in enumerate(classes)}
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

        counts = {cls: int((y_tr == cls).sum()) for cls in classes}
        max_c = max(counts.values())
        rng = np.random.RandomState(seed)
        os_idx = []
        for cls in classes:
            ci = np.where(y_tr == cls)[0]
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
        # Fit on integer-encoded labels: sklearn's early_stopping scores an
        # internal validation split with np.isnan(y_pred), which raises on string
        # arrays. clf.classes_ are then mapped back to strings for align_proba so
        # column ordering stays explicit (never positionally assumed).
        y_tr_enc = np.array([cls_to_idx[lab] for lab in y_tr])
        clf.fit(X_tr_s[os_idx], y_tr_enc[os_idx])

        clf_str_classes = np.array([classes[idx] for idx in clf.classes_])
        proba = align_proba(clf.predict_proba(X_te_s), clf_str_classes, classes)
        pred = np.array([classes[idx] for idx in proba.argmax(axis=1)])
        fm = compute_metrics(y_te, pred, proba, classes)
        fold_results.append(fm)

        pg_str = ""
        if genes is not None:
            pg = _per_gene_f1(y_te, proba, genes[te])
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


def run_mlp_binary_cv(
    X: np.ndarray, y: np.ndarray, splits: list[tuple], seed: int = 42, pos_label=1
) -> dict:
    """Binary sklearn MLP CV returning AUROC mean ± std."""
    from sklearn.neural_network import MLPClassifier

    aurocs = []
    for tr, te in splits:
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        clf = MLPClassifier(
            hidden_layer_sizes=(256,),
            max_iter=300,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
        )
        clf.fit(X_tr, y[tr])
        proba = clf.predict_proba(X_te)[:, _pos_class_col(clf.classes_, pos_label)]
        aurocs.append(float(roc_auc_score(y[te], proba)))
    if not aurocs:
        return {}
    return {
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "n_folds": len(aurocs),
    }


def pca_reduce(X_tr: np.ndarray, X_te: np.ndarray, n_components: int = 50):
    """Fit PCA on training data and transform both train and test."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=0)
    return pca.fit_transform(X_tr), pca.transform(X_te)


def run_sklearn_probe(
    clf_fn, X: np.ndarray, labels: np.ndarray, genes: np.ndarray,
    n_folds: int = 5, seed: int = 42, normalize: bool = False,
    splits: list | None = None,
) -> dict:
    """Generic gene-split CV runner for any sklearn classifier."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(labels)
    classes = le.classes_
    if splits is None:
        from esm2_mech.utils.splits import gene_split_cv
        splits = gene_split_cv(genes, n_folds=n_folds, seed=seed)

    fold_results = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_tr, X_te = X[train_idx].astype(np.float32), X[test_idx].astype(np.float32)
        y_tr, y_te = y[train_idx], y[test_idx]
        if len(set(y_tr)) < 2:
            continue
        if normalize:
            mu, std = X_tr.mean(0), X_tr.std(0) + 1e-8
            X_tr, X_te = (X_tr - mu) / std, (X_te - mu) / std
        clf = clf_fn(seed)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        fm = {"macro_f1": float(f1_score(y_te, pred, average="macro", zero_division=0))}
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_te)
            for i, cls in enumerate(classes):
                y_bin = (y_te == i).astype(int)
                if y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
                    fm[f"auroc_{cls}"] = float(roc_auc_score(y_bin, proba[:, i]))
        fold_results.append(fm)
        print(f"  Fold {fold_i+1}: macro_f1={fm['macro_f1']:.3f}")

    if not fold_results:
        return {"error": "insufficient data"}
    agg = {}
    for key in fold_results[0]:
        vals = [f[key] for f in fold_results if key in f and not np.isnan(f[key])]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
    return agg


def run_sklearn_probe_pca(
    clf_fn, X: np.ndarray, labels: np.ndarray, genes: np.ndarray,
    n_folds: int = 5, seed: int = 42, n_pca: int = 50,
    splits: list | None = None,
) -> dict:
    """Gene-split CV with per-fold PCA reduction and normalization."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(labels)
    classes = le.classes_
    if splits is None:
        from esm2_mech.utils.splits import gene_split_cv
        splits = gene_split_cv(genes, n_folds=n_folds, seed=seed)

    fold_results = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_tr, X_te = X[train_idx].astype(np.float32), X[test_idx].astype(np.float32)
        y_tr, y_te = y[train_idx], y[test_idx]
        if len(set(y_tr)) < 2:
            continue
        mu, std = X_tr.mean(0), X_tr.std(0) + 1e-8
        X_tr, X_te = (X_tr - mu) / std, (X_te - mu) / std
        pca = PCA(n_components=min(n_pca, X_tr.shape[1], X_tr.shape[0] - 1), random_state=seed)
        X_tr, X_te = pca.fit_transform(X_tr), pca.transform(X_te)
        clf = clf_fn(seed)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        fm = {"macro_f1": float(f1_score(y_te, pred, average="macro", zero_division=0))}
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_te)
            for i, cls in enumerate(classes):
                y_bin = (y_te == i).astype(int)
                if y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
                    fm[f"auroc_{cls}"] = float(roc_auc_score(y_bin, proba[:, i]))
        fold_results.append(fm)
        print(f"  Fold {fold_i+1}: macro_f1={fm['macro_f1']:.3f}")

    if not fold_results:
        return {"error": "insufficient data"}
    agg = {}
    for key in fold_results[0]:
        vals = [f[key] for f in fold_results if key in f and not np.isnan(f[key])]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
    return agg


def run_mlp_probe_cv(
    X: np.ndarray,
    labels: np.ndarray,
    splits: list[tuple],
    seed: int = 42,
    hidden: tuple = (256, 64),
    dropout: float = 0.3,
    lr: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 10,
    batch_size: int = 256,
    genes: np.ndarray | None = None,
    label: str = "",
) -> dict:
    """PyTorch MLP multi-class CV returning macro-F1 and per-class AUROC mean ± std.

    genes : if provided, the 15% validation split is gene-disjoint (recommended);
            otherwise 15% of samples are held out randomly.
    label : prefix for per-fold log lines.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import roc_auc_score, f1_score

    classes = MECHANISM_CLASSES
    n_classes = len(classes)
    cls_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    y = np.array([cls_to_idx[lab] for lab in labels])
    fold_results, pg_f1s = [], []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_tr = X[train_idx].astype(np.float32)
        X_te = X[test_idx].astype(np.float32)
        y_tr, y_te = y[train_idx], y[test_idx]
        labels_te = labels[test_idx]

        if len(set(y_tr.tolist())) < 2:
            print(f"    [{label}] Fold {fold_i+1}: skipped (< 2 classes in train)")
            continue

        rng = np.random.RandomState(seed + fold_i)
        if genes is not None:
            tr_genes = genes[train_idx]
            unique_tr_genes = np.array(sorted(set(tr_genes)))
            rng.shuffle(unique_tr_genes)
            n_val_genes = max(1, int(0.15 * len(unique_tr_genes)))
            val_gene_set = set(unique_tr_genes[:n_val_genes])
            val_mask = np.array([g in val_gene_set for g in tr_genes])
        else:
            order = np.arange(len(train_idx))
            rng.shuffle(order)
            n_val = max(1, int(0.15 * len(order)))
            val_mask = np.zeros(len(train_idx), dtype=bool)
            val_mask[order[:n_val]] = True
        fit_mask = ~val_mask

        X_fit, y_fit = X_tr[fit_mask], y_tr[fit_mask]
        X_val, y_val = X_tr[val_mask], y_tr[val_mask]
        if len(X_fit) < 10 or len(X_val) < 5:
            continue

        mu = X_fit.mean(0)
        std = X_fit.std(0) + 1e-8
        X_fit = (X_fit - mu) / std
        X_val = (X_val - mu) / std
        X_te_n = (X_te - mu) / std

        # Class weights from full training fold to avoid weight explosion when a
        # rare class is absent from the fit subset.
        class_counts = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cw = torch.tensor(1.0 / (class_counts + 1e-8)).to(device)

        layers: list = []
        prev = X_fit.shape[1]
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        model = nn.Sequential(*layers).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        crit = nn.CrossEntropyLoss(weight=cw)

        ds = TensorDataset(torch.tensor(X_fit), torch.tensor(y_fit, dtype=torch.long))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        best_val, patience_cnt, best_state = float("inf"), 0, None
        for _epoch in range(max_epochs):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                crit(model(xb.to(device)), yb.to(device)).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = crit(
                    model(torch.tensor(X_val).to(device)),
                    torch.tensor(y_val, dtype=torch.long).to(device),
                ).item()
            if vl < best_val - 1e-4:
                best_val, patience_cnt = vl, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    break
        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            proba = torch.softmax(model(torch.tensor(X_te_n).to(device)), 1).cpu().numpy()
        pred_labels = np.array([classes[idx] for idx in proba.argmax(1)])
        fm = {"macro_f1": float(f1_score(labels_te, pred_labels, average="macro", zero_division=0))}
        for col_idx, cls in enumerate(classes):
            y_bin = (labels_te == cls).astype(int)
            if y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
                fm[f"auroc_{cls}"] = float(roc_auc_score(y_bin, proba[:, col_idx]))
        fold_results.append(fm)

        pg_str = ""
        if genes is not None:
            pg = _per_gene_f1(labels_te, proba, genes[test_idx])
            pg_f1s.append(pg)
            pg_str = f"  per_gene_f1={pg:.3f}"
        print(f"    [{label}] Fold {fold_i+1}: macro_f1={fm['macro_f1']:.3f}{pg_str}")

    if not fold_results:
        return {}
    agg: dict = {}
    for key in set().union(*[set(f) for f in fold_results]):
        vals = [f[key] for f in fold_results if key in f and not np.isnan(f[key])]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
    if pg_f1s:
        agg["per_gene_f1_mean"] = float(np.mean(pg_f1s))
        agg["per_gene_f1_std"] = float(np.std(pg_f1s))
    return agg


