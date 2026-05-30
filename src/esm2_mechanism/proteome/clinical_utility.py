"""
Clinical Utility Experiment (Result 14)
========================================

Evaluates whether gene-level proteome features can identify GOF and DN
outliers within the ClinGen HI=3 gene set using HONEST out-of-sample
probabilities from family-split cross-validation.

Key design choices vs naive version:
  - Family-split CV (5-fold, same scheme as results 11-13): HI=3 genes are
    scored only when their Pfam family is in the held-out test fold.
    Singleton genes (no Pfam annotation) are each treated as their own
    family and appear in exactly one test fold.
  - Two feature sets: FULL (37 features including *_missing indicators) and
    NO-MISSING (18 features: 9 substantive + 9 family residuals only).
    Report AUROCs for both; delta shows how much signal is study-bias artefact.
  - Baselines within HI=3: mis_z alone, paralog_count alone, PPI_degree alone.
    (pLI/LOEUF are tautologically at chance within HI=3 and are reported only
    to confirm the tautology, not as a meaningful comparison.)
  - Bootstrap CIs: 1000 gene-level resamples on the HI=3 subset.
  - Calibration: reliability diagram + ECE for LogReg probabilities.
  - PR curve reported alongside ROC; operating-point stats shown prominently.

Usage
-----
    python scripts/clinical_utility.py
    python scripts/clinical_utility.py --out-dir results/my_run
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.utils import resample
from sklearn.calibration import calibration_curve
import functools

print = functools.partial(print, flush=True)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from esm2_mechanism.utils_paths import DATA_DIR, RESULTS_DIR, PROJECT_ROOT

MERGED_GENE_LIST = DATA_DIR / "merged_gene_list.tsv"
PROTEOME_FEATURES = DATA_DIR / "proteome_features_aligned.npy"
PROTEOME_COLS = DATA_DIR / "proteome_feature_columns.json"
GENE_FEATURES_TSV = DATA_DIR / "gene_proteome_features.tsv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MECH_MAP = {"LOF": "LOF", "HI": "LOF", "GOF": "GOF", "DN": "DN", "AR": None}
CLASSES = ["GOF", "DN", "LOF"]
N_FOLDS = 5
N_BOOTSTRAP = 1000
RANDOM_STATE = 42
CONF_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# Feature set definitions
# ---------------------------------------------------------------------------
def get_feature_sets(feature_names: list[str]) -> dict[str, list[int]]:
    """
    Returns column index lists for two feature sets:
      FULL    — all 37 columns (substantive + missingness indicators + residuals)
      NO_MISS — 18 columns: 9 substantive + 9 family residuals, no *_missing flags
    """
    miss_suffixes = ("_missing",)
    no_miss_idx = [
        i
        for i, n in enumerate(feature_names)
        if not any(n.endswith(s) for s in miss_suffixes) and n != "is_singleton_family"
    ]
    full_idx = list(range(len(feature_names)))
    return {"FULL": full_idx, "NO_MISS": no_miss_idx}


# ---------------------------------------------------------------------------
# Family-split CV helpers
# ---------------------------------------------------------------------------
def build_family_folds(
    families: np.ndarray,
    n_folds: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """
    Assign each unique Pfam family to one of n_folds folds.
    Singleton genes (family = NaN or '') each form their own singleton family
    and are distributed evenly across folds.
    Returns a list of n_folds arrays, each containing gene indices for that fold's test set.
    """
    unique_fams = []
    seen = set()
    singleton_counter = 0
    for f in families:
        if pd.isna(f) or f == "":
            unique_fams.append(f"__singleton_{singleton_counter}")
            singleton_counter += 1
        elif f not in seen:
            unique_fams.append(f)
            seen.add(f)

    unique_fams = np.array(unique_fams)
    rng.shuffle(unique_fams)
    fold_assignments = {fam: i % n_folds for i, fam in enumerate(unique_fams)}

    test_indices = [[] for _ in range(n_folds)]
    singleton_counter = 0
    for gene_idx, fam in enumerate(families):
        if pd.isna(fam) or fam == "":
            key = f"__singleton_{singleton_counter}"
            singleton_counter += 1
        else:
            key = fam
        fold = fold_assignments.get(key, 0)
        test_indices[fold].append(gene_idx)

    return [np.array(idx) for idx in test_indices]


def run_family_split_cv(
    X: np.ndarray,
    y: np.ndarray,
    families: np.ndarray,
    feature_set_name: str,
    feature_idx: list[int],
    le: LabelEncoder,
) -> np.ndarray:
    """
    5-fold family-split CV. Returns out-of-sample probability matrix (n_genes, 3).
    Genes not scoreable in any fold (shouldn't happen) get uniform priors.
    """
    n = len(y)
    probs = np.full((n, len(CLASSES)), np.nan)

    rng = np.random.default_rng(RANDOM_STATE)
    test_fold_indices = build_family_folds(families, N_FOLDS, rng)

    X_sub = X[:, feature_idx]

    for fold_i, test_idx in enumerate(test_fold_indices):
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]

        X_train, y_train = X_sub[train_idx], y[train_idx]
        X_test = X_sub[test_idx]

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        # LogReg
        lr = LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            multi_class="ovr",
            random_state=RANDOM_STATE,
        )
        lr.fit(X_train_sc, y_train)
        raw_proba = lr.predict_proba(X_test_sc)
        aligned = np.zeros((len(test_idx), len(CLASSES)), dtype=np.float32)
        for ci, c in enumerate(lr.classes_):
            aligned[:, c] = raw_proba[:, ci]
        probs[test_idx] = aligned

    # Any genes never assigned to a test fold keep NaN — they received no
    # prediction. Do NOT fabricate a uniform probability; downstream AUROC/ECE
    # must drop these rows rather than score a made-up prediction.
    n_unpredicted = int(np.isnan(probs[:, 0]).sum())
    if n_unpredicted:
        print(
            f"  WARNING: {n_unpredicted} genes received no LR prediction (left as NaN)"
        )

    return probs


def run_mlp_cv(
    X: np.ndarray,
    y: np.ndarray,
    families: np.ndarray,
    feature_idx: list[int],
) -> np.ndarray:
    """MLP family-split CV with oversampling on train folds."""
    n = len(y)
    probs = np.full((n, len(CLASSES)), np.nan)

    rng = np.random.default_rng(RANDOM_STATE)
    test_fold_indices = build_family_folds(families, N_FOLDS, rng)

    X_sub = X[:, feature_idx]

    for fold_i, test_idx in enumerate(test_fold_indices):
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]

        X_train, y_train = X_sub[train_idx], y[train_idx]
        X_test = X_sub[test_idx]

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        # Oversample minority classes
        from collections import Counter

        counts = Counter(y_train)
        max_count = max(counts.values())
        X_bal, y_bal = [], []
        for cls_idx, cnt in counts.items():
            mask = y_train == cls_idx
            Xc, yc = X_train_sc[mask], y_train[mask]
            if cnt < max_count:
                Xc, yc = resample(
                    Xc, yc, replace=True, n_samples=max_count, random_state=RANDOM_STATE
                )
            X_bal.append(Xc)
            y_bal.append(yc)
        X_bal = np.vstack(X_bal)
        y_bal = np.concatenate(y_bal)

        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
        )
        mlp.fit(X_bal, y_bal)
        raw_proba = mlp.predict_proba(X_test_sc)
        aligned = np.zeros((len(test_idx), len(CLASSES)), dtype=np.float32)
        for ci, c in enumerate(mlp.classes_):
            aligned[:, c] = raw_proba[:, ci]
        probs[test_idx] = aligned

    # Genes never assigned to a test fold keep NaN (no prediction). Do NOT
    # fabricate a uniform probability; downstream AUROC/ECE drop these rows.
    n_unpredicted = int(np.isnan(probs[:, 0]).sum())
    if n_unpredicted:
        print(
            f"  WARNING: {n_unpredicted} genes received no MLP prediction (left as NaN)"
        )

    return probs


# ---------------------------------------------------------------------------
# Bootstrap CI helper
# ---------------------------------------------------------------------------
def bootstrap_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_STATE,
) -> tuple[float, float, float]:
    """Returns (point_estimate, ci_low, ci_high) at 95%."""
    point = roc_auc_score(y_true, y_score)
    rng = np.random.default_rng(seed)
    boot_aurocs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_aurocs.append(roc_auc_score(yt, ys))
    boot_aurocs = np.array(boot_aurocs)
    return (
        point,
        float(np.percentile(boot_aurocs, 2.5)),
        float(np.percentile(boot_aurocs, 97.5)),
    )


# ---------------------------------------------------------------------------
# Calibration helper
# ---------------------------------------------------------------------------
def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error (binary)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(acc - conf)
    return float(ece)


# ---------------------------------------------------------------------------
# HI=3 analysis
# ---------------------------------------------------------------------------
def hi3_analysis(
    df: pd.DataFrame,
    probs_lr_full: np.ndarray,
    probs_lr_nomiss: np.ndarray,
    probs_mlp_full: np.ndarray,
    le: LabelEncoder,
    feature_names: list[str],
    out_dir: Path,
) -> dict:
    gof_idx = list(le.classes_).index("GOF")
    dn_idx = list(le.classes_).index("DN")

    hi3 = df[df["HI_score"] == 3.0].copy()
    hi3 = hi3[hi3["mechanism_3class"].notna()].copy()

    orig_idx = hi3.index.values

    hi3 = hi3.reset_index(drop=True)
    hi3["P_GOF_lr_full"] = probs_lr_full[orig_idx, gof_idx]
    hi3["P_DN_lr_full"] = probs_lr_full[orig_idx, dn_idx]
    hi3["P_GOF_lr_nomiss"] = probs_lr_nomiss[orig_idx, gof_idx]
    hi3["P_DN_lr_nomiss"] = probs_lr_nomiss[orig_idx, dn_idx]
    hi3["P_GOF_mlp_full"] = probs_mlp_full[orig_idx, gof_idx]
    hi3["P_DN_mlp_full"] = probs_mlp_full[orig_idx, dn_idx]

    results = {
        "n_hi3_total": len(hi3),
        "n_hi3_GOF": int((hi3["mechanism_3class"] == "GOF").sum()),
        "n_hi3_DN": int((hi3["mechanism_3class"] == "DN").sum()),
        "n_hi3_LOF": int((hi3["mechanism_3class"] == "LOF").sum()),
    }

    # --- H1: GOF vs LOF ---
    hi3_gl = hi3[hi3["mechanism_3class"].isin(["GOF", "LOF"])]
    y_gof = (hi3_gl["mechanism_3class"] == "GOF").astype(int).values

    for label, col in [
        ("LR_FULL", "P_GOF_lr_full"),
        ("LR_NOMISS", "P_GOF_lr_nomiss"),
        ("MLP_FULL", "P_GOF_mlp_full"),
    ]:
        # Drop genes with no prediction (NaN score) — never score a fabricated value.
        scores = hi3_gl[col].values
        valid = ~np.isnan(scores)
        y_v, scores_v = y_gof[valid], scores[valid]
        if len(np.unique(y_v)) == 2:
            pt, lo, hi_ = bootstrap_auroc(y_v, scores_v)
            results[f"H1_GOF_vs_LOF_AUROC_{label}"] = pt
            results[f"H1_GOF_vs_LOF_CI95_{label}"] = [lo, hi_]
        else:
            results[f"H1_GOF_vs_LOF_AUROC_{label}"] = float("nan")

    # Non-tautological single-feature baselines within HI=3
    for feat in ["mis_z", "paralog_count", "PPI_degree", "pLI", "LOEUF"]:
        if feat in hi3_gl.columns:
            sub = hi3_gl.dropna(subset=[feat])
            y_sub = (sub["mechanism_3class"] == "GOF").astype(int).values
            if len(np.unique(y_sub)) == 2:
                score = sub[feat].values
                # For LOEUF: lower = more constrained = more LOF, so flip
                if feat == "LOEUF":
                    score = -score
                pt, lo, hi_ = bootstrap_auroc(y_sub, score)
                results[f"H1_baseline_{feat}_AUROC"] = pt
                results[f"H1_baseline_{feat}_CI95"] = [lo, hi_]

    # --- H2: DN vs LOF ---
    hi3_dl = hi3[hi3["mechanism_3class"].isin(["DN", "LOF"])]
    y_dn = (hi3_dl["mechanism_3class"] == "DN").astype(int).values

    for label, col in [
        ("LR_FULL", "P_DN_lr_full"),
        ("LR_NOMISS", "P_DN_lr_nomiss"),
        ("MLP_FULL", "P_DN_mlp_full"),
    ]:
        # Drop genes with no prediction (NaN score) — never score a fabricated value.
        scores = hi3_dl[col].values
        valid = ~np.isnan(scores)
        y_v, scores_v = y_dn[valid], scores[valid]
        if len(np.unique(y_v)) == 2:
            pt, lo, hi_ = bootstrap_auroc(y_v, scores_v)
            results[f"H2_DN_vs_LOF_AUROC_{label}"] = pt
            results[f"H2_DN_vs_LOF_CI95_{label}"] = [lo, hi_]

    # --- H3: missingness ablation delta ---
    results["H3_missingness_ablation_delta_GOF"] = (
        results["H1_GOF_vs_LOF_AUROC_LR_FULL"]
        - results["H1_GOF_vs_LOF_AUROC_LR_NOMISS"]
    )
    results["H3_missingness_ablation_delta_DN"] = (
        results["H2_DN_vs_LOF_AUROC_LR_FULL"] - results["H2_DN_vs_LOF_AUROC_LR_NOMISS"]
    )

    # --- Calibration (GOF binary, LR_FULL) ---
    y_cal = (hi3["mechanism_3class"] == "GOF").astype(int).values
    p_cal = hi3["P_GOF_lr_full"].values
    # Drop genes with no prediction (NaN) — calibrate over real predictions only.
    cal_valid = ~np.isnan(p_cal)
    ece = (
        compute_ece(y_cal[cal_valid], p_cal[cal_valid])
        if cal_valid.any()
        else float("nan")
    )
    results["calibration_ECE_GOF_LR_FULL"] = ece

    # --- Named outlier gene report ---
    outliers = hi3[hi3["mechanism_3class"].isin(["GOF", "DN"])].copy()
    results["outlier_genes"] = (
        outliers[
            [
                "gene",
                "mechanism_3class",
                "P_GOF_lr_full",
                "P_DN_lr_full",
                "P_GOF_lr_nomiss",
                "P_DN_lr_nomiss",
                "P_GOF_mlp_full",
                "P_DN_mlp_full",
            ]
        ]
        .sort_values(["mechanism_3class", "P_GOF_lr_full"], ascending=[True, False])
        .to_dict(orient="records")
    )

    # --- Operating point stats ---
    for label, col in [("LR_FULL", "P_GOF_lr_full"), ("LR_NOMISS", "P_GOF_lr_nomiss")]:
        above = hi3[hi3[col] > CONF_THRESHOLD]
        tp = (above["mechanism_3class"] == "GOF").sum()
        fp = (above["mechanism_3class"] != "GOF").sum()
        n_gof = results["n_hi3_GOF"]
        results[f"GOF_recall_at_0.4_{label}"] = (
            float(tp / n_gof) if n_gof > 0 else float("nan")
        )
        results[f"GOF_precision_at_0.4_{label}"] = (
            float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
        )
        results[f"GOF_flagged_at_0.4_{label}"] = int(tp + fp)

    # --- Plots ---
    _plot_roc_pr(hi3, results, out_dir)
    _plot_calibration(hi3, out_dir)
    _plot_prob_distributions(hi3, out_dir)

    return results, hi3


def _plot_roc_pr(hi3: pd.DataFrame, results: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    configs = [
        (
            "GOF",
            "P_GOF_lr_full",
            "P_GOF_lr_nomiss",
            "P_GOF_mlp_full",
            "H1_GOF_vs_LOF_AUROC",
            "H1_GOF_vs_LOF_CI95",
        ),
        (
            "DN",
            "P_DN_lr_full",
            "P_DN_lr_nomiss",
            "P_DN_mlp_full",
            "H2_DN_vs_LOF_AUROC",
            "H2_DN_vs_LOF_CI95",
        ),
    ]

    for row, (pos_cls, lr_full, lr_nomiss, mlp_col, auroc_key, ci_key) in enumerate(
        configs
    ):
        subset = hi3[hi3["mechanism_3class"].isin([pos_cls, "LOF"])]
        y_true = (subset["mechanism_3class"] == pos_cls).astype(int).values

        if len(np.unique(y_true)) < 2:
            continue

        ax_roc = axes[row, 0]
        ax_pr = axes[row, 1]

        for label, col, color, ls in [
            ("LR full", lr_full, "#e74c3c", "-"),
            ("LR no-miss", lr_nomiss, "#e67e22", "--"),
            ("MLP full", mlp_col, "#3498db", "-."),
        ]:
            scores = subset[col].values
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc = roc_auc_score(y_true, scores)
            key_suffix = label.replace(" ", "_").upper().replace("-", "_")
            ci = results.get(
                f"{auroc_key}_{key_suffix.replace('LR_FULL','LR_FULL').replace('LR_NO_MISS','LR_NOMISS').replace('MLP_FULL','MLP_FULL')}",
                [None, None],
            )
            ax_roc.plot(
                fpr,
                tpr,
                color=color,
                lw=2,
                linestyle=ls,
                label=f"{label} AUC={auc:.3f}",
            )

            prec, rec, _ = precision_recall_curve(y_true, scores)
            ap = average_precision_score(y_true, scores)
            ax_pr.plot(
                rec, prec, color=color, lw=2, linestyle=ls, label=f"{label} AP={ap:.3f}"
            )

        ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.set_title(f"ROC — {pos_cls} vs LOF within HI=3")
        ax_roc.legend(fontsize=8)

        # Chance line for PR
        base_rate = y_true.mean()
        ax_pr.axhline(
            base_rate,
            color="gray",
            lw=0.8,
            linestyle="--",
            label=f"chance ({base_rate:.3f})",
        )
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title(f"PR — {pos_cls} vs LOF within HI=3")
        ax_pr.legend(fontsize=8)

    plt.suptitle(
        "ROC and PR curves — ClinGen HI=3 (out-of-sample, family-split CV)", fontsize=12
    )
    plt.tight_layout()
    plt.savefig(out_dir / "hi3_roc_pr_curves.png", dpi=150)
    plt.close()


def _plot_calibration(hi3: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, (pos_cls, col, title) in zip(
        axes,
        [
            ("GOF", "P_GOF_lr_full", "Calibration — P(GOF), LR full, HI=3"),
            ("DN", "P_DN_lr_full", "Calibration — P(DN), LR full, HI=3"),
        ],
    ):
        y_true = (hi3["mechanism_3class"] == pos_cls).astype(int).values
        y_prob = hi3[col].values

        frac_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=8, strategy="quantile"
        )
        ece = compute_ece(y_true, y_prob)

        ax.plot(
            mean_pred,
            frac_pos,
            "s-",
            color="#e74c3c",
            lw=2,
            label=f"LogReg (ECE={ece:.3f})",
        )
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="perfect")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction positive")
        ax.set_title(title)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "calibration.png", dpi=150)
    plt.close()


def _plot_prob_distributions(hi3: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    colors = {"GOF": "#e74c3c", "DN": "#3498db", "LOF": "#95a5a6"}

    for ax, (col, xlabel, title) in zip(
        axes,
        [
            (
                "P_GOF_lr_full",
                "P(GOF) — LR full features",
                "P(GOF) by true class within HI=3",
            ),
            (
                "P_DN_lr_full",
                "P(DN) — LR full features",
                "P(DN) by true class within HI=3",
            ),
        ],
    ):
        for mech in ["GOF", "DN", "LOF"]:
            vals = hi3[hi3["mechanism_3class"] == mech][col].values
            x_pos = ["GOF", "DN", "LOF"].index(mech)
            jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(vals))
            n = (hi3["mechanism_3class"] == mech).sum()
            ax.scatter(
                jitter + x_pos,
                vals,
                alpha=0.5,
                s=18,
                color=colors[mech],
                label=f"{mech} (n={n})",
            )
            ax.boxplot(
                vals,
                positions=[x_pos],
                widths=0.25,
                patch_artist=True,
                boxprops=dict(facecolor=colors[mech], alpha=0.3),
                medianprops=dict(color="black", lw=2),
                showfliers=False,
            )

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["GOF", "DN", "LOF"])
        ax.set_ylabel(xlabel)
        ax.set_title(title)
        ax.axhline(
            CONF_THRESHOLD,
            color="black",
            linestyle="--",
            lw=0.8,
            alpha=0.5,
            label=f"thresh={CONF_THRESHOLD}",
        )
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)

    plt.suptitle(
        "Predicted probabilities within ClinGen HI=3 (out-of-sample)", fontsize=12
    )
    plt.tight_layout()
    plt.savefig(out_dir / "hi3_prob_distributions.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Unannotated gene analysis (H4)
# ---------------------------------------------------------------------------
def unannotated_analysis(
    df: pd.DataFrame,
    probs_lr_full: np.ndarray,
    le: LabelEncoder,
    out_dir: Path,
) -> dict:
    gof_idx = list(le.classes_).index("GOF")
    dn_idx = list(le.classes_).index("DN")

    df = df.copy()
    df["P_GOF_lr"] = probs_lr_full[:, gof_idx]
    df["P_DN_lr"] = probs_lr_full[:, dn_idx]
    df["P_LOF_lr"] = probs_lr_full[:, list(le.classes_).index("LOF")]
    df["pred_class_lr"] = le.classes_[np.argmax(probs_lr_full, axis=1)]

    unannotated = df[df["HI_score"] != 3.0].copy()
    unannotated_labeled = unannotated[unannotated["mechanism_3class"].notna()].copy()

    results = {
        "n_unannotated_total": len(unannotated),
        "n_unannotated_HI0": int((unannotated["HI_score"] == 0.0).sum()),
        "n_unannotated_NaN": int(unannotated["HI_score"].isna().sum()),
        "n_unannotated_labeled": len(unannotated_labeled),
    }

    pred_class = le.classes_[
        np.argmax(probs_lr_full[unannotated_labeled.index], axis=1)
    ]
    results["H4_predicted_distribution_LogReg"] = {
        m: int((pred_class == m).sum()) for m in CLASSES
    }

    cols = ["gene", "mechanism_3class", "P_GOF_lr", "P_DN_lr", "P_LOF_lr", "HI_score"]
    results["H4_top20_GOF"] = unannotated_labeled.nlargest(20, "P_GOF_lr")[
        cols
    ].to_dict(orient="records")
    results["H4_top20_DN"] = unannotated_labeled.nlargest(20, "P_DN_lr")[cols].to_dict(
        orient="records"
    )

    errors = unannotated_labeled[
        (unannotated_labeled["mechanism_3class"] == "LOF")
        & (unannotated_labeled["pred_class_lr"].isin(["GOF", "DN"]))
    ]
    results["n_LOF_predicted_GOF_or_DN"] = len(errors)

    _plot_unannotated(unannotated_labeled, probs_lr_full, le, out_dir)

    return results


def _plot_unannotated(
    df: pd.DataFrame, probs: np.ndarray, le: LabelEncoder, out_dir: Path
) -> None:
    from matplotlib.patches import Patch

    gof_idx = list(le.classes_).index("GOF")
    dn_idx = list(le.classes_).index("DN")
    colors = {"GOF": "#e74c3c", "DN": "#3498db", "LOF": "#95a5a6"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, (prob_col_name, prob_idx, title) in zip(
        axes,
        [
            (
                "P_GOF_lr",
                gof_idx,
                "Top 30 predicted GOF\n(unannotated genes, colored by true label)",
            ),
            (
                "P_DN_lr",
                dn_idx,
                "Top 30 predicted DN\n(unannotated genes, colored by true label)",
            ),
        ],
    ):
        top = df.nlargest(30, prob_col_name)
        p_vals = probs[top.index, prob_idx]
        gene_names = top["gene"].values
        bar_colors = [colors.get(m, "#bdc3c7") for m in top["mechanism_3class"].values]

        ax.barh(range(len(gene_names)), p_vals[::-1], color=bar_colors[::-1])
        ax.set_yticks(range(len(gene_names)))
        ax.set_yticklabels(gene_names[::-1], fontsize=7)
        ax.axvline(CONF_THRESHOLD, color="black", linestyle="--", lw=0.8, alpha=0.5)
        ax.set_xlabel("Predicted probability (LR full, family-split CV)")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 1)

        legend_elements = [
            Patch(facecolor="#e74c3c", label="GOF (true)"),
            Patch(facecolor="#3498db", label="DN (true)"),
            Patch(facecolor="#95a5a6", label="LOF (true)"),
        ]
        ax.legend(handles=legend_elements, fontsize=7)

    plt.tight_layout()
    plt.savefig(out_dir / "unannotated_top_predictions.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Feature importance (from full-data LogReg for interpretability only)
# ---------------------------------------------------------------------------
def feature_importance_plot(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    feature_idx: list[int],
    le: LabelEncoder,
    out_dir: Path,
) -> dict:
    """
    Fit a single full-data LogReg on the NO_MISS feature set for
    interpretable coefficients (no missingness confound).
    """
    X_sub = X[:, feature_idx]
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_sub)

    lr = LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        multi_class="ovr",
        random_state=RANDOM_STATE,
    )
    lr.fit(X_sc, y)

    sub_names = [feature_names[i] for i in feature_idx]
    coef = lr.coef_  # (n_classes, n_features)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    results = {}

    for i, (cls, ax) in enumerate(zip(le.classes_, axes)):
        coefs = coef[i]
        top_idx = np.argsort(np.abs(coefs))[-15:][::-1]
        top_names = [sub_names[j] for j in top_idx]
        top_vals = coefs[top_idx]
        bar_colors = ["#e74c3c" if v > 0 else "#3498db" for v in top_vals]

        ax.barh(range(len(top_names)), top_vals[::-1], color=bar_colors[::-1])
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names[::-1], fontsize=8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(
            f"Top features for {cls}\n(full-data LR, no-miss features)", fontsize=10
        )
        ax.set_xlabel("Coefficient")

        results[f"top_features_{cls}"] = [
            {"feature": n, "coefficient": float(v)} for n, v in zip(top_names, top_vals)
        ]

    plt.suptitle(
        "LogReg feature importance (NO-MISS features, full-data fit)", fontsize=12
    )
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close()

    return results


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------
def apply_decision_rule(hi3_results: dict) -> str:
    # Missing/uncomputable AUROC must stay NaN, not 0.0 — a fabricated 0.0 would
    # emit a confident "NULL" verdict for a metric that was never computed.
    auroc = hi3_results.get("H1_GOF_vs_LOF_AUROC_LR_FULL", float("nan"))
    ci = hi3_results.get("H1_GOF_vs_LOF_CI95_LR_FULL", [None, None])
    ci_str = f" [95% CI {ci[0]:.3f}–{ci[1]:.3f}]" if ci[0] is not None else ""
    if auroc is None or np.isnan(auroc):
        return "UNDETERMINED: GOF AUROC within HI=3 could not be computed"
    if auroc >= 0.65:
        return f"INFORMATIVE: GOF AUROC within HI=3 = {auroc:.3f}{ci_str} ≥ 0.65"
    elif auroc >= 0.55:
        return f"WEAKLY INFORMATIVE: GOF AUROC within HI=3 = {auroc:.3f}{ci_str} ≥ 0.55"
    else:
        return f"NULL: GOF AUROC within HI=3 = {auroc:.3f}{ci_str} < 0.55"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    gene_list = pd.read_csv(MERGED_GENE_LIST, sep="\t")
    feat_df = pd.read_csv(GENE_FEATURES_TSV, sep="\t")
    X_all = np.load(PROTEOME_FEATURES)  # (2424, 37)

    with open(PROTEOME_COLS) as f:
        col_meta = json.load(f)
    feature_names = col_meta["numerical_columns"]

    df = gene_list.merge(feat_df, on="gene", how="left")
    df["mechanism_3class"] = df["mechanism"].map(MECH_MAP)

    labeled = df[df["mechanism_3class"].notna()].copy()
    print(
        f"Labeled genes: {len(labeled)} (GOF={( labeled['mechanism_3class']=='GOF').sum()}, "
        f"DN={(labeled['mechanism_3class']=='DN').sum()}, "
        f"LOF={(labeled['mechanism_3class']=='LOF').sum()})"
    )

    labeled_idx = labeled.index.values
    X_labeled = X_all[labeled_idx]

    le = LabelEncoder()
    le.fit(CLASSES)
    y = le.transform(labeled["mechanism_3class"].values)

    families = labeled["pfam_family"].values

    # Feature set column indices
    feat_sets = get_feature_sets(feature_names)
    full_idx = feat_sets["FULL"]
    nomiss_idx = feat_sets["NO_MISS"]

    print(f"Feature sets: FULL={len(full_idx)} cols, NO_MISS={len(nomiss_idx)} cols")

    # --- Family-split CV: out-of-sample probabilities for ALL labeled genes ---
    print("\nRunning family-split CV (LogReg FULL features)...")
    oos_probs_lr_full = run_family_split_cv(
        X_labeled, y, families, "FULL", full_idx, le
    )

    print("Running family-split CV (LogReg NO-MISS features)...")
    oos_probs_lr_nomiss = run_family_split_cv(
        X_labeled, y, families, "NO_MISS", nomiss_idx, le
    )

    print("Running family-split CV (MLP FULL features)...")
    oos_probs_mlp_full = run_mlp_cv(X_labeled, y, families, full_idx)

    # Map OOS probabilities back to full df index space
    # (oos_probs are indexed 0..len(labeled)-1; need to map to df.index)
    probs_lr_full_df = np.full((len(df), len(CLASSES)), np.nan)
    probs_lr_nomiss_df = np.full((len(df), len(CLASSES)), np.nan)
    probs_mlp_full_df = np.full((len(df), len(CLASSES)), np.nan)
    for i, orig_i in enumerate(labeled_idx):
        probs_lr_full_df[orig_i] = oos_probs_lr_full[i]
        probs_lr_nomiss_df[orig_i] = oos_probs_lr_nomiss[i]
        probs_mlp_full_df[orig_i] = oos_probs_mlp_full[i]

    # Save OOS probability TSV
    gof_idx_c = list(le.classes_).index("GOF")
    dn_idx_c = list(le.classes_).index("DN")
    lof_idx_c = list(le.classes_).index("LOF")

    probs_out = df[["gene", "mechanism", "mechanism_3class"]].copy()
    probs_out["P_GOF_lr_full"] = probs_lr_full_df[:, gof_idx_c]
    probs_out["P_DN_lr_full"] = probs_lr_full_df[:, dn_idx_c]
    probs_out["P_LOF_lr_full"] = probs_lr_full_df[:, lof_idx_c]
    has_probs = ~np.isnan(probs_lr_full_df[:, 0])
    pred_classes = np.full(len(df), None, dtype=object)
    pred_classes[has_probs] = le.classes_[
        np.argmax(probs_lr_full_df[has_probs], axis=1)
    ]
    probs_out["pred_class_lr_full"] = pred_classes
    probs_out.to_csv(out_dir / "gene_mechanism_probs.tsv", sep="\t", index=False)

    labeled[["gene", "mechanism", "mechanism_3class", "HI_score"]].to_csv(
        out_dir / "gene_labels.tsv", sep="\t", index=False
    )
    print("Saved gene_labels.tsv and gene_mechanism_probs.tsv")

    # --- HI=3 analysis ---
    print("\nRunning ClinGen HI=3 analysis...")
    hi3_results, hi3_df = hi3_analysis(
        df,
        probs_lr_full_df,
        probs_lr_nomiss_df,
        probs_mlp_full_df,
        le,
        feature_names,
        out_dir,
    )

    print(
        f"  HI=3: n={hi3_results['n_hi3_total']} (GOF={hi3_results['n_hi3_GOF']}, "
        f"DN={hi3_results['n_hi3_DN']}, LOF={hi3_results['n_hi3_LOF']})"
    )
    print(f"\n  H1 GOF-vs-LOF AUROC:")
    print(
        f"    LR FULL:    {hi3_results['H1_GOF_vs_LOF_AUROC_LR_FULL']:.3f} "
        f"[{hi3_results['H1_GOF_vs_LOF_CI95_LR_FULL'][0]:.3f}–"
        f"{hi3_results['H1_GOF_vs_LOF_CI95_LR_FULL'][1]:.3f}]"
    )
    print(
        f"    LR NO-MISS: {hi3_results['H1_GOF_vs_LOF_AUROC_LR_NOMISS']:.3f} "
        f"[{hi3_results['H1_GOF_vs_LOF_CI95_LR_NOMISS'][0]:.3f}–"
        f"{hi3_results['H1_GOF_vs_LOF_CI95_LR_NOMISS'][1]:.3f}]"
    )
    print(f"    MLP FULL:   {hi3_results['H1_GOF_vs_LOF_AUROC_MLP_FULL']:.3f}")
    print(
        f"    Missingness delta: {hi3_results['H3_missingness_ablation_delta_GOF']:+.3f}"
    )

    print(f"\n  Baselines (GOF vs LOF within HI=3):")
    for feat in ["mis_z", "paralog_count", "PPI_degree", "pLI", "LOEUF"]:
        k = f"H1_baseline_{feat}_AUROC"
        if k in hi3_results:
            ci = hi3_results.get(f"H1_baseline_{feat}_CI95", [None, None])
            ci_s = f" [{ci[0]:.3f}–{ci[1]:.3f}]" if ci[0] else ""
            print(f"    {feat:20s}: {hi3_results[k]:.3f}{ci_s}")

    print(f"\n  H2 DN-vs-LOF AUROC:")
    print(
        f"    LR FULL:    {hi3_results['H2_DN_vs_LOF_AUROC_LR_FULL']:.3f} "
        f"[{hi3_results['H2_DN_vs_LOF_CI95_LR_FULL'][0]:.3f}–"
        f"{hi3_results['H2_DN_vs_LOF_CI95_LR_FULL'][1]:.3f}]"
    )
    print(f"    LR NO-MISS: {hi3_results['H2_DN_vs_LOF_AUROC_LR_NOMISS']:.3f}")
    print(
        f"    Missingness delta: {hi3_results['H3_missingness_ablation_delta_DN']:+.3f}"
    )

    print(
        f"\n  Calibration ECE (GOF, LR full): {hi3_results['calibration_ECE_GOF_LR_FULL']:.3f}"
    )

    print(f"\n  Operating point (P_GOF > 0.4, LR FULL):")
    print(f"    Recall:    {hi3_results['GOF_recall_at_0.4_LR_FULL']:.3f}")
    print(f"    Precision: {hi3_results['GOF_precision_at_0.4_LR_FULL']:.3f}")
    print(f"    Flagged:   {hi3_results['GOF_flagged_at_0.4_LR_FULL']}")

    # --- Unannotated analysis ---
    print("\nRunning unannotated gene analysis...")
    unann_results = unannotated_analysis(df, probs_lr_full_df, le, out_dir)
    print(
        f"  Unannotated: {unann_results['n_unannotated_total']} "
        f"(HI=0: {unann_results['n_unannotated_HI0']}, NaN: {unann_results['n_unannotated_NaN']})"
    )
    print(
        f"  Predicted distribution: {unann_results['H4_predicted_distribution_LogReg']}"
    )

    # --- Feature importance (no-miss, full-data fit for interpretability) ---
    print("\nComputing feature importance (NO-MISS features)...")
    feat_results = feature_importance_plot(
        X_labeled, y, feature_names, nomiss_idx, le, out_dir
    )

    # --- Compile and save ---
    all_results = {
        "design": "family-split CV (5-fold), out-of-sample probabilities",
        "n_labeled_genes": len(labeled),
        "class_distribution": {
            "GOF": int((labeled["mechanism_3class"] == "GOF").sum()),
            "DN": int((labeled["mechanism_3class"] == "DN").sum()),
            "LOF": int((labeled["mechanism_3class"] == "LOF").sum()),
        },
        "feature_sets": {
            "FULL": len(full_idx),
            "NO_MISS": len(nomiss_idx),
        },
        "hi3": hi3_results,
        "unannotated": unann_results,
        "feature_importance": feat_results,
        "decision_rule": apply_decision_rule(hi3_results),
    }

    with open(out_dir / "clinical_utility_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nSaved clinical_utility_results.json")
    print(f"\n=== Decision rule ===")
    print(f"  {all_results['decision_rule']}")


def run_seed(
    seed: int,
    df: pd.DataFrame,
    X_all: np.ndarray,
    feature_names: list[str],
    le: LabelEncoder,
    out_dir: Path,
) -> dict:
    """Run one seed of family-split CV and return all HI=3 metrics."""
    labeled = df[df["mechanism_3class"].notna()].copy()
    labeled_idx = labeled.index.values
    X_labeled = X_all[labeled_idx]
    y = le.transform(labeled["mechanism_3class"].values)
    families = labeled["pfam_family"].values

    feat_sets = get_feature_sets(feature_names)
    full_idx = feat_sets["FULL"]
    nomiss_idx = feat_sets["NO_MISS"]

    # Override RANDOM_STATE with seed
    global RANDOM_STATE
    RANDOM_STATE = seed

    oos_lr_full = run_family_split_cv(X_labeled, y, families, "FULL", full_idx, le)
    oos_lr_nomiss = run_family_split_cv(
        X_labeled, y, families, "NO_MISS", nomiss_idx, le
    )
    oos_mlp_full = run_mlp_cv(X_labeled, y, families, full_idx)

    probs_lr_full_df = np.full((len(df), len(CLASSES)), np.nan)
    probs_lr_nomiss_df = np.full((len(df), len(CLASSES)), np.nan)
    probs_mlp_full_df = np.full((len(df), len(CLASSES)), np.nan)
    for i, orig_i in enumerate(labeled_idx):
        probs_lr_full_df[orig_i] = oos_lr_full[i]
        probs_lr_nomiss_df[orig_i] = oos_lr_nomiss[i]
        probs_mlp_full_df[orig_i] = oos_mlp_full[i]

    hi3_results, _ = hi3_analysis(
        df,
        probs_lr_full_df,
        probs_lr_nomiss_df,
        probs_mlp_full_df,
        le,
        feature_names,
        out_dir,
    )

    return {
        "seed": seed,
        "H1_GOF_vs_LOF_AUROC_LR_FULL": hi3_results["H1_GOF_vs_LOF_AUROC_LR_FULL"],
        "H1_GOF_vs_LOF_CI95_LR_FULL": hi3_results["H1_GOF_vs_LOF_CI95_LR_FULL"],
        "H1_GOF_vs_LOF_AUROC_LR_NOMISS": hi3_results["H1_GOF_vs_LOF_AUROC_LR_NOMISS"],
        "H2_DN_vs_LOF_AUROC_LR_FULL": hi3_results["H2_DN_vs_LOF_AUROC_LR_FULL"],
        "H2_DN_vs_LOF_CI95_LR_FULL": hi3_results["H2_DN_vs_LOF_CI95_LR_FULL"],
        "H2_DN_vs_LOF_AUROC_LR_NOMISS": hi3_results["H2_DN_vs_LOF_AUROC_LR_NOMISS"],
        "H3_missingness_delta_GOF": hi3_results["H3_missingness_ablation_delta_GOF"],
        "H3_missingness_delta_DN": hi3_results["H3_missingness_ablation_delta_DN"],
        "baselines": {
            feat: hi3_results.get(f"H1_baseline_{feat}_AUROC")
            for feat in ["mis_z", "paralog_count", "PPI_degree", "pLI", "LOEUF"]
        },
        "decision_rule": apply_decision_rule(hi3_results),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/clinical_utility")
    parser.add_argument(
        "--seed", type=int, default=None, help="Single seed (default: all 5 seeds 0-4)"
    )
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load shared data once
    print("Loading data...")
    gene_list = pd.read_csv(MERGED_GENE_LIST, sep="\t")
    feat_df = pd.read_csv(GENE_FEATURES_TSV, sep="\t")
    X_all = np.load(PROTEOME_FEATURES)

    with open(PROTEOME_COLS) as f:
        col_meta = json.load(f)
    feature_names = col_meta["numerical_columns"]

    df = gene_list.merge(feat_df, on="gene", how="left")
    df["mechanism_3class"] = df["mechanism"].map(MECH_MAP)

    le = LabelEncoder()
    le.fit(CLASSES)

    seeds = [args.seed] if args.seed is not None else list(range(5))

    if len(seeds) == 1:
        # Single-seed path: run full analysis with plots
        main(out_dir)
    else:
        # Multi-seed path: run HI=3 metrics per seed, save per-seed JSONs, aggregate
        seed_results = []
        for seed in seeds:
            print(f"\n{'='*50}\nSEED {seed}\n{'='*50}")
            sr = run_seed(seed, df, X_all, feature_names, le, out_dir)
            seed_results.append(sr)
            sr_path = out_dir / f"hi3_family_split_seed{seed}.json"
            with open(sr_path, "w") as f:
                json.dump(sr, f, indent=2, default=str)
            print(f"  Saved {sr_path.name}")
            print(
                f"  GOF AUROC LR_FULL: {sr['H1_GOF_vs_LOF_AUROC_LR_FULL']:.3f}  "
                f"DN AUROC LR_FULL: {sr['H2_DN_vs_LOF_AUROC_LR_FULL']:.3f}"
            )

        # Aggregate across seeds
        def mean_std(key):
            vals = [sr[key] for sr in seed_results if sr.get(key) is not None]
            return float(np.mean(vals)), float(np.std(vals))

        summary = {
            "seeds": seeds,
            "H1_GOF_AUROC_LR_FULL_mean": mean_std("H1_GOF_vs_LOF_AUROC_LR_FULL")[0],
            "H1_GOF_AUROC_LR_FULL_std": mean_std("H1_GOF_vs_LOF_AUROC_LR_FULL")[1],
            "H1_GOF_AUROC_LR_NOMISS_mean": mean_std("H1_GOF_vs_LOF_AUROC_LR_NOMISS")[0],
            "H1_GOF_AUROC_LR_NOMISS_std": mean_std("H1_GOF_vs_LOF_AUROC_LR_NOMISS")[1],
            "H2_DN_AUROC_LR_FULL_mean": mean_std("H2_DN_vs_LOF_AUROC_LR_FULL")[0],
            "H2_DN_AUROC_LR_FULL_std": mean_std("H2_DN_vs_LOF_AUROC_LR_FULL")[1],
            "H2_DN_AUROC_LR_NOMISS_mean": mean_std("H2_DN_vs_LOF_AUROC_LR_NOMISS")[0],
            "H2_DN_AUROC_LR_NOMISS_std": mean_std("H2_DN_vs_LOF_AUROC_LR_NOMISS")[1],
            "H3_missingness_delta_GOF_mean": mean_std("H3_missingness_delta_GOF")[0],
            "H3_missingness_delta_GOF_std": mean_std("H3_missingness_delta_GOF")[1],
            "H3_missingness_delta_DN_mean": mean_std("H3_missingness_delta_DN")[0],
            "H3_missingness_delta_DN_std": mean_std("H3_missingness_delta_DN")[1],
            "baselines_mean": {
                feat: float(
                    np.mean(
                        [
                            sr["baselines"][feat]
                            for sr in seed_results
                            if sr["baselines"].get(feat) is not None
                        ]
                    )
                )
                for feat in ["mis_z", "paralog_count", "PPI_degree", "pLI", "LOEUF"]
            },
            "per_seed": seed_results,
        }

        summary_path = out_dir / "hi3_family_split_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print("SUMMARY across seeds")
        print(f"{'='*60}")
        print(
            f"  H1 GOF AUROC LR FULL:    {summary['H1_GOF_AUROC_LR_FULL_mean']:.3f} ± {summary['H1_GOF_AUROC_LR_FULL_std']:.3f}"
        )
        print(
            f"  H1 GOF AUROC LR NO-MISS: {summary['H1_GOF_AUROC_LR_NOMISS_mean']:.3f} ± {summary['H1_GOF_AUROC_LR_NOMISS_std']:.3f}"
        )
        print(
            f"  H2 DN  AUROC LR FULL:    {summary['H2_DN_AUROC_LR_FULL_mean']:.3f} ± {summary['H2_DN_AUROC_LR_FULL_std']:.3f}"
        )
        print(
            f"  H2 DN  AUROC LR NO-MISS: {summary['H2_DN_AUROC_LR_NOMISS_mean']:.3f} ± {summary['H2_DN_AUROC_LR_NOMISS_std']:.3f}"
        )
        print(
            f"  H3 missingness delta GOF: {summary['H3_missingness_delta_GOF_mean']:+.3f} ± {summary['H3_missingness_delta_GOF_std']:.3f}"
        )
        print(f"\n  Baselines (mean across seeds):")
        for feat, val in summary["baselines_mean"].items():
            print(f"    {feat:20s}: {val:.3f}")
        print(f"\nSaved {summary_path.name}")
