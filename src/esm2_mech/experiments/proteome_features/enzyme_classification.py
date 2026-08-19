"""Enzyme type classification (kinase/protease/oxidoreductase/non-enzyme) from
ESM-2 WT mean-pooled embeddings, as a positive control for the mechanism arc.
"""

from __future__ import annotations

import argparse
import functools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from esm2_mech.utils.constants import (
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_N_RESAMPLES,
    N_SEEDS,
    mechanism_oof_cache_filename,
)
from esm2_mech.utils.metrics import fold_macro_f1, majority_baseline_f1
from esm2_mech.utils.splits import gene_split_cv, family_split_cv
from esm2_mech.utils.data import build_gene_to_row, load_pfam_map
from esm2_mech.utils.io import write_result_json
from esm2_mech.utils.bootstrap import (
    folds_to_arms,
    score_within_folds,
    adjudicate_equivalence,
    adjudicate_level,
    bootstrap_mechanism_metrics,
    family_or_gene_clusters,
    independent_cluster_bootstrap_diff,
    oof_permutation_pvalue,
    paired_oof_diff,
)
from esm2_mech.utils.paths import (
    EMB_WT_MEAN,
    ENZYME_CLASSIFICATION_JSON,
    ENZYME_LABELS_TSV,
    ENZYME_RESULTS_DIR,
    GENE_UNIVERSE,
    MECHANISM_AGGREGATE_JSON,
    PFAM_JSON,
    PROTEOME_FEATURES_ALIGNED,
    PROTEOME_FEATURE_COLUMNS_JSON,
    RESULTS_DIR,
    VALID_VARIANTS_JSON,
)

print = functools.partial(print, flush=True)

ENZYME_CLASSES = ["kinase", "protease", "oxidoreductase", "non-enzyme"]


def _load_mechanism_reference_f1() -> float | None:
    """Read the mechanism family-split F1 from the mechanism aggregate result."""
    if not MECHANISM_AGGREGATE_JSON.exists():
        print(f"  WARNING: {MECHANISM_AGGREGATE_JSON} not found — mechanism reference unavailable")
        return None
    with open(MECHANISM_AGGREGATE_JSON) as fh:
        agg = json.load(fh)
    val = (
        agg.get("across_seed", {})
        .get("family_split", {})
        .get("delta_mean", {})
        .get("macro_f1_seed_mean")
    )
    if val is None:
        print("  WARNING: macro_f1_seed_mean not found in mechanism aggregate — reference unavailable")
    return val


def _load_mechanism_family_oof() -> dict | None:
    """Load seed-0 mechanism family-split OOF labels for an independent CI."""
    path = RESULTS_DIR / mechanism_oof_cache_filename(0)
    if not path.exists():
        print(f"  WARNING: {path} not found — enzyme/mechanism difference CI unavailable")
        return None
    with open(path) as handle:
        cache = json.load(handle)
    oof = cache.get("delta_mean", {}).get("family_split")
    required = {"y_true", "pred", "genes"}
    if oof is None or not required.issubset(oof):
        raise ValueError(f"{path} lacks delta_mean family-split OOF fields {sorted(required)}")
    lengths = {key: len(oof[key]) for key in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{path} has misaligned mechanism OOF fields: {lengths}")
    return oof


def load_gene_embeddings() -> tuple:
    """Load per-gene WT embeddings by taking the first variant's embedding for each gene."""
    with open(VALID_VARIANTS_JSON) as fh:
        variants = json.load(fh)

    wt_mean = np.load(EMB_WT_MEAN)
    if len(variants) != wt_mean.shape[0]:
        raise ValueError(
            f"{VALID_VARIANTS_JSON} has {len(variants)} variants but "
            f"{EMB_WT_MEAN} has {wt_mean.shape[0]} rows — files are out of sync."
        )
    print(f"Loaded {len(variants)} variants, wt_mean shape: {wt_mean.shape}")

    gene_first_idx = {}
    gene_uniprot = {}
    for i, v in enumerate(variants):
        g = v["gene"]
        if g not in gene_first_idx:
            gene_first_idx[g] = i
            gene_uniprot[g] = v.get("uniprot_id", "")

    gene_list = list(gene_first_idx.keys())
    idxs = [gene_first_idx[g] for g in gene_list]
    X = wt_mean[idxs].astype(np.float32)

    print(f"Per-gene embeddings: {X.shape} ({len(gene_list)} genes)")
    return X, gene_list, [gene_uniprot[g] for g in gene_list]


def load_enzyme_labels() -> dict:
    """Load gene -> enzyme_4class from enzyme_labels.tsv."""
    import csv

    labels: dict[str, str] = {}
    with open(ENZYME_LABELS_TSV) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            labels[row["gene"]] = row["enzyme_4class"]
    print(f"Loaded enzyme labels for {len(labels)} genes")
    return labels


def load_proteome_features() -> tuple:
    """Load the proteome feature matrix aligned to gene_universe.tsv row order."""
    X = np.load(PROTEOME_FEATURES_ALIGNED).astype(np.float32)
    with open(PROTEOME_FEATURE_COLUMNS_JSON) as fh:
        cols = json.load(fh)
    genes = list(build_gene_to_row(GENE_UNIVERSE))
    if len(genes) != X.shape[0]:
        raise ValueError(
            f"{PROTEOME_FEATURES_ALIGNED} has {X.shape[0]} rows but "
            f"{GENE_UNIVERSE} lists {len(genes)} genes — not row-aligned."
        )
    print(f"Proteome features: {X.shape}, {len(genes)} genes")
    return X, genes


def _run_cv(
    clf_factory,
    scale: bool,
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple],
    classes: list[str],
    genes: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[dict, dict | None]:
    """Run CV, collecting OOF predictions for bootstrap CIs."""
    n_cls = len(classes)
    fold_f1s = []
    fold_aurocs = {c: [] for c in classes}
    oof_y, oof_proba, oof_genes, oof_rows, oof_folds = [], [], [], [], []

    for fold_i, (tr, te) in enumerate(splits):
        tr_clean = tr
        te_clean = te
        if np.isnan(X[tr]).any() or np.isnan(X[te]).any():
            tr_mask = ~np.isnan(X[tr]).any(axis=1)
            te_mask = ~np.isnan(X[te]).any(axis=1)
            tr_clean = tr[tr_mask]
            te_clean = te[te_mask]
            if len(tr_clean) == 0 or len(te_clean) == 0:
                continue

        if len(set(y[tr_clean])) < 2:
            continue
        if scale:
            sc = StandardScaler().fit(X[tr_clean])
            X_tr, X_te = sc.transform(X[tr_clean]), sc.transform(X[te_clean])
        else:
            X_tr, X_te = X[tr_clean], X[te_clean]
        clf = clf_factory(seed)
        clf.fit(X_tr, y[tr_clean])
        proba_raw = clf.predict_proba(X_te)

        proba = np.zeros((len(te_clean), n_cls), dtype=np.float32)
        for ci, c in enumerate(clf.classes_):
            if 0 <= c < n_cls:
                proba[:, c] = proba_raw[:, ci]

        pred = proba.argmax(axis=1)
        fold_f1s.append(float(f1_score(y[te_clean], pred, average="macro", zero_division=0)))

        for i, cls in enumerate(classes):
            y_bin = (y[te_clean] == i).astype(int)
            if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
                fold_aurocs[cls].append(float(roc_auc_score(y_bin, proba[:, i])))

        if genes is not None:
            oof_y.append(y[te_clean])
            oof_proba.append(proba)
            oof_genes.append(genes[te_clean])
            oof_rows.append(np.asarray(te_clean))
            oof_folds.append(np.full(len(te_clean), fold_i, dtype=int))

    result = {
        "macro_f1_mean": float(np.nanmean(fold_f1s)) if fold_f1s else None,
        "macro_f1_std": float(np.nanstd(fold_f1s)) if fold_f1s else None,
        "per_class_auroc_mean": {
            c: float(np.nanmean(v)) if v else None for c, v in fold_aurocs.items()
        },
        "per_class_auroc_std": {
            c: float(np.nanstd(v)) if v else None for c, v in fold_aurocs.items()
        },
        "n_folds": len(fold_f1s),
    }

    oof = None
    if oof_y:
        oof = {
            "y_true": np.concatenate(oof_y),
            "proba": np.concatenate(oof_proba),
            "genes": np.concatenate(oof_genes),
            "row_ids": np.concatenate(oof_rows),
            "folds": np.concatenate(oof_folds),
        }
    return result, oof


def run_logreg(X, y, splits, classes, genes=None, seed: int = 42):
    from sklearn.linear_model import LogisticRegression

    return _run_cv(
        lambda s: LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=s
        ),
        True, X, y, splits, classes, genes=genes, seed=seed,
    )


def run_mlp(X, y, splits, classes, genes=None, seed: int = 42):
    return _run_cv(
        lambda s: MLPClassifier(
            hidden_layer_sizes=(256, 64),
            activation="relu",
            alpha=1e-3,
            max_iter=300,
            random_state=s,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
        ),
        True, X, y, splits, classes, genes=genes, seed=seed,
    )


def run_histgb(X, y, splits, classes, genes=None, seed: int = 42):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return _run_cv(
        lambda s: HistGradientBoostingClassifier(
            max_iter=200, class_weight="balanced", random_state=s
        ),
        False, X, y, splits, classes, genes=genes, seed=seed,
    )


def run_multiseed(
    X: np.ndarray,
    y: np.ndarray,
    genes: list[str],
    pfam_map: dict,
    le: LabelEncoder,
    seeds: list[int],
    n_folds: int = 5,
    compute_ci: bool = True,
    n_boot: int = BOOTSTRAP_N_RESAMPLES,
    n_permutations: int = 0,
    mechanism_family_oof: dict | None = None,
) -> dict:
    """Run linear and nonlinear probes across seeds with optional cluster-bootstrap CIs."""
    linear_probe = run_logreg
    nonlinear_probe = run_mlp
    classes = list(le.classes_)
    genes_arr = np.array(genes)
    print(f"\nClasses: {classes}")
    print(f"Class distribution: {dict(Counter(y.tolist()))}")

    gs_f1s, fs_f1s, mlp_f1s = [], [], []
    gs_aurocs = {c: [] for c in classes}
    fs_aurocs = {c: [] for c in classes}
    mlp_aurocs = {c: [] for c in classes}

    seed0_fs_oof = None
    seed0_mlp_fs_oof = None

    for seed in seeds:
        print(f"\n  Seed {seed}:")

        gs_splits = gene_split_cv(genes_arr, n_folds=n_folds, seed=seed)
        gs, _ = linear_probe(X, y, gs_splits, classes, genes=genes_arr, seed=seed)
        gs_f1s.append(gs["macro_f1_mean"])
        for c in classes:
            v = gs["per_class_auroc_mean"].get(c)
            if v is not None:
                gs_aurocs[c].append(v)
        print(f"    LogReg gene-split  F1={gs['macro_f1_mean']:.3f}")

        fs_splits = family_split_cv(genes_arr, pfam_map, n_folds=n_folds, seed=seed)
        fs, fs_oof = linear_probe(X, y, fs_splits, classes, genes=genes_arr, seed=seed)
        fs_f1s.append(fs["macro_f1_mean"])
        for c in classes:
            v = fs["per_class_auroc_mean"].get(c)
            if v is not None:
                fs_aurocs[c].append(v)
        print(
            f"    LogReg family-split F1={fs['macro_f1_mean']:.3f}  "
            f"AUROC: "
            + " ".join(
                f"{c}={fs['per_class_auroc_mean'].get(c, float('nan')):.3f}"
                for c in classes
            )
        )

        if seed == seeds[0]:
            seed0_fs_oof = fs_oof

        mlp, mlp_oof = nonlinear_probe(X, y, fs_splits, classes, genes=genes_arr, seed=seed)
        mlp_f1s.append(mlp["macro_f1_mean"])
        for c in classes:
            v = mlp["per_class_auroc_mean"].get(c)
            if v is not None:
                mlp_aurocs[c].append(v)
        print(f"    MLP    family-split F1={mlp['macro_f1_mean']:.3f}")

        if seed == seeds[0]:
            seed0_mlp_fs_oof = mlp_oof

    maj_f1, _ = majority_baseline_f1(y, y)

    def _agg(vals):
        arr = np.array([v for v in vals if v is not None], dtype=float)
        if len(arr) == 0:
            return None, None
        return float(np.nanmean(arr)), float(np.nanstd(arr))

    gs_mean, gs_std = _agg(gs_f1s)
    fs_mean, fs_std = _agg(fs_f1s)
    mlp_mean, mlp_std = _agg(mlp_f1s)

    leakage_pct = None
    if gs_mean is not None and fs_mean is not None and gs_mean > maj_f1:
        leakage_pct = round(100.0 * (gs_mean - fs_mean) / (gs_mean - maj_f1), 1)

    print(f"\n  Results ({len(seeds)} seeds):")
    print(f"    Majority baseline:       F1={maj_f1:.3f}")
    print(f"    LogReg gene-split:       F1={gs_mean:.3f} +/- {gs_std:.3f}")
    print(
        f"    LogReg family-split:     F1={fs_mean:.3f} +/- {fs_std:.3f}  <- primary metric"
    )
    print(f"    MLP    family-split:     F1={mlp_mean:.3f} +/- {mlp_std:.3f}")
    if leakage_pct is not None:
        print(f"    Leakage fraction:        {leakage_pct:.1f}%")

    ci_result = None
    permutation_result = None
    oof_fs_f1 = None
    oof_mlp_f1 = None
    paired_mlp_vs_logreg = None
    independent_logreg_vs_mechanism = None

    def _oof_macro_f1(oof):
        """Seed-0 out-of-fold macro-F1, scored per fold and averaged."""
        y_str = np.array([classes[i] for i in oof["y_true"]])
        pred = np.array([classes[col] for col in oof["proba"].argmax(axis=1)])
        arms = folds_to_arms(pred, oof["folds"])

        def _fold_f1(block, arm_pred):
            return fold_macro_f1(y_str, block, arm_pred, classes)
        return y_str, score_within_folds(np.arange(len(y_str)), arms, _fold_f1)

    if seed0_fs_oof is not None:
        oof_y_str, oof_fs_f1 = _oof_macro_f1(seed0_fs_oof)
        print(f"\n  Seed-0 OOF LogReg family-split F1: {oof_fs_f1:.3f}")

    if seed0_mlp_fs_oof is not None:
        mlp_oof_y_str, oof_mlp_f1 = _oof_macro_f1(seed0_mlp_fs_oof)
        print(f"  Seed-0 OOF MLP family-split F1: {oof_mlp_f1:.3f}")

    if compute_ci and seed0_fs_oof is not None:
        print("\n  Computing cluster-bootstrap CIs (seed 0, family-split)...")
        clusters = family_or_gene_clusters(
            seed0_fs_oof["genes"], pfam_map, is_family_split=True
        )
        ci_result = bootstrap_mechanism_metrics(
            y_true=oof_y_str,
            proba=seed0_fs_oof["proba"],
            clusters=clusters,
            folds=seed0_fs_oof["folds"],
            classes=classes,
            n_resamples=n_boot,
            ci_level=BOOTSTRAP_CI_LEVEL,
            seed=0,
        )
        for metric_name, ci in ci_result.items():
            lo = ci.get("ci_lower")
            hi = ci.get("ci_upper")
            pt = ci.get("point")
            if lo is not None and hi is not None and pt is not None:
                print(f"    {metric_name}: {pt:.3f} [{lo:.3f}, {hi:.3f}]")

        if seed0_mlp_fs_oof is not None:
            print("\n  Computing paired CI: MLP minus LogReg (family-split)...")
            mlp_oof_for_diff = {**seed0_mlp_fs_oof, "y_true": mlp_oof_y_str}
            lr_oof_for_diff = {**seed0_fs_oof, "y_true": oof_y_str}
            paired_mlp_vs_logreg = paired_oof_diff(
                oof_a=mlp_oof_for_diff,
                oof_b=lr_oof_for_diff,
                pfam_map=pfam_map,
                label="2H MLP-LogReg",
                classes=classes,
                metric="macro_f1",
                is_family_split=True,
                n_resamples=n_boot,
                seed=0,
            )
            if paired_mlp_vs_logreg is not None:
                lo = paired_mlp_vs_logreg.get("ci_low")
                hi = paired_mlp_vs_logreg.get("ci_high")
                pt = paired_mlp_vs_logreg.get("point_diff")
                if lo is not None and hi is not None and pt is not None:
                    print(f"    MLP-LogReg diff: {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]")

        if mechanism_family_oof is not None:
            print("\n  Computing independent CI: enzyme LogReg minus mechanism...")
            enzyme_pred = np.array([classes[col] for col in seed0_fs_oof["proba"].argmax(axis=1)])
            mechanism_y = np.asarray(mechanism_family_oof["y_true"])
            mechanism_pred = np.asarray(mechanism_family_oof["pred"])
            enzyme_clusters = family_or_gene_clusters(
                seed0_fs_oof["genes"], pfam_map, is_family_split=True
            )
            mechanism_clusters = family_or_gene_clusters(
                np.asarray(mechanism_family_oof["genes"]), pfam_map, is_family_split=True
            )

            def _enzyme_f1(rows):
                return float(f1_score(oof_y_str[rows], enzyme_pred[rows], average="macro", zero_division=0))

            def _mechanism_f1(rows):
                return float(
                    f1_score(
                        mechanism_y[rows], mechanism_pred[rows], average="macro", zero_division=0
                    )
                )

            independent_logreg_vs_mechanism = independent_cluster_bootstrap_diff(
                enzyme_clusters,
                mechanism_clusters,
                _enzyme_f1,
                _mechanism_f1,
                n_resamples=n_boot,
                seed=0,
            )
            lo = independent_logreg_vs_mechanism.get("ci_low")
            hi = independent_logreg_vs_mechanism.get("ci_high")
            point = independent_logreg_vs_mechanism.get("point_diff")
            if lo is not None and hi is not None and point is not None:
                print(f"    enzyme-mechanism diff: {point:+.3f} [{lo:+.3f}, {hi:+.3f}]")

        if n_permutations > 0:
            print(f"\n  Computing permutation p-value ({n_permutations} reps)...")
            permutation_result = oof_permutation_pvalue(
                y_true=oof_y_str,
                proba=seed0_fs_oof["proba"],
                folds=seed0_fs_oof["folds"],
                groups=seed0_fs_oof["genes"],
                clusters=clusters,
                classes=classes,
                n_permutations=n_permutations,
                seed=0,
            )
            pval = permutation_result.get("p_value")
            print(f"    permutation p-value: {pval}")

    result = {
        "majority_f1": maj_f1,
        "logreg_gene_split": {
            "macro_f1_mean": gs_mean,
            "macro_f1_std": gs_std,
            "per_class_auroc_mean": {
                c: float(np.nanmean(v)) if v else None for c, v in gs_aurocs.items()
            },
            "n_seeds": len(seeds),
        },
        "logreg_family_split": {
            "macro_f1_mean": fs_mean,
            "macro_f1_std": fs_std,
            "per_class_auroc_mean": {
                c: float(np.nanmean(v)) if v else None for c, v in fs_aurocs.items()
            },
            "n_seeds": len(seeds),
        },
        "mlp_family_split": {
            "macro_f1_mean": mlp_mean,
            "macro_f1_std": mlp_std,
            "per_class_auroc_mean": {
                c: float(np.nanmean(v)) if v else None for c, v in mlp_aurocs.items()
            },
            "n_seeds": len(seeds),
        },
        "leakage_pct": leakage_pct,
    }

    if oof_fs_f1 is not None:
        result["logreg_family_split"]["oof_macro_f1"] = oof_fs_f1
    if oof_mlp_f1 is not None:
        result["mlp_family_split"]["oof_macro_f1"] = oof_mlp_f1
    if ci_result is not None:
        result["bootstrap_ci"] = ci_result
    if paired_mlp_vs_logreg is not None:
        result["paired_ci_mlp_minus_logreg"] = paired_mlp_vs_logreg
    if independent_logreg_vs_mechanism is not None:
        result["independent_ci_logreg_minus_mechanism"] = independent_logreg_vs_mechanism
    if permutation_result is not None:
        result["permutation_test"] = permutation_result

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=N_SEEDS,
                        help="number of seeds to run; runs 0..seeds-1 (>=1)")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--no_ci", action="store_true", help="skip cluster-bootstrap CIs")
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    parser.add_argument(
        "--n_permutations", type=int, default=0,
        help="label-permutation reps for OOF macro AUROC (0 = skip)",
    )
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")

    seeds = list(range(args.seeds))
    compute_ci = not args.no_ci

    ENZYME_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Enzyme Classification from ESM-2 WT Embeddings ===")
    print(f"Seeds: {seeds}  Folds: {args.n_folds}  CI: {compute_ci}  n_boot: {args.n_boot}")

    mechanism_ref_f1 = _load_mechanism_reference_f1()
    mechanism_family_oof = _load_mechanism_family_oof()

    X_emb, gene_list, _ = load_gene_embeddings()
    enzyme_labels = load_enzyme_labels()
    pfam_map = load_pfam_map(PFAM_JSON)

    missing = [g for g in gene_list if g not in enzyme_labels]
    if missing:
        print(
            f"  Excluding {len(missing)} genes with no enzyme label: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    labeled_mask = np.array([g in enzyme_labels for g in gene_list])
    X_emb = X_emb[labeled_mask]
    gene_list = [g for g in gene_list if g in enzyme_labels]

    y_str = [enzyme_labels[g] for g in gene_list]
    le = LabelEncoder()
    le.fit(ENZYME_CLASSES)
    y = le.transform(y_str)

    print(f"\nLabeled genes: {len(gene_list)}")
    print(f"Class distribution: {dict(Counter(y_str))}")
    print(
        f"Pfam-annotated genes: {sum(1 for g in gene_list if pfam_map.get(g))}/{len(gene_list)}"
    )

    print("\n" + "=" * 60)
    print("PART 1: ESM-2 WT embedding (1280-dim)")
    print("=" * 60)
    emb_results = run_multiseed(
        X_emb, y, gene_list, pfam_map, le, seeds=seeds, n_folds=args.n_folds,
        compute_ci=compute_ci, n_boot=args.n_boot, n_permutations=args.n_permutations,
        mechanism_family_oof=mechanism_family_oof,
    )

    print("\n" + "=" * 60)
    print("PART 2: Proteome features (37-dim) — baseline comparison")
    print("=" * 60)

    X_prot, prot_genes = load_proteome_features()
    prot_gene_to_idx = {g: i for i, g in enumerate(prot_genes)}

    gene_to_emb_idx = {g: i for i, g in enumerate(gene_list)}
    prot_aligned_idxs = [prot_gene_to_idx[g] for g in gene_list if g in prot_gene_to_idx]
    prot_aligned_genes = [g for g in gene_list if g in prot_gene_to_idx]
    prot_aligned_y = y[np.array([gene_to_emb_idx[g] for g in prot_aligned_genes])]
    X_prot_aligned = X_prot[prot_aligned_idxs]

    n_nan = int(np.isnan(X_prot_aligned).sum())
    if n_nan:
        n_rows = int(np.isnan(X_prot_aligned).any(axis=1).sum())
        print(
            f"  {n_nan} missing cells across {n_rows}/{len(X_prot_aligned)} genes "
            f"— NaN rows dropped per fold, not imputed"
        )

    print(f"Proteome-aligned genes: {len(prot_aligned_genes)}")
    proteome_results = run_multiseed(
        X_prot_aligned,
        prot_aligned_y,
        prot_aligned_genes,
        pfam_map,
        le,
        seeds=seeds,
        n_folds=args.n_folds,
        compute_ci=compute_ci,
        n_boot=args.n_boot,
        n_permutations=args.n_permutations,
    )

    print("\n" + "=" * 60)
    print("DECISION RULES (PREREGISTRATION_run_biorxiv.md, 2F-2H)")
    print("=" * 60)

    fs_f1 = emb_results["logreg_family_split"].get("oof_macro_f1")
    mlp_f1 = emb_results["mlp_family_split"].get("oof_macro_f1")
    gs_f1 = emb_results["logreg_gene_split"]["macro_f1_mean"]
    fs_ci = (emb_results.get("bootstrap_ci") or {}).get("macro_f1")
    paired_ci = emb_results.get("paired_ci_mlp_minus_logreg")
    independent_ci = emb_results.get("independent_ci_logreg_minus_mechanism")

    gate_2f = fs_f1 is not None and fs_f1 >= 0.70
    mechanism_point = independent_ci.get("point_b") if independent_ci is not None else None
    enzyme_mechanism_diff = (
        independent_ci.get("point_diff") if independent_ci is not None else None
    )
    gate_2g = enzyme_mechanism_diff is not None and enzyme_mechanism_diff > 0.10
    gate_2h = mlp_f1 is not None and fs_f1 is not None and abs(mlp_f1 - fs_f1) < 0.05

    verdict_2f = adjudicate_level(fs_f1, fs_ci, 0.70)
    if independent_ci is not None:
        verdict_2g = adjudicate_level(enzyme_mechanism_diff, independent_ci, 0.10)
    else:
        verdict_2g = "not adjudicated (independent difference CI unavailable)"
    verdict_2h = adjudicate_equivalence(gate_2h, paired_ci, 0.05)

    print(f"\n2F — family-split F1 >= 0.70:  {verdict_2f}  (F1={fs_f1:.3f})")
    if mechanism_point is not None:
        print(
            f"2G — enzyme >> mechanism floor:  {verdict_2g}  "
            f"(delta={enzyme_mechanism_diff:+.3f})"
        )
    else:
        print("2G — enzyme >> mechanism floor:  SKIPPED (mechanism reference unavailable)")
    if mlp_f1 is not None and fs_f1 is not None:
        print(
            f"2H — MLP approx LogReg family-split:  {verdict_2h}  "
            f"(delta_MLP-LR={mlp_f1 - fs_f1:+.3f})"
        )
    else:
        print("2H — MLP approx LogReg family-split:  SKIPPED (missing OOF)")

    output = {
        "description": (
            "Enzyme type classification (kinase/protease/oxidoreductase/non-enzyme) "
            "from ESM-2 WT mean-pooled embeddings. Positive control paralleling the "
            "mechanism arc (results 1-10). Primary metric: family-split LogReg macro-F1."
        ),
        "seeds": seeds,
        "n_folds": args.n_folds,
        "compute_ci": compute_ci,
        "n_boot": args.n_boot if compute_ci else None,
        "n_permutations": args.n_permutations,
        "n_genes": len(gene_list),
        "class_distribution": dict(Counter(y_str)),
        "classes": list(le.classes_),
        "esm2_wt_embedding": emb_results,
        "proteome_features": proteome_results,
        "gate_evaluation": {
            "2F_family_split_f1_ge_0.70": bool(gate_2f),
            "2F_verdict": verdict_2f,
            "2G_enzyme_beats_mechanism_by_0.10": bool(gate_2g),
            "2G_verdict": verdict_2g,
            "2H_mlp_approx_logreg": bool(gate_2h),
            "2H_verdict": verdict_2h,
            "fs_f1": fs_f1,
            "mlp_f1": mlp_f1,
            "gs_f1": gs_f1,
            "mechanism_reference_f1": mechanism_ref_f1,
            "mechanism_seed0_oof_f1": mechanism_point,
            "enzyme_minus_mechanism_f1": enzyme_mechanism_diff,
            "note": "fs_f1 and mlp_f1 are pooled-OOF macro F1 (matching the bootstrap CI)",
        },
    }

    write_result_json(ENZYME_CLASSIFICATION_JSON, output, seeds=seeds, indent=2)
    print(f"\nResults written to {ENZYME_CLASSIFICATION_JSON}")


if __name__ == "__main__":
    main()
