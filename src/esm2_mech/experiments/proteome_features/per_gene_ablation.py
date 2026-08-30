"""T2 per-gene scoring (V1/V2/V3) and T4 V2 feature-class ablation under
family-split CV. Re-scores per-gene so each gene gets one vote regardless
of variant count.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from esm2_mech.utils.data import build_gene_to_row
from esm2_mech.utils.splits import family_split_indices
from esm2_mech.utils.metrics import (
    aggregate_folds,
    align_proba,
    compute_metrics,
    empty_aggregate_metrics,
)
from esm2_mech.utils.classification import validate_complete_classification_splits
from esm2_mech.utils.paths import (
    RESULTS_DIR,
    VALID_VARIANTS_JSON,
    EMB_WT_MEAN,
    EMB_MUT_MEAN,
    GENE_UNIVERSE,
    PFAM_JSON,
    PROTEOME_FEATURES_ALIGNED,
    PROTEOME_FEATURE_COLUMNS_JSON,
    PROTEOME_FEATURES_TSV,
    BADONYI_FEATURES_ALIGNED,
    BADONYI_FEATURE_COLUMNS_JSON,
    BADONYI_FEATURES_TSV,
)
from esm2_mech.utils.constants import (
    MECHANISM_CLASSES,
    MIN_TRAIN_CLASSES,
    N_FOLDS,
    N_SEEDS,
)
from esm2_mech.utils.io import write_result_json
from esm2_mech.utils.seed_aggregation import (
    aggregate_result_contract,
    aggregate_seed_results,
    read_seed_point_estimate,
    seed_result_contract,
)
import functools

print = functools.partial(print, flush=True)

OUT_DIR = RESULTS_DIR

warnings.filterwarnings("ignore")

MERGED_VALID_VARIANTS = VALID_VARIANTS_JSON
MERGED_WT_MEAN = EMB_WT_MEAN
MERGED_MUT_MEAN = EMB_MUT_MEAN
PROTEOME_FEATURES = PROTEOME_FEATURES_ALIGNED
PROTEOME_COLS = PROTEOME_FEATURE_COLUMNS_JSON
# Row index MUST be GENE_UNIVERSE — gene_list.tsv is a longer, differently-ordered superset.
MERGED_GENE_LIST = GENE_UNIVERSE
PFAM_FAMILIES = PFAM_JSON

CLASSES = MECHANISM_CLASSES
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}

# Two classes in train suffice; requiring all len(CLASSES) silently drops valid folds.


def _encode(y: np.ndarray) -> np.ndarray:
    """String labels -> CLASSES indices for sklearn estimators that need numeric targets."""
    return np.array([CLASS_TO_IDX[lab] for lab in y])


def _decode(clf_classes: np.ndarray) -> np.ndarray:
    """Integer-encoded clf.classes_ -> string labels for align_proba."""
    return np.array([CLASSES[idx] for idx in clf_classes])

FEATURE_CLASSES = {
    "constraint": ["pLI", "LOEUF", "mis_z"],
    "paralogs": ["paralog_count"],
    "expression": ["tissue_specificity_tau"],
    "abundance": ["log_abundance_ppm"],
    "interactome": ["PPI_degree"],
    "dosage": ["HI_score", "TS_score"],
}


def get_drop_indices(feature_names: list[str], base_features: list[str]) -> list[int]:
    """Return column indices to drop for a given set of base feature names plus derived variants."""
    drop = []
    for i, name in enumerate(feature_names):
        for bf in base_features:
            if name == bf or name.startswith(bf + "_"):
                drop.append(i)
                break
    return drop


def load_all():
    with open(MERGED_VALID_VARIANTS) as f:
        variants = json.load(f)
    labels = np.array([v["label_3class"] for v in variants])
    genes = np.array([v["gene"] for v in variants])
    n = len(variants)
    wt = np.load(MERGED_WT_MEAN)[:n]
    mut = np.load(MERGED_MUT_MEAN)[:n]
    delta = (mut - wt).astype(np.float32)

    with open(PFAM_FAMILIES) as f:
        pfam_map = json.load(f)

    # Build gene→proteome row index
    gene_to_row = build_gene_to_row(MERGED_GENE_LIST)

    prot_matrix = np.load(PROTEOME_FEATURES).astype(np.float32)
    # NaN, not 0.0, for a variant whose gene has no proteome row: 0.0 is a
    # plausible real value for these features and would be indistinguishable
    # from a measurement. The proteome arms consume NaN natively.
    X_prot_var = np.full((n, prot_matrix.shape[1]), np.nan, dtype=np.float32)
    n_no_row = 0
    for i, g in enumerate(genes):
        row = gene_to_row.get(g)
        if row is not None:
            X_prot_var[i] = prot_matrix[row]
        else:
            n_no_row += 1
    if n_no_row:
        print(f"  {n_no_row}/{n} variants have no proteome row for their gene (NaN)")

    with open(PROTEOME_COLS) as f:
        col_meta = json.load(f)
    feature_names = col_meta["numerical_columns"]

    # Also load gene-level data (for V2 per-gene)
    gene_list_df = pd.read_csv(MERGED_GENE_LIST, sep="\t")
    # AR handling — KNOWN DIVERGENCE. These proteome experiments drop AR entirely;
    # the runbook pipeline (RUNBOOK_4) collapses AR -> LOF. Not reconciled because
    # these scripts are exploratory, not part of the runbook.
    mech_map = {"LOF": "LOF", "HI": "LOF", "GOF": "GOF", "DN": "DN", "AR": None}
    gene_list_df["mech3"] = gene_list_df["mechanism"].map(mech_map)
    gene_level = gene_list_df[gene_list_df["mech3"].notna()].copy()
    gene_level["pfam_family"] = gene_level["gene"].map(pfam_map)
    gene_level_idx = np.array([gene_to_row.get(g, -1) for g in gene_level["gene"]])
    valid_gene_mask = gene_level_idx >= 0
    gene_level = gene_level[valid_gene_mask].copy()
    gene_level_idx = gene_level_idx[valid_gene_mask]
    X_prot_gene = prot_matrix[gene_level_idx]

    return (
        variants,
        labels,
        genes,
        delta,
        X_prot_var,
        pfam_map,
        gene_level,
        X_prot_gene,
        prot_matrix,
        gene_to_row,
        feature_names,
    )


def run_per_gene_cv(
    *,
    # variant-level arrays (for V1 and V3 aggregation)
    delta: np.ndarray,
    X_prot_var: np.ndarray,
    var_labels: np.ndarray,
    var_genes: np.ndarray,
    # gene-level arrays (for V2)
    gene_level_df: pd.DataFrame,
    X_prot_gene: np.ndarray,
    pfam_map: dict,
    seed: int,
) -> dict:
    """Per-gene family-split CV for V1 (delta), V2 (proteome), and V3 (concat)."""
    y_var = np.asarray(var_labels)
    gene_pfam_var = np.array([pfam_map.get(g) for g in var_genes])
    has_fam_var = np.array([p is not None for p in gene_pfam_var])
    fam_idx = np.where(has_fam_var)[0]

    # Restrict variant arrays to those with Pfam (same as proteome_mechanism.py)
    delta_f = delta[fam_idx]
    X_prot_f = X_prot_var[fam_idx]
    X_concat_f = np.concatenate([delta_f, X_prot_f], axis=1)
    y_f = y_var[fam_idx]
    genes_f = var_genes[fam_idx]
    groups_f = gene_pfam_var[fam_idx]

    # Gene-level: restrict to genes with Pfam annotation
    gene_pfam_g = np.array([pfam_map.get(g) for g in gene_level_df["gene"].values])
    has_fam_g = np.array([p is not None for p in gene_pfam_g])
    gene_df_f = gene_level_df[has_fam_g].copy()
    X_prot_g_f = X_prot_gene[has_fam_g]
    y_g_f = np.asarray(gene_df_f["mech3"].values)
    groups_g = gene_pfam_g[has_fam_g]

    v1_folds, v2_folds, v3_folds = [], [], []
    splits = list(family_split_indices(groups_g, N_FOLDS, seed))
    contract = validate_complete_classification_splits(
        splits, requested_folds=N_FOLDS,
        eligible_rows=np.concatenate([test for _train, test in splits]),
        labels=y_g_f, classes=CLASSES, groups=groups_g, held_out_unit="family",
    )
    preflight_failures = []
    for fold_index, (train_rows, test_rows) in enumerate(splits):
        train_genes = set(gene_df_f.iloc[train_rows]["gene"].values)
        test_genes = set(gene_df_f.iloc[test_rows]["gene"].values)
        train_variant_rows = sum(gene in train_genes for gene in genes_f)
        test_variant_rows = sum(gene in test_genes for gene in genes_f)
        if train_variant_rows < 10 or test_variant_rows < 5:
            preflight_failures.append(
                {
                    "fold": fold_index,
                    "reason": "insufficient_variant_rows",
                    "train_rows": int(train_variant_rows),
                    "test_rows": int(test_variant_rows),
                }
            )
    if contract["status"] != "valid" or preflight_failures:
        reason = "split_validation_failed"
        unavailable = empty_aggregate_metrics(CLASSES, N_FOLDS, reason)
        unavailable.update(
            {
                "status": "unscorable",
                "split_validation": contract,
                "preflight_failures": preflight_failures,
            }
        )
        return {
            "V1_per_gene": dict(unavailable),
            "V2_per_gene": dict(unavailable),
            "V3_per_gene": dict(unavailable),
        }

    for fold_i, (tr_g, te_g) in enumerate(splits):
        train_genes_set = set(gene_df_f.iloc[tr_g]["gene"].values)
        test_genes_set = set(gene_df_f.iloc[te_g]["gene"].values)

        # --- V2: proteome features, gene-level ---
        X_tr_g, y_tr_g = X_prot_g_f[tr_g], y_g_f[tr_g]
        X_te_g, y_te_g = X_prot_g_f[te_g], y_g_f[te_g]

        # NaN-native: proteome block has real missing cells, nothing is imputed.
        lr2 = HistGradientBoostingClassifier(
            max_iter=200, class_weight="balanced", random_state=seed
        )
        lr2.fit(X_tr_g, y_tr_g)
        # String labels; align by class NAME so absent classes become zero columns.
        pr2_al = align_proba(
            lr2.predict_proba(X_te_g),
            lr2.classes_,
            CLASSES,
            allow_missing_classes=False,
        )
        pd2 = np.array([CLASSES[idx] for idx in pr2_al.argmax(axis=1)])
        # Train: all variants from train genes
        tr_var_mask = np.array([g in train_genes_set for g in genes_f])
        # Test: variants from test genes — aggregate per gene
        te_var_mask = np.array([g in test_genes_set for g in genes_f])

        v2_folds.append(compute_metrics(y_te_g, pd2, pr2_al, CLASSES))

        # MLPClassifier cannot take string targets; fit on CLASSES indices, decode for align_proba.
        X_tr_d, y_tr_d = delta_f[tr_var_mask], _encode(y_f[tr_var_mask])
        X_tr_c, y_tr_c = X_concat_f[tr_var_mask], _encode(y_f[tr_var_mask])

        def oversample(X, y, seed):
            counts = np.bincount(y, minlength=len(CLASSES))
            mc = counts.max()
            rng = np.random.RandomState(seed)
            idx = []
            for c in range(len(CLASSES)):
                ci = np.where(y == c)[0]
                if len(ci) == 0:
                    raise ValueError(
                        f"training fold is missing declared class {CLASSES[c]!r}"
                    )
                rep = mc // len(ci)
                rem = mc % len(ci)
                idx.append(np.tile(ci, rep))
                if rem > 0:
                    idx.append(rng.choice(ci, rem, replace=False))
            idx = np.concatenate(idx)
            rng.shuffle(idx)
            return X[idx], y[idx]

        sc1 = StandardScaler().fit(X_tr_d)
        X_tr_d_s = sc1.transform(X_tr_d)
        X_bal_d, y_bal_d = oversample(X_tr_d_s, y_tr_d, seed)
        mlp1 = MLPClassifier(
            (256, 64),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed,
        )
        mlp1.fit(X_bal_d, y_bal_d)

        # NaN-native: proteome columns would make the 1321-dim row unusable to an MLP.
        mlp3 = HistGradientBoostingClassifier(
            max_iter=200, class_weight="balanced", random_state=seed
        )
        mlp3.fit(X_tr_c, y_tr_c)

        test_genes_list = sorted(test_genes_set)
        y_gene_true, y_gene_pred1, y_gene_pred3 = [], [], []
        pr_gene1 = []
        pr_gene3 = []

        for gene in test_genes_list:
            gene_mask = genes_f[te_var_mask] == gene
            if gene_mask.sum() == 0:
                raise RuntimeError(
                    f"validated test gene {gene!r} has no variant rows"
                )
            # True label: gene-level (take first, all same within gene)
            gene_var_idx = np.where(te_var_mask)[0][gene_mask]
            y_gene_true.append(y_f[gene_var_idx[0]])

            # V1
            X_g_d = sc1.transform(delta_f[gene_var_idx])
            pr1_g = align_proba(
                mlp1.predict_proba(X_g_d),
                _decode(mlp1.classes_),
                CLASSES,
                allow_missing_classes=False,
            ).mean(0)
            pr_gene1.append(pr1_g)
            y_gene_pred1.append(CLASSES[int(pr1_g.argmax())])

            # V3 — unscaled, matching the NaN-native model fitted above
            pr3_g = align_proba(
                mlp3.predict_proba(X_concat_f[gene_var_idx]),
                _decode(mlp3.classes_),
                CLASSES,
                allow_missing_classes=False,
            ).mean(0)
            pr_gene3.append(pr3_g)
            y_gene_pred3.append(CLASSES[int(pr3_g.argmax())])

        y_gene_true = np.array(y_gene_true)
        y_gene_pred1 = np.array(y_gene_pred1)
        y_gene_pred3 = np.array(y_gene_pred3)
        pr_gene1 = np.array(pr_gene1)
        pr_gene3 = np.array(pr_gene3)

        v1_folds.append(compute_metrics(y_gene_true, y_gene_pred1, pr_gene1, CLASSES))
        v3_folds.append(compute_metrics(y_gene_true, y_gene_pred3, pr_gene3, CLASSES))

        print(
            f"  [T2 seed={seed}] Fold {fold_i+1}: "
            f"V1={v1_folds[-1]['macro_f1']:.3f}  "
            f"V2={v2_folds[-1]['macro_f1']:.3f}  "
            f"V3={v3_folds[-1]['macro_f1']:.3f}"
        )

    results = {
        "V1_per_gene": aggregate_folds(v1_folds, CLASSES, N_FOLDS),
        "V2_per_gene": aggregate_folds(v2_folds, CLASSES, N_FOLDS),
        "V3_per_gene": aggregate_folds(v3_folds, CLASSES, N_FOLDS),
    }
    for result in results.values():
        result["status"] = "success"
    return results


def run_v2_ablation(
    X_prot_gene: np.ndarray,
    y_gene: np.ndarray,
    groups_gene: np.ndarray,
    feature_names: list[str],
    seed: int,
) -> dict:
    """Run V2 with and without each feature class; return delta-F1."""

    def run_ablation_cv(X, y, groups):
        """NaN-native family-split CV over a proteome feature subset.

        Imputing would leak test-fold statistics; trees consume NaN directly.
        """
        folds = []
        splits = list(family_split_indices(groups, N_FOLDS, seed))
        contract = validate_complete_classification_splits(
            splits, requested_folds=N_FOLDS,
            eligible_rows=np.concatenate([test for _train, test in splits]),
            labels=y, classes=CLASSES, groups=groups, held_out_unit="family",
        )
        if contract["status"] != "valid":
            result = empty_aggregate_metrics(
                CLASSES, N_FOLDS, "split_validation_failed"
            )
            result.update({"status": "unscorable", "split_validation": contract})
            return result
        for tr, te in splits:
            X_tr, y_tr = X[tr], y[tr]
            X_te, y_te = X[te], y[te]
            clf = HistGradientBoostingClassifier(
                max_iter=200, class_weight="balanced", random_state=seed
            )
            clf.fit(X_tr, y_tr)
            pr_al = align_proba(
                clf.predict_proba(X_te),
                clf.classes_,
                CLASSES,
                allow_missing_classes=False,
            )
            pd_ = np.array([CLASSES[idx] for idx in pr_al.argmax(axis=1)])
            folds.append(compute_metrics(y_te, pd_, pr_al, CLASSES))
        result = aggregate_folds(folds, CLASSES, N_FOLDS)
        result["status"] = "success"
        result["split_validation"] = contract
        return result

    def _difference(full_value, ablated_value):
        """A delta is defined only when both arms scored. A missing metric stays
        None: neither 0.0 nor NaN, both of which read as a real 'no change'."""
        if full_value is None or ablated_value is None:
            return None
        return float(full_value - ablated_value)

    def _fmt(value):
        return f"{value:+.4f}" if value is not None else "  N/A  "

    full_result = run_ablation_cv(X_prot_gene, y_gene, groups_gene)
    full_f1 = full_result["macro_f1_mean"]
    if full_f1 is None:
        print(f"  [T4 seed={seed}] V2 FULL: unscorable")
    else:
        print(f"  [T4 seed={seed}] V2 FULL: macro_f1={full_f1:.4f}")

    ablation_results = {"FULL": full_result}

    for cls_name, base_feats in FEATURE_CLASSES.items():
        drop_idx = get_drop_indices(feature_names, base_feats)
        keep_idx = [i for i in range(X_prot_gene.shape[1]) if i not in drop_idx]
        X_abl = X_prot_gene[:, keep_idx]
        res = run_ablation_cv(X_abl, y_gene, groups_gene)
        delta_f1 = _difference(full_f1, res["macro_f1_mean"])
        delta_dn = _difference(
            full_result.get("auroc_DN_mean"), res.get("auroc_DN_mean")
        )
        ablation_results[cls_name] = {
            **res,
            "n_features_dropped": len(drop_idx),
            "n_features_kept": len(keep_idx),
            "delta_f1": delta_f1,
            "delta_auroc_DN": delta_dn,
        }
        abl_f1 = res["macro_f1_mean"]
        abl_f1_text = f"{abl_f1:.4f}" if abl_f1 is not None else "  N/A  "
        print(
            f"  [T4 seed={seed}] minus {cls_name:12s}: "
            f"f1={abl_f1_text}  ΔF1={_fmt(delta_f1)}  "
            f"ΔDN_AUROC={_fmt(delta_dn)}"
        )

    return ablation_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-t2", action="store_true")
    parser.add_argument("--skip-t4", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    (
        variants,
        var_labels,
        var_genes,
        delta,
        X_prot_var,
        pfam_map,
        gene_level_df,
        X_prot_gene,
        prot_matrix,
        gene_to_row,
        feature_names,
    ) = load_all()

    # Gene-level arrays for T4
    gene_pfam_g = np.array([pfam_map.get(g) for g in gene_level_df["gene"].values])
    has_fam_g = np.array([p is not None for p in gene_pfam_g])
    gene_df_f = gene_level_df[has_fam_g].copy()
    X_prot_g_f = X_prot_gene[has_fam_g]
    y_g_f = np.asarray(gene_df_f["mech3"].values)
    groups_g = gene_pfam_g[has_fam_g]

    seeds = [args.seed] if args.seed is not None else list(range(N_SEEDS))

    if not args.skip_t2:
        print("\n=== T2: Per-gene scoring ===")
        t2_seed_results = []
        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            res = run_per_gene_cv(
                delta=delta,
                X_prot_var=X_prot_var,
                var_labels=var_labels,
                var_genes=var_genes,
                gene_level_df=gene_level_df,
                X_prot_gene=X_prot_gene,
                pfam_map=pfam_map,
                seed=seed,
            )
            res.update(seed_result_contract(seed))
            t2_seed_results.append(res)
            out_path = OUT_DIR / f"per_gene_seed{seed}.json"
            write_result_json(out_path, res, seeds=[seed], indent=2)
            print(f"  Saved {out_path.name}")
            for v in ["V1_per_gene", "V2_per_gene", "V3_per_gene"]:
                f1 = res[v].get("macro_f1_mean")
                print(
                    f"  {v}: macro_f1={f1:.4f}" if f1 is not None
                    else f"  {v}: macro_f1=N/A"
                )

        t2_summary = {**aggregate_result_contract()}
        for v in ["V1_per_gene", "V2_per_gene", "V3_per_gene"]:
            metric_names = ["macro_f1_mean", *[f"auroc_{cls}_mean" for cls in CLASSES]]
            t2_summary[v] = {}
            for metric_name in metric_names:
                aggregate = aggregate_seed_results(
                    seeds,
                    t2_seed_results,
                    lambda result, arm=v, metric=metric_name: result[arm].get(metric),
                    status=lambda result, arm=v: result[arm]["status"],
                )
                t2_summary[v][f"{metric_name.removesuffix('_mean')}_seed_aggregate"] = (
                    aggregate.to_dict()
                )

        t2_sum_path = OUT_DIR / "per_gene_summary.json"
        write_result_json(t2_sum_path, t2_summary, seeds=seeds, indent=2)

        print("\n=== T2 SUMMARY (per-gene, mean ± std across seeds) ===")
        for v in ["V1_per_gene", "V2_per_gene", "V3_per_gene"]:
            f1 = read_seed_point_estimate(t2_summary[v]["macro_f1_seed_aggregate"])
            dn = read_seed_point_estimate(t2_summary[v]["auroc_DN_seed_aggregate"])
            if not f1.available or not dn.available:
                print(f"  {v}: N/A")
                continue
            spread = "N/A" if f1.spread is None else f"{f1.spread:.4f}"
            print(f"  {v}: macro_f1={f1.value:.4f}±{spread}  DN_AUROC={dn.value:.3f}")

    if not args.skip_t4:
        print("\n=== T4: V2 Feature-class ablation ===")
        t4_seed_results = []
        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            res = run_v2_ablation(X_prot_g_f, y_g_f, groups_g, feature_names, seed)
            res.update(seed_result_contract(seed))
            t4_seed_results.append(res)
            out_path = OUT_DIR / f"v2_ablation_seed{seed}.json"
            write_result_json(out_path, res, seeds=[seed], indent=2)
            print(f"  Saved {out_path.name}")

        # Aggregate T4
        t4_summary = {**aggregate_result_contract(), "FULL": {}}
        full_aggregate = aggregate_seed_results(
            seeds,
            t4_seed_results,
            lambda result: result["FULL"].get("macro_f1_mean"),
            status=lambda result: result["FULL"]["status"],
        )
        t4_summary["FULL"]["macro_f1_seed_aggregate"] = full_aggregate.to_dict()

        for cls_name in FEATURE_CLASSES:
            t4_summary[cls_name] = {}
            for metric_name in ("macro_f1_mean", "delta_f1", "delta_auroc_DN"):
                aggregate = aggregate_seed_results(
                    seeds,
                    t4_seed_results,
                    lambda result, arm=cls_name, metric=metric_name: result[arm].get(metric),
                    status=lambda result, arm=cls_name: result[arm]["status"],
                )
                stem = metric_name.removesuffix("_mean")
                t4_summary[cls_name][f"{stem}_seed_aggregate"] = aggregate.to_dict()

        t4_sum_path = OUT_DIR / "v2_ablation_summary.json"
        write_result_json(t4_sum_path, t4_summary, seeds=seeds, indent=2)

        print("\n=== T4 SUMMARY (V2 ablation, mean ± std across seeds) ===")
        full = read_seed_point_estimate(t4_summary["FULL"]["macro_f1_seed_aggregate"])
        if not full.available:
            print("  V2 FULL:  N/A")
        else:
            spread = "N/A" if full.spread is None else f"{full.spread:.4f}"
            print(f"  V2 FULL:  {full.value:.4f} ± {spread}")
        print(f"  {'Class':<14}  {'Abl F1':>8}  {'ΔF1':>8}  {'ΔDN AUROC':>10}")
        for cls_name in FEATURE_CLASSES:
            m = read_seed_point_estimate(t4_summary[cls_name]["macro_f1_seed_aggregate"])
            delta_f1 = read_seed_point_estimate(t4_summary[cls_name]["delta_f1_seed_aggregate"])
            delta_dn = read_seed_point_estimate(t4_summary[cls_name]["delta_auroc_DN_seed_aggregate"])
            m_s = f"{m.value:.4f}" if m.available else "  N/A  "
            delta_spread = (
                "N/A" if delta_f1.spread is None else f"{delta_f1.spread:.4f}"
            )
            df_s = (
                f"{delta_f1.value:+.4f}±{delta_spread}"
                if delta_f1.available else "  N/A  "
            )
            ddn_s = f"{delta_dn.value:+.4f}" if delta_dn.available else "  N/A  "
            print(f"  minus {cls_name:<12}  {m_s}  {df_s}  {ddn_s}")


if __name__ == "__main__":
    main()
