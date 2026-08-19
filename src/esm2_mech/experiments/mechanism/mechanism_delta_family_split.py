"""Run all baselines and delta probe under gene-split and family-split CV."""

import argparse
import json
import os
from collections import Counter

import numpy as np

from esm2_mech.utils.data import (
    embedding_variant_fingerprint,
    load_pfam_map,
    validate_embedding_variant_identity,
)
from esm2_mech.utils.paths import (
    EMB_WT_MEAN, EMB_MUT_MEAN, EMB_WT_POS, EMB_MUT_POS,
    EMB_VALID_VARIANTS_JSON,
    SEQUENCES_JSON, VARIANTS_JSON, RESULTS_DIR, PFAM_JSON,
)

OUT_DIR = RESULTS_DIR
from esm2_mech.utils.sequences import window_sequence, apply_missense, build_wt_mut_onehot
from esm2_mech.utils.constants import (
    seed_result_filename,
    mechanism_oof_cache_filename,
    MECHANISM_CLASSES,
    BOOTSTRAP_N_RESAMPLES,
)
from esm2_mech.utils.splits import (
    annotated_gene_mask, gene_split_cv, family_split_cv, fold_index_array,
)
from esm2_mech.utils.embed import unpack_run_data
from esm2_mech.utils.io import atomic_write_json, write_result_json
from esm2_mech.utils.probes import run_logreg_pca_cv
from esm2_mech.utils.bootstrap import (
    adjudicate_diff,
    bootstrap_mechanism_metrics,
    family_or_gene_clusters,
    label_permutation_pvalue,
    oof_permutation_pvalue,
    paired_oof_diff,
)

import functools

PCA_COMPONENTS = 256  # applied to embedding features (dim > PCA_COMPONENTS)

print = functools.partial(print, flush=True)


PERMUTATION_FEATURES = ("delta_mean", "wt_only_mean")

# wt_only_mean: refit permutation (macro-F1). delta_mean: OOF permutation (AUROC).
REFIT_PERMUTATION_FEATURES = ("wt_only_mean",)
OOF_PERMUTATION_FEATURES = ("delta_mean",)


def run(
    data: dict,
    out_dir: str,
    seed: int = 0,
    n_folds: int = 5,
    compute_ci: bool = True,
    n_boot: int = BOOTSTRAP_N_RESAMPLES,
    n_permutations: int = 0,
) -> dict:
    data = unpack_run_data(data)
    valid_variants = data["valid_variants"]
    emb_wt_mean = data["emb_wt_mean"]
    emb_mut_mean = data["emb_mut_mean"]
    emb_wt_pos = data["emb_wt_pos"]
    emb_mut_pos = data["emb_mut_pos"]
    labels_3class = data["labels_3class"]
    genes_arr = data["genes_arr"]
    foldx_ddg = data["foldx_ddg"]
    aa_wt_list = data["aa_wt_list"]
    aa_mut_list = data["aa_mut_list"]
    alphamissense_scores = data["alphamissense_scores"]
    deltas_mean = data["deltas_mean"]
    deltas_pos = data["deltas_pos"]
    print(f"Embedding shape: {emb_wt_mean.shape}")
    print(f"Valid variants: {len(valid_variants)}")
    print(f"Class distribution: {dict(Counter(labels_3class))}")
    print(f"Unique genes: {len(set(genes_arr))}")

    foldx_mask = ~np.isnan(foldx_ddg)
    am_mask = ~np.isnan(alphamissense_scores)

    print(f"\n=== PCA ({PCA_COMPONENTS} components) fitted per CV fold ===")

    features: dict = {
        "wt_only_mean": emb_wt_mean,
        "mut_only_mean": emb_mut_mean,
        "delta_mean": deltas_mean,
        "delta_per_residue": deltas_pos,
        "wt_concat_mut": np.concatenate([emb_wt_mean, emb_mut_mean], axis=1),
        "onehot_aa": build_wt_mut_onehot(aa_wt_list, aa_mut_list),
    }

    if foldx_mask.sum() >= 20:
        features["foldx_ddg"] = (foldx_ddg[foldx_mask].reshape(-1, 1),
                                  labels_3class[foldx_mask], genes_arr[foldx_mask])
    else:
        print(f"  [skip foldx_ddg: only {foldx_mask.sum()} observed values]")

    if am_mask.sum() >= 20:
        features["alphamissense"] = (alphamissense_scores[am_mask].reshape(-1, 1),
                                     labels_3class[am_mask], genes_arr[am_mask])
    else:
        print(f"  [skip alphamissense: only {am_mask.sum()} observed values]")

    print("\n=== Building CV splits ===")
    if not PFAM_JSON.exists():
        raise FileNotFoundError(f"{PFAM_JSON} not found — run fetch_data/fetch_annotations --step pfam first")
    pfam_map = load_pfam_map(PFAM_JSON)
    gene_splits = gene_split_cv(genes_arr, n_folds=n_folds, seed=seed)
    family_splits = family_split_cv(
        genes_arr, pfam_map, n_folds=n_folds, seed=seed
    )

    print(f"Gene-split folds:   {len(gene_splits)}")
    print(f"Family-split folds: {len(family_splits)}")
    # Only families with genes in the variant set.
    n_families = len({pfam_map.get(g) for g in genes_arr} - {None})
    print(f"Unique Pfam families: {n_families}")

    results = {
        "n_variants": len(valid_variants),
        "embedding_variant_fingerprint": embedding_variant_fingerprint(valid_variants),
        "n_genes": int(len(set(genes_arr))),
        "n_families": n_families,
        "class_distribution": dict(Counter(labels_3class)),
        "gene_split": {},
        "family_split": {},
    }

    oof_cache: dict = {}

    for name, entry in features.items():
        if isinstance(entry, tuple):
            X, feat_labels, feat_genes = entry
            feat_gene_splits = gene_split_cv(feat_genes, n_folds=n_folds, seed=seed)
            feat_family_splits = family_split_cv(feat_genes, pfam_map, n_folds=n_folds, seed=seed)
        else:
            X, feat_labels, feat_genes = entry, labels_3class, genes_arr
            feat_gene_splits, feat_family_splits = gene_splits, family_splits

        if X.shape[1] > 0 and np.allclose(X.std(), 0):
            print(f"  [skip {name}: zero variance]")
            continue
        print(f"\n--- {name} (dim={X.shape[1]}, n={len(feat_labels)}) ---")

        gs, gs_oof = run_logreg_pca_cv(
            X, feat_labels, feat_gene_splits, seed=seed, genes=feat_genes,
            label=f"{name} gene", n_pca=PCA_COMPONENTS, return_oof=True,
        )
        if compute_ci and gs_oof is not None:
            gs["ci"] = bootstrap_mechanism_metrics(
                gs_oof["y_true"], gs_oof["proba"], gs_oof["genes"], gs_oof["folds"],
                n_resamples=n_boot, seed=seed,
            )
        results["gene_split"][name] = gs
        gs_f1 = gs.get("macro_f1_mean", float("nan"))
        print(
            f"  gene-split   macro-F1 = {gs_f1:.3f} "
            f"± {gs.get('macro_f1_std', float('nan')):.3f}  "
            f"(GOF AUROC {gs.get('auroc_GOF_mean', float('nan')):.3f}, "
            f"DN {gs.get('auroc_DN_mean', float('nan')):.3f}, "
            f"LOF {gs.get('auroc_LOF_mean', float('nan')):.3f})"
        )

        if feat_family_splits:
            fs, fs_oof = run_logreg_pca_cv(
                X, feat_labels, feat_family_splits, seed=seed, genes=feat_genes,
                label=f"{name} family", n_pca=PCA_COMPONENTS, return_oof=True,
            )
            if compute_ci and fs_oof is not None:
                fs_clusters = family_or_gene_clusters(
                    fs_oof["genes"], pfam_map, is_family_split=True
                )
                fs["ci"] = bootstrap_mechanism_metrics(
                    fs_oof["y_true"], fs_oof["proba"], fs_clusters, fs_oof["folds"],
                    n_resamples=n_boot, seed=seed,
                )
            if n_permutations > 0 and name in REFIT_PERMUTATION_FEATURES:
                # feat_family_splits only ever spans rows whose gene has a Pfam
                # annotation (family_split_cv excludes the rest from both sides of
                # every fold), so the fold index it defines is not a partition of
                # every row — building fold_index_array over the full row count
                # raises. Restrict the permutation to the same annotated subset the
                # family split actually scores, remapping the splits' row indices
                # into that subset's own index space.
                ann_mask = annotated_gene_mask(feat_genes, pfam_map)
                n_excluded = int((~ann_mask).sum())
                row_map = np.where(ann_mask)[0]
                inv_map = np.full(len(feat_genes), -1, dtype=int)
                inv_map[row_map] = np.arange(len(row_map))

                X_ann = X[row_map]
                labels_ann = feat_labels[row_map]
                genes_ann = feat_genes[row_map]
                family_splits_ann = [
                    (inv_map[tr], inv_map[te]) for tr, te in feat_family_splits
                ]

                def _family_macro_f1(perm_labels, _X=X_ann, _splits=family_splits_ann, _genes=genes_ann):
                    agg_perm, _ = run_logreg_pca_cv(
                        _X, perm_labels, _splits, seed=seed, genes=_genes,
                        n_pca=PCA_COMPONENTS, return_oof=True,
                    )
                    return agg_perm.get("macro_f1_mean")

                fs["permutation"] = label_permutation_pvalue(
                    _family_macro_f1, labels_ann, statistic="macro_f1",
                    folds=fold_index_array(family_splits_ann, len(labels_ann)),
                    groups=genes_ann,
                    clusters=family_or_gene_clusters(
                        genes_ann, pfam_map, is_family_split=True
                    ),
                    n_permutations=n_permutations, seed=seed,
                )
                fs["permutation"]["n_excluded_unannotated"] = n_excluded
            elif n_permutations > 0 and name in OOF_PERMUTATION_FEATURES and fs_oof is not None:
                fs["permutation"] = oof_permutation_pvalue(
                    fs_oof["y_true"], fs_oof["proba"], fs_oof["folds"],
                    groups=fs_oof["genes"],
                    clusters=family_or_gene_clusters(
                        fs_oof["genes"], pfam_map, is_family_split=True
                    ),
                    classes=MECHANISM_CLASSES, n_permutations=n_permutations, seed=seed,
                )
            if "permutation" in fs:
                perm = fs["permutation"]
                p_value_text = (
                    f"unresolved at resolution {perm['p_value_resolution']}"
                    if perm.get("resolution_limited")
                    else str(perm.get("p_value"))
                )
                excluded_str = (
                    f", excluded {perm['n_excluded_unannotated']} unannotated-gene rows"
                    if "n_excluded_unannotated" in perm else ""
                )
                immovable_str = (
                    f", {perm['n_clusters_immovable']} immovable families"
                    if perm.get("n_clusters_immovable") is not None else ""
                )
                print(
                    f"  family-split permutation p = {p_value_text} "
                    f"({perm.get('statistic')} via {perm.get('null_type')}, "
                    f"unit {perm.get('permutation_unit')}, "
                    f"observed {perm.get('observed')}, null mean {perm.get('null_mean')}"
                    f"{immovable_str}{excluded_str})"
                )
            # Cached for every feature and every seed, not just the permutation
            # features at seed 0: the leakage fraction's headline reads each arm's
            # macro-F1 from this cache (restricted to the rows both arms scored) for
            # every feature it reports, and an interval (or headline) built from a
            # subset of seeds or features is a different quantity from the number
            # it is printed beside. Tuple-entry features (foldx_ddg, alphamissense)
            # are excluded because their row_ids index a feature-local observed
            # subset, not the row space every other feature's row_ids share.
            if (
                compute_ci
                and not isinstance(entry, tuple) and gs_oof is not None and fs_oof is not None
            ):
                def _cache_arm(oof):
                    pred = [MECHANISM_CLASSES[col] for col in oof["proba"].argmax(axis=1)]
                    return {
                        "row_ids": oof["row_ids"].tolist(),
                        "y_true": oof["y_true"].tolist(),
                        "pred": pred,
                        "genes": oof["genes"].tolist(),
                        "folds": oof["folds"].tolist(),
                    }

                oof_cache[name] = {
                    "gene_split": _cache_arm(gs_oof),
                    "family_split": _cache_arm(fs_oof),
                }

            results["family_split"][name] = fs
            fs_f1 = fs.get("macro_f1_mean", float("nan"))
            delta_macro = gs_f1 - fs_f1
            print(
                f"  family-split macro-F1 = {fs_f1:.3f} "
                f"± {fs.get('macro_f1_std', float('nan')):.3f}  "
                f"(GOF AUROC {fs.get('auroc_GOF_mean', float('nan')):.3f}, "
                f"DN {fs.get('auroc_DN_mean', float('nan')):.3f}, "
                f"LOF {fs.get('auroc_LOF_mean', float('nan')):.3f})"
            )
            print(
                f"  Δ(gene − family) macro-F1 = {delta_macro:+.3f}  "
                f"← positive ⇒ homology leakage"
            )

            if (
                compute_ci and not isinstance(entry, tuple)
                and gs_oof is not None and fs_oof is not None
            ):
                gap = paired_oof_diff(
                    gs_oof, fs_oof, pfam_map,
                    f"{name}: gene-split − family-split",
                    classes=list(MECHANISM_CLASSES),
                    cross_partition=True,
                    n_resamples=n_boot,
                    seed=seed,
                )
                if gap is not None:
                    fs["split_gap_paired"] = gap
                    if gap.get("ci_low") is None:
                        print(f"  split-gap CI suppressed ({gap['n_clusters']} families)")
                    else:
                        sensitivity = gap.get("gene_resampled_sensitivity") or {}
                        print(
                            f"  split gap = {gap['point_diff']:+.4f}  "
                            f"[{gap['ci_low']:+.4f}, {gap['ci_high']:+.4f}]  "
                            f"({gap['n_clusters']} families) — "
                            f"{adjudicate_diff(gap['point_diff'] > 0, gap, 0.0)}"
                        )
                        if sensitivity.get("ci_low") is not None:
                            print(
                                f"    gene-resampled sensitivity (not the primary "
                                f"interval): [{sensitivity['ci_low']:+.4f}, "
                                f"{sensitivity['ci_high']:+.4f}]"
                            )

    out_path = os.path.join(out_dir, seed_result_filename(seed))
    write_result_json(out_path, results, seeds=[seed], indent=2)
    print(f"\nResults written to {out_path}")

    if oof_cache:
        oof_cache_path = os.path.join(out_dir, mechanism_oof_cache_filename(seed))
        atomic_write_json(oof_cache_path, oof_cache)
        print(f"OOF cache written to {oof_cache_path}")

    return results


def _load_data_inline() -> dict:
    """Load all data needed by run() for standalone execution."""
    from esm2_mech.utils.data import load_variants as _load_variants
    from esm2_mech.experiments.mechanism.mechanism_delta_probe import _load_alphamissense_scores

    if not VARIANTS_JSON.exists():
        raise FileNotFoundError(
            f"{VARIANTS_JSON} not found — run fetch_data/fetch_variants.py --step merge first"
        )
    variants = _load_variants(VARIANTS_JSON)

    if not SEQUENCES_JSON.exists():
        raise FileNotFoundError(f"{SEQUENCES_JSON} not found — run fetch_data/fetch_sequences first")
    with open(SEQUENCES_JSON) as f:
        seq_cache = json.load(f)

    valid_variants = []
    for v in variants:
        uid = v["uniprot_id"]
        if uid not in seq_cache:
            continue
        wt_win, new_pos, _ = window_sequence(seq_cache[uid], v["aa_pos"])
        mut_win = apply_missense(wt_win, new_pos, v["aa_wt"], v["aa_mut"])
        if mut_win is None:
            continue
        valid_variants.append(v)

    print(f"Valid variant pairs: {len(valid_variants)}")
    validate_embedding_variant_identity(valid_variants, EMB_VALID_VARIANTS_JSON)

    for path in [EMB_WT_MEAN, EMB_MUT_MEAN, EMB_WT_POS, EMB_MUT_POS]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing cached embedding: {path}")

    emb_wt_mean = np.load(EMB_WT_MEAN)
    emb_mut_mean = np.load(EMB_MUT_MEAN)
    emb_wt_pos = np.load(EMB_WT_POS)
    emb_mut_pos = np.load(EMB_MUT_POS)

    labels_3class = np.array([v["label_3class"] for v in valid_variants])
    labels_4class = np.array([v["mechanism"] for v in valid_variants])
    genes_arr = np.array([v["gene"] for v in valid_variants])
    foldx_ddg = np.array(
        [v["foldx_ddg"] if v["foldx_ddg"] is not None else np.nan for v in valid_variants]
    )
    aa_wt_list = [v["aa_wt"] for v in valid_variants]
    aa_mut_list = [v["aa_mut"] for v in valid_variants]

    alphamissense_scores = _load_alphamissense_scores(valid_variants)

    return {
        "valid_variants": valid_variants,
        "emb_wt_mean": emb_wt_mean,
        "emb_mut_mean": emb_mut_mean,
        "emb_wt_pos": emb_wt_pos,
        "emb_mut_pos": emb_mut_pos,
        "labels_3class": labels_3class,
        "labels_4class": labels_4class,
        "genes_arr": genes_arr,
        "foldx_ddg": foldx_ddg,
        "aa_wt_list": aa_wt_list,
        "aa_mut_list": aa_mut_list,
        "alphamissense_scores": alphamissense_scores,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--no_ci", action="store_true", help="skip cluster-bootstrap CIs")
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    parser.add_argument(
        "--n_permutations", type=int, default=0,
        help="label-permutation reps for headline features (0 = skip; slow, refits per rep)",
    )
    args = parser.parse_args()
    data = _load_data_inline()
    run(
        data=data, out_dir=str(OUT_DIR), seed=args.seed, n_folds=args.n_folds,
        compute_ci=not args.no_ci, n_boot=args.n_boot, n_permutations=args.n_permutations,
    )


if __name__ == "__main__":
    main()
