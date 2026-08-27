"""MLP probe on raw WT and mutant embeddings (not delta).

Tests whether a nonlinear probe recovers GOF/DN/LOF from mut_only, wt_only,
or concat_wt_mut embeddings under gene-split and family-split CV.
"""

import argparse
import functools
import json
import os

import numpy as np

from esm2_mech.experiments.mechanism.loaders import _label_3class
from esm2_mech.utils.bootstrap import family_or_gene_clusters
from esm2_mech.utils.data import validate_embedding_variant_identity
from esm2_mech.utils.splits import gene_split_cv, family_split_cv
from esm2_mech.utils.probes import run_mlp_probe_cv
from esm2_mech.utils.classification import validate_complete_classification_splits
from esm2_mech.utils.constants import MECHANISM_CLASSES
from esm2_mech.utils.io import write_result_json
from esm2_mech.utils.paths import (
    EMB_MUT_MEAN,
    EMB_VALID_VARIANTS_JSON,
    EMB_WT_MEAN,
    PFAM_JSON,
    VALID_VARIANTS_JSON,
)

print = functools.partial(print, flush=True)


def load_variants_and_labels(variants_file: str | None = None):
    """Load mechanism variants row-aligned to the embeddings."""
    path = variants_file or VALID_VARIANTS_JSON
    with open(path) as f:
        variants = json.load(f)
    labels = np.array([_label_3class(v) for v in variants])
    genes = np.array([v["gene"] for v in variants])
    print(f"Loaded {len(variants):,} variants, {len(set(genes))} genes")
    return variants, labels, genes


def load_wt_mut_mean_embeddings():
    """Load WT and mutant mean-pooled embeddings as separate arrays (not delta)."""
    wt = np.load(EMB_WT_MEAN)
    mut = np.load(EMB_MUT_MEAN)
    print(f"  WT  embeddings: {wt.shape}")
    print(f"  mut embeddings: {mut.shape}")
    assert wt.shape == mut.shape, f"WT and mut embedding shapes don't match: {wt.shape} vs {mut.shape}"
    return wt.astype(np.float32), mut.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variants_file",
        default=None,
        help="Pre-filtered variants JSON (use for merged dataset)",
    )
    p.add_argument(
        "--pfam_map",
        default=None,
        help="Path to pfam_families.json (defaults to DATA_DIR/pfam_families.json)",
    )
    p.add_argument(
        "--family_split",
        action="store_true",
        help="Also run family-split CV (requires pfam_map)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument(
        "--features",
        nargs="+",
        default=["mut_only", "wt_only", "concat_wt_mut"],
        choices=["mut_only", "wt_only", "concat_wt_mut"],
        help="Which feature representations to test",
    )
    args = p.parse_args()

    np.random.seed(args.seed)

    print("=== Loading variants and labels ===")
    valid_variants, labels, genes = load_variants_and_labels(args.variants_file)
    validate_embedding_variant_identity(valid_variants, EMB_VALID_VARIANTS_JSON)

    print("\n=== Loading WT and mut mean-pooled embeddings ===")
    wt, mut = load_wt_mut_mean_embeddings()

    assert len(wt) == len(labels), (
        f"Embedding count {len(wt)} != variant count {len(labels)}. "
        f"Ensure --variants_file matches the embeddings (use valid_variants.json "
        f"from the model embeddings directory)."
    )

    # Build feature variants
    feature_specs = {}
    if "mut_only" in args.features:
        feature_specs["mut_only"] = mut
    if "wt_only" in args.features:
        feature_specs["wt_only"] = wt
    if "concat_wt_mut" in args.features:
        feature_specs["concat_wt_mut"] = np.concatenate([wt, mut], axis=1)

    print(f"\nFeatures to test: {list(feature_specs.keys())}")
    print(f"Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print(f"Unique genes: {len(set(genes))}")

    # CV splits — identical to mlp.py
    gene_splits = gene_split_cv(genes, seed=args.seed)
    gene_contract = validate_complete_classification_splits(
        gene_splits,
        requested_folds=5,
        eligible_rows=np.concatenate([test for _train, test in gene_splits]),
        labels=labels,
        classes=MECHANISM_CLASSES,
        groups=genes,
        held_out_unit="gene",
    )
    family_splits = None
    family_contract = None
    if args.family_split:
        pfam_path = args.pfam_map or PFAM_JSON
        with open(pfam_path) as f:
            pfam_map = json.load(f)
        family_splits = family_split_cv(genes, pfam_map, seed=args.seed)
        family_groups = family_or_gene_clusters(
            genes, pfam_map, is_family_split=True
        )
        family_contract = validate_complete_classification_splits(
            family_splits,
            requested_folds=5,
            eligible_rows=np.concatenate([test for _train, test in family_splits]),
            labels=labels,
            classes=MECHANISM_CLASSES,
            groups=family_groups,
            held_out_unit="family",
        )
        n_fams = len(set(pfam_map.get(g) for g in set(genes) if pfam_map.get(g)))
        print(
            f"\nFamily-split: {len(family_splits)} folds, {n_fams} unique annotated families"
        )

    # Run probe on each feature × each CV scheme
    results = {
        "config": {
            "seed": args.seed,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "n_variants": len(labels),
            "n_genes": len(set(genes)),
            "class_distribution": {
                k: int(v) for k, v in zip(*np.unique(labels, return_counts=True))
            },
            "features_tested": list(feature_specs.keys()),
        }
    }

    for feat_name, X in feature_specs.items():
        print(
            f"\n{'='*60}\n=== MLP gene-split: {feat_name} (dim={X.shape[1]}) ===\n{'='*60}"
        )
        gs = run_mlp_probe_cv(
            X,
            labels,
            gene_splits,
            MECHANISM_CLASSES,
            gene_contract,
            validation_groups=genes,
            seed=args.seed,
            genes=genes,
            max_epochs=args.max_epochs,
            patience=args.patience,
            label=f"{feat_name}_gene",
        )
        results[f"mlp_{feat_name}_gene"] = gs
        if gs["status"] == "success":
            ranking = "  ".join(
                f"{class_name} AUROC = "
                + (
                    "NA"
                    if gs[f"auroc_{class_name}_mean"] is None
                    else f"{gs[f'auroc_{class_name}_mean']:.3f}"
                )
                for class_name in MECHANISM_CLASSES
            )
            print(f"  macro_f1 = {gs['macro_f1_mean']:.3f}  {ranking}")
        else:
            print(f"  {gs['status']}")

        if family_splits:
            print(f"\n=== MLP family-split: {feat_name} ===")
            fs = run_mlp_probe_cv(
                X,
                labels,
                family_splits,
                MECHANISM_CLASSES,
                family_contract,
                validation_groups=family_or_gene_clusters(
                    genes, pfam_map, is_family_split=True
                ),
                seed=args.seed,
                genes=genes,
                max_epochs=args.max_epochs,
                patience=args.patience,
                label=f"{feat_name}_family",
            )
            results[f"mlp_{feat_name}_family"] = fs
            if fs["status"] == "success":
                ranking = "  ".join(
                    f"{class_name} AUROC = "
                    + (
                        "NA"
                        if fs[f"auroc_{class_name}_mean"] is None
                        else f"{fs[f'auroc_{class_name}_mean']:.3f}"
                    )
                    for class_name in MECHANISM_CLASSES
                )
                print(f"  macro_f1 = {fs['macro_f1_mean']:.3f}  {ranking}")
            else:
                print(f"  {fs['status']}")
            if gs["macro_f1_mean"] is not None and fs["macro_f1_mean"] is not None:
                delta_macro = gs["macro_f1_mean"] - fs["macro_f1_mean"]
                print(
                    f"  Δ(gene − family) macro-F1 = {delta_macro:+.3f}  "
                    f"← positive ⇒ homology leakage"
                )
            else:
                print("  Δ(gene − family) macro-F1 = Unscorable")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_result_json(args.out, results, seeds=[args.seed], indent=2)
    print(f"\nResults written to {args.out}")

    # Headline interpretation against the published delta numbers
    print(
        f"\n{'='*60}\nHEADLINE — compare to published MLP delta_mean numbers\n{'='*60}"
    )
    print(f"  Published MLP delta_mean family-split: 0.364 (Geras) / 0.352 (merged)")
    print(
        f"  Published MLP delta_mean GOF AUROC family-split: 0.627 (Geras) / 0.635 (merged)"
    )
    print()
    for feat in feature_specs:
        gs = results.get(f"mlp_{feat}_gene", {})
        fs = results.get(f"mlp_{feat}_family", {})
        gof_fs = fs.get("auroc_GOF_mean", float("nan"))
        f1_fs = fs.get("macro_f1_mean", float("nan"))
        if not np.isnan(gof_fs):
            verdict = (
                "STRONG NEW SIGNAL"
                if gof_fs > 0.80
                else (
                    "modest lift over delta"
                    if gof_fs > 0.70
                    else (
                        "matches delta — no new info"
                        if gof_fs > 0.55
                        else "weaker than delta"
                    )
                )
            )
            print(
                f"  {feat:18s}  family-split GOF AUROC = {gof_fs:.3f}  F1 = {f1_fs:.3f}  ⇒ {verdict}"
            )


if __name__ == "__main__":
    main()
