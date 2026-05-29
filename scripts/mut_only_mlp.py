"""
MLP probe on raw WT and mutant embeddings (not delta) — testing whether a
nonlinear probe can recover GOF / DN / LOF mechanism from mutant or WT
mean-pooled embeddings, where the delta probe gave only the small ~0.36
family-split floor.

Question: linear mut_only ≈ linear WT-only on family-split (both ~0.39 F1,
GOF AUROC ~0.73–0.80). Does an MLP find more in the mutant embedding than
the linear probe does — i.e., does the mutation-induced shift, extracted
nonlinearly from the full mutant representation, carry mechanism signal
beyond family identity?

Three features tested:
  1. mut_only_mean   — mutant sequence embedding, mean-pooled (1280-D)
  2. wt_only_mean    — WT sequence embedding, mean-pooled (1280-D)
  3. concat_wt_mut   — [WT; mut] concatenation (2560-D)

Uses the EXACT same PyTorch MLP and class-weighting as experiment_mlp.py
(imports `run_mlp_probe`), so numbers are directly comparable to the
delta_mean MLP family-split F1 = 0.364 / 0.352 published in result_7.

Usage:
    # Gerasimavicius (default)
    python mut_only_mlp.py \\
        --data_dir ../results/20260524_baseline_run/run_0/data \\
        --emb_dir ../data/embeddings \\
        --family_split \\
        --out ../results/20260524_baseline_run/run_0/mut_only_mlp_seed0.json

    # Merged dataset
    python mut_only_mlp.py \\
        --data_dir ../data \\
        --emb_dir ../data/embeddings \\
        --emb_prefix merged_ \\
        --variants_file ../data/merged_valid_variants.json \\
        --family_split \\
        --out ../results/20260524_baseline_run/run_0/mut_only_mlp_merged_seed0.json
"""

import argparse
import json
import os
import sys

import numpy as np

# Reuse helpers from experiment_mlp.py to guarantee identical methodology
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import functools
print = functools.partial(print, flush=True)

from experiment_mlp import (
    load_variants_and_labels,
    gene_split_cv,
    make_family_splits,
    run_mlp_probe,
    ESM2_MODEL_650M,
)


def load_wt_mut_mean_embeddings(emb_dir, model_name=ESM2_MODEL_650M, prefix=""):
    """Load WT and mutant mean-pooled embeddings as separate arrays (not delta)."""
    wt_path = os.path.join(emb_dir, f"{prefix}embeddings_wt_{model_name}.npy")
    mut_path = os.path.join(emb_dir, f"{prefix}embeddings_mut_{model_name}.npy")
    if not (os.path.exists(wt_path) and os.path.exists(mut_path)):
        # Try the merged naming convention if prefix didn't match
        alt_wt = os.path.join(emb_dir, f"{prefix}embeddings_wt_mean.npy")
        alt_mut = os.path.join(emb_dir, f"{prefix}embeddings_mut_mean.npy")
        if os.path.exists(alt_wt) and os.path.exists(alt_mut):
            wt_path, mut_path = alt_wt, alt_mut
        else:
            raise FileNotFoundError(
                f"Could not find WT/mut embeddings at:\n  {wt_path}\n  {mut_path}\n"
                f"(also tried {alt_wt}, {alt_mut})"
            )
    wt = np.load(wt_path)
    mut = np.load(mut_path)
    print(f"  WT  embeddings: {wt.shape}  ({wt_path})")
    print(f"  mut embeddings: {mut.shape}  ({mut_path})")
    assert wt.shape == mut.shape, f"WT and mut embedding shapes don't match: {wt.shape} vs {mut.shape}"
    return wt.astype(np.float32), mut.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True,
                   help="Directory containing variants JSON (and sequences.json, pfam_families.json if Gerasimavicius)")
    p.add_argument("--emb_dir", required=True,
                   help="Directory containing the .npy embedding files")
    p.add_argument("--model", default=ESM2_MODEL_650M)
    p.add_argument("--emb_prefix", default="",
                   help="Prefix on embedding filenames, e.g. 'merged_' for merged dataset")
    p.add_argument("--variants_file", default=None,
                   help="Pre-filtered variants JSON (use for merged dataset)")
    p.add_argument("--pfam_map", default=None,
                   help="Path to pfam_families.json (defaults to data_dir/pfam_families.json)")
    p.add_argument("--family_split", action="store_true",
                   help="Also run family-split CV (requires pfam_map)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument("--features", nargs="+",
                   default=["mut_only", "wt_only", "concat_wt_mut"],
                   choices=["mut_only", "wt_only", "concat_wt_mut"],
                   help="Which feature representations to test")
    args = p.parse_args()

    np.random.seed(args.seed)

    print("=== Loading variants and labels ===")
    valid_variants, labels, genes = load_variants_and_labels(args.data_dir, args.variants_file)

    print("\n=== Loading WT and mut mean-pooled embeddings ===")
    wt, mut = load_wt_mut_mean_embeddings(args.emb_dir, args.model, args.emb_prefix)

    assert len(wt) == len(labels), (
        f"Embedding count {len(wt)} != variant count {len(labels)}. "
        f"Ensure --variants_file matches the embeddings (use merged_valid_variants.json "
        f"with merged_ prefix)."
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

    # CV splits — identical to experiment_mlp.py
    gene_splits = gene_split_cv(genes, seed=args.seed)
    family_splits = None
    if args.family_split:
        pfam_path = args.pfam_map or os.path.join(args.data_dir, "pfam_families.json")
        with open(pfam_path) as f:
            pfam_map = json.load(f)
        family_splits = make_family_splits(genes, pfam_map, seed=args.seed)
        n_fams = len(set(pfam_map.get(g) for g in set(genes) if pfam_map.get(g)))
        print(f"\nFamily-split: {len(family_splits)} folds, {n_fams} unique annotated families")

    # Run probe on each feature × each CV scheme
    results = {
        "config": {
            "seed": args.seed,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "n_variants": len(labels),
            "n_genes": len(set(genes)),
            "class_distribution": {k: int(v) for k, v in
                                    zip(*np.unique(labels, return_counts=True))},
            "features_tested": list(feature_specs.keys()),
            "emb_prefix": args.emb_prefix,
            "data_dir": args.data_dir,
        }
    }

    for feat_name, X in feature_specs.items():
        print(f"\n{'='*60}\n=== MLP gene-split: {feat_name} (dim={X.shape[1]}) ===\n{'='*60}")
        gs = run_mlp_probe(X, labels, genes, seed=args.seed, splits=gene_splits,
                            max_epochs=args.max_epochs, patience=args.patience)
        results[f"mlp_{feat_name}_gene"] = gs
        print(f"  macro_f1 = {gs.get('macro_f1_mean', float('nan')):.3f}  "
              f"GOF AUROC = {gs.get('auroc_GOF_mean', float('nan')):.3f}  "
              f"DN AUROC = {gs.get('auroc_DN_mean', float('nan')):.3f}  "
              f"LOF AUROC = {gs.get('auroc_LOF_mean', float('nan')):.3f}")

        if family_splits:
            print(f"\n=== MLP family-split: {feat_name} ===")
            fs = run_mlp_probe(X, labels, genes, seed=args.seed, splits=family_splits,
                                max_epochs=args.max_epochs, patience=args.patience)
            results[f"mlp_{feat_name}_family"] = fs
            print(f"  macro_f1 = {fs.get('macro_f1_mean', float('nan')):.3f}  "
                  f"GOF AUROC = {fs.get('auroc_GOF_mean', float('nan')):.3f}  "
                  f"DN AUROC = {fs.get('auroc_DN_mean', float('nan')):.3f}  "
                  f"LOF AUROC = {fs.get('auroc_LOF_mean', float('nan')):.3f}")
            delta_macro = (gs.get("macro_f1_mean", float("nan"))
                           - fs.get("macro_f1_mean", float("nan")))
            print(f"  Δ(gene − family) macro-F1 = {delta_macro:+.3f}  "
                  f"← positive ⇒ homology leakage")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.out}")

    # Headline interpretation against the published delta numbers
    print(f"\n{'='*60}\nHEADLINE — compare to published MLP delta_mean numbers\n{'='*60}")
    print(f"  Published MLP delta_mean family-split: 0.364 (Geras) / 0.352 (merged)")
    print(f"  Published MLP delta_mean GOF AUROC family-split: 0.627 (Geras) / 0.635 (merged)")
    print()
    for feat in feature_specs:
        gs = results.get(f"mlp_{feat}_gene", {})
        fs = results.get(f"mlp_{feat}_family", {})
        gof_fs = fs.get("auroc_GOF_mean", float("nan"))
        f1_fs = fs.get("macro_f1_mean", float("nan"))
        if not np.isnan(gof_fs):
            verdict = (
                "STRONG NEW SIGNAL" if gof_fs > 0.80 else
                "modest lift over delta" if gof_fs > 0.70 else
                "matches delta — no new info" if gof_fs > 0.55 else
                "weaker than delta"
            )
            print(f"  {feat:18s}  family-split GOF AUROC = {gof_fs:.3f}  F1 = {f1_fs:.3f}  ⇒ {verdict}")


if __name__ == "__main__":
    main()
