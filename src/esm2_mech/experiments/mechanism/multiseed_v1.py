"""Multi-seed replication of v1 headline numbers.

Runs mechanism MLP and pathogenicity probes across seeds 0..N on Gerasimavicius
and merged datasets under gene-split and family-split CV.
"""

import argparse
import json
import os
import sys
import numpy as np
from collections import defaultdict
import functools

print = functools.partial(print, flush=True)
from esm2_mech.utils.constants import (
    DELTA_MEAN_FEATURE, DELTA_POS_FEATURE, MECHANISM_CLASSES, N_SEEDS, SPLIT_FAMILY, SPLIT_GENE, nonlinear_key,
)
from esm2_mech.utils.splits import gene_split_cv, family_split_cv
from esm2_mech.utils.probes import run_logreg_binary_cv, run_mlp_binary_cv, run_mlp_probe_cv
from esm2_mech.utils.bootstrap import family_or_gene_clusters
from esm2_mech.utils.classification import validate_complete_classification_splits
from esm2_mech.utils.io import atomic_write_json, load_json_or_discard, write_result_json
from esm2_mech.utils.paths import (
    PFAM_JSON,
    CLINVAR_PATHOGENICITY_VARIANTS_JSON,
    PATHOGENICITY_VALID_VARIANTS_JSON,
    PATH_EMB_WT_MEAN,
    PATH_EMB_MUT_MEAN,
    V1_MULTISEED_DIR,
    V1_MULTISEED_SEED0_DIR,
)

MLP_DELTA_MEAN_GENE = nonlinear_key("mlp", DELTA_MEAN_FEATURE, SPLIT_GENE)
MLP_DELTA_MEAN_FAMILY = nonlinear_key("mlp", DELTA_MEAN_FEATURE, SPLIT_FAMILY)

OUT_DIR = str(V1_MULTISEED_DIR)
SEED0_DIR = str(V1_MULTISEED_SEED0_DIR)




from esm2_mech.experiments.mechanism.loaders import load_mechanism_variants, load_merged


def load_pathogenicity(pfam_map):
    variants = load_json_or_discard(PATHOGENICITY_VALID_VARIANTS_JSON)
    if variants is None:
        with open(CLINVAR_PATHOGENICITY_VARIANTS_JSON) as f:
            variants = json.load(f)
        emb_n = np.load(PATH_EMB_WT_MEAN, mmap_mode="r").shape[0]
        if len(variants) > emb_n:
            print(
                f"  Warning: {len(variants)} pathogenicity variants but {emb_n} embeddings; "
                f"truncating to {emb_n} (23 dropped on RunPod)"
            )
            variants = variants[:emb_n]
        atomic_write_json(PATHOGENICITY_VALID_VARIANTS_JSON, variants)
        print(f"  Saved {PATHOGENICITY_VALID_VARIANTS_JSON}")

    genes = np.array([v["gene"] for v in variants])
    y = np.array([1 if v["label"] == "pathogenic" else 0 for v in variants])
    wt = np.load(PATH_EMB_WT_MEAN)
    mut = np.load(PATH_EMB_MUT_MEAN)
    delta = mut - wt
    assert len(delta) == len(
        y
    ), f"Path embedding/variant count mismatch: {len(delta)} vs {len(y)}"
    print(
        f"  Pathogenicity: {len(variants)} variants, {len(set(genes))} genes, "
        f"{int(y.sum())} pathogenic / {int((1-y).sum())} benign"
    )
    return delta, y, genes


# ── Main ──────────────────────────────────────────────────────────────────────


def run_seed(seed, pfam_map, out_dir):
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")

    geras_out = os.path.join(out_dir, f"mechanism_geras_seed{seed}.json")
    geras_results = load_json_or_discard(geras_out)
    if geras_results is not None:
        print(f"  [skip] {geras_out} already exists")
    else:
        print("\n--- Gerasimavicius mechanism ---")
        dm, dp, labels, genes = load_mechanism_variants(pfam_map)
        gs = gene_split_cv(genes, seed=seed)
        fs = family_split_cv(genes, pfam_map, seed=seed)
        family_validation_groups = family_or_gene_clusters(
            genes, pfam_map, is_family_split=True
        )
        gene_contract = validate_complete_classification_splits(
            gs, requested_folds=5,
            eligible_rows=np.concatenate([test for _train, test in gs]),
            labels=labels, classes=MECHANISM_CLASSES, groups=genes,
            held_out_unit="gene",
        )
        family_contract = validate_complete_classification_splits(
            fs, requested_folds=5,
            eligible_rows=np.concatenate([test for _train, test in fs]),
            labels=labels, classes=MECHANISM_CLASSES,
            groups=family_validation_groups, held_out_unit="family",
        )
        geras_results = {}
        for feat_name, X in [(DELTA_MEAN_FEATURE, dm), (DELTA_POS_FEATURE, dp)]:
            print(f"  MLP gene-split {feat_name}")
            geras_results[nonlinear_key("mlp", feat_name, SPLIT_GENE)] = run_mlp_probe_cv(
                X, labels, gs, MECHANISM_CLASSES, gene_contract, validation_groups=genes,
                seed=seed, genes=genes, label=f"{feat_name}_gene"
            )
            print(f"  MLP family-split {feat_name}")
            geras_results[nonlinear_key("mlp", feat_name, SPLIT_FAMILY)] = run_mlp_probe_cv(
                X, labels, fs, MECHANISM_CLASSES, family_contract, validation_groups=family_validation_groups,
                seed=seed, genes=genes, label=f"{feat_name}_family"
            )
        write_result_json(geras_out, geras_results, seeds=[seed], indent=2)
        print(f"  -> {geras_out}")

    merged_out = os.path.join(out_dir, f"mechanism_merged_seed{seed}.json")
    merged_results = load_json_or_discard(merged_out)
    if merged_results is not None:
        print(f"  [skip] {merged_out} already exists")
    else:
        print("\n--- Merged dataset mechanism ---")
        dm, labels, genes = load_merged(pfam_map)
        gs = gene_split_cv(genes, seed=seed)
        fs = family_split_cv(genes, pfam_map, seed=seed)
        family_validation_groups = family_or_gene_clusters(
            genes, pfam_map, is_family_split=True
        )
        gene_contract = validate_complete_classification_splits(
            gs, requested_folds=5,
            eligible_rows=np.concatenate([test for _train, test in gs]),
            labels=labels, classes=MECHANISM_CLASSES, groups=genes,
            held_out_unit="gene",
        )
        family_contract = validate_complete_classification_splits(
            fs, requested_folds=5,
            eligible_rows=np.concatenate([test for _train, test in fs]),
            labels=labels, classes=MECHANISM_CLASSES,
            groups=family_validation_groups, held_out_unit="family",
        )
        merged_results = {}
        print(f"  MLP gene-split delta_mean")
        merged_results[MLP_DELTA_MEAN_GENE] = run_mlp_probe_cv(
            dm, labels, gs, MECHANISM_CLASSES, gene_contract, validation_groups=genes,
            seed=seed, genes=genes, label="delta_mean_gene"
        )
        print(f"  MLP family-split delta_mean")
        merged_results[MLP_DELTA_MEAN_FAMILY] = run_mlp_probe_cv(
            dm, labels, fs, MECHANISM_CLASSES, family_contract, validation_groups=family_validation_groups,
            seed=seed, genes=genes, label="delta_mean_family"
        )
        write_result_json(merged_out, merged_results, seeds=[seed], indent=2)
        print(f"  -> {merged_out}")

    path_out = os.path.join(out_dir, f"pathogenicity_seed{seed}.json")
    path_results = load_json_or_discard(path_out)
    if path_results is not None:
        print(f"  [skip] {path_out} already exists")
    else:
        print("\n--- Pathogenicity control ---")
        delta, y, genes = load_pathogenicity(pfam_map)
        gs = gene_split_cv(genes, seed=seed)
        fs = family_split_cv(genes, pfam_map, seed=seed)
        family_validation_groups = family_or_gene_clusters(
            genes, pfam_map, is_family_split=True
        )
        binary_classes = [0, 1]
        gene_contract = validate_complete_classification_splits(
            gs, requested_folds=5,
            eligible_rows=np.concatenate([test for _train, test in gs]),
            labels=y, classes=binary_classes, groups=genes, held_out_unit="gene",
        )
        family_contract = validate_complete_classification_splits(
            fs, requested_folds=5,
            eligible_rows=np.concatenate([test for _train, test in fs]),
            labels=y, classes=binary_classes,
            groups=family_validation_groups, held_out_unit="family",
        )
        path_results = {}
        print(f"  logreg gene-split")
        path_results["logreg_gene"] = run_logreg_binary_cv(
            delta, y, gs, binary_classes, gene_contract, seed=seed
        )
        print(f"  logreg family-split")
        path_results["logreg_family"] = run_logreg_binary_cv(
            delta, y, fs, binary_classes, family_contract, seed=seed
        )
        print(f"  MLP gene-split")
        path_results["mlp_gene"] = run_mlp_binary_cv(
            delta, y, gs, binary_classes, gene_contract,
            validation_groups=genes, seed=seed
        )
        print(f"  MLP family-split")
        path_results["mlp_family"] = run_mlp_binary_cv(
            delta, y, fs, binary_classes, family_contract,
            validation_groups=family_validation_groups, seed=seed
        )
        write_result_json(path_out, path_results, seeds=[seed], indent=2)
        print(f"  -> {path_out}")

    return geras_results, merged_results, path_results


def summarise(all_seeds, out_dir):
    """Aggregate across seeds."""

    def agg(values):
        if any(value is None or np.isnan(value) for value in values):
            return None, None
        return float(np.mean(values)), float(np.std(values))

    # headline keys
    metrics = {
        "mechanism_geras_mlp_delta_mean_family_macro_f1": [],
        "mechanism_geras_mlp_delta_mean_family_auroc_GOF": [],
        "mechanism_geras_mlp_delta_mean_family_auroc_DN": [],
        "mechanism_geras_mlp_delta_mean_family_auroc_LOF": [],
        "mechanism_geras_mlp_delta_mean_gene_macro_f1": [],
        "mechanism_merged_mlp_delta_mean_family_macro_f1": [],
        "mechanism_merged_mlp_delta_mean_family_auroc_GOF": [],
        "mechanism_merged_mlp_delta_mean_family_auroc_DN": [],
        "mechanism_merged_mlp_delta_mean_family_auroc_LOF": [],
        "mechanism_merged_mlp_delta_mean_gene_macro_f1": [],
        "pathogenicity_mlp_gene_auroc": [],
        "pathogenicity_mlp_family_auroc": [],
        "pathogenicity_logreg_gene_auroc": [],
        "pathogenicity_logreg_family_auroc": [],
    }

    per_seed = {}
    for seed, geras, merged, path in all_seeds:
        per_seed[seed] = {}

        def g(d, *keys):
            v = d
            for k in keys:
                if not isinstance(v, dict) or k not in v:
                    return None
                v = v[k]
            return v

        vals = {
            "mechanism_geras_mlp_delta_mean_family_macro_f1": g(
                geras, MLP_DELTA_MEAN_FAMILY, "macro_f1_mean"
            ),
            "mechanism_geras_mlp_delta_mean_family_auroc_GOF": g(
                geras, MLP_DELTA_MEAN_FAMILY, "auroc_GOF_mean"
            ),
            "mechanism_geras_mlp_delta_mean_family_auroc_DN": g(
                geras, MLP_DELTA_MEAN_FAMILY, "auroc_DN_mean"
            ),
            "mechanism_geras_mlp_delta_mean_family_auroc_LOF": g(
                geras, MLP_DELTA_MEAN_FAMILY, "auroc_LOF_mean"
            ),
            "mechanism_geras_mlp_delta_mean_gene_macro_f1": g(
                geras, MLP_DELTA_MEAN_GENE, "macro_f1_mean"
            ),
            "mechanism_merged_mlp_delta_mean_family_macro_f1": g(
                merged, MLP_DELTA_MEAN_FAMILY, "macro_f1_mean"
            ),
            "mechanism_merged_mlp_delta_mean_family_auroc_GOF": g(
                merged, MLP_DELTA_MEAN_FAMILY, "auroc_GOF_mean"
            ),
            "mechanism_merged_mlp_delta_mean_family_auroc_DN": g(
                merged, MLP_DELTA_MEAN_FAMILY, "auroc_DN_mean"
            ),
            "mechanism_merged_mlp_delta_mean_family_auroc_LOF": g(
                merged, MLP_DELTA_MEAN_FAMILY, "auroc_LOF_mean"
            ),
            "mechanism_merged_mlp_delta_mean_gene_macro_f1": g(
                merged, MLP_DELTA_MEAN_GENE, "macro_f1_mean"
            ),
            "pathogenicity_mlp_gene_auroc": g(path, "mlp_gene", "auroc_mean"),
            "pathogenicity_mlp_family_auroc": g(path, "mlp_family", "auroc_mean"),
            "pathogenicity_logreg_gene_auroc": g(path, "logreg_gene", "auroc_mean"),
            "pathogenicity_logreg_family_auroc": g(path, "logreg_family", "auroc_mean"),
        }
        per_seed[seed] = vals
        for k, v in vals.items():
            metrics[k].append(v)

    summary = {"per_seed": per_seed, "aggregate": {}}
    print("\n" + "=" * 60)
    print("V1 MULTISEED SUMMARY")
    print("=" * 60)
    for k, vals in metrics.items():
        mean, std = agg(vals)
        summary["aggregate"][k] = {
            "mean": mean,
            "std": std,
            "n": len([v for v in vals if v is not None]),
        }
        if mean is not None:
            print(
                f"  {k:<60s}  {mean:.3f} ± {std:.3f}  (n={summary['aggregate'][k]['n']})"
            )

    gf_vals = metrics["mechanism_geras_mlp_delta_mean_gene_macro_f1"]
    fs_vals = metrics["mechanism_geras_mlp_delta_mean_family_macro_f1"]
    chance = 0.333
    leakage_fracs = []
    for gf, fs in zip(gf_vals, fs_vals):
        if gf is None or fs is None or np.isnan(gf) or np.isnan(fs):
            continue
        if (gf - chance) > 0:
            leakage_fracs.append((gf - fs) / (gf - chance))
    if leakage_fracs:
        print(
            f"\n  Leakage fraction (Gerasimavicius):  {np.mean(leakage_fracs):.1%} ± {np.std(leakage_fracs):.1%}"
        )

    path_delta_vals = []
    for s in summary["per_seed"].values():
        gv = s.get("pathogenicity_mlp_gene_auroc")
        fv = s.get("pathogenicity_mlp_family_auroc")
        if gv is None or fv is None or np.isnan(gv) or np.isnan(fv):
            continue
        path_delta_vals.append(gv - fv)
    if path_delta_vals:
        print(
            f"  Pathogenicity gene→family Δ:         {np.mean(path_delta_vals):.3f} ± {np.std(path_delta_vals):.3f}"
        )

    out = os.path.join(out_dir, "summary.json")
    write_result_json(out, summary, seeds=list(per_seed.keys()), indent=2)
    print(f"\nSummary written to {out}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=int,
        default=N_SEEDS,
        help="number of seeds to run; runs 0..seeds-1 (>=1). Seed 0 is loaded from "
             "existing results unless --include_seed0 forces a recompute.",
    )
    parser.add_argument(
        "--include_seed0",
        action="store_true",
        help="Also re-run seed 0 (normally skipped — already exists)",
    )
    parser.add_argument(
        "--summarise_only",
        action="store_true",
        help="Skip computation, just re-aggregate existing JSONs",
    )
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(PFAM_JSON) as f:
        pfam_map = json.load(f)
    print(f"Pfam map loaded: {len(pfam_map)} genes")

    seeds = list(range(args.seeds))
    all_seeds = []

    with open(os.path.join(SEED0_DIR, "mlp_results_seed0.json")) as _f:
        seed0_geras = json.load(_f)
    with open(os.path.join(SEED0_DIR, "mlp_merged_results_seed0.json")) as _f:
        seed0_merged = json.load(_f)
    with open(os.path.join(SEED0_DIR, "pathogenicity_control.json")) as _f:
        seed0_path = json.load(_f)

    def reformat_path_seed0(d):
        bf = d.get("by_feature", {})
        dm = bf.get("delta_mean", {})
        return {
            "logreg_gene": {
                "auroc_mean": dm.get("gene_split_logreg", {}).get("auroc_mean")
            },
            "logreg_family": {
                "auroc_mean": dm.get("family_split_logreg", {}).get("auroc_mean")
            },
            "mlp_gene": {"auroc_mean": dm.get("gene_split_mlp", {}).get("auroc_mean")},
            "mlp_family": {
                "auroc_mean": dm.get("family_split_mlp", {}).get("auroc_mean")
            },
        }

    if not args.include_seed0:
        all_seeds.append((0, seed0_geras, seed0_merged, reformat_path_seed0(seed0_path)))

    if not args.summarise_only:
        for seed in seeds:
            if seed == 0 and not args.include_seed0:
                continue
            g, m, p = run_seed(seed, pfam_map, OUT_DIR)
            all_seeds.append((seed, g, m, p))
    else:
        for seed in seeds:
            if seed == 0 and not args.include_seed0:
                continue
            gf = os.path.join(OUT_DIR, f"mechanism_geras_seed{seed}.json")
            mf = os.path.join(OUT_DIR, f"mechanism_merged_seed{seed}.json")
            pf = os.path.join(OUT_DIR, f"pathogenicity_seed{seed}.json")
            if os.path.exists(gf) and os.path.exists(mf) and os.path.exists(pf):
                with open(gf) as _f1, open(mf) as _f2, open(pf) as _f3:
                    all_seeds.append(
                        (seed, json.load(_f1), json.load(_f2), json.load(_f3))
                    )
            else:
                print(f"Warning: seed {seed} results not found, skipping")

    summarise(all_seeds, OUT_DIR)


if __name__ == "__main__":
    main()
