"""Multi-seed replication of v1 headline numbers.

Runs mechanism MLP and pathogenicity probes across seeds 0..N on Gerasimavicius
and merged datasets under gene-split and family-split CV.
"""

import argparse
import json
import os
import numpy as np
import functools

print = functools.partial(print, flush=True)
from esm2_mech.utils.constants import (
    DELTA_MEAN_FEATURE, DELTA_POS_FEATURE, MECHANISM_CLASSES, N_SEEDS, SPLIT_FAMILY, SPLIT_GENE, nonlinear_key,
)
from esm2_mech.utils.metrics import majority_baseline_f1
from esm2_mech.experiments.mechanism.leakage_fraction import MIN_ABOVE_CHANCE
from esm2_mech.utils.splits import gene_split_cv, family_split_cv
from esm2_mech.utils.probes import run_logreg_binary_cv, run_mlp_binary_cv, run_mlp_probe_cv
from esm2_mech.utils.bootstrap import family_or_gene_clusters
from esm2_mech.utils.classification import validate_complete_classification_splits
from esm2_mech.utils.io import atomic_write_json, load_json_or_discard, write_result_json
from esm2_mech.utils.seed_aggregation import (
    aggregate_paired_seed_difference,
    aggregate_seed_values,
    make_seed_record,
    read_seed_inference,
    seed_result_contract,
)
from esm2_mech.experiments.mechanism.seed_results import aggregate_result_contract
from esm2_mech.utils.paths import (
    PFAM_JSON,
    CLINVAR_PATHOGENICITY_VARIANTS_JSON,
    PATHOGENICITY_VALID_VARIANTS_JSON,
    PATH_EMB_WT_MEAN,
    PATH_EMB_MUT_MEAN,
    V1_MULTISEED_DIR,
)

MLP_DELTA_MEAN_GENE = nonlinear_key("mlp", DELTA_MEAN_FEATURE, SPLIT_GENE)
MLP_DELTA_MEAN_FAMILY = nonlinear_key("mlp", DELTA_MEAN_FEATURE, SPLIT_FAMILY)

OUT_DIR = str(V1_MULTISEED_DIR)




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
    geras_results = {**seed_result_contract(seed)}
    # Chance floor for the leakage fraction: the majority baseline under this
    # seed's own gene-split folds, not a fixed 1/len(classes). The classes are
    # not balanced, so the two are different numbers.
    try:
        geras_results["gene_split_majority_macro_f1"] = float(
            np.mean([
                majority_baseline_f1(labels[train], labels[test], MECHANISM_CLASSES)[0]
                for train, test in gs
            ])
        )
    except ValueError as exc:
        geras_results["gene_split_majority_macro_f1"] = None
        print(f"  Majority baseline unavailable: {exc}")
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
    merged_results = {**seed_result_contract(seed)}
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
    path_results = {**seed_result_contract(seed)}
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


def summarise(all_seeds, requested_seeds, out_dir):
    """Aggregate across seeds."""

    # headline keys
    metrics = {
        "mechanism_geras_mlp_delta_mean_family_macro_f1": [],
        "mechanism_geras_mlp_delta_mean_family_auroc_GOF": [],
        "mechanism_geras_mlp_delta_mean_family_auroc_DN": [],
        "mechanism_geras_mlp_delta_mean_family_auroc_LOF": [],
        "mechanism_geras_mlp_delta_mean_gene_macro_f1": [],
        "mechanism_geras_gene_majority_macro_f1": [],
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
            "mechanism_geras_gene_majority_macro_f1": g(
                geras, "gene_split_majority_macro_f1"
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

    summary = {
        **aggregate_result_contract(),
        "requested_seeds": list(requested_seeds),
        "per_seed": per_seed,
        "aggregate": {},
    }
    print("\n" + "=" * 60)
    print("V1 MULTISEED SUMMARY")
    print("=" * 60)
    for k, vals in metrics.items():
        records = [
            make_seed_record(seed, per_seed.get(seed, {}).get(k))
            for seed in per_seed
        ]
        aggregate = aggregate_seed_values(requested_seeds, records)
        summary["aggregate"][k] = aggregate.to_dict()
        metric = read_seed_inference(aggregate)
        if metric.available:
            print(
                f"  {k:<60s}  {metric.value:.3f} ± {metric.spread:.3f}"
            )

    gf_vals = metrics["mechanism_geras_mlp_delta_mean_gene_macro_f1"]
    fs_vals = metrics["mechanism_geras_mlp_delta_mean_family_macro_f1"]
    chance_vals = metrics["mechanism_geras_gene_majority_macro_f1"]
    leakage_records = []
    for seed, gf, fs, chance in zip(per_seed, gf_vals, fs_vals, chance_vals):
        value = None
        if all(
            v is not None and np.isfinite(v) for v in (gf, fs, chance)
        ):
            denominator = gf - chance
            value = (
                (gf - fs) / denominator
                if denominator > MIN_ABOVE_CHANCE
                else None
            )
        leakage_records.append(make_seed_record(seed, value))
    leakage = aggregate_seed_values(requested_seeds, leakage_records)
    leakage_metric = read_seed_inference(leakage)
    summary["aggregate"]["mechanism_geras_leakage_fraction"] = leakage.to_dict()
    if leakage_metric.available:
        print(
            f"\n  Leakage fraction (Gerasimavicius):  "
            f"{leakage_metric.value:.1%} ± {leakage_metric.spread:.1%}"
        )

    gene_records = [
        make_seed_record(seed, values.get("pathogenicity_mlp_gene_auroc"))
        for seed, values in per_seed.items()
    ]
    family_records = [
        make_seed_record(seed, values.get("pathogenicity_mlp_family_auroc"))
        for seed, values in per_seed.items()
    ]
    path_delta = aggregate_paired_seed_difference(
        requested_seeds, gene_records, family_records
    )
    path_delta_metric = read_seed_inference(path_delta)
    summary["aggregate"]["pathogenicity_gene_minus_family_auroc"] = path_delta.to_dict()
    if path_delta_metric.available:
        print(
            f"  Pathogenicity gene→family Δ:         "
            f"{path_delta_metric.value:.3f} ± {path_delta_metric.spread:.3f}"
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
        help="number of seeds to run; runs 0..seeds-1 (>=1)",
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

    for seed in seeds:
        geras, merged, pathogenicity = run_seed(seed, pfam_map, OUT_DIR)
        all_seeds.append((seed, geras, merged, pathogenicity))

    summarise(all_seeds, seeds, OUT_DIR)


if __name__ == "__main__":
    main()
