"""Exploratory magnitude-vs-direction decomposition of ESM-2 deltas.

Splits each delta into magnitude and direction, re-runs pathogenicity/mechanism
probes on each component, and reports a signed-stability arm when requested.
"""

import argparse
import json
import numpy as np
from collections import defaultdict
import functools

from joblib import Parallel, delayed

from esm2_mech.utils.bootstrap import (
    binary_auroc_cluster_bootstrap_ci,
    attach_mechanism_ci,
    stack_oof_over_seeds,
    family_or_gene_clusters,
)
from esm2_mech.utils.constants import BOOTSTRAP_N_RESAMPLES, MECHANISM_CLASSES, N_SEEDS
from esm2_mech.utils.data import load_pfam_map
from esm2_mech.utils.io import write_result_json
from esm2_mech.utils.paths import (
    GEOMETRY_RESULTS_DIR,
    MAGNITUDE_DIRECTION_JSON,
    NAIVE_BASELINE_JSON,
    PFAM_JSON,
)
from esm2_mech.experiments.mechanism.loaders import load_merged
from esm2_mech.utils.metrics import mean_std_n
from esm2_mech.utils.splits import gene_split_cv, family_split_cv
from esm2_mech.utils.probes import run_mlp_binary_cv, run_mlp_probe_cv, run_logreg_cv
from esm2_mech.utils.probes import run_logreg_binary_cv
from esm2_mech.utils.data import embedding_fingerprint
from esm2_mech.utils.classification import validate_complete_classification_splits
from esm2_mech.experiments.geometry.data import (
    load_pathogenicity_geometry_inputs,
    mechanism_geometry_provenance,
    pathogenicity_geometry_provenance,
)
from esm2_mech.experiments.stability.stability_data import (
    load_stability_inputs,
    variant_fingerprint as stability_variant_fingerprint,
)

print = functools.partial(print, flush=True)

GEOMETRY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STABILITY_DATASETS = {
    "none": None,
    "tsuboyama": "tsuboyama",
}
DEFAULT_STABILITY_DATASET = "none"


def load_pathogenicity_canonical(inputs):
    """Expose the validated canonical pathogenicity arrays used by this probe."""
    print(
        f"  Pathogenicity (canonical): {len(inputs.variants)} variants, "
        f"{len(set(inputs.genes))} genes, {int(inputs.labels.sum())} path / "
        f"{int((1 - inputs.labels).sum())} benign"
    )
    return inputs.delta, inputs.labels, inputs.genes


def decompose(delta):
    """Return {'full': delta, 'mag': ||d|| (N,1), 'dir': d/||d|| (N,1280)}."""
    norm = np.linalg.norm(delta, axis=1, keepdims=True)
    mag = norm.astype(np.float32)  # (N, 1)
    direction = (delta / (norm + 1e-8)).astype(np.float32)
    return {"full": delta.astype(np.float32), "mag": mag, "dir": direction}


def run_logreg_multi(
    X, labels, splits, groups, held_out_unit, seed=42, genes=None, return_oof=False
):
    contract = validate_complete_classification_splits(
        splits, requested_folds=5,
        eligible_rows=np.concatenate([test for _train, test in splits]),
        labels=labels, classes=MECHANISM_CLASSES, groups=groups,
        held_out_unit=held_out_unit,
    )
    return run_logreg_cv(
        X,
        labels,
        splits,
        MECHANISM_CLASSES,
        contract,
        seed=seed,
        genes=genes,
        return_oof=return_oof,
    )


def _read_chance_floor(strategy="most_frequent"):
    """Read the measured mechanism chance floor from naive_baseline.json."""
    with open(NAIVE_BASELINE_JSON) as handle:
        nb = json.load(handle)
    strat = nb["by_strategy"][strategy]
    return {
        "gene_split": {
            "mean": strat["gene"]["macro_f1_mean"],
            "std": strat["gene"]["macro_f1_std"],
        },
        "family_split": {
            "mean": strat["family"]["macro_f1_mean"],
            "std": strat["family"]["macro_f1_std"],
        },
    }


def agg_seeds(per_seed_vals):
    if any(value is None or not np.isfinite(value) for value in per_seed_vals):
        return {"mean": None, "std": None, "n": len([v for v in per_seed_vals if v is not None])}
    mean, std, n = mean_std_n(per_seed_vals)
    return {"mean": mean, "std": std, "n": n}


def _pathogenicity_one_seed(seed, feats, y, genes, pfam_map):
    print(f"  [pathogenicity] seed {seed} started", flush=True)
    gs = gene_split_cv(genes, seed=seed)
    fs = family_split_cv(genes, pfam_map, seed=seed)
    family_validation_groups = family_or_gene_clusters(
        genes, pfam_map, is_family_split=True
    )
    res = {}
    for fname, X in feats.items():
        for split_name, splits in [("gene_split", gs), ("family_split", fs)]:
            validation_groups = (
                genes if split_name == "gene_split" else family_validation_groups
            )
            contract = validate_complete_classification_splits(
                splits, requested_folds=5,
                eligible_rows=np.concatenate([test for _train, test in splits]),
                labels=y, classes=[0, 1], groups=validation_groups,
                held_out_unit="gene" if split_name == "gene_split" else "family",
            )
            lr_agg, lr_oof = run_logreg_binary_cv(
                X, y, splits, [0, 1], contract,
                seed=seed, genes=genes, return_oof=True
            )
            lr = lr_agg.get("auroc_mean")
            res[(fname, split_name, "logreg")] = (lr, lr_oof)
            mlp_agg, mlp_oof = run_mlp_binary_cv(
                X,
                y,
                splits,
                [0, 1],
                contract,
                validation_groups=(
                    family_validation_groups if split_name == "family_split" else genes
                ),
                seed=seed,
                genes=genes,
                return_oof=True,
            )
            mlp = mlp_agg.get("auroc_mean")
            res[(fname, split_name, "mlp")] = (mlp, mlp_oof)
            print(
                f"    [pathogenicity seed {seed}] {fname:4s} {split_name:12s} "
                f"logreg={_f(lr)} mlp={_f(mlp)}",
                flush=True,
            )
    print(f"  [pathogenicity] seed {seed} done", flush=True)
    return res


def run_pathogenicity(
    pfam_map,
    seeds,
    inputs,
    n_jobs=-1,
    compute_ci=True,
    n_boot=BOOTSTRAP_N_RESAMPLES,
):
    print("\n" + "=" * 60)
    print("PATHOGENICITY  (binary, variant-level, delta_mean)")
    print("=" * 60)
    delta, y, genes = load_pathogenicity_canonical(inputs)
    feats = decompose(delta)

    per_seed = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_pathogenicity_one_seed)(seed, feats, y, genes, pfam_map)
        for seed in seeds
    )

    collect = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    oof_collect = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for seed, res in zip(seeds, per_seed):
        for (fname, split_name, probe), (auroc, oof) in res.items():
            collect[fname][split_name][probe].append(auroc)
            if oof is not None:
                oof_collect[fname][split_name][probe].append(oof)
            print(f"  seed{seed} {fname:4s} {split_name:12s} {probe}={_f(auroc)}")

    out = {}
    for fname in feats:
        out[fname] = {}
        for split_name in ("gene_split", "family_split"):
            out[fname][split_name] = {
                "logreg_auroc": agg_seeds(collect[fname][split_name]["logreg"]),
                "mlp_auroc": agg_seeds(collect[fname][split_name]["mlp"]),
            }
            if compute_ci:
                for probe in ("logreg", "mlp"):
                    combined = stack_oof_over_seeds(
                        oof_collect[fname][split_name][probe]
                    )
                    if combined is not None:
                        clusters = family_or_gene_clusters(
                            combined["genes"],
                            pfam_map,
                            is_family_split=(split_name == "family_split"),
                        )
                        out[fname][split_name][f"{probe}_auroc"]["ci"] = (
                            binary_auroc_cluster_bootstrap_ci(
                                combined,
                                n_resamples=n_boot,
                                seed=0,
                                clusters=clusters,
                            )
                        )
    return out


def _mechanism_one_seed(seed, feats, labels, genes, pfam_map):
    print(f"  [mechanism] seed {seed} started", flush=True)
    gs = gene_split_cv(genes, seed=seed)
    fs = family_split_cv(genes, pfam_map, seed=seed)
    res = {}
    for fname, X in feats.items():
        for split_name, splits in [("gene_split", gs), ("family_split", fs)]:
            validation_groups = (
                genes
                if split_name == "gene_split"
                else family_or_gene_clusters(genes, pfam_map, is_family_split=True)
            )
            lr, lr_oof = run_logreg_multi(
                X,
                labels,
                splits,
                validation_groups,
                "gene" if split_name == "gene_split" else "family",
                seed=seed,
                genes=genes,
                return_oof=True,
            )
            contract = validate_complete_classification_splits(
                splits, requested_folds=5,
                eligible_rows=np.concatenate([test for _train, test in splits]),
                labels=labels, classes=MECHANISM_CLASSES,
                groups=validation_groups,
                held_out_unit="gene" if split_name == "gene_split" else "family",
            )
            mlp, mlp_oof = run_mlp_probe_cv(
                X,
                labels,
                splits,
                MECHANISM_CLASSES,
                contract,
                validation_groups=validation_groups,
                seed=seed,
                genes=genes,
                return_oof=True,
            )
            res[(fname, split_name)] = {
                "logreg_f1": lr.get("macro_f1_mean"),
                "mlp_f1": mlp.get("macro_f1_mean"),
                "logreg_gof": lr.get("auroc_GOF_mean"),
                "mlp_gof": mlp.get("auroc_GOF_mean"),
                "logreg_oof": lr_oof,
                "mlp_oof": mlp_oof,
            }
            print(
                f"    [mechanism seed {seed}] {fname:4s} {split_name:12s} "
                f"F1(lr={_f(lr.get('macro_f1_mean'))} mlp={_f(mlp.get('macro_f1_mean'))})",
                flush=True,
            )
    print(f"  [mechanism] seed {seed} done", flush=True)
    return res


def run_mechanism(
    pfam_map, seeds, n_jobs=-1, compute_ci=True, n_boot=BOOTSTRAP_N_RESAMPLES
):
    print("\n" + "=" * 60)
    print("MECHANISM  (3-class GOF/LOF/DN, variant-level Gerasimavicius, delta_mean)")
    print("=" * 60)
    dm, labels, genes = load_merged(pfam_map)
    feats = decompose(dm)

    per_seed = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_mechanism_one_seed)(seed, feats, labels, genes, pfam_map)
        for seed in seeds
    )

    collect = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    oof_collect = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for seed, res in zip(seeds, per_seed):
        for (fname, split_name), cell in res.items():
            for key in ("logreg_f1", "mlp_f1", "logreg_gof", "mlp_gof"):
                collect[fname][split_name][key].append(cell[key])
            if cell["logreg_oof"] is not None:
                oof_collect[fname][split_name]["logreg"].append(cell["logreg_oof"])
            if cell["mlp_oof"] is not None:
                oof_collect[fname][split_name]["mlp"].append(cell["mlp_oof"])
            print(
                f"  seed{seed} {fname:4s} {split_name:12s} "
                f"F1(lr={_f(cell['logreg_f1'])} mlp={_f(cell['mlp_f1'])})"
            )

    out = {
        "chance_floor": _read_chance_floor(),
        "input_provenance": mechanism_geometry_provenance(dm, labels, genes, pfam_map),
    }
    for fname in feats:
        out[fname] = {}
        for split_name in ("gene_split", "family_split"):
            c = collect[fname][split_name]
            cell = {
                "logreg_macro_f1": agg_seeds(c["logreg_f1"]),
                "mlp_macro_f1": agg_seeds(c["mlp_f1"]),
                "logreg_gof_auroc": agg_seeds(c["logreg_gof"]),
                "mlp_gof_auroc": agg_seeds(c["mlp_gof"]),
            }
            if compute_ci:
                for probe, out_key in (
                    ("logreg", "logreg_macro_f1"),
                    ("mlp", "mlp_macro_f1"),
                ):
                    combined = stack_oof_over_seeds(
                        oof_collect[fname][split_name][probe]
                    )
                    if combined is not None:
                        clusters = family_or_gene_clusters(
                            combined["genes"],
                            pfam_map,
                            is_family_split=(split_name == "family_split"),
                        )
                        attach_mechanism_ci(
                            cell[out_key],
                            combined,
                            clusters,
                            compute_ci=True,
                            n_resamples=n_boot,
                            seed=0,
                        )
            out[fname][split_name] = cell
    return out


def run_biophysical_direction(seeds, stability_dataset=DEFAULT_STABILITY_DATASET):
    if stability_dataset not in STABILITY_DATASETS:
        raise ValueError(f"unknown stability dataset {stability_dataset!r}")
    if stability_dataset == "none":
        print("\nStability arm not requested (stability_dataset='none').")
        return None

    print("\n" + "=" * 60)
    print(f"BIOPHYSICAL DIRECTION ({stability_dataset} signed ddG, protein-holdout)")
    print("=" * 60)
    from scipy.stats import spearmanr

    stability = load_stability_inputs()
    variants = stability.variants
    ddg = np.asarray(stability.ddg, dtype=np.float64)
    proteins = stability.proteins
    delta = stability.delta_mean
    finite = np.isfinite(ddg)
    n_dropped = int((~finite).sum())
    if n_dropped:
        print(f"  Dropped {n_dropped}/{len(ddg)} variants with non-finite ddG")
    delta, ddg, proteins = delta[finite], ddg[finite], proteins[finite]
    n = len(ddg)
    feats = decompose(delta)
    mag = feats["mag"].ravel()

    c1_rho = float(spearmanr(mag, np.abs(ddg)).correlation)

    # Tsuboyama ddG_ML is mutant minus WT folding stability. Positive values
    # therefore indicate stabilisation, while negative values indicate destabilisation.
    y_sign = (ddg > 0).astype(int)
    c2 = {}
    for fname in ("full", "dir"):
        per_seed = []
        for seed in seeds:
            splits = gene_split_cv(proteins, seed=seed)  # group-holdout by protein
            contract = validate_complete_classification_splits(
                splits, requested_folds=5,
                eligible_rows=np.concatenate([test for _train, test in splits]),
                labels=y_sign, classes=[0, 1], groups=proteins,
                held_out_unit="protein",
            )
            r = run_logreg_binary_cv(
                feats[fname], y_sign, splits, [0, 1], contract, seed=seed
            )
            per_seed.append(r.get("auroc_mean"))
        c2[fname] = agg_seeds(per_seed)

    print(f"  Spearman(||d||, |ddG|) = {c1_rho:.3f}")
    print(
        f"  sign(ddG) AUROC full={_f(c2['full']['mean'])} dir={_f(c2['dir']['mean'])}"
    )
    return {
        "n_variants": int(n),
        "n_proteins": int(len(set(proteins.tolist()))),
        "frac_stabilising": float(y_sign.mean()),
        "ddg_sign_convention": "positive ddG_ML indicates stabilisation",
        "spearman_magnitude_vs_abs_ddg": c1_rho,
        "sign_ddg_auroc": c2,
        "input_provenance": {
            "variant_fingerprint": stability_variant_fingerprint(variants),
            "delta_embedding_fingerprint": embedding_fingerprint(stability.delta_mean),
        },
    }


def _f(x):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def run(
    n_seeds=N_SEEDS,
    stability_dataset=DEFAULT_STABILITY_DATASET,
    compute_ci=True,
    n_boot=BOOTSTRAP_N_RESAMPLES,
):
    """Run the magnitude/direction decomposition over range(n_seeds)."""
    return _run_seeds(
        list(range(n_seeds)),
        stability_dataset=stability_dataset,
        compute_ci=compute_ci,
        n_boot=n_boot,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=N_SEEDS, help="number of seeds (>=1)")
    ap.add_argument(
        "--stability-dataset",
        choices=list(STABILITY_DATASETS),
        default=DEFAULT_STABILITY_DATASET,
        help="dataset for the biophysical-direction arm (default: none = skip)",
    )
    ap.add_argument("--no_ci", action="store_true", help="skip cluster-bootstrap CIs")
    ap.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    args = ap.parse_args()
    if args.seeds < 1:
        ap.error("--seeds must be >= 1")
    run(
        n_seeds=args.seeds,
        stability_dataset=args.stability_dataset,
        compute_ci=not args.no_ci,
        n_boot=args.n_boot,
    )


def _run_seeds(
    seeds,
    stability_dataset=DEFAULT_STABILITY_DATASET,
    compute_ci=True,
    n_boot=BOOTSTRAP_N_RESAMPLES,
):
    pfam_map = load_pfam_map(PFAM_JSON)
    path_inputs = load_pathogenicity_geometry_inputs()

    path_res = run_pathogenicity(
        pfam_map, seeds, path_inputs, compute_ci=compute_ci, n_boot=n_boot
    )
    mech_res = run_mechanism(pfam_map, seeds, compute_ci=compute_ci, n_boot=n_boot)
    bio_res = run_biophysical_direction(seeds, stability_dataset=stability_dataset)

    print("\n" + "=" * 60)
    print("HEADLINE — magnitude vs direction (family-split)")
    print("=" * 60)

    def pa(feat, probe):
        return path_res[feat]["family_split"][probe]["mean"]

    def me(feat):
        return mech_res[feat]["family_split"]["mlp_macro_f1"]["mean"]

    print("  Pathogenicity AUROC (family-split):")
    print(
        f"    full delta   logreg={pa('full', 'logreg_auroc'):.3f}  mlp={pa('full', 'mlp_auroc'):.3f}"
    )
    print(
        f"    magnitude    logreg={pa('mag', 'logreg_auroc'):.3f}  mlp={pa('mag', 'mlp_auroc'):.3f}"
    )
    print(
        f"    direction    logreg={pa('dir', 'logreg_auroc'):.3f}  mlp={pa('dir', 'mlp_auroc'):.3f}"
    )
    print("  Mechanism macro-F1 (family-split, MLP):")
    print(f"    chance floor = {mech_res['chance_floor']['family_split']['mean']:.3f}")
    print(f"    full delta   = {me('full'):.3f}")
    print(f"    magnitude    = {me('mag'):.3f}")
    print(f"    direction    = {me('dir'):.3f}")

    result = {
        "seeds": list(seeds),
        "pathogenicity": path_res,
        "mechanism": mech_res,
        "biophysical_direction": bio_res,
        "descriptive_family_split_summary": {
            "pathogenicity_full": path_res["full"]["family_split"],
            "pathogenicity_magnitude": path_res["mag"]["family_split"],
            "pathogenicity_direction": path_res["dir"]["family_split"],
            "mechanism_chance_floor": mech_res["chance_floor"]["family_split"],
            "mechanism_full": mech_res["full"]["family_split"],
            "mechanism_magnitude": mech_res["mag"]["family_split"],
            "mechanism_direction": mech_res["dir"]["family_split"],
        },
        "analysis_status": "exploratory",
        "input_provenance": pathogenicity_geometry_provenance(path_inputs, pfam_map),
    }
    write_result_json(MAGNITUDE_DIRECTION_JSON, result, seeds=list(seeds))
    print(f"\nResults -> {MAGNITUDE_DIRECTION_JSON}")
    return result


if __name__ == "__main__":
    main()
