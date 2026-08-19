"""Magnitude-vs-direction decomposition of ESM-2 deltas (plan_magnitude_direction.md).

Splits each delta into magnitude and direction, re-runs pathogenicity/mechanism
probes on each component, and evaluates pre-registered gates P1–P4.
"""

import argparse
import json
import numpy as np
from collections import defaultdict
import functools

print = functools.partial(print, flush=True)

from joblib import Parallel, delayed

from esm2_mech.utils.bootstrap import (
    binary_auroc_cluster_bootstrap_ci, bootstrap_mechanism_metrics_from_oof,
    stack_oof_over_seeds,
    family_or_gene_clusters,
)
from esm2_mech.utils.constants import BOOTSTRAP_N_RESAMPLES, MIN_TRAIN_CLASSES, N_SEEDS
from esm2_mech.utils.data import load_pfam_map
from esm2_mech.utils.io import write_result_json
from esm2_mech.utils.paths import (
    GEOMETRY_RESULTS_DIR,
    MAGNITUDE_DIRECTION_JSON,
    NAIVE_BASELINE_JSON,
    MEGASCALE_TSUBOYAMA_VARIANTS_JSON,
    PATH_EMB_WT_MEAN,
    PATH_EMB_MUT_MEAN,
    PATHOGENICITY_CANONICAL_VARIANTS_JSON,
    PFAM_JSON,
    MEGASCALE_EMB_WT_MEAN,
    MEGASCALE_EMB_MUT_MEAN,
)
from esm2_mech.experiments.mechanism.loaders import load_mechanism_variants
from esm2_mech.utils.metrics import mean_std_n
from esm2_mech.utils.splits import gene_split_cv, family_split_cv
from esm2_mech.utils.probes import run_mlp_binary_cv, run_mlp_probe_cv, run_logreg_cv
from esm2_mech.utils.probes import run_logreg_binary_cv

GEOMETRY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

P1_PATH_MAG_MIN = 0.85
P2_PATH_DIR_MAX = 0.70
P3_MECH_MARGIN = 0.02
P4_SIGN_AUROC_MIN = 0.65
P4_MAG_SPEARMAN_MIN = 0.30

STABILITY_DATASETS = {
    "none": None,
    "tsuboyama": (
        MEGASCALE_TSUBOYAMA_VARIANTS_JSON,
        MEGASCALE_EMB_WT_MEAN,
        MEGASCALE_EMB_MUT_MEAN,
    ),
}
DEFAULT_STABILITY_DATASET = "none"

PATH_CANON_VARIANTS = PATHOGENICITY_CANONICAL_VARIANTS_JSON
PATH_CANON_WT_EMB = PATH_EMB_WT_MEAN
PATH_CANON_MUT_EMB = PATH_EMB_MUT_MEAN


def _pathogenicity_label(label):
    """Map a canonical-set label to 1 (pathogenic) / 0 (benign); never a catch-all."""
    if label == "pathogenic":
        return 1
    if label == "benign":
        return 0
    raise ValueError(f"unexpected pathogenicity label {label!r} (expected 'pathogenic'/'benign')")


def load_pathogenicity_canonical():
    """Load the canonical pathogenicity set (row-aligned to PATH_EMB_*; matches result_6)."""
    with open(PATH_CANON_VARIANTS) as fh:
        variants = json.load(fh)
    wt = np.load(PATH_CANON_WT_EMB)
    mut = np.load(PATH_CANON_MUT_EMB)
    delta = mut - wt
    if not (len(variants) == delta.shape[0]):
        raise ValueError(
            f"variant/embedding row mismatch: {len(variants)} variants vs "
            f"{delta.shape[0]} embedding rows — canonical file is not row-aligned."
        )
    genes = np.array([v["gene"] for v in variants])
    y = np.array([_pathogenicity_label(v["label"]) for v in variants])
    print(
        f"  Pathogenicity (canonical): {len(variants)} variants, "
        f"{len(set(genes))} genes, {int(y.sum())} path / {int((1-y).sum())} benign"
    )
    return delta, y, genes



def decompose(delta):
    """Return {'full': delta, 'mag': ||d|| (N,1), 'dir': d/||d|| (N,1280)}."""
    norm = np.linalg.norm(delta, axis=1, keepdims=True)
    mag = norm.astype(np.float32)  # (N, 1)
    direction = (delta / (norm + 1e-8)).astype(np.float32)
    return {"full": delta.astype(np.float32), "mag": mag, "dir": direction}



def run_logreg_multi(X, labels, splits, seed=42, genes=None, return_oof=False):
    return run_logreg_cv(
        X, labels, splits, seed=seed, min_train_classes=MIN_TRAIN_CLASSES,
        genes=genes, return_oof=return_oof,
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
    mean, std, n = mean_std_n(per_seed_vals)
    return {"mean": mean, "std": std, "n": n}

def _pathogenicity_one_seed(seed, feats, y, genes, pfam_map):
    print(f"  [pathogenicity] seed {seed} started", flush=True)
    gs = gene_split_cv(genes, seed=seed)
    fs = family_split_cv(genes, pfam_map, seed=seed)
    res = {}
    for fname, X in feats.items():
        for split_name, splits in [("gene_split", gs), ("family_split", fs)]:
            lr_agg, lr_oof = run_logreg_binary_cv(
                X, y, splits, seed=seed, genes=genes, return_oof=True
            )
            lr = lr_agg.get("auroc_mean")
            res[(fname, split_name, "logreg")] = (lr, lr_oof)
            mlp_agg, mlp_oof = run_mlp_binary_cv(
                X, y, splits, seed=seed, genes=genes, return_oof=True
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


def run_pathogenicity(pfam_map, seeds, n_jobs=-1, compute_ci=True, n_boot=BOOTSTRAP_N_RESAMPLES):
    print("\n" + "=" * 60)
    print("PATHOGENICITY  (binary, variant-level, delta_mean)")
    print("=" * 60)
    delta, y, genes = load_pathogenicity_canonical()
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
                    combined = stack_oof_over_seeds(oof_collect[fname][split_name][probe])
                    if combined is not None:
                        clusters = family_or_gene_clusters(
                            combined["genes"], pfam_map,
                            is_family_split=(split_name == "family_split"),
                        )
                        out[fname][split_name][f"{probe}_auroc"]["ci"] = (
                            binary_auroc_cluster_bootstrap_ci(
                                combined, n_resamples=n_boot, seed=0,
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
            lr, lr_oof = run_logreg_multi(
                X, labels, splits, seed=seed, genes=genes, return_oof=True
            )
            mlp, mlp_oof = run_mlp_probe_cv(
                X, labels, splits, seed=seed, genes=genes, return_oof=True
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


def run_mechanism(pfam_map, seeds, n_jobs=-1, compute_ci=True, n_boot=BOOTSTRAP_N_RESAMPLES):
    print("\n" + "=" * 60)
    print("MECHANISM  (3-class GOF/LOF/DN, variant-level Gerasimavicius, delta_mean)")
    print("=" * 60)
    dm, _dp, labels, genes = load_mechanism_variants(pfam_map)
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

    out = {"chance_floor": _read_chance_floor()}
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
                for probe, out_key in (("logreg", "logreg_macro_f1"), ("mlp", "mlp_macro_f1")):
                    combined = stack_oof_over_seeds(oof_collect[fname][split_name][probe])
                    if combined is not None:
                        clusters = family_or_gene_clusters(
                            combined["genes"], pfam_map,
                            is_family_split=(split_name == "family_split"),
                        )
                        cell[out_key]["ci"] = bootstrap_mechanism_metrics_from_oof(
                            combined, clusters, n_resamples=n_boot, seed=0,
                        )
            out[fname][split_name] = cell
    return out



def run_biophysical_direction(seeds, stability_dataset=DEFAULT_STABILITY_DATASET):
    cfg = STABILITY_DATASETS.get(stability_dataset)
    if cfg is None:
        print(
            f"\n[Probe C] stability_dataset='{stability_dataset}' — skipping "
            "biophysical-direction arm (no stability set selected)."
        )
        return None
    variants_json, wt_emb, mut_emb = cfg
    if not (wt_emb.exists() and mut_emb.exists() and variants_json.exists()):
        print(
            f"\n[Probe C] stability_dataset='{stability_dataset}' selected but its "
            f"variants/embeddings are not present ({variants_json.name}) — skipping."
        )
        return None

    print("\n" + "=" * 60)
    print(
        f"PROBE C  biophysical direction ({stability_dataset} signed ddG, protein-holdout)"
    )
    print("=" * 60)
    from scipy.stats import spearmanr

    with open(variants_json) as fh:
        variants = json.load(fh)
    ddg = np.array(
        [v["ddg"] if v["ddg"] is not None else np.nan for v in variants],
        dtype=np.float64,
    )
    proteins = np.array([v["protein"] for v in variants])
    wt = np.load(wt_emb)
    mut = np.load(mut_emb)
    delta = mut - wt
    if not (len(delta) == len(ddg) == len(proteins)):
        raise ValueError(
            f"row mismatch in {variants_json.name}: {len(delta)} embedding rows vs "
            f"{len(ddg)} ddG values vs {len(proteins)} proteins — not row-aligned."
        )
    finite = np.isfinite(ddg)
    n_dropped = int((~finite).sum())
    if n_dropped:
        print(f"  Dropped {n_dropped}/{len(ddg)} variants with non-finite ddG")
    delta, ddg, proteins = delta[finite], ddg[finite], proteins[finite]
    n = len(ddg)
    feats = decompose(delta)
    mag = feats["mag"].ravel()

    c1_rho = float(spearmanr(mag, np.abs(ddg)).correlation)

    y_sign = (ddg > 0).astype(int)
    c2 = {}
    for fname in ("full", "dir"):
        per_seed = []
        for seed in seeds:
            splits = gene_split_cv(proteins, seed=seed)  # group-holdout by protein
            r = run_logreg_binary_cv(feats[fname], y_sign, splits, seed=seed)
            per_seed.append(r.get("auroc_mean"))
        c2[fname] = agg_seeds(per_seed)

    print(f"  C1 Spearman(||d||, |ddG|) = {c1_rho:.3f}")
    print(
        f"  C2 sign(ddG) AUROC full={_f(c2['full']['mean'])} dir={_f(c2['dir']['mean'])}"
    )
    return {
        "n_variants": int(n),
        "n_proteins": int(len(set(proteins.tolist()))),
        "frac_destabilising": float(y_sign.mean()),
        "c1_spearman_mag_absddg": c1_rho,
        "c2_sign_auroc": c2,
    }


def _f(x):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def _best(path_block, split, metric_lr, metric_mlp):
    lr = path_block[split][metric_lr]["mean"]
    mlp = path_block[split][metric_mlp]["mean"]
    vals = [v for v in (lr, mlp) if not np.isnan(v)]
    return max(vals) if vals else float("nan")

def _is_missing(x):
    return x is None or (isinstance(x, float) and np.isnan(x))


def evaluate_gates(path_res, mech_res, bio_res):
    gates = {}

    p1_val = _best(path_res["mag"], "family_split", "logreg_auroc", "mlp_auroc")
    gates["P1"] = {
        "desc": "magnitude-only pathogenicity AUROC >= 0.85 (family-split)",
        "value": p1_val,
        "threshold": P1_PATH_MAG_MIN,
        "passed": None if _is_missing(p1_val) else bool(p1_val >= P1_PATH_MAG_MIN),
    }

    p2_val = _best(path_res["dir"], "family_split", "logreg_auroc", "mlp_auroc")
    gates["P2"] = {
        "desc": "direction-only pathogenicity AUROC <= 0.70 (family-split)",
        "value": p2_val,
        "threshold": P2_PATH_DIR_MAX,
        "passed": None if _is_missing(p2_val) else bool(p2_val <= P2_PATH_DIR_MAX),
    }

    floor = mech_res["chance_floor"]["family_split"]["mean"]
    p3_val = mech_res["dir"]["family_split"]["mlp_macro_f1"]["mean"]
    p3_missing = _is_missing(floor) or _is_missing(p3_val)
    p3_thr = None if _is_missing(floor) else floor + P3_MECH_MARGIN
    gates["P3"] = {
        "desc": "direction-only mechanism macro-F1 <= chance_floor + 0.02 (family-split)",
        "value": p3_val,
        "chance_floor": floor,
        "threshold": p3_thr,
        "passed": None if p3_missing else bool(p3_val <= p3_thr),
    }

    if bio_res is not None:
        full_mean = bio_res["c2_sign_auroc"]["full"]["mean"]
        dir_mean = bio_res["c2_sign_auroc"]["dir"]["mean"]
        scorable = [v for v in (full_mean, dir_mean) if not _is_missing(v)]
        sign_auroc = max(scorable) if scorable else float("nan")
        rho = bio_res["c1_spearman_mag_absddg"]
        p4_missing = _is_missing(sign_auroc) or _is_missing(rho)
        gates["P4"] = {
            "desc": "S1724 sign(ddG) AUROC >= 0.65 AND Spearman >= 0.30",
            "sign_auroc": sign_auroc,
            "spearman": rho,
            "passed": None if p4_missing else bool(
                sign_auroc >= P4_SIGN_AUROC_MIN and rho >= P4_MAG_SPEARMAN_MIN
            ),
        }
    else:
        gates["P4"] = {
            "desc": "stability biophysical-direction arm not run — Probe C skipped",
            "passed": None,
        }

    return gates


def run(
    n_seeds=N_SEEDS, stability_dataset=DEFAULT_STABILITY_DATASET,
    compute_ci=True, n_boot=BOOTSTRAP_N_RESAMPLES,
):
    """Run the magnitude/direction decomposition over range(n_seeds)."""
    return _run_seeds(
        list(range(n_seeds)), stability_dataset=stability_dataset,
        compute_ci=compute_ci, n_boot=n_boot,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=N_SEEDS, help="number of seeds (>=1)")
    ap.add_argument(
        "--stability-dataset",
        choices=list(STABILITY_DATASETS),
        default=DEFAULT_STABILITY_DATASET,
        help="dataset for the Probe C biophysical-direction arm (default: none = skip)",
    )
    ap.add_argument("--no_ci", action="store_true", help="skip cluster-bootstrap CIs")
    ap.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    args = ap.parse_args()
    if args.seeds < 1:
        ap.error("--seeds must be >= 1")
    run(
        n_seeds=args.seeds, stability_dataset=args.stability_dataset,
        compute_ci=not args.no_ci, n_boot=args.n_boot,
    )


def _run_seeds(
    seeds, stability_dataset=DEFAULT_STABILITY_DATASET,
    compute_ci=True, n_boot=BOOTSTRAP_N_RESAMPLES,
):
    pfam_map = load_pfam_map(PFAM_JSON)

    path_res = run_pathogenicity(pfam_map, seeds, compute_ci=compute_ci, n_boot=n_boot)
    mech_res = run_mechanism(pfam_map, seeds, compute_ci=compute_ci, n_boot=n_boot)
    bio_res = run_biophysical_direction(seeds, stability_dataset=stability_dataset)

    gates = evaluate_gates(path_res, mech_res, bio_res)

    print("\n" + "=" * 60)
    print("HEADLINE — magnitude vs direction (family-split)")
    print("=" * 60)

    def pa(feat, probe):
        return path_res[feat]["family_split"][probe]["mean"]

    def me(feat):
        return mech_res[feat]["family_split"]["mlp_macro_f1"]["mean"]

    print("  Pathogenicity AUROC (family-split):")
    print(
        f"    full delta   logreg={pa('full','logreg_auroc'):.3f}  mlp={pa('full','mlp_auroc'):.3f}"
    )
    print(
        f"    magnitude    logreg={pa('mag','logreg_auroc'):.3f}  mlp={pa('mag','mlp_auroc'):.3f}"
    )
    print(
        f"    direction    logreg={pa('dir','logreg_auroc'):.3f}  mlp={pa('dir','mlp_auroc'):.3f}"
    )
    print("  Mechanism macro-F1 (family-split, MLP):")
    print(f"    chance floor = {mech_res['chance_floor']['family_split']['mean']:.3f}")
    print(f"    full delta   = {me('full'):.3f}")
    print(f"    magnitude    = {me('mag'):.3f}")
    print(f"    direction    = {me('dir'):.3f}")

    print("\n" + "=" * 60)
    print("DECISION GATES")
    print("=" * 60)
    for g, d in gates.items():
        status = "SKIP" if d["passed"] is None else ("PASS" if d["passed"] else "FAIL")
        print(f"  {g}: {d['desc']}")
        print(f"       -> {status}")

    p1, p3 = gates["P1"]["passed"], gates["P3"]["passed"]
    print(
        "\n  Load-bearing (P1 AND P3):",
        (
            "PASS — magnitude carries pathogenicity, direction carries no mechanism."
            if (p1 and p3)
            else "NOT MET — see plan failure modes."
        ),
    )

    result = {
        "seeds": list(seeds),
        "pathogenicity": path_res,
        "mechanism": mech_res,
        "biophysical_direction": bio_res,
        "gates": gates,
        "thresholds": {
            "P1_path_mag_min": P1_PATH_MAG_MIN,
            "P2_path_dir_max": P2_PATH_DIR_MAX,
            "P3_mech_margin": P3_MECH_MARGIN,
            "P4_sign_auroc_min": P4_SIGN_AUROC_MIN,
            "P4_mag_spearman_min": P4_MAG_SPEARMAN_MIN,
        },
    }
    write_result_json(MAGNITUDE_DIRECTION_JSON, result, seeds=list(seeds))
    print(f"\nResults -> {MAGNITUDE_DIRECTION_JSON}")
    return result


if __name__ == "__main__":
    main()
