"""Probe runs for perturbation scan features.

Compares scan features against baselines under gene-split and family-split CV.
"""

import argparse, functools
import json, os, sys, numpy as np
from pathlib import Path

from esm2_mech.utils.data import build_gene_to_row
from esm2_mech.utils.paths import (
    BADONYI_FEATURES_ALIGNED,
    EMB_MUT_MEAN,
    EMB_WT_MEAN,
    GENE_UNIVERSE,
    PFAM_JSON,
    PROTEOME_FEATURES_ALIGNED,
    RESULTS_DIR as _RESULTS_DIR,
    SCAN_FEATURES_META_JSON,
    SCAN_FEATURES_NPY,
    VALID_VARIANTS_JSON,
)
from esm2_mech.utils.splits import gene_split_cv, family_split_cv
from esm2_mech.utils.embed import load_gene_delta
from esm2_mech.utils.probes import run_logreg_cv, run_histgb_cv
from esm2_mech.utils.constants import MECHANISM_CLASSES, N_SEEDS
from esm2_mech.utils.classification import validate_complete_classification_splits
from esm2_mech.utils.io import write_result_json
from esm2_mech.utils.seed_aggregation import aggregate_result_contract, seed_result_contract
from esm2_mech.experiments.perturbation.seed_summary import (
    aggregate_probe_results,
    read_probe_metric,
)

print = functools.partial(print, flush=True)
OUT = _RESULTS_DIR / "perturbation_scan"
OUT.mkdir(parents=True, exist_ok=True)


DECISION_RULES = {
    "G1": ("scan_only_family_split", "macro_f1", 0.368),
    "G2": ("scan_delta_family_split", "macro_f1", 0.419),
    "G3": ("scan_proteome_family_split", "macro_f1", 0.405),
}


def load_all_features(gene_list):
    """Load all feature matrices aligned to gene_list."""
    features = {}

    # Row order pinned by meta file's gene list, not gene_universe
    scan_X = np.load(SCAN_FEATURES_NPY)
    with open(SCAN_FEATURES_META_JSON) as f:
        scan_meta = json.load(f)
    scan_idx = {g: i for i, g in enumerate(scan_meta["genes"])}

    gene_delta = load_gene_delta(VALID_VARIANTS_JSON, EMB_WT_MEAN, EMB_MUT_MEAN)

    # Proteome rows are in gene_universe.tsv order; gene_list.tsv is a differently-ordered superset
    pg_idx: dict = {}
    if PROTEOME_FEATURES_ALIGNED.exists():
        pg_idx = build_gene_to_row(GENE_UNIVERSE)

    # Require all features present so np.hstack rows stay aligned
    def _has_all(g):
        if g not in scan_idx:
            return False
        if g not in gene_delta:
            return False
        if PROTEOME_FEATURES_ALIGNED.exists() and g not in pg_idx:
            return False
        return True

    gene_mask = np.array([_has_all(g) for g in gene_list])
    scan_gene_list = gene_list[gene_mask]
    print(f"  Genes with all features: {len(scan_gene_list)}/{len(gene_list)}")

    features["scan"] = np.array(
        [scan_X[scan_idx[g]] for g in scan_gene_list], dtype=np.float32
    )
    features["delta"] = np.array(
        [np.mean(gene_delta[g], axis=0) for g in scan_gene_list], dtype=np.float32
    )

    if PROTEOME_FEATURES_ALIGNED.exists():
        proteome_X = np.load(PROTEOME_FEATURES_ALIGNED)
        if proteome_X.shape[0] != len(pg_idx):
            raise ValueError(
                f"{PROTEOME_FEATURES_ALIGNED} has {proteome_X.shape[0]} rows but "
                f"{GENE_UNIVERSE} lists {len(pg_idx)} genes — not row-aligned."
            )
        features["proteome"] = np.array(
            [proteome_X[pg_idx[g]] for g in scan_gene_list], dtype=np.float32
        )
    else:
        print(f"  {PROTEOME_FEATURES_ALIGNED} not found — skipping proteome features")

    row_counts = {name: X.shape[0] for name, X in features.items()}
    if len(set(row_counts.values())) > 1:
        raise RuntimeError(f"Feature row count mismatch after alignment: {row_counts}")

    return features, gene_mask


# Proteome block carries NaN; route these combos to the NaN-native runner
NAN_BEARING_COMBOS = {"scan_proteome"}


def run_probe(X, labels, splits, groups, held_out_unit, seed=42, combo_name=""):
    """Route to NaN-native runner if combo contains proteome block."""
    contract = validate_complete_classification_splits(
        splits, requested_folds=5,
        eligible_rows=np.concatenate([test for _train, test in splits]),
        labels=labels, classes=MECHANISM_CLASSES, groups=groups,
        held_out_unit=held_out_unit,
    )
    if combo_name in NAN_BEARING_COMBOS:
        return run_histgb_cv(
            X, labels, splits, MECHANISM_CLASSES, contract, seed=seed
        )
    return run_logreg_cv(
        X, labels, splits, MECHANISM_CLASSES, contract, seed=seed
    )


def _show_metric(metric):
    if not metric.available:
        return f"unavailable ({metric.message})"
    if metric.spread is None:
        return f"{metric.value:.3f}"
    return f"{metric.value:.3f}±{metric.spread:.3f} seed SD"


def main(n_seeds=N_SEEDS):
    requested_seeds = tuple(range(n_seeds))
    print("=== Loading data ===")

    with open(VALID_VARIANTS_JSON) as f:
        variants = json.load(f)
    for v in variants:
        if "label_3class" not in v:
            v["label_3class"] = (
                "LOF"
                if v.get("mechanism") in ("HI", "AR")
                else v.get("mechanism", "LOF")
            )

    from collections import Counter, defaultdict

    gene_labels = defaultdict(list)
    for v in variants:
        gene_labels[v["gene"].upper()].append(v["label_3class"])
    gene_list = np.array(sorted(gene_labels.keys()))
    labels = np.array([Counter(gene_labels[g]).most_common(1)[0][0] for g in gene_list])

    print(f"Genes: {len(gene_list)}  Classes: {dict(Counter(labels))}")

    # Load features
    features, gene_mask = load_all_features(gene_list)

    gene_list_scan = gene_list[gene_mask]
    labels_scan = labels[gene_mask]
    print(f"Genes with scan features: {len(gene_list_scan)}")

    with open(PFAM_JSON) as f:
        pfam_map = json.load(f)

    scan_X = features["scan"]
    delta_X = features["delta"] if "delta" in features else None
    proteome_X = features.get("proteome")

    combos = {
        "baseline_delta": delta_X,
        "scan_only": scan_X,
        "scan_delta": np.hstack([scan_X, delta_X]) if delta_X is not None else None,
        "scan_proteome": (
            np.hstack([scan_X, proteome_X]) if proteome_X is not None else None
        ),
    }
    declared_arms = [
        f"{combo_name}_{split_name}"
        for combo_name in combos
        for split_name in ("gene_split", "family_split")
    ]
    for combo_name, matrix in combos.items():
        if matrix is None:
            print(f"  [unavailable {combo_name}: its feature block was not loaded]")
    combos = {k: v for k, v in combos.items() if v is not None}

    all_results = {}
    for seed in requested_seeds:
        print(f"\n=== Seed {seed} ===")
        gs = gene_split_cv(gene_list_scan, seed=seed)
        fs = family_split_cv(gene_list_scan, pfam_map, seed=seed)
        seed_res = {}
        for combo_name, X in combos.items():
            split_specs = [
                ("gene_split", gs, gene_list_scan, "gene"),
                (
                    "family_split", fs,
                    np.array([pfam_map.get(gene) for gene in gene_list_scan], dtype=object),
                    "family",
                ),
            ]
            for split_name, splits, groups, held_out_unit in split_specs:
                key = f"{combo_name}_{split_name}"
                r = run_probe(
                    X, labels_scan, splits, groups, held_out_unit,
                    seed=seed, combo_name=combo_name
                )
                seed_res[key] = r
                f1 = r.get("macro_f1_mean")
                gof = r.get("auroc_GOF_mean")
                f1_text = "unavailable" if f1 is None else f"{f1:.3f}"
                gof_text = "unavailable" if gof is None else f"{gof:.3f}"
                print(f"  {key}: F1={f1_text}  GOF={gof_text}")
        all_results[seed] = {
            **seed_result_contract(seed),
            "results": seed_res,
        }

    print(f"\n=== {n_seeds}-SEED SUMMARY ===")
    summary = aggregate_probe_results(requested_seeds, all_results, declared_arms)
    for key in summary:
        f1 = read_probe_metric(summary, key, "macro_f1")
        gof = read_probe_metric(summary, key, "auroc_GOF")
        print(f"  {key}: F1={_show_metric(f1)}  GOF={_show_metric(gof)}")

    print("\n=== DECISION RULES ===")
    gate_results = {}
    for gate, (key, metric, threshold) in DECISION_RULES.items():
        result = read_probe_metric(summary, key, metric)
        val = result.value
        passed = None if not result.available else val > threshold
        gate_results[gate] = {
            "metric": summary[key][metric],
            "value": val,
            "threshold": threshold,
            "passed": passed,
        }
        if passed is None:
            print(f"  {gate}: {key} {metric} = Unscorable")
        else:
            status = "PASS ✓" if passed else "FAIL ✗"
            print(
                f"  {gate}: {key} {metric} = {val:.3f} "
                f"(threshold {threshold:.3f}) → {status}"
            )

    if gate_results["G1"]["passed"] is False:
        print(
            "\n  G1 failed — scan features do not improve on ClinVar-pattern baseline."
        )
        print("  Do not proceed to G2/G3 analysis.")
    elif gate_results["G1"]["passed"] is None:
        print("\n  G1 could not be evaluated because its seed summary is unavailable.")

    out = {
        **aggregate_result_contract(),
        "summary": summary,
        "gate_results": gate_results,
        "per_seed": {str(seed): all_results[seed] for seed in requested_seeds},
        "n_genes": int(len(gene_list_scan)),
        "feature_combos": list(combos.keys()),
    }
    out_path = OUT / "probe_results.json"
    write_result_json(out_path, out, seeds=list(requested_seeds))
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    arguments = parser.parse_args()
    main(n_seeds=arguments.seeds)
