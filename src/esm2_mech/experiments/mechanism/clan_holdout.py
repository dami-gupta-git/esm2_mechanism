"""Leave-one-Pfam-clan-out evaluation of ESM-2 mechanism signal.

Tests whether delta mechanism signal generalises to unseen clans or is clan-level memorisation.
"""

import argparse
import json
import os
import subprocess
import warnings
from collections import Counter, defaultdict

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import functools

from esm2_mech.utils.bootstrap import attach_mechanism_ci
from esm2_mech.utils.constants import BOOTSTRAP_N_RESAMPLES, MECHANISM_CLASSES, N_SEEDS
from esm2_mech.utils.io import load_variants_and_delta, write_result_json
from esm2_mech.utils.data import load_pfam_map
from esm2_mech.utils.classification import validate_classification_splits
from esm2_mech.utils.metrics import (
    aggregate_folds,
    align_proba,
    compute_metrics,
    empty_aggregate_metrics,
    majority_baseline_f1,
)
from esm2_mech.utils.paths import (
    CONTRASTIVE_AGGREGATE_JSON,
    EMB_VALID_VARIANTS_JSON,
    EMB_MUT_MEAN,
    EMB_WT_MEAN,
    NONLINEAR_RESULTS_SEED_JSON,
    PFAM_JSON,
    RESULTS_DIR,
    VALID_VARIANTS_JSON,
)
from esm2_mech.utils.seed_aggregation import (
    aggregate_seed_results,
    read_seed_point_estimate,
)
from esm2_mech.experiments.mechanism.seed_results import read_across_seed_metric

print = functools.partial(print, flush=True)

warnings.filterwarnings("ignore")


def load_data(data_dir, emb_dir):
    variants, labels, genes, delta, _ = load_variants_and_delta(
        VALID_VARIANTS_JSON, EMB_VALID_VARIANTS_JSON, EMB_WT_MEAN, EMB_MUT_MEAN
    )
    return variants, labels, genes, delta


def load_clan_map(clan_file):
    """Parse Pfam-A.clans.tsv.gz into pfam_acc->clan_id and clan_id->clan_name maps."""
    result = subprocess.run(
        ["gunzip", "-c", clan_file], capture_output=True, check=True
    )
    clan_map = {}
    clan_names = {}
    for line in result.stdout.decode().strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 3:
            pfam_acc, clan_id, clan_name = parts[0], parts[1], parts[2]
            clan_map[pfam_acc] = clan_id
            clan_names[clan_id] = clan_name
    print(f"Loaded {len(clan_map)} Pfam->clan mappings, {len(clan_names)} clans")
    return clan_map, clan_names


def train_mlp(X_train, y_train, seed=42):
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 64),
        activation="relu",
        alpha=1e-3,
        max_iter=300,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_probe(clf, X_test, y_test, le):
    """Returns (results, proba_aligned) with proba columns aligned to MECHANISM_CLASSES."""
    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    clf_str_classes = np.array(le.inverse_transform(clf.classes_))
    proba_aligned = align_proba(
        proba,
        clf_str_classes,
        MECHANISM_CLASSES,
        allow_missing_classes=False,
    )
    true_labels = le.inverse_transform(y_test)
    predicted_labels = le.inverse_transform(pred)
    results = compute_metrics(
        true_labels, predicted_labels, proba_aligned, MECHANISM_CLASSES
    )
    results["n_test"] = int(len(y_test))
    results["class_dist_test"] = {
        str(key): int(value) for key, value in Counter(true_labels).items()
    }
    for class_name in MECHANISM_CLASSES:
        results[f"auroc_{class_name}"] = results["per_class_auroc"][class_name]
    return results, proba_aligned


def _read_live_family_split_refs():
    """Read live family-split F1 floors from MLP and contrastive result files. None if not yet produced."""
    requested_seeds = tuple(range(N_SEEDS))
    mlp_results = []
    for seed in requested_seeds:
        mlp_path = str(NONLINEAR_RESULTS_SEED_JSON).format(seed=seed)
        if not os.path.exists(mlp_path):
            mlp_results = []
            break
        with open(mlp_path) as handle:
            mlp_results.append(json.load(handle))
    mlp_f1 = None
    if mlp_results:
        metric = read_seed_point_estimate(
            aggregate_seed_results(
                requested_seeds,
                mlp_results,
                lambda result: result.get("mlp_delta_mean_family", {}).get(
                    "macro_f1_mean"
                ),
            )
        )
        mlp_f1 = metric.value if metric.available else None

    contrastive_f1 = None
    if CONTRASTIVE_AGGREGATE_JSON.exists():
        contrastive_f1 = read_across_seed_metric(
            str(CONTRASTIVE_AGGREGATE_JSON),
            "family_split",
            "contrastive_knn",
        )
    return mlp_f1, contrastive_f1


def run_clan_holdout(delta, labels, genes, gene_clan, clan_names, le, seed=42, n_boot=BOOTSTRAP_N_RESAMPLES):
    """Leave-one-clan-out CV with clan-resampled cluster-bootstrap CI."""
    y = le.transform(labels)
    clan_to_idx = defaultdict(list)
    for i, g in enumerate(genes):
        clan = gene_clan.get(g)
        if clan:
            clan_to_idx[clan].append(i)

    qualifying = []
    for clan, idxs in clan_to_idx.items():
        clan_labels = labels[idxs]
        mech_counts = Counter(clan_labels)
        sorted_mechs = sorted(mech_counts.items(), key=lambda x: -x[1])
        if len(sorted_mechs) < 2:
            continue
        min_second = sorted_mechs[1][1]
        n_genes = len(set(genes[idxs]))
        if min_second >= 20 and n_genes >= 3:
            qualifying.append(
                {
                    "clan": clan,
                    "name": clan_names.get(clan, clan),
                    "idxs": idxs,
                    "mechs": dict(mech_counts),
                    "n_genes": n_genes,
                }
            )

    qualifying.sort(key=lambda x: -len(x["idxs"]))
    print(f"\nQualifying clans for holdout: {len(qualifying)}")
    for q in qualifying:
        print(
            f"  {q['clan']:8s} {q['name']:25s} genes={q['n_genes']:3d} "
            f"variants={len(q['idxs']):4d} mechs={q['mechs']}"
        )

    splits = []
    for item in qualifying:
        test_idx = np.asarray(item["idxs"], dtype=int)
        train_idx = np.setdiff1d(np.arange(len(delta)), test_idx)
        splits.append((train_idx, test_idx))
    eligible_rows = (
        np.concatenate([test for _train, test in splits])
        if splits
        else np.array([], dtype=int)
    )
    clan_groups = np.array([gene_clan.get(gene) for gene in genes], dtype=object)
    split_contract = validate_classification_splits(
        splits,
        requested_folds=len(qualifying),
        eligible_rows=eligible_rows,
        labels=labels,
        classes=MECHANISM_CLASSES,
        required_train_classes=MECHANISM_CLASSES,
        required_test_classes=None,
        minimum_test_classes=2,
        allow_missing_classifier_classes=False,
        groups=clan_groups,
        held_out_unit="clan",
    ) if qualifying else {
        "status": "unscorable",
        "requested_folds": 0,
        "eligible_rows": 0,
        "classes": list(MECHANISM_CLASSES),
        "held_out_unit": "clan",
        "group_count": 0,
        "failures": [{"scope": "split_set", "reason": "no_qualifying_clans"}],
    }
    if split_contract["status"] != "valid":
        return [], qualifying, None, None, split_contract

    reference_failures = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        try:
            majority_baseline_f1(
                labels[train_idx], labels[test_idx], MECHANISM_CLASSES
            )
        except ValueError as error:
            reference_failures.append(
                {"scope": "fold", "fold": fold_idx, "reason": str(error)}
            )
    if reference_failures:
        split_contract = dict(split_contract)
        split_contract["status"] = "unscorable"
        split_contract["failures"] = [
            *split_contract.get("failures", []), *reference_failures
        ]
        return [], qualifying, None, None, split_contract

    clan_results = []
    oof_y, oof_proba, oof_clan, oof_rows, oof_folds = [], [], [], [], []

    for fold_idx, (q, (train_idx, test_idx)) in enumerate(zip(qualifying, splits)):
        clan = q["clan"]
        test_idx = np.array(q["idxs"])
        y_tr = y[train_idx]
        y_te = y[test_idx]

        mu = delta[train_idx].mean(0)
        std = delta[train_idx].std(0) + 1e-8
        X_tr = (delta[train_idx] - mu) / std
        X_te = (delta[test_idx] - mu) / std

        print(
            f"\n  Clan {clan} ({q['name']})  "
            f"train={len(train_idx)} test={len(test_idx)} "
            f"test_mechs={dict(Counter(labels[test_idx]))}"
        )

        try:
            clf = train_mlp(X_tr, y_tr, seed=seed)
            mlp_res, proba_aligned = evaluate_probe(clf, X_te, y_te, le)
            oof_y.append(labels[test_idx])
            oof_proba.append(proba_aligned)
            oof_clan.append(np.full(len(test_idx), clan, dtype=object))
            oof_rows.append(test_idx)
            oof_folds.append(np.full(len(test_idx), fold_idx, dtype=int))
            k = min(10, len(X_tr) - 1)
            knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
            knn.fit(X_tr, y_tr)
            pred_knn = knn.predict(X_te)
            knn_proba = align_proba(
                knn.predict_proba(X_te),
                le.inverse_transform(knn.classes_),
                MECHANISM_CLASSES,
                allow_missing_classes=False,
            )
            knn_metrics = compute_metrics(
                labels[test_idx],
                le.inverse_transform(pred_knn),
                knn_proba,
                MECHANISM_CLASSES,
            )
            maj_f1, _ = majority_baseline_f1(
                labels[train_idx], labels[test_idx], MECHANISM_CLASSES
            )
        except Exception as error:
            clan_results.append(
                {
                    "clan": clan,
                    "fold": fold_idx,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            return clan_results, qualifying, None, None, split_contract

        knn_f1 = knn_metrics["macro_f1"]

        def metric_text(value):
            return "NA" if value is None else f"{value:.3f}"

        print(
            f"    MLP F1={metric_text(mlp_res['macro_f1'])}  "
            f"kNN F1={metric_text(knn_f1)}  "
            f"majority F1={metric_text(maj_f1)}  "
            f"GOF={metric_text(mlp_res['auroc_GOF'])}  "
            f"DN={metric_text(mlp_res['auroc_DN'])}  "
            f"LOF={metric_text(mlp_res['auroc_LOF'])}"
        )

        clan_results.append(
            {
                "clan": clan,
                "clan_name": q["name"],
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_genes_test": int(q["n_genes"]),
                "test_mechs": {k: int(v) for k, v in Counter(labels[test_idx]).items()},
                "mlp": mlp_res,
                "knn": knn_metrics,
                "knn_macro_f1": knn_f1,
                "majority_macro_f1": maj_f1,
            }
        )

    oof = None
    ci = None
    if oof_y:
        oof = {
            "y_true": np.concatenate(oof_y),
            "proba": np.concatenate(oof_proba),
            "clan": np.concatenate(oof_clan),
            "row_ids": np.concatenate(oof_rows),
            "folds": np.concatenate(oof_folds),
        }
        ci_container: dict = {}
        attach_mechanism_ci(
            ci_container,
            oof,
            oof["clan"],
            compute_ci=True,
            n_resamples=n_boot,
            seed=seed,
        )
        ci = ci_container["ci"]
        macro_interval = ci["macro_f1"]
        if macro_interval["ci_suppressed"]:
            print(
                "\n  Clan-resampled CI: unavailable "
                f"({macro_interval['reason']})"
            )
        else:
            print(
                f"\n  Clan-resampled CI (n_clusters={len(set(oof['clan'].tolist()))}, "
                f"n_resamples={n_boot}): "
                f"macro_f1 point={macro_interval['point']} "
                f"[{macro_interval['ci_low']}, {macro_interval['ci_high']}]"
            )

    return clan_results, qualifying, ci, oof, split_contract


def aggregate(clan_results, split_contract, ci=None):
    """Weighted and unweighted aggregates across qualifying clans."""
    requested_folds = split_contract["requested_folds"]
    failed_results = [
        result
        for result in clan_results
        if result.get("status") == "failed" or "mlp" not in result
    ]
    if (
        split_contract["status"] != "valid"
        or len(clan_results) != requested_folds
        or failed_results
    ):
        result = empty_aggregate_metrics(
            MECHANISM_CLASSES,
            requested_folds,
            "split_validation_failed" if split_contract["status"] != "valid" else "runtime_failure",
        )
        result.update(
            {
                "status": "unscorable" if split_contract["status"] != "valid" else "failed",
                "n_clans": requested_folds,
                "completed_folds": sum("mlp" in result for result in clan_results),
                "split_validation": split_contract,
                "ci": ci,
            }
        )
        return result

    shared = aggregate_folds(
        [result["mlp"] for result in clan_results],
        MECHANISM_CLASSES,
        requested_folds,
    )
    mlp_f1s = [result["mlp"]["macro_f1"] for result in clan_results]
    knn_f1s = [result["knn"]["macro_f1"] for result in clan_results]
    maj_f1s = [result["majority_macro_f1"] for result in clan_results]
    weights = [result["n_test"] for result in clan_results]

    def wmean(vals, ws):
        ws = np.array(ws, dtype=float)
        return float(np.average(vals, weights=ws))

    shared.update({
        "status": "success",
        "n_clans": len(clan_results),
        "mlp_macro_f1_mean": float(np.mean(mlp_f1s)),
        "mlp_macro_f1_std": float(np.std(mlp_f1s)),
        "mlp_macro_f1_weighted": wmean(mlp_f1s, weights),
        "knn_macro_f1_mean": float(np.mean(knn_f1s)),
        "majority_macro_f1_mean": float(np.mean(maj_f1s)),
        "per_class_auroc_mean": {
            class_name: shared[f"auroc_{class_name}_mean"]
            for class_name in MECHANISM_CLASSES
        },
        "ci": ci,
        "split_validation": split_contract,
    })
    return shared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clan_file", required=True, help="Path to Pfam-A.clans.tsv.gz"
    )
    parser.add_argument("--out_dir", default=str(RESULTS_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=== Loading data ===")
    variants, labels, genes, delta = load_data(None, None)
    pfam_map = load_pfam_map(PFAM_JSON)
    clan_map, clan_names = load_clan_map(args.clan_file)

    gene_clan = {}
    for gene, pfam_acc in pfam_map.items():
        if pfam_acc in clan_map:
            gene_clan[gene] = clan_map[pfam_acc]
    print(f"Genes with clan assignment: {len(gene_clan)}/{len(pfam_map)}")

    le = LabelEncoder()
    le.fit(["GOF", "DN", "LOF"])
    print(f"Classes: {list(le.classes_)}")

    print("\n=== Clan-level holdout evaluation ===")
    clan_results, qualifying, ci, _oof, split_contract = run_clan_holdout(
        delta, labels, genes, gene_clan, clan_names, le, seed=args.seed, n_boot=args.n_boot
    )

    agg = aggregate(clan_results, split_contract, ci=ci)

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    print(f"  Clans evaluated: {agg['n_clans']}")
    if agg["status"] != "success":
        print(f"  Result: {agg['status'].capitalize()}")
        for failure in split_contract.get("failures", []):
            print(f"  Reason: {failure}")

        family_split_mlp_f1, family_split_contrastive_f1 = _read_live_family_split_refs()
        results = {
            "description": (
                "Leave-one-clan-out evaluation. Train on all variants except clan X, "
                "test on clan X. Tests whether ESM-2 delta mechanism signal generalises "
                "to completely unseen protein clans (lookup vs real signal)."
            ),
            "seed": args.seed,
            "n_boot": args.n_boot,
            "aggregate": agg,
            "per_clan": clan_results,
            "references": {
                "family_split_mlp_f1": family_split_mlp_f1,
                "family_split_contrastive_f1": family_split_contrastive_f1,
            },
        }
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(
            args.out_dir, f"clan_holdout_results_seed{args.seed}.json"
        )
        write_result_json(out_path, results, seeds=[args.seed], indent=2)
        print(f"\nResults written to {out_path}")
        return
    print(
        f"  MLP macro-F1 (unweighted mean ± std): "
        f"{agg['mlp_macro_f1_mean']:.3f} ± {agg['mlp_macro_f1_std']:.3f}"
    )
    print(f"  MLP macro-F1 (weighted by clan size): {agg['mlp_macro_f1_weighted']:.3f}")
    print(f"  k-NN macro-F1 (mean):                 {agg['knn_macro_f1_mean']:.3f}")
    print(
        f"  Majority baseline (mean):             {agg['majority_macro_f1_mean']:.3f}"
    )
    print(f"  Per-class AUROC: {agg['per_class_auroc_mean']}")

    family_split_mlp_f1, family_split_contrastive_f1 = _read_live_family_split_refs()
    refs = {}
    if family_split_mlp_f1 is not None:
        refs["Family-split MLP floor (live measured)"] = family_split_mlp_f1
    if family_split_contrastive_f1 is not None:
        refs["Family-split contrastive proj (live measured)"] = family_split_contrastive_f1

    print(
        f"\nVs. cross-family baselines (clan-holdout MLP F1 = {agg['mlp_macro_f1_mean']:.3f}):"
    )
    if not refs:
        print("  (no live family-split reference files found — comparison skipped)")
    for name, val in refs.items():
        delta_f1 = agg["mlp_macro_f1_mean"] - val
        symbol = "✓" if delta_f1 > 0.02 else ("~" if delta_f1 > -0.02 else "✗")
        print(f"  {symbol} vs {name} ({val:.3f}): Δ = {delta_f1:+.3f}")

    print("\nInterpretation:")
    f1 = agg["mlp_macro_f1_mean"]
    maj = agg["majority_macro_f1_mean"]
    if family_split_contrastive_f1 is None:
        print("  (no live family-split contrastive reference — qualitative read only)")
        print(f"  Clan-holdout MLP F1 = {f1:.3f}, majority baseline = {maj:.3f}")
    elif f1 > family_split_contrastive_f1 + 0.03:
        print("  → Clan-holdout F1 exceeds cross-family baseline.")
        print("    ESM-2 delta signal generalises to unseen protein clans.")
        print("    Mechanism encoding is not purely clan-level memorisation.")
    elif f1 > maj + 0.05:
        print(f"  → Clan-holdout F1 ({f1:.3f}) is above majority ({maj:.3f})")
        print(f"    but below cross-family baseline ({family_split_contrastive_f1:.3f}).")
        print("    Partial generalisation: some real signal, some memorisation.")
    else:
        print(f"  → Clan-holdout F1 ({f1:.3f}) ≈ majority baseline ({maj:.3f}).")
        print("    Performance collapses on unseen clans.")
        print("    All apparent mechanism signal is clan/family memorisation.")
        print("    This is the definitive negative result.")

    if ci is not None and not ci["macro_f1"].get("ci_suppressed", False):
        print(
            f"\nClan-resampled CI: macro_f1 = {ci['macro_f1']['point']:.3f} "
            f"[{ci['macro_f1']['ci_low']}, {ci['macro_f1']['ci_high']}] "
            f"(n_clusters={ci['macro_f1']['n_clusters']})"
        )
    elif ci is not None:
        interval = ci["macro_f1"]
        print(
            "\nClan-resampled CI: not reported — "
            f"{interval['n_resamples']}/{interval['n_resamples_total']} resamples "
            f"were scorable ({interval.get('reason', 'no reason recorded')})"
        )

    results = {
        "description": (
            "Leave-one-clan-out evaluation. Train on all variants except clan X, "
            "test on clan X. Tests whether ESM-2 delta mechanism signal generalises "
            "to completely unseen protein clans (lookup vs real signal)."
        ),
        "seed": args.seed,
        "n_boot": args.n_boot,
        "aggregate": agg,
        "per_clan": clan_results,
        "references": {
            "family_split_mlp_f1": family_split_mlp_f1,
            "family_split_contrastive_f1": family_split_contrastive_f1,
        },
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"clan_holdout_results_seed{args.seed}.json")
    write_result_json(out_path, results, seeds=[args.seed], indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
