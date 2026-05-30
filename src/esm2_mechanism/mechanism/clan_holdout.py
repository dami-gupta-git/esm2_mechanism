"""
Clan-level holdout evaluation: does ESM-2 mechanism signal generalise to
completely unseen protein clans, or is it clan-level memorisation?

Design:
  - Load merged delta embeddings (cached .npy, no GPU needed)
  - Map each gene to its Pfam clan (one level above family)
  - Leave-one-clan-out: train contrastive projection head + MLP probe on
    all variants EXCEPT clan X, evaluate on clan X
  - Also run standard family-split as reference
  - Key question: does performance on fully held-out clans match or beat
    the cross-family family-split floor (~0.40 F1)?

Interpretation:
  - If clan-holdout F1 ≈ family-split F1 → signal generalises, not memorisation
  - If clan-holdout F1 ≈ chance → all signal is clan/family lookup

Usage:
    python clan_holdout.py --data_dir ../data --emb_dir ../data/embeddings \
        --clan_file /tmp/pfam_clans.tsv.gz --out_dir ../results/20260524_baseline_run/run_0

Output: clan_holdout_results_seed0.json
"""

import argparse
import json
import os
import subprocess
import warnings
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import functools

print = functools.partial(print, flush=True)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(data_dir, emb_dir):
    mv_path = os.path.join(emb_dir, "merged_valid_variants.json")
    if not os.path.exists(mv_path):
        mv_path = os.path.join(data_dir, "merged_valid_variants.json")
    with open(mv_path) as f:
        variants = json.load(f)

    labels = np.array([v["label_3class"] for v in variants])
    genes = np.array([v["gene"] for v in variants])

    print(f"Loaded {len(variants)} variants, {len(set(genes))} genes")
    print(f"Class distribution: {dict(Counter(labels))}")

    wt_mean = np.load(os.path.join(emb_dir, "merged_embeddings_wt_mean.npy"))
    mut_mean = np.load(os.path.join(emb_dir, "merged_embeddings_mut_mean.npy"))
    delta = (mut_mean - wt_mean).astype(np.float32)
    print(f"Delta embeddings: {delta.shape}")
    return variants, labels, genes, delta


def load_pfam(data_dir):
    with open(os.path.join(data_dir, "pfam_families.json")) as f:
        return json.load(f)  # gene -> pfam_acc


def load_clan_map(clan_file):
    """Parse Pfam-A.clans.tsv.gz -> {pfam_acc: (clan_id, clan_name)}."""
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


# ---------------------------------------------------------------------------
# Probe: MLP with same architecture as result_7
# ---------------------------------------------------------------------------


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
    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    classes = list(clf.classes_)

    results = {
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "n_test": int(len(y_test)),
        "class_dist_test": {str(k): int(v) for k, v in Counter(y_test).items()},
    }

    all_classes = list(le.classes_)
    for ai, cls in enumerate(all_classes):
        ci = le.transform([cls])[0]
        y_bin = (y_test == ci).astype(int)
        if y_bin.sum() > 0 and (1 - y_bin).sum() > 0 and ci in classes:
            pi = classes.index(ci)
            if proba[:, pi].std() > 0:
                results[f"auroc_{cls}"] = float(roc_auc_score(y_bin, proba[:, pi]))
    return results


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_clan_holdout(delta, labels, genes, gene_clan, clan_names, le, seed=42):
    """
    Leave-one-clan-out: for each qualifying clan, train on everything else,
    test on the held-out clan.
    """
    y = le.transform(labels)
    all_classes = list(le.classes_)

    # Identify qualifying clans
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

    clan_results = []

    for q in qualifying:
        clan = q["clan"]
        test_idx = np.array(q["idxs"])
        train_mask = np.ones(len(delta), dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]

        y_tr = y[train_idx]
        y_te = y[test_idx]

        # Skip if test set missing a class that appears in train
        if len(set(y_te)) < 2:
            print(f"  {clan}: skipped (single class in test)")
            continue

        # Normalise on train stats
        mu = delta[train_idx].mean(0)
        std = delta[train_idx].std(0) + 1e-8
        X_tr = (delta[train_idx] - mu) / std
        X_te = (delta[test_idx] - mu) / std

        print(
            f"\n  Clan {clan} ({q['name']})  "
            f"train={len(train_idx)} test={len(test_idx)} "
            f"test_mechs={dict(Counter(labels[test_idx]))}"
        )

        # MLP probe
        try:
            clf = train_mlp(X_tr, y_tr, seed=seed)
            mlp_res = evaluate_probe(clf, X_te, y_te, le)
        except Exception as e:
            print(f"    MLP failed: {e}")
            mlp_res = {"error": str(e)}

        # k-NN baseline (no learning)
        try:
            k = min(10, len(X_tr) - 1)
            knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
            knn.fit(X_tr, y_tr)
            pred_knn = knn.predict(X_te)
            knn_f1 = float(f1_score(y_te, pred_knn, average="macro", zero_division=0))
        except Exception as e:
            knn_f1 = float("nan")

        # Majority baseline
        majority_class = Counter(y_tr).most_common(1)[0][0]
        maj_pred = np.full(len(y_te), majority_class)
        maj_f1 = float(f1_score(y_te, maj_pred, average="macro", zero_division=0))

        print(
            f"    MLP F1={mlp_res.get('macro_f1', float('nan')):.3f}  "
            f"kNN F1={knn_f1:.3f}  "
            f"majority F1={maj_f1:.3f}  "
            f"GOF={mlp_res.get('auroc_GOF', float('nan')):.3f}  "
            f"DN={mlp_res.get('auroc_DN', float('nan')):.3f}  "
            f"LOF={mlp_res.get('auroc_LOF', float('nan')):.3f}"
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
                "knn_macro_f1": knn_f1,
                "majority_macro_f1": maj_f1,
            }
        )

    return clan_results, qualifying


def aggregate(clan_results):
    """Weighted and unweighted aggregates across qualifying clans."""
    mlp_f1s = [
        r["mlp"].get("macro_f1", float("nan"))
        for r in clan_results
        if "error" not in r["mlp"]
    ]
    knn_f1s = [r["knn_macro_f1"] for r in clan_results if "error" not in r["mlp"]]
    maj_f1s = [r["majority_macro_f1"] for r in clan_results if "error" not in r["mlp"]]
    weights = [r["n_test"] for r in clan_results if "error" not in r["mlp"]]

    per_class = defaultdict(list)
    for r in clan_results:
        for cls in ("GOF", "DN", "LOF"):
            v = r["mlp"].get(f"auroc_{cls}")
            if v is not None and not np.isnan(v):
                per_class[cls].append(v)

    def wmean(vals, ws):
        if not vals:
            return float("nan")
        ws = np.array(ws, dtype=float)
        return float(np.average(vals, weights=ws))

    return {
        "n_clans": len(clan_results),
        "mlp_macro_f1_mean": float(np.nanmean(mlp_f1s)) if mlp_f1s else float("nan"),
        "mlp_macro_f1_std": float(np.nanstd(mlp_f1s)) if mlp_f1s else float("nan"),
        "mlp_macro_f1_weighted": wmean(mlp_f1s, weights),
        "knn_macro_f1_mean": float(np.nanmean(knn_f1s)) if knn_f1s else float("nan"),
        "majority_macro_f1_mean": (
            float(np.nanmean(maj_f1s)) if maj_f1s else float("nan")
        ),
        "per_class_auroc_mean": {
            cls: float(np.mean(vs)) for cls, vs in per_class.items() if vs
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--emb_dir", default="../data/embeddings")
    parser.add_argument(
        "--clan_file", required=True, help="Path to Pfam-A.clans.tsv.gz"
    )
    parser.add_argument("--out_dir", default="../results/20260524_baseline_run/run_0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=== Loading data ===")
    variants, labels, genes, delta = load_data(args.data_dir, args.emb_dir)
    pfam_map = load_pfam(args.data_dir)
    clan_map, clan_names = load_clan_map(args.clan_file)

    # Build gene -> clan
    gene_clan = {}
    for gene, pfam_acc in pfam_map.items():
        if pfam_acc in clan_map:
            gene_clan[gene] = clan_map[pfam_acc]
    print(f"Genes with clan assignment: {len(gene_clan)}/{len(pfam_map)}")

    le = LabelEncoder()
    le.fit(["GOF", "DN", "LOF"])
    print(f"Classes: {list(le.classes_)}")

    print("\n=== Clan-level holdout evaluation ===")
    clan_results, qualifying = run_clan_holdout(
        delta, labels, genes, gene_clan, clan_names, le, seed=args.seed
    )

    agg = aggregate(clan_results)

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    print(f"  Clans evaluated: {agg['n_clans']}")
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

    # Reference numbers from prior results
    refs = {
        "Family-split MLP floor (result_7)": 0.352,
        "Family-split contrastive proj (result_9)": 0.387,
    }
    print(
        f"\nVs. cross-family baselines (clan-holdout MLP F1 = {agg['mlp_macro_f1_mean']:.3f}):"
    )
    for name, val in refs.items():
        delta_f1 = agg["mlp_macro_f1_mean"] - val
        symbol = "✓" if delta_f1 > 0.02 else ("~" if delta_f1 > -0.02 else "✗")
        print(f"  {symbol} vs {name} ({val:.3f}): Δ = {delta_f1:+.3f}")

    print("\nInterpretation:")
    f1 = agg["mlp_macro_f1_mean"]
    maj = agg["majority_macro_f1_mean"]
    ref = 0.387
    if f1 > ref + 0.03:
        print("  → Clan-holdout F1 exceeds cross-family baseline.")
        print("    ESM-2 delta signal generalises to unseen protein clans.")
        print("    Mechanism encoding is not purely clan-level memorisation.")
    elif f1 > maj + 0.05:
        print(f"  → Clan-holdout F1 ({f1:.3f}) is above majority ({maj:.3f})")
        print(f"    but below cross-family baseline ({ref:.3f}).")
        print("    Partial generalisation: some real signal, some memorisation.")
    else:
        print(f"  → Clan-holdout F1 ({f1:.3f}) ≈ majority baseline ({maj:.3f}).")
        print("    Performance collapses on unseen clans.")
        print("    All apparent mechanism signal is clan/family memorisation.")
        print("    This is the definitive negative result.")

    results = {
        "description": (
            "Leave-one-clan-out evaluation. Train on all variants except clan X, "
            "test on clan X. Tests whether ESM-2 delta mechanism signal generalises "
            "to completely unseen protein clans (lookup vs real signal)."
        ),
        "seed": args.seed,
        "aggregate": agg,
        "per_clan": clan_results,
        "references": {
            "family_split_mlp_f1_result7": 0.352,
            "family_split_contrastive_f1_result9": 0.387,
        },
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"clan_holdout_results_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
