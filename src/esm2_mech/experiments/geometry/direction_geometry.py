"""Exploratory ablation and cross-family transfer of pathogenicity directions."""

import numpy as np
import functools

from esm2_mech.utils.constants import N_FOLDS, N_SEEDS
from esm2_mech.utils.data import load_pfam_map
from esm2_mech.utils.io import write_result_json
from esm2_mech.utils.paths import (
    GEOMETRY_RESULTS_DIR,
    DIRECTION_GEOMETRY_JSON,
    PFAM_JSON,
)
from esm2_mech.utils.seed_aggregation import (
    aggregate_paired_seed_difference,
    aggregate_result_contract,
    aggregate_seed_values,
    make_seed_record,
    read_seed_point_estimate,
    seed_count,
)
from esm2_mech.utils.metrics import within_seed_summary
from esm2_mech.utils.probes import auroc_for_clf
from esm2_mech.utils.splits import family_split_cv
from esm2_mech.experiments.geometry.data import (
    load_pathogenicity_geometry_inputs,
    pathogenicity_geometry_provenance,
)

print = functools.partial(print, flush=True)

GEOMETRY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _aggregate_within_seed(seeds, summaries_by_seed):
    return aggregate_seed_values(
        seeds,
        [make_seed_record(seed, summaries_by_seed[seed]["mean"]) for seed in seeds],
    ).to_dict()


def load(pfam_map):
    inputs = load_pathogenicity_geometry_inputs()
    fam = np.array([(pfam_map.get(gene) or "NA") for gene in inputs.genes])
    print(
        f"Loaded {len(inputs.labels)} variants, {len(set(inputs.genes))} genes, "
        f"{int(inputs.labels.sum())} path / "
        f"{int((1 - inputs.labels).sum())} benign, "
        f"{len(set(fam[fam != 'NA']))} Pfam families"
    )
    return inputs, fam


def fit_direction(X, y, seed=0):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(X, y)
    w = clf.coef_.ravel()
    w_unit = w / (np.linalg.norm(w) + 1e-12)
    return w_unit, clf


def original_space_direction(scaled_direction, scaler):
    """Map standardized-feature coefficients back to the original feature space."""
    direction = np.asarray(scaled_direction) / scaler.scale_
    return direction / (np.linalg.norm(direction) + 1e-12)


def probe1_direction_ablation(delta, y, genes, pfam_map, k_max=5, seeds=range(N_SEEDS)):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    print("\n" + "=" * 60)
    print("ITERATIVE LINEAR-DIRECTION ABLATION (family-split, exploratory)")
    print("=" * 60)

    requested_seeds = tuple(seeds)
    decay = {seed: {k: [] for k in range(k_max + 1)} for seed in requested_seeds}
    for seed in requested_seeds:
        fs = family_split_cv(genes, pfam_map, seed=seed)
        for tr, te in fs:
            sc = StandardScaler().fit(delta[tr])
            Xtr = sc.transform(delta[tr])
            Xte = sc.transform(delta[te])
            ytr, yte = y[tr], y[te]
            if len(set(ytr)) < 2 or len(set(yte)) < 2:
                continue

            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(
                Xtr, ytr
            )
            decay[seed][0].append(auroc_for_clf(clf, Xte, yte))

            Rtr, Rte = Xtr.copy(), Xte.copy()
            for k in range(1, k_max + 1):
                ck = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(
                    Rtr, ytr
                )
                w = ck.coef_.ravel()
                w /= np.linalg.norm(w) + 1e-12
                Rtr = Rtr - np.outer(Rtr @ w, w)
                Rte = Rte - np.outer(Rte @ w, w)
                ck2 = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(
                    Rtr, ytr
                )
                decay[seed][k].append(auroc_for_clf(ck2, Rte, yte))

    per_seed = {
        seed: {
            k: within_seed_summary(decay[seed][k], N_FOLDS, "fold_std", "held_out_fold")
            for k in range(k_max + 1)
        }
        for seed in requested_seeds
    }
    aggregates = {
        k: _aggregate_within_seed(
            requested_seeds,
            {seed: per_seed[seed][k] for seed in requested_seeds},
        )
        for k in range(k_max + 1)
    }
    full_records = [
        make_seed_record(seed, per_seed[seed][0]["mean"]) for seed in requested_seeds
    ]
    paired_changes = {}
    for k in range(1, k_max + 1):
        residual_records = [
            make_seed_record(seed, per_seed[seed][k]["mean"])
            for seed in requested_seeds
        ]
        paired_changes[k] = aggregate_paired_seed_difference(
            requested_seeds, full_records, residual_records
        ).to_dict()

    out = {
        "full_linear_auroc": aggregates[0],
        "residual_auroc_after_removing_k": aggregates,
        "paired_full_minus_residual": paired_changes,
        "per_seed_fold_summaries": per_seed,
        "interpretation_note": (
            "This is an iterative discriminative-direction ablation. A binary "
            "linear classifier's own decision score is necessarily one-dimensional, "
            "so it is not used as evidence that the biological signal is rank one."
        ),
    }
    full = out["full_linear_auroc"]
    print(f"  full linear AUROC = {_show_seed_summary(full)}")
    for k in range(k_max + 1):
        summary = out["residual_auroc_after_removing_k"][k]
        print(
            f"  AUROC after removing {k} direction(s) = "
            f"{_show_seed_summary(summary)}"
        )
    return out


def probe2_family_transfer(delta, y, genes, fam, n_partitions=10, seeds=(0,)):
    from sklearn.preprocessing import StandardScaler

    print("\n" + "=" * 60)
    print("CROSS-FAMILY DIRECTION TRANSFER (exploratory)")
    print("=" * 60)

    mask = fam != "NA"
    X = delta[mask]
    yy = y[mask]
    ff = fam[mask]
    families = np.array(sorted(set(ff.tolist())))

    requested_seeds = tuple(seeds)
    per_seed_values = {
        seed: {
            "cosine_observed": [],
            "cosine_null_shuffled": [],
            "transfer_auroc_AtoB": [],
        }
        for seed in requested_seeds
    }
    for seed in requested_seeds:
        rng = np.random.RandomState(seed)
        for _ in range(n_partitions):
            fam_shuf = families.copy()
            rng.shuffle(fam_shuf)
            half = set(fam_shuf[: len(fam_shuf) // 2])
            a = np.array([f in half for f in ff])
            b = ~a
            if (
                yy[a].sum() < 5
                or yy[b].sum() < 5
                or (1 - yy[a]).sum() < 5
                or (1 - yy[b]).sum() < 5
            ):
                continue
            sc_a = StandardScaler().fit(X[a])
            sc_b = StandardScaler().fit(X[b])
            Xa_own = sc_a.transform(X[a])
            Xb_own = sc_b.transform(X[b])
            wA, clfA = fit_direction(Xa_own, yy[a], seed=seed)
            wB, clfB = fit_direction(Xb_own, yy[b], seed=seed)
            wA_orig = original_space_direction(wA, sc_a)
            wB_orig = original_space_direction(wB, sc_b)
            per_seed_values[seed]["cosine_observed"].append(
                float(np.dot(wA_orig, wB_orig))
            )
            Xb_via_a = sc_a.transform(X[b])
            per_seed_values[seed]["transfer_auroc_AtoB"].append(
                auroc_for_clf(clfA, Xb_via_a, yy[b])
            )

            yA_s = yy[a].copy()
            rng.shuffle(yA_s)
            yB_s = yy[b].copy()
            rng.shuffle(yB_s)
            wAn, _ = fit_direction(Xa_own, yA_s, seed=seed)
            wBn, _ = fit_direction(Xb_own, yB_s, seed=seed)
            wAn_orig = original_space_direction(wAn, sc_a)
            wBn_orig = original_space_direction(wBn, sc_b)
            per_seed_values[seed]["cosine_null_shuffled"].append(
                float(np.dot(wAn_orig, wBn_orig))
            )

    per_seed = {
        seed: {
            metric: within_seed_summary(
                values, n_partitions, "partition_std", "random_family_partition"
            )
            for metric, values in metrics.items()
        }
        for seed, metrics in per_seed_values.items()
    }
    aggregates = {
        metric: _aggregate_within_seed(
            requested_seeds,
            {seed: per_seed[seed][metric] for seed in requested_seeds},
        )
        for metric in (
            "cosine_observed",
            "cosine_null_shuffled",
            "transfer_auroc_AtoB",
        )
    }

    out = {
        "n_partitions_per_seed": n_partitions,
        **aggregates,
        "per_seed_partition_summaries": per_seed,
    }
    observed = out["cosine_observed"]
    null = out["cosine_null_shuffled"]
    transfer_summary = out["transfer_auroc_AtoB"]
    print(f"  cosine(w_A, w_B) observed = {_show_seed_summary(observed)}")
    print(f"  cosine null (shuffled y) = {_show_seed_summary(null)}")
    print(
        f"  transfer AUROC (A's direction -> B) = "
        f"{_show_seed_summary(transfer_summary)}"
    )
    return out


def run(n_seeds=N_SEEDS):
    seeds = tuple(range(n_seeds))
    pfam_map = load_pfam_map(PFAM_JSON)
    inputs, fam = load(pfam_map)

    ablation = probe1_direction_ablation(
        inputs.delta.astype(np.float64),
        inputs.labels,
        inputs.genes,
        pfam_map,
        seeds=seeds,
    )
    family_transfer = probe2_family_transfer(
        inputs.delta.astype(np.float64),
        inputs.labels,
        inputs.genes,
        fam,
        seeds=seeds,
    )

    result = {
        **aggregate_result_contract(),
        "iterative_direction_ablation": ablation,
        "cross_family_direction_transfer": family_transfer,
        "n_variants": int(len(inputs.labels)),
        "analysis_status": "exploratory",
        "input_provenance": pathogenicity_geometry_provenance(inputs, pfam_map),
    }
    write_result_json(DIRECTION_GEOMETRY_JSON, result, seeds=list(seeds))
    print(f"\nResults -> {DIRECTION_GEOMETRY_JSON}")
    return result


def _show_seed_summary(summary):
    metric = read_seed_point_estimate(summary)
    if not metric.available:
        return f"unavailable ({metric.message})"
    if metric.spread is None:
        return f"{metric.value:.3f} (seed spread unavailable)"
    return f"{metric.value:.3f} ± {metric.spread:.3f} seed SD"


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seeds", type=seed_count, default=N_SEEDS, help="number of seeds (>=1)"
    )
    args = ap.parse_args()
    result = run(n_seeds=args.seeds)
    ablation = result["iterative_direction_ablation"]
    family_transfer = result["cross_family_direction_transfer"]

    print("\n" + "=" * 60)
    print("READ")
    print("=" * 60)
    print(
        f"  Full linear AUROC is {_show_seed_summary(ablation['full_linear_auroc'])}."
    )
    print(
        "  After removing one fitted direction, residual AUROC is "
        f"{_show_seed_summary(ablation['residual_auroc_after_removing_k'][1])}."
    )
    print(
        "  Cross-family direction cosine = "
        f"{_show_seed_summary(family_transfer['cosine_observed'])}; null = "
        f"{_show_seed_summary(family_transfer['cosine_null_shuffled'])}; "
        "transfer AUROC = "
        f"{_show_seed_summary(family_transfer['transfer_auroc_AtoB'])}."
    )


if __name__ == "__main__":
    main()
