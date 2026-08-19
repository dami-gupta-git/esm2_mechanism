"""Exploratory transfer of full-delta probes across disjoint group halves.

The linear and gradient-boosted probes both consume the complete delta vector;
this analysis does not estimate or compare a single biological direction.
"""

import numpy as np
import functools

from esm2_mech.utils.data import load_pfam_map
from esm2_mech.utils.io import write_result_json
from esm2_mech.utils.paths import (
    GEOMETRY_RESULTS_DIR,
    TRANSFER_CONTRAST_JSON,
    PFAM_JSON,
)
from esm2_mech.experiments.mechanism.loaders import load_merged
from esm2_mech.utils.constants import GOF, N_SEEDS
from esm2_mech.utils.metrics import mean_std_n
from esm2_mech.utils.probes import auroc_for_clf
from esm2_mech.utils.data import embedding_fingerprint
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

STABILITY_DATASETS = {
    "none": None,
    "tsuboyama": "tsuboyama",
}
DEFAULT_STABILITY_DATASET = "none"

GEOMETRY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _make_clf(kind, seed):
    if kind == "linear":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    if kind == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, random_state=seed
        )
    raise ValueError(kind)


def transfer_test(delta, y, groups, kind="linear", n_partitions=10, seed=0, min_pos=5):
    """Fit a probe on half the groups, score the disjoint half. Returns transfer
    AUROC (mean over both directions and partitions) and a group-disjoint
    five-fold reference. `kind` in {linear, gbm}. `groups` (protein/family id per row) is
    respected by both scores — a plain label-stratified split for the reference
    would let rows from the same protein/family land on both sides of
    a fold, leaking the same signal the transfer score is designed to rule out."""
    from sklearn.preprocessing import StandardScaler

    from esm2_mech.utils.splits import family_split_indices

    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    uniq = np.array(sorted(set(groups.tolist())))
    rng = np.random.RandomState(seed)

    transfer = []
    for _ in range(n_partitions):
        gshuf = uniq.copy()
        rng.shuffle(gshuf)
        half = set(gshuf[: len(gshuf) // 2])
        a = np.array([g in half for g in groups])
        b = ~a
        for tr, te in [(a, b), (b, a)]:
            if y[tr].sum() < min_pos or (1 - y[tr]).sum() < min_pos:
                continue
            if y[te].sum() < min_pos or (1 - y[te]).sum() < min_pos:
                continue
            sc = StandardScaler().fit(delta[tr])
            Xtr = sc.transform(delta[tr])
            Xte = sc.transform(delta[te])
            clf = _make_clf(kind, seed).fit(Xtr, y[tr])
            transfer.append(auroc_for_clf(clf, Xte, y[te]))

    group_cv = []
    minority = int(min(y.sum(), (1 - y).sum()))
    n_splits = min(5, minority)
    if n_splits >= 2:
        for tr, te in family_split_indices(groups, n_splits, seed):
            if y[tr].sum() < min_pos or (1 - y[tr]).sum() < min_pos:
                continue
            if y[te].sum() < min_pos or (1 - y[te]).sum() < min_pos:
                continue
            sc = StandardScaler().fit(delta[tr])
            Xtr = sc.transform(delta[tr])
            Xte = sc.transform(delta[te])
            clf = _make_clf(kind, seed).fit(Xtr, y[tr])
            group_cv.append(auroc_for_clf(clf, Xte, y[te]))

    def agg(v):
        mean, std, n = mean_std_n(v)
        return {"mean": mean, "std": std, "n": n}

    return {"transfer_auroc": agg(transfer), "group_cv_auroc": agg(group_cv)}


def load_pathogenicity(inputs, pfam):
    groups = np.array([(pfam.get(gene) or "NA") for gene in inputs.genes])
    m = groups != "NA"
    return inputs.delta[m], inputs.labels[m], groups[m]


def load_stability(stability_dataset=DEFAULT_STABILITY_DATASET):
    if stability_dataset not in STABILITY_DATASETS:
        raise ValueError(f"unknown stability dataset {stability_dataset!r}")
    if stability_dataset == "none":
        return None
    stability = load_stability_inputs()
    ddg = np.asarray(stability.ddg, dtype=float)
    groups = stability.proteins
    finite = np.isfinite(ddg)
    n_dropped = int((~finite).sum())
    if n_dropped:
        print(f"  Dropped {n_dropped}/{len(ddg)} variants with non-finite ddG")
    delta = stability.delta_mean[finite]
    ddg = ddg[finite]
    groups = groups[finite]
    y = (ddg > np.median(ddg)).astype(int)  # median split
    provenance = {
        "variant_fingerprint": stability_variant_fingerprint(stability.variants),
        "delta_embedding_fingerprint": embedding_fingerprint(stability.delta_mean),
    }
    return (delta, y, groups), provenance


def load_mechanism_gof():
    pfam = load_pfam_map(PFAM_JSON)
    dm, labels, genes = load_merged(pfam)
    groups = np.array([(pfam.get(g) or "NA") for g in genes])
    y = (np.asarray(labels) == GOF).astype(int)
    m = groups != "NA"
    provenance = mechanism_geometry_provenance(dm, labels, genes, pfam)
    return (dm[m], y[m], groups[m]), provenance


def run(n_seeds=N_SEEDS, stability_dataset=DEFAULT_STABILITY_DATASET):
    tasks = {}
    pfam = load_pfam_map(PFAM_JSON)
    pathogenicity_inputs = load_pathogenicity_geometry_inputs()
    provenance = {
        "pathogenicity": pathogenicity_geometry_provenance(pathogenicity_inputs, pfam)
    }
    print("Loading tasks...")
    tasks["pathogenicity (path vs benign, family-split)"] = load_pathogenicity(
        pathogenicity_inputs, pfam
    )
    mechanism_task, mechanism_provenance = load_mechanism_gof()
    tasks["mechanism (GOF vs rest, family-split)"] = mechanism_task
    provenance["mechanism"] = mechanism_provenance
    stab = load_stability(stability_dataset=stability_dataset)
    if stab is not None:
        stability_task, stability_provenance = stab
        tasks["stability (ΔΔG>median, protein-split)"] = stability_task
        provenance["stability"] = stability_provenance
    else:
        print(f"  stability: dataset='{stability_dataset}' not run — skipping")

    results = {}
    print("\n" + "=" * 86)
    print(f"{'task':42s} {'probe':7s} {'group CV':>12s} {'half-transfer':>13s}")
    print("=" * 86)
    for name, (delta, y, groups) in tasks.items():
        results[name] = {}
        npart = 5 if len(y) > 5000 else 10
        for kind in ("linear", "gbm"):
            transfer_vals, group_cv_vals = [], []
            for seed in range(n_seeds):
                r = transfer_test(
                    delta, y, groups, kind=kind, n_partitions=npart, seed=seed
                )
                transfer_vals.append(r["transfer_auroc"]["mean"])
                group_cv_vals.append(r["group_cv_auroc"]["mean"])
            tm, ts, tn = mean_std_n(transfer_vals)
            gm, gs, gn = mean_std_n(group_cv_vals)
            results[name][kind] = {
                "half_group_transfer_auroc": {"mean": tm, "std": ts, "n": tn},
                "group_cv_auroc": {"mean": gm, "std": gs, "n": gn},
            }
            print(
                f"{name:42s} {kind:7s} {gm:.3f}±{gs:.3f}  "
                f"{tm:.3f}±{ts:.3f}  (seeds={n_seeds})"
            )

    result = {
        "tasks": results,
        "analysis_status": "exploratory",
        "interpretation_note": (
            "Both probes use the full delta vector. Group CV and half-group "
            "transfer use different training-set sizes, so their difference is "
            "not an isolated estimate of transfer failure."
        ),
        "input_provenance": provenance,
    }
    write_result_json(TRANSFER_CONTRAST_JSON, result, seeds=list(range(n_seeds)))
    print(f"\nResults -> {TRANSFER_CONTRAST_JSON}")
    return result


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=N_SEEDS, help="number of seeds (>=1)")
    ap.add_argument(
        "--stability-dataset",
        choices=list(STABILITY_DATASETS),
        default=DEFAULT_STABILITY_DATASET,
        help="dataset for the stability transfer arm (default: none = skip)",
    )
    args = ap.parse_args()
    run(n_seeds=args.seeds, stability_dataset=args.stability_dataset)
    print("\nRead: group CV and half-group transfer are both group-disjoint.")
    print(
        "The half-transfer arm trains on fewer groups, so the two scores are descriptive."
    )


if __name__ == "__main__":
    main()
