"""Transfer contrast: does a linear direction fit on one group of proteins/families
transfer to a disjoint group, for pathogenicity, stability, and mechanism?
"""

import json
import numpy as np
import functools

print = functools.partial(print, flush=True)

from esm2_mech.utils.io import atomic_write_json
from esm2_mech.utils.paths import (
    GEOMETRY_RESULTS_DIR,
    TRANSFER_CONTRAST_JSON,
    MEGASCALE_TSUBOYAMA_VARIANTS_JSON,
    PATH_EMB_WT_MEAN,
    PATH_EMB_MUT_MEAN,
    PATHOGENICITY_CANONICAL_VARIANTS_JSON,
    PFAM_JSON,
    MEGASCALE_EMB_WT_MEAN,
    MEGASCALE_EMB_MUT_MEAN,
)

STABILITY_DATASETS = {
    "none": None,
    "tsuboyama": (
        MEGASCALE_TSUBOYAMA_VARIANTS_JSON,
        MEGASCALE_EMB_WT_MEAN,
        MEGASCALE_EMB_MUT_MEAN,
    ),
}
DEFAULT_STABILITY_DATASET = "none"
from esm2_mech.experiments.mechanism.loaders import load_mechanism_variants
from esm2_mech.utils.constants import GOF, N_SEEDS
from esm2_mech.utils.metrics import mean_std_n
from esm2_mech.utils.probes import auroc_for_clf

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
    AUROC (mean over both directions and partitions) and a pooled random-split
    reference. `kind` in {linear, gbm}."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y).astype(int)
    # Scaler fit on all rows deliberately — shared coordinate frame for direction transfer.
    Xs = StandardScaler().fit_transform(delta)
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
            clf = _make_clf(kind, seed).fit(Xs[tr], y[tr])
            transfer.append(auroc_for_clf(clf, Xs[te], y[te]))

    # Cap n_splits at minority-class count to avoid StratifiedKFold raising.
    pooled = []
    minority = int(min(y.sum(), (1 - y).sum()))
    n_splits = min(5, minority)
    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in skf.split(Xs, y):
            clf = _make_clf(kind, seed).fit(Xs[tr], y[tr])
            pooled.append(auroc_for_clf(clf, Xs[te], y[te]))

    def agg(v):
        mean, std, n = mean_std_n(v)
        return (mean, std, n)

    return {"transfer_auroc": agg(transfer), "pooled_auroc": agg(pooled)}



def _pathogenicity_label(label):
    """Map a canonical-set label to 1 (pathogenic) / 0 (benign); never a catch-all."""
    if label == "pathogenic":
        return 1
    if label == "benign":
        return 0
    raise ValueError(f"unexpected pathogenicity label {label!r} (expected 'pathogenic'/'benign')")


def load_pathogenicity():
    with open(PATHOGENICITY_CANONICAL_VARIANTS_JSON) as _f:
        v = json.load(_f)
    delta = np.load(PATH_EMB_MUT_MEAN) - np.load(PATH_EMB_WT_MEAN)
    if len(v) != delta.shape[0]:
        raise ValueError(
            f"variant/embedding row mismatch: {len(v)} variants vs "
            f"{delta.shape[0]} embedding rows — canonical file is not row-aligned."
        )
    with open(PFAM_JSON) as _f:
        pfam = json.load(_f)
    genes = [x["gene"] for x in v]
    groups = np.array([(pfam.get(g) or "NA") for g in genes])
    y = np.array([_pathogenicity_label(x["label"]) for x in v])
    m = groups != "NA"
    return delta[m], y[m], groups[m]


def load_stability(stability_dataset=DEFAULT_STABILITY_DATASET):
    cfg = STABILITY_DATASETS.get(stability_dataset)
    if cfg is None:
        return None
    variants_json, wt_emb, mut_emb = cfg
    if not (variants_json.exists() and wt_emb.exists() and mut_emb.exists()):
        return None
    with open(variants_json) as _f:
        v = json.load(_f)
    delta = np.load(mut_emb) - np.load(wt_emb)
    if len(v) != len(delta):
        raise ValueError(
            f"row mismatch in {variants_json.name}: {len(delta)} embedding rows vs "
            f"{len(v)} variants — not row-aligned."
        )
    ddg = np.array(
        [x["ddg"] if x["ddg"] is not None else np.nan for x in v], dtype=float
    )
    groups = np.array([x["protein"] for x in v])
    finite = np.isfinite(ddg)
    n_dropped = int((~finite).sum())
    if n_dropped:
        print(f"  Dropped {n_dropped}/{len(ddg)} variants with non-finite ddG")
    delta, ddg, groups = delta[finite], ddg[finite], groups[finite]
    y = (ddg > np.median(ddg)).astype(int)  # median split
    return delta, y, groups


def load_mechanism_gof():
    with open(PFAM_JSON) as _f:
        pfam = json.load(_f)
    dm, _dp, labels, genes = load_mechanism_variants(pfam)
    groups = np.array([(pfam.get(g) or "NA") for g in genes])
    y = (np.asarray(labels) == GOF).astype(int)
    m = groups != "NA"
    return dm[m], y[m], groups[m]


def run(n_seeds=N_SEEDS, stability_dataset=DEFAULT_STABILITY_DATASET):
    tasks = {}
    print("Loading tasks...")
    tasks["pathogenicity (path vs benign, family-split)"] = load_pathogenicity()
    tasks["mechanism (GOF vs rest, family-split)"] = load_mechanism_gof()
    stab = load_stability(stability_dataset=stability_dataset)
    if stab is not None:
        tasks["stability (ΔΔG>median, protein-split)"] = stab
    else:
        print(
            f"  stability: dataset='{stability_dataset}' not run — skipping"
        )

    results = {}
    print("\n" + "=" * 86)
    print(f"{'task':42s} {'probe':7s} {'pooled':>12s} {'transfer':>13s}")
    print("=" * 86)
    for name, (delta, y, groups) in tasks.items():
        results[name] = {}
        npart = 5 if len(y) > 5000 else 10
        for kind in ("linear", "gbm"):
            transfer_vals, pooled_vals = [], []
            for seed in range(n_seeds):
                r = transfer_test(delta, y, groups, kind=kind, n_partitions=npart, seed=seed)
                transfer_vals.append(r["transfer_auroc"][0])
                pooled_vals.append(r["pooled_auroc"][0])
            tm, ts, tn = mean_std_n(transfer_vals)
            pm, ps, _ = mean_std_n(pooled_vals)
            results[name][kind] = {
                "transfer_auroc": (tm, ts, tn),
                "pooled_auroc": (pm, ps, len(pooled_vals)),
            }
            print(f"{name:42s} {kind:7s} {pm:.3f}±{ps:.3f}  {tm:.3f}±{ts:.3f}  (seeds={n_seeds})")

    atomic_write_json(TRANSFER_CONTRAST_JSON, results)
    print(f"\nResults -> {TRANSFER_CONTRAST_JSON}")
    return results


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
    print("\nRead: 'pooled' = random-split (easy). 'transfer' = probe fit on one")
    print("group-half, scored on the disjoint half. linear vs gbm shows whether")
    print("nonlinearity recovers cross-group signal (result_21: it does for stability,")
    print("not for mechanism). NOTE: stability grouped by protein (S1724 has no local")
    print(
        "Pfam map); shared-family proteins leak somewhat — result_21's Pfam-split GBM"
    )
    print("(0.750) is the authoritative stability number.")


if __name__ == "__main__":
    main()
