"""
The decider for result_23's geometric story: does a *linear direction* fit on
one group of proteins/families TRANSFER to a disjoint group — for pathogenicity,
stability, and mechanism, under one identical protocol and metric?

Protocol (same as direction_geometry.py Probe 2):
  - one shared StandardScaler per task (common coordinate frame)
  - split the task's grouping variable (Pfam family, or protein) into two
    disjoint halves, fit L2 logistic on half A, score half B (and B->A), AUROC
  - average over n_partitions random half-splits
  - report transfer AUROC vs a pooled (random-split) reference

Tasks (all binary so the AUROC is comparable):
  - pathogenicity : group=Pfam family, y = pathogenic           (canonical n=16,576)
  - stability     : group=protein,     y = ΔΔG > median         (S1724 n=1,277)
  - mechanism     : group=Pfam family, y = GOF vs rest          (Gerasimavicius)

Expectation if the geometry unifies the project:
  pathogenicity transfers (~0.81), stability drops, mechanism ~chance.

Pure CPU. Usage:
  cd esm2_mechanism
  python3 scripts/transfer_contrast.py
"""

import json
import numpy as np
import functools

print = functools.partial(print, flush=True)

from esm2_mech.utils.io import atomic_write_json
from esm2_mech.utils.paths import (
    GEOMETRY_RESULTS_DIR,
    TRANSFER_CONTRAST_JSON,
    MEGASCALE_VARIANTS_JSON,
    PATH_EMB_WT_MEAN,
    PATH_EMB_MUT_MEAN,
    PATHOGENICITY_CANONICAL_VARIANTS_JSON,
    PFAM_JSON,
    MEGASCALE_EMB_WT_MEAN,
    MEGASCALE_EMB_MUT_MEAN,
)
from esm2_mech.experiments.mechanism.loaders import load_mechanism_variants
from esm2_mech.utils.constants import GOF
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
    # Deliberate: fit the scaler ONCE on all rows so every group-half is scored in
    # one shared coordinate frame (matches direction_geometry Probe 2). The probe
    # tests whether a learned DIRECTION transfers across groups, which requires a
    # common frame — not a per-fold refit. Standardization leakage is negligible
    # here and is the intended design, not a train/test bug. Do not "fix" to
    # train-only without changing the experiment's meaning.
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
            # min_pos in BOTH train and test guarantees both classes are present
            # on each side. This guard is load-bearing: auroc_for_clf ->
            # _pos_class_col raises if the fitted clf never saw the positive class
            # (a single-class GBM/logreg fit sets classes_=[0]). Keep min_pos >= 1.
            if y[tr].sum() < min_pos or (1 - y[tr]).sum() < min_pos:
                continue
            if y[te].sum() < min_pos or (1 - y[te]).sum() < min_pos:
                continue
            clf = _make_clf(kind, seed).fit(Xs[tr], y[tr])
            transfer.append(auroc_for_clf(clf, Xs[te], y[te]))

    # Pooled (random-split) reference. StratifiedKFold(n_splits=k) raises if the
    # minority class has fewer than k members task-wide (e.g. rare GOF in the
    # mechanism task), so cap n_splits at the minority-class count and skip
    # entirely if even a 2-fold split is impossible — degrade gracefully like the
    # transfer block rather than crashing the whole task.
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
        return (mean, std if n else 0.0, n)

    return {"transfer_auroc": agg(transfer), "pooled_auroc": agg(pooled)}


# ── Task loaders ─────────────────────────────────────────────────────────────


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


def load_stability():
    if not MEGASCALE_VARIANTS_JSON.exists():
        return None
    with open(MEGASCALE_VARIANTS_JSON) as _f:
        v = json.load(_f)
    delta = np.load(MEGASCALE_EMB_MUT_MEAN) - np.load(
        MEGASCALE_EMB_WT_MEAN
    )
    n = min(len(v), len(delta))
    v, delta = v[:n], delta[:n]
    ddg = np.array([x["ddg"] for x in v], dtype=float)
    groups = np.array([x["protein"] for x in v])
    y = (ddg > np.median(ddg)).astype(int)  # median split (matches result_21)
    return delta, y, groups


def load_mechanism_gof():
    with open(PFAM_JSON) as _f:
        pfam = json.load(_f)
    dm, _dp, labels, genes = load_mechanism_variants(pfam)
    groups = np.array([(pfam.get(g) or "NA") for g in genes])
    y = (np.asarray(labels) == GOF).astype(int)
    m = groups != "NA"
    return dm[m], y[m], groups[m]


def run(n_seeds=5):
    """Run the transfer contrast across tasks. Each seed reshuffles the group
    half-splits; results are pooled over all n_seeds × partitions."""
    tasks = {}
    print("Loading tasks...")
    tasks["pathogenicity (path vs benign, family-split)"] = load_pathogenicity()
    tasks["mechanism (GOF vs rest, family-split)"] = load_mechanism_gof()
    stab = load_stability()
    if stab is not None:
        tasks["stability (ΔΔG>median, protein-split)"] = stab
    else:
        print("  stability: S1724 embeddings not found — skipping")

    results = {}
    print("\n" + "=" * 86)
    print(f"{'task':42s} {'probe':7s} {'pooled':>12s} {'transfer':>13s}")
    print("=" * 86)
    for name, (delta, y, groups) in tasks.items():
        results[name] = {}
        # GBM on the big tasks is heavy; fewer partitions there.
        npart = 5 if len(y) > 5000 else 10
        for kind in ("linear", "gbm"):
            # Pool transfer + pooled AUROCs across seeds (each seed = fresh splits).
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
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds (>=1)")
    args = ap.parse_args()
    run(n_seeds=args.seeds)
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
