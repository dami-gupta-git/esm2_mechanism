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
import os
import sys
import numpy as np
import functools
print = functools.partial(print, flush=True)

from esm2_mechanism.utils_paths import DATA_DIR as _DATA_DIR, RESULTS_DIR as _RESULTS_DIR
import esm2_mechanism.mechanism.multiseed_v1 as ms

DATA = str(_DATA_DIR)
EMB = str(_DATA_DIR / "embeddings")
OUT = str(_RESULTS_DIR / "magnitude_direction")
os.makedirs(OUT, exist_ok=True)


def _auroc(clf, X, y):
    from sklearn.metrics import roc_auc_score
    if len(set(y)) < 2:
        return np.nan
    p = clf.predict_proba(X)[:, list(clf.classes_).index(1)]
    return float(roc_auc_score(y, p))


def _make_clf(kind, seed):
    if kind == "linear":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    if kind == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                              random_state=seed)
    raise ValueError(kind)


def transfer_test(delta, y, groups, kind="linear", n_partitions=10, seed=0, min_pos=5):
    """Fit a probe on half the groups, score the disjoint half. Returns transfer
    AUROC (mean over both directions and partitions) and a pooled random-split
    reference. `kind` in {linear, gbm}."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y).astype(int)
    Xs = StandardScaler().fit_transform(delta)
    groups = np.asarray(groups)
    uniq = np.array(sorted(set(groups.tolist())))
    rng = np.random.RandomState(seed)

    transfer = []
    for _ in range(n_partitions):
        gshuf = uniq.copy(); rng.shuffle(gshuf)
        half = set(gshuf[:len(gshuf) // 2])
        a = np.array([g in half for g in groups]); b = ~a
        for tr, te in [(a, b), (b, a)]:
            if y[tr].sum() < min_pos or (1 - y[tr]).sum() < min_pos:
                continue
            if y[te].sum() < min_pos or (1 - y[te]).sum() < min_pos:
                continue
            clf = _make_clf(kind, seed).fit(Xs[tr], y[tr])
            transfer.append(_auroc(clf, Xs[te], y[te]))

    pooled = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(Xs, y):
        clf = _make_clf(kind, seed).fit(Xs[tr], y[tr])
        pooled.append(_auroc(clf, Xs[te], y[te]))

    def agg(v):
        v = [x for x in v if x is not None and not np.isnan(x)]
        return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (float("nan"), 0.0, 0)
    return {"transfer_auroc": agg(transfer), "pooled_auroc": agg(pooled)}


# ── Task loaders ─────────────────────────────────────────────────────────────

def load_pathogenicity():
    v = json.load(open(os.path.join(DATA, "pathogenicity_valid_variants_canonical.json")))
    delta = (np.load(os.path.join(EMB, "emb_mut_mean_path_canonical_n16576.npy"))
             - np.load(os.path.join(EMB, "emb_wt_mean_path_canonical_n16576.npy")))
    pfam = json.load(open(ms.PFAM_JSON))
    genes = [x["gene"] for x in v]
    groups = np.array([(pfam.get(g) or "NA") for g in genes])
    y = np.array([1 if x["label"] == "pathogenic" else 0 for x in v])
    m = groups != "NA"
    return delta[m], y[m], groups[m]


def load_stability():
    vpath = os.path.join(DATA, "megascale_variants.json")
    if not os.path.exists(vpath):
        return None
    v = json.load(open(vpath))
    delta = (np.load(os.path.join(EMB, "megascale_mut_mean.npy"))
             - np.load(os.path.join(EMB, "megascale_wt_mean.npy")))
    n = min(len(v), len(delta))
    v, delta = v[:n], delta[:n]
    ddg = np.array([x["ddg"] for x in v], dtype=float)
    groups = np.array([x["protein"] for x in v])
    y = (ddg > np.median(ddg)).astype(int)   # median split (matches result_21)
    return delta, y, groups


def load_mechanism_gof():
    dm, _dp, labels, genes = ms.load_geras(json.load(open(ms.PFAM_JSON)))
    pfam = json.load(open(ms.PFAM_JSON))
    groups = np.array([(pfam.get(g) or "NA") for g in genes])
    y = (np.asarray(labels) == "GOF").astype(int)
    m = groups != "NA"
    return dm[m], y[m], groups[m]


def main():
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
        # GBM on the big tasks is heavy; fewer partitions there
        npart = 5 if len(y) > 5000 else 10
        for kind in ("linear", "gbm"):
            r = transfer_test(delta, y, groups, kind=kind, n_partitions=npart)
            results[name][kind] = r
            pm, ps, _ = r["pooled_auroc"]; tm, ts, tn = r["transfer_auroc"]
            print(f"{name:42s} {kind:7s} {pm:.3f}±{ps:.3f}  {tm:.3f}±{ts:.3f}  (n={tn})")

    json.dump(results, open(os.path.join(OUT, "transfer_contrast.json"), "w"), indent=2)
    print(f"\nResults -> {os.path.join(OUT, 'transfer_contrast.json')}")
    print("\nRead: 'pooled' = random-split (easy). 'transfer' = probe fit on one")
    print("group-half, scored on the disjoint half. linear vs gbm shows whether")
    print("nonlinearity recovers cross-group signal (result_21: it does for stability,")
    print("not for mechanism). NOTE: stability grouped by protein (S1724 has no local")
    print("Pfam map); shared-family proteins leak somewhat — result_21's Pfam-split GBM")
    print("(0.750) is the authoritative stability number.")


if __name__ == "__main__":
    main()
