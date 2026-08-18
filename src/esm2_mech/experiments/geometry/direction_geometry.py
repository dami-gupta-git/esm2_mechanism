"""Direction geometry of the pathogenicity axis.

Probe 1: is pathogenicity one direction or a subspace? (iterative rank removal)
Probe 2: is that direction family-universal? (cross-family cosine + transfer AUROC)
"""

import json
import numpy as np
import functools

print = functools.partial(print, flush=True)

from esm2_mech.utils.constants import N_SEEDS
from esm2_mech.utils.io import atomic_write_json
from esm2_mech.utils.paths import (
    GEOMETRY_RESULTS_DIR,
    DIRECTION_GEOMETRY_JSON,
    PATH_EMB_WT_MEAN,
    PATH_EMB_MUT_MEAN,
    PATHOGENICITY_CANONICAL_VARIANTS_JSON,
    PFAM_JSON,
)
from esm2_mech.utils.metrics import mean_std_n
from esm2_mech.utils.probes import auroc_for_clf
from esm2_mech.utils.splits import family_split_cv

GEOMETRY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PATH_VARIANTS = PATHOGENICITY_CANONICAL_VARIANTS_JSON
PATH_WT = PATH_EMB_WT_MEAN
PATH_MUT = PATH_EMB_MUT_MEAN


def _pathogenicity_label(label):
    """Map a canonical-set label to 1 (pathogenic) / 0 (benign); never a catch-all."""
    if label == "pathogenic":
        return 1
    if label == "benign":
        return 0
    raise ValueError(f"unexpected pathogenicity label {label!r} (expected 'pathogenic'/'benign')")


def load():
    with open(PATH_VARIANTS) as _f:
        variants = json.load(_f)
    delta = np.load(PATH_MUT) - np.load(PATH_WT)
    if len(variants) != delta.shape[0]:
        raise ValueError(
            f"variant/embedding row mismatch: {len(variants)} variants vs "
            f"{delta.shape[0]} embedding rows — canonical file is not row-aligned."
        )
    genes = np.array([v["gene"] for v in variants])
    y = np.array([_pathogenicity_label(v["label"]) for v in variants])
    with open(PFAM_JSON) as _f:
        pfam = json.load(_f)
    fam = np.array([(pfam.get(g) or "NA") for g in genes])
    print(
        f"Loaded {len(y)} variants, {len(set(genes))} genes, "
        f"{int(y.sum())} path / {int((1-y).sum())} benign, "
        f"{len(set(fam[fam!='NA']))} Pfam families"
    )
    return delta.astype(np.float64), y, genes, fam


def fit_direction(X, y, seed=0):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(X, y)
    w = clf.coef_.ravel()
    w_unit = w / (np.linalg.norm(w) + 1e-12)
    return w_unit, clf

def probe1_rank(delta, y, genes, pfam_map, k_max=5, seeds=(0, 1, 2, 3, 4)):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    print("\n" + "=" * 60)
    print("PROBE 1 — is pathogenicity one direction or a subspace? (family-split)")
    print("=" * 60)

    decay = {k: [] for k in range(k_max + 1)}
    proj1 = []

    for seed in seeds:
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
            decay[0].append(auroc_for_clf(clf, Xte, yte))

            w1 = clf.coef_.ravel()
            w1 /= np.linalg.norm(w1) + 1e-12
            s_tr = (Xtr @ w1).reshape(-1, 1)
            s_te = (Xte @ w1).reshape(-1, 1)
            clf1 = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(
                s_tr, ytr
            )
            proj1.append(auroc_for_clf(clf1, s_te, yte))

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
                decay[k].append(auroc_for_clf(ck2, Rte, yte))

    def agg(v):
        mean, std, n = mean_std_n(v)
        return (mean, std if n else 0.0)

    out = {
        "full_auroc": agg(decay[0]),
        "proj1_auroc": agg(proj1),
        "auroc_after_removing_k": {k: agg(decay[k]) for k in decay},
    }
    m, s = out["full_auroc"]
    print(f"  full linear AUROC            = {m:.3f} ± {s:.3f}")
    m, s = out["proj1_auroc"]
    print(f"  1-D projection (top dir)    = {m:.3f} ± {s:.3f}")
    for k in range(k_max + 1):
        m, s = out["auroc_after_removing_k"][k]
        print(f"  AUROC after removing {k} dir(s) = {m:.3f} ± {s:.3f}")
    return out

def probe2_universal(delta, y, genes, fam, n_partitions=10, seeds=(0,)):
    from sklearn.preprocessing import StandardScaler

    print("\n" + "=" * 60)
    print("PROBE 2 — is the pathogenicity direction family-universal?")
    print("=" * 60)

    mask = fam != "NA"
    X = delta[mask]
    yy = y[mask]
    ff = fam[mask]
    families = np.array(sorted(set(ff.tolist())))

    cos_obs, cos_null, transfer = [], [], []
    part = 0
    for seed in seeds:
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
            wA_orig = sc_a.scale_ * wA
            wB_orig = sc_b.scale_ * wB
            wA_orig /= np.linalg.norm(wA_orig) + 1e-12
            wB_orig /= np.linalg.norm(wB_orig) + 1e-12
            cos_obs.append(float(np.dot(wA_orig, wB_orig)))
            Xb_via_a = sc_a.transform(X[b])
            transfer.append(auroc_for_clf(clfA, Xb_via_a, yy[b]))

            yA_s = yy[a].copy()
            rng.shuffle(yA_s)
            yB_s = yy[b].copy()
            rng.shuffle(yB_s)
            wAn, _ = fit_direction(Xa_own, yA_s, seed=seed)
            wBn, _ = fit_direction(Xb_own, yB_s, seed=seed)
            wAn_orig = sc_a.scale_ * wAn
            wBn_orig = sc_b.scale_ * wBn
            wAn_orig /= np.linalg.norm(wAn_orig) + 1e-12
            wBn_orig /= np.linalg.norm(wBn_orig) + 1e-12
            cos_null.append(float(np.dot(wAn_orig, wBn_orig)))
            part += 1

    def agg(v):
        mean, std, n = mean_std_n(v)
        return (mean, std if n else 0.0)

    out = {
        "n_partitions": part,
        "cosine_observed": agg(cos_obs),
        "cosine_null_shuffled": agg(cos_null),
        "transfer_auroc_AtoB": agg(transfer),
    }
    m, s = out["cosine_observed"]
    print(f"  cosine(w_A, w_B) observed   = {m:.3f} ± {s:.3f}")
    m, s = out["cosine_null_shuffled"]
    print(f"  cosine null (shuffled y)    = {m:.3f} ± {s:.3f}")
    m, s = out["transfer_auroc_AtoB"]
    print(f"  transfer AUROC (A's dir->B) = {m:.3f} ± {s:.3f}")
    return out


def run(n_seeds=N_SEEDS):
    seeds = tuple(range(n_seeds))
    delta, y, genes, fam = load()
    with open(PFAM_JSON) as _f:
        pfam_map = json.load(_f)

    r1 = probe1_rank(delta, y, genes, pfam_map, seeds=seeds)
    r2 = probe2_universal(delta, y, genes, fam, seeds=seeds)

    result = {"probe1_rank": r1, "probe2_universal": r2, "n_variants": int(len(y))}
    atomic_write_json(DIRECTION_GEOMETRY_JSON, result)
    print(f"\nResults -> {DIRECTION_GEOMETRY_JSON}")
    return result


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=N_SEEDS, help="number of seeds (>=1)")
    args = ap.parse_args()
    result = run(n_seeds=args.seeds)
    r1, r2 = result["probe1_rank"], result["probe2_universal"]

    print("\n" + "=" * 60)
    print("READ")
    print("=" * 60)
    fa = r1["full_auroc"][0]
    p1 = r1["proj1_auroc"][0]
    rem1 = r1["auroc_after_removing_k"][1][0]
    print(
        f"  One direction recovers {p1:.3f} of the full {fa:.3f}; "
        f"removing 1 direction drops to {rem1:.3f}."
    )
    co = r2["cosine_observed"][0]
    cn = r2["cosine_null_shuffled"][0]
    tr = r2["transfer_auroc_AtoB"][0]
    print(
        f"  Cross-family direction cosine = {co:.3f} (null {cn:.3f}); "
        f"transfer AUROC = {tr:.3f}."
    )


if __name__ == "__main__":
    main()
