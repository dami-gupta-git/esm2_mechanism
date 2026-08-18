"""Probe 4: what is the pathogenicity direction?

Tests context-free substitution biochemistry (BLOSUM62, hydropathy, charge, volume)
against the ESM-2 axis. Does not cover position-specific conservation (see conservation_axis).
"""

import json
import numpy as np
import functools

print = functools.partial(print, flush=True)

from esm2_mech.utils.constants import N_SEEDS
from esm2_mech.utils.data import load_pfam_map
from esm2_mech.utils.io import atomic_write_json
from esm2_mech.utils.paths import (
    GEOMETRY_RESULTS_DIR,
    PROBE4_AXIS_IDENTITY_JSON,
    PATH_EMB_WT_MEAN,
    PATH_EMB_MUT_MEAN,
    PATHOGENICITY_CANONICAL_VARIANTS_JSON,
    PFAM_JSON,
)
from esm2_mech.utils.metrics import mean_std_n
from esm2_mech.utils.probes import auroc_for_clf
from esm2_mech.utils.splits import family_split_cv

GEOMETRY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AA = "ARNDCQEGHILKMFPSTWYV"

_BL = [
    [4],
    [-1, 5],
    [-2, 0, 6],
    [-2, -2, 1, 6],
    [0, -3, -3, -3, 9],
    [-1, 1, 0, 0, -3, 5],
    [-1, 0, 0, 2, -4, 2, 5],
    [0, -2, 0, -1, -3, -2, -2, 6],
    [-2, 0, 1, -1, -3, 0, 0, -2, 8],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4],
    [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5],
    [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7],
    [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4],
    [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11],
    [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7],
    [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
]
BLOSUM = {}
for i, a in enumerate(AA):
    for j, b in enumerate(AA[: i + 1]):
        BLOSUM[(a, b)] = BLOSUM[(b, a)] = _BL[i][j]

HYDRO = dict(
    zip(
        AA,
        [
            1.8,
            -4.5,
            -3.5,
            -3.5,
            2.5,
            -3.5,
            -3.5,
            -0.4,
            -3.2,
            4.5,
            3.8,
            -3.9,
            1.9,
            2.8,
            -1.6,
            -0.8,
            -0.7,
            -0.9,
            -1.3,
            4.2,
        ],
    )
)
CHARGE = {a: 0.0 for a in AA}
CHARGE.update({"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.5})
VOLUME = dict(
    zip(
        AA,
        [
            88.6,
            173.4,
            114.1,
            111.1,
            108.5,
            143.8,
            138.4,
            60.1,
            153.2,
            166.7,
            166.7,
            168.6,
            162.9,
            189.9,
            112.7,
            89.0,
            116.1,
            227.8,
            193.6,
            140.0,
        ],
    )
)


def _pathogenicity_label(label):
    """Map a canonical-set label to 1 (pathogenic) / 0 (benign); never a catch-all."""
    if label == "pathogenic":
        return 1
    if label == "benign":
        return 0
    raise ValueError(f"unexpected pathogenicity label {label!r} (expected 'pathogenic'/'benign')")


def biochem_features(wt, mut):
    if wt not in AA or mut not in AA:
        return None
    return [
        BLOSUM[(wt, mut)],
        HYDRO[mut] - HYDRO[wt],
        abs(HYDRO[mut] - HYDRO[wt]),
        CHARGE[mut] - CHARGE[wt],
        abs(CHARGE[mut] - CHARGE[wt]),
        abs(VOLUME[mut] - VOLUME[wt]),
    ]


FEAT_NAMES = [
    "blosum62",
    "d_hydro",
    "abs_d_hydro",
    "d_charge",
    "abs_d_charge",
    "abs_d_volume",
]


def run(n_seeds=N_SEEDS):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    from scipy.stats import spearmanr

    with open(PATHOGENICITY_CANONICAL_VARIANTS_JSON) as _f:
        v = json.load(_f)
    delta = np.load(PATH_EMB_MUT_MEAN) - np.load(PATH_EMB_WT_MEAN)
    if len(v) != delta.shape[0]:
        raise ValueError(
            f"variant/embedding row mismatch: {len(v)} variants vs "
            f"{delta.shape[0]} embedding rows — canonical file is not row-aligned."
        )
    pfam = load_pfam_map(PFAM_JSON)

    bio, keep = [], []
    for i, x in enumerate(v):
        f = biochem_features(x["aa_wt"], x["aa_mut"])
        if f is not None:
            bio.append(f)
            keep.append(i)
    keep = np.array(keep)
    bio = np.array(bio, dtype=float)
    delta = delta[keep]
    y = np.array([_pathogenicity_label(v[i]["label"]) for i in keep])
    genes = np.array([v[i]["gene"] for i in keep])
    mag = np.linalg.norm(delta, axis=1)
    print(f"Variants with biochem features: {len(keep)} / {len(v)}")

    Xs = StandardScaler().fit_transform(delta)
    w = LogisticRegression(max_iter=2000, C=1.0).fit(Xs, y).coef_.ravel()
    w /= np.linalg.norm(w) + 1e-12
    s = Xs @ w
    print("\n=== A. Spearman(axis score, feature) ===")
    corrA = {}
    for j, name in enumerate(FEAT_NAMES):
        rho = float(spearmanr(s, bio[:, j]).correlation)
        corrA[name] = rho
        print(f"  {name:14s} rho = {rho:+.3f}")
    rho_mag = float(spearmanr(s, mag).correlation)
    rho_y = float(spearmanr(s, y).correlation)
    print(f"  {'magnitude ||d||':14s} rho = {rho_mag:+.3f}")
    print(f"  {'(label)':14s} rho = {rho_y:+.3f}  (sanity: axis aligns with label)")

    print("\n=== B. predict axis score from biochem features (Ridge, 5-fold) ===")
    bs = StandardScaler().fit_transform(bio)
    r2s = []
    for tr, te in KFold(5, shuffle=True, random_state=0).split(bs):
        rg = Ridge(alpha=1.0).fit(bs[tr], s[tr])
        r2s.append(r2_score(s[te], rg.predict(bs[te])))
    r2 = float(np.mean(r2s))
    print(
        f"  R^2(axis ~ biochem) = {r2:.3f}   "
        f"({'mostly context-free biochemistry' if r2 > 0.5 else 'mostly context-dependent (beyond AA identity)'})"
    )

    print("\n=== C. pathogenicity AUROC, family-split (5 seeds) ===")

    def auroc_cv(X, splits, seed):
        out = []
        for tr, te in splits:
            if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(
                sc.transform(X[tr]), y[tr]
            )
            out.append(auroc_for_clf(clf, sc.transform(X[te]), y[te]))
        return out

    cf, esm, both = [], [], []
    for seed in range(n_seeds):
        fs = family_split_cv(genes, pfam, seed=seed)
        cf += auroc_cv(bio, fs, seed)
        esm += auroc_cv(delta, fs, seed)
        both += auroc_cv(np.hstack([delta, bio]), fs, seed)

    def agg(a):
        mean, std, _ = mean_std_n(a)
        return (mean, std)

    cm, cstd = agg(cf)
    em, estd = agg(esm)
    bm, bstd = agg(both)
    print(f"  context-free biochem only : {cm:.3f} ± {cstd:.3f}")
    print(f"  ESM-2 delta only          : {em:.3f} ± {estd:.3f}")
    print(f"  ESM-2 + biochem           : {bm:.3f} ± {bstd:.3f}")

    result = {
        "n": int(len(keep)),
        "A_spearman_axis_vs_feature": corrA,
        "A_spearman_axis_vs_magnitude": rho_mag,
        "B_r2_axis_from_biochem": r2,
        "C_auroc_family_split": {
            "context_free": [cm, cstd],
            "esm2_delta": [em, estd],
            "esm2_plus_biochem": [bm, bstd],
        },
    }
    atomic_write_json(PROBE4_AXIS_IDENTITY_JSON, result)
    print(f"\nResults -> {PROBE4_AXIS_IDENTITY_JSON}")

    print("\n=== READ ===")
    print(
        f"  Axis is {'largely' if r2 > 0.5 else 'only partly'} explained by context-free biochemistry "
        f"(R^2={r2:.2f})."
    )
    print(
        f"  Context-free biochem reaches {cm:.3f} AUROC vs ESM-2's {em:.3f}: "
        f"ESM-2 adds {em - cm:+.3f} of context-dependent signal."
    )
    return result


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=N_SEEDS, help="number of seeds (>=1)")
    args = ap.parse_args()
    run(n_seeds=args.seeds)


if __name__ == "__main__":
    main()
