"""Probe 4: what is the pathogenicity direction?

Tests context-free substitution biochemistry (BLOSUM62, hydropathy, charge, volume)
against the ESM-2 axis. Does not cover position-specific conservation (see conservation_axis).
"""

import numpy as np
import functools

from esm2_mech.utils.constants import N_SEEDS
from esm2_mech.utils.data import load_pfam_map
from esm2_mech.utils.io import write_result_json
from esm2_mech.utils.paths import (
    GEOMETRY_RESULTS_DIR,
    PROBE4_AXIS_IDENTITY_JSON,
    PFAM_JSON,
)
from esm2_mech.utils.metrics import mean_std_n
from esm2_mech.utils.probes import auroc_for_clf
from esm2_mech.utils.splits import family_split_cv
from esm2_mech.experiments.geometry.axis_analysis import (
    family_held_out_axis_analysis,
    format_axis_summary,
)
from esm2_mech.experiments.geometry.data import (
    load_pathogenicity_geometry_inputs,
    pathogenicity_geometry_provenance,
)

print = functools.partial(print, flush=True)

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
    from sklearn.linear_model import LogisticRegression

    inputs = load_pathogenicity_geometry_inputs()
    v = inputs.variants
    pfam = load_pfam_map(PFAM_JSON)

    bio, keep = [], []
    for i, x in enumerate(v):
        f = biochem_features(x["aa_wt"], x["aa_mut"])
        if f is not None:
            bio.append(f)
            keep.append(i)
    keep = np.array(keep)
    bio = np.array(bio, dtype=float)
    delta = inputs.delta[keep]
    y = inputs.labels[keep]
    genes = inputs.genes[keep]
    mag = np.linalg.norm(delta, axis=1)
    print(f"Variants with biochem features: {len(keep)} / {len(v)}")

    association_features = {name: bio[:, j] for j, name in enumerate(FEAT_NAMES)}
    association_features["magnitude"] = mag
    axis_analysis = family_held_out_axis_analysis(
        delta,
        y,
        genes,
        pfam,
        association_features,
        regression_features=bio,
        seeds=range(n_seeds),
    )
    print("\n=== A. Family-held-out Spearman(axis score, feature) ===")
    for name, summary in axis_analysis["correlations"].items():
        print(f"  {name:14s} rho = {format_axis_summary(summary)}")
    r2 = axis_analysis["regression_r2"]
    print("\n=== B. Family-held-out prediction of axis score from biochemistry ===")
    print(f"  R^2(axis ~ biochem) = {format_axis_summary(r2)}")

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
        mean, std, n = mean_std_n(a)
        return {"mean": mean, "std": std, "n": n}

    context_free = agg(cf)
    esm2_delta = agg(esm)
    combined = agg(both)
    print(
        f"  context-free biochem only : {context_free['mean']:.3f} "
        f"± {context_free['std']:.3f}"
    )
    print(
        f"  ESM-2 delta only          : {esm2_delta['mean']:.3f} "
        f"± {esm2_delta['std']:.3f}"
    )
    print(
        f"  ESM-2 + biochem           : {combined['mean']:.3f} ± {combined['std']:.3f}"
    )

    result = {
        "n": int(len(keep)),
        "axis_analysis_family_held_out": axis_analysis,
        "pathogenicity_auroc_family_split": {
            "context_free": context_free,
            "esm2_delta": esm2_delta,
            "esm2_plus_biochem": combined,
        },
        "analysis_status": "exploratory",
        "input_provenance": pathogenicity_geometry_provenance(inputs, pfam),
    }
    write_result_json(PROBE4_AXIS_IDENTITY_JSON, result, seeds=list(range(n_seeds)))
    print(f"\nResults -> {PROBE4_AXIS_IDENTITY_JSON}")

    print("\n=== DESCRIPTIVE SUMMARY ===")
    print(f"  Family-held-out R^2(axis ~ biochem) = {format_axis_summary(r2)}")
    print(
        f"  Context-free biochemistry AUROC = {context_free['mean']:.3f}; "
        f"ESM-2 delta AUROC = {esm2_delta['mean']:.3f}."
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
