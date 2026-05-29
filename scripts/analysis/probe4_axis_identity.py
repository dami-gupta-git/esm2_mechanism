"""
Probe 4 (follow-up to result_23): what IS the pathogenicity direction?

result_23 found pathogenicity is a single, family-transferable linear direction
in ESM-2 perturbation space. This asks what that axis corresponds to, using
CONTEXT-FREE substitution biochemistry (computable from wt->mut identity with no
GPU): BLOSUM62, and changes in hydropathy / charge / volume.

  A. Correlate the axis-projection score with each biochemical feature.
  B. Regress the axis score on all biochemical features (Ridge R^2) — how much of
     the axis is explained by context-free biochemistry.
  C. Pathogenicity AUROC from context-free features ALONE (family-split) vs ESM-2
     (0.884) — how much pathogenicity is substitution identity vs context.

NOTE: this tests context-FREE biochemistry. Position-specific *conservation*
(ESM-2 masked log-likelihood / entropy at the variant position) is the natural
GPU follow-up (plan_loglik) and is NOT covered here.

Pure CPU. Usage:
  cd esm2_mechanism
  python3 scripts/probe4_axis_identity.py
"""

import json
import os
import sys
import numpy as np
import functools
print = functools.partial(print, flush=True)

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(SCRIPTS)
DATA = os.path.join(ROOT, "data")
EMB = os.path.join(DATA, "embeddings")
OUT = os.path.join(ROOT, "results", "magnitude_direction")

sys.path.insert(0, SCRIPTS)
import mechanism.multiseed_v1 as ms

AA = "ARNDCQEGHILKMFPSTWYV"

# BLOSUM62 lower-triangle (incl. diagonal), rows in AA order
_BL = [
 [4],
 [-1,5],
 [-2,0,6],
 [-2,-2,1,6],
 [0,-3,-3,-3,9],
 [-1,1,0,0,-3,5],
 [-1,0,0,2,-4,2,5],
 [0,-2,0,-1,-3,-2,-2,6],
 [-2,0,1,-1,-3,0,0,-2,8],
 [-1,-3,-3,-3,-1,-3,-3,-4,-3,4],
 [-1,-2,-3,-4,-1,-2,-3,-4,-3,2,4],
 [-1,2,0,-1,-3,1,1,-2,-1,-3,-2,5],
 [-1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5],
 [-2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6],
 [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7],
 [1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4],
 [0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5],
 [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11],
 [-2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7],
 [0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4],
]
BLOSUM = {}
for i, a in enumerate(AA):
    for j, b in enumerate(AA[:i + 1]):
        BLOSUM[(a, b)] = BLOSUM[(b, a)] = _BL[i][j]

HYDRO = dict(zip(AA, [1.8,-4.5,-3.5,-3.5,2.5,-3.5,-3.5,-0.4,-3.2,4.5,3.8,-3.9,1.9,2.8,-1.6,-0.8,-0.7,-0.9,-1.3,4.2]))
CHARGE = {a: 0.0 for a in AA}; CHARGE.update({"D":-1,"E":-1,"K":1,"R":1,"H":0.5})
VOLUME = dict(zip(AA, [88.6,173.4,114.1,111.1,108.5,143.8,138.4,60.1,153.2,166.7,166.7,168.6,162.9,189.9,112.7,89.0,116.1,227.8,193.6,140.0]))


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
FEAT_NAMES = ["blosum62", "d_hydro", "abs_d_hydro", "d_charge", "abs_d_charge", "abs_d_volume"]


def main():
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import KFold
    from sklearn.metrics import roc_auc_score, r2_score
    from scipy.stats import spearmanr

    v = json.load(open(os.path.join(DATA, "pathogenicity_valid_variants_canonical.json")))
    delta = (np.load(os.path.join(EMB, "emb_mut_mean_path_canonical_n16576.npy"))
             - np.load(os.path.join(EMB, "emb_wt_mean_path_canonical_n16576.npy")))
    pfam = json.load(open(ms.PFAM_JSON))

    bio, keep = [], []
    for i, x in enumerate(v):
        f = biochem_features(x["aa_wt"], x["aa_mut"])
        if f is not None:
            bio.append(f); keep.append(i)
    keep = np.array(keep)
    bio = np.array(bio, dtype=float)
    delta = delta[keep]
    y = np.array([1 if v[i]["label"] == "pathogenic" else 0 for i in keep])
    genes = np.array([v[i]["gene"] for i in keep])
    mag = np.linalg.norm(delta, axis=1)
    print(f"Variants with biochem features: {len(keep)} / {len(v)}")

    # ── axis-projection score (direction from a single full-data logreg fit) ──
    Xs = StandardScaler().fit_transform(delta)
    w = LogisticRegression(max_iter=2000, C=1.0).fit(Xs, y).coef_.ravel()
    w /= (np.linalg.norm(w) + 1e-12)
    s = Xs @ w   # pathogenicity-axis projection per variant

    # ── A. correlations ──────────────────────────────────────────────────────
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

    # ── B. how much of the axis is context-free biochemistry (Ridge R^2) ─────
    print("\n=== B. predict axis score from biochem features (Ridge, 5-fold) ===")
    bs = StandardScaler().fit_transform(bio)
    r2s = []
    for tr, te in KFold(5, shuffle=True, random_state=0).split(bs):
        rg = Ridge(alpha=1.0).fit(bs[tr], s[tr])
        r2s.append(r2_score(s[te], rg.predict(bs[te])))
    r2 = float(np.mean(r2s))
    print(f"  R^2(axis ~ biochem) = {r2:.3f}   "
          f"({'mostly context-free biochemistry' if r2 > 0.5 else 'mostly context-dependent (beyond AA identity)'})")

    # ── C. pathogenicity AUROC: context-free vs ESM-2 (family-split) ─────────
    print("\n=== C. pathogenicity AUROC, family-split (5 seeds) ===")
    def auroc_cv(X, splits, seed):
        out = []
        for tr, te in splits:
            if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(sc.transform(X[tr]), y[tr])
            p = clf.predict_proba(sc.transform(X[te]))[:, list(clf.classes_).index(1)]
            out.append(roc_auc_score(y[te], p))
        return out
    cf, esm, both = [], [], []
    for seed in range(5):
        fs = ms.family_split_cv(genes, pfam, seed=seed)
        cf += auroc_cv(bio, fs, seed)
        esm += auroc_cv(delta, fs, seed)
        both += auroc_cv(np.hstack([delta, bio]), fs, seed)
    def agg(a): return (float(np.mean(a)), float(np.std(a)))
    cm, cstd = agg(cf); em, estd = agg(esm); bm, bstd = agg(both)
    print(f"  context-free biochem only : {cm:.3f} ± {cstd:.3f}")
    print(f"  ESM-2 delta only          : {em:.3f} ± {estd:.3f}")
    print(f"  ESM-2 + biochem           : {bm:.3f} ± {bstd:.3f}")

    result = {"n": int(len(keep)),
              "A_spearman_axis_vs_feature": corrA,
              "A_spearman_axis_vs_magnitude": rho_mag,
              "B_r2_axis_from_biochem": r2,
              "C_auroc_family_split": {"context_free": [cm, cstd],
                                        "esm2_delta": [em, estd],
                                        "esm2_plus_biochem": [bm, bstd]}}
    json.dump(result, open(os.path.join(OUT, "probe4_axis_identity.json"), "w"), indent=2)
    print(f"\nResults -> {os.path.join(OUT, 'probe4_axis_identity.json')}")

    print("\n=== READ ===")
    print(f"  Axis is {'largely' if r2 > 0.5 else 'only partly'} explained by context-free biochemistry "
          f"(R^2={r2:.2f}).")
    print(f"  Context-free biochem reaches {cm:.3f} AUROC vs ESM-2's {em:.3f}: "
          f"ESM-2 adds {em - cm:+.3f} of context-dependent signal.")


if __name__ == "__main__":
    main()
