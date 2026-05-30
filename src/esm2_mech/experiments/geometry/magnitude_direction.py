"""
Magnitude-vs-direction decomposition of ESM-2 deltas (plan_magnitude_direction.md).

Question: result_6 shows the same ESM-2 delta embeddings predict pathogenicity
(AUROC 0.886) but not mechanism (chance). This script asks *why* by splitting
each delta d = mut_emb - wt_emb into:

    magnitude m = ||d||           (one scalar: how much the rep is disturbed)
    direction u = d / ||d||        (unit vector: which way it is disturbed)

and re-running the result_6 / result_4 probes on {full delta, magnitude only,
direction only}.

Probes (pre-registered):
  A  single-scalar magnitude:  m -> pathogenicity (A1) and mechanism (A2)
  B  direction only:           u -> pathogenicity (B1) and mechanism (B2)
  C  biophysical direction:    S1724 signed ddG  (C1 magnitude<->|ddG|,
                               C2 direction<->sign(ddG))  [needs result_21 cache]

Decision rules (pre-registered, family-split):
  P1: A1 magnitude-only pathogenicity AUROC >= 0.85   (pathogenicity is magnitude)
  P2: B1 direction-only pathogenicity AUROC <= 0.70   (confirms P1 from the other side)
  P3: B2 direction-only mechanism macro-F1 <= chance_floor + 0.02 (direction carries no mechanism)
  P4: C2 sign(ddG) AUROC >= 0.65 AND C1 Spearman >= 0.30 (ESM-2 has biophysical direction)

All embeddings are cached locally — no GPU required.

Usage:
  cd esm2_mechanism
  python3 scripts/magnitude_direction.py
  python3 scripts/magnitude_direction.py --seeds 0 1 2   # fewer seeds, faster
"""

import argparse
import json
import os
import sys
import numpy as np
from collections import defaultdict
import functools

print = functools.partial(print, flush=True)

from esm2_mech.utils.paths import (
    DATA_DIR as _DATA_DIR,
    RESULTS_DIR as _RESULTS_DIR,
    PATH_EMB_WT_MEAN,
    PATH_EMB_MUT_MEAN,
    MEGASCALE_EMB_WT_MEAN,
    MEGASCALE_EMB_MUT_MEAN,
)

from esm2_mech.utils.paths import PFAM_JSON
from esm2_mech.utils.splits import gene_split_cv, family_split_cv
from esm2_mech.utils.probes import run_mlp_binary_cv, run_mlp_probe_cv, run_logreg_cv
from esm2_mech.utils.probes import run_logreg_binary_cv

DATA = str(_DATA_DIR)
OUT = str(_RESULTS_DIR / "magnitude_direction")
os.makedirs(OUT, exist_ok=True)

# ── Pre-registered thresholds ────────────────────────────────────────────────
P1_PATH_MAG_MIN = 0.85  # magnitude-only pathogenicity AUROC, family-split
P2_PATH_DIR_MAX = 0.70  # direction-only pathogenicity AUROC, family-split
P3_MECH_MARGIN = 0.02  # direction-only mechanism F1 vs chance floor, family-split
P4_SIGN_AUROC_MIN = 0.65  # S1724 sign(ddG) AUROC
P4_MAG_SPEARMAN_MIN = 0.30  # S1724 Spearman(||d||, |ddG|)

# S1724 caches produced by megascale_stability.py (result_21)
S1724_WT_EMB = MEGASCALE_EMB_WT_MEAN
S1724_MUT_EMB = MEGASCALE_EMB_MUT_MEAN
S1724_VARIANTS = os.path.join(DATA, "megascale_variants.json")

# Canonical pathogenicity set (the one result_6's 0.884 family-split AUROC was
# computed on — n=16,576). NOT the older n17259 extraction in multiseed_v1.
PATH_CANON_VARIANTS = os.path.join(DATA, "pathogenicity_valid_variants_canonical.json")
PATH_CANON_WT_EMB = PATH_EMB_WT_MEAN
PATH_CANON_MUT_EMB = PATH_EMB_MUT_MEAN


def load_pathogenicity_canonical():
    """Load the canonical n=16,576 pathogenicity set (matches result_6)."""
    with open(PATH_CANON_VARIANTS) as _f:
        variants = json.load(_f)
    wt = np.load(PATH_CANON_WT_EMB)
    mut = np.load(PATH_CANON_MUT_EMB)
    delta = mut - wt
    genes = np.array([v["gene"] for v in variants])
    y = np.array([1 if v["label"] == "pathogenic" else 0 for v in variants])
    assert len(delta) == len(y), f"path emb/variant mismatch: {len(delta)} vs {len(y)}"
    print(
        f"  Pathogenicity (canonical): {len(variants)} variants, "
        f"{len(set(genes))} genes, {int(y.sum())} path / {int((1-y).sum())} benign"
    )
    return delta, y, genes


# ── Feature transforms ───────────────────────────────────────────────────────


def decompose(delta):
    """Return {'full': delta, 'mag': ||d|| (N,1), 'dir': d/||d|| (N,1280)}."""
    norm = np.linalg.norm(delta, axis=1, keepdims=True)
    mag = norm.astype(np.float32)  # (N, 1)
    direction = (delta / (norm + 1e-8)).astype(np.float32)
    return {"full": delta.astype(np.float32), "mag": mag, "dir": direction}


# ── Multiclass logreg probe (macro-F1 + per-class AUROC) ─────────────────────


def run_logreg_multi(X, labels, splits, seed=42):
    return run_logreg_cv(X, labels, splits, seed=seed)


def chance_floor_multi(labels, splits):
    """Stratified-random macro-F1 baseline (DummyClassifier) for the same splits."""
    from sklearn.dummy import DummyClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import f1_score

    le = LabelEncoder()
    y = le.fit_transform(labels)
    f1s = []
    for tr, te in splits:
        if len(set(y[tr])) < 2:
            continue
        d = DummyClassifier(strategy="stratified", random_state=0)
        d.fit(np.zeros((len(tr), 1)), y[tr])
        pred = d.predict(np.zeros((len(te), 1)))
        f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
    return float(np.mean(f1s)) if f1s else float("nan")


# ── Aggregation across seeds ─────────────────────────────────────────────────


def agg_seeds(per_seed_vals):
    vals = [v for v in per_seed_vals if v is not None and not np.isnan(v)]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}


# ── Probe A + B: pathogenicity (binary) ──────────────────────────────────────


def run_pathogenicity(pfam_map, seeds):
    print("\n" + "=" * 60)
    print("PATHOGENICITY  (binary, variant-level, delta_mean)")
    print("=" * 60)
    delta, y, genes = load_pathogenicity_canonical()
    feats = decompose(delta)

    # collect[feature][split][probe] = list of per-seed AUROC
    collect = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for seed in seeds:
        gs = gene_split_cv(genes, seed=seed)
        fs = family_split_cv(genes, pfam_map, seed=seed)
        for fname, X in feats.items():
            for split_name, splits in [("gene_split", gs), ("family_split", fs)]:
                lr = run_logreg_binary_cv(X, y, splits, seed=seed).get("auroc_mean")
                mlp = run_mlp_binary_cv(X, y, splits, seed=seed).get("auroc_mean")
                collect[fname][split_name]["logreg"].append(lr)
                collect[fname][split_name]["mlp"].append(mlp)
                print(
                    f"  seed{seed} {fname:4s} {split_name:12s} "
                    f"logreg={_f(lr)} mlp={_f(mlp)}"
                )

    out = {}
    for fname in feats:
        out[fname] = {}
        for split_name in ("gene_split", "family_split"):
            out[fname][split_name] = {
                "logreg_auroc": agg_seeds(collect[fname][split_name]["logreg"]),
                "mlp_auroc": agg_seeds(collect[fname][split_name]["mlp"]),
            }
    return out


# ── Probe A + B: mechanism (3-class) ─────────────────────────────────────────


def run_mechanism(pfam_map, seeds):
    print("\n" + "=" * 60)
    print("MECHANISM  (3-class GOF/LOF/DN, variant-level Gerasimavicius, delta_mean)")
    print("=" * 60)
    dm, _dp, labels, genes = load_geras(pfam_map)
    feats = decompose(dm)

    collect = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    floor = defaultdict(list)
    for seed in seeds:
        gs = gene_split_cv(genes, seed=seed)
        fs = family_split_cv(genes, pfam_map, seed=seed)
        for split_name, splits in [("gene_split", gs), ("family_split", fs)]:
            floor[split_name].append(chance_floor_multi(labels, splits))
        for fname, X in feats.items():
            for split_name, splits in [("gene_split", gs), ("family_split", fs)]:
                lr = run_logreg_multi(X, labels, splits, seed=seed)
                mlp = run_mlp_probe_cv(X, labels, splits, seed=seed)
                collect[fname][split_name]["logreg_f1"].append(lr.get("macro_f1_mean"))
                collect[fname][split_name]["mlp_f1"].append(mlp.get("macro_f1_mean"))
                collect[fname][split_name]["logreg_gof"].append(
                    lr.get("auroc_GOF_mean")
                )
                collect[fname][split_name]["mlp_gof"].append(mlp.get("auroc_GOF_mean"))
                print(
                    f"  seed{seed} {fname:4s} {split_name:12s} "
                    f"F1(lr={_f(lr.get('macro_f1_mean'))} "
                    f"mlp={_f(mlp.get('macro_f1_mean'))})"
                )

    out = {"chance_floor": {k: agg_seeds(v) for k, v in floor.items()}}
    for fname in feats:
        out[fname] = {}
        for split_name in ("gene_split", "family_split"):
            c = collect[fname][split_name]
            out[fname][split_name] = {
                "logreg_macro_f1": agg_seeds(c["logreg_f1"]),
                "mlp_macro_f1": agg_seeds(c["mlp_f1"]),
                "logreg_gof_auroc": agg_seeds(c["logreg_gof"]),
                "mlp_gof_auroc": agg_seeds(c["mlp_gof"]),
            }
    return out


# ── Probe C: biophysical direction on S1724 signed ddG ───────────────────────


def run_biophysical_direction(seeds):
    if not (
        os.path.exists(S1724_WT_EMB)
        and os.path.exists(S1724_MUT_EMB)
        and os.path.exists(S1724_VARIANTS)
    ):
        print(
            "\n[Probe C] S1724 embeddings not cached yet "
            "(run megascale_stability.py / result_21 first) — skipping."
        )
        return None

    print("\n" + "=" * 60)
    print("PROBE C  biophysical direction (S1724 signed ddG, protein-holdout)")
    print("=" * 60)
    from scipy.stats import spearmanr

    with open(S1724_VARIANTS) as _f:
        variants = json.load(_f)
    ddg = np.array([v["ddg"] for v in variants], dtype=np.float64)
    proteins = np.array([v["protein"] for v in variants])
    wt = np.load(S1724_WT_EMB)
    mut = np.load(S1724_MUT_EMB)
    delta = mut - wt
    n = min(len(delta), len(ddg), len(proteins))
    delta, ddg, proteins = delta[:n], ddg[:n], proteins[:n]
    feats = decompose(delta)
    mag = feats["mag"].ravel()

    # C1: global Spearman(||d||, |ddG|)
    c1_rho = float(spearmanr(mag, np.abs(ddg)).correlation)

    # C2: AUROC for sign(ddG) (destabilising vs stabilising), protein-holdout
    y_sign = (ddg > 0).astype(int)
    c2 = {}
    for fname in ("full", "dir"):
        per_seed = []
        for seed in seeds:
            splits = gene_split_cv(proteins, seed=seed)  # group-holdout by protein
            r = run_logreg_binary_cv(feats[fname], y_sign, splits, seed=seed)
            per_seed.append(r.get("auroc_mean"))
        c2[fname] = agg_seeds(per_seed)

    print(f"  C1 Spearman(||d||, |ddG|) = {c1_rho:.3f}")
    print(
        f"  C2 sign(ddG) AUROC full={_f(c2['full']['mean'])} dir={_f(c2['dir']['mean'])}"
    )
    return {
        "n_variants": int(n),
        "n_proteins": int(len(set(proteins.tolist()))),
        "frac_destabilising": float(y_sign.mean()),
        "c1_spearman_mag_absddg": c1_rho,
        "c2_sign_auroc": c2,
    }


def _f(x):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def _best(path_block, split, metric_lr, metric_mlp):
    """Best of logreg/mlp mean for a pathogenicity feature/split."""
    lr = path_block[split][metric_lr]["mean"]
    mlp = path_block[split][metric_mlp]["mean"]
    vals = [v for v in (lr, mlp) if not np.isnan(v)]
    return max(vals) if vals else float("nan")


# ── Gates ────────────────────────────────────────────────────────────────────


def evaluate_gates(path_res, mech_res, bio_res):
    gates = {}

    # P1: magnitude-only pathogenicity AUROC >= 0.85 (family-split, best probe)
    p1_val = _best(path_res["mag"], "family_split", "logreg_auroc", "mlp_auroc")
    gates["P1"] = {
        "desc": "magnitude-only pathogenicity AUROC >= 0.85 (family-split)",
        "value": p1_val,
        "threshold": P1_PATH_MAG_MIN,
        "passed": bool(p1_val >= P1_PATH_MAG_MIN),
    }

    # P2: direction-only pathogenicity AUROC <= 0.70 (family-split, best probe)
    p2_val = _best(path_res["dir"], "family_split", "logreg_auroc", "mlp_auroc")
    gates["P2"] = {
        "desc": "direction-only pathogenicity AUROC <= 0.70 (family-split)",
        "value": p2_val,
        "threshold": P2_PATH_DIR_MAX,
        "passed": bool(p2_val <= P2_PATH_DIR_MAX),
    }

    # P3: direction-only mechanism F1 <= chance_floor + 0.02 (family-split, MLP)
    floor = mech_res["chance_floor"]["family_split"]["mean"]
    p3_val = mech_res["dir"]["family_split"]["mlp_macro_f1"]["mean"]
    p3_thr = floor + P3_MECH_MARGIN
    gates["P3"] = {
        "desc": "direction-only mechanism macro-F1 <= chance_floor + 0.02 (family-split)",
        "value": p3_val,
        "chance_floor": floor,
        "threshold": p3_thr,
        "passed": bool(p3_val <= p3_thr),
    }

    # P4: S1724 sign(ddG) AUROC >= 0.65 AND Spearman(||d||,|ddG|) >= 0.30
    if bio_res is not None:
        sign_auroc = max(
            bio_res["c2_sign_auroc"]["full"]["mean"],
            bio_res["c2_sign_auroc"]["dir"]["mean"],
        )
        rho = bio_res["c1_spearman_mag_absddg"]
        gates["P4"] = {
            "desc": "S1724 sign(ddG) AUROC >= 0.65 AND Spearman >= 0.30",
            "sign_auroc": sign_auroc,
            "spearman": rho,
            "passed": bool(
                sign_auroc >= P4_SIGN_AUROC_MIN and rho >= P4_MAG_SPEARMAN_MIN
            ),
        }
    else:
        gates["P4"] = {"desc": "S1724 not cached — Probe C skipped", "passed": None}

    return gates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    with open(PFAM_JSON) as _f:
        pfam_map = json.load(_f)

    path_res = run_pathogenicity(pfam_map, args.seeds)
    mech_res = run_mechanism(pfam_map, args.seeds)
    bio_res = run_biophysical_direction(args.seeds)

    gates = evaluate_gates(path_res, mech_res, bio_res)

    # ── Headline ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("HEADLINE — magnitude vs direction (family-split)")
    print("=" * 60)

    def pa(feat, probe):
        return path_res[feat]["family_split"][probe]["mean"]

    def me(feat):
        return mech_res[feat]["family_split"]["mlp_macro_f1"]["mean"]

    print("  Pathogenicity AUROC (family-split):")
    print(
        f"    full delta   logreg={pa('full','logreg_auroc'):.3f}  mlp={pa('full','mlp_auroc'):.3f}"
    )
    print(
        f"    magnitude    logreg={pa('mag','logreg_auroc'):.3f}  mlp={pa('mag','mlp_auroc'):.3f}"
    )
    print(
        f"    direction    logreg={pa('dir','logreg_auroc'):.3f}  mlp={pa('dir','mlp_auroc'):.3f}"
    )
    print("  Mechanism macro-F1 (family-split, MLP):")
    print(f"    chance floor = {mech_res['chance_floor']['family_split']['mean']:.3f}")
    print(f"    full delta   = {me('full'):.3f}")
    print(f"    magnitude    = {me('mag'):.3f}")
    print(f"    direction    = {me('dir'):.3f}")

    print("\n" + "=" * 60)
    print("DECISION GATES")
    print("=" * 60)
    for g, d in gates.items():
        status = "SKIP" if d["passed"] is None else ("PASS" if d["passed"] else "FAIL")
        print(f"  {g}: {d['desc']}")
        print(f"       -> {status}")

    p1, p3 = gates["P1"]["passed"], gates["P3"]["passed"]
    print(
        "\n  Load-bearing (P1 AND P3):",
        (
            "PASS — magnitude carries pathogenicity, direction carries no mechanism."
            if (p1 and p3)
            else "NOT MET — see plan failure modes."
        ),
    )

    result = {
        "seeds": args.seeds,
        "pathogenicity": path_res,
        "mechanism": mech_res,
        "biophysical_direction": bio_res,
        "gates": gates,
        "thresholds": {
            "P1_path_mag_min": P1_PATH_MAG_MIN,
            "P2_path_dir_max": P2_PATH_DIR_MAX,
            "P3_mech_margin": P3_MECH_MARGIN,
            "P4_sign_auroc_min": P4_SIGN_AUROC_MIN,
            "P4_mag_spearman_min": P4_MAG_SPEARMAN_MIN,
        },
    }
    out_path = os.path.join(OUT, "probe_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults -> {out_path}")


if __name__ == "__main__":
    main()
