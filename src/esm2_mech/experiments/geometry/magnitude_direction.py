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
  python -m esm2_mech.experiments.geometry.magnitude_direction
  python -m esm2_mech.experiments.geometry.magnitude_direction --seeds 0 1 2  # fewer seeds, faster
"""

import argparse
import json
import numpy as np
from collections import defaultdict
import functools

print = functools.partial(print, flush=True)

from joblib import Parallel, delayed

from esm2_mech.utils.io import atomic_write_json
from esm2_mech.utils.paths import (
    GEOMETRY_RESULTS_DIR,
    MAGNITUDE_DIRECTION_JSON,
    MEGASCALE_VARIANTS_JSON,
    PATH_EMB_WT_MEAN,
    PATH_EMB_MUT_MEAN,
    PATHOGENICITY_CANONICAL_VARIANTS_JSON,
    PFAM_JSON,
    MEGASCALE_EMB_WT_MEAN,
    MEGASCALE_EMB_MUT_MEAN,
)
from esm2_mech.experiments.mechanism.loaders import load_mechanism_variants
from esm2_mech.utils.metrics import mean_std_n
from esm2_mech.utils.splits import gene_split_cv, family_split_cv
from esm2_mech.utils.probes import run_mlp_binary_cv, run_mlp_probe_cv, run_logreg_cv
from esm2_mech.utils.probes import run_logreg_binary_cv

GEOMETRY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Pre-registered thresholds ────────────────────────────────────────────────
P1_PATH_MAG_MIN = 0.85  # magnitude-only pathogenicity AUROC, family-split
P2_PATH_DIR_MAX = 0.70  # direction-only pathogenicity AUROC, family-split
P3_MECH_MARGIN = 0.02  # direction-only mechanism F1 vs chance floor, family-split
P4_SIGN_AUROC_MIN = 0.65  # S1724 sign(ddG) AUROC
P4_MAG_SPEARMAN_MIN = 0.30  # S1724 Spearman(||d||, |ddG|)

# S1724 caches produced by megascale_stability.py (result_21)
S1724_WT_EMB = MEGASCALE_EMB_WT_MEAN
S1724_MUT_EMB = MEGASCALE_EMB_MUT_MEAN
S1724_VARIANTS = MEGASCALE_VARIANTS_JSON

# Canonical pathogenicity set (the one result_6's 0.884 family-split AUROC was
# computed on — n=16,576).
PATH_CANON_VARIANTS = PATHOGENICITY_CANONICAL_VARIANTS_JSON
PATH_CANON_WT_EMB = PATH_EMB_WT_MEAN
PATH_CANON_MUT_EMB = PATH_EMB_MUT_MEAN


def _pathogenicity_label(label):
    """Map a canonical-set label to 1 (pathogenic) / 0 (benign).

    Explicit lookup — never a catch-all `else 0` that could silently absorb a
    missing or unexpected label as benign.
    """
    if label == "pathogenic":
        return 1
    if label == "benign":
        return 0
    raise ValueError(f"unexpected pathogenicity label {label!r} (expected 'pathogenic'/'benign')")


def load_pathogenicity_canonical():
    """Load the canonical pathogenicity set (row-aligned to PATH_EMB_*; matches result_6)."""
    with open(PATH_CANON_VARIANTS) as _f:
        variants = json.load(_f)
    wt = np.load(PATH_CANON_WT_EMB)
    mut = np.load(PATH_CANON_MUT_EMB)
    delta = mut - wt
    # The canonical variant list is guaranteed row-aligned to the embeddings by
    # build_canonical_pathogenicity (fingerprint-checked). Assert it anyway so a
    # stale/mismatched file fails loudly rather than misaligning labels.
    if not (len(variants) == delta.shape[0]):
        raise ValueError(
            f"variant/embedding row mismatch: {len(variants)} variants vs "
            f"{delta.shape[0]} embedding rows — canonical file is not row-aligned."
        )
    genes = np.array([v["gene"] for v in variants])
    y = np.array([_pathogenicity_label(v["label"]) for v in variants])
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

# Minimum distinct classes a fold's train split must have to be scored. A
# classifier only needs two classes to fit, so a fold where a rare class (e.g.
# DN) falls entirely in test is still valid. This single constant is shared by
# the logreg probe (via run_logreg_cv), the MLP probe (already skips at < 2), and
# the chance floor below — so the probe and its baseline are averaged over the
# SAME folds (CLAUDE.md: flags and computed values must use the same condition).
MIN_TRAIN_CLASSES = 2


def run_logreg_multi(X, labels, splits, seed=42):
    return run_logreg_cv(X, labels, splits, seed=seed, min_train_classes=MIN_TRAIN_CLASSES)


def chance_floor_multi(labels, splits):
    """Stratified-random macro-F1 baseline (DummyClassifier) for the same splits.

    Skips a fold on the same MIN_TRAIN_CLASSES condition the probes use, so the
    floor and the probe are averaged over an identical fold set.
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import f1_score

    le = LabelEncoder()
    y = le.fit_transform(labels)
    f1s = []
    for tr, te in splits:
        if len(set(y[tr])) < MIN_TRAIN_CLASSES:
            continue
        d = DummyClassifier(strategy="stratified", random_state=0)
        d.fit(np.zeros((len(tr), 1)), y[tr])
        pred = d.predict(np.zeros((len(te), 1)))
        f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
    return float(np.mean(f1s)) if f1s else float("nan")


# ── Aggregation across seeds ─────────────────────────────────────────────────


def agg_seeds(per_seed_vals):
    mean, std, n = mean_std_n(per_seed_vals)
    return {"mean": mean, "std": std, "n": n}


# ── Probe A + B: pathogenicity (binary) ──────────────────────────────────────


def _pathogenicity_one_seed(seed, feats, y, genes, pfam_map):
    """All feature × split × probe AUROCs for one seed. Independent across seeds,
    so seeds are dispatched in parallel. Returns {(fname, split, probe): auroc}."""
    print(f"  [pathogenicity] seed {seed} started", flush=True)
    gs = gene_split_cv(genes, seed=seed)
    fs = family_split_cv(genes, pfam_map, seed=seed)
    res = {}
    for fname, X in feats.items():
        for split_name, splits in [("gene_split", gs), ("family_split", fs)]:
            lr = run_logreg_binary_cv(X, y, splits, seed=seed).get("auroc_mean")
            res[(fname, split_name, "logreg")] = lr
            mlp = run_mlp_binary_cv(X, y, splits, seed=seed).get("auroc_mean")
            res[(fname, split_name, "mlp")] = mlp
            print(
                f"    [pathogenicity seed {seed}] {fname:4s} {split_name:12s} "
                f"logreg={_f(lr)} mlp={_f(mlp)}",
                flush=True,
            )
    print(f"  [pathogenicity] seed {seed} done", flush=True)
    return res


def run_pathogenicity(pfam_map, seeds, n_jobs=-1):
    print("\n" + "=" * 60)
    print("PATHOGENICITY  (binary, variant-level, delta_mean)")
    print("=" * 60)
    delta, y, genes = load_pathogenicity_canonical()
    feats = decompose(delta)

    # Seeds are independent — dispatch them across cores (the per-seed MLP fits
    # are the cost). Each seed re-derives its own splits from its seed, so results
    # are deterministic regardless of worker scheduling.
    per_seed = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_pathogenicity_one_seed)(seed, feats, y, genes, pfam_map)
        for seed in seeds
    )

    collect = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for seed, res in zip(seeds, per_seed):
        for (fname, split_name, probe), auroc in res.items():
            collect[fname][split_name][probe].append(auroc)
            print(f"  seed{seed} {fname:4s} {split_name:12s} {probe}={_f(auroc)}")

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


def _mechanism_one_seed(seed, feats, labels, genes, pfam_map):
    """All feature × split probe results + chance floor for one seed (parallel)."""
    print(f"  [mechanism] seed {seed} started", flush=True)
    gs = gene_split_cv(genes, seed=seed)
    fs = family_split_cv(genes, pfam_map, seed=seed)
    floor = {}
    for split_name, splits in [("gene_split", gs), ("family_split", fs)]:
        floor[split_name] = chance_floor_multi(labels, splits)
    res = {}
    for fname, X in feats.items():
        for split_name, splits in [("gene_split", gs), ("family_split", fs)]:
            lr = run_logreg_multi(X, labels, splits, seed=seed)
            mlp = run_mlp_probe_cv(X, labels, splits, seed=seed)
            res[(fname, split_name)] = {
                "logreg_f1": lr.get("macro_f1_mean"),
                "mlp_f1": mlp.get("macro_f1_mean"),
                "logreg_gof": lr.get("auroc_GOF_mean"),
                "mlp_gof": mlp.get("auroc_GOF_mean"),
            }
            print(
                f"    [mechanism seed {seed}] {fname:4s} {split_name:12s} "
                f"F1(lr={_f(lr.get('macro_f1_mean'))} mlp={_f(mlp.get('macro_f1_mean'))})",
                flush=True,
            )
    print(f"  [mechanism] seed {seed} done", flush=True)
    return floor, res


def run_mechanism(pfam_map, seeds, n_jobs=-1):
    print("\n" + "=" * 60)
    print("MECHANISM  (3-class GOF/LOF/DN, variant-level Gerasimavicius, delta_mean)")
    print("=" * 60)
    dm, _dp, labels, genes = load_mechanism_variants(pfam_map)
    feats = decompose(dm)

    per_seed = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_mechanism_one_seed)(seed, feats, labels, genes, pfam_map)
        for seed in seeds
    )

    collect = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    floor = defaultdict(list)
    for seed, (seed_floor, res) in zip(seeds, per_seed):
        for split_name, val in seed_floor.items():
            floor[split_name].append(val)
        for (fname, split_name), cell in res.items():
            for key, val in cell.items():
                collect[fname][split_name][key].append(val)
            print(
                f"  seed{seed} {fname:4s} {split_name:12s} "
                f"F1(lr={_f(cell['logreg_f1'])} mlp={_f(cell['mlp_f1'])})"
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
        S1724_WT_EMB.exists()
        and S1724_MUT_EMB.exists()
        and S1724_VARIANTS.exists()
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


def run(n_seeds=5):
    """Run the magnitude/direction decomposition over range(n_seeds)."""
    return _run_seeds(list(range(n_seeds)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    _run_seeds(args.seeds)


def _run_seeds(seeds):
    with open(PFAM_JSON) as _f:
        pfam_map = json.load(_f)

    path_res = run_pathogenicity(pfam_map, seeds)
    mech_res = run_mechanism(pfam_map, seeds)
    bio_res = run_biophysical_direction(seeds)

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
        "seeds": list(seeds),
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
    atomic_write_json(MAGNITUDE_DIRECTION_JSON, result)
    print(f"\nResults -> {MAGNITUDE_DIRECTION_JSON}")
    return result


if __name__ == "__main__":
    main()
