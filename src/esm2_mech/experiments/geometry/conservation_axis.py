"""
The conservation decider (plan_conservation_axis.md → result_24).

result_23 found pathogenicity is a single, family-transferable LINEAR direction in
ESM-2 perturbation space, and Probe 4 ruled out context-free biochemistry as its
explanation. The remaining question: is that axis just ESM-2's own *conservation*
signal (masked-LM log-likelihood), or does the embedding direction carry
pathogenicity information BEYOND the model's likelihood output?

Phase 1 (GPU): for each canonical pathogenicity variant, mask the WT position and
read log P(aa|context) for all 20 AAs → logP_wt, logP_mut, entropy.
Phase 2 (CPU): compare the conservation features to the result_23 axis on the same
variants, same Pfam family-split.

Pre-registered gates (family-split AUROC, 5-seed):
  K1: conservation-alone AUROC >= 0.85            -> axis ~ conservation
  K2: AUROC(conservation+delta) - AUROC(conservation) >= 0.02
      -> embedding carries pathogenicity BEYOND conservation (NOVEL)
  K3: Spearman(axis projection, masked_marginal)  -> descriptive

Usage:
  # Phase 1 needs a GPU (RunPod / A100 / H100):
  python3 scripts/conservation_axis.py --extract
  # Phase 2 (analysis) runs anywhere once the cache exists:
  python3 scripts/conservation_axis.py
"""

import argparse
import json
import os
import numpy as np
import functools

print = functools.partial(print, flush=True)

from esm2_mech.utils.io import atomic_write_json, save_npy, load_npy_or_discard
from esm2_mech.utils.paths import (
    GEOMETRY_RESULTS_DIR,
    CONSERVATION_AXIS_JSON,
    CONSERVATION_PATHOGENICITY_NPY,
    CONSERVATION_PATHOGENICITY_META_JSON,
    PATH_EMB_WT_MEAN,
    PATH_EMB_MUT_MEAN,
    PATHOGENICITY_CANONICAL_VARIANTS_JSON,
    PFAM_JSON,
    SEQUENCES_JSON,
)
from joblib import Parallel, delayed

from esm2_mech.embeddings.embed_variants import ESM2_MODEL_650M
from esm2_mech.utils.bootstrap import (
    adjudicate_diff,
    adjudicate_level,
    average_oof_over_seeds,
    binary_auroc_cluster_bootstrap_ci,
    family_or_gene_clusters,
    paired_oof_diff,
)
from esm2_mech.utils.constants import AA_ORDER
from esm2_mech.utils.probes import run_logreg_binary_cv
from esm2_mech.utils.sequences import window_sequence
from esm2_mech.utils.splits import family_split_cv

GEOMETRY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = PATHOGENICITY_CANONICAL_VARIANTS_JSON
WT_EMB = PATH_EMB_WT_MEAN
MUT_EMB = PATH_EMB_MUT_MEAN
SEQS = SEQUENCES_JSON
CONS_CACHE = CONSERVATION_PATHOGENICITY_NPY
CONS_META = CONSERVATION_PATHOGENICITY_META_JSON

# Pre-registered thresholds
K1_CONS_MIN = 0.85
K2_ADD_MIN = 0.02

# The pathogenic class, matching _pathogenicity_label's encoding.
PATHOGENIC = 1
# The 1280-d embedding delta needs more iterations than the shared probe default.
LOGREG_MAX_ITER = 2000


# ── Phase 1: masked-LL extraction (GPU) ──────────────────────────────────────


def extract_conservation(variants, seqs, batch_size=64, ckpt_every=2000):
    """Per variant: mask the WT position, read log P over 20 AAs at that token.
    Returns (N,3) array [logP_wt, logP_mut, entropy], aligned by variant index;
    NaN rows where sequence/position unavailable or WT mismatches."""
    import torch
    import esm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError(
            "Phase 1 needs a GPU (CUDA). Run --extract on a GPU host; "
            "Phase 2 analysis runs on CPU once the cache exists."
        )

    N = len(variants)
    out = np.full((N, 3), np.nan, dtype=np.float32)
    done = 0
    if CONS_CACHE.exists():
        cached = load_npy_or_discard(CONS_CACHE)
        if cached is not None and len(cached) == N:
            out = cached
            done = int(np.isfinite(out[:, 0]).sum())
            print(f"Resuming: {done}/{N} variants already extracted")

    model, alphabet = esm.pretrained.load_model_and_alphabet(ESM2_MODEL_650M)
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    aa_idx = {aa: alphabet.get_idx(aa) for aa in AA_ORDER}

    # Build the work list (index, masked_seq, new_pos, wt_aa, mut_aa); skip done/invalid
    work = []
    skipped = 0
    for i, v in enumerate(variants):
        if np.isfinite(out[i, 0]):
            continue
        seq = seqs.get(v.get("uniprot_id"))
        if not seq or not (1 <= v["aa_pos"] <= len(seq)):
            skipped += 1
            continue
        win, new_pos, _ = window_sequence(seq, v["aa_pos"])
        if win[new_pos - 1] != v["aa_wt"]:  # alignment / sequence mismatch
            skipped += 1
            continue
        masked = list(win)
        masked[new_pos - 1] = "<mask>"
        work.append((i, "".join(masked), new_pos, v["aa_wt"], v["aa_mut"]))
    print(
        f"To extract: {len(work)} variants ({skipped} skipped: missing seq / pos / WT mismatch)"
    )

    import torch as _t

    for bs in range(0, len(work), batch_size):
        batch = work[bs : bs + batch_size]
        data = [(f"v{idx}", mseq) for (idx, mseq, *_) in batch]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        with _t.inference_mode():
            logits = model(tokens)["logits"].cpu().float()
        for j, (idx, _mseq, new_pos, wt, mut) in enumerate(batch):
            lp = _t.log_softmax(
                logits[j, new_pos], dim=-1
            ).numpy()  # BOS at 0 → token new_pos
            p20 = np.array([np.exp(lp[aa_idx[a]]) for a in AA_ORDER])
            p20 = p20 / p20.sum()
            entropy = float(-(p20 * np.log(p20 + 1e-12)).sum())
            out[idx, 0] = float(lp[aa_idx[wt]]) if wt in aa_idx else np.nan
            out[idx, 1] = float(lp[aa_idx[mut]]) if mut in aa_idx else np.nan
            out[idx, 2] = entropy
        done = int(np.isfinite(out[:, 0]).sum())
        if (bs // batch_size) % max(1, (ckpt_every // batch_size)) == 0:
            save_npy(CONS_CACHE, out)
            print(f"  {done}/{N} done (checkpointed)")

    save_npy(CONS_CACHE, out)
    atomic_write_json(
        CONS_META,
        {
            "n": N,
            "coverage": done,
            "features": ["logP_wt", "logP_mut", "entropy"],
            "model": ESM2_MODEL_650M,
        },
    )
    print(f"Saved {CONS_CACHE}: {done}/{N} variants with conservation scores")
    return out


# ── Phase 2: analysis (CPU) ──────────────────────────────────────────────────


def _pathogenicity_label(label):
    """Map a canonical-set label to 1 (pathogenic) / 0 (benign); never a catch-all."""
    if label == "pathogenic":
        return 1
    if label == "benign":
        return 0
    raise ValueError(f"unexpected pathogenicity label {label!r} (expected 'pathogenic'/'benign')")


def _oof_one_seed(X, y, genes, pfam, seed):
    """Family-split out-of-fold positive-class probabilities for one seed."""
    splits = list(family_split_cv(genes, pfam, seed=seed))
    _, oof = run_logreg_binary_cv(
        X, y, splits, seed=seed, pos_label=PATHOGENIC, genes=genes,
        return_oof=True, max_iter=LOGREG_MAX_ITER,
    )
    return oof


def auroc_family_split(X, y, genes, pfam, seeds=range(5), n_jobs=-1):
    """Family-split AUROC from seed-averaged OOF, with its family-cluster CI.

    Returns (auroc, ci, oof). The AUROC is computed once over the seed-averaged
    out-of-fold predictions rather than as a mean of per-fold AUROCs, because the
    K1/K2/C4 claims are differences between feature sets and a difference needs
    both arms scored on the same per-variant predictions to be paired. One pooled
    quantity also keeps the point estimate and its interval measuring the same
    thing. This is the averaged-OOF convention `classify_by_mechanism` and
    `esm3_mechanism` already use for the confirmatory claims.

    Seeds run in parallel (each is an independent fold set), since this is the CPU
    cost of Phase 2. Returns (nan, None, None) if no seed produced a scorable fold.
    """
    per_seed = Parallel(n_jobs=n_jobs)(
        delayed(_oof_one_seed)(X, y, genes, pfam, seed) for seed in seeds
    )
    oof = average_oof_over_seeds(per_seed)
    if oof is None:
        return float("nan"), None, None
    clusters = family_or_gene_clusters(oof["genes"], pfam, is_family_split=True)
    ci = binary_auroc_cluster_bootstrap_ci(oof, clusters=clusters, seed=0)
    return ci["point"], ci, oof


def analyse():
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import spearmanr

    if not CONS_CACHE.exists():
        raise FileNotFoundError(CONS_CACHE)

    with open(VARIANTS) as _f:
        variants = json.load(_f)
    delta = np.load(MUT_EMB) - np.load(WT_EMB)
    cons = np.load(CONS_CACHE)
    with open(PFAM_JSON) as _f:
        pfam = json.load(_f)
    genes_all = np.array([v["gene"] for v in variants])
    y_all = np.array([_pathogenicity_label(v["label"]) for v in variants])

    valid = np.isfinite(cons).all(axis=1)
    print(f"Conservation coverage: {valid.sum()}/{len(valid)} variants")
    delta, cons, genes, y = delta[valid], cons[valid], genes_all[valid], y_all[valid]

    logP_wt, logP_mut, entropy = cons[:, 0], cons[:, 1], cons[:, 2]
    masked_marginal = logP_wt - logP_mut
    cons_feats = np.column_stack([logP_wt, logP_mut, entropy, masked_marginal])

    # result_23 axis projection (direction from a single full-data logreg fit)
    Xs = StandardScaler().fit_transform(delta)
    w = LogisticRegression(max_iter=2000, C=1.0).fit(Xs, y).coef_.ravel()
    w /= np.linalg.norm(w) + 1e-12
    s = Xs @ w

    # K3 — correlations
    k3 = {
        "masked_marginal": float(spearmanr(s, masked_marginal).correlation),
        "entropy": float(spearmanr(s, entropy).correlation),
        "logP_wt": float(spearmanr(s, logP_wt).correlation),
    }

    # AUROCs (family-split, 5 seeds). Each feature set printed as it completes.
    print("\nRunning family-split AUROCs (5 seeds)...")
    feature_sets = {
        "conservation": cons_feats,
        "delta": delta,
        "conservation_plus_delta": np.hstack([cons_feats, delta]),
        "masked_marginal_only": masked_marginal.reshape(-1, 1),
    }
    auroc = {}
    auroc_ci = {}
    oof_by_feature = {}
    for name, feat in feature_sets.items():
        value, ci, oof = auroc_family_split(feat, y, genes, pfam)
        auroc[name] = value
        auroc_ci[name] = ci
        oof_by_feature[name] = oof
        if ci is None or ci.get("ci_low") is None:
            print(f"  {name:26s} AUROC = {value:.3f}  (CI suppressed)", flush=True)
        else:
            print(
                f"  {name:26s} AUROC = {value:.3f} "
                f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]",
                flush=True,
            )

    # K2, K2b and C4 are DIFFERENCES between feature sets. All four arms are scored
    # on the same variants under the same family-split folds, so each difference is
    # a same-fold paired statistic — one family resample per replicate handed to
    # both arms — not a comparison of two independent intervals.
    print("\n=== PAIRED DIFFERENCES (family-cluster bootstrap) ===")
    contrasts = [
        ("K2_delta_beyond_conservation", "conservation_plus_delta", "conservation", K2_ADD_MIN),
        ("K2b_conservation_beyond_delta", "conservation_plus_delta", "delta", 0.0),
        ("C4_conservation_vs_delta", "conservation", "delta", 0.0),
    ]
    paired = {}
    for key, arm_a, arm_b, threshold in contrasts:
        label = f"{key}: {arm_a} − {arm_b}"
        diff = paired_oof_diff(
            oof_by_feature.get(arm_a), oof_by_feature.get(arm_b), pfam, label,
            metric="auroc_binary", is_family_split=True,
        )
        if diff is None:
            continue
        paired[key] = diff
        if diff.get("ci_low") is None:
            print(f"  {label}: diff={diff['point_diff']:+.4f}  CI suppressed")
        else:
            print(
                f"  {label}: diff={diff['point_diff']:+.4f}  "
                f"[{diff['ci_low']:+.4f}, {diff['ci_high']:+.4f}]  "
                f"({diff['n_clusters']} clusters)"
            )

    cons_a = auroc["conservation"]
    both_a = auroc["conservation_plus_delta"]
    delta_a = auroc["delta"]
    k1_passed = bool(cons_a >= K1_CONS_MIN) if np.isfinite(cons_a) else None
    k2_diff = paired.get("K2_delta_beyond_conservation")
    k2_passed = (
        bool(k2_diff["point_diff"] >= K2_ADD_MIN)
        if k2_diff and k2_diff.get("point_diff") is not None
        else None
    )
    gates = {
        "K1_conservation_is_axis": {
            "value": cons_a,
            "threshold": K1_CONS_MIN,
            "ci": auroc_ci["conservation"],
            "passed": k1_passed,
            # K1 is a level claim, not a difference: it is established when the
            # interval sits clear of the threshold, not merely above it.
            "verdict": adjudicate_level(cons_a, auroc_ci["conservation"], K1_CONS_MIN),
        },
        "K2_delta_beyond_conservation": {
            "value": k2_diff["point_diff"] if k2_diff else None,
            "threshold": K2_ADD_MIN,
            "conservation": cons_a,
            "conservation_plus_delta": both_a,
            "paired_diff": k2_diff,
            "passed": k2_passed,
            "verdict": adjudicate_diff(k2_passed, k2_diff, K2_ADD_MIN),
        },
        "K2b_conservation_beyond_delta": {
            "value": (
                paired["K2b_conservation_beyond_delta"]["point_diff"]
                if "K2b_conservation_beyond_delta" in paired
                else None
            ),
            "delta": delta_a,
            "conservation_plus_delta": both_a,
            "paired_diff": paired.get("K2b_conservation_beyond_delta"),
        },
        "C4_conservation_vs_delta": {
            "conservation": cons_a,
            "delta": delta_a,
            "paired_diff": paired.get("C4_conservation_vs_delta"),
        },
    }

    result = {
        "n_valid": int(valid.sum()),
        "k3_spearman_axis_vs": k3,
        "auroc_family_split": auroc,
        "auroc_family_split_ci": auroc_ci,
        "gates": gates,
        "thresholds": {"K1": K1_CONS_MIN, "K2": K2_ADD_MIN},
        "calibration_note": (
            "The probes are uncalibrated and measure discrimination only; the "
            "reported AUROCs are not risk estimates (R7.6)."
        ),
    }
    atomic_write_json(CONSERVATION_AXIS_JSON, result)

    print("\n" + "=" * 60)
    print("CONSERVATION DECIDER")
    print("=" * 60)
    print(f"  K3 Spearman(axis, masked_marginal) = {k3['masked_marginal']:+.3f}")
    print(f"     Spearman(axis, entropy)         = {k3['entropy']:+.3f}")
    print(f"     Spearman(axis, logP_wt)         = {k3['logP_wt']:+.3f}")
    print(f"\n  AUROC (family-split, seed-averaged OOF):")
    for name, value in auroc.items():
        ci = auroc_ci.get(name)
        interval = (
            f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"
            if ci and ci.get("ci_low") is not None
            else "(CI suppressed)"
        )
        print(f"    {name:26s} {value:.3f} {interval}")
    print(
        f"\n  K1 conservation-alone >= {K1_CONS_MIN}: {cons_a:.3f} -> "
        f"{gates['K1_conservation_is_axis']['verdict']}"
    )
    k2_value = gates["K2_delta_beyond_conservation"]["value"]
    k2_shown = f"{k2_value:+.3f}" if k2_value is not None else "no point estimate"
    print(f"  K2 delta adds over conservation >= {K2_ADD_MIN}: {k2_shown}")
    print(f"     {gates['K2_delta_beyond_conservation']['verdict']}")
    print(f"\nResults -> {CONSERVATION_AXIS_JSON}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--extract",
        action="store_true",
        help="Phase 1 masked-LL extraction (needs GPU)",
    )
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    if args.extract:
        with open(VARIANTS) as _f:
            variants = json.load(_f)
        with open(SEQS) as _f:
            seqs = json.load(_f)
        print(f"Variants: {len(variants)}  Sequences available: {len(seqs)}")
        extract_conservation(variants, seqs, batch_size=args.batch_size)

    # Phase 2 needs the cached embeddings; skip cleanly if they aren't here
    # (e.g. running --extract on a GPU pod that doesn't have the embeddings).
    if MUT_EMB.exists() and CONS_CACHE.exists():
        analyse()
    else:
        print(
            "Skipping Phase 2 analysis: "
            f"{'embeddings' if not MUT_EMB.exists() else 'conservation cache'} not present here. "
            "Run Phase 2 where the embeddings live."
        )


if __name__ == "__main__":
    main()
