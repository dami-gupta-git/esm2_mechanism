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
import sys
import numpy as np
import functools
print = functools.partial(print, flush=True)

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(SCRIPTS)
DATA = os.path.join(ROOT, "data")
EMB = os.path.join(DATA, "embeddings")
OUT = os.path.join(ROOT, "results", "magnitude_direction")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, SCRIPTS)
from embeddings.esm2_mechanism import ESM2_MODEL_650M
from utils_sequences import window_sequence
# multiseed_v1 is imported lazily inside Phase-2 analysis (Phase-1 extraction does
# not need it, so the pod only needs experiment.py + fair-esm to run --extract).

VARIANTS = os.path.join(DATA, "pathogenicity_valid_variants_canonical.json")
WT_EMB = os.path.join(EMB, "emb_wt_mean_path_canonical_n16576.npy")
MUT_EMB = os.path.join(EMB, "emb_mut_mean_path_canonical_n16576.npy")
SEQS = os.path.join(DATA, "sequences.json")
CONS_CACHE = os.path.join(DATA, "conservation_pathogenicity.npy")
CONS_META = os.path.join(DATA, "conservation_pathogenicity_meta.json")

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

# Pre-registered thresholds
K1_CONS_MIN = 0.85
K2_ADD_MIN = 0.02


# ── Phase 1: masked-LL extraction (GPU) ──────────────────────────────────────

def extract_conservation(variants, seqs, batch_size=64, ckpt_every=2000):
    """Per variant: mask the WT position, read log P over 20 AAs at that token.
    Returns (N,3) array [logP_wt, logP_mut, entropy], aligned by variant index;
    NaN rows where sequence/position unavailable or WT mismatches."""
    import torch
    import esm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Phase 1 needs a GPU (CUDA). Run --extract on a GPU host; "
                           "Phase 2 analysis runs on CPU once the cache exists.")

    N = len(variants)
    out = np.full((N, 3), np.nan, dtype=np.float32)
    done = 0
    if os.path.exists(CONS_CACHE):
        cached = np.load(CONS_CACHE)
        if len(cached) == N:
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
        win, new_pos = window_sequence(seq, v["aa_pos"])
        if win[new_pos - 1] != v["aa_wt"]:  # alignment / sequence mismatch
            skipped += 1
            continue
        masked = list(win); masked[new_pos - 1] = "<mask>"
        work.append((i, "".join(masked), new_pos, v["aa_wt"], v["aa_mut"]))
    print(f"To extract: {len(work)} variants ({skipped} skipped: missing seq / pos / WT mismatch)")

    import torch as _t
    for bs in range(0, len(work), batch_size):
        batch = work[bs:bs + batch_size]
        data = [(f"v{idx}", mseq) for (idx, mseq, *_ ) in batch]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        with _t.inference_mode():
            logits = model(tokens)["logits"].cpu().float()
        for j, (idx, _mseq, new_pos, wt, mut) in enumerate(batch):
            lp = _t.log_softmax(logits[j, new_pos], dim=-1).numpy()  # BOS at 0 → token new_pos
            p20 = np.array([np.exp(lp[aa_idx[a]]) for a in AA_ORDER])
            p20 = p20 / p20.sum()
            entropy = float(-(p20 * np.log(p20 + 1e-12)).sum())
            out[idx, 0] = float(lp[aa_idx[wt]]) if wt in aa_idx else np.nan
            out[idx, 1] = float(lp[aa_idx[mut]]) if mut in aa_idx else np.nan
            out[idx, 2] = entropy
        done = int(np.isfinite(out[:, 0]).sum())
        if (bs // batch_size) % max(1, (ckpt_every // batch_size)) == 0:
            np.save(CONS_CACHE, out)
            print(f"  {done}/{N} done (checkpointed)")

    np.save(CONS_CACHE, out)
    json.dump({"n": N, "coverage": done, "features": ["logP_wt", "logP_mut", "entropy"],
               "model": ESM2_MODEL_650M}, open(CONS_META, "w"))
    print(f"Saved {CONS_CACHE}: {done}/{N} variants with conservation scores")
    return out


# ── Phase 2: analysis (CPU) ──────────────────────────────────────────────────

def auroc_family_split(X, y, genes, pfam, seeds=range(5)):
    import mechanism.multiseed_v1 as ms
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    vals = []
    for seed in seeds:
        for tr, te in ms.family_split_cv(genes, pfam, seed=seed):
            if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(sc.transform(X[tr]), y[tr])
            p = clf.predict_proba(sc.transform(X[te]))[:, list(clf.classes_).index(1)]
            vals.append(roc_auc_score(y[te], p))
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), 0.0)


def analyse():
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import spearmanr

    if not os.path.exists(CONS_CACHE):
        raise FileNotFoundError(f"{CONS_CACHE} missing — run Phase 1 first: "
                                f"python3 scripts/conservation_axis.py --extract (needs GPU)")

    import mechanism.multiseed_v1 as ms
    variants = json.load(open(VARIANTS))
    delta = np.load(MUT_EMB) - np.load(WT_EMB)
    cons = np.load(CONS_CACHE)
    pfam = json.load(open(ms.PFAM_JSON))
    genes_all = np.array([v["gene"] for v in variants])
    y_all = np.array([1 if v["label"] == "pathogenic" else 0 for v in variants])

    valid = np.isfinite(cons).all(axis=1)
    print(f"Conservation coverage: {valid.sum()}/{len(valid)} variants")
    delta, cons, genes, y = delta[valid], cons[valid], genes_all[valid], y_all[valid]

    logP_wt, logP_mut, entropy = cons[:, 0], cons[:, 1], cons[:, 2]
    masked_marginal = logP_wt - logP_mut
    cons_feats = np.column_stack([logP_wt, logP_mut, entropy, masked_marginal])

    # result_23 axis projection (direction from a single full-data logreg fit)
    Xs = StandardScaler().fit_transform(delta)
    w = LogisticRegression(max_iter=2000, C=1.0).fit(Xs, y).coef_.ravel()
    w /= (np.linalg.norm(w) + 1e-12)
    s = Xs @ w

    # K3 — correlations
    k3 = {"masked_marginal": float(spearmanr(s, masked_marginal).correlation),
          "entropy": float(spearmanr(s, entropy).correlation),
          "logP_wt": float(spearmanr(s, logP_wt).correlation)}

    # AUROCs (family-split, 5 seeds)
    print("\nRunning family-split AUROCs (5 seeds)...")
    auroc = {
        "conservation": auroc_family_split(cons_feats, y, genes, pfam),
        "delta": auroc_family_split(delta, y, genes, pfam),
        "conservation_plus_delta": auroc_family_split(np.hstack([cons_feats, delta]), y, genes, pfam),
        "masked_marginal_only": auroc_family_split(masked_marginal.reshape(-1, 1), y, genes, pfam),
    }

    cons_a = auroc["conservation"][0]
    both_a = auroc["conservation_plus_delta"][0]
    delta_a = auroc["delta"][0]
    gates = {
        "K1_conservation_is_axis": {"value": cons_a, "threshold": K1_CONS_MIN,
                                     "passed": bool(cons_a >= K1_CONS_MIN)},
        "K2_delta_beyond_conservation": {"value": both_a - cons_a, "threshold": K2_ADD_MIN,
                                          "conservation": cons_a, "conservation_plus_delta": both_a,
                                          "passed": bool(both_a - cons_a >= K2_ADD_MIN)},
        "K2b_conservation_beyond_delta": {"value": both_a - delta_a, "delta": delta_a,
                                           "conservation_plus_delta": both_a},
    }

    result = {"n_valid": int(valid.sum()), "k3_spearman_axis_vs": k3,
              "auroc_family_split": auroc, "gates": gates,
              "thresholds": {"K1": K1_CONS_MIN, "K2": K2_ADD_MIN}}
    json.dump(result, open(os.path.join(OUT, "conservation_axis.json"), "w"), indent=2)

    print("\n" + "=" * 60)
    print("CONSERVATION DECIDER")
    print("=" * 60)
    print(f"  K3 Spearman(axis, masked_marginal) = {k3['masked_marginal']:+.3f}")
    print(f"     Spearman(axis, entropy)         = {k3['entropy']:+.3f}")
    print(f"     Spearman(axis, logP_wt)         = {k3['logP_wt']:+.3f}")
    print(f"\n  AUROC (family-split):")
    for k, (m, sd) in auroc.items():
        print(f"    {k:26s} {m:.3f} ± {sd:.3f}")
    print(f"\n  K1 conservation-alone >= {K1_CONS_MIN}: {cons_a:.3f} -> "
          f"{'PASS (axis ~ conservation)' if gates['K1_conservation_is_axis']['passed'] else 'FAIL'}")
    print(f"  K2 delta adds over conservation >= {K2_ADD_MIN}: {both_a - cons_a:+.3f} -> "
          f"{'PASS (NOVEL: embedding beyond conservation)' if gates['K2_delta_beyond_conservation']['passed'] else 'FAIL (axis ~ conservation)'}")
    print(f"\nResults -> {os.path.join(OUT, 'conservation_axis.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true", help="Phase 1 masked-LL extraction (needs GPU)")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    if args.extract:
        variants = json.load(open(VARIANTS))
        seqs = json.load(open(SEQS))
        print(f"Variants: {len(variants)}  Sequences available: {len(seqs)}")
        extract_conservation(variants, seqs, batch_size=args.batch_size)

    # Phase 2 needs the cached embeddings; skip cleanly if they aren't here
    # (e.g. running --extract on a GPU pod that doesn't have the embeddings).
    if os.path.exists(MUT_EMB) and os.path.exists(CONS_CACHE):
        analyse()
    else:
        print("Skipping Phase 2 analysis: "
              f"{'embeddings' if not os.path.exists(MUT_EMB) else 'conservation cache'} not present here. "
              "Run Phase 2 where the embeddings live.")


if __name__ == "__main__":
    main()
