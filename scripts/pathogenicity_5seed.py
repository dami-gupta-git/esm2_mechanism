"""
Re-extract pathogenicity embeddings for the canonical 16,576-variant set
and run 5-seed probe (logreg + MLP, gene-split + family-split).

Writes:
  data/embeddings/emb_wt_mean_path_canonical_n16576.npy
  data/embeddings/emb_mut_mean_path_canonical_n16576.npy
  results/pathogenicity_5seed/seed{N}.json
  results/pathogenicity_5seed/summary.json
"""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
EMB  = os.path.join(DATA, "embeddings")
OUT  = os.path.join(ROOT, "results", "pathogenicity_5seed")
os.makedirs(OUT, exist_ok=True)

from esm2_mechanism import get_esm2_embeddings_for_pairs, ESM2_MODEL_650M
from utils_sequences import window_sequence, apply_missense
from multiseed_v1 import gene_split_cv, family_split_cv, run_mlp_binary
from utils_probes import run_logreg_binary_cv as run_logreg_binary
import functools
print = functools.partial(print, flush=True)

CANONICAL = os.path.join(DATA, "pathogenicity_valid_variants_canonical.json")
WT_EMB  = os.path.join(EMB, "emb_wt_mean_path_canonical_n16576.npy")
MUT_EMB = os.path.join(EMB, "emb_mut_mean_path_canonical_n16576.npy")


def extract_embeddings(variants, seq_cache):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wt_seqs, mut_seqs, positions = [], [], []
    for v in variants:
        wt_full = seq_cache[v["uniprot_id"]]
        wt_win, new_pos = window_sequence(wt_full, v["aa_pos"])
        mut_win = apply_missense(wt_win, new_pos, v["aa_wt"], v["aa_mut"])
        wt_seqs.append(wt_win)
        mut_seqs.append(mut_win)
        positions.append(new_pos)
    print(f"Extracting embeddings for {len(wt_seqs)} pairs on {device}...")
    wt_mean, mut_mean, _, _ = get_esm2_embeddings_for_pairs(
        wt_seqs, mut_seqs, positions,
        model_name=ESM2_MODEL_650M, device=device, batch_size=128)
    return wt_mean, mut_mean


def main():
    with open(CANONICAL) as f:
        variants = json.load(f)
    with open(os.path.join(DATA, "sequences.json")) as f:
        seq_cache = json.load(f)
    with open(os.path.join(DATA, "pfam_families.json")) as f:
        pfam_map = json.load(f)

    print(f"Canonical variants: {len(variants)}")

    if os.path.exists(WT_EMB) and os.path.exists(MUT_EMB):
        print("Loading cached embeddings...")
        wt  = np.load(WT_EMB)
        mut = np.load(MUT_EMB)
    else:
        wt, mut = extract_embeddings(variants, seq_cache)
        np.save(WT_EMB, wt);  print(f"Saved {WT_EMB}")
        np.save(MUT_EMB, mut); print(f"Saved {MUT_EMB}")

    delta = mut - wt
    genes = np.array([v["gene"] for v in variants])
    y     = np.array([1 if v["label"] == "pathogenic" else 0 for v in variants])
    print(f"Pathogenic: {int(y.sum())}  Benign: {int((1-y).sum())}  Genes: {len(set(genes))}")

    all_results = {}
    for seed in range(5):
        out_f = os.path.join(OUT, f"seed{seed}.json")
        if os.path.exists(out_f):
            print(f"seed {seed}: loading cached result")
            all_results[seed] = json.load(open(out_f))
            continue
        print(f"\n=== seed {seed} ===")
        gs = gene_split_cv(genes, seed=seed)
        fs = family_split_cv(genes, pfam_map, seed=seed)
        r = {
            "logreg_gene":   run_logreg_binary(delta, y, gs, seed=seed),
            "logreg_family": run_logreg_binary(delta, y, fs, seed=seed),
            "mlp_gene":      run_mlp_binary(delta, y, gs, seed=seed),
            "mlp_family":    run_mlp_binary(delta, y, fs, seed=seed),
        }
        for k, v2 in r.items():
            print(f"  {k}: AUROC={v2.get('auroc_mean', float('nan')):.3f}")
        with open(out_f, "w") as f:
            json.dump(r, f, indent=2)
        all_results[seed] = r

    print("\n=== SUMMARY ===")
    summary = {}
    for k in ["logreg_gene", "logreg_family", "mlp_gene", "mlp_family"]:
        vals = [all_results[s][k].get("auroc_mean", float("nan")) for s in range(5)]
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "per_seed": vals}
        print(f"  {k}: {np.mean(vals):.3f} +/- {np.std(vals):.3f}  "
              f"({' '.join(f'{v:.3f}' for v in vals)})")
    deltas = [all_results[s]["mlp_gene"]["auroc_mean"] - all_results[s]["mlp_family"]["auroc_mean"]
              for s in range(5)]
    print(f"  gene->family delta: {np.mean(deltas):.3f} +/- {np.std(deltas):.3f}")

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary -> {OUT}/summary.json")


if __name__ == "__main__":
    main()
