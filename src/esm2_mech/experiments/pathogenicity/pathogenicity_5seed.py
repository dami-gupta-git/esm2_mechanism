"""
Re-extract pathogenicity embeddings for the canonical 16,576-variant set
and run 5-seed probe (logreg + MLP, gene-split + family-split).

Writes:
  data/embeddings/esm2_t33_650M_UR50D/pathogenicity_wt_mean.npy
  data/embeddings/esm2_t33_650M_UR50D/pathogenicity_mut_mean.npy
  results/pathogenicity_5seed/seed{N}.json
  results/pathogenicity_5seed/summary.json
"""
import json, os, sys, numpy as np
from esm2_mech.utils.paths import (
    DATA_DIR as _DATA_DIR,
    RESULTS_DIR as _RESULTS_DIR,
    PATH_EMB_WT_MEAN,
    PATH_EMB_MUT_MEAN,
    ESM2_MODEL,
)
from esm2_mech.utils.splits import gene_split_cv, family_split_cv
from esm2_mech.utils.probes import run_mlp_binary_cv as run_mlp_binary
from esm2_mech.utils.probes import run_logreg_binary_cv as run_logreg_binary

DATA = str(_DATA_DIR)
OUT  = str(_RESULTS_DIR / "pathogenicity_5seed")
os.makedirs(OUT, exist_ok=True)
import functools
print = functools.partial(print, flush=True)

CANONICAL = os.path.join(DATA, "pathogenicity_valid_variants_canonical.json")
WT_EMB  = PATH_EMB_WT_MEAN
MUT_EMB = PATH_EMB_MUT_MEAN


def main():
    with open(CANONICAL) as f:
        variants = json.load(f)
    with open(os.path.join(DATA, "pfam_families.json")) as f:
        pfam_map = json.load(f)

    print(f"Canonical variants: {len(variants)}")

    for path in [WT_EMB, MUT_EMB]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Embedding file missing: {path}\n"
                f"Run: python -m esm2_mech.embeddings.embed_variants --model {ESM2_MODEL}"
            )
    print("Loading embeddings...")
    wt  = np.load(WT_EMB)
    mut = np.load(MUT_EMB)

    delta = mut - wt
    genes = np.array([v["gene"] for v in variants])
    y     = np.array([1 if v["label"] == "pathogenic" else 0 for v in variants])
    print(f"Pathogenic: {int(y.sum())}  Benign: {int((1-y).sum())}  Genes: {len(set(genes))}")

    all_results = {}
    for seed in range(5):
        out_f = os.path.join(OUT, f"seed{seed}.json")
        if os.path.exists(out_f):
            print(f"seed {seed}: loading cached result")
            try:
                with open(out_f) as _f:
                    all_results[seed] = json.load(_f)
            except json.JSONDecodeError:
                print(
                    f"  WARNING: corrupt cached result at {out_f} — deleting and recomputing"
                )
                os.remove(out_f)
            else:
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
