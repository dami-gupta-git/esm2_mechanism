"""
Diagnostic: do ESM-2 embeddings cluster by Pfam family?

If yes, then gene-split CV in experiment.py was leaking via homology
(BRCA1 in train → BRCA2 in test gets recognized via family similarity),
inflating the WT-only baseline's apparent mechanism signal.

Computes, on WT, mutant, and delta embeddings:
  1. Silhouette score by Pfam family
  2. k-NN family purity (k=5, 10) vs shuffled-family null
  3. Within-family vs between-family cosine distance ratio vs null
  4. Linear probe predicting Pfam family from embedding (gene-split CV)
  5. Per-gene correlation: family-distance ratio vs mechanism-prediction accuracy

Usage:
    python family_clustering.py --run_dir run_0 --model esm2_t33_650M_UR50D
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

import functools
print = functools.partial(print, flush=True)

from experiment import (
    fetch_gerasimavicius_dataset,
    build_sequence_cache,
    window_sequence,
    apply_missense,
    fetch_pfam_families,
    gene_split_cv,
    ESM2_MODEL_650M,
)


def gene_level_embeddings(emb, genes_arr):
    """Average per-variant embeddings to one vector per gene."""
    unique_genes = sorted(set(genes_arr))
    gene_emb = np.zeros((len(unique_genes), emb.shape[1]), dtype=np.float32)
    for i, g in enumerate(unique_genes):
        mask = genes_arr == g
        gene_emb[i] = emb[mask].mean(0)
    return np.array(unique_genes), gene_emb


def knn_family_purity(emb, families, k=5, n_shuffles=20, seed=42):
    """For each point, fraction of its k nearest neighbors sharing its family."""
    n = len(emb)
    if n <= k:
        return float("nan"), float("nan"), float("nan")
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(emb)
    _, idx = nn.kneighbors(emb)
    # idx[:, 0] is the point itself; drop it
    neighbor_idx = idx[:, 1:k+1]

    purities = []
    for i in range(n):
        fam = families[i]
        nbrs = neighbor_idx[i]
        shared = sum(1 for j in nbrs if families[j] == fam)
        purities.append(shared / k)
    real_purity = float(np.mean(purities))

    rng = np.random.RandomState(seed)
    null_purities = []
    for _ in range(n_shuffles):
        shuf_fam = rng.permutation(families)
        ps = []
        for i in range(n):
            fam = shuf_fam[i]
            nbrs = neighbor_idx[i]
            ps.append(sum(1 for j in nbrs if shuf_fam[j] == fam) / k)
        null_purities.append(np.mean(ps))
    null_mean = float(np.mean(null_purities))
    null_std = float(np.std(null_purities))
    z = (real_purity - null_mean) / (null_std + 1e-10)
    return real_purity, null_mean, float(z)


def within_between_ratio(emb, families, n_shuffles=20, seed=42):
    """Mean within-family cosine distance / mean between-family cosine distance."""
    D = cdist(emb, emb, metric="cosine")
    n = len(emb)
    iu = np.triu_indices(n, k=1)
    fam_pair_same = np.array([families[i] == families[j] for i, j in zip(*iu)])
    d = D[iu]
    if fam_pair_same.sum() < 5 or (~fam_pair_same).sum() < 5:
        return float("nan"), float("nan"), float("nan")
    within = float(d[fam_pair_same].mean())
    between = float(d[~fam_pair_same].mean())
    ratio = within / (between + 1e-10)

    rng = np.random.RandomState(seed)
    null_ratios = []
    for _ in range(n_shuffles):
        shuf_fam = rng.permutation(families)
        fam_pair_same_s = np.array([shuf_fam[i] == shuf_fam[j]
                                     for i, j in zip(*iu)])
        if fam_pair_same_s.sum() < 5:
            continue
        w = d[fam_pair_same_s].mean()
        b = d[~fam_pair_same_s].mean()
        null_ratios.append(w / (b + 1e-10))
    null_mean = float(np.mean(null_ratios)) if null_ratios else float("nan")
    z = (ratio - null_mean) / (np.std(null_ratios) + 1e-10) if null_ratios else float("nan")
    return ratio, null_mean, float(z)


def family_probe(gene_emb, gene_families, gene_names, seed=42, min_family_size=3, n_folds=5):
    """Linear probe predicting Pfam family from gene-level embedding, using k-fold CV."""
    fam_counts = Counter(gene_families)
    kept = [f for f, c in fam_counts.items() if c >= min_family_size]
    mask = np.array([f in kept for f in gene_families])
    if mask.sum() < 30 or len(set(np.array(gene_families)[mask])) < 5:
        return {"note": "not enough families with min size"}
    X = gene_emb[mask]
    y = np.array(gene_families)[mask]

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(X))
    folds = np.array_split(order, n_folds)

    accs, f1s, baseline_accs = [], [], []
    majority_overall = Counter(y).most_common(1)[0][0]
    for k in range(n_folds):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        if len(set(y[train_idx])) < 2 or len(test_idx) < 2:
            continue
        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=seed)
        try:
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[test_idx])
            baseline_pred = np.full_like(y[test_idx], majority_overall)
            accs.append(float(accuracy_score(y[test_idx], pred)))
            f1s.append(float(f1_score(y[test_idx], pred, average="macro", zero_division=0)))
            baseline_accs.append(float(accuracy_score(y[test_idx], baseline_pred)))
        except Exception:
            continue

    if not accs:
        return {"note": "all folds failed"}
    return {
        "accuracy": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1": float(np.mean(f1s)),
        "majority_baseline_acc": float(np.mean(baseline_accs)),
        "n_folds": len(accs),
        "n_genes": int(mask.sum()),
        "n_families": int(len(set(y))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="run_0")
    parser.add_argument("--model", type=str, default=ESM2_MODEL_650M)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="family_clustering.json")
    args = parser.parse_args()

    data_dir = os.path.join(args.run_dir, "data")

    # Rebuild valid_variants identically to experiment.py
    print("=== Loading dataset and embeddings ===")
    variants = fetch_gerasimavicius_dataset(data_dir)
    for v in variants:
        v["label_3class"] = "LOF" if v["mechanism"] in ("HI", "AR") else v["mechanism"]
    variants = [v for v in variants
                if v["uniprot_id"] and v["aa_wt"] and v["aa_mut"] and v["aa_pos"] > 0]
    seq_cache = build_sequence_cache(variants, data_dir)

    valid_variants = []
    for v in variants:
        uid = v["uniprot_id"]
        if uid not in seq_cache:
            continue
        wt_full = seq_cache[uid]
        wt_win, new_pos = window_sequence(wt_full, v["aa_pos"])
        mut_win = apply_missense(wt_win, new_pos, v["aa_wt"], v["aa_mut"])
        if mut_win is None:
            continue
        valid_variants.append(v)

    emb_wt = np.load(os.path.join(data_dir, f"embeddings_wt_{args.model}.npy"))
    emb_mut = np.load(os.path.join(data_dir, f"embeddings_mut_{args.model}.npy"))
    emb_delta = emb_mut - emb_wt
    print(f"Variants: {len(valid_variants)}  Embedding dim: {emb_wt.shape[1]}")
    assert len(valid_variants) == emb_wt.shape[0]

    genes_arr = np.array([v["gene"] for v in valid_variants])
    labels_arr = np.array([v["label_3class"] for v in valid_variants])

    # Pfam map
    pfam_map = fetch_pfam_families(valid_variants, seq_cache, data_dir)

    # Gene-level views (one row per gene)
    gene_names, _ = gene_level_embeddings(emb_wt, genes_arr)
    gene_families = np.array([pfam_map.get(g) for g in gene_names])
    gene_mechs = np.array([labels_arr[genes_arr == g][0] for g in gene_names])
    annotated_mask = np.array([f is not None for f in gene_families])

    print(f"\nGenes: {len(gene_names)}  with Pfam annotation: {annotated_mask.sum()}")
    fam_counts = Counter(gene_families[annotated_mask])
    print(f"Unique Pfam families: {len(fam_counts)}")
    print(f"Top 5 families: {fam_counts.most_common(5)}")
    print(f"Singleton families: {sum(1 for f, c in fam_counts.items() if c == 1)}")

    results = {
        "n_variants": len(valid_variants),
        "n_genes": int(len(gene_names)),
        "n_annotated_genes": int(annotated_mask.sum()),
        "n_unique_families": int(len(fam_counts)),
        "n_singleton_families": int(sum(1 for f, c in fam_counts.items() if c == 1)),
        "by_view": {},
    }

    # Restrict to annotated, non-singleton families for meaningful clustering metrics
    nonsingleton = np.array([f is not None and fam_counts.get(f, 0) >= 2
                              for f in gene_families])
    print(f"\nGenes in non-singleton families: {nonsingleton.sum()}")

    for view_name, emb in [("wt_mean", emb_wt),
                            ("mut_mean", emb_mut),
                            ("delta_mean", emb_delta)]:
        print(f"\n=== {view_name} ===")
        # Aggregate per-variant embeddings to per-gene
        gene_emb = np.zeros((len(gene_names), emb.shape[1]), dtype=np.float32)
        for i, g in enumerate(gene_names):
            gene_emb[i] = emb[genes_arr == g].mean(0)

        # Subset to annotated non-singleton families for metrics
        ge = gene_emb[nonsingleton]
        gf = gene_families[nonsingleton]

        view_res = {}

        # 1. Silhouette
        if len(set(gf)) >= 2 and len(ge) >= 5:
            try:
                sil = float(silhouette_score(ge, gf, metric="cosine"))
            except Exception as e:
                sil = float("nan")
            view_res["silhouette_family"] = sil
            print(f"  silhouette by family (cosine): {sil:.3f}  "
                  f"(>0.3 strong, 0.1-0.3 moderate, <0.1 weak, <0 anti-clustered)")

        # 2. kNN purity
        for k in (5, 10):
            if len(ge) > k:
                p, null, z = knn_family_purity(ge, gf, k=k, seed=args.seed)
                view_res[f"knn{k}_purity"] = p
                view_res[f"knn{k}_purity_null"] = null
                view_res[f"knn{k}_purity_z"] = z
                print(f"  k={k} family purity: {p:.3f}  null {null:.3f}  z={z:+.1f}")

        # 3. Within/between
        ratio, null, z = within_between_ratio(ge, gf, seed=args.seed)
        view_res["within_between_ratio"] = ratio
        view_res["within_between_ratio_null"] = null
        view_res["within_between_ratio_z"] = z
        print(f"  within/between cosine dist ratio: {ratio:.3f}  "
              f"null {null:.3f}  z={z:+.1f}  (<1 ⇒ within tighter than between)")

        # 4. Family probe (gene-level)
        probe = family_probe(gene_emb[annotated_mask],
                              gene_families[annotated_mask].tolist(),
                              gene_names[annotated_mask].tolist(),
                              seed=args.seed)
        view_res["family_probe"] = probe
        if "accuracy" in probe:
            print(f"  family probe accuracy: {probe['accuracy']:.3f}  "
                  f"(majority baseline {probe['majority_baseline_acc']:.3f}, "
                  f"{probe['n_families']} families)")

        # 5. Per-gene: family-distance ratio vs mechanism-isolation
        #    For each gene, distance to same-family neighbors / distance to others.
        #    Then check whether genes with low ratio (= tightly family-clustered)
        #    also have mechanism that matches their family's majority mechanism.
        if nonsingleton.sum() >= 20:
            D = cdist(gene_emb[nonsingleton], gene_emb[nonsingleton], metric="cosine")
            gf_list = list(gf)
            gm = gene_mechs[nonsingleton]
            per_gene_ratio = []
            mech_matches_fam = []
            for i in range(len(gf_list)):
                same = np.array([j != i and gf_list[j] == gf_list[i]
                                  for j in range(len(gf_list))])
                diff = np.array([j != i and gf_list[j] != gf_list[i]
                                  for j in range(len(gf_list))])
                if same.sum() == 0 or diff.sum() == 0:
                    continue
                ratio_i = D[i, same].mean() / (D[i, diff].mean() + 1e-10)
                per_gene_ratio.append(ratio_i)
                # Does this gene's mechanism match the majority of its family?
                fam_mechs = gm[same]
                if len(fam_mechs) > 0:
                    majority = Counter(fam_mechs).most_common(1)[0][0]
                    mech_matches_fam.append(int(gm[i] == majority))
            if per_gene_ratio:
                mean_ratio = float(np.mean(per_gene_ratio))
                view_res["mean_per_gene_within_between_ratio"] = mean_ratio
                if mech_matches_fam:
                    frac_match = float(np.mean(mech_matches_fam))
                    view_res["frac_gene_mech_matches_family_majority"] = frac_match
                    print(f"  fraction of genes whose mechanism matches their family's "
                          f"majority mechanism: {frac_match:.3f}")
                    if len(per_gene_ratio) == len(mech_matches_fam) and len(set(mech_matches_fam)) > 1:
                        try:
                            r, p = pearsonr(per_gene_ratio, mech_matches_fam)
                            view_res["family_tightness_vs_mech_agreement_r"] = float(r)
                            view_res["family_tightness_vs_mech_agreement_p"] = float(p)
                            print(f"  Pearson r(family_tightness, mech_matches_family) "
                                  f"= {r:+.3f}  p={p:.3g}")
                        except Exception:
                            pass

        results["by_view"][view_name] = view_res

    out_path = os.path.join(args.run_dir, args.out)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")

    print("\n" + "=" * 60)
    print("HEADLINE")
    print("=" * 60)
    wt_sil = results["by_view"]["wt_mean"].get("silhouette_family", float("nan"))
    wt_knn5 = results["by_view"]["wt_mean"].get("knn5_purity", float("nan"))
    wt_knn5_null = results["by_view"]["wt_mean"].get("knn5_purity_null", float("nan"))
    delta_sil = results["by_view"]["delta_mean"].get("silhouette_family", float("nan"))
    delta_knn5 = results["by_view"]["delta_mean"].get("knn5_purity", float("nan"))
    print(f"WT  embeddings: silhouette={wt_sil:+.3f}  k=5 family purity={wt_knn5:.3f} (null {wt_knn5_null:.3f})")
    print(f"Δ   embeddings: silhouette={delta_sil:+.3f}  k=5 family purity={delta_knn5:.3f}")
    # Use k=5 purity z-score as primary signal — silhouette is unreliable in
    # high-dimensional space with uneven cluster sizes and many singletons.
    wt_knn5_z = results["by_view"]["wt_mean"].get("knn5_purity_z", float("nan"))
    if not np.isnan(wt_knn5_z):
        if wt_knn5_z > 20:
            tag = "STRONG family clustering — gene-split CV was leaking via homology"
        elif wt_knn5_z > 5:
            tag = "MODERATE family clustering — some homology leakage in gene-split CV"
        elif wt_knn5_z > 2:
            tag = "WEAK family clustering — minor homology leakage"
        else:
            tag = "NO family clustering — gene-level signal is gene-specific, not family-driven"
        print(f"\n  ⇒ {tag}  (k=5 purity z={wt_knn5_z:+.1f}; silhouette unreliable here)")


if __name__ == "__main__":
    main()
