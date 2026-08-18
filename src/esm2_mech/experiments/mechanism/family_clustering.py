"""Diagnostic: do ESM-2 embeddings cluster by Pfam family?

Measures k-NN family purity, within/between cosine ratio, and a family-prediction
linear probe on WT, mutant, and delta embeddings.
"""

import argparse
import functools
import json
from collections import Counter

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from esm2_mech.utils.bootstrap import (
    cluster_bootstrap_ci,
    cluster_subsample_ci,
    folds_to_arms,
    score_within_folds,
    within_stratum_bootstrap_ci,
)
from esm2_mech.utils.constants import BOOTSTRAP_N_RESAMPLES, N_SEEDS
from esm2_mech.utils.data import load_pfam_map, load_variants
from esm2_mech.utils.io import write_result_json
from esm2_mech.utils.metrics import fold_macro_f1, majority_baseline_f1, mean_std_n
from esm2_mech.utils.paths import (
    EMB_MUT_MEAN,
    EMB_WT_MEAN,
    FAMILY_CLUSTERING_JSON,
    PFAM_JSON,
    VALID_VARIANTS_JSON,
)

print = functools.partial(print, flush=True)

MIN_FAMILY_SIZE_CLUSTER = 2
MIN_FAMILY_SIZE_PROBE = 3


def gene_level_embeddings(emb, genes_arr):
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
    neighbor_idx = idx[:, 1 : k + 1]

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


def _knn_purity_bootstrap_metric(emb, families, k):
    """Metric closure for cluster_bootstrap_ci: k-NN purity, rebuilt on the
    resampled row subset. Unlike knn_family_purity's null-shuffle (which fixes
    the neighbor graph and only permutes labels — cheap, many repeats), a
    cluster-bootstrap resample changes row membership itself (with repeats), so
    the neighbor graph must be rebuilt per replicate.
    """
    families = np.asarray(families)

    def _metric(rows):
        sub_emb, sub_fam = emb[rows], families[rows]
        n = len(sub_emb)
        if n <= k:
            return None
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(sub_emb)
        _, idx = nn.kneighbors(sub_emb)
        neighbor_idx = idx[:, 1 : k + 1]
        purities = [
            sum(1 for j in neighbor_idx[i] if sub_fam[j] == sub_fam[i]) / k
            for i in range(n)
        ]
        return float(np.mean(purities))

    return _metric


def _within_between_bootstrap_metric(emb, families):
    """Metric closure for cluster_bootstrap_ci: within/between ratio, rebuilt on
    the resampled row subset (pairwise distances depend on which rows are drawn).
    """
    families = np.asarray(families)

    def _metric(rows):
        sub_emb, sub_fam = emb[rows], families[rows]
        n = len(sub_emb)
        D = cdist(sub_emb, sub_emb, metric="cosine")
        iu = np.triu_indices(n, k=1)
        fam_pair_same = np.array([sub_fam[i] == sub_fam[j] for i, j in zip(*iu)])
        d = D[iu]
        if fam_pair_same.sum() < 5 or (~fam_pair_same).sum() < 5:
            return None
        within = float(d[fam_pair_same].mean())
        between = float(d[~fam_pair_same].mean())
        return within / (between + 1e-10)

    return _metric


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
        fam_pair_same_s = np.array([shuf_fam[i] == shuf_fam[j] for i, j in zip(*iu)])
        if fam_pair_same_s.sum() < 5:
            continue
        w = d[fam_pair_same_s].mean()
        b = d[~fam_pair_same_s].mean()
        null_ratios.append(w / (b + 1e-10))
    null_mean = float(np.mean(null_ratios)) if null_ratios else float("nan")
    z = (
        (ratio - null_mean) / (np.std(null_ratios) + 1e-10)
        if null_ratios
        else float("nan")
    )
    return ratio, null_mean, float(z)


def family_probe(
    gene_emb, gene_families, seed=42, min_family_size=MIN_FAMILY_SIZE_PROBE,
    n_folds=5, return_oof=False,
):
    """Linear probe predicting Pfam family from gene-level embedding.

    Uses stratified k-fold CV so every fold sees each kept family in proportion,
    which a plain random split can fail to do for small families.
    return_oof : if True, return (result, oof) with out-of-fold test predictions
        {"y_true", "pred", "families"} (families == y_true here — the resampling
        unit for a downstream cluster bootstrap over the family probe's own
        accuracy/macro-F1), or None if no fold was scorable.
    """
    fam_counts = Counter(gene_families)
    kept = [f for f, c in fam_counts.items() if c >= min_family_size]
    mask = np.array([f in kept for f in gene_families])
    if mask.sum() < 30 or len(set(np.array(gene_families)[mask])) < 5:
        result = {"note": "not enough families with min size"}
        return (result, None) if return_oof else result
    X = gene_emb[mask]
    y = np.array(gene_families)[mask]
    classes = sorted(set(y))

    # n_splits cannot exceed the smallest kept-family size.
    min_kept_size = min(c for f, c in fam_counts.items() if f in kept)
    n_splits = min(n_folds, min_kept_size)
    if n_splits < 2:
        result = {"note": "smallest kept family too small for CV"}
        return (result, None) if return_oof else result

    _, majority_overall = majority_baseline_f1(y, y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    accs, f1s, baseline_accs = [], [], []
    oof_y, oof_pred, oof_folds = [], [], []
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        if len(set(y[train_idx])) < 2:
            continue
        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=seed)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        baseline_pred = np.full_like(y[test_idx], majority_overall)
        accs.append(float(accuracy_score(y[test_idx], pred)))
        f1s.append(
            float(f1_score(y[test_idx], pred, labels=classes, average="macro", zero_division=0))
        )
        baseline_accs.append(float(accuracy_score(y[test_idx], baseline_pred)))
        if return_oof:
            oof_y.append(y[test_idx])
            oof_pred.append(pred)
            oof_folds.append(np.full(len(test_idx), fold_i, dtype=int))

    if not accs:
        result = {"note": "all folds failed"}
        return (result, None) if return_oof else result
    result = {
        "accuracy": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1": float(np.mean(f1s)),
        "majority_baseline_acc": float(np.mean(baseline_accs)),
        "n_folds": len(accs),
        "n_genes": int(mask.sum()),
        "n_families": int(len(set(y))),
    }
    if not return_oof:
        return result
    oof = None
    if oof_y:
        y_true = np.concatenate(oof_y)
        oof = {
            "y_true": y_true,
            "pred": np.concatenate(oof_pred),
            "families": y_true,
            "folds": np.concatenate(oof_folds),
            "classes": classes,
        }
    return result, oof


def _family_probe_bootstrap_ci(oof, n_resamples, seed):
    """Bootstrap CI on the family probe's accuracy and macro-F1, from its OOF rows.

    Resamples genes inside each (family, fold) cell rather than inside each family as
    a whole. The family is this probe's prediction target, so dropping families from a
    draw changes the class set the macro average runs over and moves the value
    systematically; the reported point estimate then sits outside its own interval.
    Resampling within family alone does not fully fix this: a family's rows are spread
    across folds by the stratified split, and pooling them before resampling can, by
    chance, draw a family's rows entirely out of one of its folds, starving that fold's
    block in score_within_folds and forcing the whole resample to be discarded (or, for
    families that keep some rows in a fold, silently shifting that fold's class
    balance). Resampling within each (family, fold) cell keeps every family's per-fold
    row count fixed across draws, matching the point estimate, which is a fold mean.
    """
    y_true, pred, classes = oof["y_true"], oof["pred"], oof["classes"]
    arms = folds_to_arms(pred, oof["folds"])

    def _fold_accuracy(block, arm_pred):
        return float(accuracy_score(y_true[block], arm_pred[block]))

    def _fold_macro_f1(block, arm_pred):
        return fold_macro_f1(y_true, block, arm_pred, classes)

    def _scored(fold_fn):
        return lambda rows: score_within_folds(rows, arms, fold_fn)

    family_fold_strata = np.array(
        [f"{family}::{fold}" for family, fold in zip(oof["families"], oof["folds"])],
        dtype=object,
    )
    return {
        name: within_stratum_bootstrap_ci(
            family_fold_strata, _scored(fold_fn), n_resamples=n_resamples, seed=seed
        )
        for name, fold_fn in (("accuracy", _fold_accuracy), ("macro_f1", _fold_macro_f1))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, default=N_SEEDS,
        help="number of seeds for the k-NN purity / within-between / family-probe "
        "null-shuffles and CV folds; runs 0..seeds-1 (>=1)",
    )
    parser.add_argument("--no_ci", action="store_true", help="skip cluster-bootstrap CIs")
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")
    compute_ci = not args.no_ci

    # Load the pre-filtered variant list (the same row-aligned set the embeddings
    # were extracted from), so no rebuild/refilter is needed here.
    print("=== Loading dataset and embeddings ===")
    valid_variants = load_variants(VALID_VARIANTS_JSON)

    emb_wt = np.load(EMB_WT_MEAN)
    emb_mut = np.load(EMB_MUT_MEAN)
    emb_delta = emb_mut - emb_wt
    print(f"Variants: {len(valid_variants)}  Embedding dim: {emb_wt.shape[1]}")
    if len(valid_variants) != emb_wt.shape[0]:
        raise ValueError(
            f"Row mismatch: {len(valid_variants)} variants in {VALID_VARIANTS_JSON.name} "
            f"but {emb_wt.shape[0]} embedding rows."
        )

    genes_arr = np.array([v["gene"] for v in valid_variants])
    labels_arr = np.array([v["label_3class"] for v in valid_variants])

    # Pfam map
    pfam_map = load_pfam_map(PFAM_JSON)

    # Gene-level views (one row per gene). gene_names defines the canonical
    # per-gene ordering used by every per-view embedding below.
    gene_names, _ = gene_level_embeddings(emb_wt, genes_arr)
    gene_families = np.array([pfam_map.get(g) for g in gene_names])

    # label_3class is gene-level: every variant of a gene must share one label.
    # Enforce that before collapsing to one label per gene, so a future per-variant
    # label scheme can't silently pick an arbitrary value.
    gene_mechs = []
    for g in gene_names:
        gene_labels = set(labels_arr[genes_arr == g])
        if len(gene_labels) != 1:
            raise ValueError(f"gene {g} has multiple label_3class values: {gene_labels}")
        gene_mechs.append(gene_labels.pop())
    gene_mechs = np.array(gene_mechs)

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
    nonsingleton = np.array(
        [
            f is not None and fam_counts.get(f, 0) >= MIN_FAMILY_SIZE_CLUSTER
            for f in gene_families
        ]
    )
    print(f"\nGenes in non-singleton families: {nonsingleton.sum()}")

    for view_name, emb in [
        ("wt_mean", emb_wt),
        ("mut_mean", emb_mut),
        ("delta_mean", emb_delta),
    ]:
        print(f"\n=== {view_name} ===")
        # Aggregate per-variant embeddings to per-gene (same ordering as gene_names).
        view_gene_names, gene_emb = gene_level_embeddings(emb, genes_arr)
        assert np.array_equal(view_gene_names, gene_names)

        # Subset to annotated non-singleton families for metrics
        ge = gene_emb[nonsingleton]
        gf = gene_families[nonsingleton]

        view_res = {}

        # 1. Silhouette (seed-independent — computed once)
        if len(set(gf)) >= 2 and len(ge) >= 5:
            try:
                sil = float(silhouette_score(ge, gf, metric="cosine"))
            except Exception:
                sil = float("nan")
            view_res["silhouette_family"] = sil
            # Silhouette is unreliable here (high-dim, many singletons, uneven
            # cluster sizes); kNN purity / within-between ratio are the primary
            # signals. Reported for completeness only.
            print(f"  silhouette by family (cosine): {sil:.3f}  (unreliable here — see kNN purity)")

        # 2. kNN purity — multi-seed (the null-shuffle is seeded), plus a
        # cluster-bootstrap CI (resampled at the family level) computed once on
        # the fixed embeddings (the CI's own resampling has its own seed and does
        # not depend on the null-shuffle data-seed loop below).
        for k in (5, 10):
            if len(ge) > k:
                per_seed_p, per_seed_null, per_seed_z = [], [], []
                for seed in range(args.seeds):
                    p, null, z = knn_family_purity(ge, gf, k=k, seed=seed)
                    per_seed_p.append(p)
                    per_seed_null.append(null)
                    per_seed_z.append(z)
                p_mean, p_std, _ = mean_std_n(per_seed_p)
                null_mean, _, _ = mean_std_n(per_seed_null)
                z_mean, _, _ = mean_std_n(per_seed_z)
                view_res[f"knn{k}_purity"] = p_mean
                view_res[f"knn{k}_purity_std"] = p_std
                view_res[f"knn{k}_purity_null"] = null_mean
                view_res[f"knn{k}_purity_z"] = z_mean
                print(f"  k={k} family purity: {p_mean:.3f}±{p_std:.3f}  null {null_mean:.3f}  z={z_mean:+.1f}")
                if compute_ci:
                    # A distance/neighbor statistic, not an additive one — a
                    # with-replacement bootstrap would duplicate points and
                    # inflate purity (see cluster_subsample_ci's docstring).
                    view_res[f"knn{k}_purity_ci"] = cluster_subsample_ci(
                        gf, _knn_purity_bootstrap_metric(ge, gf, k),
                        n_resamples=args.n_boot, seed=0,
                    )

        # 3. Within/between — multi-seed null-shuffle + cluster-bootstrap CI.
        per_seed_ratio, per_seed_null, per_seed_z = [], [], []
        for seed in range(args.seeds):
            ratio, null, z = within_between_ratio(ge, gf, seed=seed)
            per_seed_ratio.append(ratio)
            per_seed_null.append(null)
            per_seed_z.append(z)
        ratio_mean, ratio_std, _ = mean_std_n(per_seed_ratio)
        null_mean, _, _ = mean_std_n(per_seed_null)
        z_mean, _, _ = mean_std_n(per_seed_z)
        view_res["within_between_ratio"] = ratio_mean
        view_res["within_between_ratio_std"] = ratio_std
        view_res["within_between_ratio_null"] = null_mean
        view_res["within_between_ratio_z"] = z_mean
        print(
            f"  within/between cosine dist ratio: {ratio_mean:.3f}±{ratio_std:.3f}  "
            f"null {null_mean:.3f}  z={z_mean:+.1f}  (<1 ⇒ within tighter than between)"
        )
        if compute_ci:
            # Pairwise-distance statistic — same duplicate-point problem as the
            # purity CI above, same fix.
            view_res["within_between_ratio_ci"] = cluster_subsample_ci(
                gf, _within_between_bootstrap_metric(ge, gf),
                n_resamples=args.n_boot, seed=0,
            )

        # 4. Family probe (gene-level) — multi-seed accuracy/macro-F1, plus a
        # cluster-bootstrap CI (resampled at the family level) from seed 0's OOF.
        per_seed_acc, per_seed_f1 = [], []
        probe = None
        probe_oof = None
        for seed in range(args.seeds):
            seed_probe, seed_oof = family_probe(
                gene_emb[annotated_mask],
                gene_families[annotated_mask].tolist(),
                seed=seed,
                return_oof=True,
            )
            if seed == 0:
                probe, probe_oof = seed_probe, seed_oof
            if "accuracy" in seed_probe:
                per_seed_acc.append(seed_probe["accuracy"])
                per_seed_f1.append(seed_probe["macro_f1"])
        if per_seed_acc:
            acc_mean, acc_std, n_seeds_used = mean_std_n(per_seed_acc)
            f1_mean, f1_std, _ = mean_std_n(per_seed_f1)
            seed0_accuracy = probe["accuracy"]
            seed0_macro_f1 = probe["macro_f1"]
            probe = {
                **probe,
                "accuracy": acc_mean,
                "accuracy_std": acc_std,
                "macro_f1": f1_mean,
                "macro_f1_std": f1_std,
                "n_seeds": n_seeds_used,
            }
            if compute_ci and probe_oof is not None:
                # This CI is a cluster bootstrap over seed 0's OOF rows only, so
                # it brackets seed 0's point estimate, not the multi-seed mean
                # above. Store both together so the mismatch can't be mistaken
                # for a CI on the 5-seed mean.
                probe["ci_seed0_point"] = {
                    "accuracy": seed0_accuracy,
                    "macro_f1": seed0_macro_f1,
                }
                probe["ci"] = _family_probe_bootstrap_ci(probe_oof, args.n_boot, seed=0)
        view_res["family_probe"] = probe
        if probe and "accuracy" in probe:
            print(
                f"  family probe accuracy: {probe['accuracy']:.3f}  "
                f"(majority baseline {probe['majority_baseline_acc']:.3f}, "
                f"{probe['n_families']} families)"
            )

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
                same = np.array(
                    [j != i and gf_list[j] == gf_list[i] for j in range(len(gf_list))]
                )
                diff = np.array(
                    [j != i and gf_list[j] != gf_list[i] for j in range(len(gf_list))]
                )
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
                    print(
                        f"  fraction of genes whose mechanism matches their family's "
                        f"majority mechanism: {frac_match:.3f}"
                    )
                    if (
                        len(per_gene_ratio) == len(mech_matches_fam)
                        and len(set(mech_matches_fam)) > 1
                    ):
                        try:
                            r, p = pearsonr(per_gene_ratio, mech_matches_fam)
                            view_res["family_tightness_vs_mech_agreement_r"] = float(r)
                            view_res["family_tightness_vs_mech_agreement_p"] = float(p)
                            print(
                                f"  Pearson r(family_tightness, mech_matches_family) "
                                f"= {r:+.3f}  p={p:.3g}"
                            )
                        except Exception:
                            pass

        results["by_view"][view_name] = view_res

    FAMILY_CLUSTERING_JSON.parent.mkdir(parents=True, exist_ok=True)
    write_result_json(
        FAMILY_CLUSTERING_JSON, results, seeds=list(range(args.seeds)),
        indent=2, default=str,
    )
    print(f"\nResults written to {FAMILY_CLUSTERING_JSON}")

    print("\n" + "=" * 60)
    print("HEADLINE")
    print("=" * 60)
    wt_view = results["by_view"]["wt_mean"]
    delta_view = results["by_view"]["delta_mean"]
    wt_knn5 = wt_view.get("knn5_purity", float("nan"))
    wt_knn5_null = wt_view.get("knn5_purity_null", float("nan"))
    wt_knn5_z = wt_view.get("knn5_purity_z", float("nan"))
    delta_knn5 = delta_view.get("knn5_purity", float("nan"))
    wt_probe_acc = wt_view.get("family_probe", {}).get("accuracy", float("nan"))
    wt_probe_base = wt_view.get("family_probe", {}).get("majority_baseline_acc", float("nan"))
    print(
        f"WT  embeddings: k=5 family purity={wt_knn5:.3f} (null {wt_knn5_null:.3f}, z={wt_knn5_z:+.1f})  "
        f"family-probe acc={wt_probe_acc:.3f} (majority {wt_probe_base:.3f})"
    )
    print(f"Δ   embeddings: k=5 family purity={delta_knn5:.3f}")
    # k=5 purity z-score is the primary signal — silhouette is unreliable in
    # high-dimensional space with uneven cluster sizes and many singletons.
    if not np.isnan(wt_knn5_z):
        if wt_knn5_z > 20:
            tag = "STRONG family clustering — gene-split CV was leaking via homology"
        elif wt_knn5_z > 5:
            tag = "MODERATE family clustering — some homology leakage in gene-split CV"
        elif wt_knn5_z > 2:
            tag = "WEAK family clustering — minor homology leakage"
        else:
            tag = "NO family clustering — gene-level signal is gene-specific, not family-driven"
        print(f"\n  ⇒ {tag}  (k=5 purity z={wt_knn5_z:+.1f})")


if __name__ == "__main__":
    main()
