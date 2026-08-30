"""Diagnostic: do ESM-2 embeddings cluster by Pfam family?

Measures k-NN family purity, within/between cosine ratio, and a family-prediction
linear probe on WT, mutant, and delta embeddings.
"""

import argparse
import functools
from collections import Counter

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from esm2_mech.utils.bootstrap import (
    cluster_subsample_ci,
    folds_to_arms,
    score_within_folds,
    within_stratum_bootstrap_ci,
)
from esm2_mech.utils.classification import validate_complete_classification_splits
from esm2_mech.utils.constants import BOOTSTRAP_N_RESAMPLES, N_SEEDS
from esm2_mech.utils.seed_aggregation import (
    aggregate_seed_values,
    make_seed_record,
    read_seed_inference,
    seed_count,
)
from esm2_mech.experiments.mechanism.seed_results import aggregate_result_contract
from esm2_mech.utils.data import (
    embedding_fingerprint,
    labeled_variant_fingerprint,
    load_pfam_map,
    load_variants,
    pfam_fingerprint,
    validate_embedding_variant_identity,
)
from esm2_mech.utils.io import write_result_json
from esm2_mech.utils.metrics import (
    aggregate_folds,
    align_proba,
    compute_metrics,
    fold_macro_f1,
    majority_baseline_f1,
    null_standard_score,
)
from esm2_mech.utils.paths import (
    EMB_MUT_MEAN,
    EMB_VALID_VARIANTS_JSON,
    EMB_WT_MEAN,
    FAMILY_CLUSTERING_JSON,
    PFAM_JSON,
    VALID_VARIANTS_JSON,
)

print = functools.partial(print, flush=True)

MIN_FAMILY_SIZE_CLUSTER = 2


def _show_z(value):
    return "unavailable" if value is None else f"{value:+.1f}"


def _show_metric(value):
    return "unavailable" if value is None else f"{value:.3f}"


def _read_inference_metric(metrics, key):
    return read_seed_inference(metrics.get(key, {}))


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
    null_summary = null_standard_score(real_purity, null_purities)
    return real_purity, null_summary["null_mean"], null_summary["z_score"]


def _knn_purity_bootstrap_metric(emb, families, k):
    """Metric closure for cluster_subsample_ci: k-NN purity, rebuilt on the
    resampled row subset. Unlike knn_family_purity's null-shuffle (which fixes
    the neighbor graph and only permutes labels — cheap, many repeats), a
    resample changes row membership itself, so the neighbor graph must be
    rebuilt per replicate.
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
    """Metric closure for cluster_subsample_ci: within/between ratio, rebuilt on
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
    null_summary = null_standard_score(ratio, null_ratios)
    return ratio, null_summary["null_mean"], null_summary["z_score"]


def family_probe(
    gene_emb,
    gene_families,
    seed=42,
    min_family_size=MIN_FAMILY_SIZE_PROBE,
    n_folds=3,
    return_oof=False,
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

    min_kept_size = min(c for f, c in fam_counts.items() if f in kept)
    if min_kept_size < n_folds:
        result = {
            "status": "unscorable",
            "note": "smallest kept family has fewer rows than the declared fold count",
            "requested_folds": n_folds,
            "minimum_family_rows": min_kept_size,
        }
        return (result, None) if return_oof else result

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y))
    contract = validate_complete_classification_splits(
        splits,
        requested_folds=n_folds,
        eligible_rows=np.arange(len(y)),
        labels=y,
        classes=classes,
        groups=None,
        held_out_unit=None,
    )
    if contract["status"] != "valid":
        result = {
            "status": "unscorable",
            "note": "family-probe split validation failed",
            "split_validation": contract,
            "requested_folds": n_folds,
            "completed_folds": 0,
        }
        return (result, None) if return_oof else result

    accs, baseline_accs, fold_metrics = [], [], []
    oof_y, oof_pred, oof_folds = [], [], []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=seed)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        proba = align_proba(
            clf.predict_proba(X[test_idx]),
            clf.classes_,
            classes,
            allow_missing_classes=False,
        )
        try:
            _, fold_majority = majority_baseline_f1(y[train_idx], y[test_idx], classes)
            baseline_pred = np.full_like(y[test_idx], fold_majority)
            baseline_accs.append(float(accuracy_score(y[test_idx], baseline_pred)))
        except ValueError:
            baseline_accs.append(None)
        accs.append(float(accuracy_score(y[test_idx], pred)))
        fold_metrics.append(compute_metrics(y[test_idx], pred, proba, classes))
        if return_oof:
            oof_y.append(y[test_idx])
            oof_pred.append(pred)
            oof_folds.append(np.full(len(test_idx), fold_i, dtype=int))

    aggregate = aggregate_folds(fold_metrics, classes, n_folds)
    result = {
        **aggregate,
        "status": "success",
        "accuracy": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1": aggregate["macro_f1_mean"],
        "majority_baseline_acc": (
            None
            if any(value is None for value in baseline_accs)
            else float(np.mean(baseline_accs))
        ),
        "n_folds": n_folds,
        "requested_folds": n_folds,
        "completed_folds": n_folds,
        "eligible_rows": int(len(y)),
        "out_of_fold_rows": int(len(y)),
        "split_validation": contract,
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
            "row_ids": np.arange(len(y)),
        }
    return result, oof


def _family_probe_bootstrap_ci(oof, n_resamples, seed):
    """Bootstrap CI on the family probe's accuracy and macro-F1, from its OOF rows.

    Resamples genes inside each family rather than resampling families. The family is
    this probe's prediction target, so dropping families from a draw changes the class
    set the macro average runs over and moves the value systematically; the reported
    point estimate then sits outside its own interval. Scoring stays within fold and
    averages, matching the point estimate, which is a fold mean.
    """
    y_true, pred, classes = oof["y_true"], oof["pred"], oof["classes"]
    arms = folds_to_arms(pred, oof["folds"])

    def _fold_accuracy(block, arm_pred):
        return float(accuracy_score(y_true[block], arm_pred[block]))

    def _fold_macro_f1(block, arm_pred):
        return fold_macro_f1(y_true, block, arm_pred, classes)

    def _scored(fold_fn):
        return lambda rows: score_within_folds(rows, arms, fold_fn)

    return {
        name: within_stratum_bootstrap_ci(
            oof["families"],
            _scored(fold_fn),
            n_resamples=n_resamples,
            seed=seed,
            discard_reason="a fold's resampled rows lost every row",
        )
        for name, fold_fn in (
            ("accuracy", _fold_accuracy),
            ("macro_f1", _fold_macro_f1),
        )
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=seed_count,
        default=N_SEEDS,
        help="number of seeds for the k-NN purity / within-between / family-probe "
        "null-shuffles and CV folds; runs 0..seeds-1 (>=1)",
    )
    parser.add_argument(
        "--no_ci", action="store_true", help="skip cluster-bootstrap CIs"
    )
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    args = parser.parse_args()
    compute_ci = not args.no_ci

    # Load the pre-filtered variant list (the same row-aligned set the embeddings
    # were extracted from), so no rebuild/refilter is needed here.
    print("=== Loading dataset and embeddings ===")
    valid_variants = load_variants(VALID_VARIANTS_JSON)
    validate_embedding_variant_identity(valid_variants, EMB_VALID_VARIANTS_JSON)

    emb_wt = np.load(EMB_WT_MEAN)
    emb_mut = np.load(EMB_MUT_MEAN)
    if emb_wt.shape != emb_mut.shape:
        raise ValueError(
            f"WT/mutant embedding shape mismatch: {emb_wt.shape} vs {emb_mut.shape}"
        )
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
    input_fingerprints = {
        "labeled_variants": labeled_variant_fingerprint(valid_variants, labels_arr),
        "wt_mean_embedding": embedding_fingerprint(emb_wt),
        "mut_mean_embedding": embedding_fingerprint(emb_mut),
        "pfam_assignments": pfam_fingerprint(pfam_map, genes_arr.tolist()),
    }

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
            raise ValueError(
                f"gene {g} has multiple label_3class values: {gene_labels}"
            )
        gene_mechs.append(gene_labels.pop())
    gene_mechs = np.array(gene_mechs)

    annotated_mask = np.array([f is not None for f in gene_families])

    print(f"\nGenes: {len(gene_names)}  with Pfam annotation: {annotated_mask.sum()}")
    fam_counts = Counter(gene_families[annotated_mask])
    print(f"Unique Pfam families: {len(fam_counts)}")
    print(f"Top 5 families: {fam_counts.most_common(5)}")
    print(f"Singleton families: {sum(1 for f, c in fam_counts.items() if c == 1)}")

    results = {
        **aggregate_result_contract(),
        "input_fingerprints": input_fingerprints,
        "analysis_parameters": {
            "n_seeds": args.seeds,
            "n_bootstrap_resamples": args.n_boot if compute_ci else None,
            "ci_enabled": compute_ci,
        },
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
            print(
                f"  silhouette by family (cosine): {sil:.3f}  (unreliable here — see kNN purity)"
            )

        # 2. kNN purity — multi-seed (the null-shuffle is seeded), plus a
        # subsample CI (resampled at the gene level within the fixed family set)
        # computed once on the fixed embeddings. Its own resampling has its own
        # seed and does not depend on the null-shuffle data-seed loop below, so it
        # is stored separately from the seed aggregate: it is a resampling
        # uncertainty, not a spread across model seeds.
        for k in (5, 10):
            if len(ge) > k:
                purity_records, null_records, z_records = [], [], []
                for seed in range(args.seeds):
                    p, null_value, z = knn_family_purity(ge, gf, k=k, seed=seed)
                    purity_records.append(make_seed_record(seed, p))
                    null_records.append(make_seed_record(seed, null_value))
                    z_records.append(make_seed_record(seed, z))
                purity = aggregate_seed_values(range(args.seeds), purity_records)
                null = aggregate_seed_values(range(args.seeds), null_records)
                z_score = aggregate_seed_values(range(args.seeds), z_records)
                view_res[f"knn{k}_purity_seed_aggregate"] = purity.to_dict()
                view_res[f"knn{k}_purity_null_seed_aggregate"] = null.to_dict()
                view_res[f"knn{k}_purity_z_seed_aggregate"] = z_score.to_dict()
                purity_spread = (
                    "N/A" if purity.spread is None else f"{purity.spread:.3f}"
                )
                print(
                    f"  k={k} family purity: {_show_metric(purity.mean)}±{purity_spread}  "
                    f"null {_show_metric(null.mean)}  z={_show_z(z_score.mean)}"
                )
                if compute_ci:
                    # A distance/neighbor statistic, not an additive one — a
                    # with-replacement bootstrap would duplicate points and
                    # inflate purity (see cluster_subsample_ci's docstring).
                    view_res[f"knn{k}_purity_ci"] = cluster_subsample_ci(
                        gf,
                        _knn_purity_bootstrap_metric(ge, gf, k),
                        n_resamples=args.n_boot,
                        seed=0,
                    )

        # 3. Within/between — multi-seed null-shuffle + cluster-bootstrap CI.
        ratio_records, null_records, z_records = [], [], []
        for seed in range(args.seeds):
            ratio_value, null_value, z = within_between_ratio(ge, gf, seed=seed)
            ratio_records.append(make_seed_record(seed, ratio_value))
            null_records.append(make_seed_record(seed, null_value))
            z_records.append(make_seed_record(seed, z))
        ratio = aggregate_seed_values(range(args.seeds), ratio_records)
        null = aggregate_seed_values(range(args.seeds), null_records)
        z_score = aggregate_seed_values(range(args.seeds), z_records)
        view_res["within_between_ratio_seed_aggregate"] = ratio.to_dict()
        view_res["within_between_ratio_null_seed_aggregate"] = null.to_dict()
        view_res["within_between_ratio_z_seed_aggregate"] = z_score.to_dict()
        ratio_spread = "N/A" if ratio.spread is None else f"{ratio.spread:.3f}"
        print(
            f"  within/between cosine dist ratio: {_show_metric(ratio.mean)}±{ratio_spread}  "
            f"null {_show_metric(null.mean)}  z={_show_z(z_score.mean)}  "
            "(<1 ⇒ within tighter than between)"
        )
        if compute_ci:
            # Pairwise-distance statistic — same duplicate-point problem as the
            # purity CI above, same fix.
            view_res["within_between_ratio_ci"] = cluster_subsample_ci(
                gf,
                _within_between_bootstrap_metric(ge, gf),
                n_resamples=args.n_boot,
                seed=0,
            )

        # 4. Family probe (gene-level) — multi-seed accuracy and macro-F1, plus a
        # within-family bootstrap CI from seed 0's OOF.
        # Each probe records the seed that produced it, so nothing downstream has
        # to recover a seed's identity from its position in this list.
        per_seed_probes = []
        probe_oof = None
        for seed in range(args.seeds):
            seed_probe, seed_oof = family_probe(
                gene_emb[annotated_mask],
                gene_families[annotated_mask].tolist(),
                seed=seed,
                return_oof=True,
            )
            if seed == 0:
                probe_oof = seed_oof
            per_seed_probes.append({"seed": seed, **seed_probe})
        unavailable_seeds = [
            seed_probe["seed"]
            for seed_probe in per_seed_probes
            if seed_probe.get("status") != "success"
        ]
        if not unavailable_seeds:
            accuracy = aggregate_seed_values(
                range(args.seeds),
                [
                    make_seed_record(seed_probe["seed"], seed_probe["accuracy"])
                    for seed_probe in per_seed_probes
                ],
            )
            macro_f1 = aggregate_seed_values(
                range(args.seeds),
                [
                    make_seed_record(seed_probe["seed"], seed_probe["macro_f1"])
                    for seed_probe in per_seed_probes
                ],
            )
            majority_baseline = aggregate_seed_values(
                range(args.seeds),
                [
                    make_seed_record(
                        seed_probe["seed"],
                        seed_probe.get("majority_baseline_acc"),
                    )
                    for seed_probe in per_seed_probes
                ],
            )
            probe = {
                "status": "success",
                "accuracy_seed_aggregate": accuracy.to_dict(),
                "macro_f1_seed_aggregate": macro_f1.to_dict(),
                "majority_baseline_accuracy_seed_aggregate": (
                    majority_baseline.to_dict()
                ),
                "per_seed": per_seed_probes,
            }
            if compute_ci and probe_oof is not None:
                # This interval is a within-family bootstrap over seed 0's OOF
                # rows only, so it brackets seed 0's own point estimate, not the
                # multi-seed mean in the seed aggregates above. Seed 0's points
                # are stored beside it so the two are never read as an interval
                # on the across-seed mean or as a seed spread.
                seed0_probe = next(
                    seed_probe
                    for seed_probe in per_seed_probes
                    if seed_probe["seed"] == 0
                )
                probe["ci_seed0_point"] = {
                    "accuracy": seed0_probe["accuracy"],
                    "macro_f1": seed0_probe["macro_f1"],
                }
                probe["ci_seed0"] = _family_probe_bootstrap_ci(
                    probe_oof, args.n_boot, seed=0
                )
        else:
            probe = {
                "status": "unavailable",
                "unavailable_seeds": unavailable_seeds,
                "per_seed": per_seed_probes,
            }
        view_res["family_probe"] = probe
        accuracy_metric = _read_inference_metric(probe, "accuracy_seed_aggregate")
        f1_metric = _read_inference_metric(probe, "macro_f1_seed_aggregate")
        if accuracy_metric.available and f1_metric.available:
            print(
                f"  family probe accuracy: {accuracy_metric.value:.3f}±"
                f"{accuracy_metric.spread:.3f}  macro-F1={f1_metric.value:.3f}±"
                f"{f1_metric.spread:.3f}"
            )
        elif probe:
            print("  family probe: Unscorable")

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
        FAMILY_CLUSTERING_JSON,
        results,
        seeds=list(range(args.seeds)),
        indent=2,
        default=str,
    )
    print(f"\nResults written to {FAMILY_CLUSTERING_JSON}")

    print("\n" + "=" * 60)
    print("HEADLINE")
    print("=" * 60)
    wt_view = results["by_view"]["wt_mean"]
    delta_view = results["by_view"]["delta_mean"]
    wt_knn5 = _read_inference_metric(wt_view, "knn5_purity_seed_aggregate")
    wt_knn5_null = _read_inference_metric(wt_view, "knn5_purity_null_seed_aggregate")
    wt_knn5_z = _read_inference_metric(wt_view, "knn5_purity_z_seed_aggregate")
    delta_knn5 = _read_inference_metric(delta_view, "knn5_purity_seed_aggregate")
    wt_probe_acc = _read_inference_metric(
        wt_view["family_probe"], "accuracy_seed_aggregate"
    )
    wt_probe_base = _read_inference_metric(
        wt_view["family_probe"], "majority_baseline_accuracy_seed_aggregate"
    )
    wt_probe_acc_text = (
        "unavailable" if not wt_probe_acc.available else f"{wt_probe_acc.value:.3f}"
    )
    wt_probe_base_text = (
        "unavailable" if not wt_probe_base.available else f"{wt_probe_base.value:.3f}"
    )
    if not all(
        metric.available for metric in (wt_knn5, wt_knn5_null, wt_knn5_z, delta_knn5)
    ):
        print("Family-clustering headline is unavailable")
        return
    print(
        f"WT  embeddings: k=5 family purity={wt_knn5.value:.3f} "
        f"(null {wt_knn5_null.value:.3f}, z={wt_knn5_z.value:+.1f})  "
        f"family-probe acc={wt_probe_acc_text} (majority {wt_probe_base_text})"
    )
    print(f"Δ   embeddings: k=5 family purity={delta_knn5.value:.3f}")
    # k=5 purity z-score is the primary signal — silhouette is unreliable in
    # high-dimensional space with uneven cluster sizes and many singletons.
    if wt_knn5_z.available:
        if wt_knn5_z.value > 20:
            tag = "STRONG family clustering — gene-split CV was leaking via homology"
        elif wt_knn5_z.value > 5:
            tag = "MODERATE family clustering — some homology leakage in gene-split CV"
        elif wt_knn5_z.value > 2:
            tag = "WEAK family clustering — minor homology leakage"
        else:
            tag = "NO family clustering — gene-level signal is gene-specific, not family-driven"
        print(f"\n  ⇒ {tag}  (k=5 purity z={wt_knn5_z.value:+.1f})")


if __name__ == "__main__":
    main()
