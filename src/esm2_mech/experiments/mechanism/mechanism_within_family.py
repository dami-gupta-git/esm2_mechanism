"""
Within-family mechanism classification from ESM-2 embeddings.

Cross-family mechanism classification is mostly family leakage: the classifier
recognises which protein family a gene belongs to, not the mechanism. This
experiment asks the narrower question — once family identity is held constant (so
it cannot act as a shortcut), can ESM-2 embeddings distinguish mechanism
(GOF/DN/LOF) *within* a single family? Per-family gene counts are tiny (6-16
genes), so each number is reported as mean +/- std across N seeds — the only
honest way to read results at these sample sizes.

Structure mirrors pathogenicity_control.py: phases run in sequence, the probe
phase loops features x probes x seeds and reports per-seed values plus mean/std.
Unlike pathogenicity_control (binary pathogenic-vs-benign), mechanism is
multiclass, so the multiclass probe runners run_logreg_cv / run_mlp_cv are used.

  Phase 1 (load)   - load_phase()
      Load ESM-2 embeddings, valid variants, and the Pfam map (same inputs as
      family_clustering.py); shape-check alignment; delta = mut - wt.

  Phase 2 (select) - select_families()
      Keep families with >= MIN_GENES genes AND >= MIN_CLASSES gene-level
      mechanism classes, largest first.

  Phase 3 (probe)  - probe_phase()
      For each family, within-family gene-split CV on two views (wt_only, delta)
      with two probes (logreg, mlp) across N seeds, plus an
      always-most-common-class baseline F1. -> WITHIN_FAMILY_MECHANISM_JSON.

  Inputs : VALID_VARIANTS_JSON, EMB_WT_MEAN, EMB_MUT_MEAN, PFAM_JSON
  Output : results/<run>/within_family_mechanism.json

Usage:
    python -m esm2_mech.experiments.mechanism.mechanism_within_family
        --seeds 5 --min-genes 6 --min-classes 2
"""

from __future__ import annotations

import argparse
import functools
from collections import Counter, defaultdict

import numpy as np

from esm2_mech.utils.bootstrap import (
    average_oof_over_seeds,
    bootstrap_mechanism_metrics,
    cluster_bootstrap_ci,
    label_permutation_pvalue,
)
from esm2_mech.utils.constants import (
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_N_RESAMPLES,
    CHANCE_AUROC,
    GOF,
    MECHANISM_CLASSES,
    N_FOLDS,
    N_SEEDS,
)
from esm2_mech.utils.data import load_variants
from esm2_mech.utils.io import atomic_write_json
from esm2_mech.utils.metrics import align_proba
from esm2_mech.utils.paths import (
    EMB_MUT_MEAN,
    EMB_WT_MEAN,
    PFAM_JSON,
    VALID_VARIANTS_JSON,
    WITHIN_FAMILY_MECHANISM_JSON,
)
from esm2_mech.utils.probes import run_logreg_cv, run_mlp_cv
from esm2_mech.utils.splits import gene_split_cv

from sklearn.metrics import roc_auc_score

import json

print = functools.partial(print, flush=True)

MIN_GENES = 6
MIN_CLASSES = 2

# Within-family fold size guards. gene_split_cv's defaults (>=10 train / >=5
# test) are tuned for the full 17k-variant dataset and would drop every fold of
# a 9-gene family, returning no result. Within a family the per-variant counts
# are small, so relax to the minimum that still lets a classifier fit: >=2
# classes worth of training rows and at least 1 test row.
MIN_TRAIN = 4
MIN_TEST = 1

# Feature views compared within each family.
VIEW_WT = "wt_only"
VIEW_DELTA = "delta"


# ===========================================================================
# Phase 1 - load and align
# ===========================================================================
def load_phase():
    """Load embeddings + variants + pfam map; align and return them.

    Returns wt_mean, delta, genes, labels (all row-aligned), and pfam_map.
    Mirrors family_clustering.py's loader exactly.
    """
    print("=== Phase 1: load ESM-2 embeddings + variants + pfam ===")
    valid_variants = load_variants(VALID_VARIANTS_JSON)
    wt_mean = np.load(EMB_WT_MEAN)
    mut_mean = np.load(EMB_MUT_MEAN)

    # Alignment by shape, not by a hardcoded count (project rule).
    if not (len(valid_variants) == wt_mean.shape[0] == mut_mean.shape[0]):
        raise ValueError(
            f"Row mismatch: {len(valid_variants)} variants in "
            f"{VALID_VARIANTS_JSON.name} vs wt {wt_mean.shape[0]} / "
            f"mut {mut_mean.shape[0]} embedding rows."
        )

    # label_3class is the gene-level GOF/DN/LOF label (same field
    # family_clustering.py uses); 'mechanism' is the raw multi-value source field.
    genes = np.array([v["gene"] for v in valid_variants])
    labels = np.array([v["label_3class"] for v in valid_variants])

    with open(PFAM_JSON) as handle:
        pfam_map = json.load(handle)

    delta = mut_mean - wt_mean
    print(
        f"  Loaded {len(valid_variants)} variants, {len(set(genes))} genes, "
        f"dim={wt_mean.shape[1]}  labels={dict(Counter(labels))}"
    )
    return wt_mean, delta, genes, labels, pfam_map


# ===========================================================================
# Phase 2 - select qualifying families
# ===========================================================================
def _gene_labels(genes, labels):
    """One mechanism label per gene; surface (not silence) any conflict."""
    gene_label = {}
    for gene, label in zip(genes, labels):
        prior = gene_label.get(gene)
        if prior is not None and prior != label:
            print(f"  WARNING: gene {gene} has conflicting labels {prior!r} vs {label!r}")
        gene_label[gene] = label
    return gene_label


def select_families(genes, labels, pfam_map, min_genes, min_classes):
    """Return ({family: gene_set}, gene_label) for families passing the gates.

    A family qualifies if it has >= min_genes distinct genes AND its genes span
    >= min_classes distinct mechanism classes (gene-level).
    pfam_map is a flat {gene: family_id_or_None} dict.
    """
    print("\n=== Phase 2: select qualifying families ===")
    gene_label = _gene_labels(genes, labels)

    family_to_genes = defaultdict(set)
    for gene in set(genes):
        family = pfam_map.get(gene)
        if family is not None:
            family_to_genes[family].add(gene)

    qualifying = {}
    for family, gene_set in family_to_genes.items():
        if len(gene_set) < min_genes:
            continue
        if len({gene_label[gene] for gene in gene_set}) < min_classes:
            continue
        qualifying[family] = gene_set

    ordered = dict(sorted(qualifying.items(), key=lambda kv: -len(kv[1])))
    print(f"  {len(ordered)} families pass (>= {min_genes} genes, >= {min_classes} classes)")
    for family, gene_set in ordered.items():
        counts = Counter(gene_label[gene] for gene in gene_set)
        print(f"    {family}: {len(gene_set)} genes  {dict(counts)}")
    return ordered, gene_label


# ===========================================================================
# Phase 3 - within-family probes
# ===========================================================================
def _majority_baseline_f1(y, classes):
    """Macro-F1 of always predicting the most common class in y."""
    majority = Counter(y).most_common(1)[0][0]
    preds = np.array([majority] * len(y))
    f1s = []
    for cls in classes:
        if (y == cls).sum() == 0:
            continue  # class absent from this family -> not part of macro avg
        tp = int(np.sum((preds == cls) & (y == cls)))
        fp = int(np.sum((preds == cls) & (y != cls)))
        fn = int(np.sum((preds != cls) & (y == cls)))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(np.mean(f1s)) if f1s else float("nan")


def _probe_one_family(
    features_by_view, y, genes_rows, classes, n_seeds, n_folds, compute_ci=True,
    mlp_kwargs=None,
):
    """Within-family gene-split CV for one family, both views x both probes, N seeds.

    Returns ({view: {probe: {"macro_f1": {mean,std,per_seed},
                             "auroc": {cls: {mean,std,per_seed}},
                             "ci": {macro_f1, auroc_<cls>: {point,ci_low,ci_high,...}}}}},
             {view: {probe: oof}}) where oof is the seed-averaged out-of-fold dict
    (one proba per variant) used both for the per-family CIs here and for the pooled
    cross-family GOF test downstream.

    Per-family CIs come from a gene-cluster bootstrap over the seed-averaged OOF: each
    variant's proba is averaged across the n_seeds CV repeats first (so a gene's rows
    are not duplicated n_seeds-fold, which would falsely narrow the interval), then
    whole genes are resampled. At 6-33 genes per family these intervals are wide and
    sometimes undefined (a resample can drop a whole class) - that is the honest read
    at these sizes, reported rather than hidden.
    """
    probes = {"logreg": run_logreg_cv, "mlp": run_mlp_cv}
    # Extra kwargs applied per probe. mlp_kwargs lets callers (tests) shrink the MLP
    # to make the many CV refits cheap; defaults leave the production probe unchanged.
    extra_kwargs = {"logreg": {}, "mlp": mlp_kwargs or {}}

    per_seed_f1 = {view: {p: [] for p in probes} for view in features_by_view}
    per_seed_auroc = {
        view: {p: defaultdict(list) for p in probes} for view in features_by_view
    }
    # Collect each seed's OOF per (view, probe) so it can be seed-averaged afterwards.
    oof_by_view_probe = {
        view: {p: [] for p in probes} for view in features_by_view
    }

    for seed in range(n_seeds):
        # Same folds across views/probes within a seed -> fair paired comparison.
        splits = gene_split_cv(
            genes_rows, n_folds=n_folds, seed=seed,
            min_train=MIN_TRAIN, min_test=MIN_TEST,
        )
        for view, feature_matrix in features_by_view.items():
            for probe_name, probe_fn in probes.items():
                res, oof = probe_fn(
                    feature_matrix, y, splits, classes=classes, seed=seed,
                    label=f"{view}:{probe_name}", genes=genes_rows, return_oof=True,
                    **extra_kwargs[probe_name],
                )
                oof_by_view_probe[view][probe_name].append(oof)
                # aggregate_folds returns flat keys: macro_f1_mean (the
                # across-fold mean for this seed) and auroc_{cls}_mean. None
                # (no usable fold) -> NaN, never 0.
                f1_seed = res.get("macro_f1_mean")
                per_seed_f1[view][probe_name].append(
                    float("nan") if f1_seed is None else f1_seed
                )
                for cls in classes:
                    auroc_seed = res.get(f"auroc_{cls}_mean")
                    per_seed_auroc[view][probe_name][cls].append(
                        float("nan") if auroc_seed is None else auroc_seed
                    )

    def summarize(values):
        clean = [val for val in values if not np.isnan(val)]
        return {
            "mean": float(np.mean(clean)) if clean else float("nan"),
            "std": float(np.std(clean)) if clean else float("nan"),
            "per_seed": values,
        }

    out = {}
    oof_out = {}
    for view in features_by_view:
        out[view] = {}
        oof_out[view] = {}
        for probe_name in probes:
            seed_avg_oof = average_oof_over_seeds(oof_by_view_probe[view][probe_name])
            oof_out[view][probe_name] = seed_avg_oof
            entry = {
                "macro_f1": summarize(per_seed_f1[view][probe_name]),
                "auroc": {
                    cls: summarize(per_seed_auroc[view][probe_name][cls])
                    for cls in classes
                },
            }
            if compute_ci and seed_avg_oof is not None:
                entry["ci"] = bootstrap_mechanism_metrics(
                    seed_avg_oof["y_true"], seed_avg_oof["proba"],
                    seed_avg_oof["genes"], classes=classes,
                    n_resamples=BOOTSTRAP_N_RESAMPLES, ci_level=BOOTSTRAP_CI_LEVEL,
                )
            out[view][probe_name] = entry
    return out, oof_out


def _gof_auroc_from_oof(oof, classes=MECHANISM_CLASSES):
    """Pooled one-vs-rest GOF AUROC from a stacked OOF dict, or None if undefined."""
    gof_col = classes.index(GOF)
    y_bin = (np.asarray(oof["y_true"]) == GOF).astype(int)
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return None
    return float(roc_auc_score(y_bin, np.asarray(oof["proba"])[:, gof_col]))


def _stack_oof(oof_list):
    """Concatenate per-family seed-averaged OOF dicts into one pooled OOF dict.

    Genes stay globally unique (gene ids are unique across families), so the stacked
    `genes` array is a valid cluster key for a gene-level bootstrap over the pool.
    """
    valid = [oof for oof in oof_list if oof is not None and len(oof["y_true"])]
    if not valid:
        return None
    return {
        "y_true": np.concatenate([oof["y_true"] for oof in valid]),
        "proba": np.concatenate([oof["proba"] for oof in valid]),
        "genes": np.concatenate([np.asarray(oof["genes"], dtype=object) for oof in valid]),
    }


def _run_delta_gof_auroc_for_labels(
    family_inputs, perm_labels_by_family, n_seeds, n_folds,
    mlp_hidden=(256, 64), mlp_max_iter=500,
):
    """Refit the delta MLP within every GOF-bearing family under given labels; pool GOF AUROC.

    Used by the permutation test: `perm_labels_by_family` supplies a (shuffled) label
    vector per family. Mirrors _probe_one_family's delta/MLP arm exactly — same folds,
    same seed-averaging — so the permuted statistic is comparable to the observed one.

    mlp_hidden / mlp_max_iter default to the production MLP; tests pass a smaller net
    to make the permutation loop's many refits cheap without altering the real run.
    """
    per_family_oof = []
    for family, inp in family_inputs.items():
        labels_fam = perm_labels_by_family[family]
        present = [cls for cls in MECHANISM_CLASSES if (labels_fam == cls).sum() > 0]
        if GOF not in present or len(present) < MIN_CLASSES:
            continue
        seed_oofs = []
        for seed in range(n_seeds):
            splits = gene_split_cv(
                inp["genes"], n_folds=n_folds, seed=seed,
                min_train=MIN_TRAIN, min_test=MIN_TEST,
            )
            _, oof = run_mlp_cv(
                inp["X"], labels_fam, splits, classes=present, seed=seed,
                genes=inp["genes"], return_oof=True, label="perm",
                hidden=mlp_hidden, max_iter=mlp_max_iter,
            )
            # Re-align proba to the canonical 3-class order so GOF column is stable.
            if oof is not None:
                oof["proba"] = align_proba(
                    oof["proba"], np.array(present), MECHANISM_CLASSES
                )
            seed_oofs.append(oof)
        per_family_oof.append(average_oof_over_seeds(seed_oofs))
    pooled = _stack_oof(per_family_oof)
    if pooled is None:
        return None
    return _gof_auroc_from_oof(pooled)


def pooled_gof_test(
    delta_oof_by_family, family_inputs, n_seeds, n_folds,
    compute_ci=True, n_permutations=0,
    mlp_hidden=(256, 64), mlp_max_iter=500,
):
    """Cross-family pooled test of the delta's GOF separability against chance.

    The per-family AUROCs (Table 2 of report_within_family.md) are too small to test
    individually at 6-33 genes. This pools the delta out-of-fold predictions across all
    GOF-bearing families and asks whether the delta ranks GOF above non-GOF better than
    chance (AUROC 0.5), with:
      - a gene-cluster bootstrap CI over the pooled OOF (genes are the resample unit),
      - an optional gene-level label-permutation p-value (refits the delta MLP per
        shuffle; slow, so opt-in via --n-permutations).
    Reports logreg and mlp separately. Returns a dict, or {"n_families": 0} if no
    family carries GOF.
    """
    print("\n=== Pooled cross-family test: delta GOF AUROC vs chance ===")
    probes = ["logreg", "mlp"]
    gof_families = [
        fam for fam, oof in delta_oof_by_family.items()
        if oof["mlp"] is not None and (np.asarray(oof["mlp"]["y_true"]) == GOF).any()
    ]
    if not gof_families:
        print("  no GOF-bearing family with scorable folds - skipping pooled test")
        return {"n_families": 0}

    out = {
        "n_families": len(gof_families),
        "families": gof_families,
        "chance_auroc": CHANCE_AUROC,
    }
    gof_col = MECHANISM_CLASSES.index(GOF)
    for probe_name in probes:
        pooled = _stack_oof([delta_oof_by_family[fam][probe_name] for fam in gof_families])
        if pooled is None:
            out[probe_name] = {"point": None}
            continue
        point = _gof_auroc_from_oof(pooled)
        probe_res = {
            "point": point,
            "n_variants": int(len(pooled["y_true"])),
            "n_genes": int(len(set(pooled["genes"].tolist()))),
            "n_gof_variants": int((np.asarray(pooled["y_true"]) == GOF).sum()),
        }
        if compute_ci:
            def _gof_auroc(rows, _pooled=pooled):
                y_bin = (np.asarray(_pooled["y_true"])[rows] == GOF).astype(int)
                if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                    return None
                return float(roc_auc_score(y_bin, np.asarray(_pooled["proba"])[rows, gof_col]))

            probe_res["ci"] = cluster_bootstrap_ci(
                pooled["genes"], _gof_auroc,
                n_resamples=BOOTSTRAP_N_RESAMPLES, ci_level=BOOTSTRAP_CI_LEVEL,
            )
        out[probe_name] = probe_res
        ci = probe_res.get("ci", {})
        print(
            f"  {probe_name}: GOF AUROC = {point:.3f}  "
            f"95% CI [{ci.get('ci_low')}, {ci.get('ci_high')}]  "
            f"(n={probe_res['n_genes']} genes, {probe_res['n_gof_variants']} GOF variants)"
        )

    # Permutation p-value (MLP only - it is the slow, headline arm). Labels are
    # shuffled at the gene level within each family, then the delta MLP is refit.
    if n_permutations > 0:
        inputs = {fam: family_inputs[fam] for fam in gof_families}
        observed = _run_delta_gof_auroc_for_labels(
            inputs, {fam: inputs[fam]["y"] for fam in gof_families}, n_seeds, n_folds,
            mlp_hidden=mlp_hidden, mlp_max_iter=mlp_max_iter,
        )

        def _run_metric(flat_labels, _inputs=inputs):
            # label_permutation_pvalue hands back one flat label vector; split it back
            # to per-family vectors in the same family order it was concatenated.
            by_family, cursor = {}, 0
            for fam in _inputs:
                size = len(_inputs[fam]["y"])
                by_family[fam] = flat_labels[cursor:cursor + size]
                cursor += size
            return _run_delta_gof_auroc_for_labels(
                _inputs, by_family, n_seeds, n_folds,
                mlp_hidden=mlp_hidden, mlp_max_iter=mlp_max_iter,
            )

        flat_labels = np.concatenate([inputs[fam]["y"] for fam in gof_families])
        flat_genes = np.concatenate([inputs[fam]["genes"] for fam in gof_families])
        perm = label_permutation_pvalue(
            _run_metric, flat_labels, groups=flat_genes,
            n_permutations=n_permutations, alternative="greater",
        )
        perm["observed"] = observed if observed is not None else perm.get("observed")
        out["permutation_mlp"] = perm
        print(
            f"  permutation: observed GOF AUROC {perm.get('observed')}  "
            f"null mean {perm.get('null_mean')}  p = {perm.get('p_value')}"
        )

    return out


def probe_phase(
    wt_mean, delta, genes, labels, families, gene_label, n_seeds, n_folds,
    compute_ci=True, n_permutations=0,
):
    """Phase 3. Within-family probes for every qualifying family, then a pooled test."""
    print("\n=== Phase 3: within-family probes ===")
    results = {
        "n_seeds": n_seeds,
        "n_folds": n_folds,
        "min_genes": MIN_GENES,
        "min_classes": MIN_CLASSES,
        "views": [VIEW_WT, VIEW_DELTA],
        "probes": ["logreg", "mlp"],
        "by_family": {},
    }

    # Keep the seed-averaged delta OOF per family so the pooled GOF test can stack it.
    delta_oof_by_family = {}
    family_inputs = {}

    for family, gene_set in families.items():
        row_mask = np.array([gene in gene_set for gene in genes])
        y = labels[row_mask]
        genes_rows = genes[row_mask]
        features_by_view = {VIEW_WT: wt_mean[row_mask], VIEW_DELTA: delta[row_mask]}

        # Restrict the class set to classes actually present in this family, so a
        # class with zero examples is never counted toward macro-F1 or AUROC.
        present_classes = [cls for cls in MECHANISM_CLASSES if (y == cls).sum() > 0]
        class_counts = Counter(gene_label[gene] for gene in gene_set)
        print(
            f"  {family}: {len(gene_set)} genes, {int(row_mask.sum())} variants  "
            f"{dict(class_counts)}"
        )

        family_res, family_oof = _probe_one_family(
            features_by_view, y, genes_rows, present_classes, n_seeds, n_folds,
            compute_ci=compute_ci,
        )
        results["by_family"][family] = {
            "n_genes": len(gene_set),
            "n_variants": int(row_mask.sum()),
            "classes": present_classes,
            "gene_class_counts": dict(class_counts),
            "majority_baseline_f1": _majority_baseline_f1(y, present_classes),
            **family_res,
        }
        delta_oof_by_family[family] = family_oof[VIEW_DELTA]
        # Inputs the permutation test needs to refit the delta probes per shuffle.
        family_inputs[family] = {
            "X": delta[row_mask],
            "y": y,
            "genes": genes_rows,
            "classes": present_classes,
        }

    results["pooled_gof"] = pooled_gof_test(
        delta_oof_by_family, family_inputs, n_seeds, n_folds,
        compute_ci=compute_ci, n_permutations=n_permutations,
    )

    WITHIN_FAMILY_MECHANISM_JSON.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(WITHIN_FAMILY_MECHANISM_JSON, results)
    print(f"  Results written to {WITHIN_FAMILY_MECHANISM_JSON}")
    return results


def _print_headline(results):
    print("\n" + "=" * 78)
    print("HEADLINE - within-family mechanism: delta vs wt_only macro-F1")
    print("=" * 78)
    for family, res in results["by_family"].items():
        base = res["majority_baseline_f1"]
        line = (
            f"  {family:9s} n={res['n_genes']:>2}g/{res['n_variants']:>4}v  "
            f"base={base:.3f}"
        )
        for probe_name in results["probes"]:
            wt_f1 = res[VIEW_WT][probe_name]["macro_f1"]
            delta_f1 = res[VIEW_DELTA][probe_name]["macro_f1"]
            line += (
                f"  | {probe_name}: wt={wt_f1['mean']:.3f}+/-{wt_f1['std']:.3f} "
                f"delta={delta_f1['mean']:.3f}+/-{delta_f1['std']:.3f}"
            )
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=N_SEEDS, help="number of seeds (>=1)")
    parser.add_argument("--min-genes", type=int, default=MIN_GENES)
    parser.add_argument("--min-classes", type=int, default=MIN_CLASSES)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument(
        "--no-ci", action="store_true",
        help="skip the gene-cluster bootstrap CIs (per-family and pooled)",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=0,
        help="label-permutation reps for the pooled GOF test "
             "(0 = skip; slow, refits the delta MLP per rep)",
    )
    args = parser.parse_args()

    if args.seeds < 1:
        parser.error("--seeds must be >= 1")

    wt_mean, delta, genes, labels, pfam_map = load_phase()
    families, gene_label = select_families(
        genes, labels, pfam_map, args.min_genes, args.min_classes
    )
    if not families:
        print("No qualifying families - nothing to probe.")
        return
    results = probe_phase(
        wt_mean, delta, genes, labels, families, gene_label,
        n_seeds=args.seeds, n_folds=args.n_folds,
        compute_ci=not args.no_ci, n_permutations=args.n_permutations,
    )
    _print_headline(results)


if __name__ == "__main__":
    main()
