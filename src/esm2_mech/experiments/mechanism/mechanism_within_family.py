"""Within-family mechanism classification from ESM-2 embeddings.

Holds family identity constant and asks whether ESM-2 embeddings distinguish
GOF/DN/LOF within a single Pfam family via gene-split CV.
"""

from __future__ import annotations

import argparse
import functools
from collections import Counter, defaultdict

import numpy as np

from esm2_mech.utils.bootstrap import (
    INTERVAL_GATE_REASON,
    attach_mechanism_ci,
    label_permutation_pvalue,
    oof_score_arms,
    score_within_folds,
)
from esm2_mech.utils.seed_aggregation import aggregate_oof_dicts
from esm2_mech.utils.constants import (
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_N_RESAMPLES,
    CHANCE_AUROC,
    GOF,
    MECHANISM_CLASSES,
    N_FOLDS,
    N_SEEDS,
)
from esm2_mech.utils.data import load_variants, validate_embedding_variant_identity
from esm2_mech.utils.io import write_result_json
from esm2_mech.utils.metrics import align_proba, majority_baseline_f1
from esm2_mech.utils.paths import (
    EMB_MUT_MEAN,
    EMB_VALID_VARIANTS_JSON,
    EMB_WT_MEAN,
    PFAM_JSON,
    VALID_VARIANTS_JSON,
    WITHIN_FAMILY_MECHANISM_JSON,
)
from esm2_mech.utils.probes import run_logreg_cv, run_mlp_cv
from esm2_mech.utils.splits import fold_index_array, gene_split_cv
from esm2_mech.utils.classification import validate_classification_splits

from sklearn.metrics import roc_auc_score

import json

print = functools.partial(print, flush=True)

MIN_GENES = 6
MIN_CLASSES = 2

VIEW_WT = "wt_only"
VIEW_DELTA = "delta"


def load_phase():
    """Load embeddings, variants, and pfam map; align and return them."""
    print("=== Phase 1: load ESM-2 embeddings + variants + pfam ===")
    valid_variants = load_variants(VALID_VARIANTS_JSON)
    validate_embedding_variant_identity(valid_variants, EMB_VALID_VARIANTS_JSON)
    wt_mean = np.load(EMB_WT_MEAN)
    mut_mean = np.load(EMB_MUT_MEAN)

    if not (len(valid_variants) == wt_mean.shape[0] == mut_mean.shape[0]):
        raise ValueError(
            f"Row mismatch: {len(valid_variants)} variants in "
            f"{VALID_VARIANTS_JSON.name} vs wt {wt_mean.shape[0]} / "
            f"mut {mut_mean.shape[0]} embedding rows."
        )

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
    """Return ({family: gene_set}, gene_label) for families passing size and class-count gates."""
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


def _probe_one_family(
    features_by_view, y, genes_rows, classes, n_seeds, n_folds, compute_ci=True,
    mlp_kwargs=None,
):
    """Within-family gene-split CV for one family, both views x both probes, N seeds."""
    probes = {"logreg": run_logreg_cv, "mlp": run_mlp_cv}
    extra_kwargs = {"logreg": {}, "mlp": mlp_kwargs or {}}

    per_seed_f1 = {view: {p: [] for p in probes} for view in features_by_view}
    per_seed_auroc = {
        view: {p: defaultdict(list) for p in probes} for view in features_by_view
    }
    oof_by_view_probe = {
        view: {p: {} for p in probes} for view in features_by_view
    }

    for seed in range(n_seeds):
        splits = gene_split_cv(genes_rows, n_folds=n_folds, seed=seed)
        split_contract = validate_classification_splits(
            splits,
            requested_folds=n_folds,
            eligible_rows=np.concatenate([test for _train, test in splits]),
            labels=y,
            classes=classes,
            required_train_classes=None,
            required_test_classes=None,
            allow_missing_classifier_classes=True,
            minimum_train_classes=2,
            groups=genes_rows,
            held_out_unit="gene",
        )
        for view, feature_matrix in features_by_view.items():
            for probe_name, probe_fn in probes.items():
                res, oof = probe_fn(
                    feature_matrix, y, splits, classes, split_contract, seed=seed,
                    label=f"{view}:{probe_name}", genes=genes_rows, return_oof=True,
                    **extra_kwargs[probe_name],
                )
                oof_by_view_probe[view][probe_name][seed] = oof
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
        unavailable = [index for index, value in enumerate(values) if np.isnan(value)]
        return {
            "mean": None if unavailable else float(np.mean(values)),
            "std": None if unavailable else float(np.std(values)),
            "per_seed": values,
            "missing": bool(unavailable),
            "unavailable_seeds": unavailable,
        }

    out = {}
    oof_out = {}
    for view in features_by_view:
        out[view] = {}
        oof_out[view] = {}
        for probe_name in probes:
            combined_result = aggregate_oof_dicts(
                range(n_seeds),
                oof_by_view_probe[view][probe_name],
                declared_row_ids=np.arange(len(y)),
                declared_labels=y,
                declared_clusters=genes_rows,
                class_order=classes,
                declared_fold_ids=range(n_folds),
            )
            all_seeds_scorable = combined_result.available
            seed_avg_oof = combined_result.payload
            stacked_oof = seed_avg_oof if compute_ci else None
            oof_out[view][probe_name] = seed_avg_oof
            entry = {
                "status": "success" if all_seeds_scorable else "unavailable",
                "macro_f1": summarize(per_seed_f1[view][probe_name]),
                "auroc": {
                    cls: summarize(per_seed_auroc[view][probe_name][cls])
                    for cls in classes
                },
            }
            attach_mechanism_ci(
                entry,
                stacked_oof,
                stacked_oof["genes"] if stacked_oof is not None else None,
                compute_ci=compute_ci,
                classes=classes,
                n_resamples=BOOTSTRAP_N_RESAMPLES,
                ci_level=BOOTSTRAP_CI_LEVEL,
            )
            out[view][probe_name] = entry
    return out, oof_out


def _gof_auroc_from_oof(oof, classes=MECHANISM_CLASSES):
    """Mean one-vs-rest GOF AUROC across fitted seed/fold blocks."""
    gof_col = classes.index(GOF)
    y_true = np.asarray(oof["y_true"])
    arms = oof_score_arms(oof, "within-family pooled GOF AUROC")

    def _fold_auroc(block, probabilities):
        y_binary = (y_true[block] == GOF).astype(int)
        if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
            return None
        return float(roc_auc_score(y_binary, probabilities[block, gof_col]))

    return score_within_folds(np.arange(len(y_true)), arms, _fold_auroc)


def _stack_oof(oof_list):
    """Concatenate families while preserving each fitted seed/fold block."""
    if not oof_list or any(oof is None for oof in oof_list):
        return None
    valid = [oof for oof in oof_list if len(oof["y_true"])]
    if len(valid) != len(oof_list):
        return None
    output = {
        "y_true": np.concatenate([oof["y_true"] for oof in valid]),
        "genes": np.concatenate([np.asarray(oof["genes"], dtype=object) for oof in valid]),
        "row_ids": np.arange(sum(len(oof["y_true"]) for oof in valid)),
    }
    if all("oof_by_seed" in oof for oof in valid):
        requested_seed_sets = {tuple(oof["requested_seeds"]) for oof in valid}
        if len(requested_seed_sets) != 1:
            raise ValueError("families have different requested OOF seeds")
        requested_seeds = requested_seed_sets.pop()
        output["requested_seeds"] = list(requested_seeds)
        output["oof_by_seed"] = {}
        for seed in requested_seeds:
            proba = np.concatenate(
                [oof["oof_by_seed"][seed]["proba"] for oof in valid]
            )
            fold_blocks = []
            fold_offset = 0
            for oof in valid:
                folds = np.asarray(oof["oof_by_seed"][seed]["folds"], dtype=int)
                fold_blocks.append(folds + fold_offset)
                fold_offset += int(folds.max()) + 1
            output["oof_by_seed"][seed] = {
                "seed": seed,
                "proba": proba,
                "folds": np.concatenate(fold_blocks),
            }
        return output
    if not all("proba" in oof and "folds" in oof for oof in valid):
        raise KeyError("pooled OOF inputs must retain folds and probabilities")
    output["proba"] = np.concatenate([oof["proba"] for oof in valid])
    fold_blocks = []
    fold_offset = 0
    for oof in valid:
        folds = np.asarray(oof["folds"], dtype=int)
        fold_blocks.append(folds + fold_offset)
        fold_offset += int(folds.max()) + 1
    output["folds"] = np.concatenate(fold_blocks)
    return output


def _run_delta_gof_auroc_for_labels(
    family_inputs, perm_labels_by_family, n_seeds, n_folds,
    mlp_hidden=(256, 64), mlp_max_iter=500,
):
    """Refit the delta MLP within every GOF-bearing family under given labels; pool GOF AUROC."""
    per_family_oof = []
    for family, inp in family_inputs.items():
        labels_fam = perm_labels_by_family[family]
        present = [cls for cls in MECHANISM_CLASSES if (labels_fam == cls).sum() > 0]
        if GOF not in present or len(present) < MIN_CLASSES:
            continue
        seed_oofs = []
        for seed in range(n_seeds):
            splits = gene_split_cv(inp["genes"], n_folds=n_folds, seed=seed)
            split_contract = validate_classification_splits(
                splits,
                requested_folds=n_folds,
                eligible_rows=np.concatenate([test for _train, test in splits]),
                labels=labels_fam,
                classes=present,
                required_train_classes=None,
                required_test_classes=None,
                allow_missing_classifier_classes=True,
                minimum_train_classes=2,
                groups=inp["genes"],
                held_out_unit="gene",
            )
            _, oof = run_mlp_cv(
                inp["X"], labels_fam, splits, present, split_contract, seed=seed,
                genes=inp["genes"], return_oof=True, label="perm",
                hidden=mlp_hidden, max_iter=mlp_max_iter,
            )
            if oof is not None:
                oof["proba"] = align_proba(
                    oof["proba"],
                    np.array(present),
                    MECHANISM_CLASSES,
                    allow_missing_classes=True,
                )
                oof["classes"] = list(MECHANISM_CLASSES)
            seed_oofs.append(oof)
        combined = aggregate_oof_dicts(
            range(n_seeds),
            {seed: oof for seed, oof in enumerate(seed_oofs)},
            declared_row_ids=np.arange(len(labels_fam)),
            declared_labels=labels_fam,
            declared_clusters=inp["genes"],
            class_order=MECHANISM_CLASSES,
            declared_fold_ids=range(n_folds),
        )
        per_family_oof.append(combined.payload)
    pooled = _stack_oof(per_family_oof)
    if pooled is None:
        return None
    return _gof_auroc_from_oof(pooled)


def pooled_gof_test(
    delta_oof_by_family, family_inputs, n_seeds, n_folds,
    compute_ci=True, n_permutations=0,
    mlp_hidden=(256, 64), mlp_max_iter=500,
):
    """Pool delta OOF across GOF-bearing families and test GOF AUROC against chance."""
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
            probe_res["ci"] = {
                "point": point,
                "ci_low": None,
                "ci_high": None,
                "missing": True,
                "reason": INTERVAL_GATE_REASON,
                "n_resamples": 0,
                "n_resamples_total": 0,
                "n_clusters": probe_res["n_genes"],
            }
        out[probe_name] = probe_res
        ci = probe_res.get("ci", {})
        if ci.get("ci_low") is not None and ci.get("ci_high") is not None:
            ci_str = f"95% CI [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"
        else:
            ci_str = "95% CI suppressed (too few valid resamples)"
        if point is None:
            print(
                f"  {probe_name}: GOF AUROC = Unscorable "
                "(at least one required fold has a constant GOF target)"
            )
        else:
            print(
                f"  {probe_name}: GOF AUROC = {point:.3f}  {ci_str}  "
                f"(n={probe_res['n_genes']} genes, "
                f"{probe_res['n_gof_variants']} GOF variants)"
            )

    if n_permutations > 0 and out["mlp"]["point"] is None:
        out["permutation_mlp"] = {
            "observed": None,
            "p_value": None,
            "missing": True,
            "reason": "observed_fold_aware_gof_auroc_unavailable",
        }
    elif n_permutations > 0:
        inputs = {fam: family_inputs[fam] for fam in gof_families}
        observed = _run_delta_gof_auroc_for_labels(
            inputs, {fam: inputs[fam]["y"] for fam in gof_families}, n_seeds, n_folds,
            mlp_hidden=mlp_hidden, mlp_max_iter=mlp_max_iter,
        )

        def _run_metric(flat_labels, _inputs=inputs):
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
        flat_folds = []
        fold_offset = 0
        for family in gof_families:
            family_splits = gene_split_cv(
                inputs[family]["genes"], n_folds=n_folds, seed=0
            )
            family_folds = fold_index_array(
                family_splits, len(inputs[family]["genes"])
            )
            flat_folds.append(family_folds + fold_offset)
            fold_offset += n_folds
        perm = label_permutation_pvalue(
            _run_metric, flat_labels, statistic="auroc_GOF", groups=flat_genes,
            folds=np.concatenate(flat_folds),
            n_permutations=n_permutations, alternative="greater",
        )
        perm["observed"] = observed if observed is not None else perm.get("observed")
        out["permutation_mlp"] = perm
        p_value_text = (
            f"unresolved at resolution {perm['p_value_resolution']}"
            if perm.get("resolution_limited")
            else str(perm.get("p_value"))
        )
        print(
            f"  permutation: observed GOF AUROC {perm.get('observed')}  "
            f"null mean {perm.get('null_mean')}  p = {p_value_text}"
        )

    return out


def _within_family_majority_reference(y, genes, classes, n_seeds, n_folds):
    """Calculate the class-only reference from each training fold."""
    per_seed = []
    seed_values = []
    for seed in range(n_seeds):
        splits = gene_split_cv(genes, n_folds=n_folds, seed=seed)
        contract = validate_classification_splits(
            splits,
            requested_folds=n_folds,
            eligible_rows=np.concatenate([test for _train, test in splits]),
            labels=y,
            classes=classes,
            required_train_classes=None,
            required_test_classes=None,
            allow_missing_classifier_classes=True,
            minimum_train_classes=2,
            groups=genes,
            held_out_unit="gene",
        )
        if contract["status"] != "valid":
            per_seed.append(
                {
                    "seed": seed,
                    "status": "unscorable",
                    "split_validation": contract,
                }
            )
            continue
        fold_values = []
        fold_majorities = []
        try:
            for train_rows, test_rows in splits:
                value, majority_class = majority_baseline_f1(
                    y[train_rows], y[test_rows], classes
                )
                fold_values.append(value)
                fold_majorities.append(majority_class)
        except ValueError as error:
            per_seed.append(
                {
                    "seed": seed,
                    "status": "unscorable",
                    "reason": str(error),
                    "split_validation": contract,
                }
            )
            continue
        seed_value = float(np.mean(fold_values))
        seed_values.append(seed_value)
        per_seed.append(
            {
                "seed": seed,
                "status": "success",
                "macro_f1_mean": seed_value,
                "fold_macro_f1": fold_values,
                "fold_majority_classes": fold_majorities,
                "split_validation": contract,
            }
        )
    if len(seed_values) != n_seeds:
        return {
            "status": "unavailable",
            "classes": list(classes),
            "macro_f1_mean": None,
            "macro_f1_std": None,
            "per_seed": per_seed,
            "reason": "one or more required seeds are unscorable",
        }
    return {
        "status": "success",
        "classes": list(classes),
        "macro_f1_mean": float(np.mean(seed_values)),
        "macro_f1_std": float(np.std(seed_values)),
        "per_seed": per_seed,
        "reason": None,
    }


def probe_phase(
    wt_mean, delta, genes, labels, families, gene_label, n_seeds, n_folds,
    compute_ci=True, n_permutations=0,
):
    """Within-family probes for every qualifying family, then a pooled test."""
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

    delta_oof_by_family = {}
    family_inputs = {}

    for family, gene_set in families.items():
        row_mask = np.array([gene in gene_set for gene in genes])
        y = labels[row_mask]
        genes_rows = genes[row_mask]
        features_by_view = {VIEW_WT: wt_mean[row_mask], VIEW_DELTA: delta[row_mask]}

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
            "majority_reference": _within_family_majority_reference(
                y, genes_rows, present_classes, n_seeds, n_folds
            ),
            **family_res,
        }
        delta_oof_by_family[family] = family_oof[VIEW_DELTA]
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
    write_result_json(WITHIN_FAMILY_MECHANISM_JSON, results, seeds=list(range(n_seeds)))
    print(f"  Results written to {WITHIN_FAMILY_MECHANISM_JSON}")
    return results


def _print_headline(results):
    print("\n" + "=" * 78)
    print("HEADLINE - within-family mechanism: delta vs wt_only macro-F1")
    print("=" * 78)
    for family, res in results["by_family"].items():
        base = res["majority_reference"]["macro_f1_mean"]
        base_text = "Unscorable" if base is None else f"{base:.3f}"
        line = (
            f"  {family:9s} n={res['n_genes']:>2}g/{res['n_variants']:>4}v  "
            f"base={base_text}"
        )
        for probe_name in results["probes"]:
            wt_f1 = res[VIEW_WT][probe_name]["macro_f1"]
            delta_f1 = res[VIEW_DELTA][probe_name]["macro_f1"]
            if wt_f1["mean"] is None or delta_f1["mean"] is None:
                line += f"  | {probe_name}: Unscorable"
            else:
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
