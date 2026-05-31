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

from esm2_mech.utils.constants import MECHANISM_CLASSES
from esm2_mech.utils.data import load_variants
from esm2_mech.utils.io import atomic_write_json
from esm2_mech.utils.paths import (
    EMB_MUT_MEAN,
    EMB_WT_MEAN,
    PFAM_JSON,
    VALID_VARIANTS_JSON,
    WITHIN_FAMILY_MECHANISM_JSON,
)
from esm2_mech.utils.probes import run_logreg_cv, run_mlp_cv
from esm2_mech.utils.splits import gene_split_cv

import json

print = functools.partial(print, flush=True)

N_SEEDS = 5
MIN_GENES = 6
MIN_CLASSES = 2
N_FOLDS = 5

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


def _probe_one_family(features_by_view, y, genes_rows, classes, n_seeds, n_folds):
    """Within-family gene-split CV for one family, both views x both probes, N seeds.

    Returns {view: {probe: {"macro_f1": {mean,std,per_seed},
                            "auroc": {cls: {mean,std,per_seed}}}}}.
    """
    probes = {"logreg": run_logreg_cv, "mlp": run_mlp_cv}

    per_seed_f1 = {view: {p: [] for p in probes} for view in features_by_view}
    per_seed_auroc = {
        view: {p: defaultdict(list) for p in probes} for view in features_by_view
    }

    for seed in range(n_seeds):
        # Same folds across views/probes within a seed -> fair paired comparison.
        splits = gene_split_cv(
            genes_rows, n_folds=n_folds, seed=seed,
            min_train=MIN_TRAIN, min_test=MIN_TEST,
        )
        for view, feature_matrix in features_by_view.items():
            for probe_name, probe_fn in probes.items():
                res = probe_fn(
                    feature_matrix, y, splits, classes=classes, seed=seed,
                    label=f"{view}:{probe_name}",
                )
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
    for view in features_by_view:
        out[view] = {}
        for probe_name in probes:
            out[view][probe_name] = {
                "macro_f1": summarize(per_seed_f1[view][probe_name]),
                "auroc": {
                    cls: summarize(per_seed_auroc[view][probe_name][cls])
                    for cls in classes
                },
            }
    return out


def probe_phase(wt_mean, delta, genes, labels, families, gene_label, n_seeds, n_folds):
    """Phase 3. Within-family probes for every qualifying family."""
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

        family_res = _probe_one_family(
            features_by_view, y, genes_rows, present_classes, n_seeds, n_folds
        )
        results["by_family"][family] = {
            "n_genes": len(gene_set),
            "n_variants": int(row_mask.sum()),
            "classes": present_classes,
            "gene_class_counts": dict(class_counts),
            "majority_baseline_f1": _majority_baseline_f1(y, present_classes),
            **family_res,
        }

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
    )
    _print_headline(results)


if __name__ == "__main__":
    main()
