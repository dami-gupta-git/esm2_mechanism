"""
Homology-partition robustness panel (Task 2b, biorxiv/FOLLOWUP_biorxiv.md).

Promotes `clan_holdout.py` (leave-one-Pfam-clan-out) and
`mmseqs_cluster_holdout.py` (MMseqs2 20% sequence-identity cluster-holdout)
alongside the default Pfam-family split into one consolidated table: for each
partition, does the ESM-2 delta_mean MLP macro-F1 sit at the measured chance
floor (the mechanism null), and what fraction of its gene-split score is
homology leakage (the leakage fraction)?

All three rows use the SAME probe architecture — sklearn MLPClassifier,
hidden=(256, 64) — so the comparison isolates the partition (increasing
strictness: family < clan < MMseqs2 20% identity), not a change in model. This
is why the family row here is computed fresh rather than read from mlp.py's
nonlinear_results_seed{seed}.json: that file's mlp_delta_mean_family is a
PyTorch MLP (a different implementation), and mixing it with the sklearn MLP
used by clan_holdout.py/mmseqs_cluster_holdout.py would make the "increasing
strictness" comparison confounded by a model change. mlp.py's number is still
cited alongside as a cross-check, never as the row's own CI source.

Per pre-registration §1.2, each row's CI resamples the unit that row's split actually holds
out: family-split resamples Pfam families, clan-split resamples Pfam clans,
MMseqs2-split resamples MMseqs2 clusters. The leakage fraction for each row is
LF = (gene_split_f1 - partition_split_f1) / (gene_split_f1 - chance), jointly
resampled at the partition's own unit per replicate (mirrors
leakage_fraction.py's leakage_fraction_ci, generalised from family to
clan/cluster) rather than combined from two separately-computed CIs.

Output: results/<RUN_NAME>/homology_partition_panel/panel.json

Usage:
    python -m esm2_mech.experiments.mechanism.homology_partition_panel \\
        --clan_file data/downloads/Pfam-A.clans.tsv.gz --n_boot 1000
"""

from __future__ import annotations

import argparse
import functools
import json

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from esm2_mech.experiments.mechanism.clan_holdout import (
    load_clan_map,
    load_pfam,
    run_clan_holdout,
)
from esm2_mech.experiments.mechanism.leakage_fraction import MIN_ABOVE_CHANCE
from esm2_mech.experiments.mechanism.mmseqs_cluster_holdout import (
    cluster_split_indices,
    load_clusters,
    run_mlp as run_mmseqs_mlp,
)
from esm2_mech.utils.bootstrap import (
    bootstrap_mechanism_metrics,
    cluster_bootstrap_ci,
    family_or_gene_clusters,
)
from esm2_mech.utils.constants import BOOTSTRAP_N_RESAMPLES, MECHANISM_CLASSES
from esm2_mech.utils.metrics import majority_baseline_f1
from esm2_mech.utils.io import load_variants_and_delta, write_result_json
from esm2_mech.utils.paths import (
    EMB_MUT_MEAN,
    EMB_VALID_VARIANTS_JSON,
    EMB_WT_MEAN,
    HOMOLOGY_PARTITION_PANEL_DIR,
    HOMOLOGY_PARTITION_PANEL_JSON,
    NAIVE_BASELINE_JSON,
    NONLINEAR_RESULTS_SEED_JSON,
    VALID_VARIANTS_JSON,
)
from esm2_mech.utils.probes import run_mlp_cv
from esm2_mech.utils.splits import family_split_cv, gene_split_cv

print = functools.partial(print, flush=True)

MLP_HIDDEN = (256, 64)
PARTITION_FAMILY = "pfam_family"
PARTITION_CLAN = "pfam_clan"
PARTITION_MMSEQS = "mmseqs2_cluster"


def load_data():
    _variants, labels, genes, delta, _ = load_variants_and_delta(
        VALID_VARIANTS_JSON, EMB_VALID_VARIANTS_JSON, EMB_WT_MEAN, EMB_MUT_MEAN
    )
    return labels, genes, delta


def _measured_chance_floors():
    """Measured majority-class macro-F1 floors from naive_baseline.json.

    `gene` is the LF-ratio denominator's chance term (matches
    leakage_fraction.py's convention: the ratio's numerator is always the
    gene-split score, so its chance term is the gene-split floor). `family` is
    the single "measured floor" every row's mechanism-null is compared
    against, matching the floor the rest of the project cites for 2A — not
    recomputed per partition, so it cannot silently diverge.
    """
    with open(NAIVE_BASELINE_JSON) as f:
        nb = json.load(f)
    gene_chance = nb["by_strategy"]["most_frequent"]["gene"]["macro_f1_mean"]
    family_chance = nb["by_strategy"]["most_frequent"]["family"]["macro_f1_mean"]
    return gene_chance, family_chance


def _live_mlp_py_family_reference(seed):
    """mlp.py's PyTorch mlp_delta_mean_family, cited as a cross-check only —
    never the source of this panel's family row (see module docstring)."""
    path = str(NONLINEAR_RESULTS_SEED_JSON).format(seed=seed)
    try:
        with open(path) as f:
            d = json.load(f)
    except FileNotFoundError:
        return None
    return d.get("mlp_delta_mean_family", {}).get("macro_f1_mean")


def _pred_from_proba(proba):
    return np.array([MECHANISM_CLASSES[col] for col in proba.argmax(axis=1)])


def leakage_fraction_ci_for_partition(oof_gene, oof_partition, partition_clusters, n_boot, seed):
    """CI on LF = (gene_f1 - partition_f1) / (gene_f1 - chance), jointly
    resampled at the partition's own held-out unit per replicate (pre-registration §1.2).

    The chance floor (majority-class macro-F1) is recomputed from the resampled
    gene-split labels on every bootstrap replicate, because resampling clusters
    changes the class balance and therefore the majority-class baseline.

    `oof_gene`/`oof_partition` are {"y_true", "proba", "row_ids"} with row_ids
    in a SHARED global row space (the same indexing the caller used for both
    gene_split_cv and the partition's own split). `partition_clusters` is a
    row-aligned array of partition-unit ids (family/clan/cluster), indexed the
    same way as oof_partition's rows.
    """
    gene_pos = {int(row): pos for pos, row in enumerate(oof_gene["row_ids"])}
    part_pos = {int(row): pos for pos, row in enumerate(oof_partition["row_ids"])}
    shared_rows = sorted(set(gene_pos) & set(part_pos))
    if not shared_rows:
        return None

    gene_y = np.asarray(oof_gene["y_true"])
    gene_pred = _pred_from_proba(np.asarray(oof_gene["proba"]))
    part_y = np.asarray(oof_partition["y_true"])
    part_pred = _pred_from_proba(np.asarray(oof_partition["proba"]))

    gene_idx_all = np.array([gene_pos[row] for row in shared_rows])
    part_idx_all = np.array([part_pos[row] for row in shared_rows])
    clusters = np.asarray(partition_clusters)[part_idx_all]

    def _ratio(rows):
        g_idx = gene_idx_all[rows]
        p_idx = part_idx_all[rows]
        gene_f1 = float(f1_score(gene_y[g_idx], gene_pred[g_idx], average="macro", zero_division=0))
        part_f1 = float(f1_score(part_y[p_idx], part_pred[p_idx], average="macro", zero_division=0))
        resample_chance, _ = majority_baseline_f1(gene_y[g_idx], gene_y[g_idx])
        denom = gene_f1 - resample_chance
        if denom <= MIN_ABOVE_CHANCE:
            return None
        return (gene_f1 - part_f1) / denom

    return cluster_bootstrap_ci(clusters, _ratio, n_resamples=n_boot, seed=seed)


def _partition_row(
    name, oof_gene, oof_partition, partition_clusters, gene_chance, family_chance, n_boot, seed
):
    """One robustness-table row: mechanism-null CI + leakage-fraction CI, both
    resampled at `partition_clusters` (the row's own held-out unit)."""
    null_ci = bootstrap_mechanism_metrics(
        oof_partition["y_true"], oof_partition["proba"], partition_clusters,
        n_resamples=n_boot, seed=seed,
    )
    lf_ci = leakage_fraction_ci_for_partition(
        oof_gene, oof_partition, partition_clusters, n_boot, seed
    )
    return {
        "partition": name,
        "n_clusters": null_ci["macro_f1"]["n_clusters"],
        "measured_floor": family_chance,
        "mechanism_null_macro_f1": null_ci["macro_f1"],
        "leakage_fraction_ci": lf_ci,
    }


def run_family_row(labels, genes, delta, pfam_map, seed, n_boot, n_folds=5):
    gene_splits = gene_split_cv(genes, n_folds=n_folds, seed=seed)
    family_splits = family_split_cv(genes, pfam_map, n_folds=n_folds, seed=seed)
    _, oof_gene = run_mlp_cv(
        delta, labels, gene_splits, hidden=MLP_HIDDEN, seed=seed, genes=genes,
        label="panel-gene", return_oof=True,
    )
    _, oof_family = run_mlp_cv(
        delta, labels, family_splits, hidden=MLP_HIDDEN, seed=seed, genes=genes,
        label="panel-family", return_oof=True,
    )
    if oof_gene is None or oof_family is None:
        raise RuntimeError("family/gene-split MLP produced no scorable fold")

    family_clusters = family_or_gene_clusters(oof_family["genes"], pfam_map, is_family_split=True)
    gene_chance, family_chance = _measured_chance_floors()
    return oof_gene, _partition_row(
        PARTITION_FAMILY, oof_gene, oof_family, family_clusters,
        gene_chance, family_chance, n_boot, seed,
    )


def run_clan_row(labels, genes, delta, gene_clan, clan_names, oof_gene, gene_chance, family_chance, seed, n_boot):
    le = LabelEncoder()
    le.fit(MECHANISM_CLASSES)
    _clan_results, _qualifying, _ci, oof_clan = run_clan_holdout(
        delta, labels, genes, gene_clan, clan_names, le, seed=seed, n_boot=n_boot
    )
    if oof_clan is None:
        return None
    return _partition_row(
        PARTITION_CLAN, oof_gene, oof_clan, oof_clan["clan"],
        gene_chance, family_chance, n_boot, seed,
    )


def run_mmseqs_row(labels, genes, delta, gene_to_cluster, oof_gene, gene_chance, family_chance, seed, n_boot, n_folds=5):
    cluster_of = np.array([gene_to_cluster.get(g) for g in genes])
    has_cluster = np.array([c is not None for c in cluster_of])
    idx = np.where(has_cluster)[0]
    labels_f = labels[idx]
    genes_f = genes[idx]
    groups = cluster_of[idx]
    delta_f = delta[idx]

    _agg, oof_local = run_mmseqs_mlp(
        delta_f, labels_f, genes_f, groups, MLP_HIDDEN, n_folds, seed, "panel-mmseqs",
        return_oof=True,
    )
    if oof_local is None:
        return None
    # oof_local["row_ids"] are positions into the cluster-filtered subset
    # (delta_f/genes_f/groups), not the global row space gene_split_cv used —
    # remap to global indices via `idx` so the LF joint-resample can match
    # rows against oof_gene's global row_ids.
    oof_global = {
        "y_true": oof_local["y_true"],
        "proba": oof_local["proba"],
        "row_ids": idx[oof_local["row_ids"]],
    }
    partition_clusters = groups[oof_local["row_ids"]]
    return _partition_row(
        PARTITION_MMSEQS, oof_gene, oof_global, partition_clusters,
        gene_chance, family_chance, n_boot, seed,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clan_file", required=True, help="Path to Pfam-A.clans.tsv.gz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=== Loading data ===")
    labels, genes, delta = load_data()
    pfam_map = load_pfam()
    clan_map, clan_names = load_clan_map(args.clan_file)
    gene_clan = {g: clan_map[acc] for g, acc in pfam_map.items() if acc in clan_map}
    print(f"Genes with clan assignment: {len(gene_clan)}/{len(pfam_map)}")
    gene_to_cluster = load_clusters()

    gene_chance, family_chance = _measured_chance_floors()
    print(f"Measured chance floors: gene={gene_chance:.4f} family={family_chance:.4f}")

    print("\n=== Family row (Pfam family, current default) ===")
    oof_gene, family_row = run_family_row(labels, genes, delta, pfam_map, args.seed, args.n_boot)
    print(
        f"  mechanism_null macro_f1={family_row['mechanism_null_macro_f1']['point']:.4f} "
        f"[{family_row['mechanism_null_macro_f1']['ci_low']}, "
        f"{family_row['mechanism_null_macro_f1']['ci_high']}] "
        f"n_clusters(families)={family_row['n_clusters']}"
    )

    print("\n=== Clan row (Pfam clan, stricter) ===")
    clan_row = run_clan_row(
        labels, genes, delta, gene_clan, clan_names, oof_gene, gene_chance, family_chance,
        args.seed, args.n_boot,
    )
    if clan_row is not None:
        print(
            f"  mechanism_null macro_f1={clan_row['mechanism_null_macro_f1']['point']:.4f} "
            f"[{clan_row['mechanism_null_macro_f1']['ci_low']}, "
            f"{clan_row['mechanism_null_macro_f1']['ci_high']}] "
            f"n_clusters(clans)={clan_row['n_clusters']}"
        )
    else:
        print("  No qualifying clans produced OOF — clan row omitted.")

    print("\n=== MMseqs2 row (20% identity cluster, strictest) ===")
    mmseqs_row = run_mmseqs_row(
        labels, genes, delta, gene_to_cluster, oof_gene, gene_chance, family_chance,
        args.seed, args.n_boot,
    )
    if mmseqs_row is not None:
        print(
            f"  mechanism_null macro_f1={mmseqs_row['mechanism_null_macro_f1']['point']:.4f} "
            f"[{mmseqs_row['mechanism_null_macro_f1']['ci_low']}, "
            f"{mmseqs_row['mechanism_null_macro_f1']['ci_high']}] "
            f"n_clusters(mmseqs)={mmseqs_row['n_clusters']}"
        )
    else:
        print("  No scorable MMseqs2 fold produced OOF — MMseqs2 row omitted.")

    rows = [row for row in (family_row, clan_row, mmseqs_row) if row is not None]
    results = {
        "description": (
            "Homology-partition robustness panel (Task 2b): the mechanism null "
            "(delta_mean MLP macro-F1 vs. the measured family-split chance floor) "
            "and the leakage fraction, under three partition definitions of "
            "increasing strictness — Pfam family, Pfam clan, MMseqs2 20% "
            "sequence-identity cluster. Every row uses the same sklearn MLP "
            f"(hidden={MLP_HIDDEN}) so only the partition varies."
        ),
        "seed": args.seed,
        "n_boot": args.n_boot,
        "measured_chance_floors": {"gene": gene_chance, "family": family_chance},
        "mlp_py_torch_family_reference": _live_mlp_py_family_reference(args.seed),
        "rows": rows,
    }

    HOMOLOGY_PARTITION_PANEL_DIR.mkdir(parents=True, exist_ok=True)
    write_result_json(HOMOLOGY_PARTITION_PANEL_JSON, results, seeds=[args.seed], indent=2)
    print(f"\nResults written to {HOMOLOGY_PARTITION_PANEL_JSON}")


if __name__ == "__main__":
    main()
