"""
Homology-partition robustness panel.

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

The analysis plan assigns each row the unit held out by that split: Pfam families,
Pfam clans, or MMseqs2 clusters. The leakage-fraction point is
LF = (gene_split_f1 - partition_split_f1) / (gene_split_f1 - chance) on rows
shared by both arms, resampled over the row's own held-out unit. Each draw
recomputes its own chance floor, so numerator and denominator always come from
the same draw.

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
from sklearn.preprocessing import LabelEncoder

from esm2_mech.experiments.mechanism.clan_holdout import (
    load_clan_map,
    run_clan_holdout,
)
from esm2_mech.experiments.mechanism.leakage_fraction import MIN_ABOVE_CHANCE
from esm2_mech.experiments.mechanism.mmseqs_cluster_holdout import (
    load_clusters,
    run_mlp as run_mmseqs_mlp,
)
from esm2_mech.utils.bootstrap import (
    attach_mechanism_ci,
    cluster_bootstrap_ci,
    family_or_gene_clusters,
    score_within_folds,
)
from esm2_mech.utils.constants import BOOTSTRAP_N_RESAMPLES, MECHANISM_CLASSES
from esm2_mech.utils.data import load_pfam_map
from esm2_mech.utils.metrics import fold_macro_f1, majority_baseline_f1
from esm2_mech.utils.io import load_variants_and_delta, write_result_json
from esm2_mech.utils.paths import (
    EMB_MUT_MEAN,
    EMB_VALID_VARIANTS_JSON,
    EMB_WT_MEAN,
    HOMOLOGY_PARTITION_PANEL_DIR,
    homology_partition_panel_json,
    NAIVE_BASELINE_JSON,
    NONLINEAR_RESULTS_SEED_JSON,
    PFAM_JSON,
    VALID_VARIANTS_JSON,
)
from esm2_mech.utils.seed_aggregation import (
    read_seed_point_estimate,
    read_seed_result_contract,
)
from esm2_mech.utils.probes import run_mlp_cv
from esm2_mech.utils.splits import family_split_cv, gene_split_cv
from esm2_mech.utils.classification import validate_complete_classification_splits

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
    against, matching the floor the rest of the project cites for the
    mechanism-above-chance assessment — not
    recomputed per partition, so it cannot silently diverge.
    """
    with open(NAIVE_BASELINE_JSON) as f:
        nb = json.load(f)
    gene = read_seed_point_estimate(
        nb["by_strategy"]["most_frequent"]["gene"]["macro_f1_seed_aggregate"]
    )
    family = read_seed_point_estimate(
        nb["by_strategy"]["most_frequent"]["family"]["macro_f1_seed_aggregate"]
    )
    if not gene.available or not family.available:
        raise ValueError("measured chance floors are unavailable")
    gene_chance = gene.value
    family_chance = family.value
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
    status = read_seed_result_contract(seed, path, d)
    if status != "success":
        return None
    value = d.get("mlp_delta_mean_family", {}).get("macro_f1_mean")
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        return None
    return float(value)


def _pred_from_proba(proba):
    return np.array([MECHANISM_CLASSES[col] for col in proba.argmax(axis=1)])


def _oof_macro_f1(oof):
    predictions = _pred_from_proba(np.asarray(oof["proba"]))
    fold_values = []
    for fold in sorted(np.unique(oof["folds"]).tolist()):
        rows = np.where(np.asarray(oof["folds"]) == fold)[0]
        fold_values.append(
            fold_macro_f1(
                np.asarray(oof["y_true"]), rows, predictions, MECHANISM_CLASSES
            )
        )
    return float(np.mean(fold_values))


def leakage_fraction_ci_for_partition(
    oof_gene, oof_partition, partition_clusters, n_boot, seed
):
    """Return the shared-cohort leakage-fraction point with its interval gated.

    `oof_gene`/`oof_partition` are {"y_true", "proba", "row_ids"} with row_ids
    in a SHARED global row space (the same indexing the caller used for both
    gene_split_cv and the partition's own split). `partition_clusters` is a
    row-aligned array of partition-unit ids (family/clan/cluster), indexed the
    same way as oof_partition's rows.

    The ratio's chance floor is refitted from each fold's training labels on the
    rows in hand, so no externally measured floor is accepted here: a floor
    computed on a different cohort would not match the numerator's rows. Each
    arm therefore needs at least two folds, since a single fold leaves no
    training rows to select a majority class from.
    """
    gene_pos = {int(row): pos for pos, row in enumerate(oof_gene["row_ids"])}
    part_pos = {int(row): pos for pos, row in enumerate(oof_partition["row_ids"])}
    shared_rows = sorted(set(gene_pos) & set(part_pos))
    if not shared_rows:
        return None

    gene_positions = np.array([gene_pos[row] for row in shared_rows], dtype=int)
    partition_positions = np.array([part_pos[row] for row in shared_rows], dtype=int)

    def _shared_oof(oof, positions):
        return {
            "y_true": np.asarray(oof["y_true"])[positions],
            "proba": np.asarray(oof["proba"])[positions],
            "row_ids": np.asarray(oof["row_ids"])[positions],
            "folds": np.asarray(oof["folds"])[positions],
        }

    gene_shared = _shared_oof(oof_gene, gene_positions)
    partition_shared = _shared_oof(oof_partition, partition_positions)
    if not np.array_equal(gene_shared["y_true"], partition_shared["y_true"]):
        raise ValueError("shared homology-panel rows have inconsistent observed labels")

    gene_predictions = _pred_from_proba(gene_shared["proba"])
    partition_predictions = _pred_from_proba(partition_shared["proba"])
    gene_arms = [(gene_predictions, gene_shared["folds"], np.unique(gene_shared["folds"]))]
    partition_arms = [
        (
            partition_predictions,
            partition_shared["folds"],
            np.unique(partition_shared["folds"]),
        )
    ]
    observed = gene_shared["y_true"]

    def _fold_f1(block, arm_pred):
        return fold_macro_f1(observed, block, arm_pred, MECHANISM_CLASSES)

    gene_folds = gene_shared["folds"]
    gene_fold_ids = gene_arms[0][2]

    def _majority_floor(rows):
        """Training-fold majority rule, scored out of fold on the drawn rows.

        The floor must be the same kind of quantity as the probe score it is
        subtracted from: fitted on a fold's training rows and scored on that
        fold's held-out rows. Choosing the majority class from the rows being
        scored would fit the floor in sample and inflate it.
        """
        fold_of_row = gene_folds[rows]
        fold_values = []
        for fold in gene_fold_ids:
            test_rows = rows[fold_of_row == fold]
            train_rows = rows[fold_of_row != fold]
            if test_rows.size == 0 or train_rows.size == 0:
                return None
            try:
                value, _majority = majority_baseline_f1(
                    observed[train_rows], observed[test_rows], MECHANISM_CLASSES
                )
            except ValueError:
                # The drawn training rows have no single majority class, so the
                # floor — and with it the ratio's denominator — is undefined here.
                return None
            fold_values.append(value)
        return float(np.mean(fold_values))

    def _ratio(rows):
        # Both arms keep their own fold assignment on the drawn rows, and the
        # chance floor is refitted fold by fold on the same draw, so numerator and
        # denominator come from one draw rather than from mixed cohorts.
        gene_f1 = score_within_folds(rows, gene_arms, _fold_f1)
        partition_f1 = score_within_folds(rows, partition_arms, _fold_f1)
        if gene_f1 is None or partition_f1 is None:
            return None
        resample_chance = _majority_floor(rows)
        if resample_chance is None:
            return None
        denominator = gene_f1 - resample_chance
        if denominator <= MIN_ABOVE_CHANCE:
            return None
        return (gene_f1 - partition_f1) / denominator

    interval = cluster_bootstrap_ci(
        np.asarray(partition_clusters)[partition_positions],
        _ratio,
        n_resamples=n_boot,
        seed=seed,
        discard_reason=(
            "a fold of the draw had no single training-side majority class, or "
            "the gene arm's lift over "
            "the resampled chance floor collapsed, so the leakage fraction had no "
            "denominator on that draw"
        ),
        metric_name="leakage_fraction",
    )
    interval["n_shared_rows"] = int(len(shared_rows))
    return interval


def _partition_row(
    name, oof_gene, oof_partition, partition_clusters, family_chance, n_boot, seed
):
    """One robustness-table row: mechanism-null CI + leakage-fraction CI, both
    resampled at `partition_clusters` (the row's own held-out unit)."""
    ci_container: dict = {}
    attach_mechanism_ci(
        ci_container,
        oof_partition,
        partition_clusters,
        compute_ci=True,
        n_resamples=n_boot,
        seed=seed,
    )
    null_ci = ci_container["ci"]
    lf_ci = leakage_fraction_ci_for_partition(
        oof_gene, oof_partition, partition_clusters, n_boot, seed
    )
    return {
        "partition": name,
        "status": "success",
        "n_clusters": int(len(np.unique(partition_clusters))),
        "measured_floor": family_chance,
        "mechanism_null_macro_f1": null_ci["macro_f1"],
        "leakage_fraction_ci": lf_ci,
    }


def _print_partition_summary(row, unit_name):
    """Print one panel row without presenting a suppressed interval as a CI."""
    if row["status"] != "success":
        print(f"  {row['partition']}: {row['status'].capitalize()}")
        return
    interval = row["mechanism_null_macro_f1"]
    point_text = "NA" if interval["point"] is None else f"{interval['point']:.4f}"
    if interval["ci_suppressed"]:
        print(
            f"  mechanism_null macro_f1={point_text}; interval unavailable "
            f"({interval['reason']}); n_clusters({unit_name})={row['n_clusters']}"
        )
        return
    print(
        f"  mechanism_null macro_f1={point_text} "
        f"[{interval['ci_low']}, {interval['ci_high']}] "
        f"n_clusters({unit_name})={row['n_clusters']}"
    )


def run_family_row(labels, genes, delta, pfam_map, seed, n_boot, n_folds=5):
    gene_splits = gene_split_cv(genes, n_folds=n_folds, seed=seed)
    family_splits = family_split_cv(genes, pfam_map, n_folds=n_folds, seed=seed)
    gene_contract = validate_complete_classification_splits(
        gene_splits,
        requested_folds=n_folds,
        eligible_rows=np.concatenate([test for _train, test in gene_splits]),
        labels=labels,
        classes=MECHANISM_CLASSES,
        groups=genes,
        held_out_unit="gene",
    )
    family_groups = family_or_gene_clusters(
        genes, pfam_map, is_family_split=True
    )
    family_contract = validate_complete_classification_splits(
        family_splits,
        requested_folds=n_folds,
        eligible_rows=np.concatenate([test for _train, test in family_splits]),
        labels=labels,
        classes=MECHANISM_CLASSES,
        groups=family_groups,
        held_out_unit="family",
    )
    _, oof_gene = run_mlp_cv(
        delta, labels, gene_splits, MECHANISM_CLASSES, gene_contract,
        hidden=MLP_HIDDEN, seed=seed, genes=genes,
        label="panel-gene", return_oof=True, compute_per_gene=False,
    )
    _, oof_family = run_mlp_cv(
        delta, labels, family_splits, MECHANISM_CLASSES, family_contract,
        hidden=MLP_HIDDEN, seed=seed, genes=genes,
        label="panel-family", return_oof=True, compute_per_gene=False,
    )
    if oof_gene is None or oof_family is None:
        raise RuntimeError("family/gene-split MLP produced no scorable fold")

    family_clusters = family_or_gene_clusters(oof_family["genes"], pfam_map, is_family_split=True)
    _gene_chance, family_chance = _measured_chance_floors()
    return oof_gene, _partition_row(
        PARTITION_FAMILY, oof_gene, oof_family, family_clusters,
        family_chance, n_boot, seed,
    )


def run_clan_row(labels, genes, delta, gene_clan, clan_names, oof_gene, family_chance, seed, n_boot):
    le = LabelEncoder()
    le.fit(MECHANISM_CLASSES)
    _clan_results, _qualifying, _ci, oof_clan, _split_contract = run_clan_holdout(
        delta, labels, genes, gene_clan, clan_names, le, seed=seed, n_boot=n_boot
    )
    if oof_clan is None:
        return None
    return _partition_row(
        PARTITION_CLAN, oof_gene, oof_clan, oof_clan["clan"],
        family_chance, n_boot, seed,
    )


def run_mmseqs_row(labels, genes, delta, gene_to_cluster, oof_gene, family_chance, seed, n_boot, n_folds=5):
    cluster_of = np.array([gene_to_cluster.get(g) for g in genes])
    has_cluster = np.array([c is not None for c in cluster_of])
    idx = np.where(has_cluster)[0]
    labels_f = labels[idx]
    genes_f = genes[idx]
    groups = cluster_of[idx]
    delta_f = delta[idx]

    _agg, oof_local = run_mmseqs_mlp(
        delta_f, labels_f, genes_f, groups, MLP_HIDDEN, n_folds, seed, "panel-mmseqs",
        return_oof=True, compute_per_gene=False,
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
        "folds": oof_local["folds"],
    }
    partition_clusters = groups[oof_local["row_ids"]]
    return _partition_row(
        PARTITION_MMSEQS, oof_gene, oof_global, partition_clusters,
        family_chance, n_boot, seed,
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
    pfam_map = load_pfam_map(PFAM_JSON)
    clan_map, clan_names = load_clan_map(args.clan_file)
    gene_clan = {g: clan_map[acc] for g, acc in pfam_map.items() if acc in clan_map}
    print(f"Genes with clan assignment: {len(gene_clan)}/{len(pfam_map)}")
    gene_to_cluster = load_clusters()

    gene_chance, family_chance = _measured_chance_floors()
    print(f"Measured chance floors: gene={gene_chance:.4f} family={family_chance:.4f}")

    print("\n=== Family row (Pfam family, current default) ===")
    oof_gene, family_row = run_family_row(labels, genes, delta, pfam_map, args.seed, args.n_boot)
    _print_partition_summary(family_row, "families")

    print("\n=== Clan row (Pfam clan, stricter) ===")
    clan_row = run_clan_row(
        labels, genes, delta, gene_clan, clan_names, oof_gene, family_chance,
        args.seed, args.n_boot,
    )
    if clan_row is not None:
        _print_partition_summary(clan_row, "clans")
    else:
        print("  No qualifying clans produced OOF — clan row omitted.")

    print("\n=== MMseqs2 row (20% identity cluster, strictest) ===")
    mmseqs_row = run_mmseqs_row(
        labels, genes, delta, gene_to_cluster, oof_gene, family_chance,
        args.seed, args.n_boot,
    )
    if mmseqs_row is not None:
        _print_partition_summary(mmseqs_row, "mmseqs")
    else:
        print("  No scorable MMseqs2 fold produced OOF — MMseqs2 row omitted.")

    rows = [row for row in (family_row, clan_row, mmseqs_row) if row is not None]
    results = {
        "description": (
            "Homology-partition robustness panel: the mechanism null "
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
    panel_path = homology_partition_panel_json(args.seed)
    write_result_json(panel_path, results, seeds=[args.seed], indent=2)
    print(f"\nResults written to {panel_path}")


if __name__ == "__main__":
    main()
