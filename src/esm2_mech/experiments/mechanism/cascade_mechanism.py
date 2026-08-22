"""Two-stage cascade mechanism classifier on ESM-2 delta embeddings.

Stage A separates LOF from {GOF, DN}. Stage B separates GOF from DN and is fitted
only on the non-LOF rows of the same training fold. The two stages are combined
into one three-class posterior per variant:

    P(LOF) = pA,  P(GOF) = (1 - pA) * pB,  P(DN) = (1 - pA) * (1 - pB)

Stage A's training fold is resampled two ways, run as separate arms so the effect
of the resampling is readable rather than assumed. The ``family_matched`` arm keeps
only Pfam families that contain both a LOF and a non-LOF variant, and inside each
such family downsamples both sides to that family's minority count, so no family
carries a class prevalence a probe could memorise in place of the mutation. Which
LOF rows survive is decided round-robin over k-means clusters fitted on the
training fold's LOF delta embeddings, so the retained rows keep the spread of the
discarded ones. The ``unbalanced`` arm leaves the training fold alone and is the
ablation the first arm is read against.

Every fit is a small MLP trained with focal loss. Resampling, clustering, scaling
and the early-stopping holdout are all confined to the training rows of a fold;
test folds keep the real class distribution, so the reported metrics are on the
population the classifier would face.
"""

from __future__ import annotations

import argparse
import functools
from collections import defaultdict

import numpy as np

from esm2_mech.utils.bootstrap import attach_mechanism_ci, family_or_gene_clusters
from esm2_mech.utils.constants import (
    BOOTSTRAP_N_RESAMPLES,
    CASCADE_ARM_FAMILY_MATCHED,
    CASCADE_ARM_SIZE_MATCHED,
    CASCADE_FOCAL_GAMMA,
    CASCADE_LOF_CLUSTER_PCA,
    CASCADE_LOF_N_CLUSTERS,
    CASCADE_LOF_TARGET_RATIO,
    CASCADE_SAMPLING_ARMS,
    CASCADE_STAGE_A,
    CASCADE_STAGE_B,
    DN,
    GOF,
    LOF,
    MECHANISM_CLASSES,
    N_FOLDS,
    N_SEEDS,
    SPLIT_FAMILY,
    SPLIT_GENE,
)
from esm2_mech.utils.data import (
    embedding_fingerprint,
    labeled_variant_fingerprint,
    load_pfam_map,
    pfam_fingerprint,
)
from esm2_mech.utils.io import load_variants_and_delta, write_result_json
from esm2_mech.utils.metrics import (
    aggregate_folds,
    compute_metrics,
    imbalance_metrics,
    majority_baseline_f1,
    mean_std_n,
    standardize,
)
from esm2_mech.utils.paths import (
    CASCADE_MECHANISM_AGGREGATE_JSON,
    CASCADE_MECHANISM_DIR,
    EMB_MUT_MEAN,
    EMB_VALID_VARIANTS_JSON,
    EMB_WT_MEAN,
    PFAM_JSON,
    VALID_VARIANTS_JSON,
)
from esm2_mech.utils.probes import aggregate_fold_dicts, require_no_nan
from esm2_mech.utils.probes import _validation_group_mask
from esm2_mech.utils.splits import family_split_cv, gene_split_cv

print = functools.partial(print, flush=True)

GOF_COLUMN = MECHANISM_CLASSES.index(GOF)
DN_COLUMN = MECHANISM_CLASSES.index(DN)
LOF_COLUMN = MECHANISM_CLASSES.index(LOF)

# A fold too small to hold out whole groups for early stopping, or left with one
# class on either side of a stage, is recorded under one of these reasons instead
# of being scored on a fabricated probability.
SKIP_NO_VALIDATION_GROUPS = "fewer than two training groups for the early-stopping holdout"
SKIP_ONE_CLASS_IN_FIT = "the fitting subset carried only one class"
SKIP_TOO_FEW_ROWS = "the fitting or validation subset was too small to train on"


# ── Training-fold resampling ─────────────────────────────────────────────────


def _round_robin_by_key(
    rows: list[int], key_of: dict[int, object], n_keep: int, rng: np.random.RandomState
) -> list[int]:
    """Take n_keep of `rows`, cycling over the groups `key_of` assigns them to.

    Every group gives up one row per cycle, so a group holding a small share of
    `rows` still contributes until it is exhausted. Taking a plain random sample
    instead would retain the groups in proportion to their size, which is what
    reduces a downsampled LOF set to its largest cluster.
    """
    if n_keep >= len(rows):
        return list(rows)
    if n_keep <= 0:
        return []
    buckets: dict[object, list[int]] = defaultdict(list)
    for row in rows:
        buckets[key_of[row]].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    ordered_keys = sorted(buckets, key=str)
    picked: list[int] = []
    while len(picked) < n_keep:
        drew_this_cycle = False
        for key in ordered_keys:
            if not buckets[key]:
                continue
            picked.append(buckets[key].pop())
            drew_this_cycle = True
            if len(picked) == n_keep:
                break
        if not drew_this_cycle:
            break
    return picked


def lof_cluster_assignment(
    delta: np.ndarray,
    lof_rows: np.ndarray,
    n_clusters: int,
    n_pca: int,
    seed: int,
) -> tuple[dict[int, int], dict]:
    """k-means cluster id per LOF training row, fitted on those rows alone.

    The scaler, the PCA and the k-means are all fitted inside the training fold, so
    nothing about the held-out families reaches the clustering. Returns the row to
    cluster map and the realised design (how many clusters were actually fitted and
    how the rows fell across them), so a degenerate clustering is visible in the
    result file rather than silently flattening the round-robin into a random draw.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    n_rows = len(lof_rows)
    effective_clusters = min(n_clusters, n_rows)
    if effective_clusters < 2:
        return (
            {int(row): 0 for row in lof_rows},
            {
                "n_lof_rows": int(n_rows),
                "n_clusters_requested": int(n_clusters),
                "n_clusters_fitted": 1,
                "cluster_sizes": [int(n_rows)],
            },
        )

    features = delta[lof_rows].astype(np.float32)
    (features,) = standardize(features)
    n_components = min(n_pca, features.shape[1], n_rows - 1)
    if n_components >= 1:
        features = PCA(n_components=n_components, random_state=seed).fit_transform(features)
    kmeans = KMeans(n_clusters=effective_clusters, n_init=10, random_state=seed)
    cluster_ids = kmeans.fit_predict(features)
    sizes = np.bincount(cluster_ids, minlength=effective_clusters)
    return (
        {int(row): int(cluster) for row, cluster in zip(lof_rows, cluster_ids)},
        {
            "n_lof_rows": int(n_rows),
            "n_clusters_requested": int(n_clusters),
            "n_clusters_fitted": int(effective_clusters),
            "cluster_sizes": [int(size) for size in sizes],
        },
    )


def family_matched_training_rows(
    train_rows: np.ndarray,
    is_lof: np.ndarray,
    labels: np.ndarray,
    families: np.ndarray,
    cluster_of: dict[int, int],
    target_ratio: float,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, dict]:
    """Select stage-A training rows so no family carries a LOF/non-LOF prevalence.

    A family holding only one of the two classes is dropped: it can teach which
    family a variant belongs to and nothing about the substitution. In every
    remaining family both sides are cut to that family's minority count, LOF rows
    chosen round-robin over their k-means clusters and non-LOF rows round-robin
    over GOF and DN so stage B still sees both.

    `families` is the Pfam family of each row and is the matching unit under both
    splits. The gene is not a usable matching unit here: mechanism labels are
    assigned per gene, so every variant in a gene carries one class and no gene
    ever holds both sides to match.

    `target_ratio` above 1.0 tops the LOF side back up from the dropped
    single-class families. It defaults to 1.0, at which the matched rows already
    meet the target and no such row is drawn — raising it trades the per-family
    guarantee for LOF volume, so the realised counts are returned alongside.
    """
    per_family: dict[object, dict[str, list[int]]] = defaultdict(
        lambda: {"lof": [], "other": []}
    )
    for row in train_rows:
        row = int(row)
        per_family[families[row]]["lof" if is_lof[row] else "other"].append(row)

    label_of = {int(row): labels[int(row)] for row in train_rows}
    kept_lof: list[int] = []
    kept_other: list[int] = []
    single_class_lof: list[int] = []
    n_mixed_families = 0
    n_lof_only_families = 0
    n_non_lof_only_families = 0

    for family in sorted(per_family, key=str):
        members = per_family[family]
        lof_rows, other_rows = members["lof"], members["other"]
        if not lof_rows:
            n_non_lof_only_families += 1
            continue
        if not other_rows:
            n_lof_only_families += 1
            single_class_lof.extend(lof_rows)
            continue
        n_mixed_families += 1
        n_keep = min(len(lof_rows), len(other_rows))
        kept_lof.extend(_round_robin_by_key(lof_rows, cluster_of, n_keep, rng))
        kept_other.extend(_round_robin_by_key(other_rows, label_of, n_keep, rng))

    target_lof = int(round(target_ratio * len(kept_other)))
    n_topped_up = max(0, target_lof - len(kept_lof))
    topped_up = _round_robin_by_key(single_class_lof, cluster_of, n_topped_up, rng)

    selected = np.array(sorted(kept_lof + kept_other + topped_up), dtype=int)
    design = {
        "n_train_rows_available": int(len(train_rows)),
        "n_train_rows_selected": int(len(selected)),
        "n_mixed_families": n_mixed_families,
        "n_lof_only_families_dropped": n_lof_only_families,
        "n_non_lof_only_families_dropped": n_non_lof_only_families,
        "n_lof_matched_within_family": len(kept_lof),
        "n_non_lof_kept": len(kept_other),
        "n_lof_topped_up_from_single_class_families": len(topped_up),
        "target_lof_to_non_lof_ratio": float(target_ratio),
        "realised_lof_to_non_lof_ratio": (
            float((len(kept_lof) + len(topped_up)) / len(kept_other))
            if kept_other
            else None
        ),
    }
    return selected, design


def size_matched_training_rows(
    train_rows: np.ndarray,
    is_lof: np.ndarray,
    labels: np.ndarray,
    cluster_of: dict[int, int],
    n_lof: int,
    n_non_lof: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, dict]:
    """Draw `n_lof` and `n_non_lof` rows from the whole training fold, unpaired.

    The counts come from what family matching produced on the same fold, so this
    arm differs from it in one respect only: which family a row belongs to plays
    no part in the selection. Family matching changes both the size of the
    training pool and whether family identity predicts the label; comparing it
    against the untouched fold cannot say which of the two moved the result, and
    comparing it against this arm can.

    LOF rows are still drawn round-robin over the k-means clusters and non-LOF
    rows round-robin over GOF and DN, so the only removed ingredient is the
    within-family pairing rather than the cluster and class coverage as well.
    """
    lof_rows = [int(row) for row in train_rows if is_lof[int(row)]]
    non_lof_rows = [int(row) for row in train_rows if not is_lof[int(row)]]
    label_of = {int(row): labels[int(row)] for row in non_lof_rows}

    kept_lof = _round_robin_by_key(lof_rows, cluster_of, n_lof, rng)
    kept_non_lof = _round_robin_by_key(non_lof_rows, label_of, n_non_lof, rng)
    selected = np.array(sorted(kept_lof + kept_non_lof), dtype=int)
    design = {
        "n_train_rows_available": int(len(train_rows)),
        "n_train_rows_selected": int(len(selected)),
        "n_lof_requested": int(n_lof),
        "n_lof_drawn": len(kept_lof),
        "n_non_lof_requested": int(n_non_lof),
        "n_non_lof_drawn": len(kept_non_lof),
        "realised_lof_to_non_lof_ratio": (
            float(len(kept_lof) / len(kept_non_lof)) if kept_non_lof else None
        ),
    }
    return selected, design


# ── Focal-loss binary MLP ────────────────────────────────────────────────────


def _focal_loss(logits, targets, alpha: float, gamma: float):
    """Binary focal loss on raw logits.

    The alpha term reweights the two classes; the (1 - p_t)^gamma term shrinks the
    contribution of rows the model already classifies confidently, which is what
    keeps a large easy class from dominating the gradient once it is well fitted.
    gamma of 0 reduces this to weighted cross-entropy.
    """
    import torch
    import torch.nn.functional as functional

    bce = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = torch.sigmoid(logits)
    p_t = probability * targets + (1 - probability) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t).pow(gamma) * bce).mean()


def fit_focal_mlp(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_test: np.ndarray,
    seed: int,
    hidden: tuple[int, ...],
    dropout: float,
    lr: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    gamma: float,
) -> np.ndarray:
    """Fit a binary MLP with focal loss and return P(positive) on X_test.

    X_fit, X_validation and X_test are standardized here with statistics taken from
    X_fit alone. The alpha weight is the negative-class share of the fitting subset,
    so the rarer class carries the larger weight without any row being duplicated.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    X_fit, X_validation, X_test = standardize(X_fit, X_validation, X_test)
    positive_fraction = float(y_fit.mean())
    alpha = 1.0 - positive_fraction

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    layers: list = []
    previous = X_fit.shape[1]
    for width in hidden:
        layers += [nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)]
        previous = width
    layers.append(nn.Linear(previous, 1))
    model = nn.Sequential(*layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    fit_dataset = TensorDataset(
        torch.tensor(X_fit, dtype=torch.float32),
        torch.tensor(y_fit, dtype=torch.float32),
    )
    shuffle_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        fit_dataset, batch_size=batch_size, shuffle=True, generator=shuffle_generator
    )
    validation_features = torch.tensor(X_validation, dtype=torch.float32).to(device)
    validation_targets = torch.tensor(y_validation, dtype=torch.float32).to(device)

    best_validation_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    for _epoch in range(max_epochs):
        model.train()
        for features, targets in loader:
            optimizer.zero_grad()
            logits = model(features.to(device)).squeeze(-1)
            _focal_loss(logits, targets.to(device), alpha, gamma).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = _focal_loss(
                model(validation_features).squeeze(-1),
                validation_targets,
                alpha,
                gamma,
            ).item()
        if validation_loss < best_validation_loss - 1e-4:
            best_validation_loss = validation_loss
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    if best_state is None:
        raise RuntimeError("focal-loss MLP early stopping produced no fitted checkpoint")
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32).to(device)).squeeze(-1)
        return torch.sigmoid(logits).cpu().numpy().astype(np.float64)


def fit_stage(
    X: np.ndarray,
    y_binary: np.ndarray,
    fit_pool: np.ndarray,
    test_rows: np.ndarray,
    groups: np.ndarray,
    seed: int,
    args,
) -> tuple[np.ndarray | None, str | None]:
    """Fit one cascade stage on `fit_pool` and score `test_rows`.

    The early-stopping holdout takes whole groups (Pfam families under the family
    split, genes under the gene split) out of `fit_pool`, so a validation row never
    shares a group with a row the weights were fitted on. Returns
    (probabilities, None) or (None, reason) — a stage that cannot be fitted is
    reported as unfitted rather than scored on a stand-in probability.
    """
    validation_mask = _validation_group_mask(groups[fit_pool], seed, validation_fraction=0.15)
    if validation_mask is None:
        return None, SKIP_NO_VALIDATION_GROUPS
    fit_rows = fit_pool[~validation_mask]
    validation_rows = fit_pool[validation_mask]
    if len(fit_rows) < 10 or len(validation_rows) < 5:
        return None, SKIP_TOO_FEW_ROWS
    if len(set(y_binary[fit_rows].tolist())) < 2:
        return None, SKIP_ONE_CLASS_IN_FIT
    if len(set(y_binary[validation_rows].tolist())) < 2:
        return None, SKIP_ONE_CLASS_IN_FIT

    proba = fit_focal_mlp(
        X[fit_rows].astype(np.float32),
        y_binary[fit_rows].astype(np.float32),
        X[validation_rows].astype(np.float32),
        y_binary[validation_rows].astype(np.float32),
        X[test_rows].astype(np.float32),
        seed=seed,
        hidden=tuple(args.hidden),
        dropout=args.dropout,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        gamma=args.focal_gamma,
    )
    return proba, None


# ── Fold and seed drivers ────────────────────────────────────────────────────


def _binary_fold_metrics(y_binary: np.ndarray, proba: np.ndarray) -> dict:
    """AUROC plus the imbalance metrics for one stage on one fold."""
    from sklearn.metrics import f1_score, roc_auc_score

    metrics: dict = {"n": int(len(y_binary))}
    if len(set(y_binary.tolist())) < 2:
        return metrics
    metrics["auroc"] = float(roc_auc_score(y_binary, proba))
    metrics["macro_f1"] = float(
        f1_score(y_binary, (proba >= 0.5).astype(int), average="macro", zero_division=0)
    )
    imbalance = imbalance_metrics(y_binary.astype(int), proba)
    if imbalance is not None:
        for name in ("auprc", "prevalence", "ppv", "npv"):
            if imbalance[name] is not None:
                metrics[name] = float(imbalance[name])
    return metrics


def run_fold(
    fold_index: int,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    labels: np.ndarray,
    delta: np.ndarray,
    families: np.ndarray,
    validation_groups: np.ndarray,
    arm: str,
    seed: int,
    args,
) -> dict:
    """Run both stages on one fold and combine them into a three-class posterior.

    `families` is the matching unit for the stage-A resampling; `validation_groups`
    is the split's own unit (gene or family), which the early-stopping holdout is
    taken over so a validation row never shares a dependency unit with a fitted one.
    """
    is_lof = labels == LOF
    fold_seed = seed * 1000 + fold_index
    rng = np.random.RandomState(fold_seed)

    if arm in (CASCADE_ARM_FAMILY_MATCHED, CASCADE_ARM_SIZE_MATCHED):
        train_lof_rows = train_rows[is_lof[train_rows]]
        cluster_of, cluster_design = lof_cluster_assignment(
            delta, train_lof_rows, args.lof_clusters, args.lof_cluster_pca, fold_seed
        )
        # The size-matched control takes its row counts from what family matching
        # produced on this same fold, so the matched selection is computed either
        # way and only the arm decides which of the two is trained on.
        matched_rows, matched_design = family_matched_training_rows(
            train_rows, is_lof, labels, families, cluster_of,
            target_ratio=args.lof_ratio, rng=rng,
        )
        if arm == CASCADE_ARM_FAMILY_MATCHED:
            stage_a_pool, sampling_design = matched_rows, matched_design
        else:
            n_lof_matched = int((labels[matched_rows] == LOF).sum())
            stage_a_pool, sampling_design = size_matched_training_rows(
                train_rows, is_lof, labels, cluster_of,
                n_lof=n_lof_matched,
                n_non_lof=len(matched_rows) - n_lof_matched,
                rng=rng,
            )
            sampling_design["counts_taken_from_family_matched_arm"] = matched_design
    else:
        cluster_design = None
        stage_a_pool = train_rows
        sampling_design = {
            "n_train_rows_available": int(len(train_rows)),
            "n_train_rows_selected": int(len(train_rows)),
        }

    fold: dict = {
        "fold": fold_index,
        "n_train_rows": int(len(train_rows)),
        "n_test_rows": int(len(test_rows)),
        "lof_clustering": cluster_design,
        "stage_a_sampling": sampling_design,
    }

    if len(stage_a_pool) == 0:
        fold["skipped_reason"] = "the stage-A training pool was empty after resampling"
        return fold

    stage_a_proba, stage_a_skip = fit_stage(
        delta, is_lof.astype(int), stage_a_pool, test_rows, validation_groups,
        fold_seed, args,
    )
    if stage_a_proba is None:
        fold["skipped_reason"] = f"stage A was not fitted: {stage_a_skip}"
        return fold
    fold[CASCADE_STAGE_A] = _binary_fold_metrics(is_lof[test_rows].astype(int), stage_a_proba)

    # Stage B takes every non-LOF training row, not the stage-A pool's share of
    # them. The resampling exists to strip the LOF class imbalance out of stage A;
    # GOF versus DN does not have that imbalance, and passing the reduced pool on
    # would discard the scarcest data in the study for no reason.
    stage_b_pool = train_rows[~is_lof[train_rows]]
    fold["stage_b_n_training_rows"] = int(len(stage_b_pool))
    is_gof = labels == GOF
    stage_b_proba, stage_b_skip = fit_stage(
        delta, is_gof.astype(int), stage_b_pool, test_rows, validation_groups,
        fold_seed + 1, args,
    )
    if stage_b_proba is None:
        fold["skipped_reason"] = f"stage B was not fitted: {stage_b_skip}"
        return fold
    # Stage B is scored only where it is defined: the test rows that are not LOF.
    non_lof_test = ~is_lof[test_rows]
    fold[CASCADE_STAGE_B] = _binary_fold_metrics(
        is_gof[test_rows][non_lof_test].astype(int), stage_b_proba[non_lof_test]
    )

    proba_three_class = np.zeros((len(test_rows), len(MECHANISM_CLASSES)), dtype=np.float64)
    proba_three_class[:, LOF_COLUMN] = stage_a_proba
    proba_three_class[:, GOF_COLUMN] = (1.0 - stage_a_proba) * stage_b_proba
    proba_three_class[:, DN_COLUMN] = (1.0 - stage_a_proba) * (1.0 - stage_b_proba)

    y_test = labels[test_rows]
    predictions = np.array(
        [MECHANISM_CLASSES[column] for column in proba_three_class.argmax(axis=1)]
    )
    cascade = compute_metrics(y_test, predictions, proba_three_class, MECHANISM_CLASSES)
    baseline_f1, majority_class = majority_baseline_f1(labels[train_rows], y_test)
    cascade["majority_baseline_macro_f1"] = baseline_f1
    cascade["majority_class"] = majority_class
    fold["cascade"] = cascade
    fold["proba"] = proba_three_class
    fold["y_true"] = y_test
    return fold


def run_arm(
    arm: str,
    split_name: str,
    splits: list[tuple],
    labels: np.ndarray,
    delta: np.ndarray,
    genes: np.ndarray,
    pfam_map: dict,
    seed: int,
    args,
) -> dict:
    """Run every fold of one split under one sampling arm and aggregate."""
    # Two different units. The resampling matches within Pfam families under both
    # splits, because a gene never holds both stage-A classes. The early-stopping
    # holdout follows whichever unit the outer split used.
    families = family_or_gene_clusters(genes, pfam_map, is_family_split=True)
    validation_groups = family_or_gene_clusters(
        genes, pfam_map, is_family_split=(split_name == SPLIT_FAMILY)
    )
    print(f"\n=== {arm} / {split_name}-split / seed {seed} ===")

    fold_records, skipped = [], []
    cascade_folds, stage_a_folds, stage_b_folds = [], [], []
    oof_y, oof_proba, oof_genes, oof_rows, oof_folds = [], [], [], [], []
    for fold_index, (train_rows, test_rows) in enumerate(splits):
        train_rows = np.asarray(train_rows)
        test_rows = np.asarray(test_rows)
        fold = run_fold(
            fold_index, train_rows, test_rows,
            labels, delta, families, validation_groups, arm, seed, args,
        )
        proba = fold.pop("proba", None)
        y_true = fold.pop("y_true", None)
        fold_records.append(fold)
        if "skipped_reason" in fold:
            skipped.append({"fold": fold_index, "reason": fold["skipped_reason"]})
            print(f"  Fold {fold_index + 1}: skipped — {fold['skipped_reason']}")
            continue
        cascade_folds.append(fold["cascade"])
        stage_a_folds.append(fold[CASCADE_STAGE_A])
        stage_b_folds.append(fold[CASCADE_STAGE_B])
        oof_y.append(y_true)
        oof_proba.append(proba)
        oof_genes.append(genes[test_rows])
        oof_rows.append(test_rows)
        oof_folds.append(np.full(len(test_rows), fold_index, dtype=int))
        print(
            f"  Fold {fold_index + 1}: cascade macro_f1="
            f"{fold['cascade']['macro_f1']:.3f} "
            f"(majority baseline {fold['cascade']['majority_baseline_macro_f1']:.3f}), "
            f"stage A auroc={fold[CASCADE_STAGE_A].get('auroc', float('nan')):.3f}, "
            f"stage B auroc={fold[CASCADE_STAGE_B].get('auroc', float('nan')):.3f}"
        )

    result: dict = {
        "arm": arm,
        "split": split_name,
        "n_folds_requested": len(splits),
        "n_folds_scored": len(cascade_folds),
        "skipped_folds": skipped,
        "per_fold": fold_records,
    }
    if not cascade_folds:
        result["cascade"] = None
        result["unscorable_reason"] = "no fold produced both stages"
        return result

    result["cascade"] = aggregate_folds(cascade_folds, MECHANISM_CLASSES)
    baseline_mean, baseline_std, baseline_n = mean_std_n(
        [fold["majority_baseline_macro_f1"] for fold in cascade_folds]
    )
    result["cascade"]["majority_baseline_macro_f1_mean"] = (
        baseline_mean if baseline_n else None
    )
    result["cascade"]["majority_baseline_macro_f1_std"] = (
        baseline_std if baseline_n else None
    )
    result[CASCADE_STAGE_A] = aggregate_fold_dicts(stage_a_folds)
    result[CASCADE_STAGE_B] = aggregate_fold_dicts(stage_b_folds)

    oof = {
        "y_true": np.concatenate(oof_y),
        "proba": np.concatenate(oof_proba),
        "genes": np.concatenate(oof_genes),
        "row_ids": np.concatenate(oof_rows),
        "folds": np.concatenate(oof_folds),
    }
    clusters = family_or_gene_clusters(
        oof["genes"], pfam_map, is_family_split=(split_name == SPLIT_FAMILY)
    )
    attach_mechanism_ci(
        result["cascade"], oof, clusters,
        compute_ci=not args.no_ci, n_resamples=args.n_boot, seed=seed,
    )
    print(
        f"  {arm}/{split_name}: cascade macro_f1="
        f"{result['cascade']['macro_f1_mean']:.3f} "
        f"(majority baseline {baseline_mean:.3f})"
    )
    return result


def run_seed(
    seed: int, labels, genes, delta, pfam_map, input_fingerprints, args
) -> dict:
    gene_splits = gene_split_cv(genes, n_folds=args.n_folds, seed=seed)
    family_splits = family_split_cv(genes, pfam_map, n_folds=args.n_folds, seed=seed)
    print(
        f"\nSeed {seed}: {len(gene_splits)} gene-split folds, "
        f"{len(family_splits)} family-split folds"
    )
    splits_by_name = [(SPLIT_GENE, gene_splits), (SPLIT_FAMILY, family_splits)]

    arms: dict[str, dict] = {}
    for arm in args.arms:
        for split_name, splits in splits_by_name:
            arms[f"{arm}_{split_name}"] = run_arm(
                arm, split_name, splits, labels, delta, genes, pfam_map, seed, args
            )

    result = {
        "seed": seed,
        "arms": arms,
        "input_fingerprints": input_fingerprints,
        "analysis_parameters": analysis_parameters(args),
    }
    CASCADE_MECHANISM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CASCADE_MECHANISM_DIR / f"cascade_seed{seed}.json"
    write_result_json(out_path, result, seeds=[seed])
    print(f"\nWrote {out_path}")
    return result


def analysis_parameters(args) -> dict:
    """Every setting that changes the numbers, recorded with the result."""
    return {
        "arms": list(args.arms),
        "n_folds": args.n_folds,
        "hidden": list(args.hidden),
        "dropout": args.dropout,
        "lr": args.lr,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "focal_gamma": args.focal_gamma,
        "lof_clusters": args.lof_clusters,
        "lof_cluster_pca": args.lof_cluster_pca,
        "lof_to_non_lof_target_ratio": args.lof_ratio,
        "n_bootstrap_resamples": None if args.no_ci else args.n_boot,
        "ci_enabled": not args.no_ci,
    }


# ── Across-seed aggregation ──────────────────────────────────────────────────


def aggregate_seeds(seed_results: list[dict]) -> dict:
    """Mean and standard deviation of each arm's headline metrics across seeds."""
    arm_keys = sorted({key for result in seed_results for key in result["arms"]})
    across: dict[str, dict] = {}
    for arm_key in arm_keys:
        arm_results = [
            result["arms"][arm_key]
            for result in seed_results
            if result["arms"].get(arm_key, {}).get("cascade") is not None
        ]
        seeds_present = [
            result["seed"] for result in seed_results if arm_key in result["arms"]
        ]
        if not arm_results:
            across[arm_key] = {
                "n_seeds_scored": 0,
                "seeds_requested": seeds_present,
                "unscorable_reason": "no seed produced a scorable cascade for this arm",
            }
            continue
        summary: dict = {
            "n_seeds_scored": len(arm_results),
            "seeds_requested": seeds_present,
        }
        headline = [
            ("cascade", "macro_f1_mean"),
            ("cascade", "majority_baseline_macro_f1_mean"),
            ("cascade", f"auroc_{GOF}_mean"),
            ("cascade", f"auroc_{DN}_mean"),
            ("cascade", f"auroc_{LOF}_mean"),
            (CASCADE_STAGE_A, "auroc_mean"),
            (CASCADE_STAGE_A, "auprc_mean"),
            (CASCADE_STAGE_A, "prevalence_mean"),
            (CASCADE_STAGE_B, "auroc_mean"),
            (CASCADE_STAGE_B, "auprc_mean"),
            (CASCADE_STAGE_B, "prevalence_mean"),
        ]
        for section, metric in headline:
            values = [
                arm[section][metric]
                for arm in arm_results
                if arm.get(section) is not None and arm[section].get(metric) is not None
            ]
            mean, std, count = mean_std_n(values)
            summary[f"{section}.{metric}"] = {
                "mean": mean if count else None,
                "std": std if count else None,
                "n_seeds": count,
            }
        across[arm_key] = summary
    return across


def print_summary(across: dict) -> None:
    print("\n=== Across-seed summary ===")
    header = f"{'arm':<32} {'cascade F1':>12} {'baseline':>10} {'A auroc':>9} {'B auroc':>9}"
    print(header)
    print("-" * len(header))
    for arm_key in sorted(across):
        summary = across[arm_key]
        if summary["n_seeds_scored"] == 0:
            print(f"{arm_key:<32} {'not scorable':>12}")
            continue

        def cell(key: str) -> str:
            entry = summary.get(key, {})
            if entry.get("mean") is None:
                return "NA"
            return f"{entry['mean']:.3f}±{entry['std']:.3f}"

        print(
            f"{arm_key:<32} {cell('cascade.macro_f1_mean'):>12} "
            f"{cell('cascade.majority_baseline_macro_f1_mean'):>10} "
            f"{cell(CASCADE_STAGE_A + '.auroc_mean'):>9} "
            f"{cell(CASCADE_STAGE_B + '.auroc_mean'):>9}"
        )


# ── Entry point ──────────────────────────────────────────────────────────────


def load_data():
    variants, labels, genes, delta_mean, _delta_pos = load_variants_and_delta(
        VALID_VARIANTS_JSON, EMB_VALID_VARIANTS_JSON, EMB_WT_MEAN, EMB_MUT_MEAN
    )
    require_no_nan(delta_mean, "cascade_mechanism.load_data")
    pfam_map = load_pfam_map(PFAM_JSON)
    input_fingerprints = {
        "labeled_variants": labeled_variant_fingerprint(variants, labels),
        "delta_mean_embedding": embedding_fingerprint(delta_mean),
        "pfam_assignments": pfam_fingerprint(pfam_map, genes.tolist()),
    }
    return labels, genes, delta_mean, pfam_map, input_fingerprints


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--seeds", type=int, default=N_SEEDS,
                        help="number of seeds to run; runs 0..seeds-1 (>=1)")
    parser.add_argument("--n_folds", type=int, default=N_FOLDS)
    parser.add_argument("--arms", nargs="+", default=list(CASCADE_SAMPLING_ARMS),
                        choices=list(CASCADE_SAMPLING_ARMS),
                        help="stage-A training-fold sampling arms to run")
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 64])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--focal_gamma", type=float, default=CASCADE_FOCAL_GAMMA,
                        help="focal-loss focusing exponent; 0 = weighted cross-entropy")
    parser.add_argument("--lof_clusters", type=int, default=CASCADE_LOF_N_CLUSTERS)
    parser.add_argument("--lof_cluster_pca", type=int, default=CASCADE_LOF_CLUSTER_PCA)
    parser.add_argument("--lof_ratio", type=float, default=CASCADE_LOF_TARGET_RATIO,
                        help="target LOF-to-non-LOF row ratio in a family-matched "
                             "training fold; above 1.0 tops the LOF side up from "
                             "families that hold no GOF or DN variant")
    parser.add_argument("--no_ci", action="store_true", help="skip cluster-bootstrap CIs")
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")
    if args.lof_ratio <= 0:
        parser.error("--lof_ratio must be > 0")

    labels, genes, delta, pfam_map, input_fingerprints = load_data()
    CASCADE_MECHANISM_DIR.mkdir(parents=True, exist_ok=True)

    seed_results = []
    for seed in range(args.seeds):
        print("\n" + "#" * 60)
        print(f"# SEED {seed}")
        print("#" * 60)
        seed_results.append(
            run_seed(seed, labels, genes, delta, pfam_map, input_fingerprints, args)
        )

    for result in seed_results:
        if result["input_fingerprints"] != input_fingerprints:
            raise ValueError(f"seed {result['seed']} was produced from different inputs")

    across = aggregate_seeds(seed_results)
    write_result_json(
        CASCADE_MECHANISM_AGGREGATE_JSON,
        {
            "n_seeds": len(seed_results),
            "seed_files": [f"cascade_seed{result['seed']}.json" for result in seed_results],
            "input_fingerprints": input_fingerprints,
            "analysis_parameters": analysis_parameters(args),
            "across_seed": across,
        },
        seeds=list(range(args.seeds)),
    )
    print_summary(across)
    print(f"\nWrote {CASCADE_MECHANISM_AGGREGATE_JSON}")


if __name__ == "__main__":
    main()
