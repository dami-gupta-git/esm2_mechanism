"""CV split generators: random, gene-disjoint and family-disjoint cross-validation."""

from __future__ import annotations

import numpy as np


def random_split_cv(n: int, n_folds: int = 5, seed: int = 42) -> list[tuple]:
    """Random k-fold CV over n rows (no grouping).

    Shuffles the row indices and splits them into n_folds disjoint test folds;
    each fold's train set is the complement. Used as the in-distribution
    reference against which grouped (gene/family) splits are compared.
    """
    idx = np.arange(n)
    np.random.RandomState(seed).shuffle(idx)
    splits = []
    for fold in np.array_split(idx, n_folds):
        tr = np.setdiff1d(idx, fold)
        splits.append((tr, fold))
    return splits


def gene_split_cv(
    genes: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    min_train: int = 10,
    min_test: int = 5,
) -> list[tuple]:
    """Gene-disjoint CV: each fold holds out a disjoint set of genes.

    min_train/min_test are the minimum row counts a fold must have to be kept.
    The defaults (10/5) suit the full dataset; within-family CV passes smaller
    values because a single family has too few variants to clear the defaults
    (which would otherwise drop every fold and return an empty split list).
    """
    unique = np.array(sorted(set(genes)))
    np.random.RandomState(seed).shuffle(unique)
    splits = []
    for fold in np.array_split(unique, n_folds):
        tr = np.where(~np.isin(genes, fold))[0]
        te = np.where(np.isin(genes, fold))[0]
        if len(tr) >= min_train and len(te) >= min_test:
            splits.append((tr, te))
    return splits


def family_split_cv(
    genes: np.ndarray, pfam_map: dict, n_folds: int = 5, seed: int = 42
) -> list[tuple]:
    """Family-disjoint CV using a {gene: pfam_family} map.

    Only annotated genes (present in pfam_map) are included in train/test;
    unannotated genes are excluded from both sides.
    """
    g2p = {g: pfam_map[g] for g in np.unique(genes) if pfam_map.get(g)}
    fams = np.array(sorted(set(g2p.values())))
    np.random.RandomState(seed).shuffle(fams)
    n = len(genes)
    splits = []
    for fold_fams in np.array_split(fams, n_folds):
        fs = set(fold_fams)
        te = np.array([genes[i] in g2p and g2p[genes[i]] in fs for i in range(n)])
        tr = np.array([genes[i] in g2p and g2p[genes[i]] not in fs for i in range(n)])
        if tr.sum() >= 10 and te.sum() >= 5:
            splits.append((np.where(tr)[0], np.where(te)[0]))
    return splits


def family_split_indices(groups: np.ndarray, n_folds: int, seed: int):
    """Yield (train_idx, test_idx) for family-disjoint CV.

    groups is a 1-D array of Pfam family strings (or None for unannotated).
    None-group samples are distributed evenly across folds rather than excluded.
    """
    rng = np.random.RandomState(seed)
    unique_fams = np.array(sorted(f for f in set(groups) if f is not None))
    rng.shuffle(unique_fams)
    fam_fold = {f: i % n_folds for i, f in enumerate(unique_fams)}

    none_positions = [i for i, g in enumerate(groups) if g is None]
    rng.shuffle(none_positions)
    none_fold = {pos: i % n_folds for i, pos in enumerate(none_positions)}

    fold_of = np.array(
        [
            fam_fold[g] if g is not None else none_fold.get(i, 0)
            for i, g in enumerate(groups)
        ]
    )
    for k in range(n_folds):
        test = np.where(fold_of == k)[0]
        train = np.where(fold_of != k)[0]
        yield train, test


def fold_index_array(splits: list[tuple], n_rows: int) -> np.ndarray:
    """Which test fold each row falls in, for the permutation tests.

    A permutation that refits per draw scores each fold separately, so the shuffle
    has to know the fold layout even though the out-of-fold predictions are rebuilt
    every draw. Every row must land in exactly one test fold; a row in none or in two
    means the splits are not a partition and the within-fold shuffle would be
    ill-defined.
    """
    folds = np.full(n_rows, -1, dtype=int)
    for fold_i, (_, test_idx) in enumerate(splits):
        test_idx = np.asarray(test_idx)
        if (folds[test_idx] != -1).any():
            raise ValueError("fold_index_array: a row appears in two test folds")
        folds[test_idx] = fold_i
    if (folds == -1).any():
        raise ValueError(
            f"fold_index_array: {(folds == -1).sum()} of {n_rows} rows are in no test fold"
        )
    return folds
