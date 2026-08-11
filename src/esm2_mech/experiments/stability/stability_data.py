"""Shared input loading for the Megascale stability probes.

megascale_stability.py, megascale_mlp.py and stability_baselines.py all need the
same inputs: the Tsuboyama variants, their ΔΔG labels, the Pfam family map (loaded
from cache or built), and the mean-pooled embedding delta. This module loads them
once so the three probes do not each repeat the boilerplate.
"""

import functools
from collections import namedtuple

import numpy as np

from esm2_mech.experiments.stability.tsuboyama_loader import load_tsuboyama_variants
from esm2_mech.experiments.stability.build_domain_families import build_family_map
from esm2_mech.utils.constants import N_FOLDS
from esm2_mech.utils.io import load_json_or_discard
from esm2_mech.utils.splits import random_split_cv, gene_split_cv, family_split_cv
from esm2_mech.utils.paths import (
    MEGASCALE_EMB_WT_MEAN,
    MEGASCALE_EMB_MUT_MEAN,
    MEGASCALE_EMB_WT_POS,
    MEGASCALE_EMB_MUT_POS,
    MEGASCALE_DOMAIN_FAMILIES_JSON,
)

print = functools.partial(print, flush=True)

# delta_pos is None unless include_pos=True (only the linear probe uses the
# per-residue delta); n_families/n_orphans are derived from the family map so the
# callers do not each recompute them.
StabilityInputs = namedtuple(
    "StabilityInputs",
    [
        "variants",
        "proteins",
        "ddg",
        "family_map",
        "delta_mean",
        "delta_pos",
        "n_families",
        "n_orphans",
    ],
)


def stability_splits(seed, n_variants, proteins, family_map, n_folds=N_FOLDS):
    """The three CV schemes for one seed, as a name→splits dict.

    random (in-distribution), domain-holdout (never train+test on the same
    domain), family-holdout (never train+test on related Pfam families). Shared by
    every stability probe so the scheme/arg triple lives in one place.
    """
    return {
        "random": random_split_cv(n_variants, n_folds, seed),
        "domain": gene_split_cv(proteins, n_folds=n_folds, seed=seed),
        "family": family_split_cv(proteins, family_map, n_folds=n_folds, seed=seed),
    }


def _load_family_map(variants):
    """Domain → Pfam family map, from cache if present else built via HMMER.

    Orphan domains (no Pfam hit) are absent from the map and so are excluded from
    the family-split only.
    """
    cached = load_json_or_discard(MEGASCALE_DOMAIN_FAMILIES_JSON)
    if cached is not None:
        return cached
    return build_family_map(variants=variants)


def _check_alignment(embedding, n_variants, path):
    if len(embedding) != n_variants:
        raise ValueError(
            f"embedding/variant row mismatch: {len(embedding)} embedding rows vs "
            f"{n_variants} variants — {path} is not row-aligned."
        )


def load_stability_inputs(include_pos=False):
    """Load the inputs shared by every stability probe.

    Returns a StabilityInputs namedtuple. delta_pos is None unless include_pos is
    True (the per-residue delta is only used by the linear probe).
    """
    variants = load_tsuboyama_variants()
    proteins = np.array([variant["protein"] for variant in variants])
    ddg = np.array([variant["ddg"] for variant in variants])
    print(
        f"Loaded {len(variants)} Tsuboyama variants across "
        f"{len(set(proteins))} natural domains"
    )

    family_map = _load_family_map(variants)
    n_families = len(set(family_map.values()))
    n_orphans = len(set(proteins)) - len(
        {protein for protein in proteins if protein in family_map}
    )
    print(
        f"Pfam families: {len(set(proteins))} domains → {n_families} families "
        f"({n_orphans} orphans excluded from family-split)"
    )

    wt_mean = np.load(MEGASCALE_EMB_WT_MEAN)
    mut_mean = np.load(MEGASCALE_EMB_MUT_MEAN)
    _check_alignment(wt_mean, len(variants), MEGASCALE_EMB_WT_MEAN)
    _check_alignment(mut_mean, len(variants), MEGASCALE_EMB_MUT_MEAN)
    delta_mean = mut_mean - wt_mean

    delta_pos = None
    if include_pos:
        wt_pos = np.load(MEGASCALE_EMB_WT_POS)
        mut_pos = np.load(MEGASCALE_EMB_MUT_POS)
        _check_alignment(wt_pos, len(variants), MEGASCALE_EMB_WT_POS)
        _check_alignment(mut_pos, len(variants), MEGASCALE_EMB_MUT_POS)
        delta_pos = mut_pos - wt_pos

    return StabilityInputs(
        variants=variants,
        proteins=proteins,
        ddg=ddg,
        family_map=family_map,
        delta_mean=delta_mean,
        delta_pos=delta_pos,
        n_families=n_families,
        n_orphans=n_orphans,
    )
