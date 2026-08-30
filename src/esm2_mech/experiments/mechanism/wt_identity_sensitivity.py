"""Test the WT-only split gap on proteins embedded without sequence windowing.

The canonical WT mean embedding is variant-window-dependent for proteins longer
than ESM-2's residue limit. This sensitivity analysis restricts the existing
valid-variant rows to proteins at or below that limit, where every variant from
one protein has the same full-protein WT embedding. It then runs only the
``wt_only_mean`` probe under the existing gene and Pfam-family splits.

This is a subset robustness test. It does not isolate windowing from differences
between short and long proteins.

Usage:
    python -m esm2_mech.experiments.mechanism.wt_identity_sensitivity \
        --seeds 5 --n_boot 1000
"""

from __future__ import annotations

import argparse
import functools
import glob
import hashlib
import json
import os
from collections import Counter

import numpy as np

from esm2_mech.experiments.mechanism.classify_by_mechanism import (
    load_data,
    summarize_split_gap,
)
from esm2_mech.experiments.mechanism.mechanism_delta_family_split import (
    mechanism_input_fingerprints,
    run as run_family_split,
)
from esm2_mech.utils.constants import (
    BOOTSTRAP_N_RESAMPLES,
    MAX_SEQ_LEN,
    MECHANISM_CLASSES,
    MECHANISM_OOF_CACHE_GLOB,
    N_FOLDS,
    N_SEEDS,
    SEED_RESULT_GLOB,
)
from esm2_mech.utils.data import (
    embedding_fingerprint,
    embedding_variant_fingerprint,
    subset_data,
)
from esm2_mech.utils.io import write_result_json
from esm2_mech.utils.paths import (
    EMB_WT_MEAN,
    PFAM_JSON,
    SEQUENCES_JSON,
    VALID_VARIANTS_JSON,
    WT_IDENTITY_SENSITIVITY_AGGREGATE_JSON,
    WT_IDENTITY_SENSITIVITY_DIR,
)
from esm2_mech.utils.seed_aggregation import load_seed_files, read_seed_point_estimate
from esm2_mech.experiments.mechanism.seed_results import (
    aggregate_across_seeds,
    aggregate_result_contract,
    print_table,
)
from esm2_mech.utils.data import load_pfam_map

print = functools.partial(print, flush=True)

WT_ONLY_FEATURE = "wt_only_mean"


def build_short_protein_mask(
    variants: list[dict], sequences: dict[str, str], max_seq_len: int = MAX_SEQ_LEN
) -> np.ndarray:
    """Select rows whose complete UniProt sequence fits within ESM-2's limit."""
    mask = np.zeros(len(variants), dtype=bool)
    for row_index, variant in enumerate(variants):
        uniprot_id = variant.get("uniprot_id")
        if not uniprot_id:
            raise ValueError(f"variant row {row_index} has no uniprot_id")
        if uniprot_id not in sequences:
            raise ValueError(
                f"sequence cache lacks {uniprot_id!r} from valid variant row {row_index}"
            )
        mask[row_index] = len(sequences[uniprot_id]) <= max_seq_len
    return mask


def sequence_selection_fingerprint(variants: list[dict], sequences: dict[str, str]) -> str:
    """Hash the UniProt IDs and sequence content used to define the subset."""
    digest = hashlib.sha256()
    for uniprot_id in sorted({variant["uniprot_id"] for variant in variants}):
        digest.update(uniprot_id.encode())
        digest.update(b"\x00")
        digest.update(sequences[uniprot_id].encode())
        digest.update(b"\x00")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", type=int, default=N_SEEDS,
        help="number of seeds to run; runs 0..seeds-1 (>=1)",
    )
    parser.add_argument("--n_folds", type=int, default=N_FOLDS)
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    parser.add_argument("--no_ci", action="store_true", help="skip cluster-bootstrap CIs")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")
    requested_seeds = range(args.seeds)
    if args.n_folds < 2:
        parser.error("--n_folds must be >= 2")
    if args.n_boot < 1:
        parser.error("--n_boot must be >= 1")

    with open(SEQUENCES_JSON) as handle:
        sequences = json.load(handle)
    data = load_data()
    mask = build_short_protein_mask(data["valid_variants"], sequences)
    if int(mask.sum()) < 50:
        raise ValueError(f"short-protein subset has only {int(mask.sum())} variants")
    subset = subset_data(data, mask)
    input_fingerprints = mechanism_input_fingerprints(
        subset, load_pfam_map(PFAM_JSON)
    )

    WT_IDENTITY_SENSITIVITY_DIR.mkdir(parents=True, exist_ok=True)
    stale = []
    for output_glob in (SEED_RESULT_GLOB, MECHANISM_OOF_CACHE_GLOB):
        stale.extend(glob.glob(os.path.join(str(WT_IDENTITY_SENSITIVITY_DIR), output_glob)))
    for path in stale:
        os.remove(path)
    if stale:
        print(f"Removed {len(stale)} stale seed file(s) from {WT_IDENTITY_SENSITIVITY_DIR}")

    full_class_distribution = dict(Counter(data["labels_3class"]))
    subset_class_distribution = dict(Counter(subset["labels_3class"]))
    print(
        f"Short-protein subset: {int(mask.sum())} / {mask.size} variants; "
        f"{len(set(subset['genes_arr']))} genes; classes {subset_class_distribution}"
    )

    for seed in range(args.seeds):
        print(f"\n--- Seed {seed} ---")
        run_family_split(
            data=subset,
            out_dir=str(WT_IDENTITY_SENSITIVITY_DIR),
            seed=seed,
            n_folds=args.n_folds,
            compute_ci=not args.no_ci,
            n_boot=args.n_boot,
            n_permutations=0,
            feature_names=(WT_ONLY_FEATURE,),
            input_fingerprints=input_fingerprints,
        )

    seed_results = load_seed_files(
        str(WT_IDENTITY_SENSITIVITY_DIR),
        SEED_RESULT_GLOB,
        expected_seeds=requested_seeds,
    )
    aggregated = aggregate_across_seeds(
        seed_results,
        requested_seeds,
        confusion_matrix_class_order=MECHANISM_CLASSES,
    )
    split_gap_summary = summarize_split_gap(seed_results, requested_seeds)
    payload = {
        **aggregate_result_contract(),
        "analysis": "WT-only gene-minus-family split gap on unwindowed proteins",
        "interpretation_limit": (
            "Subset robustness test; excluding long proteins does not isolate windowing "
            "from protein-length and dataset-composition differences."
        ),
        "selection": {
            "rule": "full UniProt sequence length <= MAX_SEQ_LEN",
            "max_sequence_length": MAX_SEQ_LEN,
            "n_full_variants": len(data["valid_variants"]),
            "n_subset_variants": len(subset["valid_variants"]),
            "n_excluded_variants": int((~mask).sum()),
            "n_full_genes": len(set(data["genes_arr"])),
            "n_subset_genes": len(set(subset["genes_arr"])),
            "full_class_distribution": full_class_distribution,
            "subset_class_distribution": subset_class_distribution,
        },
        "inputs": {
            "valid_variants_path": str(VALID_VARIANTS_JSON),
            "sequences_path": str(SEQUENCES_JSON),
            "wt_embedding_path": str(EMB_WT_MEAN),
            "full_variant_fingerprint": embedding_variant_fingerprint(data["valid_variants"]),
            "subset_variant_fingerprint": embedding_variant_fingerprint(subset["valid_variants"]),
            "selection_sequence_fingerprint": sequence_selection_fingerprint(
                data["valid_variants"], sequences
            ),
            "subset_wt_embedding_fingerprint": embedding_fingerprint(subset["emb_wt_mean"]),
        },
        "n_seeds": len(seed_results),
        "n_folds": args.n_folds,
        "n_bootstrap_resamples": args.n_boot,
        "input_fingerprints": input_fingerprints,
        "seed_files": [filename for _seed, filename, _result in seed_results],
        "split_gap_summary": split_gap_summary,
        "across_seed": aggregated,
    }
    write_result_json(
        WT_IDENTITY_SENSITIVITY_AGGREGATE_JSON,
        payload,
        seeds=list(range(args.seeds)),
        indent=2,
    )
    print_table(aggregated)
    split_gap = read_seed_point_estimate(
        split_gap_summary["gene_minus_family_seed_aggregate"]
    )
    if split_gap.available:
        print(
            f"\nClaim 2B sensitivity: gene-minus-family macro-F1 = "
            f"{split_gap.value:.3f} (seed mean of the row-aligned paired gap); "
            "each seed's own interval is in per_seed_interval."
        )
    else:
        print(f"\nClaim 2B sensitivity is unavailable ({split_gap.message}).")
    print(f"Wrote {WT_IDENTITY_SENSITIVITY_AGGREGATE_JSON}")


if __name__ == "__main__":
    main()
