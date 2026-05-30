"""
Validate the cached UniProt sequences against the variants that reference them.

Two independent checks are run over data/cache/sequences.json (plus the optional
uniprot_sequences_extended.json overlay):

  1. Sequence sanity — every cached sequence is non-empty and contains only valid
     amino-acid letters (the 20 standard residues plus U = selenocysteine, the
     only non-standard residue UniProt encodes inline).

  2. WT-residue agreement — for each variant in variants.json, the cached
     sequence for its uniprot_id must carry the stated wild-type amino acid at
     the 1-indexed aa_pos. A disagreement means the cached sequence is a
     different isoform than the one the variant was numbered against. Such
     variants are silently dropped downstream (apply_missense returns None), so
     this check surfaces how many — and which proteins — are affected.

Disagreements are grouped per protein because isoform mismatches are systematic
(all/most variants of one protein fail together), not scattered errors.

This is a diagnostic, not a gate: WT mismatches are expected for a few proteins
and are handled safely downstream. The script still exits non-zero if a cached
sequence fails the sanity check (an invalid sequence is a real corruption), so
it can guard a pipeline run against that case.

Usage:
    python -m esm2_mechanism.fetch_data.check_sequence_cache
"""

from __future__ import annotations

import argparse
import functools
import json
from collections import defaultdict
from pathlib import Path

from esm2_mechanism.utils_paths import (
    SEQUENCES_JSON,
    SEQUENCES_EXTENDED_JSON,
    VARIANTS_JSON,
)

print = functools.partial(print, flush=True)

# 20 standard amino acids + U (selenocysteine), the only non-standard residue
# UniProt writes inline in canonical sequences.
VALID_AA = set("ACDEFGHIKLMNPQRSTVWYU")
MAX_EXAMPLES = 15


def load_json(path: Path):
    """Load a JSON file, surfacing a clear error if it is missing or corrupt."""
    if not path.exists():
        raise FileNotFoundError(f"required input not found: {path}")
    try:
        with open(path) as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc})") from exc


def load_sequences(
    seq_path: Path, ext_path: Path
) -> dict[str, str]:
    """Return uniprot_id -> sequence, with the extended overlay taking precedence."""
    sequences: dict[str, str] = dict(load_json(seq_path))
    if ext_path.exists():
        extended = load_json(ext_path)
        print(f"  overlaying {len(extended)} extended sequences from {ext_path.name}")
        sequences.update(extended)
    return sequences


def check_sequence_sanity(sequences: dict[str, str]) -> list[str]:
    """Return a list of uniprot_ids whose sequence is empty or has invalid letters."""
    bad: list[str] = []
    for uniprot_id, seq in sequences.items():
        if not seq:
            bad.append(uniprot_id)
            continue
        invalid = set(seq) - VALID_AA
        if invalid:
            print(
                f"  [BAD] {uniprot_id}: {len(seq)} aa, invalid letters {sorted(invalid)}"
            )
            bad.append(uniprot_id)
    return bad


def check_wt_agreement(
    sequences: dict[str, str], variants: list[dict]
) -> tuple[int, int, int, int, dict[str, list[int]]]:
    """Cross-check each variant's WT residue against the cached sequence.

    Returns (checked, missing_seq, out_of_bounds, mismatched, per_protein) where
    per_protein maps uniprot_id -> [total_checked, mismatched] for proteins with
    at least one mismatch.
    """
    checked = missing_seq = out_of_bounds = mismatched = 0
    per_protein: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for variant in variants:
        uniprot_id = variant.get("uniprot_id")
        if not uniprot_id:
            continue
        seq = sequences.get(uniprot_id)
        if not seq:
            missing_seq += 1
            continue
        pos = variant["aa_pos"]
        wt_aa = variant["aa_wt"]
        if pos < 1 or pos > len(seq):
            out_of_bounds += 1
            continue
        checked += 1
        per_protein[uniprot_id][0] += 1
        if seq[pos - 1] != wt_aa:
            mismatched += 1
            per_protein[uniprot_id][1] += 1

    mismatched_proteins = {
        uniprot_id: counts
        for uniprot_id, counts in per_protein.items()
        if counts[1] > 0
    }
    return checked, missing_seq, out_of_bounds, mismatched, mismatched_proteins


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sequences", type=Path, default=SEQUENCES_JSON)
    parser.add_argument("--extended", type=Path, default=SEQUENCES_EXTENDED_JSON)
    parser.add_argument("--variants", type=Path, default=VARIANTS_JSON)
    args = parser.parse_args()

    print("=== Sequence-cache validation ===\n")

    print(f"Loading sequences from {args.sequences} ...")
    sequences = load_sequences(args.sequences, args.extended)
    print(f"  {len(sequences)} cached sequences\n")

    print("Check 1: sequence sanity (alphabet + non-empty)")
    bad_sequences = check_sequence_sanity(sequences)
    if bad_sequences:
        print(f"  [FAIL] {len(bad_sequences)} invalid sequence(s): {bad_sequences[:MAX_EXAMPLES]}")
    else:
        print(f"  [PASS] all {len(sequences)} sequences non-empty with valid letters")

    print("\nCheck 2: WT-residue agreement with variants")
    variants = load_json(args.variants)
    print(f"  {len(variants)} variants in {args.variants.name}")
    checked, missing_seq, out_of_bounds, mismatched, mismatched_proteins = (
        check_wt_agreement(sequences, variants)
    )
    print(
        f"  checked: {checked}  no_sequence: {missing_seq}  "
        f"out_of_bounds: {out_of_bounds}  WT_mismatch: {mismatched}"
    )
    if mismatched_proteins:
        print(
            f"  {len(mismatched_proteins)} protein(s) with WT disagreement "
            "(likely isoform mismatch — these variants are dropped downstream):"
        )
        for uniprot_id, (total, mism) in sorted(
            mismatched_proteins.items(), key=lambda item: -item[1][1]
        ):
            print(f"    {uniprot_id}: {mism}/{total} variants mismatch")
    else:
        print("  [PASS] every variant's WT residue matches the cached sequence")

    print("\n=== Summary ===")
    print(f"  sequences checked:   {len(sequences)}")
    print(f"  invalid sequences:   {len(bad_sequences)}")
    print(f"  variants checked:    {checked}")
    print(f"  WT mismatches:       {mismatched} across {len(mismatched_proteins)} proteins")
    print(f"  out-of-bounds:       {out_of_bounds}")

    # Only invalid sequences are a hard failure. WT mismatches are expected for a
    # few isoform-shifted proteins and are handled safely downstream.
    if bad_sequences:
        print("  RESULT: FAIL — cached sequences contain invalid/empty entries.")
        return 1
    print("  RESULT: PASS — all cached sequences are well-formed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
