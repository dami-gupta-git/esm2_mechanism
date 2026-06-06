"""
Parser for the full Tsuboyama 2023 point-mutant stability dataset.

Source: Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv
(Tsuboyama et al. 2023, "Mega-scale experimental analysis of protein folding
stability in biology and design"). This is the scaled-up replacement for the
27-protein S1724 benchmark used in result_21.

Scope decisions (see for_me/explain_stability.md):
  - Natural domains only. De novo designed mini-proteins (WT_name like "EA|run2…",
    "HHH", "XX|run1") are dropped — they have no Pfam family, so they would break
    the family-split's comparability to the mechanism/pathogenicity experiments.
    A natural domain is identified positively by a real 4-char PDB id in WT_name.
  - Single-point missense only. Insertions ("ins…"), deletions ("del…"),
    multi-mutants, and the wild-type rows themselves are excluded from the variant
    set (WT rows are used only to recover each domain's reference sequence).
  - ddG_ML == "-" (unreliable fit) rows are DROPPED, never imputed — 0.0 is a
    plausible real ΔΔG, so imputing would contaminate the probe.

Relevant columns:
  WT_name   parent domain id (e.g. "1A0N.pdb", or "1BK2.pdb_L7S" for a sub-variant)
  mut_type  "wt" | substitution "D1Q" | "insX.." | "delX.." | ...
  aa_seq    the variant's amino-acid sequence (the MUTANT sequence for mutant rows)
  ddG_ML    ML-fit ΔΔG (mutation effect); "-" when the fit was unreliable

Output: list of dicts {protein, mutation_code, wt_seq, mut_seq, var_pos, ddg},
cached to MEGASCALE_TSUBOYAMA_VARIANTS_JSON.
"""

import csv
import functools
import math
import os
import re
import sys

from esm2_mech.utils.constants import (
    MEGASCALE_PDB_ID_REGEX,
    MEGASCALE_SUBSTITUTION_REGEX,
    MEGASCALE_WT_MUT_TYPE,
    MEGASCALE_DDG_MISSING,
)
from esm2_mech.utils.io import atomic_write_json, load_json_or_discard
from esm2_mech.utils.paths import (
    MEGASCALE_TSUBOYAMA_CSV,
    MEGASCALE_TSUBOYAMA_VARIANTS_JSON,
)

print = functools.partial(print, flush=True)

# The dna_seq / aa_seq_full columns can be long; lift the csv field-size cap so
# the DictReader never truncates a row mid-parse.
csv.field_size_limit(sys.maxsize)

_PDB_ID = re.compile(MEGASCALE_PDB_ID_REGEX)
_SUBSTITUTION = re.compile(MEGASCALE_SUBSTITUTION_REGEX)


def _is_bare_natural_domain(wt_name: str) -> bool:
    """True if WT_name is a bare natural-domain reference (a real PDB id, no suffix).

    We accept ONLY bare WT_names like "1BK2.pdb". A suffixed name such as
    "1BK2.pdb_L7S" denotes a row measured against the L7S MUTANT background, not
    the wild type — its mutation code and ddG_ML are relative to that mutant, so
    folding it into the parent domain would pair the variant with the wrong
    reference sequence. Such rows are excluded entirely (both the alt-reference
    "wt" rows and the substitutions built on them).
    """
    return "_" not in wt_name and bool(_PDB_ID.match(wt_name))


def load_tsuboyama_variants(csv_path=None, cache_path=None):
    """Parse the Tsuboyama point-mutant table into single-point missense variants.

    Two passes over the CSV, both restricted to BARE natural-domain WT_names
    (a real PDB id with no '_MUT' background suffix — see _is_bare_natural_domain):
      1. Collect each domain's wild-type sequence from its mut_type == "wt" row.
      2. Build one variant record per single-substitution row with a parseable
         numeric ddG_ML that passes the wt/mut residue sanity check.

    Returns the variant list and also writes it to cache_path.
    """
    csv_path = str(csv_path or MEGASCALE_TSUBOYAMA_CSV)
    cache_path = str(cache_path or MEGASCALE_TSUBOYAMA_VARIANTS_JSON)

    if os.path.exists(cache_path):
        print(f"Loading cached Tsuboyama variants from {cache_path} ...")
        # A run killed mid-write can leave a truncated cache; load_json_or_discard
        # deletes it and returns None so we fall through and re-parse.
        cached = load_json_or_discard(cache_path)
        if cached is not None:
            return cached

    # ── Pass 1: wild-type reference sequence per bare natural domain ──────────
    # If two "wt" rows give the SAME domain DIFFERENT sequences we cannot know
    # which reference its variants belong to — that is a data-integrity failure,
    # not a benign first-wins, so we raise rather than silently keep one.
    wt_seqs = {}
    wt_collisions = {}
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["mut_type"] != MEGASCALE_WT_MUT_TYPE:
                continue
            wt_name = row["WT_name"].strip()
            if not _is_bare_natural_domain(wt_name):
                continue
            seq = row["aa_seq"].strip()
            if not seq:
                continue
            if wt_name in wt_seqs:
                if wt_seqs[wt_name] != seq:
                    wt_collisions.setdefault(wt_name, set()).add(seq)
                continue
            wt_seqs[wt_name] = seq
    if wt_collisions:
        examples = ", ".join(sorted(wt_collisions)[:5])
        raise ValueError(
            f"{len(wt_collisions)} domain(s) have conflicting wild-type sequences "
            f"across 'wt' rows — cannot determine the reference for their variants: "
            f"{examples}"
        )
    print(f"Pass 1: {len(wt_seqs)} natural domains with a wild-type sequence")

    # ── Pass 2: single-substitution variants ─────────────────────────────────
    variants = []
    skipped = {
        "not_bare_natural": 0,   # designs + _MUT-background rows
        "no_wt_seq": 0,
        "not_substitution": 0,
        "ddg_missing": 0,
        "ddg_unparseable": 0,
        "length_mismatch": 0,
        "pos_out_of_range": 0,
        "wt_residue_mismatch": 0,
        "mut_residue_mismatch": 0,
    }
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mut_type = row["mut_type"].strip()
            if mut_type == MEGASCALE_WT_MUT_TYPE:
                continue

            wt_name = row["WT_name"].strip()
            if not _is_bare_natural_domain(wt_name):
                skipped["not_bare_natural"] += 1
                continue
            wt_seq = wt_seqs.get(wt_name)
            if wt_seq is None:
                skipped["no_wt_seq"] += 1
                continue

            match = _SUBSTITUTION.match(mut_type)
            if not match:
                skipped["not_substitution"] += 1
                continue
            aa_wt, pos_str, aa_mut = match.groups()
            var_pos = int(pos_str)  # 1-indexed

            ddg_raw = row["ddG_ML"].strip()
            if ddg_raw == MEGASCALE_DDG_MISSING or ddg_raw == "":
                skipped["ddg_missing"] += 1
                continue
            try:
                ddg = float(ddg_raw)
            except ValueError:
                # Messy published CSV — a non-"-" unparseable value is dropped,
                # never imputed or allowed to abort the whole parse.
                skipped["ddg_unparseable"] += 1
                continue
            if not math.isfinite(ddg):
                # "nan"/"inf"/"-inf" parse via float() but are not real ΔΔG values;
                # drop them so a non-finite label never enters the variant set.
                skipped["ddg_unparseable"] += 1
                continue

            mut_seq = row["aa_seq"].strip()
            # A true single substitution preserves length. A WT/mut length
            # disagreement means aa_seq is not the matching point-mutant (an indel
            # or wrong row) — surface it under its own bucket rather than hiding it
            # in pos_out_of_range or letting a longer mutant slip past the bound.
            if len(mut_seq) != len(wt_seq):
                skipped["length_mismatch"] += 1
                continue
            if var_pos < 1 or var_pos > len(wt_seq):
                skipped["pos_out_of_range"] += 1
                continue
            # Sanity: WT residue at var_pos matches the mutation code's source aa,
            # and the mutant sequence carries the substituted aa at the same spot.
            if wt_seq[var_pos - 1] != aa_wt:
                skipped["wt_residue_mismatch"] += 1
                continue
            if mut_seq[var_pos - 1] != aa_mut:
                skipped["mut_residue_mismatch"] += 1
                continue

            variants.append(
                {
                    "protein": wt_name,
                    "mutation_code": mut_type,
                    "wt_seq": wt_seq,
                    "mut_seq": mut_seq,
                    "var_pos": var_pos,
                    "ddg": ddg,
                }
            )

    n_domains = len({v["protein"] for v in variants})
    print(
        f"Tsuboyama: {len(variants)} single-point missense variants across "
        f"{n_domains} natural domains"
    )
    print(f"  skipped: {skipped}")
    if skipped["no_wt_seq"]:
        print(
            f"  NOTE: {skipped['no_wt_seq']} substitution rows had no bare-WT "
            f"sequence for their domain (dropped)"
        )

    atomic_write_json(cache_path, variants)
    print(f"Cached to {cache_path}")
    return variants


if __name__ == "__main__":
    load_tsuboyama_variants()
