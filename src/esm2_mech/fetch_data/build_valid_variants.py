"""Build the filtered variant list used by embedding and analysis scripts."""

import functools
import json

from esm2_mech.utils.io import atomic_write_json
from esm2_mech.utils.paths import VARIANTS_JSON, SEQUENCES_JSON, DATA_DIR
from esm2_mech.utils.sequences import window_sequence, apply_missense

print = functools.partial(print, flush=True)

VALID_VARIANTS_OUT = DATA_DIR / "valid_variants.json"


def build_valid_variants() -> list[dict]:
    if not VARIANTS_JSON.exists():
        raise FileNotFoundError(
            f"{VARIANTS_JSON} not found — run fetch_data/fetch_variants.py --step merge first"
        )
    with open(VARIANTS_JSON) as f:
        variants = json.load(f)
    print(f"Loaded {len(variants):,} variants")

    if not SEQUENCES_JSON.exists():
        raise FileNotFoundError(
            f"{SEQUENCES_JSON} not found — run fetch_data/fetch_sequences first"
        )
    with open(SEQUENCES_JSON) as f:
        seq_cache = json.load(f)
    print(f"Loaded {len(seq_cache):,} sequences")

    valid = []
    skipped_no_uid = 0
    skipped_no_seq = 0
    skipped_no_fields = 0
    skipped_invalid = 0
    for v in variants:
        uid = v.get("uniprot_id")
        if not uid:
            skipped_no_uid += 1
            continue
        if uid not in seq_cache:
            skipped_no_seq += 1
            continue
        aa_pos = v.get("aa_pos")
        aa_wt = v.get("aa_wt")
        aa_mut = v.get("aa_mut")
        if aa_pos is None or not aa_wt or not aa_mut:
            skipped_no_fields += 1
            continue
        wt_win, new_pos, _ = window_sequence(seq_cache[uid], aa_pos)
        mut_win = apply_missense(wt_win, new_pos, aa_wt, aa_mut)
        if mut_win is None:
            skipped_invalid += 1
            continue
        valid.append(v)

    if skipped_no_uid:
        print(f"WARNING: {skipped_no_uid} variants skipped — missing UniProt ID")
    if skipped_no_seq:
        print(f"WARNING: {skipped_no_seq} variants skipped — UniProt ID not in sequence cache")
    if skipped_no_fields:
        print(f"WARNING: {skipped_no_fields} variants skipped — missing aa_pos/aa_wt/aa_mut")
    if skipped_invalid:
        print(f"WARNING: {skipped_invalid} variants skipped — invalid WT/mut window")
    print(f"Valid variants: {len(valid):,}")
    return valid


def main():
    valid = build_valid_variants()
    atomic_write_json(VALID_VARIANTS_OUT, valid)
    print(f"Wrote {VALID_VARIANTS_OUT}")


if __name__ == "__main__":
    main()
