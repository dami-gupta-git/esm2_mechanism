"""
Download the ClinGen gene-dosage curation file used by build_proteome_features.

Tries each ClinGen URL in turn and saves the raw TSV to the downloads dir the
feature builder reads first (data/downloads/ClinGen_gene_curation_list_GRCh38.tsv).
After download it verifies the header is parseable (gene / HI-description /
TS-description columns locatable by name) so a bad or moved file fails here
rather than silently yielding empty HI_score/TS_score columns downstream.

Usage:
    python -m esm2_mechanism.fetch_data.fetch_clingen
    python -m esm2_mechanism.fetch_data.fetch_clingen --force   # re-download
"""

from __future__ import annotations

import argparse
import functools

from esm2_mechanism.fetch_data.build_proteome_features import (
    CLINGEN_MANUAL,
    CLINGEN_URLS,
    _download_file,
    find_clingen_columns,
)

print = functools.partial(print, flush=True)


def verify(path) -> bool:
    """Return True if the file's header exposes gene/HI/TS columns by name."""
    with open(path) as f:
        lines = f.readlines()
    idx_gene, idx_hi, idx_ts = find_clingen_columns(lines)
    print(f"  header columns → gene={idx_gene}, HI={idx_hi}, TS={idx_ts}")
    if idx_gene is None or idx_hi is None or idx_ts is None:
        print("ERROR: ClinGen header not recognised — file will not parse in build_proteome_features")
        return False
    n_data = sum(1 for ln in lines if ln.strip() and not ln.strip().startswith("#"))
    print(f"  data rows: {n_data}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the ClinGen gene-dosage file.")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached.")
    args = parser.parse_args()

    CLINGEN_MANUAL.parent.mkdir(parents=True, exist_ok=True)

    print("=== Downloading ClinGen gene-dosage curation file ===")
    ok = False
    for url in CLINGEN_URLS:
        ok = _download_file(url, CLINGEN_MANUAL, force=args.force)
        if ok:
            break

    if not ok:
        print("ERROR: all ClinGen download URLs failed.")
        print("Place the file manually at:")
        print(f"  {CLINGEN_MANUAL}")
        return 1

    print(f"Saved: {CLINGEN_MANUAL}")
    if not verify(CLINGEN_MANUAL):
        return 1
    print("OK — file is parseable by build_proteome_features.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
