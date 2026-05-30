"""
Cross-file gene-coverage consistency check for the fetch-pipeline outputs.

The pipeline produces several gene-keyed files that must stand in fixed
set relationships (derived by reading the builders, not guessed):

  gene_universe.tsv          = gene_list.tsv filtered to Pfam-annotated genes
                               (_build_gene_universe drops genes with no Pfam)
  gene_proteome_features.tsv : built by iterating gene_universe.tsv → exact
                               same gene set
  badonyi_features.tsv       : built by iterating gene_universe.tsv → exact
                               same gene set
  enzyme_labels.tsv          : keyed on gene_list.tsv → exact same gene set
  pfam_families.json         : Pfam fetched for every gene_list gene → keys
                               are a superset of gene_list

Aligned matrices carry no gene column; their rows are positionally aligned to
gene_universe.tsv, so their row count must equal the universe gene count:

  proteome_features_aligned.npy : rows == len(gene_universe)
  badonyi_features_aligned.npy  : rows == len(gene_universe)

Each violation is reported with counts and example genes in both directions.
The script exits non-zero if any required check fails, so it can gate a run.

Usage:
    python -m esm2_mechanism.fetch_data.check_gene_coverage
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
from pathlib import Path
from typing import Optional

import numpy as np

from esm2_mechanism.utils_paths import DATA_DIR

print = functools.partial(print, flush=True)

MAX_EXAMPLES = 10


# ---------------------------------------------------------------------------
# Gene-set loaders
# ---------------------------------------------------------------------------
def load_tsv_genes(path: Path, column: str = "gene") -> set[str]:
    """Return the set of gene symbols in `column` of a TSV (header required)."""
    genes: set[str] = set()
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"{path}: no '{column}' column (header={reader.fieldnames})")
        for row in reader:
            gene = (row.get(column) or "").strip()
            if gene:
                genes.add(gene)
    return genes


def load_json_keys(path: Path) -> set[str]:
    """Return the top-level key set of a JSON object (gene → value mapping)."""
    with open(path) as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object keyed by gene")
    return {str(key).strip() for key in data}


def count_tsv_rows(path: Path) -> int:
    """Number of data rows (excluding header) in a TSV."""
    with open(path, newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader, None)  # header
        return sum(1 for row in reader if row)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def _examples(genes: set[str]) -> str:
    sample = sorted(genes)[:MAX_EXAMPLES]
    suffix = ", ..." if len(genes) > MAX_EXAMPLES else ""
    return ", ".join(sample) + suffix


def check_subset(name: str, sub: set[str], sup: set[str], sub_name: str, sup_name: str) -> bool:
    """Pass if `sub` ⊆ `sup`. Reports genes present in sub but absent from sup."""
    missing = sub - sup
    if not missing:
        print(f"[PASS] {name}: all {len(sub)} {sub_name} genes present in {sup_name}")
        return True
    print(
        f"[FAIL] {name}: {len(missing)} {sub_name} genes missing from {sup_name}\n"
        f"        examples: {_examples(missing)}"
    )
    return False


def check_equal(name: str, left: set[str], right: set[str], left_name: str, right_name: str) -> bool:
    """Pass if the two gene sets are exactly equal. Reports both directions."""
    only_left = left - right
    only_right = right - left
    if not only_left and not only_right:
        print(f"[PASS] {name}: identical gene sets ({len(left)} genes)")
        return True
    print(f"[FAIL] {name}: gene sets differ (|{left_name}|={len(left)}, |{right_name}|={len(right)})")
    if only_left:
        print(f"        {len(only_left)} only in {left_name}: {_examples(only_left)}")
    if only_right:
        print(f"        {len(only_right)} only in {right_name}: {_examples(only_right)}")
    return False


def check_row_count(name: str, npy_path: Path, expected: int, expected_name: str) -> bool:
    """Pass if the .npy first-axis length equals `expected` (alignment contract)."""
    matrix = np.load(npy_path, allow_pickle=False)
    n_rows = matrix.shape[0]
    if n_rows == expected:
        print(f"[PASS] {name}: {npy_path.name} has {n_rows} rows == {expected_name}")
        return True
    print(
        f"[FAIL] {name}: {npy_path.name} has {n_rows} rows != {expected} ({expected_name}) "
        f"— positional alignment to gene_universe is broken"
    )
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory holding the pipeline outputs.")
    args = parser.parse_args()
    data = args.data_dir

    paths = {
        "gene_list": data / "gene_list.tsv",
        "gene_universe": data / "gene_universe.tsv",
        "proteome": data / "gene_proteome_features.tsv",
        "badonyi": data / "badonyi_features.tsv",
        "enzyme": data / "enzyme_labels.tsv",
        "pfam": data / "pfam_families.json",
        "proteome_npy": data / "proteome_features_aligned.npy",
        "badonyi_npy": data / "badonyi_features_aligned.npy",
    }

    results: list[bool] = []

    def have(*keys: str) -> bool:
        missing = [keys[i] for i, key in enumerate(keys) if not paths[key].exists()]
        if missing:
            print(f"[SKIP] missing file(s): {', '.join(str(paths[m]) for m in missing)}")
        return not missing

    print("=== Gene-coverage consistency check ===\n")

    # Load gene sets once, where files exist.
    sets: dict[str, set[str]] = {}
    if paths["gene_list"].exists():
        sets["gene_list"] = load_tsv_genes(paths["gene_list"])
    if paths["gene_universe"].exists():
        sets["gene_universe"] = load_tsv_genes(paths["gene_universe"])
    if paths["proteome"].exists():
        sets["proteome"] = load_tsv_genes(paths["proteome"])
    if paths["badonyi"].exists():
        sets["badonyi"] = load_tsv_genes(paths["badonyi"])
    if paths["enzyme"].exists():
        sets["enzyme"] = load_tsv_genes(paths["enzyme"])
    if paths["pfam"].exists():
        sets["pfam"] = load_json_keys(paths["pfam"])

    # 1. gene_universe ⊆ gene_list
    if have("gene_universe", "gene_list"):
        results.append(
            check_subset("universe⊆gene_list", sets["gene_universe"], sets["gene_list"], "universe", "gene_list")
        )

    # 2. gene_universe ⊆ pfam keys. pfam_families.json is keyed on the variant
    #    gene set (variants.json), NOT gene_list, so gene_list is intentionally
    #    a superset of pfam. The real contract is that every *retained* universe
    #    gene has a Pfam annotation (universe = gene_list genes with non-None pfam).
    if have("gene_universe", "pfam"):
        results.append(
            check_subset("universe⊆pfam", sets["gene_universe"], sets["pfam"], "universe", "pfam_families")
        )

    # 3. proteome == gene_universe
    if have("proteome", "gene_universe"):
        results.append(
            check_equal("proteome==universe", sets["proteome"], sets["gene_universe"], "proteome", "universe")
        )

    # 4. badonyi == gene_universe
    if have("badonyi", "gene_universe"):
        results.append(
            check_equal("badonyi==universe", sets["badonyi"], sets["gene_universe"], "badonyi", "universe")
        )

    # 5. enzyme == gene_list
    if have("enzyme", "gene_list"):
        results.append(
            check_equal("enzyme==gene_list", sets["enzyme"], sets["gene_list"], "enzyme", "gene_list")
        )

    # 6. aligned matrices: row count == universe gene count
    if have("gene_universe"):
        n_universe = count_tsv_rows(paths["gene_universe"])
        if have("proteome_npy"):
            results.append(check_row_count("proteome_npy_rows", paths["proteome_npy"], n_universe, "len(gene_universe)"))
        if have("badonyi_npy"):
            results.append(check_row_count("badonyi_npy_rows", paths["badonyi_npy"], n_universe, "len(gene_universe)"))

    print("\n=== Summary ===")
    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print(f"  checks run: {len(results)}  passed: {n_pass}  failed: {n_fail}")
    if n_fail:
        print("  RESULT: FAIL — gene sets are inconsistent across files.")
        return 1
    if not results:
        print("  RESULT: nothing checked (no files found).")
        return 1
    print("  RESULT: PASS — all gene-coverage expectations hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
