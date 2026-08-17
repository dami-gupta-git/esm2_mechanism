"""Cross-file gene-coverage consistency check for the fetch-pipeline outputs."""

from __future__ import annotations

import argparse
import csv
import functools
import json
from pathlib import Path
from typing import Optional

import numpy as np

from esm2_mech.utils.paths import (
    BADONYI_FEATURES_ALIGNED,
    BADONYI_FEATURES_TSV,
    DATA_DIR,
    ENZYME_LABELS_TSV,
    GENE_LIST_TSV,
    GENE_UNIVERSE,
    PFAM_JSON,
    PROTEOME_FEATURES_ALIGNED,
    PROTEOME_FEATURES_TSV,
)

print = functools.partial(print, flush=True)

MAX_EXAMPLES = 10


def load_tsv_genes(path: Path, column: str = "gene") -> set[str]:
    """Return the set of gene symbols in a TSV column."""
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
    """Return the top-level key set of a JSON object."""
    with open(path) as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object keyed by gene")
    return {str(key).strip() for key in data}


def count_tsv_rows(path: Path) -> int:
    """Number of data rows (excluding header)."""
    with open(path, newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader, None)  # header
        return sum(1 for row in reader if row)


def _examples(genes: set[str]) -> str:
    sample = sorted(genes)[:MAX_EXAMPLES]
    suffix = ", ..." if len(genes) > MAX_EXAMPLES else ""
    return ", ".join(sample) + suffix


def check_subset(name: str, sub: set[str], sup: set[str], sub_name: str, sup_name: str) -> bool:
    """Pass if sub is a subset of sup."""
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
    """Pass if the two gene sets are exactly equal."""
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
    """Pass if the .npy first-axis length equals expected."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory holding the pipeline outputs.")
    args = parser.parse_args()
    data = args.data_dir

    paths = {
        "gene_list": data / GENE_LIST_TSV.name,
        "gene_universe": data / GENE_UNIVERSE.name,
        "proteome": data / PROTEOME_FEATURES_TSV.name,
        "badonyi": data / BADONYI_FEATURES_TSV.name,
        "enzyme": data / ENZYME_LABELS_TSV.name,
        "pfam": data / PFAM_JSON.name,
        "proteome_npy": data / PROTEOME_FEATURES_ALIGNED.name,
        "badonyi_npy": data / BADONYI_FEATURES_ALIGNED.name,
    }

    results: list[bool] = []

    def have(*keys: str) -> bool:
        missing = [keys[i] for i, key in enumerate(keys) if not paths[key].exists()]
        if missing:
            print(f"[SKIP] missing file(s): {', '.join(str(paths[m]) for m in missing)}")
        return not missing

    print("=== Gene-coverage consistency check ===\n")

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

    if have("gene_universe", "gene_list"):
        results.append(
            check_subset("universe⊆gene_list", sets["gene_universe"], sets["gene_list"], "universe", "gene_list")
        )

    if have("gene_universe", "pfam"):
        results.append(
            check_subset("universe⊆pfam", sets["gene_universe"], sets["pfam"], "universe", "pfam_families")
        )

    if have("proteome", "gene_universe"):
        results.append(
            check_equal("proteome==universe", sets["proteome"], sets["gene_universe"], "proteome", "universe")
        )

    if have("badonyi", "gene_universe"):
        results.append(
            check_equal("badonyi==universe", sets["badonyi"], sets["gene_universe"], "badonyi", "universe")
        )

    if have("enzyme", "gene_list"):
        results.append(
            check_equal("enzyme==gene_list", sets["enzyme"], sets["gene_list"], "enzyme", "gene_list")
        )

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
