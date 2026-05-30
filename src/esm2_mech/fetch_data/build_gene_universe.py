"""
Build gene_universe.tsv — step 2 of the gene universe pipeline.

Filters gene_list.tsv to genes with a Pfam family assignment. The output
gene_universe.tsv is the canonical aligned row order for all feature matrices
(proteome_features_aligned.npy, badonyi_features_aligned.npy, etc.).

Must run after fetch_annotations --step pfam.

Inputs : data/gene_list.tsv
         data/pfam_families.json
Output : data/gene_universe.tsv
         Columns: gene, mechanism, uniprot_id, source, g2p_disagrees, pfam_family

Usage:
    python -m esm2_mech.fetch_data.build_gene_universe
"""

from __future__ import annotations

import csv
import functools
import json

from esm2_mech.utils.paths import GENE_LIST_TSV, GENE_UNIVERSE, PFAM_JSON

print = functools.partial(print, flush=True)


def main() -> None:
    with open(PFAM_JSON) as f:
        pfam: dict[str, str | None] = json.load(f)
    n_annotated = sum(1 for v in pfam.values() if v is not None)
    print(f"Pfam families loaded: {len(pfam)} genes, {n_annotated} annotated")

    rows_in: list[dict] = []
    with open(GENE_LIST_TSV, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows_in.append(row)
    print(f"gene_list: {len(rows_in)} genes")

    rows_out: list[dict] = []
    dropped: list[str] = []
    for row in rows_in:
        gene = row["gene"]
        pfam_family = pfam.get(gene)
        if pfam_family is None:
            dropped.append(gene)
            continue
        rows_out.append({**row, "pfam_family": pfam_family})

    if dropped:
        print(f"Dropped {len(dropped)} genes with no Pfam annotation: {sorted(dropped)}")
    print(f"gene_universe: {len(rows_out)} genes retained")

    fieldnames = ["gene", "mechanism", "uniprot_id", "source", "g2p_disagrees", "pfam_family"]
    with open(GENE_UNIVERSE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Wrote {GENE_UNIVERSE}")


if __name__ == "__main__":
    main()
