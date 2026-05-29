"""
Fetch enzyme class annotations from UniProt for all genes in the merged dataset
and produce a 4-class label file (kinase / protease / oxidoreductase / non-enzyme).

EC number hierarchy:
  1.x.x.x  Oxidoreductase
  2.7.x.x  Kinase (phosphotransferases — subset of EC 2, flagged via KW-0418)
  3.4.x.x  Protease (peptidases — subset of EC 3)
  other    Non-enzyme (or enzyme class outside the 4-class scheme)

The 4-class scheme focuses on classes that are (a) well-populated in human disease
genes and (b) structurally stereotyped, making them a clean positive control for
ESM-2 WT embedding classification.

Outputs:
  data/enzyme_labels.tsv  — one row per gene in merged_gene_list
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import time
import urllib.request
import urllib.error
from collections import defaultdict
from pathlib import Path
from typing import Optional

print = functools.partial(print, flush=True)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/{acc}?format=json"
BATCH_SIZE = 50
RETRIES = 3
BACKOFF_BASE = 2.0

ENZYME_4CLASS_PRIORITY = ["kinase", "protease", "oxidoreductase"]


# ---------------------------------------------------------------------------
# UniProt fetch
# ---------------------------------------------------------------------------

def fetch_uniprot_entry(acc: str) -> Optional[dict]:
    url = UNIPROT_SEARCH_URL.format(acc=acc)
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "dami-mechanism-prediction/0.1 (academic research)",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            print(f"  HTTP {e.code} for {acc} (attempt {attempt+1})")
        except Exception as e:
            print(f"  Error fetching {acc} (attempt {attempt+1}): {e}")
        if attempt < RETRIES - 1:
            time.sleep(BACKOFF_BASE * (attempt + 1))
    return None


def parse_ec_and_keywords(entry: dict) -> tuple:
    """Extract EC numbers and keyword IDs from a UniProt JSON entry."""
    ec_numbers = []
    keyword_ids = []

    # EC numbers live in proteinDescription -> recommendedName -> ecNumbers
    # and also in comments (catalytic activity)
    try:
        prot = entry.get("proteinDescription", {})
        for name_block in [prot.get("recommendedName", {})] + prot.get("alternativeNames", []):
            for ec_obj in name_block.get("ecNumbers", []):
                val = ec_obj.get("value", "")
                if val:
                    ec_numbers.append(val)
    except Exception:
        pass

    # Also check catalytic activity comments for EC numbers
    try:
        for comment in entry.get("comments", []):
            if comment.get("commentType") == "CATALYTIC ACTIVITY":
                reaction = comment.get("reaction", {})
                for ec_obj in reaction.get("ecNumber", []):
                    val = ec_obj if isinstance(ec_obj, str) else ec_obj.get("value", "")
                    if val and val not in ec_numbers:
                        ec_numbers.append(val)
    except Exception:
        pass

    # Keywords
    try:
        for kw in entry.get("keywords", []):
            kid = kw.get("id", "")
            if kid:
                keyword_ids.append(kid)
    except Exception:
        pass

    return ec_numbers, keyword_ids


def assign_4class(ec_numbers: list[str], keyword_ids: list[str]) -> str:
    """
    Assign one of: kinase, protease, oxidoreductase, non-enzyme.

    Priority order (most specific first):
      1. kinase     — KW-0418 (kinase keyword) OR EC 2.7.x.x
      2. protease   — EC 3.4.x.x
      3. oxidoreductase — EC 1.x.x.x
      4. non-enzyme — everything else
    """
    # Kinase: UniProt keyword KW-0418 is the authoritative source
    if "KW-0418" in keyword_ids:
        return "kinase"
    # Also catch EC 2.7 without keyword (some entries lack it)
    for ec in ec_numbers:
        if ec.startswith("2.7."):
            return "kinase"

    # Protease: EC 3.4.x.x
    for ec in ec_numbers:
        if ec.startswith("3.4."):
            return "protease"

    # Oxidoreductase: EC 1.x.x.x
    for ec in ec_numbers:
        if ec.startswith("1."):
            return "oxidoreductase"

    return "non-enzyme"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch enzyme class labels from UniProt")
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--cache_dir", default=str(DATA_DIR / "cache" / "enzyme_uniprot"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load gene -> uniprot_id from merged_valid_variants
    variants_path = data_dir / "merged_valid_variants.json"
    print(f"Loading variants from {variants_path}")
    with open(variants_path) as f:
        variants = json.load(f)

    gene_to_uniprot = {}
    for v in variants:
        g = v["gene"]
        uid = v.get("uniprot_id", "")
        if g not in gene_to_uniprot and uid:
            gene_to_uniprot[g] = uid

    print(f"Unique genes: {len(gene_to_uniprot)}")

    # Also load merged_gene_list for canonical gene order
    gene_list_path = data_dir / "merged_gene_list.tsv"
    canonical_genes: list[str] = []
    with open(gene_list_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            canonical_genes.append(row["gene"])
    print(f"Canonical gene list: {len(canonical_genes)} genes")

    # Fetch UniProt entries (with per-accession disk cache)
    print("\nFetching UniProt entries...")
    results: dict[str, dict] = {}  # gene -> parsed result

    unique_accs = list(set(gene_to_uniprot.values()))
    acc_to_data = {}  # acc -> dict or None

    for i, acc in enumerate(unique_accs):
        cache_file = cache_dir / f"{acc}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                acc_to_data[acc] = json.load(f)
            continue

        if (i + 1) % 50 == 0:
            print(f"  Fetched {i+1}/{len(unique_accs)} accessions...")

        entry = fetch_uniprot_entry(acc)
        # Cache raw entry (None stored as null sentinel)
        with open(cache_file, "w") as f:
            json.dump(entry, f)
        acc_to_data[acc] = entry

        # Polite rate limiting: ~3 requests/sec
        time.sleep(0.35)

    print(f"Fetched {sum(1 for v in acc_to_data.values() if v is not None)}/{len(unique_accs)} entries")

    # Parse each gene
    rows = []
    class_counts = defaultdict(int)
    multi_class_count = 0

    for gene in canonical_genes:
        acc = gene_to_uniprot.get(gene, "")
        if not acc:
            rows.append({
                "gene": gene,
                "uniprot_id": "",
                "ec_numbers": "",
                "keyword_ids": "",
                "enzyme_4class": "non-enzyme",
                "multi_class_flag": "0",
                "uniprot_missing_flag": "1",
            })
            class_counts["non-enzyme"] += 1
            continue

        entry = acc_to_data.get(acc)
        if entry is None:
            rows.append({
                "gene": gene,
                "uniprot_id": acc,
                "ec_numbers": "",
                "keyword_ids": "",
                "enzyme_4class": "non-enzyme",
                "multi_class_flag": "0",
                "uniprot_missing_flag": "1",
            })
            class_counts["non-enzyme"] += 1
            continue

        ec_numbers, keyword_ids = parse_ec_and_keywords(entry)
        enzyme_class = assign_4class(ec_numbers, keyword_ids)

        # Multi-class flag: gene has EC numbers that map to >1 of our named classes
        mapped_classes = set()
        for ec in ec_numbers:
            c = assign_4class([ec], keyword_ids)
            if c != "non-enzyme":
                mapped_classes.add(c)
        is_multi = len(mapped_classes) > 1
        if is_multi:
            multi_class_count += 1

        rows.append({
            "gene": gene,
            "uniprot_id": acc,
            "ec_numbers": ";".join(ec_numbers),
            "keyword_ids": ";".join(keyword_ids[:10]),  # cap to avoid huge fields
            "enzyme_4class": enzyme_class,
            "multi_class_flag": "1" if is_multi else "0",
            "uniprot_missing_flag": "0",
        })
        class_counts[enzyme_class] += 1

    # Write output
    out_path = data_dir / "enzyme_labels.tsv"
    fieldnames = ["gene", "uniprot_id", "ec_numbers", "keyword_ids",
                  "enzyme_4class", "multi_class_flag", "uniprot_missing_flag"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_path}")
    print("\nClass distribution:")
    for cls in ["kinase", "protease", "oxidoreductase", "non-enzyme"]:
        n = class_counts[cls]
        print(f"  {cls:20s}: {n:4d}  ({100*n/len(rows):.1f}%)")
    print(f"  Multi-class genes (flagged): {multi_class_count}")


if __name__ == "__main__":
    main()
