"""
Fetch gene-level annotation sources.

Three pipeline steps:

  Step 1 — pfam  (run after fetch_variants --step merge)
    Fetch primary Pfam family for each gene via UniProt REST API.
    Input : data/variants.json
    Output: data/pfam_families.json

  Step 2 — uniprot  (run after fetch_variants --step merge)
    Fetch UniProt sequences for all variants in variants.json.
    Input : data/variants.json
    Output: data/cache/uniprot_sequences_extended.json

  Step 3 — enzyme  (run after step 1 or independently)
    Fetch enzyme class annotations (kinase / protease / oxidoreductase /
    non-enzyme) from UniProt for all genes in gene_list.tsv.
    Input : data/variants.json, data/gene_list.tsv
    Output: data/enzyme_labels.tsv

Usage:
    python -m esm2_mech.fetch_data.fetch_annotations --step pfam
    python -m esm2_mech.fetch_data.fetch_annotations --step uniprot
    python -m esm2_mech.fetch_data.fetch_annotations --step enzyme
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import sys
import time
import urllib.request
import urllib.parse
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Optional

from Bio import SeqIO

from esm2_mech.utils.paths import (
    DATA_DIR,
    ENZYME_CACHE_DIR,
    ENZYME_LABELS_TSV,
    GENE_LIST_TSV,
    SEQUENCES_EXTENDED_JSON,
    SEQUENCES_JSON,
    VARIANTS_JSON,
)
from esm2_mech.utils.constants import HTTP_USER_AGENT, UNIPROT_REST
from esm2_mech.utils.io import atomic_write_json as _atomic_write_json
from esm2_mech.fetch_data.uniprot_fetch import (
    TransientFetchError as _TransientFetchError,
    fetch_with_retries,
    parse_pfam_id,
)

print = functools.partial(print, flush=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MERGED_VARIANTS = VARIANTS_JSON
MERGED_GENE_LIST = GENE_LIST_TSV
PFAM_OUT = DATA_DIR / "pfam_families.json"
SEQUENCES_EXTENDED_OUT = SEQUENCES_EXTENDED_JSON
ENZYME_OUT = ENZYME_LABELS_TSV

# ===========================================================================
# Step 1 — Pfam families
# ===========================================================================
_PFAM_DELAY = 0.3
_PFAM_RETRIES = 3


def fetch_pfam_for_uniprot(uniprot_id: str) -> Optional[str]:
    """Return the first Pfam cross-reference ID for a UniProt entry, or None if genuinely absent.

    Raises _TransientFetchError on network/server failure so callers can skip caching.
    """
    body = fetch_with_retries(
        f"{UNIPROT_REST}/{uniprot_id}.json",
        headers={"User-Agent": HTTP_USER_AGENT},
        timeout=20,
        retries=_PFAM_RETRIES,
        delay=_PFAM_DELAY,
        label=uniprot_id,
    )
    if body is None:
        return None  # genuine 404 — safe to cache
    return parse_pfam_id(json.loads(body))


def main_pfam(from_scratch: bool = False) -> None:
    if not MERGED_VARIANTS.exists():
        raise FileNotFoundError(f"Required input not found: {MERGED_VARIANTS}")

    with open(MERGED_VARIANTS) as f:
        variants = json.load(f)

    unique_pairs: set[tuple[str, str]] = {
        (v["gene"], v["uniprot_id"])
        for v in variants
        if v.get("gene") and v.get("uniprot_id")
    }
    print(f"Unique (gene, uniprot_id) pairs: {len(unique_pairs)}")

    existing: dict[str, Optional[str]] = {}
    if not from_scratch and PFAM_OUT.exists():
        with open(PFAM_OUT) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing entries from {PFAM_OUT.name}")

    to_fetch = sorted((gene, uid) for gene, uid in unique_pairs if gene not in existing)
    print(f"Already have: {len(existing)}, need to fetch: {len(to_fetch)}")

    results: dict[str, Optional[str]] = dict(existing)

    for i, (gene, uniprot_id) in enumerate(to_fetch):
        try:
            results[gene] = fetch_pfam_for_uniprot(uniprot_id)
        except _TransientFetchError:
            pass  # not written to results — will be retried next run
        time.sleep(_PFAM_DELAY)
        if (i + 1) % 100 == 0 or (i + 1) == len(to_fetch):
            print(f"  {i+1}/{len(to_fetch)} fetched")
            _atomic_write_json(PFAM_OUT, results)

    if not to_fetch:
        print("Nothing to fetch.")
        _atomic_write_json(PFAM_OUT, results)

    n_annotated = sum(1 for v in results.values() if v is not None)
    print(f"\nPfam annotations: {n_annotated}/{len(results)} genes assigned")
    print(f"Wrote: {PFAM_OUT}")


# ===========================================================================
# Step 2 — UniProt sequences
# ===========================================================================
_UNIPROT_BATCH_SIZE = 100
_UNIPROT_RETRIES = 3
_UNIPROT_USER_AGENT = "dami-mechanism-prediction/0.1 (academic research)"


def _fetch_uniprot_batch(accs: list[str]) -> str:
    query = " OR ".join(f"accession:{a}" for a in accs)
    params = {"query": query, "format": "fasta", "size": len(accs)}
    url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url, headers={"User-Agent": _UNIPROT_USER_AGENT, "Accept": "text/plain"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode()


def parse_fasta(text: str) -> dict[str, str]:
    out = {}
    for record in SeqIO.parse(StringIO(text), "fasta"):
        parts = record.id.split("|")
        acc = parts[1].split("-")[0] if len(parts) > 1 else parts[0].split("-")[0]
        out[acc] = str(record.seq)
    return out


def select_extended(
    base: dict[str, str],
    extended_existing: dict[str, str],
    new_results: dict[str, str],
    needed: set[str],
) -> dict[str, str]:
    merged = {**extended_existing, **new_results}
    return {k: v for k, v in merged.items() if k in needed and k not in base}


def main_uniprot(from_scratch: bool = False) -> None:
    if not MERGED_VARIANTS.exists():
        raise FileNotFoundError(f"Required input not found: {MERGED_VARIANTS}")

    with open(MERGED_VARIANTS) as f:
        variants = json.load(f)

    all_uniprots = {v["uniprot_id"] for v in variants if v.get("uniprot_id")}
    print(f"Unique UniProt IDs in variants.json: {len(all_uniprots)}")

    base: dict[str, str] = {}
    extended_existing: dict[str, str] = {}

    if not from_scratch:
        if SEQUENCES_JSON.exists():
            with open(SEQUENCES_JSON) as f:
                base = json.load(f)
            print(f"Loaded {len(base)} sequences from sequences.json")
        if SEQUENCES_EXTENDED_OUT.exists():
            with open(SEQUENCES_EXTENDED_OUT) as f:
                extended_existing = json.load(f)
            print(
                f"Loaded {len(extended_existing)} sequences from uniprot_sequences_extended.json"
            )

    new_results: dict[str, str] = {}
    todo = sorted(all_uniprots - set(base) - set(extended_existing))
    print(f"Already have: {len(all_uniprots) - len(todo)}, need to fetch: {len(todo)}")

    SEQUENCES_EXTENDED_OUT.parent.mkdir(parents=True, exist_ok=True)

    if not todo:
        print("Nothing to fetch.")
        _atomic_write_json(
            SEQUENCES_EXTENDED_OUT,
            select_extended(base, extended_existing, new_results, all_uniprots),
        )
        print(f"Wrote: {SEQUENCES_EXTENDED_OUT}")
        return

    n_batches = (len(todo) + _UNIPROT_BATCH_SIZE - 1) // _UNIPROT_BATCH_SIZE
    t0 = time.time()

    for bi in range(n_batches):
        batch = todo[bi * _UNIPROT_BATCH_SIZE : (bi + 1) * _UNIPROT_BATCH_SIZE]
        for attempt in range(_UNIPROT_RETRIES):
            try:
                text = _fetch_uniprot_batch(batch)
                new_results.update(parse_fasta(text))
                break
            except Exception as e:
                print(f"  batch {bi+1}/{n_batches} attempt {attempt+1} error: {e}")
                time.sleep(2 * (attempt + 1))
        else:
            print(f"  GAVE UP on batch {bi+1}/{n_batches}")

        elapsed = time.time() - t0
        print(
            f"  batch {bi+1}/{n_batches}: {len(new_results)} fetched ({elapsed:.0f}s)"
        )
        _atomic_write_json(
            SEQUENCES_EXTENDED_OUT,
            select_extended(base, extended_existing, new_results, all_uniprots),
        )
        time.sleep(0.5)

    _atomic_write_json(
        SEQUENCES_EXTENDED_OUT,
        select_extended(base, extended_existing, new_results, all_uniprots),
    )

    have_all = set(base) | set(extended_existing) | set(new_results)
    found = len(all_uniprots & have_all)
    missing = sorted(all_uniprots - have_all)
    print(f"\nFetched {found}/{len(all_uniprots)} sequences")
    if missing:
        print(f"Missing ({len(missing)}): {missing[:20]}")
    print(f"Wrote: {SEQUENCES_EXTENDED_OUT}")


# ===========================================================================
# Step 3 — Enzyme labels
# ===========================================================================
_ENZYME_RETRIES = 3
_ENZYME_BACKOFF = 2.0
ENZYME_4CLASS_PRIORITY = ["kinase", "protease", "oxidoreductase"]


def _fetch_uniprot_entry(acc: str) -> Optional[dict]:
    """Return the UniProt JSON entry, or None on a genuine 404.

    Raises _TransientFetchError on network/server failure so callers can skip caching.
    """
    body = fetch_with_retries(
        f"{UNIPROT_REST}/{acc}?format=json",
        headers={
            "User-Agent": "dami-mechanism-prediction/0.1 (academic research)",
            "Accept": "application/json",
        },
        timeout=30,
        retries=_ENZYME_RETRIES,
        delay=_ENZYME_BACKOFF,
        label=acc,
    )
    return None if body is None else json.loads(body)


def parse_ec_and_keywords(entry: dict) -> tuple:
    """Extract EC numbers and keyword IDs from a UniProt JSON entry."""
    ec_numbers = []
    keyword_ids = []

    try:
        prot = entry.get("proteinDescription", {})
        name_blocks = [prot.get("recommendedName") or {}] + (
            prot.get("alternativeNames") or []
        )
        for name_block in name_blocks:
            for ec_obj in name_block.get("ecNumbers", []):
                val = ec_obj.get("value", "")
                if val:
                    ec_numbers.append(val)
    except Exception:
        pass

    try:
        for comment in entry.get("comments", []):
            if comment.get("commentType") == "CATALYTIC ACTIVITY":
                reaction = comment.get("reaction", {})
                ec_val = reaction.get("ecNumber")
                if isinstance(ec_val, str) and ec_val and ec_val not in ec_numbers:
                    ec_numbers.append(ec_val)
                elif isinstance(ec_val, list):
                    for ec_obj in ec_val:
                        val = (
                            ec_obj
                            if isinstance(ec_obj, str)
                            else ec_obj.get("value", "")
                        )
                        if val and val not in ec_numbers:
                            ec_numbers.append(val)
    except Exception:
        pass

    try:
        for kw in entry.get("keywords", []):
            kid = kw.get("id", "")
            if kid:
                keyword_ids.append(kid)
    except Exception:
        pass

    return ec_numbers, keyword_ids


def assign_4class(ec_numbers: list[str], keyword_ids: list[str]) -> str | None:
    """Assign one of: kinase, protease, oxidoreductase, non-enzyme.

    Priority order (most specific first):
      1. kinase        — KW-0418 OR EC 2.7.x.x
      2. protease      — EC 3.4.x.x
      3. oxidoreductase— EC 1.x.x.x
      4. non-enzyme    — no EC annotation

    EC-annotated enzymes outside the three selected subclasses return None and
    are excluded from the simplified four-class experiment.
    """
    if "KW-0418" in keyword_ids:
        return "kinase"
    for ec in ec_numbers:
        if ec.startswith("2.7."):
            return "kinase"
    for ec in ec_numbers:
        if ec.startswith("3.4."):
            return "protease"
    for ec in ec_numbers:
        parts = ec.split(".")
        if len(parts) >= 2 and parts[0] == "1" and parts[1]:
            return "oxidoreductase"
    if ec_numbers:
        return None
    return "non-enzyme"


def main_enzyme() -> None:
    missing = [p for p in [MERGED_VARIANTS, MERGED_GENE_LIST] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required input(s) not found:\n" + "\n".join(f"  {p}" for p in missing)
        )

    ENZYME_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with open(MERGED_VARIANTS) as f:
        variants = json.load(f)

    gene_to_uniprot: dict[str, str] = {}
    for v in variants:
        g = v["gene"]
        uid = v.get("uniprot_id", "")
        if g not in gene_to_uniprot and uid:
            gene_to_uniprot[g] = uid

    print(f"Unique genes: {len(gene_to_uniprot)}")

    canonical_genes: list[str] = []
    with open(MERGED_GENE_LIST) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            canonical_genes.append(row["gene"])
    print(f"Canonical gene list: {len(canonical_genes)} genes")

    print("\nFetching UniProt entries...")
    unique_accs = list(set(gene_to_uniprot.values()))
    acc_to_data: dict[str, Optional[dict]] = {}
    fetched = 0
    loaded_from_cache = 0
    transient_failures: list[str] = []

    for i, acc in enumerate(unique_accs):
        cache_file = ENZYME_CACHE_DIR / f"{acc}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    acc_to_data[acc] = json.load(f)
                loaded_from_cache += 1
                if (i + 1) % 50 == 0:
                    print(f"  Loaded {i+1}/{len(unique_accs)} accessions (cached)...")
                continue
            except Exception as e:
                print(f"  WARNING: corrupt cache for {acc}: {e} — re-fetching")
                cache_file.unlink(missing_ok=True)

        if (i + 1) % 50 == 0:
            print(f"  Fetched {i+1}/{len(unique_accs)} accessions...")

        try:
            entry = _fetch_uniprot_entry(acc)
        except _TransientFetchError:
            transient_failures.append(acc)
            time.sleep(0.35)
            continue
        _atomic_write_json(cache_file, entry)
        acc_to_data[acc] = entry
        fetched += 1
        time.sleep(0.35)

    if transient_failures:
        examples = sorted(transient_failures)[:5]
        raise RuntimeError(
            f"UniProt annotation fetch had {len(transient_failures)} transient "
            f"failure(s); {ENZYME_OUT} was not written. Re-run to retry. "
            f"Examples: {examples}"
        )

    n_found = sum(1 for v in acc_to_data.values() if v is not None)
    print(
        f"Done: {n_found}/{len(unique_accs)} entries found "
        f"({fetched} fetched from network, {loaded_from_cache} from cache)"
    )

    rows = []
    class_counts: dict[str, int] = defaultdict(int)
    multi_class_count = 0

    for gene in canonical_genes:
        acc = gene_to_uniprot.get(gene)
        if not acc:
            rows.append(
                {
                    "gene": gene,
                    "uniprot_id": "",
                    "ec_numbers": "",
                    "keyword_ids": "",
                    "enzyme_4class": "",
                    "multi_class_flag": "",
                    "uniprot_missing_flag": "1",
                    "enzyme_4class_excluded_flag": "0",
                }
            )
            class_counts["missing"] += 1
            continue

        entry = acc_to_data.get(acc)
        if entry is None:
            rows.append(
                {
                    "gene": gene,
                    "uniprot_id": acc,
                    "ec_numbers": "",
                    "keyword_ids": "",
                    "enzyme_4class": "",
                    "multi_class_flag": "",
                    "uniprot_missing_flag": "1",
                    "enzyme_4class_excluded_flag": "0",
                }
            )
            class_counts["missing"] += 1
            continue

        ec_numbers, keyword_ids = parse_ec_and_keywords(entry)
        enzyme_class = assign_4class(ec_numbers, keyword_ids)

        # Check each class independently using the same evidence rules as assign_4class,
        # but without priority resolution. A gene with kinase + protease evidence is
        # flagged multi even though enzyme_4class resolves to "kinase".
        def _ec_class(ec: str) -> str | None:
            parts = ec.split(".")
            if len(parts) >= 2 and parts[1]:
                if parts[0] == "2" and ec.startswith("2.7."):
                    return "kinase"
                if parts[0] == "3" and ec.startswith("3.4."):
                    return "protease"
                if parts[0] == "1":
                    return "oxidoreductase"
            return None

        mapped_classes = set()
        if "KW-0418" in keyword_ids:
            mapped_classes.add("kinase")
        for ec in ec_numbers:
            c = _ec_class(ec)
            if c is not None:
                mapped_classes.add(c)
        is_multi = len(mapped_classes) > 1
        if is_multi:
            multi_class_count += 1

        rows.append(
            {
                "gene": gene,
                "uniprot_id": acc,
                "ec_numbers": ";".join(ec_numbers),
                "keyword_ids": ";".join(keyword_ids),
                "enzyme_4class": "" if enzyme_class is None else enzyme_class,
                "multi_class_flag": "1" if is_multi else "0",
                "uniprot_missing_flag": "0",
                "enzyme_4class_excluded_flag": "1" if enzyme_class is None else "0",
            }
        )
        if enzyme_class is None:
            class_counts["excluded_other_enzyme"] += 1
        else:
            class_counts[enzyme_class] += 1

    fieldnames = [
        "gene",
        "uniprot_id",
        "ec_numbers",
        "keyword_ids",
        "enzyme_4class",
        "multi_class_flag",
        "uniprot_missing_flag",
        "enzyme_4class_excluded_flag",
    ]
    with open(ENZYME_OUT, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            none_fields = [key for key, value in row.items() if value is None]
            if none_fields:
                raise ValueError(f"enzyme label row for {row['gene']} has None fields: {none_fields}")
            writer.writerow(row)

    print(f"\nWrote {len(rows)} rows to {ENZYME_OUT}")
    print("\nClass distribution:")
    for cls in [
        "kinase",
        "protease",
        "oxidoreductase",
        "non-enzyme",
        "excluded_other_enzyme",
        "missing",
    ]:
        n = class_counts[cls]
        print(f"  {cls:20s}: {n:4d}  ({100*n/len(rows):.1f}%)")
    print(f"  Multi-class genes (flagged): {multi_class_count}")


# ===========================================================================
# Main dispatch
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch gene-level annotation sources. See module docstring for step order."
    )
    parser.add_argument(
        "--step",
        choices=["pfam", "uniprot", "enzyme"],
        required=True,
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="(pfam / uniprot) Ignore existing cache and re-fetch.",
    )
    args = parser.parse_args()

    if args.step == "pfam":
        main_pfam(from_scratch=args.from_scratch)
    elif args.step == "uniprot":
        main_uniprot(from_scratch=args.from_scratch)
    elif args.step == "enzyme":
        main_enzyme()


if __name__ == "__main__":
    main()
