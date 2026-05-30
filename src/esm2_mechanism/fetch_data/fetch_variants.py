"""
Fetch and merge variant datasets.

Three pipeline steps:

  Step 1 — gerasimavicius
    Parse Gerasimavicius et al. variant table from the local Excel file.
    Input : data/downloads/DiseaseMech_Stability_VEPS.xlsx
    Output: data/gerasimavicius_variants.json

  Step 2 — clinvar
    Fetch ClinVar pathogenic/likely-pathogenic missense variants for all genes
    in gene_list.tsv.  Resume-safe; rate-limited to ≤3 NCBI req/s.
    Input : data/gene_list.tsv
    Output: data/clinvar_variants.tsv
    Cache : data/cache/clinvar/<gene>.json, data/cache/uniprot/<gene>.json

  Step 3 — merge
    Merge Gerasimavicius + G2P/ClinVar into a single variant dataset.
    Priority: Gerasimavicius for genes present in both (has FoldX ddG).
    Input : data/gerasimavicius_variants.json, data/gene_list.tsv,
            data/clinvar_variants.tsv
    Output: data/variants.json

Usage:
    python -m esm2_mechanism.fetch_data.fetch_variants --step gerasimavicius
    python -m esm2_mechanism.fetch_data.fetch_variants --step clinvar
    python -m esm2_mechanism.fetch_data.fetch_variants --step merge [--pathogenic_only]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import functools
from collections import Counter
from pathlib import Path
from typing import Optional

import openpyxl
import requests

from esm2_mechanism.utils_paths import DATA_DIR, GENE_LIST_TSV, VARIANTS_JSON

print = functools.partial(print, flush=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
XLSX_PATH = DATA_DIR / "downloads" / "DiseaseMech_Stability_VEPS.xlsx"
GERAS_OUT = DATA_DIR / "gerasimavicius_variants.json"
CLINVAR_OUT = DATA_DIR / "clinvar_variants.tsv"
MERGED_OUT = VARIANTS_JSON
CACHE_DIR = DATA_DIR / "cache"
UNIPROT_CACHE = CACHE_DIR / "uniprot"
CLINVAR_CACHE = CACHE_DIR / "clinvar"

# ===========================================================================
# Step 1 — Gerasimavicius variants
# ===========================================================================
_VARIANT_PAT = re.compile(r"^([A-Z])(\d+)([A-Z])$")

_GERAS_MECH_MAP = {
    "GOF": "GOF",
    "DN": "DN",
    "HI": "HI",
    "AR": "AR",
    "AR, HET": "AR",
    "AR, HOM": "AR",
}


def parse_gerasimavicius_variants(xlsx_path: Path) -> list[dict]:
    print(f"Loading {xlsx_path.name}...")
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    try:
        ws = wb["ClinVar_gene_level"]
        rows = list(ws.iter_rows(values_only=True))
    finally:
        wb.close()

    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    col = {h: i for i, h in enumerate(header)}
    print(f"  Sheet columns: {header[:10]}")

    required = {"Gene", "Uniprot_id", "Uniprot_variant", "Disease_mechanism", "Class"}
    missing_cols = required - col.keys()
    if missing_cols:
        raise ValueError(
            f"Missing expected columns in ClinVar_gene_level sheet: {missing_cols}"
        )

    variants = []
    skipped = 0
    for row in rows[1:]:
        try:
            row_class = str(row[col["Class"]] or "").strip().upper()
            if "CLINVAR" not in row_class:
                continue

            gene = row[col["Gene"]]
            uniprot = row[col["Uniprot_id"]]
            variant_str = row[col["Uniprot_variant"]]
            mech_raw = row[col["Disease_mechanism"]]
            foldx_raw = (
                row[col["raw_FoldX_Monomer"]] if "raw_FoldX_Monomer" in col else None
            )

            if not all([gene, uniprot, variant_str, mech_raw]):
                skipped += 1
                continue

            mech = _GERAS_MECH_MAP.get(str(mech_raw).strip().upper())
            if mech is None:
                skipped += 1
                continue

            m = _VARIANT_PAT.match(str(variant_str).strip())
            if not m:
                skipped += 1
                continue
            aa_wt, aa_pos_str, aa_mut = m.groups()

            foldx_ddg = None
            if foldx_raw is not None:
                try:
                    foldx_ddg = float(foldx_raw)
                except (ValueError, TypeError):
                    pass

            variants.append(
                {
                    "gene": str(gene).upper(),
                    "uniprot_id": str(uniprot).strip(),
                    "aa_pos": int(aa_pos_str),
                    "aa_wt": aa_wt.upper(),
                    "aa_mut": aa_mut.upper(),
                    "mechanism": mech,
                    "foldx_ddg": foldx_ddg,
                    "clinvar_id": "",
                }
            )
        except Exception as exc:
            print(f"WARNING: skipping row due to error: {exc}")
            skipped += 1

    print(f"  Parsed {len(variants)} variants ({skipped} rows skipped)")
    mechs = Counter(v["mechanism"] for v in variants)
    n_genes = len(set(v["gene"] for v in variants))
    print(f"  Genes: {n_genes} | Mechanism counts: {dict(mechs)}")
    return variants


def main_gerasimavicius() -> None:
    if not XLSX_PATH.exists():
        raise FileNotFoundError(f"Required input not found: {XLSX_PATH}")
    variants = parse_gerasimavicius_variants(XLSX_PATH)
    if not variants:
        raise ValueError("No variants parsed — check the Excel file and sheet name")
    with open(GERAS_OUT, "w") as f:
        json.dump(variants, f)
    print(f"Written to {GERAS_OUT}")


# ===========================================================================
# Step 2 — ClinVar variants
# ===========================================================================
_NCBI_MIN_INTERVAL = 1.0 / 3.0
_last_ncbi_call: float = 0.0

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "fetch_clinvar/1.0 (dami.gupta@gmail.com)"})

NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"

_CLINSIG_KEEP = {"pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"}

_PROT_CHANGE_RE = re.compile(
    r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})" r"|p\.([A-Z])(\d+)([A-Z])"
)

_THREE_TO_ONE = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
    "Sec": "U",
    "Pyl": "O",
    "Asx": "B",
    "Glx": "Z",
    "Xaa": "X",
    "Ter": "*",
}

_seq_cache: dict[str, str] = {}


def _ncbi_wait() -> None:
    global _last_ncbi_call
    elapsed = time.monotonic() - _last_ncbi_call
    gap = _NCBI_MIN_INTERVAL - elapsed
    if gap > 0:
        time.sleep(gap)
    _last_ncbi_call = time.monotonic()


def _get_json(
    url: str, params: dict, ncbi: bool = False, retries: int = 4
) -> Optional[dict]:
    skip_rate_limit = False
    for attempt in range(retries):
        if ncbi and not skip_rate_limit:
            _ncbi_wait()
        skip_rate_limit = False
        try:
            r = _SESSION.get(url, params=params, timeout=30)
            if r.status_code in (429, 503):
                wait = 10 * (attempt + 1)
                print(f"WARNING: HTTP {r.status_code} – sleeping {wait}s")
                time.sleep(wait)
                skip_rate_limit = True
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            print(f"WARNING: Request error (attempt {attempt + 1}): {exc}")
            time.sleep(5 * (attempt + 1))
    return None


def _get_text(url: str, params: dict, retries: int = 4) -> Optional[str]:
    for attempt in range(retries):
        try:
            r = _SESSION.get(url, params=params, timeout=30)
            if r.status_code in (429, 503):
                wait = 10 * (attempt + 1)
                print(f"WARNING: HTTP {r.status_code} – sleeping {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.text
        except requests.RequestException as exc:
            print(f"WARNING: Request error (attempt {attempt + 1}): {exc}")
            time.sleep(5 * (attempt + 1))
    return None


def _parse_hgvsp(hgvs_p: str) -> Optional[tuple]:
    m = _PROT_CHANGE_RE.search(hgvs_p)
    if not m:
        return None
    if m.group(1):
        wt = _THREE_TO_ONE.get(m.group(1))
        pos = int(m.group(2))
        mut = _THREE_TO_ONE.get(m.group(3))
    else:
        wt = m.group(4)
        pos = int(m.group(5))
        mut = m.group(6)
    if wt is None or mut is None or wt == mut or mut == "*" or wt == "*":
        return None
    return (wt, pos, mut)


def fetch_uniprot_id(gene: str, prefilled: Optional[str]) -> Optional[str]:
    cache_file = UNIPROT_CACHE / f"{gene}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text()).get("uniprot_id")

    if prefilled:
        cache_file.write_text(json.dumps({"uniprot_id": prefilled}))
        return prefilled

    params = {
        "query": f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true",
        "fields": "accession,gene_names",
        "format": "json",
        "size": 5,
    }
    data = _get_json(UNIPROT_SEARCH, params)
    if data is None:
        print(
            f"  WARNING: {gene}: UniProt request failed — not caching, will retry next run"
        )
        return None

    results = data.get("results", [])
    if not results:
        cache_file.write_text(json.dumps({"uniprot_id": None}))
        return None

    acc = None
    for entry in results:
        primary_names = [
            g["geneName"]["value"] for g in entry.get("genes", []) if "geneName" in g
        ]
        if any(n.upper() == gene.upper() for n in primary_names):
            acc = entry["primaryAccession"]
            break

    if acc is None:
        print(
            f"  WARNING: {gene}: no exact gene name match in UniProt results — skipping"
        )
        cache_file.write_text(json.dumps({"uniprot_id": None}))
        return None

    cache_file.write_text(json.dumps({"uniprot_id": acc}))
    return acc


def fetch_protein_sequence(uniprot_id: str) -> Optional[str]:
    if uniprot_id in _seq_cache:
        return _seq_cache[uniprot_id]
    seq_file = UNIPROT_CACHE / f"{uniprot_id}_seq.txt"
    if seq_file.exists():
        seq = seq_file.read_text().strip()
        if seq:
            _seq_cache[uniprot_id] = seq
            return seq
        print(f"  WARNING: empty sequence cache for {uniprot_id} — re-fetching")
        seq_file.unlink()
    time.sleep(0.1)
    text = _get_text(UNIPROT_FASTA.format(acc=uniprot_id), {})
    if not text:
        return None
    lines = text.strip().splitlines()
    seq = "".join(line for line in lines if not line.startswith(">"))
    tmp = seq_file.with_suffix(".tmp")
    tmp.write_text(seq)
    os.replace(tmp, seq_file)
    _seq_cache[uniprot_id] = seq
    return seq


def fetch_clinvar_variants(gene: str) -> list:
    cache_file = CLINVAR_CACHE / f"{gene}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    variants: list = []
    search_params = {
        "db": "clinvar",
        "term": (
            f"{gene}[Gene Name] AND "
            '("pathogenic"[ClinSig] OR "likely pathogenic"[ClinSig]) AND '
            '"single nucleotide variant"[Type of variation]'
        ),
        "retmax": 10000,
        "retmode": "json",
    }
    if NCBI_API_KEY:
        search_params["api_key"] = NCBI_API_KEY

    result = _get_json(ESEARCH_URL, search_params, ncbi=True)
    if result is None:
        print(
            f"  WARNING: {gene}: NCBI esearch failed — not caching, will retry next run"
        )
        return []

    ids = result.get("esearchresult", {}).get("idlist", [])
    if not ids:
        cache_file.write_text(json.dumps([]))
        return []

    print(f"  {gene}: {len(ids)} ClinVar IDs found")

    BATCH = 200
    any_batch_failed = False
    for batch_start in range(0, len(ids), BATCH):
        batch_ids = ids[batch_start : batch_start + BATCH]
        summary_params = {"db": "clinvar", "id": ",".join(batch_ids), "retmode": "json"}
        if NCBI_API_KEY:
            summary_params["api_key"] = NCBI_API_KEY

        summ = _get_json(ESUMMARY_URL, summary_params, ncbi=True)
        if summ is None:
            print(
                f"  WARNING: {gene}: esummary batch at offset {batch_start} failed — will not cache result"
            )
            any_batch_failed = True
            continue

        result_obj = summ.get("result", {})
        for uid in result_obj.get("uids", []):
            rec = result_obj.get(uid, {})

            clinsig_raw = (
                rec.get("germline_classification", {}).get("description", "")
                or rec.get("clinical_significance", {}).get("description", "")
            ).lower()
            if clinsig_raw not in _CLINSIG_KEEP:
                continue

            mol_cons = [
                (mc if isinstance(mc, str) else mc.get("type", "")).lower()
                for mc in rec.get("molecular_consequence_list", [])
            ]
            if not any("missense" in mc for mc in mol_cons):
                continue

            parsed = _parse_hgvsp(rec.get("title", ""))
            if parsed is None:
                continue
            wt_aa, pos, mut_aa = parsed
            variants.append(
                {
                    "hgvs_p": rec.get("title", ""),
                    "wt_aa": wt_aa,
                    "pos": pos,
                    "mut_aa": mut_aa,
                    "clinsig": clinsig_raw,
                }
            )

    seen: set = set()
    deduped: list = []
    for v in variants:
        key = (v["wt_aa"], v["pos"], v["mut_aa"])
        if key not in seen:
            seen.add(key)
            deduped.append(v)

    if any_batch_failed:
        print(
            f"  WARNING: {gene}: result is incomplete due to batch failure — not caching, will retry next run"
        )
        return deduped
    cache_file.write_text(json.dumps(deduped))
    return deduped


def validate_wt(variant: dict, sequence: str) -> bool:
    pos = variant["pos"]
    if pos < 1 or pos > len(sequence):
        return False
    return sequence[pos - 1] == variant["wt_aa"]


def _load_gene_list(path: Path) -> list:
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            gene = row.get("gene", "").strip()
            if not gene:
                continue
            rows.append(
                {
                    "gene": gene,
                    "mechanism": row.get("mechanism", "").strip(),
                    "uniprot_id": row.get("uniprot_id", "").strip() or None,
                }
            )
    return rows


def main_clinvar() -> None:
    if not GENE_LIST_TSV.exists():
        raise FileNotFoundError(f"Required input not found: {GENE_LIST_TSV}")

    for d in (UNIPROT_CACHE, CLINVAR_CACHE):
        d.mkdir(parents=True, exist_ok=True)

    genes = _load_gene_list(GENE_LIST_TSV)
    print(f"Loaded {len(genes)} genes from {GENE_LIST_TSV}")

    done_genes: set = set()
    if CLINVAR_OUT.exists() and CLINVAR_OUT.stat().st_size > 0:
        raw = CLINVAR_OUT.read_bytes()
        if not raw.endswith(b"\n"):
            truncate_at = raw.rfind(b"\n") + 1
            if truncate_at > 0:
                print(
                    "WARNING: Output file has partial last line; truncating to last complete row"
                )
                with open(CLINVAR_OUT, "r+b") as fh:
                    fh.truncate(truncate_at)
        with open(CLINVAR_OUT, newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                done_genes.add(row.get("gene", ""))

    write_header = not CLINVAR_OUT.exists() or CLINVAR_OUT.stat().st_size == 0
    total_variants = 0

    with open(CLINVAR_OUT, "a", newline="") as out_fh:
        writer = csv.writer(out_fh, delimiter="\t")
        if write_header:
            writer.writerow(
                ["gene", "uniprot_id", "aa_pos", "aa_wt", "aa_mut", "clinsig"]
            )

        for idx, gdata in enumerate(genes, 1):
            gene = gdata["gene"]
            if gene in done_genes:
                print(f"[{idx}/{len(genes)}] {gene} – already in output, skipping")
                continue

            print(f"[{idx}/{len(genes)}] {gene}")
            uniprot_id = fetch_uniprot_id(gene, gdata["uniprot_id"])
            if uniprot_id is None:
                print(f"WARNING: {gene}: no UniProt ID found, skipping")
                continue

            sequence = fetch_protein_sequence(uniprot_id)
            if sequence is None:
                print(
                    f"WARNING: {gene}: sequence unavailable, writing variants without WT validation"
                )

            variants = fetch_clinvar_variants(gene)
            print(f"  {gene}: {len(variants)} missense P/LP variants")

            gene_written = 0
            for v in variants:
                if sequence and not validate_wt(v, sequence):
                    print(f"  WT mismatch {gene} pos {v['pos']}: expected {v['wt_aa']}")
                    continue
                writer.writerow(
                    [gene, uniprot_id, v["pos"], v["wt_aa"], v["mut_aa"], v["clinsig"]]
                )
                gene_written += 1

            total_variants += gene_written
            out_fh.flush()

    print(f"Done. {total_variants} total variants written to {CLINVAR_OUT}")


# ===========================================================================
# Step 3 — Merge datasets
# ===========================================================================
def main_merge(pathogenic_only: bool = False) -> None:
    missing = [p for p in [GERAS_OUT, GENE_LIST_TSV, CLINVAR_OUT] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required input(s) not found:\n" + "\n".join(f"  {p}" for p in missing)
        )

    with open(GERAS_OUT) as f:
        geras = json.load(f)
    geras_genes = set(v["gene"].upper() for v in geras)
    print(f"Gerasimavicius: {len(geras)} variants, {len(geras_genes)} genes")

    with open(GENE_LIST_TSV, newline="") as f:
        gene_mech_map = {
            r["gene"].upper(): r["mechanism"] for r in csv.DictReader(f, delimiter="\t")
        }

    with open(CLINVAR_OUT, newline="") as f:
        clinvar_rows = list(csv.DictReader(f, delimiter="\t"))

    if pathogenic_only:
        clinvar_rows = [
            r for r in clinvar_rows if r.get("clinsig", "").lower() == "pathogenic"
        ]
        print(f"Filtered to pathogenic only: {len(clinvar_rows)} variants")

    new_variants = []
    skipped_no_mech = 0
    skipped_no_uniprot = 0
    for r in clinvar_rows:
        gene = r["gene"].upper()
        if gene in geras_genes:
            continue
        mech = gene_mech_map.get(gene)
        if not mech:
            skipped_no_mech += 1
            continue
        # G2P uses "LOF" where Gerasimavicius distinguishes "HI" and "AR".
        # Map to "HI" so the 3-class collapse (HI+AR→LOF) works correctly.
        # WARNING: all G2P LOF genes become mechanism="HI" in the merged dataset.
        # Do NOT run the 4-class HI/AR secondary probe on merged data.
        if mech == "LOF":
            mech = "HI"
        if mech not in ("GOF", "DN", "HI", "AR"):
            continue
        try:
            aa_pos = int(r["aa_pos"])
        except (ValueError, TypeError):
            continue
        uniprot_id = r.get("uniprot_id", "").strip()
        if not uniprot_id:
            skipped_no_uniprot += 1
            continue
        new_variants.append(
            {
                "gene": gene,
                "uniprot_id": uniprot_id,
                "aa_pos": aa_pos,
                "aa_wt": r["aa_wt"].upper(),
                "aa_mut": r["aa_mut"].upper(),
                "mechanism": mech,
                "foldx_ddg": None,
                "clinvar_id": "",
                "source": "clinvar_g2p",
            }
        )

    print(
        f"ClinVar new-gene variants: {len(new_variants)} "
        f"(skipped {skipped_no_mech} with no mechanism label, "
        f"{skipped_no_uniprot} with no UniProt ID)"
    )

    for v in geras:
        v["source"] = "gerasimavicius"

    merged = geras + new_variants
    for v in merged:
        v["label_3class"] = "LOF" if v["mechanism"] in ("HI", "AR") else v["mechanism"]

    mechs = Counter(v["mechanism"] for v in merged)
    mechs3 = Counter(v["label_3class"] for v in merged)
    n_genes = len(set(v["gene"] for v in merged))
    sources = Counter(v["source"] for v in merged)

    print(f"\nMerged dataset: {len(merged)} variants, {n_genes} genes")
    print(f"Mechanism (4-class): {dict(mechs)}")
    print(f"Mechanism (3-class): {dict(mechs3)}")
    print(f"Sources: {dict(sources)}")

    with open(MERGED_OUT, "w") as f:
        json.dump(merged, f)
    print(f"Written to {MERGED_OUT}")


# ===========================================================================
# Main dispatch
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and merge variant datasets. See module docstring for step order."
    )
    parser.add_argument(
        "--step",
        choices=["gerasimavicius", "clinvar", "merge"],
        required=True,
    )
    parser.add_argument(
        "--pathogenic_only",
        action="store_true",
        help="(merge step only) Restrict ClinVar variants to 'pathogenic' only.",
    )
    args = parser.parse_args()

    if args.step == "gerasimavicius":
        main_gerasimavicius()
    elif args.step == "clinvar":
        main_clinvar()
    else:
        main_merge(pathogenic_only=args.pathogenic_only)


if __name__ == "__main__":
    main()
