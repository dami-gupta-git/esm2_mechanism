"""
Fetch ClinVar pathogenic/likely-pathogenic missense variants for a list of genes.

Input : merged_gene_list.tsv  (columns: gene, mechanism, uniprot_id, source, g2p_disagrees)
Output: clinvar_variants.tsv  (columns: gene, uniprot_id, aa_pos, aa_wt, aa_mut)
Cache : cache/clinvar/<gene>.json  and  cache/uniprot/<gene>.json

Resume-safe: genes whose cache files already exist are skipped.
Rate limit: ≤3 NCBI requests / second (enforced via token bucket).
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
import logging
import requests
from pathlib import Path
from typing import Optional
import functools
print = functools.partial(print, flush=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR.parents[1] / "data"
INPUT_TSV   = DATA_DIR / "merged_gene_list.tsv"
OUTPUT_TSV  = DATA_DIR / "clinvar_variants.tsv"
CACHE_DIR   = DATA_DIR / "cache"
UNIPROT_CACHE = CACHE_DIR / "uniprot"
CLINVAR_CACHE = CACHE_DIR / "clinvar"

for d in (UNIPROT_CACHE, CLINVAR_CACHE):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Rate limiter  (≤3 NCBI requests / second)
# ---------------------------------------------------------------------------
_NCBI_MIN_INTERVAL = 1.0 / 3.0
_last_ncbi_call: float = 0.0

def _ncbi_wait() -> None:
    global _last_ncbi_call
    elapsed = time.monotonic() - _last_ncbi_call
    gap = _NCBI_MIN_INTERVAL - elapsed
    if gap > 0:
        time.sleep(gap)
    _last_ncbi_call = time.monotonic()

# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "fetch_clinvar/1.0 (dami.gupta@gmail.com)"})

def get_json(url: str, params: dict, ncbi: bool = False, retries: int = 4) -> Optional[dict]:
    for attempt in range(retries):
        if ncbi:
            _ncbi_wait()
        try:
            r = SESSION.get(url, params=params, timeout=30)
            if r.status_code == 429 or r.status_code == 503:
                wait = 10 * (attempt + 1)
                log.warning("HTTP %s – sleeping %ds", r.status_code, wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            log.warning("Request error (attempt %d): %s", attempt + 1, exc)
            time.sleep(5 * (attempt + 1))
    return None

def get_text(url: str, params: dict, ncbi: bool = False, retries: int = 4) -> Optional[str]:
    for attempt in range(retries):
        if ncbi:
            _ncbi_wait()
        try:
            r = SESSION.get(url, params=params, timeout=30)
            if r.status_code == 429 or r.status_code == 503:
                wait = 10 * (attempt + 1)
                log.warning("HTTP %s – sleeping %ds", r.status_code, wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.text
        except requests.RequestException as exc:
            log.warning("Request error (attempt %d): %s", attempt + 1, exc)
            time.sleep(5 * (attempt + 1))
    return None

# ---------------------------------------------------------------------------
# UniProt lookup
# ---------------------------------------------------------------------------
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"

def fetch_uniprot_id(gene: str, prefilled: Optional[str]) -> Optional[str]:
    """Return canonical human UniProt accession, using cache then REST API."""
    cache_file = UNIPROT_CACHE / f"{gene}.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        return data.get("uniprot_id")

    # Use pre-filled value if available (from Gerasimavicius sheet)
    if prefilled:
        cache_file.write_text(json.dumps({"uniprot_id": prefilled}))
        return prefilled

    # Query UniProt REST
    params = {
        "query": f'gene_exact:{gene} AND organism_id:9606 AND reviewed:true',
        "fields": "accession,gene_names",
        "format": "json",
        "size": 5,
    }
    data = get_json(UNIPROT_SEARCH, params)
    if data is None:
        cache_file.write_text(json.dumps({"uniprot_id": None}))
        return None

    results = data.get("results", [])
    if not results:
        cache_file.write_text(json.dumps({"uniprot_id": None}))
        return None

    # Prefer entry whose primary gene name matches exactly (case-insensitive)
    acc = None
    for entry in results:
        genes_obj = entry.get("genes", [])
        primary_names = [
            g["geneName"]["value"]
            for g in genes_obj
            if "geneName" in g
        ]
        if any(n.upper() == gene.upper() for n in primary_names):
            acc = entry["primaryAccession"]
            break
    if acc is None and results:
        acc = results[0]["primaryAccession"]

    cache_file.write_text(json.dumps({"uniprot_id": acc}))
    return acc

# ---------------------------------------------------------------------------
# UniProt protein sequence (for WT amino-acid lookup)
# ---------------------------------------------------------------------------
UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"

_seq_cache: dict[str, str] = {}

def fetch_protein_sequence(uniprot_id: str) -> Optional[str]:
    if uniprot_id in _seq_cache:
        return _seq_cache[uniprot_id]
    # Persist to disk so resumes don't re-fetch
    seq_file = UNIPROT_CACHE / f"{uniprot_id}_seq.txt"
    if seq_file.exists():
        seq = seq_file.read_text().strip()
        _seq_cache[uniprot_id] = seq
        return seq
    time.sleep(0.1)  # rate-limit UniProt fetches
    url = UNIPROT_FASTA.format(acc=uniprot_id)
    text = get_text(url, {})
    if not text:
        return None
    lines = text.strip().splitlines()
    seq = "".join(l for l in lines if not l.startswith(">"))
    seq_file.write_text(seq)
    _seq_cache[uniprot_id] = seq
    return seq

# ---------------------------------------------------------------------------
# ClinVar E-utilities fetch
# ---------------------------------------------------------------------------
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EFETCH_URL   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")  # optional; raises rate limit to 10/s

# Exact-match set: clinsig_raw must equal one of these (not merely contain them)
_CLINSIG_KEEP = {"pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"}

# Regex to pull protein change out of various formats
# Matches NP_xxx.x:p.X123Y  or  p.X123Y  or  X123Y  (1-letter AA codes)
_PROT_CHANGE_RE = re.compile(
    r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})"   # 3-letter: p.Val600Glu
    r"|p\.([A-Z])(\d+)([A-Z])"                    # 1-letter: p.V600E
)

_THREE_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Sec": "U", "Pyl": "O", "Asx": "B", "Glx": "Z", "Xaa": "X",
    "Ter": "*",
}


def _parse_hgvsp(hgvs_p: str) -> Optional[tuple]:
    """
    Parse an HGVSp string and return (wt_aa, position, mut_aa) in 1-letter codes.
    Returns None for synonymous, stop-gain, frameshifts, or unparseable variants.
    """
    m = _PROT_CHANGE_RE.search(hgvs_p)
    if not m:
        return None
    if m.group(1):  # 3-letter form
        wt  = _THREE_TO_ONE.get(m.group(1))
        pos = int(m.group(2))
        mut = _THREE_TO_ONE.get(m.group(3))
    else:           # 1-letter form
        wt  = m.group(4)
        pos = int(m.group(5))
        mut = m.group(6)
    if wt is None or mut is None:
        return None
    if wt == mut:   # synonymous
        return None
    if mut == "*":  # stop-gain – keep? user said missense only
        return None
    if wt == "*":
        return None
    return (wt, pos, mut)


def fetch_clinvar_variants(gene: str) -> list:
    """
    Return list of dicts with keys: hgvs_p, wt_aa, pos, mut_aa, clinsig.
    Results are cached to disk.
    """
    cache_file = CLINVAR_CACHE / f"{gene}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    variants: list = []

    # Step 1 – search for ClinVar variation IDs for this gene
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

    result = get_json(ESEARCH_URL, search_params, ncbi=True)
    if result is None:
        cache_file.write_text(json.dumps([]))
        return []

    ids = result.get("esearchresult", {}).get("idlist", [])
    if not ids:
        cache_file.write_text(json.dumps([]))
        return []

    log.info("  %s: %d ClinVar IDs found", gene, len(ids))

    # Step 2 – fetch summaries in batches of 200
    BATCH = 200
    for batch_start in range(0, len(ids), BATCH):
        batch_ids = ids[batch_start : batch_start + BATCH]
        summary_params = {
            "db": "clinvar",
            "id": ",".join(batch_ids),
            "retmode": "json",
        }
        if NCBI_API_KEY:
            summary_params["api_key"] = NCBI_API_KEY

        summ = get_json(ESUMMARY_URL, summary_params, ncbi=True)
        if summ is None:
            continue

        result_obj = summ.get("result", {})
        uid_list = result_obj.get("uids", [])

        for uid in uid_list:
            rec = result_obj.get(uid, {})

            # Clinical significance: esummary v3 uses germline_classification
            clinsig_raw = (
                rec.get("germline_classification", {}).get("description", "")
                or rec.get("clinical_significance", {}).get("description", "")
            ).lower()
            if clinsig_raw not in _CLINSIG_KEEP:
                continue

            # Molecular consequence: list of strings in v3 (e.g. ["missense variant"])
            mol_cons_raw = rec.get("molecular_consequence_list", [])
            mol_cons = [
                (mc if isinstance(mc, str) else mc.get("type", "")).lower()
                for mc in mol_cons_raw
            ]
            if not any("missense" in mc for mc in mol_cons):
                continue

            # HGVSp: parse from title field, e.g. "NM_xxx(GENE):c.xxx (p.Val600Glu)"
            title = rec.get("title", "")
            hgvs_p_raw = title  # _parse_hgvsp searches within the string

            parsed = _parse_hgvsp(hgvs_p_raw)
            if parsed is None:
                continue

            wt_aa, pos, mut_aa = parsed
            variants.append({
                "hgvs_p": hgvs_p_raw,
                "wt_aa":  wt_aa,
                "pos":    pos,
                "mut_aa": mut_aa,
                "clinsig": clinsig_raw,
            })

    # Deduplicate by (wt, pos, mut)
    seen: set = set()
    deduped: list = []
    for v in variants:
        key = (v["wt_aa"], v["pos"], v["mut_aa"])
        if key not in seen:
            seen.add(key)
            deduped.append(v)

    cache_file.write_text(json.dumps(deduped))
    return deduped


# ---------------------------------------------------------------------------
# Sequence-based WT validation
# ---------------------------------------------------------------------------

def validate_wt(variant: dict, sequence: str) -> bool:
    """Cross-check the WT amino acid against the UniProt sequence (1-indexed)."""
    pos = variant["pos"]
    if pos < 1 or pos > len(sequence):
        return False
    return sequence[pos - 1] == variant["wt_aa"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_gene_list(path: Path) -> list:
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            gene = row.get("gene", "").strip()
            if not gene:
                continue
            mech = row.get("mechanism", "").strip()
            uniprot = row.get("uniprot_id", "").strip() or None
            rows.append({"gene": gene, "mechanism": mech, "uniprot_id": uniprot})
    return rows


def main():
    genes = load_gene_list(INPUT_TSV)
    log.info("Loaded %d genes from %s", len(genes), INPUT_TSV)

    # Determine which genes already have output rows (for partial-resume at output level)
    done_genes: set = set()
    if OUTPUT_TSV.exists():
        with open(OUTPUT_TSV, newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                done_genes.add(row.get("gene", ""))

    write_header = not OUTPUT_TSV.exists() or OUTPUT_TSV.stat().st_size == 0
    out_fh = open(OUTPUT_TSV, "a", newline="")
    writer = csv.writer(out_fh, delimiter="\t")
    if write_header:
        writer.writerow(["gene", "uniprot_id", "aa_pos", "aa_wt", "aa_mut", "clinsig"])

    total_variants = 0

    for idx, gdata in enumerate(genes, 1):
        gene = gdata["gene"]
        prefilled_uniprot = gdata["uniprot_id"]

        if gene in done_genes:
            log.info("[%d/%d] %s – already in output, skipping", idx, len(genes), gene)
            continue

        log.info("[%d/%d] %s", idx, len(genes), gene)

        # 1. Resolve UniProt accession
        uniprot_id = fetch_uniprot_id(gene, prefilled_uniprot)
        if uniprot_id is None:
            log.warning("  %s: no UniProt ID found", gene)

        # 2. Fetch protein sequence for WT validation (best-effort)
        sequence: str | None = None
        if uniprot_id:
            sequence = fetch_protein_sequence(uniprot_id)

        # 3. Fetch ClinVar variants
        variants = fetch_clinvar_variants(gene)
        log.info("  %s: %d missense P/LP variants", gene, len(variants))

        # 4. Write output rows
        gene_written = 0
        for v in variants:
            if sequence and not validate_wt(v, sequence):
                log.debug("  WT mismatch %s pos %d: expected %s", gene, v["pos"], v["wt_aa"])
                continue
            if uniprot_id is None:
                continue  # skip rows with no UniProt ID — downstream can't fetch sequence
            writer.writerow([gene, uniprot_id, v["pos"], v["wt_aa"], v["mut_aa"], v["clinsig"]])
            gene_written += 1

        total_variants += gene_written

        # Flush after each gene so partial results are saved even on crash
        out_fh.flush()

    out_fh.close()
    log.info("Done. %d total variants written to %s", total_variants, OUTPUT_TSV)


if __name__ == "__main__":
    main()
