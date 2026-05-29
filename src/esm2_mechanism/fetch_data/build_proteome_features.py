"""
Phase 1 + 2 feature collection — Experiment 11
===============================================

Collects gene-level proteome features for all 1985 genes in
data/merged_gene_list.tsv.  Sources:

  1. gnomAD v4.1 constraint  (pLI, LOEUF, mis_z)
  2. Ensembl Compara         (paralog_count)
  3. Human Protein Atlas     (tissue_specificity_tau, n_tissues_expressed)
  4. PaxDb integrated human  (log_abundance_ppm)
  5. BioPlex 3.0 PPI network (PPI_degree)
  6. ClinGen dosage          (HI_score, TS_score)

Missing-data policy (from plan_experiment.md §Phase 2):
  - Binary <feature>_missing indicator for every numerical feature.
  - Median imputation applied only to the .npy matrix (raw NaN kept in TSV).
  - Family-mean-centred residuals computed for every continuous feature.

Outputs:
  data/gene_proteome_features.tsv          human-readable gene × feature table
  data/proteome_features_aligned.npy       float32 matrix aligned to merged_gene_list.tsv order
  data/proteome_feature_columns.json       column metadata

Usage:
  python -m esm2_mechanism.fetch_data.build_proteome_features
  python -m esm2_mechanism.fetch_data.build_proteome_features --force-redownload
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import functools

from esm2_mechanism.utils_paths import DATA_DIR

print = functools.partial(print, flush=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_proteome_features")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CACHE_DIR = DATA_DIR / "cache" / "proteome_features"
PILOT_CACHE_DIR = DATA_DIR / "cache" / "proteome_pilot"
PILOT_PARALOG_CACHE = PILOT_CACHE_DIR / "paralogs"

MERGED_GENE_LIST = DATA_DIR / "merged_gene_list.tsv"
PFAM_FAMILIES = DATA_DIR / "pfam_families.json"

OUT_TSV = DATA_DIR / "gene_proteome_features.tsv"
OUT_NPY = DATA_DIR / "proteome_features_aligned.npy"
OUT_COLS = DATA_DIR / "proteome_feature_columns.json"

# ---------------------------------------------------------------------------
# Source URLs
# ---------------------------------------------------------------------------
GNOMAD_CONSTRAINT_URL = (
    "https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/"
    "constraint/gnomad.v4.1.constraint_metrics.tsv"
)
GNOMAD_CACHE = CACHE_DIR / "gnomad_v4.1_constraint.tsv"
PILOT_GNOMAD_CACHE = PILOT_CACHE_DIR / "gnomad_v4.1_constraint.tsv"

ENSEMBL_PARALOG_URL = (
    "https://rest.ensembl.org/homology/symbol/human/{gene}"
    "?type=paralogues;format=condensed"
)

HPA_PROTEINATLAS_URL = (
    "https://www.proteinatlas.org/download/proteinatlas.tsv.zip"
)
HPA_PROTEINATLAS_CACHE = CACHE_DIR / "proteinatlas.tsv.zip"

PAXDB_URL = (
    "https://pax-db.org/download/5.0/datasets/9606/9606-WHOLE_ORGANISM-integrated.txt"
)
PAXDB_CACHE = CACHE_DIR / "paxdb_9606_integrated.txt"
# Manually placed file (PaxDb blocks automated download)
PAXDB_MANUAL = DATA_DIR / "9606-WHOLE_ORGANISM-integrated.txt"

BIOMART_MAPPING_CACHE = CACHE_DIR / "biomart_ensp_to_gene.tsv"

BIOPLEX_URL = (
    "https://bioplex.hms.harvard.edu/data/BioPlex_293T_Network_10K_Dec_2019.tsv"
)
BIOPLEX_CACHE = CACHE_DIR / "BioPlex_293T_Network_10K.tsv"

CLINGEN_URLS = [
    "https://ftp.clinicalgenome.org/ClinGen_gene_curation_list_GRCh38.tsv",
    "https://search.clinicalgenome.org/kb/gene-dosage/download",
]
CLINGEN_CACHE = CACHE_DIR / "clingen_gene_curation.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": "*/*", "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _download_file(url: str, dest: Path, force: bool = False) -> bool:
    """Download url → dest.  Returns True on success."""
    if dest.exists() and dest.stat().st_size > 1000 and not force:
        log.info(f"  cached: {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return True
    log.info(f"  downloading: {url}")
    t0 = time.time()
    try:
        urllib.request.urlretrieve(url, dest)
        log.info(f"  saved {dest.stat().st_size/1e6:.1f} MB in {time.time()-t0:.1f}s → {dest}")
        return True
    except Exception as e:
        log.warning(f"  download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def _fnum(v: str) -> Optional[float]:
    v = v.strip()
    if v in ("", "NA", "nan", "NaN", ".", "None", "null"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Step 1 — Load full gene universe from merged_gene_list.tsv
# ---------------------------------------------------------------------------
def load_gene_universe(tsv_path: Path) -> list[str]:
    """
    Return all gene symbols in order of first appearance in the TSV.
    Includes ALL genes (labeled or not) — 1985 entries, deduplicated preserving order.
    """
    seen: set[str] = set()
    genes: list[str] = []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            g = row["gene"].strip()
            if g and g not in seen:
                seen.add(g)
                genes.append(g)
    log.info(f"Gene universe: {len(genes)} unique genes from {tsv_path.name}")
    return genes


# ---------------------------------------------------------------------------
# Step 2 — Pfam family lookup
# ---------------------------------------------------------------------------
def load_pfam_families(path: Path) -> dict[str, Optional[str]]:
    with open(path) as f:
        d = json.load(f)
    log.info(f"Pfam families loaded: {len(d)} genes")
    return d


# ---------------------------------------------------------------------------
# Source 1 — gnomAD v4.1 constraint
# ---------------------------------------------------------------------------
def get_gnomad_constraint(force: bool = False) -> dict[str, dict]:
    """Returns {gene: {pLI, LOEUF, mis_z}}."""
    log.info("=== gnomAD v4.1 constraint ===")

    # Reuse pilot cache if it exists and is large enough
    cache_path = GNOMAD_CACHE
    if not cache_path.exists() or force:
        if PILOT_GNOMAD_CACHE.exists() and PILOT_GNOMAD_CACHE.stat().st_size > 1_000_000 and not force:
            log.info(f"  reusing pilot cache: {PILOT_GNOMAD_CACHE}")
            cache_path = PILOT_GNOMAD_CACHE
        else:
            ok = _download_file(GNOMAD_CONSTRAINT_URL, GNOMAD_CACHE, force=force)
            if not ok:
                log.warning("  gnomAD download failed — all gnomAD features will be None")
                return {}
    elif GNOMAD_CACHE.stat().st_size <= 1_000_000:
        # Stale/partial cache, try pilot
        if PILOT_GNOMAD_CACHE.exists() and PILOT_GNOMAD_CACHE.stat().st_size > 1_000_000:
            log.info(f"  reusing pilot cache: {PILOT_GNOMAD_CACHE}")
            cache_path = PILOT_GNOMAD_CACHE
        else:
            ok = _download_file(GNOMAD_CONSTRAINT_URL, GNOMAD_CACHE, force=True)
            if not ok:
                return {}

    log.info(f"  parsing {cache_path}")
    with open(cache_path) as f:
        header = f.readline().rstrip("\n").split("\t")

    def find_col(*names: str) -> Optional[int]:
        for n in names:
            if n in header:
                return header.index(n)
        return None

    idx_gene = find_col("gene")
    idx_mane = find_col("mane_select", "canonical")
    idx_lof_exp = find_col("lof.exp")
    idx_pli = find_col("lof.pLI", "pLI")
    idx_loeuf = find_col("lof.oe_ci.upper", "oe_lof_upper", "LOEUF")
    idx_misz = find_col("mis.z_score", "mis_z", "mis.z")

    missing_cols = []
    if idx_gene is None:
        missing_cols.append("gene")
    if idx_pli is None:
        missing_cols.append("pLI")
    if idx_loeuf is None:
        missing_cols.append("LOEUF")
    if idx_misz is None:
        missing_cols.append("mis_z")
    if missing_cols:
        log.error(f"  gnomAD TSV missing columns: {missing_cols}; header: {header}")
        return {}

    by_gene: dict[str, dict] = {}
    n_rows = 0
    with open(cache_path) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            needed = max(idx_gene, idx_pli, idx_loeuf, idx_misz)
            if len(parts) <= needed:
                continue
            gene = parts[idx_gene].strip()
            if not gene or gene == "NA":
                continue
            is_mane = (
                parts[idx_mane].strip().lower() in ("true", "1", "yes")
                if idx_mane is not None else False
            )
            lof_exp = _fnum(parts[idx_lof_exp]) if idx_lof_exp is not None else 0.0
            row = {
                "pLI": _fnum(parts[idx_pli]),
                "LOEUF": _fnum(parts[idx_loeuf]),
                "mis_z": _fnum(parts[idx_misz]),
                "_mane": is_mane,
                "_lof_exp": lof_exp if lof_exp is not None else 0.0,
            }
            prev = by_gene.get(gene)
            if prev is None:
                by_gene[gene] = row
            elif row["_mane"] and not prev["_mane"]:
                by_gene[gene] = row
            elif row["_mane"] == prev["_mane"] and row["_lof_exp"] > prev["_lof_exp"]:
                by_gene[gene] = row
            n_rows += 1

    log.info(f"  parsed {n_rows} rows → {len(by_gene)} unique genes")
    return by_gene


# ---------------------------------------------------------------------------
# Source 2 — Ensembl Compara paralog count
# ---------------------------------------------------------------------------
def _fetch_paralog_count_rest(gene: str, own_cache_dir: Path) -> Optional[int]:
    """Single REST call; writes to own_cache_dir/{gene}.json."""
    cache_file = own_cache_dir / f"{gene}.json"
    if cache_file.exists():
        try:
            d = json.loads(cache_file.read_text())
            return d.get("paralog_count")
        except Exception:
            pass
    url = ENSEMBL_PARALOG_URL.format(gene=gene)
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read().decode()
        data = json.loads(body)
        homologies: list = []
        for entry in data.get("data", []):
            homologies.extend(entry.get("homologies", []))
        count = sum(1 for h in homologies if "paralog" in h.get("type", "").lower())
        cache_file.write_text(json.dumps({"paralog_count": count}))
        return count
    except Exception as e:
        log.debug(f"  paralog fetch failed for {gene}: {e}")
        cache_file.write_text(json.dumps({"paralog_count": None, "error": str(e)}))
        return None


def get_paralogs(genes: list[str]) -> dict[str, Optional[int]]:
    """
    Reuse pilot cache (data/cache/proteome_pilot/paralogs/) where available.
    Fetch missing genes via REST at 10 req/s.
    """
    log.info("=== Ensembl Compara paralogs ===")
    own_cache = CACHE_DIR / "paralogs"
    out: dict[str, Optional[int]] = {}
    to_fetch: list[str] = []

    for gene in genes:
        # Check pilot cache first
        pilot_file = PILOT_PARALOG_CACHE / f"{gene}.json"
        own_file = own_cache / f"{gene}.json"
        if pilot_file.exists():
            try:
                d = json.loads(pilot_file.read_text())
                out[gene] = d.get("paralog_count")
                continue
            except Exception:
                pass
        if own_file.exists():
            try:
                d = json.loads(own_file.read_text())
                out[gene] = d.get("paralog_count")
                continue
            except Exception:
                pass
        to_fetch.append(gene)

    log.info(f"  {len(out)} genes from cache, {len(to_fetch)} need REST fetch")
    t0 = time.time()
    for i, gene in enumerate(to_fetch):
        c = _fetch_paralog_count_rest(gene, own_cache)
        out[gene] = c
        time.sleep(0.1)  # 10 req/s
        if (i + 1) % 100 == 0:
            log.info(f"  fetched {i+1}/{len(to_fetch)} (elapsed {time.time()-t0:.1f}s)")

    n_ok = sum(1 for v in out.values() if v is not None)
    log.info(f"  paralog_count coverage: {n_ok}/{len(genes)}")
    return out


# ---------------------------------------------------------------------------
# Source 3 — Human Protein Atlas
# ---------------------------------------------------------------------------
def get_hpa_features(genes: list[str], force: bool = False) -> dict[str, dict]:
    """
    Returns {gene: {tissue_specificity_tau: float|None, n_tissues_expressed: int|None}}.

    Downloads proteinatlas.tsv.zip (bulk export).  Parses:
      - "RNA tissue specificity" column  → tau proxy via text label mapping
      - "RNA tissue category" or tissue expression columns → n_tissues_expressed

    Falls back gracefully if download fails or columns are absent.
    """
    log.info("=== Human Protein Atlas ===")
    result: dict[str, dict] = {g: {"tissue_specificity_tau": None, "n_tissues_expressed": None} for g in genes}
    genes_set = set(genes)

    TAU_MAP = {
        "tissue enriched": 0.8,
        "group enriched": 0.7,
        "tissue enhanced": 0.6,
        "low tissue specificity": 0.2,
        "not detected": 0.0,
    }

    ok = _download_file(HPA_PROTEINATLAS_URL, HPA_PROTEINATLAS_CACHE, force=force)
    if not ok:
        log.warning("  HPA download failed — tissue features will be None for all genes")
        return result

    try:
        import zipfile
        with zipfile.ZipFile(HPA_PROTEINATLAS_CACHE, "r") as zf:
            # Find the TSV inside the zip
            tsv_names = [n for n in zf.namelist() if n.endswith(".tsv")]
            if not tsv_names:
                log.warning(f"  HPA zip contains no TSV: {zf.namelist()}")
                return result
            tsv_name = tsv_names[0]
            log.info(f"  reading {tsv_name} from zip")
            with zf.open(tsv_name) as raw:
                import io
                f = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
                reader = csv.DictReader(f, delimiter="\t")
                fieldnames = reader.fieldnames or []
                log.info(f"  HPA columns ({len(fieldnames)}): {fieldnames[:10]}")

                # Find gene name column and text specificity label column only.
                # "RNA tissue specificity score" is NOT Yanai τ — it is an HPA-internal
                # integer (range 0–∞); we use only the text label mapped to [0, 0.8].
                col_gene = None
                col_tau_text = None
                for name in fieldnames:
                    nl = name.lower().strip()
                    if nl in ("gene name", "gene_name", "gene"):
                        col_gene = name
                    if nl == "rna tissue specificity":
                        col_tau_text = name

                if col_gene is None:
                    log.warning(f"  HPA: gene column not found; header: {fieldnames[:15]}")
                    return result

                n_tau = 0
                for row in reader:
                    g = row.get(col_gene, "").strip()
                    if g not in genes_set:
                        continue
                    tau = None
                    if col_tau_text:
                        label = row.get(col_tau_text, "").strip().lower()
                        tau = TAU_MAP.get(label)
                    result[g]["tissue_specificity_tau"] = tau
                    if tau is not None:
                        n_tau += 1

                log.info(f"  tau: {n_tau} genes assigned")
    except Exception as e:
        log.warning(f"  HPA parse failed: {e}")

    n_tau = sum(1 for v in result.values() if v["tissue_specificity_tau"] is not None)
    log.info(f"  HPA coverage: tau={n_tau}/{len(genes)} (n_tissues_expressed not available from bulk export)")
    return result


# ---------------------------------------------------------------------------
# Source 4 — PaxDb protein abundance
# ---------------------------------------------------------------------------
def _fetch_biomart_ensp_to_gene() -> dict[str, str]:
    """
    BioMart bulk query: ENSP → HGNC gene symbol.
    Returns {ensp_id: gene_symbol}.
    Caches result to BIOMART_MAPPING_CACHE.
    """
    if BIOMART_MAPPING_CACHE.exists() and BIOMART_MAPPING_CACHE.stat().st_size > 1000:
        log.info(f"  BioMart mapping cached: {BIOMART_MAPPING_CACHE}")
        mapping: dict[str, str] = {}
        with open(BIOMART_MAPPING_CACHE) as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 2 and row[0] and row[1]:
                    mapping[row[0]] = row[1]
        log.info(f"  loaded {len(mapping)} ENSP→gene mappings")
        return mapping

    log.info("  fetching ENSP→gene symbol mapping from BioMart ...")
    biomart_url = (
        "https://www.ensembl.org/biomart/martservice?query="
        "<?xml version='1.0' encoding='UTF-8'?>"
        "<!DOCTYPE Query>"
        "<Query virtualSchemaName='default' formatter='TSV' header='0' uniqueRows='1' count='' datasetConfigVersion='0.6'>"
        "<Dataset name='hsapiens_gene_ensembl' interface='default'>"
        "<Attribute name='ensembl_peptide_id'/>"
        "<Attribute name='hgnc_symbol'/>"
        "</Dataset>"
        "</Query>"
    )
    try:
        data = _http_get(biomart_url, timeout=120)
        lines = data.decode("utf-8", errors="replace").strip().split("\n")
        mapping: dict[str, str] = {}
        with open(BIOMART_MAPPING_CACHE, "w") as f:
            for line in lines:
                parts = line.split("\t")
                if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                    ensp = parts[0].strip()
                    gene = parts[1].strip()
                    mapping[ensp] = gene
                    f.write(f"{ensp}\t{gene}\n")
        log.info(f"  BioMart: {len(mapping)} ENSP→gene mappings")
        return mapping
    except Exception as e:
        log.warning(f"  BioMart query failed: {e}")
        return {}


def get_paxdb_abundance(genes: list[str], force: bool = False) -> dict[str, Optional[float]]:
    """
    Returns {gene: log10(abundance_ppm + 1e-3)} or {gene: None}.

    This version of the file has gene symbol in col 0, string ID in col 1,
    abundance_ppm in col 2.  We use the gene symbol directly.

    File resolution order:
      1. PAXDB_MANUAL (data/9606-WHOLE_ORGANISM-integrated.txt) — manually placed
      2. PAXDB_CACHE   (data/cache/proteome_features/...)        — previously downloaded
      3. HTTP download (blocked by PaxDb as of 2026-05)
    """
    log.info("=== PaxDb abundance ===")
    result: dict[str, Optional[float]] = {g: None for g in genes}
    genes_set = set(genes)

    # Resolve file path
    paxdb_path = None
    if PAXDB_MANUAL.exists() and PAXDB_MANUAL.stat().st_size > 1000:
        paxdb_path = PAXDB_MANUAL
        log.info(f"  using manually placed file: {paxdb_path}")
    elif PAXDB_CACHE.exists() and PAXDB_CACHE.stat().st_size > 1000 and not force:
        paxdb_path = PAXDB_CACHE
        log.info(f"  using cached file: {paxdb_path}")
    else:
        ok = _download_file(PAXDB_URL, PAXDB_CACHE, force=force)
        if ok:
            paxdb_path = PAXDB_CACHE
        else:
            log.warning("  PaxDb not available — log_abundance_ppm will be None for all genes")
            return result

    try:
        n_hit = 0
        with open(paxdb_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                gene = parts[0].strip()
                abund = _fnum(parts[2])
                if gene in genes_set and abund is not None and result[gene] is None:
                    result[gene] = math.log10(abund + 1e-3)
                    n_hit += 1
        n_covered = sum(1 for v in result.values() if v is not None)
        log.info(f"  PaxDb coverage: {n_covered}/{len(genes)} genes assigned log_abundance_ppm")
        return result
    except Exception as e:
        log.warning(f"  PaxDb parse failed: {e}")
        return result


# ---------------------------------------------------------------------------
# Source 5 — BioPlex 3.0 PPI degree
# ---------------------------------------------------------------------------
def get_bioplex_degree(genes: list[str], force: bool = False) -> dict[str, Optional[int]]:
    """
    Returns {gene: degree} (number of unique interaction partners).
    Handles TSV with GeneA/GeneB columns (gene symbols).
    """
    log.info("=== BioPlex 3.0 PPI ===")
    result: dict[str, Optional[int]] = {g: None for g in genes}
    genes_set = set(genes)

    ok = _download_file(BIOPLEX_URL, BIOPLEX_CACHE, force=force)
    if not ok:
        log.warning("  BioPlex download failed — PPI_degree will be None for all genes")
        return result

    try:
        degree: dict[str, set[str]] = {}
        with open(BIOPLEX_CACHE) as f:
            reader = csv.DictReader(f, delimiter="\t")
            fieldnames = reader.fieldnames or []
            # Find GeneA / GeneB columns (case-insensitive)
            col_a = None
            col_b = None
            for name in fieldnames:
                nl = name.lower().strip()
                if nl in ("genea", "gene_a", "symbola", "symbol_a", "gene a", "symbol a"):
                    col_a = name
                elif nl in ("geneb", "gene_b", "symbolb", "symbol_b", "gene b", "symbol b"):
                    col_b = name
            if col_a is None or col_b is None:
                log.warning(f"  BioPlex: symbol columns not found in {fieldnames}; trying any gene/symbol col")
                for name in fieldnames:
                    nl = name.lower().strip()
                    if ("gene" in nl or "symbol" in nl) and col_a is None:
                        col_a = name
                    elif ("gene" in nl or "symbol" in nl) and col_b is None and name != col_a:
                        col_b = name
            if col_a is None or col_b is None:
                log.warning(f"  BioPlex: cannot find gene symbol columns; skipping. Header: {fieldnames}")
                return result

            for row in reader:
                ga = row.get(col_a, "").strip()
                gb = row.get(col_b, "").strip()
                if not ga or not gb:
                    continue
                # Count degree for all genes (not just our universe) then filter
                degree.setdefault(ga, set()).add(gb)
                degree.setdefault(gb, set()).add(ga)

        n_hit = 0
        for gene in genes_set:
            if gene in degree:
                result[gene] = len(degree[gene])
                n_hit += 1

        log.info(f"  BioPlex coverage: {n_hit}/{len(genes)} genes with PPI_degree")
    except Exception as e:
        log.warning(f"  BioPlex parse failed: {e}")

    return result


# ---------------------------------------------------------------------------
# Source 6 — ClinGen dosage sensitivity
# ---------------------------------------------------------------------------
def get_clingen_dosage(genes: list[str], force: bool = False) -> dict[str, dict]:
    """
    Returns {gene: {HI_score: int|None, TS_score: int|None}}.
    Scores are 0-3: 0=no evidence, 1=little, 2=some, 3=sufficient.
    """
    log.info("=== ClinGen dosage sensitivity ===")
    result: dict[str, dict] = {g: {"HI_score": None, "TS_score": None} for g in genes}
    genes_set = set(genes)

    ok = False
    for url in CLINGEN_URLS:
        ok = _download_file(url, CLINGEN_CACHE, force=force)
        if ok:
            break
    if not ok:
        log.warning("  ClinGen download failed — HI_score/TS_score will be None for all genes")
        return result

    try:
        # ClinGen file: comment lines start with '#', then headerless data rows.
        # Positional columns (verified from downloaded file):
        #   col 0  = gene symbol
        #   col 5  = HI description text  ("Sufficient evidence for dosage pathogenicity", etc.)
        #   col 12 = TS description text
        IDX_GENE = 0
        IDX_HI = 5
        IDX_TS = 12

        TEXT_MAP = {
            "sufficient evidence for dosage pathogenicity": 3,
            "sufficient evidence": 3,
            "some evidence": 2,
            "emerging evidence": 1,
            "little evidence": 1,
            "no evidence available": 0,
            "not yet evaluated": None,
            "gene associated with autosomal recessive phenotype": None,
        }

        def extract_score_text(val: str) -> Optional[int]:
            vl = val.strip().lower()
            if vl in TEXT_MAP:
                return TEXT_MAP[vl]
            # Try numeric
            try:
                iv = int(float(vl))
                return iv if 0 <= iv <= 3 else None
            except ValueError:
                return None

        with open(CLINGEN_CACHE) as f:
            lines = f.readlines()

        n_hit = 0
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split("\t")
            if len(parts) <= IDX_GENE:
                continue
            gene = parts[IDX_GENE].strip()
            if gene not in genes_set:
                continue
            hi_raw = parts[IDX_HI].strip() if len(parts) > IDX_HI else ""
            ts_raw = parts[IDX_TS].strip() if len(parts) > IDX_TS else ""
            hi = extract_score_text(hi_raw)
            ts = extract_score_text(ts_raw)
            if result[gene]["HI_score"] is None:
                result[gene]["HI_score"] = hi
            if result[gene]["TS_score"] is None:
                result[gene]["TS_score"] = ts
            n_hit += 1

        n_hi = sum(1 for v in result.values() if v["HI_score"] is not None)
        n_ts = sum(1 for v in result.values() if v["TS_score"] is not None)
        log.info(f"  ClinGen coverage: HI={n_hi}/{len(genes)}, TS={n_ts}/{len(genes)}")
    except Exception as e:
        log.warning(f"  ClinGen parse failed: {e}")

    return result


# ---------------------------------------------------------------------------
# Phase 2 — Build feature table and aligned matrix
# ---------------------------------------------------------------------------
# Raw continuous feature columns (in the order they'll appear in the TSV)
CONT_FEATURES = [
    "pLI",
    "LOEUF",
    "mis_z",
    "paralog_count",
    "tissue_specificity_tau",   # HPA text label mapped to {0, 0.2, 0.6, 0.7, 0.8}
    # n_tissues_expressed: dropped — not in HPA bulk export (0% coverage)
    "log_abundance_ppm",        # PaxDb integrated human, manually downloaded
    "PPI_degree",
    "HI_score",
    "TS_score",
]


def build_feature_table(
    genes: list[str],
    families: dict[str, Optional[str]],
    gnomad: dict[str, dict],
    paralogs: dict[str, Optional[int]],
    hpa: dict[str, dict],
    paxdb: dict[str, Optional[float]],
    bioplex: dict[str, Optional[int]],
    clingen: dict[str, dict],
) -> tuple[list[dict], list[str]]:
    """
    Build per-gene row dicts with:
      - raw continuous features (NaN where missing)
      - family-mean-centred residuals  (<feature>_familyresid)
      - missingness indicators         (<feature>_missing)
      - is_singleton_family indicator

    Returns (rows, column_names) — rows are dicts, columns are ordered.
    TSV contains raw values (NaN as empty); residuals and missingness also included.
    """
    rows: list[dict] = []

    for gene in genes:
        family = families.get(gene)
        row: dict = {
            "gene": gene,
            "pfam_family": family if family is not None else "",
        }
        # gnomAD
        g = gnomad.get(gene, {})
        row["pLI"] = g.get("pLI")
        row["LOEUF"] = g.get("LOEUF")
        row["mis_z"] = g.get("mis_z")
        # paralogs
        row["paralog_count"] = paralogs.get(gene)
        # HPA — text-label specificity category mapped to {0, 0.2, 0.6, 0.7, 0.8}
        h = hpa.get(gene, {})
        row["tissue_specificity_tau"] = h.get("tissue_specificity_tau")
        # n_tissues_expressed dropped — not in HPA bulk export (0% coverage)
        # PaxDb abundance
        row["log_abundance_ppm"] = paxdb.get(gene)
        # BioPlex
        row["PPI_degree"] = bioplex.get(gene)
        # ClinGen
        cl = clingen.get(gene, {})
        row["HI_score"] = cl.get("HI_score")
        row["TS_score"] = cl.get("TS_score")
        rows.append(row)

    # --- Family-mean-centred residuals ---
    # Group genes by family; compute family mean for each continuous feature
    family_groups: dict[str, list[int]] = {}  # family → list of row indices
    for i, row in enumerate(rows):
        fam = row["pfam_family"]
        if fam:
            family_groups.setdefault(fam, []).append(i)

    for feat in CONT_FEATURES:
        # Family means: {family: mean_value_or_None}
        fam_means: dict[str, Optional[float]] = {}
        for fam, idxs in family_groups.items():
            vals = [rows[i][feat] for i in idxs if rows[i][feat] is not None]
            fam_means[fam] = float(np.mean(vals)) if vals else None

        for i, row in enumerate(rows):
            raw = row[feat]
            fam = row["pfam_family"]
            is_singleton = (fam != "" and fam in family_groups and len(family_groups[fam]) == 1)
            if is_singleton:
                row[f"{feat}_familyresid"] = 0.0 if raw is not None else None
            elif fam and fam_means.get(fam) is not None and raw is not None:
                row[f"{feat}_familyresid"] = raw - fam_means[fam]
            else:
                row[f"{feat}_familyresid"] = None

    # --- is_singleton_family indicator ---
    for row in rows:
        fam = row["pfam_family"]
        is_singleton = (fam != "" and fam in family_groups and len(family_groups[fam]) == 1)
        row["is_singleton_family"] = 1 if is_singleton else 0

    # --- Missingness indicators ---
    for feat in CONT_FEATURES:
        for row in rows:
            row[f"{feat}_missing"] = 0 if row[feat] is not None else 1
        for row in rows:
            row[f"{feat}_familyresid_missing"] = 0 if row.get(f"{feat}_familyresid") is not None else 1

    # Build column list in a defined order
    col_names: list[str] = ["gene", "pfam_family", "is_singleton_family"]
    for feat in CONT_FEATURES:
        col_names.append(feat)
        col_names.append(f"{feat}_missing")
        col_names.append(f"{feat}_familyresid")
        col_names.append(f"{feat}_familyresid_missing")

    return rows, col_names


def save_tsv(rows: list[dict], col_names: list[str], path: Path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(col_names)
        for row in rows:
            line = []
            for col in col_names:
                v = row.get(col)
                if v is None:
                    line.append("")
                else:
                    line.append(str(v))
            writer.writerow(line)
    log.info(f"Wrote TSV: {path} ({len(rows)} rows × {len(col_names)} cols)")


def build_aligned_matrix(
    rows: list[dict],
    col_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Build float32 matrix.  Numerical columns only (skip 'gene', 'pfam_family').
    Median-impute missing values (NaN) column-wise.
    Returns (matrix, numerical_col_names).
    """
    num_cols = [c for c in col_names if c not in ("gene", "pfam_family")]

    X = np.full((len(rows), len(num_cols)), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        for j, col in enumerate(num_cols):
            v = row.get(col)
            if v is not None:
                try:
                    X[i, j] = float(v)
                except (ValueError, TypeError):
                    pass

    # Median imputation column-wise
    for j in range(X.shape[1]):
        col_data = X[:, j]
        nan_mask = np.isnan(col_data)
        if nan_mask.any():
            finite = col_data[~nan_mask]
            if finite.size > 0:
                med = float(np.median(finite))
            else:
                med = 0.0
            X[nan_mask, j] = med

    return X.astype(np.float32), num_cols


def print_coverage_summary(rows: list[dict], genes: list[str]):
    """Print per-source coverage table to stdout."""
    n = len(genes)
    sources = {
        "gnomAD (pLI)": "pLI",
        "gnomAD (LOEUF)": "LOEUF",
        "gnomAD (mis_z)": "mis_z",
        "paralogs": "paralog_count",
        "HPA (specificity cat.)": "tissue_specificity_tau",
        "PaxDb (log_abund)": "log_abundance_ppm",
        "BioPlex (degree)": "PPI_degree",
        "ClinGen (HI)": "HI_score",
        "ClinGen (TS)": "TS_score",
    }
    gene_to_row = {r["gene"]: r for r in rows}
    print("\n" + "=" * 55)
    print(f"{'Source':<25}  {'Covered':>8}  {'Total':>6}  {'%':>6}")
    print("-" * 55)
    for label, feat in sources.items():
        covered = sum(1 for g in genes if gene_to_row.get(g, {}).get(feat) is not None)
        pct = 100.0 * covered / n if n > 0 else 0.0
        print(f"{label:<25}  {covered:>8}  {n:>6}  {pct:>5.1f}%")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build gene-level proteome feature matrix for Experiment 11."
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Bypass all caches and re-download every source.",
    )
    args = parser.parse_args()
    force = args.force_redownload

    missing = [p for p in [MERGED_GENE_LIST, PFAM_FAMILIES] if not p.exists()]
    if missing:
        raise FileNotFoundError("Required input(s) not found:\n" + "\n".join(f"  {p}" for p in missing))

    for d in (CACHE_DIR, CACHE_DIR / "paralogs"):
        d.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("build_proteome_features.py — Experiment 11 Phase 1+2")
    log.info("=" * 60)

    # 1. Gene universe
    genes = load_gene_universe(MERGED_GENE_LIST)

    # 2. Pfam families
    families = load_pfam_families(PFAM_FAMILIES)

    # 3. Sources — each wrapped in try/except so one failure doesn't abort
    gnomad: dict[str, dict] = {}
    try:
        gnomad = get_gnomad_constraint(force=force)
    except Exception as e:
        log.warning(f"gnomAD source failed entirely: {e} — using empty dict")

    paralogs: dict[str, Optional[int]] = {g: None for g in genes}
    try:
        paralogs = get_paralogs(genes)
    except Exception as e:
        log.warning(f"Paralog source failed entirely: {e} — using None for all genes")

    hpa: dict[str, dict] = {g: {"tissue_specificity_tau": None, "n_tissues_expressed": None} for g in genes}
    try:
        hpa = get_hpa_features(genes, force=force)
    except Exception as e:
        log.warning(f"HPA source failed entirely: {e} — using None for all genes")

    paxdb: dict[str, Optional[float]] = {g: None for g in genes}
    try:
        paxdb = get_paxdb_abundance(genes, force=force)
    except Exception as e:
        log.warning(f"PaxDb source failed entirely: {e} — using None for all genes")

    bioplex: dict[str, Optional[int]] = {g: None for g in genes}
    try:
        bioplex = get_bioplex_degree(genes, force=force)
    except Exception as e:
        log.warning(f"BioPlex source failed entirely: {e} — using None for all genes")

    clingen: dict[str, dict] = {g: {"HI_score": None, "TS_score": None} for g in genes}
    try:
        clingen = get_clingen_dosage(genes, force=force)
    except Exception as e:
        log.warning(f"ClinGen source failed entirely: {e} — using None for all genes")

    # 4. Phase 2: build feature table
    log.info("=== Phase 2: building feature table ===")
    rows, col_names = build_feature_table(
        genes, families, gnomad, paralogs, hpa, paxdb, bioplex, clingen
    )

    # 5. Save TSV (raw values, NaN as empty string)
    save_tsv(rows, col_names, OUT_TSV)

    # 6. Build aligned numpy matrix (median-imputed)
    log.info("=== Building aligned numpy matrix ===")
    X, num_cols = build_aligned_matrix(rows, col_names)
    np.save(OUT_NPY, X)
    log.info(f"Wrote numpy matrix: {OUT_NPY} shape={X.shape} dtype={X.dtype}")

    # 7. Save column metadata
    col_metadata = {
        "all_columns": col_names,
        "numerical_columns": num_cols,
        "continuous_features": CONT_FEATURES,
        "n_genes": len(genes),
        "n_numerical_cols": len(num_cols),
        "notes": {
            "gene_order": "Same as merged_gene_list.tsv order of first appearance",
            "imputation": "Median imputation applied only to .npy; raw NaN kept in TSV",
            "residuals": "family-mean-centred, named <feat>_familyresid",
            "missingness": "Binary <feat>_missing indicators included",
            "scaling": "NOT applied here — fit scaler on training fold during modelling",
        },
    }
    OUT_COLS.write_text(json.dumps(col_metadata, indent=2))
    log.info(f"Wrote column metadata: {OUT_COLS}")

    # 8. Coverage summary
    print_coverage_summary(rows, genes)

    log.info("Done.")


if __name__ == "__main__":
    main()
