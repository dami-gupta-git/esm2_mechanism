"""
Phase 1 + 2 feature collection — Experiment 11
===============================================

Collects gene-level proteome features for all genes in
data/gene_universe.tsv.  Sources:

  1. gnomAD v4.1 constraint  (pLI, LOEUF, mis_z)
  2. Ensembl Compara         (paralog_count)
  3. Human Protein Atlas     (tissue_specificity_tau)
  4. PaxDb integrated human  (log_abundance_ppm)
  5. BioPlex 3.0 PPI network (PPI_degree)
  6. GeneBayes s_het         (s_het) — replaces ClinGen HI/TS (80%/63% missing)

Missing-data policy (from plan_experiment.md §Phase 2):
  - Binary <feature>_missing indicator for every numerical feature.
  - Median imputation applied only to the .npy matrix (raw NaN kept in TSV).
  - Family-mean-centred residuals computed for every continuous feature.

Outputs:
  data/gene_proteome_features.tsv          human-readable gene × feature table
  data/proteome_features_aligned.npy       float32 matrix aligned to gene_universe.tsv order
  data/proteome_feature_columns.json       column metadata

Usage:
  python -m esm2_mechanism.fetch_data.build_proteome_features
  python -m esm2_mechanism.fetch_data.build_proteome_features --force-redownload
"""

from __future__ import annotations

import argparse
import csv
import functools
import gzip
import io
import itertools
import json
import math
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

from esm2_mechanism.utils_paths import DATA_DIR, GNOMAD_LOF_FILE, PAXDB_FILE, S_HET_FILE

print = functools.partial(print, flush=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CACHE_DIR = DATA_DIR / "cache" / "proteome_features"
PILOT_CACHE_DIR = DATA_DIR / "cache" / "proteome_pilot"
PILOT_PARALOG_CACHE = PILOT_CACHE_DIR / "paralogs"

GENE_UNIVERSE = DATA_DIR / "gene_universe.tsv"

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

HPA_PROTEINATLAS_URL = "https://www.proteinatlas.org/download/proteinatlas.tsv.zip"
HPA_PROTEINATLAS_CACHE = CACHE_DIR / "proteinatlas.tsv.zip"

PAXDB_URL = (
    "https://pax-db.org/download/5.0/datasets/9606/9606-WHOLE_ORGANISM-integrated.txt"
)
PAXDB_CACHE = CACHE_DIR / "paxdb_9606_integrated.txt"
# Manually placed file (PaxDb blocks automated download)
PAXDB_MANUAL = PAXDB_FILE

BIOPLEX_URL = (
    "https://bioplex.hms.harvard.edu/data/BioPlex_293T_Network_10K_Dec_2019.tsv"
)
BIOPLEX_CACHE = CACHE_DIR / "BioPlex_293T_Network_10K.tsv"

# Manually placed file (Zeng et al. 2023 GeneBayes, https://doi.org/10.5281/zenodo.7939767)
SHET_MANUAL = S_HET_FILE

# gnomAD v2.1.1 — used only to build ensg→gene_symbol map for s_het join
# (gnomAD v4.1 gene_id column is a bare integer, not an Ensembl ID)
GNOMAD_V2_MANUAL = GNOMAD_LOF_FILE
GNOMAD_V2_ENSG_CACHE = CACHE_DIR / "gnomad_v2.1.1_ensg_symbol.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _download_file(url: str, dest: Path, force: bool = False) -> bool:
    """Download url → dest.  Returns True on success."""
    if dest.exists() and dest.stat().st_size > 1_000_000 and not force:
        print(f"  cached: {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return True
    print(f"  downloading: {url}")
    t0 = time.time()
    try:
        urllib.request.urlretrieve(url, dest)
        print(
            f"  saved {dest.stat().st_size/1e6:.1f} MB in {time.time()-t0:.1f}s → {dest}"
        )
        return True
    except Exception as e:
        print(f"WARNING: download failed: {e}")
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
# Step 1 — Load gene universe from gene_universe.tsv
# ---------------------------------------------------------------------------
def load_gene_universe(tsv_path: Path) -> tuple[list[str], dict[str, str]]:
    """
    Return (genes_in_order, {gene: pfam_family}) from gene_universe.tsv.
    All genes in the file have a pfam_family (filtered upstream by build_gene_universe.py).
    """
    genes: list[str] = []
    families: dict[str, str] = {}
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            g = row["gene"].strip()
            if g and g not in families:
                genes.append(g)
                families[g] = row["pfam_family"].strip()
    print(f"Gene universe: {len(genes)} genes from {tsv_path.name}")
    return genes, families


# ---------------------------------------------------------------------------
# Source 1 — gnomAD v4.1 constraint
# ---------------------------------------------------------------------------
def get_gnomad_constraint(force: bool = False) -> dict[str, dict]:
    """Returns {gene: {pLI, LOEUF, mis_z}}."""
    print("=== gnomAD v4.1 constraint ===")

    # Reuse pilot cache if it exists and is large enough
    cache_path = GNOMAD_CACHE
    if not cache_path.exists() or force:
        if (
            PILOT_GNOMAD_CACHE.exists()
            and PILOT_GNOMAD_CACHE.stat().st_size > 1_000_000
            and not force
        ):
            print(f"  reusing pilot cache: {PILOT_GNOMAD_CACHE}")
            cache_path = PILOT_GNOMAD_CACHE
        else:
            ok = _download_file(GNOMAD_CONSTRAINT_URL, GNOMAD_CACHE, force=force)
            if not ok:
                print(
                    "WARNING: gnomAD download failed — all gnomAD features will be None"
                )
                return {}
    elif GNOMAD_CACHE.stat().st_size <= 1_000_000:
        # Stale/partial cache, try pilot
        if (
            PILOT_GNOMAD_CACHE.exists()
            and PILOT_GNOMAD_CACHE.stat().st_size > 1_000_000
        ):
            print(f"  reusing pilot cache: {PILOT_GNOMAD_CACHE}")
            cache_path = PILOT_GNOMAD_CACHE
        else:
            ok = _download_file(GNOMAD_CONSTRAINT_URL, GNOMAD_CACHE, force=True)
            if not ok:
                return {}

    print(f"  parsing {cache_path}")
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
        print(f"ERROR: gnomAD TSV missing columns: {missing_cols}; header: {header}")
        return {}

    by_gene: dict[str, dict] = {}
    n_rows = 0
    # Every column index we will dereference per row, including the optional
    # _mane and lof.exp tie-break columns — a row must be long enough for all
    # of them or we skip it (otherwise parts[idx_mane]/parts[idx_lof_exp] raises).
    needed = max(
        i
        for i in (idx_gene, idx_pli, idx_loeuf, idx_misz, idx_mane, idx_lof_exp)
        if i is not None
    )
    n_skipped = 0
    with open(cache_path) as f:
        next(f)
        for line in f:
            n_rows += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= needed:
                n_skipped += 1
                continue
            gene = parts[idx_gene].strip()
            if not gene or gene == "NA":
                n_skipped += 1
                continue
            is_mane = (
                parts[idx_mane].strip().lower() in ("true", "1", "yes")
                if idx_mane is not None
                else False
            )
            lof_exp = _fnum(parts[idx_lof_exp]) if idx_lof_exp is not None else None
            row = {
                "pLI": _fnum(parts[idx_pli]),
                "LOEUF": _fnum(parts[idx_loeuf]),
                "mis_z": _fnum(parts[idx_misz]),
                "_mane": is_mane,
                "_lof_exp": lof_exp,
            }
            prev = by_gene.get(gene)
            if prev is None:
                by_gene[gene] = row
            elif row["_mane"] and not prev["_mane"]:
                by_gene[gene] = row
            elif row["_mane"] == prev["_mane"] and (
                (row["_lof_exp"] is not None and prev["_lof_exp"] is None)
                or (
                    row["_lof_exp"] is not None
                    and prev["_lof_exp"] is not None
                    and row["_lof_exp"] > prev["_lof_exp"]
                )
            ):
                by_gene[gene] = row

    print(f"  parsed {n_rows} rows ({n_skipped} skipped) → {len(by_gene)} unique genes")
    return by_gene


# ---------------------------------------------------------------------------
# Source 2 — Ensembl Compara paralog count
# ---------------------------------------------------------------------------
def _load_paralog_cache(cache_file: Path) -> tuple[bool, Optional[int]]:
    """
    Read a paralog cache file. Returns (is_usable, paralog_count).

    A cache entry is only usable if it represents a real REST success: an int
    count >= 0 with no "error" tag. Anything else — parse error, count=None,
    or tagged "error" — returns (False, None) so callers re-fetch instead of
    treating a poisoned zero/None as truth.
    """
    if not cache_file.exists():
        return False, None
    try:
        d = json.loads(cache_file.read_text())
    except Exception:
        return False, None
    if "error" in d:
        return False, None
    val = d.get("paralog_count")
    if not isinstance(val, int) or val < 0:
        return False, None
    return True, val


def _fetch_paralog_count_rest(gene: str, own_cache_dir: Path) -> Optional[int]:
    """Single REST call; writes to own_cache_dir/{gene}.json."""
    cache_file = own_cache_dir / f"{gene}.json"
    usable, cached = _load_paralog_cache(cache_file)
    if usable:
        return cached

    url = ENSEMBL_PARALOG_URL.format(gene=gene)
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read().decode()
        data = json.loads(body)
        # An empty "data" list means the gene symbol resolved to no entry —
        # could be a real "no paralogs" answer OR a transient Ensembl hiccup.
        # We can't tell them apart, so write an error-tagged sentinel that the
        # next run will retry, rather than caching a permanent 0.
        entries = data.get("data")
        if not entries:
            err = "empty data payload (HTTP 200 but no entries)"
            print(f"  paralog fetch ambiguous for {gene}: {err}")
            cache_file.write_text(json.dumps({"paralog_count": None, "error": err}))
            return None
        homologies: list = []
        for entry in entries:
            homologies.extend(entry.get("homologies", []))
        count = sum(1 for h in homologies if "paralog" in h.get("type", "").lower())
        cache_file.write_text(json.dumps({"paralog_count": count}))
        return count
    except Exception as e:
        print(
            f"  paralog fetch failed for {gene}: {e} — not caching, will retry next run"
        )
        return None


def get_paralogs(genes: list[str]) -> dict[str, Optional[int]]:
    """
    Reuse pilot cache (data/cache/proteome_pilot/paralogs/) where available.
    Fetch missing genes via REST at 10 req/s.
    """
    print("=== Ensembl Compara paralogs ===")
    own_cache = CACHE_DIR / "paralogs"
    out: dict[str, Optional[int]] = {}
    to_fetch: list[str] = []

    for gene in genes:
        # Check pilot cache first, then own. Both go through _load_paralog_cache
        # so that error-tagged or malformed entries fall through to a re-fetch
        # instead of silently propagating None.
        pilot_file = PILOT_PARALOG_CACHE / f"{gene}.json"
        own_file = own_cache / f"{gene}.json"
        usable, cached = _load_paralog_cache(pilot_file)
        if usable:
            out[gene] = cached
            continue
        usable, cached = _load_paralog_cache(own_file)
        if usable:
            out[gene] = cached
            continue
        to_fetch.append(gene)

    print(f"  {len(out)} genes from cache, {len(to_fetch)} need REST fetch")
    t0 = time.time()
    for i, gene in enumerate(to_fetch):
        c = _fetch_paralog_count_rest(gene, own_cache)
        out[gene] = c
        time.sleep(0.1)  # 10 req/s
        if (i + 1) % 100 == 0:
            print(f"  fetched {i+1}/{len(to_fetch)} (elapsed {time.time()-t0:.1f}s)")

    n_ok = sum(1 for v in out.values() if v is not None)
    print(f"  paralog_count coverage: {n_ok}/{len(genes)}")
    return out


# ---------------------------------------------------------------------------
# Source 3 — Human Protein Atlas
# ---------------------------------------------------------------------------
def get_hpa_features(genes: list[str], force: bool = False) -> dict[str, dict]:
    """
    Returns {gene: {tissue_specificity_tau: float|None}}.

    Downloads proteinatlas.tsv.zip (bulk export).  Parses:
      - "RNA tissue specificity" column  → tau proxy via text label mapping

    Falls back gracefully if download fails or columns are absent.
    """
    print("=== Human Protein Atlas ===")
    result: dict[str, dict] = {g: {"tissue_specificity_tau": None} for g in genes}
    genes_set = set(genes)

    TAU_MAP = {
        "tissue enriched": 0.8,
        "group enriched": 0.7,
        "tissue enhanced": 0.6,
        "low tissue specificity": 0.2,
        # "not detected" is absent from the map — it means no tissue data, not tau=0.
        # Leaving it unmapped returns None, which the _missing indicator will flag correctly.
    }

    ok = _download_file(HPA_PROTEINATLAS_URL, HPA_PROTEINATLAS_CACHE, force=force)
    if not ok:
        print(
            "WARNING: HPA download failed — tissue features will be None for all genes"
        )
        return result

    try:
        import zipfile

        with zipfile.ZipFile(HPA_PROTEINATLAS_CACHE, "r") as zf:
            # Find the TSV inside the zip
            tsv_names = [n for n in zf.namelist() if n.endswith(".tsv")]
            if not tsv_names:
                print(f"WARNING: HPA zip contains no TSV: {zf.namelist()}")
                return result
            tsv_name = tsv_names[0]
            print(f"  reading {tsv_name} from zip")
            with zf.open(tsv_name) as raw:
                f = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
                reader = csv.DictReader(f, delimiter="\t")
                fieldnames = reader.fieldnames or []
                print(f"  HPA columns ({len(fieldnames)}): {fieldnames[:10]}")

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
                    print(
                        f"WARNING: HPA: gene column not found; header: {fieldnames[:15]}"
                    )
                    return result
                if col_tau_text is None:
                    print(
                        f"WARNING: HPA: 'RNA tissue specificity' column not found; tau will be None for all genes. Header: {fieldnames[:15]}"
                    )

                n_tau = 0
                for row in reader:
                    g = row.get(col_gene, "").strip()
                    if g not in genes_set:
                        continue
                    tau = None
                    if col_tau_text:
                        label = row.get(col_tau_text, "").strip().lower()
                        tau = TAU_MAP.get(label)
                    if tau is not None and result[g]["tissue_specificity_tau"] is None:
                        result[g]["tissue_specificity_tau"] = tau
                        n_tau += 1

                print(f"  tau: {n_tau} genes assigned")
    except Exception as e:
        print(f"WARNING: HPA parse failed: {e}")

    n_tau = sum(1 for v in result.values() if v["tissue_specificity_tau"] is not None)
    print(
        f"  HPA coverage: tau={n_tau}/{len(genes)} (n_tissues_expressed not available from bulk export)"
    )
    return result


# ---------------------------------------------------------------------------
# Source 4 — PaxDb protein abundance
# ---------------------------------------------------------------------------
def get_paxdb_abundance(
    genes: list[str], force: bool = False
) -> dict[str, Optional[float]]:
    """
    Returns {gene: log10(abundance_ppm)} or {gene: None}.
    Genes with abundance_ppm < 1e-3 (below detection floor) are returned as None.

    The manually placed file (data/downloads/9606-WHOLE_ORGANISM-integrated.txt)
    has been pre-processed to: gene_symbol\tstring_id\tabundance_ppm.
    This differs from the raw PaxDb download which has numeric internal IDs in col 0.

    File resolution order:
      1. PAXDB_MANUAL (data/9606-WHOLE_ORGANISM-integrated.txt) — manually placed
      2. PAXDB_CACHE   (data/cache/proteome_features/...)        — previously downloaded
      3. HTTP download (blocked by PaxDb as of 2026-05)
    """
    print("=== PaxDb abundance ===")
    result: dict[str, Optional[float]] = {g: None for g in genes}
    genes_set = set(genes)

    # Resolve file path
    paxdb_path = None
    if PAXDB_MANUAL.exists() and PAXDB_MANUAL.stat().st_size > 1000:
        paxdb_path = PAXDB_MANUAL
        print(f"  using manually placed file: {paxdb_path}")
    elif PAXDB_CACHE.exists() and PAXDB_CACHE.stat().st_size > 1000 and not force:
        paxdb_path = PAXDB_CACHE
        print(f"  using cached file: {paxdb_path}")
    else:
        ok = _download_file(PAXDB_URL, PAXDB_CACHE, force=force)
        if ok:
            paxdb_path = PAXDB_CACHE
        else:
            print(
                "WARNING: PaxDb not available — log_abundance_ppm will be None for all genes"
            )
            return result

    try:
        with open(paxdb_path) as f:
            header_skipped = False
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if not header_skipped:
                    header_skipped = True
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                gene = parts[0].strip()
                abund = _fnum(parts[2])
                if (
                    gene in genes_set
                    and abund is not None
                    and abund >= 1e-3
                    and result[gene] is None
                ):
                    result[gene] = math.log10(abund)
        n_covered = sum(1 for v in result.values() if v is not None)
        print(
            f"  PaxDb coverage: {n_covered}/{len(genes)} genes assigned log_abundance_ppm"
        )
        return result
    except Exception as e:
        print(f"WARNING: PaxDb parse failed: {e}")
        return result


# ---------------------------------------------------------------------------
# Source 5 — BioPlex 3.0 PPI degree
# ---------------------------------------------------------------------------
def get_bioplex_degree(
    genes: list[str], force: bool = False
) -> dict[str, Optional[int]]:
    """
    Returns {gene: degree} (number of unique interaction partners).
    Handles TSV with GeneA/GeneB columns (gene symbols).
    """
    print("=== BioPlex 3.0 PPI ===")
    result: dict[str, Optional[int]] = {g: None for g in genes}
    genes_set = set(genes)

    ok = _download_file(BIOPLEX_URL, BIOPLEX_CACHE, force=force)
    if not ok:
        print(
            "WARNING: BioPlex download failed — PPI_degree will be None for all genes"
        )
        return result

    try:
        degree: dict[str, set[str]] = {}
        with open(BIOPLEX_CACHE) as f:
            reader = csv.DictReader(f, delimiter="\t")
            fieldnames = reader.fieldnames or []
            # Strict exact-name match only. Substring fallbacks ("gene" in name)
            # are unsafe — they can pair an ID column with a description column
            # and silently produce garbage degrees.
            col_a = None
            col_b = None
            for name in fieldnames:
                nl = name.lower().strip()
                if nl in (
                    "genea",
                    "gene_a",
                    "symbola",
                    "symbol_a",
                    "gene a",
                    "symbol a",
                ):
                    col_a = name
                elif nl in (
                    "geneb",
                    "gene_b",
                    "symbolb",
                    "symbol_b",
                    "gene b",
                    "symbol b",
                ):
                    col_b = name
            if col_a is None or col_b is None:
                print(
                    f"WARNING: BioPlex: expected gene-symbol columns not found. "
                    f"Header: {fieldnames}; skipping PPI degree."
                )
                return result

            for row in reader:
                ga = row.get(col_a, "").strip()
                gb = row.get(col_b, "").strip()
                if not ga or not gb:
                    continue
                # Count degree for all genes (not just our universe) then filter
                degree.setdefault(ga, set()).add(gb)
                degree.setdefault(gb, set()).add(ga)

        # Sanity check: verify matched column values look like HGNC gene symbols,
        # not database IDs (ENSG*, 6-char UniProt). Sample the first 50 keys from
        # the degree dict — they come directly from the matched columns.
        _ENSG_RE = re.compile(r"^ENSG\d{8,}$")
        _UNIPROT_RE = re.compile(r"^[A-NR-Z]\d[A-Z\d]{3}\d$|^[OPQ]\d[A-Z\d]{3}\d$")
        _sample = list(itertools.islice(degree, 50))
        _n_db_ids = sum(1 for k in _sample if _ENSG_RE.match(k) or _UNIPROT_RE.match(k))
        if degree and _n_db_ids > len(_sample) * 0.5:
            print(
                f"WARNING: BioPlex: column values look like database IDs, not HGNC symbols "
                f"(sample: {_sample[:5]}). Discarding to avoid fabricated degrees."
            )
            return {g: None for g in genes}

        n_hit = 0
        for gene in genes_set:
            if gene in degree:
                result[gene] = len(degree[gene])
                n_hit += 1

        print(f"  BioPlex coverage: {n_hit}/{len(genes)} genes with PPI_degree")
    except Exception as e:
        print(f"WARNING: BioPlex parse failed: {e}")

    return result


# ---------------------------------------------------------------------------
# Source 6 — GeneBayes s_het (Zeng et al. 2023)
# ---------------------------------------------------------------------------
def _load_ensg_to_symbol() -> dict[str, str]:
    """
    Returns {ensg: gene_symbol} built from gnomAD v2.1.1 (which uses proper
    ENSG IDs in its gene_id column, unlike v4.1 which uses bare integers).
    Result is cached to GNOMAD_V2_ENSG_CACHE as JSON to avoid re-parsing.
    Reads from GNOMAD_V2_MANUAL (manually placed bgz file).
    """
    if GNOMAD_V2_ENSG_CACHE.exists() and GNOMAD_V2_ENSG_CACHE.stat().st_size > 1000:
        print(f"  ensg→symbol map: loading from cache {GNOMAD_V2_ENSG_CACHE}")
        try:
            with open(GNOMAD_V2_ENSG_CACHE) as f:
                return json.load(f)
        except Exception as e:
            print(f"WARNING: corrupt ENSG cache ({e}) — deleting and re-parsing")
            GNOMAD_V2_ENSG_CACHE.unlink()

    if not GNOMAD_V2_MANUAL.exists() or GNOMAD_V2_MANUAL.stat().st_size < 1000:
        print(
            f"WARNING: {GNOMAD_V2_MANUAL} not found — s_het will be None for all genes.\n"
            "  Download gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz from gnomAD release 2.1.1 "
            "and place it at that path."
        )
        return {}

    print(f"  ensg→symbol map: parsing {GNOMAD_V2_MANUAL}")
    mapping: dict[str, str] = {}
    try:
        with gzip.open(GNOMAD_V2_MANUAL, "rb") as gz:
            header = gz.readline().decode().strip().split("\t")
            idx_gene = header.index("gene") if "gene" in header else None
            idx_ensg = header.index("gene_id") if "gene_id" in header else None
            if idx_gene is None or idx_ensg is None:
                print(
                    f"WARNING: gnomAD v2.1.1 missing gene/gene_id columns: {header[:10]}"
                )
                return {}
            needed = max(idx_gene, idx_ensg)
            for line in gz:
                parts = line.decode().strip().split("\t")
                if len(parts) <= needed:
                    continue
                ensg = parts[idx_ensg].strip().split(".")[0]
                gene = parts[idx_gene].strip()
                if ensg and gene and ensg not in mapping:
                    mapping[ensg] = gene
        GNOMAD_V2_ENSG_CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = GNOMAD_V2_ENSG_CACHE.with_suffix(".tmp")
        tmp.write_text(json.dumps(mapping))
        os.replace(tmp, GNOMAD_V2_ENSG_CACHE)
        print(f"  ensg→symbol map: {len(mapping)} entries cached")
    except Exception as e:
        print(f"WARNING: could not build ensg→symbol map: {e}")
    return mapping


def get_shet(genes: list[str]) -> dict[str, Optional[float]]:
    """
    Returns {gene: s_het_post_mean} or {gene: None}.

    Reads data/downloads/s_het_estimates.genebayes.tsv (manually placed).
    Joins on Ensembl gene ID via the gnomAD v2.1.1 ensg→symbol map.
    """
    print("=== GeneBayes s_het ===")
    result: dict[str, Optional[float]] = {g: None for g in genes}

    if not SHET_MANUAL.exists() or SHET_MANUAL.stat().st_size < 100:
        print(
            f"WARNING: {SHET_MANUAL} not found — s_het will be None for all genes.\n"
            "  Download from https://doi.org/10.5281/zenodo.7939767 and place at that path."
        )
        return result

    ensg_to_gene = _load_ensg_to_symbol()
    if not ensg_to_gene:
        print("WARNING: ensg→symbol map empty — s_het will be None for all genes")
        return result

    genes_set = set(genes)
    try:
        with open(SHET_MANUAL) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                ensg = row.get("ensg", "").strip().split(".")[0]
                gene = ensg_to_gene.get(ensg)
                if gene is None or gene not in genes_set:
                    continue
                val = _fnum(row.get("post_mean", ""))
                if val is not None and result[gene] is None:
                    result[gene] = val
    except Exception as e:
        print(f"WARNING: s_het parse failed: {e}")
        return result

    n_covered = sum(1 for v in result.values() if v is not None)
    print(f"  s_het coverage: {n_covered}/{len(genes)} genes assigned")
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
    "tissue_specificity_tau",  # HPA text label mapped to {0, 0.2, 0.6, 0.7, 0.8}
    # n_tissues_expressed: dropped — not in HPA bulk export (0% coverage)
    "log_abundance_ppm",  # PaxDb integrated human, manually downloaded
    "PPI_degree",
    "s_het",  # GeneBayes posterior mean (Zeng et al. 2023); replaces ClinGen HI/TS
]


def build_feature_table(
    genes: list[str],
    families: dict[str, Optional[str]],
    gnomad: dict[str, dict],
    paralogs: dict[str, Optional[int]],
    hpa: dict[str, dict],
    paxdb: dict[str, Optional[float]],
    bioplex: dict[str, Optional[int]],
    shet: dict[str, Optional[float]],
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
        # GeneBayes s_het
        row["s_het"] = shet.get(gene)
        rows.append(row)

    # --- Family-mean-centred residuals ---
    # Group genes by family; compute family mean for each continuous feature
    family_groups: dict[str, list[int]] = {}  # family → list of row indices
    for i, row in enumerate(rows):
        fam = row["pfam_family"]
        if fam:
            family_groups.setdefault(fam, []).append(i)

    singleton_families: set[str] = {
        fam for fam, idxs in family_groups.items() if len(idxs) == 1
    }

    for feat in CONT_FEATURES:
        # Family means: {family: mean_value_or_None}. Require ≥2 observed members
        # so a single observed gene doesn't produce a zero residual that's
        # indistinguishable from "no signal" downstream.
        fam_means: dict[str, Optional[float]] = {}
        for fam, idxs in family_groups.items():
            vals = [rows[i][feat] for i in idxs if rows[i][feat] is not None]
            fam_means[fam] = float(np.mean(vals)) if len(vals) >= 2 else None

        for row in rows:
            raw = row[feat]
            fam = row["pfam_family"]
            if fam and fam_means.get(fam) is not None and raw is not None:
                row[f"{feat}_familyresid"] = raw - fam_means[fam]
            else:
                # Singletons, families with <2 observed members, no-family genes,
                # or missing raw values: residual is undefined — leave None (not 0.0)
                # so _familyresid_missing flags it as missing downstream.
                row[f"{feat}_familyresid"] = None

    # --- is_singleton_family indicator ---
    for row in rows:
        row["is_singleton_family"] = (
            1 if row["pfam_family"] in singleton_families else 0
        )

    # --- Missingness indicators ---
    for feat in CONT_FEATURES:
        for row in rows:
            row[f"{feat}_missing"] = 0 if row[feat] is not None else 1
        for row in rows:
            row[f"{feat}_familyresid_missing"] = (
                0 if row.get(f"{feat}_familyresid") is not None else 1
            )

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
    print(f"Wrote TSV: {path} ({len(rows)} rows × {len(col_names)} cols)")


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

    # Median imputation column-wise. Fully-missing columns (source failed entirely)
    # are left as NaN — imputing them would fabricate values with no basis.
    for j in range(X.shape[1]):
        col_data = X[:, j]
        nan_mask = np.isnan(col_data)
        if nan_mask.any():
            finite = col_data[~nan_mask]
            if finite.size == 0:
                print(
                    f"WARNING: column '{num_cols[j]}' has no observed values — "
                    f"source likely failed; leaving as NaN in matrix."
                )
                continue
            X[nan_mask, j] = float(np.median(finite))

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
        "GeneBayes (s_het)": "s_het",
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
    args = parser.parse_args([])
    force = args.force_redownload

    if not GENE_UNIVERSE.exists():
        raise FileNotFoundError(
            f"Required input not found: {GENE_UNIVERSE}\n"
            "  Run: python -m esm2_mechanism.fetch_data.build_gene_universe --step universe"
        )

    for d in (CACHE_DIR, CACHE_DIR / "paralogs"):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("build_proteome_features.py — Experiment 11 Phase 1+2")
    print("=" * 60)

    # 1. Gene universe + Pfam families (gene_universe.tsv already filtered to Pfam-annotated genes)
    genes, families = load_gene_universe(GENE_UNIVERSE)
    print(f"Proceeding with {len(genes)} genes")

    # 3. Sources — each wrapped in try/except so one failure doesn't abort
    gnomad: dict[str, dict] = {}
    try:
        gnomad = get_gnomad_constraint(force=force)
    except Exception as e:
        print(f"WARNING: gnomAD source failed entirely: {e} — using empty dict")

    paralogs: dict[str, Optional[int]] = {g: None for g in genes}
    try:
        paralogs = get_paralogs(genes)
    except Exception as e:
        print(
            f"WARNING: Paralog source failed entirely: {e} — using None for all genes"
        )

    hpa: dict[str, dict] = {g: {"tissue_specificity_tau": None} for g in genes}
    try:
        hpa = get_hpa_features(genes, force=force)
    except Exception as e:
        print(f"WARNING: HPA source failed entirely: {e} — using None for all genes")

    paxdb: dict[str, Optional[float]] = {g: None for g in genes}
    try:
        paxdb = get_paxdb_abundance(genes, force=force)
    except Exception as e:
        print(f"WARNING: PaxDb source failed entirely: {e} — using None for all genes")

    bioplex: dict[str, Optional[int]] = {g: None for g in genes}
    try:
        bioplex = get_bioplex_degree(genes, force=force)
    except Exception as e:
        print(
            f"WARNING: BioPlex source failed entirely: {e} — using None for all genes"
        )

    shet: dict[str, Optional[float]] = {g: None for g in genes}
    try:
        shet = get_shet(genes)
    except Exception as e:
        print(f"WARNING: s_het source failed entirely: {e} — using None for all genes")

    # 4. Phase 2: build feature table
    print("=== Phase 2: building feature table ===")
    rows, col_names = build_feature_table(
        genes, families, gnomad, paralogs, hpa, paxdb, bioplex, shet
    )

    # 5. Save TSV (raw values, NaN as empty string)
    save_tsv(rows, col_names, OUT_TSV)

    # 6. Build aligned numpy matrix (median-imputed)
    print("=== Building aligned numpy matrix ===")
    X, num_cols = build_aligned_matrix(rows, col_names)
    np.save(OUT_NPY, X)
    print(f"Wrote numpy matrix: {OUT_NPY} shape={X.shape} dtype={X.dtype}")

    # 7. Save column metadata
    col_metadata = {
        "all_columns": col_names,
        "numerical_columns": num_cols,
        "continuous_features": CONT_FEATURES,
        "n_genes": len(genes),
        "n_numerical_cols": len(num_cols),
        "notes": {
            "gene_order": "Same as gene_universe.tsv row order",
            "imputation": "Median imputation applied only to .npy; raw NaN kept in TSV",
            "residuals": "family-mean-centred, named <feat>_familyresid",
            "missingness": "Binary <feat>_missing indicators included",
            "scaling": "NOT applied here — fit scaler on training fold during modelling",
        },
    }
    OUT_COLS.write_text(json.dumps(col_metadata, indent=2))
    print(f"Wrote column metadata: {OUT_COLS}")

    # 8. Coverage summary
    print_coverage_summary(rows, genes)

    print("Done.")


if __name__ == "__main__":
    main()
