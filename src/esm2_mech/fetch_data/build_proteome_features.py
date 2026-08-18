"""Collect gene-level proteome features for all genes in gene_universe.tsv."""

from __future__ import annotations

import argparse
import csv
import functools
import gzip
import io
import itertools
import json
import math
import re
import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

from esm2_mech.utils.data import load_gene_universe
from esm2_mech.utils.paths import (
    DATA_DIR,
    GENE_UNIVERSE,
    PROTEOME_FEATURES_ALIGNED,
    PROTEOME_FEATURES_TSV,
    PROTEOME_FEATURE_COLUMNS_JSON,
    GNOMAD_LOF_FILE,
    PAXDB_FILE,
    PROTEOME_FEATURES_CACHE_DIR,
    PROTEOME_PILOT_CACHE_DIR,
    S_HET_FILE,
)
from esm2_mech.utils.io import atomic_write_json, save_npy

print = functools.partial(print, flush=True)

CACHE_DIR = PROTEOME_FEATURES_CACHE_DIR
PILOT_CACHE_DIR = PROTEOME_PILOT_CACHE_DIR
PILOT_PARALOG_CACHE = PILOT_CACHE_DIR / "paralogs"

OUT_TSV = PROTEOME_FEATURES_TSV
OUT_NPY = PROTEOME_FEATURES_ALIGNED
OUT_COLS = PROTEOME_FEATURE_COLUMNS_JSON

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
PAXDB_MANUAL = PAXDB_FILE

BIOPLEX_URL = (
    "https://bioplex.hms.harvard.edu/data/BioPlex_293T_Network_10K_Dec_2019.tsv"
)
BIOPLEX_CACHE = CACHE_DIR / "BioPlex_293T_Network_10K.tsv"

SHET_MANUAL = S_HET_FILE

GNOMAD_V2_MANUAL = GNOMAD_LOF_FILE
GNOMAD_V2_ENSG_CACHE = CACHE_DIR / "gnomad_v2.1.1_ensg_symbol.json"


def _download_file(url: str, dest: Path, force: bool = False) -> bool:
    """Download url to dest; returns True on success."""
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


def get_gnomad_constraint(force: bool = False) -> dict[str, dict]:
    """Return {gene: {pLI, LOEUF, mis_z}}."""
    print("=== gnomAD v4.1 constraint ===")

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


def _load_paralog_cache(cache_file: Path) -> tuple[bool, Optional[int]]:
    """Read a paralog cache file; returns (is_usable, paralog_count)."""
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
    """Single REST call; caches to own_cache_dir/{gene}.json."""
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
        # Empty "data" is ambiguous (no paralogs vs transient hiccup); write error-tagged sentinel to retry.
        entries = data.get("data")
        if not entries:
            err = "empty data payload (HTTP 200 but no entries)"
            print(f"  paralog fetch ambiguous for {gene}: {err}")
            atomic_write_json(cache_file, {"paralog_count": None, "error": err})
            return None
        homologies: list = []
        for entry in entries:
            homologies.extend(entry.get("homologies", []))
        count = sum(1 for h in homologies if "paralog" in h.get("type", "").lower())
        atomic_write_json(cache_file, {"paralog_count": count})
        return count
    except Exception as e:
        print(
            f"  paralog fetch failed for {gene}: {e} — not caching, will retry next run"
        )
        return None


def get_paralogs(genes: list[str]) -> dict[str, Optional[int]]:
    """Reuse pilot cache where available; fetch missing genes via REST at 10 req/s."""
    print("=== Ensembl Compara paralogs ===")
    own_cache = CACHE_DIR / "paralogs"
    out: dict[str, Optional[int]] = {}
    to_fetch: list[str] = []

    for gene in genes:
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


def get_hpa_features(genes: list[str], force: bool = False) -> dict[str, dict]:
    """Return {gene: {tissue_specificity_tau: float|None}} from HPA bulk export."""
    print("=== Human Protein Atlas ===")
    result: dict[str, dict] = {g: {"tissue_specificity_tau": None} for g in genes}
    genes_set = set(genes)

    TAU_MAP = {
        "tissue enriched": 0.8,
        "group enriched": 0.7,
        "tissue enhanced": 0.6,
        "low tissue specificity": 0.2,
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


def get_paxdb_abundance(
    genes: list[str], force: bool = False
) -> dict[str, Optional[float]]:
    """Return {gene: log10(abundance_ppm)} or {gene: None}."""
    print("=== PaxDb abundance ===")
    result: dict[str, Optional[float]] = {g: None for g in genes}
    genes_set = set(genes)

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


def get_bioplex_degree(
    genes: list[str], force: bool = False
) -> dict[str, Optional[int]]:
    """Return {gene: degree} (number of unique interaction partners)."""
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
                degree.setdefault(ga, set()).add(gb)
                degree.setdefault(gb, set()).add(ga)

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


def _load_ensg_to_symbol() -> dict[str, str]:
    """Return {ensg: gene_symbol} from gnomAD v2.1.1; cached to JSON."""
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
        atomic_write_json(GNOMAD_V2_ENSG_CACHE, mapping)
        print(f"  ensg→symbol map: {len(mapping)} entries cached")
    except Exception as e:
        print(f"WARNING: could not build ensg→symbol map: {e}")
    return mapping


def get_shet(genes: list[str]) -> dict[str, Optional[float]]:
    """Return {gene: s_het_post_mean} or {gene: None}."""
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


CONT_FEATURES = [
    "pLI",
    "LOEUF",
    "mis_z",
    "paralog_count",
    "tissue_specificity_tau",
    "log_abundance_ppm",
    "PPI_degree",
    "s_het",
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
    """Build per-gene feature rows with raw values, family residuals, and missingness indicators."""
    rows: list[dict] = []

    for gene in genes:
        family = families.get(gene)
        row: dict = {
            "gene": gene,
            "pfam_family": family if family is not None else "",
        }
        g = gnomad.get(gene, {})
        row["pLI"] = g.get("pLI")
        row["LOEUF"] = g.get("LOEUF")
        row["mis_z"] = g.get("mis_z")
        row["paralog_count"] = paralogs.get(gene)
        h = hpa.get(gene, {})
        row["tissue_specificity_tau"] = h.get("tissue_specificity_tau")
        row["log_abundance_ppm"] = paxdb.get(gene)
        row["PPI_degree"] = bioplex.get(gene)
        row["s_het"] = shet.get(gene)
        rows.append(row)

    family_groups: dict[str, list[int]] = {}  # family → list of row indices
    for i, row in enumerate(rows):
        fam = row["pfam_family"]
        if fam:
            family_groups.setdefault(fam, []).append(i)

    singleton_families: set[str] = {
        fam for fam, idxs in family_groups.items() if len(idxs) == 1
    }

    for feat in CONT_FEATURES:
        # Require >=2 observed members so a singleton doesn't produce a zero residual.
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
                row[f"{feat}_familyresid"] = None

    for row in rows:
        row["is_singleton_family"] = (
            1 if row["pfam_family"] in singleton_families else 0
        )

    for feat in CONT_FEATURES:
        for row in rows:
            row[f"{feat}_missing"] = 0 if row[feat] is not None else 1
        for row in rows:
            row[f"{feat}_familyresid_missing"] = (
                0 if row.get(f"{feat}_familyresid") is not None else 1
            )

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
    """Build float32 matrix from numerical columns; missing values left as NaN."""
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

    empty_cols = []
    for j, col in enumerate(num_cols):
        n_obs = int((~np.isnan(X[:, j])).sum())
        if n_obs == 0:
            empty_cols.append(col)
    if empty_cols:
        print(
            f"WARNING: {len(empty_cols)} column(s) have NO observed values — the "
            f"source likely failed to download. They are left as NaN and will "
            f"contribute nothing to any probe: {', '.join(empty_cols)}"
        )

    return X.astype(np.float32), num_cols


def print_coverage_summary(rows: list[dict], genes: list[str]):
    """Print per-source coverage table."""
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

    if not GENE_UNIVERSE.exists():
        raise FileNotFoundError(
            f"Required input not found: {GENE_UNIVERSE}\n"
            "  Run: python -m esm2_mech.fetch_data.build_gene_universe --step universe"
        )

    for d in (CACHE_DIR, CACHE_DIR / "paralogs"):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("build_proteome_features.py — Experiment 11 Phase 1+2")
    print("=" * 60)

    genes, families = load_gene_universe(GENE_UNIVERSE)
    print(f"Proceeding with {len(genes)} genes")

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

    print("=== Phase 2: building feature table ===")
    rows, col_names = build_feature_table(
        genes, families, gnomad, paralogs, hpa, paxdb, bioplex, shet
    )

    save_tsv(rows, col_names, OUT_TSV)

    print("=== Building aligned numpy matrix ===")
    X, num_cols = build_aligned_matrix(rows, col_names)
    save_npy(OUT_NPY, X)
    print(f"Wrote numpy matrix: {OUT_NPY} shape={X.shape} dtype={X.dtype}")

    col_metadata = {
        "all_columns": col_names,
        "numerical_columns": num_cols,
        "continuous_features": CONT_FEATURES,
        "n_genes": len(genes),
        "n_numerical_cols": len(num_cols),
        "notes": {
            "gene_order": "Same as gene_universe.tsv row order",
            "imputation": "None — raw NaN kept in both .npy and TSV; consumers must restrict to the observed subset per feature and recompute CV splits",
            "residuals": "family-mean-centred, named <feat>_familyresid",
            "missingness": "Binary <feat>_missing indicators included",
            "scaling": "NOT applied here — fit scaler on training fold during modelling",
        },
    }
    atomic_write_json(OUT_COLS, col_metadata, indent=2)
    print(f"Wrote column metadata: {OUT_COLS}")

    print_coverage_summary(rows, genes)

    print("Done.")


if __name__ == "__main__":
    main()
