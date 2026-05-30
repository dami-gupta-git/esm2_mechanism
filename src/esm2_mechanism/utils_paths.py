import functools
import sys
from pathlib import Path
print = functools.partial(print, flush=True)

PACKAGE_ROOT = Path(__file__).parent.resolve()  # src/esm2_mechanism/
PROJECT_ROOT = PACKAGE_ROOT.parent.parent  # esm2_mechanism/

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
PAPERS_DIR = PROJECT_ROOT / "papers"
DOCS_DIR = PROJECT_ROOT / "docs"

# Canonical model identifiers.
ESM2_MODEL = "esm2_t33_650M_UR50D"
ESM3_MODEL = "esm3-sm-open-v1"

# Per-model embedding directories.
EMB_DIR = DATA_DIR / "embeddings" / ESM2_MODEL
ESM3_EMB_DIR = DATA_DIR / "embeddings" / ESM3_MODEL

# Top-level data files written by the fetch pipeline.
VARIANTS_JSON = DATA_DIR / "variants.json"
GENE_LIST_TSV = DATA_DIR / "gene_list.tsv"

# ── ESM-2 Gerasimavicius embeddings (embed_variants.py) ──────────────────────
VALID_VARIANTS_JSON = EMB_DIR / "valid_variants.json"
EMB_WT_MEAN = EMB_DIR / "embeddings_wt_mean.npy"
EMB_MUT_MEAN = EMB_DIR / "embeddings_mut_mean.npy"
EMB_WT_POS = EMB_DIR / "embeddings_wt_pos.npy"
EMB_MUT_POS = EMB_DIR / "embeddings_mut_pos.npy"

# ── Pathogenicity control embeddings (pathogenicity_control.py) ───────────────
PATH_EMB_WT_MEAN = EMB_DIR / "pathogenicity_wt_mean.npy"
PATH_EMB_MUT_MEAN = EMB_DIR / "pathogenicity_mut_mean.npy"
PATH_EMB_META = EMB_DIR / "pathogenicity_meta.json"

# ── Megascale S1724 embeddings (megascale_stability.py) ──────────────────────
MEGASCALE_EMB_WT_MEAN = EMB_DIR / "megascale_wt_mean.npy"
MEGASCALE_EMB_MUT_MEAN = EMB_DIR / "megascale_mut_mean.npy"
MEGASCALE_EMB_WT_POS = EMB_DIR / "megascale_wt_pos.npy"
MEGASCALE_EMB_MUT_POS = EMB_DIR / "megascale_mut_pos.npy"
MEGASCALE_DELTAS = EMB_DIR / "megascale_deltas.npy"
MEGASCALE_DDG = EMB_DIR / "megascale_ddg.npy"

# ── Stability subspace (esm2_mechanism.py) ───────────────────────────────────
STABILITY_SUBSPACE = EMB_DIR / "stability_subspace.npy"

# ── Perturbation scan embeddings (perturbation_scan.py) ──────────────────────
SCAN_EMB_WT = EMB_DIR / "scan_wt.npy"
SCAN_EMB_MUT = EMB_DIR / "scan_mut.npy"
SCAN_CKPT_WT = EMB_DIR / "scan_ckpt_wt.npy"
SCAN_CKPT_MUT = EMB_DIR / "scan_ckpt_mut.npy"

# ── ESM-3 embeddings (esm3_mechanism.py) ─────────────────────────────────────
ESM3_EMB_SEQ = ESM3_EMB_DIR / "seq_mean.npy"
ESM3_EMB_SEQ_STRUCT = ESM3_EMB_DIR / "seq_struct_mean.npy"
ESM3_VALID_IDX = ESM3_EMB_DIR / "valid_idx.npy"
ESM3_STRUCT_META = ESM3_EMB_DIR / "struct_meta.json"

DOWNLOADS_DIR = DATA_DIR / "downloads"

# Manually-placed prerequisite files — must exist in DOWNLOADS_DIR before the pipeline starts.
DISEASE_MECH_STABILITY_VEPS_FILE = DOWNLOADS_DIR / "DiseaseMech_Stability_VEPS.xlsx"
ALL_G2P_FILE = DOWNLOADS_DIR / "AllG2P.csv"
TABLE_S3_FILE = DOWNLOADS_DIR / "table_S3.xlsx"
PAXDB_FILE = DOWNLOADS_DIR / "9606-WHOLE_ORGANISM-integrated.txt"
S_HET_FILE = DOWNLOADS_DIR / "s_het_estimates.genebayes.tsv"
GNOMAD_LOF_FILE = DOWNLOADS_DIR / "gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz"

PREREQUISITE_FILES = [
    DISEASE_MECH_STABILITY_VEPS_FILE,
    ALL_G2P_FILE,
    TABLE_S3_FILE,
    PAXDB_FILE,
    S_HET_FILE,
    GNOMAD_LOF_FILE,
]


def check_prerequisites() -> None:
    """Verify all manually-placed prerequisite files exist in DOWNLOADS_DIR.

    Logs each missing file and exits with a non-zero status if any are absent.
    """
    missing = [path for path in PREREQUISITE_FILES if not path.exists()]
    if missing:
        for path in missing:
           print("Missing prerequisite file: %s", path)
        return False
    print("All %d prerequisite files present.", len(PREREQUISITE_FILES))
    return True
