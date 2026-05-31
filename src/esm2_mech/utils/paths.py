import functools
from pathlib import Path

from esm2_mech.utils.constants import ESM2_MODEL, ESM3_MODEL, GENE_UNIVERSE_FILENAME, PFAM_FAMILIES_FILENAME

print = functools.partial(print, flush=True)

PACKAGE_ROOT = Path(__file__).parent.parent.resolve()  # src/esm2_mech/
PROJECT_ROOT = PACKAGE_ROOT.parent.parent              # esm2_mechanism/

RUN_NAME="run2"

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / RUN_NAME
REPORTS_DIR = PROJECT_ROOT / "reports"
PAPERS_DIR = PROJECT_ROOT / "papers"
DOCS_DIR = PROJECT_ROOT / "docs"

# ── Per-model embedding directories ──────────────────────────────────────────
EMB_DIR = DATA_DIR / "embeddings" / ESM2_MODEL
ESM3_EMB_DIR = DATA_DIR / "embeddings" / ESM3_MODEL

# ── Top-level data files written by the fetch pipeline ───────────────────────
GENE_UNIVERSE = DATA_DIR / GENE_UNIVERSE_FILENAME
VARIANTS_JSON = DATA_DIR / "variants.json"
GENE_LIST_TSV = DATA_DIR / "gene_list.tsv"
PFAM_JSON = DATA_DIR / "pfam_families.json"
GERAS_VALID_VARIANTS_JSON = DATA_DIR / "geras_valid_variants.json"
ALPHAMISSENSE_SCORES_JSON = DATA_DIR / "alphamissense_scores_full.json"
VALID_VARIANTS_JSON = DATA_DIR / "valid_variants.json"

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_DIR = DATA_DIR / "cache"
SEQUENCES_JSON = CACHE_DIR / "sequences.json"
SEQUENCES_EXTENDED_JSON = CACHE_DIR / "uniprot_sequences_extended.json"
AM_CACHE_FILE = CACHE_DIR / "AlphaMissense_aa_substitutions.tsv.gz"
BADONYI_CACHE_DIR = CACHE_DIR / "badonyi"
PROTEINGYM_CACHE_DIR = CACHE_DIR / "proteingym"
PROTEOME_FEATURES_CACHE_DIR = CACHE_DIR / "proteome_features"
PROTEOME_PILOT_CACHE_DIR = CACHE_DIR / "proteome_pilot"
LL_CKPT_JSON = CACHE_DIR / "ll_ckpt.json"
ENZYME_CACHE_DIR = CACHE_DIR / "enzyme_uniprot"
SCAN_PROBE_CACHE_JSON = CACHE_DIR / "scan_probes.json"
ESM3_STRUCT_TOKENS_JSON = CACHE_DIR / "esm3_struct_tokens.json"
CLINVAR_PATHOGENICITY_VARIANTS_JSON = DATA_DIR / "clinvar_pathogenicity_variants.json"

# ── ESM-2 Gerasimavicius embeddings (embed_variants.py) ──────────────────────
EMB_WT_MEAN = EMB_DIR / "embeddings_wt_mean.npy"
EMB_MUT_MEAN = EMB_DIR / "embeddings_mut_mean.npy"
EMB_WT_POS = EMB_DIR / "embeddings_wt_pos.npy"
EMB_MUT_POS = EMB_DIR / "embeddings_mut_pos.npy"

# ── Pathogenicity control embeddings (embed_pathogenicity.py) ────────────────
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

# ── Downloads (manually-placed prerequisite files) ───────────────────────────
DOWNLOADS_DIR = DATA_DIR / "downloads"

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


def check_prerequisites() -> bool:
    missing = [path for path in PREREQUISITE_FILES if not path.exists()]
    if missing:
        for path in missing:
            print(f"Missing prerequisite file: {path}")
        return False
    print(f"All {len(PREREQUISITE_FILES)} prerequisite files present.")
    return True
