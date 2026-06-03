import functools
from pathlib import Path

from esm2_mech.utils.constants import ESM2_MODEL, ESM3_MODEL, GENE_UNIVERSE_FILENAME, PFAM_FAMILIES_FILENAME

print = functools.partial(print, flush=True)

PACKAGE_ROOT = Path(__file__).parent.parent.resolve()  # src/esm2_mech/
PROJECT_ROOT = PACKAGE_ROOT.parent.parent              # esm2_mechanism/

RUN_NAME="run6"

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / RUN_NAME
REPORTS_DIR = PROJECT_ROOT / "reports"
RUN_REPORTS_DIR = REPORTS_DIR / RUN_NAME
FIGURES_DIR = RUN_REPORTS_DIR / "figures"

# ── Result files written under RESULTS_DIR ───────────────────────────────────
MECHANISM_AGGREGATE_JSON = RESULTS_DIR / "aggregate.json"
FAMILY_CLUSTERING_JSON = RESULTS_DIR / "family_clustering.json"
NAIVE_BASELINE_JSON = RESULTS_DIR / "naive_baseline.json"
PATHOGENICITY_CONTROL_JSON = RESULTS_DIR / "pathogenicity_control.json"
LEAKAGE_FRACTION_JSON = RESULTS_DIR / "leakage_fraction.json"
# Per-seed probe outputs, written as each seed completes (resume + progress).
# Format with .format(seed=N).
PATHOGENICITY_CONTROL_SEED_JSON = str(RESULTS_DIR / "pathogenicity_control_seed{seed}.json")
WITHIN_FAMILY_MECHANISM_JSON = RESULTS_DIR / "within_family_mechanism.json"
# Per-seed ESM-2 nonlinear-probe results (MLP/GBM/RF/kNN on delta features). Format
# with .format(seed=N). The ESM-3 experiment reads mlp_delta_mean_family from these to
# derive the matched ESM-2 family-split floor instead of hardcoding it.
NONLINEAR_RESULTS_SEED_JSON = str(RESULTS_DIR / "nonlinear_results_seed{seed}.json")

# ── Geometry experiments (magnitude/direction, result_23) ────────────────────
GEOMETRY_RESULTS_DIR = RESULTS_DIR / "magnitude_direction"
MAGNITUDE_DIRECTION_JSON = GEOMETRY_RESULTS_DIR / "probe_results.json"
DIRECTION_GEOMETRY_JSON = GEOMETRY_RESULTS_DIR / "geometry_results.json"
TRANSFER_CONTRAST_JSON = GEOMETRY_RESULTS_DIR / "transfer_contrast.json"
CONSERVATION_AXIS_JSON = GEOMETRY_RESULTS_DIR / "conservation_axis.json"
PROBE4_AXIS_IDENTITY_JSON = GEOMETRY_RESULTS_DIR / "probe4_axis_identity.json"

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
ALPHAMISSENSE_SCORES_JSON = DATA_DIR / "alphamissense_scores_full.json"
VALID_VARIANTS_JSON = DATA_DIR / "valid_variants.json"

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_DIR = DATA_DIR / "cache"
SEQUENCES_JSON = CACHE_DIR / "sequences.json"
SEQUENCES_NOT_FOUND_JSON = CACHE_DIR / "sequences_not_found.json"
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
# Sidecar recording the fetch params the cache above was built with, so a param
# change (cap / seed) re-fetches instead of silently serving the stale set.
CLINVAR_PATHOGENICITY_PARAMS_JSON = DATA_DIR / "clinvar_pathogenicity_variants.params.json"

# ── ESM-2 Gerasimavicius embeddings (embed_variants.py) ──────────────────────
# Row-aligned subset of variants actually embedded (one row per .npy array row).
# Distinct from VALID_VARIANTS_JSON (the pre-filter input written by build_valid_variants).
# Write-only sanity/provenance artifact — NO code reads it; always identical to
# VALID_VARIANTS_JSON because the embed step re-applies the same filters. See the
# _flush_checkpoint docstring in utils/embed.py for the full rationale.
EMB_VALID_VARIANTS_JSON = EMB_DIR / "embedded_variants.json"
EMB_WT_MEAN = EMB_DIR / "embeddings_wt_mean.npy"
EMB_MUT_MEAN = EMB_DIR / "embeddings_mut_mean.npy"
EMB_WT_POS = EMB_DIR / "embeddings_wt_pos.npy"
EMB_MUT_POS = EMB_DIR / "embeddings_mut_pos.npy"

# ── Pathogenicity control embeddings (embed_pathogenicity.py) ────────────────
PATH_EMB_WT_MEAN = EMB_DIR / "pathogenicity_wt_mean.npy"
PATH_EMB_MUT_MEAN = EMB_DIR / "pathogenicity_mut_mean.npy"
PATH_EMB_META = EMB_DIR / "pathogenicity_meta.json"

# ── Canonical pathogenicity set + masked-LL conservation (geometry experiments) ─
# The n=16,576 canonical pathogenicity variant set whose embeddings are PATH_EMB_*.
# Row-aligned to PATH_EMB_WT_MEAN / PATH_EMB_MUT_MEAN.
PATHOGENICITY_CANONICAL_VARIANTS_JSON = DATA_DIR / "pathogenicity_valid_variants_canonical.json"
# Masked-LM conservation readouts per canonical variant: [logP_wt, logP_mut, entropy].
CONSERVATION_PATHOGENICITY_NPY = DATA_DIR / "conservation_pathogenicity.npy"
CONSERVATION_PATHOGENICITY_META_JSON = DATA_DIR / "conservation_pathogenicity_meta.json"
# Megascale S1724 stability variants (Probe C / stability transfer).
MEGASCALE_VARIANTS_JSON = DATA_DIR / "megascale_variants.json"
# Source benchmark archive (S1724 + TED) and the cached protein→cluster map
# used for the family-split analogue.
MEGASCALE_BENCHMARKS_ZIP = DATA_DIR / "megascale" / "benchmarks.zip"
MEGASCALE_PROTEIN_CLUSTERS_JSON = DATA_DIR / "megascale_protein_clusters.json"

# ── Full Tsuboyama 2023 point-mutant stability dataset (scaled-up control) ────
# The processed point-mutant ΔΔG table: per row a mutant aa_seq, mut_type code,
# parent domain (WT_name) and ddG_ML label. Natural domains only (de novo designs
# are dropped). Parsed/cached to MEGASCALE_TSUBOYAMA_VARIANTS_JSON.
MEGASCALE_DOWNLOAD_DIR = DATA_DIR / "downloads" / "megascale"
MEGASCALE_TSUBOYAMA_CSV = (
    MEGASCALE_DOWNLOAD_DIR
    / "Processed_K50_dG_datasets"
    / "Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
)
MEGASCALE_TSUBOYAMA_VARIANTS_JSON = DATA_DIR / "megascale_tsuboyama_variants.json"
# Domain → Pfam family map (HMMER hmmscan against Pfam-A); domains with no Pfam
# hit are absent from this map and excluded from family-split only.
MEGASCALE_DOMAIN_FAMILIES_JSON = DATA_DIR / "megascale_domain_families.json"
# Pfam-A profile database (hmmpress-ed) used by the domain-family mapping step.
PFAM_A_HMM = MEGASCALE_DOWNLOAD_DIR / "Pfam-A.hmm"

# ── Megascale stability embeddings (megascale_stability.py) ──────────────────
MEGASCALE_EMB_WT_MEAN = EMB_DIR / "megascale_wt_mean.npy"
MEGASCALE_EMB_MUT_MEAN = EMB_DIR / "megascale_mut_mean.npy"
MEGASCALE_EMB_WT_POS = EMB_DIR / "megascale_wt_pos.npy"
MEGASCALE_EMB_MUT_POS = EMB_DIR / "megascale_mut_pos.npy"
MEGASCALE_DELTAS = EMB_DIR / "megascale_deltas.npy"
MEGASCALE_DDG = EMB_DIR / "megascale_ddg.npy"
# Checkpoint subdir for the megascale embedding job. The shared embed helper
# writes fixed names (embeddings_{wt,mut}_{mean,pos}.npy); isolating them here
# avoids colliding with the mechanism embeddings in EMB_DIR. On completion the
# driver promotes them to the MEGASCALE_EMB_* names above.
MEGASCALE_EMB_CKPT_DIR = EMB_DIR / "megascale_ckpt"

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
