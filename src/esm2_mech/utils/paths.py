import functools
from pathlib import Path

from esm2_mech.utils.constants import ESM2_MODEL, ESM3_MODEL, GENE_UNIVERSE_FILENAME, PFAM_FAMILIES_FILENAME

print = functools.partial(print, flush=True)

def _find_project_root() -> Path:
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "pyproject.toml").exists():
            return p
        p = p.parent
    raise RuntimeError(
        "Cannot locate project root (no pyproject.toml found above "
        f"{Path(__file__).resolve()}). Is the package installed editable?"
    )

PROJECT_ROOT = _find_project_root()
PACKAGE_ROOT = PROJECT_ROOT / "src" / "esm2_mech"

RUN_NAME="run_biorxiv"

DATA_DIR = PROJECT_ROOT / "data"
ALL_RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR = ALL_RESULTS_DIR / RUN_NAME
REPORTS_DIR = PROJECT_ROOT / "reports"
RUN_REPORTS_DIR = REPORTS_DIR / RUN_NAME
FIGURES_DIR = RUN_REPORTS_DIR / "figures"

LOGS_DIR = PROJECT_ROOT / "logs" / "biorxiv_18Aug_2026"


def results_dir_for_run(run_name: str) -> Path:
    """The results directory of an arbitrary run, not necessarily the live one.

    Everything else keys off RUN_NAME; this exists for tools that read two runs at
    once (compare_runs.py diffing an old run against the new one), so they do not
    build a run directory inline.
    """
    return ALL_RESULTS_DIR / run_name

# ── LLM-judge mechanism eval (Langfuse-traced agent) ─────────────────────────
# An LLM-as-judge predicts label_3class per variant; predictions are scored
# against ground truth and every model call is traced to Langfuse. Lives in its
# own subdir so it never clobbers the probe seed files under RESULTS_DIR.
LLM_JUDGE_DIR = RESULTS_DIR / "llm_judge"
LLM_JUDGE_PREDICTIONS_JSONL = LLM_JUDGE_DIR / "predictions.jsonl"
LLM_JUDGE_SUMMARY_JSON = LLM_JUDGE_DIR / "summary.json"

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

# ── Enzyme classification (positive control for the mechanism arc) ───────────
ENZYME_RESULTS_DIR = RESULTS_DIR / "enzyme_classification"
ENZYME_CLASSIFICATION_JSON = ENZYME_RESULTS_DIR / "enzyme_classification_summary.json"
# Per-seed ESM-2 nonlinear-probe results (MLP/GBM/RF/kNN on delta features). Format
# with .format(seed=N). The ESM-3 experiment reads mlp_delta_mean_family from these to
# derive the matched ESM-2 family-split floor instead of hardcoding it.
NONLINEAR_RESULTS_SEED_JSON = str(RESULTS_DIR / "nonlinear_results_seed{seed}.json")

# Per-seed contrastive metric-learning results live directly under RESULTS_DIR
# (= results/<run>); filenames come from constants.contrastive_seed_result_filename.
# contrastive_mechanism.main pools them into CONTRASTIVE_AGGREGATE_JSON.
CONTRASTIVE_RESULTS_DIR = RESULTS_DIR
CONTRASTIVE_AGGREGATE_JSON = RESULTS_DIR / "contrastive_aggregate.json"

# ── Homology-partition robustness panel (Task 2b) ─────────────────────────────
# Consolidates leave-one-clan-out and MMseqs2-cluster-holdout as first-class
# run_biorxiv deliverables alongside the default Pfam-family split. Lives in its
# own subdir so the panel JSON never clobbers the per-module seed files.
HOMOLOGY_PARTITION_PANEL_DIR = RESULTS_DIR / "homology_partition_panel"
HOMOLOGY_PARTITION_PANEL_JSON = HOMOLOGY_PARTITION_PANEL_DIR / "panel.json"
MMSEQS_CLUSTERS_JSON = DATA_DIR / "mmseqs_clusters.json"
# PFAM_CLANS_TSV_GZ is defined below, once DOWNLOADS_DIR exists.

# ── V1 multi-seed replication (multiseed_v1.py) ───────────────────────────────
V1_MULTISEED_DIR = RESULTS_DIR / "v1_multiseed"
# Seed-0 numbers are sourced from a frozen prior baseline run, not recomputed
# under the current RUN_NAME — hence the literal run directory rather than a
# RESULTS_DIR-derived path.
V1_MULTISEED_SEED0_DIR = PROJECT_ROOT / "results" / "20260524_baseline_run" / "run_0"

# ── Single-source robustness check (Gerasimavicius-only mechanism re-run) ─────
# Re-runs the gene/family-split mechanism probe on the single-source subset to
# confirm the mechanism null is not a curation/source confound. Lives in its own
# subdir so it never clobbers the merged-dataset seed files written to RESULTS_DIR.
SINGLE_SOURCE_DIR = RESULTS_DIR / "single_source_gerasimavicius"
SINGLE_SOURCE_AGGREGATE_JSON = SINGLE_SOURCE_DIR / "aggregate.json"
SINGLE_SOURCE_NAIVE_BASELINE_JSON = SINGLE_SOURCE_DIR / "naive_baseline.json"

# ── WT identity sensitivity (proteins that do not require windowing) ────────
WT_IDENTITY_SENSITIVITY_DIR = RESULTS_DIR / "wt_identity_short_proteins"
WT_IDENTITY_SENSITIVITY_AGGREGATE_JSON = WT_IDENTITY_SENSITIVITY_DIR / "aggregate.json"

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
# GENE_UNIVERSE is the CANONICAL row order for every aligned feature matrix
# below — build_gene_universe.py filters GENE_LIST_TSV to Pfam-annotated genes,
# so the two files have different lengths AND different orderings. Anything that
# indexes an *_FEATURES_ALIGNED matrix must key off GENE_UNIVERSE; GENE_LIST_TSV
# is the unfiltered gene/mechanism table and is a row index for nothing.
GENE_UNIVERSE = DATA_DIR / GENE_UNIVERSE_FILENAME
VARIANTS_JSON = DATA_DIR / "variants.json"
GENE_LIST_TSV = DATA_DIR / "gene_list.tsv"
PFAM_JSON = DATA_DIR / "pfam_families.json"
ALPHAMISSENSE_SCORES_JSON = DATA_DIR / "alphamissense_scores_full.json"
VALID_VARIANTS_JSON = DATA_DIR / "valid_variants.json"

# ── Gene-level feature matrices (row-aligned to GENE_UNIVERSE) ───────────────
# The .npy matrices carry raw NaN for missing cells — nothing is imputed. See
# build_proteome_features.build_aligned_matrix for why.
PROTEOME_FEATURES_TSV = DATA_DIR / "gene_proteome_features.tsv"
PROTEOME_FEATURES_ALIGNED = DATA_DIR / "proteome_features_aligned.npy"
PROTEOME_FEATURE_COLUMNS_JSON = DATA_DIR / "proteome_feature_columns.json"
BADONYI_FEATURES_TSV = DATA_DIR / "badonyi_features.tsv"
BADONYI_FEATURES_ALIGNED = DATA_DIR / "badonyi_features_aligned.npy"
BADONYI_FEATURE_COLUMNS_JSON = DATA_DIR / "badonyi_feature_columns.json"
# Gene → enzyme 4-class labels (fetch_annotations --step enzyme). Keyed on gene
# symbol, not row-aligned to anything.
ENZYME_LABELS_TSV = DATA_DIR / "enzyme_labels.tsv"

# ── Perturbation scan features (perturbation_scan.py phase 3) ────────────────
# Row order is pinned by the "genes" list inside SCAN_FEATURES_META_JSON, not by
# GENE_UNIVERSE.
SCAN_FEATURES_NPY = DATA_DIR / "scan_features.npy"
SCAN_FEATURES_META_JSON = DATA_DIR / "scan_features_meta.json"

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
# Required row-identity sidecar. Every mechanism embedding loader compares it to
# the current variant list before using the arrays.
EMB_VALID_VARIANTS_JSON = EMB_DIR / "embedded_variants.json"
EMB_WT_MEAN = EMB_DIR / "embeddings_wt_mean.npy"
EMB_MUT_MEAN = EMB_DIR / "embeddings_mut_mean.npy"
EMB_WT_POS = EMB_DIR / "embeddings_wt_pos.npy"
EMB_MUT_POS = EMB_DIR / "embeddings_mut_pos.npy"

# ── Pathogenicity control embeddings (embed_pathogenicity.py) ────────────────
PATH_EMB_WT_MEAN = EMB_DIR / "pathogenicity_wt_mean.npy"
PATH_EMB_MUT_MEAN = EMB_DIR / "pathogenicity_mut_mean.npy"
PATH_EMB_META = EMB_DIR / "pathogenicity_meta.json"

# Row-count-validated pathogenicity variant list (dropped rows beyond the
# embedding count trimmed off), cached once and reused across seeds. Distinct
# from PATHOGENICITY_CANONICAL_VARIANTS_JSON below (the geometry-experiment set).
PATHOGENICITY_VALID_VARIANTS_JSON = DATA_DIR / "pathogenicity_valid_variants.json"

# ── Canonical pathogenicity set + masked-LL conservation (geometry experiments) ─
# The n=16,576 canonical pathogenicity variant set whose embeddings are PATH_EMB_*.
# Row-aligned to PATH_EMB_WT_MEAN / PATH_EMB_MUT_MEAN.
PATHOGENICITY_CANONICAL_VARIANTS_JSON = DATA_DIR / "pathogenicity_valid_variants_canonical.json"
# Masked-LM conservation readouts per canonical variant: [logP_wt, logP_mut, entropy].
CONSERVATION_PATHOGENICITY_NPY = DATA_DIR / "conservation_pathogenicity.npy"
CONSERVATION_PATHOGENICITY_META_JSON = DATA_DIR / "conservation_pathogenicity_meta.json"
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
MEGASCALE_EMB_FINGERPRINT = EMB_DIR / "megascale_fingerprint.json"

# ── Stability subspace (esm2_mechanism.py) ───────────────────────────────────
STABILITY_SUBSPACE = EMB_DIR / "stability_subspace.npy"
# Sidecar recording the Megascale inputs the subspace above was fitted from, so
# a changed variant set / ΔΔG table / embedding file refits instead of silently
# serving a subspace fitted against different data. A cache with no sidecar is
# unverifiable and is treated as a miss.
STABILITY_SUBSPACE_PARAMS_JSON = EMB_DIR / "stability_subspace.params.json"

# ── Perturbation scan embeddings (perturbation_scan.py) ──────────────────────
SCAN_EMB_WT = EMB_DIR / "scan_wt.npy"
SCAN_EMB_MUT = EMB_DIR / "scan_mut.npy"
SCAN_CKPT_WT = EMB_DIR / "scan_ckpt_wt.npy"
SCAN_CKPT_MUT = EMB_DIR / "scan_ckpt_mut.npy"

# ESM-3 embeddings live one level below ESM3_EMB_DIR, in a per-dataset subdir
# (geras | merged) chosen at runtime by esm3_mechanism.configure_dataset. There
# is no single canonical seq_mean.npy to name here.

# ── Downloads (manually-placed prerequisite files) ───────────────────────────
DOWNLOADS_DIR = DATA_DIR / "downloads"

# Pfam clan map (homology-partition robustness panel, Task 2b) — fetched from
# https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.clans.tsv.gz,
# not committed (too large; matches the existing DOWNLOADS_DIR convention below).
PFAM_CLANS_TSV_GZ = DOWNLOADS_DIR / "Pfam-A.clans.tsv.gz"

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
