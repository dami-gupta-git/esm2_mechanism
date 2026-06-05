import functools

print = functools.partial(print, flush=True)

# ── Model identifiers ─────────────────────────────────────────────────────────
ESM2_MODEL = "esm2_t33_650M_UR50D"
ESM2_MODEL_3B = "esm2_t36_3B_UR50D"
ESM3_MODEL = "esm3-sm-open-v1"

# ── Mechanism class label constants ──────────────────────────────────────────
GOF = "GOF"
DN = "DN"
LOF = "LOF"
MECHANISM_CLASSES = [GOF, DN, LOF]

# ── Variant label sources (the `source` field in valid_variants.json) ─────────
SOURCE_GERASIMAVICIUS = "gerasimavicius"
SOURCE_CLINVAR_G2P = "clinvar_g2p"

# ── ESM-2 sequence limits ─────────────────────────────────────────────────────
MAX_SEQ_LEN = 1022          # ESM-2 token limit: 1024 - <cls> - <eos>
WINDOW_HALF = MAX_SEQ_LEN // 2  # 511: centre variant with full budget used

# ── Data file names ───────────────────────────────────────────────────────────
GENE_UNIVERSE_FILENAME = "gene_universe.tsv"
SEQUENCES_FILENAME = "sequences.json"
PFAM_FAMILIES_FILENAME = "pfam_families.json"

# ── Mechanism per-seed baseline result files ──────────────────────────────────
# Written by mechanism_delta_family_split.run; pooled by classify_by_mechanism.
# Both the writer and the aggregator derive their names from this single source,
# so the format lives in exactly one place.
SEED_RESULT_PREFIX = "family_split_baselines_seed"
SEED_RESULT_EXT = ".json"
SEED_RESULT_GLOB = f"{SEED_RESULT_PREFIX}*{SEED_RESULT_EXT}"


def seed_result_filename(seed: int) -> str:
    """Filename for one seed's family-split baseline results."""
    return f"{SEED_RESULT_PREFIX}{seed}{SEED_RESULT_EXT}"


# ── Contrastive metric-learning per-seed result files ─────────────────────────
# Written by contrastive_mechanism.run; pooled by contrastive_mechanism.main.
# Same single-source pattern as SEED_RESULT_* above.
CONTRASTIVE_SEED_RESULT_PREFIX = "contrastive_results_seed"
CONTRASTIVE_SEED_RESULT_EXT = ".json"
CONTRASTIVE_SEED_RESULT_GLOB = (
    f"{CONTRASTIVE_SEED_RESULT_PREFIX}*{CONTRASTIVE_SEED_RESULT_EXT}"
)

# Feature key under which the MLP delta_mean family-split floor is stored in the
# run's aggregate.json (written by classify_by_mechanism). The contrastive
# verdict compares against this floor; never hardcode the floor value.
DELTA_MEAN_FEATURE = "delta_mean"


def contrastive_seed_result_filename(seed: int) -> str:
    """Filename for one seed's contrastive metric-learning results."""
    return (
        f"{CONTRASTIVE_SEED_RESULT_PREFIX}{seed}{CONTRASTIVE_SEED_RESULT_EXT}"
    )


# ── Megascale / Tsuboyama 2023 stability parsing ──────────────────────────────
# A natural domain's WT_name base is a real 4-char PDB id (digit + 3 alphanumeric)
# followed by ".pdb"; de novo designs use synthetic names (e.g. "EA|run2_...",
# "HHH", "XX|run1") that do not match. We keep only natural domains.
MEGASCALE_PDB_ID_REGEX = r"^[0-9][A-Za-z0-9]{3}\.pdb$"
# A single-point substitution mutation code, e.g. "D1Q" (wt-aa, 1-indexed pos, mut-aa).
MEGASCALE_SUBSTITUTION_REGEX = r"^([A-Z])(\d+)([A-Z])$"
# Tsuboyama marks the wild-type row with this mut_type value.
MEGASCALE_WT_MUT_TYPE = "wt"
# ddG_ML uses this string for rows whose fit was unreliable — dropped, never imputed.
MEGASCALE_DDG_MISSING = "-"

# ── Cluster-bootstrap / permutation inference (pre-preprint statistics) ───────
# Dependency-aware inference resamples whole genes/families (the label unit), not
# variants. See reports/run6/STATS_PLAN.md for the rationale.
BOOTSTRAP_N_RESAMPLES = 1000
BOOTSTRAP_CI_LEVEL = 0.95
PERMUTATION_N_RESAMPLES = 1000
# No-signal reference for a one-vs-rest AUROC (ranking metric): a CI clearing this
# from above, or a permutation p-value against it, marks above-chance separation.
CHANCE_AUROC = 0.5

# ── External API roots ────────────────────────────────────────────────────────
UNIPROT_REST = "https://rest.uniprot.org/uniprotkb"
