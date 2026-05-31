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


# ── External API roots ────────────────────────────────────────────────────────
UNIPROT_REST = "https://rest.uniprot.org/uniprotkb"
