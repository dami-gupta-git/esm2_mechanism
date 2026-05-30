import functools

print = functools.partial(print, flush=True)

# ── Model identifiers ─────────────────────────────────────────────────────────
ESM2_MODEL = "esm2_t33_650M_UR50D"
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

# ── External API roots ────────────────────────────────────────────────────────
UNIPROT_REST = "https://rest.uniprot.org/uniprotkb"
