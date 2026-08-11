import functools

print = functools.partial(print, flush=True)

# ── Model identifiers ─────────────────────────────────────────────────────────
ESM2_MODEL = "esm2_t33_650M_UR50D"
ESM2_MODEL_3B = "esm2_t36_3B_UR50D"
ESM3_MODEL = "esm3-sm-open-v1"

# ── Amino acids ───────────────────────────────────────────────────────────────
# Canonical 20 amino acids in fixed order. Single source of truth for any
# AA->index encoding (e.g. WT/MUT one-hot). Never inline the literal string.
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AA_ORDER)}

# ── Seeds ─────────────────────────────────────────────────────────────────────
# Number of random seeds for every multi-seed probe. The `--seeds` CLI flag is a
# COUNT (default N_SEEDS): seeds 0..N_SEEDS-1 are run. Single source of truth —
# never hardcode 5 or [0, 1, 2, 3, 4] inline.
N_SEEDS = 5

# Number of CV folds for every multi-seed probe. Single source of truth — never
# hardcode 5 inline.
N_FOLDS = 5

# ── Numerical floors ──────────────────────────────────────────────────────────
# Divide-by-zero / norm floor for ratios and projections. Single source of truth —
# never inline 1e-10. (Distinct from STD_EPS below and from the tighter 1e-12 used
# in the geometry probes, which is a deliberately different role.)
NORM_EPS = 1e-10
# Per-column standardization std floor (avoids divide-by-zero on a constant column).
STD_EPS = 1e-8

# ── Mechanism class label constants ──────────────────────────────────────────
GOF = "GOF"
DN = "DN"
LOF = "LOF"
MECHANISM_CLASSES = [GOF, DN, LOF]

# ── Variant label sources (the `source` field in valid_variants.json) ─────────
SOURCE_GERASIMAVICIUS = "gerasimavicius"
SOURCE_CLINVAR_G2P = "clinvar_g2p"

# ── LLM-judge mechanism eval ──────────────────────────────────────────────────
# Model and batch parameters for the Langfuse-traced LLM-as-judge that predicts
# label_3class per variant. Single source of truth — never inline these.
LLM_JUDGE_MODEL = "claude-haiku-4-5"
LLM_JUDGE_MAX_TOKENS = 1024
LLM_JUDGE_TOOL_NAME = "report_mechanism"
# Concurrency for the batched judge run. Deliberately set to exercise the
# provider's rate limits under load.
LLM_JUDGE_CONCURRENCY = 8
# Linear-backoff retry budget for transient provider errors (429 / 5xx / overloaded).
LLM_JUDGE_MAX_RETRIES = 5
LLM_JUDGE_BACKOFF_SECONDS = 2.0
# The field in valid_variants.json / variants.json holding the ground-truth class.
LABEL_3CLASS_FIELD = "label_3class"

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


# Seed-0 gene-split/family-split OOF cache (row_ids, y_true, pred, genes) for the
# PERMUTATION_FEATURES headline features, written by mechanism_delta_family_split.run
# alongside its seed-result file (same out_dir) so leakage_fraction.py can bootstrap
# the leakage-fraction RATIO jointly (shared resample across both arms) instead of
# combining two separately-computed CIs.
MECHANISM_OOF_CACHE_PREFIX = "mechanism_oof_cache_seed"
MECHANISM_OOF_CACHE_EXT = ".json"


def mechanism_oof_cache_filename(seed: int) -> str:
    """Filename for one seed's mechanism OOF cache."""
    return f"{MECHANISM_OOF_CACHE_PREFIX}{seed}{MECHANISM_OOF_CACHE_EXT}"


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


# ── Nonlinear delta-probe result keys ─────────────────────────────────────────
# Keys under which experiments/mechanism/mlp stores each nonlinear probe's metrics
# in nonlinear_results_seed{seed}.json. The convention is <model>_<feat>_<split>,
# e.g. "mlp_delta_mean_family". Both the producer (mlp.py) and the consumers
# (multiseed_v1, esm3_mechanism) build keys via nonlinear_key() so the format
# lives in exactly one place — never hardcode the literal strings inline.
DELTA_POS_FEATURE = "delta_pos"
NONLINEAR_MODELS = ("mlp", "gbm", "rf", "knn")
NONLINEAR_FEATURES = (DELTA_MEAN_FEATURE, DELTA_POS_FEATURE)
SPLIT_GENE = "gene"
SPLIT_FAMILY = "family"
NONLINEAR_SPLITS = (SPLIT_GENE, SPLIT_FAMILY)


def nonlinear_key(model: str, feature: str, split: str) -> str:
    """Result key for one nonlinear delta probe: <model>_<feature>_<split>.

    Validates each token against the known vocabularies so a typo fails loudly
    here rather than silently reading a missing key downstream.
    """
    if model not in NONLINEAR_MODELS:
        raise ValueError(f"unknown nonlinear model {model!r}; expected one of {NONLINEAR_MODELS}")
    if feature not in NONLINEAR_FEATURES:
        raise ValueError(f"unknown nonlinear feature {feature!r}; expected one of {NONLINEAR_FEATURES}")
    if split not in NONLINEAR_SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {NONLINEAR_SPLITS}")
    return f"{model}_{feature}_{split}"


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
# A cluster-bootstrap metric can be undefined on a resample (e.g. a rare class is
# absent, so one-vs-rest AUROC is undefined) and is dropped from the percentile. If
# too few resamples survive, the CI is built on a biased, thinned subset and must not
# be trusted. Below this surviving fraction, cluster_bootstrap_ci returns no CI and
# flags it so the dropout is visible instead of silently narrowing the interval.
BOOTSTRAP_MIN_VALID_FRAC = 0.8
# No-signal reference for a one-vs-rest AUROC (ranking metric): a CI clearing this
# from above, or a permutation p-value against it, marks above-chance separation.
CHANCE_AUROC = 0.5

# ── External API roots ────────────────────────────────────────────────────────
UNIPROT_REST = "https://rest.uniprot.org/uniprotkb"
