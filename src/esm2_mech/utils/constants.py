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

# Claim 2B is supported only when the paired split-gap interval excludes zero
# in at least three of the five preregistered seeds.
SPLIT_GAP_MIN_SUPPORTING_SEEDS = 3

# Claim 2A's equivalence margin above the measured family-split chance floor.
MECHANISM_NULL_FLOOR_MARGIN = 0.05
# Claim 2A is affirmed only when this many of the five seed-specific intervals
# have an upper bound below the measured floor plus the equivalence margin.
MECHANISM_NULL_MIN_AFFIRMING_SEEDS = 3

# Claim 2G's minimum material enzyme-minus-mechanism macro-F1 difference.
ENZYME_MECHANISM_MIN_F1_GAP = 0.05

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


# Per-seed gene-split/family-split OOF cache for every scored feature. The cache
# is bound to its exact seed result and lets leakage_fraction.py align both arms
# to one row space before computing the headline or its interval.
MECHANISM_OOF_CACHE_PREFIX = "mechanism_oof_cache_seed"
MECHANISM_OOF_CACHE_EXT = ".json"
MECHANISM_OOF_CACHE_GLOB = f"{MECHANISM_OOF_CACHE_PREFIX}*{MECHANISM_OOF_CACHE_EXT}"
# Increment whenever the cache envelope or feature-arm schema changes. Every
# reader must require this exact version before using cached scientific data.
MECHANISM_OOF_CACHE_SCHEMA_VERSION = 2


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


# ── Two-stage cascade mechanism classifier ────────────────────────────────────
# Stage A is LOF vs {GOF, DN}; stage B is GOF vs DN fitted on the non-LOF rows.
# Cascade posteriors are P(LOF)=pA, P(GOF)=(1-pA)*pB, P(DN)=(1-pA)*(1-pB).
CASCADE_STAGE_A = "lof_vs_rest"
CASCADE_STAGE_B = "gof_vs_dn"
CASCADE_STAGES = (CASCADE_STAGE_A, CASCADE_STAGE_B)
# Training-fold sampling arms. "family_matched" equalizes LOF against non-LOF
# inside each Pfam family so family identity cannot predict the stage-A label;
# "unbalanced" keeps the training fold as-is and is the ablation it is read
# against. Never compare a family_matched number to a single-stage result
# without also reading the unbalanced arm.
CASCADE_ARM_FAMILY_MATCHED = "family_matched"
CASCADE_ARM_UNBALANCED = "unbalanced"
# "size_matched" is the control that separates the two things family matching
# does at once. It draws the same number of rows at the same LOF-to-non-LOF
# ratio as the family_matched arm produced on that fold, but samples them across
# the whole training fold instead of pairing within a family. Reading
# family_matched against size_matched isolates the removal of the family
# shortcut; reading it against unbalanced confounds that with the row count.
CASCADE_ARM_SIZE_MATCHED = "size_matched"
CASCADE_SAMPLING_ARMS = (
    CASCADE_ARM_FAMILY_MATCHED,
    CASCADE_ARM_SIZE_MATCHED,
    CASCADE_ARM_UNBALANCED,
)
# k-means clusters fitted on the training fold's LOF delta embeddings. LOF
# downsampling draws round-robin across them so the retained LOF rows keep the
# spread of the discarded ones rather than collapsing onto one region.
CASCADE_LOF_N_CLUSTERS = 8
# PCA components the LOF rows are reduced to before k-means, fitted on the
# training fold only. k-means on the raw 1280-d delta is dominated by the
# high-variance directions and is slow enough to matter across 5 seeds x 5 folds.
CASCADE_LOF_CLUSTER_PCA = 50
# Target LOF-to-non-LOF row ratio in a family_matched training fold. Matched
# rows from mixed families are taken first; only if they fall short of the
# target is the remainder drawn from LOF-only families, which carry no
# within-family contrast and can only teach family identity.
CASCADE_LOF_TARGET_RATIO = 1.0
# Focal-loss focusing exponent. 0.0 reduces the loss to weighted cross-entropy.
CASCADE_FOCAL_GAMMA = 2.0

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
# Claim 2A's preregistered across-seed permutation rule. A result is evaluable
# only when all N_SEEDS requested seed runs produced a finite p-value.
PERMUTATION_SIGNIFICANCE_THRESHOLD = 0.05
PERMUTATION_MIN_SIGNIFICANT_SEEDS = 3
# A cluster-bootstrap metric can be undefined on a resample (e.g. a rare class is
# absent, so one-vs-rest AUROC is undefined) and is dropped from the percentile. If
# too few resamples survive, the CI is built on a biased, thinned subset and must not
# be trusted. Below this surviving fraction, cluster_bootstrap_ci returns no CI and
# flags it so the dropout is visible instead of silently narrowing the interval.
BOOTSTRAP_MIN_VALID_FRAC = 0.8
# Every fold-aware metric scores each class inside each fold and averages. A resample
# in which any fold has lost a class entirely is discarded rather than scored over the
# folds that survive, so every resample scores the same statistic (all folds, all
# classes). In the real splits every fold already carries every class with several
# families to spare, so a discard needs a draw that drops all of a fold's families for
# one class at once and should be rare. A discard rate above this fraction is a fault
# signal — most likely the resampling unit or the fold construction — and must be
# investigated rather than absorbed.
BOOTSTRAP_MAX_DISCARD_FRAC = 0.01
# No-signal reference for a one-vs-rest AUROC (ranking metric): a CI clearing this
# from above, or a permutation p-value against it, marks above-chance separation.
CHANCE_AUROC = 0.5
# Minimum distinct classes a CV fold's train split needs for a classifier to fit.
# A fold where a rare class falls entirely in test is still fittable and must be
# kept: skipping at n_classes silently averages arms over different fold sets.
MIN_TRAIN_CLASSES = 2

# ── External API roots ────────────────────────────────────────────────────────
UNIPROT_REST = "https://rest.uniprot.org/uniprotkb"
# Sent on every outbound HTTP request; some endpoints reject the default urllib UA.
HTTP_USER_AGENT = "Mozilla/5.0"
