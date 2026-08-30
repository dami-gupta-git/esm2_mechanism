"""Builders shared by more than one test module.

The seed-aggregate builders construct their dicts through the production
SeedAggregate rather than restating its fields, so a schema change fails the
tests instead of leaving them asserting against a stale hand-written shape.
"""

from esm2_mech.utils.seed_aggregation import (
    SEED_STATUS_SUCCESS,
    SeedAggregate,
    SeedUnavailableReason,
)

# The standard five-seed set, and a value ladder over it whose spreads are
# known exactly, so a test can assert the reduction rather than a rounded number.
FIVE_SEEDS = (0, 1, 2, 3, 4)
FIVE_VALUES_BY_SEED = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5}
MEAN_OF_FIVE_VALUES = 0.3
SAMPLE_SPREAD_OF_FIVE_VALUES = 0.15811388300841897
POPULATION_SPREAD_OF_FIVE_VALUES = 0.1414213562373095

DEFAULT_SEEDS = FIVE_SEEDS


def seed_result(macro_f1_mean, *, split="gene_split", feature="esm2"):
    """A minimal per-seed result dict for one split/feature."""
    return {
        split: {
            feature: {
                "status": SEED_STATUS_SUCCESS,
                "macro_f1_mean": macro_f1_mean,
                "macro_f1_std": 0.01,
            }
        }
    }


def available_seed_aggregate(mean, spread=0.01, seeds=DEFAULT_SEEDS):
    """A stored seed aggregate whose every requested seed contributed."""
    return SeedAggregate(
        state="available",
        reason=None,
        requested_seeds=tuple(seeds),
        contributing_seeds=tuple(seeds),
        affected_seeds=(),
        mean=mean,
        spread=spread,
        sampling_unit="model_seed",
        message=None,
    ).to_dict()


def unavailable_seed_aggregate(seeds=DEFAULT_SEEDS, affected=(4,)):
    """A stored seed aggregate withheld because some requested seed failed."""
    return SeedAggregate(
        state="unavailable",
        reason=SeedUnavailableReason.FAILED_SEED,
        requested_seeds=tuple(seeds),
        contributing_seeds=tuple(seed for seed in seeds if seed not in affected),
        affected_seeds=tuple(affected),
        mean=None,
        spread=None,
        sampling_unit="model_seed",
        message="a requested seed failed",
    ).to_dict()
