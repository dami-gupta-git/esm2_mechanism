"""Shared model-seed summaries for perturbation probes."""

from esm2_mech.utils.seed_aggregation import (
    SEED_STATUS_FAILED,
    SEED_STATUSES,
    aggregate_seed_results,
    read_seed_point_estimate,
)


METRICS = ("macro_f1", "auroc_GOF")


def aggregate_probe_results(requested_seeds, per_seed_results, requested_arms):
    """Summarize each declared probe arm across the requested model seeds.

    The caller declares which arms the experiment set out to run, so an arm that
    no seed produced is reported as unavailable rather than disappearing from the
    summary. Each metric carries its own availability: one metric being undefined
    does not withhold another that aggregated cleanly.
    """
    requested = tuple(requested_seeds)
    seed_results = list(per_seed_results.values())
    summary = {}
    for key in requested_arms:
        summary[key] = {
            metric: aggregate_seed_results(
                requested,
                seed_results,
                lambda result, key=key, metric=metric: result["results"]
                .get(key, {})
                .get(f"{metric}_mean"),
                status=lambda result, key=key: _metric_status(result, key),
            ).to_dict()
            for metric in METRICS
        }
    return summary


def read_probe_metric(summary, key, metric):
    return read_seed_point_estimate(summary[key][metric])


def _metric_status(seed_result, key):
    cell = seed_result["results"].get(key)
    if cell is None:
        return SEED_STATUS_FAILED
    status = cell.get("status")
    if status not in SEED_STATUSES:
        raise ValueError(f"{key}: invalid probe status {status!r}")
    return status
