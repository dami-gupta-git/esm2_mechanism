"""Batched, Langfuse-traced run of the LLM-as-judge over a sample of variants."""

import argparse
import functools
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from langfuse import get_client

from esm2_mech.experiments.llm_judge.judge import judge_variant, make_client
from esm2_mech.utils.constants import (
    LABEL_3CLASS_FIELD,
    LLM_JUDGE_CONCURRENCY,
    MECHANISM_CLASSES,
)
from esm2_mech.utils.io import atomic_write_json
from esm2_mech.utils.paths import (
    LLM_JUDGE_DIR,
    LLM_JUDGE_PREDICTIONS_JSONL,
    LLM_JUDGE_SUMMARY_JSON,
    VARIANTS_JSON,
)

print = functools.partial(print, flush=True)


def _load_labelled_variants() -> list:
    """Load variants with a usable ground-truth mechanism class."""
    with open(VARIANTS_JSON) as handle:
        variants = json.load(handle)
    labelled = [
        variant
        for variant in variants
        if variant.get(LABEL_3CLASS_FIELD) in MECHANISM_CLASSES
    ]
    print(f"[run] {len(labelled)}/{len(variants)} variants have a 3-class label")
    return labelled


def _sample(variants: list, n_items: int, seed: int) -> list:
    rng = random.Random(seed)
    if n_items >= len(variants):
        return list(variants)
    return rng.sample(variants, n_items)


def _summarize(records: list) -> dict:
    """Score predictions against ground truth and return observability metrics."""
    scored = [rec for rec in records if rec["correct"] is not None]
    n_correct = sum(1 for rec in scored if rec["correct"])
    structured_failures = [rec for rec in records if not rec["structured_ok"]]
    retried = [rec for rec in records if rec["attempts"] > 1]
    latencies = sorted(rec["latency_ms"] for rec in records)

    def _pct(sorted_vals, frac):
        if not sorted_vals:
            return None
        idx = min(len(sorted_vals) - 1, int(frac * len(sorted_vals)))
        return sorted_vals[idx]

    pred_counts = {cls: 0 for cls in MECHANISM_CLASSES}
    for rec in records:
        if rec["predicted"] in pred_counts:
            pred_counts[rec["predicted"]] += 1

    return {
        "n_total": len(records),
        "n_scored": len(scored),
        "n_correct": n_correct,
        "accuracy": (n_correct / len(scored)) if scored else None,
        "n_structured_failures": len(structured_failures),
        "n_retried": len(retried),
        "total_input_tokens": sum(rec["input_tokens"] for rec in records),
        "total_output_tokens": sum(rec["output_tokens"] for rec in records),
        "latency_ms_p50": _pct(latencies, 0.50),
        "latency_ms_p95": _pct(latencies, 0.95),
        "prediction_counts": pred_counts,
    }


def main() -> None:
    # Load LANGFUSE_* / ANTHROPIC_API_KEY from .env before any SDK client initialises.
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="number of variants to judge")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    args = parser.parse_args()

    if os.environ.get("TRACING_ENABLED", "true").lower() in ("0", "false", "no"):
        print("[run] WARNING: TRACING_ENABLED is off; Langfuse will not record")

    LLM_JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    client = make_client()
    langfuse = get_client()

    variants = _sample(_load_labelled_variants(), args.n, args.seed)
    print(
        f"[run] judging {len(variants)} variants with concurrency "
        f"{LLM_JUDGE_CONCURRENCY}"
    )

    records = []
    with open(LLM_JUDGE_PREDICTIONS_JSONL, "w") as out:
        with ThreadPoolExecutor(max_workers=LLM_JUDGE_CONCURRENCY) as pool:
            futures = {
                pool.submit(judge_variant, client, variant): variant
                for variant in variants
            }
            for done, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                records.append(record)
                out.write(json.dumps(record) + "\n")
                out.flush()
                if done % 25 == 0 or done == len(variants):
                    print(f"[run] {done}/{len(variants)} done")

    langfuse.flush()

    summary = _summarize(records)
    atomic_write_json(LLM_JUDGE_SUMMARY_JSON, summary, indent=2)
    print(f"[run] summary -> {LLM_JUDGE_SUMMARY_JSON}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
