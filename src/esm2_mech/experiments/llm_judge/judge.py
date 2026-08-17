"""LLM-as-judge that predicts a variant's mechanism class via forced tool call, traced to Langfuse."""

import functools
import os
import time

import anthropic
from langfuse import get_client, observe

from esm2_mech.utils.constants import (
    LABEL_3CLASS_FIELD,
    LLM_JUDGE_BACKOFF_SECONDS,
    LLM_JUDGE_MAX_RETRIES,
    LLM_JUDGE_MAX_TOKENS,
    LLM_JUDGE_MODEL,
    LLM_JUDGE_TOOL_NAME,
    MECHANISM_CLASSES,
)

print = functools.partial(print, flush=True)


class TransientProviderError(Exception):
    """A retryable provider failure (429 / 5xx / overloaded / timeout)."""


# Class set comes from MECHANISM_CLASSES, never hardcoded inline.
_MECHANISM_TOOL = {
    "name": LLM_JUDGE_TOOL_NAME,
    "description": (
        "Report the predicted molecular mechanism class for this missense "
        "variant. GOF = gain of function, DN = dominant-negative, LOF = loss of "
        "function."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "mechanism": {
                "type": "string",
                "enum": list(MECHANISM_CLASSES),
                "description": "The predicted mechanism class.",
            },
            "rationale": {
                "type": "string",
                "description": "One sentence justifying the call.",
            },
        },
        "required": ["mechanism", "rationale"],
    },
}


def _build_prompt(variant: dict) -> str:
    """Render a single variant into the judge prompt text."""
    ddg = variant.get("foldx_ddg")
    ddg_line = (
        f"FoldX folding stability change (ddG): {ddg:.2f} kcal/mol"
        if isinstance(ddg, (int, float))
        else "FoldX folding stability change (ddG): not available"
    )
    return (
        f"Gene: {variant['gene']} (UniProt {variant.get('uniprot_id', 'unknown')})\n"
        f"Missense substitution: {variant['aa_wt']}{variant['aa_pos']}"
        f"{variant['aa_mut']}\n"
        f"{ddg_line}\n\n"
        f"Predict the molecular mechanism class for this variant. Call the "
        f"{LLM_JUDGE_TOOL_NAME} tool with exactly one of "
        f"{', '.join(MECHANISM_CLASSES)}."
    )


def _classify_provider_error(err: Exception) -> bool:
    """Return True if err is transient and worth retrying."""
    if isinstance(err, (anthropic.RateLimitError, anthropic.InternalServerError)):
        return True
    if isinstance(err, anthropic.APIStatusError):
        return err.status_code in (429, 500, 502, 503, 504, 529)
    if isinstance(err, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
        return True
    return False


@observe(name="judge_variant")
def judge_variant(client: anthropic.Anthropic, variant: dict) -> dict:
    """Predict one variant's mechanism with linear-backoff retry on transient errors."""
    prompt = _build_prompt(variant)
    langfuse = get_client()
    truth = variant.get(LABEL_3CLASS_FIELD)

    last_err = None
    started = time.monotonic()
    for attempt in range(1, LLM_JUDGE_MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=LLM_JUDGE_MODEL,
                max_tokens=LLM_JUDGE_MAX_TOKENS,
                tools=[_MECHANISM_TOOL],
                tool_choice={"type": "tool", "name": LLM_JUDGE_TOOL_NAME},
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except Exception as err:  # noqa: BLE001 — re-raised below if not transient
            last_err = err
            if _classify_provider_error(err) and attempt < LLM_JUDGE_MAX_RETRIES:
                delay = LLM_JUDGE_BACKOFF_SECONDS * attempt
                print(
                    f"[judge] {variant['gene']} "
                    f"{variant['aa_wt']}{variant['aa_pos']}{variant['aa_mut']}: "
                    f"transient error {type(err).__name__} "
                    f"(attempt {attempt}/{LLM_JUDGE_MAX_RETRIES}), "
                    f"backing off {delay:.1f}s"
                )
                time.sleep(delay)
                continue
            raise
    else:
        raise TransientProviderError(str(last_err))

    latency_ms = (time.monotonic() - started) * 1000.0

    predicted = None
    rationale = None
    for block in response.content:
        if block.type == "tool_use" and block.name == LLM_JUDGE_TOOL_NAME:
            predicted = block.input.get("mechanism")
            rationale = block.input.get("rationale")
            break

    structured_ok = predicted in MECHANISM_CLASSES
    usage = response.usage

    record = {
        "gene": variant["gene"],
        "uniprot_id": variant.get("uniprot_id"),
        "variant": f"{variant['aa_wt']}{variant['aa_pos']}{variant['aa_mut']}",
        "truth": truth,
        "predicted": predicted,
        "rationale": rationale,
        # None when no ground truth or no valid prediction — never coerce unanswerable to False.
        "correct": (predicted == truth) if (structured_ok and truth) else None,
        "structured_ok": structured_ok,
        "attempts": attempt,
        "latency_ms": latency_ms,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "stop_reason": response.stop_reason,
    }

    langfuse.update_current_span(
        metadata={
            "gene": variant["gene"],
            "truth": truth,
            "predicted": predicted,
            "structured_ok": structured_ok,
            "attempts": attempt,
        }
    )
    return record


def make_client() -> anthropic.Anthropic:
    """Construct the Anthropic client, raising if ANTHROPIC_API_KEY is absent."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment")
    return anthropic.Anthropic()
