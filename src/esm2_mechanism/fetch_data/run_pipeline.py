"""
Run the full fetch pipeline in order (steps 1–11).

State is saved to data/.pipeline_state.json after each successful step.
By default the pipeline resumes from the step after the last recorded success.
Use --from-step N to override.

Usage:
    python -m esm2_mechanism.fetch_data.run_pipeline
    python -m esm2_mechanism.fetch_data.run_pipeline --from-step 5
    python -m esm2_mechanism.fetch_data.run_pipeline --from-step 5 --pathogenic-only

Steps:
     1  build_gene_universe  gene-list   — merge Gerasimavicius + G2P
     2  fetch_variants       gerasimavicius
     3  fetch_variants       clinvar
     4  fetch_variants       merge
     5  fetch_annotations    pfam
     6  build_gene_universe  universe
     7  fetch_annotations    uniprot
     8  fetch_annotations    enzyme
     9  build_proteome_features
    10  build_badonyi_features
    11  fetch_annotations    alphamissense  (requires merged_valid_variants.json)
"""

from __future__ import annotations

import argparse
import datetime
import functools
import json
import os
import sys
import traceback
from pathlib import Path

from esm2_mechanism.fetch_data.build_gene_universe import main_gene_list, main_universe
from esm2_mechanism.fetch_data.fetch_variants import (
    main_gerasimavicius,
    main_clinvar,
    main_merge,
)
from esm2_mechanism.fetch_data.fetch_annotations import (
    main_pfam,
    main_uniprot,
    main_enzyme,
    main_alphamissense,
)
from esm2_mechanism.fetch_data.build_proteome_features import main as main_proteome
from esm2_mechanism.fetch_data.build_badonyi_features import main as main_badonyi
from esm2_mechanism.utils_paths import DATA_DIR

print = functools.partial(print, flush=True)

FIRST_STEP = 1
LAST_STEP = 11
STATE_FILE = DATA_DIR / ".pipeline_state.json"


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception as exc:
        print(f"WARNING: could not read pipeline state file ({exc}) — treating as no prior state")
        return {}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, STATE_FILE)


def _make_steps(pathogenic_only: bool, from_scratch: bool) -> list[tuple[int, str, callable]]:
    return [
        (1,  "build_gene_universe gene-list",   main_gene_list),
        (2,  "fetch_variants gerasimavicius",    main_gerasimavicius),
        (3,  "fetch_variants clinvar",           main_clinvar),
        (4,  "fetch_variants merge",             lambda: main_merge(pathogenic_only=pathogenic_only)),
        (5,  "fetch_annotations pfam",           lambda: main_pfam(from_scratch=from_scratch)),
        (6,  "build_gene_universe universe",     main_universe),
        (7,  "fetch_annotations uniprot",        lambda: main_uniprot(from_scratch=from_scratch)),
        (8,  "fetch_annotations enzyme",         main_enzyme),
        (9,  "build_proteome_features",          main_proteome),
        (10, "build_badonyi_features",           main_badonyi),
        (11, "fetch_annotations alphamissense",  main_alphamissense),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full fetch pipeline. Resumes automatically from the step after "
            "the last recorded success. Exits immediately on any step failure."
        )
    )
    parser.add_argument(
        "--from-step",
        type=int,
        default=None,
        metavar="N",
        help=(
            f"Force start from step N ({FIRST_STEP}–{LAST_STEP}), "
            "ignoring saved state. Omit to auto-resume."
        ),
    )
    parser.add_argument(
        "--pathogenic-only",
        action="store_true",
        help="(step 4) Restrict ClinVar variants to 'pathogenic' only.",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="(steps 5, 7) Ignore existing cache and re-fetch.",
    )
    args = parser.parse_args()

    if args.from_step is not None and not (FIRST_STEP <= args.from_step <= LAST_STEP):
        parser.error(f"--from-step must be between {FIRST_STEP} and {LAST_STEP}")

    state = _load_state()
    last_completed = state.get("last_completed_step", 0)

    if args.from_step is not None:
        resume_from = args.from_step
        print(f"Resuming from step {resume_from} (--from-step override)")
    elif last_completed >= LAST_STEP:
        print(
            f"All steps already completed (last run: {state.get('last_completed_at', 'unknown')}). "
            "Use --from-step to re-run."
        )
        return
    elif last_completed > 0:
        resume_from = last_completed + 1
        print(
            f"Resuming from step {resume_from} "
            f"(last completed: step {last_completed} at {state.get('last_completed_at', 'unknown')})"
        )
    else:
        resume_from = FIRST_STEP
        print("No prior state found — starting from step 1")

    steps = _make_steps(
        pathogenic_only=args.pathogenic_only,
        from_scratch=args.from_scratch,
    )

    for step_num, label, fn in steps:
        if step_num < resume_from:
            print(f"[step {step_num:2d}] skipping: {label}")
            continue

        print(f"\n{'='*60}")
        print(f"[step {step_num:2d}] {label}")
        print(f"{'='*60}")
        try:
            fn()
        except Exception:
            print(f"\nERROR: step {step_num} ({label}) failed:", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            print(
                f"\nPipeline aborted at step {step_num}. "
                f"Re-run without arguments to resume from this step.",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)

        state["last_completed_step"] = step_num
        state["last_completed_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        state[f"step_{step_num}_completed_at"] = state["last_completed_at"]
        _save_state(state)
        print(f"[step {step_num:2d}] done: {label}")

    print(f"\n{'='*60}")
    print("Pipeline complete (all steps finished).")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
