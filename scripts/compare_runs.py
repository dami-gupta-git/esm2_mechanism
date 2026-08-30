"""Diff every number in one run's result files against another's.

Compare point estimates in two current-schema result trees.

  python scripts/compare_runs.py run6 run_biorxiv

Every numeric and string leaf in every JSON under `results/<run>/` is compared by its
dotted path. A shared aggregate's `mean` is judged against its `seed_std` only when
the aggregate declares the current schema and `model_seed` sampling unit. Other
numeric leaves use --abs-threshold.

Keys present in only one run are reported separately from movement. run_biorxiv adds
CI keys throughout, so folding additions into the movement count would bury the
signal this script exists to surface.

Self-diff is the invariant: comparing a run against itself must report no movement,
no additions and no removals.
"""

import argparse
import functools
import json
import math
import sys
from pathlib import Path

from esm2_mech.utils.seed_aggregation import read_seed_inference
from esm2_mech.utils.paths import results_dir_for_run

print = functools.partial(print, flush=True)

# Only used when a metric has no sibling `_std` to judge it against; every such
# comparison is counted and reported so this never passes silently as a seed-std test.
DEFAULT_ABS_THRESHOLD = 0.005

MEAN_SUFFIX = ".mean"


def _flatten(node, prefix: str, out: dict) -> None:
    """Collect every scalar leaf as {dotted path: value}.

    Lists are indexed positionally. A list whose length differs between runs shows up
    as added/removed keys at the tail, which is the honest reading — there is no
    element-wise correspondence to assume once the lengths differ.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            _flatten(value, f"{prefix}.{key}" if prefix else str(key), out)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _flatten(value, f"{prefix}[{index}]", out)
    else:
        out[prefix] = node


def load_run(run_dir: Path) -> dict:
    """Flatten every JSON under `run_dir` into one {path: leaf} mapping.

    Keys are prefixed with the file's path relative to the run directory, so two runs
    are comparable regardless of where they live on disk. A file that fails to parse
    is reported and skipped rather than aborting the diff — a corrupt file in one
    corner should not hide movement everywhere else.
    """
    leaves: dict = {}
    unreadable = []
    for path in sorted(run_dir.rglob("*.json")):
        try:
            with open(path) as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            unreadable.append((path, exc))
            continue
        _flatten(data, str(path.relative_to(run_dir)), leaves)
    for path, exc in unreadable:
        print(f"  WARNING: could not read {path}: {exc}")
    return leaves


def _is_number(value) -> bool:
    # bool is an int subclass; treat it as a categorical value, not a measurement.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


_AGGREGATE_SCALARS = (
    "schema_version",
    "state",
    "reason",
    "mean",
    "seed_std",
    "sampling_unit",
    "message",
)
_AGGREGATE_SEED_LISTS = ("requested_seeds", "contributing_seeds", "affected_seeds")


def _aggregate_at(path: str, leaves: dict) -> dict:
    """Rebuild one stored seed aggregate from its flattened leaves.

    Flattening indexes lists positionally, so the seed lists are reassembled from
    their indexed leaves. An empty list contributes no leaves and comes back empty,
    which is what an available aggregate stores for its affected seeds.
    """
    record = {
        field: leaves[f"{path}.{field}"]
        for field in _AGGREGATE_SCALARS
        if f"{path}.{field}" in leaves
    }
    for field in _AGGREGATE_SEED_LISTS:
        values = []
        while f"{path}.{field}[{len(values)}]" in leaves:
            values.append(leaves[f"{path}.{field}[{len(values)}]"])
        record[field] = values
    return record


def _threshold_for(key: str, old: dict, abs_threshold: float) -> tuple[float, bool]:
    """(threshold, used_seed_std) for one metric.

    The shared reader decides whether a stored aggregate is a complete seed
    aggregate, so this script does not carry its own copy of that rule. Fold,
    protein, partition, and other spreads never stand in for seed variation, and a
    zero spread is no threshold at all.
    """
    if key.endswith(MEAN_SUFFIX):
        aggregate_path = key[: -len(MEAN_SUFFIX)]
        metric = read_seed_inference(_aggregate_at(aggregate_path, old))
        if metric.available and metric.spread > 0:
            return float(metric.spread), True
    return abs_threshold, False


def compare(old: dict, new: dict, abs_threshold: float) -> dict:
    """Diff two flattened runs into moved / changed / added / removed buckets."""
    moved, changed, unchanged_count, threshold_fallbacks = [], [], 0, 0
    incomparable = []

    for key in sorted(set(old) & set(new)):
        old_value, new_value = old[key], new[key]

        if _is_number(old_value) and _is_number(new_value):
            # NaN == NaN is False, but two NaNs are the same reading, not movement.
            if math.isnan(old_value) and math.isnan(new_value):
                unchanged_count += 1
                continue
            if math.isnan(old_value) != math.isnan(new_value):
                incomparable.append((key, old_value, new_value))
                continue
            delta = new_value - old_value
            threshold, used_std = _threshold_for(key, old, abs_threshold)
            if not used_std:
                threshold_fallbacks += 1
            if abs(delta) > threshold:
                moved.append((key, old_value, new_value, delta, threshold, used_std))
            else:
                unchanged_count += 1
            continue

        # Non-numeric leaves: gate verdicts, labels, nulls. Any change is reported,
        # since a verdict flipping is exactly what this is meant to catch.
        if old_value == new_value:
            unchanged_count += 1
        else:
            changed.append((key, old_value, new_value))

    return {
        "moved": moved,
        "changed": changed,
        "incomparable": incomparable,
        "added": sorted(set(new) - set(old)),
        "removed": sorted(set(old) - set(new)),
        "unchanged": unchanged_count,
        "threshold_fallbacks": threshold_fallbacks,
    }


def format_report(
    old_run: str, new_run: str, result: dict, abs_threshold: float
) -> str:
    """The delta note: a table of old / new / delta with the material-movement flag."""
    lines = [
        f"# Delta note — {old_run} to {new_run}",
        "",
        f"Compared {result['unchanged'] + len(result['moved']) + len(result['changed'])} "
        f"shared leaves. Movement is judged against the {old_run} shared `seed_std` "
        f"where one exists, otherwise against an absolute threshold of {abs_threshold}; "
        f"{result['threshold_fallbacks']} numeric comparisons used the absolute "
        "threshold.",
        "",
    ]

    if result["moved"]:
        lines += [
            f"## Material movement ({len(result['moved'])})",
            "",
            "| Key | Old | New | Delta | Threshold | Basis |",
            "|---|---|---|---|---|---|",
        ]
        for key, old_value, new_value, delta, threshold, used_std in result["moved"]:
            basis = "seed SD" if used_std else "absolute"
            lines.append(
                f"| `{key}` | {old_value:.6g} | {new_value:.6g} | {delta:+.6g} | "
                f"{threshold:.6g} | {basis} |"
            )
        lines.append("")
    else:
        lines += ["## Material movement", "", "None.", ""]

    if result["changed"]:
        lines += [
            f"## Non-numeric changes ({len(result['changed'])})",
            "",
            "| Key | Old | New |",
            "|---|---|---|",
        ]
        lines += [f"| `{k}` | {o!r} | {n!r} |" for k, o, n in result["changed"]]
        lines.append("")

    if result["incomparable"]:
        lines += [
            f"## NaN appeared or disappeared ({len(result['incomparable'])})",
            "",
            "A metric that became NaN, or stopped being NaN, is a change in what could "
            "be measured at all — not a movement in its value.",
            "",
            "| Key | Old | New |",
            "|---|---|---|",
        ]
        lines += [f"| `{k}` | {o} | {n} |" for k, o, n in result["incomparable"]]
        lines.append("")

    for bucket, title, note in (
        (
            "added",
            "Keys added",
            f"Present in {new_run} only — expected for the new CI "
            "and paired-difference keys.",
        ),
        (
            "removed",
            "Keys removed",
            f"Present in {old_run} only. Each one is either an "
            "intended cut or a silently dropped output.",
        ),
    ):
        entries = result[bucket]
        lines += [f"## {title} ({len(entries)})", ""]
        if entries:
            lines += [note, ""] + [f"- `{key}`" for key in entries] + [""]
        else:
            lines += ["None.", ""]

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("old_run", help="baseline run name, e.g. run6")
    parser.add_argument("new_run", help="run being checked, e.g. run_biorxiv")
    parser.add_argument(
        "--abs-threshold",
        type=float,
        default=DEFAULT_ABS_THRESHOLD,
        help="movement threshold for metrics with no sibling _std",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the delta note here as well as to stdout",
    )
    parser.add_argument(
        "--fail-on-movement",
        action="store_true",
        help="exit non-zero if anything moved materially (the self-diff invariant)",
    )
    args = parser.parse_args()

    old_dir = results_dir_for_run(args.old_run)
    new_dir = results_dir_for_run(args.new_run)
    for run_name, run_dir in ((args.old_run, old_dir), (args.new_run, new_dir)):
        if not run_dir.is_dir():
            raise FileNotFoundError(f"{run_dir} (results for run {run_name!r})")

    print(f"Reading {old_dir}")
    old = load_run(old_dir)
    print(f"Reading {new_dir}")
    new = load_run(new_dir)
    print(f"  {len(old)} leaves in {args.old_run}, {len(new)} in {args.new_run}")

    result = compare(old, new, args.abs_threshold)
    report = format_report(args.old_run, args.new_run, result, args.abs_threshold)
    print("\n" + report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"Delta note -> {args.out}")

    if args.fail_on_movement and (result["moved"] or result["changed"]):
        print(
            f"FAIL: {len(result['moved'])} metrics moved and "
            f"{len(result['changed'])} non-numeric values changed"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
