"""
Run the result_23 geometry probes — single entry point, single --seeds flag.

result_23 ("pathogenicity is a direction = conservation") is a set of CPU probes
on the cached pathogenicity / mechanism ESM-2 embeddings. Each probe lives in its
own module; this orchestrator calls them so the whole set runs from one command
with one seed count.

Probes (all CPU; each writes its own JSON under results/<run>/magnitude_direction/):
  magnitude  — magnitude vs direction decomposition (magnitude_direction.run)
  geometry   — rank + family-transfer of the direction (direction_geometry.run)
  transfer   — direction transfer across path/stability/mechanism (transfer_contrast.run)
  biochem    — is the axis context-free biochemistry? (probe4_axis_identity.run)

The conservation decider (conservation_axis.py) is NOT included here — it has a
GPU masked-LL extraction phase and is run separately.

Usage:
    python -m esm2_mech.experiments.geometry.run_geometry --seeds 5
    python -m esm2_mech.experiments.geometry.run_geometry --seeds 5 --probe magnitude geometry
"""

from __future__ import annotations

import argparse
import functools

from esm2_mech.experiments.geometry import (
    magnitude_direction,
    direction_geometry,
    transfer_contrast,
    probe4_axis_identity,
)
from esm2_mech.utils.constants import N_SEEDS

print = functools.partial(print, flush=True)

# Probe name -> the run(n_seeds) callable that executes it and writes its JSON.
PROBES = {
    "magnitude": magnitude_direction.run,
    "geometry": direction_geometry.run,
    "transfer": transfer_contrast.run,
    "biochem": probe4_axis_identity.run,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=N_SEEDS, help="number of seeds (>=1)")
    parser.add_argument(
        "--probe",
        nargs="+",
        choices=list(PROBES) + ["all"],
        default=["all"],
        help="which probe(s) to run (default: all)",
    )
    parser.add_argument(
        "--stability-dataset",
        choices=["none", "tsuboyama"],
        default="none",
        help=(
            "dataset for the stability arm of the magnitude and transfer probes "
            "(Probe C / P4 and the transfer-contrast stability row); default none = skip"
        ),
    )
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")

    selected = list(PROBES) if "all" in args.probe else args.probe
    # Only these two probes have a stability arm; the rest take n_seeds only.
    stability_aware = {"magnitude", "transfer"}
    print(
        f"=== geometry probes: {selected}  (seeds={args.seeds}, "
        f"stability_dataset={args.stability_dataset}) ==="
    )
    for name in selected:
        print(f"\n########## {name} ##########")
        if name in stability_aware:
            PROBES[name](n_seeds=args.seeds, stability_dataset=args.stability_dataset)
        else:
            PROBES[name](n_seeds=args.seeds)
    print(f"\n=== done: {selected} ===")


if __name__ == "__main__":
    main()
