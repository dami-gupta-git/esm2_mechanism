"""Orchestrator for geometry probes (all CPU). Conservation decider runs separately (GPU)."""

from __future__ import annotations

import argparse
import functools

from esm2_mech.experiments.geometry import (
    magnitude_direction,
    direction_geometry,
    transfer_contrast,
    probe4_axis_identity,
)
from esm2_mech.utils.constants import BOOTSTRAP_N_RESAMPLES, N_SEEDS

print = functools.partial(print, flush=True)

PROBES = {
    "magnitude": magnitude_direction.run,
    "geometry": direction_geometry.run,
    "transfer": transfer_contrast.run,
    "biochem": probe4_axis_identity.run,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", type=int, default=N_SEEDS, help="number of seeds (>=1)"
    )
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
            "(default none = skip); a selected dataset must have validated inputs"
        ),
    )
    parser.add_argument(
        "--no_ci",
        action="store_true",
        help="skip cluster-bootstrap CIs (magnitude probe only — the others are "
        "exploratory correlation/transfer probes without bootstrap CIs)",
    )
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_N_RESAMPLES)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")

    selected = list(PROBES) if "all" in args.probe else args.probe
    stability_aware = {"magnitude", "transfer"}
    ci_aware = {"magnitude"}
    print(
        f"=== geometry probes: {selected}  (seeds={args.seeds}, "
        f"stability_dataset={args.stability_dataset}) ==="
    )
    for name in selected:
        print(f"\n########## {name} ##########")
        kwargs = {"n_seeds": args.seeds}
        if name in stability_aware:
            kwargs["stability_dataset"] = args.stability_dataset
        if name in ci_aware:
            kwargs["compute_ci"] = not args.no_ci
            kwargs["n_boot"] = args.n_boot
        PROBES[name](**kwargs)
    print(f"\n=== done: {selected} ===")


if __name__ == "__main__":
    main()
